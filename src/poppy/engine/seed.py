import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from poppy.db import apply_row_factory, rollback_and_close, write_txn
from poppy.db import connect as connect_db
from poppy.engine._closet_marker import (
    NOT_CLOSET_SQL,
    STALE_VECTOR_MODEL_ID,
    chunked,
    clear_closet_tombstones,
    delete_marked_closets,
    ensure_closet_side_tables,
    is_marked_closet_row,
    is_proven_unmarked_copy,
    migrate_closet_marker,
    note_leaked_cloud_copy,
    utc_iso,
)
from poppy.engine._timestamps import expiry_passed, normalise_stored_timestamps
from poppy.engine.interface import ConsolidationResult, EngineStats, RetrievalEngine
from poppy.models import Filters, Memory, ScoredMemory, Source

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    project TEXT,
    source_type TEXT NOT NULL,
    source_session_id TEXT,
    source_timestamp TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    related_to TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT,
    -- Provenance marker for bloom's synthetic per-speaker rows. Seed never sets
    -- it, but it declares the column so a seed-created store the default engine
    -- later opens already has the shape.
    is_closet INTEGER NOT NULL DEFAULT 0
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    id UNINDEXED,
    content
);

CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memory_fts(id, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memories BEGIN
    DELETE FROM memory_fts WHERE id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memories BEGIN
    DELETE FROM memory_fts WHERE id = old.id;
    INSERT INTO memory_fts(id, content) VALUES (new.id, new.content);
END;
"""


def _migrate_expires_at(conn: sqlite3.Connection) -> None:
    """Idempotently add expires_at to memories table for DBs created before lifecycle work."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(memories)")}
    if "expires_at" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN expires_at TEXT")
        conn.commit()


def _fts_triggers_enriched(conn: sqlite3.Connection) -> bool:
    """Whether all three FTS triggers exist in their enriched_content form.

    Inspected from sqlite_master so a partially applied prior migration (any
    trigger missing, or the legacy content-form wiring still live) is detected
    on the next construction and repaired.
    """
    rows = dict(
        conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'trigger'"
            " AND name IN ('memory_ai', 'memory_ad', 'memory_au')"
        )
    )
    if set(rows) != {"memory_ai", "memory_ad", "memory_au"}:
        return False
    # memory_ad only deletes by id and never references a content column, so
    # the enriched wiring is asserted on the two inserting triggers.
    return all("enriched_content" in (rows[name] or "") for name in ("memory_ai", "memory_au"))


def _migrate_enriched_content(conn: sqlite3.Connection) -> None:
    """Idempotently add ``enriched_content`` and rewire FTS triggers to point at it.

    The ``bloom`` engine indexes a derived enrichment of each memory (a
    preamble + per-turn formatting) instead of the raw content. It expects
    ``memories.enriched_content`` to exist and the FTS triggers to write that
    column into ``memory_fts``. DBs created by ``seed`` predate this column.

    Migration steps (all idempotent):
      1. ALTER TABLE memories ADD COLUMN enriched_content TEXT (nullable so the
         add succeeds on a populated table — SQLite forbids NOT NULL without a
         DEFAULT here).
      2. Backfill enriched_content := content for every row that still has
         NULL, so legacy memories are FTS-searchable on something meaningful.
      3. DROP the three FTS triggers and recreate them pointing at
         enriched_content. The existing memory_fts rows already match the
         backfilled enriched_content (since enriched_content == content for
         legacy rows), so no FTS rebuild is needed — only new ingests need the
         updated trigger wiring.

    Schema-only migration; embeddings are handled separately by
    ``_migrate_embedding_model_id`` and ``poppy migrate-engine``.
    """
    # Fast path: nothing to do, so don't take the write lock on every bloom
    # construction. Only when work remains do we serialize.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
    has_null = bool(
        "enriched_content" in cols
        and conn.execute("SELECT 1 FROM memories WHERE enriched_content IS NULL LIMIT 1").fetchone()
    )
    if "enriched_content" in cols and not has_null and _fts_triggers_enriched(conn):
        return

    # The whole migration runs under one BEGIN IMMEDIATE so no other writer can
    # observe the intermediate state where the FTS triggers have been dropped
    # but not yet recreated — a row inserted in that window would land with no
    # memory_fts entry and stay permanently unsearchable. Seed
    # ingest also takes the write lock, so it waits for this to finish. State is
    # re-read inside the lock because another process may have migrated while we
    # waited.
    conn.execute("BEGIN IMMEDIATE")
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
        if "enriched_content" not in cols:
            conn.execute("ALTER TABLE memories ADD COLUMN enriched_content TEXT")
        # Backfill is cheap; run it whenever any row still has NULL so a partial
        # prior migration completes itself.
        conn.execute("UPDATE memories SET enriched_content = content WHERE enriched_content IS NULL")
        if not _fts_triggers_enriched(conn):
            # Rebuild whenever the live triggers are not the enriched form. The
            # decision cannot key off "column was just added": a construction
            # that failed between the ALTER and this rewrite leaves the column
            # present with the legacy content-form triggers still in place — a
            # retry must self-heal that, not skip the rewrite.
            for name in ("memory_ai", "memory_ad", "memory_au"):
                conn.execute(f"DROP TRIGGER IF EXISTS {name}")
            # Individual execute() calls, not executescript(): the latter issues
            # an implicit COMMIT that would end the BEGIN IMMEDIATE early.
            conn.execute(
                "CREATE TRIGGER memory_ai AFTER INSERT ON memories BEGIN"
                " INSERT INTO memory_fts(id, content) VALUES (new.id, new.enriched_content); END"
            )
            conn.execute(
                "CREATE TRIGGER memory_ad AFTER DELETE ON memories BEGIN DELETE FROM memory_fts WHERE id = old.id; END"
            )
            conn.execute(
                "CREATE TRIGGER memory_au AFTER UPDATE ON memories BEGIN"
                " DELETE FROM memory_fts WHERE id = old.id;"
                " INSERT INTO memory_fts(id, content) VALUES (new.id, new.enriched_content); END"
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _has_enriched_content(conn: sqlite3.Connection) -> bool:
    """Whether ``memories`` carries bloom's ``enriched_content`` column."""
    return "enriched_content" in {row[1] for row in conn.execute("PRAGMA table_info(memories)")}


# Written into memory_embeddings.model_id when a seed write replaces the text a
# vector was computed from. Any value that is not a live engine's model_id would
# do; a named sentinel makes the reason legible in the table and in doctor output.
# Defined beside the closet marker, which retags rebuilt closets the same way.
SEED_INVALIDATED_MODEL_ID = STALE_VECTOR_MODEL_ID


def _has_memory_embeddings(conn: sqlite3.Connection) -> bool:
    """Whether the store has bloom's ``memory_embeddings`` table. Seed never creates it."""
    return (
        conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_embeddings'").fetchone()
        is not None
    )


# Two explicit statement variants per operation rather than assembled SQL: the
# enriched pair targets a store bloom created (NOT NULL enriched_content, FTS
# triggers indexing it), the plain pair a seed-only store.
#
# Every one of them writes ``is_closet = 0`` as a literal. Seed has no speaker
# expansion, so a row it writes is by definition a real memory — including a
# write that lands on an id a marked closet used to occupy. Leaving the marker
# set there would hide the user's note from every list and then destroy it when
# the unrelated parent memory was deleted. The marker is only ever set
# by bloom's closet synthesis; every other write of a row clears it.
_INSERT_SQL = """INSERT INTO memories (id, content, memory_type, project, source_type,
    source_session_id, source_timestamp, confidence, related_to, created_at, updated_at, expires_at,
    is_closet)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)"""

_INSERT_SQL_ENRICHED = """INSERT INTO memories (id, content, memory_type, project, source_type,
    source_session_id, source_timestamp, confidence, related_to, created_at, updated_at, expires_at,
    enriched_content, is_closet)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)"""

# Metadata-only: the text is unchanged, so `is_closet` is left exactly as it is.
# Clearing it here would strip the marker off a row that still holds the parent's
# speaker turns — it would then list, push live, and survive the parent's
# redaction. A `poppy edit <closet id> --project x` is the reachable case
# (lifecycle.edit_memory refuses it outright; this is the second line).
_UPDATE_SQL = """UPDATE memories SET content=?, memory_type=?, project=?, source_type=?,
    source_session_id=?, source_timestamp=?, confidence=?, related_to=?, updated_at=?, expires_at=?
    WHERE id=?"""

# Content write: the row now holds text seed put there, so it is a real memory
# whatever occupied the id before.
_UPDATE_SQL_NEW_CONTENT = """UPDATE memories SET content=?, memory_type=?, project=?, source_type=?,
    source_session_id=?, source_timestamp=?, confidence=?, related_to=?, updated_at=?, expires_at=?,
    is_closet=0
    WHERE id=?"""

_UPDATE_SQL_ENRICHED = """UPDATE memories SET content=?, memory_type=?, project=?, source_type=?,
    source_session_id=?, source_timestamp=?, confidence=?, related_to=?, updated_at=?, expires_at=?,
    enriched_content=?, is_closet=0
    WHERE id=?"""


def _migrate_embedding_model_id(conn: sqlite3.Connection) -> None:
    """Idempotently add model_id to memory_embeddings.

    Tags every BLOB with the bi-encoder that produced it. retrieve() filters
    to matching rows so a later engine swap doesn't silently mix vector
    spaces; ``poppy migrate-engine`` uses the same column to find rows that
    need re-embedding. NULL model_id (legacy rows) is treated as untrusted —
    excluded from cosine scoring until re-embedded. The seed engine never
    creates memory_embeddings, so this is a no-op there.
    """
    table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_embeddings'"
    ).fetchone()
    if not table_exists:
        return
    # Use positional access (PRAGMA returns cid, name, type, ...) so this works
    # whether or not the caller set row_factory = sqlite3.Row.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(memory_embeddings)")}
    if "model_id" not in cols:
        conn.execute("ALTER TABLE memory_embeddings ADD COLUMN model_id TEXT")
        conn.commit()


class SeedEngine(RetrievalEngine):
    """FTS5-only retrieval — no ML deps, no model downloads. The universal fallback."""

    # SeedEngine has no embedding model; migration tooling uses model_id to
    # decide which rows to re-embed, so it must be None here.
    model_id = None
    # See the note on the default engine: sync checks this rather than the
    # signature before passing ``remote_event_ts``.
    accepts_remote_event_ts = True
    _engine_name = "seed"

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        self._conn = connect_db(db_path, check_same_thread=False)
        try:
            apply_row_factory(self._conn)
            # Seed never creates either artefact, so this answers "has the
            # default engine ever written here?". A store that only ever ran
            # seed cannot hold a per-speaker copy, whatever its ids look like.
            had_bloom_schema = _has_memory_embeddings(self._conn) or _has_enriched_content(self._conn)
            self._conn.executescript(SCHEMA)
            # Before the migration: its backfill writes to both side tables.
            ensure_closet_side_tables(self._conn)
            _migrate_expires_at(self._conn)
            # A store bloom created before the marker existed still holds
            # unmarked closets; seed must know which rows those are to keep them
            # out of list_all/sync and to clear them on a redaction.
            migrate_closet_marker(self._conn, had_bloom_schema=had_bloom_schema)
            # After the marker migration, whose backfill copies a parent's
            # timestamps onto its copies verbatim: this then puts the whole store
            # — those rows included — into one spelling.
            normalise_stored_timestamps(self._conn)
        except Exception:
            rollback_and_close(self._conn)
            raise

    def note_leaked_cloud_copy(self, memory: Memory) -> bool:
        """Queue a cleanup announcement for an incoming row that is a leaked copy.

        Under the write transaction: called from sync's pull outside any write of
        ours, so an uncommitted insert would be invisible to the TombstoneStore's
        own connection and lost when this one closes.
        """
        with self._lock, write_txn(self._conn):
            return note_leaked_cloud_copy(
                self._conn,
                memory.id,
                content=memory.content,
                related_to=memory.related_to,
                created_at=memory.created_at.isoformat(),
                updated_at=memory.updated_at.isoformat(),
            )

    def is_closet_row(self, memory_id: str) -> bool:
        """Whether ``memory_id`` names a derived copy the default engine wrote."""
        with self._lock:
            return is_marked_closet_row(self._conn, memory_id)

    def is_proven_copy_row(self, memory_id: str) -> bool:
        """Whether an UNMARKED row at ``memory_id`` is, on full proof, a copy of its live parent."""
        with self._lock:
            return is_proven_unmarked_copy(self._conn, memory_id)

    def _clear_closets(self, memory_id: str, *, event_ts: datetime | None = None) -> None:
        """Remove the marked per-speaker copies bloom derived from ``memory_id``.

        Seed has no speaker expansion, so it cannot rebuild them — but leaving
        them is the redaction hole: after a fallback-engine edit or delete the
        copies keep the OLD text and stay recallable the moment the user switches
        back to the default engine. Clearing is lossless, because closets are
        derived data that bloom regenerates from the parent on its next ingest,
        and because they are excluded from ``list_all`` so no cloud copy of a
        locally-cleared closet can resurrect through pull.

        Scoped to rows carrying the provenance marker, so a real memory whose id
        happens to sit under this parent's prefix is never touched.

        Both marker states go, including copies adopted on inference: this runs
        only when the memory's text has CHANGED or the memory is being deleted,
        and no copy of text that is being redacted may survive. Adopted rows are
        snapshotted first, inside ``delete_marked_closets``.
        """
        delete_marked_closets(
            self._conn, memory_id, has_embeddings=_has_memory_embeddings(self._conn), event_ts=event_ts
        )

    def _invalidate_parent_embedding(self, memory_id: str) -> None:
        """Retag ``memory_id``'s vector as stale instead of deleting it.

        The vector was computed from the text this write is replacing, so it no
        longer describes the row. Deleting it would be data loss with no repair
        path: ``migrate-engine`` and ``doctor`` both reach rows by joining
        memory_embeddings, so a row with no embedding is invisible to the very
        tooling meant to fix it. Retagging keeps the row visible and repairable
        — ``_build_where_clause`` selects on ``model_id IS NULL OR model_id !=
        active``, so the sentinel makes it a migrate-engine target and a doctor
        stale count, while retrieve's model_id filter already keeps the stale
        vector out of cosine scoring in the meantime.

        Scope is this row's own vector; the closets bloom derived from it are
        cleared outright by ``_clear_closets``, not retagged, because their text
        is a copy of the text this write is removing.

        No-op when the store has no memory_embeddings table, no model_id column
        (those rows already count as unknown and are migrate targets anyway),
        or no vector for this id.
        """
        if not _has_memory_embeddings(self._conn):
            return
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(memory_embeddings)")}
        if "model_id" not in cols:
            return
        self._conn.execute(
            "UPDATE memory_embeddings SET model_id = ? WHERE id = ?",
            (SEED_INVALIDATED_MODEL_ID, memory_id),
        )

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        expires_at_raw = row["expires_at"] if "expires_at" in row.keys() else None
        return Memory(
            id=row["id"],
            content=row["content"],
            memory_type=row["memory_type"],
            source=Source(
                type=row["source_type"],
                session_id=row["source_session_id"],
                timestamp=datetime.fromisoformat(row["source_timestamp"]),
            ),
            project=row["project"],
            related_to=json.loads(row["related_to"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            confidence=row["confidence"],
            expires_at=datetime.fromisoformat(expires_at_raw) if expires_at_raw else None,
        )

    def ingest(self, memory: Memory, *, remote_event_ts: datetime | None = None) -> str:
        # Every timestamp column goes in as UTC text, in the one canonical
        # spelling. These columns are compared AS TEXT by the push watermark
        # filter, the expiry purge and the `since` filter, so a row stamped
        # `12:00+02:00` by an importer or a research harness would sort above a
        # later `11:00+00:00` and make each of those answer wrongly.
        expires_iso = utc_iso(memory.expires_at)
        source_iso = utc_iso(memory.source.timestamp)
        created_iso = utc_iso(memory.created_at)
        updated_iso = utc_iso(memory.updated_at)
        # A store first written by ``bloom`` carries a NOT NULL ``enriched_content``
        # column, and its FTS triggers index that column rather than ``content``.
        # Seed has no enrichment step, so it mirrors the raw content there: the
        # write succeeds and the row stays keyword-searchable under both engines.
        # Probed per write rather than cached at construction because another
        # process can migrate the store to the bloom schema while this engine is
        # alive; the write transaction is what makes probe-then-write atomic.
        with self._lock, write_txn(self._conn):
            enriched_schema = _has_enriched_content(self._conn)
            existing = self.get(memory.id)
            # Only a changed body invalidates what bloom derived from this
            # memory. A metadata-only re-ingest — the project/type/expiry edits
            # lifecycle.edit_memory replays with identical content — leaves
            # bloom's enrichment and vector accurate, so touching them would
            # spend good data for nothing.
            content_changed = existing is not None and existing.content != memory.content
            write_enriched = enriched_schema and (existing is None or content_changed)
            if content_changed:
                # Gated on content change alone, not on enriched_schema:
                # ``_invalidate_parent_embedding`` self-guards a missing table
                # or column, and a store that has ``memory_embeddings`` but not
                # ``enriched_content`` (bloom construction died mid-migration)
                # still holds a vector for the old text that must be invalidated.
                # The marked per-speaker copies bloom derived
                # from this memory hold the OLD text verbatim, so the edit clears
                # them too: leaving them is what let a redaction on the fallback
                # engine stay recallable under the default one.
                self._invalidate_parent_embedding(memory.id)
                self._clear_closets(memory.id, event_ts=remote_event_ts)
            if existing is not None:
                # A same-text write onto a marked copy keeps the marker, so it
                # has to keep the copy's back-reference to its parent as well:
                # that link is what the parent's redaction reads to find its own
                # copies (``_owned_by``). The caller's Memory usually arrives
                # with an empty related_to, and writing it verbatim would leave
                # a marked copy no parent can reach. Bloom does the
                # same on its marker-keeping path.
                #
                # An UNMARKED row that GRADES as a copy keeps it too. A store
                # that never ran bloom never adopts the copy it pulled, so the
                # link is the only thing identifying it — and erasing it on a
                # same-text re-ingest (an external writer replaying the row with
                # no related_to) made the parent's forget grade it a stranger and
                # leave the speaker text live. Graded before the write,
                # while the stored row is still the evidence.
                keeps_link = not content_changed and (
                    self.is_closet_row(memory.id) or is_proven_unmarked_copy(self._conn, memory.id)
                )
                related_to = existing.related_to if keeps_link else memory.related_to
                params = (
                    memory.content,
                    memory.memory_type,
                    memory.project,
                    memory.source.type,
                    memory.source.session_id,
                    source_iso,
                    memory.confidence,
                    json.dumps(related_to),
                    updated_iso,
                    expires_iso,
                )
                if write_enriched:
                    # Raw content is the best enrichment seed can produce; it
                    # only replaces bloom's when the old one described text
                    # this write is removing.
                    self._conn.execute(_UPDATE_SQL_ENRICHED, (*params, memory.content, memory.id))
                elif content_changed:
                    self._conn.execute(_UPDATE_SQL_NEW_CONTENT, (*params, memory.id))
                else:
                    # Metadata only: leave the marker alone.
                    self._conn.execute(_UPDATE_SQL, (*params, memory.id))
                if content_changed:
                    # The id now holds text this engine wrote, so it is a real
                    # memory again: any record of it having been deleted as a
                    # derived copy is stale and must not make pull skip it.
                    clear_closet_tombstones(self._conn, [memory.id])
            else:
                params = (
                    memory.id,
                    memory.content,
                    memory.memory_type,
                    memory.project,
                    memory.source.type,
                    memory.source.session_id,
                    source_iso,
                    memory.confidence,
                    json.dumps(memory.related_to),
                    created_iso,
                    updated_iso,
                    expires_iso,
                )
                if enriched_schema:
                    self._conn.execute(_INSERT_SQL_ENRICHED, (*params, memory.content))
                else:
                    self._conn.execute(_INSERT_SQL, params)
                # A fresh row reclaims the id for a real memory (see above).
                clear_closet_tombstones(self._conn, [memory.id])
            return memory.id

    def retrieve(self, query: str, filters: Filters | None = None, limit: int = 10) -> list[ScoredMemory]:
        escaped_query = '"' + query.replace('"', '""') + '"'
        with self._lock:
            try:
                rows = self._conn.execute(
                    """SELECT m.*, rank FROM memory_fts fts
                       JOIN memories m ON fts.id = m.id
                       WHERE memory_fts MATCH ?
                       ORDER BY rank
                       LIMIT ?""",
                    (escaped_query, limit * 3),
                ).fetchall()
            except Exception:
                return []

        now = datetime.now(timezone.utc)
        include_expired = bool(filters and filters.include_expired)
        results = []
        for row in rows:
            mem = self._row_to_memory(row)
            if not include_expired and mem.expires_at is not None and mem.expires_at <= now:
                continue
            if filters:
                if filters.project and mem.project != filters.project:
                    continue
                if filters.memory_type and mem.memory_type != filters.memory_type:
                    continue
                if filters.since and mem.created_at < filters.since:
                    continue
                if filters.min_confidence and mem.confidence < filters.min_confidence:
                    continue
            score = 1.0 / (1.0 + abs(row["rank"]))
            results.append(ScoredMemory(memory=mem, score=score))
            if len(results) >= limit:
                break
        return results

    def get(self, memory_id: str) -> Memory | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def delete(self, memory_id: str, *, remote_event_ts: datetime | None = None) -> bool:
        with self._lock, write_txn(self._conn):
            # Delete the parent row. Its own ``memory_embeddings`` vector is
            # local-only (never synced, recomputed per engine), so it goes here:
            # leaving it lets a later seed insert of the same id silently inherit
            # a vector computed from the deleted text.
            # Copies FIRST, and unconditionally: an unmarked one is graded
            # against this parent's text, so the parent row has to still be here.
            # Unconditional because a copy whose parent row is already gone is
            # exactly the orphan that stays recallable while list_all merely
            # hides it, so a repeat delete sweeps the marked ones.
            self._clear_closets(memory_id, event_ts=remote_event_ts)
            cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            deleted = cursor.rowcount > 0
            if deleted and _has_memory_embeddings(self._conn):
                self._conn.execute("DELETE FROM memory_embeddings WHERE id = ?", (memory_id,))
        return deleted

    def list_all(self, filters: Filters | None = None, limit: int = 50) -> list[Memory]:
        # On a store the default engine created, ``memories`` also holds its
        # synthetic per-speaker rows. They are derived data, excluded here (by
        # the provenance marker, so a real memory whose id merely contains
        # ``_closet_`` is never hidden) so they never reach ``poppy list`` or
        # sync push. Seed's retrieve() FTS-matches memories directly, so keyword
        # recall is unaffected.
        query = f"SELECT * FROM memories WHERE {NOT_CLOSET_SQL}"
        params: list = []
        if filters:
            if filters.project:
                query += " AND project = ?"
                params.append(filters.project)
            if filters.memory_type:
                query += " AND memory_type = ?"
                params.append(filters.memory_type)
            if filters.since:
                # created_at is stored as canonical UTC text, so the SQL string
                # comparison is only correct when the bound is spelled the same
                # way. Through the one helper that does it, so the bound and the
                # stored values can never drift apart; it also reads a naive bound
                # as UTC, the convention the rest of the store uses.
                query += " AND created_at >= ?"
                params.append(utc_iso(filters.since))
            if filters.min_confidence:
                query += " AND confidence >= ?"
                params.append(filters.min_confidence)
        if not (filters and filters.include_expired):
            query += " AND (expires_at IS NULL OR expires_at > ?)"
            params.append(datetime.now(timezone.utc).isoformat())
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def purge_expired(self) -> int:
        """Hard-delete memories whose expires_at is in the past. Returns rowcount.

        The count is memories, not rows: on a store the default engine wrote, an
        expired parent's per-speaker copies expire with it and are swept in the
        same pass, but they were never memories in their own right.
        """
        now = datetime.now(timezone.utc)
        with self._lock, write_txn(self._conn):
            # Expiry is decided by PARSING the stored stamp, not by comparing it
            # as text against ``now``. Writes normalise to UTC, but a row that
            # reached the store another way can carry any offset, and as text
            # ``2026-07-01T13:00:00-12:00`` (an hour in the future) sorts below
            # noon UTC — which purged live memories on the spot. An
            # unparseable stamp is left alone rather than treated as expired.
            expired = [
                (row[0], row[1])
                for row in self._conn.execute(
                    f"SELECT id, {NOT_CLOSET_SQL}, expires_at FROM memories WHERE expires_at IS NOT NULL"
                ).fetchall()
                if expiry_passed(row[2], now)
            ]
            if not expired:
                return 0
            # Cascade to the marked copies of every expiring parent, mirroring
            # the default engine. A parent whose expiry was moved by a
            # fallback-engine edit outlives its copies' stored expires_at (or the
            # reverse), so purging by timestamp alone can leave orphans standing
            # — hidden from list_all but still recallable.
            has_embeddings = _has_memory_embeddings(self._conn)
            for mid, is_memory in expired:
                if is_memory:
                    # No tombstones: expiry is not a redaction, and the cloud row
                    # carries the same expires_at.
                    delete_marked_closets(self._conn, mid, has_embeddings=has_embeddings, tombstone=False)
            # Deleted BY ID, from the list the parse decided: a second text
            # comparison here would remove exactly the rows the parse just
            # spared.
            for batch in chunked([mid for mid, _ in expired]):
                placeholders = ",".join("?" * len(batch))
                if has_embeddings:
                    self._conn.execute(f"DELETE FROM memory_embeddings WHERE id IN ({placeholders})", batch)
                self._conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", batch)
            return sum(1 for _, is_memory in expired if is_memory)

    def consolidate(self) -> ConsolidationResult:
        return ConsolidationResult(merged=0, removed=0, updated=0)

    def stats(self) -> EngineStats:
        with self._lock:
            # Memories, not rows — see the note on the default engine's stats().
            count = self._conn.execute(f"SELECT COUNT(*) FROM memories WHERE {NOT_CLOSET_SQL}").fetchone()[0]
        storage = self._db_path.stat().st_size if self._db_path.exists() else 0
        return EngineStats(
            memory_count=count,
            storage_bytes=storage,
            engine_name="seed",
            engine_version="0.1.0",
        )
