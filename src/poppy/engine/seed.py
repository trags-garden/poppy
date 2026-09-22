import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from poppy.db import apply_row_factory, open_with_upgrades, write_txn
from poppy.engine._timestamps import (
    TIMESTAMP_MIGRATION,
    chunked,
    expiry_passed,
    migration_applied,
    normalise_stored_timestamps,
    unexpired_sql,
    utc_iso,
)
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
    expires_at TEXT
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


def _enriched_content_current(conn: sqlite3.Connection) -> bool:
    """Whether ``enriched_content`` and its FTS triggers are fully in place.

    Both the migration's fast path and the pre-open currency check ask this, so
    they can never disagree about what "already migrated" means. Runs on a
    read-only probe connection too, hence positional row access.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
    if "enriched_content" not in cols:
        return False
    if conn.execute("SELECT 1 FROM memories WHERE enriched_content IS NULL LIMIT 1").fetchone():
        return False
    return _fts_triggers_enriched(conn)


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
    if _enriched_content_current(conn):
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
SEED_INVALIDATED_MODEL_ID = "seed-invalidated"


def _has_memory_embeddings(conn: sqlite3.Connection) -> bool:
    """Whether the store has bloom's ``memory_embeddings`` table. Seed never creates it."""
    return (
        conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_embeddings'").fetchone()
        is not None
    )


# Two explicit statement variants per operation rather than assembled SQL: the
# enriched pair targets a store bloom created (NOT NULL enriched_content, FTS
# triggers indexing it), the plain pair a seed-only store.
_INSERT_SQL = """INSERT INTO memories (id, content, memory_type, project, source_type,
    source_session_id, source_timestamp, confidence, related_to, created_at, updated_at, expires_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

_INSERT_SQL_ENRICHED = """INSERT INTO memories (id, content, memory_type, project, source_type,
    source_session_id, source_timestamp, confidence, related_to, created_at, updated_at, expires_at,
    enriched_content)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

_UPDATE_SQL = """UPDATE memories SET content=?, memory_type=?, project=?, source_type=?,
    source_session_id=?, source_timestamp=?, confidence=?, related_to=?, updated_at=?, expires_at=?
    WHERE id=?"""

_UPDATE_SQL_ENRICHED = """UPDATE memories SET content=?, memory_type=?, project=?, source_type=?,
    source_session_id=?, source_timestamp=?, confidence=?, related_to=?, updated_at=?, expires_at=?,
    enriched_content=?
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


def _seed_store_is_current(conn: sqlite3.Connection) -> bool:
    """Whether a store already holds everything a seed open would write.

    Asked of a read-only probe before any lock is taken, so it must answer from
    reads alone and must be conservative: anything it cannot confirm counts as
    work owed, and the open takes the gate.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
    if "expires_at" not in cols:
        return False
    return migration_applied(conn, TIMESTAMP_MIGRATION)


class SeedEngine(RetrievalEngine):
    """FTS5-only retrieval — no ML deps, no model downloads. The universal fallback."""

    # SeedEngine has no embedding model; migration tooling uses model_id to
    # decide which rows to re-embed, so it must be None here.
    model_id = None
    _engine_name = "seed"

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        # Opening a store that still owes its one-time upgrades is a write, and
        # several processes doing it at once collide; ``open_with_upgrades`` owns
        # that policy for both engines. A store that is already current takes no
        # lock at all, so the hook and read paths never wait on a writer.
        self._conn = open_with_upgrades(
            db_path,
            store_is_current=_seed_store_is_current,
            apply_upgrades=self._apply_open_time_work,
            check_same_thread=False,
        )

    @staticmethod
    def _apply_open_time_work(conn: sqlite3.Connection) -> None:
        """Bring a just-opened store up to date. Idempotent, and re-run inside the gate."""
        apply_row_factory(conn)
        conn.executescript(SCHEMA)
        _migrate_expires_at(conn)
        normalise_stored_timestamps(conn)

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

    def ingest(self, memory: Memory) -> str:
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
                self._invalidate_parent_embedding(memory.id)
            if existing is not None:
                params = (
                    memory.content,
                    memory.memory_type,
                    memory.project,
                    memory.source.type,
                    memory.source.session_id,
                    source_iso,
                    memory.confidence,
                    json.dumps(memory.related_to),
                    updated_iso,
                    expires_iso,
                )
                if write_enriched:
                    # Raw content is the best enrichment seed can produce; it
                    # only replaces bloom's when the old one described text
                    # this write is removing.
                    self._conn.execute(_UPDATE_SQL_ENRICHED, (*params, memory.content, memory.id))
                else:
                    self._conn.execute(_UPDATE_SQL, (*params, memory.id))
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

    def delete(self, memory_id: str) -> bool:
        with self._lock, write_txn(self._conn):
            # Delete the row. Its own ``memory_embeddings`` vector is
            # local-only (never synced, recomputed per engine), so it goes here:
            # leaving it lets a later seed insert of the same id silently inherit
            # a vector computed from the deleted text.
            cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            deleted = cursor.rowcount > 0
            if deleted and _has_memory_embeddings(self._conn):
                self._conn.execute("DELETE FROM memory_embeddings WHERE id = ?", (memory_id,))
        return deleted

    def list_all(self, filters: Filters | None = None, limit: int = 50) -> list[Memory]:
        clauses: list[str] = []
        params: list = []
        if filters:
            if filters.project:
                clauses.append("project = ?")
                params.append(filters.project)
            if filters.memory_type:
                clauses.append("memory_type = ?")
                params.append(filters.memory_type)
            if filters.since:
                # created_at is stored as canonical UTC text, so the SQL string
                # comparison is only correct when the bound is spelled the same
                # way. Through the one helper that does it, so the bound and the
                # stored values can never drift apart; it also reads a naive bound
                # as UTC, the convention the rest of the store uses.
                clauses.append("created_at >= ?")
                params.append(utc_iso(filters.since))
            if filters.min_confidence:
                clauses.append("confidence >= ?")
                params.append(filters.min_confidence)
        if not (filters and filters.include_expired):
            clause, bound = unexpired_sql()
            clauses.append(clause)
            params.append(bound)
        query = "SELECT * FROM memories"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def purge_expired(self) -> int:
        """Hard-delete expired memories and any stored embeddings atomically."""
        now = datetime.now(timezone.utc)
        with self._lock, write_txn(self._conn):
            expired = [
                row[0]
                for row in self._conn.execute(
                    "SELECT id, expires_at FROM memories WHERE expires_at IS NOT NULL"
                ).fetchall()
                if expiry_passed(row[1], now)
            ]
            has_embeddings = _has_memory_embeddings(self._conn)
            for batch in chunked(expired):
                placeholders = ",".join("?" * len(batch))
                if has_embeddings:
                    self._conn.execute(f"DELETE FROM memory_embeddings WHERE id IN ({placeholders})", batch)
                self._conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", batch)
            return len(expired)

    def consolidate(self) -> ConsolidationResult:
        return ConsolidationResult(merged=0, removed=0, updated=0)

    def stats(self) -> EngineStats:
        # Counts what `list_all` shows, through the same clause: expired rows are
        # already invisible to every reader, so counting them would report a
        # total nothing in the UI or the CLI can account for.
        clause, bound = unexpired_sql()
        with self._lock:
            count = self._conn.execute(f"SELECT COUNT(*) FROM memories WHERE {clause}", (bound,)).fetchone()[0]
        storage = self._db_path.stat().st_size if self._db_path.exists() else 0
        return EngineStats(
            memory_count=count,
            storage_bytes=storage,
            engine_name="seed",
            engine_version="0.1.0",
        )
