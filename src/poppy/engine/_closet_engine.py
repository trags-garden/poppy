"""Shared closet-hybrid retrieval engine (runtime-agnostic).

This module holds the retrieval architecture once, separate from the embedding
runtime that feeds it: ``bloom`` supplies ONNX encoders here, and the
experimental backends in poppy-lab supply their own. Keeping the pipeline in one
place is what makes a runtime comparison honest — any recall delta is the
encoder, not a drifted pipeline.

Two-stage retrieval (hybrid FTS5+embeddings -> cross-encoder rerank) with a
per-speaker content expansion at ingest. For each multi-speaker session, the
engine stores a synthetic memory containing only one speaker's turns alongside
the full session, so queries like "When did Alice go to X?" can score the
Alice-only memory higher than the joint session.

    main memory:    id=D1,               content=[alice turns + bob turns]
    alice closet:   id=D1_closet_alice,  content=[alice turns only]
    bob closet:     id=D1_closet_bob,    content=[bob turns only]

All variants compete equally in retrieval; cross-speaker queries still tend to
prefer the full session because only that one contains both speakers' turns.

Nothing here imports an ML runtime: subclasses supply ``_embed`` and ``_rerank``
(plus their own model loading), so this module stays cheap to import and usable
by any backend.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from poppy.db import apply_row_factory, rollback_and_close, write_txn
from poppy.db import connect as connect_db

# The marker, and the pure-text closet derivation it shares with the one-time
# migration. Both live beside each other in `_closet_marker` so the
# dependency-free ``seed`` engine can use them without importing this module.
from poppy.engine._closet_marker import (
    CLOSET_ADOPTED_UNVERIFIED,
    CLOSET_DERIVED,
    MARKER_COLUMN,
    NOT_CLOSET_SQL,
    TIER_LIKELY,
    TIER_PROVEN,
    _columns,
    _table_exists,
    back_up_migrated_rows,
    chunked,
    classify_closet_row,
    clear_closet_deletion_records,
    clear_closet_tombstones,
    closet_id,
    delete_marked_closets,
    derive_closets,
    ensure_closet_side_tables,
    is_marked_closet_row,
    is_proven_unmarked_copy,
    migrate_closet_marker,
    note_leaked_cloud_copy,
    record_closet_tombstones,
    record_legacy_closet_ids,
    speaker_of,
    utc_iso,
)
from poppy.engine._timestamps import expiry_passed, normalise_stored_timestamps
from poppy.engine.interface import ConsolidationResult, EngineStats, RetrievalEngine
from poppy.models import Filters, Memory, ScoredMemory, Source

logger = logging.getLogger(__name__)

FIRST_STAGE_K = 100
RRF_K = 60

STOPWORDS = frozenset(
    "a an the is was were be been being am are do does did have has had "
    "will would shall should may might can could of in to for on with at by "
    "from as into about between through during before after above below "
    "and or but not no nor so yet both either neither each every all any "
    "what which who whom whose when where why how that this these those "
    "i me my we us our you your he him his she her it its they them their".split()
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    enriched_content TEXT NOT NULL,
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
    -- Provenance marker: 1 only on the synthetic per-speaker rows this engine
    -- creates. Set at creation, never inferred from the id.
    is_closet INTEGER NOT NULL DEFAULT 0
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    id UNINDEXED,
    content
);

CREATE TRIGGER IF NOT EXISTS memory_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memory_fts(id, content) VALUES (new.id, new.enriched_content);
END;

CREATE TRIGGER IF NOT EXISTS memory_ad AFTER DELETE ON memories BEGIN
    DELETE FROM memory_fts WHERE id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS memory_au AFTER UPDATE ON memories BEGIN
    DELETE FROM memory_fts WHERE id = old.id;
    INSERT INTO memory_fts(id, content) VALUES (new.id, new.enriched_content);
END;

CREATE TABLE IF NOT EXISTS memory_embeddings (
    id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL
);
"""


def _tokenize_for_fts(query: str) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", query.lower())
    terms = [w for w in words if w not in STOPWORDS and len(w) > 1]
    if not terms:
        return ""
    return " OR ".join(f'"{t}"' for t in terms)


def _enrich_full_content(content: str, session_timestamp: str | None = None) -> str:
    """Champion's enrichment — full session with all speakers."""
    try:
        turns = json.loads(content)
        if not isinstance(turns, list):
            return content
    except (json.JSONDecodeError, TypeError):
        return content

    # ``speaker_of`` falls back to "Unknown" for anything that is not a
    # non-empty string. Whatever JSON is in the store reaches here — a list, a
    # dict, a number — and enrichment must degrade, not raise: a TypeError here
    # would take down every ingest of that memory.
    speakers = []
    seen: set[str] = set()
    for turn in turns:
        # Skipped in the rendering loop below too: counting a non-dict turn here
        # would name a speaker in the preamble that has no lines under it.
        if not isinstance(turn, dict):
            continue
        speaker = speaker_of(turn) or "Unknown"
        if speaker not in seen:
            speakers.append(speaker)
            seen.add(speaker)

    date_str = ""
    if session_timestamp:
        try:
            dt = datetime.fromisoformat(session_timestamp)
            date_str = dt.strftime("%B %d, %Y at %I:%M %p")
        except ValueError:
            date_str = session_timestamp

    speaker_str = " and ".join(speakers) if len(speakers) <= 2 else ", ".join(speakers[:-1]) + f", and {speakers[-1]}"

    lines = []
    if date_str and speakers:
        lines.append(f"Conversation on {date_str} between {speaker_str}.")
    elif date_str:
        lines.append(f"Conversation on {date_str}.")
    elif speakers:
        lines.append(f"Conversation between {speaker_str}.")

    lines.append("")

    for turn in turns:
        if not isinstance(turn, dict):
            continue
        speaker = speaker_of(turn) or "Unknown"
        dia_id = turn.get("dia_id", "")
        text = turn.get("text", "")
        if dia_id and text:
            lines.append(f"{dia_id} {speaker}: {text}")
        elif text:
            lines.append(f"{speaker}: {text}")

    return "\n".join(lines) if lines else content


class ClosetHybridEngine(RetrievalEngine):
    """Champion retrieval + per-speaker closet memories, model-runtime agnostic.

    Subclasses provide the embedding runtime by implementing ``_embed`` (a single
    text -> normalized ``np.float32`` vector) and ``_rerank`` (query + docs ->
    per-doc relevance scores), and set ``model_id`` / ``_engine_name`` /
    ``_engine_version``. This base owns the SQLite schema, migrations, ingest
    (incl. closet synthesis), hybrid FTS5+cosine RRF retrieval, cross-encoder
    rerank, and lifecycle.
    """

    # Subclasses override. ``model_id`` tags every BLOB this engine writes so a
    # later swap to a backend with different vectors never mixes incompatible
    # vector spaces in the shared memory_embeddings table.
    model_id: str | None = None
    # This engine accepts ``remote_event_ts`` on ingest and delete. Sync checks
    # the flag rather than the signature, so an engine that predates it (or a
    # third-party one) is called the old way instead of raising.
    accepts_remote_event_ts = True
    _engine_name = "closet-hybrid"
    _engine_version = "1.0.0"

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        # One connection, usable from any thread (the UI serves requests on a
        # pool, and sync runs on its own), so every write sequence is serialized
        # on this lock. Without it two threads mistake each other for nested
        # callers of ``write_txn``: the second one's write joins the first one's
        # transaction and reports success, and the first one's failure rolls it
        # back. Reentrant so a write path may call another one on the
        # same thread; seed holds the same discipline.
        self._lock = threading.RLock()
        self._conn = connect_db(db_path, check_same_thread=False)
        try:
            apply_row_factory(self._conn)
            # Captured BEFORE this engine's own schema work: `executescript`
            # below creates memory_embeddings and the migrations add
            # enriched_content, so afterwards every store looks like one this
            # engine has written. A store that only ever ran `seed` cannot hold a
            # per-speaker copy, and the marker migration must not grade its rows.
            had_bloom_schema = _table_exists(self._conn, "memory_embeddings") or "enriched_content" in _columns(
                self._conn, "memories"
            )
            self._conn.executescript(SCHEMA)
            # Before the migration: its backfill writes to both side tables.
            ensure_closet_side_tables(self._conn)
            from poppy.engine.seed import (
                _migrate_embedding_model_id,
                _migrate_enriched_content,
                _migrate_expires_at,
            )

            # Order matters: expires_at and enriched_content are column-level
            # schema upgrades that must complete before any read path runs. The
            # enriched_content migration also rewires the FTS triggers to point at
            # the new column; the ingest/update paths rely on that.
            _migrate_expires_at(self._conn)
            _migrate_enriched_content(self._conn)
            _migrate_embedding_model_id(self._conn)
            # Last: its backfill rebuilds legacy closets from their parents and
            # writes enriched_content, so it needs the two column upgrades above
            # to have landed.
            migrate_closet_marker(self._conn, had_bloom_schema=had_bloom_schema)
            # After the marker migration, whose backfill copies a parent's
            # timestamps onto its copies verbatim: this then puts the whole store
            # — those rows included — into one spelling.
            normalise_stored_timestamps(self._conn)
        except Exception:
            rollback_and_close(self._conn)
            raise

    # --- embedding runtime seams (subclass responsibility) -----------------

    def _embed(self, text: str) -> np.ndarray:
        """Return a normalized float32 embedding for ``text``."""
        raise NotImplementedError

    def _rerank(self, query: str, docs: list[str]) -> list[float]:
        """Return one relevance score per doc (higher = more relevant)."""
        raise NotImplementedError

    # --- row mapping / filtering ------------------------------------------

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

    def _row_to_memory_enriched(self, row: sqlite3.Row) -> Memory:
        expires_at_raw = row["expires_at"] if "expires_at" in row.keys() else None
        return Memory(
            id=row["id"],
            content=row["enriched_content"],
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

    def _passes_filters(self, mem: Memory, filters: Filters | None) -> bool:
        include_expired = bool(filters and filters.include_expired)
        if not include_expired and mem.expires_at is not None and mem.expires_at <= datetime.now(timezone.utc):
            return False
        if not filters:
            return True
        if filters.project and mem.project != filters.project:
            return False
        if filters.memory_type and mem.memory_type != filters.memory_type:
            return False
        if filters.since and mem.created_at < filters.since:
            return False
        if filters.min_confidence and mem.confidence < filters.min_confidence:
            return False
        return True

    # --- ingest ------------------------------------------------------------

    def _insert_memory(
        self,
        memory_id: str,
        raw_content: str,
        enriched: str,
        memory_type: str,
        project: str | None,
        source: Source,
        related_to: list[str],
        created_at: datetime,
        updated_at: datetime,
        confidence: float,
        expires_at: datetime | None = None,
        is_closet: int = 0,
        embedding: np.ndarray | None = None,
    ) -> None:
        # The vector is computed by the CALLER, outside the write transaction,
        # and passed in. Embedding is the one fallible non-database step in a
        # write: the encoder loads lazily, unloads when idle and can be missing
        # or offline, so raising it from in here aborted the write with the
        # transaction open. Still computed here when a caller omits
        # it, so a subclass or a future call site cannot write an unembedded row.
        emb_blob = (self._embed(enriched) if embedding is None else embedding).tobytes()
        # UTC text for every timestamp column, in the one canonical spelling.
        # These columns are compared AS TEXT by the push watermark filter, the
        # expiry purge and the `since` filter, so a row stamped `12:00+02:00` by
        # an importer or a research harness would sort above a later
        # `11:00+00:00` and make each of those answer wrongly.
        expires_iso = utc_iso(expires_at)
        # ``is_closet`` is written on every insert, not only the closet ones: a
        # re-ingest of a parent must actively clear the marker, or a row that was
        # once derived could keep a stale state and stay hidden from list/sync.
        # It is a STATE, so it is passed through rather than coerced: an
        # unchanged-body re-ingest of a copy must not silently downgrade an
        # adopted-unverified row to a derived one, which would make the next
        # parent write regenerate it.
        #
        # REPLACE resolves its conflict by deleting the old row WITHOUT firing
        # the ``memory_ad`` delete trigger (SQLite only fires it under
        # ``PRAGMA recursive_triggers``), while ``memory_ai`` still fires for the
        # new row — so a re-ingest used to leave the pre-edit enriched_content
        # standing in memory_fts next to the new text, and a redacted secret
        # stayed live in FTS/RRF retrieval. Clearing the id here is the
        # narrow fix: it is a no-op for a first insert, and it does not change
        # trigger semantics anywhere else in the store.
        self._conn.execute("DELETE FROM memory_fts WHERE id = ?", (memory_id,))
        self._conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, content, enriched_content, memory_type, project, source_type,
                source_session_id, source_timestamp, confidence, related_to, created_at, updated_at,
                expires_at, is_closet)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                memory_id,
                raw_content,
                enriched,
                memory_type,
                project,
                source.type,
                source.session_id,
                utc_iso(source.timestamp),
                confidence,
                json.dumps(related_to),
                utc_iso(created_at),
                utc_iso(updated_at),
                expires_iso,
                int(is_closet),
            ),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO memory_embeddings (id, embedding, model_id) VALUES (?, ?, ?)",
            (memory_id, emb_blob, self.model_id),
        )

    def ingest(self, memory: Memory, *, remote_event_ts: datetime | None = None) -> str:
        """Write a memory and re-derive its per-speaker copies.

        ``remote_event_ts`` is when the write HAPPENED, for one sync is applying
        on behalf of another device. Any copy this write removes is recorded with
        that time rather than ours; see ``delete_marked_closets``.
        """
        enriched = _enrich_full_content(
            memory.content,
            session_timestamp=memory.source.timestamp.isoformat(),
        )
        # Every embedding for this write, computed BEFORE the transaction opens.
        # ``_embed`` is the one fallible non-database step on the write path: the
        # encoder loads lazily and unloads when idle, so a cold model cache with
        # no network raises here on an ordinary write in a long-lived `poppy
        # serve`. Inside the transaction that left the write lock held on a
        # connection nobody closes — every other process got "database is
        # locked", and the next successful write committed what this one had
        # already deleted. The closet texts are a pure function of the
        # content, so deriving them here costs nothing and needs no lock.
        parent_embedding = self._embed(enriched)
        closets = [
            (slug, closet_raw, closet_enriched, self._embed(closet_enriched))
            for slug, closet_raw, closet_enriched in derive_closets(memory.content, memory.source.timestamp.isoformat())
        ]

        # One transaction for the whole read-decide-write sequence, rolled back
        # in full if any statement in it fails, and one writer at a time.
        with self._lock, write_txn(self._conn):
            # Whether this write leaves the row's TEXT untouched. A metadata-only
            # re-ingest of a marked closet — the shape `poppy edit <closet id>
            # --project x` produces — must not strip the marker off a row that still
            # holds the parent's speaker turns, or it would list, push live, and
            # survive the parent's redaction. lifecycle.edit_memory refuses
            # that edit outright; this is the second line of defence, at the write.
            # Any CHANGE of the text means the row is a real memory now, whatever
            # occupied the id before, so the marker is cleared as normal.
            prior = self._conn.execute(
                "SELECT content, is_closet, related_to, created_at FROM memories WHERE id = ?", (memory.id,)
            ).fetchone()
            prior_state = int(prior["is_closet"] or 0) if prior is not None else 0
            parent_content_changed = prior is not None and prior["content"] != memory.content
            # A same-text write is an update of the memory that is already here, so
            # it keeps that memory's creation time; the caller's value is whatever
            # the surface built. The creation time is evidence: an unmarked copy of
            # this parent (pulled into a store that never ran bloom) is proven by
            # sharing it, and replacing it here would strand that copy past the
            # parent's redaction. A content change is a different memory at the id
            # and takes the caller's value.
            created_at = memory.created_at
            if prior is not None and not parent_content_changed:
                try:
                    created_at = datetime.fromisoformat(prior["created_at"])
                except (TypeError, ValueError):
                    # A row written outside Poppy with an unparseable stamp must not
                    # wedge every same-text write to it (and with it, sync).
                    created_at = memory.created_at
            keeps_marker = prior is not None and prior["content"] == memory.content and prior_state >= CLOSET_DERIVED
            # A copy's `related_to` is the back-reference to its parent, written when
            # the copy was derived. The caller's Memory is whatever the surface built
            # — often with an empty related_to — so writing it verbatim on a
            # marker-keeping re-ingest would erase the link the migration and the
            # shape test both read. Keep what is stored.
            #
            # An UNMARKED row that GRADES as a copy of its live parent keeps the link
            # for the same reason and more urgently: nothing else identifies it, so
            # erasing it on a same-text re-ingest left the parent's redaction grading
            # its own copy a stranger, with the speaker text live. Graded
            # from the stored row, before this write replaces it.
            keeps_link = keeps_marker or (
                prior is not None
                and prior["content"] == memory.content
                and not prior_state
                and is_proven_unmarked_copy(self._conn, memory.id)
            )
            related_to = json.loads(prior["related_to"] or "[]") if keeps_link else memory.related_to

            # Clear this memory's prior closets (idempotent re-ingest). A content edit
            # replays through here, so this is also what drops the old speaker text
            # before the new closets are synthesised below. Only MARKED rows go: a
            # real memory whose id happens to sit under this prefix is untouched.
            #
            # No tombstones yet: most of these ids are about to be rewritten with the
            # same speakers, and announcing a deletion per closet on every ingest
            # would push a burst of pointless soft-deletes. Only the ids this write
            # does NOT re-create are real deletions, and they are recorded below.
            #
            # `derived_only` turns on whether the TEXT changed, and that is the whole
            # rule: redaction beats preservation.
            #
            # A metadata-only write replays the same body through here, so nothing
            # about the memory's text has moved and a copy adopted on inference is
            # left alone — keeping its text is the point of that grade.
            #
            # A CONTENT change is a redaction of the old text, so every copy of it
            # goes, adopted ones included. Otherwise editing a secret out of a memory
            # left an adopted copy holding it, still returned by recall. The adopted
            # rows are snapshotted before they go, since they might be a curated
            # split rather than a copy.
            previous_closets = delete_marked_closets(
                self._conn,
                memory.id,
                has_embeddings=True,
                tombstone=False,
                derived_only=not parent_content_changed,
                event_ts=remote_event_ts,
            )

            self._insert_memory(
                memory.id,
                memory.content,
                enriched,
                memory.memory_type,
                memory.project,
                memory.source,
                related_to,
                created_at,
                memory.updated_at,
                memory.confidence,
                memory.expires_at,
                is_closet=prior_state if keeps_marker else 0,
                embedding=parent_embedding,
            )
            if not keeps_marker:
                # The id now holds a real memory, so any record of it having been
                # deleted as a derived copy is stale: leaving it would make pull skip
                # every future update to this memory.
                clear_closet_tombstones(self._conn, [memory.id])

            # Synthesize a closet per speaker if the content has the expected turn
            # shape. ``derive_closets`` is shared with the one-time marker migration,
            # so what is written here and what that migration re-derives can never
            # drift apart.
            rebuilt: set[str] = set()
            for slug, closet_raw, closet_enriched, closet_embedding in closets:
                cid = closet_id(memory.id, slug)
                # Something unmarked can already occupy this id, and it is one of two
                # very different things.
                #
                # A LEAKED COPY OF THIS VERY PARENT. A client at or below 0.2.4
                # pushed its per-speaker copies to the cloud as ordinary memories, so
                # a fresh install pulling that account receives them unmarked — and
                # if the copy arrives before its parent, refusing to derive here
                # would leave it unmarked for ever: listed, synced, and untouched by
                # any redaction of the parent. It is adopted instead.
                #
                # A REAL MEMORY that merely holds the id. Nothing reserves the
                # derived id space and ids arrive from the cloud, the web API,
                # importers and research harnesses. Overwriting it would replace the
                # user's memory with a speaker projection, hide it, and let a later
                # redaction of the parent delete it with no way back. It keeps the
                # id; per-speaker recall for that speaker falls back to the parent's
                # own row, which still contains the turns.
                #
                # The same conjunctive test the one-time migration uses tells them
                # apart, and it is PROVABLE here in a way it is not in general: this
                # parent is in hand, deriving this exact id, at the one moment it is
                # written. Only MARK, never purge, and only for this parent's ids —
                # so this is not the perpetual reclassification that was removed.
                existing = self._conn.execute(
                    "SELECT is_closet, content, related_to, created_at, updated_at FROM memories WHERE id = ?",
                    (cid,),
                ).fetchone()
                if existing is not None and existing["is_closet"] == CLOSET_ADOPTED_UNVERIFIED:
                    # Adopted on inference and deliberately not rewritten. It is
                    # already treated as a copy everywhere else — hidden, and cleared
                    # when this parent is redacted — so the only thing to do here is
                    # leave it alone. Only the explicit repair converts it.
                    logger.debug("not deriving %s: an adopted copy holds that id, unverified", cid)
                    continue
                if existing is not None and not existing["is_closet"]:
                    tier, _p, _raw, _enr, _meta = classify_closet_row(
                        self._conn,
                        cid,
                        content=existing["content"],
                        related_raw=existing["related_to"],
                        created_at=existing["created_at"],
                    )
                    if tier == TIER_PROVEN:
                        # Its text IS what this parent derives, so writing the copy
                        # over it changes nothing but the marker. It came down from
                        # the cloud, so the cloud holds it too and owes a delete,
                        # announced with the timestamp it arrived carrying.
                        logger.debug("adopting %s: text matches what %s derives", cid, memory.id)
                        record_legacy_closet_ids(self._conn, [(cid, existing["updated_at"])])
                    elif tier == TIER_LIKELY:
                        # Strong, but the text differs, so it could be a curated
                        # split rather than a leak. Mark it — that is what hides it
                        # and makes the parent's redaction reach it — and keep the
                        # text exactly as it is. Not announced: an announcement
                        # deletes the cloud's row everywhere, which no inference
                        # earns. Pre-image kept so even the marker is reversible.
                        logger.debug("adopting %s as a likely copy of %s; text kept", cid, memory.id)
                        back_up_migrated_rows(self._conn, [cid], "adopted")
                        self._conn.execute(
                            f"UPDATE memories SET {MARKER_COLUMN} = ? WHERE id = ?",
                            (CLOSET_ADOPTED_UNVERIFIED, cid),
                        )
                        continue
                    else:
                        logger.debug("not deriving %s: a real memory already holds that id", cid)
                        continue
                rebuilt.add(cid)
                closet_source = Source(
                    type=memory.source.type,
                    session_id=memory.source.session_id,
                    timestamp=memory.source.timestamp,
                )
                self._insert_memory(
                    cid,
                    closet_raw,
                    closet_enriched,
                    memory.memory_type,
                    memory.project,
                    closet_source,
                    [memory.id],
                    created_at,  # the parent's kept creation time, which the copies share
                    memory.updated_at,
                    memory.confidence,
                    memory.expires_at,
                    # THE provenance marker. Set here, at the only place this engine
                    # derives a copy, and read everywhere else instead of guessing
                    # from ids. Always the DERIVED state: an adopted-unverified row
                    # never reaches this line, because synthesis skips its slug.
                    is_closet=CLOSET_DERIVED,
                    embedding=closet_embedding,
                )

            # A speaker this write removed leaves a closet id that existed a moment
            # ago and does not now. That is a redaction — an edit that drops a
            # speaker, or replaces the turns with prose — so the deletion is recorded
            # content-free and push carries it to any cloud copy.
            record_closet_tombstones(
                self._conn,
                [cid for cid in previous_closets if cid not in rebuilt],
                when=remote_event_ts,
                applying_remote_deletion=remote_event_ts is not None,
            )
            # Every id just (re)created is live derived data again, so a record
            # saying it was deleted earlier is stale. A restore regenerates copies,
            # and the old record would then win the freshness comparison against a
            # cloud row written after that deletion but before this one — letting a
            # leaked copy through. Any announcement owed for the id survives: a
            # restore does not un-leak what an older client already pushed.
            clear_closet_deletion_records(self._conn, sorted(rebuilt))

        return memory.id

    # --- retrieval ---------------------------------------------------------

    def retrieve(self, query: str, filters: Filters | None = None, limit: int = 10) -> list[ScoredMemory]:
        candidates = self._hybrid_retrieve(query, filters, k=FIRST_STAGE_K)
        if not candidates:
            return []

        scores = self._rerank(query, [c.memory.content for c in candidates])

        reranked = [ScoredMemory(memory=c.memory, score=float(s)) for c, s in zip(candidates, scores)]
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:limit]

    def _hybrid_retrieve(self, query: str, filters: Filters | None, k: int) -> list[ScoredMemory]:
        fts_ranks: dict[str, int] = {}
        fts_query = _tokenize_for_fts(query)
        if fts_query:
            try:
                fts_rows = self._conn.execute(
                    """SELECT m.*, rank FROM memory_fts fts
                       JOIN memories m ON fts.id = m.id
                       WHERE memory_fts MATCH ?
                       ORDER BY rank
                       LIMIT ?""",
                    (fts_query, k * 5),
                ).fetchall()
                for rank_pos, row in enumerate(fts_rows):
                    mem = self._row_to_memory_enriched(row)
                    if self._passes_filters(mem, filters):
                        fts_ranks[row["id"]] = rank_pos
            except Exception:
                pass

        query_emb = self._embed(query)
        # Embedding-channel filter on engine fingerprint. Rows from another
        # engine's model are excluded from RRF here and only contribute via
        # FTS5 above until re-embedded.
        rows = self._conn.execute(
            "SELECT m.*, e.embedding FROM memories m JOIN memory_embeddings e ON m.id = e.id WHERE e.model_id = ?",
            (self.model_id,),
        ).fetchall()

        emb_scored: list[tuple[str, float, sqlite3.Row]] = []
        for row in rows:
            mem = self._row_to_memory_enriched(row)
            if not self._passes_filters(mem, filters):
                continue
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
            score = float(np.dot(query_emb, emb))
            emb_scored.append((row["id"], score, row))

        emb_scored.sort(key=lambda x: x[1], reverse=True)
        emb_ranks: dict[str, int] = {mid: rank for rank, (mid, _, _) in enumerate(emb_scored)}

        all_ids = set(fts_ranks.keys()) | set(emb_ranks.keys())
        max_rank = len(all_ids) + 1

        rrf_scores: dict[str, float] = {}
        for mid in all_ids:
            fts_r = fts_ranks.get(mid, max_rank)
            emb_r = emb_ranks.get(mid, max_rank)
            rrf_scores[mid] = 1.0 / (RRF_K + fts_r) + 1.0 / (RRF_K + emb_r)

        row_map = {row["id"]: row for _, _, row in emb_scored}
        if fts_query:
            for mid in fts_ranks:
                if mid not in row_map:
                    r = self._conn.execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone()
                    if r:
                        row_map[mid] = r

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]
        results = []
        for mid in sorted_ids:
            row = row_map.get(mid)
            if row:
                mem = self._row_to_memory_enriched(row)
                results.append(ScoredMemory(memory=mem, score=rrf_scores[mid]))
        return results

    # --- single-item accessors / lifecycle --------------------------------

    def get(self, memory_id: str) -> Memory | None:
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def note_leaked_cloud_copy(self, memory: Memory) -> bool:
        """Queue a cleanup announcement for an incoming row that is a leaked copy.

        Under the write transaction, which is committed for BOTH outcomes: this
        is called from sync's pull outside any write of ours, so an uncommitted
        insert would be invisible to the TombstoneStore's own connection and lost
        when this one closes. Both branches write — one queues or refreshes the
        announcement, the other cancels it — so committing only the queueing one
        would leave the write lock held on the cancel path and block every other
        connection to the store. (Called from inside another write transaction on
        this thread it commits nothing, leaving that to the outer owner, which is
        what makes the whole sequence one unit; sync's pull is not such a
        caller.)
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
        """Whether ``memory_id`` names one of this engine's own derived copies."""
        return is_marked_closet_row(self._conn, memory_id)

    def is_proven_copy_row(self, memory_id: str) -> bool:
        """Whether an UNMARKED row at ``memory_id`` is, on full proof, a copy of its live parent."""
        return is_proven_unmarked_copy(self._conn, memory_id)

    def delete(self, memory_id: str, *, remote_event_ts: datetime | None = None) -> bool:
        # One transaction: a failure between the copies and the parent would
        # otherwise commit half a redaction — or, worse, leave it uncommitted on
        # an open transaction for the next write to commit.
        with self._lock, write_txn(self._conn):
            # Copies FIRST: an unmarked one is graded against this parent's text,
            # so the parent row has to still be here when that happens.
            delete_marked_closets(self._conn, memory_id, has_embeddings=True, event_ts=remote_event_ts)
            cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._conn.execute("DELETE FROM memory_embeddings WHERE id = ?", (memory_id,))
            return cursor.rowcount > 0

    def list_all(self, filters: Filters | None = None, limit: int = 50) -> list[Memory]:
        # Closets are derived data, like embeddings: excluded here so they never
        # reach ``poppy list`` or sync push (push iterates list_all). Keyed on the
        # provenance marker, so a real memory whose id merely contains
        # ``_closet_`` is never hidden. retrieve()/RRF is untouched — it queries
        # memory_fts and memory_embeddings directly — so recall of a single
        # speaker's turns still works.
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
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def purge_expired(self) -> int:
        """Hard-delete memories whose expires_at is in the past. Returns rowcount.

        Also drops the cascade of synthetic ``_closet_<speaker>`` rows for any
        expired parent — closets share lifetime with their parent by construction.
        """
        now = datetime.now(timezone.utc)
        # The write lock is claimed BEFORE the rows are read, and held to the
        # commit. The decision is made in Python now, so the SELECT and the
        # DELETEs are two statements: without the lock, a `poppy edit --ttl` that
        # pushed a memory's expiry out in between would be overwritten by a delete
        # whose evidence is the expiry it just replaced. Under BEGIN IMMEDIATE that
        # writer waits on the busy timeout instead. Same helper seed's purge uses.
        with self._lock, write_txn(self._conn):
            # Expiry is decided by PARSING the stored stamp, not by comparing it as
            # text against ``now``. Writes normalise to UTC, but a row that reached
            # the store another way can carry any offset, and as text
            # ``2026-07-01T13:00:00-12:00`` (an hour in the future) sorts below
            # noon UTC — which hard-deleted live memories on the spot.
            expired = [
                (row[0], bool(row[1]))
                for row in self._conn.execute(
                    f"SELECT id, {NOT_CLOSET_SQL}, expires_at FROM memories WHERE expires_at IS NOT NULL"
                ).fetchall()
                if expiry_passed(row[2], now)
            ]
            # Parents only — closets are counted as the cascade they are, not as
            # memories the user lost.
            parents = [mid for mid, is_memory in expired if is_memory]
            for mid in parents:
                self._conn.execute("DELETE FROM memory_embeddings WHERE id = ?", (mid,))
                # No closet tombstones: expiry is not a redaction. The cloud copy
                # carries the same expires_at and ages out on its own.
                delete_marked_closets(self._conn, mid, has_embeddings=True, tombstone=False)
            # Deleted BY ID, from the list the parse decided — a second text
            # comparison here would remove exactly the rows the parse spared. Any
            # closet still standing outlived a parent whose own expiry moved; it
            # shares the parent's lifetime by construction, so it goes too.
            for batch in chunked([mid for mid, _ in expired]):
                placeholders = ",".join("?" * len(batch))
                self._conn.execute(f"DELETE FROM memory_embeddings WHERE id IN ({placeholders})", batch)
                self._conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", batch)
            return len(parents)

    def consolidate(self) -> ConsolidationResult:
        return ConsolidationResult(merged=0, removed=0, updated=0)

    def stats(self) -> EngineStats:
        # Memories, not rows: the per-speaker copies are excluded from list_all,
        # so counting them here would make `poppy doctor` and the dashboard
        # disagree with what the user can actually see.
        count = self._conn.execute(f"SELECT COUNT(*) FROM memories WHERE {NOT_CLOSET_SQL}").fetchone()[0]
        storage = self._db_path.stat().st_size if self._db_path.exists() else 0
        return EngineStats(
            memory_count=count,
            storage_bytes=storage,
            engine_name=self._engine_name,
            engine_version=self._engine_version,
        )
