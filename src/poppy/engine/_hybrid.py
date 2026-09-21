"""Shared hybrid retrieval over full memories, independent of the model runtime.

Subclasses supply embeddings and reranking. This module owns storage, full-text
indexing, and the two-stage retrieval pipeline.
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

from poppy.db import apply_row_factory, rollback_and_close, write_gate, write_txn
from poppy.db import connect as connect_db
from poppy.engine._closet_marker import (
    clear_marked_copies,
    clear_retired_records,
    ensure_closet_side_tables,
)
from poppy.engine._legacy_copies import mark_legacy_copies_for_cleanup
from poppy.engine._timestamps import (
    _columns,
    _table_exists,
    chunked,
    expiry_passed,
    normalise_stored_timestamps,
    utc_iso,
)
from poppy.engine.interface import ConsolidationResult, EngineStats, RetrievalEngine
from poppy.models import Filters, Memory, ScoredMemory, Source

logger = logging.getLogger(__name__)

FIRST_STAGE_K = 100
RRF_K = 60

# Rows an older release marked as derived per-speaker copies. They are kept
# out of listings, counts and recall, and removed when the memory they were
# copied from is redacted or deleted.
_NOT_MARKED_SQL = "COALESCE(is_closet, 0) = 0"

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
    -- Retained for cleanup of stores written by older releases.
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


def _speaker_of(turn: dict) -> str | None:
    speaker = turn.get("speaker")
    return speaker if isinstance(speaker, str) and speaker else None


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
        speaker = _speaker_of(turn) or "Unknown"
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
        speaker = _speaker_of(turn) or "Unknown"
        dia_id = turn.get("dia_id", "")
        text = turn.get("text", "")
        if dia_id and text:
            lines.append(f"{dia_id} {speaker}: {text}")
        elif text:
            lines.append(f"{speaker}: {text}")

    return "\n".join(lines) if lines else content


class HybridEngine(RetrievalEngine):
    """FTS5 and cosine retrieval followed by cross-encoder reranking."""

    # Subclasses override. ``model_id`` tags every BLOB this engine writes so a
    # later swap to a backend with different vectors never mixes incompatible
    # vector spaces in the shared memory_embeddings table.
    model_id: str | None = None
    # This engine accepts ``remote_event_ts`` on ingest and delete. Sync checks
    # the flag rather than the signature, so an engine that predates it (or a
    # third-party one) is called the old way instead of raising.
    accepts_remote_event_ts = True
    _engine_name = "hybrid"
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
            # Retained while older caller surfaces still read these tables.
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
            from poppy.sync.state import remove_derived_rows

            # Classification and removal share one gate and transaction, so an
            # interrupted upgrade cannot leave classified copies behind.
            with write_gate(db_path.parent), write_txn(self._conn):
                mark_legacy_copies_for_cleanup(self._conn, had_bloom_schema=had_bloom_schema)
                remove_derived_rows(self._conn, db_path.parent, gate_held=True)
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
                expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            ),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO memory_embeddings (id, embedding, model_id) VALUES (?, ?, ?)",
            (memory_id, emb_blob, self.model_id),
        )

    def ingest(self, memory: Memory, *, remote_event_ts: datetime | None = None) -> str:
        """Store one memory with its full-content index and embedding."""
        enriched = _enrich_full_content(memory.content, memory.source.timestamp.isoformat())
        # Model loading may fail. Do it before taking the database write lock.
        embedding = self._embed(enriched)
        with self._lock, write_txn(self._conn):
            prior = self._conn.execute("SELECT content, created_at FROM memories WHERE id = ?", (memory.id,)).fetchone()
            created_at = memory.created_at
            if prior is not None and prior["content"] == memory.content:
                try:
                    created_at = datetime.fromisoformat(prior["created_at"])
                except (TypeError, ValueError):
                    pass
            elif prior is not None:
                # The text this memory held is being replaced, so a copy of that
                # text left over from an older release is being redacted too. A
                # write that only changes metadata replays the same body and
                # leaves them alone.
                clear_marked_copies(self._conn, memory.id, has_embeddings=True)
            self._insert_memory(
                memory.id,
                memory.content,
                enriched,
                memory.memory_type,
                memory.project,
                memory.source,
                memory.related_to,
                created_at,
                memory.updated_at,
                memory.confidence,
                memory.expires_at,
                embedding=embedding,
            )
            clear_retired_records(self._conn, memory.id)
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
                       WHERE memory_fts MATCH ? AND COALESCE(m.is_closet, 0) = 0
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
            "SELECT m.*, e.embedding FROM memories m JOIN memory_embeddings e ON m.id = e.id "
            "WHERE e.model_id = ? AND COALESCE(m.is_closet, 0) = 0",
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

    def is_proven_copy_row(self, memory_id: str) -> bool:
        """Compatibility check for lifecycle callers inspecting old copies."""
        from poppy.engine._closet_marker import is_proven_unmarked_copy

        with self._lock:
            return is_proven_unmarked_copy(self._conn, memory_id)

    def delete(self, memory_id: str, *, remote_event_ts: datetime | None = None) -> bool:
        # One transaction: a failure between the copies and the memory itself
        # would otherwise commit half a redaction.
        with self._lock, write_txn(self._conn):
            clear_marked_copies(self._conn, memory_id, has_embeddings=True)
            cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._conn.execute("DELETE FROM memory_embeddings WHERE id = ?", (memory_id,))
            return cursor.rowcount > 0

    def list_all(self, filters: Filters | None = None, limit: int = 50) -> list[Memory]:
        # Rows an older release left marked as per-speaker copies are excluded,
        # here and from stats and recall, so nothing shows text a memory in this
        # store already holds and push (which reads this) never sends one. Keyed
        # on the marker, so a real memory whose id merely looks like a copy's is
        # never hidden.
        query = f"SELECT * FROM memories WHERE {_NOT_MARKED_SQL}"
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
        """Hard-delete expired memories and their embeddings in one transaction."""
        now = datetime.now(timezone.utc)
        with self._lock, write_txn(self._conn):
            expired = [
                (row[0], bool(row[1]))
                for row in self._conn.execute(
                    f"SELECT id, {_NOT_MARKED_SQL}, expires_at FROM memories WHERE expires_at IS NOT NULL"
                ).fetchall()
                if expiry_passed(row[2], now)
            ]
            # Counted as memories the user lost: a leftover copy is not one, and
            # it goes with the memory it was copied from rather than on its own.
            # No deletion record for these: expiry is not a redaction.
            memories = [mid for mid, is_memory in expired if is_memory]
            for mid in memories:
                clear_marked_copies(self._conn, mid, has_embeddings=True, tombstone=False)
            for batch in chunked([mid for mid, _ in expired]):
                placeholders = ",".join("?" * len(batch))
                self._conn.execute(f"DELETE FROM memory_embeddings WHERE id IN ({placeholders})", batch)
                self._conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", batch)
            return len(memories)

    def consolidate(self) -> ConsolidationResult:
        return ConsolidationResult(merged=0, removed=0, updated=0)

    def stats(self) -> EngineStats:
        # Memories, not rows: leftover copies are hidden from list_all, so
        # counting them here would disagree with what the user can see.
        count = self._conn.execute(f"SELECT COUNT(*) FROM memories WHERE {_NOT_MARKED_SQL}").fetchone()[0]
        storage = self._db_path.stat().st_size if self._db_path.exists() else 0
        return EngineStats(
            memory_count=count,
            storage_bytes=storage,
            engine_name=self._engine_name,
            engine_version=self._engine_version,
        )
