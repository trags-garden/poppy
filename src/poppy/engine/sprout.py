"""Sprout — mid-tier retrieval engine.

Architecture:
  Stage 1: Hybrid retrieval (FTS5 BM25 + embedding cosine + RRF fusion) → K=100
  Stage 2: Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) → top limit

Models (all local, no API):
  - Bi-encoder: all-MiniLM-L6-v2 (384-dim, ~80MB)
  - Cross-encoder: ms-marco-MiniLM-L-6-v2 (6-layer, ~80MB)

Lighter than bloom: same cross-encoder but a smaller bi-encoder and no per-speaker
content expansion at ingest. Useful when memory churn matters more than the last
few points of recall.
"""

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from poppy.db import connect as connect_db
from poppy.engine._st_loader import announce_first_run_download, load_st_model
from poppy.engine.interface import ConsolidationResult, EngineStats, RetrievalEngine
from poppy.models import Filters, Memory, ScoredMemory, Source

BI_ENCODER = "all-MiniLM-L6-v2"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
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

CREATE TABLE IF NOT EXISTS memory_embeddings (
    id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL
);
"""


def _tokenize_for_fts(query: str) -> str:
    """Convert a natural language query to FTS5 OR query with stopword removal."""
    words = re.findall(r"[a-zA-Z0-9]+", query.lower())
    terms = [w for w in words if w not in STOPWORDS and len(w) > 1]
    if not terms:
        return ""
    return " OR ".join(f'"{t}"' for t in terms)


class SproutEngine(RetrievalEngine):
    """Two-stage retrieval: Hybrid (FTS5+embeddings+RRF) → Cross-Encoder reranking."""

    # Tags every BLOB this engine writes into memory_embeddings.model_id so a
    # later engine swap doesn't silently mix vector spaces.
    model_id = BI_ENCODER

    def __init__(
        self,
        db_path: Path,
        bi_encoder: SentenceTransformer | None = None,
        cross_encoder: CrossEncoder | None = None,
    ) -> None:
        self._db_path = db_path
        self._conn = connect_db(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        from poppy.engine.seed import _migrate_embedding_model_id, _migrate_expires_at

        _migrate_expires_at(self._conn)
        _migrate_embedding_model_id(self._conn)
        # Go through the offline-safe loader, not the bare constructors.
        # Cached models load with local_files_only (no network HEAD); a cold
        # cache offline raises ModelUnavailableError instead of a traceback.
        if bi_encoder is None or cross_encoder is None:
            announce_first_run_download((BI_ENCODER, CROSS_ENCODER))
        self._bi_encoder = bi_encoder or load_st_model("bi", BI_ENCODER)
        self._cross_encoder = cross_encoder or load_st_model("cross", CROSS_ENCODER)

    def _embed(self, text: str) -> np.ndarray:
        return self._bi_encoder.encode(text, normalize_embeddings=True)

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

    def ingest(self, memory: Memory) -> str:
        emb = self._embed(memory.content)
        emb_blob = emb.tobytes()
        expires_iso = memory.expires_at.isoformat() if memory.expires_at else None

        existing = self.get(memory.id)
        if existing is not None:
            self._conn.execute(
                """UPDATE memories SET content=?, memory_type=?, project=?, source_type=?,
                   source_session_id=?, source_timestamp=?, confidence=?, related_to=?, updated_at=?,
                   expires_at=?
                   WHERE id=?""",
                (
                    memory.content,
                    memory.memory_type,
                    memory.project,
                    memory.source.type,
                    memory.source.session_id,
                    memory.source.timestamp.isoformat(),
                    memory.confidence,
                    json.dumps(memory.related_to),
                    memory.updated_at.isoformat(),
                    expires_iso,
                    memory.id,
                ),
            )
            self._conn.execute(
                "UPDATE memory_embeddings SET embedding=?, model_id=? WHERE id=?",
                (emb_blob, self.model_id, memory.id),
            )
        else:
            self._conn.execute(
                """INSERT INTO memories (id, content, memory_type, project, source_type,
                   source_session_id, source_timestamp, confidence, related_to, created_at, updated_at,
                   expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    memory.id,
                    memory.content,
                    memory.memory_type,
                    memory.project,
                    memory.source.type,
                    memory.source.session_id,
                    memory.source.timestamp.isoformat(),
                    memory.confidence,
                    json.dumps(memory.related_to),
                    memory.created_at.isoformat(),
                    memory.updated_at.isoformat(),
                    expires_iso,
                ),
            )
            self._conn.execute(
                "INSERT INTO memory_embeddings (id, embedding, model_id) VALUES (?, ?, ?)",
                (memory.id, emb_blob, self.model_id),
            )
        self._conn.commit()
        return memory.id

    def retrieve(self, query: str, filters: Filters | None = None, limit: int = 10) -> list[ScoredMemory]:
        # Stage 1: Hybrid (FTS5 + embeddings + RRF)
        candidates = self._hybrid_retrieve(query, filters, k=FIRST_STAGE_K)
        if not candidates:
            return []

        # Stage 2: Cross-encoder reranking
        pairs = [(query, c.memory.content) for c in candidates]
        scores = self._cross_encoder.predict(pairs)

        reranked = [ScoredMemory(memory=c.memory, score=float(s)) for c, s in zip(candidates, scores)]
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:limit]

    def _hybrid_retrieve(self, query: str, filters: Filters | None, k: int) -> list[ScoredMemory]:
        """First stage: FTS5 BM25 + embedding cosine similarity + RRF fusion."""
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
                    mem = self._row_to_memory(row)
                    if self._passes_filters(mem, filters):
                        fts_ranks[row["id"]] = rank_pos
            except Exception:
                pass

        query_emb = self._embed(query)
        # Filter to embeddings produced by this engine's bi-encoder. Rows
        # written by a different engine (different model_id) or with NULL
        # model_id are skipped here so they can't poison RRF; they still
        # contribute via FTS5 above, so unmigrated rows degrade to FTS-only
        # quality rather than producing meaningless cosines.
        rows = self._conn.execute(
            "SELECT m.*, e.embedding FROM memories m JOIN memory_embeddings e ON m.id = e.id WHERE e.model_id = ?",
            (self.model_id,),
        ).fetchall()

        emb_scored: list[tuple[str, float, sqlite3.Row]] = []
        for row in rows:
            mem = self._row_to_memory(row)
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
                mem = self._row_to_memory(row)
                results.append(ScoredMemory(memory=mem, score=rrf_scores[mid]))
        return results

    def get(self, memory_id: str) -> Memory | None:
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def delete(self, memory_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.execute("DELETE FROM memory_embeddings WHERE id = ?", (memory_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def list_all(self, filters: Filters | None = None, limit: int = 50) -> list[Memory]:
        query = "SELECT * FROM memories WHERE 1=1"
        params: list = []
        if filters:
            if filters.project:
                query += " AND project = ?"
                params.append(filters.project)
            if filters.memory_type:
                query += " AND memory_type = ?"
                params.append(filters.memory_type)
            if filters.since:
                # created_at is stored as a '+00:00' ISO string, so the SQL
                # string comparison is only correct when the bound shares that
                # offset. Normalize aware datetimes to UTC; naive ones are
                # already treated as UTC by convention.
                since = filters.since
                if since.tzinfo is not None:
                    since = since.astimezone(timezone.utc)
                query += " AND created_at >= ?"
                params.append(since.isoformat())
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
        """Hard-delete memories whose expires_at is in the past. Returns rowcount."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            "SELECT id FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        )
        ids = [row["id"] for row in cursor.fetchall()]
        for mid in ids:
            self._conn.execute("DELETE FROM memory_embeddings WHERE id = ?", (mid,))
        deleted = self._conn.execute(
            "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        )
        self._conn.commit()
        return deleted.rowcount

    def consolidate(self) -> ConsolidationResult:
        return ConsolidationResult(merged=0, removed=0, updated=0)

    def stats(self) -> EngineStats:
        count = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        storage = self._db_path.stat().st_size if self._db_path.exists() else 0
        return EngineStats(
            memory_count=count,
            storage_bytes=storage,
            engine_name="sprout",
            engine_version="0.1.0",
        )
