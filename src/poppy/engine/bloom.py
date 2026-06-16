"""Bloom — local champion retrieval engine.

Two-stage retrieval (hybrid FTS5+embeddings → cross-encoder rerank) with a
per-speaker content expansion at ingest. For each multi-speaker session, the
engine stores a synthetic memory containing only one speaker's turns alongside
the full session, so queries like "When did Alice go to X?" can score the
Alice-only memory higher than the joint session.

    main memory:    id=D1,               content=[alice turns + bob turns]
    alice closet:   id=D1_closet_alice,  content=[alice turns only]
    bob closet:     id=D1_closet_bob,    content=[bob turns only]

All variants compete equally in retrieval; cross-speaker queries still tend to
prefer the full session because only that one contains both speakers' turns.

Models (all local, no API):
  - Bi-encoder: BAAI/bge-small-en-v1.5 (384-dim, ~130MB)
  - Cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2 (6-layer, ~80MB)
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

BI_ENCODER = "BAAI/bge-small-en-v1.5"
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
    expires_at TEXT
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

    speakers = []
    seen: set[str] = set()
    for turn in turns:
        speaker = turn.get("speaker", "Unknown")
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
        speaker = turn.get("speaker", "Unknown")
        dia_id = turn.get("dia_id", "")
        text = turn.get("text", "")
        if dia_id and text:
            lines.append(f"{dia_id} {speaker}: {text}")
        elif text:
            lines.append(f"{speaker}: {text}")

    return "\n".join(lines) if lines else content


def _enrich_closet_content(
    content: str, session_timestamp: str | None, target_speaker: str, other_speakers: list[str]
) -> tuple[str, str]:
    """Return (closet_raw_json, closet_enriched_text) for one speaker's turns.

    Returns ('', '') if the speaker has no turns in this session.
    """
    try:
        turns = json.loads(content)
        if not isinstance(turns, list):
            return "", ""
    except (json.JSONDecodeError, TypeError):
        return "", ""

    speaker_turns = [t for t in turns if isinstance(t, dict) and t.get("speaker") == target_speaker]
    if not speaker_turns:
        return "", ""

    date_str = ""
    if session_timestamp:
        try:
            dt = datetime.fromisoformat(session_timestamp)
            date_str = dt.strftime("%B %d, %Y at %I:%M %p")
        except ValueError:
            date_str = session_timestamp

    # Preamble names both the speaker and the counterpart(s) so cross-speaker
    # context is not lost entirely — the closet is Alice's contributions, but
    # names Bob so queries mentioning Bob can still match.
    if other_speakers:
        if len(other_speakers) == 1:
            partner = other_speakers[0]
        else:
            partner = ", ".join(other_speakers[:-1]) + f", and {other_speakers[-1]}"
        if date_str:
            preamble = f"{target_speaker}'s contributions to a conversation with {partner} on {date_str}."
        else:
            preamble = f"{target_speaker}'s contributions to a conversation with {partner}."
    else:
        if date_str:
            preamble = f"{target_speaker}'s contributions on {date_str}."
        else:
            preamble = f"{target_speaker}'s contributions."

    lines = [preamble, ""]
    for turn in speaker_turns:
        dia_id = turn.get("dia_id", "")
        text = turn.get("text", "")
        if dia_id and text:
            lines.append(f"{dia_id} {target_speaker}: {text}")
        elif text:
            lines.append(f"{target_speaker}: {text}")

    return json.dumps(speaker_turns), "\n".join(lines)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "x"


class BloomEngine(RetrievalEngine):
    """Champion retrieval + per-speaker closet memories added at ingest."""

    # Stamps every BLOB this engine writes so a later switch to/from `sprout`
    # (which uses all-MiniLM-L6-v2) doesn't mix incompatible vector spaces in
    # the shared memory_embeddings table.
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
        from poppy.engine.seed import (
            _migrate_embedding_model_id,
            _migrate_enriched_content,
            _migrate_expires_at,
        )

        # Order matters: expires_at and enriched_content are column-level
        # schema upgrades that must complete before any read path runs. The
        # enriched_content migration also rewires the FTS triggers to point at
        # the new column; bloom's INSERT/UPDATE paths rely on that.
        _migrate_expires_at(self._conn)
        _migrate_enriched_content(self._conn)
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
    ) -> None:
        emb_blob = self._embed(enriched).tobytes()
        expires_iso = expires_at.isoformat() if expires_at else None
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
                source.timestamp.isoformat(),
                confidence,
                json.dumps(related_to),
                created_at.isoformat(),
                updated_at.isoformat(),
                expires_iso,
            ),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO memory_embeddings (id, embedding, model_id) VALUES (?, ?, ?)",
            (memory_id, emb_blob, self.model_id),
        )

    def ingest(self, memory: Memory) -> str:
        enriched = _enrich_full_content(
            memory.content,
            session_timestamp=memory.source.timestamp.isoformat(),
        )
        # Clean up any prior synthetic closets for this memory (idempotent re-ingest)
        self._conn.execute(
            "DELETE FROM memories WHERE id LIKE ?",
            (f"{memory.id}_closet_%",),
        )
        self._conn.execute(
            "DELETE FROM memory_embeddings WHERE id LIKE ?",
            (f"{memory.id}_closet_%",),
        )

        self._insert_memory(
            memory.id,
            memory.content,
            enriched,
            memory.memory_type,
            memory.project,
            memory.source,
            memory.related_to,
            memory.created_at,
            memory.updated_at,
            memory.confidence,
            memory.expires_at,
        )

        # Synthesize a closet per speaker if the content has the expected turn shape.
        try:
            turns = json.loads(memory.content)
        except (json.JSONDecodeError, TypeError):
            turns = []

        if isinstance(turns, list) and turns:
            speakers: list[str] = []
            seen: set[str] = set()
            for t in turns:
                if not isinstance(t, dict):
                    continue
                sp = t.get("speaker")
                if sp and sp not in seen:
                    speakers.append(sp)
                    seen.add(sp)

            # Only build closets when there is actually more than one speaker —
            # a single-speaker session would produce an identical closet.
            if len(speakers) >= 2:
                for sp in speakers:
                    others = [o for o in speakers if o != sp]
                    closet_raw, closet_enriched = _enrich_closet_content(
                        memory.content,
                        memory.source.timestamp.isoformat(),
                        sp,
                        others,
                    )
                    if not closet_raw:
                        continue
                    closet_id = f"{memory.id}_closet_{_slug(sp)}"
                    closet_source = Source(
                        type=memory.source.type,
                        session_id=memory.source.session_id,
                        timestamp=memory.source.timestamp,
                    )
                    self._insert_memory(
                        closet_id,
                        closet_raw,
                        closet_enriched,
                        memory.memory_type,
                        memory.project,
                        closet_source,
                        [memory.id],
                        memory.created_at,
                        memory.updated_at,
                        memory.confidence,
                        memory.expires_at,
                    )

        self._conn.commit()
        return memory.id

    def retrieve(self, query: str, filters: Filters | None = None, limit: int = 10) -> list[ScoredMemory]:
        candidates = self._hybrid_retrieve(query, filters, k=FIRST_STAGE_K)
        if not candidates:
            return []

        pairs = [(query, c.memory.content) for c in candidates]
        scores = self._cross_encoder.predict(pairs)

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

    def get(self, memory_id: str) -> Memory | None:
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def delete(self, memory_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.execute("DELETE FROM memory_embeddings WHERE id = ?", (memory_id,))
        # Also remove any closets tied to this parent memory.
        self._conn.execute("DELETE FROM memories WHERE id LIKE ?", (f"{memory_id}_closet_%",))
        self._conn.execute("DELETE FROM memory_embeddings WHERE id LIKE ?", (f"{memory_id}_closet_%",))
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
        """Hard-delete memories whose expires_at is in the past. Returns rowcount.

        Also drops the cascade of synthetic ``_closet_<speaker>`` rows for any
        expired parent — closets share lifetime with their parent by construction.
        """
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            "SELECT id FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        )
        ids = [row["id"] for row in cursor.fetchall()]
        for mid in ids:
            self._conn.execute("DELETE FROM memory_embeddings WHERE id = ?", (mid,))
            self._conn.execute("DELETE FROM memory_embeddings WHERE id LIKE ?", (f"{mid}_closet_%",))
            self._conn.execute("DELETE FROM memories WHERE id LIKE ?", (f"{mid}_closet_%",))
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
            engine_name="bloom",
            engine_version="1.0.0",
        )
