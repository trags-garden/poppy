import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from poppy.db import connect as connect_db
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


def _migrate_enriched_content(conn: sqlite3.Connection) -> None:
    """Idempotently add ``enriched_content`` and rewire FTS triggers to point at it.

    The ``bloom`` engine indexes a derived enrichment of each memory (a
    preamble + per-turn formatting) instead of the raw content. It expects
    ``memories.enriched_content`` to exist and the FTS triggers to write that
    column into ``memory_fts``. DBs created by ``seed``/``sprout`` predate
    this column.

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
    cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
    added_column = False
    if "enriched_content" not in cols:
        conn.execute("ALTER TABLE memories ADD COLUMN enriched_content TEXT")
        added_column = True
    # Backfill is cheap; run it whenever any row still has NULL so a partial
    # prior migration completes itself.
    conn.execute("UPDATE memories SET enriched_content = content WHERE enriched_content IS NULL")
    if added_column:
        # Triggers are rebuilt only when we just added the column — otherwise
        # they were either already rewired by a prior bloom open, or
        # they're the bloom form on a fresh DB created by it directly.
        # Either way no rewire is needed.
        for name in ("memory_ai", "memory_ad", "memory_au"):
            conn.execute(f"DROP TRIGGER IF EXISTS {name}")
        conn.executescript(
            """
            CREATE TRIGGER memory_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memory_fts(id, content) VALUES (new.id, new.enriched_content);
            END;
            CREATE TRIGGER memory_ad AFTER DELETE ON memories BEGIN
                DELETE FROM memory_fts WHERE id = old.id;
            END;
            CREATE TRIGGER memory_au AFTER UPDATE ON memories BEGIN
                DELETE FROM memory_fts WHERE id = old.id;
                INSERT INTO memory_fts(id, content) VALUES (new.id, new.enriched_content);
            END;
            """
        )
    conn.commit()


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

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        self._conn = connect_db(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        _migrate_expires_at(self._conn)

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
        expires_iso = memory.expires_at.isoformat() if memory.expires_at else None
        with self._lock:
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
            self._conn.commit()
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
        with self._lock:
            cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
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
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def purge_expired(self) -> int:
        """Hard-delete memories whose expires_at is in the past. Returns rowcount."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
            self._conn.commit()
            return cursor.rowcount

    def consolidate(self) -> ConsolidationResult:
        return ConsolidationResult(merged=0, removed=0, updated=0)

    def stats(self) -> EngineStats:
        with self._lock:
            count = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        storage = self._db_path.stat().st_size if self._db_path.exists() else 0
        return EngineStats(
            memory_count=count,
            storage_bytes=storage,
            engine_name="seed",
            engine_version="0.1.0",
        )
