"""Sidecar tombstone storage for the UI.

Soft-delete promise from the marketing copy: "Tombstoned — you can restore for 7
days." We honor it without touching the immutable RetrievalEngine ABC or Memory
dataclass: tombstones live in a separate SQLite table in the same DB file,
managed entirely by the UI layer. On soft-delete the row is snapshotted into
`ui_tombstones` and removed from the engine. On restore it's re-ingested.
Tombstones older than the TTL are purged on UI startup.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from poppy.db import connect as connect_db
from poppy.models import Memory, Source

TTL_DAYS = 7

SCHEMA = """
CREATE TABLE IF NOT EXISTS ui_tombstones (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    project TEXT,
    source_type TEXT NOT NULL,
    source_session_id TEXT,
    source_timestamp TEXT NOT NULL,
    confidence REAL NOT NULL,
    related_to TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    tombstoned_at TEXT NOT NULL,
    superseded_by TEXT
);
"""


def _migrate_superseded_by(conn: sqlite3.Connection) -> None:
    """Idempotently add superseded_by to ui_tombstones for DBs created before lifecycle work."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(ui_tombstones)").fetchall()}
    if "superseded_by" not in cols:
        conn.execute("ALTER TABLE ui_tombstones ADD COLUMN superseded_by TEXT")


@dataclass
class Tombstone:
    memory: Memory
    tombstoned_at: datetime
    superseded_by: str | None = None

    @property
    def expires_at(self) -> datetime:
        return self.tombstoned_at + timedelta(days=TTL_DAYS)


class TombstoneStore:
    """Manages the `ui_tombstones` sidecar table in the Poppy SQLite DB."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        self._conn = connect_db(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        _migrate_superseded_by(self._conn)

    def add(self, memory: Memory, *, superseded_by: str | None = None) -> Tombstone:
        now = datetime.now(timezone.utc)
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO ui_tombstones (
                    id, content, memory_type, project, source_type, source_session_id,
                    source_timestamp, confidence, related_to, created_at, updated_at, tombstoned_at,
                    superseded_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    now.isoformat(),
                    superseded_by,
                ),
            )
            self._conn.commit()
        return Tombstone(memory=memory, tombstoned_at=now, superseded_by=superseded_by)

    def remove(self, memory_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM ui_tombstones WHERE id = ?", (memory_id,))
            self._conn.commit()

    def get(self, memory_id: str) -> Tombstone | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM ui_tombstones WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_tombstone(row)

    def list_all(self) -> list[Tombstone]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM ui_tombstones ORDER BY tombstoned_at DESC").fetchall()
        return [self._row_to_tombstone(r) for r in rows]

    def purge_expired(self) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=TTL_DAYS)).isoformat()
        with self._lock:
            cursor = self._conn.execute("DELETE FROM ui_tombstones WHERE tombstoned_at < ?", (cutoff,))
            self._conn.commit()
            return cursor.rowcount

    @staticmethod
    def _row_to_tombstone(row: sqlite3.Row) -> Tombstone:
        memory = Memory(
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
        )
        keys = row.keys()
        return Tombstone(
            memory=memory,
            tombstoned_at=datetime.fromisoformat(row["tombstoned_at"]),
            superseded_by=row["superseded_by"] if "superseded_by" in keys else None,
        )
