"""Hold incoming legacy candidates privately until their parent can grade them."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from poppy.db import write_txn
from poppy.engine._legacy_copies import TIER_LIKELY, TIER_ORPHAN, TIER_PROVEN, classify_legacy_copy
from poppy.engine._timestamps import _table_exists, decode_text_columns, raw_text_columns, utc_iso
from poppy.models import Memory
from poppy.sync.serializer import deletion_time, wire_to_memory
from poppy.sync.state import record_local_deletion

DDL = """
CREATE TABLE IF NOT EXISTS sync_pending_copies (
    id TEXT PRIMARY KEY, row_json TEXT NOT NULL, remote_url TEXT NOT NULL, updated_at TEXT NOT NULL
)
"""


def grade_incoming(conn: sqlite3.Connection, memory: Memory) -> tuple[str, bool]:
    tier, parent = classify_legacy_copy(
        conn,
        memory.id,
        content=memory.content,
        related_raw=json.dumps(memory.related_to),
        created_at=memory.created_at.isoformat(),
    )
    # ORPHAN also describes a parent that has no matching projection. Only a
    # missing parent needs staging; a readable unrelated parent provides no proof.
    waiting = tier == TIER_ORPHAN and not conn.execute("SELECT 1 FROM memories WHERE id = ?", (parent,)).fetchone()
    return tier, waiting


def defer_copy(conn: sqlite3.Connection, row: dict, remote_url: str) -> None:
    with write_txn(conn):
        conn.execute(DDL)
        conn.execute(
            "INSERT INTO sync_pending_copies (id, row_json, remote_url, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET row_json = excluded.row_json, remote_url = excluded.remote_url, "
            "updated_at = excluded.updated_at WHERE excluded.updated_at >= sync_pending_copies.updated_at",
            (row["id"], json.dumps(row), remote_url, utc_iso(row["updated_at"])),
        )


def clear_pending(conn: sqlite3.Connection, memory_id: str, *, through: datetime | None = None) -> None:
    if _table_exists(conn, "sync_pending_copies"):
        if through is None:
            conn.execute("DELETE FROM sync_pending_copies WHERE id = ?", (memory_id,))
        else:
            # A write at this id only supersedes deferred versions at or below
            # its own time. A newer candidate still needs its parent's evidence.
            conn.execute(
                "DELETE FROM sync_pending_copies WHERE id = ? AND updated_at <= ?", (memory_id, utc_iso(through))
            )


def grade_pending(conn: sqlite3.Connection) -> list[tuple[dict, str]]:
    """Refuse copies atomically; return ordinary rows for sync's freshness checks.

    Store opening calls this without ingesting anything. An ordinary row waits
    for the next pull, which applies all normal deletion and conflict checks.
    Undecodable staging records stay private and cannot prevent store opening.
    """
    if not _table_exists(conn, "sync_pending_copies"):
        return []
    ready = []
    with write_txn(conn):
        columns = raw_text_columns("row_json", "remote_url")
        for stored in conn.execute(f"SELECT {columns} FROM sync_pending_copies").fetchall():
            try:
                payload, remote_url = decode_text_columns(tuple(stored))
                row = json.loads(payload)
                memory = wire_to_memory(row)
                tier, waiting = grade_incoming(conn, memory)
            except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
                continue
            if tier in (TIER_PROVEN, TIER_LIKELY):
                if tier == TIER_PROVEN:
                    record_local_deletion(conn, memory.id, deletion_time(row) or memory.updated_at)
                clear_pending(conn, memory.id)
            elif not waiting:
                ready.append((row, remote_url))
    return ready
