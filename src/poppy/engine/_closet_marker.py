"""Adapters for callers that still inspect legacy copy records, and the
cleanup that stops a marked copy outliving the memory it was copied from.

No copies are created here. Keep the adapters only until the CLI, lifecycle,
Trash and sync callers have moved to their final legacy-data interfaces; keep
:func:`clear_marked_copies` and the marker filters in the engines for as long as
stores written before this release are supported.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from poppy.engine._legacy_copies import (
    BACKUP_DDL as CLOSET_BACKUP_DDL,
)
from poppy.engine._legacy_copies import (
    SEPARATOR as CLOSET_SEPARATOR,
)
from poppy.engine._legacy_copies import (
    TIER_LIKELY,
    TIER_NONE,
    TIER_ORPHAN,
    TIER_PROVEN,
    back_up_inferred_copy,
    classify_legacy_copy,
    has_copy_provenance,
    legacy_copy_grades,
)
from poppy.engine._legacy_copies import (
    has_marker as has_marker,
)
from poppy.engine._timestamps import _columns, _table_exists
from poppy.engine._timestamps import chunked as chunked
from poppy.engine._timestamps import utc_iso as utc_iso

CLOSET_TOMBSTONE_TABLE = "closet_tombstones"
LEGACY_CLOSET_TABLE = "legacy_closet_ids"
IS_CLOSET_SQL = "COALESCE(is_closet, 0) >= 1"
NOT_CLOSET_SQL = "COALESCE(is_closet, 0) = 0"
CLOSET_TOMBSTONE_DDL = """
CREATE TABLE IF NOT EXISTS closet_tombstones (
    id TEXT PRIMARY KEY, tombstoned_at TEXT NOT NULL, is_local INTEGER NOT NULL DEFAULT 0
);
"""
LEGACY_CLOSET_DDL = """
CREATE TABLE IF NOT EXISTS legacy_closet_ids (
    id TEXT PRIMARY KEY, announce_pending INTEGER NOT NULL DEFAULT 1,
    announced_at TEXT, legacy_updated_at TEXT
);
"""


def ensure_closet_side_tables(conn: sqlite3.Connection) -> None:
    conn.execute(CLOSET_TOMBSTONE_DDL)
    conn.execute(CLOSET_BACKUP_DDL)
    conn.execute(LEGACY_CLOSET_DDL)
    if "is_local" not in _columns(conn, CLOSET_TOMBSTONE_TABLE):
        conn.execute(f"ALTER TABLE {CLOSET_TOMBSTONE_TABLE} ADD COLUMN is_local INTEGER NOT NULL DEFAULT 0")


def pending_legacy_announcements(conn: sqlite3.Connection) -> list[tuple[str, str | None]]:
    if not _table_exists(conn, LEGACY_CLOSET_TABLE):
        return []
    rows = conn.execute(
        f"SELECT id, legacy_updated_at FROM {LEGACY_CLOSET_TABLE} WHERE announce_pending = 1 ORDER BY id"
    )
    return [(row[0], row[1]) for row in rows]


def announced_copy_claim(conn: sqlite3.Connection, memory_id: str) -> str | None:
    if not _table_exists(conn, LEGACY_CLOSET_TABLE):
        return None
    row = conn.execute(f"SELECT legacy_updated_at FROM {LEGACY_CLOSET_TABLE} WHERE id = ?", (memory_id,)).fetchone()
    return row[0] if row is not None else None


def mark_legacy_announced(
    conn: sqlite3.Connection, memory_ids: list[str] | list[tuple[str, str | None]], *, when: datetime | None = None
) -> None:
    if not memory_ids or not _table_exists(conn, LEGACY_CLOSET_TABLE):
        return
    stamp = (when or datetime.now(timezone.utc)).isoformat()
    for entry in memory_ids:
        if isinstance(entry, tuple):
            memory_id, sent = entry
            conn.execute(
                f"UPDATE {LEGACY_CLOSET_TABLE} SET announce_pending = 0, announced_at = ? WHERE id = "
                f"? AND COALESCE(legacy_updated_at, '') = COALESCE(?, '')",
                (stamp, memory_id, utc_iso(sent)),
            )
        else:
            conn.execute(
                f"UPDATE {LEGACY_CLOSET_TABLE} SET announce_pending = 0, announced_at = ? WHERE id = ?", (stamp, entry)
            )


def record_closet_tombstones(
    conn: sqlite3.Connection,
    memory_ids: list[str],
    *,
    when: datetime | None = None,
    applying_remote_deletion: bool = False,
    authoritative: bool = False,
) -> None:
    if not memory_ids:
        return
    stamp = utc_iso(when or datetime.now(timezone.utc))
    if authoritative and when is not None:
        capped = when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc)
        stamp = utc_iso(min(capped, datetime.now(timezone.utc)))
        conn.executemany(
            f"INSERT INTO {CLOSET_TOMBSTONE_TABLE} (id, tombstoned_at, is_local) VALUES (?, ?, 0) ON "
            f"CONFLICT(id) DO UPDATE SET tombstoned_at = MAX({CLOSET_TOMBSTONE_TABLE}.tombstoned_at, "
            f"excluded.tombstoned_at)",
            [(mid, stamp) for mid in memory_ids],
        )
        return
    if applying_remote_deletion:
        conn.executemany(
            f"INSERT INTO {CLOSET_TOMBSTONE_TABLE} (id, tombstoned_at, is_local) VALUES (?, ?, 0) ON "
            f"CONFLICT(id) DO UPDATE SET tombstoned_at = excluded.tombstoned_at, is_local = 0",
            [(mid, stamp) for mid in memory_ids],
        )
        return
    if when is None:
        conn.executemany(
            f"INSERT INTO {CLOSET_TOMBSTONE_TABLE} (id, tombstoned_at, is_local) VALUES (?, ?, 1) ON "
            f"CONFLICT(id) DO UPDATE SET is_local = 1, tombstoned_at = "
            f"MAX({CLOSET_TOMBSTONE_TABLE}.tombstoned_at, excluded.tombstoned_at)",
            [(mid, stamp) for mid in memory_ids],
        )
        return
    conn.executemany(
        f"INSERT INTO {CLOSET_TOMBSTONE_TABLE} (id, tombstoned_at, is_local) VALUES (?, ?, 0) ON "
        f"CONFLICT(id) DO UPDATE SET tombstoned_at = CASE WHEN {CLOSET_TOMBSTONE_TABLE}.is_local = 1 "
        f"THEN {CLOSET_TOMBSTONE_TABLE}.tombstoned_at ELSE "
        f"MIN({CLOSET_TOMBSTONE_TABLE}.tombstoned_at, excluded.tombstoned_at) END",
        [(mid, stamp) for mid in memory_ids],
    )


def is_marked_closet_row(conn: sqlite3.Connection, memory_id: str) -> bool:
    return conn.execute(f"SELECT 1 FROM memories WHERE id = ? AND {IS_CLOSET_SQL}", (memory_id,)).fetchone() is not None


def is_marked_closet(engine: object, memory_id: str) -> bool:
    # Keep the existing caller guard without requiring an engine-specific API.
    conn = getattr(engine, "_conn", None)
    return conn is not None and has_marker(conn) and is_marked_closet_row(conn, memory_id)


def _owned_by(parent_id: str, related_raw: str | None) -> bool:
    """Whether a marked row's back-reference names exactly ``parent_id``.

    A copy's id always sits under its parent's, but that prefix is a coarse net:
    it also catches the copies of another memory whose own id starts the same
    way. The back-reference every copy carries is what decides ownership.
    """
    try:
        return json.loads(related_raw or "[]") == [parent_id]
    except (TypeError, ValueError):
        return False


def clear_marked_copies(
    conn: sqlite3.Connection, parent_id: str, *, has_embeddings: bool, tombstone: bool = True
) -> list[str]:
    """Remove the marked legacy copies of ``parent_id``. The caller holds the write lock.

    A row carrying the marker is kept out of listings, counts and recall, so the
    only thing keeping its text reachable is the memory it was copied from. When
    that memory's text is edited away or deleted, no copy of the old text may
    outlive it: without this, editing a secret out of a memory left a copy of the
    secret in the store for good.

    Nothing is lost by removing them. A copy holds a projection of text its
    parent already holds, and no version derives them any more.

    Content-free on the way out: the ids and a time, never the text. Writing the
    speaker text into Trash here would put back exactly what a redaction removes.

    ``tombstone=False`` is for expiry, which is not a redaction: the cloud row
    carries the same expiry and ages out on its own.

    Scoped to rows carrying the marker, so a real memory whose id merely looks
    like a copy's is never touched.
    """
    if not has_marker(conn):
        return []
    rows = conn.execute(
        f"SELECT id, related_to FROM memories WHERE {IS_CLOSET_SQL} AND instr(id, ?) > 0",
        (CLOSET_SEPARATOR,),
    ).fetchall()
    ids = sorted(row[0] for row in rows if _owned_by(parent_id, row[1]))
    if not ids:
        return []
    if tombstone:
        record_closet_tombstones(conn, ids)
    for batch in chunked(ids):
        placeholders = ",".join("?" * len(batch))
        if has_embeddings:
            conn.execute(f"DELETE FROM memory_embeddings WHERE id IN ({placeholders})", batch)
        conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", batch)
    return ids


def rearm_legacy_announcement(conn: sqlite3.Connection, memory_id: str, updated_at: str) -> None:
    """Compatibility claim for callers still using the legacy retention queue.

    Trash still retains deletion evidence while this flag is pending. Retire
    this adapter together with that consumer; clearing it early admits old
    speaker snapshots after the retention window. Store opening retires it.
    """
    conn.execute(
        "INSERT INTO legacy_closet_ids (id, announce_pending, legacy_updated_at) VALUES (?, 1, ?) "
        "ON CONFLICT(id) DO UPDATE SET announce_pending = 1, announced_at = NULL, legacy_updated_at = "
        "MAX(COALESCE(legacy_closet_ids.legacy_updated_at, excluded.legacy_updated_at), excluded.legacy_updated_at)",
        (memory_id, utc_iso(updated_at)),
    )


def _stored_grade(conn: sqlite3.Connection, memory_id: str, table: str) -> tuple[str, str | None]:
    row = conn.execute(f"SELECT content, related_to, created_at FROM {table} WHERE id = ?", (memory_id,)).fetchone()
    if row is None:
        return TIER_NONE, None
    return classify_legacy_copy(conn, memory_id, content=row[0], related_raw=row[1], created_at=row[2])


def is_proven_unmarked_copy(conn: sqlite3.Connection, memory_id: str) -> bool:
    if not conn.execute(f"SELECT 1 FROM memories WHERE id = ? AND {NOT_CLOSET_SQL}", (memory_id,)).fetchone():
        return False
    return _stored_grade(conn, memory_id, "memories")[0] == TIER_PROVEN


def claim_proven_unmarked_copy(conn: sqlite3.Connection, memory_id: str) -> bool:
    if not is_proven_unmarked_copy(conn, memory_id):
        return False
    stamp = conn.execute("SELECT updated_at FROM memories WHERE id = ?", (memory_id,)).fetchone()[0]
    rearm_legacy_announcement(conn, memory_id, stamp)
    return True


def grade_copy_snapshot(
    conn: sqlite3.Connection, memory_id: str, *, content: str, related_to: list[str], created_at: str
) -> str | None:
    tier, _ = classify_legacy_copy(
        conn, memory_id, content=content, related_raw=json.dumps(list(related_to)), created_at=created_at
    )
    return tier if tier in (TIER_PROVEN, TIER_LIKELY) else None


def refuse_restorable_copy_snapshot(conn: sqlite3.Connection, memory_id: str) -> str | None:
    if not _table_exists(conn, "ui_tombstones"):
        return None
    tier, parent = _stored_grade(conn, memory_id, "ui_tombstones")
    if tier != TIER_PROVEN:
        return None
    _claim_snapshot(conn, memory_id)
    conn.execute("DELETE FROM ui_tombstones WHERE id = ?", (memory_id,))
    return parent


def clear_copy_snapshot(conn: sqlite3.Connection, copy_id: str) -> bool:
    if not _table_exists(conn, "ui_tombstones"):
        return False
    tier, _ = _stored_grade(conn, copy_id, "ui_tombstones")
    if tier != TIER_PROVEN:
        # A marked inferred copy can have text differing from its live parent.
        # Preserve its pre-image before clearing an identical Trash snapshot.
        row = conn.execute(
            "SELECT m.content, t.related_to, t.created_at, m.related_to FROM memories m "
            "JOIN ui_tombstones t ON t.id = m.id AND t.content = m.content WHERE m.id = ? AND m.is_closet = 2",
            (copy_id,),
        ).fetchone()
        if row is None:
            return False
        try:
            parents = json.loads(row[3] or "[]")
        except (TypeError, ValueError):
            return False
        if not isinstance(parents, list) or len(parents) != 1 or not isinstance(parents[0], str):
            return False
        parent = conn.execute("SELECT created_at FROM memories WHERE id = ?", (parents[0],)).fetchone()
        if parent is None or not copy_id.startswith(parents[0] + CLOSET_SEPARATOR):
            return False
        if not has_copy_provenance(
            parents[0],
            copy_id[len(parents[0] + CLOSET_SEPARATOR) :],
            content=row[0],
            related_raw=row[1],
            created_at=row[2],
            parent_created_at=parent[0],
        ):
            return False
        back_up_inferred_copy(conn, copy_id, "cleared")
    if tier == TIER_PROVEN:
        _claim_snapshot(conn, copy_id)
    conn.execute("DELETE FROM ui_tombstones WHERE id = ?", (copy_id,))
    return True


def count_unmarked_derivable_copies(conn: sqlite3.Connection) -> int:
    return sum(tier in (TIER_PROVEN, TIER_LIKELY) for _, tier in legacy_copy_grades(conn))


def count_adopted_pending_rebuild(conn: sqlite3.Connection) -> int:
    if not has_marker(conn):
        return 0
    return conn.execute("SELECT COUNT(*) FROM memories WHERE is_closet = 2").fetchone()[0]


def count_orphan_shaped_rows(conn: sqlite3.Connection) -> int:
    return sum(tier == TIER_ORPHAN for _, tier in legacy_copy_grades(conn))


def clear_retired_records(conn: sqlite3.Connection, memory_id: str) -> None:
    """Release historical claims when a caller writes a normal memory at the id."""
    for table in ("closet_tombstones", "legacy_closet_ids"):
        if _table_exists(conn, table):
            conn.execute(f"DELETE FROM {table} WHERE id = ?", (memory_id,))


def _claim_snapshot(conn: sqlite3.Connection, memory_id: str) -> None:
    row = conn.execute("SELECT tombstoned_at, updated_at FROM ui_tombstones WHERE id = ?", (memory_id,)).fetchone()
    rearm_legacy_announcement(conn, memory_id, later_stamp(row[0], row[1]))


def later_stamp(a: str | None, b: str | None) -> str | None:
    if a is None:
        return b
    if b is None:
        return a
    try:
        left, right = (datetime.fromisoformat(a), datetime.fromisoformat(b))
    except (ValueError, OverflowError, TypeError):
        return a
    if left.tzinfo is None:
        left = left.replace(tzinfo=timezone.utc)
    if right.tzinfo is None:
        right = right.replace(tzinfo=timezone.utc)
    return a if left >= right else b
