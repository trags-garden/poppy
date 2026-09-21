"""One-time cleanup of legacy derived copies, using the frozen pre-0.3.1 format.

Delete this module once stores from before this release are no longer supported.
The classifier is only for store opening and checks of legacy live/cloud/Trash
rows. Ingest only clears historical claims and cascades parent redactions; it
never classifies new writes. Classification requires provenance AND exact
content equality before authorizing cleanup. Inferred copies retain a backup
and stay hidden until their parent is edited, deleted or expires.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone

from poppy.db import write_txn
from poppy.engine._timestamps import _columns, _table_exists, chunked, later_stamp, utc_iso

SEPARATOR = "_closet_"
MARKER_COLUMN = "is_closet"
TIER_PROVEN = "proven"
TIER_LIKELY = "likely"
TIER_ORPHAN = "orphan"
TIER_NONE = "none"

BACKUP_DDL = """
CREATE TABLE IF NOT EXISTS closet_migration_backup (
    id TEXT PRIMARY KEY, content TEXT, enriched_content TEXT, related_to TEXT,
    created_at TEXT, updated_at TEXT, action TEXT NOT NULL, migrated_at TEXT NOT NULL
);
"""


def has_marker(conn: sqlite3.Connection) -> bool:
    return MARKER_COLUMN in {row[1] for row in conn.execute("PRAGMA table_info(memories)")}


def _turns(content: str) -> list:
    try:
        value = json.loads(content)
    except (ValueError, TypeError):
        return []
    return value if isinstance(value, list) else []


def _speaker(turn: object) -> str | None:
    speaker = turn.get("speaker") if isinstance(turn, dict) else None
    return speaker if isinstance(speaker, str) and speaker else None


def _slug(speaker: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", speaker.lower()).strip("_") or "x"


def _projected_texts(content: str) -> dict[str, str]:
    """Reproduce only the old raw JSON and collision-safe speaker slugs."""
    turns = _turns(content)
    speakers = list(dict.fromkeys(speaker for turn in turns if (speaker := _speaker(turn))))
    if len(speakers) < 2:
        return {}
    projected: dict[str, str] = {}
    for speaker in speakers:
        base = slug = _slug(speaker)
        suffix = 2
        while slug in projected:
            slug = f"{base}_{suffix}"
            suffix += 1
        projected[slug] = json.dumps([turn for turn in turns if _speaker(turn) == speaker])
    return projected


def _parent_splits(memory_id: str) -> list[tuple[str, str]]:
    # Either the parent id or the slug may itself contain the separator.
    splits = []
    start = 0
    while (index := memory_id.find(SEPARATOR, start)) >= 0:
        splits.append((memory_id[:index], memory_id[index + len(SEPARATOR) :]))
        start = index + 1
    return splits


def same_instant(left: str | None, right: str | None) -> bool:
    if left == right:
        return True
    if left is None or right is None:
        return False
    try:
        return datetime.fromisoformat(left) == datetime.fromisoformat(right)
    except (ValueError, TypeError):
        return False


def has_copy_provenance(
    parent_id: str,
    slug: str,
    *,
    content: str,
    related_raw: str | None,
    created_at: str | None = None,
    parent_created_at: str | None = None,
) -> bool:
    """Require the old back-reference, speaker slug, and parent's creation time."""
    try:
        related = json.loads(related_raw or "[]")
    except (ValueError, TypeError):
        return False
    turns = _turns(content)
    if related != [parent_id] or not turns:
        return False
    speakers = {_speaker(turn) for turn in turns}
    if None in speakers or len(speakers) != 1:
        return False
    base = _slug(speakers.pop())
    if slug != base and not re.fullmatch(rf"{re.escape(base)}_\d+", slug):
        return False
    return parent_created_at is None or same_instant(created_at, parent_created_at)


def classify_legacy_copy(
    conn: sqlite3.Connection,
    memory_id: str,
    *,
    content: str,
    related_raw: str | None,
    created_at: str | None,
) -> tuple[str, str | None]:
    """Grade a stored or incoming row without mutating it or its parent.

    PROVEN requires a deterministic id, exact back-reference, matching creation
    instant, and byte-equal projection from the live parent. LIKELY has the same
    provenance but different text. Text alone earns no grade.
    """
    for parent_id, slug in _parent_splits(memory_id):
        parent = conn.execute("SELECT content, created_at FROM memories WHERE id = ?", (parent_id,)).fetchone()
        if not has_copy_provenance(
            parent_id,
            slug,
            content=content,
            related_raw=related_raw,
            created_at=created_at,
            parent_created_at=parent[1] if parent else None,
        ):
            continue
        raw = _projected_texts(parent[0]).get(slug) if parent else None
        if raw is None:
            return TIER_ORPHAN, parent_id
        return (TIER_PROVEN if content == raw else TIER_LIKELY), parent_id
    return TIER_NONE, None


def back_up_inferred_copy(conn: sqlite3.Connection, memory_id: str, action: str = "adopted") -> None:
    """Keep the existing recoverable pre-image format for inference only."""
    conn.execute(BACKUP_DDL)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
    enriched = "enriched_content" if "enriched_content" in columns else "NULL"
    conn.execute(
        "INSERT OR REPLACE INTO closet_migration_backup "
        "(id, content, enriched_content, related_to, created_at, updated_at, action, migrated_at) "
        f"SELECT id, content, {enriched}, related_to, created_at, updated_at, ?, ? FROM memories WHERE id = ?",
        (action, datetime.now(timezone.utc).isoformat(), memory_id),
    )


def _decoded(value: object) -> str | None:
    """Text for a column read as raw bytes, or a raise the caller turns into no grade.

    Read as bytes on purpose. A column declared TEXT can still hold bytes that
    are not valid text, and the database driver raises while BUILDING the result
    set, which is before any per-row guard can run. Decoding one row at a time
    moves that failure inside the guard.
    """
    if value is None or isinstance(value, str):
        return value
    return bytes(value).decode()


def legacy_copy_grades(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Read unmarked candidates once, before any marker writes change the store.

    Each field is read twice: as raw bytes, and as the storage class the value
    actually has. Only a value stored AS TEXT is a value a release that wrote
    copies could have written, so only that is decoded and graded. A row holding
    anything else in a field this reads gets no grade at all, whatever those
    bytes would spell: reading it as text would let a row nothing here wrote
    look like a copy, and the strongest grade authorises deleting it without
    keeping a pre-image.
    """
    rows = conn.execute(
        "SELECT CAST(id AS BLOB), CAST(content AS BLOB), CAST(related_to AS BLOB), "
        "CAST(created_at AS BLOB), typeof(id), typeof(content), typeof(related_to), "
        "typeof(created_at) FROM memories "
        "WHERE instr(id, ?) > 0 AND COALESCE(is_closet, 0) = 0",
        (SEPARATOR,),
    ).fetchall()
    grades = []
    for raw_id, raw_content, raw_related, raw_created, *storage in rows:
        if any(kind != "text" for kind in storage):
            continue
        try:
            memory_id = _decoded(raw_id)
            tier, _ = classify_legacy_copy(
                conn,
                memory_id,
                content=_decoded(raw_content),
                related_raw=_decoded(raw_related),
                created_at=_decoded(raw_created),
            )
        except Exception:
            # Grading runs inside the transaction that adds the marker column, so
            # a row this cannot read must not raise: that rolls the column back
            # too, and every later open fails the same way, leaving the store
            # unopenable. A row that cannot be graded is left exactly as it is.
            continue
        grades.append((memory_id, tier))
    return grades


def mark_legacy_copies_for_cleanup(conn: sqlite3.Connection, *, had_bloom_schema: bool) -> None:
    """Classify only stores predating the marker, within the caller's transaction.

    The engine holds the write gate and removes proven rows before committing.
    Do not rebuild text, copy parent metadata, or queue uploads: deletion records
    must retain each removed row's own timestamp, and no proven text is backed up.
    """
    if has_marker(conn):
        return
    with write_txn(conn):
        if has_marker(conn):
            return
        conn.execute("ALTER TABLE memories ADD COLUMN is_closet INTEGER NOT NULL DEFAULT 0")
        if not had_bloom_schema:
            return
        for memory_id, tier in legacy_copy_grades(conn):
            if tier == TIER_PROVEN:
                conn.execute("UPDATE memories SET is_closet = 1 WHERE id = ?", (memory_id,))
            elif tier == TIER_LIKELY:
                back_up_inferred_copy(conn, memory_id)
                conn.execute("UPDATE memories SET is_closet = 2 WHERE id = ?", (memory_id,))


COPY_DELETION_TABLE = "closet_tombstones"
COPY_CLAIM_TABLE = "legacy_closet_ids"
MARKED_COPY_SQL = "COALESCE(is_closet, 0) >= 1"
COPY_DELETION_DDL = """
CREATE TABLE IF NOT EXISTS closet_tombstones (
    id TEXT PRIMARY KEY, tombstoned_at TEXT NOT NULL, is_local INTEGER NOT NULL DEFAULT 0
);
"""
COPY_CLAIM_DDL = """
CREATE TABLE IF NOT EXISTS legacy_closet_ids (
    id TEXT PRIMARY KEY, announce_pending INTEGER NOT NULL DEFAULT 1,
    announced_at TEXT, legacy_updated_at TEXT
);
"""


def ensure_legacy_copy_tables(conn: sqlite3.Connection) -> None:
    conn.execute(COPY_DELETION_DDL)
    conn.execute(BACKUP_DDL)
    conn.execute(COPY_CLAIM_DDL)
    if "is_local" not in _columns(conn, COPY_DELETION_TABLE):
        conn.execute(f"ALTER TABLE {COPY_DELETION_TABLE} ADD COLUMN is_local INTEGER NOT NULL DEFAULT 0")


def pending_legacy_announcements(conn: sqlite3.Connection) -> list[tuple[str, str | None]]:
    if not _table_exists(conn, COPY_CLAIM_TABLE):
        return []
    rows = conn.execute(f"SELECT id, legacy_updated_at FROM {COPY_CLAIM_TABLE} WHERE announce_pending = 1 ORDER BY id")
    return [(row[0], row[1]) for row in rows]


def announced_copy_claim(conn: sqlite3.Connection, memory_id: str) -> str | None:
    if not _table_exists(conn, COPY_CLAIM_TABLE):
        return None
    row = conn.execute(f"SELECT legacy_updated_at FROM {COPY_CLAIM_TABLE} WHERE id = ?", (memory_id,)).fetchone()
    return row[0] if row is not None else None


def mark_legacy_announced(
    conn: sqlite3.Connection, memory_ids: list[str] | list[tuple[str, str | None]], *, when: datetime | None = None
) -> None:
    if not memory_ids or not _table_exists(conn, COPY_CLAIM_TABLE):
        return
    stamp = (when or datetime.now(timezone.utc)).isoformat()
    for entry in memory_ids:
        if isinstance(entry, tuple):
            memory_id, sent = entry
            conn.execute(
                f"UPDATE {COPY_CLAIM_TABLE} SET announce_pending = 0, announced_at = ? WHERE id = "
                f"? AND COALESCE(legacy_updated_at, '') = COALESCE(?, '')",
                (stamp, memory_id, utc_iso(sent)),
            )
        else:
            conn.execute(
                f"UPDATE {COPY_CLAIM_TABLE} SET announce_pending = 0, announced_at = ? WHERE id = ?", (stamp, entry)
            )


def record_copy_deletions(
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
            f"INSERT INTO {COPY_DELETION_TABLE} (id, tombstoned_at, is_local) VALUES (?, ?, 0) ON "
            f"CONFLICT(id) DO UPDATE SET tombstoned_at = MAX({COPY_DELETION_TABLE}.tombstoned_at, "
            f"excluded.tombstoned_at)",
            [(mid, stamp) for mid in memory_ids],
        )
        return
    if applying_remote_deletion:
        conn.executemany(
            f"INSERT INTO {COPY_DELETION_TABLE} (id, tombstoned_at, is_local) VALUES (?, ?, 0) ON "
            f"CONFLICT(id) DO UPDATE SET tombstoned_at = excluded.tombstoned_at, is_local = 0",
            [(mid, stamp) for mid in memory_ids],
        )
        return
    if when is None:
        conn.executemany(
            f"INSERT INTO {COPY_DELETION_TABLE} (id, tombstoned_at, is_local) VALUES (?, ?, 1) ON "
            f"CONFLICT(id) DO UPDATE SET is_local = 1, tombstoned_at = "
            f"MAX({COPY_DELETION_TABLE}.tombstoned_at, excluded.tombstoned_at)",
            [(mid, stamp) for mid in memory_ids],
        )
        return
    conn.executemany(
        f"INSERT INTO {COPY_DELETION_TABLE} (id, tombstoned_at, is_local) VALUES (?, ?, 0) ON "
        f"CONFLICT(id) DO UPDATE SET tombstoned_at = CASE WHEN {COPY_DELETION_TABLE}.is_local = 1 "
        f"THEN {COPY_DELETION_TABLE}.tombstoned_at ELSE "
        f"MIN({COPY_DELETION_TABLE}.tombstoned_at, excluded.tombstoned_at) END",
        [(mid, stamp) for mid in memory_ids],
    )


def is_marked_copy(engine: object, memory_id: str) -> bool:
    """Whether a stored row is hidden by a marker from legacy cleanup."""
    conn = getattr(engine, "_conn", None)
    return (
        conn is not None
        and has_marker(conn)
        and conn.execute(f"SELECT 1 FROM memories WHERE id = ? AND {MARKED_COPY_SQL}", (memory_id,)).fetchone()
        is not None
    )


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

    Inferred copies can differ from their parent. Their marker and ownership
    link still make them subject to the parent's redaction; retained pre-images
    follow the existing migration backup retention policy.

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
        f"SELECT id, related_to FROM memories WHERE {MARKED_COPY_SQL} AND instr(id, ?) > 0",
        (SEPARATOR,),
    ).fetchall()
    ids = sorted(row[0] for row in rows if _owned_by(parent_id, row[1]))
    if not ids:
        return []
    # A snapshot left by an older client would become visible after its live
    # marker disappears. Clear matching snapshots while the parent and copy
    # still supply the evidence, using the same guard as direct forget.
    for memory_id in ids:
        clear_copy_snapshot(conn, memory_id)
    if tombstone:
        record_copy_deletions(conn, ids)
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
    if not conn.execute("SELECT 1 FROM memories WHERE id = ? AND COALESCE(is_closet, 0) = 0", (memory_id,)).fetchone():
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
        if parent is None or not copy_id.startswith(parents[0] + SEPARATOR):
            return False
        if not has_copy_provenance(
            parents[0],
            copy_id[len(parents[0] + SEPARATOR) :],
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


def clear_retired_records(conn: sqlite3.Connection, memory_id: str) -> None:
    """Release historical claims when a caller writes a normal memory at the id."""
    for table in ("closet_tombstones", "legacy_closet_ids"):
        if _table_exists(conn, table):
            conn.execute(f"DELETE FROM {table} WHERE id = ?", (memory_id,))


def _claim_snapshot(conn: sqlite3.Connection, memory_id: str) -> None:
    row = conn.execute("SELECT tombstoned_at, updated_at FROM ui_tombstones WHERE id = ?", (memory_id,)).fetchone()
    rearm_legacy_announcement(conn, memory_id, later_stamp(row[0], row[1]))
