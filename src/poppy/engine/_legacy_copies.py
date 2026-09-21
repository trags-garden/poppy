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
from poppy.engine._timestamps import (
    _columns,
    _table_exists,
    chunked,
    decode_text_columns,
    later_stamp,
    raw_text_columns,
    utc_iso,
)

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
    if not all(isinstance(value, str) for value in (memory_id, content, related_raw, created_at)):
        return TIER_NONE, None
    try:
        datetime.fromisoformat(created_at)
    except (ValueError, TypeError, OverflowError):
        return TIER_NONE, None
    for parent_id, slug in _parent_splits(memory_id):
        try:
            parent = _text_row(conn, "memories", parent_id, "content", "created_at")
            if parent is not None:
                datetime.fromisoformat(parent[1])
        except (ValueError, TypeError, OverflowError):
            return TIER_NONE, None
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


def back_up_cleared_snapshot(conn: sqlite3.Connection, copy_id: str) -> None:
    """Keep the Trash body about to be deleted, when nothing is held for this id.

    ``back_up_inferred_copy`` reads the STORED row, which is the right pre-image
    for a row being removed but the wrong one for a Trash entry: where the two
    differ, and that is the case this exists for, the entry's own text is what
    disappears. So this reads the entry.

    One pre-image per id, and the slot may already hold the stored row's text
    from when it was marked. That earlier pre-image is kept: it is the text the
    user was working with, where a Trash entry is something already deleted, and
    replacing one recoverable row with another gains nothing. So when a
    pre-image already exists for the id, the entry is deleted WITHOUT one.

    Nothing a user or an agent can reach reads this table, and it ages out on
    the same clock as Trash.
    """
    conn.execute(BACKUP_DDL)
    conn.execute(
        "INSERT OR IGNORE INTO closet_migration_backup "
        "(id, content, enriched_content, related_to, created_at, updated_at, action, migrated_at) "
        "SELECT id, content, NULL, related_to, created_at, updated_at, 'cleared', ? "
        "FROM ui_tombstones WHERE id = ?",
        (datetime.now(timezone.utc).isoformat(), copy_id),
    )


def _text_row(conn: sqlite3.Connection, table: str, memory_id: str, *columns: str) -> tuple | None:
    row = conn.execute(f"SELECT {raw_text_columns(*columns)} FROM {table} WHERE id = ?", (memory_id,)).fetchone()
    return decode_text_columns(tuple(row)) if row is not None else None


def legacy_copy_grades(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Grade TEXT candidates only; unreadable rows stay unmarked and untouched."""
    columns = raw_text_columns("id", "content", "related_to", "created_at", "updated_at")
    rows = conn.execute(
        f"SELECT {columns} FROM memories WHERE instr(id, ?) > 0 AND COALESCE(is_closet, 0) = 0",
        (SEPARATOR,),
    ).fetchall()
    grades = []
    for row in rows:
        try:
            memory_id, content, related, created, updated = decode_text_columns(tuple(row))
            datetime.fromisoformat(updated)
            tier, _ = classify_legacy_copy(
                conn,
                memory_id,
                content=content,
                related_raw=related,
                created_at=created,
            )
        except Exception:
            # The marker column is added in this transaction. A bad row must
            # not roll it back and make every subsequent open fail again.
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


def announced_copy_claim(conn: sqlite3.Connection, memory_id: str) -> str | None:
    if not _table_exists(conn, COPY_CLAIM_TABLE):
        return None
    row = conn.execute(f"SELECT legacy_updated_at FROM {COPY_CLAIM_TABLE} WHERE id = ?", (memory_id,)).fetchone()
    return row[0] if row is not None else None


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
    conn: sqlite3.Connection,
    parent_id: str,
    *,
    has_embeddings: bool,
    tombstone: bool = True,
    remote_event_ts: datetime | None = None,
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
        record_copy_deletions(conn, ids, when=remote_event_ts, applying_remote_deletion=remote_event_ts is not None)
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
    row = _text_row(conn, table, memory_id, "content", "related_to", "created_at")
    if row is None:
        return TIER_NONE, None
    if not all(isinstance(value, str) for value in row):
        raise ValueError("Incomplete legacy row")
    datetime.fromisoformat(row[2])
    return classify_legacy_copy(conn, memory_id, content=row[0], related_raw=row[1], created_at=row[2])


def is_proven_unmarked_copy(conn: sqlite3.Connection, memory_id: str) -> bool:
    if not conn.execute("SELECT 1 FROM memories WHERE id = ? AND COALESCE(is_closet, 0) = 0", (memory_id,)).fetchone():
        return False
    try:
        return _stored_grade(conn, memory_id, "memories")[0] == TIER_PROVEN
    except (ValueError, TypeError, OverflowError):
        return False


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
    try:
        tier, parent = _stored_grade(conn, memory_id, "ui_tombstones")
    except (ValueError, TypeError, OverflowError):
        return None
    if tier != TIER_PROVEN:
        return None
    _claim_snapshot(conn, memory_id)
    conn.execute("DELETE FROM ui_tombstones WHERE id = ?", (memory_id,))
    return parent


def clear_copy_snapshot(conn: sqlite3.Connection, copy_id: str) -> bool:
    if not _table_exists(conn, "ui_tombstones"):
        return False
    try:
        tier, _ = _stored_grade(conn, copy_id, "ui_tombstones")
    except (ValueError, TypeError, OverflowError):
        # An unreadable snapshot cannot be graded. A readable marked copy at
        # this id still proves that the snapshot must not become public.
        marked = conn.execute(
            "SELECT 1 FROM memories WHERE id = ? AND COALESCE(is_closet, 0) >= 1", (copy_id,)
        ).fetchone()
        try:
            live_tier, _ = _stored_grade(conn, copy_id, "memories")
        except (ValueError, TypeError, OverflowError):
            return False
        if not marked or live_tier not in (TIER_PROVEN, TIER_LIKELY):
            return False
        conn.execute("DELETE FROM ui_tombstones WHERE id = ?", (copy_id,))
        return True
    if tier != TIER_PROVEN:
        # A retained copy's text can differ from its live parent's, and a
        # snapshot an older client left can differ from BOTH: it holds that
        # copy's text from an earlier point. Requiring the snapshot to match
        # the stored row byte for byte let those older snapshots through, and
        # a snapshot only stays hidden while a marked row exists at its id to
        # hide it. The moment the row went, the snapshot was listed in Trash,
        # restorable as an ordinary memory, and pushed with the speaker text
        # in its body. So the snapshot is graded on ITS OWN fields, and any
        # snapshot at a marked id that proves to be that copy's text goes,
        # with a recoverable pre-image kept first.
        if not conn.execute(
            "SELECT 1 FROM memories WHERE id = ? AND COALESCE(is_closet, 0) >= 1", (copy_id,)
        ).fetchone():
            return False
        try:
            snapshot = _text_row(conn, "ui_tombstones", copy_id, "content", "related_to", "created_at")
            live = _text_row(conn, "memories", copy_id, "related_to")
            if snapshot is None or live is None:
                return False
            parents = json.loads(live[0] or "[]")
            if not isinstance(parents, list) or len(parents) != 1 or not isinstance(parents[0], str):
                return False
            parent = _text_row(conn, "memories", parents[0], "created_at")
            if parent is None or not copy_id.startswith(parents[0] + SEPARATOR):
                return False
            if not has_copy_provenance(
                parents[0],
                copy_id[len(parents[0] + SEPARATOR) :],
                content=snapshot[0],
                related_raw=snapshot[1],
                created_at=snapshot[2],
                parent_created_at=parent[0],
            ):
                return False
        except (ValueError, TypeError, OverflowError):
            return False
        back_up_cleared_snapshot(conn, copy_id)
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
    try:
        row = _text_row(conn, "ui_tombstones", memory_id, "tombstoned_at", "updated_at")
        stamp = later_stamp(row[0], row[1])
        datetime.fromisoformat(stamp)
    except (ValueError, TypeError, OverflowError):
        return
    rearm_legacy_announcement(conn, memory_id, stamp)
