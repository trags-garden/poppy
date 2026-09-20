"""One-time cleanup of legacy derived copies, using the frozen pre-0.3.1 format.

Delete this module once stores from before this release are no longer supported.
Only store opening and checks of legacy live/cloud/Trash rows use this code;
ingest must never call it. Classification requires provenance AND exact content
equality before authorizing removal. Inferred copies keep their text and backup.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone

from poppy.db import write_txn

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


def legacy_copy_grades(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Read unmarked candidates once, before any marker writes change the store."""
    rows = conn.execute(
        "SELECT id, content, related_to, created_at FROM memories "
        "WHERE instr(id, ?) > 0 AND COALESCE(is_closet, 0) = 0",
        (SEPARATOR,),
    ).fetchall()
    grades = []
    for memory_id, content, related, created in rows:
        try:
            tier, _ = classify_legacy_copy(conn, memory_id, content=content, related_raw=related, created_at=created)
        except Exception:
            # Grading runs inside the transaction that adds the marker column, so
            # a row this cannot read must not raise: that rolls the column back
            # too, and every later open fails the same way, leaving the store
            # unopenable. An ungradeable row is left alone, which is what the
            # weakest grade already means.
            tier = TIER_NONE
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
