"""Memory lifecycle helpers: TTL parsing, edit, supersede.

These compose the RetrievalEngine ABC with the UI's TombstoneStore so supersede
soft-deletes the old memory through the same 7-day undo window as a manual
delete. CLI and MCP both call into here so the behavior stays identical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from poppy.engine.interface import RetrievalEngine
from poppy.models import Memory

_DURATION_RE = re.compile(r"(\d+)\s*([wdhms])")


def parse_ttl(text: str) -> timedelta:
    """Parse a TTL string like '30d', '12h', '1w3d', or a bare integer (days).

    Raises ValueError on negative, zero, or unparseable input.
    """
    s = text.strip().lower()
    if not s:
        raise ValueError("empty TTL")

    if s.isdigit():
        days = int(s)
        if days <= 0:
            raise ValueError(f"TTL must be positive: {text!r}")
        return timedelta(days=days)

    matches = _DURATION_RE.findall(s)
    if not matches:
        raise ValueError(f"unparseable TTL: {text!r} (use forms like 30d, 12h, 1w3d)")
    if "".join(f"{n}{u}" for n, u in matches) != re.sub(r"\s+", "", s):
        raise ValueError(f"unparseable TTL: {text!r}")

    units = {"w": "weeks", "d": "days", "h": "hours", "m": "minutes", "s": "seconds"}
    total = timedelta()
    for n, u in matches:
        total += timedelta(**{units[u]: int(n)})
    if total.total_seconds() <= 0:
        raise ValueError(f"TTL must be positive: {text!r}")
    return total


def parse_expires_at(text: str) -> datetime:
    """Parse an ISO-8601 datetime string. Naive datetimes are assumed UTC."""
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_since(text: str, *, now: datetime | None = None) -> datetime:
    """Parse a --since value into an inclusive lower bound for created_at.

    Accepts either an ISO-8601 date/datetime ('2026-06-01',
    '2026-06-01T12:00:00') or a relative duration in TTL syntax ('7d',
    '12h', '1w3d', bare integer = days) meaning "the last N". Naive
    datetimes are assumed UTC. Raises ValueError on unparseable input.
    """
    s = text.strip()
    if not s:
        raise ValueError("empty --since value")
    try:
        return parse_expires_at(s)
    except ValueError:
        pass
    try:
        delta = parse_ttl(s)
    except ValueError:
        raise ValueError(
            f"invalid --since value: {text!r} (use an ISO date like 2026-06-01 or a duration like 7d, 12h, 1w3d)"
        ) from None
    base = now or datetime.now(timezone.utc)
    return base - delta


def resolve_expiry(
    ttl: str | None,
    expires_at: str | None,
    *,
    now: datetime | None = None,
) -> datetime | None:
    """Resolve --ttl / --expires-at into a datetime, or None if neither given.

    Mutually exclusive — caller should validate beforehand. Returns None if both
    are None so callers can pass through unchanged.
    """
    if ttl and expires_at:
        raise ValueError("--ttl and --expires-at are mutually exclusive")
    if ttl is None and expires_at is None:
        return None
    base = now or datetime.now(timezone.utc)
    if ttl is not None:
        return base + parse_ttl(ttl)
    return parse_expires_at(expires_at)  # type: ignore[arg-type]


@dataclass
class EditResult:
    memory: Memory
    changed: bool


def edit_memory(
    engine: RetrievalEngine,
    memory_id: str,
    *,
    content: str | None = None,
    memory_type: str | None = None,
    project: str | None = None,
    expires_at: datetime | None = None,
    clear_expiry: bool = False,
    project_unset: bool = False,
) -> EditResult:
    """Apply a partial update to an existing memory. Preserves id, created_at, source.

    `project_unset=True` clears project to None (since `project=None` means "don't change").
    `clear_expiry=True` clears expires_at to None (since `expires_at=None` means "don't change").
    """
    existing = engine.get(memory_id)
    if existing is None:
        raise KeyError(f"memory not found: {memory_id}")

    if expires_at is not None and clear_expiry:
        raise ValueError("expires_at and clear_expiry are mutually exclusive")

    new_content = content if content is not None else existing.content
    new_type = memory_type if memory_type is not None else existing.memory_type
    if project_unset:
        new_project = None
    elif project is not None:
        new_project = project
    else:
        new_project = existing.project
    if clear_expiry:
        new_expires = None
    elif expires_at is not None:
        new_expires = expires_at
    else:
        new_expires = existing.expires_at

    changed = (
        new_content != existing.content
        or new_type != existing.memory_type
        or new_project != existing.project
        or new_expires != existing.expires_at
    )
    if not changed:
        return EditResult(memory=existing, changed=False)

    updated = Memory(
        id=existing.id,
        content=new_content,
        memory_type=new_type,
        source=existing.source,
        project=new_project,
        related_to=existing.related_to,
        created_at=existing.created_at,
        updated_at=datetime.now(timezone.utc),
        confidence=existing.confidence,
        expires_at=new_expires,
    )
    engine.ingest(updated)
    return EditResult(memory=updated, changed=True)


@dataclass
class SupersedeResult:
    new_id: str
    old_id: str
    tombstoned: bool


def supersede_memory(
    engine: RetrievalEngine,
    new_memory: Memory,
    old_id: str,
    *,
    poppy_dir: Path,
) -> SupersedeResult:
    """Tombstone the old memory and ingest the new, linking them via related_to.

    The tombstone makes supersede reversible for the standard 7-day window —
    `poppy ui` (or a future `poppy restore`) can bring the old back.
    """
    from poppy.ui.tombstones import TombstoneStore

    old = engine.get(old_id)
    if old is None:
        raise KeyError(f"memory not found: {old_id}")

    db_path = poppy_dir / "memories.db"
    tombstones = TombstoneStore(db_path)
    tombstones.add(old, superseded_by=new_memory.id)
    engine.delete(old_id)

    related = list(new_memory.related_to)
    if old_id not in related:
        related.append(old_id)
    new_memory.related_to = related
    engine.ingest(new_memory)

    return SupersedeResult(new_id=new_memory.id, old_id=old_id, tombstoned=True)
