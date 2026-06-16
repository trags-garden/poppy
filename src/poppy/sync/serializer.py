"""Memory / Tombstone ↔ Trags wire-format conversions.

Trags `MemoryIn` shape (mirrored from `trags-apps/api/trags/api/routes/memories.py`):
  id, content, memory_type, project,
  source_type, source_session_id, source_timestamp,
  confidence, related_to, expires_at, superseded_by,
  created_at, updated_at, deleted_at

Mapping rules:
  Live Memory             → deleted_at=null, superseded_by=null
  Tombstone (plain)       → deleted_at=tombstoned_at, superseded_by=null
  Tombstone (superseded)  → deleted_at=tombstoned_at, superseded_by=<id>
"""

from __future__ import annotations

from datetime import datetime

from poppy.models import Memory, Source
from poppy.ui.tombstones import Tombstone


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def memory_to_wire(memory: Memory) -> dict:
    return {
        "id": memory.id,
        "content": memory.content,
        "memory_type": memory.memory_type,
        "project": memory.project,
        "source_type": memory.source.type,
        "source_session_id": memory.source.session_id,
        "source_timestamp": _iso(memory.source.timestamp),
        "confidence": memory.confidence,
        "related_to": list(memory.related_to),
        "expires_at": _iso(memory.expires_at),
        "superseded_by": None,
        "created_at": _iso(memory.created_at),
        "updated_at": _iso(memory.updated_at),
        "deleted_at": None,
    }


def tombstone_to_wire(tombstone: Tombstone) -> dict:
    """Serialize a tombstone as a soft-deleted Trags row.

    `updated_at` is bumped to `tombstoned_at` so the freshness check on pull
    treats the tombstone as a more recent state-change than the live row's
    original `updated_at`. This matches Trags' own `soft_delete()` semantics,
    which also bumps `updated_at` to `now()` on delete.
    """
    row = memory_to_wire(tombstone.memory)
    row["deleted_at"] = _iso(tombstone.tombstoned_at)
    row["updated_at"] = _iso(tombstone.tombstoned_at)
    row["superseded_by"] = tombstone.superseded_by
    return row


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def wire_to_memory(row: dict) -> Memory:
    """Always returns a Memory — caller checks row['deleted_at'] to decide
    whether to ingest live or write a tombstone."""
    timestamp = _parse_iso(row.get("source_timestamp"))
    created = _parse_iso(row.get("created_at"))
    if created is None:
        raise ValueError(f"Trags row missing created_at: {row.get('id')}")
    updated = _parse_iso(row.get("updated_at")) or created
    return Memory(
        id=row["id"],
        content=row["content"],
        memory_type=row["memory_type"],
        source=Source(
            type=row.get("source_type") or "trags-sync",
            session_id=row.get("source_session_id"),
            timestamp=timestamp or created,
        ),
        project=row.get("project"),
        related_to=list(row.get("related_to") or []),
        created_at=created,
        updated_at=updated,
        confidence=float(row.get("confidence") or 1.0),
        expires_at=_parse_iso(row.get("expires_at")),
    )


def is_tombstone(row: dict) -> bool:
    return row.get("deleted_at") is not None
