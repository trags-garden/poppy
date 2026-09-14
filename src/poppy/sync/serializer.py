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

# Stand-in body for a closet tombstone. The Trags route rejects a falsy
# `content` outright (`id, content, memory_type required` -> 400), so a
# soft-delete still has to carry a string; this one is a fixed literal that
# reveals nothing about the memory whose speaker copy is being deleted.
#
# Deliberately not something a person would type. `is_closet_tombstone` reads it
# back to recognise our own deletions, and a user's memory that happened to
# match would be misfiled — so the string is made implausible as a memory body
# rather than merely short.
CLOSET_TOMBSTONE_CONTENT = "[poppy: derived per-speaker copy removed]"
CLOSET_TOMBSTONE_MEMORY_TYPE = "fact"


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


# Used when a queued announcement has no recorded pre-migration timestamp, which
# can only happen for a store migrated by an unreleased build of this branch.
# An epoch-stamped delete loses the server's freshness comparison to any live
# row, so the announcement is answered and consumed without ever overwriting
# something. The leaked copy then falls to the human-inspected server-side
# cleanup rather than to a guess made here.
_UNKNOWN_LEGACY_TIMESTAMP = "1970-01-01T00:00:00+00:00"


def closet_tombstone_to_wire(memory_id: str, when: datetime | str | None) -> dict:
    """Serialize a leaked copy's deletion as a soft-deleted row carrying no text.

    Every field is either the id, the time, or a fixed constant — no content, no
    project, no source, no ``related_to``. A client at or below 0.3.0 pushed the
    per-speaker copy to the cloud as an ordinary memory; this is what deletes it
    there, and it must not re-send a character of the text.

    ``when`` is the LEAKED ROW'S OWN ``updated_at``, as the 0.2.4 client wrote
    it — captured at migration time, before the rebuild overwrote it. Not the
    local deletion time, and emphatically not now.

    That is what makes the announcement safe. The server resolves writes by
    freshness, so this timestamp is an ownership claim: it matches the
    untouched leak exactly, and the delete applies. If another device has since
    reclaimed the id for a real memory and uploaded it, that row is newer, the
    write is ignored, and the memory survives. A now-stamped delete would have
    won that comparison and destroyed it on every device.

    RESIDUAL, and it is not fixable from here: the tombstone the server stores
    keeps this old timestamp, so a device whose pull watermark has already passed
    it never fetches the tombstone and keeps its own copy of the row. Stamping
    applied deletions forward server-side was tried and reverted — it lost
    client-time ordering for offline restores — so this stands.

    It is acceptable because these announcements are best-effort convergence, not
    the erasure guarantee. The authoritative cloud erasure is the human-run
    server-side cleanup (trags #134, migration 033), whose tombstones ARE stamped
    forward and therefore reach every device.

    The nulls are accepted: the Trags route requires only id, content and
    memory_type (web/app/api/memories/route.ts), and the memories table takes
    project, source_type and source_session_id as nullable TEXT (migration 009).
    """
    iso = when if isinstance(when, str) else _iso(when)
    iso = iso or _UNKNOWN_LEGACY_TIMESTAMP
    return {
        "id": memory_id,
        "content": CLOSET_TOMBSTONE_CONTENT,
        "memory_type": CLOSET_TOMBSTONE_MEMORY_TYPE,
        "project": None,
        "source_type": None,
        "source_session_id": None,
        "source_timestamp": iso,
        "confidence": 1.0,
        "related_to": [],
        "expires_at": None,
        "superseded_by": None,
        "created_at": iso,
        "updated_at": iso,
        "deleted_at": iso,
    }


def is_tombstone(row: dict) -> bool:
    return row.get("deleted_at") is not None


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def deletion_time(row: dict) -> datetime | None:
    """When a soft-deleted wire row was DELETED, not when we heard about it.

    ``deleted_at`` is the deletion's own clock; ``updated_at`` is bumped to match
    it by every writer of a tombstone, so it is the fallback for a row that
    somehow carries only one of the two.
    """
    return _parse_iso(row.get("deleted_at")) or _parse_iso(row.get("updated_at"))


def is_closet_tombstone(row: dict) -> bool:
    """Whether a soft-deleted wire row is one of OUR content-free closet tombstones.

    Recognised by the two constants this client writes into them, on a row that
    is already deleted — not by the id's shape, and never applied to live data.
    A device that does not hold that copy locally would otherwise file the
    deletion in ``ui_tombstones`` and show a restorable Trash entry whose body is
    the placeholder; restoring it would create junk and push it back.

    Relies on the server storing ``content`` verbatim, which it does: the
    memories table takes it as plain TEXT and the upsert writes it through
    unchanged (trags migrations 009, 027 and 032), so the placeholder comes back
    byte-for-byte.

    The caller must also check that no live row holds the id. Content equality is
    a strong hint, not a proof of provenance: a real memory whose body happened
    to match would otherwise have its deletion filed as a closet's and never
    applied to the live row.
    """
    return (
        is_tombstone(row)
        and row.get("content") == CLOSET_TOMBSTONE_CONTENT
        and row.get("memory_type") == CLOSET_TOMBSTONE_MEMORY_TYPE
    )


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
