"""Bidirectional sync between Poppy and a Trags `memories` KV store.

The Trags side is hosted side: memories KV table + 5 CRUD endpoints). The Poppy
side reads from BOTH `memories` (live) AND `ui_tombstones` (soft-deletes
and supersede chains) so the round-trip preserves the full state.

Watermarks are tracked per remote URL in ``~/.poppy/sync_state.json`` —
push sends everything updated after ``last_pushed_at``, pull fetches
everything with ``updated_at >= last_pulled_at``.

Conflict policy: last-writer-wins by ``updated_at``. On push, we use
upsert (POST) which the Trags server resolves server-side. On pull,
local rows whose ``updated_at`` is newer than the incoming row are
preserved (the incoming row is skipped).

Watermark semantics (watermark logic): items are processed in ``updated_at`` ASC
order. The watermark only advances while every preceding item has
succeeded. On the first failure the watermark freezes, so the failed
item — and everything that came after it in time — is retried on the
next sync. Successful retries are idempotent on the server side.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from poppy.engine.interface import RetrievalEngine
from poppy.sync.client import (
    TragsAuthError,
    TragsClient,
    TragsConflictError,
    TragsError,
)
from poppy.sync.serializer import (
    is_tombstone,
    memory_to_wire,
    tombstone_to_wire,
    wire_to_memory,
)
from poppy.sync.state import RemoteState, SyncState, get_remote, latest_iso, load, save
from poppy.ui.tombstones import TombstoneStore

__all__ = [
    "PushResult",
    "PullResult",
    "SyncResult",
    "push",
    "pull",
    "sync",
    "TragsClient",
    "TragsAuthError",
    "TragsConflictError",
    "TragsError",
]


@dataclass
class PushResult:
    sent_live: int
    sent_tombstones: int
    skipped: int
    errors: int


@dataclass
class PullResult:
    applied_live: int
    applied_tombstones: int
    skipped_stale: int
    errors: int


@dataclass
class SyncResult:
    push: PushResult
    pull: PullResult


def push(
    *,
    engine: RetrievalEngine,
    tombstones: TombstoneStore,
    client: TragsClient,
    state: SyncState,
    poppy_dir: Path,
    dry_run: bool = False,
) -> PushResult:
    """Send local memories + tombstones to Trags.

    Only sends rows whose ``updated_at`` is newer than ``last_pushed_at``.
    On first run (state empty), everything is sent. Items are sent in
    ``updated_at`` ASC order; the watermark freezes on the first failure
    so failed items retry on the next sync.
    """
    remote = get_remote(state, client.base_url)
    watermark = remote.last_pushed_at

    sent_live = 0
    sent_tombstones = 0
    skipped = 0
    errors = 0
    new_watermark = watermark
    watermark_locked = False

    # Collect candidates above watermark. Items are tagged with their
    # kind so we know which wire serializer to use.
    candidates: list[tuple[str, str, object]] = []

    for memory in engine.list_all(filters=None, limit=10**9):
        iso = memory.updated_at.isoformat()
        if watermark and iso <= watermark:
            skipped += 1
            continue
        candidates.append((iso, "live", memory))

    for ts in tombstones.list_all():
        iso = ts.tombstoned_at.isoformat()
        if watermark and iso <= watermark:
            skipped += 1
            continue
        candidates.append((iso, "tomb", ts))

    # Oldest first — watermark advances monotonically through successes.
    candidates.sort(key=lambda c: c[0])

    for iso, kind, payload in candidates:
        if dry_run:
            if kind == "live":
                sent_live += 1
            else:
                sent_tombstones += 1
            if not watermark_locked:
                new_watermark = latest_iso(new_watermark, iso)
            continue

        try:
            if kind == "live":
                client.upsert(memory_to_wire(payload))  # type: ignore[arg-type]
                sent_live += 1
            else:
                client.upsert(tombstone_to_wire(payload))  # type: ignore[arg-type]
                sent_tombstones += 1
        except (TragsError, TragsConflictError):
            errors += 1
            watermark_locked = True
            continue

        if not watermark_locked:
            new_watermark = latest_iso(new_watermark, iso)

    if not dry_run:
        remote.last_pushed_at = new_watermark
        remote.pushed_count += sent_live + sent_tombstones
        remote.last_synced_at = datetime.now(timezone.utc).isoformat()
        save(poppy_dir, state)

    return PushResult(
        sent_live=sent_live,
        sent_tombstones=sent_tombstones,
        skipped=skipped,
        errors=errors,
    )


def pull(
    *,
    engine: RetrievalEngine,
    tombstones: TombstoneStore,
    client: TragsClient,
    state: SyncState,
    poppy_dir: Path,
    dry_run: bool = False,
) -> PullResult:
    """Apply Trags rows that are newer than our last_pulled_at watermark.

    Routing:
      row with deleted_at=null  → engine.ingest(memory)   (insert/update)
      row with deleted_at set   → tombstones.add(...)     (mirror soft-delete)
                                  + engine.delete(id) if a live row exists

    Skips rows whose updated_at <= local copy's updated_at to avoid clobbering
    locally-newer state (last-writer-wins). Incoming rows are sorted
    ``updated_at`` ASC so the watermark advances contiguously; a parse error
    on any row freezes the watermark so unparsed rows get retried later.
    """
    remote = get_remote(state, client.base_url)
    watermark = remote.last_pulled_at

    applied_live = 0
    applied_tombstones = 0
    skipped_stale = 0
    errors = 0
    new_watermark = watermark
    watermark_locked = False

    # Collect + parse, then sort oldest-first. Server returns newest-first,
    # so we have to materialize before processing.
    candidates: list[tuple[str, dict, object]] = []
    for row in client.iter_all_since(watermark):
        iso = row.get("updated_at")
        try:
            incoming = wire_to_memory(row)
        except (KeyError, ValueError):
            errors += 1
            watermark_locked = True
            continue
        if not iso:
            errors += 1
            watermark_locked = True
            continue
        candidates.append((iso, row, incoming))

    candidates.sort(key=lambda c: c[0])

    for iso, row, incoming in candidates:
        if is_tombstone(row):
            if not dry_run:
                local_live = engine.get(incoming.id)  # type: ignore[attr-defined]
                if local_live is not None and local_live.updated_at >= incoming.updated_at:  # type: ignore[attr-defined]
                    skipped_stale += 1
                    if not watermark_locked:
                        new_watermark = latest_iso(new_watermark, iso)
                    continue
                if local_live is not None:
                    engine.delete(incoming.id)  # type: ignore[attr-defined]
                superseded_by = row.get("superseded_by")
                tombstones.add(incoming, superseded_by=superseded_by)  # type: ignore[arg-type]
            applied_tombstones += 1
        else:
            if not dry_run:
                existing = engine.get(incoming.id)  # type: ignore[attr-defined]
                if existing is not None and existing.updated_at > incoming.updated_at:  # type: ignore[attr-defined]
                    skipped_stale += 1
                    if not watermark_locked:
                        new_watermark = latest_iso(new_watermark, iso)
                    continue
                engine.ingest(incoming)  # type: ignore[arg-type]
            applied_live += 1

        if not watermark_locked:
            new_watermark = latest_iso(new_watermark, iso)

    if not dry_run:
        remote.last_pulled_at = new_watermark
        remote.pulled_count += applied_live + applied_tombstones
        remote.last_synced_at = datetime.now(timezone.utc).isoformat()
        save(poppy_dir, state)

    return PullResult(
        applied_live=applied_live,
        applied_tombstones=applied_tombstones,
        skipped_stale=skipped_stale,
        errors=errors,
    )


def sync(
    *,
    engine: RetrievalEngine,
    tombstones: TombstoneStore,
    client: TragsClient,
    poppy_dir: Path,
    dry_run: bool = False,
) -> SyncResult:
    """Pull-then-push. Pull first so any incoming deletes apply before we
    re-upload tombstones we may have just learned about."""
    state = load(poppy_dir)
    pull_res = pull(
        engine=engine,
        tombstones=tombstones,
        client=client,
        state=state,
        poppy_dir=poppy_dir,
        dry_run=dry_run,
    )
    push_res = push(
        engine=engine,
        tombstones=tombstones,
        client=client,
        state=state,
        poppy_dir=poppy_dir,
        dry_run=dry_run,
    )
    return SyncResult(push=push_res, pull=pull_res)


def make_client_from_config(*, base_url: str, api_key: str) -> TragsClient:
    return TragsClient(base_url=base_url, api_key=api_key)


# Public re-export helpers used by the CLI ----------------------------------


def remote_state_for(poppy_dir: Path, url: str) -> RemoteState:
    """Read-only snapshot of the watermark for `poppy sync status`."""
    state = load(poppy_dir)
    return get_remote(state, url)
