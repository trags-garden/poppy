"""Core Remember and Forget write flows shared by CLI, MCP, and UI.

CLAUDE.md says CLI and MCP "share a core library." That was true for the
primitives (`runtime.get_engine`, `lifecycle.supersede_memory`/edit/TTL) but the
write *flows* composing them used to be re-implemented per surface. This module
owns them once — memory-id generation, conflict-mode resolution, tombstoning,
telemetry, and autosync triggering each get exactly one home, the same move
already made for supersede in `lifecycle.py`.

CLI, MCP, and UI are thin adapters over `remember()` and `forget()`: they collect
inputs, translate the returned dataclass into their own presentation, and map the
raised `ValueError` (bad expiry) / `KeyError` (unknown supersede target) into
their own error contract (click exceptions vs `{"error": ...}` dicts vs HTTP
codes).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from poppy.engine.interface import RetrievalEngine
from poppy.models import Memory, Source

if TYPE_CHECKING:
    from poppy.capture.reconciler import Conflict
    from poppy.config import PoppyConfig
    from poppy.ui.tombstones import Tombstone, TombstoneStore


def make_memory_id() -> str:
    """Generate a memory id: ``mem_`` + 12 hex chars. Single home for the scheme."""
    return f"mem_{uuid.uuid4().hex[:12]}"


def _emit_memory_write(poppy_dir: Path, memory: Memory) -> None:
    """Emit the ``memory_write`` telemetry event for a successful write.

    Opt-out and offline-safety are handled inside ``telemetry.capture``.
    It never raises or blocks. Same event name and properties
    across every surface now that the flow is shared. Privacy: send whether a
    project was set, never the project name itself (project labels can carry
    client or codename information).
    """
    from poppy import telemetry

    telemetry.capture(
        poppy_dir,
        "memory_write",
        {"memory_type": memory.memory_type, "has_project": memory.project is not None, "source": memory.source.type},
    )


def _trigger_autosync(poppy_dir: Path) -> None:
    from poppy.sync.auto import trigger

    trigger(poppy_dir)


@dataclass
class RememberResult:
    """Outcome of a remember flow, for adapters to render.

    ``memory`` is the written memory, or ``None`` in ``check`` mode (a dry run
    that never writes). ``mode`` is the resolved conflict mode
    (``off``/``suggest``/``auto``/``check``). ``conflicts`` are the detected
    conflicts (empty unless detection ran). ``superseded_id`` is the id of the
    memory that was tombstoned, if this write superseded one. ``conflict_error``
    carries a human message when conflict detection raised — the write still
    proceeds, mirroring the pre-extraction behavior.
    """

    memory: Memory | None
    wrote: bool
    mode: str
    conflicts: list["Conflict"] = field(default_factory=list)
    superseded_id: str | None = None
    conflict_error: str | None = None


def remember(
    engine: RetrievalEngine,
    poppy_dir: Path,
    *,
    content: str,
    memory_type: str = "fact",
    project: str | None = None,
    related_to: list[str] | None = None,
    ttl: str | None = None,
    expires_at: str | None = None,
    supersedes: str | None = None,
    check_conflicts: bool = False,
    auto_supersede: bool = False,
    source: str = "manual",
    now: datetime | None = None,
    cfg: "PoppyConfig | None" = None,
) -> RememberResult:
    """Run the complete remember flow: build → conflict-resolve → write → sync.

    Raises ``ValueError`` on an unparseable/mutually-exclusive TTL and ``KeyError``
    when an explicit ``supersedes`` target does not exist — adapters translate
    both into their own error contract.
    """
    from poppy.config import load_config
    from poppy.lifecycle import resolve_expiry, supersede_memory

    now = now or datetime.now(timezone.utc)
    expiry = resolve_expiry(ttl, expires_at, now=now)

    memory = Memory(
        id=make_memory_id(),
        content=content,
        memory_type=memory_type,
        source=Source(type=source, session_id=None, timestamp=now),
        project=project,
        related_to=list(related_to or []),
        created_at=now,
        updated_at=now,
        confidence=1.0,
        expires_at=expiry,
    )

    cfg = cfg if cfg is not None else load_config(poppy_dir)
    if check_conflicts:
        mode = "check"
    elif auto_supersede:
        mode = "auto"
    else:
        mode = cfg.auto_supersede  # off | suggest | auto

    conflicts: list["Conflict"] = []
    conflict_error: str | None = None
    if mode != "off" and not supersedes:
        from poppy.capture.reconciler import detect_conflicts, pick_auto_supersede

        try:
            conflicts = detect_conflicts(engine, memory, cfg=cfg)
        except Exception as exc:
            conflicts = []
            conflict_error = f"conflict detection failed: {exc}"

        if mode == "auto":
            picked = pick_auto_supersede(conflicts)
            if picked is not None:
                supersedes = picked.memory.id

    if mode == "check":
        # Dry run: never write, whether or not an explicit supersedes was passed.
        # "check" means check — tombstoning a supersede target during a
        # "conflict check" would be a silent, destructive misreport.
        return RememberResult(memory=None, wrote=False, mode=mode, conflicts=conflicts, conflict_error=conflict_error)

    superseded_id: str | None = None
    if supersedes:
        result = supersede_memory(engine, memory, supersedes, poppy_dir=poppy_dir)
        superseded_id = result.old_id
    else:
        engine.ingest(memory)

    _emit_memory_write(poppy_dir, memory)
    _trigger_autosync(poppy_dir)

    return RememberResult(
        memory=memory,
        wrote=True,
        mode=mode,
        conflicts=conflicts,
        superseded_id=superseded_id,
        conflict_error=conflict_error,
    )


@dataclass
class ForgetResult:
    """Outcome of a forget flow, for adapters to render.

    ``memory`` is the memory that was tombstoned+deleted, or ``None`` when the id
    was not live. In the not-live case ``already_tombstoned`` tells whether a
    tombstone already exists for it (an idempotent re-delete), and ``tombstone``
    is that existing tombstone. On a real delete ``tombstone`` is the freshly
    added one and carries the restore-window metadata.
    """

    deleted: bool
    tombstoned: bool
    memory: Memory | None = None
    tombstone: "Tombstone | None" = None
    already_tombstoned: bool = False


def forget(
    engine: RetrievalEngine,
    poppy_dir: Path,
    memory_id: str,
    *,
    reader: RetrievalEngine | None = None,
    tombstones: "TombstoneStore | None" = None,
) -> ForgetResult:
    """Tombstone-then-delete a memory, then trigger autosync on success.

    The tombstone makes the delete durable (restorable for the 7-day window) AND
    propagates via sync: push only sends deletions it finds as tombstones, so a
    bare ``engine.delete`` is invisible to sync and the still-live cloud row
    resurrects on the next pull. This is the one place that invariant
    lives.

    ``reader`` defaults to ``engine`` but lets the UI read existence through its
    fast FTS-only engine while deleting through the heavy write engine.
    ``tombstones`` lets a caller reuse an already-open store.
    """
    from poppy.db import write_gate
    from poppy.engine._legacy_copies import is_marked_copy
    from poppy.ui.tombstones import TombstoneStore

    reader = reader if reader is not None else engine
    store = tombstones if tombstones is not None else TombstoneStore(poppy_dir / "memories.db")

    # Read, tombstone and delete under the write gate, the same lock restore and
    # pull take. Interleaved, the two flows destroy the memory: forget reads the
    # live row and writes its tombstone, restore sees both and clears that
    # tombstone as stale, then forget deletes the row — no memory, no tombstone.
    with write_gate(poppy_dir):
        mem = reader.get(memory_id)
        if mem is None:
            existing = store.get(memory_id)
            return ForgetResult(
                deleted=False,
                tombstoned=False,
                memory=None,
                tombstone=existing,
                already_tombstoned=existing is not None,
            )

        # Rows retained from older releases must be deleted without snapshotting
        # their speaker text. The frozen classifier also protects unmarked copies
        # left by an older client sharing the store.
        if is_marked_copy(engine, memory_id) or store.claim_proven_unmarked_copy(memory_id):
            # An older client may already have left a Trash snapshot. Clear it
            # while the live row still supplies the provenance needed to grade it.
            store.clear_copy_snapshot(memory_id)
            deleted = engine.delete(memory_id)
            if deleted:
                store.add_copy_deletions([memory_id])
            ts = None
            tombstoned = deleted
        else:
            # engine.delete also clears the derived per-speaker copies and
            # records their content-free tombstones, so the deletion of any
            # cloud copy travels without this flow tracking their ids.
            # Snapshot the memory before the live row goes. Push checks the
            # per-ID remote provenance recorded by uploads, pulls, or upgrade.
            ts = store.add(mem)
            deleted = engine.delete(memory_id)
            tombstoned = True

    if deleted:
        _trigger_autosync(poppy_dir)
    return ForgetResult(deleted=deleted, tombstoned=tombstoned, memory=mem, tombstone=ts)


@dataclass
class RestoreResult:
    """Outcome of a restore flow, for adapters to render.

    ``found`` is whether a tombstone existed at all (a 404 for the UI when not).
    ``memory`` is the live row afterwards — freshly restored, or the pre-existing
    one when the id was already live. ``expired`` marks the one case with no live
    row: the memory's own TTL ran out while it sat in the tombstone table, so the
    tombstone is cleared and nothing is re-ingested. ``raced`` means another
    writer resolved this tombstone while the row was being written: the restore
    still stands, but a tombstone left by that writer may sit beside it until the
    next sync or delete resolves the pair.
    """

    found: bool
    memory: Memory | None = None
    expired: bool = False
    already_live: bool = False
    raced: bool = False


def restore(
    engine: RetrievalEngine,
    poppy_dir: Path,
    memory_id: str,
    *,
    tombstones: "TombstoneStore | None" = None,
    now: datetime | None = None,
) -> RestoreResult:
    """Bring a tombstoned memory back as a live row, then trigger autosync.

    The restored row is stamped with an ``updated_at`` strictly later than the
    tombstone's ``tombstoned_at``. That is the whole point: push only sends rows
    above the ``last_pushed_at`` watermark, so re-ingesting the memory with its
    original ``updated_at`` left the restore purely local — the cloud row stayed
    soft-deleted and the next pull re-applied that tombstone, deleting the
    memory again. A fresh ``updated_at`` makes push carry it as a live
    upsert (``deleted_at: null``), which the server's ``upsert_memory_with_quota``
    resolves as an un-delete, and makes pull's freshness guards keep the local
    live row instead of the older server tombstone. Wall-clock time is not
    trusted for that ordering: an NTP step or a VM resume can put ``now`` behind
    ``tombstoned_at``, so the stamp is ``max(now, tombstoned_at + 1µs)``.

    A memory whose own TTL elapsed while it was tombstoned is NOT brought back.
    Restoring it live would either hide it anyway (an expired row is filtered
    from list/retrieve and from push) or, if the expiry were cleared, keep data
    the user had scheduled to disappear — the wrong answer for a memory product
    that promises expiry. Nothing is ingested and the tombstone is left where it
    is, to age out of the 7-day window on its own; a recovery action must not be
    what destroys the last copy. The caller is told the TTL elapsed. A
    still-future expiry is kept exactly as it was.

    Returns a :class:`RestoreResult`; ``found=False`` when no tombstone exists.
    """
    from poppy.db import write_gate
    from poppy.lifecycle import refuse_if_legacy_copy
    from poppy.ui.tombstones import TombstoneStore

    store = tombstones if tombstones is not None else TombstoneStore(poppy_dir / "memories.db")

    # The whole read-decide-write sequence runs under the write gate so a pull in
    # the sync worker, or a concurrent forget, cannot land between the check and
    # the ingest.
    with write_gate(poppy_dir):
        ts = store.get_public(memory_id)
        if ts is None:
            return RestoreResult(found=False)

        # Defence in depth. Nothing writes a per-speaker copy into ui_tombstones
        # any more, but a tombstone left by an older build could still name one,
        # and restoring it would re-ingest the copy as an ordinary memory —
        # unmarked, listed, and pushed live with the redacted text.
        refuse_if_legacy_copy(engine, memory_id, "restore")

        live = engine.get_public(memory_id)

        # That guard reads the LIVE row, and the case it cannot see is the one
        # with no live row at all: a 0.2.4 client forgot a copy and KEPT its
        # parent, so the entry sits there with the speaker text and nothing at
        # the id to be marked. The SNAPSHOT's own provenance is what decides,
        # graded against the parent that is still here. PROVEN only — a lesser
        # grade is a memory of the user's and comes back.
        #
        # ONLY when nothing is live at the id. With a live row the question is not
        # what the Trash entry is: the already-live branch below simply clears the
        # entry, and it has to stay that way. Grading first deleted the entry and
        # queued a cloud cleanup stamped with ITS deletion time — against the id of
        # an independent note the user or an importer had written there, which push
        # then overwrote with the placeholder on every device.
        if live is None:
            parent = store.refuse_restorable_copy_snapshot(memory_id)
            if parent is not None:
                raise ValueError(
                    f"{memory_id} is a derived per-speaker copy of {parent}, not a memory in its own "
                    "right; it cannot be restored on its own"
                )

        now = now or datetime.now(timezone.utc)
        stamp = max(now, ts.tombstoned_at + timedelta(microseconds=1))
        expiry = ts.memory.expires_at

        if live is not None:
            # Never clobber a live row with the tombstoned snapshot: it would undo
            # an edit or a re-create. Just clear the tombstone that shouldn't be
            # there.
            store.remove(memory_id, token=ts.token)
            result = RestoreResult(found=True, memory=live, already_live=True)

        elif expiry is not None and expiry <= stamp:
            # Left standing, not destroyed. The tombstone is the only copy of the
            # memory, and a recovery action must never be the thing that deletes
            # it. Expiry elsewhere works the same way: an expired memory is hidden,
            # not erased, until something purges it. So nothing is ingested and
            # nothing is removed; the tombstone ages out of the 7-day window
            # through purge_expired like any other, and stays recoverable until it
            # does. Autosync still runs below, so the tombstone reaches the cloud
            # and soft-deletes a row that would otherwise sit there live with a
            # past expires_at.
            result = RestoreResult(found=True, expired=True)

        else:
            memory = replace(ts.memory, updated_at=stamp)

            # Ingest BEFORE clearing the tombstone. Anything can fail in between
            # — a model load on a bloom store, SQLITE_BUSY past the timeout, a
            # full disk, a SIGKILL — and the tombstone is the only copy of the
            # memory left. Clearing it first would make a crash in that window
            # lose the memory outright, locally and (if the tombstone was already
            # pushed) everywhere. Writing first can only ever leave a duplicate
            # state that the next restore resolves.
            engine.ingest(memory)

            # Then clear exactly the tombstone we read. Losing that conditional
            # delete means another writer resolved this tombstone while we wrote:
            # a second restore that got there first, or a fresh delete. Either way
            # the row we just ingested stays. Deleting it to "undo" would be
            # unrecoverable — two restores racing on the unlocked path (Windows, or
            # after the write gate's timeout) would each ingest and the loser would
            # wipe the winner's row, leaving no memory and no tombstone. A live row
            # beside a stale tombstone is recoverable instead: pull drops a
            # tombstone once a newer live row exists, and the UI resolves the pair
            # by updated_at.
            raced = not store.remove(memory_id, token=ts.token)
            result = RestoreResult(found=True, memory=memory, raced=raced)

    _trigger_autosync(poppy_dir)
    return result
