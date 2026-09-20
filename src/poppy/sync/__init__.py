"""Bidirectional sync between Poppy and a Trags `memories` KV store.

The Trags side is hosted side: memories KV table + 5 CRUD endpoints). The Poppy
side reads from BOTH `memories` (live) AND `ui_tombstones` (soft-deletes
and supersede chains) so the round-trip preserves the full state.

Watermarks are tracked per remote URL in ``~/.poppy/sync_state.json`` —
push sends live rows updated after ``last_pushed_at`` and tombstones with
known IDs that have not been sent to that remote. Pull fetches
everything with ``updated_at >= last_pulled_at``.

Conflict policy: last-writer-wins by ``updated_at``. On push, we use
upsert (POST) which the Trags server resolves server-side. On pull,
local rows whose ``updated_at`` is newer than the incoming row are
preserved (the incoming row is skipped).

Watermark semantics (watermark logic): live rows are processed in ``updated_at`` ASC
order. Tombstones use sent marks and never affect this watermark. The watermark
only advances while every preceding live row has succeeded. On the first failure the watermark freezes, so the failed
item — and everything that came after it in time — is retried on the
next sync. Successful retries are idempotent on the server side.

Those comparisons are on ISO-8601 TEXT, which compares wall clock, so every
timestamp on both sides of them is spelled in ONE offset — UTC, via ``utc_iso``.
That is a spelling rule, not a semantic one: what is pushed, in what order, and
how the watermark advances are exactly as described above.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from poppy.db import write_gate
from poppy.engine._closet_marker import TIER_PROVEN, is_marked_closet, utc_iso
from poppy.engine.interface import RetrievalEngine
from poppy.sync.client import (
    TragsAuthError,
    TragsClient,
    TragsConflictError,
    TragsError,
    TragsQuotaError,
    TragsTransportError,
)
from poppy.sync.serializer import (
    deletion_time,
    is_redacted_deletion,
    is_tombstone,
    memory_to_wire,
    tombstone_to_wire,
    wire_to_memory,
)
from poppy.sync.state import (
    PUSH_WATERMARK_VERSION,
    RemoteState,
    SyncState,
    get_remote,
    latest_iso,
    load,
    mutate_remote,
    stamp_error,
)
from poppy.ui.tombstones import Tombstone, TombstoneStore

__all__ = [
    "PushResult",
    "PullResult",
    "SyncResult",
    "push",
    "pull",
    "sync",
    "remote_is_configured",
    "TragsClient",
    "TragsAuthError",
    "TragsConflictError",
    "TragsError",
    "TragsQuotaError",
    "TragsTransportError",
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
    # Incoming rows for ids this device holds as derived per-speaker copies, or
    # has just deleted as one. Never applied in either direction.
    # Defaulted so existing call sites and tests are unaffected.
    skipped_closets: int = 0
    # This store's OWN deletions, served back by the inclusive pull watermark.
    # Counted apart from `skipped_stale`, which means "local state is newer" and
    # is a different thing, and deliberately absent from what `poppy sync run`
    # prints: nothing happened, and a converged store's report says so by staying
    # at zero across the board rather than carrying a column that never clears.
    skipped_echoes: int = 0
    # Virtual Trash after a dry-run pull, consumed by the same cycle's push.
    tombstone_preview: dict[str, Tombstone | None] = field(default_factory=dict, repr=False)
    known_ids: set[str] = field(default_factory=set, repr=False)


@dataclass
class SyncResult:
    push: PushResult
    pull: PullResult


# After this many CONSECUTIVE transport failures (timeouts / connection errors),
# a push stops rather than grinding the whole backlog against a dead/black-holing
# host — each doomed upsert burns the full httpx timeout, and the worker holds
# `sync.lock` + its `writers.registered` entry (which blocks `poppy encrypt`) the
# entire time. Bounds the lock hold to ~K x timeout; the worker then backs off.
MAX_CONSECUTIVE_TRANSPORT_FAILURES = 3

logger = logging.getLogger(__name__)


def _note_transport_progress(
    exc: Exception, *, pulled: int = 0, pushed: int = 0, unknown: bool = False, simulated: bool = False
) -> Exception:
    """Record on a transport error what the run completed before the host died.

    The CLI builds its offline line from these, so a run that applied pulled
    rows, sent part of its backlog, or lost a response mid-write is never
    summarised as "nothing happened". Accumulates, because a `sync`
    that pulled before its push failed has to report both halves on the one
    error that reaches the caller. A no-op on anything else: `push` also carries
    untyped faults in the same slot, and those escape unannotated as before.

    `simulated` marks the counts as a `--dry-run`'s. Both loops count what they
    WOULD do so the CLI can show it, and those numbers are not evidence that
    anything happened - the caller has to know which kind it is holding.
    """
    if isinstance(exc, TragsTransportError):
        exc.applied_pulled += pulled
        exc.sent_pushed += pushed
        exc.outcome_unknown = exc.outcome_unknown or unknown
        exc.simulated = exc.simulated or simulated
    return exc


def _remote_kwargs(engine: RetrievalEngine, when: datetime | None) -> dict:
    """``remote_event_ts`` for engines that take it, nothing for those that do not.

    Sync applies another device's writes through the ordinary ``ingest`` and
    ``delete``, and the engine has to date anything it removes from the event
    rather than from this device's clock. Passing it unconditionally would break
    an engine that predates the argument, and catching TypeError would swallow
    real ones from inside the call, so the engines advertise it instead.
    """
    if when is None or not getattr(engine, "accepts_remote_event_ts", False):
        return {}
    return {"remote_event_ts": when}


def _note_leaked_copy(engine: RetrievalEngine, incoming: object) -> bool:
    """Ask the engine to queue a cleanup announcement for a leaked copy row.

    Duck-typed like the other closet seams: an engine with no copy expansion has
    nothing to recognise. No error tolerance — a raising engine must surface
    rather than quietly leave leaked text in the cloud.
    """
    note = getattr(engine, "note_leaked_cloud_copy", None)
    if note is None:
        return False
    return bool(note(incoming))


def _fresh_row_to_push(engine: RetrievalEngine, memory: object) -> object | None:
    """Re-read a row snapshotted by push's candidate scan: the row to send, or None.

    ``list_all`` materialises every candidate before the first upsert, and the
    uploads that follow take as long as the network does. A forget landing in that
    window is invisible: its own tombstone is newer and the cloud converges on the
    next cycle, but the row is uploaded once MORE after the user deleted it — and
    for a legacy unmarked copy of a multi-speaker memory that means the speaker
    text reaching the cloud after the parent was redacted.

    Re-read immediately before the upsert, so the window is a database read rather
    than a whole push. Two ways a candidate stops being a memory to send: the row
    is gone, or the id is now MARKED derived data (a bloom ingest of the parent
    re-derived a copy there). Both are the same question ``list_all`` answered at
    scan time, asked again at the last possible moment.

    The FRESH row is what gets sent, not the scanned one. An edit landing in the same
    window (redacting text, say) would otherwise put the old text on the wire once
    more, with the newer version following a cycle later.

    Nothing about the watermark changes: it still advances to the SCANNED timestamp,
    never the fresh row's. Advancing to a stamp no candidate list was built from
    would skip a sibling written in between; re-sending the fresh row next cycle is
    idempotent. A row that has since gone was pushed as far as it ever will be, and
    its deletion travels as a tombstone with a strictly later timestamp of its own.

    Errors are NOT swallowed. The caller treats a failure here as a local read
    failure: recorded, watermark frozen, retried next cycle.
    """
    memory_id = memory.id  # type: ignore[attr-defined]
    fresh = engine.get(memory_id)  # type: ignore[attr-defined]
    if fresh is None or is_marked_closet(engine, memory_id):
        return None
    return fresh


def _closet_deletion_wins(
    engine: RetrievalEngine,
    tombstones: TombstoneStore,
    memory_id: str,
    incoming: object,
    *,
    dry_run: bool,
) -> bool:
    """Whether a recorded copy-deletion should suppress an incoming row for this id.

    Only when the deletion is the newer fact — the same last-writer-wins rule the
    ui tombstones get. A cloud row at or below the deletion's timestamp is the
    stale copy we removed, and applying it would bring the redacted speaker text
    back as an ordinary memory. A STRICTLY NEWER row is a legitimate recreation
    of the id: another device wrote an independent note there, and it must be
    ingested, because the watermark advances past it either way and a skip loses
    it for good. That ingest clears the deletion record as part of reclaiming the
    id, so the suppression does not come back.

    A live local row means the id already belongs to a real memory, so ordinary
    pull rules apply and nothing is suppressed.

    The ANNOUNCEMENT CLAIM is an EXISTENCE HINT, never a comparator. It says this id
    was once PROVEN to be a leaked copy here; it does not say when anything happened,
    because it can be advanced from an incoming row's own ``updated_at``
    (``note_leaked_cloud_copy``) and a 0.2.4 client could put any stamp there. Used as
    a floor, a claim dated in the future hid every honestly-stamped row written at
    that id until that date passed — and since the queue is never purged, hid them for
    ever on this device, with the pull watermark moving past them. Clamping it to now
    did not help: honest rows are stamped at or before now.

    So when there is a claim and no deletion record left, the decision is by EVIDENCE
    rather than by clock: grade the incoming row against its LIVE PARENT with the same
    tiered test used everywhere. PROVEN means this row IS the leaked copy, so it is
    suppressed however it is stamped. Anything else — a real note at that id, or no
    local parent to prove anything against — is ingested exactly as it was before this
    fix. That is what the record's seven-day lifetime used to bound, done positively.

    Why the claim is consulted at all: the record ages out seven days after the
    deletion once the announcement has landed, while the cloud row survives until the
    server applies that delete. So a row this pull captured before it waited for the
    write gate can find the record already purged by a concurrent push. With the
    parent still here the evidence is available and the copy stays out.

    A row that grades PROVEN under a claim is suppressed WHATEVER it is stamped, record
    or no record. Comparing it with the record's own timestamp admitted a republished
    leak: a 0.2.4 client re-pushing the copy after the record had aged out gave it a
    newer stamp, and pull ingested the speaker text as an ordinary memory and dropped
    the claim reclaiming the id.

    And suppressing is not enough on its own: the cloud is still holding that row. So
    the announcement is RE-ARMED at the stamp just seen, exactly as
    ``note_leaked_cloud_copy`` does for a marked copy — the sighting proves both that
    the leak is still up there and what timestamp it now carries. Skipping without
    re-arming is what left the cloud copy live for good, which main did not do.
    """
    # Side-table lookups first, and the ROW only if one of them hit: on a large
    # first pull this runs for every incoming row, and the engine read is the
    # expensive one.
    deletion = tombstones.get_closet(memory_id)
    claim = tombstones.announced_copy_claim(memory_id)
    if deletion is None and claim is None:
        return False
    if engine.get(memory_id) is not None:  # type: ignore[attr-defined]
        return False

    incoming_updated_at: datetime = incoming.updated_at  # type: ignore[attr-defined]
    by_record = deletion is not None and incoming_updated_at <= deletion.tombstoned_at
    if claim is None:
        return by_record
    if tombstones.grade_copy_snapshot(incoming) != TIER_PROVEN:  # type: ignore[arg-type]
        # Not provably the leak: the claim contributes nothing, and the record's own
        # rule decides exactly as it did on main.
        return by_record
    if not dry_run:
        # The sighting IS the proof, for the same reasons a marked copy's is: the leak
        # is still in the cloud, and this is the stamp it carries now.
        tombstones.claim_leaked_copy(memory_id, incoming_updated_at)
    return True


def push(
    *,
    engine: RetrievalEngine,
    tombstones: TombstoneStore,
    client: TragsClient,
    state: SyncState,
    poppy_dir: Path,
    dry_run: bool = False,
    already_authenticated: bool = False,
    tombstone_preview: dict[str, Tombstone | None] | None = None,
    known_ids_preview: set[str] | None = None,
) -> PushResult:
    """Send local memories + tombstones to Trags.

    Live rows newer than ``last_pushed_at`` are sent, with the watermark frozen
    strictly below the first failed LIVE row. Deletions travel only for IDs the
    remote is known to hold, from successful sends, live pulls or the one-time
    legacy backfill, and only until acknowledged. Tombstones never move or freeze
    the live watermark. They carry the full snapshot, including any local edits.
    Residuals: a lost upload response stays unknown until a pull sees the row;
    incremental pull cannot see it below the pull cursor. Deletions include
    edits never pushed. Pre-upgrade memories are treated as known, so the full
    guarantee covers memories created after upgrade and stores with no remote
    at upgrade.

    ``already_authenticated`` says a request has ALREADY succeeded against this
    remote with this client in this same cycle — ``sync()`` sets it after a clean
    pull — so a push with nothing to send skips its own probe rather than buying
    the same proof a second time.
    """
    remote = get_remote(state, client.base_url)

    # The watermark NEVER MOVES PAST NOW. A row stamped in the future — a skewed
    # clock, a bad import — is a real local write and is still sent, but letting it
    # carry the watermark to 2030 puts every later `remember`, edit and restore
    # below the mark, and the ``iso <= watermark`` filter then skips them silently
    # for ever: one bad row stops the device pushing anything again.
    #
    # Holding the mark at the newest row that HAS ALREADY HAPPENED keeps the scalar
    # honest — no per-row sent set, no hole for a single string to express. The
    # price is re-sending, on every push, every candidate the mark will not move
    # past: the future row itself, and — on a device whose own clock LAGS — the
    # whole set of rows just pulled from other devices, which carry those devices'
    # correct and locally still-future stamps. That worst case is a SET, not a row.
    # Both are bounded by the skew and end when it does; upserts are idempotent and
    # the server's freshness gate is the arbiter, so the cost is requests only.
    now = datetime.now(timezone.utc)

    def carries_watermark(iso: str) -> bool:
        """Whether ``iso`` may BECOME the watermark: a real instant, already past.

        PARSED, never compared as text. ``isoformat()`` omits ``.ffffff`` when the
        microsecond is exactly 0, and ``'.'`` sorts above ``'+'``, so a text compare
        against now reads an honest same-second row as future about once in a
        million pushes — costing either the mark for that push or a needless full
        re-push. Parsing removes the class instead of narrowing it.

        A stamp that does not parse is refused outright. ``utc_iso`` passes such a
        value through verbatim, and any non-digit first character sorts above
        ``2026…``, so letting one carry the mark would hand the watermark a string
        every real row sorts under — exactly the silence this fix exists to remove.
        The bounded cost taken instead: that row alone is re-upserted once per push,
        for as long as the store holds a corrupt ``updated_at``. Rejecting such a
        value belongs at the write boundary (``lifecycle.parse_expires_at``), not
        here, where the only safe move is to refuse it the watermark.
        """
        try:
            stamp = datetime.fromisoformat(iso)
        except (TypeError, ValueError, OverflowError):
            return False
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp <= now

    def advanced(current: str | None, iso: str) -> str | None:
        """``latest_iso``, except a stamp that cannot carry the watermark leaves it."""
        return latest_iso(current, iso) if carries_watermark(iso) else current

    # ONE full pass, per remote, when the watermark cannot be trusted to mean what
    # it says. Decided before the watermark is used for anything.
    #
    # The watermark records how far push got under the OLD text ordering, and it is
    # a single string: it cannot say WHICH rows above it had already succeeded. Two
    # things invalidate it, and neither can be repaired by re-spelling it:
    #
    #   * the UTC rewrite moved a push candidate. A row that failed and was being
    #     retried BECAUSE it sorted above the watermark (`11:00+02:00` over a
    #     watermark of `10:00+00:00`) drops BELOW it once spelled `09:00+00:00`, and
    #     is then skipped for ever with push reporting no errors. The store records
    #     that event once and never clears it; what each REMOTE has done about it is
    #     `repush_done_for`, written by the same state save as the watermark below,
    #     so there is no window in which one has landed and the other has not.
    #   * the stored watermark itself carries a non-UTC offset, which needs no
    #     rewrite to hide a row: `10:00-05:00` is 15:00Z, and every canonical row
    #     between those two readings sorts under it as text. Re-spelling the
    #     watermark to `15:00+00:00` would CONFIRM the skip rather than undo it, so
    #     that spelling is not trusted either — one pass, then the saved watermark
    #     is canonical and this never fires for this remote again.
    #
    # Upserts are idempotent and the server's freshness gate rejects stale rows, so
    # the cost is one pass of requests and nothing else.
    repush_stamp = tombstones.repush_stamp()
    stored_watermark = remote.last_pushed_at
    canonical_watermark = utc_iso(stored_watermark)
    owes_repush = repush_stamp is not None and remote.repush_done_for != repush_stamp
    watermark_respelled = canonical_watermark != stored_watermark
    # A THIRD, and the reason it is a RECORDED MARKER and not a reading of the
    # value: a watermark left above now by a client that let a future-dated row
    # carry it hides every honestly stamped write underneath it — and it STOPS
    # LOOKING WRONG the moment the clock passes it. At 12:00 the old client pushes
    # a row dated 12:05 and the mark jumps there; a write at 12:01 falls under it
    # and is skipped; by 12:06 the mark is an ordinary past timestamp, nothing in
    # it reads as wrong any more, and that write is stranded for ever with push
    # reporting no errors. A poisoned `12:05` and an honest one are one string. So
    # the state records WHICH CLIENT wrote the mark, and a remote below
    # `PUSH_WATERMARK_VERSION` buys exactly one full pass — the same pass, for the
    # same reason, as the two above.
    owes_watermark_pass = remote.push_watermark_v < PUSH_WATERMARK_VERSION
    # And the live guard behind that migration: a stored watermark that CANNOT BE
    # TRUE — an instant that has not happened, or text that is not a timestamp at
    # all. This client's own watermark is never either, so once the pass above is
    # made the ways back are a downgrade and up again, or a merge taking the max
    # over a pre-fix duplicate key. Rare, and neither is worth trusting.
    watermark_unusable = canonical_watermark is not None and not carries_watermark(canonical_watermark)
    # A dry run READS all four, so `sync --dry-run` reports the pass that is
    # actually coming; it writes nothing, so it consumes none of them.
    watermark = (
        None
        if (owes_repush or watermark_respelled or owes_watermark_pass or watermark_unusable)
        else canonical_watermark
    )
    # Error-slot GENERATIONS this cycle STARTED with. Clearing compares against
    # these so a clean push only resolves the exact failures it saw — never one a
    # concurrent sync recorded after this snapshot, even if the text is identical.
    gens_at_start = dict(remote.error_gens)

    sent_live = 0
    sent_tombstones = 0
    accepted_ids: set[str] = set()
    accepted_tombstones: list[Tombstone] = []
    # Sends this push will have to make AGAIN next time: rows whose stamp can never
    # carry the watermark, so nothing on disk will record them as done. Deducted
    # from the cumulative pushed total below.
    resent = 0
    skipped = 0
    errors = 0
    requests_made = 0
    new_watermark = watermark
    watermark_locked = False
    auth_exc: TragsAuthError | None = None
    quota_exc: TragsQuotaError | None = None
    transport_exc: Exception | None = None
    transport_failed = False
    # True once ANY failure this push left its request's fate unknowable (a lost
    # response, not a refused connection). Sticky across rows, because the LAST
    # transport error is the one that gets raised and it may well be the one
    # that never reached the host, while an earlier row's write did land.
    unknown_outcome = False
    consecutive_transport = 0
    last_soft_error: str | None = None
    # Whether the key was proved good this cycle without sending a memory —
    # either by the probe below, or by the pull that ran just before it.
    probe_ok = False
    # A failed probe's message. Deliberately NOT ``last_soft_error`` (which is
    # the push slot's text): the probe sent nothing, so it neither stamps nor is
    # stamped over a real row failure.
    probe_error: str | None = None
    # ISO of the first LIVE row that failed. Everything processed in ``updated_at``
    # ASC order, so this is the smallest failed timestamp; the watermark is
    # frozen STRICTLY BELOW it (see below) so a sibling row sharing that exact
    # timestamp is never skipped by the ``iso <= watermark`` candidate filter.
    first_fail_iso: str | None = None

    # Collect LIVE candidates strictly above the watermark. ``iso <= watermark`` is
    # correct precisely BECAUSE ``first_fail_iso`` freezes the watermark below any
    # failed timestamp: no row at exactly ``last_pushed_at`` is ever left unsent,
    # so there is nothing to recover by re-sending the boundary. A ``<`` filter
    # would instead re-upsert every boundary row on every push forever (one paid
    # write per local write) and could RESURRECT a cross-device deletion (re-send
    # a stale row the server upserts back to deleted_at=null).
    candidates: list[tuple[str, str, object]] = []

    # Canonical UTC spelling keeps the live filter and candidate ordering
    # independent of the offset an importer supplied.
    for memory in engine.list_all(filters=None, limit=10**9):
        if dry_run and tombstone_preview and tombstone_preview.get(memory.id) is not None:
            continue  # The dry-run pull would have deleted this live row.
        iso = utc_iso(memory.updated_at)
        if watermark and iso <= watermark:
            skipped += 1
            continue
        candidates.append((iso, "live", memory))

    # A deletion travels as a TOMBSTONE, and a tombstone on the wire is the
    # memory's whole body plus a `deleted_at` — the body is load-bearing, because
    # cross-device Trash restore rebuilds its entry from the remote row. Sent to a
    # remote that holds nothing at that id, the upsert CREATES the row it means to
    # remove. So a store's whole deletion history, every one of them carrying its
    # text, used to go up on the first push after `poppy setup trags`: memories
    # that had only ever lived on this machine reached the cloud for the first
    # time BECAUSE the user deleted them.
    #
    # Evidence belongs to the ID and remote, not a forget-time snapshot or a
    # watermark. A later pull of this ID can heal an unacknowledged live upload.
    known_ids = tombstones.known_ids(client.base_url)
    if dry_run and known_ids_preview:
        known_ids.update(known_ids_preview)
    remote_url = client.base_url.rstrip("/")
    local_tombstones = {ts.memory.id: ts for ts in tombstones.list_all()}
    if dry_run and tombstone_preview:
        local_tombstones.update(tombstone_preview)
    for ts in local_tombstones.values():
        if ts is None:
            continue
        iso = utc_iso(ts.tombstoned_at)
        if ts.memory.id not in known_ids or remote_url in ts.sent_remotes:
            skipped += 1
            continue
        candidates.append((iso, "tomb", ts))

    # Oldest first; only live candidates participate in the watermark.
    candidates.sort(key=lambda c: c[0])

    for iso, kind, payload in candidates:
        if dry_run:
            if kind == "live":
                sent_live += 1
            else:
                sent_tombstones += 1
            if kind == "live" and not watermark_locked:
                new_watermark = advanced(new_watermark, iso)
            continue

        # Attempt EVERY candidate. The server only quota-gates a NEW-live CREATE
        # (trags `upsert_memory_with_quota` gates `v_creates_new_live`); a POST
        # that UPDATES an already-synced memory (e.g. a sensitive-text edit) and a
        # TOMBSTONE are never gated and still land. A still-capped create just
        # re-402s and `continue`s below, keeping the watermark frozen; this
        # SELF-CONVERGES the moment the cap lifts (a create 201s next push) with
        # no per-id state to invalidate. Amplification (a capped free user
        # auto-pushing all N doomed creates on every write) is bounded at the
        # worker level by a quota backoff, not here.
        if kind == "live":
            # The re-read touches the LOCAL store, so its failures are not the
            # network's: counted as a transport fault they would trip the circuit
            # breaker, skip the announcement drain and re-raise a raw sqlite error at
            # the CLI. Recorded in the push slot with the watermark frozen instead,
            # which is what every other per-item failure gets. Raised out of the loop
            # (where it used to sit) it left push before `_persist` and lost all
            # three.
            try:
                payload = _fresh_row_to_push(engine, payload)
            except Exception as exc:
                # sqlite3.Error, or SQLCipher's own Error class on an encrypted
                # store — a different hierarchy, hence not a typed except. Nothing
                # in this call touches the network.
                errors += 1
                watermark_locked = True
                last_soft_error = f"local read failed: {exc}"
                if first_fail_iso is None:
                    first_fail_iso = iso
                continue
            if payload is None:
                # Deleted (or re-derived as a copy) since the candidate scan. Not an
                # error and not a failure: nothing to send, and the watermark moves
                # on exactly as it would have.
                skipped += 1
                if not watermark_locked:
                    new_watermark = advanced(new_watermark, iso)
                continue

        try:
            requests_made += 1
            if kind == "live":
                client.upsert(memory_to_wire(payload))  # type: ignore[arg-type]
                sent_live += 1
            else:
                client.upsert(tombstone_to_wire(payload))  # type: ignore[arg-type]
                sent_tombstones += 1
                accepted_tombstones.append(payload)  # type: ignore[arg-type]
            accepted_ids.add(payload.id if kind == "live" else payload.memory.id)
            consecutive_transport = 0  # any success resets the circuit breaker
        except TragsAuthError as exc:
            # A revoked/invalid key fails every remaining upsert too. Freeze the
            # watermark and stop the push rather than hammering the server once
            # per pending item; re-raised below after state is persisted so the
            # caller can surface it and stop retrying.
            errors += 1
            if kind == "live":
                watermark_locked = True
            auth_exc = exc
            if kind == "live" and first_fail_iso is None:
                first_fail_iso = iso
            break
        except TragsQuotaError as exc:
            # The account hit its memory cap on a new-live CREATE. Record it and
            # freeze the watermark, but KEEP DRAINING: TOMBSTONES (deletions) and
            # UPDATES to already-synced memories consume no quota and still land
            # this push. Re-raised below so the caller shows the upgrade prompt
            # once; the frozen watermark retries the deferred creates next push.
            errors += 1
            if kind == "live":
                watermark_locked = True
            quota_exc = exc
            consecutive_transport = 0  # a 402 is a response, not a transport fault
            if kind == "live" and first_fail_iso is None:
                first_fail_iso = iso
            continue
        except TragsError as exc:
            # `TragsConflictError` included: it subclasses `TragsError`, and a 409
            # is a server response like any other per-row refusal.
            errors += 1
            if kind == "live":
                watermark_locked = True
            last_soft_error = str(exc)
            if kind == "live" and first_fail_iso is None:
                first_fail_iso = iso
            if not isinstance(exc, TragsTransportError):
                consecutive_transport = 0  # a server response, not a transport fault
                continue
            # A wrapped httpx transport fault: the HOST is the problem,
            # not this row, so it gets exactly what the untyped transport failures
            # below get (flagged for retry, bounded by the circuit breaker) rather
            # than being read as a per-row refusal from a live host.
            transport_exc = exc
            transport_failed = True
            unknown_outcome = unknown_outcome or exc.outcome_unknown
            consecutive_transport += 1
            if consecutive_transport >= MAX_CONSECUTIVE_TRANSPORT_FAILURES:
                break
            continue
        except Exception as exc:
            # An unexpected error — a transport failure like httpx.ReadTimeout is
            # NOT a TragsError. Record it and KEEP DRAINING so a later TOMBSTONE
            # (a deletion) still gets its attempt, UNLESS we hit
            # MAX_CONSECUTIVE_TRANSPORT_FAILURES in a row — then the host is likely
            # dead and we STOP (circuit breaker) instead of grinding the backlog
            # and holding the worker lock for ~N x timeout. The persist block
            # records what we saw (quota keeps precedence for the push slot); the
            # transport failure is flagged so the caller can retry it even when a
            # quota was also seen.
            errors += 1
            if kind == "live":
                watermark_locked = True
            last_soft_error = str(exc)
            transport_exc = exc
            transport_failed = True
            # Unrecognised, so it cannot prove the request did not land either.
            unknown_outcome = True
            consecutive_transport += 1
            if kind == "live" and first_fail_iso is None:
                first_fail_iso = iso
            if consecutive_transport >= MAX_CONSECUTIVE_TRANSPORT_FAILURES:
                break
            continue

        if kind == "live" and not carries_watermark(iso):
            # Sent, and it will be sent AGAIN on every push until the clock reaches
            # it — or for ever, if the stamp is not a time at all. Real work for the
            # server, none for this device: the watermark is this device's whole
            # ledger of what is done, and this row will never be in it. Kept out of
            # the cumulative total rather than drifting it upward on an idle store,
            # and counted once, later, if the clock ever catches up.
            resent += 1
        if kind == "live" and not watermark_locked:
            new_watermark = advanced(new_watermark, iso)

    if first_fail_iso is not None:
        stop_iso = first_fail_iso
        # A row at ``stop_iso`` failed, so only rows STRICTLY BELOW
        # it are proven pushed. Freeze the watermark at the largest candidate iso below
        # it — never at the failing timestamp itself. Otherwise a same-tick
        # success would advance the watermark onto that timestamp and its failed
        # sibling would be skipped forever by the ``iso <= watermark`` filter.
        # Everything at ``stop_iso`` (including any success there, re-sent
        # idempotently) then retries on the next push.
        # Restricted to candidates that may carry the mark at all, for the same
        # reason the advance checks it: a future-dated or unparseable candidate
        # below the failure would otherwise freeze the watermark in 2030.
        below = [
            c_iso for c_iso, kind, _ in candidates if kind == "live" and c_iso < stop_iso and carries_watermark(c_iso)
        ]
        new_watermark = max(below) if below else watermark

    # Nothing above the watermark, nothing to announce: this push has not spoken
    # to the server at all, and without a probe it reports `0 live, 0 tombstones,
    # N skipped, 0 errors` and exits 0 against a revoked key or a deleted account.
    # One cheap authenticated read settles that. Its failure is recorded (auth in
    # its own slot via the re-raise below, everything else in the PROBE slot), so
    # `poppy sync status` shows a `last error:` line either way.
    #
    # A transport failure here is NOT flagged for the caller's re-raise: there is
    # no pending row to retry, and an offline machine should get a recorded error
    # and a non-zero exit, not a raw traceback out of `poppy sync push`.
    if not dry_run and requests_made == 0 and errors == 0:
        if already_authenticated:
            # The pull in this same cycle just completed an authenticated request
            # through this very client. That is the fact the probe would buy, so
            # probing again would double every idle cycle's request count — the
            # common case for the auto worker — and learn nothing new.
            probe_ok = True
        else:
            # Looked up OUTSIDE the try: a client object with no ``ping`` is a
            # programming error, not a sync failure, and swallowing it recorded
            # `probe failed: 'X' object has no attribute 'ping'` as if the server
            # had said it. Let it raise where someone will see it.
            probe = client.ping
            try:
                probe()
                probe_ok = True
            except TragsAuthError as exc:
                errors += 1
                auth_exc = exc
            except TragsQuotaError as exc:
                # A 402 on a read is not expected, but if the server ever sends
                # one, take the path the upsert loop takes: re-raised below so the
                # caller shows the upgrade prompt instead of a bare failure.
                errors += 1
                quota_exc = exc
                probe_error = str(exc)
            except TragsError as exc:
                errors += 1
                probe_error = str(exc)
            except Exception as exc:
                # A transport fault: dead host, DNS, offline laptop. ``str`` on
                # those is routinely empty, so name the class as well.
                errors += 1
                probe_error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__

    if not dry_run:
        # Durable provenance BEFORE the watermark: an interrupted state save may
        # cause retries, but can never strand a deletion of an accepted ID.
        tombstones.note_remote_memories(accepted_ids, client.base_url)
        tombstones.mark_sent(accepted_tombstones, client.base_url)

        def _persist(r: RemoteState) -> None:
            # Runs on a FRESH read under the state lock, so counter deltas
            # accumulate and a concurrent sync's error slots survive.
            r.last_pushed_at = new_watermark
            # The generation the watermark on disk was written under, recorded by
            # the SAME save for the same reason ``repush_done_for`` is: a save that
            # never lands leaves the pass still owed. Unconditional — a pass that
            # sent nothing (an empty store; one whose every row failed) is still a
            # pass this client made, and what the marker claims is only that the
            # watermark beside it is one THIS client wrote.
            r.push_watermark_v = PUSH_WATERMARK_VERSION
            if repush_stamp is not None:
                # Recorded by the SAME save as the watermark, which is what makes
                # the recovery safe: if this save never lands, neither does the
                # marker, and the next push does the whole pass again. A pass that
                # failed part way saves a watermark frozen below the failure
                # together with the marker, so the failed row is above it and
                # retries in the ordinary way — no second full pass needed.
                r.repush_done_for = repush_stamp
            r.pushed_count += sent_live + sent_tombstones - resent
            r.last_synced_at = datetime.now(timezone.utc).isoformat()
            if probe_error is not None:
                # The probe is the ONLY thing that can have failed here: it runs
                # solely when nothing was sent and nothing had failed, so this
                # branch can never be masking a row failure. Its own slot keeps it
                # out of the push slot, where an idle store has no upload to clear
                # it with and one transient 500 wedged every later push at exit 1.
                # A quota seen by the probe still re-raises below.
                stamp_error(r, "probe", probe_error)
            elif quota_exc is not None:
                # Record the cap in the push slot so `poppy sync status` shows it
                # and the worker log reflects it instead of a healthy `sync ok`.
                stamp_error(r, "push", str(quota_exc))
            elif auth_exc is not None:
                # Auth can fail on the pull side too; worker/CLI own that slot.
                pass
            elif errors > 0:
                # A partial/failed push keeps its own slot so `sync status` still
                # shows a push failure. Only a clean push clears the push slot.
                stamp_error(r, "push", last_soft_error or f"push failed: {errors} error(s)")
            elif requests_made > 0:
                # A clean push that ACTUALLY sent something clears its push slot,
                # the auth slot and the probe slot (a successful authenticated
                # WRITE is strictly stronger evidence than the probe's read) — but
                # ONLY the errors THIS cycle observed whose GENERATION is
                # unchanged, so a concurrently-recorded failure (newer generation,
                # even identical text) survives. It never touches the pull slot.
                #
                # The older rule here was that a no-op push clears NOTHING,
                # because a push that sent nothing had proved nothing. It now
                # sends a probe (above), so it does prove something — but only
                # about the key and the host, never about a pending row: the
                # branch below clears the auth and probe slots and still leaves
                # the push slot alone. The reversal is deliberate, not
                # a regression to restore.
                #
                # A push with any 402 lands in the quota branch above (banner
                # kept), so this only runs when nothing 402'd — i.e. the cap has
                # lifted.
                for kind in ("push", "auth", "probe"):
                    if kind in gens_at_start and r.error_gens.get(kind) == gens_at_start[kind]:
                        r.errors.pop(kind, None)
                        r.error_gens.pop(kind, None)
            elif probe_ok:
                # The probe sent no memory, so it says nothing about a frozen push
                # row and leaves the push slot alone, but it DID reach the server
                # with this key — which resolves an auth banner AND a probe
                # failure an earlier idle run recorded (same generation rule as
                # above). Otherwise a user who fixed a revoked key, or whose 5xx
                # has passed, could never clear the banner with `poppy sync push`
                # while there was nothing new to send.
                for kind in ("auth", "probe"):
                    if kind in gens_at_start and r.error_gens.get(kind) == gens_at_start[kind]:
                        r.errors.pop(kind, None)
                        r.error_gens.pop(kind, None)

        mutate_remote(poppy_dir, client.base_url, _persist)

    # State is already persisted. Quota keeps precedence as the USER-facing signal
    # (upgrade prompt / non-zero exit) — but quota and a transport failure are
    # INDEPENDENT: a row that transport-failed (especially a tombstone/deletion)
    # still needs a retry. Flag that on the quota error so the worker re-arms even
    # while it stops-on-quota; a transport error only re-raises on its own (for
    # the worker's backoff retry) when no quota was seen this push.
    if auth_exc is not None:
        raise auth_exc
    if quota_exc is not None:
        quota_exc.transport_retry = transport_failed  # type: ignore[attr-defined]
        raise quota_exc
    if transport_exc is not None:
        raise _note_transport_progress(
            transport_exc, pushed=sent_live + sent_tombstones, unknown=unknown_outcome, simulated=dry_run
        )

    return PushResult(
        sent_live=sent_live,
        sent_tombstones=sent_tombstones,
        skipped=skipped,
        errors=errors,
    )


def _stored_utc(when: datetime) -> datetime:
    """A stored timestamp as UTC-aware, ready to compare with one off the wire.

    ``TombstoneStore.add`` normalises what it writes, so every record a current
    client filed carries an offset. One left by an older build can still be naive,
    and comparing that with an incoming ``deleted_at`` raises ``TypeError`` out of
    the pull loop — losing the whole cycle over one row. Naive means UTC here: it
    is the only thing the column has ever held.
    """
    return when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc)


def _same_deletion_snapshot(record, incoming, deleted_at: datetime, superseded_by: str | None) -> bool:
    """Whether filing this snapshot would leave the Trash row saying what it says.

    Compared whole, so a column ``ui_tombstones`` gains later is covered by
    construction — except ``updated_at``, which a deletion carries BUMPED to its
    own clock (``tombstone_to_wire``, and the server's soft delete does the same)
    so the tombstone outranks the live row it replaces. The record itself
    remembers when the memory was last EDITED, an earlier time, so that bump is a
    serialization detail rather than news.

    Any other value IS news. A row whose ``updated_at`` sits after its
    ``deleted_at`` was touched since the deletion: the cloud holds a corrected
    body for a memory deleted at this same instant, and Trash has to take it or
    Restore hands the user back the draft it superseded.
    """
    if record.superseded_by != superseded_by:
        return False
    stored = record.memory
    if incoming.updated_at != stored.updated_at and not (
        incoming.updated_at == deleted_at and _stored_utc(stored.updated_at) <= deleted_at
    ):
        return False
    return replace(incoming, updated_at=stored.updated_at) == stored


def _deletion_already_held(
    tombstones: TombstoneStore,
    incoming: object,
    record,
    deleted_at: datetime | None,
    superseded_by: str | None,
) -> str | None:
    """Why a pulled deletion needs no write, or ``None`` when it has to be applied.

    ``"stale"`` — the record here is of a STRICTLY LATER deletion. ``add`` keeps
    it and drops the older snapshot, which describes an earlier life of the id,
    so the pull can only agree with it and say so.

    ``"echo"`` — this store's own deletion, coming back. The pull watermark is
    inclusive (``updated_at >= last_pulled_at``), so the newest row pulled is
    served again by every later pull: a device that deletes a memory re-reads its
    own tombstone forever. Re-recording it rewrote the row and burned a fresh
    token (the one thing that tells two deletions of an id apart) every sync, and
    reported "pull: 1 tombstones" on a store that had long since converged.

    Both answers require nothing live at the id, which is why the caller reads
    ``record`` only then: with a live row present the deletion applies whatever
    Trash holds. And neither answer survives a snapshot that grades PROVEN. That
    one is a leaked per-speaker copy, and the copy-cleanup branch in the caller
    replaces the entry with a content-free record and claims the cloud row, which
    is work rather than a no-op.
    """
    if record is None or deleted_at is None:
        return None
    stored_at = _stored_utc(record.tombstoned_at)
    if deleted_at > stored_at:
        return None
    if deleted_at < stored_at:
        outcome = "stale"
    elif _same_deletion_snapshot(record, incoming, deleted_at, superseded_by):
        outcome = "echo"
    else:
        return None
    if tombstones.grade_copy_snapshot(incoming) == TIER_PROVEN:  # type: ignore[arg-type]
        return None
    return outcome


def _apply_pulled_row(
    engine: RetrievalEngine,
    tombstones: TombstoneStore,
    row: dict,
    incoming: object,
    *,
    dry_run: bool,
    remote_url: str,
    tombstone_preview: dict[str, Tombstone | None],
    seen_deletions: list[Tombstone],
) -> str:
    """Decide and apply one pulled row. Caller holds the write gate (unless dry run).

    Returns ``"closet"`` (skipped: local derived data or a copy deletion wins),
    ``"stale"`` (skipped: local state is newer), ``"echo"`` (skipped: this store
    already holds exactly this deletion), ``"redacted"`` (a non-restorable
    deletion), ``"tombstone"`` or ``"live"`` (applied).
    Every local read here happens under
    the same gate as the write it leads to, so no concurrent forget, restore or
    edit can invalidate a decision between the two.
    """
    # A deletion in the retired format, for an id this device holds nothing live
    # at. Recorded as what it is: filing it in ``ui_tombstones`` would put a
    # restorable Trash entry in front of the user whose body is the placeholder,
    # and restoring that would create junk and push it back up.
    #
    # The live-row check is what keeps this safe, and it is not optional. The
    # shape match is a strong hint, not proof: a real memory that happened to
    # match would otherwise be deleted here with no Trash entry and its id
    # suppressed for good. With a live row present the ordinary tombstone branch
    # below runs instead, freshness checks and all.
    if is_redacted_deletion(row):
        local_live = engine.get(incoming.id)  # type: ignore[attr-defined]
        # Applied to a LIVE row only on proof that the row really is a derived
        # copy. The shape match says the deletion came from an older client; it
        # says nothing about what this device holds at the id. A real memory
        # whose body happened to match would otherwise be destroyed here with no
        # Trash entry and its id suppressed for good, so proof is the local
        # conjunctive test against the live parent, never the body.
        if local_live is None or tombstones.is_proven_unmarked_copy(incoming.id):  # type: ignore[attr-defined]
            if local_live is not None and local_live.updated_at > incoming.updated_at:
                return "stale"
            if not dry_run:
                # The DELETION's timestamp, not this device's clock, so a
                # recreation of the id written after it is not refused.
                deleted_at = deletion_time(row) or incoming.updated_at  # type: ignore[attr-defined]
                tombstones.record_local_deletion(incoming.id, deleted_at)  # type: ignore[attr-defined]
                if local_live is not None:
                    engine.delete(incoming.id, **_remote_kwargs(engine, deleted_at))  # type: ignore[attr-defined]
                previous = tombstones.get(incoming.id)  # type: ignore[attr-defined]
                if previous is not None and previous.tombstoned_at <= deleted_at:
                    tombstones.remove(incoming.id, token=previous.token)  # type: ignore[attr-defined]
            return "redacted"

    if tombstones.local_deletion_wins(incoming.id, incoming.updated_at):  # type: ignore[attr-defined]
        return "stale"

    # Newer than the record, but is it really someone reclaiming the id? A device
    # still on the older release goes on deriving these copies and publishing
    # them, and each publication carries a fresher stamp than the row this store
    # removed, so the record's own timestamp cannot tell the two apart. Taken at
    # face value the speaker text comes back as an ordinary memory, drops the
    # record, and is pushed up again.
    #
    # So it is graded against the live parent, the same conjunctive test used
    # everywhere else: provably the same derived copy is still ours to refuse,
    # and the record moves up to the stamp just seen so the next publication of
    # it is covered without grading again. Anything else is a real memory at that
    # id and takes the ordinary path below, which clears the record as it lands.
    if (
        not is_tombstone(row)
        and tombstones.local_deletion_at(incoming.id) is not None  # type: ignore[attr-defined]
        and tombstones.grade_copy_snapshot(incoming) == TIER_PROVEN  # type: ignore[arg-type]
    ):
        if not dry_run:
            tombstones.record_local_deletion(incoming.id, incoming.updated_at)  # type: ignore[attr-defined]
        return "redacted"

    # A marked closet is LOCAL-ONLY derived data: bloom re-derives it from
    # the parent on this device, so sync never applies an incoming row for
    # one, live or tombstone. A live row would overwrite the marker and turn
    # the copy into a real syncable memory that the parent's redaction then
    # misses; an incoming TOMBSTONE would delete a copy bloom legitimately
    # owns (exactly what the server-side cleanup will be sending).
    #
    # First test: a marked copy is local derived data, skipped unconditionally.
    # Second: see ``_closet_deletion_wins``.
    #
    # Both read LOCAL state, never the incoming id's shape. A real cloud
    # memory whose id merely looks like a copy is unmarked here and flows
    # through untouched — unless this device derives a copy at exactly that
    # id, in which case the local derived row wins.
    if is_marked_closet(engine, incoming.id) or _closet_deletion_wins(  # type: ignore[attr-defined]
        engine,
        tombstones,
        incoming.id,  # type: ignore[attr-defined]
        incoming,
        dry_run=dry_run,
    ):
        if not dry_run and not is_tombstone(row) and is_marked_closet(engine, incoming.id):  # type: ignore[attr-defined]
            # A LIVE cloud row at an id this device derives a copy at. If it
            # is provably a leaked copy of that same parent, the cloud is
            # holding text a client at or below 0.2.4 pushed, and nothing
            # else would ever queue it: the one-time migration does not run
            # on a store created after the fix, and adoption only fires when
            # the copy arrives BEFORE its parent. The engine applies the
            # conjunctive test, so a real memory another device wrote at that
            # id is never announced.
            _note_leaked_copy(engine, incoming)
        return "closet"

    if is_tombstone(row):
        local_live = engine.get(incoming.id)  # type: ignore[attr-defined]
        deleted_at = deletion_time(row)
        superseded_by = row.get("superseded_by")
        # Read only where the question can be answered yes: with a live row at
        # the id the deletion applies whatever Trash holds, and a row carrying no
        # deletion timestamp has nothing to compare. Everything else falls
        # through to `add`, which reads the same row itself.
        existing_record = None
        if local_live is None and deleted_at is not None:
            existing_record = tombstones.get(incoming.id)  # type: ignore[attr-defined]
        # Asked BEFORE the dry-run exit and answered from local state alone, so
        # `poppy sync run --dry-run` reports a deletion this store already holds
        # the way the real run does, instead of promising a tombstone it would
        # never apply.
        held = _deletion_already_held(tombstones, incoming, existing_record, deleted_at, superseded_by)
        if held is not None:
            if held == "echo" and remote_url not in existing_record.sent_remotes:
                # This exact deletion is already at R, including when it was
                # first learned from a different remote. Do not re-announce it.
                if dry_run:
                    tombstone_preview[incoming.id] = replace(  # type: ignore[attr-defined]
                        existing_record, sent_remotes=existing_record.sent_remotes | {remote_url}
                    )
                else:
                    seen_deletions.append(existing_record)
            return held
        # STRICTLY newer keeps the local row. An equal timestamp
        # lets the DELETION win, matching the server's own rule
        # (migration 032): an equal-timestamp delete of a live row
        # applies. The two must agree, or a row the cloud has already
        # deleted survives here and the next push re-uploads it —
        # which is exactly how a device that pulled leaked copies
        # from a 0.2.4 account re-published the redacted text, since
        # the cleanup announcement carries the copy's own timestamp
        # and so ties with the local row.
        if local_live is not None and local_live.updated_at > incoming.updated_at:  # type: ignore[attr-defined]
            return "stale"
        if dry_run:
            tombstone_preview[incoming.id] = Tombstone(  # type: ignore[attr-defined]
                memory=incoming,  # type: ignore[arg-type]
                tombstoned_at=deleted_at or incoming.updated_at,  # type: ignore[attr-defined]
                superseded_by=superseded_by,
                sent_remotes={remote_url},
            )
            return "tombstone"
        if local_live is not None:
            # engine.delete clears the parent's derived copies and
            # records their content-free tombstones itself, so a
            # deletion pulled from another device also removes any
            # cloud copy of them. It is told WHEN the deletion
            # happened, so every record it writes — for copies it had
            # marked and for unmarked ones it grades on the way — is
            # dated from the event rather than from this device's
            # clock. Receipt time would make them look newer than a
            # recreation of one of those ids that came after, and the
            # freshness rule would then skip it for good.
            deletion_ts = deletion_time(row) or incoming.updated_at  # type: ignore[attr-defined]
            engine.delete(incoming.id, **_remote_kwargs(engine, deletion_ts))  # type: ignore[attr-defined]
        # A soft-delete carrying a COPY's text. Only a 0.2.4 client deleting a
        # copy row by hand produces one, and with no local row at the id the
        # branch below filed it as an ordinary Trash entry: the user was offered
        # the speaker text as restorable, and once the parent was forgotten
        # nothing could tell what the entry was any more. Graded here, while the
        # parent is still present, a PROVEN one is recorded as what it is —
        # content-free, dated from the deletion. A likely one keeps its
        # entry; restoring it is the user's call, as it is for any row short of
        # proof.
        if local_live is None and tombstones.grade_copy_snapshot(incoming) == TIER_PROVEN:  # type: ignore[arg-type]
            copy_deleted_at = deleted_at or incoming.updated_at  # type: ignore[attr-defined]
            tombstones.add_closets([incoming.id], now=copy_deleted_at)  # type: ignore[attr-defined]
            # AND the cloud cleanup. The remote row is soft-deleted but its BODY
            # still holds the speaker text, and on main this entry went to Trash
            # where the parent's own forget graded it and queued the delete.
            # Recording it locally and stopping there left that text up there for
            # good: a device pulling it after the parent was gone graded it ORPHAN
            # and filed it as restorable. PROVEN is the tier that may claim, and the
            # claim must not be older than the row it names — the server's freshness
            # gate compares it with that row's own updated_at, so a cleanup row
            # whose deleted_at predates its updated_at would be answered
            # stale_ignored with the text still up there.
            tombstones.claim_leaked_copy(
                incoming.id,  # type: ignore[attr-defined]
                max(copy_deleted_at, incoming.updated_at),  # type: ignore[attr-defined]
            )
            # An entry may already sit at this id, and ``tombstones.add`` — what ran
            # here before — would have REPLACED it. Recording content-free beside it
            # would leave the older text restorable and pushable, so it goes; unless
            # it is strictly newer than this deletion, which is the one case add
            # would have kept (a real note deleted at that id since).
            existing_entry = tombstones.get(incoming.id)  # type: ignore[attr-defined]
            if existing_entry is not None and existing_entry.tombstoned_at <= copy_deleted_at:
                tombstones.remove(incoming.id, token=existing_entry.token)  # type: ignore[attr-defined]
            return "tombstone"
        # The DELETION's own timestamp, never this device's clock.
        # Receipt time promotes a deletion that happened yesterday
        # into a brand-new one, and the next push sends it back up as
        # newer than anything written since — overwriting an
        # independent note another device wrote at that id while this
        # pull was in flight, and leaving only the placeholder in
        # Trash. Mark it sent to its source remote at creation: a deletion
        # we merely LEARNED about is not re-announced as ours.
        tombstones.add(
            incoming,  # type: ignore[arg-type]
            superseded_by=superseded_by,
            tombstoned_at=deleted_at,
            sent_to=remote_url,
        )
        return "tombstone"

    # A local tombstone that is at least as new as the incoming live
    # row is an authoritative delete that push hasn't propagated yet
    # (sync is pull-then-push). Skip the ingest so a forgotten memory
    # cannot resurrect; push will carry the tombstone to the cloud.
    # A genuinely newer live row (re-create after delete) still wins,
    # mirroring the tombstone branch's updated_at comparison.
    local_tomb = tombstones.get(incoming.id)  # type: ignore[attr-defined]
    if local_tomb is not None and local_tomb.tombstoned_at >= incoming.updated_at:  # type: ignore[attr-defined]
        return "stale"
    existing = engine.get(incoming.id)  # type: ignore[attr-defined]
    if existing is not None and existing.updated_at > incoming.updated_at:  # type: ignore[attr-defined]
        return "stale"
    if dry_run:
        if local_tomb is not None:
            tombstone_preview[incoming.id] = None  # type: ignore[attr-defined]
        return "live"
    # Applying someone else's EDIT. If it changes the text, the
    # engine clears the copies of the old text; it is told when
    # the edit happened so those records are dated from it. Dated
    # today, a deletion that actually happened in July looks
    # newer than an independent note written at that id in
    # between, so the note is skipped and the watermark advances
    # past it for good.
    engine.ingest(incoming, **_remote_kwargs(engine, incoming.updated_at))  # type: ignore[arg-type]
    # This id now holds a real memory that beat any deletion recorded for it, so
    # the record has done its job and must not outlive the thing it described.
    tombstones.clear_local_deletion(incoming.id)  # type: ignore[attr-defined]
    if local_tomb is not None:
        # The live row won over an older local tombstone: a restore
        # (or re-create) that happened elsewhere. Drop the stale
        # tombstone, or the UI keeps showing the memory as deleted
        # and push re-uploads the tombstone, undoing the restore.
        # Conditional on the exact tombstone we compared against: a
        # delete made locally while this pull ran leaves a different
        # tombstone, which must survive and propagate on the next
        # push. Keyed on the token, since two deletes of one memory
        # can share a tombstoned_at.
        tombstones.remove(incoming.id, token=local_tomb.token)  # type: ignore[attr-defined]
    return "live"


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
    # Error-slot generations this cycle started with (see push for the rationale).
    gens_at_start = dict(remote.error_gens)

    applied_live = 0
    applied_tombstones = 0
    skipped_stale = 0
    errors = 0
    skipped_closets = 0
    skipped_echoes = 0
    new_watermark = watermark
    watermark_locked = False
    auth_exc: TragsAuthError | None = None
    transport_exc: TragsTransportError | None = None
    last_pull_error: str | None = None
    tombstone_preview: dict[str, Tombstone | None] = {}
    known_ids: set[str] = set()
    seen_deletions: list[Tombstone] = []

    # Collect + parse, then sort oldest-first. Server returns newest-first,
    # so we have to materialize before processing. The pagination itself makes
    # network calls, so guard it (push guards its per-item calls the same way):
    # a mid-pull failure must freeze the watermark and surface, never escape as a
    # raw traceback.
    candidates: list[tuple[str, dict, object]] = []
    try:
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
    except TragsAuthError as exc:
        # Revoked/invalid key: stop paginating. Persist and apply whatever was
        # already collected, then re-raise below so the caller stops retrying.
        errors += 1
        watermark_locked = True
        auth_exc = exc
    except TragsTransportError as exc:
        # The host gave no usable answer, so this is not a row the server
        # refused. Apply what we gathered, freeze the watermark, and re-raise
        # below exactly as `push` does. Recording it and returning
        # normally is not enough: the auto worker keys its retry-with-backoff off
        # the exception, a dry run persists no error for the CLI to notice, and
        # the caller needs the object to say what this run did before the host
        # went away. Swallowing it is what made an offline pull look clean.
        errors += 1
        watermark_locked = True
        last_pull_error = str(exc)
        transport_exc = exc
    except TragsError as exc:
        # Transient network/server error mid-pagination. Freeze the watermark and
        # apply what we gathered; the rest retries on the next sync.
        errors += 1
        watermark_locked = True
        last_pull_error = str(exc)

    candidates.sort(key=lambda c: c[0])

    for iso, row, incoming in candidates:
        # ONE gate per row, around the WHOLE read-decide-write sequence. Every
        # test below reads local state that another process can change: a
        # `poppy forget` of the parent between "is this id a marked copy / does
        # a copy deletion suppress it?" and the ingest would delete the copy and
        # record its deletion, and a pull that had already decided "ordinary
        # live row" would then ingest the stale cloud text over it, clearing the
        # deletion record on the way and pushing the redacted text back up on
        # the next cycle. A check made outside the gate is a hint; only the one
        # made inside it is a decision.
        gate = nullcontext() if dry_run else write_gate(poppy_dir)
        with gate:
            outcome = _apply_pulled_row(
                engine,
                tombstones,
                row,
                incoming,
                dry_run=dry_run,
                remote_url=client.base_url.rstrip("/"),
                tombstone_preview=tombstone_preview,
                seen_deletions=seen_deletions,
            )
        # Even a stale row proves this ID exists. Pulled deletions are marked
        # sent at creation, so their sightings add no pending work.
        if outcome not in {"closet", "redacted"}:
            known_ids.add(incoming.id)
        if outcome == "closet":
            skipped_closets += 1
        elif outcome in {"echo", "redacted"}:
            skipped_echoes += 1
        elif outcome == "stale":
            skipped_stale += 1
        elif outcome == "tombstone":
            applied_tombstones += 1
        else:
            applied_live += 1
        if not watermark_locked:
            new_watermark = latest_iso(new_watermark, iso)

    if not dry_run:
        tombstones.note_remote_memories(known_ids, client.base_url)
        tombstones.mark_sent(seen_deletions, client.base_url)

        def _persist(r: RemoteState) -> None:
            r.last_pulled_at = new_watermark
            r.pulled_count += applied_live + applied_tombstones
            r.last_synced_at = datetime.now(timezone.utc).isoformat()
            if auth_exc is not None:
                # Auth slot is recorded/cleared by the worker/CLI (source="auth").
                pass
            elif errors > 0:
                # Record the pull failure (a 503 mid-pagination, an unparseable
                # row) in its OWN slot so it persists like push's.
                stamp_error(r, "pull", last_pull_error or f"pull failed: {errors} error(s)")
            else:
                # A clean pull authenticated and completed, so it clears its own
                # pull slot AND the auth slot — but only the errors THIS cycle
                # observed whose GENERATION is unchanged, so a concurrent failure
                # survives. It never touches the push slot.
                for kind in ("pull", "auth"):
                    if kind in gens_at_start and r.error_gens.get(kind) == gens_at_start[kind]:
                        r.errors.pop(kind, None)
                        r.error_gens.pop(kind, None)

        mutate_remote(poppy_dir, client.base_url, _persist)

    if auth_exc is not None:
        raise auth_exc
    if transport_exc is not None:
        raise _note_transport_progress(transport_exc, pulled=applied_live + applied_tombstones, simulated=dry_run)

    return PullResult(
        applied_live=applied_live,
        applied_tombstones=applied_tombstones,
        skipped_stale=skipped_stale,
        errors=errors,
        skipped_closets=skipped_closets,
        skipped_echoes=skipped_echoes,
        tombstone_preview=tombstone_preview,
        known_ids=known_ids,
    )


def sync(
    *,
    engine: RetrievalEngine,
    tombstones: TombstoneStore,
    client: TragsClient,
    poppy_dir: Path,
    dry_run: bool = False,
) -> SyncResult:
    """Pull-then-push, so incoming deletions apply before uploading local writes."""
    state = load(poppy_dir)
    pull_res = pull(
        engine=engine,
        tombstones=tombstones,
        client=client,
        state=state,
        poppy_dir=poppy_dir,
        dry_run=dry_run,
    )
    try:
        push_res = push(
            engine=engine,
            tombstones=tombstones,
            client=client,
            state=state,
            poppy_dir=poppy_dir,
            dry_run=dry_run,
            # A clean pull just made at least one authenticated request through
            # this same client, which is exactly what an idle push's probe would
            # ask for. Without this an idle cycle paid for two identical GETs
            # instead of one.
            already_authenticated=pull_res.errors == 0,
            tombstone_preview=pull_res.tombstone_preview,
            known_ids_preview=pull_res.known_ids,
        )
    except TragsTransportError as exc:
        # The pull that just ran applied rows to the local store and the push
        # then found the host gone. Carry the pull's count onto the error so the
        # offline line describes the whole run, not just its failed half: a
        # pulled tombstone DELETES a local memory, and a user told nothing
        # happened has no reason to go looking for it inside the retention
        # window.
        _note_transport_progress(exc, pulled=pull_res.applied_live + pull_res.applied_tombstones, simulated=dry_run)
        raise
    if not dry_run:
        # AFTER push, never before. A tombstone is the only thing that carries a
        # deletion to the cloud, so purging one that has not been sent leaves the
        # cloud row live and the next pull re-ingests the forgotten memory.
        # Sent marks, not the live watermark, prove a deletion is done. Known
        # IDs with an unsent deletion for any remote survive; unknown IDs have
        # no pending work. Closet announcements have their own pending guard.
        #
        # This runs here because the local web UI's startup was otherwise the
        # only caller, and a user who never opens the dashboard kept expired
        # records forever, including the migration's plaintext pre-images.
        try:
            tombstones.purge_expired(pushed_through=datetime.now(timezone.utc).isoformat(), require_sent=True)
        except Exception as exc:
            # Housekeeping must never be what fails a sync; retried next cycle.
            # Logged rather than silent so a store that never ages out is
            # diagnosable.
            logger.warning("closet/tombstone purge skipped: %s", exc)

    # No blanket clear here: pull and push each compare-and-clear ONLY the error
    # slots they observed and resolved (pull -> pull+auth, push -> push+auth),
    # under the state lock. A fully-clean cycle therefore clears every kind that
    # was present, while an error a concurrent sync recorded mid-cycle survives —
    # an unconditional `clear_error()` here would wipe it.
    return SyncResult(push=push_res, pull=pull_res)


def remote_is_configured(poppy_dir: Path) -> bool:
    """Whether this store has somewhere to push to.

    The same question ``sync.auto.run_once`` asks before it does anything — a
    resolved Trags API key — so the two can never disagree about whether a
    deletion still has somewhere to travel.
    """
    from poppy.config import load_config, resolve_trags_api_key

    try:
        return bool(resolve_trags_api_key(load_config(poppy_dir)))
    except Exception:
        # Unreadable config: assume a remote exists. The cautious answer keeps
        # unsent deletions, which is the recoverable direction.
        return True


def make_client_from_config(*, base_url: str, api_key: str) -> TragsClient:
    return TragsClient(base_url=base_url, api_key=api_key)


# Public re-export helpers used by the CLI ----------------------------------


def remote_state_for(poppy_dir: Path, url: str) -> RemoteState:
    """Read-only snapshot of the watermark for `poppy sync status`."""
    state = load(poppy_dir)
    return get_remote(state, url)
