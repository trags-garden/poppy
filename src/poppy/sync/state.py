"""Watermark persistence for `poppy sync`.

State lives at ``$POPPY_DIR/sync_state.json``. Schema:

```json
{
  "<trags-url>": {
    "last_pulled_at": "<iso8601>",     // server-side updated_at of latest pulled row
    "last_pushed_at": "<iso8601>",     // local updated_at of latest pushed row
    "last_synced_at": "<iso8601>",     // when sync last completed
    "pushed_count": 0,
    "pulled_count": 0
  }
}
```

Keyed by URL so a user can sync to multiple Trags instances if desired.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

from poppy.db import write_gate, write_txn
from poppy.engine._timestamps import utc_iso
from poppy.paths import ensure_poppy_dir, write_text_atomic

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

STATE_FILENAME = "sync_state.json"
LOCK_FILENAME = "sync_state.json.lock"


# Recognised error origins. A single state record must hold a push AND a pull
# failure at once — one slot per kind — so clearing one never erases the other.
#
# "probe" is the key check a push with nothing to send makes (see ``push()``). It
# uploads no memory, so what it sees says nothing about a pending row: kept in
# the push slot it could only be discharged by an actual upload, and an idle
# store has none to make — one transient 500 on an idle push then wedged every
# later push at exit 1 for good. Its own slot is written and cleared by the
# probe's own evidence.
ERROR_KINDS = ("push", "pull", "auth", "probe")

# The GENERATION of push-watermark semantics a stored watermark was written
# under. A remote recording anything less owes ONE full push pass before its
# watermark may be trusted again (see ``push``). Bumped only when a fix changes
# what an already-written watermark can MEAN — the one thing that makes a value
# still looking perfectly ordinary unsafe to reuse.
PUSH_WATERMARK_VERSION = 1


@dataclass
class RemoteState:
    last_pulled_at: str | None = None
    last_pushed_at: str | None = None
    last_synced_at: str | None = None
    pushed_count: int = 0
    pulled_count: int = 0
    # Outstanding sync failures keyed by origin ("push" | "pull" | "auth" | "probe").
    # SEPARATE slots because a push failure and a pull failure can be unresolved
    # simultaneously; clearing one must not erase the other. "auth" clears on any
    # clean authenticated op. Surfaced by `poppy sync status`.
    errors: dict[str, str] = field(default_factory=dict)
    # Monotonic GENERATION per error slot, bumped every time a slot is (re)recorded
    # from ``error_seq``. Compare-and-clear matches on the generation, not the
    # message text, so a re-recorded failure with IDENTICAL text (an ABA) is a
    # different generation and is never mistaken for the one a cycle observed.
    error_gens: dict[str, int] = field(default_factory=dict)
    error_seq: int = 0
    # The UTC-rewrite event this remote has already made its one full re-push for
    # (the stamp of the store's ``utc_timestamp_text_repush`` row), or None.
    #
    # PER REMOTE, because the watermark is: re-spelling a row can drop it below one
    # remote's watermark and not another's, and a single consumable flag in the
    # store let the first remote to sync consume the recovery while every other one
    # skipped the row for ever. Optional, so a ``sync_state.json`` written before
    # this loads unchanged and reads as "not done yet".
    repush_done_for: str | None = None

    # The generation of watermark semantics ``last_pushed_at`` was written under,
    # or 0 for a state file written before this field existed.
    #
    # A watermark is worth exactly what the client that wrote it was worth, and a
    # client that let a FUTURE-dated row carry the mark left one that silently
    # hides every honest write underneath it. The value cannot be read to find
    # that out: once the clock has passed the bad stamp, a poisoned ``12:05`` and
    # an honest ``12:05`` are the same string. So what is recorded is WHICH CLIENT
    # wrote it, and a remote below ``PUSH_WATERMARK_VERSION`` buys exactly one full
    # pass — written by the same state save as the watermark itself, so a pass
    # whose save never lands is a pass still owed.
    push_watermark_v: int = 0

    # -- Back-compat read helpers (single-slot callers/tests) ---------------
    @property
    def last_error(self) -> str | None:
        """A representative outstanding message (auth > push > pull), or None."""
        for kind in ERROR_KINDS:
            if kind in self.errors:
                return self.errors[kind]
        return next(iter(self.errors.values()), None)

    @property
    def last_error_source(self) -> str | None:
        for kind in ERROR_KINDS:
            if kind in self.errors:
                return kind
        return next(iter(self.errors), None)


@dataclass
class SyncState:
    remotes: dict[str, RemoteState] = field(default_factory=dict)


def _state_path(poppy_dir: Path) -> Path:
    return poppy_dir / STATE_FILENAME


def normalize_remote_url(url: str) -> str:
    """Canonical state key for a remote URL.

    `TragsClient` strips the trailing slash from its `base_url`, so state written
    under `client.base_url` must be read under the same normalization or a
    `https://host/` config would address a different record than `https://host`
    and a recorded failure would silently vanish.
    """
    return url.rstrip("/")


def load(poppy_dir: Path, *, strict: bool = False) -> SyncState:
    """Load state, optionally refusing unreadable data for one-shot migrations.

    Missing state means no remotes even in strict mode. Existing unreadable or
    malformed state must not silently authorize an irreversible empty backfill.
    """
    path = _state_path(poppy_dir)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return SyncState()
    except (OSError, ValueError) as exc:
        if strict:
            raise ValueError(f"Cannot load sync state from {path}") from exc
        return SyncState()
    if strict and (
        not isinstance(data, dict)
        or (data.get("remotes") is not None and not isinstance(data["remotes"], dict))
        or any(value is not None and not isinstance(value, dict) for value in (data.get("remotes") or {}).values())
    ):
        raise ValueError(f"Invalid sync state in {path}")
    # Ignore unknown keys so a state file written by a newer Poppy (extra fields)
    # loads instead of raising an unexpected-keyword TypeError.
    known = {f.name for f in fields(RemoteState)}
    remotes: dict[str, RemoteState] = {}
    for url, values in (data.get("remotes") or {}).items():
        values = values or {}
        rs = RemoteState(**{k: v for k, v in values.items() if k in known})
        # Migrate the deprecated single-slot fields into the per-kind dict. The
        # old `last_error` was auth-only, so an untagged legacy banner is auth.
        if not rs.errors and values.get("last_error"):
            src = values.get("last_error_source")
            rs.errors[src if src in ERROR_KINDS else "auth"] = values["last_error"]
        # Ensure every error slot carries a generation (legacy/older state files,
        # or a file written before generations existed).
        for kind in rs.errors:
            if kind not in rs.error_gens:
                rs.error_seq += 1
                rs.error_gens[kind] = rs.error_seq
        key = normalize_remote_url(url)
        # Old code stored state under BOTH the client-normalized host and the raw
        # trailing-slash URL. Now that they collapse to one key, MERGE colliding
        # records (most-advanced wins) so an empty duplicate never wipes the
        # populated one — which would reset watermarks/counters and force a full
        # replay.
        remotes[key] = _merge_remote(remotes[key], rs) if key in remotes else rs
    return SyncState(remotes=remotes)


def _merge_remote(a: RemoteState, b: RemoteState) -> RemoteState:
    """Combine two records that normalized to the same key, keeping the most
    advanced values: max watermarks, max counters, and the union of error slots.
    """
    errors = dict(a.errors)
    error_gens = dict(a.error_gens)
    for kind, message in b.errors.items():
        if kind not in errors:
            errors[kind] = message
            error_gens[kind] = b.error_gens.get(kind, 0)
    # error_seq must stay above every live generation so future records get a
    # strictly higher one.
    error_seq = max(a.error_seq, b.error_seq, max(error_gens.values(), default=0))
    return RemoteState(
        last_pulled_at=_later_verbatim(a.last_pulled_at, b.last_pulled_at),
        last_pushed_at=_later_verbatim(a.last_pushed_at, b.last_pushed_at),
        last_synced_at=_later_verbatim(a.last_synced_at, b.last_synced_at),
        pushed_count=max(a.pushed_count, b.pushed_count),
        pulled_count=max(a.pulled_count, b.pulled_count),
        errors=errors,
        error_gens=error_gens,
        error_seq=error_seq,
        # Kept ONLY when both keys agree. The two records carry independent
        # watermarks, so "done" under one of them says nothing about the other: a
        # key holding the stamp with NO watermark (its recovery pass failed at the
        # first row) merged with a key holding an old watermark would claim the pass
        # was done and skip every row under that watermark for ever. Disagreement
        # means the merged remote owes the pass.
        repush_done_for=a.repush_done_for if a.repush_done_for == b.repush_done_for else None,
        # The LOWER generation, for the same reason: the merged watermark is a MAX
        # over both records, so it can be the one the older client wrote, and a
        # merge claiming the newer generation would retire the recovery pass while
        # carrying the very value that pass exists to distrust.
        push_watermark_v=min(a.push_watermark_v, b.push_watermark_v),
    )


def _later_verbatim(*values: str | None) -> str | None:
    """The later INSTANT, spelled exactly as it was stored.

    ``latest_iso`` answers which is later and returns it CANONICALISED, which is
    right for a watermark push has just computed and wrong for one being read back:
    push decides it owes a recovery pass by seeing that the stored spelling is not
    canonical, and a merge that quietly canonicalised it hid that. A stored
    ``2026-07-02T10:00:00-05:00`` looked already-canonical after merging, so the
    pass never ran and a pending tombstone at ``11:00+00:00`` stayed under the
    watermark until Trash aged it out.
    """
    winner = latest_iso(*values)
    if winner is None:
        return None
    for value in values:
        if value is not None and utc_iso(value) == winner:
            return value
    return winner


def save(poppy_dir: Path, state: SyncState) -> None:
    path = _state_path(poppy_dir)
    ensure_poppy_dir(path.parent)
    payload = {"remotes": {url: asdict(rs) for url, rs in state.remotes.items()}}
    # Atomic: a torn file reads as "no state" to the lenient ``load``, which would
    # silently reset every remote's watermarks on the next sync.
    write_text_atomic(path, json.dumps(payload, indent=2))


def get_remote(state: SyncState, url: str) -> RemoteState:
    url = normalize_remote_url(url)
    if url not in state.remotes:
        state.remotes[url] = RemoteState()
    return state.remotes[url]


def latest_iso(*values: str | datetime | None) -> str | None:
    """Return the LATEST INSTANT among strings / datetimes / Nones, as UTC text.

    ``max`` over raw ISO strings compares WALL CLOCK, so it only answers "which
    is later" when every value is spelled in the same offset. A watermark taken
    from a row stamped ``12:00+02:00`` (10:00Z) sits above a later tombstone at
    ``11:00+00:00``, and push's ``iso <= watermark`` filter then skips the newer
    deletion for good. Every value is therefore normalised to the one canonical
    spelling before the comparison, and the watermark is stored in it — which is
    also what the rows it is compared against now carry.

    A value that does not parse is kept verbatim, as ``utc_iso`` leaves it: this
    is a watermark, and dropping one because a clock wrote something odd would
    re-push the whole store.
    """
    isos = [iso for iso in (utc_iso(v) for v in values) if iso is not None]
    return max(isos) if isos else None


@contextmanager
def state_lock(poppy_dir: Path):
    """Exclusive lock around the state file so a read-modify-write is atomic.

    Sync entry points (CLI push/pull/run, the auto-worker) are separate
    processes and are NOT serialized by the worker's sync.lock, so their
    load->mutate->save on ``sync_state.json`` would otherwise race and a clean
    sync could clobber a concurrent sync's freshly-recorded error. Best effort:
    if ``fcntl`` is unavailable the body still runs.
    """
    ensure_poppy_dir(poppy_dir)
    lock_path = poppy_dir / LOCK_FILENAME
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
            except OSError:
                pass
        yield
    finally:
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


def mutate_remote(poppy_dir: Path, url: str, mutate: Callable[[RemoteState], None]) -> None:
    """Atomically load -> ``mutate(remote)`` -> save under the state lock, on a
    FRESH read from disk so a concurrent sync's writes to other slots/remotes
    survive (they aren't overwritten by a stale in-memory snapshot)."""
    with state_lock(poppy_dir):
        state = load(poppy_dir)
        mutate(get_remote(state, url))
        save(poppy_dir, state)


def stamp_error(remote: RemoteState, kind: str, message: str) -> None:
    """Record ``message`` in the ``kind`` slot with a fresh monotonic generation.

    Compare-and-clear matches on this generation, not the text, so a re-recorded
    failure (even with identical text) is a distinct generation and is never
    mistaken for the one a cycle observed. Call inside a ``mutate_remote``.
    """
    remote.error_seq += 1
    remote.errors[kind] = message
    remote.error_gens[kind] = remote.error_seq


def record_error(poppy_dir: Path, url: str, message: str, source: str = "push") -> None:
    """Record a sync failure in its own slot so `sync status` can surface it.

    ``source`` ("push"/"pull"/"auth") keys the slot so recording one origin's
    failure never overwrites another's.
    """
    mutate_remote(poppy_dir, url, lambda remote: stamp_error(remote, source, message))


def clear_resolved_errors(poppy_dir: Path, url: str, kinds: tuple[str, ...], observed_gens: dict[str, int]) -> None:
    """Compare-and-clear: pop only the ``kinds`` this cycle both STARTED WITH
    (present in ``observed_gens``) and whose slot still carries the SAME
    GENERATION it observed. An error written concurrently — even with identical
    text — has a newer generation and survives. Atomic under the state lock.
    """

    def _clear(remote: RemoteState) -> None:
        for kind in kinds:
            if kind in observed_gens and remote.error_gens.get(kind) == observed_gens[kind]:
                remote.errors.pop(kind, None)
                remote.error_gens.pop(kind, None)

    mutate_remote(poppy_dir, url, _clear)


def clear_error(poppy_dir: Path, url: str) -> None:
    """Clear ALL recorded errors for a remote. No-op if unset. Atomic."""

    def _clear(remote: RemoteState) -> None:
        remote.errors.clear()
        remote.error_gens.clear()

    mutate_remote(poppy_dir, url, _clear)


# Ids this store deleted locally and must never accept back from a remote at or
# below the recorded instant. The stamp is compared as TEXT by the upsert's MAX,
# so both writers below put it through `utc_iso` first: two spellings of one
# instant sort the wrong way round and would lower a record instead of raising it.
LOCAL_DELETIONS_DDL = "CREATE TABLE IF NOT EXISTS sync_local_deletions (id TEXT PRIMARY KEY, deleted_at TEXT NOT NULL)"

_RECORD_LOCAL_DELETION_SQL = (
    "INSERT INTO sync_local_deletions (id, deleted_at) VALUES (?, ?) "
    "ON CONFLICT(id) DO UPDATE SET deleted_at = MAX(deleted_at, excluded.deleted_at)"
)


# Sorts below every real timestamp as text and as an instant. Used for a row
# whose own timestamps are unreadable, so the record exists without claiming an
# instant it cannot support.
_MIN_STAMP = datetime.min.replace(tzinfo=timezone.utc).isoformat()


def _first_readable(*values: str | None) -> str | None:
    """The first of these stored timestamps that parses, in canonical UTC."""
    for value in values:
        stamp = utc_iso(value)
        if stamp is None:
            continue
        try:
            datetime.fromisoformat(stamp)
        except (TypeError, ValueError, OverflowError):
            continue
        return stamp
    return None


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)).fetchone() is not None


def remove_derived_rows(conn: sqlite3.Connection, poppy_dir: Path, *, gate_held: bool = False) -> None:
    """Remove stored derived duplicates without creating an upload or Trash entry.

    Keep a durable deletion record carrying no text: a cloud copy of one of these
    rows can arrive long after the ordinary Trash retention window, and there
    would be nothing left locally to recognise it by.

    Also retires the announcement queue. Nothing in this version drains it, but a
    client on the previous release sharing the same store still would, and every
    entry in it uploads a deletion in the retired wire format.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
    has_marked = bool(
        "is_closet" in columns and conn.execute("SELECT 1 FROM memories WHERE is_closet = 1 LIMIT 1").fetchone()
    )
    has_queue = bool(
        _table_exists(conn, "legacy_closet_ids")
        and conn.execute("SELECT 1 FROM legacy_closet_ids WHERE announce_pending = 1 LIMIT 1").fetchone()
    )
    if not has_marked and not has_queue:
        return
    # ``gate_held`` is for the engine, which takes the gate once around the
    # marker migration and this, so the two commit as one. Taking it again here
    # would be a second flock on the same file from the same process, which
    # blocks until the attempt loop gives up.
    gate = nullcontext() if gate_held else write_gate(poppy_dir)
    with gate, write_txn(conn):
        now_iso = utc_iso(datetime.now(timezone.utc))
        if _table_exists(conn, "legacy_closet_ids"):
            # Answered, not deleted, for every id EXCEPT the ones removed below:
            # those lose their entry along with the row. That is fine, because
            # what keeps a republished copy out is no longer this table but the
            # deletion record plus the grading pull does against the live parent.
            # Clearing the pending flag is what stops an older client sharing
            # this store from uploading the entry in the retired format.
            conn.execute(
                "UPDATE legacy_closet_ids SET announce_pending = 0, announced_at = ? WHERE announce_pending = 1",
                (now_iso,),
            )
        rows = conn.execute(
            "SELECT m.id, COALESCE(l.legacy_updated_at, m.updated_at), m.created_at FROM memories m "
            "LEFT JOIN legacy_closet_ids l ON l.id = m.id WHERE m.is_closet = 1"
            if _table_exists(conn, "legacy_closet_ids")
            else "SELECT id, updated_at, created_at FROM memories WHERE is_closet = 1"
        ).fetchall()
        if not rows:
            return
        conn.execute(LOCAL_DELETIONS_DDL)
        deletions = []
        for memory_id, updated_at, created_at in rows:
            # The ROW'S OWN time, never this upgrade's clock. A record stamped now
            # sits above every version of this id written before the upgrade, so a
            # real memory another device wrote at the id months ago would be
            # refused on the next pull and the watermark would move past it: the
            # remote version would be lost here for good. Stamped with what the
            # row actually carried, the record covers exactly the copy that was
            # removed and anything newer is graded rather than refused.
            #
            # A row whose times cannot be read at all falls back to the earliest
            # representable instant, never to now: a record in the past can only
            # let something through, and what it would let through is caught by
            # the grading on the pull side instead. A record dated now would
            # silently refuse a legitimate older version of the id for good.
            stamp = _first_readable(updated_at, created_at) or _MIN_STAMP
            deletions.append((memory_id, stamp))
        conn.executemany(_RECORD_LOCAL_DELETION_SQL, deletions)
        # No pre-image is kept, deliberately. Only rows marked as DERIVED are
        # removed here, and a derived row is reconstructible from the parent that
        # is still sitting in the store: either this engine wrote it at ingest, or
        # the one-time migration proved its text equals what that parent projects.
        # The inferred tier is marked differently, is not touched below, and keeps
        # its own pre-image. Copying the speaker text into a backup table on the
        # way out would put the very text this removal exists to get rid of back
        # into the store.
        # Subqueries avoid SQLite's parameter limit on large stores. The engine's
        # existing DELETE trigger removes the corresponding full-text entries.
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "ui_tombstones" in tables:
            # A snapshot of the same derived row is not a restorable memory.
            # Preserve an independent note that previously occupied this ID.
            snapshots = conn.execute(
                "SELECT t.id, t.created_at, m.created_at FROM ui_tombstones t "
                "JOIN memories m ON m.id = t.id WHERE m.is_closet = 1 "
                "AND m.content = t.content AND m.related_to = t.related_to "
                "AND m.source_type = t.source_type "
                "AND m.source_session_id IS t.source_session_id"
            ).fetchall()
            # A timestamp can have different UTC offsets in older stores.
            conn.executemany(
                "DELETE FROM ui_tombstones WHERE id = ?",
                [(mid,) for mid, saved, live in snapshots if utc_iso(saved) == utc_iso(live)],
            )
        for table in ("memory_embeddings", "legacy_closet_ids"):
            if table in tables:
                conn.execute(f"DELETE FROM {table} WHERE id IN (SELECT id FROM memories WHERE is_closet = 1)")
        conn.execute("DELETE FROM memories WHERE is_closet = 1")


def local_deletion_at(conn: sqlite3.Connection | None, memory_id: str) -> datetime | None:
    """When this store deleted ``memory_id`` locally, or None if it never did."""
    if conn is None or not _table_exists(conn, "sync_local_deletions"):
        return None
    row = conn.execute("SELECT deleted_at FROM sync_local_deletions WHERE id = ?", (memory_id,)).fetchone()
    if row is None:
        return None
    try:
        return datetime.fromisoformat(row[0])
    except (TypeError, ValueError, OverflowError):  # pragma: no cover - both writers spell it
        return None


def local_deletion_wins(conn: sqlite3.Connection | None, memory_id: str, updated_at: datetime) -> bool:
    """Whether a durable local deletion supersedes this incoming version."""
    deleted_at = local_deletion_at(conn, memory_id)
    if deleted_at is None:
        return False
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    return deleted_at >= updated_at


def clear_local_deletion(conn: sqlite3.Connection, memory_id: str) -> None:
    """Drop the record for an id a real memory has legitimately taken back."""
    if not _table_exists(conn, "sync_local_deletions"):
        return
    with write_txn(conn):
        conn.execute("DELETE FROM sync_local_deletions WHERE id = ?", (memory_id,))


def record_local_deletion(conn: sqlite3.Connection, memory_id: str, deleted_at: datetime) -> None:
    """Remember a non-restorable deletion without queuing a cloud write."""
    with write_txn(conn):
        conn.execute(LOCAL_DELETIONS_DDL)
        conn.execute(_RECORD_LOCAL_DELETION_SQL, (memory_id, utc_iso(deleted_at)))
