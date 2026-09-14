"""Auto-sync: run a bidirectional `pull-then-push` after every local write.

Design:

  trigger(poppy_dir)             — fast, called at every write site.
      1. Check config; bail if auto_sync != "on" or no trags_api_key.
      2. Touch a "pending" flag (atomic). Any newer write resets it.
      3. Spawn a detached worker (`poppy sync _auto-worker`) and return.
         If a worker is already running it'll see the pending flag on
         its next loop iteration; if not, the new worker picks it up.

  run_worker(poppy_dir)          — invoked by the hidden CLI subcommand.
      1. Non-blocking flock on ~/.poppy/sync.lock. If held, exit (another
         worker is running; it'll observe our pending flag).
      2. Loop:
           - if `pending` doesn't exist, break.
           - delete `pending` (any new write re-touches it).
           - sleep DEBOUNCE_S to coalesce bursts.
           - run `sync.sync()` (pull-then-push); log to sync-worker.log.
      3. Release lock.

The trigger MUST NEVER raise — write paths swallow exceptions defensively.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from poppy.paths import ensure_poppy_dir

PENDING_FILENAME = "sync.pending"
LOCK_FILENAME = "sync.lock"
LOG_FILENAME = "sync-worker.log"
# Subprocess stdout/stderr (model-load chatter, tracebacks) goes here so the
# structured LOG_FILENAME stays parseable. Capped at MAX_RAW_LOG_BYTES.
RAW_LOG_FILENAME = "sync-worker.raw.log"
MAX_RAW_LOG_BYTES = 1 * 1024 * 1024  # 1 MB rolling truncate

# Coalesce bursts of writes. Long enough to absorb a consolidate() that
# inserts ~10 facts back-to-back, short enough to feel "automatic".
DEBOUNCE_S = float(os.environ.get("POPPY_AUTOSYNC_DEBOUNCE", "1.0"))


def _is_enabled(poppy_dir: Path) -> bool:
    """Whether auto-sync should fire for this poppy_dir."""
    if os.environ.get("POPPY_AUTOSYNC_DISABLE") == "1":
        return False
    try:
        from poppy.config import load_config, resolve_trags_api_key

        cfg = load_config(poppy_dir)
    except Exception:
        return False
    if cfg.auto_sync != "on":
        return False
    if not resolve_trags_api_key(cfg):
        return False
    return True


def _touch_pending(poppy_dir: Path) -> None:
    ensure_poppy_dir(poppy_dir)
    (poppy_dir / PENDING_FILENAME).touch()


def _open_raw_log(poppy_dir: Path):
    """Open the raw stdout/stderr sink, truncating if it has grown unbounded.

    Model-load reports and tracebacks land here so the structured LOG_FILENAME
    keeps one entry per line.
    """
    raw_path = poppy_dir / RAW_LOG_FILENAME
    try:
        if raw_path.exists() and raw_path.stat().st_size > MAX_RAW_LOG_BYTES:
            # Keep the most recent quarter — cheap rolling truncate.
            with open(raw_path, "rb") as f:
                f.seek(-MAX_RAW_LOG_BYTES // 4, 2)
                tail = f.read()
            with open(raw_path, "wb") as f:
                f.write(b"... (truncated)\n")
                f.write(tail)
    except OSError:
        pass
    try:
        return open(raw_path, "ab")
    except OSError:
        return subprocess.DEVNULL  # type: ignore[return-value]


def _spawn_worker(poppy_dir: Path) -> None:
    """Spawn `poppy sync _auto-worker` detached. Best effort."""
    poppy_bin = sys.argv[0] if sys.argv and sys.argv[0] else "poppy"
    raw_fd = _open_raw_log(poppy_dir)
    env = {**os.environ, "POPPY_DIR": str(poppy_dir)}
    try:
        subprocess.Popen(
            [poppy_bin, "sync", "_auto-worker"],
            stdin=subprocess.DEVNULL,
            stdout=raw_fd,
            stderr=raw_fd,
            start_new_session=True,
            close_fds=True,
            env=env,
        )
    except OSError:
        pass
    finally:
        if raw_fd is not subprocess.DEVNULL:
            try:
                raw_fd.close()
            except Exception:
                pass


def trigger(poppy_dir: Path, *, _spawn=None) -> bool:
    """Mark sync pending and (best effort) spawn a worker.

    Returns True if a sync was triggered, False if disabled/misconfigured.
    Never raises — write paths call this for its side effects only.
    """
    spawn = _spawn if _spawn is not None else _spawn_worker
    try:
        if not _is_enabled(poppy_dir):
            return False
        _touch_pending(poppy_dir)
        spawn(poppy_dir)
        return True
    except Exception:
        return False


def _log(poppy_dir: Path, msg: str) -> None:
    try:
        log_path = poppy_dir / LOG_FILENAME
        ts = datetime.now(timezone.utc).isoformat()
        with open(log_path, "a") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


def _do_sync(poppy_dir: Path) -> dict:
    """Run a single bidirectional sync round (pull-then-push).

    Pull first so any incoming deletes/edits land before we re-upload the
    resulting tombstones; matches `poppy sync run` semantics.
    """
    from poppy.config import load_config, resolve_trags_api_key
    from poppy.runtime import get_engine
    from poppy.sync import TragsClient
    from poppy.sync import sync as do_sync
    from poppy.sync.client import TragsAuthError, auth_error_message
    from poppy.sync.state import clear_resolved_errors, get_remote, load, record_error
    from poppy.ui.tombstones import TombstoneStore

    cfg = load_config(poppy_dir)
    api_key = resolve_trags_api_key(cfg)
    if not api_key:
        return {"skipped": "no_api_key"}
    engine = get_engine(poppy_dir)
    tombstones = TombstoneStore(poppy_dir / "memories.db")
    client = TragsClient(base_url=cfg.trags_api_url, api_key=api_key)
    # Snapshot the error-slot generations this cycle starts with so a clean sync
    # clears only what it saw — not a failure recorded concurrently mid-cycle
    # (a newer generation, even with identical text).
    gens_at_start = dict(get_remote(load(poppy_dir), cfg.trags_api_url).error_gens)
    try:
        with client:
            res = do_sync(
                engine=engine,
                tombstones=tombstones,
                client=client,
                poppy_dir=poppy_dir,
                dry_run=False,
            )
    except TragsAuthError as exc:
        # Persist the failure so `sync status` shows it, then re-raise so the
        # worker stops instead of retrying a key the server keeps rejecting.
        record_error(poppy_dir, cfg.trags_api_url, auth_error_message(exc), source="auth")
        raise
    # Only a FULLY clean cycle clears the banner. A cycle that still has push or
    # pull errors (a 409/500 on a row, a mid-pagination hiccup) synced nothing
    # for that row, so `push()` recorded the failure — keep it, or `sync status`
    # would look healthy while the row stays stuck until the next local write.
    total_errors = res.pull.errors + res.push.errors
    if total_errors == 0:
        # A fully-clean cycle: compare-and-clear only the slots this cycle
        # observed and whose generation is unchanged; a failure a concurrent CLI
        # sync recorded mid-cycle (newer generation) survives.
        clear_resolved_errors(poppy_dir, cfg.trags_api_url, tuple(gens_at_start), gens_at_start)
    return {
        "pull_live": res.pull.applied_live,
        "pull_tombstones": res.pull.applied_tombstones,
        "pull_skipped": res.pull.skipped_stale,
        "push_live": res.push.sent_live,
        "push_tombstones": res.push.sent_tombstones,
        "errors": total_errors,
    }


def run_worker(poppy_dir: Path, *, max_rounds: int = 32) -> None:
    """Drain the pending flag, pushing on each round. Single-flight via flock."""
    from poppy.sync.client import TragsAuthError, TragsQuotaError

    ensure_poppy_dir(poppy_dir)
    pending_path = poppy_dir / PENDING_FILENAME
    lock_path = poppy_dir / LOCK_FILENAME

    lock_fd = open(lock_path, "w")
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # Another worker is in flight; it will see our pending flag.
            return

        # Register as a live writer so `poppy encrypt` refuses to migrate under a
        # running sync (kernel releases the lock if this worker dies).
        from poppy import writers

        with writers.registered(poppy_dir, "sync"):
            rounds = 0
            while rounds < max_rounds:
                if not pending_path.exists():
                    break
                try:
                    pending_path.unlink()
                except FileNotFoundError:
                    pass
                # Coalesce: any writes during this sleep re-touch pending.
                if DEBOUNCE_S > 0:
                    time.sleep(DEBOUNCE_S)
                try:
                    summary = _do_sync(poppy_dir)
                    if summary.get("errors"):
                        # Some rows failed (recorded by `push()`); don't claim a
                        # clean sync. The frozen watermark retries them later.
                        _log(poppy_dir, f"sync incomplete {json.dumps(summary)}")
                    else:
                        _log(poppy_dir, f"sync ok {json.dumps(summary)}")
                except TragsQuotaError as exc:
                    # The account hit its memory cap. Live CREATES stay 402'd, but
                    # a `forget` (tombstone) or an edit-to-synced queued MID-CYCLE
                    # is NOT quota-gated and must still flush this run — otherwise
                    # the deletion waits for the next trigger and, after tombstone
                    # retention, is lost. Re-drain WHILE `pending` is set at the
                    # end of a pass: it was cleared at pass start and is only
                    # re-set by NEW external work, so a newly-queued tombstone
                    # earns another pass (drains it -> pending clears -> loop
                    # ends), while quota-blocked creates from a prior snapshot
                    # never set pending and so cannot spin. `max_rounds` remains a
                    # safety belt.
                    _log(poppy_dir, f"sync stopped: quota exceeded: {exc}")
                    if getattr(exc, "transport_retry", False):
                        # Quota is terminal for the worker, but a row (especially
                        # a tombstone/deletion) transport-failed this cycle and
                        # still needs a retry — re-arm so the drain below runs
                        # again and re-attempts it. Independent of the cap.
                        _touch_pending(poppy_dir)
                    if pending_path.exists():
                        rounds += 1
                        continue
                    break
                except TragsAuthError as exc:
                    # A revoked/invalid key will not fix itself by retrying, so
                    # stop the worker and do NOT re-arm the pending flag —
                    # otherwise every write spins this loop up to max_rounds with
                    # backoff, invisibly. `_do_sync` recorded the error;
                    # `poppy sync status` surfaces it. A fresh `poppy setup trags`
                    # re-arms sync.
                    _log(poppy_dir, f"sync stopped: auth failed: {exc!r}")
                    break
                except Exception as exc:
                    _log(poppy_dir, f"sync failed: {exc!r}")
                    # Re-arm so a later trigger retries. Backoff prevents tight loop.
                    _touch_pending(poppy_dir)
                    time.sleep(min(2**rounds, 30))
                rounds += 1

        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError:
            pass
    finally:
        try:
            lock_fd.close()
        except Exception:
            pass
