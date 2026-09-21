"""Per-session capture state file for ADR-0001's watermark-after-success rule.

A single JSON file under the Poppy data directory holds the per-session capture
state — the watermark and the turn cadence counters.
It is keyed by session id so concurrent sessions never corrupt each other's
progress.

Writes go through ``save`` which does an atomic ``os.replace`` so a crash mid-write
can never leave a torn file. But the counters (``turns``/``captures``) are updated
read-modify-write from separate short-lived hook processes — ``register_turn``
runs in a fresh process on every UserPromptSubmit, outside the mid-session
single-flight lock — so a bare load→mutate→save loses increments when two fire
close together, desyncing cadence and suppressing capture. ``update`` serializes
the whole read-modify-write under an exclusive ``flock`` to close that race.

This file is a per-device cache of capture progress — it is NOT synced (the sync
boundary keeps the capture journal / state local).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator, TypeVar

from poppy.paths import ensure_poppy_dir

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX (Windows)
    fcntl = None  # type: ignore[assignment]

STATE_FILENAME = "capture_state.json"
# Dedicated lock file (never the state file itself: ``save`` replaces the state
# inode, which would detach any lock held on it).
LOCK_FILENAME = STATE_FILENAME + ".lock"

_T = TypeVar("_T")


def state_path(poppy_dir: Path) -> Path:
    return poppy_dir / STATE_FILENAME


def load(poppy_dir: Path) -> dict:
    path = state_path(poppy_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save(poppy_dir: Path, data: dict) -> None:
    ensure_poppy_dir(poppy_dir)
    path = state_path(poppy_dir)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


@contextmanager
def _locked(poppy_dir: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock on the capture-state lock file.

    Degrades to a no-op where ``fcntl`` is unavailable (Windows), where the
    atomic ``save`` still prevents torn files even if it cannot prevent a lost
    update.
    """
    ensure_poppy_dir(poppy_dir)
    if fcntl is None:
        yield
        return
    fd = os.open(str(poppy_dir / LOCK_FILENAME), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def update(poppy_dir: Path, mutate: Callable[[dict], _T]) -> _T:
    """Atomically load → mutate → save the capture state under an exclusive lock.

    ``mutate`` receives the loaded dict, edits it in place, and may return a
    value (e.g. the new counter), which ``update`` returns. The flock serializes
    the whole read-modify-write across the concurrent short-lived hook processes,
    so increments are never lost.
    """
    with _locked(poppy_dir):
        data = load(poppy_dir)
        result = mutate(data)
        save(poppy_dir, data)
    return result


def stamp_last_seen(poppy_dir: Path, session_id: str) -> str:
    """Record a session activity timestamp under the capture-state flock.

    The timestamp is read inside the lock and never regresses: two Stop hooks
    racing for the lock could otherwise let the one that acquired it later
    overwrite a newer stamp with an older one. The stored value is the max of the
    fresh and existing timestamps (ISO-8601 UTC strings sort chronologically), so
    ``last_seen_at`` is monotonic. Returns the value actually stored.
    """

    def _mutate(data: dict) -> str:
        timestamp = datetime.now(UTC).isoformat()
        entry = data.setdefault(session_id, {})
        previous = entry.get("last_seen_at")
        if isinstance(previous, str) and previous > timestamp:
            timestamp = previous
        entry["last_seen_at"] = timestamp
        return timestamp

    return update(poppy_dir, _mutate)
