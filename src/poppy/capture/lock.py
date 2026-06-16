"""Single-flight capture lock.

Never two capture workers for the same session at once: a fast typer can stack
UserPromptSubmit fires, and overlapping extractions would double-read turns and
race on the host CLI's auth/lock state (a known cause of silent 0-memory
extractions). Each capture worker acquires a per-session lock for its whole run;
a worker that cannot acquire it skips the fire — the watermark catches up next
time.

The lock is an ``O_CREAT | O_EXCL`` lock file under the Poppy directory. A lock
older than ``LOCK_TTL_S`` is treated as stale (a crashed worker) and stolen, so a
crash can never wedge capture permanently.
"""

from __future__ import annotations

import errno
import os
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# A held lock older than this (seconds) is assumed to belong to a crashed worker
# and is stolen. Comfortably longer than a capture's host-CLI timeout (120s).
LOCK_TTL_S = 300


def _lock_path(poppy_dir: Path, session_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id) or "session"
    return poppy_dir / f"capture-{safe}.lock"


def _try_open(path: Path) -> int | None:
    try:
        return os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            return None
    # Lock exists — steal it only if it is stale (crashed worker).
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return None
    if age <= LOCK_TTL_S:
        return None
    try:
        path.unlink()
    except OSError:
        return None
    try:
        return os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except OSError:
        return None


@contextmanager
def single_flight(poppy_dir: Path, session_id: str) -> Iterator[bool]:
    """Hold the per-session capture lock for the duration of the block.

    Yields ``True`` if the lock was acquired (caller should do the capture) or
    ``False`` if another worker holds it (caller should skip). Always releases a
    lock it acquired, even on error.
    """
    poppy_dir.mkdir(parents=True, exist_ok=True)
    path = _lock_path(poppy_dir, session_id)
    fd = _try_open(path)
    acquired = fd is not None
    if fd is not None:
        os.close(fd)
    try:
        yield acquired
    finally:
        if acquired:
            try:
                path.unlink()
            except OSError:
                pass
