"""Single-flight capture lock.

Never two capture workers for the same session at once: a fast typer can stack
UserPromptSubmit fires, and overlapping extractions would double-read turns and
race on the host CLI's auth/lock state (a known cause of silent 0-memory
extractions). Each capture worker acquires a per-session lock for its whole run;
a worker that cannot acquire it skips the fire — the watermark catches up next
time.

The lock is an exclusive ``flock`` on a per-session lock file under the Poppy
directory, held for the whole run. The kernel releases it when the holder exits,
so a crashed worker can never wedge capture, and a live worker can never lose it
mid-run however long it takes. The file itself is never deleted: unlinking a
locked file would let a second worker create and lock a fresh one alongside it.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from poppy.paths import ensure_poppy_dir

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX (Windows)
    fcntl = None  # type: ignore[assignment]


def _lock_path(poppy_dir: Path, session_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id) or "session"
    return poppy_dir / f"capture-{safe}.lock"


def _try_lock(fd: int) -> bool:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def is_held(path: Path) -> bool:
    """Whether a live worker currently holds the lock file at ``path``."""
    if fcntl is None:
        return False
    try:
        fd = os.open(str(path), os.O_RDWR)
    except OSError:
        return False
    try:
        return not _try_lock(fd)
    finally:
        os.close(fd)  # closing drops a probe lock we may have taken


@contextmanager
def single_flight(poppy_dir: Path, session_id: str) -> Iterator[bool]:
    """Hold the per-session capture lock for the duration of the block.

    Yields ``True`` if the lock was acquired (caller should do the capture) or
    ``False`` if another worker holds it (caller should skip). Always releases a
    lock it acquired, even on error. Degrades to always acquiring where
    ``fcntl`` is unavailable (Windows).
    """
    ensure_poppy_dir(poppy_dir)
    if fcntl is None:
        yield True
        return
    try:
        fd = os.open(str(_lock_path(poppy_dir, session_id)), os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:
        yield False
        return
    try:
        acquired = _try_lock(fd)
        try:
            yield acquired
        finally:
            if acquired:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
    finally:
        os.close(fd)
