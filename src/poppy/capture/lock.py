"""Single-flight capture lock.

Never two capture workers for the same session at once: a fast typer can stack
UserPromptSubmit fires, and overlapping extractions would double-read turns and
race on the host CLI's auth/lock state (a known cause of silent 0-memory
extractions). Each capture worker acquires a per-session lock for its whole run;
a worker that cannot acquire it skips the fire — the watermark catches up next
time.

The lock is an exclusive ``flock`` on a per-session ``.flock`` file under the
Poppy directory, held for the whole run. The kernel releases it when the holder
exits, so a crashed worker can never wedge capture, and a live worker can never
lose it mid-run however long it takes. The file itself is never deleted:
unlinking a locked file would let a second worker create and lock a fresh one
alongside it. The holder writes its pid into the file while it runs, so
``poppy doctor`` can see a capture in flight without touching the lock.

Older versions used a ``.lock`` file created exclusively and deleted on release,
treated as abandoned after ``LEGACY_LOCK_TTL_S``. A worker started by an older
version may still be running across an upgrade, so a fresh legacy file counts
as held. New code never creates or deletes legacy files; the separate name
keeps an older worker's delete from touching the ``.flock`` file.
"""

from __future__ import annotations

import errno
import os
import re
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from poppy.paths import ensure_poppy_dir

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX (Windows)
    fcntl = None  # type: ignore[assignment]

# Older versions stole a ``.lock`` file older than this (seconds).
LEGACY_LOCK_TTL_S = 300

_OPEN_FLAGS = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)

# ``poppy doctor`` from an older version briefly takes the lock to test it, so
# a contended lock is retried for this long (seconds, in steps) before another
# worker is taken to hold it.
CONTENDED_RETRY_S = 0.1
_CONTENDED_STEPS = 4


def _safe(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", session_id) or "session"


def _lock_path(poppy_dir: Path, session_id: str) -> Path:
    return poppy_dir / f"capture-{_safe(session_id)}.flock"


def _legacy_lock_path(poppy_dir: Path, session_id: str) -> Path:
    return poppy_dir / f"capture-{_safe(session_id)}.lock"


def _legacy_held(poppy_dir: Path, session_id: str) -> bool:
    try:
        mtime = _legacy_lock_path(poppy_dir, session_id).stat().st_mtime
    except OSError:
        return False
    return time.time() - mtime <= LEGACY_LOCK_TTL_S


def _try_lock(fd: int) -> bool | None:
    """Take the lock: ``True`` if taken, ``False`` if held elsewhere, ``None`` if
    the filesystem cannot lock at all (some network mounts)."""
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
            return False
        return None
    return True


def _write_pid(fd: int, pid: int | None) -> None:
    """Record the holder's pid in the lock file, or clear it."""
    try:
        os.ftruncate(fd, 0)
        if pid is not None:
            os.pwrite(fd, f"{pid}\n".encode(), 0)
    except OSError:
        pass  # only ``is_held`` reads it


def is_held(path: Path) -> bool:
    """Whether a live worker currently holds the ``.flock`` file at ``path``.

    Reads the holder's pid instead of probing the lock, so a check can never
    make a capture skip. A pid reused since a crash can show a capture in flight
    that is not; the answer is informational only.
    """
    try:
        fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return False
    try:
        pid = int(os.read(fd, 32).strip() or 0)
    except (OSError, ValueError):
        return False
    finally:
        os.close(fd)
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True  # alive, owned by another user
    except OSError:
        return False
    return True


@contextmanager
def single_flight(poppy_dir: Path, session_id: str) -> Iterator[bool]:
    """Hold the per-session capture lock for the duration of the block.

    Yields ``True`` if the lock was acquired (caller should do the capture) or
    ``False`` if another worker still holds it after ``CONTENDED_RETRY_S``
    (caller should skip). Always releases a lock it acquired, even on error.
    Where locking is unavailable (Windows, or a filesystem that refuses locks)
    it acquires without exclusion rather than skipping every capture.
    """
    ensure_poppy_dir(poppy_dir)
    if _legacy_held(poppy_dir, session_id):
        yield False
        return
    if fcntl is None:
        yield True
        return
    try:
        fd = os.open(str(_lock_path(poppy_dir, session_id)), _OPEN_FLAGS, 0o600)
    except OSError:
        yield False
        return
    try:
        acquired = _try_lock(fd)
        for _ in range(_CONTENDED_STEPS):
            if acquired is not False:
                break
            time.sleep(CONTENDED_RETRY_S / _CONTENDED_STEPS)
            acquired = _try_lock(fd)
        if acquired is None:
            sys.stderr.write("poppy capture: file locking unavailable here; capturing without the session lock\n")
        if acquired:
            _write_pid(fd, os.getpid())
        try:
            yield acquired is not False
        finally:
            if acquired:
                _write_pid(fd, None)
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
    finally:
        os.close(fd)
