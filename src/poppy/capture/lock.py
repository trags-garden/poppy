"""Single-flight capture lock.

Never two capture workers for the same session at once: a fast typer can stack
UserPromptSubmit fires, and overlapping extractions would double-read turns and
race on the host CLI's auth/lock state (a known cause of silent 0-memory
extractions). Each capture worker acquires a per-session lock for its whole run;
a worker that cannot acquire it skips the fire — the watermark catches up next
time.

The lock is an ``O_CREAT | O_EXCL`` lock file under the Poppy directory. A lock
older than ``LOCK_TTL_S`` is treated as stale (a crashed worker) and stolen, so a
crash can never wedge capture permanently. A live worker calls ``renew()`` at
each progress point of its pass, so only a single step, never the whole pass,
has to fit inside ``LOCK_TTL_S``.
"""

from __future__ import annotations

import errno
import os
import re
import secrets
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from poppy.paths import ensure_poppy_dir

# A held lock older than this (seconds) is assumed to belong to a crashed worker
# and is stolen. Holders renew it after every step of a pass, so it only has to
# outlast the slowest single step: one extraction (120s) or one model request.
LOCK_TTL_S = 300

# The lock this context holds, as (path, token). The token written into the lock
# file tells our lock apart from one another worker created after stealing ours.
_held: ContextVar[tuple[Path, str] | None] = ContextVar("poppy_capture_lock", default=None)


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
    ensure_poppy_dir(poppy_dir)
    path = _lock_path(poppy_dir, session_id)
    fd = _try_open(path)
    acquired = fd is not None
    token = secrets.token_hex(16)
    if fd is not None:
        try:
            os.write(fd, token.encode())
        finally:
            os.close(fd)
    held = _held.set((path, token) if acquired else None)
    try:
        yield acquired
    finally:
        _held.reset(held)
        # A stalled holder whose lock was stolen must not delete the new owner's.
        if acquired and _owns(path, token):
            try:
                path.unlink()
            except OSError:
                pass


def _owns(path: Path, token: str) -> bool:
    try:
        return path.read_text() == token
    except OSError:
        return False


def renew() -> bool:
    """Refresh the lock this context holds so it is not taken for abandoned.

    Only touches a lock file that still carries our token: if another worker
    stole it, or it is gone, nothing is created or changed. Returns whether the
    lock was renewed. A no-op outside ``single_flight``.
    """
    held = _held.get()
    if held is None:
        return False
    path, token = held
    if not _owns(path, token):
        return False
    try:
        now = time.time()
        os.utime(path, (now, now))
    except OSError:
        return False
    return True
