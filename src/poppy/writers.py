"""Live-writer registry for the local store.

Every long-lived process that writes the store (the MCP server, the web UI, the
auto-sync worker, and each capture worker) registers itself here for its whole
run by holding an exclusive ``flock`` on a per-process file under
``$POPPY_DIR/writers/``. The kernel releases that lock automatically when the
process dies, so liveness needs no mtime heuristics or TTLs.

Encryption migration (``poppy encrypt enable/disable``) enumerates the directory
and probes each lock non-blockingly:

* acquiring the lock means the owning process is gone, so the file is stale and
  is unlinked;
* failing to acquire means the writer is live, so migration refuses and names
  the surface.

Because each writer owns its own file, probing one never disturbs another (a
failed non-blocking acquire does not touch the holder's lock). This is a
``fcntl``-based, POSIX-only mechanism; on platforms without ``fcntl`` (Windows)
it degrades to a no-op and migration relies on its other guards.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from poppy.paths import ensure_poppy_dir

WRITERS_DIRNAME = "writers"

# Human-readable labels for the surfaces that register. The key is the prefix
# used in the lock filename (``<surface>.<pid>.lock``).
SURFACE_LABELS = {
    "serve": "the MCP server (`poppy serve`)",
    "daemon": "the MCP daemon (`poppy daemon`)",
    "ui": "the web UI (`poppy ui`)",
    "sync": "an auto-sync worker (`poppy sync`)",
    "capture": "a capture worker (`poppy hook _capture-worker`)",
    "post-compact": "a post-compact worker (`poppy hook _post-compact-worker`)",
    "session-end": "a session-end worker (`poppy hook _session-end-worker`)",
}


def _writers_dir(poppy_dir: Path) -> Path:
    return poppy_dir / WRITERS_DIRNAME


def _fcntl():
    try:
        import fcntl  # noqa: PLC0415

        return fcntl
    except ImportError:  # pragma: no cover - Windows only
        return None


@contextmanager
def registered(poppy_dir: Path, surface: str) -> Iterator[None]:
    """Hold this process's writer lock for the duration of the block.

    A no-op where ``fcntl`` is unavailable, and never fatal: registration is a
    best-effort signal to migration, so a failure to register must not stop a
    writer from doing its job.
    """
    fcntl = _fcntl()
    if fcntl is None:
        yield
        return
    fd = None
    path = _writers_dir(poppy_dir) / f"{surface}.{os.getpid()}.lock"
    try:
        ensure_poppy_dir(poppy_dir)
        ensure_poppy_dir(_writers_dir(poppy_dir))
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)  # own per-pid file: uncontended
    except OSError:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
            fd = None
    try:
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)  # releases the flock
            except OSError:
                pass
            try:
                path.unlink()
            except OSError:
                pass


def live_writers(poppy_dir: Path) -> list[str]:
    """Return labels of surfaces with a live registered writer.

    Probes each ``writers/*.lock`` non-blockingly, unlinking any whose owner is
    gone. Returns the de-duplicated human-readable labels of the live ones.
    """
    fcntl = _fcntl()
    if fcntl is None:
        return []
    directory = _writers_dir(poppy_dir)
    if not directory.exists():
        return []
    live: set[str] = set()
    for path in sorted(directory.glob("*.lock")):
        surface = path.name.rsplit(".", 2)[0]  # "<surface>.<pid>.lock" -> surface
        try:
            fd = os.open(str(path), os.O_RDWR)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Held by a live process.
            live.add(SURFACE_LABELS.get(surface, surface))
            os.close(fd)
            continue
        # Acquired: the owner is gone. Release and remove the stale file.
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)
        try:
            path.unlink()
        except OSError:
            pass
    return sorted(live)
