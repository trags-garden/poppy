"""Filesystem helpers for the Poppy data directory.

``~/.poppy`` holds owner-only material: the local memory database (plaintext
unless encryption is enabled), the capture journals, config.json, and the
headless fallbacks for the Trags / consolidate API keys. The default umask
leaves a freshly created directory group- and world-readable (0755), which would
expose all of that to other local accounts on a shared host.

Every code path that may be the first to create the directory goes through
``ensure_poppy_dir`` so the 0700 tightening is guaranteed, rather than being an
accident of whichever site ran first. Kept stdlib-only so the capture hot path,
the sync worker, and the MCP server can all import it without pulling in engines
or config.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

# Owner-only (rwx) directory mode. mkdir's own mode argument is masked by umask
# for new dirs and ignored entirely for existing ones, so the explicit chmod is
# what actually guarantees the permission.
POPPY_DIR_MODE = stat.S_IRWXU  # 0o700


def ensure_poppy_dir(poppy_dir: Path) -> Path:
    """Create ``poppy_dir`` (with parents) if missing and strip group/other access.

    Tightens even a pre-existing directory that was created world-readable by an
    older Poppy or by a non-interactive ``poppy setup`` that never touched the
    perm-fixing config write. Loosen-only in the other direction: owner bits are
    preserved as-is, so a deliberately read-only directory (or one mid-repair by
    the encryption gate) is never granted write back. Best-effort: a chmod
    failure (an exotic filesystem, or Windows where the mode is a no-op) never
    blocks the caller. Returns the directory for call-site convenience.
    """
    if not poppy_dir.exists():
        poppy_dir.mkdir(parents=True, exist_ok=True)
        try:
            poppy_dir.chmod(POPPY_DIR_MODE)
        except OSError:
            pass
        return poppy_dir
    try:
        mode = stat.S_IMODE(poppy_dir.stat().st_mode)
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            poppy_dir.chmod(mode & POPPY_DIR_MODE)
    except OSError:
        pass
    return poppy_dir


def write_text_atomic(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` so readers see the old file or the new one, never a torn one.

    A plain ``write_text`` truncates first, so a crash, kill or full disk part way
    through leaves a half-written file. Lenient loaders read that as "no data"
    and the next save makes the loss permanent. Here the text goes to a temporary
    file in the same directory (``os.replace`` only renames atomically within one
    filesystem), is flushed to disk, and then renamed over the target. The
    temporary name is unique per writer, so two processes never publish each
    other's half-written bytes, and a failed write never leaves it behind.
    ``mkstemp`` creates the file owner-only (0600), matching the directory.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
