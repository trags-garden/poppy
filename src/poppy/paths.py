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

import errno
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


# What a directory fsync fails with where the platform or filesystem does not
# support it: EINVAL (Linux filesystems without directory fsync), EBADF (systems
# that refuse fsync on a read-only descriptor), ENOTSUP / EOPNOTSUPP (network and
# FUSE mounts). The rename has happened and there is nothing more to do, so these
# are not failures. Anything else, EIO above all, means the rename may not be on
# disk and must reach the caller.
_DIR_FSYNC_UNSUPPORTED = frozenset({errno.EINVAL, errno.EBADF, errno.ENOTSUP, errno.EOPNOTSUPP})


def write_text_atomic(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` so readers see the old file or the new one, never a torn one.

    A plain ``write_text`` truncates first, so a crash, kill or full disk part way
    through leaves a half-written file. Lenient loaders read that as "no data"
    and the next save makes the loss permanent. Here the text goes to a temporary
    file in the same directory (``os.replace`` only renames atomically within one
    filesystem), is flushed to disk, and then renamed over the target, and the
    directory is flushed so the rename survives a power loss. The temporary name
    is unique per writer, so two processes never publish each other's
    half-written bytes, and a failed write never leaves it behind.

    Consequences callers should know:

    - The target becomes a new regular file created owner-only (0600) by
      ``mkstemp``. A symlink at ``path`` is replaced, not written through.
    - Saving needs write permission on the directory, not just on the file.
    - On Windows the replace raises ``PermissionError`` while another process
      has the target open; the old file is left as it was.
    - An error from flushing the directory is raised after the rename, so the
      target may already hold ``text`` when this raises.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        # Raw writes on the descriptor mkstemp returned: no file object ever takes
        # it over, so it is closed exactly once here whatever fails.
        try:
            remaining = memoryview(text.encode("utf-8"))
            while remaining:
                remaining = remaining[os.write(fd, remaining) :]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY)
    except PermissionError:
        # Windows cannot open a directory at all, and POSIX cannot open one we
        # may write into but not read. Either way there is no handle to flush.
        return
    try:
        os.fsync(dir_fd)
    except OSError as exc:
        if exc.errno not in _DIR_FSYNC_UNSUPPORTED:
            raise
    finally:
        os.close(dir_fd)
