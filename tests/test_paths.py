"""Tests for the shared ``ensure_poppy_dir`` helper and its call sites.

The Poppy data directory holds owner-only material (the local memory DB,
journals, config, key fallbacks). Every site that may create it must tighten it
to 0700, or a non-interactive setup that never hits the config write leaves it
world-readable on a shared host.
"""

from __future__ import annotations

import errno
import os
import stat
import sys
import tempfile
from pathlib import Path

import pytest

from poppy.paths import _DIR_FSYNC_UNSUPPORTED, ensure_poppy_dir, write_text_atomic

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX dir modes")


def _mode(p: Path) -> int:
    return stat.S_IMODE(p.stat().st_mode)


def test_ensure_poppy_dir_creates_owner_only(tmp_path):
    d = tmp_path / "poppy"
    assert ensure_poppy_dir(d) == d
    assert d.is_dir()
    assert _mode(d) == 0o700


def test_ensure_poppy_dir_tightens_existing_loose_dir(tmp_path):
    d = tmp_path / "poppy"
    d.mkdir()
    d.chmod(0o755)  # a world-readable dir left by an older bare mkdir
    ensure_poppy_dir(d)
    assert _mode(d) == 0o700


def test_ensure_poppy_dir_preserves_readonly_owner_bits(tmp_path):
    # Loosen-only: a deliberately read-only dir (encryption-gate repair
    # scenarios simulate exactly this) must not be granted owner write back.
    d = tmp_path / "poppy"
    d.mkdir()
    d.chmod(0o500)
    ensure_poppy_dir(d)
    assert _mode(d) == 0o500
    d.chmod(0o700)  # restore so tmp_path cleanup can remove it


def test_ensure_poppy_dir_strips_group_other_but_keeps_owner(tmp_path):
    d = tmp_path / "poppy"
    d.mkdir()
    d.chmod(0o555)  # read-only AND group/other-readable
    ensure_poppy_dir(d)
    assert _mode(d) == 0o500
    d.chmod(0o700)  # restore so tmp_path cleanup can remove it


def test_config_save_tightens_dir(tmp_path):
    from poppy.config import PoppyConfig, save_config

    d = tmp_path / "store"
    d.mkdir()
    d.chmod(0o755)
    save_config(PoppyConfig(poppy_dir=d))
    assert _mode(d) == 0o700


def test_get_fast_engine_tightens_dir(tmp_path):
    from poppy.runtime import get_fast_engine

    d = tmp_path / "store"
    d.mkdir()
    d.chmod(0o755)
    get_fast_engine(d)
    assert _mode(d) == 0o700


def test_journal_record_tightens_dir(tmp_path):
    from poppy.capture import journal

    d = tmp_path / "store"
    d.mkdir()
    d.chmod(0o755)
    journal.record(d, session_id="s1", project=None, count=0, items=[])
    assert _mode(d) == 0o700


def test_write_text_atomic_replaces_a_loose_file_with_an_owner_only_one(tmp_path):
    target = tmp_path / "state.json"
    target.write_text("old")
    target.chmod(0o644)
    write_text_atomic(target, "new")
    assert target.read_text() == "new"
    assert _mode(target) == 0o600
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def _watch_dir_fsync(monkeypatch, target: Path, *, fail_with: int | None = None) -> dict[str, list[int]]:
    """Record the descriptors opened on ``target``'s directory and fsynced; optionally fail the directory fsync."""
    seen: dict[str, list[int]] = {"opened": [], "fsynced": []}
    real_open, real_fsync = os.open, os.fsync

    def recording_open(file, *args, **kwargs):
        fd = real_open(file, *args, **kwargs)
        if Path(file) == target.parent:
            seen["opened"].append(fd)
        return fd

    def recording_fsync(fd):
        if fd in seen["opened"]:
            seen["fsynced"].append(fd)
            if fail_with is not None:
                raise OSError(fail_with, os.strerror(fail_with))
        return real_fsync(fd)

    monkeypatch.setattr(os, "open", recording_open)
    monkeypatch.setattr(os, "fsync", recording_fsync)
    return seen


def _is_closed(fd: int) -> bool:
    try:
        os.fstat(fd)
    except OSError as exc:
        return exc.errno == errno.EBADF
    return False


def test_write_text_atomic_flushes_and_closes_the_directory(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    seen = _watch_dir_fsync(monkeypatch, target)
    write_text_atomic(target, "new")
    assert len(seen["opened"]) == 1
    assert seen["fsynced"] == seen["opened"]
    assert _is_closed(seen["opened"][0])


def test_write_text_atomic_propagates_a_directory_fsync_io_error(tmp_path, monkeypatch):
    """EIO means the rename may not be on disk: the caller must hear about it.

    The rename itself has already happened, so the target holds the new text.
    """
    target = tmp_path / "state.json"
    target.write_text("old")
    seen = _watch_dir_fsync(monkeypatch, target, fail_with=errno.EIO)
    with pytest.raises(OSError) as raised:
        write_text_atomic(target, "new")
    assert raised.value.errno == errno.EIO
    assert _is_closed(seen["opened"][0])
    assert target.read_text() == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


@pytest.mark.parametrize("code", sorted(_DIR_FSYNC_UNSUPPORTED))
def test_write_text_atomic_ignores_an_unsupported_directory_fsync(tmp_path, monkeypatch, code):
    target = tmp_path / "state.json"
    seen = _watch_dir_fsync(monkeypatch, target, fail_with=code)
    write_text_atomic(target, "new")
    assert seen["fsynced"] == seen["opened"]
    assert _is_closed(seen["opened"][0])
    assert target.read_text() == "new"


def test_write_text_atomic_skips_a_directory_it_cannot_open(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    real_open = os.open

    def no_directory_handles(file, *args, **kwargs):
        if Path(file) == tmp_path:
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(os, "open", no_directory_handles)
    write_text_atomic(target, "new")
    assert target.read_text() == "new"


def test_write_text_atomic_closes_the_temp_descriptor_when_the_write_fails(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    target.write_text("old")
    created: list[int] = []
    real_mkstemp = tempfile.mkstemp

    def recording_mkstemp(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        created.append(fd)
        return fd, name

    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(tempfile, "mkstemp", recording_mkstemp)
    # Fail at the first step after mkstemp, however the descriptor is written to.
    monkeypatch.setattr(os, "write", interrupted)
    monkeypatch.setattr(os, "fdopen", interrupted)
    with pytest.raises(KeyboardInterrupt):
        write_text_atomic(target, "new")
    monkeypatch.undo()
    assert len(created) == 1 and _is_closed(created[0])
    assert target.read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]
