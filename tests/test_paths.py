"""Tests for the shared ``ensure_poppy_dir`` helper and its call sites.

The Poppy data directory holds owner-only material (the local memory DB,
journals, config, key fallbacks). Every site that may create it must tighten it
to 0700, or a non-interactive setup that never hits the config write leaves it
world-readable on a shared host.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

from poppy.paths import ensure_poppy_dir, write_text_atomic

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
