"""Tests for single-flight capture lock."""

from __future__ import annotations

import os
import time
from pathlib import Path

from poppy.capture.lock import _lock_path, single_flight


def test_grants_then_releases(tmp_path: Path) -> None:
    with single_flight(tmp_path, "s1") as acquired:
        assert acquired is True
        assert _lock_path(tmp_path, "s1").exists()
    assert not _lock_path(tmp_path, "s1").exists()


def test_second_acquire_while_held_is_skipped(tmp_path: Path) -> None:
    with single_flight(tmp_path, "s1") as first:
        assert first is True
        with single_flight(tmp_path, "s1") as second:
            assert second is False


def test_independent_sessions_both_acquire(tmp_path: Path) -> None:
    with single_flight(tmp_path, "s1") as a, single_flight(tmp_path, "s2") as b:
        assert a is True
        assert b is True


def test_stale_lock_is_stolen(tmp_path: Path) -> None:
    path = _lock_path(tmp_path, "s1")
    path.write_text("")
    stale = time.time() - 10_000  # older than LOCK_TTL_S
    os.utime(path, (stale, stale))

    with single_flight(tmp_path, "s1") as acquired:
        assert acquired is True
