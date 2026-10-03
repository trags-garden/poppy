"""Tests for single-flight capture lock."""

from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path

from poppy.capture.lock import _lock_path, is_held, single_flight


def test_grants_then_releases(tmp_path: Path) -> None:
    with single_flight(tmp_path, "s1") as acquired:
        assert acquired is True
        assert is_held(_lock_path(tmp_path, "s1"))
    assert not is_held(_lock_path(tmp_path, "s1"))
    with single_flight(tmp_path, "s1") as again:
        assert again is True


def test_lock_file_persists_across_release(tmp_path: Path) -> None:
    with single_flight(tmp_path, "s1") as acquired:
        assert acquired is True
    assert _lock_path(tmp_path, "s1").exists()


def test_every_other_acquirer_is_skipped_while_held(tmp_path: Path) -> None:
    with single_flight(tmp_path, "s1") as first:
        assert first is True
        with single_flight(tmp_path, "s1") as second:
            assert second is False
            with single_flight(tmp_path, "s1") as third:
                assert third is False
        # A skipped worker leaving does not release the holder's lock.
        with single_flight(tmp_path, "s1") as fourth:
            assert fourth is False


def test_holder_keeps_the_lock_however_old_the_file_is(tmp_path: Path) -> None:
    path = _lock_path(tmp_path, "s1")
    with single_flight(tmp_path, "s1") as first:
        assert first is True
        ancient = time.time() - 10_000
        os.utime(path, (ancient, ancient))
        with single_flight(tmp_path, "s1") as second:
            assert second is False


def test_independent_sessions_both_acquire(tmp_path: Path) -> None:
    with single_flight(tmp_path, "s1") as a, single_flight(tmp_path, "s2") as b:
        assert a is True
        assert b is True


def test_lock_file_left_by_an_older_version_does_not_block(tmp_path: Path) -> None:
    path = _lock_path(tmp_path, "s1")
    path.write_text("")
    os.utime(path, (time.time(), time.time()))  # fresh, as an older version's live lock looked
    with single_flight(tmp_path, "s1") as acquired:
        assert acquired is True


def test_crashed_holder_releases_the_lock(tmp_path: Path) -> None:
    path = _lock_path(tmp_path, "s1")
    tmp_path.mkdir(exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with single_flight(tmp_path, "s1") as while_held:
        assert while_held is False
    os.close(fd)  # the holder dies without unlocking or deleting anything
    with single_flight(tmp_path, "s1") as acquired:
        assert acquired is True
