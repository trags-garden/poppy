"""Tests for single-flight capture lock."""

from __future__ import annotations

import errno
import fcntl
import os
import time
from pathlib import Path
from types import SimpleNamespace

from poppy.capture.lock import (
    CONTENDED_RETRY_S,
    LEGACY_LOCK_TTL_S,
    _legacy_lock_path,
    _lock_path,
    is_held,
    single_flight,
)


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


def _legacy_holder(tmp_path: Path, *, age_s: float = 0) -> Path:
    """Create the lock file an older version's worker holds while it runs."""
    path = _legacy_lock_path(tmp_path, "s1")
    path.write_text("")
    then = time.time() - age_s
    os.utime(path, (then, then))
    return path


def test_a_running_older_version_capture_is_respected(tmp_path: Path) -> None:
    legacy = _legacy_holder(tmp_path)  # A: an older version's live capture
    with single_flight(tmp_path, "s1") as b:
        assert b is False
    legacy.unlink()  # A finishes the way older versions release
    with single_flight(tmp_path, "s1") as b:
        assert b is True


def test_an_abandoned_older_version_lock_does_not_block(tmp_path: Path) -> None:
    legacy = _legacy_holder(tmp_path, age_s=LEGACY_LOCK_TTL_S + 1)
    with single_flight(tmp_path, "s1") as acquired:
        assert acquired is True
    assert legacy.exists()  # never deleted by new code


def test_an_older_version_release_cannot_free_the_new_lock(tmp_path: Path) -> None:
    legacy = _legacy_holder(tmp_path, age_s=LEGACY_LOCK_TTL_S + 1)
    with single_flight(tmp_path, "s1") as b:
        assert b is True
        legacy.unlink()  # an older worker releasing (or stealing) its own file
        with single_flight(tmp_path, "s1") as c:
            assert c is False
    assert _lock_path(tmp_path, "s1").exists()


def test_a_filesystem_without_locks_captures_instead_of_skipping(tmp_path: Path, monkeypatch, capsys) -> None:
    def refuse(fd, op):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", refuse)
    with single_flight(tmp_path, "s1") as acquired:
        assert acquired is True
        assert not is_held(_lock_path(tmp_path, "s1"))
    assert "file locking unavailable" in capsys.readouterr().err


def test_a_symlinked_lock_file_is_not_followed(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    target.write_text("keep")
    tmp_path.joinpath(_lock_path(tmp_path, "s1").name).symlink_to(target)
    with single_flight(tmp_path, "s1") as acquired:
        assert acquired is False
    assert target.read_text() == "keep"


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


def _probe(tmp_path: Path, monkeypatch, *, release_after: int | None) -> tuple[list[float], int]:
    """Hold the lock the way ``is_held`` probes it, dropping it on a given retry."""
    tmp_path.mkdir(exist_ok=True)
    fd = os.open(str(_lock_path(tmp_path, "s1")), os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        if len(slept) == release_after:
            fcntl.flock(fd, fcntl.LOCK_UN)  # the probe finishes

    monkeypatch.setattr("poppy.capture.lock.time", SimpleNamespace(time=time.time, sleep=sleep))
    return slept, fd


def test_a_doctor_probe_does_not_make_a_capture_skip(tmp_path: Path, monkeypatch) -> None:
    slept, fd = _probe(tmp_path, monkeypatch, release_after=1)
    with single_flight(tmp_path, "s1") as acquired:
        assert acquired is True
    os.close(fd)
    assert len(slept) == 1


def test_a_lock_held_past_the_retry_window_still_skips(tmp_path: Path, monkeypatch) -> None:
    slept, fd = _probe(tmp_path, monkeypatch, release_after=None)
    with single_flight(tmp_path, "s1") as acquired:
        assert acquired is False
    os.close(fd)
    assert slept and sum(slept) <= CONTENDED_RETRY_S + 1e-9
