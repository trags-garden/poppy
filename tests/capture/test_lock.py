"""Tests for single-flight capture lock."""

from __future__ import annotations

import contextvars
import os
import time
from pathlib import Path
from types import SimpleNamespace

from poppy.capture.lock import LOCK_TTL_S, _lock_path, renew, single_flight


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


def _fake_clock(monkeypatch):
    clock = SimpleNamespace(elapsed=0.0)
    started = time.time()
    monkeypatch.setattr("poppy.capture.lock.time", SimpleNamespace(time=lambda: started + clock.elapsed))
    return clock


def test_renewed_lock_is_not_stolen(tmp_path: Path, monkeypatch) -> None:
    clock = _fake_clock(monkeypatch)
    with single_flight(tmp_path, "s1") as first:
        assert first is True
        for _ in range(4):
            clock.elapsed += LOCK_TTL_S - 1
            assert renew() is True
            with single_flight(tmp_path, "s1") as second:
                assert second is False


def test_stalled_holder_is_still_stolen(tmp_path: Path, monkeypatch) -> None:
    clock = _fake_clock(monkeypatch)
    path = _lock_path(tmp_path, "s1")
    with single_flight(tmp_path, "s1") as first:
        assert first is True
        clock.elapsed += LOCK_TTL_S - 1
        assert renew() is True
        clock.elapsed += LOCK_TTL_S + 1  # no progress for longer than the lifetime
        with single_flight(tmp_path, "s1") as second:
            assert second is True
        # The stalled holder's lock is gone; renewing must not recreate it.
        assert renew() is False
        assert not path.exists()


def test_renew_does_not_touch_a_lock_another_worker_owns(tmp_path: Path, monkeypatch) -> None:
    clock = _fake_clock(monkeypatch)
    path = _lock_path(tmp_path, "s1")
    with single_flight(tmp_path, "s1") as acquired:
        assert acquired is True
        path.write_text("someone-else")
        before = path.stat().st_mtime
        clock.elapsed += 60
        assert renew() is False
        assert path.stat().st_mtime == before


def test_renew_outside_a_lock_is_a_no_op(tmp_path: Path) -> None:
    assert renew() is False
    with single_flight(tmp_path, "s1") as first, single_flight(tmp_path, "s1") as second:
        assert first is True and second is False
        # The skipped block holds nothing, so it renews nothing.
        assert renew() is False


def test_stalled_holder_leaves_the_new_owners_lock_in_place(tmp_path: Path) -> None:
    path = _lock_path(tmp_path, "s1")
    other_worker = contextvars.Context()  # B runs as a separate worker
    b = single_flight(tmp_path, "s1")
    with single_flight(tmp_path, "s1") as a:
        assert a is True
        stale = time.time() - LOCK_TTL_S - 1  # A stalls
        os.utime(path, (stale, stale))
        assert other_worker.run(b.__enter__) is True  # B steals and keeps running
    try:
        # A has exited; B's lock must survive it.
        assert path.exists()
        with single_flight(tmp_path, "s1") as c:
            assert c is False
    finally:
        other_worker.run(b.__exit__, None, None, None)
    assert not path.exists()
