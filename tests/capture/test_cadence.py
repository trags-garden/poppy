"""Tests for TurnCadence (ADR-0001: every-Nth-turn cadence + soft cap)."""

from __future__ import annotations

from pathlib import Path

from poppy.capture.cadence import (
    capture_count,
    record_capture,
    register_turn,
    should_capture,
    soft_cap_reached,
)
from poppy.capture.watermark import reset_session


def test_register_turn_increments(tmp_path: Path) -> None:
    assert register_turn(tmp_path, "s1") == 1
    assert register_turn(tmp_path, "s1") == 2
    assert register_turn(tmp_path, "s1") == 3


def test_should_capture_fires_every_nth() -> None:
    assert should_capture(3) is True
    assert should_capture(6) is True
    assert should_capture(0) is False
    assert should_capture(1) is False
    assert should_capture(4) is False
    assert should_capture(4, n=2) is True


def test_reset_zeroes_the_turn_counter(tmp_path: Path) -> None:
    register_turn(tmp_path, "s1")
    register_turn(tmp_path, "s1")
    reset_session(tmp_path, "s1")
    assert register_turn(tmp_path, "s1") == 1


def test_concurrent_sessions_do_not_interfere(tmp_path: Path) -> None:
    register_turn(tmp_path, "s1")
    register_turn(tmp_path, "s1")
    assert register_turn(tmp_path, "s2") == 1
    assert register_turn(tmp_path, "s1") == 3


def test_soft_cap_tracks_capture_count(tmp_path: Path) -> None:
    assert soft_cap_reached(tmp_path, "s1", k=2) is False
    record_capture(tmp_path, "s1")
    assert soft_cap_reached(tmp_path, "s1", k=2) is False
    record_capture(tmp_path, "s1")
    assert soft_cap_reached(tmp_path, "s1", k=2) is True
    assert capture_count(tmp_path, "s1") == 2


def test_register_turn_no_lost_increments_under_concurrency(tmp_path: Path) -> None:
    """register_turn runs in a fresh process per UserPromptSubmit, all mutating
    the same state file. The flock'd read-modify-write must not lose increments
    when several fire at once (regression for the cadence/watermark race)."""
    import threading

    from poppy.capture import _state

    n_workers = 8
    per_worker = 25
    barrier = threading.Barrier(n_workers)

    def worker() -> None:
        barrier.wait()  # maximize contention: everyone starts together
        for _ in range(per_worker):
            register_turn(tmp_path, "sess")

    threads = [threading.Thread(target=worker) for _ in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert _state.load(tmp_path)["sess"]["turns"] == n_workers * per_worker
