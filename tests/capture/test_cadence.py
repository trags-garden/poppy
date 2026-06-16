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
