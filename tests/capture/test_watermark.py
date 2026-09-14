"""Tests for per-session capture watermark (ADR-0001)."""

from __future__ import annotations

from pathlib import Path

from poppy.capture.watermark import get_watermark, reset_session, set_watermark


def test_default_is_zero(tmp_path: Path) -> None:
    assert get_watermark(tmp_path, "s1") == 0


def test_set_and_get_roundtrip(tmp_path: Path) -> None:
    set_watermark(tmp_path, "s1", 5)
    assert get_watermark(tmp_path, "s1") == 5


def test_sessions_are_isolated(tmp_path: Path) -> None:
    set_watermark(tmp_path, "s1", 5)
    set_watermark(tmp_path, "s2", 9)
    assert get_watermark(tmp_path, "s1") == 5
    assert get_watermark(tmp_path, "s2") == 9


def test_reset_zeroes_the_session(tmp_path: Path) -> None:
    set_watermark(tmp_path, "s1", 5)
    reset_session(tmp_path, "s1")
    assert get_watermark(tmp_path, "s1") == 0


def test_set_clamps_negative_to_zero(tmp_path: Path) -> None:
    set_watermark(tmp_path, "s1", -3)
    assert get_watermark(tmp_path, "s1") == 0
