"""Tests for TranscriptWindow (ADR-0001: watermark-after-success capture window).

The window reader returns exactly the turns in ``(watermark, now]`` — clamping
safely when compaction shrinks the transcript below the watermark. JSONL parsing
is covered apart, in ``test_transcript.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from poppy.capture.window import read_window

CODEX_FIXTURE = Path(__file__).parents[1] / "fixtures" / "codex" / "rollout-resumed.jsonl"
CURSOR_FIXTURE = Path(__file__).parents[1] / "fixtures" / "cursor" / "run-after-resumes.jsonl"


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_window_returns_only_new_turns(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "t.jsonl",
        [{"type": "user", "message": {"role": "user", "content": f"turn number {i} content"}} for i in range(5)],
    )
    w = read_window(p, watermark=2)
    assert w.total_turns == 5
    assert w.new_watermark == 5
    assert [t["text"] for t in w.messages] == [
        "turn number 2 content",
        "turn number 3 content",
        "turn number 4 content",
    ]


def test_window_empty_on_refire(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "t.jsonl",
        [{"type": "user", "message": {"role": "user", "content": f"turn {i} body"}} for i in range(3)],
    )
    w = read_window(p, watermark=3)
    assert w.messages == []
    assert w.new_watermark == 3


def test_window_clamps_when_transcript_shrinks(tmp_path: Path) -> None:
    """Post-compaction the transcript can be shorter than the watermark."""
    p = _write(
        tmp_path / "t.jsonl",
        [{"type": "user", "message": {"role": "user", "content": f"turn {i} body"}} for i in range(2)],
    )
    w = read_window(p, watermark=10)
    assert w.messages == []
    assert w.new_watermark == 2  # clamped down to what remains


def test_window_missing_file_is_empty(tmp_path: Path) -> None:
    w = read_window(tmp_path / "nope.jsonl", watermark=0)
    assert w.messages == []
    assert w.new_watermark == 0
    assert not w


def test_window_dispatches_codex_from_session_meta_shape() -> None:
    window = read_window(CODEX_FIXTURE, watermark=2)

    assert window.total_turns == 4
    assert [turn["text"] for turn in window.messages] == [
        "Resume the session and retain the existing deployment decision.",
        "The resumed turn keeps the release and environment cache key.",
    ]


def test_window_dispatches_cursor_from_role_message_shape() -> None:
    window = read_window(CURSOR_FIXTURE, watermark=3)

    assert window.total_turns == 7
    messages = [turn["text"] for turn in window.messages]
    assert messages[0] == "run the shell command `echo resumed_probe` and report output"
    assert "resumed_probe" in messages[1]
    assert messages[2:] == [
        "run the shell command `echo continue_probe`",
        "Output:\n\n```\ncontinue_probe\n```\n\n[REDACTED]",
    ]
