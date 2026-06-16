"""Tests for TranscriptWindow (ADR-0001: incremental capture window).

The window reader skips compact-summary entries, keeps user/assistant turns,
flattens text blocks, drops tool-only and sub-threshold turns, truncates each
message, and returns exactly the turns in ``(watermark, now]`` — clamping safely
when compaction shrinks the transcript below the watermark.
"""

from __future__ import annotations

import json
from pathlib import Path

from poppy.capture.window import read_turns, read_window


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_read_turns_filters_roles_tools_and_compact(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "t.jsonl",
        [
            {"type": "user", "message": {"role": "user", "content": "first user turn here"}},
            # thinking-only assistant turn flattens to empty -> dropped
            {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "x"}]}},
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "assistant reply text"}, {"type": "tool_use", "id": "1"}],
                },
            },
            {"type": "summary", "summary": "a compaction summary"},  # not user/assistant
            # explicit compact-summary marker on a user-typed row -> dropped
            {"type": "user", "isCompactSummary": True, "message": {"role": "user", "content": "COMPACT BOUNDARY"}},
            {"type": "system", "message": {"role": "system", "content": "noise"}},
        ],
    )
    turns = read_turns(p)
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[0]["text"] == "first user turn here"
    assert turns[1]["text"] == "assistant reply text"


def test_read_turns_drops_sub_threshold(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "t.jsonl",
        [
            {"type": "user", "message": {"role": "user", "content": "a"}},  # len 1 < min_chars
            {"type": "assistant", "message": {"role": "assistant", "content": "ok now"}},
        ],
    )
    assert [t["text"] for t in read_turns(p, min_chars=2)] == ["ok now"]


def test_read_turns_truncates_each_message(tmp_path: Path) -> None:
    p = _write(tmp_path / "t.jsonl", [{"type": "user", "message": {"role": "user", "content": "x" * 5000}}])
    turns = read_turns(p, per_message_chars=100)
    assert len(turns[0]["text"]) <= 101  # 100 + ellipsis
    assert turns[0]["text"].endswith("…")


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
