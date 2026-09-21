"""Tests for the transcript reader seam.

One module, one source-app adapter today — the Claude Code JSONL reader — plus
the shared ``extract_text``. Parsing is tested here, apart from ADR-0001's
watermark-after-success windowing (``test_window.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

from poppy.capture.transcript import (
    detect_transcript_host,
    extract_text,
    read_claude_code_turns,
    read_codex_turns,
    read_cursor_turns,
)

CODEX_FIXTURE = Path(__file__).parents[1] / "fixtures" / "codex" / "rollout-resumed.jsonl"
CURSOR_FIXTURES = Path(__file__).parents[1] / "fixtures" / "cursor"


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


# ---------- extract_text (shared) ----------


def test_extract_text_passthrough_string():
    assert extract_text("plain string") == "plain string"


def test_extract_text_keeps_only_text_blocks():
    content = [
        {"type": "text", "text": "keep me"},
        {"type": "thinking", "thinking": "drop me"},
        {"type": "tool_use", "id": "1"},
        {"type": "text", "text": "and me"},
    ]
    assert extract_text(content) == "keep me\nand me"


def test_extract_text_non_list_non_str_is_empty():
    assert extract_text(None) == ""
    assert extract_text(42) == ""


# ---------- Claude Code JSONL adapter ----------


def test_read_claude_code_turns_filters_roles_tools_and_compact(tmp_path: Path) -> None:
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
    turns = read_claude_code_turns(p)
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert turns[0]["text"] == "first user turn here"
    assert turns[1]["text"] == "assistant reply text"


def test_read_claude_code_turns_drops_sub_threshold(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "t.jsonl",
        [
            {"type": "user", "message": {"role": "user", "content": "a"}},  # len 1 < min_chars
            {"type": "assistant", "message": {"role": "assistant", "content": "ok now"}},
        ],
    )
    assert [t["text"] for t in read_claude_code_turns(p, min_chars=2)] == ["ok now"]


def test_read_claude_code_turns_truncates_each_message(tmp_path: Path) -> None:
    p = _write(tmp_path / "t.jsonl", [{"type": "user", "message": {"role": "user", "content": "x" * 5000}}])
    turns = read_claude_code_turns(p, per_message_chars=100)
    assert len(turns[0]["text"]) <= 101  # 100 + ellipsis
    assert turns[0]["text"].endswith("…")


def test_read_claude_code_turns_missing_file_is_empty(tmp_path: Path) -> None:
    assert read_claude_code_turns(tmp_path / "nope.jsonl") == []


# ---------- Codex rollout JSONL adapter ----------


def test_read_codex_turns_filters_synthetic_and_tolerates_unknown_rows() -> None:
    turns = read_codex_turns(CODEX_FIXTURE)

    assert detect_transcript_host(CODEX_FIXTURE) == "codex"
    assert [turn["role"] for turn in turns] == ["user", "assistant", "user", "assistant"]
    assert "recommended_plugins" not in " ".join(turn["text"] for turn in turns)
    assert turns[-1]["text"] == "The resumed turn keeps the release and environment cache key."


def test_read_codex_turns_sees_turns_appended_on_resume(tmp_path: Path) -> None:
    lines = CODEX_FIXTURE.read_text().splitlines()
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text("\n".join(lines[:-2]) + "\n")
    assert len(read_codex_turns(rollout)) == 2

    with rollout.open("a") as fh:
        fh.write("\n".join(lines[-2:]) + "\n")
    assert len(read_codex_turns(rollout)) == 4


# ---------- Cursor agent transcript JSONL adapter ----------


def test_read_cursor_turns_filters_wrappers_tools_redacted_and_unknown_rows() -> None:
    turns = read_cursor_turns(CURSOR_FIXTURES / "run-rich.jsonl")

    assert detect_transcript_host(CURSOR_FIXTURES / "run-rich.jsonl") == "cursor"
    assert turns == [
        {"role": "user", "text": "run a command, then create and edit dd.txt."},
        {"role": "assistant", "text": "Ran the command and updated `dd.txt` to `ab`.\n\n[REDACTED]"},
    ]
    rendered = " ".join(turn["text"] for turn in turns)
    assert "timestamp" not in rendered
    assert "user_query" not in rendered
    assert "REDACTED" in rendered
    assert "StrReplace" not in rendered


def test_read_cursor_turns_only_drops_marker_only_redacted_blocks(tmp_path: Path) -> None:
    transcript = tmp_path / "redacted.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "role": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": " [REDACTED] "},
                        {"type": "text", "text": "The literal [REDACTED] is part of this real answer."},
                    ]
                },
            }
        )
        + "\n"
    )

    assert read_cursor_turns(transcript) == [
        {"role": "assistant", "text": "The literal [REDACTED] is part of this real answer."}
    ]


def test_read_cursor_turns_sees_resume_appends_naturally(tmp_path: Path) -> None:
    lines = (CURSOR_FIXTURES / "run-after-resumes.jsonl").read_text().splitlines()
    transcript = tmp_path / "cursor.jsonl"
    transcript.write_text("\n".join(lines[:4]) + "\n")
    assert len(read_cursor_turns(transcript)) == 3

    with transcript.open("a") as fh:
        fh.write("\n".join(lines[4:]) + "\n")
    turns = read_cursor_turns(transcript)
    assert len(turns) == 7
    assert "continue_probe" in turns[-1]["text"]


def test_read_cursor_turns_tolerates_bom_blank_and_partial_lines(tmp_path: Path) -> None:
    transcript = tmp_path / "partial.jsonl"
    transcript.write_text(
        "\ufeff"
        + json.dumps(
            {
                "role": "user",
                "message": {"content": [{"type": "text", "text": "plain user text without wrappers"}]},
            }
        )
        + '\n\n{"role":"assistant"'
    )
    assert read_cursor_turns(transcript) == [{"role": "user", "text": "plain user text without wrappers"}]


def test_detect_transcript_host_empty_and_partial_are_unknown(tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    partial = tmp_path / "partial.jsonl"
    partial.write_text('{"role":"user"')

    assert detect_transcript_host(empty) is None
    assert detect_transcript_host(partial) is None


# ---------- Fixes: BOM detection, developer exclusion, per-block synthetic ----------


def test_detect_transcript_host_strips_bom_before_detection(tmp_path: Path) -> None:
    """A UTF-8 BOM on the first record must not misroute a Codex rollout to the
    Claude reader; detection re-runs on every read, so a late session_meta recovers."""
    from poppy.capture.transcript import read_transcript_turns

    p = tmp_path / "rollout.jsonl"
    p.write_text(
        "\ufeff"
        + json.dumps({"type": "session_meta", "payload": {"type": "session_meta", "id": "x"}})
        + "\n"
        + json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "real user question about cache keys"}],
                },
            }
        )
    )
    assert detect_transcript_host(p) == "codex"
    assert [t["role"] for t in read_transcript_turns(p)] == ["user"]


def test_read_codex_turns_excludes_developer_role(tmp_path: Path) -> None:
    """A non-synthetic developer turn is injected instruction content, never a
    memory — it is excluded like the Claude Code reader excludes non-user/assistant."""
    p = _write(
        tmp_path / "r.jsonl",
        [
            {"type": "session_meta", "payload": {"type": "session_meta", "id": "x"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "a real developer instruction, not synthetic"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "real user question about cache keys"}],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "real assistant answer about cache keys"}],
                },
            },
        ],
    )
    turns = read_codex_turns(p)
    assert [t["role"] for t in turns] == ["user", "assistant"]
    assert "developer instruction" not in " ".join(t["text"] for t in turns)


def test_read_codex_turns_filters_synthetic_per_block(tmp_path: Path) -> None:
    """Synthetic-context blocks are dropped individually: a real block in the same
    message survives, and a message of only synthetic blocks is dropped entirely."""
    p = _write(
        tmp_path / "r.jsonl",
        [
            {"type": "session_meta", "payload": {"type": "session_meta", "id": "x"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "<environment_context>synthetic env dump</environment_context>"},
                        {"type": "input_text", "text": "the real question that must survive"},
                    ],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "<cwd>/only/synthetic</cwd>"}],
                },
            },
        ],
    )
    turns = read_codex_turns(p)
    assert [t["text"] for t in turns] == ["the real question that must survive"]
    assert "synthetic env dump" not in " ".join(t["text"] for t in turns)
