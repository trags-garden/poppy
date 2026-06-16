"""Integration: SessionEnd backstop reads the incremental window.

The SessionEnd consolidator now reads ``(watermark, end]`` through TranscriptWindow
instead of the last-60-messages slice. This fixes the long-session truncation bug
(early turns are captured), uses the watermark as the idempotency key, and flushes
only the tail the mid-session loop has not already captured.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from poppy.capture.watermark import get_watermark, set_watermark
from poppy.consolidation import consolidate_stop_event
from poppy.engine.seed import SeedEngine


def _transcript(tmp_path: Path, n: int, *, marker_first: str | None = None) -> Path:
    rows = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        text = marker_first if (i == 0 and marker_first) else f"turn {i} discusses a durable decision about module {i}"
        rows.append({"type": role, "message": {"role": role, "content": text}})
    p = tmp_path / "transcript.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


@pytest.fixture
def enabled_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SeedEngine:
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    engine = SeedEngine(db_path=tmp_path / "memories.db")
    # Inject the FTS-only engine so the test stays fast and offline.
    monkeypatch.setattr("poppy.consolidation.get_engine", lambda *_a, **_k: engine)
    return engine


def test_captures_early_turns_of_long_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled_env: SeedEngine
) -> None:
    """Regression: a >60-message session no longer drops its early turns."""
    seen: dict = {}

    def fake_call_llm(prompt, *, transcript_path, cfg):
        seen["prompt"] = prompt
        return [{"type": "decision", "content": "Use ruff for formatting."}]

    monkeypatch.setattr("poppy.consolidation.call_llm", fake_call_llm)
    p = _transcript(tmp_path, 70, marker_first="EARLY_MARKER_DECISION about the auth architecture")

    n = consolidate_stop_event({"session_id": "s1", "transcript_path": str(p), "cwd": str(tmp_path / "proj")})

    assert n == 1
    # The old last-60 reader would have dropped turn 0; the window includes it.
    assert "EARLY_MARKER_DECISION" in seen["prompt"]


def test_watermark_makes_refire_a_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled_env: SeedEngine
) -> None:
    monkeypatch.setattr(
        "poppy.consolidation.call_llm",
        lambda *a, **k: [{"type": "fact", "content": "The team uses uv for packaging."}],
    )
    p = _transcript(tmp_path, 6)
    payload = {"session_id": "s1", "transcript_path": str(p), "cwd": str(tmp_path / "proj")}

    n1 = consolidate_stop_event(payload)
    n2 = consolidate_stop_event(payload)

    assert n1 == 1
    assert n2 == 0  # window (6, 6] is empty — idempotent via the watermark
    assert get_watermark(tmp_path, "s1") == 6


def test_flushes_only_the_tail_after_watermark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled_env: SeedEngine
) -> None:
    captured: dict = {}

    def fake_call_llm(prompt, *, transcript_path, cfg):
        captured["prompt"] = prompt
        return [{"type": "fact", "content": "A durable tail fact about deployment."}]

    monkeypatch.setattr("poppy.consolidation.call_llm", fake_call_llm)
    p = _transcript(tmp_path, 10)
    # Pretend the mid-session loop already captured the first 4 turns.
    set_watermark(tmp_path, "s1", 4)

    n = consolidate_stop_event({"session_id": "s1", "transcript_path": str(p), "cwd": str(tmp_path / "proj")})

    assert n == 1
    assert "turn 9" in captured["prompt"]  # tail present
    assert "turn 0 " not in captured["prompt"]  # already-captured head excluded
    assert get_watermark(tmp_path, "s1") == 10


def test_tail_below_min_turns_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled_env: SeedEngine
) -> None:
    monkeypatch.setattr("poppy.consolidation.call_llm", lambda *a, **k: [{"type": "fact", "content": "x durable"}])
    p = _transcript(tmp_path, 6)
    set_watermark(tmp_path, "s1", 4)  # only 2 tail turns remain (< MIN_BACKSTOP_TURNS)

    n = consolidate_stop_event({"session_id": "s1", "transcript_path": str(p), "cwd": str(tmp_path / "proj")})

    assert n == 0


def test_session_start_resets_watermark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from click.testing import CliRunner

    from poppy.cli.hooks import hook

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    set_watermark(tmp_path, "s1", 5)

    payload = json.dumps({"cwd": str(tmp_path), "session_id": "s1", "hook_event_name": "SessionStart"})
    result = CliRunner().invoke(hook, ["session-start"], input=payload)

    assert result.exit_code == 0
    assert get_watermark(tmp_path, "s1") == 0
