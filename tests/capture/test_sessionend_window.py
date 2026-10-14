"""Integration: SessionEnd backstop reads the incremental window.

The SessionEnd consolidator now reads ``(watermark, end]`` through TranscriptWindow
instead of the last-60-messages slice. This fixes the long-session truncation bug
(early turns are captured), uses the watermark as the idempotency key, and flushes
only the tail the mid-session loop has not already captured.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from poppy.capture import lock
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


@pytest.mark.skipif(lock.fcntl is None, reason="flock requires POSIX")
@pytest.mark.parametrize("hook_name", ["SessionEnd", "sessionEnd"])
def test_session_end_waits_for_capture_and_extracts_appended_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled_env: SeedEngine, hook_name: str
) -> None:
    prompts = []

    def extract(prompt, **kwargs):
        prompts.append(prompt)
        return [{"type": "decision", "content": "Use a shared cache for rate limits."}]

    monkeypatch.setattr("poppy.consolidation.call_llm", extract)
    p = _transcript(tmp_path, 6)
    ready = threading.Event()
    contended = threading.Event()
    errors = []
    real_try_lock = lock._try_lock

    def try_lock(fd):
        acquired = real_try_lock(fd)
        if acquired is False:
            contended.set()
        return acquired

    monkeypatch.setattr(lock, "_try_lock", try_lock)

    def capture_holder():
        try:
            with lock.single_flight(tmp_path, "s1") as acquired:
                assert acquired
                ready.set()
                assert contended.wait(5)
                # New turns arrive while the earlier capture is still processing its window.
                with p.open("a") as transcript:
                    for i in range(6, 10):
                        role = "user" if i % 2 == 0 else "assistant"
                        transcript.write(
                            "\n" + json.dumps({"type": role, "message": {"role": role, "content": f"tail turn {i}"}})
                        )
                time.sleep(0.5)
                set_watermark(tmp_path, "s1", 6)
        except BaseException as exc:
            errors.append(exc)
            ready.set()

    holder = threading.Thread(target=capture_holder)
    holder.start()
    try:
        assert ready.wait(5)
        assert not errors
        n = consolidate_stop_event(
            {"session_id": "s1", "transcript_path": str(p), "cwd": str(tmp_path), "hook_event_name": hook_name}
        )
    finally:
        holder.join(timeout=5)
    assert not holder.is_alive()
    assert not errors
    assert n == 1
    assert len(prompts) == 1
    assert "tail turn 6" in prompts[0] and "tail turn 9" in prompts[0]
    assert "turn 0 " not in prompts[0] and "turn 5 " not in prompts[0]
    assert get_watermark(tmp_path, "s1") == 10


@pytest.mark.skipif(lock.fcntl is None, reason="flock requires POSIX")
@pytest.mark.parametrize("hook_name", ["SessionEnd", "sessionEnd", "preCompact", "Stop"])
def test_backstop_lock_wait_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled_env: SeedEngine, capsys, hook_name: str
) -> None:
    p = _transcript(tmp_path, 6)
    sleeps = []

    def unexpected_extraction(*args, **kwargs):
        raise AssertionError("a contended capture must skip extraction")

    monkeypatch.setattr("poppy.consolidation.call_llm", unexpected_extraction)
    with lock.single_flight(tmp_path, "s1") as acquired:
        assert acquired
        monkeypatch.setattr(
            lock, "time", SimpleNamespace(time=time.time, monotonic=lambda: sum(sleeps), sleep=sleeps.append)
        )
        assert (
            consolidate_stop_event({"session_id": "s1", "transcript_path": str(p), "hook_event_name": hook_name}) == 0
        )
    session_end = hook_name in ("SessionEnd", "sessionEnd")
    assert sum(sleeps) == pytest.approx(30 if session_end else 0.1)
    assert max(sleeps) == pytest.approx(0.25 if session_end else 0.025)
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == (1 if session_end else 0)
    if session_end:
        assert "SessionEnd" in lines[0] and "30s" in lines[0] and "skipping" in lines[0]
    assert get_watermark(tmp_path, "s1") == 0


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


def test_codex_resume_session_start_preserves_watermark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from click.testing import CliRunner

    from poppy.cli.hooks import hook

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    set_watermark(tmp_path, "s1", 5)

    # The session is identified as Codex from the rollout shape (a session_meta
    # record), never a directory name; a Codex resume appends to this same file.
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(json.dumps({"type": "session_meta", "payload": {"type": "session_meta", "id": "s1"}}))
    payload = json.dumps(
        {
            "cwd": str(tmp_path),
            "session_id": "s1",
            "hook_event_name": "SessionStart",
            "source": "resume",
            "transcript_path": str(transcript),
        }
    )
    result = CliRunner().invoke(hook, ["session-start"], input=payload)

    assert result.exit_code == 0
    assert get_watermark(tmp_path, "s1") == 5
