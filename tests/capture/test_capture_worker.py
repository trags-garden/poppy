"""Integration: the silent mid-session capture worker.

Covers the worker end-to-end (window-only extraction, reconcile, journal,
watermark advance, provenance), the single-flight lock, the minimum-content gate,
the per-session soft cap, no-op-when-disabled, the no-double-extract guarantee
across a mid-session fire and the SessionEnd backstop, and the UserPromptSubmit
cadence trigger.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from poppy.capture.cadence import DEFAULT_SOFT_CAP_K, capture_count, record_capture
from poppy.capture.journal import read_last
from poppy.capture.lock import _lock_path
from poppy.capture.watermark import get_watermark
from poppy.consolidation import consolidate_capture_event, consolidate_stop_event
from poppy.engine.seed import SeedEngine


def _transcript(tmp_path: Path, n: int, *, prefix: str = "turn") -> Path:
    rows = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        text = f"{prefix} {i}: a substantive discussion of the deployment architecture and schema decision {i}"
        rows.append({"type": role, "message": {"role": role, "content": text}})
    p = tmp_path / "transcript.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


@pytest.fixture
def enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SeedEngine:
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    engine = SeedEngine(db_path=tmp_path / "memories.db")
    monkeypatch.setattr("poppy.consolidation.get_engine", lambda *_a, **_k: engine)
    return engine


def test_worker_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: SeedEngine) -> None:
    monkeypatch.setattr(
        "poppy.consolidation.call_llm",
        lambda *a, **k: [{"type": "decision", "content": "Adopt blue-green deployments for the API."}],
    )
    p = _transcript(tmp_path, 6)
    payload = {"session_id": "sess-1", "transcript_path": str(p), "cwd": str(tmp_path / "proj")}

    n = consolidate_capture_event(payload)

    assert n == 1
    assert get_watermark(tmp_path, "sess-1") == 6  # window-only extraction advanced the watermark
    stored = enabled.list_all(limit=50)
    assert len(stored) == 1
    # Provenance: real source app + real session id.
    assert stored[0].source.type == "claude-code"
    assert stored[0].source.session_id == "sess-1"
    # Journal recorded the capture.
    rec = read_last(tmp_path)
    assert rec is not None and rec.session_id == "sess-1" and rec.count == 1
    assert capture_count(tmp_path, "sess-1") == 1


def test_no_turn_extracted_twice_across_fire_and_backstop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: SeedEngine
) -> None:
    facts = [
        "Adopt blue-green deployments for the API.",
        "Cache rate-limit counters in Redis with a 60 second TTL.",
    ]
    prompts: list[str] = []

    def fake_call_llm(prompt, *, transcript_path, cfg):
        prompts.append(prompt)
        return [{"type": "fact", "content": facts[(len(prompts) - 1) % len(facts)]}]

    monkeypatch.setattr("poppy.consolidation.call_llm", fake_call_llm)
    p = _transcript(tmp_path, 6)
    payload = {"session_id": "sess-2", "transcript_path": str(p), "cwd": str(tmp_path / "proj")}

    consolidate_capture_event(payload)  # captures turns 0..5, watermark -> 6
    assert get_watermark(tmp_path, "sess-2") == 6

    # The session continues: append a 4-turn tail.
    with p.open("a") as fh:
        for i in range(6, 10):
            role = "user" if i % 2 == 0 else "assistant"
            fh.write(
                "\n"
                + json.dumps(
                    {
                        "type": role,
                        "message": {
                            "role": role,
                            "content": f"tail turn {i}: more decisions about caching and the gateway rate limits",
                        },
                    }
                )
            )

    consolidate_stop_event(payload)  # backstop flushes only the tail (6, 10]

    assert get_watermark(tmp_path, "sess-2") == 10
    last = prompts[-1]
    assert "tail turn 9" in last  # tail present
    assert "turn 0:" not in last  # head already captured — not re-extracted


def test_backstop_journals_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: SeedEngine) -> None:
    """The SessionEnd backstop must journal too — else a short session's
    only capture is invisible to the banner / `poppy doctor`."""
    monkeypatch.setattr(
        "poppy.consolidation.call_llm",
        lambda *a, **k: [{"type": "decision", "content": "Pin the answerer LLM when comparing experiments."}],
    )
    p = _transcript(tmp_path, 6)
    payload = {"session_id": "backstop-1", "transcript_path": str(p), "cwd": str(tmp_path / "proj")}

    n = consolidate_stop_event(payload)

    assert n == 1
    rec = read_last(tmp_path)
    assert rec is not None
    assert rec.session_id == "backstop-1"
    assert rec.count == 1


def test_single_flight_skips_when_lock_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: SeedEngine
) -> None:
    monkeypatch.setattr(
        "poppy.consolidation.call_llm", lambda *a, **k: [{"type": "fact", "content": "must not be stored"}]
    )
    p = _transcript(tmp_path, 6)
    # Simulate an in-flight worker by pre-holding the lock (fresh mtime → not stale).
    _lock_path(tmp_path, "sess-3").write_text("")

    n = consolidate_capture_event({"session_id": "sess-3", "transcript_path": str(p), "cwd": str(tmp_path)})

    assert n == 0
    assert get_watermark(tmp_path, "sess-3") == 0


def test_minimum_content_gate_skips_thin_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: SeedEngine
) -> None:
    called: list = []
    monkeypatch.setattr(
        "poppy.consolidation.call_llm", lambda *a, **k: called.append(1) or [{"type": "fact", "content": "x"}]
    )
    p = tmp_path / "transcript.jsonl"
    p.write_text(
        "\n".join(json.dumps({"type": "user", "message": {"role": "user", "content": "hi"}}) for _ in range(2))
    )

    n = consolidate_capture_event({"session_id": "sess-4", "transcript_path": str(p), "cwd": str(tmp_path)})

    assert n == 0
    assert called == []  # never reached the LLM


def test_soft_cap_defers_to_backstop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: SeedEngine) -> None:
    monkeypatch.setattr("poppy.consolidation.call_llm", lambda *a, **k: [{"type": "fact", "content": "past the cap"}])
    p = _transcript(tmp_path, 6)
    for _ in range(DEFAULT_SOFT_CAP_K):
        record_capture(tmp_path, "sess-5")

    n = consolidate_capture_event({"session_id": "sess-5", "transcript_path": str(p), "cwd": str(tmp_path)})

    assert n == 0


def test_noop_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.delenv("POPPY_CONSOLIDATE", raising=False)
    p = _transcript(tmp_path, 6)

    n = consolidate_capture_event({"session_id": "s", "transcript_path": str(p), "cwd": str(tmp_path)})

    assert n == 0


def _fake_popen(spawned: list):
    class FakeProc:
        def __init__(self, *a, **kw):
            spawned.append(a[0] if a else kw.get("args"))

            class _Stdin:
                def write(self, b):
                    pass

                def close(self):
                    pass

            self.stdin = _Stdin()

    return FakeProc


def test_user_prompt_submit_fires_capture_on_nth_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from click.testing import CliRunner

    from poppy.cli.hooks import hook

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    spawned: list = []
    monkeypatch.setattr("subprocess.Popen", _fake_popen(spawned))

    p = _transcript(tmp_path, 3)
    payload = json.dumps(
        {
            "cwd": str(tmp_path),
            "session_id": "hk",
            "transcript_path": str(p),
            "prompt": "a long enough prompt to pass the recall gate",
        }
    )
    runner = CliRunner()
    for _ in range(3):  # turns 1, 2, 3 — only the 3rd fires
        runner.invoke(hook, ["user-prompt-submit"], input=payload)

    capture_spawns = [s for s in spawned if s and list(s[1:]) == ["hook", "_capture-worker"]]
    assert len(capture_spawns) == 1


def test_user_prompt_submit_no_capture_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from click.testing import CliRunner

    from poppy.cli.hooks import hook

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.delenv("POPPY_CONSOLIDATE", raising=False)
    spawned: list = []
    monkeypatch.setattr("subprocess.Popen", _fake_popen(spawned))

    p = _transcript(tmp_path, 3)
    payload = json.dumps(
        {"cwd": str(tmp_path), "session_id": "hk", "transcript_path": str(p), "prompt": "a long enough prompt here"}
    )
    runner = CliRunner()
    for _ in range(3):
        runner.invoke(hook, ["user-prompt-submit"], input=payload)

    assert spawned == []  # disabled → never spawns a capture worker
