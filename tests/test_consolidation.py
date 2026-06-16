import json

from poppy.config import PoppyConfig
from poppy.consolidation import (
    _compact_event_id,
    consolidate_compact_event,
    detect_host_cli,
    format_transcript,
    is_enabled,
    parse_json_array,
    read_transcript_messages,
)


def test_is_enabled_respects_env(tmp_path, monkeypatch):
    monkeypatch.delenv("POPPY_CONSOLIDATE", raising=False)
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    cfg = PoppyConfig(poppy_dir=tmp_path)
    assert is_enabled(cfg) is False
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    assert is_enabled(cfg) is True


def test_is_enabled_respects_config(tmp_path, monkeypatch):
    monkeypatch.delenv("POPPY_CONSOLIDATE", raising=False)
    # A legacy consolidate-enabled true is grandfathered as consent, and
    # default-on then needs a free host-CLI backend present.
    monkeypatch.setattr("poppy.capture.policy.host_cli_available", lambda: True)
    cfg = PoppyConfig(poppy_dir=tmp_path, consolidate_enabled=True)
    assert is_enabled(cfg) is True
    # An explicit opt-out stays off even with a backend present.
    assert is_enabled(PoppyConfig(poppy_dir=tmp_path, consent="denied")) is False


def test_parse_json_array_strips_fences():
    raw = '```json\n[{"type": "decision", "content": "use ruff"}]\n```'
    assert parse_json_array(raw) == [{"type": "decision", "content": "use ruff"}]


def test_parse_json_array_handles_malformed():
    assert parse_json_array("not json at all") == []
    assert parse_json_array("[]") == []
    assert parse_json_array('[{"type": "fact"}]') == []  # missing content


def test_parse_json_array_normalizes_unknown_type():
    assert parse_json_array('[{"type": "weird", "content": "x"}]') == [{"type": "fact", "content": "x"}]


def test_read_transcript_messages_filters_tool_results(tmp_path):
    path = tmp_path / "t.jsonl"
    rows = [
        {"type": "user", "message": {"role": "user", "content": "hello"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "skip"}]}},
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "world"}, {"type": "tool_use", "id": "x"}],
            },
        },
        {"type": "system", "message": {"role": "system", "content": "noise"}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    msgs = read_transcript_messages(path)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["text"] == "hello"
    assert msgs[1]["text"] == "world"


def test_format_transcript_respects_char_budget():
    messages = [{"role": "user", "text": "x" * 5000}, {"role": "assistant", "text": "y" * 5000}]
    out = format_transcript(messages, char_budget=2000)
    # First message gets truncated to 1500 chars; budget stops the second.
    assert len(out) <= 2000 + 200
    assert "USER:" in out


def test_detect_host_cli_prefers_claude_for_claude_transcripts(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}" if name == "claude" else None)
    assert detect_host_cli("/Users/x/.claude/projects/abc/session.jsonl") == "claude"


def test_detect_host_cli_returns_none_when_no_cli(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert detect_host_cli("/some/path.jsonl") is None


def test_compact_event_id_is_stable_per_summary():
    a = _compact_event_id("sess1", "summary X")
    b = _compact_event_id("sess1", "summary X")
    c = _compact_event_id("sess1", "summary Y")
    assert a == b
    assert a != c
    assert a.startswith("sess1:compact:")


def test_consolidate_compact_event_disabled_is_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("POPPY_CONSOLIDATE", raising=False)
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    payload = {
        "session_id": "s1",
        "compact_summary": "We decided to use ruff for formatting.",
        "cwd": "/tmp/poppy",
    }
    assert consolidate_compact_event(payload) == 0


def test_consolidate_compact_event_skips_without_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    assert consolidate_compact_event({"session_id": "s1", "cwd": "/tmp/poppy"}) == 0
    assert consolidate_compact_event({"compact_summary": "x" * 100, "cwd": "/tmp/poppy"}) == 0


def test_consolidate_compact_event_stores_and_is_idempotent(tmp_path, monkeypatch):
    """Stub the LLM, run twice with the same payload, expect 1 fire only."""
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))

    fake_items = [{"type": "decision", "content": "Use ruff for formatting."}]
    calls = {"n": 0}

    def fake_call_llm(prompt, *, transcript_path, cfg):
        calls["n"] += 1
        return fake_items

    monkeypatch.setattr("poppy.consolidation.call_llm", fake_call_llm)

    payload = {
        "session_id": "s1",
        "compact_summary": "User and assistant agreed to adopt ruff for code formatting.",
        "cwd": str(tmp_path / "poppy"),
        "transcript_path": "/dev/null",
    }
    n1 = consolidate_compact_event(payload)
    n2 = consolidate_compact_event(payload)
    assert n1 == 1
    assert n2 == 0  # idempotent re-fire
    assert calls["n"] == 1  # second call short-circuited before LLM


def test_consolidate_compact_event_distinct_summaries_both_fire(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))

    # Distinct content per fire so the dedup-on-capture reconciler does
    # not skip the second as a duplicate — this test guards the per-compact-event
    # idempotency key, not the no-dedup behaviour the reconciler replaces.
    contents = [
        "The build uses Cargo for the Rust workspace crates.",
        "Deployments run on Kubernetes via Helm charts in us-east-1.",
    ]
    calls = {"n": 0}

    def fake_call_llm(prompt, *, transcript_path, cfg):
        out = [{"type": "fact", "content": contents[calls["n"] % len(contents)]}]
        calls["n"] += 1
        return out

    monkeypatch.setattr("poppy.consolidation.call_llm", fake_call_llm)

    base = {"session_id": "s1", "cwd": str(tmp_path / "poppy"), "transcript_path": "/dev/null"}
    n1 = consolidate_compact_event({**base, "compact_summary": "first compact"})
    n2 = consolidate_compact_event({**base, "compact_summary": "second compact"})
    assert n1 == 1
    assert n2 == 1
