import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from poppy.config import PoppyConfig
from poppy.consolidation import (
    _compact_event_id,
    call_host_cli,
    call_llm,
    consolidate_compact_event,
    consolidate_stop_event,
    detect_host_cli,
    format_transcript,
    get_engine,
    get_poppy_dir,
    is_enabled,
    parse_json_array,
)
from poppy.models import Filters

CURSOR_FIXTURE = Path(__file__).parent / "fixtures" / "cursor" / "run-shell.jsonl"


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


def test_is_enabled_respects_per_project_off_switch(tmp_path, monkeypatch):
    """A deny-listed project is off even when global capture is on."""
    monkeypatch.delenv("POPPY_CONSOLIDATE", raising=False)
    monkeypatch.setattr("poppy.capture.policy.host_cli_available", lambda: True)
    cfg = PoppyConfig(poppy_dir=tmp_path, consent="granted", disabled_projects=["client-secret"])
    assert is_enabled(cfg, project="client-secret") is False  # gated by the deny-list
    assert is_enabled(cfg, project="my-oss") is True  # other projects unaffected
    assert is_enabled(cfg) is True  # no project context → global state


def test_parse_json_array_strips_fences():
    raw = '```json\n[{"type": "decision", "content": "use ruff"}]\n```'
    assert parse_json_array(raw) == [{"type": "decision", "content": "use ruff"}]


def test_parse_json_array_handles_malformed():
    assert parse_json_array("not json at all") == []
    assert parse_json_array("[]") == []
    assert parse_json_array('[{"type": "fact"}]') == []  # missing content


def test_parse_json_array_normalizes_unknown_type():
    assert parse_json_array('[{"type": "weird", "content": "x"}]') == [{"type": "fact", "content": "x"}]


def test_format_transcript_respects_char_budget():
    messages = [{"role": "user", "text": "x" * 5000}, {"role": "assistant", "text": "y" * 5000}]
    out = format_transcript(messages, char_budget=2000)
    # First message gets truncated to 1500 chars; budget stops the second.
    assert len(out) <= 2000 + 200
    assert "USER:" in out


def test_detect_host_cli_prefers_claude_for_claude_transcripts(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}" if name == "claude" else None)
    assert detect_host_cli("/Users/x/.claude/projects/abc/session.jsonl") == "claude"


def test_detect_host_cli_uses_cursor_agent_for_cursor_transcripts(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}" if name == "cursor-agent" else None)
    assert detect_host_cli(str(CURSOR_FIXTURE)) == "cursor-agent"


def test_detect_host_cli_returns_none_when_no_cli(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert detect_host_cli("/some/path.jsonl") is None


@pytest.mark.parametrize("cli_name", ["claude", "cursor-agent", "codex", "gemini"])
def test_call_host_cli_suppresses_hooks_in_child_environment(monkeypatch, cli_name):
    calls = []

    class Result:
        returncode = 0
        stdout = "[]"
        stderr = ""

    monkeypatch.setattr("poppy.consolidation.subprocess.run", lambda *args, **kwargs: calls.append(kwargs) or Result())

    assert call_host_cli("extract", cli=cli_name) == "[]"
    assert calls[0]["env"]["POPPY_SUPPRESS_HOOKS"] == "1"


def _host_cli_stub(monkeypatch, stdout: str, *, returncode: int = 0):
    """Make `claude` the detected host CLI and hand its process the given output.

    Both stand-ins are bound onto ``poppy.consolidation`` itself rather than onto
    the shared ``shutil`` / ``subprocess`` module objects, so no other importer
    sees a stubbed PATH lookup or a stubbed process spawn.
    """
    result = SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")
    monkeypatch.setattr(
        "poppy.consolidation.shutil",
        SimpleNamespace(which=lambda name: f"/usr/bin/{name}" if name == "claude" else None),
    )
    monkeypatch.setattr(
        "poppy.consolidation.subprocess",
        SimpleNamespace(run=lambda *args, **kwargs: result, TimeoutExpired=subprocess.TimeoutExpired),
    )


@pytest.mark.parametrize("stdout", ["[]", "```json\n[]\n```", "  [ ]  "])
def test_call_llm_never_spends_on_remote_when_host_cli_returns_empty(tmp_path, monkeypatch, stdout):
    """An empty array from the host CLI ends the extraction; it is not a failure.

    Most capture windows hold nothing durable, so `[]` is the common answer. Falling
    through to the openai-compat backend would auto-spend on a paid remote model on
    the majority of capture cycles for anyone with both backends configured.
    """
    _host_cli_stub(monkeypatch, stdout)
    remote_calls = []
    monkeypatch.setattr(
        "poppy.consolidation.call_openai_compat",
        lambda prompt, **kwargs: remote_calls.append(kwargs) or '[{"type": "fact", "content": "paid"}]',
    )
    cfg = PoppyConfig(poppy_dir=tmp_path, consolidate_model="gpt-4o-mini", consolidate_api_key="sk-test")

    assert call_llm("extract", transcript_path="/Users/x/.claude/projects/abc/session.jsonl", cfg=cfg) == []
    assert remote_calls == []


@pytest.mark.parametrize(
    "stdout",
    [
        '["Use PostgreSQL for persistent storage."]',  # bare strings, not objects
        '[{"type": "decision", "text": "Use PostgreSQL."}]',  # no "content" key
        '[{"type": "fact", "content": "   "}]',  # blank content
    ],
)
def test_call_llm_falls_back_to_remote_when_host_array_is_unusable(tmp_path, monkeypatch, capsys, stdout):
    """A non-empty array the parser cannot use is a broken answer, not "nothing durable".

    Only a genuinely empty array means the host CLI found nothing. When its output
    shape drifts from the prompt the array is non-empty but yields no memories, and
    the remote fallback has to stay reachable or the drift captures zero memories
    every cycle with a log line that claims there was nothing to keep.
    """
    _host_cli_stub(monkeypatch, stdout)
    remote_calls = []
    monkeypatch.setattr(
        "poppy.consolidation.call_openai_compat",
        lambda prompt, **kwargs: remote_calls.append(kwargs) or '[{"type": "decision", "content": "from remote"}]',
    )
    cfg = PoppyConfig(poppy_dir=tmp_path, consolidate_model="gpt-4o-mini", consolidate_api_key="sk-test")

    items = call_llm("extract", transcript_path="/Users/x/.claude/projects/abc/session.jsonl", cfg=cfg)
    assert items == [{"type": "decision", "content": "from remote"}]
    assert len(remote_calls) == 1
    assert "unparseable as JSON array" in capsys.readouterr().err


def test_call_llm_still_falls_back_to_remote_when_host_cli_fails(tmp_path, monkeypatch):
    """The fallback survives for the case it exists for: no usable host-CLI answer."""
    _host_cli_stub(monkeypatch, "", returncode=1)
    remote_calls = []
    monkeypatch.setattr(
        "poppy.consolidation.call_openai_compat",
        lambda prompt, **kwargs: remote_calls.append(kwargs) or '[{"type": "fact", "content": "from remote"}]',
    )
    cfg = PoppyConfig(poppy_dir=tmp_path, consolidate_model="gpt-4o-mini", consolidate_api_key="sk-test")

    items = call_llm("extract", transcript_path="/Users/x/.claude/projects/abc/session.jsonl", cfg=cfg)
    assert items == [{"type": "fact", "content": "from remote"}]
    assert len(remote_calls) == 1


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


def test_capture_scopes_project_by_marker_not_basename(tmp_path, monkeypatch):
    """Auto-capture must scope memories with the shared marker-walking resolver,
    not the naive ``Path(cwd).name``.

    Regression for the mis-scoping bug: a capture in a directory that is not a
    project root (``~/code/personal``, ``~/scratch``) was tagged with the bare
    basename, so the memory never resurfaced under recall's marker-walked scope.
    """
    from poppy.capture import journal

    store = tmp_path / "store"
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    monkeypatch.setenv("POPPY_DIR", str(store))
    monkeypatch.setattr(
        "poppy.consolidation.call_llm",
        lambda prompt, *, transcript_path, cfg: [{"type": "fact", "content": "The store is postgres."}],
    )

    # A bare directory with no project marker anywhere up the chain: the project
    # tag must be None, not "personal".
    bare = tmp_path / "code" / "personal"
    bare.mkdir(parents=True)
    consolidate_compact_event(
        {"session_id": "bare", "compact_summary": "one", "cwd": str(bare), "transcript_path": "/dev/null"}
    )
    assert journal.read_all(store)[-1]["project"] is None

    # A directory carrying a marker is tagged with that directory's name.
    marked = tmp_path / "code" / "myrepo"
    (marked / "src").mkdir(parents=True)
    (marked / "pyproject.toml").write_text("")
    consolidate_compact_event(
        {"session_id": "marked", "compact_summary": "two", "cwd": str(marked / "src"), "transcript_path": "/dev/null"}
    )
    assert journal.read_all(store)[-1]["project"] == "myrepo"


def test_consolidate_stop_event_stamps_source_from_host_cli(tmp_path, monkeypatch):
    """A Codex session's captured memory is stamped `codex`, not `claude-code`.

    The bug hardcoded ``source_type="claude-code"`` at every capture site even
    though ``detect_host_cli`` already classifies the host CLI. The transcript
    uses Codex's rollout shape and only ``codex`` resolves on PATH, so provenance
    must follow the real client without relying on the directory name.
    """
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    # Only `codex` is on PATH — detect_host_cli must classify this as codex.
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}" if name == "codex" else None)

    # A real Codex rollout shape under a Codex path.
    codex_dir = tmp_path / ".codex" / "sessions"
    codex_dir.mkdir(parents=True)
    transcript = codex_dir / "rollout.jsonl"
    rows = [
        {"type": "session_meta", "payload": {"type": "session_meta", "id": "codex-sess-1"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "How should we format Python?"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Use ruff for formatting and linting."}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "And for the package manager?"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Standardize on uv."}],
            },
        },
    ]
    transcript.write_text("\n".join(json.dumps(r) for r in rows))

    monkeypatch.setattr(
        "poppy.consolidation.call_llm",
        lambda prompt, *, transcript_path, cfg: [{"type": "decision", "content": "Use ruff for Python formatting."}],
    )

    # A marked project dir so project_from_cwd tags it (it returns None without a marker).
    proj = tmp_path / "myproj"
    proj.mkdir()
    (proj / "pyproject.toml").write_text("")

    payload = {
        "session_id": "codex-sess-1",
        "transcript_path": str(transcript),
        "cwd": str(proj),
    }
    stored = consolidate_stop_event(payload)
    assert stored == 1

    engine = get_engine(get_poppy_dir())
    memories = engine.list_all(filters=Filters(project="myproj"), limit=10)
    assert len(memories) == 1
    assert memories[0].source.type == "codex"
