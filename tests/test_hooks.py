import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from poppy.cli.hooks import _tool_file_paths, hook

CURSOR_RESUMED_FIXTURE = Path(__file__).parent / "fixtures" / "cursor" / "run-after-resumes.jsonl"


def _seed_seed_engine(
    tmp_path,
    *,
    content: str,
    project: str,
    source_type: str = "manual",
    session_id: str | None = None,
):
    """Seed a memory via the model-free SeedEngine (the same DB the hooks read).

    ``source_type`` / ``session_id`` default to a manual write; pass a host-CLI
    source and a session id to simulate an auto-captured memory.
    """
    import datetime as dt

    from poppy.engine.seed import SeedEngine
    from poppy.models import Memory, Source

    engine = SeedEngine(db_path=tmp_path / "memories.db")
    now = dt.datetime.now(dt.UTC)
    engine.ingest(
        Memory(
            id=f"mem_{abs(hash(content)) & 0xFFFFFF:06x}",
            content=content,
            memory_type="preference",
            source=Source(type=source_type, session_id=session_id, timestamp=now),
            project=project,
            related_to=[],
            created_at=now,
            updated_at=now,
            confidence=1.0,
        )
    )


def test_session_start_silent_when_disabled_and_no_memories(tmp_path, monkeypatch):
    """A deliberate off-state (env-off) with no memories produces no output —
    the banner only nags when capture is broken or pending, never when the user
    turned it off on purpose."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "0")  # explicit off → silent banner
    runner = CliRunner()
    payload = json.dumps({"cwd": str(tmp_path), "session_id": "s1", "hook_event_name": "SessionStart"})
    result = runner.invoke(hook, ["session-start"], input=payload)
    assert result.exit_code == 0
    assert result.stdout == ""  # no banner + no memories → no output


def test_session_start_active_banner_shows_counts(tmp_path, monkeypatch):
    """When capture is active the banner reports the project memory count and the
    last session's capture total (user stories 14-15)."""
    from poppy.capture import journal

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")  # forces active regardless of PATH/consent

    project_dir = tmp_path / "myproj"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text("")  # project marker → project="myproj"
    _seed_seed_engine(tmp_path, content="prefer asyncpg over psycopg", project="myproj")
    # A prior session captured 2 memories.
    journal.record(tmp_path, session_id="prev", project="myproj", count=2, items=[])

    runner = CliRunner()
    payload = json.dumps({"cwd": str(project_dir), "session_id": "now", "hook_event_name": "SessionStart"})
    result = runner.invoke(hook, ["session-start"], input=payload)
    assert result.exit_code == 0
    ctx = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "active" in ctx.lower()
    assert "1 memory for this project" in ctx
    assert "2 memories captured last session" in ctx
    assert "asyncpg" in ctx  # the memory list still follows the banner


def test_session_start_consent_pending_banner_points_to_enable(tmp_path, monkeypatch):
    """A fresh install (consent pending) shows the consent nudge, not silence and
    not a misleading '0 captured' (consent and default-on precedence)."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.delenv("POPPY_CONSOLIDATE", raising=False)  # default: consent pending
    runner = CliRunner()
    payload = json.dumps({"cwd": str(tmp_path), "session_id": "s1", "hook_event_name": "SessionStart"})
    result = runner.invoke(hook, ["session-start"], input=payload)
    assert result.exit_code == 0
    envelope = json.loads(result.stdout)
    ctx = envelope["hookSpecificOutput"]["additionalContext"]
    assert "pending your consent" in ctx
    assert "poppy autocapture on --global" in ctx
    # The nudge must also reach the human via systemMessage, not only the
    # agent-facing context channel.
    assert "pending your consent" in envelope["systemMessage"]
    assert "poppy autocapture on --global" in envelope["systemMessage"]


def test_session_start_inactive_when_consented_but_no_backend(tmp_path, monkeypatch):
    """Consented but no extraction backend → a loud INACTIVE banner, the
    silent-breakage case the banner exists to catch (user story 16)."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.delenv("POPPY_CONSOLIDATE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / "config.json").write_text(json.dumps({"consent": "granted"}))
    monkeypatch.setattr("poppy.capture.policy.host_cli_available", lambda: False)

    runner = CliRunner()
    payload = json.dumps({"cwd": str(tmp_path), "session_id": "s1", "hook_event_name": "SessionStart"})
    result = runner.invoke(hook, ["session-start"], input=payload)
    assert result.exit_code == 0
    envelope = json.loads(result.stdout)
    ctx = envelope["hookSpecificOutput"]["additionalContext"]
    assert "INACTIVE" in ctx
    assert "no extraction backend" in ctx
    # The INACTIVE breakage warning must reach the human too.
    assert "INACTIVE" in envelope["systemMessage"]
    assert "no extraction backend" in envelope["systemMessage"]


def test_session_start_shows_project_off_banner_without_system_message(tmp_path, monkeypatch):
    """Inside a deny-listed repo the SessionStart banner shows 'off for this
    project' in agent context, but does NOT nag via systemMessage (deliberate off-state)."""
    import json as _json

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.delenv("POPPY_CONSOLIDATE", raising=False)  # env must not override the per-project switch

    project_dir = tmp_path / "client-secret"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text("")  # project marker → project="client-secret"
    (tmp_path / "config.json").write_text(_json.dumps({"consent": "granted", "disabled_projects": ["client-secret"]}))

    runner = CliRunner()
    payload = json.dumps({"cwd": str(project_dir), "session_id": "s1", "hook_event_name": "SessionStart"})
    result = runner.invoke(hook, ["session-start"], input=payload)
    assert result.exit_code == 0
    envelope = json.loads(result.stdout)
    ctx = envelope["hookSpecificOutput"]["additionalContext"]
    assert "off for this project" in ctx
    assert "poppy autocapture on" in ctx
    assert "systemMessage" not in envelope


def test_session_start_active_banner_not_surfaced_as_system_message(tmp_path, monkeypatch):
    """The reassuring ACTIVE line stays agent-context only — no
    systemMessage, so a working install never nags the human every session."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")  # forces active
    _seed_seed_engine(tmp_path, content="prefer asyncpg over psycopg", project="proj")

    runner = CliRunner()
    payload = json.dumps({"cwd": str(tmp_path), "session_id": "s1", "hook_event_name": "SessionStart"})
    result = runner.invoke(hook, ["session-start"], input=payload)
    assert result.exit_code == 0
    envelope = json.loads(result.stdout)
    assert "active" in envelope["hookSpecificOutput"]["additionalContext"].lower()
    assert "systemMessage" not in envelope


def test_session_start_broken_engine_surfaced_as_system_message(tmp_path, monkeypatch):
    """A failed memory engine is the loudest INACTIVE case and must reach
    the human via systemMessage, not just agent context."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")

    def _boom(*_a, **_k):
        raise RuntimeError("engine down")

    monkeypatch.setattr("poppy.cli.hooks.get_fast_engine", _boom)

    runner = CliRunner()
    payload = json.dumps({"cwd": str(tmp_path), "session_id": "s1", "hook_event_name": "SessionStart"})
    result = runner.invoke(hook, ["session-start"], input=payload)
    assert result.exit_code == 0
    envelope = json.loads(result.stdout)
    assert "INACTIVE" in envelope["hookSpecificOutput"]["additionalContext"]
    assert "could not load its memory engine" in envelope["systemMessage"]


def test_session_start_with_memories(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    # Seed a memory via CLI runner's same engine path
    import datetime as dt

    from poppy.models import Memory, Source
    from poppy.runtime import get_engine

    engine = get_engine(tmp_path)
    project_dir = tmp_path / "myproj"
    project_dir.mkdir()
    now = dt.datetime.now(dt.UTC)
    engine.ingest(
        Memory(
            id="mem_test1",
            content="prefer asyncpg over psycopg",
            memory_type="preference",
            source=Source(type="manual", session_id=None, timestamp=now),
            project="myproj",
            related_to=[],
            created_at=now,
            updated_at=now,
            confidence=1.0,
        )
    )

    runner = CliRunner()
    payload = json.dumps({"cwd": str(project_dir), "session_id": "s1", "hook_event_name": "SessionStart"})
    result = runner.invoke(hook, ["session-start"], input=payload)
    assert result.exit_code == 0
    envelope = json.loads(result.stdout)
    assert envelope["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "asyncpg" in envelope["hookSpecificOutput"]["additionalContext"]


def _mem(content: str, memory_type: str = "fact"):
    from types import SimpleNamespace

    return SimpleNamespace(content=content, memory_type=memory_type)


def test_render_session_memories_truncates_each_memory():
    from poppy.cli.hooks import _SESSION_START_MEMORY_CHARS, _render_session_memories

    block = _render_session_memories([_mem("x" * 5000)], project="proj")
    line = block.splitlines()[1]
    assert line.endswith("…")
    # The rendered content is capped at the per-memory char budget, not 5000.
    assert len(line) <= len("- [fact] ") + _SESSION_START_MEMORY_CHARS + 1


def test_render_session_memories_stays_under_budget_and_marks_omissions():
    from poppy.cli.hooks import _SESSION_START_BLOCK_BUDGET, _render_session_memories

    memories = [_mem(f"memory number {i} " + "y" * 280) for i in range(30)]
    block = _render_session_memories(memories, project="proj")

    # The block never blows up: a bounded overshoot for the last line + marker
    # is fine, but it must stay near the budget, not balloon to KBs.
    assert len(block.encode("utf-8")) <= _SESSION_START_BLOCK_BUDGET + 512
    rendered = [ln for ln in block.splitlines() if ln.startswith("- [")]
    assert len(rendered) < len(memories)
    assert "more memories not shown" in block


def test_render_session_memories_shows_all_when_small():
    from poppy.cli.hooks import _render_session_memories

    memories = [_mem("short one"), _mem("short two"), _mem("short three")]
    block = _render_session_memories(memories, project="proj")
    assert "more memories not shown" not in block
    assert "…" not in block
    assert block.count("- [") == 3


def test_render_session_memories_giant_first_memory_is_truncated_not_dropped():
    from poppy.cli.hooks import _SESSION_START_BLOCK_BUDGET, _render_session_memories

    # A single enormous memory is per-memory truncated (so it fits the budget)
    # and still emitted; the block never balloons and later memories survive.
    block = _render_session_memories([_mem("z" * 10000), _mem("second")], project=None)
    first_line = block.splitlines()[1]
    assert first_line.startswith("- [") and first_line.endswith("…")
    assert "second" in block
    assert len(block.encode("utf-8")) <= _SESSION_START_BLOCK_BUDGET


def test_render_session_memories_empty_returns_none():
    from poppy.cli.hooks import _render_session_memories

    assert _render_session_memories([], project="proj") is None


def test_session_start_caps_a_single_long_memory(tmp_path, monkeypatch):
    """A single huge memory must not blow the injected block up."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    project_dir = tmp_path / "myproj"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text("")
    long_content = "SENTINEL_HEAD " + ("x" * 5000) + " SENTINEL_TAIL"
    _seed_seed_engine(tmp_path, content=long_content, project="myproj")

    runner = CliRunner()
    payload = json.dumps({"cwd": str(project_dir), "session_id": "s1", "hook_event_name": "SessionStart"})
    result = runner.invoke(hook, ["session-start"], input=payload)
    assert result.exit_code == 0
    ctx = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "SENTINEL_HEAD" in ctx  # the memory is still surfaced
    assert "SENTINEL_TAIL" not in ctx  # but truncated well before the tail
    assert "…" in ctx


def test_session_start_drops_and_marks_when_many_long_memories(tmp_path, monkeypatch):
    """Many long memories are capped to the budget with an explicit marker."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    project_dir = tmp_path / "myproj"
    project_dir.mkdir()
    (project_dir / "pyproject.toml").write_text("")
    for i in range(20):
        _seed_seed_engine(tmp_path, content=f"memory {i} " + ("q" * 280), project="myproj")

    runner = CliRunner()
    payload = json.dumps({"cwd": str(project_dir), "session_id": "s1", "hook_event_name": "SessionStart"})
    result = runner.invoke(hook, ["session-start"], input=payload)
    assert result.exit_code == 0
    ctx = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "more memories not shown" in ctx
    assert len([ln for ln in ctx.splitlines() if ln.startswith("- [")]) < 20


def test_session_start_invalid_json_fails_open(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(hook, ["session-start"], input="not json at all")
    assert result.exit_code == 0


def test_stop_silent(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.delenv("POPPY_CONSOLIDATE_MODEL", raising=False)
    runner = CliRunner()
    payload = json.dumps({"cwd": str(tmp_path), "session_id": "s1", "hook_event_name": "Stop"})
    result = runner.invoke(hook, ["stop"], input=payload)
    assert result.exit_code == 0
    assert result.stdout == ""


def _seed_memory(tmp_path, *, content: str, project: str = "myproj"):
    import datetime as dt

    from poppy.models import Memory, Source
    from poppy.runtime import get_engine

    engine = get_engine(tmp_path)
    now = dt.datetime.now(dt.UTC)
    engine.ingest(
        Memory(
            id=f"mem_{abs(hash(content)) & 0xFFFFFF:06x}",
            content=content,
            memory_type="preference",
            source=Source(type="manual", session_id=None, timestamp=now),
            project=project,
            related_to=[],
            created_at=now,
            updated_at=now,
            confidence=1.0,
        )
    )


def test_user_prompt_submit_short_prompt_no_op(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    runner = CliRunner()
    payload = json.dumps({"cwd": str(tmp_path), "prompt": "hi", "hook_event_name": "UserPromptSubmit"})
    result = runner.invoke(hook, ["user-prompt-submit"], input=payload)
    assert result.exit_code == 0
    assert result.stdout == ""


def test_user_prompt_submit_returns_envelope(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    project_dir = tmp_path / "myproj"
    project_dir.mkdir()
    _seed_memory(tmp_path, content="prefer asyncpg over psycopg")

    runner = CliRunner()
    payload = json.dumps(
        {"cwd": str(project_dir), "prompt": "should we use asyncpg here?", "hook_event_name": "UserPromptSubmit"}
    )
    result = runner.invoke(hook, ["user-prompt-submit"], input=payload)
    assert result.exit_code == 0
    envelope = json.loads(result.stdout)
    assert envelope["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "asyncpg" in envelope["hookSpecificOutput"]["additionalContext"]


def test_cursor_user_prompt_submit_runs_cadence_but_prints_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    _seed_memory(tmp_path, content="prefer asyncpg over psycopg")
    cadence_payloads = []
    monkeypatch.setattr("poppy.cli.hooks._maybe_fire_capture", lambda payload: cadence_payloads.append(payload))
    payload = json.dumps(
        {
            "cwd": str(tmp_path),
            "session_id": "cursor-prompt",
            "prompt": "should we use asyncpg here?",
            "hook_event_name": "beforeSubmitPrompt",
            "cursor_version": "2026.07.20-test",
        }
    )

    result = CliRunner().invoke(hook, ["user-prompt-submit"], input=payload)

    assert result.exit_code == 0
    assert result.stdout == ""
    assert len(cadence_payloads) == 1


def test_user_prompt_submit_falls_back_cross_project(tmp_path, monkeypatch):
    """When the cwd-derived project filter yields nothing, the hook should
    retry without the project filter and surface globally relevant memories."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    # Memory belongs to project "poppy"
    _seed_memory(tmp_path, content="prefer asyncpg over psycopg", project="poppy")

    # cwd is "personal" — no project marker, no memories scoped to it.
    unrelated = tmp_path / "personal"
    unrelated.mkdir()

    runner = CliRunner()
    payload = json.dumps(
        {"cwd": str(unrelated), "prompt": "should we use asyncpg here?", "hook_event_name": "UserPromptSubmit"}
    )
    result = runner.invoke(hook, ["user-prompt-submit"], input=payload)
    assert result.exit_code == 0
    envelope = json.loads(result.stdout)
    assert "asyncpg" in envelope["hookSpecificOutput"]["additionalContext"]


def test_project_from_cwd_walks_up_to_marker(tmp_path):
    from poppy.project import project_from_cwd

    proj = tmp_path / "code" / "myrepo"
    deep = proj / "src" / "feature"
    deep.mkdir(parents=True)
    (proj / "pyproject.toml").write_text("")
    assert project_from_cwd(str(deep)) == "myrepo"


def test_project_from_cwd_canonicalizes_symlink_alias(tmp_path):
    """A repo reached via a differently-named symlink alias must resolve to the
    same project name as its physical path: the per-project capture
    deny-list is exact-string membership, so a symlink-dependent name would let
    `poppy capture --off` be silently bypassed when the hook cwd is unresolved."""
    from poppy.project import project_from_cwd

    physical = tmp_path / "real-repo"
    physical.mkdir()
    (physical / "pyproject.toml").write_text("")
    alias = tmp_path / "aliased-name"
    alias.symlink_to(physical, target_is_directory=True)

    # Both the physical path and the alias path resolve to the physical basename.
    assert project_from_cwd(str(physical)) == "real-repo"
    assert project_from_cwd(str(alias)) == "real-repo"


def test_capture_off_survives_symlink_alias_cwd(tmp_path, monkeypatch):
    """End-to-end: `poppy capture --off` recorded at the physical path,
    then a SessionStart hook invoked with a symlink-alias cwd, still reports the
    project as disabled — the resolver canonicalizes both to one identity."""
    from poppy.cli.main import cli

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.delenv("POPPY_CONSOLIDATE", raising=False)

    physical = tmp_path / "client-secret"
    physical.mkdir()
    (physical / "pyproject.toml").write_text("")
    (tmp_path / "config.json").write_text(json.dumps({"consent": "granted"}))

    # Disable capture from the physical path.
    off = CliRunner().invoke(cli, ["capture", "--off", "--project", "client-secret"], env={"POPPY_DIR": str(tmp_path)})
    assert off.exit_code == 0

    # A session whose payload cwd is a symlink alias with a *different* basename
    # must still be recognized as the disabled project.
    alias = tmp_path / "totally-different-name"
    alias.symlink_to(physical, target_is_directory=True)
    payload = json.dumps({"cwd": str(alias), "session_id": "s1", "hook_event_name": "SessionStart"})
    result = CliRunner().invoke(hook, ["session-start"], input=payload)
    assert result.exit_code == 0
    ctx = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "off for this project" in ctx  # deny-list matched despite the alias name


def test_project_from_cwd_returns_none_without_marker(tmp_path):
    from poppy.project import project_from_cwd

    arbitrary = tmp_path / "code" / "personal"
    arbitrary.mkdir(parents=True)
    # No marker anywhere up the chain — must return None, not "personal".
    # The walk stops at the temp root which has no marker.
    result = project_from_cwd(str(arbitrary))
    # Allow either None or the basename if pytest tmp_path happens to live
    # under a directory containing a marker; prefer to assert "personal" did
    # not leak as a false positive.
    assert result != "personal" or result is None


def test_user_prompt_submit_no_results_silent(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    runner = CliRunner()
    payload = json.dumps(
        {"cwd": str(tmp_path), "prompt": "completely unrelated query", "hook_event_name": "UserPromptSubmit"}
    )
    result = runner.invoke(hook, ["user-prompt-submit"], input=payload)
    assert result.exit_code == 0
    assert result.stdout == ""


def test_pre_tool_use_no_file_path_silent(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    runner = CliRunner()
    payload = json.dumps({"cwd": str(tmp_path), "tool_name": "Edit", "tool_input": {}})
    result = runner.invoke(hook, ["pre-tool-use"], input=payload)
    assert result.exit_code == 0
    assert result.stdout == ""


def test_pre_tool_use_surfaces_by_filename(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    project_dir = tmp_path / "myproj"
    project_dir.mkdir()
    _seed_memory(tmp_path, content="auth.py uses JWT — never switch back to session cookies")

    runner = CliRunner()
    payload = json.dumps(
        {
            "cwd": str(project_dir),
            "tool_name": "Edit",
            "tool_input": {"file_path": str(project_dir / "src" / "auth.py")},
        }
    )
    result = runner.invoke(hook, ["pre-tool-use"], input=payload)
    assert result.exit_code == 0
    envelope = json.loads(result.stdout)
    assert "auth.py" in envelope["hookSpecificOutput"]["additionalContext"]


def test_cursor_pre_tool_use_emits_exact_native_allow_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    project_dir = tmp_path / "myproj"
    project_dir.mkdir()
    _seed_memory(tmp_path, content="auth.py uses JWT and must retain token validation")
    payload = json.dumps(
        {
            "cwd": str(project_dir),
            "session_id": "cursor-write",
            "tool_name": "Write",
            "tool_input": {"file_path": str(project_dir / "src" / "auth.py"), "content": "updated"},
            "hook_event_name": "preToolUse",
            "cursor_version": "2026.07.20-test",
        }
    )

    result = CliRunner().invoke(hook, ["pre-tool-use"], input=payload)

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "permission": "allow",
        "additional_context": "## Poppy memories touching `auth.py`:\n"
        "- [preference] auth.py uses JWT and must retain token validation",
    }
    assert "deny" not in result.stdout


@pytest.mark.parametrize(
    ("tool_name", "tool_input"),
    [("Read", {"file_path": "/workspace/auth.py"}), ("Shell", {"file_path": "/workspace/auth.py"})],
)
def test_cursor_pre_tool_use_skips_non_write_tools_before_engine_load(tmp_path, monkeypatch, tool_name, tool_input):
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setattr(
        "poppy.cli.hooks.get_fast_engine", lambda *_args, **_kwargs: pytest.fail("engine should not load")
    )
    payload = json.dumps(
        {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "hook_event_name": "preToolUse",
            "cursor_version": "2026.07.20-test",
        }
    )

    result = CliRunner().invoke(hook, ["pre-tool-use"], input=payload)

    assert result.exit_code == 0
    assert result.stdout == ""


def test_real_cursor_hook_fixture_preserves_relied_on_snake_case_keys():
    payload = json.loads((Path(__file__).parent / "fixtures" / "cursor" / "hook-pre-tool-use.json").read_text())

    assert set(payload) == {
        "session_id",
        "transcript_path",
        "cwd",
        "hook_event_name",
        "tool_name",
        "tool_input",
        "cursor_version",
    }
    assert payload["hook_event_name"] == "preToolUse"
    assert all(key not in payload for key in ("sessionId", "transcriptPath", "hookEventName", "toolName", "toolInput"))


def test_all_hook_entrypoints_exit_immediately_when_suppressed(monkeypatch):
    monkeypatch.setenv("POPPY_SUPPRESS_HOOKS", "1")
    monkeypatch.setattr("poppy.cli.hooks._read_hook_input", lambda: pytest.fail("hook input should not be read"))

    for command in hook.commands:
        result = CliRunner().invoke(hook, [command], input="not read")
        assert result.exit_code == 0, (command, result.output)
        assert result.output == "", command


@pytest.mark.parametrize("operation", ["Update", "Add", "Delete"])
def test_apply_patch_extracts_all_file_header_variants(operation):
    patch = f"*** Begin Patch\n*** {operation} File: src/poppy/capture/window.py\n*** End Patch"

    assert _tool_file_paths("apply_patch", {"command": patch}) == ["src/poppy/capture/window.py"]


def test_apply_patch_extracts_multiple_unique_paths():
    patch = """*** Begin Patch
*** Update File: src/a.py
*** Delete File: src/b.py
*** Add File: src/a.py
*** End Patch"""

    assert _tool_file_paths("apply_patch", {"command": patch}) == ["src/a.py", "src/b.py"]


def test_stop_only_stamps_last_seen(tmp_path, monkeypatch):
    from poppy.capture import _state

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")  # consent on → stamp allowed
    payload = json.dumps({"session_id": "codex-session", "hook_event_name": "Stop", "turn_id": "turn-1"})

    result = CliRunner().invoke(hook, ["stop"], input=payload)

    assert result.exit_code == 0
    entry = _state.load(tmp_path)["codex-session"]
    assert entry["last_seen_at"]
    assert "watermark" not in entry


def _stub_popen_no_op(monkeypatch):
    """Stub subprocess.Popen so detached workers don't actually spawn in tests."""

    class _Stub:
        def __init__(self, *a, **kw):
            class _Stdin:
                def write(self, b):
                    pass

                def close(self):
                    pass

            self.stdin = _Stdin()

    monkeypatch.setattr("subprocess.Popen", _Stub)


def test_session_end_consolidation_disabled_when_not_opted_in(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.delenv("POPPY_CONSOLIDATE", raising=False)
    _stub_popen_no_op(monkeypatch)
    payload = json.dumps(
        {
            "cwd": str(tmp_path),
            "session_id": "s1",
            "transcript_path": "/nonexistent",
            "hook_event_name": "SessionEnd",
        }
    )
    runner = CliRunner()
    result = runner.invoke(hook, ["session-end"], input=payload)
    assert result.exit_code == 0


def test_session_end_logs_payload_and_spawns_worker(tmp_path, monkeypatch):
    """session-end must log the payload AND attempt a detached worker spawn."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    spawned = {"called": False, "argv": None}

    class FakeProc:
        def __init__(self, *a, **kw):
            spawned["called"] = True
            spawned["argv"] = a[0] if a else kw.get("args")

            class _Stdin:
                def write(self, b):
                    pass

                def close(self):
                    pass

            self.stdin = _Stdin()

    monkeypatch.setattr("subprocess.Popen", FakeProc)

    payload = json.dumps(
        {
            "cwd": str(tmp_path / "proj"),
            "session_id": "abc",
            "transcript_path": "/dev/null",
            "hook_event_name": "SessionEnd",
            "reason": "exit",
        }
    )
    runner = CliRunner()
    result = runner.invoke(hook, ["session-end"], input=payload)
    assert result.exit_code == 0
    log_path = tmp_path / "sessionend-debug.log"
    assert log_path.exists()
    assert "abc" in log_path.read_text()
    assert spawned["called"] is True
    assert spawned["argv"][1:] == ["hook", "_session-end-worker"]


def test_session_end_worker_consolidates_when_enabled(tmp_path, monkeypatch):
    """The internal _session-end-worker is where the LLM call + storage runs."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")

    transcript_path = tmp_path / "transcript.jsonl"
    lines = []
    for i in range(6):
        role = "user" if i % 2 == 0 else "assistant"
        lines.append(json.dumps({"type": role, "message": {"role": role, "content": f"message body {i}"}}))
    transcript_path.write_text("\n".join(lines))

    monkeypatch.setattr(
        "poppy.consolidation.call_llm",
        lambda *a, **kw: [{"type": "decision", "content": "use ruff for formatting"}],
    )

    payload = json.dumps(
        {
            "cwd": str(tmp_path / "proj"),
            "session_id": "session-xyz",
            "transcript_path": str(transcript_path),
            "hook_event_name": "SessionEnd",
        }
    )
    runner = CliRunner()
    result = runner.invoke(hook, ["_session-end-worker"], input=payload)
    assert result.exit_code == 0

    from poppy.models import Filters
    from poppy.runtime import get_engine

    engine = get_engine(tmp_path)
    matched = [m for m in engine.list_all(filters=Filters(), limit=50) if m.source.session_id == "session-xyz"]
    assert len(matched) == 1
    assert matched[0].content == "use ruff for formatting"
    assert matched[0].memory_type == "decision"


def test_stop_hook_is_now_a_noop_even_when_consolidation_enabled(tmp_path, monkeypatch):
    """Regression guard: the legacy Stop hook must not run consolidation, even
    when POPPY_CONSOLIDATE=1 is set, because it fires every assistant turn."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    transcript_path = tmp_path / "transcript.jsonl"
    lines = [json.dumps({"type": "user", "message": {"role": "user", "content": f"m{i}"}}) for i in range(6)]
    transcript_path.write_text("\n".join(lines))
    monkeypatch.setattr(
        "poppy.consolidation.call_llm",
        lambda *a, **kw: [{"type": "fact", "content": "should never be stored via Stop"}],
    )
    payload = json.dumps(
        {
            "cwd": str(tmp_path / "proj"),
            "session_id": "stop-noop-session",
            "transcript_path": str(transcript_path),
            "hook_event_name": "Stop",
        }
    )
    runner = CliRunner()
    result = runner.invoke(hook, ["stop"], input=payload)
    assert result.exit_code == 0

    from poppy.models import Filters
    from poppy.runtime import get_engine

    engine = get_engine(tmp_path)
    memories = engine.list_all(filters=Filters(), limit=50)
    assert not any(m.source.session_id == "stop-noop-session" for m in memories), (
        "Stop hook must not run consolidation — consolidation lives in SessionEnd"
    )


def test_session_end_worker_consolidation_idempotent(tmp_path, monkeypatch):
    """Running the worker twice for the same session_id must not double-store."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")

    transcript_path = tmp_path / "transcript.jsonl"
    lines = [json.dumps({"type": "user", "message": {"role": "user", "content": f"m{i}"}}) for i in range(6)]
    transcript_path.write_text("\n".join(lines))

    monkeypatch.setattr(
        "poppy.consolidation.call_llm",
        lambda *a, **kw: [{"type": "fact", "content": "team uses ruff"}],
    )

    payload = json.dumps(
        {
            "cwd": str(tmp_path / "proj"),
            "session_id": "session-abc",
            "transcript_path": str(transcript_path),
        }
    )
    runner = CliRunner()
    runner.invoke(hook, ["_session-end-worker"], input=payload)
    runner.invoke(hook, ["_session-end-worker"], input=payload)

    from poppy.models import Filters
    from poppy.runtime import get_engine

    engine = get_engine(tmp_path)
    matched = [m for m in engine.list_all(filters=Filters(), limit=50) if m.source.session_id == "session-abc"]
    assert len(matched) == 1


def test_replay_session_end_runs_consolidator_against_logged_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")

    transcript_path = tmp_path / "transcript.jsonl"
    transcript_path.write_text(
        "\n".join(json.dumps({"type": "user", "message": {"role": "user", "content": f"m{i}"}}) for i in range(6))
    )
    monkeypatch.setattr(
        "poppy.consolidation.call_llm",
        lambda *a, **kw: [{"type": "decision", "content": "ship the feature"}],
    )
    log_path = tmp_path / "sessionend-debug.log"
    log_path.write_text(
        json.dumps(
            {
                "ts": "2026-05-04T18:00:00+00:00",
                "session_id": "abc",
                "transcript_bytes": 100,
                "payload": {
                    "session_id": "abc",
                    "cwd": str(tmp_path / "proj"),
                    "transcript_path": str(transcript_path),
                },
            }
        )
        + "\n"
    )

    runner = CliRunner()
    result = runner.invoke(hook, ["replay-session-end"])
    assert result.exit_code == 0
    assert "stored 1 memories" in result.output


def test_post_compact_logs_payload_and_spawns_worker(tmp_path, monkeypatch):
    """post-compact should write the debug log AND attempt a detached worker spawn."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    spawned = {"called": False, "argv": None}

    class FakeProc:
        def __init__(self, *a, **kw):
            spawned["called"] = True
            spawned["argv"] = a[0] if a else kw.get("args")

            class _Stdin:
                def write(self, b):
                    pass

                def close(self):
                    pass

            self.stdin = _Stdin()

    monkeypatch.setattr("subprocess.Popen", FakeProc)

    payload = json.dumps(
        {
            "session_id": "abc",
            "compact_summary": "we decided to use ruff",
            "cwd": str(tmp_path / "proj"),
            "transcript_path": "/dev/null",
        }
    )
    runner = CliRunner()
    result = runner.invoke(hook, ["post-compact"], input=payload)
    assert result.exit_code == 0
    log_path = tmp_path / "postcompact-debug.log"
    assert log_path.exists()
    assert "we decided to use ruff" in log_path.read_text()
    assert spawned["called"] is True
    assert spawned["argv"][1:] == ["hook", "_post-compact-worker"]


def test_postcompact_debug_log_hardened_and_trimmed(tmp_path, monkeypatch):
    """The post-compact debug log is written 0600 and persists only the
    replay-relevant payload keys, not the whole raw hook blob."""
    import stat as _stat
    import sys as _sys

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))

    class FakeProc:
        def __init__(self, *a, **kw):
            class _Stdin:
                def write(self, b):
                    pass

                def close(self):
                    pass

            self.stdin = _Stdin()

    monkeypatch.setattr("subprocess.Popen", FakeProc)

    payload = json.dumps(
        {
            "session_id": "abc",
            "compact_summary": "we decided to use ruff",
            "cwd": str(tmp_path / "proj"),
            "transcript_path": "/dev/null",
            "unrelated_secret": "should-not-persist",
        }
    )
    runner = CliRunner()
    result = runner.invoke(hook, ["post-compact"], input=payload)
    assert result.exit_code == 0

    log_path = tmp_path / "postcompact-debug.log"
    entry = json.loads(log_path.read_text().strip())
    # Replay-relevant fields survive so `replay-compact` still works...
    assert entry["payload"]["session_id"] == "abc"
    assert entry["payload"]["compact_summary"] == "we decided to use ruff"
    # ...but arbitrary extra keys from the raw blob are dropped.
    assert "unrelated_secret" not in entry["payload"]

    if _sys.platform != "win32":
        assert _stat.S_IMODE(log_path.stat().st_mode) == 0o600


def test_replay_compact_runs_consolidator_against_logged_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    monkeypatch.setattr(
        "poppy.consolidation.call_llm",
        lambda *a, **kw: [{"type": "decision", "content": "use ruff"}],
    )
    log_path = tmp_path / "postcompact-debug.log"
    log_path.write_text(
        json.dumps(
            {
                "ts": "2026-05-03T18:00:00+00:00",
                "session_id": "abc",
                "summary_len": 50,
                "payload": {
                    "session_id": "abc",
                    "compact_summary": "x" * 50,
                    "cwd": str(tmp_path / "proj"),
                    "transcript_path": "/dev/null",
                },
            }
        )
        + "\n"
    )

    runner = CliRunner()
    result = runner.invoke(hook, ["replay-compact"])
    assert result.exit_code == 0
    assert "stored 1 memories" in result.output


def test_session_start_floor_skips_recency_dump_but_keeps_banner(tmp_path, monkeypatch):
    """With a SessionStart floor configured, the unscored
    recency dump is suppressed while the status banner still shows."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")  # forces an active banner
    import datetime as dt

    from poppy.config import load_config, save_config
    from poppy.models import Memory, Source
    from poppy.runtime import get_engine

    engine = get_engine(tmp_path)
    project_dir = tmp_path / "myproj"
    project_dir.mkdir()
    now = dt.datetime.now(dt.UTC)
    engine.ingest(
        Memory(
            id="mem_floor1",
            content="prefer asyncpg over psycopg",
            memory_type="preference",
            source=Source(type="manual", session_id=None, timestamp=now),
            project="myproj",
            related_to=[],
            created_at=now,
            updated_at=now,
            confidence=1.0,
        )
    )
    cfg = load_config(tmp_path)
    cfg.session_start_min_score = 0.3
    save_config(cfg)

    runner = CliRunner()
    payload = json.dumps({"cwd": str(project_dir), "session_id": "s1", "hook_event_name": "SessionStart"})
    result = runner.invoke(hook, ["session-start"], input=payload)
    assert result.exit_code == 0
    ctx = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    # Recency dump suppressed by the floor...
    assert "asyncpg" not in ctx
    assert "## Poppy memories" not in ctx
    # ...but the banner (active) is still injected.
    assert "Poppy" in ctx


def test_session_start_floor_off_still_injects_memories(tmp_path, monkeypatch):
    """Default floor (0.0) leaves the recency dump behaviour unchanged."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    import datetime as dt

    from poppy.models import Memory, Source
    from poppy.runtime import get_engine

    engine = get_engine(tmp_path)
    project_dir = tmp_path / "myproj"
    project_dir.mkdir()
    now = dt.datetime.now(dt.UTC)
    engine.ingest(
        Memory(
            id="mem_floor2",
            content="prefer asyncpg over psycopg",
            memory_type="preference",
            source=Source(type="manual", session_id=None, timestamp=now),
            project="myproj",
            related_to=[],
            created_at=now,
            updated_at=now,
            confidence=1.0,
        )
    )
    runner = CliRunner()
    payload = json.dumps({"cwd": str(project_dir), "session_id": "s1", "hook_event_name": "SessionStart"})
    result = runner.invoke(hook, ["session-start"], input=payload)
    assert result.exit_code == 0
    ctx = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "asyncpg" in ctx


def test_session_start_floor_suppresses_even_a_giant_memory(tmp_path, monkeypatch):
    """The cap runs only when injection happens: with the floor set, a huge
    memory is fully suppressed — no truncated fragment or omission marker leaks
    (the gate short-circuits before the cap)."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    import datetime as dt

    from poppy.config import load_config, save_config
    from poppy.models import Memory, Source
    from poppy.runtime import get_engine

    engine = get_engine(tmp_path)
    project_dir = tmp_path / "myproj"
    project_dir.mkdir()
    now = dt.datetime.now(dt.UTC)
    engine.ingest(
        Memory(
            id="mem_floor_giant",
            content="SENTINEL_HEAD " + ("x" * 5000) + " SENTINEL_TAIL",
            memory_type="preference",
            source=Source(type="manual", session_id=None, timestamp=now),
            project="myproj",
            related_to=[],
            created_at=now,
            updated_at=now,
            confidence=1.0,
        )
    )
    cfg = load_config(tmp_path)
    cfg.session_start_min_score = 0.3
    save_config(cfg)

    runner = CliRunner()
    payload = json.dumps({"cwd": str(project_dir), "session_id": "s1", "hook_event_name": "SessionStart"})
    result = runner.invoke(hook, ["session-start"], input=payload)
    assert result.exit_code == 0
    ctx = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    # Nothing from the recency dump leaks: not even a per-memory-truncated head.
    assert "SENTINEL_HEAD" not in ctx
    assert "## Poppy memories" not in ctx
    assert "more memories not shown" not in ctx
    # ...but the banner (active) is still injected.
    assert "Poppy" in ctx


# --- activation funnel telemetry --------------------------------


def test_is_prior_session_autocapture_predicate():
    """The closed-loop predicate: only auto-captured memories (session id set) from
    a session other than the current one count."""
    import datetime as dt

    from poppy.cli.hooks import _is_prior_session_autocapture
    from poppy.models import Memory, Source

    def _mem(session_id):
        now = dt.datetime.now(dt.UTC)
        return Memory(
            id="m",
            content="c",
            memory_type="fact",
            source=Source(type="claude-code", session_id=session_id, timestamp=now),
            project="p",
            related_to=[],
            created_at=now,
            updated_at=now,
        )

    assert _is_prior_session_autocapture(_mem("prev"), "now") is True
    assert _is_prior_session_autocapture(_mem(None), "now") is False  # manual write
    assert _is_prior_session_autocapture(_mem("now"), "now") is False  # same session
    # A PostCompact capture from THIS session stamps "<session>:compact:<hash>".
    assert _is_prior_session_autocapture(_mem("now:compact:abc123"), "now") is False
    # ...but the same shape from a prior session does count.
    assert _is_prior_session_autocapture(_mem("prev:compact:abc123"), "now") is True


def _record_milestones(monkeypatch):
    """Capture telemetry.capture_once calls as (event, properties) tuples."""
    calls: list[tuple[str, dict | None]] = []
    monkeypatch.setattr("poppy.telemetry.capture_once", lambda _dir, event, props=None: calls.append((event, props)))
    return calls


def test_session_start_emits_closed_loop_for_prior_session_autocapture(tmp_path, monkeypatch):
    """The activation event fires when an auto-captured memory from a PRIOR session
    is injected into this one."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    calls = _record_milestones(monkeypatch)

    _seed_seed_engine(
        tmp_path, content="staging uses asyncpg", project="p", source_type="claude-code", session_id="prev"
    )
    runner = CliRunner()
    payload = json.dumps({"cwd": str(tmp_path), "session_id": "now", "hook_event_name": "SessionStart"})
    result = runner.invoke(hook, ["session-start"], input=payload)
    assert result.exit_code == 0

    closed = [props for (event, props) in calls if event == "closed_loop"]
    assert len(closed) == 1
    assert closed[0]["channel"] == "session_start"
    assert closed[0]["loop_count"] == 1


def test_session_start_no_closed_loop_for_manual_or_same_session(tmp_path, monkeypatch):
    """No activation for a manual memory (no session id) or a same-session capture."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    calls = _record_milestones(monkeypatch)

    _seed_seed_engine(tmp_path, content="a manual note", project="p")  # manual: session_id=None
    _seed_seed_engine(
        tmp_path, content="captured this session", project="p", source_type="claude-code", session_id="now"
    )
    runner = CliRunner()
    payload = json.dumps({"cwd": str(tmp_path), "session_id": "now", "hook_event_name": "SessionStart"})
    result = runner.invoke(hook, ["session-start"], input=payload)
    assert result.exit_code == 0
    assert not [e for (e, _p) in calls if e == "closed_loop"]


@pytest.mark.parametrize("source_type", ["claude-memory", "hermes-memory"])
def test_session_start_no_closed_loop_for_imported_memory(tmp_path, monkeypatch, source_type):
    """Synthetic import session ids are provenance, not auto-capture sessions."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    calls = _record_milestones(monkeypatch)

    _seed_seed_engine(
        tmp_path,
        content="imported memory",
        project="p",
        source_type=source_type,
        session_id="imported-project/MEMORY",
    )
    runner = CliRunner()
    payload = json.dumps({"cwd": str(tmp_path), "session_id": "now", "hook_event_name": "SessionStart"})
    result = runner.invoke(hook, ["session-start"], input=payload)
    assert result.exit_code == 0
    assert not [e for (e, _p) in calls if e == "closed_loop"]


def test_session_start_emits_consent_pending_shown(tmp_path, monkeypatch):
    """A fresh (consent-pending) install records that the human was shown the nudge."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.delenv("POPPY_CONSOLIDATE", raising=False)
    calls = _record_milestones(monkeypatch)

    runner = CliRunner()
    payload = json.dumps({"cwd": str(tmp_path), "session_id": "s1", "hook_event_name": "SessionStart"})
    result = runner.invoke(hook, ["session-start"], input=payload)
    assert result.exit_code == 0
    pending = [props for (event, props) in calls if event == "consent_pending_shown"]
    assert len(pending) == 1
    assert pending[0]["channel"] == "session_start"


def test_user_prompt_submit_emits_closed_loop_on_prior_session_recall(tmp_path, monkeypatch):
    """Recall injecting a prior-session auto-capture is also a closed loop."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    calls = _record_milestones(monkeypatch)

    _seed_seed_engine(
        tmp_path, content="prefer asyncpg over psycopg", project="p", source_type="claude-code", session_id="prev"
    )
    runner = CliRunner()
    payload = json.dumps({"cwd": str(tmp_path), "session_id": "now", "prompt": "how do we use asyncpg here?"})
    result = runner.invoke(hook, ["user-prompt-submit"], input=payload)
    assert result.exit_code == 0
    assert "asyncpg" in json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    closed = [props for (event, props) in calls if event == "closed_loop"]
    assert len(closed) == 1
    assert closed[0]["channel"] == "recall_prompt"


def test_emit_first_autocapture_records_trigger(tmp_path, monkeypatch):
    """The capture worker's helper records first_autocapture_stored with its path enum."""
    from poppy.capture.orchestrator import _emit_first_autocapture

    calls = _record_milestones(monkeypatch)
    _emit_first_autocapture(tmp_path, "mid_session")
    assert calls == [("first_autocapture_stored", {"trigger": "mid_session"})]


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_stop_consent_disabled_writes_no_state(tmp_path, monkeypatch):
    """An opted-out repo (consent off) must write no session state at all —
    the Stop liveness stamp is gated by the same consent check as capture."""
    from poppy.capture import _state

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "0")  # explicit off
    payload = json.dumps({"session_id": "codex-session", "cwd": str(tmp_path), "hook_event_name": "Stop"})

    result = CliRunner().invoke(hook, ["stop"], input=payload)

    assert result.exit_code == 0
    assert _state.load(tmp_path) == {}


def test_session_start_resets_watermark_for_claude_code_unconditionally(tmp_path, monkeypatch):
    """Claude Code resets the watermark on every SessionStart, including resume —
    unchanged from origin/main (zero behavior change on CC)."""
    from poppy.capture.watermark import get_watermark, set_watermark

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    set_watermark(tmp_path, "s1", 7)
    transcript = _write_jsonl(
        tmp_path / "cc.jsonl", [{"type": "user", "message": {"role": "user", "content": "hello there"}}]
    )
    payload = json.dumps(
        {
            "cwd": str(tmp_path),
            "session_id": "s1",
            "source": "resume",
            "transcript_path": str(transcript),
            "hook_event_name": "SessionStart",
        }
    )
    CliRunner().invoke(hook, ["session-start"], input=payload)
    assert get_watermark(tmp_path, "s1") == 0


def test_session_start_preserves_codex_watermark_on_missing_source(tmp_path, monkeypatch):
    """A Codex SessionStart with no source preserves the watermark — the rollout
    is appended in place, so resetting would re-extract the whole appended tail."""
    from poppy.capture.watermark import get_watermark, set_watermark

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    set_watermark(tmp_path, "s1", 7)
    transcript = _write_jsonl(
        tmp_path / "rollout.jsonl", [{"type": "session_meta", "payload": {"type": "session_meta", "id": "x"}}]
    )
    payload = json.dumps(
        {
            "cwd": str(tmp_path),
            "session_id": "s1",
            "transcript_path": str(transcript),
            "hook_event_name": "SessionStart",
        }
    )
    CliRunner().invoke(hook, ["session-start"], input=payload)
    assert get_watermark(tmp_path, "s1") == 7


def test_session_start_resets_codex_watermark_on_startup(tmp_path, monkeypatch):
    """A fresh Codex startup does reset the watermark."""
    from poppy.capture.watermark import get_watermark, set_watermark

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    set_watermark(tmp_path, "s1", 7)
    transcript = _write_jsonl(
        tmp_path / "rollout.jsonl", [{"type": "session_meta", "payload": {"type": "session_meta", "id": "x"}}]
    )
    payload = json.dumps(
        {
            "cwd": str(tmp_path),
            "session_id": "s1",
            "source": "startup",
            "transcript_path": str(transcript),
            "hook_event_name": "SessionStart",
        }
    )
    CliRunner().invoke(hook, ["session-start"], input=payload)
    assert get_watermark(tmp_path, "s1") == 0


def test_stamp_last_seen_never_regresses(tmp_path):
    """A later Stop that grabbed the lock must not overwrite a newer stamp."""
    from poppy.capture import _state

    future = "2099-01-01T00:00:00+00:00"
    _state.update(tmp_path, lambda d: d.setdefault("s", {}).update({"last_seen_at": future}))
    returned = _state.stamp_last_seen(tmp_path, "s")
    assert returned == future
    assert _state.load(tmp_path)["s"]["last_seen_at"] == future


def test_stamp_last_seen_advances_when_newer(tmp_path):
    from poppy.capture import _state

    past = "2000-01-01T00:00:00+00:00"
    _state.update(tmp_path, lambda d: d.setdefault("s", {}).update({"last_seen_at": past}))
    assert _state.stamp_last_seen(tmp_path, "s") > past


def test_cursor_session_start_null_transcript_resets_from_payload_host(tmp_path, monkeypatch):
    from poppy.capture.watermark import get_watermark, set_watermark

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.delenv("CURSOR_TRANSCRIPT_PATH", raising=False)
    set_watermark(tmp_path, "cursor-session", 8)
    payload = json.dumps(
        {
            "session_id": "cursor-session",
            "hook_event_name": "sessionStart",
            "cursor_version": "2026.07.20-test",
            "workspace_roots": [str(tmp_path)],
            "transcript_path": None,
        }
    )

    result = CliRunner().invoke(hook, ["session-start"], input=payload)

    assert result.exit_code == 0
    assert result.stdout == ""
    assert get_watermark(tmp_path, "cursor-session") == 0


def test_cursor_before_submit_uses_env_transcript_and_keeps_soft_cap(tmp_path, monkeypatch):
    from poppy.capture.cadence import DEFAULT_SOFT_CAP_K, record_capture

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    monkeypatch.setenv("CURSOR_TRANSCRIPT_PATH", str(CURSOR_RESUMED_FIXTURE))
    spawned: list = []
    monkeypatch.setattr(
        "poppy.cli.hooks._spawn_detached_worker",
        lambda subcommand, payload, log_name: spawned.append(subcommand),
    )
    for _ in range(DEFAULT_SOFT_CAP_K):
        record_capture(tmp_path, "cursor-capped")
    payload = json.dumps(
        {
            "session_id": "cursor-capped",
            "hook_event_name": "beforeSubmitPrompt",
            "cursor_version": "2026.07.20-test",
            "workspace_roots": [str(tmp_path)],
            "transcript_path": None,
            "prompt": "a long enough Cursor prompt for the cadence hook",
        }
    )

    for _ in range(3):
        CliRunner().invoke(hook, ["user-prompt-submit"], input=payload)

    assert "_capture-worker" not in spawned


def test_cursor_resume_without_session_start_preserves_existing_watermark(tmp_path, monkeypatch):
    from poppy.capture.watermark import get_watermark, set_watermark

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    set_watermark(tmp_path, "cursor-resume", 4)
    payload = json.dumps(
        {
            "session_id": "cursor-resume",
            "hook_event_name": "beforeSubmitPrompt",
            "cursor_version": "2026.07.20-test",
            "workspace_roots": [str(tmp_path)],
            "transcript_path": str(CURSOR_RESUMED_FIXTURE),
            "prompt": "continue the existing Cursor session without a new sessionStart",
        }
    )

    CliRunner().invoke(hook, ["user-prompt-submit"], input=payload)

    assert get_watermark(tmp_path, "cursor-resume") == 4


def test_cursor_session_end_worker_backstops_and_stamps_cursor_source(tmp_path, monkeypatch):
    from poppy.models import Filters
    from poppy.runtime import get_engine

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    monkeypatch.setattr(
        "poppy.consolidation.call_llm",
        lambda *a, **kw: [{"type": "decision", "content": "Cursor sessions retain the stable session id."}],
    )
    payload = json.dumps(
        {
            "session_id": "cursor-end",
            "hook_event_name": "sessionEnd",
            "cursor_version": "2026.07.20-test",
            "workspace_roots": [str(tmp_path)],
            "transcript_path": str(CURSOR_RESUMED_FIXTURE),
        }
    )

    result = CliRunner().invoke(hook, ["_session-end-worker"], input=payload)

    assert result.exit_code == 0
    stored = get_engine(tmp_path).list_all(filters=Filters(), limit=10)
    assert len(stored) == 1
    assert stored[0].source.type == "cursor"


def test_cursor_session_end_null_transcript_stamps_liveness_and_fails_open(tmp_path, monkeypatch):
    from poppy.capture import _state

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    monkeypatch.delenv("CURSOR_TRANSCRIPT_PATH", raising=False)
    _stub_popen_no_op(monkeypatch)
    payload = json.dumps(
        {
            "session_id": "cursor-null-end",
            "hook_event_name": "sessionEnd",
            "cursor_version": "2026.07.20-test",
            "workspace_roots": [str(tmp_path)],
            "transcript_path": None,
        }
    )

    result = CliRunner().invoke(hook, ["session-end"], input=payload)

    assert result.exit_code == 0
    assert _state.load(tmp_path)["cursor-null-end"]["last_seen_at"]


def test_cursor_precompact_routes_to_transcript_backstop(tmp_path, monkeypatch):
    from poppy.models import Filters
    from poppy.runtime import get_engine

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    monkeypatch.setattr(
        "poppy.consolidation.call_llm",
        lambda *a, **kw: [{"type": "fact", "content": "Cursor preCompact flushes the live transcript."}],
    )
    payload = json.dumps(
        {
            "session_id": "cursor-compact",
            "hook_event_name": "preCompact",
            "cursor_version": "2026.07.20-test",
            "workspace_roots": [str(tmp_path)],
            "transcript_path": str(CURSOR_RESUMED_FIXTURE),
        }
    )

    result = CliRunner().invoke(hook, ["_post-compact-worker"], input=payload)

    assert result.exit_code == 0
    stored = get_engine(tmp_path).list_all(filters=Filters(), limit=10)
    assert len(stored) == 1
    assert stored[0].source.type == "cursor"


def test_cursor_precompact_skips_extraction_when_session_lock_is_held(tmp_path, monkeypatch):
    from poppy.capture.lock import _lock_path
    from poppy.capture.watermark import get_watermark

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    calls = []
    monkeypatch.setattr(
        "poppy.consolidation.call_llm",
        lambda *args, **kwargs: calls.append((args, kwargs)) or [{"type": "fact", "content": "duplicate"}],
    )
    _lock_path(tmp_path, "cursor-compact-locked").write_text("")
    payload = json.dumps(
        {
            "session_id": "cursor-compact-locked",
            "hook_event_name": "preCompact",
            "cursor_version": "2026.07.20-test",
            "workspace_roots": [str(tmp_path)],
            "transcript_path": str(CURSOR_RESUMED_FIXTURE),
        }
    )

    result = CliRunner().invoke(hook, ["_post-compact-worker"], input=payload)

    assert result.exit_code == 0
    assert calls == []
    assert get_watermark(tmp_path, "cursor-compact-locked") == 0
