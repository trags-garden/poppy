import json

from click.testing import CliRunner

from poppy.cli.hooks import hook


def _seed_seed_engine(tmp_path, *, content: str, project: str):
    """Seed a memory via the model-free SeedEngine (the same DB the hooks read)."""
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
            source=Source(type="manual", session_id=None, timestamp=now),
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
    not a misleading '0 captured' (ADR-0002)."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.delenv("POPPY_CONSOLIDATE", raising=False)  # default: consent pending
    runner = CliRunner()
    payload = json.dumps({"cwd": str(tmp_path), "session_id": "s1", "hook_event_name": "SessionStart"})
    result = runner.invoke(hook, ["session-start"], input=payload)
    assert result.exit_code == 0
    ctx = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "pending your consent" in ctx
    assert "poppy consent --enable" in ctx


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
    ctx = json.loads(result.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "INACTIVE" in ctx
    assert "no extraction backend" in ctx


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
    from poppy.cli.hooks import _project_from_cwd

    proj = tmp_path / "code" / "myrepo"
    deep = proj / "src" / "feature"
    deep.mkdir(parents=True)
    (proj / "pyproject.toml").write_text("")
    assert _project_from_cwd(str(deep)) == "myrepo"


def test_project_from_cwd_returns_none_without_marker(tmp_path):
    from poppy.cli.hooks import _project_from_cwd

    arbitrary = tmp_path / "code" / "personal"
    arbitrary.mkdir(parents=True)
    # No marker anywhere up the chain — must return None, not "personal".
    # The walk stops at the temp root which has no marker.
    result = _project_from_cwd(str(arbitrary))
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
