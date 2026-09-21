import json
import re
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from poppy.cli.main import cli
from poppy.config import load_config


@pytest.mark.parametrize(
    "args",
    [
        ["forget", "{id}"],
        ["forget", "{id}", "--yes"],
        ["edit", "{id}", "--project", "other"],
        ["remember", "replacement", "--supersedes", "{id}"],
    ],
)
def test_hidden_id_has_the_unknown_id_response(hidden_memory_store, monkeypatch, args):
    engine, memory = hidden_memory_store
    monkeypatch.setattr("poppy.cli.main._get_engine", lambda: engine)
    runner = CliRunner()
    hidden = runner.invoke(cli, [arg.format(id=memory.id) for arg in args])
    missing = runner.invoke(cli, [arg.format(id="missing") for arg in args])
    assert hidden.exit_code == missing.exit_code
    assert hidden.output.replace(memory.id, "missing") == missing.output
    assert "not found" in hidden.output.lower()
    assert memory.content not in hidden.stdout + hidden.stderr
    assert engine.get(memory.id) == memory


def test_doctor_omits_retired_copy_counts(hidden_memory_store, tmp_path):
    engine, memory = hidden_memory_store
    with engine._conn:
        engine._conn.execute(
            "INSERT INTO closet_migration_backup (id, content, action, migrated_at) VALUES (?, ?, ?, ?)",
            (memory.id, memory.content, "adopted", memory.updated_at.isoformat()),
        )
    result = CliRunner().invoke(cli, ["doctor"], env={"POPPY_DIR": str(tmp_path), "HOME": str(tmp_path)})
    assert result.exit_code == 0, result.output
    assert "storage" in result.output.lower()
    assert "per-speaker copies" not in result.output.lower()
    assert "closet" not in result.output.lower()
    assert memory.content not in result.stdout + result.stderr
    # The count and the deadline still get reported, because nothing else tells
    # a user that recoverable rows are sitting in the store. Count only, never
    # the rows, and never the name of the feature they came from.
    assert "kept pre-images" in result.output
    assert "1 row(s)" in result.output


@pytest.mark.parametrize("migrated_at", ["9999-12-31T23:59:59+00:00", "not-a-timestamp", ""])
def test_doctor_finishes_on_an_undateable_pre_image(hidden_memory_store, tmp_path, migrated_at):
    """One unreadable row in a side table must not cut the check list short."""
    engine, memory = hidden_memory_store
    with engine._conn:
        engine._conn.execute(
            "INSERT INTO closet_migration_backup (id, content, action, migrated_at) VALUES (?, ?, ?, ?)",
            (memory.id, memory.content, "adopted", migrated_at),
        )
    result = CliRunner().invoke(cli, ["doctor"], env={"POPPY_DIR": str(tmp_path), "HOME": str(tmp_path)})
    assert result.exit_code == 0, result.output
    # The checks that come after this line still ran.
    assert "custom redaction" in result.output
    assert memory.content not in result.stdout + result.stderr


def test_mcp_setup_commands_warn_when_poppy_executable_cannot_be_resolved(tmp_path, monkeypatch):
    monkeypatch.setattr("poppy.setup.claude_code.shutil.which", lambda _name: None)
    monkeypatch.setattr("poppy.setup.claude_code.sys.executable", str(tmp_path / "missing" / "python"))

    cases = (
        (["setup", "cursor"], {}),
        (["setup", "goose"], {"POPPY_GOOSE_CONFIG_DIR": str(tmp_path / "goose")}),
        (
            ["setup", "claude-desktop"],
            {"POPPY_CLAUDE_DESKTOP_CONFIG": str(tmp_path / "desktop.json")},
        ),
    )
    for index, (args, extra_env) in enumerate(cases):
        result = CliRunner().invoke(
            cli,
            args,
            env={
                "HOME": str(tmp_path),
                "POPPY_DIR": str(tmp_path / f".poppy-{index}"),
                "POPPY_TELEMETRY_OFF": "1",
                **extra_env,
            },
        )

        assert result.exit_code == 0, result.output
        assert "Warning: poppy could not be resolved to an absolute path" in result.output
        assert "client may not find poppy on its PATH" in result.output


def test_setup_codex_installs_hooks_and_explains_trust_gate(tmp_path):
    codex_home = tmp_path / "codex-home"
    result = CliRunner().invoke(
        cli,
        ["setup", "codex"],
        env={
            "HOME": str(tmp_path),
            "CODEX_HOME": str(codex_home),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )

    assert result.exit_code == 0, result.output
    hooks = json.loads((codex_home / "hooks.json").read_text())["hooks"]
    assert set(hooks) == {"SessionStart", "UserPromptSubmit", "PreToolUse", "Stop"}
    assert "one-time interactive approval" in result.output
    assert "in the TUI once" in result.output
    assert "silently skips" in result.output
    assert "requires approval again" in result.output


def test_setup_codex_no_hooks_skips_trust_message(tmp_path):
    codex_home = tmp_path / "codex-home"
    result = CliRunner().invoke(
        cli,
        ["setup", "codex", "--no-hooks"],
        env={
            "HOME": str(tmp_path),
            "CODEX_HOME": str(codex_home),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )

    assert result.exit_code == 0, result.output
    assert not (codex_home / "hooks.json").exists()
    assert "hook trust" not in result.output


def test_setup_cursor_installs_hooks_without_trust_gate(tmp_path):
    cursor_home = tmp_path / "cursor-home"
    result = CliRunner().invoke(
        cli,
        ["setup", "cursor"],
        env={
            "HOME": str(tmp_path),
            "CURSOR_HOME": str(cursor_home),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )

    assert result.exit_code == 0, result.output
    assert f"Cursor hooks are installed at {cursor_home / 'hooks.json'}" in result.output
    assert (cursor_home / "mcp.json").exists()
    assert "Capture starts immediately" in result.output
    assert "No hook trust step is required" in result.output
    assert "interactive approval" not in result.output


def test_setup_cursor_warns_foreign_event_and_suppresses_capture_success_copy(tmp_path):
    cursor_home = tmp_path / "cursor-home"
    cursor_home.mkdir()
    hooks_path = cursor_home / "hooks.json"
    hooks_path.write_text(json.dumps({"version": 1, "hooks": {"futureEvent": [{"command": "user hook"}]}}))

    result = CliRunner().invoke(
        cli,
        ["setup", "cursor"],
        env={
            "HOME": str(tmp_path),
            "CURSOR_HOME": str(cursor_home),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )

    assert result.exit_code == 0, result.output
    assert "Cursor will ignore the entire hooks file until these unrecognized event keys are removed" in result.output
    assert "futureEvent" in result.output
    assert "Capture starts immediately" not in result.output
    assert json.loads(hooks_path.read_text())["hooks"]["futureEvent"] == [{"command": "user hook"}]


def test_setup_cursor_no_hooks(tmp_path):
    cursor_home = tmp_path / "cursor-home"
    result = CliRunner().invoke(
        cli,
        ["setup", "cursor", "--no-hooks"],
        env={
            "CURSOR_HOME": str(cursor_home),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )

    assert result.exit_code == 0, result.output
    assert not (cursor_home / "hooks.json").exists()
    assert "Capture starts immediately" not in result.output


def test_setup_claude_desktop_reports_msix_dual_write(tmp_path):
    normal_config = tmp_path / "normal" / "claude_desktop_config.json"
    local_appdata = tmp_path / "LocalAppData"
    msix_dir = local_appdata / "Packages" / "Claude_random-id" / "LocalCache" / "Roaming" / "Claude"
    msix_dir.mkdir(parents=True)
    msix_config = msix_dir / "claude_desktop_config.json"

    result = CliRunner().invoke(
        cli,
        ["setup", "claude-desktop"],
        env={
            "HOME": str(tmp_path),
            "LOCALAPPDATA": str(local_appdata),
            "POPPY_CLAUDE_DESKTOP_CONFIG": str(normal_config),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )

    assert result.exit_code == 0, result.output
    assert f"MCP config: {normal_config}" in result.output
    assert f"MSIX MCP config: {msix_config}" in result.output
    assert "Microsoft Store/MSIX" in result.output
    assert "https://github.com/anthropics/claude-code/issues/26073" in result.output
    assert normal_config.exists()
    assert msix_config.exists()


def test_doctor_reports_msix_claude_desktop_registration(tmp_path):
    local_appdata = tmp_path / "LocalAppData"
    msix_dir = local_appdata / "Packages" / "Claude_random-id" / "LocalCache" / "Roaming" / "Claude"
    msix_dir.mkdir(parents=True)
    msix_config = msix_dir / "claude_desktop_config.json"
    msix_config.write_text(json.dumps({"mcpServers": {"poppy": {"command": "/resolved/poppy"}}}))

    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "HOME": str(tmp_path),
            "LOCALAPPDATA": str(local_appdata),
            "POPPY_CLAUDE_DESKTOP_CONFIG": str(tmp_path / "normal.json"),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )

    assert result.exit_code == 0, result.output
    assert "Claude desktop MSIX MCP: OK" in result.output
    assert str(msix_config) in result.output


def test_h_is_a_help_alias(tmp_path):
    """`poppy -h` must show help, not error. The group context also
    propagates the alias to subcommands, so `poppy remember -h` works too."""
    runner = CliRunner()
    root = runner.invoke(cli, ["-h"], env={"POPPY_DIR": str(tmp_path)})
    assert root.exit_code == 0
    assert "Usage:" in root.output
    assert "remember what matters" in root.output.lower()

    sub = runner.invoke(cli, ["remember", "-h"], env={"POPPY_DIR": str(tmp_path)})
    assert sub.exit_code == 0
    assert "Usage:" in sub.output


def _all_command_paths(cmd, prefix=()):
    """Yield the argv path to every command in the tree, root first."""
    yield list(prefix)
    if isinstance(cmd, click.Group):
        for name, sub in cmd.commands.items():
            yield from _all_command_paths(sub, (*prefix, name))


def test_no_internal_ids_in_help(tmp_path):
    """No internal tracker id (ADR-/TRA-/HP-####) may appear in the rendered
    --help of the root group or any subcommand. These ids are
    meaningless to a PyPI user; keep them in code comments, never on public
    CLI surfaces."""
    id_pattern = re.compile(r"\b(?:ADR|TRA|HP)-\d+", re.IGNORECASE)
    runner = CliRunner()
    offenders = []
    for path in _all_command_paths(cli):
        result = runner.invoke(cli, [*path, "--help"], env={"POPPY_DIR": str(tmp_path)})
        assert result.exit_code == 0, f"`poppy {' '.join(path)} --help` failed:\n{result.output}"
        for match in id_pattern.findall(result.output):
            offenders.append(f"poppy {' '.join(path) or '(root)'} --help contains {match!r}")
    assert not offenders, "internal ids leaked into public help:\n" + "\n".join(offenders)


def test_remember(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["remember", "use Pydantic for validation"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert "Remembered" in result.output


def test_remember_with_type(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["remember", "chose FastAPI over Flask", "--type", "decision"],
        env={"POPPY_DIR": str(tmp_path)},
    )
    assert result.exit_code == 0
    assert "decision" in result.output.lower()


def test_remember_check_with_supersedes_is_dry_run(tmp_path):
    """--check-conflicts with --supersedes must not tombstone the target.

    Regression: the dry-run early return once sat behind a `not supersedes`
    guard, so this combo silently tombstoned the target."""
    from poppy.ui.tombstones import TombstoneStore

    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path)}
    runner.invoke(cli, ["config", "set", "engine", "seed"], env=env)
    runner.invoke(cli, ["remember", "old fact"], env=env)
    mem_id = json.loads(runner.invoke(cli, ["list", "--json"], env=env).output)[0]["id"]

    result = runner.invoke(cli, ["remember", "new fact", "--supersedes", mem_id, "--check-conflicts"], env=env)
    assert result.exit_code == 0
    assert "Nothing was written" in result.output
    assert TombstoneStore(tmp_path / "memories.db").get(mem_id) is None
    listed = json.loads(runner.invoke(cli, ["list", "--json"], env=env).output)
    assert any(m["id"] == mem_id for m in listed), "target must survive a dry run"


def test_remember_bad_ttl_fails_before_engine(tmp_path, monkeypatch):
    """A bad --ttl fails fast (BadParameter, exit 2) before the engine is built."""

    def _boom(*a, **k):
        raise AssertionError("engine constructed before --ttl validation")

    monkeypatch.setattr("poppy.cli.main._runtime_get_engine", _boom)
    runner = CliRunner()
    result = runner.invoke(cli, ["remember", "x", "--ttl", "not-a-duration"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 2, result.output


def test_recall(tmp_path):
    runner = CliRunner()
    runner.invoke(cli, ["remember", "always use Pydantic validation"], env={"POPPY_DIR": str(tmp_path)})
    result = runner.invoke(cli, ["recall", "Pydantic"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert "Pydantic" in result.output


def test_recall_no_results(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["recall", "nonexistent topic"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert "No memories found" in result.output


def test_list(tmp_path):
    runner = CliRunner()
    runner.invoke(cli, ["remember", "memory one"], env={"POPPY_DIR": str(tmp_path)})
    runner.invoke(cli, ["remember", "memory two"], env={"POPPY_DIR": str(tmp_path)})
    result = runner.invoke(cli, ["list"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert "memory one" in result.output
    assert "memory two" in result.output


def test_forget(tmp_path):
    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path)}
    runner.invoke(cli, ["remember", "temporary memory"], env=env)
    # Get the memory ID from list --json (extract JSON from output in case of model loading noise)
    result = runner.invoke(cli, ["list", "--json"], env=env)
    # Find the JSON array — skip progress bars that may contain [ characters
    import re

    match = re.search(r"(\[\s*\{.*\}\s*\])", result.output, re.DOTALL)
    assert match, f"No JSON array found in output: {result.output[:200]}"
    memories = json.loads(match.group(1))
    mem_id = memories[0]["id"]

    result = runner.invoke(cli, ["forget", mem_id, "--yes"], env=env)
    assert result.exit_code == 0
    assert "Forgotten" in result.output


def test_stats(tmp_path):
    runner = CliRunner()
    runner.invoke(cli, ["remember", "test memory"], env={"POPPY_DIR": str(tmp_path)})
    result = runner.invoke(cli, ["stats"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert "1" in result.output  # memory count
    # Engine is whichever resolved on this machine: bloom (the default) or seed
    # as the final floor.
    assert any(e in result.output for e in ("bloom", "seed"))


def test_capture_off_and_on_with_explicit_project(tmp_path):
    """`poppy capture --off/--on --project X` toggles the per-project deny-list."""
    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path)}

    off = runner.invoke(cli, ["capture", "--off", "--project", "client-secret"], env=env)
    assert off.exit_code == 0
    assert "off for 'client-secret'" in off.output
    assert load_config(tmp_path).is_project_disabled("client-secret") is True

    # Status reflects the off state.
    status = runner.invoke(cli, ["capture", "--status", "--project", "client-secret"], env=env)
    assert status.exit_code == 0
    assert "off" in status.output.lower()

    on = runner.invoke(cli, ["capture", "--on", "--project", "client-secret"], env=env)
    assert on.exit_code == 0
    assert "back on for 'client-secret'" in on.output
    assert load_config(tmp_path).is_project_disabled("client-secret") is False


def test_capture_off_outside_a_project_errors(tmp_path, monkeypatch):
    """--off with no detectable project (and no --project) is a usage error, not a
    silent no-op that writes an empty/wrong tag."""
    workdir = tmp_path / "not-a-repo"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    runner = CliRunner()
    result = runner.invoke(cli, ["capture", "--off"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code != 0
    assert "Could not identify a project" in result.output


def test_capture_autodetects_project_from_cwd(tmp_path, monkeypatch):
    """Run inside a repo, `poppy capture --off` tags that repo without --project."""
    repo = tmp_path / "myrepo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("")  # project marker → project="myrepo"
    monkeypatch.chdir(repo)
    runner = CliRunner()
    result = runner.invoke(cli, ["capture", "--off"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert "off for 'myrepo'" in result.output
    assert load_config(tmp_path).is_project_disabled("myrepo") is True


def test_consent_enable_emits_consent_granted(tmp_path, monkeypatch):
    """`poppy consent --enable` records the consent_granted funnel milestone."""
    calls: list[tuple[str, dict | None]] = []
    monkeypatch.setattr("poppy.telemetry.capture_once", lambda _dir, event, props=None: calls.append((event, props)))
    result = CliRunner().invoke(cli, ["consent", "--enable"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert ("consent_granted", {"via": "consent_cmd"}) in calls


def test_record_agent_setup_emits_both_events(tmp_path, monkeypatch):
    """Every setup path routes through _record_agent_setup, which emits the
    per-install agent_setup event plus the once-per-device setup_completed
    funnel milestone."""
    from poppy.cli.main import _record_agent_setup

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    events: list[str] = []
    monkeypatch.setattr("poppy.telemetry.capture", lambda _dir, event, props=None: events.append(event))
    monkeypatch.setattr("poppy.telemetry.capture_once", lambda _dir, event, props=None: events.append(event) or True)
    _record_agent_setup("cursor")
    assert events == ["agent_setup", "setup_completed"]


def test_config_set_and_get(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["config", "set", "obsidian-vault", "/Users/test/cortex"],
        env={"POPPY_DIR": str(tmp_path)},
    )
    assert result.exit_code == 0


def test_setup_claude_code(tmp_path):
    # Create a fake claude settings directory
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["setup", "claude-code"],
        env={"POPPY_DIR": str(tmp_path), "CLAUDE_CONFIG_DIR": str(claude_dir)},
    )
    assert result.exit_code == 0
    assert "MCP config" in result.output
    assert "Poppy is ready" in result.output

    # Verify MCP config was written to ~/.claude.json (sibling of ~/.claude/),
    # NOT to ~/.claude/settings.json which holds hooks only.
    mcp_config_path = claude_dir.parent / ".claude.json"
    assert mcp_config_path.exists()
    mcp_data = json.loads(mcp_config_path.read_text())
    assert "poppy" in mcp_data["mcpServers"]

    # Verify the extraction backstop hooks were written to settings.json
    settings_path = claude_dir / "settings.json"
    assert settings_path.exists()
    settings_data = json.loads(settings_path.read_text())
    assert any(
        h.get("command") == "poppy hook session-start"
        for group in settings_data.get("hooks", {}).get("SessionStart", [])
        for h in group.get("hooks", [])
    )
    assert any(
        h.get("command") == "poppy hook session-end"
        for group in settings_data.get("hooks", {}).get("SessionEnd", [])
        for h in group.get("hooks", [])
    )
    assert any(
        h.get("command") == "poppy hook post-compact"
        for group in settings_data.get("hooks", {}).get("PostCompact", [])
        for h in group.get("hooks", [])
    )

    doctor_result = runner.invoke(
        cli,
        ["doctor"],
        env={"POPPY_DIR": str(tmp_path), "CLAUDE_CONFIG_DIR": str(claude_dir)},
    )
    assert doctor_result.exit_code == 0
    assert "PostCompact hook" in doctor_result.output

    # Verify CLAUDE.md block was written
    md_path = claude_dir / "CLAUDE.md"
    assert md_path.exists()
    assert "POPPY:BEGIN" in md_path.read_text()


def test_setup_claude_desktop_print_instructions(tmp_path):
    """--print-instructions emits the primer and skips installation."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["setup", "claude-desktop", "--print-instructions"],
        env={"POPPY_CLAUDE_DESKTOP_CONFIG": str(tmp_path / "should-not-exist.json")},
    )
    assert result.exit_code == 0
    # Points at the actual setting label in Claude desktop, not "Personal Preferences".
    assert "Instructions for Claude" in result.output
    assert "Poppy memory" in result.output  # body header
    assert "remember" in result.output and "recall_index" in result.output
    # Print mode must not write the config.
    assert not (tmp_path / "should-not-exist.json").exists()


def test_setup_claude_desktop_print_import_prompt(tmp_path):
    """--print-import-prompt emits the backfill prompt and skips installation."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["setup", "claude-desktop", "--print-import-prompt"],
        env={"POPPY_CLAUDE_DESKTOP_CONFIG": str(tmp_path / "should-not-exist.json")},
    )
    assert result.exit_code == 0
    assert "Export all of my stored memories" in result.output
    assert "remember(content, memory_type, project)" in result.output
    assert "recall_index" in result.output  # dedupe step
    assert not (tmp_path / "should-not-exist.json").exists()


def test_setup_claude_desktop_writes_config(tmp_path):
    """Plain invocation writes the MCP config and prints the primer hint."""
    target = tmp_path / "claude_desktop_config.json"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["setup", "claude-desktop"],
        env={"POPPY_CLAUDE_DESKTOP_CONFIG": str(target)},
    )
    assert result.exit_code == 0
    assert target.exists()
    settings = json.loads(target.read_text())
    assert "poppy" in settings["mcpServers"]
    assert "--print-instructions" in result.output
    assert "--print-import-prompt" in result.output


def test_doctor_reports_capture_consent_pending(tmp_path, monkeypatch):
    """doctor reports the granular capture status: a fresh (consent-pending) install
    shows the consent nudge, not a bare 'disabled' with a stale hint."""
    monkeypatch.delenv("POPPY_CONSOLIDATE", raising=False)
    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path), "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude")}
    runner.invoke(cli, ["config", "set", "engine", "seed"], env=env)
    result = runner.invoke(cli, ["doctor"], env=env)
    assert "auto-capture" in result.output
    assert "poppy autocapture on --global" in result.output
    # The old, wrong hint must be gone.
    assert "config set consolidate-enabled true" not in result.output


def test_doctor_no_backend_hint_lists_every_supported_host_cli(tmp_path, monkeypatch):
    monkeypatch.setattr("poppy.capture.policy.host_cli_available", lambda: False)
    (tmp_path / "config.json").write_text(json.dumps({"consent": "granted", "engine": "seed"}))

    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={"POPPY_DIR": str(tmp_path), "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude")},
    )

    assert result.exit_code == 0, result.output
    assert "install a supported host CLI (claude/cursor-agent/codex/gemini)" in result.output


def test_doctor_reports_last_capture(tmp_path):
    """doctor surfaces last-capture freshness from the journal."""
    from poppy.capture import journal

    journal.record(tmp_path, session_id="s1", project="poppy", count=3, items=[])
    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path), "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude")}
    runner.invoke(cli, ["config", "set", "engine", "seed"], env=env)
    result = runner.invoke(cli, ["doctor"], env=env)
    assert "last capture" in result.output
    assert "3 stored for poppy" in result.output


def test_doctor_reports_codex_hooks_and_capture_liveness(tmp_path):
    from poppy.capture import journal
    from poppy.setup.claude_code import install_codex_hooks

    poppy_dir = tmp_path / ".poppy"
    codex_home = tmp_path / "codex-home"
    install_codex_hooks(codex_home)
    runner = CliRunner()
    env = {
        "HOME": str(tmp_path),
        "CODEX_HOME": str(codex_home),
        "POPPY_DIR": str(poppy_dir),
        "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude"),
        "POPPY_TELEMETRY_OFF": "1",
        # Force capture on so it is genuinely expected to run: the liveness WARN
        # only fires when capture is active, so a missing capture is a real fault
        # here (deterministic, no host-CLI-on-PATH dependency).
        "POPPY_CONSOLIDATE": "1",
    }
    runner.invoke(cli, ["config", "set", "engine", "seed"], env=env)

    no_capture = runner.invoke(cli, ["doctor"], env=env)
    assert no_capture.exit_code == 0, no_capture.output
    assert "Codex hooks.json: OK" in no_capture.output
    assert "hooks installed but no capture observed" in no_capture.output
    assert "likely untrusted" in no_capture.output

    # A claude-code capture must NOT make the Codex liveness line read OK.
    journal.record(poppy_dir, session_id="cc-session", project="poppy", count=9, items=[], source="claude-code")
    still_dead = runner.invoke(cli, ["doctor"], env=env)
    assert "hooks installed but no capture observed" in still_dead.output

    journal.record(poppy_dir, session_id="codex-session", project="poppy", count=2, items=[], source="codex")
    live = runner.invoke(cli, ["doctor"], env=env)
    assert live.exit_code == 0, live.output
    assert "Codex capture liveness: OK" in live.output
    assert "2 stored" in live.output


def test_doctor_reports_cursor_hooks_account_flag_and_capture_liveness(tmp_path):
    from poppy.capture import journal
    from poppy.setup.claude_code import install_cursor_hooks

    poppy_dir = tmp_path / ".poppy"
    cursor_home = tmp_path / "cursor-home"
    install_cursor_hooks(cursor_home)
    runner = CliRunner()
    env = {
        "HOME": str(tmp_path),
        "CURSOR_HOME": str(cursor_home),
        "CODEX_HOME": str(tmp_path / "no-codex"),
        "POPPY_DIR": str(poppy_dir),
        "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude"),
        "POPPY_TELEMETRY_OFF": "1",
        # Force capture on so it is genuinely expected to run: the liveness WARN
        # only fires when capture is active, so a missing capture is a real fault
        # here (deterministic, no host-CLI-on-PATH dependency).
        "POPPY_CONSOLIDATE": "1",
    }
    runner.invoke(cli, ["config", "set", "engine", "seed"], env=env)

    no_capture = runner.invoke(cli, ["doctor"], env=env)
    assert no_capture.exit_code == 0, no_capture.output
    assert "Cursor hooks.json: OK" in no_capture.output
    assert "hooks installed but no capture observed" in no_capture.output
    assert "enable_execute_hook_exec" in no_capture.output

    journal.record(poppy_dir, session_id="cc", project="poppy", count=1, items=[], source="claude-code")
    still_dead = runner.invoke(cli, ["doctor"], env=env)
    assert "hooks installed but no capture observed" in still_dead.output

    journal.record(poppy_dir, session_id="cursor", project="poppy", count=2, items=[], source="cursor")
    live = runner.invoke(cli, ["doctor"], env=env)
    assert "Cursor capture liveness: OK" in live.output
    assert "2 stored" in live.output


def test_doctor_warns_unknown_cursor_event_disables_all_hooks(tmp_path):
    from poppy.setup.claude_code import install_cursor_hooks

    cursor_home = tmp_path / "cursor-home"
    hooks_path = install_cursor_hooks(cursor_home)
    settings = json.loads(hooks_path.read_text())
    settings["hooks"]["notADocumentedCursorEvent"] = [{"command": "user hook"}]
    hooks_path.write_text(json.dumps(settings))
    env = {
        "CURSOR_HOME": str(cursor_home),
        "CODEX_HOME": str(tmp_path / "no-codex"),
        "POPPY_DIR": str(tmp_path / ".poppy"),
        "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude"),
        "POPPY_TELEMETRY_OFF": "1",
    }
    CliRunner().invoke(cli, ["config", "set", "engine", "seed"], env=env)

    result = CliRunner().invoke(cli, ["doctor"], env=env)

    assert result.exit_code == 0, result.output
    assert "Cursor hooks.json: WARN" in result.output
    assert "Cursor will ignore the entire hooks file until these unrecognized event keys are removed" in result.output
    assert "notADocumentedCursorEvent" in result.output


def test_readme_documents_headless_cursor_backstop_only_capture():
    readme = (Path(__file__).parents[1] / "README.md").read_text()

    assert "In headless\n`cursor-agent -p` runs" in readme
    assert "capture is\nbackstop-only" in readme
    assert "`beforeSubmitPrompt` never fires" in readme
    assert "`sessionEnd` still flushes" in readme


def test_doctor_warns_invalid_cursor_hooks_json(tmp_path):
    cursor_home = tmp_path / "cursor-home"
    cursor_home.mkdir()
    # A real Poppy-in-Cursor user: the MCP entry registers Poppy (the footprint
    # doctor gates on), and the hooks.json is corrupt — which must be surfaced.
    (cursor_home / "mcp.json").write_text(json.dumps({"mcpServers": {"poppy": {"command": "poppy"}}}))
    (cursor_home / "hooks.json").write_text("{invalid")
    env = {
        "CURSOR_HOME": str(cursor_home),
        "CODEX_HOME": str(tmp_path / "no-codex"),
        "POPPY_DIR": str(tmp_path / ".poppy"),
        "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude"),
        "POPPY_TELEMETRY_OFF": "1",
    }
    CliRunner().invoke(cli, ["config", "set", "engine", "seed"], env=env)

    result = CliRunner().invoke(cli, ["doctor"], env=env)

    assert result.exit_code == 0, result.output
    assert "Cursor hooks.json: WARN" in result.output
    assert "invalid JSON or native Cursor hook shape" in result.output


def test_doctor_reports_capture_state(tmp_path):
    """doctor surfaces the per-session watermark/lock state."""
    from poppy.capture.watermark import set_watermark

    set_watermark(tmp_path, "s1", 4)
    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path), "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude")}
    runner.invoke(cli, ["config", "set", "engine", "seed"], env=env)
    result = runner.invoke(cli, ["doctor"], env=env)
    assert "capture state" in result.output
    assert "1 session(s) tracked" in result.output


def test_doctor_checks_hermes_soul_md_guidance(tmp_path):
    from poppy.setup.hermes import install_for_hermes

    hermes_home = tmp_path / ".hermes"
    install_for_hermes(hermes_home)
    (hermes_home / "SOUL.md").unlink()
    (hermes_home / "AGENTS.md").write_text("<!-- POPPY:BEGIN -->\nlegacy\n<!-- POPPY:END -->\n")
    runner = CliRunner()
    env = {
        "HOME": str(tmp_path),
        "HERMES_HOME": str(hermes_home),
        "POPPY_DIR": str(tmp_path / ".poppy"),
        "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude"),
    }
    runner.invoke(cli, ["config", "set", "engine", "seed"], env=env)

    missing_result = runner.invoke(cli, ["doctor"], env=env)
    assert missing_result.exit_code == 0
    assert f"Hermes SOUL.md guidance: WARN — {hermes_home / 'SOUL.md'}" in missing_result.output
    assert "install SOUL.md guidance" in missing_result.output

    install_for_hermes(hermes_home)
    installed_result = runner.invoke(cli, ["doctor"], env=env)
    assert installed_result.exit_code == 0
    assert f"Hermes SOUL.md guidance: OK — {hermes_home / 'SOUL.md'}" in installed_result.output


# --- main() console-script wrapper (offline error path) ---


def test_main_renders_model_unavailable_cleanly(monkeypatch, capsys):
    """The console-script wrapper renders ModelUnavailableError as one line, not a traceback."""
    import pytest

    from poppy.cli import main as main_mod
    from poppy.errors import ModelUnavailableError

    def boom():
        raise ModelUnavailableError(
            "Couldn't load retrieval model 'BAAI/bge-small-en-v1.5'. Connect to the internet "
            "for the one-time model download, or run `poppy engines use seed` for offline "
            "keyword-only search."
        )

    monkeypatch.setattr(main_mod, "cli", boom)
    with pytest.raises(SystemExit) as exc:
        main_mod.main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("poppy: ")
    assert "poppy engines use seed" in err
    assert "Traceback" not in err
    assert len(err.strip().splitlines()) == 1


def test_main_model_unavailable_from_engine_load_path(monkeypatch, capsys):
    """End to end: an offline cold-cache failure during engine construction inside a
    real command exits 1 with the clean message and no traceback."""
    import sys

    import pytest

    from poppy.cli import main as main_mod
    from poppy.errors import ModelUnavailableError

    def boom(_poppy_dir, **_kwargs):
        raise ModelUnavailableError(
            "Couldn't load retrieval model 'BAAI/bge-small-en-v1.5'. Connect to the internet "
            "for the one-time model download, or run `poppy engines use seed` for offline "
            "keyword-only search."
        )

    monkeypatch.setattr(main_mod, "_runtime_get_engine", boom)
    monkeypatch.setattr(sys, "argv", ["poppy", "recall", "anything"])
    with pytest.raises(SystemExit) as exc:
        main_mod.main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "poppy: Couldn't load retrieval model" in err
    assert "Traceback" not in err


def test_main_normal_run_unaffected(monkeypatch, capsys):
    """A healthy invocation through the wrapper behaves exactly like cli()."""
    import sys

    import pytest

    monkeypatch.setattr(sys, "argv", ["poppy", "--version"])
    from poppy.cli import main as main_mod

    with pytest.raises(SystemExit) as exc:
        main_mod.main()
    assert exc.value.code == 0
    assert "poppy" in capsys.readouterr().out


def test_console_script_points_at_wrapper():
    """[project.scripts] must target main() so ModelUnavailableError renders cleanly
    instead of dumping a raw traceback on an offline first run."""
    import tomllib
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as f:
        scripts = tomllib.load(f)["project"]["scripts"]
    assert scripts["poppy"] == "poppy.cli.main:main"


def test_doctor_skips_codex_section_when_codex_absent(tmp_path):
    """A Claude-Code-only user with no Codex config gets no Codex hooks/liveness
    lines — the section is gated on Codex being present, so no misleading WARN."""
    from poppy.cli.main import cli

    runner = CliRunner()
    env = {
        "CODEX_HOME": str(tmp_path / "no-codex-here"),  # does not exist
        "POPPY_DIR": str(tmp_path / ".poppy"),
        "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude"),
        "POPPY_TELEMETRY_OFF": "1",
    }
    runner.invoke(cli, ["config", "set", "engine", "seed"], env=env)
    result = runner.invoke(cli, ["doctor"], env=env)
    assert result.exit_code == 0, result.output
    assert "Codex hooks.json" not in result.output
    assert "Codex capture liveness" not in result.output


def test_doctor_skips_cursor_section_when_cursor_absent(tmp_path):
    runner = CliRunner()
    env = {
        "CURSOR_HOME": str(tmp_path / "no-cursor-here"),
        "CODEX_HOME": str(tmp_path / "no-codex-here"),
        "POPPY_DIR": str(tmp_path / ".poppy"),
        "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude"),
        "POPPY_TELEMETRY_OFF": "1",
    }
    runner.invoke(cli, ["config", "set", "engine", "seed"], env=env)

    result = runner.invoke(cli, ["doctor"], env=env)

    assert result.exit_code == 0, result.output
    assert "Cursor hooks.json" not in result.output
    assert "Cursor capture liveness" not in result.output


def test_doctor_shows_codex_section_when_codex_present(tmp_path):
    """When Poppy is set up in Codex, the doctor shows the Codex hooks section.
    (A bare ~/.codex/config.toml with no Poppy stays silent.)"""
    from poppy.cli.main import cli
    from poppy.setup.claude_code import install_codex_hooks

    codex_home = tmp_path / "codex-home"
    install_codex_hooks(codex_home)
    runner = CliRunner()
    env = {
        "CODEX_HOME": str(codex_home),
        "POPPY_DIR": str(tmp_path / ".poppy"),
        "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude"),
        "POPPY_TELEMETRY_OFF": "1",
    }
    runner.invoke(cli, ["config", "set", "engine", "seed"], env=env)
    result = runner.invoke(cli, ["doctor"], env=env)
    assert result.exit_code == 0, result.output
    assert "Codex hooks.json" in result.output


def test_doctor_skips_codex_section_when_config_present_but_no_poppy(tmp_path):
    """A bare ~/.codex/config.toml (e.g. just `model = ...`) with no Poppy setup
    must NOT trigger Codex WARNs — config-file existence is not a Poppy footprint."""
    from poppy.cli.main import cli

    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text('model = "gpt-5"\n')
    runner = CliRunner()
    env = {
        "HOME": str(tmp_path),
        "CODEX_HOME": str(codex_home),
        "POPPY_DIR": str(tmp_path / ".poppy"),
        "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude"),
        "POPPY_TELEMETRY_OFF": "1",
    }
    runner.invoke(cli, ["config", "set", "engine", "seed"], env=env)
    result = runner.invoke(cli, ["doctor"], env=env)
    assert result.exit_code == 0, result.output
    assert "Codex hooks.json" not in result.output
    assert "Codex MCP" not in result.output
    assert "Codex primer" not in result.output


def _fresh_setup_env(tmp_path):
    """Isolated env for a fresh single-client install on a REALISTIC machine:
    Claude Code is already present (its config dir + ~/.claude.json exist) but
    Poppy was never set up there. ~/.claude.json exists for every Claude Code
    user, so gating the doctor's Claude Code section on that file's existence
    would wrongly warn a Cursor-only user. Plus a bogus daemon
    port so inspect_daemon can't reach a real running daemon."""
    claude_dir = tmp_path / "claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    # A bare Claude Code settings.json (hooks/permissions live here) with no
    # Poppy hooks, and the sibling ~/.claude.json (MCP servers + project history)
    # with no "poppy" server — exactly what a Claude Code user who never ran
    # `poppy setup claude-code` has on disk.
    (claude_dir / "settings.json").write_text(json.dumps({"permissions": {"allow": []}}))
    (claude_dir / ".claude.json").write_text(json.dumps({"mcpServers": {}, "projects": {}}))
    return {
        "HOME": str(tmp_path),
        "POPPY_DIR": str(tmp_path / ".poppy"),
        "CLAUDE_CONFIG_DIR": str(claude_dir),
        "CURSOR_HOME": str(tmp_path / "cursor"),
        "CODEX_HOME": str(tmp_path / "codex"),
        # A port nothing listens on, so inspect_daemon can't reach a real daemon.
        "POPPY_DAEMON_PORT": "59321",
        "POPPY_TELEMETRY_OFF": "1",
    }


def test_doctor_clean_after_fresh_cursor_setup(tmp_path):
    """Done-when: `poppy setup cursor` then `poppy doctor` prints ZERO
    WARN lines — no daemon warns (daemon is opt-in) and no Claude Code warns
    (the user chose Cursor)."""
    runner = CliRunner()
    env = _fresh_setup_env(tmp_path)
    runner.invoke(cli, ["config", "set", "engine", "seed"], env=env)

    setup = runner.invoke(cli, ["setup", "cursor"], env=env)
    assert setup.exit_code == 0, setup.output

    doctor = runner.invoke(cli, ["doctor"], env=env)
    assert doctor.exit_code == 0, doctor.output
    assert "WARN" not in doctor.output, doctor.output
    # Cursor-only: no Claude Code section, no daemon section.
    assert "claude config dir" not in doctor.output
    assert "MCP server registered" not in doctor.output
    assert "daemon installed" not in doctor.output
    # The client the user did choose is present and healthy.
    assert "Cursor MCP: OK" in doctor.output


def test_doctor_clean_after_fresh_codex_setup(tmp_path):
    """Done-when: `poppy setup codex` then `poppy doctor` prints ZERO WARN."""
    runner = CliRunner()
    env = _fresh_setup_env(tmp_path)
    runner.invoke(cli, ["config", "set", "engine", "seed"], env=env)

    setup = runner.invoke(cli, ["setup", "codex"], env=env)
    assert setup.exit_code == 0, setup.output

    doctor = runner.invoke(cli, ["doctor"], env=env)
    assert doctor.exit_code == 0, doctor.output
    assert "WARN" not in doctor.output, doctor.output
    assert "Codex hooks.json: OK" in doctor.output
    assert "daemon installed" not in doctor.output


def test_doctor_clean_after_fresh_claude_code_setup(tmp_path):
    """Done-when: `poppy setup claude-code` then `poppy doctor` prints
    ZERO WARN (consent stays pending non-interactively → INFO, not WARN)."""
    runner = CliRunner()
    env = _fresh_setup_env(tmp_path)
    runner.invoke(cli, ["config", "set", "engine", "seed"], env=env)

    setup = runner.invoke(cli, ["setup", "claude-code"], env=env)
    assert setup.exit_code == 0, setup.output

    doctor = runner.invoke(cli, ["doctor"], env=env)
    assert doctor.exit_code == 0, doctor.output
    assert "WARN" not in doctor.output, doctor.output
    assert "PostCompact hook: OK" in doctor.output
    assert "daemon installed" not in doctor.output


def test_doctor_daemon_section_silent_without_daemon(tmp_path):
    """Daemon checks stay silent unless a service agent is installed or a daemon
    is running — a default (embedded) install shows no daemon lines."""
    runner = CliRunner()
    env = _fresh_setup_env(tmp_path)
    runner.invoke(cli, ["config", "set", "engine", "seed"], env=env)

    doctor = runner.invoke(cli, ["doctor"], env=env)
    assert doctor.exit_code == 0, doctor.output
    assert "daemon installed" not in doctor.output
    assert "daemon running" not in doctor.output
    assert "daemon reachable" not in doctor.output


def test_doctor_warns_when_client_points_at_dead_daemon(tmp_path):
    """A client wired to the shared HTTP daemon (`setup --daemon`) whose daemon
    is gone (uninstalled, or dotfiles restored onto a machine with no service
    agent) must NOT read healthy: the daemon checks run even with no agent/lock/
    liveness, and the client MCP line WARNs that the server is unreachable and
    names the fix — instead of a false `Cursor MCP: OK`."""
    cursor_home = tmp_path / "cursor"
    cursor_home.mkdir(parents=True)
    # Poppy registered against the daemon URL, but nothing is listening on 59321.
    (cursor_home / "mcp.json").write_text(
        json.dumps(
            {"mcpServers": {"poppy": {"url": "http://127.0.0.1:59321/mcp", "headers": {"Authorization": "Bearer x"}}}}
        )
    )
    runner = CliRunner()
    env = {
        "HOME": str(tmp_path),
        "CURSOR_HOME": str(cursor_home),
        "CODEX_HOME": str(tmp_path / "no-codex"),
        "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
        "POPPY_DIR": str(tmp_path / ".poppy"),
        "POPPY_DAEMON_PORT": "59321",  # nothing listening → not reachable
        "POPPY_LAUNCH_AGENTS_DIR": str(tmp_path / "no-agents"),  # no service agent
        "POPPY_SYSTEMD_USER_DIR": str(tmp_path / "no-systemd"),
        "POPPY_TELEMETRY_OFF": "1",
    }
    runner.invoke(cli, ["config", "set", "engine", "seed"], env=env)

    result = runner.invoke(cli, ["doctor"], env=env)
    # Daemon health checks ran despite no agent/lock/liveness...
    assert "daemon reachable: WARN" in result.output, result.output
    # ...and the client MCP line is not a false OK.
    assert "Cursor MCP: WARN" in result.output
    assert "not reachable" in result.output
    # The fix is named.
    assert "without --daemon" in result.output


def test_doctor_warns_when_poppy_client_mcp_entry_deleted(tmp_path):
    """A registered Poppy client (Cursor hooks present) whose MCP entry was
    deleted must WARN — the client can no longer reach Poppy — instead of being
    silently skipped."""
    from poppy.setup.claude_code import install_cursor_hooks

    cursor_home = tmp_path / "cursor"
    install_cursor_hooks(cursor_home)  # Poppy footprint via hooks; no mcp.json.
    runner = CliRunner()
    env = {
        "HOME": str(tmp_path),
        "CURSOR_HOME": str(cursor_home),
        "CODEX_HOME": str(tmp_path / "no-codex"),
        "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
        "POPPY_DIR": str(tmp_path / ".poppy"),
        "POPPY_TELEMETRY_OFF": "1",
    }
    runner.invoke(cli, ["config", "set", "engine", "seed"], env=env)

    result = runner.invoke(cli, ["doctor"], env=env)
    assert result.exit_code == 0, result.output
    assert "Cursor MCP: WARN" in result.output
    assert "MCP entry is missing" in result.output
    assert "re-register the MCP server" in result.output


def test_doctor_client_daemon_reachability_is_per_port(tmp_path, monkeypatch):
    """Reachability is probed on the SPECIFIC port each client's URL points at,
    not one global probe: a Cursor URL on 7679 must WARN even when a daemon
    answers on a different port 7680."""
    import poppy.mcp_server.lifecycle as lifecycle

    cursor_home = tmp_path / "cursor"
    cursor_home.mkdir(parents=True)
    (cursor_home / "mcp.json").write_text(
        json.dumps(
            {"mcpServers": {"poppy": {"url": "http://127.0.0.1:7679/mcp", "headers": {"Authorization": "Bearer x"}}}}
        )
    )

    def fake_probe(poppy_dir, port=None, timeout=0.25, token=None, host="127.0.0.1", auth_header=None):
        resolved = lifecycle.daemon_port() if port is None else port
        return ({"version": "x"}, None) if resolved == 7680 else (None, "connection refused")

    monkeypatch.setattr(lifecycle, "probe_status", fake_probe)

    env = {
        "HOME": str(tmp_path),
        "CURSOR_HOME": str(cursor_home),
        "CODEX_HOME": str(tmp_path / "no-codex"),
        "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
        "POPPY_DIR": str(tmp_path / ".poppy"),
        "POPPY_DAEMON_PORT": "7680",  # the global daemon answers here, not 7679
        "POPPY_LAUNCH_AGENTS_DIR": str(tmp_path / "no-agents"),
        "POPPY_SYSTEMD_USER_DIR": str(tmp_path / "no-systemd"),
        "POPPY_TELEMETRY_OFF": "1",
    }
    CliRunner().invoke(cli, ["config", "set", "engine", "seed"], env=env)

    result = CliRunner().invoke(cli, ["doctor"], env=env)
    # The configured daemon (7680) answers, so its health line is OK...
    assert "daemon reachable: OK" in result.output, result.output
    # ...but Cursor points at 7679, which is dead → WARN, not a false OK.
    assert "Cursor MCP: WARN" in result.output
    assert "port is not reachable" in result.output


def test_doctor_survives_malformed_client_daemon_url(tmp_path):
    """A client config with a malformed daemon URL (out-of-range / non-numeric
    port) must NOT abort `poppy doctor` with a traceback — it is the one command
    that has to survive broken configs. It WARNs and keeps running."""
    cursor_home = tmp_path / "cursor"
    cursor_home.mkdir(parents=True)
    (cursor_home / "mcp.json").write_text(
        json.dumps({"mcpServers": {"poppy": {"url": "http://127.0.0.1:99999/mcp", "headers": {}}}})
    )
    runner = CliRunner()
    env = {
        "HOME": str(tmp_path),
        "CURSOR_HOME": str(cursor_home),
        "CODEX_HOME": str(tmp_path / "no-codex"),
        "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
        "POPPY_DIR": str(tmp_path / ".poppy"),
        "POPPY_DAEMON_PORT": "59321",
        "POPPY_LAUNCH_AGENTS_DIR": str(tmp_path / "no-agents"),
        "POPPY_SYSTEMD_USER_DIR": str(tmp_path / "no-systemd"),
        "POPPY_TELEMETRY_OFF": "1",
    }
    runner.invoke(cli, ["config", "set", "engine", "seed"], env=env)

    result = runner.invoke(cli, ["doctor"], env=env)
    # Did not crash: later diagnostics still printed, no traceback.
    assert result.exception is None, result.output
    assert "capture state" in result.output  # a line emitted near the very end
    assert "Cursor MCP: WARN" in result.output
    assert "malformed" in result.output


def test_doctor_warns_when_client_daemon_token_is_stale(tmp_path, monkeypatch):
    """Each client is probed with the bearer baked into ITS OWN config, not the
    store's current daemon.token: a client holding a stale token the daemon 401s
    must WARN even though a client with the current token reads OK."""
    import poppy.mcp_server.lifecycle as lifecycle

    cursor_home = tmp_path / "cursor"
    cursor_home.mkdir(parents=True)
    codex_home = tmp_path / "codex"
    codex_home.mkdir(parents=True)
    # Cursor holds a STALE token; Codex holds the CURRENT token.
    (cursor_home / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "poppy": {"url": "http://127.0.0.1:7679/mcp", "headers": {"Authorization": "Bearer STALE"}}
                }
            }
        )
    )
    (codex_home / "config.toml").write_text(
        "[mcp_servers.poppy]\n"
        'url = "http://127.0.0.1:7679/mcp"\n'
        "[mcp_servers.poppy.http_headers]\n"
        'Authorization = "Bearer CURRENT"\n'
    )

    def fake_probe(poppy_dir, port=None, timeout=0.25, token=None, host="127.0.0.1", auth_header=None):
        # The daemon matches the header by partitioning on the
        # FIRST space, so only the byte-exact "Bearer CURRENT" authenticates.
        scheme, sep, supplied = (auth_header or "").partition(" ")
        if sep and scheme.lower() == "bearer" and supplied == "CURRENT":
            return {"version": "x"}, None
        return None, "HTTP Error 401: Unauthorized"

    monkeypatch.setattr(lifecycle, "probe_status", fake_probe)

    env = {
        "HOME": str(tmp_path),
        "CURSOR_HOME": str(cursor_home),
        "CODEX_HOME": str(codex_home),
        "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
        "POPPY_DIR": str(tmp_path / ".poppy"),
        "POPPY_DAEMON_PORT": "7679",
        "POPPY_LAUNCH_AGENTS_DIR": str(tmp_path / "no-agents"),
        "POPPY_SYSTEMD_USER_DIR": str(tmp_path / "no-systemd"),
        "POPPY_TELEMETRY_OFF": "1",
    }
    CliRunner().invoke(cli, ["config", "set", "engine", "seed"], env=env)

    result = CliRunner().invoke(cli, ["doctor"], env=env)
    # Cursor's stale token → 401 → WARN naming the token; Codex's current token → OK.
    assert "Cursor MCP: WARN" in result.output
    assert "token is stale" in result.output
    assert "Codex MCP: OK" in result.output


def test_doctor_warns_when_msix_registration_broken_but_standard_ok(tmp_path):
    """On Windows, Claude Desktop reads the MSIX-virtualized config. A broken MSIX
    registration (poppy removed) must WARN independently even when the standard
    config is still healthy — else the app's real, broken config is hidden."""
    local_appdata = tmp_path / "LocalAppData"
    msix_dir = local_appdata / "Packages" / "Claude_random-id" / "LocalCache" / "Roaming" / "Claude"
    msix_dir.mkdir(parents=True)
    # MSIX config exists but poppy was removed; the standard config still has it.
    (msix_dir / "claude_desktop_config.json").write_text(json.dumps({"mcpServers": {"other": {}}}))
    standard_config = tmp_path / "standard_claude_desktop_config.json"
    standard_config.write_text(json.dumps({"mcpServers": {"poppy": {"command": "poppy"}}}))

    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "HOME": str(tmp_path),
            "LOCALAPPDATA": str(local_appdata),
            "POPPY_CLAUDE_DESKTOP_CONFIG": str(standard_config),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )
    assert result.exit_code == 0, result.output
    # Standard config is healthy...
    assert "Claude desktop MCP: OK" in result.output
    # ...but the MSIX virtualized config, which the app actually reads, is broken.
    assert "Claude desktop MSIX MCP: WARN" in result.output


def test_doctor_rejects_substring_loopback_daemon_url(tmp_path):
    """SECURITY: a daemon URL whose host merely CONTAINS '127.0.0.1'/'localhost'
    (e.g. `http://127.0.0.1.evil.com/mcp`) must NOT be trusted as local — doctor
    must not probe 127.0.0.1 and print OK while the client ships its bearer token
    off-box. It WARNs that the host is non-loopback."""
    cursor_home = tmp_path / "cursor"
    cursor_home.mkdir(parents=True)
    (cursor_home / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "poppy": {
                        "url": "http://127.0.0.1.evil.com:7679/mcp",
                        "headers": {"Authorization": "Bearer secret"},
                    }
                }
            }
        )
    )
    runner = CliRunner()
    env = {
        "HOME": str(tmp_path),
        "CURSOR_HOME": str(cursor_home),
        "CODEX_HOME": str(tmp_path / "no-codex"),
        "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
        "POPPY_DIR": str(tmp_path / ".poppy"),
        "POPPY_DAEMON_PORT": "59321",
        "POPPY_LAUNCH_AGENTS_DIR": str(tmp_path / "no-agents"),
        "POPPY_SYSTEMD_USER_DIR": str(tmp_path / "no-systemd"),
        "POPPY_TELEMETRY_OFF": "1",
    }
    runner.invoke(cli, ["config", "set", "engine", "seed"], env=env)

    result = runner.invoke(cli, ["doctor"], env=env)
    assert result.exception is None, result.output
    assert "Cursor MCP: WARN" in result.output
    assert "non-loopback" in result.output.lower()
    # The attacker host is named, and it is never reported OK.
    assert "127.0.0.1.evil.com" in result.output
    assert "Cursor MCP: OK" not in result.output


def test_doctor_warns_when_msix_daemon_registration_is_dead(tmp_path, monkeypatch):
    """MSIX daemon registrations get the SAME reachability/token check as any
    other client: an MSIX-only config pointing at a dead port must WARN, not
    print OK just because the entry exists."""
    import poppy.mcp_server.lifecycle as lifecycle

    local_appdata = tmp_path / "LocalAppData"
    msix_dir = local_appdata / "Packages" / "Claude_random-id" / "LocalCache" / "Roaming" / "Claude"
    msix_dir.mkdir(parents=True)
    (msix_dir / "claude_desktop_config.json").write_text(
        json.dumps(
            {"mcpServers": {"poppy": {"url": "http://127.0.0.1:7679/mcp", "headers": {"Authorization": "Bearer x"}}}}
        )
    )

    # The daemon never answers.
    monkeypatch.setattr(lifecycle, "probe_status", lambda *a, **k: (None, "connection refused"))

    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "HOME": str(tmp_path),
            "LOCALAPPDATA": str(local_appdata),
            # No standard desktop config → daemon_clients would be empty pre-fix.
            "POPPY_CLAUDE_DESKTOP_CONFIG": str(tmp_path / "absent.json"),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_DAEMON_PORT": "7679",
            "POPPY_LAUNCH_AGENTS_DIR": str(tmp_path / "no-agents"),
            "POPPY_SYSTEMD_USER_DIR": str(tmp_path / "no-systemd"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )
    assert "Claude desktop MSIX MCP: WARN" in result.output
    assert "not reachable" in result.output
    assert "Claude desktop MSIX MCP: OK" not in result.output


def test_doctor_survives_unreadable_unrelated_client_config(tmp_path):
    """An unreadable config (mode 000) for a client the user never chose must not
    abort `poppy doctor` — one bad unrelated config is data, not a crash. The
    Cursor-only user still gets their full report."""
    from poppy.setup.claude_code import install_cursor_hooks

    cursor_home = tmp_path / "cursor"
    install_cursor_hooks(cursor_home)  # a real Poppy footprint

    # ~/.gemini/settings.json is one of the nine configs the daemon scan reads;
    # Path.home() honors the HOME env below, so this resolves into tmp_path.
    gemini_dir = tmp_path / ".gemini"
    gemini_dir.mkdir(parents=True)
    unreadable = gemini_dir / "settings.json"
    unreadable.write_text(json.dumps({"mcpServers": {"poppy": {"command": "poppy"}}}))
    unreadable.chmod(0o000)

    try:
        result = CliRunner().invoke(
            cli,
            ["doctor"],
            env={
                "HOME": str(tmp_path),
                "CURSOR_HOME": str(cursor_home),
                "CODEX_HOME": str(tmp_path / "no-codex"),
                "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
                "POPPY_DIR": str(tmp_path / ".poppy"),
                "POPPY_TELEMETRY_OFF": "1",
            },
        )
        assert result.exception is None, result.output
        # Full report still produced (a near-end line printed), no traceback.
        assert "capture state" in result.output
        assert "Cursor MCP" in result.output
    finally:
        unreadable.chmod(0o600)  # let pytest clean tmp_path up


def test_doctor_cannot_warn_fully_erased_mcp_only_client(tmp_path):
    """Narrowed guarantee: an MCP-only client (windsurf) whose
    entry is fully removed leaves no footprint, so it is indistinguishable from
    'never set up' and correctly produces NO Windsurf line — while a client with
    residual hooks/primer DOES still warn about a missing MCP entry (covered by
    test_doctor_warns_when_poppy_client_mcp_entry_deleted)."""
    windsurf_config = tmp_path / ".codeium" / "windsurf" / "mcp_config.json"
    windsurf_config.parent.mkdir(parents=True)
    windsurf_config.write_text(json.dumps({"mcpServers": {"somethingElse": {}}}))  # no poppy
    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "HOME": str(tmp_path),
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )
    assert result.exit_code == 0, result.output
    assert "Windsurf MCP" not in result.output


def test_setup_claude_code_does_not_truncate_unreadable_settings(tmp_path):
    """CRITICAL (data loss): `poppy setup claude-code` over an
    UNREADABLE ~/.claude/settings.json must NOT overwrite it with a Poppy-only
    stub. The shared `_read_json` raises on OSError, so the hook-install write
    path aborts with the user's config bytes intact rather than truncating it."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    settings = claude_dir / "settings.json"
    original = json.dumps(
        {
            "permissions": {"allow": ["Bash(ls)"]},
            "hooks": {"UserPromptSubmit": [{"hooks": [{"command": "my own hook"}]}]},
        }
    )
    settings.write_text(original)
    settings.chmod(0o000)
    try:
        result = CliRunner().invoke(
            cli,
            ["setup", "claude-code", "--no-claude-md"],
            env={
                "HOME": str(tmp_path),
                "CLAUDE_CONFIG_DIR": str(claude_dir),
                "POPPY_DIR": str(tmp_path / ".poppy"),
                "POPPY_TELEMETRY_OFF": "1",
            },
        )
        assert result.exit_code != 0  # fails loudly, does not silently truncate
        settings.chmod(0o600)
        assert settings.read_text() == original  # bytes unchanged
    finally:
        settings.chmod(0o600)


def test_doctor_survives_non_object_json_client_config(tmp_path):
    """A client config that is valid JSON but not an object (`[]`, `null`, a
    string) must not crash doctor via `settings.get(...)`."""
    cursor_home = tmp_path / "cursor"
    cursor_home.mkdir(parents=True)
    (cursor_home / "mcp.json").write_text("[]")  # a JSON array, not an object
    gemini_dir = tmp_path / ".gemini"
    gemini_dir.mkdir()
    (gemini_dir / "settings.json").write_text('"just a string"')

    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "HOME": str(tmp_path),
            "CURSOR_HOME": str(cursor_home),
            "CODEX_HOME": str(tmp_path / "no-codex"),
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )
    assert result.exception is None, result.output
    assert "capture state" in result.output  # ran to the end, no crash


def test_doctor_warns_when_daemon_url_scheme_or_path_wrong(tmp_path, monkeypatch):
    """Host+port match is not enough: an `https://` scheme or a non-`/mcp` path
    means the client cannot reach Poppy's daemon even though a probe of the same
    host:port answers. Must WARN, not print OK."""
    import poppy.mcp_server.lifecycle as lifecycle

    cursor_home = tmp_path / "cursor"
    cursor_home.mkdir(parents=True)
    (cursor_home / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "poppy": {"url": "https://localhost:7679/not-mcp", "headers": {"Authorization": "Bearer x"}}
                }
            }
        )
    )
    # A daemon WOULD answer — the URL is still unusable by the client.
    monkeypatch.setattr(lifecycle, "probe_status", lambda *a, **k: ({"version": "x"}, None))

    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "HOME": str(tmp_path),
            "CURSOR_HOME": str(cursor_home),
            "CODEX_HOME": str(tmp_path / "no-codex"),
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_DAEMON_PORT": "7679",
            "POPPY_LAUNCH_AGENTS_DIR": str(tmp_path / "no-agents"),
            "POPPY_SYSTEMD_USER_DIR": str(tmp_path / "no-systemd"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )
    assert "Cursor MCP: WARN" in result.output
    assert "does not serve" in result.output
    assert "Cursor MCP: OK" not in result.output


def test_doctor_survives_non_utf8_client_config(tmp_path):
    """A non-UTF-8 client config raises UnicodeDecodeError (a ValueError, NOT
    OSError) in read_text; doctor must catch the whole read/parse family and not
    abort for a client the user never chose."""
    gemini_dir = tmp_path / ".gemini"
    gemini_dir.mkdir()
    (gemini_dir / "settings.json").write_bytes(b"\xff\xfe{\x00}\x00")  # UTF-16 bytes

    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "HOME": str(tmp_path),
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )
    assert result.exception is None, result.output
    assert "capture state" in result.output  # ran to the end, no crash


def test_doctor_warns_unreadable_mcp_config_for_footprinted_client(tmp_path):
    """A footprinted Cursor client (hooks present) whose mcp.json is UNREADABLE
    must still show its diagnostics AND WARN that Poppy's registration can't be
    verified — not be silently skipped nor mislabeled 'MCP entry missing'."""
    from poppy.setup.claude_code import install_cursor_hooks

    cursor_home = tmp_path / "cursor"
    install_cursor_hooks(cursor_home)  # Poppy hooks footprint
    mcp = cursor_home / "mcp.json"
    mcp.write_text(json.dumps({"mcpServers": {"poppy": {"command": "poppy"}}}))
    mcp.chmod(0o000)
    try:
        result = CliRunner().invoke(
            cli,
            ["doctor"],
            env={
                "HOME": str(tmp_path),
                "CURSOR_HOME": str(cursor_home),
                "CODEX_HOME": str(tmp_path / "no-codex"),
                "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
                "POPPY_DIR": str(tmp_path / ".poppy"),
                "POPPY_TELEMETRY_OFF": "1",
            },
        )
        assert result.exception is None, result.output
        # Cursor diagnostics NOT suppressed by the unreadable MCP file.
        assert "Cursor hooks.json" in result.output
        # Unreadable, not missing.
        assert "Cursor MCP: WARN" in result.output
        assert "unreadable" in result.output
        assert "MCP entry is missing" not in result.output
    finally:
        mcp.chmod(0o600)


def test_doctor_warns_when_msix_config_deleted_but_standard_ok(tmp_path):
    """The MSIX Claude app is installed (its dir exists) but its config file was
    deleted, while the standard config still registers Poppy: doctor must WARN
    that the MSIX registration is missing, independent of the healthy standard
    config."""
    local_appdata = tmp_path / "LocalAppData"
    msix_dir = local_appdata / "Packages" / "Claude_random-id" / "LocalCache" / "Roaming" / "Claude"
    msix_dir.mkdir(parents=True)  # app dir present, but NO config file inside
    standard_config = tmp_path / "standard_claude_desktop_config.json"
    standard_config.write_text(json.dumps({"mcpServers": {"poppy": {"command": "poppy"}}}))

    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "HOME": str(tmp_path),
            "LOCALAPPDATA": str(local_appdata),
            "POPPY_CLAUDE_DESKTOP_CONFIG": str(standard_config),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )
    assert result.exit_code == 0, result.output
    assert "Claude desktop MCP: OK" in result.output
    assert "Claude desktop MSIX MCP: WARN" in result.output
    assert "missing" in result.output


def test_doctor_warns_when_daemon_url_path_not_exactly_mcp(tmp_path, monkeypatch):
    """Path must be EXACTLY the daemon's /mcp endpoint: `/wrong/mcp` (last
    segment 'mcp' but not the /mcp path) must WARN even though a probe of the
    same host:port answers."""
    import poppy.mcp_server.lifecycle as lifecycle

    cursor_home = tmp_path / "cursor"
    cursor_home.mkdir(parents=True)
    (cursor_home / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "poppy": {"url": "http://127.0.0.1:7679/wrong/mcp", "headers": {"Authorization": "Bearer x"}}
                }
            }
        )
    )
    monkeypatch.setattr(lifecycle, "probe_status", lambda *a, **k: ({"version": "x"}, None))

    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "HOME": str(tmp_path),
            "CURSOR_HOME": str(cursor_home),
            "CODEX_HOME": str(tmp_path / "no-codex"),
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_DAEMON_PORT": "7679",
            "POPPY_LAUNCH_AGENTS_DIR": str(tmp_path / "no-agents"),
            "POPPY_SYSTEMD_USER_DIR": str(tmp_path / "no-systemd"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )
    assert "Cursor MCP: WARN" in result.output
    assert "does not serve" in result.output
    assert "Cursor MCP: OK" not in result.output


def test_doctor_survives_malformed_bearer_token(tmp_path):
    """A bearer token that cannot form a valid HTTP header (embedded newline)
    raises when the probe request is built; doctor must catch it and WARN, not
    abort."""
    cursor_home = tmp_path / "cursor"
    cursor_home.mkdir(parents=True)
    (cursor_home / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "poppy": {
                        "url": "http://127.0.0.1:59321/mcp",
                        "headers": {"Authorization": "Bearer first\nsecond"},
                    }
                }
            }
        )
    )
    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "HOME": str(tmp_path),
            "CURSOR_HOME": str(cursor_home),
            "CODEX_HOME": str(tmp_path / "no-codex"),
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_DAEMON_PORT": "59321",
            "POPPY_LAUNCH_AGENTS_DIR": str(tmp_path / "no-agents"),
            "POPPY_SYSTEMD_USER_DIR": str(tmp_path / "no-systemd"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )
    assert result.exception is None, result.output
    assert "capture state" in result.output  # ran to the end, no crash
    assert "Cursor MCP: WARN" in result.output
    assert "not a valid HTTP header" in result.output


def test_doctor_shows_cursor_diagnostics_on_partial_hook_install(tmp_path):
    """A PARTIAL Cursor hook install (one Poppy hook removed) + a missing MCP
    entry must still show Cursor's diagnostics and WARN about exactly what's
    missing — 'any hook', not 'all hooks', keeps the footprint."""
    from poppy.setup.claude_code import install_cursor_hooks

    cursor_home = tmp_path / "cursor"
    hooks_path = install_cursor_hooks(cursor_home)
    settings = json.loads(hooks_path.read_text())
    settings["hooks"].pop("sessionEnd", None)  # drop one Poppy hook → partial
    hooks_path.write_text(json.dumps(settings))

    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "HOME": str(tmp_path),
            "CURSOR_HOME": str(cursor_home),
            "CODEX_HOME": str(tmp_path / "no-codex"),
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )
    assert result.exit_code == 0, result.output
    # Not suppressed: the partial hooks and the missing MCP entry both surface.
    assert "Cursor hooks.json: WARN" in result.output
    assert "Cursor MCP: WARN" in result.output
    assert "MCP entry is missing" in result.output


def test_doctor_probes_exact_ipv6_loopback_host(tmp_path, monkeypatch):
    """A `::1` registration is probed at `::1`, not rewritten to 127.0.0.1: if the
    daemon answers only on IPv4, the IPv6 client registration WARNs (its own
    connection would fail) instead of a false OK."""
    import poppy.mcp_server.lifecycle as lifecycle

    cursor_home = tmp_path / "cursor"
    cursor_home.mkdir(parents=True)
    (cursor_home / "mcp.json").write_text(
        json.dumps(
            {"mcpServers": {"poppy": {"url": "http://[::1]:7679/mcp", "headers": {"Authorization": "Bearer x"}}}}
        )
    )

    def fake_probe(poppy_dir, port=None, timeout=0.25, token=None, host="127.0.0.1", auth_header=None):
        return ({"version": "x"}, None) if host == "127.0.0.1" else (None, "connection refused")

    monkeypatch.setattr(lifecycle, "probe_status", fake_probe)

    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "HOME": str(tmp_path),
            "CURSOR_HOME": str(cursor_home),
            "CODEX_HOME": str(tmp_path / "no-codex"),
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_DAEMON_PORT": "7679",
            "POPPY_LAUNCH_AGENTS_DIR": str(tmp_path / "no-agents"),
            "POPPY_SYSTEMD_USER_DIR": str(tmp_path / "no-systemd"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )
    assert "Cursor MCP: WARN" in result.output
    assert "not reachable" in result.output
    assert "Cursor MCP: OK" not in result.output


def test_probe_status_does_not_follow_redirects_and_leak_token(tmp_path):
    """SECURITY: the authenticated probe must NOT follow HTTP
    redirects — a `/status` that 302s to another host would otherwise forward the
    client's bearer token off-box. The redirect target is never contacted and the
    probe reports failure."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from poppy.mcp_server.lifecycle import probe_status

    hits: list[str] = []
    leaked_auth: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            hits.append(self.path)
            if self.path == "/status":
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{self.server.server_address[1]}/leak")
                self.end_headers()
            else:  # /leak — the "attacker" endpoint the redirect points at
                leaked_auth.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")

        def log_message(self, *_args):  # silence the test server
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload, error = probe_status(tmp_path, port, timeout=1.0, token="secret", host="127.0.0.1")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert payload is None  # a 302 is not "healthy"
    assert error is not None
    assert "/leak" not in hits  # redirect NOT followed
    assert leaked_auth == []  # the bearer token never reached the redirect target


def test_doctor_survives_null_nested_hook_value(tmp_path):
    """A hooks.json whose event value is null (`{"hooks":{"Stop":null}}`) must not
    crash the footprint scan (iterating None → TypeError, which safe_read doesn't
    catch) for a client the user never configured."""
    codex_home = tmp_path / "codex"
    codex_home.mkdir(parents=True)
    (codex_home / "hooks.json").write_text(json.dumps({"hooks": {"Stop": None}}))

    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "HOME": str(tmp_path),
            "CODEX_HOME": str(codex_home),
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )
    assert result.exception is None, result.output
    assert "capture state" in result.output  # ran to the end, no crash
    # No Poppy registration in Codex → the section stays silent (never chose it).
    assert "Codex hooks.json" not in result.output


def test_probe_status_ignores_http_proxy_env(tmp_path, monkeypatch):
    """SECURITY: the authenticated probe must never route through
    $http_proxy — a proxy is not matched by `no_proxy=127.0.0.1,localhost` for an
    IPv6-loopback URL, so the bearer token would go to the proxy. The probe's
    opener uses an empty ProxyHandler, so the token reaches only the exact
    loopback endpoint and the proxy is never contacted."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from poppy.mcp_server.lifecycle import probe_status

    proxy_hits: list = []
    real_auth: list = []

    class Real(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            real_auth.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *_a):
            pass

    class Proxy(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            proxy_hits.append((self.path, self.headers.get("Authorization")))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *_a):
            pass

    real = HTTPServer(("127.0.0.1", 0), Real)
    proxy = HTTPServer(("127.0.0.1", 0), Proxy)
    threads = [threading.Thread(target=srv.serve_forever, daemon=True) for srv in (real, proxy)]
    for thread in threads:
        thread.start()
    monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{proxy.server_address[1]}")
    monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy.server_address[1]}")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv("NO_PROXY", "")
    try:
        payload, _error = probe_status(tmp_path, real.server_address[1], timeout=1.0, token="secret", host="127.0.0.1")
    finally:
        for srv in (real, proxy):
            srv.shutdown()
            srv.server_close()
        for thread in threads:
            thread.join(timeout=2)

    assert payload == {}  # reached the real endpoint directly
    assert proxy_hits == []  # the bearer token NEVER went to the proxy
    assert real_auth == ["Bearer secret"]  # token only to the validated endpoint


def test_runtime_loopback_clients_bypass_http_proxy(monkeypatch):
    """The RUNTIME clients that reach the loopback daemon — the `poppy serve`
    forwarder and the daemon-models inference client — must NOT route through
    $http_proxy, matching doctor's no-proxy probe. Otherwise a proxy that 403s
    loopback would break the real path while doctor reported OK."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import anyio

    from poppy.engine._daemon_models import _new_http_client
    from poppy.mcp_server.forwarder import _loopback_http_client

    real_hits: list = []
    proxy_hits: list = []

    def _handler(hits):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                hits.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *_a):
                pass

        return Handler

    real = HTTPServer(("127.0.0.1", 0), _handler(real_hits))
    proxy = HTTPServer(("127.0.0.1", 0), _handler(proxy_hits))
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (real, proxy)]
    for thread in threads:
        thread.start()
    monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{proxy.server_address[1]}")
    monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy.server_address[1]}")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv("NO_PROXY", "")
    url = f"http://127.0.0.1:{real.server_address[1]}/x"
    try:
        with _new_http_client(None) as client:  # sync daemon-models client
            assert client.trust_env is False
            assert client.get(url).status_code == 200

        async def _go():
            async with _loopback_http_client() as ac:  # async forwarder client
                assert ac.trust_env is False
                response = await ac.get(url)
                assert response.status_code == 200

        anyio.run(_go)
    finally:
        for srv in (real, proxy):
            srv.shutdown()
            srv.server_close()
        for thread in threads:
            thread.join(timeout=2)

    assert real_hits, "both clients reached the loopback server directly"
    assert proxy_hits == [], "neither runtime client routed through the proxy"


def test_doctor_shows_hermes_diagnostics_on_partial_install(tmp_path):
    """A PARTIAL Hermes install (plugin files present, but no config.yaml provider
    and no SOUL.md) must still show Hermes diagnostics and WARN about exactly
    what's missing — the footprint gate is 'ANY component', not 'all'."""
    hermes_home = tmp_path / ".hermes"
    plugin_init = hermes_home / "plugins" / "poppy" / "__init__.py"
    plugin_init.parent.mkdir(parents=True)
    plugin_init.write_text("# poppy hermes plugin\n")  # plugin present, nothing else

    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "HOME": str(tmp_path),
            "HERMES_HOME": str(hermes_home),
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )
    assert result.exit_code == 0, result.output
    # Not suppressed: the un-activated provider and missing SOUL.md both surface.
    assert "Hermes Agent plugin: WARN" in result.output
    assert "Hermes SOUL.md guidance: WARN" in result.output


def test_doctor_shows_hermes_diagnostics_on_config_provider_only(tmp_path):
    """The OTHER partial Hermes shape: config.yaml activates `provider: poppy` but
    the plugin files and SOUL.md are absent. The config-provider setting is a
    footprint component, so Hermes is still footprinted and WARNs the plugin is
    missing — this shape previously disappeared entirely."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True)
    # provider configured, but NO plugins/poppy/__init__.py and NO SOUL.md.
    (hermes_home / "config.yaml").write_text("memory:\n  provider: poppy\n")

    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "HOME": str(tmp_path),
            "HERMES_HOME": str(hermes_home),
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )
    assert result.exit_code == 0, result.output
    assert "Hermes Agent plugin: WARN" in result.output  # plugin files missing
    assert "Hermes SOUL.md guidance: WARN" in result.output  # SOUL.md missing


def test_doctor_shows_goose_diagnostics_on_partial_install(tmp_path):
    """The same 'any component' rule for the other multi-file client: a partial
    Goose install (managed .goosehints present, but the MCP extension not in
    config.yaml) still shows Goose diagnostics."""
    goose_dir = tmp_path / "goose"
    goose_dir.mkdir(parents=True)
    (goose_dir / ".goosehints").write_text("<!-- POPPY:BEGIN -->\nprimer\n<!-- POPPY:END -->\n")

    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "HOME": str(tmp_path),
            "POPPY_GOOSE_CONFIG_DIR": str(goose_dir),
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )
    assert result.exit_code == 0, result.output
    assert "Goose MCP: WARN" in result.output  # extension not registered
    assert "Goose primer: OK" in result.output  # the piece that IS installed


def test_doctor_survives_non_dict_mcpservers_in_desktop_config(tmp_path):
    """A Claude Desktop config with a non-object `mcpServers` (`null`) must not
    crash doctor via `"poppy" in None`."""
    standard = tmp_path / "standard_desktop.json"
    standard.write_text(json.dumps({"mcpServers": None}))

    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "HOME": str(tmp_path),
            "POPPY_CLAUDE_DESKTOP_CONFIG": str(standard),
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )
    assert result.exception is None, result.output
    assert "capture state" in result.output  # ran to the end, no crash


def test_doctor_warns_unreadable_codex_config_as_present_not_missing(tmp_path):
    """Codex's TOML reader swallows OSError, so an unreadable config.toml for a
    footprinted Codex client (hooks present) must be reported as 'present but
    unreadable', not 'MCP entry is missing' — the fix is permissions, not `poppy
    setup`."""
    from poppy.setup.claude_code import install_codex_hooks

    codex_home = tmp_path / "codex"
    install_codex_hooks(codex_home)  # Poppy footprint via hooks
    config = codex_home / "config.toml"
    config.write_text('[mcp_servers.poppy]\nurl = "http://127.0.0.1:7679/mcp"\n')
    config.chmod(0o000)
    try:
        result = CliRunner().invoke(
            cli,
            ["doctor"],
            env={
                "HOME": str(tmp_path),
                "CODEX_HOME": str(codex_home),
                "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
                "POPPY_DIR": str(tmp_path / ".poppy"),
                "POPPY_TELEMETRY_OFF": "1",
            },
        )
        assert result.exception is None, result.output
        assert "Codex MCP: WARN" in result.output
        assert "unreadable" in result.output
        assert "MCP entry is missing" not in result.output
    finally:
        config.chmod(0o600)


def test_doctor_prints_desktop_line_when_only_msix_registered(tmp_path):
    """When only the MSIX config registers Poppy and the standard desktop config
    file doesn't exist, the standard 'Claude desktop MCP' line still prints
    (previously swallowed by the exists() guard)."""
    local_appdata = tmp_path / "LocalAppData"
    msix_dir = local_appdata / "Packages" / "Claude_random-id" / "LocalCache" / "Roaming" / "Claude"
    msix_dir.mkdir(parents=True)
    (msix_dir / "claude_desktop_config.json").write_text(json.dumps({"mcpServers": {"poppy": {"command": "poppy"}}}))

    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "HOME": str(tmp_path),
            "LOCALAPPDATA": str(local_appdata),
            "POPPY_CLAUDE_DESKTOP_CONFIG": str(tmp_path / "absent_standard.json"),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )
    assert result.exit_code == 0, result.output
    assert "Claude desktop MCP" in result.output  # the standard line now prints
    assert "Claude desktop MSIX MCP: OK" in result.output


def test_doctor_daemon_reachable_ok_when_client_on_custom_port_healthy(tmp_path, monkeypatch):
    """The daemon summary line probes the configured/default port, but a client
    set up on a custom --daemon port answers elsewhere. When every client reached
    its daemon, the summary must not print a spurious 'daemon reachable: WARN'
    that the per-client checks already contradict."""
    import poppy.mcp_server.lifecycle as lifecycle

    cursor_home = tmp_path / "cursor"
    cursor_home.mkdir(parents=True)
    (cursor_home / "mcp.json").write_text(
        json.dumps(
            {"mcpServers": {"poppy": {"url": "http://127.0.0.1:8888/mcp", "headers": {"Authorization": "Bearer x"}}}}
        )
    )

    def fake_probe(poppy_dir, port=None, timeout=0.25, token=None, host="127.0.0.1", auth_header=None):
        # The daemon answers only on the client's custom port 8888.
        return ({"version": "x"}, None) if port == 8888 else (None, "connection refused")

    monkeypatch.setattr(lifecycle, "probe_status", fake_probe)

    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "HOME": str(tmp_path),
            "CURSOR_HOME": str(cursor_home),
            "CODEX_HOME": str(tmp_path / "no-codex"),
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_DAEMON_PORT": "7679",  # the summary probe hits 7679 (dead)
            "POPPY_LAUNCH_AGENTS_DIR": str(tmp_path / "no-agents"),
            "POPPY_SYSTEMD_USER_DIR": str(tmp_path / "no-systemd"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )
    assert "Cursor MCP: OK" in result.output  # client healthy on its own port
    assert "daemon reachable: OK" in result.output
    assert "daemon reachable: WARN" not in result.output


def test_probe_status_sends_authorization_header_verbatim(tmp_path):
    """probe_status transmits the given Authorization header byte-for-byte — no
    strip/normalize — so doctor reproduces exactly what a client sends (a double
    space is preserved)."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from poppy.mcp_server.lifecycle import probe_status

    received: list = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            received.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *_a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        probe_status(tmp_path, server.server_address[1], timeout=1.0, auth_header="Bearer  secret")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert received == ["Bearer  secret"]  # double space preserved, never normalized


def test_doctor_warns_when_client_authorization_header_is_malformed(tmp_path, monkeypatch):
    """The BLOCKER case: a client whose stored header is `"Bearer  secret"` (double
    space) is 401'd by the daemon (which partitions on the first space),
    but doctor previously normalized it to `secret` and probed a false OK. Doctor
    now probes the EXACT header, reproduces the 401, and WARNs — while a correctly
    formed `"Bearer secret"` client reads OK."""
    import poppy.mcp_server.lifecycle as lifecycle

    cursor_home = tmp_path / "cursor"
    cursor_home.mkdir(parents=True)
    (cursor_home / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "poppy": {"url": "http://127.0.0.1:7679/mcp", "headers": {"Authorization": "Bearer  secret"}}
                }
            }
        )
    )
    codex_home = tmp_path / "codex"
    codex_home.mkdir(parents=True)
    (codex_home / "config.toml").write_text(
        "[mcp_servers.poppy]\n"
        'url = "http://127.0.0.1:7679/mcp"\n'
        "[mcp_servers.poppy.http_headers]\n"
        'Authorization = "Bearer secret"\n'
    )

    def fake_probe(poppy_dir, port=None, timeout=0.25, token=None, host="127.0.0.1", auth_header=None):
        # Mirror the daemon's bearer_token_matches: partition on the FIRST
        # space, so only the byte-exact "Bearer secret" authenticates.
        scheme, sep, supplied = (auth_header or "").partition(" ")
        if sep and scheme.lower() == "bearer" and supplied == "secret":
            return {"version": "x"}, None
        return None, "HTTP Error 401: Unauthorized"

    monkeypatch.setattr(lifecycle, "probe_status", fake_probe)

    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "HOME": str(tmp_path),
            "CURSOR_HOME": str(cursor_home),
            "CODEX_HOME": str(codex_home),
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_DAEMON_PORT": "7679",
            "POPPY_LAUNCH_AGENTS_DIR": str(tmp_path / "no-agents"),
            "POPPY_SYSTEMD_USER_DIR": str(tmp_path / "no-systemd"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )
    assert "Cursor MCP: WARN" in result.output  # "Bearer  secret" → 401
    assert "token is stale" in result.output
    assert "Codex MCP: OK" in result.output  # "Bearer secret" → 200


def test_probe_status_client_probe_sends_no_credential_not_store_token(tmp_path, monkeypatch):
    """SECURITY: a client probe (auth_header=None + token=None)
    must send NO Authorization header — never the store daemon.token. Otherwise
    doctor would disclose the daemon's real token to a client's listener and mask
    a no-credential client's 401 as OK. The store-token fallback belongs only to
    the daemon's own self-check (default token)."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from poppy.mcp_server.lifecycle import probe_status

    monkeypatch.delenv("POPPY_DAEMON_TOKEN", raising=False)
    (tmp_path / "daemon.token").write_text("STORETOKEN\n")  # a real store token on disk
    seen: list = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            seen.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *_a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        probe_status(tmp_path, port, timeout=1.0, token=None, auth_header=None)  # client probe: no credential
        probe_status(tmp_path, port, timeout=1.0)  # daemon self-check: uses the store token
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert seen[0] is None  # client probe sent NO Authorization
    assert seen[1] == "Bearer STORETOKEN"  # only the self-check sends the store token


def test_doctor_warns_client_with_no_auth_header(tmp_path, monkeypatch):
    """A daemon-mode client registration with NO Authorization header must WARN
    (its probe sends no credential → the daemon 401s), not be probed with the
    store token and reported OK. A correctly-configured client
    still reads OK."""
    import poppy.mcp_server.lifecycle as lifecycle

    cursor_home = tmp_path / "cursor"
    cursor_home.mkdir(parents=True)
    # Daemon URL but NO headers at all.
    (cursor_home / "mcp.json").write_text(json.dumps({"mcpServers": {"poppy": {"url": "http://127.0.0.1:7679/mcp"}}}))
    codex_home = tmp_path / "codex"
    codex_home.mkdir(parents=True)
    (codex_home / "config.toml").write_text(
        "[mcp_servers.poppy]\n"
        'url = "http://127.0.0.1:7679/mcp"\n'
        "[mcp_servers.poppy.http_headers]\n"
        'Authorization = "Bearer good"\n'
    )

    sent_headers: list = []

    def fake_probe(poppy_dir, port=None, timeout=0.25, token=None, host="127.0.0.1", auth_header=None):
        sent_headers.append(auth_header)
        return ({"version": "x"}, None) if auth_header == "Bearer good" else (None, "HTTP Error 401: Unauthorized")

    monkeypatch.setattr(lifecycle, "probe_status", fake_probe)

    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "HOME": str(tmp_path),
            "CURSOR_HOME": str(cursor_home),
            "CODEX_HOME": str(codex_home),
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_DAEMON_PORT": "7679",
            "POPPY_LAUNCH_AGENTS_DIR": str(tmp_path / "no-agents"),
            "POPPY_SYSTEMD_USER_DIR": str(tmp_path / "no-systemd"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )
    assert "Cursor MCP: WARN" in result.output
    assert "no Authorization credential" in result.output
    assert "Codex MCP: OK" in result.output
    # The no-header client's probe carried NO credential (not the store token).
    assert None in sent_headers


def _serve_raw_once(raw_response: bytes):
    """Start a one-shot raw-socket HTTP-ish server that replies with
    ``raw_response`` then closes; returns (port, server_socket, thread)."""
    import socket
    import threading

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    srv.settimeout(5)  # so accept() never blocks the thread forever at teardown
    port = srv.getsockname()[1]

    def serve():
        # Accept exactly one connection (each test makes a single probe), then the
        # thread exits — no blocking accept lingers to trip a teardown abort.
        try:
            conn, _ = srv.accept()
        except OSError:
            return
        try:
            conn.recv(65536)
            if raw_response:
                conn.sendall(raw_response)
        except OSError:
            pass
        finally:
            conn.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return port, srv, thread


def test_probe_status_is_total_on_odd_protocol_failures(tmp_path):
    """probe_status is TOTAL: any odd HTTP failure — a truncated body
    (IncompleteRead), a non-HTTP line (BadStatusLine), an immediate close
    (RemoteDisconnected) — returns (None, error) and never raises."""
    from poppy.mcp_server.lifecycle import probe_status

    bad_responses = [
        b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\nContent-Type: application/json\r\n\r\n{}",  # IncompleteRead
        b"garbage not an http response\r\n\r\n",  # BadStatusLine
        b"",  # immediate close → RemoteDisconnected
    ]
    for raw in bad_responses:
        port, srv, thread = _serve_raw_once(raw)
        try:
            payload, error = probe_status(tmp_path, port, timeout=1.0, token=None)
        finally:
            srv.close()
            thread.join(timeout=2)
        assert payload is None, raw
        assert error is not None, raw  # a reason, not a crash


def test_doctor_survives_incomplete_read_from_broken_endpoint(tmp_path):
    """A client whose /status advertises Content-Length: 100 but sends only `{}`
    then closes raises http.client.IncompleteRead (an HTTPException the older
    guards missed). The probe is now total: doctor WARNs the client and STILL
    completes the rest of the report (storage/engines/capture)."""
    # inspect_daemon probes a dead port; only the Cursor client hits the broken one.
    port, srv, thread = _serve_raw_once(
        b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\nContent-Type: application/json\r\n\r\n{}"
    )
    cursor_home = tmp_path / "cursor"
    cursor_home.mkdir(parents=True)
    (cursor_home / "mcp.json").write_text(
        json.dumps(
            {"mcpServers": {"poppy": {"url": f"http://127.0.0.1:{port}/mcp", "headers": {"Authorization": "Bearer x"}}}}
        )
    )
    try:
        result = CliRunner().invoke(
            cli,
            ["doctor"],
            env={
                "HOME": str(tmp_path),
                "CURSOR_HOME": str(cursor_home),
                "CODEX_HOME": str(tmp_path / "no-codex"),
                "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
                "POPPY_DIR": str(tmp_path / ".poppy"),
                "POPPY_DAEMON_PORT": "59321",  # dead — inspect_daemon won't hit the broken server
                "POPPY_LAUNCH_AGENTS_DIR": str(tmp_path / "no-agents"),
                "POPPY_SYSTEMD_USER_DIR": str(tmp_path / "no-systemd"),
                "POPPY_TELEMETRY_OFF": "1",
            },
        )
    finally:
        srv.close()
        thread.join(timeout=2)
    assert result.exception is None, result.output  # did NOT abort
    assert "Cursor MCP: WARN" in result.output  # broken endpoint → unhealthy
    assert "capture state" in result.output  # the rest of the report still printed


def _serve_slowloris(stop):
    """A server that returns 200 with NO Content-Length, then trickles one byte
    every 50 ms without closing — defeats a per-read socket timeout. Returns
    (port, server_socket, thread); the caller sets ``stop`` and closes to end it."""
    import socket
    import threading

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    srv.settimeout(5)
    port = srv.getsockname()[1]

    def serve():
        try:
            conn, _ = srv.accept()
        except OSError:
            return
        try:
            conn.recv(65536)
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n")
            while not stop.is_set():
                try:
                    conn.sendall(b" ")
                except OSError:
                    return
                stop.wait(0.05)
        finally:
            conn.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return port, srv, thread


def test_probe_status_bounds_slowloris_with_a_deadline(tmp_path, monkeypatch):
    """A slowloris /status (trickles bytes, never closes) defeats the per-read
    socket timeout, but the whole-probe wall-clock deadline bounds it: the probe
    returns a failure quickly instead of hanging forever."""
    import threading
    import time

    import poppy.mcp_server.lifecycle as lifecycle
    from poppy.mcp_server.lifecycle import probe_status

    monkeypatch.setattr(lifecycle, "_PROBE_DEADLINE_S", 0.3)  # keep the test fast
    stop = threading.Event()
    port, srv, thread = _serve_slowloris(stop)
    try:
        start = time.monotonic()
        payload, error = probe_status(tmp_path, port, timeout=1.0, token=None)
        elapsed = time.monotonic() - start
    finally:
        stop.set()
        srv.close()
        thread.join(timeout=3)
    assert payload is None
    assert error is not None
    assert elapsed < 2.0  # bounded by the 0.3s deadline, not the endless trickle


def test_probe_status_caps_an_oversized_response(tmp_path):
    """A flood /status (a body far larger than the tiny status JSON) is rejected
    by the size cap rather than read unbounded into memory."""
    from poppy.mcp_server.lifecycle import _PROBE_MAX_BYTES, probe_status

    huge = b"x" * (_PROBE_MAX_BYTES + 4096)
    raw = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n" + huge
    port, srv, thread = _serve_raw_once(raw)
    try:
        payload, error = probe_status(tmp_path, port, timeout=1.0, token=None)
    finally:
        srv.close()
        thread.join(timeout=2)
    assert payload is None
    assert error is not None


def test_doctor_survives_slowloris_status_endpoint(tmp_path, monkeypatch):
    """End-to-end: a client whose /status slowloris-trickles must not freeze
    doctor — it WARNs the client within the deadline and completes the rest of
    the report."""
    import threading

    import poppy.mcp_server.lifecycle as lifecycle

    monkeypatch.setattr(lifecycle, "_PROBE_DEADLINE_S", 0.3)
    stop = threading.Event()
    port, srv, thread = _serve_slowloris(stop)
    cursor_home = tmp_path / "cursor"
    cursor_home.mkdir(parents=True)
    (cursor_home / "mcp.json").write_text(
        json.dumps(
            {"mcpServers": {"poppy": {"url": f"http://127.0.0.1:{port}/mcp", "headers": {"Authorization": "Bearer x"}}}}
        )
    )
    try:
        result = CliRunner().invoke(
            cli,
            ["doctor"],
            env={
                "HOME": str(tmp_path),
                "CURSOR_HOME": str(cursor_home),
                "CODEX_HOME": str(tmp_path / "no-codex"),
                "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
                "POPPY_DIR": str(tmp_path / ".poppy"),
                "POPPY_DAEMON_PORT": "59321",  # dead — inspect_daemon won't hit the slowloris
                "POPPY_LAUNCH_AGENTS_DIR": str(tmp_path / "no-agents"),
                "POPPY_SYSTEMD_USER_DIR": str(tmp_path / "no-systemd"),
                "POPPY_TELEMETRY_OFF": "1",
            },
        )
    finally:
        stop.set()
        srv.close()
        thread.join(timeout=3)
    assert result.exception is None, result.output  # did NOT hang/abort
    assert "Cursor MCP: WARN" in result.output
    assert "capture state" in result.output  # the rest of the report still printed


def test_doctor_reads_authorization_header_case_insensitively(tmp_path, monkeypatch):
    """HTTP header names are case-insensitive (RFC 7230): an `AUTHORIZATION` or
    `authorization` key authenticates fine, so doctor must find it and report OK,
    not drop it, probe with no credential, and WARN a working client."""
    import poppy.mcp_server.lifecycle as lifecycle

    cursor_home = tmp_path / "cursor"
    cursor_home.mkdir(parents=True)
    (cursor_home / "mcp.json").write_text(  # UPPERCASE header key
        json.dumps(
            {"mcpServers": {"poppy": {"url": "http://127.0.0.1:7679/mcp", "headers": {"AUTHORIZATION": "Bearer good"}}}}
        )
    )
    codex_home = tmp_path / "codex"
    codex_home.mkdir(parents=True)
    (codex_home / "config.toml").write_text(  # lowercase header key
        "[mcp_servers.poppy]\n"
        'url = "http://127.0.0.1:7679/mcp"\n'
        "[mcp_servers.poppy.http_headers]\n"
        'authorization = "Bearer good"\n'
    )

    def fake_probe(poppy_dir, port=None, timeout=0.25, token=None, host="127.0.0.1", auth_header=None):
        return ({"version": "x"}, None) if auth_header == "Bearer good" else (None, "HTTP Error 401: Unauthorized")

    monkeypatch.setattr(lifecycle, "probe_status", fake_probe)

    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "HOME": str(tmp_path),
            "CURSOR_HOME": str(cursor_home),
            "CODEX_HOME": str(codex_home),
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_DAEMON_PORT": "7679",
            "POPPY_LAUNCH_AGENTS_DIR": str(tmp_path / "no-agents"),
            "POPPY_SYSTEMD_USER_DIR": str(tmp_path / "no-systemd"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )
    assert "Cursor MCP: OK" in result.output  # uppercase key found
    assert "Codex MCP: OK" in result.output  # lowercase key found
    assert "no Authorization credential" not in result.output


def _doctor_env_for_daemon_url(tmp_path, url):
    cursor_home = tmp_path / "cursor"
    cursor_home.mkdir(parents=True)
    (cursor_home / "mcp.json").write_text(
        json.dumps({"mcpServers": {"poppy": {"url": url, "headers": {"Authorization": "Bearer x"}}}})
    )
    return {
        "HOME": str(tmp_path),
        "CURSOR_HOME": str(cursor_home),
        "CODEX_HOME": str(tmp_path / "no-codex"),
        "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
        "POPPY_DIR": str(tmp_path / ".poppy"),
        "POPPY_DAEMON_PORT": "7679",
        "POPPY_LAUNCH_AGENTS_DIR": str(tmp_path / "no-agents"),
        "POPPY_SYSTEMD_USER_DIR": str(tmp_path / "no-systemd"),
        "POPPY_TELEMETRY_OFF": "1",
    }


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://127.0.0.1:7679/mcp;",  # bare trailing `;` — urlparse peels this off, urlsplit keeps it
        "http://127.0.0.1:7679/mcp;broken",  # path parameter — part of the request path
        "http://127.0.0.1:7679/wrong/mcp",  # different path
    ],
)
def test_doctor_warns_when_daemon_url_path_is_not_exactly_mcp(tmp_path, monkeypatch, bad_url):
    """A path parameter (`;` / `;broken`) or a different path changes the route the
    server resolves, so the client's real endpoint 404s even though a probe of
    the same host:port /status answers. Must WARN, not OK."""
    import poppy.mcp_server.lifecycle as lifecycle

    monkeypatch.setattr(lifecycle, "probe_status", lambda *a, **k: ({"version": "x"}, None))  # would say OK
    result = CliRunner().invoke(cli, ["doctor"], env=_doctor_env_for_daemon_url(tmp_path, bad_url))
    assert "Cursor MCP: WARN" in result.output, bad_url
    assert "does not serve" in result.output
    assert "Cursor MCP: OK" not in result.output


@pytest.mark.parametrize(
    "ok_url",
    [
        "http://127.0.0.1:7679/mcp?x=1",  # query — ignored for route matching
        "http://127.0.0.1:7679/mcp#frag",  # fragment — never sent to the server
    ],
)
def test_doctor_allows_daemon_url_query_and_fragment(tmp_path, monkeypatch, ok_url):
    """A query or fragment does NOT change the endpoint the server resolves, so a
    `/mcp?x=1` or `/mcp#frag` client is healthy — doctor must NOT falsely WARN
    it."""
    import poppy.mcp_server.lifecycle as lifecycle

    monkeypatch.setattr(lifecycle, "probe_status", lambda *a, **k: ({"version": "x"}, None))
    result = CliRunner().invoke(cli, ["doctor"], env=_doctor_env_for_daemon_url(tmp_path, ok_url))
    assert "Cursor MCP: OK" in result.output, ok_url
    assert "does not serve" not in result.output


@pytest.mark.parametrize(
    "bad_url",
    [
        "http:/127.0.0.1:59321/mcp",  # single slash → no netloc / no host
        "http:///mcp",  # empty host
        "ftp://127.0.0.1:59321/mcp",  # wrong scheme
    ],
)
def test_doctor_warns_when_daemon_url_is_malformed(tmp_path, monkeypatch, bad_url):
    """A url-bearing entry IS a daemon registration even when the URL is malformed
    (no netloc / empty host / wrong scheme): it must WARN, not collapse to None
    and print a bare `Cursor MCP: OK` with the daemon checks suppressed."""
    import poppy.mcp_server.lifecycle as lifecycle

    monkeypatch.setattr(lifecycle, "probe_status", lambda *a, **k: ({"version": "x"}, None))  # would say OK
    result = CliRunner().invoke(cli, ["doctor"], env=_doctor_env_for_daemon_url(tmp_path, bad_url))
    assert "Cursor MCP: WARN" in result.output, bad_url
    assert "Cursor MCP: OK" not in result.output


def test_doctor_ok_for_stdio_command_entry(tmp_path):
    """The non-daemon case: a poppy entry with a `command` (stdio) and no `url` is
    a normal local registration → OK, no daemon section."""
    cursor_home = tmp_path / "cursor"
    cursor_home.mkdir(parents=True)
    (cursor_home / "mcp.json").write_text(
        json.dumps({"mcpServers": {"poppy": {"command": "poppy", "args": ["serve", "--source", "cursor"]}}})
    )
    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "HOME": str(tmp_path),
            "CURSOR_HOME": str(cursor_home),
            "CODEX_HOME": str(tmp_path / "no-codex"),
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
            "POPPY_DIR": str(tmp_path / ".poppy"),
            "POPPY_DAEMON_PORT": "59321",
            "POPPY_LAUNCH_AGENTS_DIR": str(tmp_path / "no-agents"),
            "POPPY_SYSTEMD_USER_DIR": str(tmp_path / "no-systemd"),
            "POPPY_TELEMETRY_OFF": "1",
        },
    )
    assert result.exit_code == 0, result.output
    assert "Cursor MCP: OK" in result.output
    assert "daemon installed" not in result.output  # not a daemon client → no daemon section


# ---------- Review fixes (PR #63) ----------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("true", "on"),
        ("yes", "on"),
        ("1", "on"),
        ("on", "on"),
        ("false", "off"),
        ("no", "off"),
        ("0", "off"),
        ("off", "off"),
    ],
)
def test_config_set_telemetry_echoes_normalized(tmp_path, value, expected):
    """`config set telemetry <alias>` echoes the canonical on/off (byte-identical
    to the pre-registry special-case), not the raw alias string."""
    result = CliRunner().invoke(cli, ["config", "set", "telemetry", value], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert f"Set telemetry = {expected}" in result.output


def test_config_set_invalid_value_does_not_touch_config_json(tmp_path):
    """An invalid value is rejected before any load/save, so it never creates or
    rewrites config.json (load_config can migrate a plaintext key as a side effect)."""
    env = {"POPPY_DIR": str(tmp_path)}
    cfg_path = tmp_path / "config.json"

    # No pre-existing config: an invalid set must not create one.
    result = CliRunner().invoke(cli, ["config", "set", "auto-sync", "bogus"], env=env)
    assert result.exit_code == 2
    assert not cfg_path.exists()

    # Pre-existing config: an invalid set must leave it byte-for-byte unchanged.
    CliRunner().invoke(cli, ["config", "set", "auto-sync", "off"], env=env)
    before = cfg_path.read_bytes()
    result = CliRunner().invoke(cli, ["config", "set", "recall-min-score", "nope"], env=env)
    assert result.exit_code == 2
    assert cfg_path.read_bytes() == before
