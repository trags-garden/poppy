import json

from click.testing import CliRunner

from poppy.cli.main import cli


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
    # Engine is whichever resolved on this machine: bloom (default,
    # fallback chain), sprout, or seed as the final ML-deps-missing floor.
    assert any(e in result.output for e in ("bloom", "best", "seed"))


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

    # Verify SessionStart and SessionEnd hooks were written to settings.json
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
    assert "poppy consent --enable" in result.output
    # The old, wrong hint must be gone.
    assert "config set consolidate-enabled true" not in result.output


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

    def boom(_poppy_dir):
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
