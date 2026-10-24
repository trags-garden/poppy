"""The Claude Code plugin in plugins/poppy and how setup and doctor treat it.

The plugin is static JSON, so these tests keep it in step with what
``poppy setup claude-code`` writes: the same MCP arguments, hook events,
matchers and hook subcommands, and the same primer text.
"""

from __future__ import annotations

import json
import os
import shlex
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from poppy.cli.main import cli
from poppy.setup.claude_code import (
    _HOOK_DEFS,
    CLAUDE_CODE_PLUGIN_ID,
    CLAUDE_MD_BODY,
    _mcp_entry,
    is_claude_code_plugin_installed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = REPO_ROOT / "plugins" / "poppy"
SHIM = PLUGIN_ROOT / "scripts" / "poppy-run"
SHIM_COMMAND = "${CLAUDE_PLUGIN_ROOT}/scripts/poppy-run"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_marketplace_lists_the_plugin_under_the_detected_id():
    marketplace = _load(REPO_ROOT / ".claude-plugin" / "marketplace.json")
    manifest = _load(PLUGIN_ROOT / ".claude-plugin" / "plugin.json")
    (entry,) = marketplace["plugins"]
    assert (REPO_ROOT / entry["source"]).resolve() == PLUGIN_ROOT
    assert entry["name"] == manifest["name"]
    assert f"{entry['name']}@{marketplace['name']}" == CLAUDE_CODE_PLUGIN_ID


def test_mcp_server_runs_the_same_command_as_setup():
    (server,) = _load(PLUGIN_ROOT / ".mcp.json")["mcpServers"].values()
    expected = _mcp_entry("claude-code")
    assert server["command"] == SHIM_COMMAND
    assert server["args"] == expected["args"]
    assert server["type"] == expected["type"]


def test_hooks_match_setup_events_matchers_and_subcommands():
    hooks = _load(PLUGIN_ROOT / "hooks" / "hooks.json")["hooks"]
    assert set(hooks) == set(_HOOK_DEFS)
    for event, (matcher, canonical) in _HOOK_DEFS.items():
        (group,) = hooks[event]
        (handler,) = group["hooks"]
        assert group["matcher"] == matcher
        assert handler["type"] == "command"
        argv = shlex.split(handler["command"])
        assert argv[0] == SHIM_COMMAND
        assert argv[1:] == canonical.split()[1:]


def test_skill_body_is_the_setup_primer():
    text = (PLUGIN_ROOT / "skills" / "memory" / "SKILL.md").read_text()
    _, frontmatter, body = text.split("---\n", 2)
    assert "name: memory" in frontmatter
    assert body.strip() == CLAUDE_MD_BODY.strip()


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell shim")
def test_shim_is_executable():
    assert SHIM.stat().st_mode & stat.S_IXUSR


def _fake_poppy(directory: Path) -> Path:
    directory.mkdir(parents=True)
    fake = directory / "poppy"
    fake.write_text('#!/bin/sh\necho "fake poppy $*"\n')
    fake.chmod(0o755)
    return fake


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell shim")
def test_shim_finds_poppy_outside_path(tmp_path):
    _fake_poppy(tmp_path / ".local" / "bin")
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    result = subprocess.run([str(SHIM), "hook", "session-start"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "fake poppy hook session-start\n"


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell shim")
def test_shim_prefers_poppy_on_path(tmp_path):
    _fake_poppy(tmp_path / ".local" / "bin")
    on_path = tmp_path / "on-path"
    on_path.mkdir()
    (on_path / "poppy").write_text('#!/bin/sh\necho "path poppy"\n')
    (on_path / "poppy").chmod(0o755)
    env = {"HOME": str(tmp_path), "PATH": f"{on_path}:/usr/bin:/bin"}
    result = subprocess.run([str(SHIM), "serve"], env=env, capture_output=True, text=True)
    assert result.stdout == "path poppy\n"


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell shim")
@pytest.mark.skipif(
    sys.platform == "darwin" and any(Path(p).exists() for p in ("/opt/homebrew/bin/poppy", "/usr/local/bin/poppy")),
    reason="a system-wide poppy is installed",
)
def test_shim_explains_a_missing_poppy_without_blocking(tmp_path):
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    result = subprocess.run([str(SHIM), "hook", "user-prompt-submit"], env=env, capture_output=True, text=True)
    # Exit 2 would block the prompt or tool call that fired the hook.
    assert result.returncode == 127
    assert "pipx install poppy-memory" in result.stderr


def _record_install(claude_dir: Path, *, scope: str = "user", enabled: bool | None = True) -> None:
    plugins = claude_dir / "plugins"
    plugins.mkdir(parents=True, exist_ok=True)
    record = {"version": 2, "plugins": {CLAUDE_CODE_PLUGIN_ID: [{"scope": scope, "version": "0.3.4"}]}}
    (plugins / "installed_plugins.json").write_text(json.dumps(record))
    settings_path = claude_dir / "settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    if enabled is not None:
        settings["enabledPlugins"] = {CLAUDE_CODE_PLUGIN_ID: enabled}
    settings_path.write_text(json.dumps(settings))


@pytest.mark.parametrize(
    ("scope", "enabled", "expected"),
    [
        ("user", True, True),
        ("user", None, True),
        ("user", False, False),
        ("project", True, False),
    ],
)
def test_plugin_detection(tmp_path, scope, enabled, expected):
    _record_install(tmp_path, scope=scope, enabled=enabled)
    assert is_claude_code_plugin_installed(tmp_path) is expected


def test_plugin_detection_tolerates_missing_and_odd_records(tmp_path):
    assert is_claude_code_plugin_installed(tmp_path) is False
    (tmp_path / "plugins").mkdir()
    for content in ("[]", '{"plugins": []}', '{"plugins": {"poppy@trags-garden": {}}}', "not json"):
        (tmp_path / "plugins" / "installed_plugins.json").write_text(content)
        assert is_claude_code_plugin_installed(tmp_path) is False


def _env(tmp_path: Path) -> dict:
    claude_dir = tmp_path / "claude"
    claude_dir.mkdir()
    (claude_dir / "settings.json").write_text(json.dumps({"permissions": {"allow": []}}))
    return {
        "HOME": str(tmp_path),
        "POPPY_DIR": str(tmp_path / ".poppy"),
        "CLAUDE_CONFIG_DIR": str(claude_dir),
        "CURSOR_HOME": str(tmp_path / "cursor"),
        "CODEX_HOME": str(tmp_path / "codex"),
        # A port nothing listens on, so doctor cannot reach a real daemon.
        "POPPY_DAEMON_PORT": "59321",
        "POPPY_TELEMETRY_OFF": "1",
    }


def test_setup_skips_when_the_plugin_is_installed(tmp_path):
    env = _env(tmp_path)
    claude_dir = Path(env["CLAUDE_CONFIG_DIR"])
    _record_install(claude_dir)
    settings_before = (claude_dir / "settings.json").read_text()

    result = CliRunner().invoke(cli, ["setup", "claude-code"], env=env)

    assert result.exit_code == 0, result.output
    assert "plugin for Claude Code is installed" in result.output
    assert (claude_dir / "settings.json").read_text() == settings_before
    assert not (claude_dir / ".claude.json").exists()
    assert not (claude_dir / "CLAUDE.md").exists()


def test_setup_installs_when_the_plugin_is_disabled(tmp_path):
    env = _env(tmp_path)
    claude_dir = Path(env["CLAUDE_CONFIG_DIR"])
    _record_install(claude_dir, enabled=False)

    result = CliRunner().invoke(cli, ["setup", "claude-code"], env=env)

    assert result.exit_code == 0, result.output
    assert "plugin for Claude Code is installed" not in result.output
    assert "SessionStart" in json.loads((claude_dir / "settings.json").read_text())["hooks"]
    assert (claude_dir / "CLAUDE.md").exists()


def test_doctor_accepts_the_plugin_as_the_claude_code_wiring(tmp_path):
    env = _env(tmp_path)
    _record_install(Path(env["CLAUDE_CONFIG_DIR"]))
    runner = CliRunner()
    runner.invoke(cli, ["config", "set", "engine", "seed"], env=env)

    result = runner.invoke(cli, ["doctor"], env=env)

    assert result.exit_code == 0, result.output
    assert f"Claude Code plugin: OK, {CLAUDE_CODE_PLUGIN_ID}" in result.output
    assert "WARN" not in result.output, result.output


def test_doctor_warns_when_setup_entries_sit_next_to_the_plugin(tmp_path):
    env = _env(tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["config", "set", "engine", "seed"], env=env)
    assert runner.invoke(cli, ["setup", "claude-code"], env=env).exit_code == 0
    _record_install(Path(env["CLAUDE_CONFIG_DIR"]))

    result = runner.invoke(cli, ["doctor"], env=env)

    assert "hooks run twice" in result.output
