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


# The shim always searches these, whatever HOME and PATH say.
_SYSTEM_POPPY = any(Path(p).exists() for p in ("/opt/homebrew/bin/poppy", "/usr/local/bin/poppy"))
needs_no_system_poppy = pytest.mark.skipif(_SYSTEM_POPPY, reason="a system-wide poppy is installed")


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell shim")
@needs_no_system_poppy
def test_shim_explains_a_missing_poppy_without_blocking(tmp_path):
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    result = subprocess.run([str(SHIM), "hook", "user-prompt-submit"], env=env, capture_output=True, text=True)
    # Exit 2 would block the prompt or tool call that fired the hook.
    assert result.returncode == 127
    assert "pipx install poppy-memory" in result.stderr


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell shim")
@pytest.mark.parametrize(("args", "expected"), [(["hook", "pre-tool-use"], 1), (["serve"], 2), (["hook", "x"], 1)])
def test_shim_never_lets_a_hook_exit_2(tmp_path, args, expected):
    bin_dir = tmp_path / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "poppy").write_text("#!/bin/sh\nexit 2\n")
    (bin_dir / "poppy").chmod(0o755)
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    result = subprocess.run([str(SHIM), *args], env=env, capture_output=True, text=True)
    # Exit 2 blocks the prompt or tool call behind a hook; other commands keep poppy's code.
    assert result.returncode == expected


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell shim")
def test_shim_keeps_other_hook_exit_codes(tmp_path):
    _fake_poppy(tmp_path / ".local" / "bin")
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    result = subprocess.run([str(SHIM), "hook", "session-end"], env=env, capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout == "fake poppy hook session-end\n"


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell shim")
def test_shim_finds_a_pip_user_install_on_macos(tmp_path):
    _fake_poppy(tmp_path / "Library" / "Python" / "3.12" / "bin")
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    result = subprocess.run([str(SHIM), "--version"], env=env, capture_output=True, text=True)
    assert result.stdout == "fake poppy --version\n"


def _record_install(
    claude_dir: Path,
    *,
    scope: str = "user",
    enabled: bool | None = True,
    plugins_root: Path | None = None,
) -> None:
    plugins = plugins_root or claude_dir / "plugins"
    plugins.mkdir(parents=True, exist_ok=True)
    entry = {"scope": scope, "installPath": str(PLUGIN_ROOT), "version": "0.3.4"}
    record = {"version": 2, "plugins": {CLAUDE_CODE_PLUGIN_ID: [entry]}}
    (plugins / "installed_plugins.json").write_text(json.dumps(record))
    claude_dir.mkdir(parents=True, exist_ok=True)
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


def test_plugin_detection_follows_the_plugin_cache_dir(tmp_path, monkeypatch):
    moved = tmp_path / "elsewhere"
    _record_install(tmp_path / "claude", plugins_root=moved)
    assert is_claude_code_plugin_installed(tmp_path / "claude") is False
    monkeypatch.setenv("CLAUDE_CODE_PLUGIN_CACHE_DIR", str(moved))
    assert is_claude_code_plugin_installed(tmp_path / "claude") is True


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
    assert "setup did not add them" in result.output
    assert (claude_dir / "settings.json").read_text() == settings_before
    assert not (claude_dir / ".claude.json").exists()
    assert not (claude_dir / "CLAUDE.md").exists()


def test_setup_installs_when_the_plugin_is_disabled(tmp_path):
    env = _env(tmp_path)
    claude_dir = Path(env["CLAUDE_CONFIG_DIR"])
    _record_install(claude_dir, enabled=False)

    result = CliRunner().invoke(cli, ["setup", "claude-code"], env=env)

    assert result.exit_code == 0, result.output
    assert "setup did not add them" not in result.output
    assert "SessionStart" in json.loads((claude_dir / "settings.json").read_text())["hooks"]
    assert (claude_dir / "CLAUDE.md").exists()


def test_setup_skip_follows_the_plugin_cache_dir(tmp_path):
    env = _env(tmp_path)
    claude_dir = Path(env["CLAUDE_CONFIG_DIR"])
    env["CLAUDE_CODE_PLUGIN_CACHE_DIR"] = str(tmp_path / "plugin-cache")
    _record_install(claude_dir, plugins_root=tmp_path / "plugin-cache")

    result = CliRunner().invoke(cli, ["setup", "claude-code"], env=env)

    assert result.exit_code == 0, result.output
    assert "setup did not add them" in result.output
    assert "hooks" not in json.loads((claude_dir / "settings.json").read_text())


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs POSIX file modes and a non-root user")
def test_setup_continues_when_plugin_records_are_unreadable(tmp_path):
    env = _env(tmp_path)
    claude_dir = Path(env["CLAUDE_CONFIG_DIR"])
    _record_install(claude_dir)
    record = claude_dir / "plugins" / "installed_plugins.json"
    record.chmod(0)
    try:
        result = CliRunner().invoke(cli, ["setup", "claude-code"], env=env)
    finally:
        record.chmod(0o600)

    assert result.exit_code == 0, result.output
    assert "could not read Claude Code's plugin records" in result.output
    assert "SessionStart" in json.loads((claude_dir / "settings.json").read_text())["hooks"]


def test_setup_with_the_plugin_adds_claude_md_only_when_asked(tmp_path):
    env = _env(tmp_path)
    claude_dir = Path(env["CLAUDE_CONFIG_DIR"])
    _record_install(claude_dir)
    runner = CliRunner()

    assert runner.invoke(cli, ["setup", "claude-code"], env=env).exit_code == 0
    assert not (claude_dir / "CLAUDE.md").exists()

    result = runner.invoke(cli, ["setup", "claude-code", "--claude-md"], env=env)
    assert result.exit_code == 0, result.output
    assert "CLAUDE.md block" in result.output
    assert "<!-- POPPY:BEGIN -->" in (claude_dir / "CLAUDE.md").read_text()
    assert "hooks" not in json.loads((claude_dir / "settings.json").read_text())


def test_setup_with_the_plugin_explains_an_ignored_daemon_flag(tmp_path, monkeypatch):
    import poppy.cli.main as main_mod

    def no_daemon():
        raise AssertionError("the daemon must not be set up when the plugin is installed")

    monkeypatch.setattr(main_mod, "_daemon_setup_kwargs", no_daemon)
    env = _env(tmp_path)
    _record_install(Path(env["CLAUDE_CONFIG_DIR"]))

    result = CliRunner().invoke(cli, ["setup", "claude-code", "--daemon"], env=env)

    assert result.exit_code == 0, result.output
    assert "--daemon was ignored" in result.output


def test_setup_with_the_plugin_records_the_plugin_path(tmp_path, monkeypatch):
    import poppy.cli.main as main_mod

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(main_mod.telemetry, "capture", lambda _dir, event, props=None: events.append((event, props)))
    monkeypatch.setattr(
        main_mod.telemetry, "capture_once", lambda _dir, event, props=None: events.append((event, props)) or True
    )
    env = _env(tmp_path)
    _record_install(Path(env["CLAUDE_CONFIG_DIR"]))

    assert CliRunner().invoke(cli, ["setup", "claude-code"], env=env).exit_code == 0

    expected = {"agent": "claude-code", "via": "plugin"}
    assert ("agent_setup", expected) in events
    assert ("setup_completed", expected) in events


def _doctor(env: dict) -> str:
    runner = CliRunner()
    runner.invoke(cli, ["config", "set", "engine", "seed"], env=env)
    result = runner.invoke(cli, ["doctor"], env=env)
    return result.output


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell shim")
def test_doctor_accepts_the_plugin_as_the_claude_code_wiring(tmp_path):
    env = _env(tmp_path)
    _fake_poppy(tmp_path / ".local" / "bin")
    _record_install(Path(env["CLAUDE_CONFIG_DIR"]))

    output = _doctor(env)

    assert f"Claude Code plugin: OK, {CLAUDE_CODE_PLUGIN_ID}" in output
    assert "WARN" not in output, output


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell shim")
@needs_no_system_poppy
def test_doctor_fails_when_the_plugin_cannot_find_poppy(tmp_path):
    env = {**_env(tmp_path), "PATH": "/usr/bin:/bin"}
    _record_install(Path(env["CLAUDE_CONFIG_DIR"]))

    output = _doctor(env)

    assert "Claude Code plugin: FAIL" in output
    assert "cannot find poppy" in output


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell shim")
@needs_no_system_poppy
def test_doctor_warns_when_the_plugin_finds_poppy_only_on_this_path(tmp_path):
    on_path = tmp_path / "custom-bin"
    _fake_poppy(on_path)
    env = {**_env(tmp_path), "PATH": f"{on_path}:/usr/bin:/bin"}
    _record_install(Path(env["CLAUDE_CONFIG_DIR"]))

    output = _doctor(env)

    assert "Claude Code plugin: WARN, the plugin finds poppy only through this shell's PATH" in output


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell shim")
def test_doctor_names_a_duplicate_mcp_server_without_claiming_duplicate_hooks(tmp_path):
    env = _env(tmp_path)
    _fake_poppy(tmp_path / ".local" / "bin")
    assert CliRunner().invoke(cli, ["setup", "claude-code", "--no-hooks"], env=env).exit_code == 0
    _record_install(Path(env["CLAUDE_CONFIG_DIR"]))

    output = _doctor(env)

    assert "registered a second Poppy MCP server" in output
    assert "claude mcp remove poppy -s user" in output
    assert "run twice" not in output


@pytest.mark.skipif(os.name == "nt", reason="POSIX shell shim")
def test_doctor_names_duplicate_hooks_and_still_checks_them(tmp_path):
    env = _env(tmp_path)
    claude_dir = Path(env["CLAUDE_CONFIG_DIR"])
    _fake_poppy(tmp_path / ".local" / "bin")
    assert CliRunner().invoke(cli, ["setup", "claude-code"], env=env).exit_code == 0
    # Point setup's hooks at a binary that no longer exists.
    settings = json.loads((claude_dir / "settings.json").read_text())
    for groups in settings["hooks"].values():
        for group in groups:
            for handler in group["hooks"]:
                handler["command"] = "/nonexistent/poppy " + " ".join(handler["command"].split()[1:])
    (claude_dir / "settings.json").write_text(json.dumps(settings))
    _record_install(claude_dir)

    output = _doctor(env)

    assert "so Poppy's hooks run twice" in output
    assert "hook command: FAIL" in output
