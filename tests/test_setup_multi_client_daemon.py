"""Multi-client native daemon registration."""

import json
import tomllib
from pathlib import Path

import pytest
from click.testing import CliRunner

from poppy.cli.main import cli
from poppy.setup.claude_code import (
    CorruptConfigError,
    get_vscode_mcp_config_path,
    install_for_client,
)

TOKEN = "test-daemon-token"
PORT = 8765
URL = f"http://127.0.0.1:{PORT}/mcp"


@pytest.fixture(autouse=True)
def isolate_codex_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))


def _config_path(tmp_path: Path, client: str) -> Path:
    return {
        "cursor": tmp_path / ".cursor" / "mcp.json",
        "vscode": tmp_path / "vscode-mcp.json",
        "codex": tmp_path / ".codex" / "config.toml",
        "gemini": tmp_path / ".gemini" / "settings.json",
    }[client]


def _setup_env(tmp_path: Path, client: str) -> dict[str, str]:
    env = {
        "HOME": str(tmp_path),
        "POPPY_DIR": str(tmp_path / ".poppy"),
        "POPPY_TELEMETRY_OFF": "1",
    }
    if client == "vscode":
        env["POPPY_VSCODE_MCP_CONFIG"] = str(_config_path(tmp_path, client))
    return env


def _ready_probe(_path, _port=None, timeout=0.25):
    """A probe_status stand-in that reports the daemon authenticated and ready."""
    return {"version": "t"}, None


def test_cursor_daemon_entry_has_exact_remote_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    install_for_client(client="cursor", daemon=True, daemon_port=PORT, daemon_token=TOKEN)

    settings = json.loads(_config_path(tmp_path, "cursor").read_text())
    assert settings["mcpServers"]["poppy"] == {
        "url": URL,
        "headers": {"Authorization": f"Bearer {TOKEN}"},
    }


def test_gemini_stdio_and_daemon_entries_use_native_dialect(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("poppy.setup.claude_code.get_poppy_executable", lambda: "/test/poppy")

    install_for_client(client="gemini")
    path = _config_path(tmp_path, "gemini")
    settings = json.loads(path.read_text())
    assert settings["mcpServers"]["poppy"] == {
        "command": "/test/poppy",
        "args": ["serve", "--source", "gemini"],
    }

    install_for_client(client="gemini", daemon=True, daemon_port=PORT, daemon_token=TOKEN)
    settings = json.loads(path.read_text())
    assert settings["mcpServers"]["poppy"] == {
        "httpUrl": URL,
        "headers": {"Authorization": f"Bearer {TOKEN}"},
    }
    assert "url" not in settings["mcpServers"]["poppy"]


def test_vscode_stdio_and_daemon_entries_use_servers_key(tmp_path, monkeypatch):
    path = _config_path(tmp_path, "vscode")
    monkeypatch.setenv("POPPY_VSCODE_MCP_CONFIG", str(path))
    monkeypatch.setattr("poppy.setup.claude_code.get_poppy_executable", lambda: "/test/poppy")

    install_for_client(client="vscode")
    settings = json.loads(path.read_text())
    assert settings == {
        "servers": {
            "poppy": {
                "command": "/test/poppy",
                "args": ["serve", "--source", "vscode"],
                "type": "stdio",
            }
        }
    }

    install_for_client(client="vscode", daemon=True, daemon_port=PORT, daemon_token=TOKEN)
    settings = json.loads(path.read_text())
    assert settings["servers"]["poppy"] == {
        "type": "http",
        "url": URL,
        "headers": {"Authorization": f"Bearer {TOKEN}"},
    }
    assert "mcpServers" not in settings


def test_codex_daemon_uses_static_headers_and_preserves_unrelated_toml(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    path = _config_path(tmp_path, "codex")
    path.parent.mkdir(parents=True, exist_ok=True)
    original = '# preserve this comment\nmodel = "gpt-test"\n\n[mcp_servers.other]\ncommand = "other"\n'
    path.write_text(original)

    paths = install_for_client(client="codex", daemon=True, daemon_port=PORT, daemon_token=TOKEN)

    backup = path.with_name(path.name + ".pre-poppy.bak")
    assert paths["backup"] == backup
    assert backup.read_text() == original
    rendered = path.read_text()
    assert "# preserve this comment" in rendered
    parsed = tomllib.loads(rendered)
    assert parsed["model"] == "gpt-test"
    assert parsed["mcp_servers"]["other"] == {"command": "other"}
    assert parsed["mcp_servers"]["poppy"] == {
        "url": URL,
        "http_headers": {"Authorization": f"Bearer {TOKEN}"},
    }


def test_codex_daemon_keeps_legacy_json_migration(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    legacy = tmp_path / ".codex" / "config.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"mcpServers": {"poppy": {"command": "old"}, "other": {"command": "keep"}}}))

    paths = install_for_client(client="codex", daemon=True, daemon_port=PORT, daemon_token=TOKEN)

    assert paths["Removed legacy MCP config"] == legacy
    assert json.loads(legacy.read_text()) == {"mcpServers": {"other": {"command": "keep"}}}
    poppy = tomllib.loads(_config_path(tmp_path, "codex").read_text())["mcp_servers"]["poppy"]
    assert poppy["http_headers"] == {"Authorization": f"Bearer {TOKEN}"}


@pytest.mark.parametrize("client", ["cursor", "vscode", "gemini"])
def test_json_client_backup_is_written_once(tmp_path, monkeypatch, client):
    path = _config_path(tmp_path, client)
    monkeypatch.setenv("HOME", str(tmp_path))
    if client == "vscode":
        monkeypatch.setenv("POPPY_VSCODE_MCP_CONFIG", str(path))
        servers_key = "servers"
    else:
        servers_key = "mcpServers"
    path.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps({servers_key: {"other": {"command": "keep"}}, "marker": "original"})
    path.write_text(original)

    install_for_client(client=client, daemon=True, daemon_port=PORT, daemon_token=TOKEN)
    backup = path.with_name(path.name + ".pre-poppy.bak")
    assert backup.read_text() == original

    path.write_text(path.read_text().replace('"marker": "original"', '"marker": "changed"'))
    install_for_client(client=client, daemon=True, daemon_port=PORT, daemon_token=TOKEN)
    assert backup.read_text() == original


@pytest.mark.parametrize("client", ["vscode", "gemini"])
def test_new_json_clients_abort_on_corrupt_config(tmp_path, monkeypatch, client):
    path = _config_path(tmp_path, client)
    monkeypatch.setenv("HOME", str(tmp_path))
    if client == "vscode":
        monkeypatch.setenv("POPPY_VSCODE_MCP_CONFIG", str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    corrupt = "{ invalid json"
    path.write_text(corrupt)

    with pytest.raises(CorruptConfigError):
        install_for_client(client=client, daemon=True, daemon_port=PORT, daemon_token=TOKEN)

    assert path.read_text() == corrupt
    assert not path.with_name(path.name + ".pre-poppy.bak").exists()


def test_codex_daemon_aborts_on_corrupt_toml(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    path = _config_path(tmp_path, "codex")
    path.parent.mkdir(parents=True)
    corrupt = "[mcp_servers.poppy\nurl = nope"
    path.write_text(corrupt)

    with pytest.raises(CorruptConfigError):
        install_for_client(client="codex", daemon=True, daemon_port=PORT, daemon_token=TOKEN)

    assert path.read_text() == corrupt
    assert not path.with_name(path.name + ".pre-poppy.bak").exists()


def test_vscode_path_override_and_platform_defaults(tmp_path, monkeypatch):
    override = tmp_path / "override.json"
    monkeypatch.setenv("POPPY_VSCODE_MCP_CONFIG", str(override))
    assert get_vscode_mcp_config_path() == override

    monkeypatch.delenv("POPPY_VSCODE_MCP_CONFIG")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("poppy.setup.claude_code.sys.platform", "darwin")
    assert get_vscode_mcp_config_path() == tmp_path / "Library/Application Support/Code/User/mcp.json"

    monkeypatch.setattr("poppy.setup.claude_code.sys.platform", "linux")
    assert get_vscode_mcp_config_path() == tmp_path / ".config/Code/User/mcp.json"

    appdata = tmp_path / "AppData/Roaming"
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setattr("poppy.setup.claude_code.sys.platform", "win32")
    assert get_vscode_mcp_config_path() == appdata / "Code/User/mcp.json"


@pytest.mark.parametrize("client", ["cursor", "vscode", "codex", "gemini"])
def test_setup_polls_until_the_rotated_token_is_served(tmp_path, monkeypatch, client):
    """setup waits for authenticated readiness: an async service rotation (env B) is baked, not the pre-start file A.

    install_agent starts the service, which holds the run-lock and rotates the file
    to B asynchronously; setup's own start_daemon short-circuits on the held lock and
    returns before the rotation lands. setup must poll /status until the daemon serves
    the settled token, then bake that — never the stale pre-rotation value.
    """
    monkeypatch.delenv("POPPY_DAEMON_TOKEN", raising=False)
    poppy_dir = tmp_path / ".poppy"
    poppy_dir.mkdir(parents=True, exist_ok=True)
    token_file = poppy_dir / "daemon.token"
    token_file.write_text("stale-A\n")  # the pre-rotation on-disk token
    state = {"probes": 0}

    monkeypatch.setattr("poppy.mcp_server.auth.ensure_daemon_token", lambda path, *, rotate_from_env=False: "stale-A")
    monkeypatch.setattr("poppy.mcp_server.lifecycle.install_agent", lambda path: None)
    # start_daemon short-circuits on the lock the async service already holds.
    monkeypatch.setattr("poppy.mcp_server.lifecycle.start_daemon", lambda path: "Poppy daemon is already running.")
    monkeypatch.setattr("poppy.mcp_server.lifecycle.daemon_port", lambda: PORT)

    def fake_probe(path, port=None, timeout=0.25):
        state["probes"] += 1
        if state["probes"] < 2:
            return None, "starting"  # not ready yet; the file still holds stale-A
        # The service finished initializing: it rotated the file to served-B and serves it.
        token_file.write_text("served-B\n")
        return {"version": "test", "port": port}, None

    monkeypatch.setattr("poppy.mcp_server.lifecycle.probe_status", fake_probe)

    result = CliRunner().invoke(cli, ["setup", client, "--daemon"], env=_setup_env(tmp_path, client))

    assert result.exit_code == 0, result.output
    assert state["probes"] >= 2, "setup did not poll for authenticated readiness"
    config_text = _config_path(tmp_path, client).read_text()
    assert "served-B" in config_text, "the client was not configured with the served (rotated) token"
    assert "stale-A" not in config_text, "the client baked a pre-rotation token the daemon no longer serves"
    if client == "codex":
        entry = tomllib.loads(config_text)["mcp_servers"]["poppy"]
        assert entry == {"url": URL, "http_headers": {"Authorization": "Bearer served-B"}}
    else:
        settings = json.loads(config_text)
        servers_key = "servers" if client == "vscode" else "mcpServers"
        string_values = {value for value in settings[servers_key]["poppy"].values() if isinstance(value, str)}
        assert URL in string_values


def test_setup_seeds_and_bakes_the_env_token_on_a_fresh_machine(tmp_path, monkeypatch):
    """POPPY_DAEMON_TOKEN=X with no token file -> setup seeds X and bakes X (uses the real ensure/seed)."""
    client = "codex"
    monkeypatch.setattr("poppy.mcp_server.lifecycle.install_agent", lambda path: None)
    monkeypatch.setattr("poppy.mcp_server.lifecycle.start_daemon", lambda path: "started")
    monkeypatch.setattr("poppy.mcp_server.lifecycle.daemon_port", lambda: PORT)
    # The env-less service reads the seeded token from disk and serves it; /status authenticates.
    monkeypatch.setattr("poppy.mcp_server.lifecycle.probe_status", _ready_probe)

    env = {**_setup_env(tmp_path, client), "POPPY_DAEMON_TOKEN": "fresh-X"}
    result = CliRunner().invoke(cli, ["setup", client, "--daemon"], env=env)

    assert result.exit_code == 0, result.output
    assert (tmp_path / ".poppy" / "daemon.token").read_text().strip() == "fresh-X"  # seeded absent file
    entry = tomllib.loads(_config_path(tmp_path, client).read_text())["mcp_servers"]["poppy"]
    assert entry == {"url": URL, "http_headers": {"Authorization": "Bearer fresh-X"}}  # baked what will be served


def test_setup_raises_when_the_daemon_never_becomes_ready(tmp_path, monkeypatch):
    """A daemon that never authenticates -> setup fails and leaves the client config untouched."""
    client = "codex"
    path = _config_path(tmp_path, client)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('model = "keep"\n')
    monkeypatch.delenv("POPPY_DAEMON_TOKEN", raising=False)
    poppy_dir = tmp_path / ".poppy"
    poppy_dir.mkdir(parents=True, exist_ok=True)
    (poppy_dir / "daemon.token").write_text("A\n")

    monkeypatch.setattr("poppy.mcp_server.auth.ensure_daemon_token", lambda path, *, rotate_from_env=False: "A")
    monkeypatch.setattr("poppy.mcp_server.lifecycle.install_agent", lambda path: None)
    monkeypatch.setattr("poppy.mcp_server.lifecycle.start_daemon", lambda path: "started")
    monkeypatch.setattr("poppy.mcp_server.lifecycle.daemon_port", lambda: PORT)
    monkeypatch.setattr("poppy.mcp_server.lifecycle.probe_status", lambda path, port=None, timeout=0.25: (None, "down"))
    monkeypatch.setattr("poppy.mcp_server.lifecycle.DEFAULT_LIFECYCLE_TIMEOUT", 0.2)  # keep the test fast

    result = CliRunner().invoke(cli, ["setup", client, "--daemon"], env=_setup_env(tmp_path, client))

    assert result.exit_code == 1
    assert "did not become ready" in result.output
    assert path.read_text() == 'model = "keep"\n'


@pytest.mark.parametrize("client", ["cursor", "vscode", "codex", "gemini"])
def test_setup_daemon_lifecycle_failure_leaves_config_untouched(tmp_path, monkeypatch, client):
    path = _config_path(tmp_path, client)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = 'model = "keep"\n' if client == "codex" else '{"marker": "keep"}\n'
    path.write_text(original)
    monkeypatch.setattr("poppy.mcp_server.auth.ensure_daemon_token", lambda _path: TOKEN)
    monkeypatch.setattr("poppy.mcp_server.lifecycle.install_agent", lambda _path: None)
    monkeypatch.setattr(
        "poppy.mcp_server.lifecycle.start_daemon", lambda _path: (_ for _ in ()).throw(RuntimeError("start failed"))
    )

    result = CliRunner().invoke(cli, ["setup", client, "--daemon"], env=_setup_env(tmp_path, client))

    assert result.exit_code == 1
    assert "client config was not changed" in result.output
    assert path.read_text() == original
