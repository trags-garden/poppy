"""Client removal keeps user settings and never opens the memory store."""

import json
import os
import tomllib

import pytest
from click.testing import CliRunner

from poppy.cli.main import cli
from poppy.setup import claude_code as setup

CLIENTS = ["claude-code", "cursor", "codex"]


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("CLAUDE_CONFIG_DIR", "CURSOR_HOME", "CODEX_HOME", "CLAUDE_CODE_PLUGIN_CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("POPPY_LAUNCH_AGENTS_DIR", str(tmp_path / "services"))
    monkeypatch.setenv("POPPY_SYSTEMD_USER_DIR", str(tmp_path / "services"))
    monkeypatch.setattr(setup, "get_poppy_executable", lambda **_: "/test/bin/poppy")


def paths(home, client):
    directory = home / {"claude-code": ".claude", "cursor": ".cursor", "codex": ".codex"}[client]
    config = {
        "claude-code": home / ".claude.json",
        "cursor": directory / "mcp.json",
        "codex": directory / "config.toml",
    }[client]
    hooks = directory / ("settings.json" if client == "claude-code" else "hooks.json")
    primer = directory / ("AGENTS.md" if client == "codex" else "CLAUDE.md")
    return config, hooks, primer


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def snapshot(home):
    return {
        str(path.relative_to(home)): ("link", os.readlink(path)) if path.is_symlink() else path.read_bytes()
        for path in home.rglob("*")
        if path.is_symlink() or path.is_file()
    }


def remove(client, *options):
    return CliRunner().invoke(cli, ["setup", client, "--remove", *options])


@pytest.mark.parametrize("client", CLIENTS)
@pytest.mark.parametrize("user_settings", [False, True])
def test_install_remove_round_trip(tmp_path, client, user_settings):
    config, hooks, primer = paths(tmp_path, client)
    original_config = {"mcpServers": {"other": {"command": "keep"}}, "theme": "dark"} if user_settings else {}
    user_hook = {"command": "echo keep"}
    event = "sessionStart" if client == "cursor" else "SessionStart"
    group = user_hook if client == "cursor" else {"matcher": "user", "hooks": [user_hook]}
    original_hooks = {"hooks": {event: [group]}, "user": True} if user_settings else {}
    if client == "codex":
        original_toml = (
            '# Keep this comment\nmodel = "keep"\n\n[mcp_servers.other]\ncommand = "keep"\n' if user_settings else ""
        )
        write(config, original_toml)
    else:
        write(config, json.dumps(original_config, indent=2))
    write(hooks, json.dumps(original_hooks, indent=2))

    setup.install_for_client(client=client)
    result = remove(client)

    assert result.exit_code == 0, result.output
    assert "Removed" in result.output
    assert str(config) in result.output
    assert str(hooks) in result.output
    if client == "codex":
        # TOML setup adds a separator after an existing table. tomlkit keeps
        # that whitespace with the user's table when Poppy's table is removed.
        assert config.read_text() == original_toml + ("\n" if user_settings else "")
        assert tomllib.loads(config.read_text()) == tomllib.loads(original_toml)
    else:
        assert json.loads(config.read_text()) == original_config
    expected_hooks = {**original_hooks, "version": 1} if client == "cursor" else original_hooks
    assert json.loads(hooks.read_text()) == expected_hooks
    assert not primer.exists()

    before = snapshot(tmp_path)
    second = remove(client)
    assert second.exit_code == 0, second.output
    assert "Nothing was installed" in second.output
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("client", CLIENTS)
def test_install_remove_from_missing_configs(tmp_path, client):
    config, hooks, primer = paths(tmp_path, client)
    assert not config.exists()
    assert not hooks.exists()
    setup.install_for_client(client=client)

    result = remove(client)

    assert result.exit_code == 0, result.output
    assert (tomllib.loads(config.read_text()) if client == "codex" else json.loads(config.read_text())) == {}
    assert json.loads(hooks.read_text()) == ({"version": 1} if client == "cursor" else {})
    assert not primer.exists()
    # Config files have no creation marker. Keep empty files and the backups
    # rather than guessing that they did not exist before setup.
    assert config.exists()
    assert hooks.exists()


@pytest.mark.parametrize("client", CLIENTS)
def test_remove_never_configured_creates_nothing(tmp_path, monkeypatch, client):
    before = list(tmp_path.rglob("*"))
    monkeypatch.setattr(setup, "_write_text", lambda *a, **kw: pytest.fail("unexpected write"))
    result = remove(client)
    assert result.exit_code == 0, result.output
    assert "Nothing was installed" in result.output
    assert list(tmp_path.rglob("*")) == before


@pytest.mark.parametrize("client", CLIENTS)
def test_second_removal_writes_nothing(tmp_path, monkeypatch, client):
    setup.install_for_client(client=client)
    assert remove(client).exit_code == 0
    before = snapshot(tmp_path)
    monkeypatch.setattr(setup, "_write_text", lambda *a, **kw: pytest.fail("unexpected write"))
    monkeypatch.setattr(setup, "_backup_once", lambda *a, **kw: pytest.fail("unexpected backup"))
    result = remove(client)
    assert result.exit_code == 0, result.output
    assert "Nothing was installed" in result.output
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("client", CLIENTS)
def test_remove_shared_group_and_absolute_legacy_commands(tmp_path, client):
    _, hooks, _ = paths(tmp_path, client)
    user_command = {"command": "poppy-debug hook session-start", "timeout": 20}
    commands = [
        {"command": "poppy hook session-start"},
        {"command": "'/old path/poppy-0.3.3' hook stop"},
        {"command": "POPPY_DIR='/old store' /old/bin/poppy hook pre-tool-use"},
        user_command,
    ]
    group = {"matcher": "older-matcher", "hooks": commands, "timeout": 10}
    empty_group = {"matcher": "empty", "hooks": []}
    initial = {
        "hooks": {
            "sessionStart" if client == "cursor" else "SessionStart": commands
            if client == "cursor"
            else [group, empty_group],
            "userEmpty": [],
        },
        "userSetting": True,
    }
    write(hooks, json.dumps(initial))
    result = remove(client)
    assert result.exit_code == 0, result.output
    expected = [user_command] if client == "cursor" else [{**group, "hooks": [user_command]}, empty_group]
    assert json.loads(hooks.read_text()) == {
        "hooks": {"sessionStart" if client == "cursor" else "SessionStart": expected, "userEmpty": []},
        "userSetting": True,
    }
    assert "hook (" in result.output


@pytest.mark.parametrize("client", CLIENTS)
def test_remove_keeps_existing_empty_containers(tmp_path, monkeypatch, client):
    config, hooks, _ = paths(tmp_path, client)
    write(config, "[mcp_servers]\n" if client == "codex" else '{"mcpServers": {}}')
    write(hooks, '{"hooks": {"userEmpty": []}}')
    before = snapshot(tmp_path)
    monkeypatch.setattr(setup, "_write_text", lambda *a, **kw: pytest.fail("unexpected write"))
    result = remove(client)
    assert result.exit_code == 0, result.output
    assert "Nothing was installed" in result.output
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize(
    ("client", "file_kind"),
    [
        (client, kind)
        for client in CLIENTS
        for kind in ["config", "hooks", "primer"]
        if (client, kind) != ("cursor", "primer")
    ],
)
def test_remove_preserves_symlink(tmp_path, client, file_kind):
    setup.install_for_client(client=client)
    config, hooks, primer = paths(tmp_path, client)
    link = {"config": config, "hooks": hooks, "primer": primer}[file_kind]
    original = link.read_bytes()
    target = tmp_path / "dotfiles" / link.name
    write(target, link.read_text())
    link.unlink()
    link.symlink_to(os.path.relpath(target, link.parent))
    link_text = os.readlink(link)

    result = remove(client)
    assert result.exit_code == 0, result.output
    assert link.is_symlink()
    assert os.readlink(link) == link_text
    assert target.read_bytes() != original
    assert link.with_name(link.name + setup.PREVIOUS_CONTENT_BACKUP_SUFFIX).read_bytes() == original
    assert str(link) in result.output


@pytest.mark.parametrize(
    ("client", "file_kind"),
    [
        (client, kind)
        for client in CLIENTS
        for kind in ["config", "hooks", "primer", "legacy"]
        if (kind != "primer" or client != "cursor") and (kind != "legacy" or client == "codex")
    ],
)
@pytest.mark.parametrize("broken_link", [False, True])
def test_remove_preflight_refuses_before_any_changes(tmp_path, client, file_kind, broken_link):
    setup.install_for_client(client=client)
    config, hooks, primer = paths(tmp_path, client)
    path = {"config": config, "hooks": hooks, "primer": primer, "legacy": config.with_suffix(".json")}[file_kind]
    path.unlink(missing_ok=True)
    if broken_link:
        path.symlink_to("missing-target")
    elif file_kind == "primer":
        path.write_bytes(b"\xff")
    else:
        path.write_text("{ broken config")
    before = snapshot(tmp_path)

    result = remove(client)
    assert result.exit_code == 1, result.output
    assert result.output.startswith("Error: Refusing to ")
    assert str(path) in result.output
    assert len(result.output.splitlines()) == 1
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("client", CLIENTS)
@pytest.mark.parametrize("flag", ["--daemon", "--hooks", "--no-hooks", "--claude-md", "--no-claude-md", "--yes"])
def test_remove_rejects_install_flags(tmp_path, client, flag):
    result = remove(client, flag)
    assert result.exit_code == 2
    if client == "claude-code" or flag in {"--daemon", "--hooks", "--no-hooks"}:
        assert "Error: --remove cannot be combined with" in result.output
    else:
        assert f"Error: No such option '{flag}'" in result.output
    assert list(tmp_path.rglob("*")) == []


@pytest.mark.parametrize("client", ["claude-code", "codex"])
def test_remove_primer_preserves_user_text(tmp_path, client):
    _, _, primer = paths(tmp_path, client)
    original = "# My instructions\n\n"
    write(primer, original)
    setup.install_for_client(client=client)
    result = remove(client)
    assert result.exit_code == 0, result.output
    # The existing helper leaves the newline after the closing marker.
    assert primer.read_text() == original + "\n"
    assert "managed primer block" in result.output


def test_codex_remove_legacy_registration(tmp_path):
    config, _, _ = paths(tmp_path, "codex")
    legacy = config.with_suffix(".json")
    write(legacy, json.dumps({"mcpServers": {"poppy": {}, "other": {"command": "keep"}}, "user": True}))
    result = remove("codex")
    assert result.exit_code == 0, result.output
    assert json.loads(legacy.read_text()) == {"mcpServers": {"other": {"command": "keep"}}, "user": True}
    assert str(legacy) in result.output
    assert not config.exists()


def test_codex_remove_retains_explicit_empty_table_and_comments(tmp_path):
    config, _, _ = paths(tmp_path, "codex")
    original = '# User comment\nmodel = "keep"\n\n[mcp_servers] # My servers\n# Keep this table\n\n'
    write(config, original)
    setup.install_for_client(client="codex")
    result = remove("codex")
    assert result.exit_code == 0, result.output
    assert config.read_text() == original


@pytest.mark.parametrize("client", CLIENTS)
@pytest.mark.parametrize("invalid", ["[]", '{"mcpServers": []}', '{"hooks": []}'])
def test_remove_refuses_invalid_structure(tmp_path, client, invalid):
    config, hooks, _ = paths(tmp_path, client)
    setup.install_for_client(client=client)
    if invalid == '{"mcpServers": []}' and client == "codex":
        write(config, "mcp_servers = []\n")
    else:
        write(
            hooks if invalid == '{"hooks": []}' else (config.with_suffix(".json") if client == "codex" else config),
            invalid,
        )
    before = snapshot(tmp_path)
    result = remove(client)
    assert result.exit_code == 1, result.output
    assert result.output.startswith("Error: Refusing to overwrite invalid")
    assert len(result.output.splitlines()) == 1
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("client", CLIENTS)
def test_remove_honors_client_home_override(tmp_path, monkeypatch, client):
    override = tmp_path / "custom-client"
    variable = {"claude-code": "CLAUDE_CONFIG_DIR", "cursor": "CURSOR_HOME", "codex": "CODEX_HOME"}[client]
    monkeypatch.setenv(variable, str(override))
    setup.install_for_client(client=client)
    result = remove(client)
    assert result.exit_code == 0, result.output
    assert str(override) in result.output
    assert not setup.is_mcp_installed(client=client)
    assert not setup.poppy_hook_commands_in(
        json.loads((override / ("settings.json" if client == "claude-code" else "hooks.json")).read_text())
    )
    assert not (tmp_path / ".claude").exists()
    assert not (tmp_path / ".cursor").exists()
    assert not (tmp_path / ".codex").exists()


@pytest.mark.parametrize("with_setup", [False, True])
def test_remove_with_plugin_keeps_plugin_and_prints_uninstall(tmp_path, with_setup):
    if with_setup:
        setup.install_for_client(client="claude-code")
    plugin = tmp_path / ".claude" / "plugins" / "installed_plugins.json"
    plugin_text = json.dumps({"plugins": {"poppy@trags-garden": [{"scope": "user", "installPath": "/keep"}]}})
    write(plugin, plugin_text)
    result = remove("claude-code")
    assert result.exit_code == 0, result.output
    assert "claude plugin uninstall poppy@trags-garden" in result.output
    assert plugin.read_text() == plugin_text
    assert ("Removed" if with_setup else "Nothing was installed") in result.output


@pytest.mark.parametrize("client", CLIENTS)
def test_remove_does_not_touch_store_consent_or_daemon(tmp_path, monkeypatch, client):
    from poppy.mcp_server.lifecycle import agent_path

    setup.install_for_client(client=client)
    service = agent_path(platform="darwin")
    write(service, "keep service")
    monkeypatch.setattr("poppy.mcp_server.lifecycle.agent_path", lambda: service)
    store = tmp_path / ".poppy"
    write(store / "config.json", '{"capture_consent": "granted", "trags_api_key": "keep"}')
    write(store / "memories.db", "keep memories")
    monkeypatch.setenv("POPPY_DIR", str(store))
    monkeypatch.setattr("poppy.cli.main.load_config", lambda *a: pytest.fail("opened Poppy config"))
    monkeypatch.setattr("poppy.cli.main._record_agent_setup", lambda *a, **kw: pytest.fail("recorded installation"))
    before = snapshot(store)
    result = remove(client)
    assert result.exit_code == 0, result.output
    assert (
        result.output.splitlines()[-1]
        == "The daemon service is still installed. Remove it with `poppy daemon uninstall`."
    )
    assert service.read_text() == "keep service"
    assert snapshot(store) == before


@pytest.mark.parametrize("client", CLIENTS)
def test_remove_skips_telemetry_prompt_and_update_notice(tmp_path, monkeypatch, client):
    monkeypatch.setattr(
        "poppy.cli.main.telemetry.maybe_prompt_for_consent", lambda *a: pytest.fail("prompted for consent")
    )
    monkeypatch.setattr(
        "poppy.cli.main.update_check.maybe_print_update_notice", lambda *a: pytest.fail("opened update cache")
    )
    result = remove(client)
    assert result.exit_code == 0, result.output
    assert "Nothing was installed" in result.output
    assert list(tmp_path.rglob("*")) == []


def test_remove_reports_a_filesystem_error_in_one_line(monkeypatch):
    from click.testing import CliRunner

    from poppy.cli.main import cli
    from poppy.setup import claude_code

    def refuse(**_kwargs):
        raise PermissionError(13, "Permission denied", "settings.json")

    monkeypatch.setattr(claude_code, "remove_for_client", refuse)
    result = CliRunner().invoke(cli, ["setup", "codex", "--remove"])

    assert result.exit_code == 1
    assert "Could not finish removing Poppy from codex" in result.output
    assert not isinstance(result.exception, PermissionError)
