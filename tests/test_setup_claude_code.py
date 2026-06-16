import json

from poppy.setup.claude_code import (
    CLAUDE_MD_BEGIN,
    CLAUDE_MD_END,
    install_claude_md_block,
    install_for_client,
    install_mcp_config,
    install_pre_tool_use_hook,
    install_session_end_hook,
    install_session_start_hook,
    install_user_prompt_submit_hook,
    is_hook_installed,
    is_mcp_installed,
    managed_claude_md_present,
    remove_legacy_hooks,
)


def test_install_mcp_config_claude_code(tmp_path):
    # Claude Code stores MCP server registrations in ~/.claude.json (sibling
    # of the ~/.claude directory), not in ~/.claude/settings.json.
    install_mcp_config(claude_config_dir=tmp_path, client="claude-code")
    settings = json.loads((tmp_path / ".claude.json").read_text())
    assert "poppy" in settings["mcpServers"]
    assert settings["mcpServers"]["poppy"]["command"] == "poppy"
    assert settings["mcpServers"]["poppy"]["args"] == ["serve", "--source", "claude-code"]
    assert is_mcp_installed(tmp_path, client="claude-code")


def test_install_session_start_hook(tmp_path):
    install_session_start_hook(claude_config_dir=tmp_path)
    settings = json.loads((tmp_path / "settings.json").read_text())
    hooks = settings["hooks"]["SessionStart"]
    assert any(h.get("command") == "poppy hook session-start" for group in hooks for h in group.get("hooks", []))
    assert is_hook_installed(tmp_path, "SessionStart")


def test_install_session_end_hook(tmp_path):
    install_session_end_hook(claude_config_dir=tmp_path)
    assert is_hook_installed(tmp_path, "SessionEnd")
    settings = json.loads((tmp_path / "settings.json").read_text())
    hooks = settings["hooks"]["SessionEnd"]
    assert any(h.get("command") == "poppy hook session-end" for group in hooks for h in group.get("hooks", []))


def test_remove_legacy_stop_hook_preserves_user_hooks(tmp_path):
    settings = {
        "hooks": {
            "Stop": [
                {
                    "matcher": "",
                    "hooks": [
                        {"type": "command", "command": "poppy hook stop"},
                        {"type": "command", "command": "user's other hook"},
                    ],
                }
            ]
        }
    }
    (tmp_path / "settings.json").write_text(json.dumps(settings))
    removed = remove_legacy_hooks(tmp_path)
    assert "Stop:poppy hook stop" in removed

    final = json.loads((tmp_path / "settings.json").read_text())
    stop_cmds = [h.get("command") for g in final.get("hooks", {}).get("Stop", []) for h in g.get("hooks", [])]
    assert "poppy hook stop" not in stop_cmds
    assert "user's other hook" in stop_cmds


def test_install_hook_idempotent(tmp_path):
    install_session_start_hook(claude_config_dir=tmp_path)
    install_session_start_hook(claude_config_dir=tmp_path)
    settings = json.loads((tmp_path / "settings.json").read_text())
    matching = [
        h
        for group in settings["hooks"]["SessionStart"]
        for h in group.get("hooks", [])
        if h.get("command") == "poppy hook session-start"
    ]
    assert len(matching) == 1


def test_install_claude_md_block_creates_file(tmp_path):
    install_claude_md_block(claude_config_dir=tmp_path)
    text = (tmp_path / "CLAUDE.md").read_text()
    assert CLAUDE_MD_BEGIN in text
    assert CLAUDE_MD_END in text
    assert "Poppy memory" in text
    assert managed_claude_md_present(tmp_path)


def test_install_claude_md_block_preserves_existing(tmp_path):
    md = tmp_path / "CLAUDE.md"
    md.write_text("# Existing CLAUDE.md\n\nUser content here.\n")
    install_claude_md_block(claude_config_dir=tmp_path)
    text = md.read_text()
    assert "User content here." in text
    assert CLAUDE_MD_BEGIN in text


def test_install_claude_md_block_replaces_old_block(tmp_path):
    md = tmp_path / "CLAUDE.md"
    md.write_text(f"prefix\n{CLAUDE_MD_BEGIN}\nstale content\n{CLAUDE_MD_END}\nsuffix\n")
    install_claude_md_block(claude_config_dir=tmp_path)
    text = md.read_text()
    assert "prefix" in text
    assert "suffix" in text
    assert "stale content" not in text
    assert "Poppy memory" in text


def test_install_for_client_claude_code_full(tmp_path):
    paths = install_for_client(client="claude-code", claude_config_dir=tmp_path)
    assert "MCP config" in paths
    assert "SessionStart hook" in paths
    assert "UserPromptSubmit hook" in paths
    assert "PreToolUse hook" in paths
    assert "SessionEnd hook" in paths
    assert "CLAUDE.md block" in paths


def test_install_user_prompt_submit_hook(tmp_path):
    install_user_prompt_submit_hook(claude_config_dir=tmp_path)
    settings = json.loads((tmp_path / "settings.json").read_text())
    hooks = settings["hooks"]["UserPromptSubmit"]
    assert any(h.get("command") == "poppy hook user-prompt-submit" for group in hooks for h in group.get("hooks", []))
    assert is_hook_installed(tmp_path, "UserPromptSubmit")


def test_install_pre_tool_use_hook_has_matcher(tmp_path):
    install_pre_tool_use_hook(claude_config_dir=tmp_path)
    settings = json.loads((tmp_path / "settings.json").read_text())
    groups = settings["hooks"]["PreToolUse"]
    # The PreToolUse hook must scope to Edit|Write|MultiEdit, not fire on every tool call.
    matcher_for_poppy = next(
        g.get("matcher") for g in groups if any(h.get("command") == "poppy hook pre-tool-use" for h in g["hooks"])
    )
    assert matcher_for_poppy == "Edit|Write|MultiEdit"
    assert is_hook_installed(tmp_path, "PreToolUse")


def test_install_for_client_no_hooks(tmp_path):
    paths = install_for_client(client="claude-code", claude_config_dir=tmp_path, install_hooks=False)
    assert "MCP config" in paths
    assert "SessionStart hook" not in paths


def test_install_for_client_cursor(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    # Cursor writes to ~/.cursor/mcp.json regardless of claude_config_dir
    install_for_client(client="cursor", claude_config_dir=tmp_path)
    cursor_config = tmp_path / ".cursor" / "mcp.json"
    assert cursor_config.exists()
    settings = json.loads(cursor_config.read_text())
    assert "poppy" in settings["mcpServers"]


def test_install_for_client_copilot_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    install_for_client(client="copilot-cli", claude_config_dir=tmp_path)
    config = tmp_path / ".copilot" / "mcp-config.json"
    assert config.exists()
    settings = json.loads(config.read_text())
    assert settings["mcpServers"]["poppy"]["command"] == "poppy"
    assert settings["mcpServers"]["poppy"]["args"] == ["serve", "--source", "copilot-cli"]


def test_install_copilot_cli_preserves_existing_servers(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = tmp_path / ".copilot" / "mcp-config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"mcpServers": {"playwright": {"command": "playwright-mcp"}}}))
    install_for_client(client="copilot-cli", claude_config_dir=tmp_path)
    settings = json.loads(config.read_text())
    assert settings["mcpServers"]["playwright"]["command"] == "playwright-mcp"
    assert settings["mcpServers"]["poppy"]["command"] == "poppy"


def test_install_for_client_pi(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    install_for_client(client="pi", claude_config_dir=tmp_path)
    config = tmp_path / ".pi" / "agent" / "mcp.json"
    assert config.exists()
    settings = json.loads(config.read_text())
    assert settings["mcpServers"]["poppy"]["command"] == "poppy"
    assert settings["mcpServers"]["poppy"]["args"] == ["serve", "--source", "pi"]


def test_install_copilot_cli_writes_primer(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    paths = install_for_client(client="copilot-cli", claude_config_dir=tmp_path)
    primer = tmp_path / ".copilot" / "AGENTS.md"
    assert paths["Primer (AGENTS.md)"] == primer
    assert primer.exists()
    text = primer.read_text()
    assert CLAUDE_MD_BEGIN in text
    assert CLAUDE_MD_END in text
    assert "Poppy memory" in text


def test_install_pi_writes_primer(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    paths = install_for_client(client="pi", claude_config_dir=tmp_path)
    primer = tmp_path / ".pi" / "AGENTS.md"
    assert paths["Primer (AGENTS.md)"] == primer
    assert primer.exists()
    assert "Poppy memory" in primer.read_text()


def test_primer_block_preserves_existing_agents_md(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    agents = tmp_path / ".copilot" / "AGENTS.md"
    agents.parent.mkdir(parents=True)
    agents.write_text("# Existing instructions\n\nMy own rules here.\n")
    install_for_client(client="copilot-cli", claude_config_dir=tmp_path)
    text = agents.read_text()
    assert "My own rules here." in text
    assert CLAUDE_MD_BEGIN in text


def test_primer_block_replaces_stale_block(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    agents = tmp_path / ".pi" / "AGENTS.md"
    agents.parent.mkdir(parents=True)
    agents.write_text(f"prefix\n{CLAUDE_MD_BEGIN}\nstale\n{CLAUDE_MD_END}\nsuffix\n")
    install_for_client(client="pi", claude_config_dir=tmp_path)
    text = agents.read_text()
    assert "prefix" in text
    assert "suffix" in text
    assert "stale" not in text
    assert "Poppy memory" in text


# ---------- claude-desktop integration ----------


def test_install_for_client_claude_desktop_writes_config(tmp_path, monkeypatch):
    target = tmp_path / "Claude" / "claude_desktop_config.json"
    monkeypatch.setenv("POPPY_CLAUDE_DESKTOP_CONFIG", str(target))

    paths = install_for_client(client="claude-desktop")
    assert paths["MCP config"] == target
    settings = json.loads(target.read_text())
    assert settings["mcpServers"]["poppy"]["command"] == "poppy"
    assert settings["mcpServers"]["poppy"]["args"] == ["serve", "--source", "claude-desktop"]
    assert is_mcp_installed(client="claude-desktop")


def test_install_claude_desktop_backs_up_existing_config(tmp_path, monkeypatch):
    target = tmp_path / "claude_desktop_config.json"
    target.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}, "userField": 42}))
    monkeypatch.setenv("POPPY_CLAUDE_DESKTOP_CONFIG", str(target))

    paths = install_for_client(client="claude-desktop")
    backup = target.with_name(target.name + ".pre-poppy.bak")
    assert paths["backup"] == backup
    assert backup.exists()
    # Backup must be the byte-identical pre-merge config.
    assert json.loads(backup.read_text()) == {"mcpServers": {"other": {"command": "x"}}, "userField": 42}
    # Merged config preserves the user's other entries.
    merged = json.loads(target.read_text())
    assert merged["userField"] == 42
    assert merged["mcpServers"]["other"]["command"] == "x"
    assert merged["mcpServers"]["poppy"]["args"] == ["serve", "--source", "claude-desktop"]


def test_install_claude_desktop_backup_is_idempotent(tmp_path, monkeypatch):
    target = tmp_path / "claude_desktop_config.json"
    original = {"mcpServers": {}, "marker": "v1"}
    target.write_text(json.dumps(original))
    monkeypatch.setenv("POPPY_CLAUDE_DESKTOP_CONFIG", str(target))

    install_for_client(client="claude-desktop")
    # Mutate target to ensure a second backup, if it ran, would clobber.
    target.write_text(json.dumps({"mcpServers": {"poppy": {"command": "poppy"}}, "marker": "v2"}))

    paths = install_for_client(client="claude-desktop")
    # Second run finds the .bak already present and does not re-backup.
    assert "backup" not in paths
    backup = target.with_name(target.name + ".pre-poppy.bak")
    assert json.loads(backup.read_text()) == original


def test_install_claude_desktop_no_backup_when_absent(tmp_path, monkeypatch):
    target = tmp_path / "claude_desktop_config.json"
    monkeypatch.setenv("POPPY_CLAUDE_DESKTOP_CONFIG", str(target))

    paths = install_for_client(client="claude-desktop")
    assert "backup" not in paths
    assert not target.with_name(target.name + ".pre-poppy.bak").exists()
    assert target.exists()


def test_get_claude_desktop_config_path_env_override(tmp_path, monkeypatch):
    from poppy.setup.claude_code import get_claude_desktop_config_path

    monkeypatch.setenv("POPPY_CLAUDE_DESKTOP_CONFIG", str(tmp_path / "x.json"))
    assert get_claude_desktop_config_path() == tmp_path / "x.json"


def test_get_claude_desktop_config_path_macos_default(monkeypatch, tmp_path):
    from poppy.setup.claude_code import get_claude_desktop_config_path

    monkeypatch.delenv("POPPY_CLAUDE_DESKTOP_CONFIG", raising=False)
    monkeypatch.setattr("os.name", "posix")
    monkeypatch.setenv("HOME", str(tmp_path))
    expected = tmp_path / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    assert get_claude_desktop_config_path() == expected
