import errno
import json
import os
import shutil
import stat
import subprocess
import sys
import tomllib

import pytest
from click.testing import CliRunner

from poppy.cli.main import cli
from poppy.setup.claude_code import (
    CLAUDE_MD_BEGIN,
    CLAUDE_MD_END,
    CURSOR_DOCUMENTED_HOOK_EVENTS,
    CorruptConfigError,
    get_poppy_executable,
    install_claude_md_block,
    install_codex_hooks,
    install_cursor_hooks,
    install_for_client,
    install_mcp_config,
    install_post_compact_hook,
    install_pre_tool_use_hook,
    install_session_end_hook,
    install_session_start_hook,
    install_user_prompt_submit_hook,
    is_codex_hooks_installed,
    is_cursor_hooks_installed,
    is_hook_installed,
    is_mcp_installed,
    managed_claude_md_present,
    remove_legacy_hooks,
)


@pytest.fixture(autouse=True)
def deterministic_poppy_path(tmp_path, monkeypatch):
    executable = tmp_path / "resolved-bin" / "poppy"
    monkeypatch.setattr("poppy.setup.claude_code.shutil.which", lambda _name: str(executable))


@pytest.fixture(autouse=True)
def isolate_client_home_overrides(monkeypatch):
    monkeypatch.delenv("COPILOT_HOME", raising=False)
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("CURSOR_HOME", raising=False)


def expected_poppy_path(tmp_path):
    return str((tmp_path / "resolved-bin" / "poppy").resolve())


def test_get_poppy_executable_prefers_path_hit(tmp_path, monkeypatch):
    executable = tmp_path / "path-bin" / "poppy"
    monkeypatch.setattr("poppy.setup.claude_code.shutil.which", lambda _name: str(executable))

    assert get_poppy_executable() == str(executable.resolve())


def test_get_poppy_executable_falls_back_to_python_sibling(tmp_path, monkeypatch):
    python = tmp_path / "venv" / "bin" / "python"
    sibling = python.parent / "poppy"
    sibling.parent.mkdir(parents=True)
    sibling.touch()
    monkeypatch.setattr("poppy.setup.claude_code.shutil.which", lambda _name: None)
    monkeypatch.setattr("poppy.setup.claude_code.sys.executable", str(python))

    assert get_poppy_executable() == str(sibling.resolve())


def test_get_poppy_executable_returns_bare_command_when_unresolved(tmp_path, monkeypatch):
    monkeypatch.setattr("poppy.setup.claude_code.shutil.which", lambda _name: None)
    monkeypatch.setattr("poppy.setup.claude_code.sys.executable", str(tmp_path / "venv" / "bin" / "python"))

    assert get_poppy_executable() == "poppy"


def test_install_mcp_config_claude_code(tmp_path):
    # Claude Code stores MCP server registrations in ~/.claude.json (sibling
    # of the ~/.claude directory), not in ~/.claude/settings.json.
    install_mcp_config(claude_config_dir=tmp_path, client="claude-code")
    settings = json.loads((tmp_path / ".claude.json").read_text())
    assert "poppy" in settings["mcpServers"]
    assert settings["mcpServers"]["poppy"]["command"] == expected_poppy_path(tmp_path)
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


def test_install_post_compact_hook(tmp_path):
    install_post_compact_hook(claude_config_dir=tmp_path)
    assert is_hook_installed(tmp_path, "PostCompact")
    settings = json.loads((tmp_path / "settings.json").read_text())
    hooks = settings["hooks"]["PostCompact"]
    assert any(h.get("command") == "poppy hook post-compact" for group in hooks for h in group.get("hooks", []))


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


def test_install_pre_tool_use_hook_migrates_owned_stale_matcher(tmp_path):
    settings_path = tmp_path / "settings.json"
    user_group = {
        "matcher": "Bash",
        "hooks": [{"type": "command", "command": "user's hook"}],
    }
    settings_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Edit|Write|MultiEdit",
                            "hooks": [{"type": "command", "command": "poppy hook pre-tool-use"}],
                        },
                        user_group,
                    ]
                }
            }
        )
    )

    install_pre_tool_use_hook(tmp_path)
    migrated = json.loads(settings_path.read_text())
    assert migrated["hooks"]["PreToolUse"][0]["matcher"] == "Edit|Write|NotebookEdit"
    assert migrated["hooks"]["PreToolUse"][1] == user_group

    after_migration = settings_path.read_text()
    install_pre_tool_use_hook(tmp_path)
    assert settings_path.read_text() == after_migration


def test_install_pre_tool_use_hook_leaves_mixed_user_group_untouched(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Edit|Write|MultiEdit",
                    "hooks": [
                        {"type": "command", "command": "poppy hook pre-tool-use"},
                        {"type": "command", "command": "user's other hook"},
                    ],
                }
            ]
        }
    }
    settings_path.write_text(json.dumps(settings))

    install_pre_tool_use_hook(tmp_path)

    assert json.loads(settings_path.read_text()) == settings


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
    assert "PostCompact hook" in paths
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
    # The PreToolUse hook must scope to write-capable tools, not fire on every tool call.
    matcher_for_poppy = next(
        g.get("matcher") for g in groups if any(h.get("command") == "poppy hook pre-tool-use" for h in g["hooks"])
    )
    assert matcher_for_poppy == "Edit|Write|NotebookEdit"
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
    assert (tmp_path / ".cursor" / "hooks.json").exists()


def test_install_cursor_hooks_fresh_native_shape_and_allowlist(tmp_path):
    hooks_path = install_cursor_hooks(tmp_path)
    settings = json.loads(hooks_path.read_text())

    assert settings["version"] == 1
    assert set(settings["hooks"]) == {
        "sessionStart",
        "beforeSubmitPrompt",
        "preToolUse",
        "sessionEnd",
        "preCompact",
    }
    assert set(settings["hooks"]) <= CURSOR_DOCUMENTED_HOOK_EVENTS
    assert all(set(entry) == {"command"} for entries in settings["hooks"].values() for entry in entries)
    assert is_cursor_hooks_installed(tmp_path)


def test_install_cursor_hooks_merge_preserves_user_content_and_is_idempotent(tmp_path):
    hooks_path = tmp_path / "hooks.json"
    original_user_entry = {"command": "user-hook", "timeout": 17, "matcher": "Write"}
    unknown_user_value = [{"command": "future-user-hook", "custom": {"keep": True}}]
    hooks_path.write_text(
        json.dumps(
            {
                "version": 1,
                "userTopLevel": {"keep": True},
                "hooks": {
                    "preToolUse": [
                        original_user_entry,
                        {"command": "poppy hook pre-tool-use"},
                        {"command": "poppy hook pre-tool-use"},
                    ],
                    "futureEvent": unknown_user_value,
                },
            }
        )
    )

    install_cursor_hooks(tmp_path)
    first = hooks_path.read_text()
    install_cursor_hooks(tmp_path)
    assert hooks_path.read_text() == first

    merged = json.loads(first)
    assert merged["userTopLevel"] == {"keep": True}
    assert merged["hooks"]["futureEvent"] == unknown_user_value
    assert original_user_entry in merged["hooks"]["preToolUse"]
    assert [entry["command"] for entry in merged["hooks"]["preToolUse"]].count("poppy hook pre-tool-use") == 1


def test_install_cursor_hooks_backs_up_malformed_json_and_writes_fresh(tmp_path):
    hooks_path = tmp_path / "hooks.json"
    hooks_path.write_text("{malformed")

    install_cursor_hooks(tmp_path)

    assert (tmp_path / "hooks.json.bak").read_text() == "{malformed"
    assert is_cursor_hooks_installed(tmp_path)


def test_install_cursor_hooks_backs_up_non_list_owned_event_and_writes_fresh(tmp_path, capsys):
    hooks_path = tmp_path / "hooks.json"
    hooks_path.write_text(json.dumps({"userTopLevel": "discard malformed file", "hooks": {"preToolUse": {}}}))

    install_cursor_hooks(tmp_path)

    captured = capsys.readouterr()
    assert "non-list values for Poppy-owned hook events: preToolUse" in captured.err
    assert str(tmp_path / "hooks.json.bak") in captured.err
    assert json.loads((tmp_path / "hooks.json.bak").read_text())["userTopLevel"] == "discard malformed file"
    assert "userTopLevel" not in json.loads(hooks_path.read_text())
    assert is_cursor_hooks_installed(tmp_path)


def test_install_cursor_hooks_rotates_backups_without_clobbering(tmp_path):
    hooks_path = tmp_path / "hooks.json"
    hooks_path.write_text("{first malformed")
    install_cursor_hooks(tmp_path)
    hooks_path.write_text("{second malformed")

    install_cursor_hooks(tmp_path)

    assert (tmp_path / "hooks.json.bak").read_text() == "{first malformed"
    assert (tmp_path / "hooks.json.bak-1").read_text() == "{second malformed"


def test_install_cursor_no_hooks(tmp_path, monkeypatch):
    cursor_home = tmp_path / "cursor-home"
    monkeypatch.setenv("CURSOR_HOME", str(cursor_home))

    paths = install_for_client(client="cursor", install_hooks=False)

    assert "Cursor hooks" not in paths
    assert not (cursor_home / "hooks.json").exists()


def test_install_codex_writes_toml_and_primer_idempotently(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert not is_mcp_installed(client="codex")

    paths = install_for_client(client="codex")
    config = tmp_path / ".codex" / "config.toml"
    primer = tmp_path / ".codex" / "AGENTS.md"
    assert paths["MCP config"] == config
    assert paths["Primer (AGENTS.md)"] == primer
    assert paths["Codex hooks"] == tmp_path / ".codex" / "hooks.json"
    assert tomllib.loads(config.read_text())["mcp_servers"]["poppy"] == {
        "command": expected_poppy_path(tmp_path),
        "args": ["serve", "--source", "codex"],
    }
    assert is_mcp_installed(client="codex")
    assert is_codex_hooks_installed()

    install_for_client(client="codex")
    primer_text = primer.read_text()
    assert primer_text.count(CLAUDE_MD_BEGIN) == 1
    assert primer_text.count(CLAUDE_MD_END) == 1


def test_install_codex_hooks_merges_migrates_and_is_idempotent(tmp_path):
    hooks_path = tmp_path / "hooks.json"
    hooks_path.write_text(
        json.dumps(
            {
                "unrelated": {"keep": True},
                "hooks": {
                    "SessionStart": [
                        {"matcher": "", "hooks": [{"type": "command", "command": "poppy hook session-start"}]},
                        {"matcher": "", "hooks": [{"type": "command", "command": "user start hook"}]},
                    ],
                    "PreToolUse": [
                        {
                            "matcher": "Edit|Write|NotebookEdit",
                            "hooks": [
                                {"type": "command", "command": "poppy hook pre-tool-use"},
                                {"type": "command", "command": "user edit hook"},
                            ],
                        }
                    ],
                    "SessionEnd": [
                        {
                            "matcher": "",
                            "hooks": [
                                {"type": "command", "command": "poppy hook session-end"},
                                {"type": "command", "command": "user end hook"},
                            ],
                        }
                    ],
                    "PostCompact": [
                        {"matcher": "", "hooks": [{"type": "command", "command": "poppy hook post-compact"}]}
                    ],
                },
            }
        )
    )

    assert install_codex_hooks(tmp_path) == hooks_path
    first = hooks_path.read_text()
    assert is_codex_hooks_installed(tmp_path)
    install_codex_hooks(tmp_path)
    assert hooks_path.read_text() == first

    settings = json.loads(first)
    assert settings["unrelated"] == {"keep": True}
    assert "PostCompact" not in settings["hooks"]
    commands = {
        event: [h["command"] for group in groups for h in group.get("hooks", [])]
        for event, groups in settings["hooks"].items()
    }
    assert commands["SessionStart"].count("poppy hook session-start") == 1
    assert "user start hook" in commands["SessionStart"]
    assert "poppy hook session-end" not in commands["SessionEnd"]
    assert "user end hook" in commands["SessionEnd"]
    poppy_patch_group = next(
        group
        for group in settings["hooks"]["PreToolUse"]
        if any(h.get("command") == "poppy hook pre-tool-use" for h in group["hooks"])
    )
    assert poppy_patch_group["matcher"] == "apply_patch"
    assert commands["Stop"] == ["poppy hook stop"]


def test_install_codex_no_hooks(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))

    paths = install_for_client(client="codex", install_hooks=False)

    assert "Codex hooks" not in paths
    assert not (tmp_path / "codex-home" / "hooks.json").exists()


def test_install_codex_preserves_toml_content_comments_and_backup(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = tmp_path / ".codex" / "config.toml"
    config.parent.mkdir(parents=True)
    original = '# keep this comment\nmodel = "gpt-test"\n\n[mcp_servers.other]\ncommand = "other"\n'
    config.write_text(original)

    paths = install_for_client(client="codex")
    backup = config.with_name(config.name + ".pre-poppy.bak")
    assert paths["backup"] == backup
    assert backup.read_text() == original
    merged = config.read_text()
    assert "# keep this comment" in merged
    assert tomllib.loads(merged)["mcp_servers"]["other"]["command"] == "other"

    config.write_text(config.read_text().replace('model = "gpt-test"', 'model = "gpt-new"'))
    second_paths = install_for_client(client="codex")
    assert second_paths["backup"] == config.with_name(config.name + ".pre-poppy.bak-1")
    assert backup.read_text() == original
    assert 'model = "gpt-new"' in second_paths["backup"].read_text()


def test_install_codex_aborts_on_corrupt_toml_without_modifying_it(tmp_path, monkeypatch):
    from poppy.setup.claude_code import CorruptConfigError

    monkeypatch.setenv("HOME", str(tmp_path))
    config = tmp_path / ".codex" / "config.toml"
    config.parent.mkdir(parents=True)
    corrupt = "[mcp_servers.poppy\ncommand = nope"
    config.write_text(corrupt)

    with pytest.raises(CorruptConfigError):
        install_for_client(client="codex")
    assert config.read_text() == corrupt
    assert not config.with_name(config.name + ".pre-poppy.bak").exists()


def test_install_codex_migrates_only_legacy_poppy_json_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    legacy = tmp_path / ".codex" / "config.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "poppy": {"command": "poppy", "type": "stdio"},
                    "other": {"command": "other"},
                },
                "unrelated": {"keep": True},
            }
        )
    )

    paths = install_for_client(client="codex")
    assert paths["Removed legacy MCP config"] == legacy
    migrated = json.loads(legacy.read_text())
    assert "poppy" not in migrated["mcpServers"]
    assert migrated["mcpServers"]["other"] == {"command": "other"}
    assert migrated["unrelated"] == {"keep": True}


def test_install_for_client_copilot_cli(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    install_for_client(client="copilot-cli", claude_config_dir=tmp_path)
    config = tmp_path / ".copilot" / "mcp-config.json"
    assert config.exists()
    settings = json.loads(config.read_text())
    assert settings["mcpServers"]["poppy"]["command"] == expected_poppy_path(tmp_path)
    assert settings["mcpServers"]["poppy"]["args"] == ["serve", "--source", "copilot-cli"]


def test_install_copilot_cli_preserves_existing_servers(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = tmp_path / ".copilot" / "mcp-config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"mcpServers": {"playwright": {"command": "playwright-mcp"}}}))
    install_for_client(client="copilot-cli", claude_config_dir=tmp_path)
    settings = json.loads(config.read_text())
    assert settings["mcpServers"]["playwright"]["command"] == "playwright-mcp"
    assert settings["mcpServers"]["poppy"]["command"] == expected_poppy_path(tmp_path)


def test_install_for_client_pi(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    install_for_client(client="pi", claude_config_dir=tmp_path)
    config = tmp_path / ".pi" / "agent" / "mcp.json"
    assert config.exists()
    settings = json.loads(config.read_text())
    assert settings["mcpServers"]["poppy"]["command"] == expected_poppy_path(tmp_path)
    assert settings["mcpServers"]["poppy"]["args"] == ["serve", "--source", "pi"]


def test_install_copilot_cli_writes_primer(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    paths = install_for_client(client="copilot-cli", claude_config_dir=tmp_path)
    primer = tmp_path / ".copilot" / "copilot-instructions.md"
    assert paths["Primer (copilot-instructions.md)"] == primer
    assert primer.exists()
    text = primer.read_text()
    assert CLAUDE_MD_BEGIN in text
    assert CLAUDE_MD_END in text
    assert "Poppy memory" in text


def test_install_pi_writes_primer(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    paths = install_for_client(client="pi", claude_config_dir=tmp_path)
    primer = tmp_path / ".pi" / "agent" / "AGENTS.md"
    assert paths["Primer (AGENTS.md)"] == primer
    assert primer.exists()
    assert "Poppy memory" in primer.read_text()


def test_primer_block_preserves_existing_agents_md(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    primer = tmp_path / ".copilot" / "copilot-instructions.md"
    primer.parent.mkdir(parents=True)
    primer.write_text("# Existing instructions\n\nMy own rules here.\n")
    install_for_client(client="copilot-cli", claude_config_dir=tmp_path)
    text = primer.read_text()
    assert "My own rules here." in text
    assert CLAUDE_MD_BEGIN in text


def test_primer_block_replaces_stale_block_at_new_pi_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    agents = tmp_path / ".pi" / "agent" / "AGENTS.md"
    agents.parent.mkdir(parents=True)
    agents.write_text(f"prefix\n{CLAUDE_MD_BEGIN}\nstale\n{CLAUDE_MD_END}\nsuffix\n")
    install_for_client(client="pi", claude_config_dir=tmp_path)
    text = agents.read_text()
    assert "prefix" in text
    assert "suffix" in text
    assert "stale" not in text
    assert "Poppy memory" in text


@pytest.mark.parametrize(
    ("client", "legacy_relative"),
    [
        ("copilot-cli", ".copilot/AGENTS.md"),
        ("pi", ".pi/AGENTS.md"),
    ],
)
def test_install_removes_managed_block_from_legacy_primer_preserving_user_content(
    tmp_path, monkeypatch, client, legacy_relative
):
    monkeypatch.setenv("HOME", str(tmp_path))
    legacy = tmp_path / legacy_relative
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(f"# User rules\n{CLAUDE_MD_BEGIN}\nstale\n{CLAUDE_MD_END}\nKeep this.\n")

    install_for_client(client=client)
    install_for_client(client=client)

    text = legacy.read_text()
    assert "# User rules" in text
    assert "Keep this." in text
    assert CLAUDE_MD_BEGIN not in text
    assert CLAUDE_MD_END not in text


@pytest.mark.parametrize(
    ("client", "legacy_relative"),
    [
        ("copilot-cli", ".copilot/AGENTS.md"),
        ("pi", ".pi/AGENTS.md"),
    ],
)
def test_install_deletes_legacy_primer_containing_only_managed_block(tmp_path, monkeypatch, client, legacy_relative):
    monkeypatch.setenv("HOME", str(tmp_path))
    legacy = tmp_path / legacy_relative
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(f" \n{CLAUDE_MD_BEGIN}\nstale\n{CLAUDE_MD_END}\n\t")

    install_for_client(client=client)

    assert not legacy.exists()


def test_copilot_home_overrides_mcp_and_primer_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    copilot_home = tmp_path / "custom-copilot"
    monkeypatch.setenv("COPILOT_HOME", str(copilot_home))
    legacy = copilot_home / "AGENTS.md"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(f"{CLAUDE_MD_BEGIN}\nstale\n{CLAUDE_MD_END}\n")

    paths = install_for_client(client="copilot-cli")

    assert paths["MCP config"] == copilot_home / "mcp-config.json"
    assert paths["Primer (copilot-instructions.md)"] == copilot_home / "copilot-instructions.md"
    assert (copilot_home / "mcp-config.json").exists()
    assert (copilot_home / "copilot-instructions.md").exists()
    assert not legacy.exists()


def test_pi_agent_dir_overrides_mcp_and_primer_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    agent_dir = tmp_path / "custom-pi-agent"
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))

    paths = install_for_client(client="pi")

    assert paths["MCP config"] == agent_dir / "mcp.json"
    assert paths["Primer (AGENTS.md)"] == agent_dir / "AGENTS.md"
    assert (agent_dir / "mcp.json").exists()
    assert (agent_dir / "AGENTS.md").exists()


# ---------- claude-desktop integration ----------


def test_install_for_client_claude_desktop_writes_config(tmp_path, monkeypatch):
    target = tmp_path / "Claude" / "claude_desktop_config.json"
    monkeypatch.setenv("POPPY_CLAUDE_DESKTOP_CONFIG", str(target))

    paths = install_for_client(client="claude-desktop")
    assert paths["MCP config"] == target
    settings = json.loads(target.read_text())
    assert settings["mcpServers"]["poppy"]["command"] == expected_poppy_path(tmp_path)
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


def test_install_claude_desktop_backup_rotates(tmp_path, monkeypatch):
    target = tmp_path / "claude_desktop_config.json"
    original = {"mcpServers": {}, "marker": "v1"}
    target.write_text(json.dumps(original))
    monkeypatch.setenv("POPPY_CLAUDE_DESKTOP_CONFIG", str(target))

    install_for_client(client="claude-desktop")
    # Mutate target so the second backup represents a distinct pre-merge state.
    target.write_text(json.dumps({"mcpServers": {"poppy": {"command": "poppy"}}, "marker": "v2"}))

    paths = install_for_client(client="claude-desktop")
    backup = target.with_name(target.name + ".pre-poppy.bak")
    rotated = target.with_name(target.name + ".pre-poppy.bak-1")
    assert paths["backup"] == rotated
    assert json.loads(backup.read_text()) == original
    assert json.loads(rotated.read_text())["marker"] == "v2"


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
    # Pin the platform: on a Linux runner the real sys.platform would take the
    # Linux branch and this test would assert the wrong default.
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.setenv("HOME", str(tmp_path))
    expected = tmp_path / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    assert get_claude_desktop_config_path() == expected


def test_get_claude_desktop_config_path_linux_default(monkeypatch, tmp_path):
    from poppy.setup.claude_code import get_claude_desktop_config_path

    monkeypatch.delenv("POPPY_CLAUDE_DESKTOP_CONFIG", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr("os.name", "posix")
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    expected = tmp_path / ".config" / "Claude" / "claude_desktop_config.json"
    assert get_claude_desktop_config_path() == expected


def test_get_claude_desktop_config_path_linux_respects_xdg(monkeypatch, tmp_path):
    from poppy.setup.claude_code import get_claude_desktop_config_path

    monkeypatch.delenv("POPPY_CLAUDE_DESKTOP_CONFIG", raising=False)
    monkeypatch.setattr("os.name", "posix")
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    expected = tmp_path / "xdg" / "Claude" / "claude_desktop_config.json"
    assert get_claude_desktop_config_path() == expected


def test_get_claude_desktop_msix_config_path_detects_virtualized_directory(tmp_path, monkeypatch):
    from poppy.setup.claude_code import get_claude_desktop_msix_config_path

    local_appdata = tmp_path / "LocalAppData"
    virtualized_dir = local_appdata / "Packages" / "Claude_random-id" / "LocalCache" / "Roaming" / "Claude"
    virtualized_dir.mkdir(parents=True)
    monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))

    assert get_claude_desktop_msix_config_path() == virtualized_dir / "claude_desktop_config.json"


def test_get_claude_desktop_msix_config_path_returns_none_without_install(tmp_path, monkeypatch):
    from poppy.setup.claude_code import get_claude_desktop_msix_config_path

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))

    assert get_claude_desktop_msix_config_path() is None


def test_install_claude_desktop_also_merges_msix_config(tmp_path, monkeypatch):
    normal_config = tmp_path / "normal" / "claude_desktop_config.json"
    msix_dir = tmp_path / "LocalAppData" / "Packages" / "Claude_random-id" / "LocalCache" / "Roaming" / "Claude"
    msix_dir.mkdir(parents=True)
    msix_config = msix_dir / "claude_desktop_config.json"
    msix_config.write_text(json.dumps({"mcpServers": {"other": {"command": "other"}}, "marker": "keep"}))
    monkeypatch.setenv("POPPY_CLAUDE_DESKTOP_CONFIG", str(normal_config))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))

    paths = install_for_client(client="claude-desktop")

    assert paths["MCP config"] == normal_config
    assert paths["MSIX MCP config"] == msix_config
    msix_backup = msix_config.with_name(msix_config.name + ".pre-poppy.bak")
    assert paths["MSIX backup"] == msix_backup
    assert json.loads(msix_backup.read_text())["marker"] == "keep"
    normal = json.loads(normal_config.read_text())
    msix = json.loads(msix_config.read_text())
    assert normal["mcpServers"]["poppy"]["command"] == expected_poppy_path(tmp_path)
    assert msix["mcpServers"]["poppy"]["command"] == expected_poppy_path(tmp_path)
    assert msix["mcpServers"]["other"] == {"command": "other"}
    assert msix["marker"] == "keep"


# --- Corrupt-config safety, universal backup, atomic write ---


def test_install_aborts_on_corrupt_config_without_overwriting(tmp_path):
    from poppy.setup.claude_code import CorruptConfigError

    config = tmp_path / ".claude.json"
    garbage = "{ this is not valid json "
    config.write_text(garbage)

    with pytest.raises(CorruptConfigError):
        install_mcp_config(claude_config_dir=tmp_path, client="claude-code")
    # The corrupt file is left byte-for-byte intact — not truncated to a stub.
    assert config.read_text() == garbage


@pytest.mark.parametrize(
    "installer",
    [
        install_session_start_hook,
        install_user_prompt_submit_hook,
        install_pre_tool_use_hook,
        install_session_end_hook,
        install_post_compact_hook,
        remove_legacy_hooks,
    ],
)
def test_hook_writers_refuse_malformed_settings(tmp_path, installer):
    settings_path = tmp_path / "settings.json"
    original = b'{"permissions": {"allow": []},\r\n'
    settings_path.write_bytes(original)

    with pytest.raises(CorruptConfigError, match="Refusing to overwrite unparseable config") as exc:
        installer(tmp_path)

    assert str(settings_path) in str(exc.value)
    assert settings_path.read_bytes() == original
    assert not is_hook_installed(tmp_path)


@pytest.mark.parametrize("client", ["claude-code", "codex"])
@pytest.mark.parametrize("daemon_mode", [False, True])
def test_setup_refuses_malformed_hooks_before_any_changes(tmp_path, monkeypatch, client, daemon_mode):
    client_dir = tmp_path / (".claude" if client == "claude-code" else ".codex")
    client_dir.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(client_dir))
    monkeypatch.setenv("CODEX_HOME", str(client_dir))
    settings_path = client_dir / ("settings.json" if client == "claude-code" else "hooks.json")
    settings_path.write_bytes(b'{"hooks": {},\r\n')
    mcp_path = tmp_path / ".claude.json" if client == "claude-code" else client_dir / "config.toml"
    mcp_path.write_text(
        '{"mcpServers": {"other": {"command": "keep"}}}' if client == "claude-code" else 'model = "keep"\n'
    )
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    def unexpected_daemon_setup():
        pytest.fail("Daemon bootstrap must not run with malformed client config")

    monkeypatch.setattr("poppy.cli.main._daemon_setup_kwargs", unexpected_daemon_setup)
    message = (
        f"Refusing to overwrite unparseable config at {settings_path}. Fix or remove it, then re-run `poppy setup`."
    )
    with pytest.raises(CorruptConfigError) as exc:
        install_for_client(client=client)
    assert str(exc.value) == message

    args = ["setup", client] + (["--daemon"] if daemon_mode else [])
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1
    assert result.output == f"Error: {message}\n"
    assert isinstance(result.exception, SystemExit)
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


_INVALID_CLAUDE_HOOKS = [
    ([], "config root must be an object"),
    ({"hooks": []}, '"hooks" must be an object'),
    ({"hooks": None}, '"hooks" must be an object'),
    ({"hooks": {"UserPromptSubmit": None}}, '"hooks.UserPromptSubmit" must be a list'),
    ({"hooks": {"UserPromptSubmit": {}}}, '"hooks.UserPromptSubmit" must be a list'),
    ({"hooks": {"UserPromptSubmit": "wrong"}}, '"hooks.UserPromptSubmit" must be a list'),
    ({"hooks": {"UserPromptSubmit": [None]}}, '"hooks.UserPromptSubmit[0]" must be an object'),
    ({"hooks": {"UserPromptSubmit": [[]]}}, '"hooks.UserPromptSubmit[0]" must be an object'),
    ({"hooks": {"UserPromptSubmit": [{"hooks": None}]}}, '"hooks.UserPromptSubmit[0].hooks" must be a list'),
    ({"hooks": {"UserPromptSubmit": [{"hooks": {}}]}}, '"hooks.UserPromptSubmit[0].hooks" must be a list'),
    ({"hooks": {"UserPromptSubmit": [{"hooks": [None]}]}}, '"hooks.UserPromptSubmit[0].hooks[0]" must be an object'),
    ({"hooks": {"Stop": None}}, '"hooks.Stop" must be a list'),
]

_REFUSED_CLIENT_CONFIGS = (
    [
        ("claude-code", ".claude/settings.json", json.dumps(settings).encode(), reason)
        for settings, reason in _INVALID_CLAUDE_HOOKS
    ]
    + [
        (client, f".{client}/hooks.json", json.dumps({"hooks": value}).encode(), '"hooks" must be an object')
        for client in ("cursor", "codex")
        for value in ([], "wrong", 1)
    ]
    + [
        ("codex", ".codex/hooks.json", b"[]", "config root must be an object"),
    ]
    + [
        (client, path, b"\xff\xfe", None)
        for client, path in (
            ("claude-code", ".claude/settings.json"),
            ("claude-code", ".claude.json"),
            ("cursor", ".cursor/hooks.json"),
            ("cursor", ".cursor/mcp.json"),
            ("codex", ".codex/hooks.json"),
            ("codex", ".codex/config.json"),
            ("codex", ".codex/config.toml"),
        )
    ]
)


@pytest.mark.parametrize("client,relative_path,original,reason", _REFUSED_CLIENT_CONFIGS)
@pytest.mark.parametrize("daemon_mode", [False, True])
@pytest.mark.parametrize("existing_mcp", [False, True])
def test_setup_refuses_invalid_config_before_any_changes(
    tmp_path, monkeypatch, client, relative_path, original, reason, daemon_mode, existing_mcp
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("POPPY_DIR", str(tmp_path / ".poppy"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claude"))
    monkeypatch.setenv("CURSOR_HOME", str(tmp_path / ".cursor"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))
    mcp_path = (
        tmp_path
        / {
            "claude-code": ".claude.json",
            "cursor": ".cursor/mcp.json",
            "codex": ".codex/config.toml",
        }[client]
    )
    if existing_mcp:
        mcp_path.parent.mkdir(parents=True, exist_ok=True)
        mcp_path.write_bytes(
            b'model = "keep"\r\n' if client == "codex" else b'{"mcpServers":{"other":{"command":"keep"}}}\r\n'
        )
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(original)
    (tmp_path / "unrelated.txt").write_bytes(b"keep\r\n")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    def unexpected_daemon_setup():
        pytest.fail("Daemon bootstrap must not run with invalid client config")

    monkeypatch.setattr("poppy.cli.main._daemon_setup_kwargs", unexpected_daemon_setup)
    prefix = (
        f"Refusing to overwrite unparseable config at {path}. "
        if reason is None
        else f"Refusing to overwrite invalid hooks config at {path}: {reason}. "
    )
    message = prefix + "Fix or remove it, then re-run `poppy setup`."
    with pytest.raises(CorruptConfigError) as exc:
        install_for_client(client=client, daemon=daemon_mode, daemon_token="test-token" if daemon_mode else None)
    assert str(exc.value) == message
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before

    result = CliRunner().invoke(cli, ["setup", client] + (["--daemon"] if daemon_mode else []))
    assert result.exit_code == 1
    assert result.output == f"Error: {message}\n"
    assert isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("settings,reason", _INVALID_CLAUDE_HOOKS)
@pytest.mark.parametrize(
    "installer",
    [
        install_session_start_hook,
        install_user_prompt_submit_hook,
        install_pre_tool_use_hook,
        install_session_end_hook,
        install_post_compact_hook,
        remove_legacy_hooks,
    ],
)
def test_claude_hook_writers_refuse_invalid_shapes(tmp_path, settings, reason, installer):
    path = tmp_path / "settings.json"
    original = json.dumps(settings).encode()
    path.write_bytes(original)

    with pytest.raises(CorruptConfigError) as exc:
        installer(tmp_path)

    assert str(path) in str(exc.value)
    assert reason in str(exc.value)
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == {path: original}


@pytest.mark.parametrize("installer", [install_cursor_hooks, install_codex_hooks])
@pytest.mark.parametrize("original", [b'{"hooks": []}', b"\xff\xfe"])
def test_native_hook_writers_refuse_invalid_configs(tmp_path, installer, original):
    path = tmp_path / "hooks.json"
    path.write_bytes(original)

    with pytest.raises(CorruptConfigError) as exc:
        installer(tmp_path)

    assert str(path) in str(exc.value)
    assert "Refusing to overwrite" in str(exc.value)
    assert {p: p.read_bytes() for p in tmp_path.iterdir()} == {path: original}


@pytest.mark.parametrize("original", [b"{malformed", b"[]", b'{"hooks":{"preToolUse":{}}}'])
def test_cursor_setup_keeps_intentional_backup_and_reset(tmp_path, monkeypatch, original):
    monkeypatch.setenv("HOME", str(tmp_path))
    cursor_home = tmp_path / ".cursor"
    cursor_home.mkdir()
    path = cursor_home / "hooks.json"
    path.write_bytes(original)

    result = CliRunner().invoke(cli, ["setup", "cursor"])

    assert result.exit_code == 0, result.output
    assert path.with_suffix(".json.bak").read_bytes() == original
    assert is_cursor_hooks_installed(cursor_home)


@pytest.mark.parametrize("status", [is_hook_installed, is_cursor_hooks_installed, is_codex_hooks_installed])
def test_hook_status_tolerates_non_utf8(tmp_path, status):
    (tmp_path / "settings.json").write_bytes(b"\xff\xfe")
    (tmp_path / "hooks.json").write_bytes(b"\xff\xfe")

    assert not status(tmp_path)


def test_install_preserves_user_settings(tmp_path):
    user_hook = {"matcher": "custom", "hooks": [{"type": "command", "command": "user-hook"}]}
    original = {
        "permissions": {"allow": ["Bash(ls)"]},
        "env": {"USER_SETTING": "keep"},
        "model": "user-model",
        "custom": {"keep": True},
        "hooks": {"SessionStart": [user_hook], "CustomEvent": [user_hook]},
    }
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps(original))

    install_for_client(claude_config_dir=tmp_path)

    final = json.loads(settings_path.read_text())
    for key in ("permissions", "env", "model", "custom"):
        assert final[key] == original[key]
    assert final["hooks"]["CustomEvent"] == [user_hook]
    assert user_hook in final["hooks"]["SessionStart"]
    for event in ("SessionStart", "UserPromptSubmit", "PreToolUse", "SessionEnd", "PostCompact"):
        assert is_hook_installed(tmp_path, event)


def test_install_without_hooks_leaves_malformed_settings_alone(tmp_path):
    settings_path = tmp_path / "settings.json"
    original = b'{"permissions":\r\n'
    settings_path.write_bytes(original)

    install_for_client(claude_config_dir=tmp_path, install_hooks=False)

    assert settings_path.read_bytes() == original
    assert is_mcp_installed(tmp_path)


def test_desktop_refuses_malformed_msix_config_before_any_changes(tmp_path, monkeypatch):
    config = tmp_path / "desktop.json"
    original = b'{"mcpServers": {"other": {"command": "keep"}}}'
    config.write_bytes(original)
    msix_config = tmp_path / "msix.json"
    malformed = b'{"mcpServers":\r\n'
    msix_config.write_bytes(malformed)
    monkeypatch.setenv("POPPY_CLAUDE_DESKTOP_CONFIG", str(config))
    monkeypatch.setattr("poppy.setup.claude_code.get_claude_desktop_msix_config_path", lambda: msix_config)

    with pytest.raises(CorruptConfigError, match="Refusing to overwrite unparseable config"):
        install_for_client(client="claude-desktop")

    assert config.read_bytes() == original
    assert msix_config.read_bytes() == malformed
    assert set(tmp_path.iterdir()) == {config, msix_config}


def test_install_backs_up_existing_config_for_non_desktop_client(tmp_path):
    from poppy.setup.claude_code import CONFIG_BACKUP_SUFFIX

    config = tmp_path / ".claude.json"
    config.write_text(json.dumps({"mcpServers": {"existing": {"command": "x"}}, "userField": 7}))

    install_mcp_config(claude_config_dir=tmp_path, client="claude-code")

    backup = config.with_name(config.name + CONFIG_BACKUP_SUFFIX)
    assert backup.exists()
    assert json.loads(backup.read_text()) == {"mcpServers": {"existing": {"command": "x"}}, "userField": 7}
    # Live config gained poppy while preserving the user's other entries.
    live = json.loads(config.read_text())
    assert "poppy" in live["mcpServers"]
    assert live["mcpServers"]["existing"]["command"] == "x"
    assert live["userField"] == 7


def test_install_no_backup_when_config_absent(tmp_path):
    from poppy.setup.claude_code import CONFIG_BACKUP_SUFFIX

    config = tmp_path / ".claude.json"
    install_mcp_config(claude_config_dir=tmp_path, client="claude-code")

    assert not config.with_name(config.name + CONFIG_BACKUP_SUFFIX).exists()
    assert config.exists()


def test_write_json_leaves_no_temp_file(tmp_path):
    install_mcp_config(claude_config_dir=tmp_path, client="claude-code")
    # Atomic write must clean up its same-dir temp file.
    assert not list(tmp_path.glob(".claude.json.poppy-tmp-*"))


def _link_to_dotfiles(tmp_path, link, content):
    """Place ``content`` in a dotfiles directory and symlink ``link`` to it."""
    target = tmp_path / "dots" / link.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    os.chmod(target, 0o640)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(target, link.parent))
    return target


@pytest.mark.parametrize(
    ("client", "relative_path", "original"),
    [
        ("claude-code", ".claude.json", '{"userField": "keep-me"}'),
        ("claude-code", ".claude/settings.json", '{"model": "keep-me"}'),
        ("cursor", ".cursor/mcp.json", '{"userField": "keep-me"}'),
        ("cursor", ".cursor/hooks.json", '{"version": 1, "hooks": {"stop": [{"command": "keep-me"}]}}'),
        ("codex", ".codex/config.toml", 'model = "keep-me"\n'),
        ("codex", ".codex/hooks.json", '{"hooks": {}, "userField": "keep-me"}'),
    ],
    ids=["claude-json", "claude-settings", "cursor-mcp", "cursor-hooks", "codex-toml", "codex-hooks"],
)
def test_install_writes_through_symlinked_client_config(tmp_path, monkeypatch, client, relative_path, original):
    """A dotfiles-managed config stays a symlink and the change lands in its target."""
    home = tmp_path / "home"
    monkeypatch.setenv("CURSOR_HOME", str(home / ".cursor"))
    monkeypatch.setenv("CODEX_HOME", str(home / ".codex"))
    link = home / relative_path
    target = _link_to_dotfiles(tmp_path, link, original)
    link_text = os.readlink(link)

    install_for_client(client=client, claude_config_dir=home / ".claude", install_claude_md=False)

    assert link.is_symlink()
    assert os.readlink(link) == link_text
    content = target.read_text()
    assert "poppy" in content
    assert "keep-me" in content
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert not list(tmp_path.rglob("*.poppy-tmp-*"))


def test_install_writes_through_chained_symlink(tmp_path):
    config = tmp_path / "claude" / ".claude.json"
    target = _link_to_dotfiles(tmp_path, tmp_path / "stow" / ".claude.json", '{"userField": 7}')
    config.parent.mkdir()
    config.symlink_to(tmp_path / "stow" / ".claude.json")

    install_mcp_config(claude_config_dir=tmp_path / "claude", client="claude-code")

    assert config.is_symlink()
    assert (tmp_path / "stow" / ".claude.json").is_symlink()
    settings = json.loads(target.read_text())
    assert "poppy" in settings["mcpServers"]
    assert settings["userField"] == 7


@pytest.mark.parametrize(
    "broken", ["dangling", "missing-directory", "dot-dot-through-missing-directory", "loop", "directory"]
)
def test_install_refuses_unfollowable_symlink_before_changing_files(tmp_path, broken):
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    dots = tmp_path / "dots"
    dots.mkdir()
    sibling = dots / "settings.json"
    sibling.write_text('{"permissions": {"deny": ["Bash(rm:*)"]}}')
    original_sibling = sibling.read_bytes()
    settings = claude_dir / "settings.json"
    if broken == "dangling":
        settings.symlink_to("../dots/absent.json")
    elif broken == "missing-directory":
        settings.symlink_to(tmp_path / "unmounted" / "settings.json")
    elif broken == "dot-dot-through-missing-directory":
        # The OS cannot open this link, but collapsing ".." as text would land
        # on the real sibling file.
        settings.symlink_to(dots / "missing" / ".." / "settings.json")
    elif broken == "directory":
        settings.symlink_to(dots)
    else:
        settings.symlink_to(claude_dir / "loop.json")
        (claude_dir / "loop.json").symlink_to(settings)
    before = {path: os.readlink(path) for path in claude_dir.iterdir()}

    with pytest.raises(CorruptConfigError, match="Refusing to write through symlink"):
        install_for_client(client="claude-code", claude_config_dir=claude_dir)

    assert {path: os.readlink(path) for path in claude_dir.iterdir()} == before
    assert not (tmp_path / ".claude.json").exists()
    assert sibling.read_bytes() == original_sibling
    assert list(dots.iterdir()) == [sibling]
    assert not (tmp_path / "unmounted").exists()


_RUNNING_AS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


@pytest.mark.skipif(_RUNNING_AS_ROOT, reason="root can write to read-only files")
def test_install_refuses_read_only_symlink_target_before_changing_files(tmp_path):
    claude_dir = tmp_path / ".claude"
    target = _link_to_dotfiles(tmp_path, claude_dir / "settings.json", '{"model": "keep-me"}')
    original = target.read_bytes()
    target.chmod(0o444)
    try:
        with pytest.raises(CorruptConfigError, match="is not writable"):
            install_for_client(client="claude-code", claude_config_dir=claude_dir)
    finally:
        target.chmod(0o640)

    assert (claude_dir / "settings.json").is_symlink()
    assert target.read_bytes() == original
    assert not (tmp_path / ".claude.json").exists()


@pytest.mark.skipif(_RUNNING_AS_ROOT, reason="root can write to read-only directories")
def test_install_writes_symlink_target_inside_read_only_directory(tmp_path):
    claude_dir = tmp_path / ".claude"
    target = _link_to_dotfiles(tmp_path, claude_dir / "settings.json", '{"model": "keep-me"}')
    target.parent.chmod(0o555)
    try:
        install_session_start_hook(claude_dir)
    finally:
        target.parent.chmod(0o755)

    assert "poppy" in target.read_text()
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize("when", ["after-read", "during-backup"])
def test_install_hook_refuses_symlink_retargeted_between_read_and_write(tmp_path, monkeypatch, when):
    from poppy.setup import claude_code

    claude_dir = tmp_path / ".claude"
    settings = claude_dir / "settings.json"
    first = _link_to_dotfiles(tmp_path, settings, '{"model": "first"}')
    second = tmp_path / "other" / "settings.json"
    second.parent.mkdir()
    second.write_text('{"permissions": {"deny": ["Bash(rm:*)"]}}')
    original_first = first.read_bytes()
    original_second = second.read_bytes()

    def retarget():
        settings.unlink()
        settings.symlink_to(second)

    if when == "after-read":
        validate = claude_code._validate_claude_hooks_config

        def hook(config, path):
            retarget()
            return validate(config, path)

        monkeypatch.setattr(claude_code, "_validate_claude_hooks_config", hook)
    else:
        backup_once = claude_code._backup_once

        def hook(path, suffix):
            result = backup_once(path, suffix)
            retarget()
            return result

        monkeypatch.setattr(claude_code, "_backup_once", hook)

    with pytest.raises(CorruptConfigError, match="while `poppy setup` was running"):
        install_session_start_hook(claude_dir)

    assert second.read_bytes() == original_second
    assert first.read_bytes() == original_first


def _assign_other_group(target):
    """Give ``target`` a group a fresh file beside it would not get, if one is available."""
    probe = target.with_name("group-probe")
    probe.touch()
    new_file_gid = probe.stat().st_gid
    probe.unlink()
    for gid in sorted(set(os.getgroups()) | {os.getegid()}):
        if gid == new_file_gid:
            continue
        try:
            os.chown(target, -1, gid)
        except OSError:
            continue
        return


def test_install_rewrites_symlink_target_in_place(tmp_path):
    """The target keeps its inode, owner, group and mode, and nothing is created beside it."""
    claude_dir = tmp_path / ".claude"
    # Padding makes the new content shorter, so a missing truncate leaves invalid JSON.
    target = _link_to_dotfiles(tmp_path, claude_dir / "settings.json", '{"model": "keep-me"' + " " * 4096 + "}")
    if hasattr(os, "getgroups"):
        _assign_other_group(target)
    before = target.stat()

    install_session_start_hook(claude_dir)

    after = target.stat()
    assert (after.st_dev, after.st_ino, after.st_uid, after.st_gid) == (
        before.st_dev,
        before.st_ino,
        before.st_uid,
        before.st_gid,
    )
    assert stat.S_IMODE(after.st_mode) == 0o640
    settings = json.loads(target.read_text())
    assert settings["model"] == "keep-me"
    assert "SessionStart" in settings["hooks"]
    assert list(target.parent.iterdir()) == [target]


def test_install_leaves_planted_temp_symlink_beside_target_alone(tmp_path):
    claude_dir = tmp_path / ".claude"
    target = _link_to_dotfiles(tmp_path, claude_dir / "settings.json", '{"model": "keep-me"}')
    victim = tmp_path / "unrelated.txt"
    victim.write_text("unrelated content")
    planted = target.with_name(f"{target.name}.poppy-tmp-{os.getpid()}")
    planted.symlink_to(victim)

    install_session_start_hook(claude_dir)

    assert victim.read_text() == "unrelated content"
    assert planted.is_symlink()
    assert os.readlink(planted) == str(victim)
    assert (claude_dir / "settings.json").is_symlink()
    assert "poppy" in target.read_text()


def test_install_saves_latest_content_before_each_in_place_write(tmp_path, monkeypatch):
    from poppy.setup import claude_code
    from poppy.setup.claude_code import CONFIG_BACKUP_SUFFIX, PREVIOUS_CONTENT_BACKUP_SUFFIX

    claude_dir = tmp_path / ".claude"
    link = claude_dir / "settings.json"
    target = _link_to_dotfiles(tmp_path, link, "{}")
    install_session_start_hook(claude_dir)
    edited = '{"hooks": {}, "recent": "IRREPLACEABLE"}'
    target.write_text(edited)
    real_write = os.write
    target_inode = target.stat().st_ino

    def write_then_fail(fd, data):
        if os.fstat(fd).st_ino != target_inode:
            return real_write(fd, data)
        real_write(fd, bytes(data[:80]))
        raise OSError(errno.ENOSPC, "simulated write failure")

    monkeypatch.setattr(claude_code.os, "write", write_then_fail)

    with pytest.raises(OSError, match="simulated write failure"):
        install_session_start_hook(claude_dir)
    monkeypatch.undo()

    with pytest.raises(json.JSONDecodeError):
        json.loads(target.read_text())
    assert link.with_name(link.name + CONFIG_BACKUP_SUFFIX).read_text() == "{}"
    previous = link.with_name(link.name + PREVIOUS_CONTENT_BACKUP_SUFFIX)
    assert previous.read_text() == edited
    assert stat.S_IMODE(previous.stat().st_mode) == 0o600
    shutil.copyfile(previous, link)
    assert link.is_symlink()
    assert json.loads(target.read_text())["recent"] == "IRREPLACEABLE"


def test_install_refuses_in_place_write_when_content_cannot_be_saved(tmp_path, monkeypatch):
    from poppy.setup import claude_code
    from poppy.setup.claude_code import PREVIOUS_CONTENT_BACKUP_SUFFIX

    claude_dir = tmp_path / ".claude"
    link = claude_dir / "settings.json"
    target = _link_to_dotfiles(tmp_path, link, '{"model": "keep-me"}')
    original = target.read_bytes()

    def no_space(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(claude_code.tempfile, "mkstemp", no_space)

    with pytest.raises(CorruptConfigError, match="could not save its current content"):
        install_session_start_hook(claude_dir)

    assert target.read_bytes() == original
    assert not link.with_name(link.name + PREVIOUS_CONTENT_BACKUP_SUFFIX).exists()


def test_install_never_exposes_stale_tail_when_content_shrinks(tmp_path, monkeypatch):
    from poppy.setup import claude_code

    claude_dir = tmp_path / ".claude"
    target = _link_to_dotfiles(tmp_path, claude_dir / "settings.json", '{"model": "keep-me"' + " " * 4096 + "}")
    real_ftruncate = os.ftruncate
    seen_before_truncate = []

    def parse_then_truncate(fd, length):
        seen_before_truncate.append(json.loads(target.read_text()))
        return real_ftruncate(fd, length)

    monkeypatch.setattr(claude_code.os, "ftruncate", parse_then_truncate)

    install_session_start_hook(claude_dir)

    assert len(seen_before_truncate) == 1
    assert "SessionStart" in seen_before_truncate[0]["hooks"]
    assert json.loads(target.read_text()) == seen_before_truncate[0]


@pytest.mark.skipif(sys.platform != "darwin", reason="uses macOS `chmod +a` ACLs")
def test_install_keeps_acl_on_symlink_target(tmp_path):
    claude_dir = tmp_path / ".claude"
    target = _link_to_dotfiles(tmp_path, claude_dir / "settings.json", '{"model": "keep-me"}')
    subprocess.run(["chmod", "+a", "everyone deny delete", str(target)], check=True)

    def acl_entries():
        listing = subprocess.run(["ls", "-le", str(target)], capture_output=True, text=True, check=True).stdout
        return listing.splitlines()[1:]

    before = acl_entries()
    assert before

    install_session_start_hook(claude_dir)

    assert acl_entries() == before
    assert "poppy" in target.read_text()


def test_install_backs_up_symlinked_config_next_to_the_link(tmp_path):
    from poppy.setup.claude_code import CONFIG_BACKUP_SUFFIX

    config = tmp_path / "claude" / ".claude.json"
    target = _link_to_dotfiles(tmp_path, config, '{"userField": 7}')

    install_mcp_config(claude_config_dir=tmp_path / "claude", client="claude-code")

    backup = config.with_name(config.name + CONFIG_BACKUP_SUFFIX)
    assert backup.is_file()
    assert not backup.is_symlink()
    assert json.loads(backup.read_text()) == {"userField": 7}
    assert list(target.parent.iterdir()) == [target]


def test_install_codex_hooks_preserves_group_without_hooks_key(tmp_path):
    """A user-authored group with no "hooks" key must not be silently dropped."""
    hooks_path = tmp_path / "hooks.json"
    hooks_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "UserPromptSubmit": [
                        {"matcher": "custom", "disabled": True},  # no "hooks" key
                        {"matcher": "", "hooks": [{"type": "command", "command": "poppy hook user-prompt-submit"}]},
                    ]
                }
            }
        )
    )
    install_codex_hooks(tmp_path)
    settings = json.loads(hooks_path.read_text())
    groups = settings["hooks"]["UserPromptSubmit"]
    assert {"matcher": "custom", "disabled": True} in groups
    assert any(h.get("command") == "poppy hook user-prompt-submit" for g in groups for h in g.get("hooks", []))
    assert is_codex_hooks_installed(tmp_path)


def test_install_codex_hooks_preserves_non_list_event_value(tmp_path):
    """A malformed non-list event value must be preserved verbatim, not crash."""
    hooks_path = tmp_path / "hooks.json"
    hooks_path.write_text(json.dumps({"hooks": {"Stop": "poppy hook stop"}}))  # non-list value
    # Must not raise.
    install_codex_hooks(tmp_path)
    settings = json.loads(hooks_path.read_text())
    assert settings["hooks"]["Stop"] == "poppy hook stop"  # user value untouched
    # Other events still got their poppy hooks installed.
    assert any(
        h.get("command") == "poppy hook session-start"
        for g in settings["hooks"]["SessionStart"]
        for h in g.get("hooks", [])
    )
