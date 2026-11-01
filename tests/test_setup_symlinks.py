"""Setup preserves context and config files managed through dotfiles symlinks."""

import json
import os
import stat

import pytest
from click.testing import CliRunner

from poppy.cli.main import cli
from poppy.config import PoppyConfig, save_config
from poppy.setup.claude_code import (
    CLAUDE_MD_BEGIN,
    CLAUDE_MD_END,
    CONFIG_BACKUP_SUFFIX,
    CorruptConfigError,
    _remove_managed_primer_block,
    install_primer_md_block,
)
from poppy.setup.goose import install_for_goose
from poppy.setup.hermes import _remove_legacy_agents_block, install_for_hermes


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("POPPY_DIR", str(tmp_path / "poppy"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))
    monkeypatch.setenv("COPILOT_HOME", str(tmp_path / ".copilot"))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / ".pi" / "agent"))
    monkeypatch.setenv("POPPY_GOOSE_CONFIG_DIR", str(tmp_path / "goose"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("poppy.setup.claude_code.shutil.which", lambda _: str(tmp_path / "bin" / "poppy"))


def link_file(tmp_path, link, content):
    target = tmp_path / "dotfiles" / link.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(target, link.parent))
    return target


@pytest.mark.parametrize("name", ["CLAUDE.md", "AGENTS.md", "copilot-instructions.md", ".goosehints"])
def test_primer_install_preserves_symlink_and_backs_up_target(tmp_path, name):
    link = tmp_path / "client" / name
    original = "# My instructions\n"
    target = link_file(tmp_path, link, original)
    link_text = os.readlink(link)

    install_primer_md_block(link)
    install_primer_md_block(link)

    assert link.is_symlink()
    assert os.readlink(link) == link_text
    assert target.read_text().startswith(original)
    assert target.read_text().count(CLAUDE_MD_BEGIN) == 1
    assert link.with_name(link.name + CONFIG_BACKUP_SUFFIX).read_text() == original


@pytest.mark.parametrize("remove", [_remove_managed_primer_block, _remove_legacy_agents_block])
@pytest.mark.parametrize("prefix", ["", "# My instructions\n"])
def test_primer_removal_preserves_symlink_and_edits_target(tmp_path, remove, prefix):
    link = tmp_path / "client" / "AGENTS.md"
    original = f"{prefix}{CLAUDE_MD_BEGIN}\nold guidance\n{CLAUDE_MD_END}\n"
    target = link_file(tmp_path, link, original)
    link_text = os.readlink(link)

    remove(link)
    remove(link)

    assert link.is_symlink()
    assert os.readlink(link) == link_text
    assert target.read_text() == (prefix + "\n" if prefix else "")
    assert link.with_name(link.name + CONFIG_BACKUP_SUFFIX).read_text() == original


@pytest.mark.parametrize("remove", [_remove_managed_primer_block, _remove_legacy_agents_block])
def test_primer_removal_refuses_broken_symlink(tmp_path, remove):
    link = tmp_path / "AGENTS.md"
    target = tmp_path / "missing.md"
    link.symlink_to(target)

    with pytest.raises(CorruptConfigError, match=str(link)):
        remove(link)

    assert link.is_symlink()
    assert not target.exists()
    assert list(tmp_path.iterdir()) == [link]


@pytest.mark.parametrize(
    ("install", "name", "original", "expected"),
    [
        (install_for_goose, "config.yaml", "GOOSE_MODEL: keep-me\n", "poppy:"),
        (install_for_goose, ".goosehints", "# keep-me\n", CLAUDE_MD_BEGIN),
        (install_for_hermes, "config.yaml", "model: keep-me\n", "provider: poppy"),
        (install_for_hermes, "SOUL.md", "# keep-me\n", "poppy_recall"),
        (install_for_hermes, "AGENTS.md", f"{CLAUDE_MD_BEGIN}\nold\n{CLAUDE_MD_END}\n", ""),
    ],
)
def test_client_install_preserves_symlink_and_backs_up_target(tmp_path, install, name, original, expected):
    home = tmp_path / "client"
    link = home / name
    target = link_file(tmp_path, link, original)
    link_text = os.readlink(link)

    install(home)
    install(home)

    assert link.is_symlink()
    assert os.readlink(link) == link_text
    if expected:
        assert "keep-me" in target.read_text()
        assert expected in target.read_text()
    else:
        assert target.read_text() == ""
    assert link.with_name(link.name + CONFIG_BACKUP_SUFFIX).read_text() == original


def snapshot(root):
    return {
        str(path.relative_to(root)): os.readlink(path)
        if path.is_symlink()
        else (path.read_bytes() if path.is_file() else None)
        for path in root.rglob("*")
    }


@pytest.mark.parametrize(
    ("client", "relative_path", "options"),
    [
        ("claude-code", ".claude/CLAUDE.md", []),
        ("claude-code", ".claude/CLAUDE.md", ["--daemon"]),
        ("codex", ".codex/AGENTS.md", []),
        ("codex", ".codex/AGENTS.md", ["--daemon"]),
        ("copilot-cli", ".copilot/copilot-instructions.md", []),
        ("copilot-cli", ".copilot/AGENTS.md", []),
        ("pi", ".pi/agent/AGENTS.md", []),
        ("pi", ".pi/AGENTS.md", []),
        ("goose", "goose/config.yaml", []),
        ("goose", "goose/.goosehints", []),
        ("hermes-agent", "hermes/config.yaml", []),
        ("hermes-agent", "hermes/SOUL.md", []),
        ("hermes-agent", "hermes/AGENTS.md", []),
    ],
)
def test_setup_refuses_broken_symlink_before_any_changes(tmp_path, monkeypatch, client, relative_path, options):
    link = tmp_path / relative_path
    link.parent.mkdir(parents=True)
    link.symlink_to("missing-target")
    (link.parent / "user-file").write_text("Keep this.\n")
    before = snapshot(tmp_path)
    monkeypatch.setattr("poppy.cli.main._daemon_setup_kwargs", lambda: pytest.fail("daemon setup ran before preflight"))

    result = CliRunner().invoke(cli, ["setup", client, *options])

    assert result.exit_code == 1
    assert result.output.startswith("Error: Refusing to write through symlink")
    assert str(link) in result.output
    assert len(result.output.splitlines()) == 1
    assert snapshot(tmp_path) == before


def test_save_config_preserves_symlink_and_updates_target(tmp_path):
    poppy_dir = tmp_path / "poppy"
    link = poppy_dir / "config.json"
    target = link_file(tmp_path, link, "{}")
    target.chmod(0o644)
    link_text = os.readlink(link)
    config = PoppyConfig(poppy_dir=poppy_dir, telemetry_enabled=False)

    save_config(config)

    assert link.is_symlink()
    assert os.readlink(link) == link_text
    assert json.loads(target.read_text())["telemetry_enabled"] is False
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not list(tmp_path.rglob(".config-*.tmp"))


def test_setup_with_plugin_refuses_broken_primer_before_any_changes(tmp_path, monkeypatch):
    link = tmp_path / ".claude" / "CLAUDE.md"
    link.parent.mkdir()
    link.symlink_to("missing-target")
    before = snapshot(tmp_path)
    monkeypatch.setattr("poppy.setup.claude_code.is_claude_code_plugin_installed", lambda _: True)

    result = CliRunner().invoke(cli, ["setup", "claude-code", "--claude-md"])

    assert result.exit_code == 1
    assert result.output.startswith("Error: Refusing to write through symlink")
    assert str(link) in result.output
    assert len(result.output.splitlines()) == 1
    assert snapshot(tmp_path) == before
