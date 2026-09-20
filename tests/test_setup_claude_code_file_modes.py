"""Client config writes keep daemon tokens private."""

import json
import os
import stat

import pytest
from click.testing import CliRunner

from poppy.cli.main import cli
from poppy.mcp_server import lifecycle
from poppy.setup import claude_code as claude_code_module


@pytest.mark.parametrize("daemon", [False, True], ids=["stdio", "daemon"])
@pytest.mark.parametrize(
    "existing_mode", [None, 0o600, 0o640, 0o644], ids=["new", "private", "group-readable", "world-readable"]
)
def test_setup_preserves_config_and_backup_modes(tmp_path, monkeypatch, daemon, existing_mode):
    claude_dir = tmp_path / "claude"
    claude_dir.mkdir()
    config_path = claude_dir / ".claude.json"
    original = '{"mcpServers": {"existing": {"command": "keep"}}}\n'
    if existing_mode is not None:
        config_path.write_text(original)
        os.chmod(config_path, existing_mode)

    # Never install a real service or invoke launchctl/systemctl from this test.
    monkeypatch.setattr(lifecycle, "install_agent", lambda *_a, **_k: None)
    monkeypatch.setattr(lifecycle, "start_daemon", lambda *_a, **_k: "started")
    monkeypatch.setattr(lifecycle, "probe_status", lambda *_a, **_k: ({"version": "t"}, None))
    env = {
        "HOME": str(tmp_path / "home"),
        "POPPY_DIR": str(tmp_path / "poppy"),
        "CLAUDE_CONFIG_DIR": str(claude_dir),
        "CURSOR_HOME": str(tmp_path / "cursor"),
        "CODEX_HOME": str(tmp_path / "codex"),
        "POPPY_LAUNCH_AGENTS_DIR": str(tmp_path / "agents"),
        "POPPY_SYSTEMD_USER_DIR": str(tmp_path / "units"),
        "POPPY_TELEMETRY_OFF": "1",
    }
    args = ["setup", "claude-code", "--yes", "--no-hooks", "--no-claude-md"]
    if daemon:
        args.append("--daemon")
    runner = CliRunner()
    expected_mode = 0o600 if daemon or existing_mode is None else existing_mode

    # Inspect only the client config, since os is shared with other writers.
    # The replacement must already have its final permissions when published.
    replace_calls = []
    original_replace = os.replace

    def _capture_replace(src, dst):
        if os.path.basename(str(dst)) == config_path.name:
            replace_calls.append(stat.S_IMODE(os.stat(src).st_mode))
        return original_replace(src, dst)

    monkeypatch.setattr(claude_code_module.os, "replace", _capture_replace)

    # A permissive umask makes the regression deterministic; always restore it.
    previous_umask = os.umask(0o022)
    try:
        for run in range(2):
            before = config_path.read_text() if config_path.exists() else None
            result = runner.invoke(cli, args, env=env)
            assert result.exit_code == 0, result.output
            assert stat.S_IMODE(config_path.stat().st_mode) == expected_mode
            tightened = daemon and existing_mode in (0o640, 0o644) and run == 0
            # The notice goes to stderr only, so stdout stays parseable.
            assert result.stderr.count("Tightened permissions") == int(tightened)
            assert "Tightened permissions" not in result.stdout
            if tightened:
                assert f"Tightened permissions on {config_path} to owner-only (0600)" in result.stderr
            settings = json.loads(config_path.read_text())
            entry = settings["mcpServers"]["poppy"]
            if daemon:
                token = (tmp_path / "poppy" / "daemon.token").read_text().strip()
                assert token
                assert entry["type"] == "http"
                assert entry["headers"]["Authorization"] == f"Bearer {token}"
                # Reporting the change must never echo the credential itself.
                assert token not in result.stdout
                assert token not in result.stderr
            else:
                assert entry["type"] == "stdio"
            if existing_mode is not None:
                assert settings["mcpServers"]["existing"] == {"command": "keep"}

            backups = sorted(claude_dir.glob(".claude.json.pre-poppy.bak*"))
            assert len(backups) == run + (existing_mode is not None)
            for index, backup in enumerate(backups):
                backup_mode = existing_mode if index == 0 and existing_mode is not None else expected_mode
                assert stat.S_IMODE(backup.stat().st_mode) == backup_mode
            if before is not None:
                assert backups[-1].read_text() == before
    finally:
        os.umask(previous_umask)

    assert replace_calls
    assert all(captured_mode == expected_mode for captured_mode in replace_calls)
    assert not (tmp_path / "agents").exists()
    assert not (tmp_path / "units").exists()


@pytest.mark.parametrize(
    "client", ["claude-code", "claude-desktop", "cursor", "vscode", "windsurf", "codex", "copilot-cli", "pi", "gemini"]
)
@pytest.mark.parametrize("symlink", [False, True], ids=["regular", "symlink"])
@pytest.mark.parametrize("daemon", [False, True], ids=["stdio", "daemon"])
@pytest.mark.parametrize("existing_mode", [0o640, 0o644], ids=["group-readable", "world-readable"])
def test_client_token_permissions(tmp_path, monkeypatch, capsys, client, symlink, daemon, existing_mode):
    monkeypatch.setenv("HOME", str(tmp_path))
    for name, directory in {
        "CODEX_HOME": "codex",
        "CURSOR_HOME": "cursor",
        "COPILOT_HOME": "copilot",
        "PI_CODING_AGENT_DIR": "pi",
        "POPPY_CLAUDE_DESKTOP_CONFIG": "desktop.json",
        "POPPY_VSCODE_MCP_CONFIG": "vscode.json",
    }.items():
        monkeypatch.setenv(name, str(tmp_path / directory))
    claude_dir = tmp_path / "claude"
    path = claude_code_module._client_settings_path(client, claude_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "dotfile" if symlink else path
    target.write_text('model = "keep"\n' if client == "codex" else '{"marker": "keep"}\n')
    target.chmod(existing_mode)
    if symlink:
        path.symlink_to(target)
    inode = target.stat().st_ino
    expected_mode = 0o600 if daemon else existing_mode
    original_write = os.write

    def capture_write(fd, data):
        if os.fstat(fd).st_ino == inode and b"test-token" in bytes(data):
            assert stat.S_IMODE(os.fstat(fd).st_mode) == 0o600
        return original_write(fd, data)

    monkeypatch.setattr(claude_code_module.os, "write", capture_write)
    for run in range(2):
        claude_code_module.install_mcp_config(claude_dir, client, daemon=daemon, daemon_token="test-token")
        captured = capsys.readouterr()
        assert stat.S_IMODE(target.stat().st_mode) == expected_mode
        # The notice goes to stderr only, and never carries the credential.
        assert captured.err.count("Tightened permissions") == int(daemon and run == 0)
        assert "Tightened permissions" not in captured.out
        assert "test-token" not in captured.err
        assert "test-token" not in captured.out
        assert ("test-token" in target.read_text()) == daemon
        assert "keep" in target.read_text()
        if symlink:
            assert path.is_symlink()
            assert target.stat().st_ino == inode


@pytest.mark.parametrize("client", ["cursor", "codex"])
def test_existing_token_backup_is_private(tmp_path, monkeypatch, client):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CURSOR_HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    path = claude_code_module.install_mcp_config(client=client, daemon=True, daemon_token="old-token")
    path.chmod(0o644)
    claude_code_module.install_mcp_config(client=client, daemon=True, daemon_token="new-token")
    backup = path.with_name(path.name + claude_code_module.CONFIG_BACKUP_SUFFIX)
    assert "old-token" in backup.read_text()
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600


def test_chained_relative_symlink_target_is_tightened(tmp_path, monkeypatch, capsys):
    """A config reached through two relative links still ends owner-only."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CURSOR_HOME", str(tmp_path / "cursor"))
    claude_dir = tmp_path / "claude"
    path = claude_code_module._client_settings_path("cursor", claude_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    dotfiles = tmp_path / "dotfiles"
    dotfiles.mkdir()
    real = dotfiles / "mcp.json"
    real.write_text('{"marker": "keep"}\n')
    real.chmod(0o644)
    middle = dotfiles / "middle.json"
    middle.symlink_to("mcp.json")
    path.symlink_to(os.path.relpath(middle, path.parent))
    inode = real.stat().st_ino

    claude_code_module.install_mcp_config(claude_dir, "cursor", daemon=True, daemon_token="test-token")

    assert stat.S_IMODE(real.stat().st_mode) == 0o600
    # Both links survive and the same file was rewritten, not replaced.
    assert path.is_symlink() and middle.is_symlink()
    assert real.stat().st_ino == inode
    assert "test-token" in real.read_text()
    assert "keep" in real.read_text()
    # The notice names the file whose mode actually changed.
    assert f"Tightened permissions on {real} to owner-only (0600)" in capsys.readouterr().err


def test_token_write_refused_when_mode_cannot_be_tightened(tmp_path, monkeypatch):
    """Rather than leave the token in a file it cannot protect, setup aborts.

    A config can be writable through its group while still being owned by
    another user, and only the owner may change a file's mode.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CURSOR_HOME", str(tmp_path / "cursor"))
    claude_dir = tmp_path / "claude"
    path = claude_code_module._client_settings_path("cursor", claude_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "dotfile.json"
    target.write_text('{"marker": "keep"}\n')
    target.chmod(0o644)
    path.symlink_to(target)
    inode = target.stat().st_ino
    original_fchmod = os.fchmod

    def refuse_fchmod(fd, mode):
        if os.fstat(fd).st_ino == inode:
            raise PermissionError(1, "Operation not permitted")
        return original_fchmod(fd, mode)

    monkeypatch.setattr(claude_code_module.os, "fchmod", refuse_fchmod)

    with pytest.raises(claude_code_module.CorruptConfigError) as excinfo:
        claude_code_module.install_mcp_config(claude_dir, "cursor", daemon=True, daemon_token="test-token")

    assert "Operation not permitted" in str(excinfo.value)
    assert "test-token" not in str(excinfo.value)
    # Nothing was written, so nothing leaked.
    assert "test-token" not in target.read_text()
    assert "keep" in target.read_text()
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
