"""Client config writes and backups preserve permissions."""

import json
import os
import stat

import pytest
from click.testing import CliRunner

from poppy.cli.main import cli
from poppy.mcp_server import lifecycle
from poppy.setup import claude_code as claude_code_module


@pytest.mark.parametrize("daemon", [False, True], ids=["stdio", "daemon"])
@pytest.mark.parametrize("existing_mode", [None, 0o600, 0o640], ids=["new", "private", "group-readable"])
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
    expected_mode = existing_mode if existing_mode is not None else 0o600

    # The temp file must already be at its final mode by the time os.replace
    # publishes it — proves the token/content was never written at a wider,
    # umask-default mode first (Greptile). `os` is a shared module
    # object, so filter to the client config's own replace calls; Poppy's own
    # internal config/token files are correctly always 0600 and aren't the
    # thing under test here.
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
            settings = json.loads(config_path.read_text())
            entry = settings["mcpServers"]["poppy"]
            if daemon:
                token = (tmp_path / "poppy" / "daemon.token").read_text().strip()
                assert token
                assert entry["type"] == "http"
                assert entry["headers"]["Authorization"] == f"Bearer {token}"
            else:
                assert entry["type"] == "stdio"
            if existing_mode is not None:
                assert settings["mcpServers"]["existing"] == {"command": "keep"}

            backups = sorted(claude_dir.glob(".claude.json.pre-poppy.bak*"))
            assert len(backups) == run + (existing_mode is not None)
            for backup in backups:
                assert stat.S_IMODE(backup.stat().st_mode) == expected_mode
            if before is not None:
                assert backups[-1].read_text() == before
    finally:
        os.umask(previous_umask)

    assert replace_calls
    assert all(captured_mode == expected_mode for captured_mode in replace_calls)
    assert not (tmp_path / "agents").exists()
    assert not (tmp_path / "units").exists()
