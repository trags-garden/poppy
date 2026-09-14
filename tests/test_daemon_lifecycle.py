from __future__ import annotations

import fcntl
import json
import os
import plistlib
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from click.testing import CliRunner

from poppy.cli.main import cli
from poppy.mcp_server import lifecycle


def _record_commands(monkeypatch, returncodes=None):
    calls = []
    codes = returncodes or {}

    def record(argv):
        calls.append(argv)
        code = codes.get(argv[1], 0) if len(argv) > 1 else 0
        return subprocess.CompletedProcess(argv, code, "", "")

    monkeypatch.setattr(lifecycle, "_run_service_command", record)
    return calls


def test_launchd_install_and_uninstall_content(tmp_path, monkeypatch):
    agents = tmp_path / "LaunchAgents"
    poppy_dir = tmp_path / "poppy"
    monkeypatch.setenv("POPPY_LAUNCH_AGENTS_DIR", str(agents))
    calls = _record_commands(monkeypatch)

    path, _ = lifecycle.install_agent(poppy_dir, platform="darwin", executable="/opt/poppy/bin/poppy")

    data = plistlib.loads(path.read_bytes())
    assert data["Label"] == lifecycle.LAUNCHD_LABEL
    assert data["ProgramArguments"] == ["/opt/poppy/bin/poppy", "daemon", "run"]
    assert data["RunAtLoad"] is True
    assert data["KeepAlive"] is True
    assert data["StandardOutPath"] == str(poppy_dir / "logs" / "daemon.log")
    assert data["StandardErrorPath"] == str(poppy_dir / "logs" / "daemon.log")
    target = f"gui/{os.getuid()}/{lifecycle.LAUNCHD_LABEL}"
    # Default stub returncode 0 means the label reads as already loaded, so a
    # re-install bootouts the stale definition before bootstrapping the new one.
    assert calls == [
        ["launchctl", "print", target],
        ["launchctl", "bootout", target],
        ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)],
    ]

    lifecycle.uninstall_agent(platform="darwin")
    assert not path.exists()
    assert calls[-1] == ["launchctl", "bootout", target]


def test_launchd_install_and_uninstall_when_label_not_loaded(tmp_path, monkeypatch):
    agents = tmp_path / "LaunchAgents"
    poppy_dir = tmp_path / "poppy"
    monkeypatch.setenv("POPPY_LAUNCH_AGENTS_DIR", str(agents))
    calls = _record_commands(monkeypatch, returncodes={"print": 1})

    path, _ = lifecycle.install_agent(poppy_dir, platform="darwin", executable="/opt/poppy/bin/poppy")
    target = f"gui/{os.getuid()}/{lifecycle.LAUNCHD_LABEL}"
    assert calls == [
        ["launchctl", "print", target],
        ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)],
    ]

    del calls[:]
    lifecycle.uninstall_agent(platform="darwin")
    assert not path.exists()
    assert calls == [["launchctl", "print", target]]


def test_systemd_install_and_uninstall_content(tmp_path, monkeypatch):
    units = tmp_path / "systemd" / "user"
    poppy_dir = tmp_path / "poppy"
    monkeypatch.setenv("POPPY_SYSTEMD_USER_DIR", str(units))
    calls = _record_commands(monkeypatch)

    path, _ = lifecycle.install_agent(poppy_dir, platform="linux", executable="/opt/poppy/bin/poppy")

    unit = path.read_text()
    assert "ExecStart=/opt/poppy/bin/poppy daemon run" in unit
    assert "Restart=on-failure" in unit
    assert "WantedBy=default.target" in unit
    assert calls == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", lifecycle.SYSTEMD_FILENAME],
    ]

    lifecycle.uninstall_agent(platform="linux")
    assert not path.exists()
    assert calls[-2:] == [
        ["systemctl", "--user", "disable", "--now", lifecycle.SYSTEMD_FILENAME],
        ["systemctl", "--user", "daemon-reload"],
    ]


def test_installed_start_stop_drive_systemd(tmp_path, monkeypatch):
    units = tmp_path / "units"
    monkeypatch.setenv("POPPY_SYSTEMD_USER_DIR", str(units))
    path = lifecycle.agent_path("linux")
    path.parent.mkdir(parents=True)
    path.write_text("unit")
    calls = _record_commands(monkeypatch)
    monkeypatch.setattr(
        lifecycle,
        "inspect_daemon",
        lambda *_args, **_kwargs: lifecycle.DaemonState(True, False, None, False, None),
    )
    monkeypatch.setattr(lifecycle, "_wait_for_daemon", lambda *_args: True)

    assert "systemd" in lifecycle.start_daemon(tmp_path, platform="linux")
    assert "systemd" in lifecycle.stop_daemon(tmp_path, platform="linux")
    assert calls == [
        ["systemctl", "--user", "start", lifecycle.SYSTEMD_FILENAME],
        ["systemctl", "--user", "stop", lifecycle.SYSTEMD_FILENAME],
    ]


def test_installed_start_stop_drive_launchctl(tmp_path, monkeypatch):
    agents = tmp_path / "agents"
    monkeypatch.setenv("POPPY_LAUNCH_AGENTS_DIR", str(agents))
    path = lifecycle.agent_path("darwin")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"plist")
    calls = _record_commands(monkeypatch, returncodes={"print": 1})
    monkeypatch.setattr(
        lifecycle,
        "inspect_daemon",
        lambda *_args, **_kwargs: lifecycle.DaemonState(True, False, None, False, None),
    )
    monkeypatch.setattr(lifecycle, "_wait_for_daemon", lambda *_args: True)

    assert "launchd" in lifecycle.start_daemon(tmp_path, platform="darwin")
    assert "launchd" in lifecycle.stop_daemon(tmp_path, platform="darwin")
    target = f"gui/{os.getuid()}/{lifecycle.LAUNCHD_LABEL}"
    assert calls == [
        ["launchctl", "print", target],
        ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)],
        ["launchctl", "kickstart", target],
        ["launchctl", "bootout", target],
    ]


def test_start_daemon_skips_bootstrap_when_label_already_loaded(tmp_path, monkeypatch):
    agents = tmp_path / "agents"
    monkeypatch.setenv("POPPY_LAUNCH_AGENTS_DIR", str(agents))
    path = lifecycle.agent_path("darwin")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"plist")
    calls = _record_commands(monkeypatch)
    monkeypatch.setattr(
        lifecycle,
        "inspect_daemon",
        lambda *_args, **_kwargs: lifecycle.DaemonState(True, False, None, False, None),
    )
    monkeypatch.setattr(lifecycle, "_wait_for_daemon", lambda *_args: True)

    assert "launchd" in lifecycle.start_daemon(tmp_path, platform="darwin")
    target = f"gui/{os.getuid()}/{lifecycle.LAUNCHD_LABEL}"
    assert calls == [
        ["launchctl", "print", target],
        ["launchctl", "kickstart", target],
    ]


@pytest.mark.parametrize(
    "platform,expected",
    [
        ("darwin", ["launchctl", "kickstart", f"gui/{os.getuid()}/{lifecycle.LAUNCHD_LABEL}"]),
        ("linux", ["systemctl", "--user", "start", lifecycle.SYSTEMD_FILENAME]),
    ],
)
def test_kickstart_agent_uses_service_command_seam(monkeypatch, platform, expected):
    calls = _record_commands(monkeypatch)

    assert lifecycle.kickstart_agent(platform=platform) == expected
    assert calls == [expected]


def test_fallback_start_and_stop_use_process_seams(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_SYSTEMD_USER_DIR", str(tmp_path / "missing-units"))
    spawned = []
    signaled = []
    monkeypatch.setattr(
        lifecycle,
        "inspect_daemon",
        lambda *_args, **_kwargs: lifecycle.DaemonState(False, False, None, False, None),
    )
    monkeypatch.setattr(lifecycle, "resolve_poppy_executable", lambda: "/bin/poppy")
    monkeypatch.setattr(lifecycle, "_spawn_detached", lambda argv, log: spawned.append((argv, log)))
    monkeypatch.setattr(lifecycle, "_wait_for_daemon", lambda *_args: True)

    assert "detached" in lifecycle.start_daemon(tmp_path, platform="linux")
    assert spawned == [(["/bin/poppy", "daemon", "run"], tmp_path / "logs" / "daemon.log")]

    monkeypatch.setattr(lifecycle, "lock_status", lambda _path: (True, 4321))
    existence = iter([True, False, False])
    monkeypatch.setattr(lifecycle, "_process_exists", lambda _pid: next(existence))
    monkeypatch.setattr(lifecycle, "_signal_process", lambda pid, sig: signaled.append((pid, sig)))
    assert "PID 4321" in lifecycle.stop_daemon(tmp_path, platform="linux")
    assert signaled[0][0] == 4321


@pytest.mark.parametrize("command", ["install", "start", "stop"])
def test_daemon_service_command_failure_exits_one(tmp_path, monkeypatch, command):
    agents = tmp_path / "agents"
    monkeypatch.setenv("POPPY_LAUNCH_AGENTS_DIR", str(agents))
    monkeypatch.setattr(lifecycle.sys, "platform", "darwin")
    monkeypatch.setattr(
        lifecycle,
        "inspect_daemon",
        lambda *_args, **_kwargs: lifecycle.DaemonState(True, False, None, False, None),
    )
    if command != "install":
        path = lifecycle.agent_path("darwin")
        path.parent.mkdir(parents=True)
        path.write_bytes(b"plist")

    def fail(argv):
        return subprocess.CompletedProcess(argv, 17, "", "service manager refused the request")

    monkeypatch.setattr(lifecycle, "_run_service_command", fail)
    result = CliRunner().invoke(cli, ["daemon", command], env={"POPPY_DIR": str(tmp_path / "poppy")})

    assert result.exit_code == 1
    assert "service manager refused the request" in result.output
    assert "status 17" in result.output


def test_lifecycle_error_carries_service_failure_details(monkeypatch):
    argv = ["systemctl", "--user", "start", lifecycle.SYSTEMD_FILENAME]
    stderr = "  " + ("x" * (lifecycle.SERVICE_STDERR_TAIL + 20)) + " final error  \n"
    monkeypatch.setattr(
        lifecycle,
        "_run_service_command",
        lambda _argv: subprocess.CompletedProcess(argv, 23, "", stderr),
    )

    with pytest.raises(lifecycle.LifecycleError) as raised:
        lifecycle._checked_service_command("start the Poppy daemon", argv)

    assert raised.value.action == "start the Poppy daemon"
    assert raised.value.argv == argv
    assert raised.value.returncode == 23
    assert raised.value.stderr == stderr.strip()
    assert "final error" in str(raised.value)
    assert len(str(raised.value).split("stderr: ", 1)[1]) == lifecycle.SERVICE_STDERR_TAIL


def test_start_command_success_without_reachability_raises(tmp_path, monkeypatch):
    units = tmp_path / "units"
    monkeypatch.setenv("POPPY_SYSTEMD_USER_DIR", str(units))
    path = lifecycle.agent_path("linux")
    path.parent.mkdir(parents=True)
    path.write_text("unit")
    monkeypatch.setattr(
        lifecycle,
        "inspect_daemon",
        lambda *_args, **_kwargs: lifecycle.DaemonState(True, False, None, False, None),
    )
    monkeypatch.setattr(
        lifecycle,
        "_run_service_command",
        lambda argv: subprocess.CompletedProcess(argv, 0, "", ""),
    )
    monkeypatch.setattr(lifecycle, "lock_status", lambda _path: (False, None))
    monkeypatch.setattr(lifecycle, "probe_status", lambda _path: (None, "offline"))

    with pytest.raises(lifecycle.LifecycleError, match="daemon.log"):
        lifecycle.start_daemon(tmp_path, platform="linux", timeout=0)


def test_wait_for_daemon_token_returns_the_settled_authenticated_token(tmp_path, monkeypatch):
    """A single authenticated, stable read returns the served token."""
    (tmp_path / "daemon.token").write_text("served-B\n")
    monkeypatch.delenv("POPPY_DAEMON_TOKEN", raising=False)
    monkeypatch.setattr(lifecycle, "probe_status", lambda *_a, **_k: ({"version": "t"}, None))

    assert lifecycle.wait_for_daemon_token(tmp_path, port=7679, timeout=1.0) == "served-B"


def test_wait_for_daemon_token_rejects_a_mid_rotation_read(tmp_path, monkeypatch):
    """A rotation landing between the probe and the re-read is not captured; the settled value is returned."""
    monkeypatch.delenv("POPPY_DAEMON_TOKEN", raising=False)
    token_file = tmp_path / "daemon.token"
    token_file.write_text("old-A\n")
    state = {"n": 0}

    def fake_probe(_path, _port=None, timeout=0.25):
        state["n"] += 1
        # On the first authenticated probe, the file rotates to B underneath us:
        # the read-after-probe won't match old-A, so this attempt is discarded.
        if state["n"] == 1:
            token_file.write_text("new-B\n")
        return {"version": "t"}, None

    monkeypatch.setattr(lifecycle, "probe_status", fake_probe)

    assert lifecycle.wait_for_daemon_token(tmp_path, port=7679, timeout=1.0) == "new-B"
    assert state["n"] >= 2


def test_wait_for_daemon_token_times_out_to_none(tmp_path, monkeypatch):
    """A daemon that never authenticates yields None within the bounded timeout."""
    (tmp_path / "daemon.token").write_text("A\n")
    monkeypatch.delenv("POPPY_DAEMON_TOKEN", raising=False)
    monkeypatch.setattr(lifecycle, "probe_status", lambda *_a, **_k: (None, "down"))

    assert lifecycle.wait_for_daemon_token(tmp_path, port=7679, timeout=0.15) is None


def test_restart_stop_timeout_exits_one_without_start(tmp_path, monkeypatch):
    units = tmp_path / "units"
    monkeypatch.setenv("POPPY_SYSTEMD_USER_DIR", str(units))
    monkeypatch.setattr(lifecycle.sys, "platform", "linux")
    path = lifecycle.agent_path("linux")
    path.parent.mkdir(parents=True)
    path.write_text("unit")
    calls = _record_commands(monkeypatch)
    monkeypatch.setattr(lifecycle, "lock_status", lambda _path: (True, 4321))
    monkeypatch.setattr(lifecycle, "_wait_for_daemon_stop", lambda *_args: False)

    result = CliRunner().invoke(cli, ["daemon", "restart"], env={"POPPY_DIR": str(tmp_path / "poppy")})

    assert result.exit_code == 1
    assert "PID 4321" in result.output
    assert calls == [["systemctl", "--user", "stop", lifecycle.SYSTEMD_FILENAME]]


def test_daemon_status_reports_lock_reachable_and_installed(tmp_path, monkeypatch):
    units = tmp_path / "units"
    monkeypatch.setenv("POPPY_SYSTEMD_USER_DIR", str(units))
    path = lifecycle.agent_path("linux")
    path.parent.mkdir(parents=True)
    path.write_text("unit")
    monkeypatch.setattr(lifecycle.sys, "platform", "linux")
    lock = tmp_path / "daemon.lock"
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    os.write(fd, f"{os.getpid()}\n".encode())
    fcntl.flock(fd, fcntl.LOCK_EX)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"version": "1.2.3"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError:
        os.close(fd)
        pytest.skip("sandbox forbids loopback socket binds")
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = CliRunner().invoke(
            cli,
            ["daemon", "status"],
            env={"POPPY_DIR": str(tmp_path), "POPPY_DAEMON_PORT": str(port)},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        os.close(fd)

    assert result.exit_code == 0, result.output
    assert "Agent installed: yes" in result.output
    assert f"Lock held: yes (PID {os.getpid()})" in result.output
    assert "HTTP reachable: yes" in result.output


def test_daemon_status_nothing_running_always_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_SYSTEMD_USER_DIR", str(tmp_path / "units"))
    monkeypatch.setattr(lifecycle.sys, "platform", "linux")
    monkeypatch.setattr(lifecycle, "probe_status", lambda *_args, **_kwargs: (None, "offline"))
    result = CliRunner().invoke(cli, ["daemon", "status"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert "Agent installed: no" in result.output
    assert "Lock held: no" in result.output
    assert "HTTP reachable: no" in result.output


def test_setup_claude_code_daemon_is_idempotent_and_secures_token(tmp_path, monkeypatch):
    poppy_dir = tmp_path / "poppy"
    claude_dir = tmp_path / ".claude"
    calls = []
    monkeypatch.setattr(lifecycle, "install_agent", lambda path: calls.append(("install", path)))
    monkeypatch.setattr(lifecycle, "start_daemon", lambda path: calls.append(("start", path)) or "started")
    # The daemon is mocked away, so make the readiness probe report authenticated:
    # setup bakes the token load_daemon_token reads from the real (minted) file.
    monkeypatch.setattr(lifecycle, "probe_status", lambda *_args, **_kwargs: ({"version": "t"}, None))

    env = {"POPPY_DIR": str(poppy_dir), "CLAUDE_CONFIG_DIR": str(claude_dir)}
    runner = CliRunner()
    args = ["setup", "claude-code", "--daemon", "--no-hooks", "--no-claude-md"]
    first = runner.invoke(cli, args, env=env)
    second = runner.invoke(cli, args, env=env)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    token_path = poppy_dir / "daemon.token"
    token = token_path.read_text().strip()
    assert len(token) == 64
    assert token_path.stat().st_mode & 0o777 == 0o600
    settings = json.loads((tmp_path / ".claude.json").read_text())
    assert settings["mcpServers"]["poppy"] == {
        "type": "http",
        "url": "http://127.0.0.1:7679/mcp",
        "headers": {"Authorization": f"Bearer {token}"},
    }
    assert token not in first.output + second.output
    assert calls == [("install", poppy_dir), ("start", poppy_dir)] * 2


def test_setup_claude_code_daemon_failure_preserves_client_config(tmp_path, monkeypatch):
    poppy_dir = tmp_path / "poppy"
    claude_dir = tmp_path / ".claude"
    config_path = tmp_path / ".claude.json"
    original = '{"mcpServers": {"existing": {"command": "keep"}}}\n'
    config_path.write_text(original)
    monkeypatch.setattr(lifecycle, "install_agent", lambda _path: None)

    def fail_start(_path):
        raise lifecycle.LifecycleError(
            "start the Poppy daemon",
            ["systemctl", "--user", "start", lifecycle.SYSTEMD_FILENAME],
            1,
            "unit failed",
        )

    monkeypatch.setattr(lifecycle, "start_daemon", fail_start)
    result = CliRunner().invoke(
        cli,
        ["setup", "claude-code", "--daemon", "--no-hooks", "--no-claude-md"],
        env={"POPPY_DIR": str(poppy_dir), "CLAUDE_CONFIG_DIR": str(claude_dir)},
    )

    assert result.exit_code == 1
    assert "client config was not changed" in result.output
    assert config_path.read_text() == original


def test_default_setup_entry_remains_stdio(tmp_path):
    from poppy.setup.claude_code import install_mcp_config

    path = install_mcp_config(tmp_path / ".claude", client="claude-code")
    entry = json.loads(path.read_text())["mcpServers"]["poppy"]
    assert entry["type"] == "stdio"
    assert entry["args"] == ["serve", "--source", "claude-code"]
