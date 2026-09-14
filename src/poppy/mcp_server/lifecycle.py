"""OS service installation, process control, and daemon status probing."""

from __future__ import annotations

import json
import os
import plistlib
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from poppy.mcp_server.auth import load_daemon_token
from poppy.mcp_server.daemon import DAEMON_LOCK_FILENAME

LAUNCHD_LABEL = "ai.trags.poppy.daemon"
LAUNCHD_FILENAME = f"{LAUNCHD_LABEL}.plist"
SYSTEMD_FILENAME = "poppy-daemon.service"
DEFAULT_DAEMON_PORT = 7679
DEFAULT_LIFECYCLE_TIMEOUT = 5.0
SERVICE_STDERR_TAIL = 1000


class LifecycleError(Exception):
    """A service action failed or did not reach its expected state."""

    def __init__(
        self,
        action: str,
        argv: list[str],
        returncode: int | None,
        stderr: str = "",
        *,
        detail: str | None = None,
    ) -> None:
        self.action = action
        self.argv = argv
        self.returncode = returncode
        self.stderr = stderr.strip()
        command = " ".join(argv)
        if detail is None:
            detail = f"Failed to {action}: `{command}` exited with status {returncode}."
        if self.stderr:
            detail = f"{detail}\nstderr: {self.stderr[-SERVICE_STDERR_TAIL:]}"
        super().__init__(detail)


@dataclass(frozen=True)
class DaemonState:
    """A non-mutating snapshot used by the CLI and doctor."""

    installed: bool
    lock_held: bool
    pid: int | None
    reachable: bool
    status: dict[str, Any] | None
    error: str | None = None


def daemon_port() -> int:
    """Return the configured daemon port, falling back on malformed input."""
    try:
        return int(os.environ.get("POPPY_DAEMON_PORT", DEFAULT_DAEMON_PORT))
    except ValueError:
        return DEFAULT_DAEMON_PORT


def launch_agents_dir() -> Path:
    override = os.environ.get("POPPY_LAUNCH_AGENTS_DIR")
    return Path(override) if override else Path.home() / "Library" / "LaunchAgents"


def systemd_user_dir() -> Path:
    override = os.environ.get("POPPY_SYSTEMD_USER_DIR")
    return Path(override) if override else Path.home() / ".config" / "systemd" / "user"


def agent_path(platform: str | None = None) -> Path:
    platform = sys.platform if platform is None else platform
    if platform == "darwin":
        return launch_agents_dir() / LAUNCHD_FILENAME
    if platform.startswith("linux"):
        return systemd_user_dir() / SYSTEMD_FILENAME
    raise RuntimeError(f"Daemon service installation is not supported on {platform}.")


def resolve_poppy_executable() -> str:
    """Resolve the console script without mistaking pytest/python for Poppy."""
    argv0 = Path(sys.argv[0])
    if argv0.name == "poppy" and argv0.exists():
        return str(argv0.resolve())
    found = shutil.which("poppy")
    if found:
        return str(Path(found).resolve())
    sibling = Path(sys.executable).parent / "poppy"
    if sibling.is_file():
        return str(sibling.resolve())
    return "poppy"


def _run_service_command(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Single monkeypatch seam for every launchctl/systemctl invocation."""
    return subprocess.run(argv, check=False, capture_output=True, text=True)


def _checked_service_command(action: str, argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a service command and turn a nonzero exit into a lifecycle failure."""
    result = _run_service_command(argv)
    if result.returncode != 0:
        raise LifecycleError(action, argv, result.returncode, result.stderr)
    return result


def _spawn_detached(argv: list[str], log_path: Path) -> subprocess.Popen[bytes]:
    """Single monkeypatch seam for the unsupervised fallback process."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab")
    try:
        return subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log.close()


def _signal_process(pid: int, sig: signal.Signals) -> None:
    """Monkeypatch seam for process signaling."""
    os.kill(pid, sig)


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_pid(poppy_dir: Path) -> int | None:
    try:
        return int((poppy_dir / DAEMON_LOCK_FILENAME).read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return None


def lock_status(poppy_dir: Path) -> tuple[bool, int | None]:
    """Return whether the daemon lock is held and the PID written into it."""
    path = poppy_dir / DAEMON_LOCK_FILENAME
    pid = _read_pid(poppy_dir)
    try:
        import fcntl

        fd = os.open(path, os.O_RDWR)
    except (FileNotFoundError, OSError, ImportError):
        return False, pid
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True, pid
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        os.close(fd)
    return False, pid


class _Unset:
    """Sentinel type: `token` unspecified means 'use the store's daemon token',
    distinct from an explicit ``None`` (no auth). A dedicated type keeps a type
    checker happy where a bare ``Any`` object would not."""


_TOKEN_UNSET = _Unset()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Block HTTP redirects on the authenticated probe so a 3xx never re-sends the
    client's bearer token to another host. Without this, a `/status` that returns
    `302 Location: http://collector.example/leak` would make urllib forward the
    Authorization header off-box (security). Raising turns the
    redirect into an error the caller reports as an unhealthy registration; the
    token is only ever sent to the exact validated loopback host:port."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        raise urllib.error.HTTPError(req.full_url, code, f"redirect blocked ({code})", headers, fp)


# A LOCKED-DOWN opener for the authenticated probe: the bearer token must reach
# ONLY the exact validated loopback host:port and nowhere else. build_opener's
# defaults would (a) follow redirects and (b) route through $http_proxy — and a
# proxy is not matched by `no_proxy=127.0.0.1,localhost` for an IPv6-loopback
# (`[::1]`) URL, so the token would be sent to the proxy. We therefore pass an
# EMPTY ProxyHandler (no proxies, ignore env) and the no-redirect handler; the
# default set adds no cookie or auth-retry handler.
_NO_REDIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)

# Whole-probe wall-clock deadline and a response-size cap. The /status body is
# tiny, so a few KB is ample; the deadline bounds a slowloris peer that the
# per-read socket timeout can't. Module-level so tests can tune.
_PROBE_DEADLINE_S = 3.0
_PROBE_MAX_BYTES = 64 * 1024


def probe_status(
    poppy_dir: Path,
    port: int | None = None,
    timeout: float = 0.25,
    *,
    token: str | None | _Unset = _TOKEN_UNSET,
    host: str = "127.0.0.1",
    auth_header: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Probe the authenticated local status endpoint without third-party HTTP dependencies.

    ``token`` defaults to the store's current ``daemon.token`` and is sent as a
    reconstructed ``Bearer <token>`` header. ``host`` is the EXACT loopback host
    to connect to; doctor passes the client's registered host (e.g. ``::1``)
    rather than rewriting it, so an IPv6 registration is checked against the
    address the client will actually use.

    ``auth_header``, when given, is sent as the ``Authorization`` header VERBATIM
    (and ``token`` is ignored) — doctor passes a client's stored header byte-for-
    byte so a malformed credential (``"Bearer  secret"`` with a double space,
    wrong case, ...) reproduces the SAME 401 the client gets instead of being
    normalized into a false 200.
    """
    # TOTAL by design: this is a best-effort, side-effect-free reachability check
    # whose only question is "reachable + authenticated?". ANY failure — a network
    # error (OSError/URLError), a blocked redirect (HTTPError), a truncated or
    # garbage body (http.client.IncompleteRead / JSONDecodeError), an unencodable
    # header, or anything else — means "no". A blanket except can't mask a real
    # bug (no side effects; the result is only data the caller renders as OK/WARN)
    # and it guarantees a broken endpoint can never abort doctor.
    result: list[tuple[dict[str, Any] | None, str | None]] = []

    def _run() -> None:
        try:
            resolved_port = daemon_port() if port is None else port
            # Bracket an IPv6 literal so it forms a valid URL authority (``[::1]``).
            host_for_url = f"[{host}]" if ":" in host else host
            request = urllib.request.Request(f"http://{host_for_url}:{resolved_port}/status")
            if auth_header is not None:
                request.add_header("Authorization", auth_header)  # verbatim, as the client sends it
            else:
                resolved_token = load_daemon_token(poppy_dir) if isinstance(token, _Unset) else token
                if resolved_token:
                    request.add_header("Authorization", f"Bearer {resolved_token}")
            # No-redirect opener: a 3xx must never forward the bearer off-box. Cap
            # the read at MAX_BYTES+1 so a flood peer can't OOM us — the /status
            # body is tiny.
            with _NO_REDIRECT_OPENER.open(request, timeout=timeout) as response:
                body = response.read(_PROBE_MAX_BYTES + 1)
                # `read(amt)` returns a PARTIAL body instead of raising
                # IncompleteRead, so a positive remaining length means the server
                # promised (Content-Length) more than it sent — truncated/broken.
                remaining = getattr(response, "length", None)
            if len(body) > _PROBE_MAX_BYTES:
                result.append((None, "status response too large"))
                return
            if remaining:
                result.append((None, "incomplete response body"))
                return
            payload = json.loads(body)
            result.append((payload if isinstance(payload, dict) else None, None))
        except Exception as exc:  # noqa: BLE001 — intentional total guard (see above)
            # A blocked redirect raises HTTPError holding the live response fp;
            # close it so the socket isn't leaked (no ResourceWarning).
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
            result.append((None, str(exc)))

    # The per-read socket timeout resets on every byte, so a slowloris peer
    # (trickling one byte at a time, never closing) would hang the read forever.
    # Bound the WHOLE probe with a wall-clock deadline: run it in a daemon thread
    # and abandon it if it overruns, so a hostile endpoint can never freeze doctor.
    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(_PROBE_DEADLINE_S)
    if worker.is_alive():
        return None, f"probe exceeded {_PROBE_DEADLINE_S:g}s deadline"
    return result[0] if result else (None, "probe produced no result")


def wait_for_daemon_token(poppy_dir: Path, *, port: int | None = None, timeout: float | None = None) -> str | None:
    """Poll ``/status`` until the daemon serves a settled token, then return it.

    `setup --daemon` must configure a client with the token the RUNNING daemon
    serves. But when the service is started by install_agent, THAT process holds
    the run-lock and rotates the file to its own env token asynchronously, while
    setup's own ``start_daemon`` short-circuits on the held lock and returns before
    the rotation lands. So do not trust a bare file read: poll the
    authenticated status endpoint, re-reading ``daemon.token`` each attempt (so a
    rotation is picked up), until a probe authenticates AND the on-disk token is
    unchanged across the probe. A 200 means the on-disk token IS what the daemon
    serves; the stability check rejects a value captured mid-rotation. Returns the
    settled token, or None if the daemon never became ready within ``timeout``.
    """
    port = daemon_port() if port is None else port
    timeout = DEFAULT_LIFECYCLE_TIMEOUT if timeout is None else timeout
    deadline = time.monotonic() + timeout
    while True:
        token = load_daemon_token(poppy_dir)
        status, _ = probe_status(poppy_dir, port)
        # 200 (status is not None) proves the token probe_status just sent
        # authenticated; the re-read guards against a rotation landing mid-probe.
        if token and status is not None and load_daemon_token(poppy_dir) == token:
            return token
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(0.05, remaining))


def inspect_daemon(poppy_dir: Path, *, port: int | None = None, platform: str | None = None) -> DaemonState:
    """Collect all daemon signals; failures are data so status always succeeds."""
    try:
        installed = agent_path(platform).is_file()
    except RuntimeError:
        installed = False
    held, pid = lock_status(poppy_dir)
    status, error = probe_status(poppy_dir, port)
    return DaemonState(installed, held, pid, status is not None, status, error)


def _wait_for_daemon(poppy_dir: Path, timeout: float) -> bool:
    """Wait until the lock is held and the authenticated status probe succeeds."""
    deadline = time.monotonic() + timeout
    while True:
        held, _ = lock_status(poppy_dir)
        status, _ = probe_status(poppy_dir)
        if held and status is not None:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))


def _wait_for_daemon_stop(poppy_dir: Path, pid: int | None, timeout: float) -> bool:
    """Wait until the original daemon PID exits or releases the daemon lock."""
    deadline = time.monotonic() + timeout
    while True:
        held, _ = lock_status(poppy_dir)
        if not held or (pid is not None and not _process_exists(pid)):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))


def _launchd_plist(poppy_dir: Path, executable: str) -> bytes:
    log_path = poppy_dir / "logs" / "daemon.log"
    payload = {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": [executable, "daemon", "run"],
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(log_path),
        "StandardErrorPath": str(log_path),
    }
    return plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=False)


def _systemd_unit(executable: str) -> str:
    return (
        "[Unit]\n"
        "Description=Poppy MCP daemon\n\n"
        "[Service]\n"
        f"ExecStart={executable} daemon run\n"
        "Restart=on-failure\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def install_agent(
    poppy_dir: Path,
    *,
    platform: str | None = None,
    executable: str | None = None,
) -> tuple[Path, list[list[str]]]:
    """Write and load the current platform's per-user service definition."""
    platform = sys.platform if platform is None else platform
    executable = resolve_poppy_executable() if executable is None else executable
    path = agent_path(platform)
    path.parent.mkdir(parents=True, exist_ok=True)
    (poppy_dir / "logs").mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []
    if platform == "darwin":
        path.write_bytes(_launchd_plist(poppy_dir, executable))
        # Re-installing over a loaded label: bootstrap fails with EIO, and the
        # plist may have changed, so bootout first and let launchd re-read it.
        target = f"gui/{os.getuid()}/{LAUNCHD_LABEL}"
        if _run_service_command(["launchctl", "print", target]).returncode == 0:
            commands.append(["launchctl", "bootout", target])
        commands.append(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(path)])
    elif platform.startswith("linux"):
        path.write_text(_systemd_unit(executable), encoding="utf-8")
        commands.extend(
            [
                ["systemctl", "--user", "daemon-reload"],
                ["systemctl", "--user", "enable", "--now", SYSTEMD_FILENAME],
            ]
        )
    else:  # agent_path already rejects this, retained for type narrowing
        raise RuntimeError(f"Daemon service installation is not supported on {platform}.")
    for command in commands:
        _checked_service_command("install the Poppy daemon agent", command)
    return path, commands


def uninstall_agent(*, platform: str | None = None) -> tuple[Path, list[list[str]]]:
    """Unload, disable, and remove the current platform's service definition."""
    platform = sys.platform if platform is None else platform
    path = agent_path(platform)
    commands: list[list[str]] = []
    if platform == "darwin":
        # Only bootout a label launchd actually has loaded; a plist file left
        # on disk without a loaded label must not fail the uninstall.
        target = f"gui/{os.getuid()}/{LAUNCHD_LABEL}"
        if path.exists() and _run_service_command(["launchctl", "print", target]).returncode == 0:
            commands.append(["launchctl", "bootout", target])
    else:
        if path.exists():
            commands.append(["systemctl", "--user", "disable", "--now", SYSTEMD_FILENAME])
    for command in commands:
        _checked_service_command("uninstall the Poppy daemon agent", command)
    path.unlink(missing_ok=True)
    if platform.startswith("linux"):
        reload_command = ["systemctl", "--user", "daemon-reload"]
        _checked_service_command("reload the systemd user manager", reload_command)
        commands.append(reload_command)
    return path, commands


def kickstart_agent(*, platform: str | None = None) -> list[str]:
    """Ask the installed per-user service manager to start the daemon."""
    platform = sys.platform if platform is None else platform
    if platform == "darwin":
        command = ["launchctl", "kickstart", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"]
    elif platform.startswith("linux"):
        command = ["systemctl", "--user", "start", SYSTEMD_FILENAME]
    else:
        raise RuntimeError(f"Daemon service installation is not supported on {platform}.")
    _checked_service_command("kickstart daemon", command)
    return command


def start_daemon(
    poppy_dir: Path,
    *,
    platform: str | None = None,
    timeout: float = DEFAULT_LIFECYCLE_TIMEOUT,
) -> str:
    """Start the service manager unit, or a detached foreground runner fallback."""
    platform = sys.platform if platform is None else platform
    state = inspect_daemon(poppy_dir, platform=platform)
    if state.lock_held or state.reachable:
        return f"Poppy daemon is already running{f' (PID {state.pid})' if state.pid else ''}."
    try:
        installed = agent_path(platform).is_file()
    except RuntimeError:
        installed = False
    command: list[str]
    result: subprocess.CompletedProcess[str] | None = None
    if installed and platform == "darwin":
        # `stop` uses bootout, which unregisters the job, so start must bootstrap
        # the definition again before asking launchd to kickstart it. But
        # install_agent (or a previous start) may have left the label loaded,
        # and bootstrapping a loaded label fails with EIO — probe first so
        # start stays idempotent in the install -> start flow.
        probe = _run_service_command(["launchctl", "print", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"])
        if probe.returncode != 0:
            bootstrap = ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(agent_path(platform))]
            _checked_service_command("bootstrap the Poppy daemon agent", bootstrap)
        command = ["launchctl", "kickstart", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"]
        result = _checked_service_command("start the Poppy daemon", command)
        success = "Started Poppy daemon with launchd."
    elif installed and platform.startswith("linux"):
        command = ["systemctl", "--user", "start", SYSTEMD_FILENAME]
        result = _checked_service_command("start the Poppy daemon", command)
        success = "Started Poppy daemon with systemd."
    else:
        command = [resolve_poppy_executable(), "daemon", "run"]
        _spawn_detached(command, poppy_dir / "logs" / "daemon.log")
        success = "Started Poppy daemon as a detached process."
    if not _wait_for_daemon(poppy_dir, timeout):
        log_path = poppy_dir / "logs" / "daemon.log"
        raise LifecycleError(
            "start the Poppy daemon",
            command,
            result.returncode if result is not None else None,
            result.stderr if result is not None else "",
            detail=f"Poppy daemon did not become reachable within {timeout:g} seconds. Check {log_path}.",
        )
    return success


def stop_daemon(poppy_dir: Path, *, platform: str | None = None, timeout: float = 5.0) -> str:
    """Stop the service manager unit, or terminate the lock-owning fallback PID."""
    platform = sys.platform if platform is None else platform
    try:
        installed = agent_path(platform).is_file()
    except RuntimeError:
        installed = False
    held, pid = lock_status(poppy_dir)
    command: list[str] | None = None
    result: subprocess.CompletedProcess[str] | None = None
    if installed and platform == "darwin":
        command = ["launchctl", "bootout", f"gui/{os.getuid()}/{LAUNCHD_LABEL}"]
        result = _checked_service_command("stop the Poppy daemon", command)
        success = "Stopped Poppy daemon with launchd."
    elif installed and platform.startswith("linux"):
        command = ["systemctl", "--user", "stop", SYSTEMD_FILENAME]
        result = _checked_service_command("stop the Poppy daemon", command)
        success = "Stopped Poppy daemon with systemd."
    elif not held or pid is None or not _process_exists(pid):
        return "Poppy daemon is already stopped."
    else:
        _signal_process(pid, signal.SIGTERM)
        command = ["signal", str(pid), "SIGTERM"]
        success = f"Stopped Poppy daemon PID {pid}."
    if not _wait_for_daemon_stop(poppy_dir, pid, timeout):
        pid_detail = f" PID {pid}" if pid is not None else ""
        raise LifecycleError(
            "stop the Poppy daemon",
            command,
            result.returncode if result is not None else None,
            result.stderr if result is not None else "",
            detail=f"Timed out waiting for Poppy daemon{pid_detail} to stop.",
        )
    return success
