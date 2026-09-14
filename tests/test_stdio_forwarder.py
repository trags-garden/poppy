from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest
import uvicorn
from click.testing import CliRunner
from mcp import ClientSession, types
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.message import SessionMessage

from poppy import __version__
from poppy.cli.main import cli
from poppy.engine.seed import SeedEngine
from poppy.mcp_server import lifecycle
from poppy.mcp_server.auth import BearerAuthMiddleware
from poppy.mcp_server.daemon import bind_socket, create_http_app
from poppy.mcp_server.forwarder import _bridge, _BridgeState
from poppy.mcp_server.server import create_mcp_server


class _FakeMcp:
    def __init__(self) -> None:
        self.run_calls: list[dict[str, str]] = []

    def run(self, **kwargs) -> None:
        self.run_calls.append(kwargs)


def _patch_local_server(monkeypatch) -> _FakeMcp:
    fake = _FakeMcp()
    monkeypatch.setattr("poppy.mcp_server.server.create_mcp_server", lambda **_kwargs: fake)
    return fake


def test_daemon_down_without_agent_falls_back_without_kickstart(tmp_path, monkeypatch):
    fake = _patch_local_server(monkeypatch)
    monkeypatch.setattr(lifecycle, "probe_status", lambda *_args, **_kwargs: (None, "offline"))
    monkeypatch.setattr(lifecycle, "agent_path", lambda: tmp_path / "missing-agent")
    monkeypatch.setattr(lifecycle, "kickstart_agent", lambda: pytest.fail("kickstart must not run"))

    result = CliRunner().invoke(cli, ["serve"], env={"POPPY_DIR": str(tmp_path)})

    assert result.exit_code == 0, result.output
    assert "Poppy daemon unavailable; using in-process server." in result.output
    assert fake.run_calls == [{"transport": "stdio"}]


def test_installed_agent_is_kickstarted_then_bounded_fallback(tmp_path, monkeypatch):
    fake = _patch_local_server(monkeypatch)
    agents = tmp_path / "agents"
    monkeypatch.setenv("POPPY_LAUNCH_AGENTS_DIR", str(agents))
    monkeypatch.setattr(lifecycle.sys, "platform", "darwin")
    agent = lifecycle.agent_path()
    agent.parent.mkdir(parents=True)
    agent.write_text("installed")
    probes: list[int] = []
    monkeypatch.setattr(
        lifecycle,
        "probe_status",
        lambda *_args, **_kwargs: (probes.append(1) and None, "offline"),
    )
    commands: list[list[str]] = []

    def _record_success(argv: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(lifecycle, "_run_service_command", _record_success)

    result = CliRunner().invoke(
        cli,
        ["serve"],
        env={"POPPY_DIR": str(tmp_path), "POPPY_DAEMON_POLL_TIMEOUT": "0"},
    )

    assert result.exit_code == 0, result.output
    assert len(probes) == 2
    assert commands == [["launchctl", "kickstart", f"gui/{os.getuid()}/{lifecycle.LAUNCHD_LABEL}"]]
    assert "Poppy daemon unavailable after kickstart; using in-process server." in result.output
    assert fake.run_calls == [{"transport": "stdio"}]


@pytest.mark.parametrize(
    "args,extra_env",
    [
        (["--no-daemon"], {}),
        ([], {"POPPY_SERVE_NO_DAEMON": "1"}),
    ],
)
def test_no_daemon_switches_skip_probe(tmp_path, monkeypatch, args, extra_env):
    fake = _patch_local_server(monkeypatch)
    monkeypatch.setattr(lifecycle, "probe_status", lambda *_args, **_kwargs: pytest.fail("probe must not run"))

    result = CliRunner().invoke(cli, ["serve", *args], env={"POPPY_DIR": str(tmp_path), **extra_env})

    assert result.exit_code == 0, result.output
    assert fake.run_calls == [{"transport": "stdio"}]


def test_reachable_daemon_forwards_and_warns_on_version_skew(tmp_path, monkeypatch):
    monkeypatch.setattr(lifecycle, "probe_status", lambda *_args, **_kwargs: ({"version": "99.0"}, None))
    calls: list[tuple[Path, int]] = []
    monkeypatch.setattr("poppy.mcp_server.forwarder.run_forwarder", lambda path, port: calls.append((path, port)) or 0)
    monkeypatch.setattr(
        "poppy.mcp_server.server.create_mcp_server",
        lambda **_kwargs: pytest.fail("forwarding must not construct the local MCP server"),
    )

    result = CliRunner().invoke(cli, ["serve", "--port", "8123"], env={"POPPY_DIR": str(tmp_path)})

    assert result.exit_code == 0, result.output
    assert calls == [(tmp_path, 8123)]
    assert f"Warning: Poppy CLI {__version__} is forwarding to daemon 99.0." in result.output


def test_reachable_daemon_with_old_mcp_sdk_falls_back_in_process(tmp_path, monkeypatch):
    fake = _patch_local_server(monkeypatch)
    monkeypatch.setattr(lifecycle, "probe_status", lambda *_args, **_kwargs: ({"version": __version__}, None))
    monkeypatch.setitem(sys.modules, "poppy.mcp_server.forwarder", None)

    result = CliRunner().invoke(cli, ["serve"], env={"POPPY_DIR": str(tmp_path)})

    assert result.exit_code == 0, result.output
    assert "MCP SDK too old for daemon forwarding; using in-process server." in result.output
    assert fake.run_calls == [{"transport": "stdio"}]


@pytest.mark.asyncio
async def test_bridge_drains_late_response_after_clean_stdin_eof():
    local_send, local_read = anyio.create_memory_object_stream[SessionMessage | Exception](1)
    local_write, local_receive = anyio.create_memory_object_stream[SessionMessage](1)
    daemon_send, daemon_read = anyio.create_memory_object_stream[SessionMessage | Exception](1)
    daemon_write, daemon_receive = anyio.create_memory_object_stream[SessionMessage](1)
    request = SessionMessage(
        types.JSONRPCMessage(
            types.JSONRPCRequest(
                jsonrpc="2.0",
                id="original-id",
                method="initialize",
                params={"clientInfo": {"name": "stdio-client", "version": "1"}},
            )
        )
    )
    response = SessionMessage(
        types.JSONRPCMessage(types.JSONRPCResponse(jsonrpc="2.0", id="original-id", result={"ok": True}))
    )
    result: list[int] = []
    state = _BridgeState()

    async with anyio.create_task_group() as task_group:

        async def run_bridge() -> None:
            result.append(await _bridge(local_read, local_write, daemon_read, daemon_write, state))

        task_group.start_soon(run_bridge)
        await local_send.send(request)
        assert await daemon_receive.receive() is request
        await local_send.aclose()
        with anyio.fail_after(1):
            while not state.stdin_eof:
                await anyio.sleep(0)
        await daemon_send.send(response)
        assert await local_receive.receive() is response

    assert result == [0]


@pytest.mark.asyncio
async def test_bridge_times_out_clean_eof_drain_with_one_connection_error():
    local_send, local_read = anyio.create_memory_object_stream[SessionMessage | Exception](1)
    local_write, local_receive = anyio.create_memory_object_stream[SessionMessage](1)
    _daemon_send, daemon_read = anyio.create_memory_object_stream[SessionMessage | Exception](1)
    daemon_write, daemon_receive = anyio.create_memory_object_stream[SessionMessage](1)
    request = SessionMessage(
        types.JSONRPCMessage(types.JSONRPCRequest(jsonrpc="2.0", id="pending-id", method="tools/list", params={}))
    )
    result: list[int] = []

    async with anyio.create_task_group() as task_group:

        async def run_bridge() -> None:
            result.append(await _bridge(local_read, local_write, daemon_read, daemon_write, _BridgeState(), 0.01))

        task_group.start_soon(run_bridge)
        await local_send.send(request)
        assert await daemon_receive.receive() is request
        await local_send.aclose()
        error = await local_receive.receive()
        assert isinstance(error.message.root, types.JSONRPCError)
        assert error.message.root.id == "pending-id"
        assert error.message.root.error.code == types.CONNECTION_CLOSED

    assert result == [1]
    with pytest.raises(anyio.WouldBlock):
        local_receive.receive_nowait()


@pytest.mark.asyncio
async def test_bridge_claims_each_pending_id_once_when_both_pumps_fail():
    fail = anyio.Event()

    async def failing_read():
        await fail.wait()
        raise OSError("stream failed")
        yield  # pragma: no cover - makes this an async generator

    class RacingWrite:
        def __init__(self) -> None:
            self.arrivals = 0
            self.release = anyio.Event()
            self.messages: list[SessionMessage] = []

        async def send(self, message: SessionMessage) -> None:
            self.arrivals += 1
            if self.arrivals == 2:
                self.release.set()
            await self.release.wait()
            self.messages.append(message)

    local_write = RacingWrite()
    daemon_write, _daemon_receive = anyio.create_memory_object_stream[SessionMessage](1)
    state = _BridgeState(pending={1, 2})
    fail.set()

    result = await _bridge(failing_read(), local_write, failing_read(), daemon_write, state)

    roots = [message.message.root for message in local_write.messages]
    assert result == 1
    assert all(isinstance(root, types.JSONRPCError) for root in roots)
    assert sorted(root.id for root in roots) == [1, 2]
    assert all(root.error.code == types.CONNECTION_CLOSED for root in roots)


@pytest.mark.asyncio
async def test_bridge_returns_error_for_pending_request_when_daemon_disconnects():
    local_send, local_read = anyio.create_memory_object_stream[SessionMessage | Exception](1)
    local_write, local_receive = anyio.create_memory_object_stream[SessionMessage](1)
    daemon_send, daemon_read = anyio.create_memory_object_stream[SessionMessage | Exception](1)
    daemon_write, daemon_receive = anyio.create_memory_object_stream[SessionMessage](1)
    request = SessionMessage(
        types.JSONRPCMessage(types.JSONRPCRequest(jsonrpc="2.0", id=42, method="tools/list", params={}))
    )
    result: list[int] = []

    async with anyio.create_task_group() as task_group:

        async def run_bridge() -> None:
            result.append(await _bridge(local_read, local_write, daemon_read, daemon_write, _BridgeState()))

        task_group.start_soon(run_bridge)
        await local_send.send(request)
        assert await daemon_receive.receive() is request
        await daemon_send.aclose()
        error = await local_receive.receive()
        assert isinstance(error.message.root, types.JSONRPCError)
        assert error.message.root.id == 42
        assert error.message.root.error.code == types.CONNECTION_CLOSED

    assert result == [1]


@pytest.mark.asyncio
async def test_daemon_down_without_agent_runs_working_local_stdio_server(tmp_path):
    poppy_dir = tmp_path / "poppy"
    poppy_dir.mkdir()
    (poppy_dir / "config.json").write_text('{"engine": "seed"}')
    stderr_path = tmp_path / "fallback.stderr"
    executable = Path(sys.executable).parent / "poppy"
    env = {
        "POPPY_DIR": str(poppy_dir),
        "POPPY_DAEMON_PORT": "1",
        "POPPY_LAUNCH_AGENTS_DIR": str(tmp_path / "agents"),
        "POPPY_TELEMETRY_OFF": "1",
    }
    params = StdioServerParameters(command=str(executable), args=["serve"], env=env, cwd=Path.cwd())

    with stderr_path.open("w+") as stderr:
        async with stdio_client(params, errlog=stderr) as (read_stream, write_stream):
            async with ClientSession(
                read_stream,
                write_stream,
                client_info=types.Implementation(name="fallback-client", version="1"),
            ) as session:
                await session.initialize()
                remembered = await session.call_tool("remember", {"content": "local fallback works"})
                assert remembered.isError is not True
                context = await session.call_tool("context", {"limit": 5})
                assert context.isError is not True
                assert "local fallback works" in context.content[0].text

    assert "Poppy daemon unavailable; using in-process server." in stderr_path.read_text()
    memory = SeedEngine(poppy_dir / "memories.db").list_all()[0]
    assert memory.source.type == "fallback-client"


def _start_daemon(poppy_dir: Path, token: str):
    engine = SeedEngine(poppy_dir / "memories.db")
    mcp = create_mcp_server(
        poppy_dir=poppy_dir,
        source="mcp",
        engine=engine,
        host="127.0.0.1",
        port=0,
        json_response=True,
        stateless_http=False,
    )
    try:
        listener = bind_socket("127.0.0.1", 0)
    except PermissionError:
        pytest.skip("sandbox forbids loopback socket binds")
    port = listener.getsockname()[1]
    app = BearerAuthMiddleware(
        create_http_app(mcp.streamable_http_app(), engine=engine, poppy_dir=poppy_dir, port=port),
        token=token,
        path=(mcp.settings.streamable_http_path, "/status"),
    )
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    errors: list[BaseException] = []

    def run() -> None:
        try:
            server.run(sockets=[listener])
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started, errors
    return SimpleNamespace(engine=engine, listener=listener, port=port, server=server, thread=thread, errors=errors)


def _stop_daemon(harness) -> None:
    harness.server.should_exit = True
    harness.thread.join(timeout=5)
    harness.listener.close()
    assert not harness.thread.is_alive()
    assert not harness.errors


@pytest.mark.asyncio
async def test_stdio_subprocess_forwards_client_identity_and_loads_no_engine(tmp_path):
    token = "forwarder-test-token"
    poppy_dir = tmp_path / "poppy"
    poppy_dir.mkdir()
    (poppy_dir / "daemon.token").write_text(f"{token}\n")
    harness = _start_daemon(poppy_dir, token)
    stderr_path = tmp_path / "forwarder.stderr"
    executable = Path(sys.executable).parent / "poppy"
    env = {
        "POPPY_DIR": str(poppy_dir),
        "POPPY_DAEMON_PORT": str(harness.port),
        "POPPY_LAUNCH_AGENTS_DIR": str(tmp_path / "agents"),
        "POPPY_TEST_FAIL_ON_ENGINE": "1",
        "POPPY_TELEMETRY_OFF": "1",
    }
    params = StdioServerParameters(command=str(executable), args=["serve"], env=env, cwd=Path.cwd())

    try:
        with stderr_path.open("w+") as stderr:
            async with stdio_client(params, errlog=stderr) as (read_stream, write_stream):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    client_info=types.Implementation(name="forwarder-e2e", version="1"),
                ) as session:
                    initialized = await session.initialize()
                    assert initialized.serverInfo.name == "poppy"
                    remembered = await session.call_tool("remember", {"content": "forwarded identity"})
                    assert remembered.isError is not True
                    context = await session.call_tool("context", {"limit": 5})
                    assert context.isError is not True
                    assert "forwarded identity" in context.content[0].text
    finally:
        _stop_daemon(harness)

    memories = harness.engine.list_all()
    assert len(memories) == 1
    assert memories[0].source.type == "forwarder-e2e"
    assert memories[0].source.type not in {"mcp", "claude-code"}
    assert "POPPY_TEST_FAIL_ON_ENGINE" not in stderr_path.read_text()


@pytest.mark.asyncio
async def test_bridge_drops_late_response_once_drain_timeout_errored_it():
    local_send, local_read = anyio.create_memory_object_stream[SessionMessage | Exception](1)
    daemon_send, daemon_read = anyio.create_memory_object_stream[SessionMessage | Exception](1)
    daemon_write, daemon_receive = anyio.create_memory_object_stream[SessionMessage](1)

    class GatedRecordingWrite:
        def __init__(self) -> None:
            self.messages: list[SessionMessage] = []
            self.release = anyio.Event()

        async def send(self, message: SessionMessage) -> None:
            self.messages.append(message)
            await self.release.wait()

    local_write = GatedRecordingWrite()
    request = SessionMessage(
        types.JSONRPCMessage(types.JSONRPCRequest(jsonrpc="2.0", id="late-id", method="tools/call", params={}))
    )
    response = SessionMessage(
        types.JSONRPCMessage(types.JSONRPCResponse(jsonrpc="2.0", id="late-id", result={"ok": True}))
    )
    state = _BridgeState()
    result: list[int] = []

    async with anyio.create_task_group() as task_group:

        async def run_bridge() -> None:
            result.append(await _bridge(local_read, local_write, daemon_read, daemon_write, state, 0.01))

        task_group.start_soon(run_bridge)
        await local_send.send(request)
        assert await daemon_receive.receive() is request
        await local_send.aclose()
        # Wait until the drain deadline claimed the id and the connection error
        # send is in flight, blocked on the gate.
        with anyio.fail_after(1):
            while state.pending or not local_write.messages:
                await anyio.sleep(0)
        # The daemon answers only now, after the timeout already errored the id.
        await daemon_send.send(response)
        await daemon_send.aclose()
        # Deterministic sync point: the relay either forwards the duplicate
        # (bug, recorded as a second message) or drains and closes its daemon
        # side after dropping it.
        with anyio.fail_after(1):
            while len(local_write.messages) < 2:
                try:
                    daemon_receive.receive_nowait()
                except anyio.WouldBlock:
                    await anyio.sleep(0)
                except anyio.EndOfStream:
                    break
        local_write.release.set()

    assert result == [1]
    assert len(local_write.messages) == 1
    root = local_write.messages[0].message.root
    assert isinstance(root, types.JSONRPCError)
    assert root.id == "late-id"
    assert root.error.code == types.CONNECTION_CLOSED


@pytest.mark.asyncio
async def test_bridge_completes_in_flight_response_when_stdin_closes_mid_send():
    local_send, local_read = anyio.create_memory_object_stream[SessionMessage | Exception](1)
    local_write, local_receive = anyio.create_memory_object_stream[SessionMessage](0)
    daemon_send, daemon_read = anyio.create_memory_object_stream[SessionMessage | Exception](1)
    daemon_write, daemon_receive = anyio.create_memory_object_stream[SessionMessage](1)
    request = SessionMessage(
        types.JSONRPCMessage(types.JSONRPCRequest(jsonrpc="2.0", id="mid-send", method="tools/call", params={}))
    )
    response = SessionMessage(
        types.JSONRPCMessage(types.JSONRPCResponse(jsonrpc="2.0", id="mid-send", result={"ok": True}))
    )
    state = _BridgeState()
    result: list[int] = []

    async with anyio.create_task_group() as task_group:

        async def run_bridge() -> None:
            result.append(await _bridge(local_read, local_write, daemon_read, daemon_write, state))

        task_group.start_soon(run_bridge)
        await local_send.send(request)
        assert await daemon_receive.receive() is request
        await daemon_send.send(response)
        # The relay claims the id, then blocks delivering on the rendezvous
        # stream; stdin closes in exactly that window.
        with anyio.fail_after(1):
            while not state.in_flight:
                await anyio.sleep(0)
        await local_send.aclose()
        with anyio.fail_after(1):
            delivered = await local_receive.receive()
        assert delivered is response

    assert result == [0]


@pytest.mark.asyncio
async def test_bridge_errors_pending_request_when_daemon_ends_after_eof():
    local_send, local_read = anyio.create_memory_object_stream[SessionMessage | Exception](1)
    local_write, local_receive = anyio.create_memory_object_stream[SessionMessage](1)
    daemon_send, daemon_read = anyio.create_memory_object_stream[SessionMessage | Exception](1)
    daemon_write, daemon_receive = anyio.create_memory_object_stream[SessionMessage](1)
    request = SessionMessage(
        types.JSONRPCMessage(types.JSONRPCRequest(jsonrpc="2.0", id=7, method="tools/call", params={}))
    )
    result: list[int] = []

    async with anyio.create_task_group() as task_group:

        async def run_bridge() -> None:
            result.append(await _bridge(local_read, local_write, daemon_read, daemon_write, _BridgeState()))

        task_group.start_soon(run_bridge)
        await local_send.send(request)
        assert await daemon_receive.receive() is request
        # Client half-closes stdin while the request is still pending, then the
        # daemon stream ends cleanly without ever answering.
        await local_send.aclose()
        await daemon_send.aclose()
        error = await local_receive.receive()
        assert isinstance(error.message.root, types.JSONRPCError)
        assert error.message.root.id == 7
        assert error.message.root.error.code == types.CONNECTION_CLOSED

    assert result == [1]
