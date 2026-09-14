"""Raw stdio-to-streamable-HTTP bridge for the machine daemon."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import anyio
import httpx
from mcp import types
from mcp.client.streamable_http import streamablehttp_client
from mcp.server.stdio import stdio_server
from mcp.shared.message import SessionMessage

from poppy.mcp_server.auth import load_daemon_token

_DEFAULT_DRAIN_GRACE = 10.0


def _loopback_http_client(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    """httpx client for the LOOPBACK daemon that never uses an HTTP proxy.

    ``trust_env=False`` ignores ``$http_proxy`` so the connection to
    ``127.0.0.1`` is always DIRECT — matching doctor's no-proxy probe. A proxy
    that 403s loopback would otherwise break ``poppy serve`` (the forwarder) while
    doctor reported OK, and could exfiltrate the bearer token.
    Keeps the MCP SDK defaults (follow_redirects, 30s timeout) otherwise.
    """
    kwargs: dict = {"follow_redirects": True, "trust_env": False}
    kwargs["timeout"] = timeout if timeout is not None else httpx.Timeout(30.0)
    if headers is not None:
        kwargs["headers"] = headers
    if auth is not None:
        kwargs["auth"] = auth
    return httpx.AsyncClient(**kwargs)


class _DaemonDisconnected(Exception):
    """Internal control flow that cancels the SDK's blocking stdin reader."""


@dataclass
class _BridgeState:
    """Mutable state shared by the two one-way pumps."""

    pending: set[str | int] = field(default_factory=set)
    in_flight: bool = False
    stdin_eof: bool = False
    exit_code: int = 0


def _request_id(message: SessionMessage) -> str | int | None:
    root = message.message.root
    if isinstance(root, types.JSONRPCRequest):
        return root.id
    return None


def _response_id(message: SessionMessage) -> str | int | None:
    root = message.message.root
    if isinstance(root, types.JSONRPCResponse | types.JSONRPCError):
        return root.id
    return None


async def _send_connection_errors(write_stream, pending: set[str | int]) -> None:
    """Finish outstanding requests when the daemon transport disappears."""
    while pending:
        request_id = pending.pop()
        error = types.JSONRPCError(
            jsonrpc="2.0",
            id=request_id,
            error=types.ErrorData(code=types.CONNECTION_CLOSED, message="Poppy daemon connection closed"),
        )
        await write_stream.send(SessionMessage(types.JSONRPCMessage(error)))


async def _bridge(
    local_read,
    local_write,
    daemon_read,
    daemon_write,
    state: _BridgeState,
    drain_grace: float = _DEFAULT_DRAIN_GRACE,
) -> int:
    finished = anyio.Event()
    eof = anyio.Event()
    state_changed = anyio.Event()

    async def client_to_daemon() -> None:
        clean_eof = False
        try:
            async for message in local_read:
                if isinstance(message, Exception):
                    continue
                request_id = _request_id(message)
                if request_id is not None:
                    state.pending.add(request_id)
                await daemon_write.send(message)
            clean_eof = True
        except Exception:
            if not state.stdin_eof:
                state.exit_code = 1
                await _send_connection_errors(local_write, state.pending)
        finally:
            state.stdin_eof = True
            if clean_eof:
                eof.set()
                state_changed.set()
            # in_flight covers a claimed response mid-delivery: it is no longer
            # in pending but must not be cancelled by finishing the drain early.
            if not clean_eof or not (state.pending or state.in_flight):
                await daemon_write.aclose()
                finished.set()
                state_changed.set()

    async def daemon_to_client() -> None:
        try:
            async for message in daemon_read:
                if isinstance(message, Exception):
                    raise message
                response_id = _response_id(message)
                if response_id is None:
                    await local_write.send(message)
                elif response_id in state.pending:
                    # Claim the id synchronously before awaiting the send so the
                    # drain timeout can never also error it: each request gets
                    # exactly one terminal message.
                    state.pending.discard(response_id)
                    state.in_flight = True
                    try:
                        await local_write.send(message)
                    finally:
                        state.in_flight = False
                # else: this id already received a terminal error (drain timeout
                # or pump failure); forwarding the late result would hand the
                # client two responses for one request id.
                if state.stdin_eof and not state.pending:
                    break
        except Exception:
            if not state.stdin_eof:
                state.exit_code = 1
                await _send_connection_errors(local_write, state.pending)
        finally:
            if not state.stdin_eof:
                state.exit_code = 1
                await _send_connection_errors(local_write, state.pending)
            else:
                # The daemon stream can also end cleanly (no exception) while
                # drained-after-EOF requests are still outstanding; those must
                # get errors too, not a silent exit 0.
                if state.pending:
                    state.exit_code = 1
                    await _send_connection_errors(local_write, state.pending)
                await daemon_write.aclose()
            finished.set()
            state_changed.set()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(client_to_daemon)
        task_group.start_soon(daemon_to_client)
        await state_changed.wait()
        if eof.is_set() and not finished.is_set():
            with anyio.move_on_after(drain_grace) as drain_scope:
                await finished.wait()
            if drain_scope.cancel_called:
                state.exit_code = 1
                await _send_connection_errors(local_write, state.pending)
                await daemon_write.aclose()
        task_group.cancel_scope.cancel()
    return state.exit_code


async def _run_forwarder(poppy_dir: Path, port: int, drain_grace: float = _DEFAULT_DRAIN_GRACE) -> int:
    headers: dict[str, str] = {}
    token = load_daemon_token(poppy_dir)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    state = _BridgeState()
    try:
        async with stdio_server() as (local_read, local_write):
            try:
                async with streamablehttp_client(
                    f"http://127.0.0.1:{port}/mcp",
                    headers=headers,
                    terminate_on_close=True,
                    httpx_client_factory=_loopback_http_client,  # never proxy the loopback daemon
                ) as (daemon_read, daemon_write, _get_session_id):
                    # The daemon returns every request response on its POST
                    # (json_response=True). The bridge therefore never waits on
                    # the standalone GET/SSE listener, avoiding SDK issue #1675's
                    # post-initialize subscription race.
                    exit_code = await _bridge(
                        local_read,
                        local_write,
                        daemon_read,
                        daemon_write,
                        state,
                        drain_grace,
                    )
            except Exception:
                if not state.stdin_eof:
                    state.exit_code = 1
                    await _send_connection_errors(local_write, state.pending)
                exit_code = state.exit_code
            finally:
                await local_write.aclose()
            if exit_code:
                # Raising through stdio_server makes its task group cancel the
                # blocking stdin reader instead of waiting for another line.
                raise _DaemonDisconnected
    except _DaemonDisconnected:
        return 1
    return 0


def run_forwarder(poppy_dir: Path, port: int) -> int:
    """Forward this process's stdio MCP stream to the local daemon."""
    try:
        configured_grace = float(os.environ.get("POPPY_FORWARDER_DRAIN_S", _DEFAULT_DRAIN_GRACE))
    except ValueError:
        drain_grace = _DEFAULT_DRAIN_GRACE
    else:
        drain_grace = max(0.0, configured_grace) if math.isfinite(configured_grace) else _DEFAULT_DRAIN_GRACE
    return anyio.run(_run_forwarder, poppy_dir, port, drain_grace)
