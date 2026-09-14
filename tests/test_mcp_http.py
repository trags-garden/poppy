from __future__ import annotations

import fcntl
import os
import socket
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from click.testing import CliRunner
from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from poppy.cli.main import cli
from poppy.mcp_server.auth import BearerAuthMiddleware, load_daemon_token
from poppy.mcp_server.daemon import ALREADY_RUNNING_MESSAGE, DAEMON_LOCK_FILENAME, create_http_app


async def _ok_app(scope, receive, send):
    await JSONResponse({"ok": True})(scope, receive, send)


@pytest.mark.parametrize(
    "token,header,status",
    [
        (None, None, 200),
        ("secret", None, 401),
        ("secret", "Bearer wrong", 401),
        ("secret", "Bearer secret", 200),
    ],
)
def test_bearer_auth(token, header, status):
    client = TestClient(BearerAuthMiddleware(_ok_app, token))
    headers = {"Authorization": header} if header else {}
    response = client.post("/mcp", headers=headers)
    assert response.status_code == status


def test_bearer_auth_does_not_gate_other_paths():
    client = TestClient(BearerAuthMiddleware(_ok_app, "secret"))
    assert client.get("/health").status_code == 200


class _StatusEngine:
    _engine_name = "status-test"


def test_status_endpoint_schema_and_mcp_sibling(tmp_path):
    async def mcp_route(_request):
        return JSONResponse({"mcp": True})

    mcp_app = Starlette(routes=[Route("/mcp", mcp_route)])
    app = create_http_app(mcp_app, engine=_StatusEngine(), poppy_dir=tmp_path, port=8123, started_at=time.monotonic())
    client = TestClient(app)

    status = client.get("/status")
    assert status.status_code == 200
    assert set(status.json()) == {
        "version",
        "pid",
        "uptime_s",
        "store_path",
        "engine",
        "engine_error",
        "models_loaded",
        "model_idle_deadline",
        "port",
    }
    assert status.json()["engine_error"] is None
    assert status.json()["models_loaded"] is False
    assert status.json()["port"] == 8123
    assert client.get("/mcp").json() == {"mcp": True}


def test_status_endpoint_surfaces_engine_error(tmp_path):
    app = create_http_app(
        Starlette(),
        engine=_StatusEngine(),
        engine_error="bloom: ImportError: onnxruntime unavailable",
        poppy_dir=tmp_path,
        port=8123,
    )

    status = TestClient(app).get("/status")

    assert status.status_code == 200
    assert status.json()["engine_error"] == "bloom: ImportError: onnxruntime unavailable"


def test_create_mcp_server_exposes_captured_engine_error(tmp_path, monkeypatch):
    from poppy.mcp_server import server as server_module

    observed_errors = []

    def get_engine(_poppy_dir, *, error_sink):
        error_sink("bloom: ImportError: onnxruntime unavailable")
        return _StatusEngine()

    monkeypatch.setattr(server_module, "get_engine", get_engine)

    mcp = server_module.create_mcp_server(
        poppy_dir=tmp_path,
        engine_error_sink=observed_errors.append,
    )

    assert observed_errors == ["bloom: ImportError: onnxruntime unavailable"]
    assert mcp._poppy_engine_error == "bloom: ImportError: onnxruntime unavailable"


def test_status_auth_and_loaded_models(tmp_path):
    from poppy.engine._model_holder import IdleUnloadingModels

    class Engine:
        _engine_name = "hot"
        _models = IdleUnloadingModels(bi_encoder=object())

    mcp_app = Starlette()
    composed = create_http_app(mcp_app, engine=Engine(), poppy_dir=tmp_path, port=7679)
    client = TestClient(BearerAuthMiddleware(composed, "secret", path=("/mcp", "/status")))

    assert client.get("/status").status_code == 401
    response = client.get("/status", headers={"Authorization": "Bearer secret"})
    assert response.status_code == 200
    assert response.json()["models_loaded"] is True
    assert response.json()["model_idle_deadline"] is None


def test_daemon_token_file_is_authoritative_over_env(tmp_path, monkeypatch):
    """The on-disk token wins over a divergent POPPY_DAEMON_TOKEN so a client sends what the daemon serves.

    ensure_daemon_token serves the FILE value; a reader that preferred the env
    would send a different bearer and get 401.
    """
    (tmp_path / "daemon.token").write_text("file-token\n")
    monkeypatch.setenv("POPPY_DAEMON_TOKEN", "env-token")
    assert load_daemon_token(tmp_path) == "file-token"


def test_daemon_token_env_only_fills_an_absent_file(tmp_path, monkeypatch):
    """With no token file, the (stripped) env token is the fallback that ensure_daemon_token then persists."""
    monkeypatch.setenv("POPPY_DAEMON_TOKEN", "  env-token  ")
    assert load_daemon_token(tmp_path) == "env-token"


def test_daemon_token_file_is_stripped(tmp_path, monkeypatch):
    monkeypatch.delenv("POPPY_DAEMON_TOKEN", raising=False)
    (tmp_path / "daemon.token").write_text("  file-token\n")
    assert load_daemon_token(tmp_path) == "file-token"


def test_daemon_run_rotates_the_token_from_env(tmp_path, monkeypatch):
    """ROTATION (round-9): the serving daemon overwrites a compromised token with a new env value.

    `poppy daemon run` (rotate_from_env=True) persists and serves POPPY_DAEMON_TOKEN
    so a leaked credential A can be revoked; clients then follow the new file.
    """
    from poppy.mcp_server.auth import ensure_daemon_token

    (tmp_path / "daemon.token").write_text("compromised-A\n")
    monkeypatch.setenv("POPPY_DAEMON_TOKEN", "rotated-B")
    assert ensure_daemon_token(tmp_path, rotate_from_env=True) == "rotated-B"
    assert (tmp_path / "daemon.token").read_text().strip() == "rotated-B"
    # A client resolves the rotated token from the file.
    assert load_daemon_token(tmp_path) == "rotated-B"


def test_client_read_ignores_transient_env_and_is_non_destructive(tmp_path, monkeypatch):
    """TRANSIENT CLIENT ENV (round-6): a one-off env on a client command neither 401s nor rewrites the file.

    `POPPY_DAEMON_TOKEN=x poppy daemon status` reads via load_daemon_token, which
    returns the live daemon's file token A and never writes.
    """
    (tmp_path / "daemon.token").write_text("live-A\n")
    monkeypatch.setenv("POPPY_DAEMON_TOKEN", "transient-x")
    assert load_daemon_token(tmp_path) == "live-A"
    assert (tmp_path / "daemon.token").read_text().strip() == "live-A"


def test_setup_seed_mode_never_clobbers_a_live_token(tmp_path, monkeypatch):
    """SETUP (round-7): `setup <client> --daemon` bakes the live token, an env cannot overwrite it.

    Seed mode (rotate_from_env=False) returns the on-disk token untouched so the
    client config matches what a running daemon already serves.
    """
    from poppy.mcp_server.auth import ensure_daemon_token

    (tmp_path / "daemon.token").write_text("live-A\n")
    monkeypatch.setenv("POPPY_DAEMON_TOKEN", "setup-B")
    assert ensure_daemon_token(tmp_path) == "live-A"  # baked into the client config
    assert (tmp_path / "daemon.token").read_text().strip() == "live-A"


def test_setup_seed_mode_seeds_a_fresh_machine(tmp_path, monkeypatch):
    """`POPPY_DAEMON_TOKEN=X setup --daemon` on a machine with no token file seeds and bakes X.

    The env-less service daemon that launchd/systemd starts later reads X from disk.
    """
    from poppy.mcp_server.auth import ensure_daemon_token

    monkeypatch.setenv("POPPY_DAEMON_TOKEN", "fresh-X")
    assert ensure_daemon_token(tmp_path) == "fresh-X"  # seeded + baked into the client config
    assert (tmp_path / "daemon.token").read_text().strip() == "fresh-X"
    # The env-less service daemon resolves the same value from disk.
    monkeypatch.delenv("POPPY_DAEMON_TOKEN")
    assert load_daemon_token(tmp_path) == "fresh-X"


def test_seed_mode_read_is_serialized_under_the_token_lock(tmp_path, monkeypatch):
    """The seed-mode read+return holds _token_lock, so it cannot interleave with a rotation overwrite.

    The seed decision must be mutually exclusive with the daemon's locked
    rotate-overwrite; a read outside the lock could return a stale token the
    daemon no longer serves. Assert the lock is held while the
    seed path reads by proving an independent non-blocking acquire fails.
    """
    import poppy.mcp_server.auth as auth_mod

    monkeypatch.delenv("POPPY_DAEMON_TOKEN", raising=False)
    (tmp_path / "daemon.token").write_text("live-A\n")
    lock_path = tmp_path / auth_mod.DAEMON_TOKEN_LOCK_FILENAME
    observed = {}
    real_read = auth_mod._read_token

    def spy_read(token_path):
        # A separate open file description trying LOCK_EX|LOCK_NB must fail while
        # the seed path holds the token lock (flock conflicts across fds).
        probe = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            observed["held"] = False
            fcntl.flock(probe, fcntl.LOCK_UN)
        except OSError:
            observed["held"] = True
        finally:
            os.close(probe)
        return real_read(token_path)

    monkeypatch.setattr(auth_mod, "_read_token", spy_read)
    assert auth_mod.ensure_daemon_token(tmp_path) == "live-A"
    assert observed.get("held") is True, "the seed-mode read ran without holding _token_lock"


def test_seed_read_after_rotation_is_consistent_with_disk(tmp_path, monkeypatch):
    """After a rotation, a seed-mode read returns exactly what is on disk — never a torn/stale value."""
    from poppy.mcp_server.auth import ensure_daemon_token

    (tmp_path / "daemon.token").write_text("old-A\n")
    monkeypatch.setenv("POPPY_DAEMON_TOKEN", "rotated-B")
    assert ensure_daemon_token(tmp_path, rotate_from_env=True) == "rotated-B"

    # A later setup (seed mode, env cleared as an env-less service context) reads
    # the rotated value, matching disk and what the daemon now serves.
    monkeypatch.delenv("POPPY_DAEMON_TOKEN")
    seeded = ensure_daemon_token(tmp_path)
    assert seeded == "rotated-B"
    assert seeded == (tmp_path / "daemon.token").read_text().strip()


class _FakeMcp:
    settings = SimpleNamespace(streamable_http_path="/mcp")

    def __init__(self):
        self.run_calls = []

    def run(self, **kwargs):
        self.run_calls.append(kwargs)

    def streamable_http_app(self):
        return _ok_app


def test_serve_defaults_to_unchanged_stdio(tmp_path, monkeypatch):
    fake_mcp = _FakeMcp()
    monkeypatch.setattr("poppy.mcp_server.server.create_mcp_server", lambda **_kwargs: fake_mcp)

    result = CliRunner().invoke(cli, ["serve"], env={"POPPY_DIR": str(tmp_path)})

    assert result.exit_code == 0, result.output
    assert fake_mcp.run_calls == [{"transport": "stdio"}]


def test_daemon_run_reuses_http_serve_path(tmp_path, monkeypatch):
    calls = {}

    def fake_bind(host, port):
        calls["bind"] = (host, port)
        return socket.socket()

    monkeypatch.setattr("poppy.mcp_server.server.create_mcp_server", lambda **_kwargs: _FakeMcp())
    monkeypatch.setattr("poppy.mcp_server.daemon.bind_socket", fake_bind)
    monkeypatch.setattr(
        "poppy.mcp_server.daemon.run_server", lambda _app, _listener, host, port: calls.setdefault("run", (host, port))
    )

    result = CliRunner().invoke(
        cli,
        ["daemon", "run", "--port", "8765"],
        env={"POPPY_DIR": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    assert calls == {"bind": ("127.0.0.1", 8765), "run": ("127.0.0.1", 8765)}


def test_daemon_run_enforces_token_without_setup(tmp_path, monkeypatch):
    """`daemon run` mints a token so /mcp and /status refuse anonymous callers."""
    monkeypatch.delenv("POPPY_DAEMON_TOKEN", raising=False)
    captured = {}

    monkeypatch.setattr("poppy.mcp_server.server.create_mcp_server", lambda **_kwargs: _FakeMcp())
    monkeypatch.setattr("poppy.mcp_server.daemon.bind_socket", lambda *_args: socket.socket())
    monkeypatch.setattr(
        "poppy.mcp_server.daemon.run_server", lambda app, _listener, _host, _port: captured.setdefault("app", app)
    )

    result = CliRunner().invoke(cli, ["daemon", "run"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0, result.output

    # A token file must exist even though `setup --daemon` was never run.
    token = (tmp_path / "daemon.token").read_text().strip()
    assert token

    app = captured["app"]
    assert isinstance(app, BearerAuthMiddleware)
    assert app.token == token

    client = TestClient(app)

    # /mcp: 401 without the bearer, reaches the app with it.
    assert client.post("/mcp").status_code == 401
    assert client.post("/mcp", headers={"Authorization": f"Bearer {token}"}).status_code == 200

    # /status: does not leak store_path unauthenticated, but returns it with the token.
    assert client.get("/status").status_code == 401
    authed = client.get("/status", headers={"Authorization": f"Bearer {token}"})
    assert authed.status_code == 200
    assert "store_path" in authed.json()


@pytest.mark.parametrize(
    "args,extra_env,expected",
    [
        (["--transport", "http"], {}, ("127.0.0.1", 7679)),
        (["--transport", "streamable-http", "--host", "localhost", "--port", "8123"], {}, ("localhost", 8123)),
        (["--transport", "http"], {"POPPY_DAEMON_PORT": "8345"}, ("127.0.0.1", 8345)),
    ],
)
def test_serve_http_flag_parsing(tmp_path, monkeypatch, args, extra_env, expected):
    calls = {}
    fake_mcp = _FakeMcp()

    def fake_create(**kwargs):
        calls["create"] = kwargs
        return fake_mcp

    def fake_bind(host, port):
        calls["bind"] = (host, port)
        return socket.socket()

    def fake_run(_app, _listener, host, port):
        calls["run"] = (host, port)

    monkeypatch.setattr("poppy.mcp_server.server.create_mcp_server", fake_create)
    monkeypatch.setattr("poppy.mcp_server.daemon.bind_socket", fake_bind)
    monkeypatch.setattr("poppy.mcp_server.daemon.run_server", fake_run)
    env = {"POPPY_DIR": str(tmp_path), **extra_env}

    result = CliRunner().invoke(cli, ["serve", *args], env=env)

    assert result.exit_code == 0, result.output
    assert calls["bind"] == expected
    assert calls["run"] == expected
    assert calls["create"]["json_response"] is True
    assert calls["create"]["stateless_http"] is False


def test_http_concrete_source_warns_and_uses_client_attribution(tmp_path, monkeypatch):
    calls = {}

    def fake_create(**kwargs):
        calls.update(kwargs)
        return _FakeMcp()

    monkeypatch.setattr("poppy.mcp_server.server.create_mcp_server", fake_create)
    monkeypatch.setattr("poppy.mcp_server.daemon.bind_socket", lambda *_args: socket.socket())
    monkeypatch.setattr("poppy.mcp_server.daemon.run_server", lambda *_args: None)

    result = CliRunner().invoke(
        cli,
        ["serve", "--transport", "http", "--source", "claude-code"],
        env={"POPPY_DIR": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    assert "per-client clientInfo attribution takes precedence" in result.output
    assert calls["source"] == "mcp"


def test_serve_http_exits_zero_when_daemon_lock_is_held(tmp_path):
    lock_path = tmp_path / DAEMON_LOCK_FILENAME
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    # A live daemon already serves token A. A losing `daemon run` — even one with
    # a rotation env token — must NOT rewrite the file, because it holds no lock
    # and will not serve.
    (tmp_path / "daemon.token").write_text("live-A\n")
    try:
        result = CliRunner().invoke(
            cli,
            ["serve", "--transport", "http"],
            env={"POPPY_DIR": str(tmp_path), "POPPY_DAEMON_TOKEN": "rotated-B"},
        )
    finally:
        os.close(fd)

    assert result.exit_code == 0
    assert ALREADY_RUNNING_MESSAGE in result.output
    assert (tmp_path / "daemon.token").read_text().strip() == "live-A"


def test_serve_http_exits_zero_when_port_is_in_use(tmp_path):
    try:
        occupied = socket.create_server(("127.0.0.1", 0))
    except PermissionError:
        pytest.skip("sandbox forbids loopback socket binds")
    port = occupied.getsockname()[1]
    try:
        result = CliRunner().invoke(
            cli,
            ["serve", "--transport", "http", "--port", str(port)],
            env={"POPPY_DIR": str(tmp_path)},
        )
    finally:
        occupied.close()

    assert result.exit_code == 0
    assert ALREADY_RUNNING_MESSAGE in result.output


@pytest.mark.asyncio
async def test_streamable_http_asgi_smoke_and_origin_rejection(tmp_path, monkeypatch):
    from poppy.engine.seed import SeedEngine
    from poppy.mcp_server import server as server_module

    engine = SeedEngine(db_path=tmp_path / "memories.db")
    monkeypatch.setattr(server_module, "get_engine", lambda _poppy_dir: engine)
    mcp = server_module.create_mcp_server(
        poppy_dir=tmp_path,
        host="127.0.0.1",
        port=7679,
        json_response=True,
        stateless_http=False,
    )
    mcp_app = mcp.streamable_http_app()
    composed = create_http_app(mcp_app, engine=engine, poppy_dir=tmp_path, port=7679)
    app = BearerAuthMiddleware(composed, token=None, path=("/mcp", "/status"))
    transport = httpx.ASGITransport(app=app)

    def asgi_client_factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(
            transport=transport,
            headers=headers,
            timeout=timeout,
            auth=auth,
            follow_redirects=True,
        )

    async with composed.router.lifespan_context(composed):
        async with streamablehttp_client(
            "http://127.0.0.1:7679/mcp",
            httpx_client_factory=asgi_client_factory,
        ) as (read_stream, write_stream, _get_session_id):
            async with ClientSession(
                read_stream,
                write_stream,
                client_info=types.Implementation(name="poppy-http-test", version="1"),
            ) as session:
                initialized = await session.initialize()
                assert initialized.serverInfo.name == "poppy"
                result = await session.call_tool("context", {"limit": 1})
                assert result.isError is not True

        async with httpx.AsyncClient(transport=transport) as client:
            status = await client.get("http://127.0.0.1:7679/status")
            response = await client.post(
                "http://127.0.0.1:7679/mcp",
                headers={"Origin": "http://evil.example"},
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            )
        assert status.status_code == 200
        assert response.status_code == 403


@pytest.mark.asyncio
async def test_streamable_http_ephemeral_server(tmp_path, monkeypatch):
    from poppy.engine.seed import SeedEngine
    from poppy.mcp_server import server as server_module
    from poppy.mcp_server.daemon import bind_socket

    engine = SeedEngine(db_path=tmp_path / "memories.db")
    monkeypatch.setattr(server_module, "get_engine", lambda _poppy_dir: engine)
    mcp = server_module.create_mcp_server(
        poppy_dir=tmp_path,
        host="127.0.0.1",
        port=0,
        json_response=True,
        stateless_http=False,
    )
    assert mcp.settings.json_response is True
    assert mcp.settings.stateless_http is False
    assert mcp.settings.transport_security.enable_dns_rebinding_protection is True

    app = BearerAuthMiddleware(mcp.streamable_http_app(), token=None)
    try:
        listener = bind_socket("127.0.0.1", 0)
    except PermissionError:
        pytest.skip("sandbox forbids loopback socket binds")
    port = listener.getsockname()[1]
    uvicorn_server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    server_errors = []

    def run_server():
        try:
            uvicorn_server.run(sockets=[listener])
        except BaseException as exc:  # pragma: no cover - surfaced by the assertion below
            server_errors.append(exc)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not uvicorn_server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)

    try:
        assert uvicorn_server.started, server_errors
        url = f"http://127.0.0.1:{port}/mcp"
        async with streamablehttp_client(url) as (read_stream, write_stream, _get_session_id):
            async with ClientSession(
                read_stream,
                write_stream,
                client_info=types.Implementation(name="poppy-http-test", version="1"),
            ) as session:
                initialized = await session.initialize()
                assert initialized.serverInfo.name == "poppy"
                result = await session.call_tool("context", {"limit": 1})
                assert result.isError is not True

    finally:
        uvicorn_server.should_exit = True
        thread.join(timeout=5)
        listener.close()

    assert not thread.is_alive()
    assert not server_errors


def test_ensure_daemon_token_is_race_safe_cross_process(tmp_path):
    """Concurrent processes (daemon run vs setup) must all agree on one token."""
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        f"""
        import os, sys
        from pathlib import Path
        os.environ.pop("POPPY_DAEMON_TOKEN", None)
        from poppy.mcp_server.auth import ensure_daemon_token
        sys.stdout.write(ensure_daemon_token(Path({str(tmp_path)!r})))
        """
    )
    procs = [subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True) for _ in range(8)]
    tokens = {p.communicate()[0].strip() for p in procs}
    assert all(p.returncode == 0 for p in procs)
    # All eight processes agree on exactly one token, and it is what is on disk.
    assert len(tokens) == 1, tokens
    assert (tmp_path / "daemon.token").read_text().strip() == next(iter(tokens))


def test_ensure_daemon_token_repairs_an_empty_file(tmp_path, monkeypatch):
    """A 0-byte daemon.token (touch/crash) must be repaired, never returned as \"\"."""
    monkeypatch.delenv("POPPY_DAEMON_TOKEN", raising=False)
    from poppy.mcp_server.auth import ensure_daemon_token

    (tmp_path / "daemon.token").write_text("")  # empty file
    token = ensure_daemon_token(tmp_path)
    assert token, "an empty token file must be repaired, not returned empty"
    assert (tmp_path / "daemon.token").read_text().strip() == token


def test_ensure_daemon_token_does_not_follow_a_symlinked_target(tmp_path, monkeypatch):
    """A symlink planted at the token path must be replaced, not written through to a victim."""
    monkeypatch.setenv("POPPY_DAEMON_TOKEN", "chosen-secret")
    from poppy.mcp_server.auth import ensure_daemon_token

    victim = tmp_path / "victim.txt"
    victim.write_text("")  # empty target -> token treated as absent -> published
    token_path = tmp_path / "daemon.token"
    token_path.symlink_to(victim)

    assert ensure_daemon_token(tmp_path) == "chosen-secret"
    # os.replace put a real file at the link name; the victim was not written through.
    assert not token_path.is_symlink()
    assert token_path.read_text().strip() == "chosen-secret"
    assert victim.read_text() == "", "the symlink target was overwritten"


def test_env_token_fills_absent_file_but_never_overwrites(tmp_path, monkeypatch):
    """POPPY_DAEMON_TOKEN fills an absent token but never clobbers a live daemon's minted one."""
    from poppy.mcp_server.auth import ensure_daemon_token, load_daemon_token

    token_path = tmp_path / "daemon.token"

    # (a) Absent file: the env token is persisted so an env-less service reads it.
    monkeypatch.setenv("POPPY_DAEMON_TOKEN", "chosen-secret")
    assert ensure_daemon_token(tmp_path) == "chosen-secret"
    assert token_path.read_text().strip() == "chosen-secret"
    monkeypatch.delenv("POPPY_DAEMON_TOKEN")
    assert load_daemon_token(tmp_path) == "chosen-secret"

    # (b) An existing DIFFERENT token wins: a transient env token must NOT overwrite
    #     it (that would 401 every baked client), and the on-disk value is returned.
    token_path.write_text("minted-live-token\n")
    monkeypatch.setenv("POPPY_DAEMON_TOKEN", "transient")
    assert ensure_daemon_token(tmp_path) == "minted-live-token"
    assert token_path.read_text().strip() == "minted-live-token"


def test_env_token_is_stripped_consistently(tmp_path, monkeypatch):
    """A whitespace-padded env token round-trips: the disk value equals the client's Bearer value."""
    from poppy.mcp_server.auth import ensure_daemon_token, load_daemon_token

    monkeypatch.setenv("POPPY_DAEMON_TOKEN", "  padded-secret  ")
    # The value baked into a client's Authorization header.
    assert ensure_daemon_token(tmp_path) == "padded-secret"
    assert (tmp_path / "daemon.token").read_text().strip() == "padded-secret"
    # A reader with the same padded env resolves the identical stripped token.
    assert load_daemon_token(tmp_path) == "padded-secret"
    # And so does the env-less service daemon reading from disk.
    monkeypatch.delenv("POPPY_DAEMON_TOKEN")
    assert load_daemon_token(tmp_path) == "padded-secret"


def test_bearer_token_matches_rejects_an_empty_token():
    """A bare `Bearer ` must never authenticate against an empty configured token."""
    from poppy.mcp_server.auth import bearer_token_matches

    assert bearer_token_matches("Bearer ", "") is False
    assert bearer_token_matches("Bearer ", None) is False
    assert bearer_token_matches("Bearer abc", "abc") is True


def test_ensure_daemon_token_never_publishes_an_empty_file(tmp_path, monkeypatch):
    """The token file only ever appears with content, so a reader never caches a None token."""
    monkeypatch.delenv("POPPY_DAEMON_TOKEN", raising=False)
    from poppy.mcp_server.auth import ensure_daemon_token, load_daemon_token

    token = ensure_daemon_token(tmp_path)
    assert token and load_daemon_token(tmp_path) == token
    # No temp files left behind.
    assert [p.name for p in tmp_path.glob("daemon.token*")] == ["daemon.token"]
