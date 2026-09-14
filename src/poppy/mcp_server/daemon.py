"""Process and socket primitives for the streamable-HTTP daemon transport."""

from __future__ import annotations

import errno
import fcntl
import os
import socket
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.types import ASGIApp

from poppy import __version__
from poppy.engine._model_holder import model_runtime_status
from poppy.mcp_server.auth import bearer_token_matches, load_daemon_token
from poppy.paths import ensure_poppy_dir

ALREADY_RUNNING_MESSAGE = "poppy daemon already running (lock held)"
DAEMON_LOCK_FILENAME = "daemon.lock"


@contextmanager
def daemon_lock(poppy_dir: Path) -> Iterator[bool]:
    """Try to hold the store's daemon lock, yielding whether it was acquired."""
    ensure_poppy_dir(poppy_dir)
    fd = os.open(str(poppy_dir / DAEMON_LOCK_FILENAME), os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
        if not acquired:
            yield False
            return

        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"{os.getpid()}\n".encode())
        yield True
    finally:
        os.close(fd)


def bind_socket(host: str, port: int) -> socket.socket:
    """Bind the daemon listener so address conflicts remain inspectable."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    listener = socket.socket(family=family)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind((host, port))
    except OSError:
        listener.close()
        raise
    listener.set_inheritable(True)
    return listener


def run_server(app: ASGIApp, listener: socket.socket, host: str, port: int) -> None:
    """Run uvicorn with the already-bound listener until shutdown."""
    config = uvicorn.Config(app, host=host, port=port)
    uvicorn.Server(config).run(sockets=[listener])


def create_http_app(
    mcp_app: Starlette,
    *,
    engine: object,
    engine_error: str | None = None,
    poppy_dir: Path,
    port: int,
    started_at: float | None = None,
) -> Starlette:
    """Compose daemon status and inference routes beside the MCP app."""
    started_at = time.monotonic() if started_at is None else started_at

    def engine_name() -> str:
        return getattr(engine, "_engine_name", engine.__class__.__name__)

    def inference_holder() -> object | None:
        holder = getattr(engine, "_models", None)
        if holder is None or not callable(getattr(holder, "embed", None)):
            return None
        return holder

    def authorize_internal(request: Request) -> JSONResponse | None:
        try:
            token = load_daemon_token(poppy_dir)
        except OSError:
            token = None
        if bearer_token_matches(request.headers.get("authorization", ""), token):
            return None
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    async def embed(request: Request) -> JSONResponse:
        unauthorized = authorize_internal(request)
        if unauthorized is not None:
            return unauthorized
        holder = inference_holder()
        if holder is None:
            return JSONResponse({"error": "inference_unavailable", "engine": engine_name()}, status_code=503)
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        texts = payload.get("texts") if isinstance(payload, dict) else None
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        try:
            vectors = [holder.embed(text).tolist() for text in texts]
        except Exception:
            return JSONResponse({"error": "inference_failed", "engine": engine_name()}, status_code=503)
        return JSONResponse({"engine": engine_name(), "vectors": vectors})

    async def rerank(request: Request) -> JSONResponse:
        unauthorized = authorize_internal(request)
        if unauthorized is not None:
            return unauthorized
        holder = inference_holder()
        if holder is None:
            return JSONResponse({"error": "inference_unavailable", "engine": engine_name()}, status_code=503)
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        query = payload.get("query") if isinstance(payload, dict) else None
        docs = payload.get("docs") if isinstance(payload, dict) else None
        if not isinstance(query, str) or not isinstance(docs, list) or any(not isinstance(doc, str) for doc in docs):
            return JSONResponse({"error": "invalid_request"}, status_code=400)
        try:
            if not docs:
                scores = []
            elif callable(getattr(holder, "rerank", None)):
                scores = holder.rerank(query, docs)
            elif callable(getattr(holder, "score_pairs", None)):
                scores = holder.score_pairs([(query, doc) for doc in docs])
            else:
                return JSONResponse({"error": "inference_unavailable", "engine": engine_name()}, status_code=503)
            serialized_scores = [float(score) for score in scores]
        except Exception:
            return JSONResponse({"error": "inference_failed", "engine": engine_name()}, status_code=503)
        if len(serialized_scores) != len(docs):
            return JSONResponse({"error": "inference_failed", "engine": engine_name()}, status_code=503)
        return JSONResponse({"engine": engine_name(), "scores": serialized_scores})

    async def status(_request: Request) -> JSONResponse:
        models_loaded, idle_deadline = model_runtime_status(engine)
        return JSONResponse(
            {
                "version": __version__,
                "pid": os.getpid(),
                "uptime_s": max(0.0, time.monotonic() - started_at),
                "store_path": str(poppy_dir / "memories.db"),
                "engine": engine_name(),
                "engine_error": engine_error,
                "models_loaded": models_loaded,
                "model_idle_deadline": idle_deadline,
                "port": port,
            }
        )

    @asynccontextmanager
    async def lifespan(_app: Starlette):
        async with mcp_app.router.lifespan_context(mcp_app):
            yield

    return Starlette(
        routes=[
            Route("/status", status),
            Route("/internal/embed", embed, methods=["POST"]),
            Route("/internal/rerank", rerank, methods=["POST"]),
            Mount("/", app=mcp_app),
        ],
        lifespan=lifespan,
    )
