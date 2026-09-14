"""Capture-only inference delegation to the local Poppy daemon."""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from poppy.mcp_server.auth import load_daemon_token

_DEFAULT_DAEMON_PORT = 7679
_PROBE_TIMEOUT_S = 0.5


class DaemonInferenceUnavailable(RuntimeError):
    """The daemon cannot safely serve inference for this engine."""


def _daemon_port() -> int:
    try:
        return int(os.environ.get("POPPY_DAEMON_PORT", _DEFAULT_DAEMON_PORT))
    except ValueError:
        return _DEFAULT_DAEMON_PORT


def _new_http_client(token: str | None) -> httpx.Client:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.Client(
        base_url=f"http://127.0.0.1:{_daemon_port()}",
        headers=headers,
        timeout=httpx.Timeout(30.0, connect=_PROBE_TIMEOUT_S, pool=_PROBE_TIMEOUT_S),
        # A loopback daemon connection must NEVER go through an HTTP proxy:
        # trust_env=False ignores $http_proxy, matching doctor's no-proxy probe.
        # A proxy could 403 loopback (breaking the runtime path while doctor says
        # OK) or exfiltrate the bearer token.
        trust_env=False,
    )


class DaemonModels:
    """Models-holder duck type backed by the already-running daemon."""

    def __init__(self, poppy_dir: Path, expected_engine: str, *, client: httpx.Client | None = None) -> None:
        self._poppy_dir = poppy_dir
        self._expected_engine = expected_engine
        self._client = client
        self._verified = False

    def _json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            if self._client is None:
                self._client = _new_http_client(load_daemon_token(self._poppy_dir))
            response = self._client.request(method, path, **kwargs)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001 - every transport/protocol failure triggers local fallback
            raise DaemonInferenceUnavailable(f"{type(exc).__name__}: {exc}") from exc
        if not isinstance(payload, dict):
            raise DaemonInferenceUnavailable("daemon returned a non-object response")
        return payload

    def _check_engine(self, payload: dict[str, Any]) -> None:
        actual = payload.get("engine")
        if actual != self._expected_engine:
            raise DaemonInferenceUnavailable(
                f"daemon engine mismatch (expected {self._expected_engine!r}, got {actual!r})"
            )

    def _verify(self) -> None:
        if self._verified:
            return
        payload = self._json("GET", "/status", timeout=_PROBE_TIMEOUT_S)
        self._check_engine(payload)
        self._verified = True

    def embed(self, text: str) -> np.ndarray:
        self._verify()
        payload = self._json("POST", "/internal/embed", json={"texts": [text]})
        self._check_engine(payload)
        vectors = payload.get("vectors")
        if not isinstance(vectors, list) or len(vectors) != 1 or not isinstance(vectors[0], list):
            raise DaemonInferenceUnavailable("daemon returned malformed vectors")
        try:
            vector = np.asarray(vectors[0], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise DaemonInferenceUnavailable("daemon returned non-numeric vectors") from exc
        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            raise DaemonInferenceUnavailable("daemon returned an invalid vector")
        return vector

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        self._verify()
        payload = self._json("POST", "/internal/rerank", json={"query": query, "docs": docs})
        self._check_engine(payload)
        scores = payload.get("scores")
        if not isinstance(scores, list) or len(scores) != len(docs):
            raise DaemonInferenceUnavailable("daemon returned malformed scores")
        try:
            values = [float(score) for score in scores]
        except (TypeError, ValueError) as exc:
            raise DaemonInferenceUnavailable("daemon returned non-numeric scores") from exc
        if not np.all(np.isfinite(values)):
            raise DaemonInferenceUnavailable("daemon returned invalid scores")
        return values

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        query = pairs[0][0]
        if any(pair_query != query for pair_query, _doc in pairs):
            raise DaemonInferenceUnavailable("daemon rerank requires one shared query")
        return self.rerank(query, [doc for _query, doc in pairs])


class DaemonFallbackModels:
    """Use daemon inference until its first failure, then stay local."""

    def __init__(self, remote: DaemonModels, local_factory: Callable[[], Any]) -> None:
        self._remote = remote
        self._local_factory = local_factory
        self._local: Any = None
        self._lock = threading.Lock()

    def _target(self) -> Any:
        return self._local if self._local is not None else self._remote

    def _fall_back(self, exc: DaemonInferenceUnavailable) -> Any:
        with self._lock:
            if self._local is None:
                print(
                    f"poppy: daemon inference unavailable ({exc}); loading models in capture worker.",
                    file=sys.stderr,
                    flush=True,
                )
                self._local = self._local_factory()
        return self._local

    def _call(self, method: str, *args: Any) -> Any:
        target = self._target()
        try:
            return getattr(target, method)(*args)
        except DaemonInferenceUnavailable as exc:
            return getattr(self._fall_back(exc), method)(*args)

    def embed(self, text: str) -> np.ndarray:
        return self._call("embed", text)

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        return self._call("rerank", query, docs)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        return self._call("score_pairs", pairs)


def daemon_fallback_models(
    poppy_dir: Path,
    expected_engine: str,
    local_factory: Callable[[], Any],
) -> DaemonFallbackModels:
    """Build the capture-only proxy while keeping local holder creation lazy."""
    return DaemonFallbackModels(DaemonModels(poppy_dir, expected_engine), local_factory)
