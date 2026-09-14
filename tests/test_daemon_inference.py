from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
from starlette.applications import Starlette
from starlette.testclient import TestClient

import poppy.engine._daemon_models as daemon_models_module
from poppy.config import PoppyConfig, save_config
from poppy.consolidation import consolidate_compact_event
from poppy.engine._daemon_models import DaemonFallbackModels, DaemonModels
from poppy.mcp_server.daemon import create_http_app


class _FakeModels:
    def __init__(self) -> None:
        self.embed_calls: list[str] = []
        self.rerank_calls: list[tuple[str, list[str]]] = []

    def embed(self, text: str) -> np.ndarray:
        self.embed_calls.append(text)
        return np.asarray([0.6, 0.8, 0.0], dtype=np.float32)

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        self.rerank_calls.append((query, docs))
        return [float(len(doc)) for doc in docs]

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        return self.rerank(pairs[0][0], [doc for _query, doc in pairs]) if pairs else []


def _authed_inference_app(tmp_path, *, engine_name: str = "bloom", models=None):
    (tmp_path / "daemon.token").write_text("secret\n")
    engine = SimpleNamespace(_engine_name=engine_name)
    if models is not None:
        engine._models = models
    return create_http_app(Starlette(), engine=engine, poppy_dir=tmp_path, port=7679)


def test_internal_inference_requires_auth_and_round_trips_fake_holder(tmp_path):
    models = _FakeModels()
    app = _authed_inference_app(tmp_path, models=models)
    client = TestClient(app)

    assert client.get("/status").status_code == 200
    assert client.post("/internal/embed", json={"texts": ["alpha"]}).status_code == 401
    assert client.post("/internal/rerank", json={"query": "q", "docs": ["a"]}).status_code == 401

    remote = DaemonModels(
        tmp_path,
        "bloom",
        client=TestClient(app, headers={"Authorization": "Bearer secret"}),
    )
    vector = remote.embed("alpha")
    scores = remote.rerank("query", ["one", "three"])

    assert vector.dtype == np.float32
    np.testing.assert_array_equal(vector, np.asarray([0.6, 0.8, 0.0], dtype=np.float32))
    assert scores == [3.0, 5.0]
    assert models.embed_calls == ["alpha"]
    assert models.rerank_calls == [("query", ["one", "three"])]


def test_internal_inference_returns_unavailable_without_models_holder(tmp_path):
    app = _authed_inference_app(tmp_path)
    client = TestClient(app, headers={"Authorization": "Bearer secret"})

    response = client.post("/internal/embed", json={"texts": ["alpha"]})

    assert response.status_code == 503
    assert response.json() == {"error": "inference_unavailable", "engine": "bloom"}

    local = _FakeModels()
    models = DaemonFallbackModels(DaemonModels(tmp_path, "bloom", client=client), lambda: local)
    np.testing.assert_array_equal(models.embed("fallback"), np.asarray([0.6, 0.8, 0.0], dtype=np.float32))
    assert local.embed_calls == ["fallback"]


def test_daemon_models_falls_back_permanently_on_connection_failure(capsys):
    requests = {"count": 0}

    def unavailable(request: httpx.Request) -> httpx.Response:
        requests["count"] += 1
        raise httpx.ConnectError("daemon down", request=request)

    local = _FakeModels()
    remote = DaemonModels(
        poppy_dir=Path("/unused"),
        expected_engine="bloom",
        client=httpx.Client(transport=httpx.MockTransport(unavailable), base_url="http://daemon"),
    )
    models = DaemonFallbackModels(remote, lambda: local)

    first = models.embed("first")
    models.embed("second")

    assert first.dtype == np.float32
    assert local.embed_calls == ["first", "second"]
    assert requests["count"] == 1
    assert capsys.readouterr().err.count("daemon inference unavailable") == 1


def test_daemon_models_falls_back_on_engine_mismatch(capsys):
    def mismatch(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/status"
        return httpx.Response(200, json={"engine": "seed"})

    local = _FakeModels()
    remote = DaemonModels(
        poppy_dir=Path("/unused"),
        expected_engine="bloom",
        client=httpx.Client(transport=httpx.MockTransport(mismatch), base_url="http://daemon"),
    )
    models = DaemonFallbackModels(remote, lambda: local)

    vector = models.embed("local")

    np.testing.assert_array_equal(vector, np.asarray([0.6, 0.8, 0.0], dtype=np.float32))
    assert "engine mismatch" in capsys.readouterr().err


def test_daemon_models_falls_back_when_token_cannot_be_loaded(tmp_path, monkeypatch, capsys):
    def unreadable_token(_poppy_dir):
        raise PermissionError("token unreadable")

    monkeypatch.setattr(daemon_models_module, "load_daemon_token", unreadable_token)
    local = _FakeModels()
    models = DaemonFallbackModels(DaemonModels(tmp_path, "bloom"), lambda: local)

    models.embed("local")

    assert local.embed_calls == ["local"]
    assert "PermissionError" in capsys.readouterr().err


def test_compact_capture_uses_daemon_without_local_model_load(tmp_path, monkeypatch):
    cfg = PoppyConfig(poppy_dir=tmp_path)
    cfg.set("engine", "bloom")
    save_config(cfg)
    (tmp_path / "daemon.token").write_text("secret\n")
    daemon_models = _FakeModels()
    app = _authed_inference_app(tmp_path, models=daemon_models)

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    monkeypatch.setenv("POPPY_TEST_FAIL_ON_MODEL_LOAD", "1")
    monkeypatch.setattr(
        "poppy.engine._daemon_models._new_http_client",
        lambda token: TestClient(app, headers={"Authorization": f"Bearer {token}"}),
    )
    monkeypatch.setattr(
        "poppy.consolidation.call_llm",
        lambda *_args, **_kwargs: [{"type": "decision", "content": "Use daemon inference for capture workers."}],
    )

    stored = consolidate_compact_event(
        {
            "session_id": "capture-daemon",
            "compact_summary": "The team decided how capture inference should execute.",
            "cwd": str(tmp_path),
            "transcript_path": "/dev/null",
        }
    )

    assert stored == 1
    assert daemon_models.embed_calls
