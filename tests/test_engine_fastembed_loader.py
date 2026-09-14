"""Offline-safe fastembed (ONNX) model loading + lazy/idle-unload tests.

``_load_fastembed`` threads ``local_files_only`` from the pinned cache state,
raises an actionable
ModelUnavailableError on a cold cache with no network, and re-raises genuine
failures for cached models as-is. Plus FastembedModels' lazy-load + idle-unload
behavior for the always-on process.

The real fastembed classes are monkeypatched so nothing downloads.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import poppy.engine._fastembed_loader as fe
from poppy.engine._model_cache import is_fastembed_model_cached
from poppy.errors import ModelUnavailableError

BI = "BAAI/bge-small-en-v1.5"


@pytest.fixture(autouse=True)
def _reset_first_run_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fe, "_FIRST_RUN_NOTICE_SHOWN", False)


def _patch_ctors(monkeypatch: pytest.MonkeyPatch, ctor: Any) -> None:
    """Point both fastembed constructors at ``ctor`` without importing real weights."""
    monkeypatch.setattr("fastembed.TextEmbedding", ctor, raising=True)
    monkeypatch.setattr("fastembed.rerank.cross_encoder.TextCrossEncoder", ctor, raising=True)


# --- _load_fastembed offline semantics ---------------------------------------


def test_cold_offline_load_raises_actionable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise OSError("connection refused")

    _patch_ctors(monkeypatch, boom)
    monkeypatch.setattr(fe, "is_fastembed_model_cached", lambda *_a, **_k: False)

    with pytest.raises(ModelUnavailableError, match="poppy engines use seed"):
        fe._load_fastembed("bi", BI, tmp_path)


def test_cached_load_failure_reraises_original(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("corrupt onnx graph")

    _patch_ctors(monkeypatch, boom)
    monkeypatch.setattr(fe, "is_fastembed_model_cached", lambda *_a, **_k: True)

    with pytest.raises(RuntimeError, match="corrupt onnx graph") as excinfo:
        fe._load_fastembed("bi", BI, tmp_path)
    assert not isinstance(excinfo.value, ModelUnavailableError)


@pytest.mark.parametrize("cached", [True, False])
def test_load_threads_local_files_only_by_cache_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cached: bool
) -> None:
    captured: dict[str, Any] = {}

    class _Recorder:
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs

    _patch_ctors(monkeypatch, _Recorder)
    monkeypatch.setattr(fe, "is_fastembed_model_cached", lambda *_a, **_k: cached)

    fe._load_fastembed("bi", BI, tmp_path)

    assert captured["model_name"] == BI
    assert captured["kwargs"]["local_files_only"] is cached
    assert captured["kwargs"]["cache_dir"] == str(tmp_path)
    assert captured["kwargs"]["providers"] == ["CPUExecutionProvider"]


def test_unset_override_keeps_exact_stock_constructor_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class _Recorder:
        def __init__(self, model_name: str, **kwargs: Any) -> None:
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs

    _patch_ctors(monkeypatch, _Recorder)
    monkeypatch.setattr(fe, "is_fastembed_model_cached", lambda *_a, **_k: True)

    fe._load_fastembed("cross", fe.CROSS_ENCODER, tmp_path)

    assert captured == {
        "model_name": fe.CROSS_ENCODER,
        "kwargs": {
            "cache_dir": str(tmp_path),
            "providers": ["CPUExecutionProvider"],
            "local_files_only": True,
        },
    }


@pytest.mark.parametrize(
    ("kind", "dir_env", "file_env"),
    [
        ("bi", "POPPY_ONNX_BI_MODEL_DIR", "POPPY_ONNX_BI_MODEL_FILE"),
        ("cross", "POPPY_ONNX_CE_MODEL_DIR", "POPPY_ONNX_CE_MODEL_FILE"),
    ],
)
def test_local_onnx_override_registers_and_constructs_exact_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: str,
    dir_env: str,
    file_env: str,
) -> None:
    model_dir = tmp_path / kind
    model_path = model_dir / "onnx" / "candidate.onnx"
    model_path.parent.mkdir(parents=True)
    model_path.touch()
    monkeypatch.setenv(dir_env, str(model_dir))
    monkeypatch.setenv(file_env, "onnx/candidate.onnx")
    registered: list[dict[str, Any]] = []
    constructed: list[tuple[str, dict[str, Any]]] = []

    class _Recorder:
        @classmethod
        def list_supported_models(cls) -> list[dict[str, Any]]:
            return []

        @classmethod
        def add_custom_model(cls, **kwargs: Any) -> None:
            registered.append(kwargs)

        def __init__(self, model_name: str, **kwargs: Any) -> None:
            constructed.append((model_name, kwargs))

    _patch_ctors(monkeypatch, _Recorder)
    stock_name = fe.BI_ENCODER if kind == "bi" else fe.CROSS_ENCODER

    fe._load_fastembed(kind, stock_name, tmp_path / "cache")

    alias, kwargs = constructed[0]
    assert alias.startswith(f"poppy-tra-442/{kind}-")
    assert kwargs == {
        "cache_dir": str(tmp_path / "cache"),
        "providers": ["CPUExecutionProvider"],
        "local_files_only": True,
        "specific_model_path": str(model_dir.resolve()),
    }
    assert registered[0]["model"] == alias
    assert registered[0]["model_file"] == "onnx/candidate.onnx"
    assert registered[0]["sources"].hf == stock_name


@pytest.mark.parametrize(
    ("dir_value", "file_value", "message"),
    [
        (None, "onnx/model.onnx", "set POPPY_ONNX_BI_MODEL_DIR"),
        ("missing", "onnx/model.onnx", "does not exist or is not a directory"),
    ],
)
def test_bad_local_onnx_override_fails_actionably_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dir_value: str | None,
    file_value: str,
    message: str,
) -> None:
    if dir_value is not None:
        monkeypatch.setenv("POPPY_ONNX_BI_MODEL_DIR", str(tmp_path / dir_value))
    monkeypatch.setenv("POPPY_ONNX_BI_MODEL_FILE", file_value)
    monkeypatch.setattr(
        fe,
        "is_fastembed_model_cached",
        lambda *_a, **_k: pytest.fail("bad override must not inspect or fall back to the stock cache"),
    )

    with pytest.raises(ModelUnavailableError, match=message) as excinfo:
        fe._load_fastembed("bi", fe.BI_ENCODER, tmp_path / "cache")
    assert "fallback" in str(excinfo.value)


# --- first-run notice --------------------------------------------------------


def test_first_run_notice_prints_once_for_cold_models(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(fe, "is_fastembed_model_cached", lambda *_a, **_k: False)
    fe.announce_first_run_download(("model-a", "model-b"))
    fe.announce_first_run_download(("model-a", "model-b"))
    err = capsys.readouterr().err
    assert err.count("downloading ONNX retrieval models") == 1
    assert "model-a" in err and "model-b" in err
    assert "This happens once." in err


def test_no_first_run_notice_when_cached(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(fe, "is_fastembed_model_cached", lambda *_a, **_k: True)
    fe.announce_first_run_download(("model-a", "model-b"))
    assert capsys.readouterr().err == ""


# --- cache probe -------------------------------------------------------------


def test_is_fastembed_model_cached_maps_to_onnx_repo_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """bge-small maps to fastembed's quantized ONNX repo folder on disk."""
    monkeypatch.setenv("POPPY_FASTEMBED_CACHE", str(tmp_path))
    assert not is_fastembed_model_cached(BI)
    (tmp_path / "models--qdrant--bge-small-en-v1.5-onnx-q").mkdir()
    assert is_fastembed_model_cached(BI)


# --- FastembedModels lazy-load + idle-unload ---------------------------------


class _FakeBi:
    def embed(self, texts):
        for t in texts:
            yield np.ones(4, dtype=np.float32)


class _FakeCross:
    def rerank(self, query, docs, batch_size: int | None = None):
        return [1.0 for _ in docs]


class _RecordingCross:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query, docs):
        documents = list(docs)
        self.calls.append((query, documents))
        return [float(index) for index, _ in enumerate(documents)]


class _BatchRecordingCross:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int | None]] = []

    def rerank(self, query, docs, batch_size: int | None = None):
        documents = list(docs)
        self.calls.append((query, documents, batch_size))
        return [float(index) for index, _ in enumerate(documents)]


def test_injected_reranker_keeps_legacy_call_contract(tmp_path: Path) -> None:
    cross = _RecordingCross()
    models = fe.FastembedModels(bi_encoder=_FakeBi(), cross_encoder=cross, cache_dir=tmp_path)
    assert models.rerank("query", ["a", "b"]) == [0.0, 1.0]
    assert cross.calls == [("query", ["a", "b"])]


def test_rerank_uses_memory_bounded_batch_and_preserves_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cross = _BatchRecordingCross()
    monkeypatch.setattr(fe, "_load_fastembed", lambda kind, model_name, cache_dir: cross)
    monkeypatch.setattr(fe, "announce_first_run_download", lambda names: None)
    models = fe.FastembedModels(cache_dir=tmp_path, idle_timeout_s=0)
    documents = [f"doc-{index}" for index in range(13)]

    scores = models.rerank("query", documents)

    assert scores == [float(index) for index in range(13)]
    assert cross.calls == [("query", documents, 8)]


def test_stock_rerank_env_restores_fastembed_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(fe._STOCK_RERANK_ENV, "1")
    cross = _BatchRecordingCross()
    monkeypatch.setattr(fe, "_load_fastembed", lambda kind, model_name, cache_dir: cross)
    monkeypatch.setattr(fe, "announce_first_run_download", lambda names: None)
    models = fe.FastembedModels(cache_dir=tmp_path, idle_timeout_s=0)

    assert models.rerank("query", ["a", "b"]) == [0.0, 1.0]
    assert cross.calls == [("query", ["a", "b"], None)]


def test_lazy_loads_once_then_reuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loads: list[tuple[str, str]] = []

    def fake_load(kind: str, model_name: str, cache_dir: Path) -> Any:
        loads.append((kind, model_name))
        return _FakeBi() if kind == "bi" else _FakeCross()

    monkeypatch.setattr(fe, "_load_fastembed", fake_load)
    monkeypatch.setattr(fe, "announce_first_run_download", lambda names: None)

    models = fe.FastembedModels(cache_dir=tmp_path, idle_timeout_s=0)
    # No load at construction time.
    assert loads == []
    models.embed("hello")
    models.embed("world")
    # Bi-encoder loaded exactly once, cross-encoder not at all (no rerank yet).
    assert loads == [("bi", fe.BI_ENCODER)]
    models.rerank("q", ["a", "b"])
    assert loads == [("bi", fe.BI_ENCODER), ("cross", fe.CROSS_ENCODER)]


def test_idle_unload_then_reload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loads: list[str] = []

    def fake_load(kind: str, model_name: str, cache_dir: Path) -> Any:
        loads.append(kind)
        return _FakeBi() if kind == "bi" else _FakeCross()

    monkeypatch.setattr(fe, "_load_fastembed", fake_load)
    monkeypatch.setattr(fe, "announce_first_run_download", lambda names: None)

    # Inject the monotonic clock so the idle-unload path is deterministic across
    # platforms (the real monotonic epoch is arbitrary — a freshly-booted CI
    # runner can read below the 60s rate-limit interval and mask the unload).
    clock = {"t": 1000.0}
    models = fe.FastembedModels(cache_dir=tmp_path, idle_timeout_s=600.0, clock=lambda: clock["t"])
    models.embed("first")
    assert loads == ["bi"]
    assert models._bi is not None

    clock["t"] += 601.0
    models.check_idle_unload()
    assert models._bi is None

    # Next call lazily reloads from cache.
    models.embed("second")
    assert loads == ["bi", "bi"]


def test_injected_models_never_load_or_unload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("loader must not be called for injected encoders")

    monkeypatch.setattr(fe, "_load_fastembed", boom)

    models = fe.FastembedModels(
        bi_encoder=_FakeBi(), cross_encoder=_FakeCross(), cache_dir=tmp_path, idle_timeout_s=0.01
    )
    assert models._injected is True
    v = models.embed("x")
    assert v.dtype == np.float32 and v.shape == (4,)
    assert models._unload_thread is None
    models.check_idle_unload()
    assert models._bi is not None
    assert models._cross is not None


def test_loaded_but_never_called_model_unloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clock = {"t": 10.0}
    monkeypatch.setattr(fe, "_load_fastembed", lambda kind, model_name, cache_dir: _FakeBi())
    monkeypatch.setattr(fe, "announce_first_run_download", lambda names: None)
    models = fe.FastembedModels(cache_dir=tmp_path, idle_timeout_s=5.0, clock=lambda: clock["t"])

    models._model("bi")
    clock["t"] = 15.0

    assert models.check_idle_unload()
    assert models._bi is None


def test_idle_timeout_env_and_explicit_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("POPPY_MODEL_IDLE_S", "2.5")
    assert fe.FastembedModels(cache_dir=tmp_path)._idle_timeout_s == 2.5

    monkeypatch.setenv("POPPY_MODEL_IDLE_S", "garbage")
    assert fe.FastembedModels(cache_dir=tmp_path)._idle_timeout_s == 0.0

    monkeypatch.setenv("POPPY_MODEL_IDLE_S", "0")
    disabled = fe.FastembedModels(cache_dir=tmp_path)
    assert disabled._idle_timeout_s == 0.0
    assert fe.FastembedModels(cache_dir=tmp_path, idle_timeout_s=7.0)._idle_timeout_s == 7.0


def test_disabled_timeout_never_starts_timer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("POPPY_MODEL_IDLE_S", "0")
    monkeypatch.setattr(fe, "_load_fastembed", lambda kind, model_name, cache_dir: _FakeBi())
    monkeypatch.setattr(fe, "announce_first_run_download", lambda names: None)
    models = fe.FastembedModels(cache_dir=tmp_path)

    models.embed("hello")

    assert models._bi is not None
    assert models._unload_thread is None


def test_timer_is_daemon_and_exits_after_unload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fe, "_load_fastembed", lambda kind, model_name, cache_dir: _FakeBi())
    monkeypatch.setattr(fe, "announce_first_run_download", lambda names: None)
    models = fe.FastembedModels(cache_dir=tmp_path, idle_timeout_s=0.05)

    models.embed("hello")
    thread = models._unload_thread
    assert thread is not None and thread.daemon

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and (models._bi is not None or thread.is_alive()):
        time.sleep(0.01)

    assert models._bi is None
    assert models._cross is None
    assert not thread.is_alive()
    assert models._unload_thread is None
