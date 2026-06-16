"""Offline-safe model loading tests.

``load_st_model`` threads ``local_files_only`` from the HF cache state, raises
an actionable ModelUnavailableError on a cold cache with no network, and
re-raises genuine failures for cached models as-is. The first-run download
notice prints once per process, and only when a model is actually cold.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

import poppy.engine._st_loader as st_loader
from poppy.engine._model_cache import is_model_cached
from poppy.errors import ModelUnavailableError


@pytest.fixture(autouse=True)
def _reset_first_run_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(st_loader, "_FIRST_RUN_NOTICE_SHOWN", False)


def _fake_st_module(ctor: Any) -> types.ModuleType:
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = ctor  # type: ignore[attr-defined]
    mod.CrossEncoder = ctor  # type: ignore[attr-defined]
    return mod


def test_cold_offline_model_load_raises_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cold cache with no network surfaces an actionable error, not a raw traceback."""

    def boom(*_a: Any, **_k: Any) -> Any:
        raise OSError("connection refused")

    monkeypatch.setitem(sys.modules, "sentence_transformers", _fake_st_module(boom))
    monkeypatch.setattr(st_loader, "is_model_cached", lambda _repo: False)

    with pytest.raises(ModelUnavailableError, match="poppy engines use seed"):
        st_loader.load_st_model("bi", "BAAI/bge-small-en-v1.5")


def test_cached_model_load_failure_reraises_original(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the model IS cached, a load failure is a genuine error; surface it as-is."""

    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("corrupt weights")

    monkeypatch.setitem(sys.modules, "sentence_transformers", _fake_st_module(boom))
    monkeypatch.setattr(st_loader, "is_model_cached", lambda _repo: True)

    with pytest.raises(RuntimeError, match="corrupt weights") as excinfo:
        st_loader.load_st_model("bi", "BAAI/bge-small-en-v1.5")
    assert not isinstance(excinfo.value, ModelUnavailableError)


@pytest.mark.parametrize("cached", [True, False])
def test_load_threads_local_files_only_by_cache_state(monkeypatch: pytest.MonkeyPatch, cached: bool) -> None:
    """A cached model is loaded with local_files_only=True so the load-time
    huggingface.co HEAD (which crashes when the network is down) is skipped; an
    uncached model keeps local_files_only=False so a first-run download still works."""
    captured: dict[str, Any] = {}

    class _Recorder:
        def __init__(self, repo: str, **kwargs: Any) -> None:
            captured["repo"] = repo
            captured["kwargs"] = kwargs

    monkeypatch.setitem(sys.modules, "sentence_transformers", _fake_st_module(_Recorder))
    monkeypatch.setattr(st_loader, "is_model_cached", lambda _repo: cached)

    st_loader.load_st_model("bi", "BAAI/bge-small-en-v1.5")

    assert captured["repo"] == "BAAI/bge-small-en-v1.5"
    assert captured["kwargs"] == {"local_files_only": cached}


def test_first_run_notice_prints_once_for_cold_models(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(st_loader, "is_model_cached", lambda _repo: False)

    st_loader.announce_first_run_download(("model-a", "model-b"))
    st_loader.announce_first_run_download(("model-a", "model-b"))

    err = capsys.readouterr().err
    assert err.count("downloading retrieval models") == 1
    assert "model-a" in err
    assert "model-b" in err
    assert "This happens once." in err


def test_no_first_run_notice_when_models_cached(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(st_loader, "is_model_cached", lambda _repo: True)

    st_loader.announce_first_run_download(("model-a", "model-b"))

    assert capsys.readouterr().err == ""


def test_is_model_cached_checks_both_namespaced_forms(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bare model names resolve to the sentence-transformers/ org on disk."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    assert not is_model_cached("all-MiniLM-L6-v2")
    (tmp_path / "models--sentence-transformers--all-MiniLM-L6-v2").mkdir()
    assert is_model_cached("all-MiniLM-L6-v2")

    assert not is_model_cached("BAAI/bge-small-en-v1.5")
    (tmp_path / "models--BAAI--bge-small-en-v1.5").mkdir()
    assert is_model_cached("BAAI/bge-small-en-v1.5")
