"""Offline-aware fastembed (ONNX) model loading for the ``bloom`` engine.

Two guarantees:

* A model already in the pinned fastembed cache loads with
  ``local_files_only=True`` so no network HEAD fires (works fully offline once
  cached). An uncached model downloads normally.
* A cold cache with no network raises :class:`poppy.errors.ModelUnavailableError`
  with an actionable message instead of a raw huggingface_hub traceback; the CLI
  entrypoint renders it as one line.

On top of loading, :class:`FastembedModels` lazy-loads each encoder and starts a
daemon timer after the first load. The timer releases both ONNX sessions after
an idle stretch, so an inactive capture / MCP process reclaims their RAM without
waiting for another inference call.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from poppy.engine._model_cache import fastembed_cache_dir, is_fastembed_model_cached
from poppy.engine._model_holder import IdleUnloadingModels
from poppy.errors import ModelUnavailableError

# fastembed model names. fastembed serves bge-small as a quantized ONNX export
# whose vectors are not interchangeable with the torch build's, so bloom tags
# them with the ONNX-specific ``MODEL_ID`` below.
BI_ENCODER = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER = "Xenova/ms-marco-MiniLM-L-6-v2"
MODEL_ID = "BAAI/bge-small-en-v1.5-onnx"

# CPU-only. CoreML leaks native memory per-op in long-running processes (OMEGA
# profiled this) and the speed difference for single 384-dim embeddings is
# negligible, so pin the CPU provider for the always-on process.
_PROVIDERS = ["CPUExecutionProvider"]

# Default idle-unload timeout. ``POPPY_MODEL_IDLE_S`` overrides it in the shared
# holder when no explicit constructor value is supplied.
_IDLE_TIMEOUT_S = 600.0

# Fastembed defaults cross-encoder reranking to 64 documents per ONNX run. On a
# real 3,112-memory store (2026-07-15), batch 8 cut settled post-recall-5 RSS
# from 2.95-3.01 GB to 1.09-1.12 GB and improved steady recall latency from
# 2.17-2.19s to 1.39-1.73s by reducing activation memory and sequence padding.
# Keep a one-variable rollback for unusual CPUs/document-length distributions.
_RERANK_BATCH_SIZE = 8
_STOCK_RERANK_ENV = "POPPY_ONNX_STOCK_RERANK"

_FIRST_RUN_NOTICE_SHOWN = False


def _silence_fastembed_logs() -> None:
    """Mute fastembed's loguru sink so a cold-cache/offline load fails with our
    single actionable line, not fastembed's internal ERROR dumps.

    fastembed logs the "could not find model / could not load from any source"
    failures at ERROR before raising, which would otherwise print alongside the
    clean message we render in ``_load_fastembed``. Best-effort: loguru ships
    with fastembed, but a missing/renamed logger must never break loading.
    """
    try:
        from loguru import logger  # noqa: PLC0415

        logger.disable("fastembed")
    except Exception:
        pass


def announce_first_run_download(model_names: Sequence[str]) -> None:
    """Print a one-time stderr notice if any of ``model_names`` still needs downloading."""
    global _FIRST_RUN_NOTICE_SHOWN
    if _FIRST_RUN_NOTICE_SHOWN:
        return
    _FIRST_RUN_NOTICE_SHOWN = True
    cold = [m for m in model_names if not is_fastembed_model_cached(m)]
    if cold:
        print(
            f"poppy: downloading ONNX retrieval models ({', '.join(cold)}) on first use. This happens once.",
            file=sys.stderr,
            flush=True,
        )


def _load_fastembed(kind: str, model_name: str, cache_dir: Path):
    """Construct one fastembed model with offline-safe semantics.

    ``kind`` is ``'bi'`` (TextEmbedding) or ``'cross'`` (TextCrossEncoder). A
    cached model loads with ``local_files_only=True`` (no network); an uncached
    model downloads. On a cold cache with no network, raise ModelUnavailableError
    with an actionable message rather than a raw traceback.
    """
    try:
        from fastembed import TextEmbedding
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError as exc:  # pragma: no cover - fastembed is a base dependency
        raise ImportError(
            "fastembed is not installed. It ships in the base `poppy-memory` install; "
            "reinstall it, or run `poppy engines use seed` for keyword-only search."
        ) from exc

    _silence_fastembed_logs()
    ctor = TextEmbedding if kind == "bi" else TextCrossEncoder
    cache = str(cache_dir)
    cached = is_fastembed_model_cached(model_name, cache_dir)
    try:
        return ctor(model_name, cache_dir=cache, providers=_PROVIDERS, local_files_only=cached)
    except Exception as exc:  # noqa: BLE001 - re-raised as an actionable error below
        if cached:
            raise  # present locally; a different failure — surface it as-is
        raise ModelUnavailableError(
            f"Couldn't load ONNX retrieval model {model_name!r} ({type(exc).__name__}: {exc}). "
            "Connect to the internet for the one-time model download, or run "
            "`poppy engines use seed` for offline keyword-only search."
        ) from exc


class FastembedModels(IdleUnloadingModels):
    """Lazy, idle-unloading holder for bloom's bi-encoder + cross-encoder.

    Encoders load on first use (deferring the download-check and ONNX session
    off the import/instantiation path) and unload after ``idle_timeout_s`` of
    inactivity to return the session's RAM to the always-on process; the next
    call reloads them from the local cache. Injected encoders (tests, benchmarks)
    are kept resident and never lazily loaded or unloaded.
    """

    def __init__(
        self,
        bi_encoder=None,
        cross_encoder=None,
        cache_dir: Path | None = None,
        idle_timeout_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cache_dir = cache_dir or fastembed_cache_dir()
        self._announced = False
        self._cross_injected = cross_encoder is not None
        super().__init__(
            bi_encoder=bi_encoder,
            cross_encoder=cross_encoder,
            idle_timeout_s=idle_timeout_s,
            clock=clock,
        )

    # --- lazy load / idle unload ------------------------------------------

    def _announce_once(self) -> None:
        if self._injected or self._announced:
            return
        self._announced = True
        announce_first_run_download((BI_ENCODER, CROSS_ENCODER))

    def _load_model(self, kind: str):
        if os.environ.get("POPPY_TEST_FAIL_ON_MODEL_LOAD") == "1":
            raise RuntimeError("POPPY_TEST_FAIL_ON_MODEL_LOAD: local model loading is forbidden")
        model_name = BI_ENCODER if kind == "bi" else CROSS_ENCODER
        return _load_fastembed(kind, model_name, self._cache_dir)

    # --- inference ---------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        """Return a normalized float32 embedding for ``text``."""
        model = self._model("bi")
        vec = next(iter(model.embed([text])))
        self._mark_used()
        return np.asarray(vec, dtype=np.float32)

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        """Return one relevance score per doc (higher = more relevant)."""
        model = self._model("cross")
        documents = list(docs)
        if self._cross_injected or os.environ.get(_STOCK_RERANK_ENV) == "1":
            raw_scores = model.rerank(query, documents)
        else:
            raw_scores = model.rerank(query, documents, batch_size=_RERANK_BATCH_SIZE)
        scores = [float(s) for s in raw_scores]
        self._mark_used()
        return scores
