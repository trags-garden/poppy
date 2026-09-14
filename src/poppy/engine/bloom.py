"""Bloom — the local retrieval engine (ONNX / fastembed, no torch).

Bloom is the closet-hybrid architecture
(:class:`poppy.engine._closet_engine.ClosetHybridEngine`): two-stage retrieval
(hybrid FTS5+embeddings -> cross-encoder rerank) with a per-speaker content
expansion at ingest, served over fastembed's ONNX runtime. That keeps it
installable from a plain ``pip install poppy-memory`` (fastembed + onnxruntime
are ~200 MB total) and cheap to keep resident in the always-on capture / MCP
process.

Models (all local, no API):
  - Bi-encoder:    BAAI/bge-small-en-v1.5  (fastembed's quantized ONNX export)
  - Cross-encoder: Xenova/ms-marco-MiniLM-L-6-v2

Bloom tags every vector it writes with the ONNX ``model_id``
(``BAAI/bge-small-en-v1.5-onnx``). The shared ``memory_embeddings`` table keys
on model_id, so rows written by a different bi-encoder — including the torch
engines Poppy shipped before 0.3.0 — go FTS-only until ``poppy migrate-engine``
re-embeds them, and two vector spaces never mix inside RRF.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from poppy.db import rollback_and_close
from poppy.engine._closet_engine import ClosetHybridEngine
from poppy.engine._fastembed_loader import MODEL_ID, FastembedModels

__all__ = ["BloomEngine"]


class BloomEngine(ClosetHybridEngine):
    """Retrieval over fastembed ONNX encoders."""

    # Unchanged across the 0.3.0 engine collapse: this is the same ONNX
    # fingerprint the engine has always written, so stores indexed by earlier
    # versions keep their vectors and need no re-embedding.
    model_id = MODEL_ID
    _engine_name = "bloom"
    _engine_version = "1.0.0"

    def __init__(
        self,
        db_path: Path,
        bi_encoder=None,
        cross_encoder=None,
        *,
        delegate_models_to_daemon: bool = False,
    ) -> None:
        super().__init__(db_path)
        try:
            # FastembedModels owns lazy-load + idle-unload; injected encoders (tests,
            # benchmarks) bypass both and stay resident.
            def local_factory():
                return FastembedModels(bi_encoder=bi_encoder, cross_encoder=cross_encoder)

            if delegate_models_to_daemon:
                from poppy.engine._daemon_models import daemon_fallback_models

                self._models = daemon_fallback_models(db_path.parent, self._engine_name, local_factory)
            else:
                self._models = local_factory()
        except Exception:
            rollback_and_close(self._conn)
            raise

    def _embed(self, text: str) -> np.ndarray:
        return self._models.embed(text)

    def _rerank(self, query: str, docs: list[str]) -> list[float]:
        return self._models.rerank(query, docs)
