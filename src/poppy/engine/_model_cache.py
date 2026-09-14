"""Offline-safe model cache probing.

When a model is already in the local cache, ``huggingface_hub`` still fires a
``HEAD`` to huggingface.co on load to check for a newer revision. With the
network fully down (DNS failing), its retry path raises ``RuntimeError: Cannot
send a request, as the client has been closed`` instead of falling back to the
cache, which aborts model loading.

The fix is to load cached models with ``local_files_only=True`` (see
``poppy.engine._fastembed_loader``), which skips the network HEAD entirely so
retrieval works fully offline whenever the model is cached. A genuinely
missing model is left to download normally / surface a clear error.

These helpers deliberately avoid importing ``huggingface_hub`` so they are
cheap to call from read-only paths and safe to call before it is imported.

fastembed downloads its ONNX weights via ``huggingface_hub.snapshot_download``
into an HF-style ``models--<org>--<name>`` layout, but under fastembed's own
``cache_dir`` (which defaults to an ephemeral tempdir). Poppy pins a persistent
cache_dir instead; these helpers resolve that dir and probe it so the offline
guard can tell a cold cache from a merely-network-down one before touching the
network.
"""

from __future__ import annotations

import os
from pathlib import Path

# fastembed maps a friendly model name (what the engine passes) onto the actual
# HF repo it downloads (its quantized ONNX export). ONNX output is NOT
# bit-identical to the same model under torch, which is why the engine tags its
# vectors with an ONNX-specific model_id.
FASTEMBED_HF_SOURCE = {
    "BAAI/bge-small-en-v1.5": "qdrant/bge-small-en-v1.5-onnx-q",
    "Xenova/ms-marco-MiniLM-L-6-v2": "Xenova/ms-marco-MiniLM-L-6-v2",
}


def fastembed_cache_dir() -> Path:
    """Return the persistent cache dir Poppy pins for fastembed downloads.

    ``POPPY_FASTEMBED_CACHE`` overrides (the seam tests use); otherwise
    ``XDG_CACHE_HOME``/``~/.cache`` ``/fastembed``. Deliberately NOT fastembed's
    own default (``$TMPDIR/fastembed_cache``), which macOS purges — a purged
    cache would silently re-download the models on every cold boot.
    """
    explicit = os.environ.get("POPPY_FASTEMBED_CACHE")
    if explicit:
        return Path(explicit)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "fastembed"


def is_fastembed_model_cached(model_name: str, cache_dir: Path | None = None) -> bool:
    """No-network check for whether fastembed's ONNX weights for ``model_name``
    are already in the pinned cache.

    Resolves ``model_name`` to the HF repo fastembed actually downloads (its
    quantized ONNX export) and checks for that snapshot's ``models--<org>--<name>``
    directory. Unknown names fall back to the name itself so the probe degrades
    to a best-effort check rather than crashing.
    """
    cache = cache_dir or fastembed_cache_dir()
    hf_repo = FASTEMBED_HF_SOURCE.get(model_name, model_name)
    return (cache / ("models--" + hf_repo.replace("/", "--"))).is_dir()
