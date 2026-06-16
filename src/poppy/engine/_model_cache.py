"""Offline-safe Hugging Face model cache probing.

When an embedding/cross-encoder model is already in the local HF cache,
``sentence-transformers`` / ``huggingface_hub`` still fire a ``HEAD`` to
huggingface.co on load to check for a newer revision. With the network fully
down (DNS failing), huggingface_hub's retry path raises
``RuntimeError: Cannot send a request, as the client has been closed`` instead
of falling back to the cache, which aborts model loading.

The fix is to load cached models with ``local_files_only=True`` (see
``poppy.engine._st_loader``), which skips the network HEAD entirely so
retrieval works fully offline whenever the model is cached. A genuinely
missing model is left to download normally / surface a clear error.

These helpers deliberately avoid importing ``huggingface_hub`` so they are
cheap to call from read-only paths and safe to call before it is imported.
"""

from __future__ import annotations

import os
from pathlib import Path


def hf_hub_cache_dir() -> Path:
    """Return the HF hub cache directory without importing huggingface_hub.

    Mirrors huggingface_hub's own resolution order (``HF_HUB_CACHE`` >
    ``HF_HOME`` > ``XDG_CACHE_HOME`` > ``~/.cache``) so this stays correct while
    remaining import-free.
    """
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return Path(explicit)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "huggingface" / "hub"


def is_model_cached(repo_id: str) -> bool:
    """Best-effort, no-network check for whether the HF cache holds ``repo_id``.

    A bare model name (no ``/``) is resolved by sentence-transformers to the
    ``sentence-transformers/`` org, so the on-disk cache dir is namespaced even
    though the engine refers to it unqualified (e.g. ``all-MiniLM-L6-v2`` ->
    ``models--sentence-transformers--all-MiniLM-L6-v2``). Check both forms.
    """
    cache = hf_hub_cache_dir()
    candidates = [repo_id]
    if "/" not in repo_id:
        candidates.append(f"sentence-transformers/{repo_id}")
    return any((cache / ("models--" + c.replace("/", "--"))).is_dir() for c in candidates)
