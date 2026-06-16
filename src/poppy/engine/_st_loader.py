"""Offline-aware sentence-transformers model loading.

Shared by the engines that download Hugging Face models (bloom, sprout).
Two behaviors:

* A model already in the local HF cache is loaded with ``local_files_only=True``
  so the load-time huggingface.co HEAD update-check (which crashes when the
  network is fully down) is skipped. An uncached model downloads normally.
* On a cold cache with no network, the raw stack trace is replaced by a
  :class:`poppy.errors.ModelUnavailableError` carrying an actionable message;
  the CLI entrypoint (``poppy.cli.main.main``) renders it as one line.

A one-time stderr notice announces the first-run model download so a hung-
looking first ``poppy recall`` is explained.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from poppy.engine._model_cache import is_model_cached
from poppy.errors import ModelUnavailableError

_FIRST_RUN_NOTICE_SHOWN = False


def announce_first_run_download(repo_ids: Sequence[str]) -> None:
    """Print a one-time stderr notice if any of ``repo_ids`` still needs downloading."""
    global _FIRST_RUN_NOTICE_SHOWN
    if _FIRST_RUN_NOTICE_SHOWN:
        return
    _FIRST_RUN_NOTICE_SHOWN = True
    cold = [r for r in repo_ids if not is_model_cached(r)]
    if cold:
        print(
            f"poppy: downloading retrieval models ({', '.join(cold)}) on first use. This happens once.",
            file=sys.stderr,
            flush=True,
        )


def load_st_model(kind: str, repo_id: str):
    """Construct one sentence-transformers model with offline-safe semantics.

    ``kind`` is ``'bi'`` (SentenceTransformer) or ``'cross'`` (CrossEncoder).
    A cached model is loaded with ``local_files_only=True`` so the load-time
    huggingface.co HEAD (which crashes when the network is fully down) is
    skipped; an uncached model keeps ``local_files_only=False`` so a first-run
    download still works. On a cold cache with no network, raise
    ModelUnavailableError with an actionable message rather than a raw stack
    trace.
    """
    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer

        local_only = is_model_cached(repo_id)
        if kind == "bi":
            return SentenceTransformer(repo_id, local_files_only=local_only)
        return CrossEncoder(repo_id, local_files_only=local_only)
    except Exception as exc:  # noqa: BLE001 - re-raised as an actionable error below
        if is_model_cached(repo_id):
            raise  # cached locally; this is a different failure, surface it as-is
        raise ModelUnavailableError(
            f"Couldn't load retrieval model {repo_id!r} ({type(exc).__name__}: {exc}). "
            "Connect to the internet for the one-time model download, or run "
            "`poppy engines use seed` for offline keyword-only search."
        ) from exc
