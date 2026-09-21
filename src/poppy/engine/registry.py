"""Engine registry — discoverable, swappable retrieval engines.

Two built-in engines, hardcoded:

  - bloom — the default. Hybrid (FTS5 + bge-small embeddings + RRF) into a
            cross-encoder rerank.
            Runs on fastembed's ONNX runtime, so it works on the plain
            ``pip install poppy-memory`` with no extra.
  - seed  — FTS5 only. No ML deps, no model downloads. The universal floor and
            the automatic fallback when bloom cannot start.

Users pick via ``poppy config set engine <name>`` or
``poppy engines use <name>``. The runtime reads the choice from PoppyConfig
and dispatches through ``resolve_engine``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from poppy.engine.interface import RetrievalEngine

BUILTIN_NAMES = ("bloom", "seed")

# Engine names from earlier releases and the dev-era vocabulary, mapped to their
# current builtin equivalents so a config.json written by an older install keeps
# working. `petal` was this engine's own name before 0.3.0 — same architecture,
# same models, same model_id — so it maps silently, as do speaker_closet/best
# (the promoted champion) and baseline (the FTS-only floor).
LEGACY_ALIASES = {
    "petal": "bloom",
    "speaker_closet": "bloom",
    "best": "bloom",
    "baseline": "seed",
}

# Removed in 0.3.0 (the torch stack moved out of the publish repo). These map to
# bloom too, but loudly: their vectors were written by a different bi-encoder, so
# those rows fall back to FTS-only until `poppy migrate-engine` re-embeds them.
RETIRED_ALIASES = {
    "sprout": "bloom",
}

_retired_notices_emitted: set[str] = set()


def canonical_name(name: str) -> str:
    """Map a legacy engine name to its current builtin; pass others through.

    Retired names get a one-time stderr notice per process — silence would hide
    that the stored vectors no longer match the active engine.
    """
    if name in RETIRED_ALIASES:
        target = RETIRED_ALIASES[name]
        if name not in _retired_notices_emitted:
            _retired_notices_emitted.add(name)
            print(
                f"poppy: engine {name!r} was removed in 0.3.0; using {target!r}. "
                f"Memories embedded by {name!r} are keyword-only until you run `poppy migrate-engine`.",
                file=sys.stderr,
                flush=True,
            )
        return target
    return LEGACY_ALIASES.get(name, name)


_DESCRIPTIONS = {
    "bloom": "Default: hybrid retrieval + cross-encoder rerank (local ONNX models).",
    "seed": "FTS5 only, no ML deps, no model downloads.",
}


@dataclass
class EngineInfo:
    """One row in the engine catalog."""

    name: str
    description: str
    builtin: bool  # always True here; kept for forward-compatibility with the dev registry.
    deps_ok: bool
    deps_error: str | None


def _probe(name: str) -> EngineInfo:
    desc = _DESCRIPTIONS[name]
    if name == "seed":
        return EngineInfo(name=name, description=desc, builtin=True, deps_ok=True, deps_error=None)
    # bloom needs fastembed + numpy, both base dependencies — so it is available
    # on a plain install. A failure here means a broken environment, not a
    # missing extra, so the underlying error is what the user needs to see.
    try:
        import fastembed  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as exc:
        return EngineInfo(name=name, description=desc, builtin=True, deps_ok=False, deps_error=str(exc))
    return EngineInfo(name=name, description=desc, builtin=True, deps_ok=True, deps_error=None)


def list_engines() -> list[EngineInfo]:
    return [_probe(n) for n in BUILTIN_NAMES]


def known_names() -> list[str]:
    return list(BUILTIN_NAMES)


def resolve_engine(name: str, db_path: Path, *, delegate_models_to_daemon: bool = False) -> RetrievalEngine:
    """Instantiate the engine `name` against the real `db_path`.

    Raises ValueError for unknown names and ImportError if the engine's
    dependencies are broken on this machine. The caller decides whether to fall
    back. Legacy names (petal, speaker_closet, best, baseline) are mapped
    silently to their builtin equivalents; retired ones (sprout) map with a
    notice.
    """
    name = canonical_name(name)
    if name == "bloom":
        from poppy.engine.bloom import BloomEngine  # noqa: PLC0415

        return BloomEngine(db_path=db_path, delegate_models_to_daemon=delegate_models_to_daemon)
    if name == "seed":
        from poppy.engine.seed import SeedEngine  # noqa: PLC0415

        return SeedEngine(db_path=db_path)
    raise ValueError(f"Unknown engine: {name!r}. Run `poppy engines` to see the full list.")
