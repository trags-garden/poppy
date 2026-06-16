"""Engine registry — discoverable, swappable retrieval engines.

Three built-in engines, hardcoded:

  - bloom  — local champion. Hybrid (FTS5 + bge-small embeddings + RRF) →
             cross-encoder rerank, with per-speaker content expansion at
             ingest. Default. Floats to whatever the current champion is.
  - sprout — mid tier. Same two-stage architecture as bloom but a lighter
             bi-encoder (all-MiniLM-L6-v2) and no closet expansion.
  - seed   — FTS5 only. No ML deps, no model downloads. The universal floor.

Users pick via ``poppy config set engine <name>`` or
``poppy engines use <name>``. The runtime reads the choice from PoppyConfig
and dispatches through ``resolve_engine``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from poppy.engine.interface import RetrievalEngine

BUILTIN_NAMES = ("bloom", "sprout", "seed")

# Engine names from the dev-era vocabulary, mapped silently to their current
# builtin equivalents so a config.json written by an older install keeps
# working: speaker_closet and best both point at the promoted champion
# (bloom), baseline at the FTS-only floor (seed).
LEGACY_ALIASES = {
    "speaker_closet": "bloom",
    "best": "bloom",
    "baseline": "seed",
}


def canonical_name(name: str) -> str:
    """Map a legacy engine name to its current builtin; pass others through."""
    return LEGACY_ALIASES.get(name, name)


_DESCRIPTIONS = {
    "bloom": "Local champion: hybrid + cross-encoder rerank + per-speaker expansion (~600MB deps).",
    "sprout": "Mid: hybrid (smaller bi-encoder) + cross-encoder rerank.",
    "seed": "FTS5 only — no ML deps, no model downloads.",
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
    # bloom and sprout both need sentence-transformers + numpy (in base deps).
    try:
        import numpy  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError as exc:
        return EngineInfo(name=name, description=desc, builtin=True, deps_ok=False, deps_error=str(exc))
    return EngineInfo(name=name, description=desc, builtin=True, deps_ok=True, deps_error=None)


def list_engines() -> list[EngineInfo]:
    return [_probe(n) for n in BUILTIN_NAMES]


def known_names() -> list[str]:
    return list(BUILTIN_NAMES)


def resolve_engine(name: str, db_path: Path) -> RetrievalEngine:
    """Instantiate the engine `name` against the real `db_path`.

    Raises ValueError for unknown names and ImportError if the engine's
    optional dependencies are missing on this machine. The caller decides
    whether to fall back. Legacy names (speaker_closet, best, baseline) are
    mapped silently to their builtin equivalents.
    """
    name = canonical_name(name)
    if name == "bloom":
        from poppy.engine.bloom import BloomEngine  # noqa: PLC0415

        return BloomEngine(db_path=db_path)
    if name == "sprout":
        from poppy.engine.sprout import SproutEngine  # noqa: PLC0415

        return SproutEngine(db_path=db_path)
    if name == "seed":
        from poppy.engine.seed import SeedEngine  # noqa: PLC0415

        return SeedEngine(db_path=db_path)
    raise ValueError(f"Unknown engine: {name!r}. Run `poppy engines` to see the full list.")
