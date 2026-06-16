"""Shared runtime helpers used by CLI, MCP server, and hooks."""

import os
from pathlib import Path

from poppy.engine.interface import RetrievalEngine
from poppy.engine.seed import SeedEngine


def get_poppy_dir() -> Path:
    env_dir = os.environ.get("POPPY_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".poppy"


def get_engine(poppy_dir: Path | None = None) -> RetrievalEngine:
    """Build the active engine for the product surface.

    Reads the engine choice from PoppyConfig (`poppy config set engine <name>`
    / `poppy engines use <name>`). Default is ``bloom`` — the local champion.

    Fallback semantics:

    * ``ImportError`` / unknown name -> walk down ``bloom`` -> ``sprout`` ->
      ``seed`` with a stderr warning. Covers minimal installs missing the
      optional ML wheels. ML deps ship in base, so this path should fire
      only on genuinely broken envs.
    * Any other exception (schema migration error, disk I/O, sqlite
      corruption) is raised loud. Silent fallback would mask a broken DB by
      serving toy FTS-only results from a different schema; the user must
      see the failure.
    """
    import sys

    from poppy.config import load_config
    from poppy.engine.registry import resolve_engine

    poppy_dir = poppy_dir or get_poppy_dir()
    poppy_dir.mkdir(parents=True, exist_ok=True)
    db_path = poppy_dir / "memories.db"

    name = load_config(poppy_dir).engine
    try:
        return resolve_engine(name, db_path)
    except (ImportError, ValueError) as e:
        if name not in ("bloom", "sprout"):
            print(f"poppy: engine {name!r} unavailable ({e}); falling back.", file=sys.stderr)
        if name != "bloom":
            try:
                from poppy.engine.bloom import BloomEngine

                return BloomEngine(db_path=db_path)
            except ImportError:
                pass
        try:
            from poppy.engine.sprout import SproutEngine

            return SproutEngine(db_path=db_path)
        except ImportError:
            return SeedEngine(db_path=db_path)


def get_fast_engine(poppy_dir: Path | None = None) -> RetrievalEngine:
    """Return a model-load-free engine. Used in hooks where per-call latency matters.

    SeedEngine is FTS5 + SQLite only — no embedding model, no cross-encoder.
    Recall quality is lower than BloomEngine but the hook starts in well
    under a second, which keeps SessionStart / UserPromptSubmit / PreToolUse
    hooks snappy.
    """
    poppy_dir = poppy_dir or get_poppy_dir()
    poppy_dir.mkdir(parents=True, exist_ok=True)
    db_path = poppy_dir / "memories.db"
    return SeedEngine(db_path=db_path)
