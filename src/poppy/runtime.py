"""Shared runtime helpers used by CLI, MCP server, and hooks."""

import os
from collections.abc import Callable
from pathlib import Path

from poppy.engine.interface import RetrievalEngine
from poppy.engine.seed import SeedEngine
from poppy.errors import ModelUnavailableError
from poppy.paths import ensure_poppy_dir


def get_poppy_dir() -> Path:
    env_dir = os.environ.get("POPPY_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".poppy"


def _engine_failure_description(name: str, exc: Exception) -> str:
    detail_lines = str(exc).splitlines()
    detail = detail_lines[0].strip() if detail_lines else ""
    description = f"{name}: {type(exc).__name__}"
    if detail:
        description += f": {detail}"
    return description if len(description) <= 240 else description[:237] + "..."


def get_engine(
    poppy_dir: Path | None = None,
    *,
    delegate_models_to_daemon: bool = False,
    download_models: bool = True,
    error_sink: Callable[[str], None] | None = None,
) -> RetrievalEngine:
    """Build the full pipeline, falling back to keyword search when models cannot load.

    Legacy engine settings have no effect. Model checks happen here so an
    offline cold cache falls back before a read or write starts. Database and
    other unexpected errors still propagate rather than hiding a broken store.
    """
    if os.environ.get("POPPY_TEST_FAIL_ON_ENGINE") == "1":
        raise RuntimeError("POPPY_TEST_FAIL_ON_ENGINE: local engine construction is forbidden")

    from poppy.config import load_config
    from poppy.engine.registry import resolve_engine

    poppy_dir = poppy_dir or get_poppy_dir()
    ensure_poppy_dir(poppy_dir)
    load_config(poppy_dir)
    db_path = poppy_dir / "memories.db"

    engine = None
    try:
        engine = resolve_engine("bloom", db_path, delegate_models_to_daemon=delegate_models_to_daemon)
        engine.check_models(download=download_models)
        return engine
    except (ImportError, ModelUnavailableError) as exc:
        if error_sink is not None:
            error_sink(_engine_failure_description("bloom", exc))
        if engine is not None:
            engine._conn.close()
        return resolve_engine("seed", db_path)
    except Exception:
        if engine is not None:
            engine._conn.close()
        raise


def get_fast_engine(poppy_dir: Path | None = None) -> RetrievalEngine:
    """Return a model-load-free engine. Used in hooks where per-call latency matters.

    SeedEngine is FTS5 + SQLite only — no embedding model, no cross-encoder.
    Recall quality is lower than BloomEngine but the hook starts in well
    under a second, which keeps SessionStart / UserPromptSubmit / PreToolUse
    hooks snappy.
    """
    poppy_dir = poppy_dir or get_poppy_dir()
    ensure_poppy_dir(poppy_dir)
    db_path = poppy_dir / "memories.db"
    return SeedEngine(db_path=db_path)
