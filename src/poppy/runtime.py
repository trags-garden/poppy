"""Shared runtime helpers used by CLI, MCP server, and hooks."""

import os
from collections.abc import Callable
from pathlib import Path

from poppy.engine.interface import RetrievalEngine
from poppy.engine.seed import SeedEngine
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
    error_sink: Callable[[str], None] | None = None,
) -> RetrievalEngine:
    """Build the active engine for the product surface.

    Reads the engine choice from PoppyConfig (`poppy config set engine <name>`
    / `poppy engines use <name>`). Default is ``bloom``.

    Fallback semantics:

    * ``ImportError`` / unknown name -> walk a best-first chain
      ``bloom`` -> ``seed`` (skipping the one that just failed), reporting
      failures through ``error_sink`` when supplied.
    * ``ModelUnavailableError`` is deliberately NOT caught: an offline cold
      cache on the selected (or fallen-to) embedding engine surfaces its own
      actionable "connect to download / use seed" message instead of silently
      degrading to keyword search.
    * Any other exception (schema migration error, disk I/O, sqlite
      corruption) is raised loud. Silent fallback would mask a broken DB by
      serving toy FTS-only results from a different schema; the user must
      see the failure.
    """
    if os.environ.get("POPPY_TEST_FAIL_ON_ENGINE") == "1":
        raise RuntimeError("POPPY_TEST_FAIL_ON_ENGINE: local engine construction is forbidden")

    from poppy.config import load_config
    from poppy.engine.registry import resolve_engine

    poppy_dir = poppy_dir or get_poppy_dir()
    ensure_poppy_dir(poppy_dir)
    db_path = poppy_dir / "memories.db"

    name = load_config(poppy_dir).engine
    try:
        return resolve_engine(name, db_path, delegate_models_to_daemon=delegate_models_to_daemon)
    except (ImportError, ValueError) as e:
        if error_sink is not None:
            error_sink(_engine_failure_description(name, e))
        # Best-first substitutes: bloom is the full closet-hybrid engine; seed is
        # the guaranteed FTS-only floor.
        last_exc: Exception = e
        for candidate in ("bloom", "seed"):
            if candidate == name:
                continue
            try:
                return resolve_engine(candidate, db_path, delegate_models_to_daemon=delegate_models_to_daemon)
            except (ImportError, ValueError) as ex:
                if error_sink is not None:
                    error_sink(_engine_failure_description(candidate, ex))
                last_exc = ex
                continue
        # seed never raises ImportError/ValueError, so this is effectively
        # unreachable; keep it defensive rather than swallow the failure.
        raise last_exc


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
