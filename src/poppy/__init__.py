"""Poppy package metadata.

The version is single-sourced from pyproject.toml: installed distributions
resolve it via importlib.metadata; bare source checkouts (no dist-info on
sys.path) fall back to reading pyproject.toml next to the package.
"""

from importlib import metadata as _metadata


def _resolve_version() -> str:
    try:
        return _metadata.version("poppy-memory")
    except _metadata.PackageNotFoundError:
        pass
    try:
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        with pyproject.open("rb") as f:
            return tomllib.load(f)["project"]["version"]
    except Exception:
        return "0.0.0+unknown"


__version__ = _resolve_version()
