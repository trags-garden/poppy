"""Version single-sourcing tests.

pyproject.toml is the canonical version. poppy.__version__ resolves it via
importlib.metadata when installed, with a pyproject.toml fallback for bare
source checkouts. The mcpb manifest keeps a placeholder that the build stamps.
"""

from __future__ import annotations

import importlib.metadata
import json
import tomllib
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

import poppy
from poppy.cli.main import cli

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def test_dunder_version_matches_installed_metadata() -> None:
    assert poppy.__version__ == importlib.metadata.version("poppy-memory")


def test_dunder_version_matches_pyproject() -> None:
    assert poppy.__version__ == _pyproject_version()


def test_resolve_version_falls_back_to_pyproject_when_not_installed() -> None:
    with patch.object(
        poppy._metadata,
        "version",
        side_effect=importlib.metadata.PackageNotFoundError("poppy-memory"),
    ):
        assert poppy._resolve_version() == _pyproject_version()


def test_cli_version_option(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert f"poppy, version {poppy.__version__}" in result.output


def test_repo_manifest_version_is_placeholder() -> None:
    """The checked-in mcpb manifest must not hardcode a release version; the
    build stamps the pyproject version via _sync_manifest_version()."""
    manifest = json.loads((REPO_ROOT / "mcpb" / "manifest.json").read_text())
    assert manifest["version"] == "0.0.0-dev"
