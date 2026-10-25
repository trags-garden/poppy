"""Version single-sourcing tests.

pyproject.toml is the canonical version. poppy.__version__ resolves it via
importlib.metadata when installed, with a pyproject.toml fallback for bare
source checkouts.
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


def test_registry_versions_match_pyproject() -> None:
    server = json.loads((REPO_ROOT / "server.json").read_text())
    version = _pyproject_version()
    assert server["version"] == version
    assert server["packages"][0]["version"] == version


def test_registry_package_starts_stdio_server() -> None:
    server = json.loads((REPO_ROOT / "server.json").read_text())
    package = server["packages"][0]
    assert server["name"] == "ai.trags/poppy"
    assert package["registryType"] == "pypi"
    assert package["identifier"] == "poppy-memory"
    assert package["transport"] == {"type": "stdio"}
    command = [package["runtimeHint"]]
    for arg in package.get("runtimeArguments", []):
        if arg["type"] == "named":
            command.append(arg["name"])
        command.append(arg["value"])
    command.append(f"{package['identifier']}@{package['version']}")
    for arg in package["packageArguments"]:
        if arg["type"] == "named":
            command.append(arg["name"])
        command.append(arg["value"])
    assert command == ["uvx", f"poppy-memory@{_pyproject_version()}", "serve"]
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        scripts = tomllib.load(f)["project"]["scripts"]
    assert scripts[package["identifier"]] == scripts["poppy"] == "poppy.cli.main:main"
    result = CliRunner().invoke(cli, [*command[2:], "--help"], prog_name=package["identifier"])
    assert result.exit_code == 0
    assert "Usage: poppy-memory serve" in result.output
    assert "<!-- mcp-name: ai.trags/poppy -->" in (REPO_ROOT / "README.md").read_text().splitlines()


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


def test_cli_version_short_alias(tmp_path: Path) -> None:
    """`poppy -v` must be an alias for `--version`, not error."""
    runner = CliRunner()
    result = runner.invoke(cli, ["-v"], env={"POPPY_DIR": str(tmp_path)})
    assert result.exit_code == 0
    assert f"poppy, version {poppy.__version__}" in result.output


def test_claude_code_plugin_version_matches_pyproject() -> None:
    manifest = json.loads((REPO_ROOT / "plugins" / "poppy" / ".claude-plugin" / "plugin.json").read_text())
    assert manifest["version"] == _pyproject_version()
