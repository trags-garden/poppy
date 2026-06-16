"""Build a `.mcpb` (MCP Bundle) from the Poppy source tree.

The bundle is what Claude Desktop installs with a double-click.
We assemble a minimal staging directory (manifest, server entry shim,
pyproject.toml, src/poppy) and shell out to the `mcpb` CLI to pack it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

# Files copied verbatim from the repo root into the staging dir.
ROOT_FILES = ("pyproject.toml", "README.md", "LICENSE", "uv.lock")

# Subdirectories to ignore when copying src/poppy.
SRC_IGNORES = ("__pycache__", ".pytest_cache", ".ruff_cache")


def _copy_source(repo_root: Path, stage: Path) -> None:
    src_dst = stage / "src" / "poppy"
    shutil.copytree(
        repo_root / "src" / "poppy",
        src_dst,
        ignore=shutil.ignore_patterns(*SRC_IGNORES),
    )

    for name in ROOT_FILES:
        f = repo_root / name
        if f.exists():
            shutil.copy2(f, stage / name)


def _copy_mcpb_inputs(repo_root: Path, stage: Path) -> None:
    mcpb_src = repo_root / "mcpb"
    shutil.copy2(mcpb_src / "manifest.json", stage / "manifest.json")
    shutil.copy2(mcpb_src / "icon.png", stage / "icon.png")
    shutil.copytree(mcpb_src / "server", stage / "server")


def _read_version(repo_root: Path) -> str:
    pyproject = (repo_root / "pyproject.toml").read_text()
    for line in pyproject.splitlines():
        if line.startswith("version") and "=" in line:
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("version not found in pyproject.toml")


def _sync_manifest_version(stage: Path, version: str) -> None:
    manifest_path = stage / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != version:
        manifest["version"] = version
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def build_mcpb(repo_root: Path, output_dir: Path) -> Path:
    """Build a .mcpb bundle and return the path to the produced file.

    Raises RuntimeError if the `mcpb` CLI is missing or pack fails.
    """
    if shutil.which("mcpb") is None:
        raise RuntimeError("`mcpb` CLI not found — install with `npm install -g @anthropic-ai/mcpb`")

    version = _read_version(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = output_dir / f"poppy-memory-{version}-stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    _copy_mcpb_inputs(repo_root, stage)
    _copy_source(repo_root, stage)
    _sync_manifest_version(stage, version)

    output_path = output_dir / f"poppy-memory-{version}.mcpb"
    if output_path.exists():
        output_path.unlink()

    result = subprocess.run(
        ["mcpb", "pack", str(stage), str(output_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"mcpb pack failed:\n{result.stderr}\n{result.stdout}")

    shutil.rmtree(stage)
    return output_path
