"""Tests for `poppy build mcpb`."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from poppy import build_mcpb as build_mcpb_mod

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_read_version_matches_pyproject() -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text()
    expected = next(
        line.split("=", 1)[1].strip().strip('"').strip("'")
        for line in text.splitlines()
        if line.startswith("version") and "=" in line
    )
    assert build_mcpb_mod._read_version(REPO_ROOT) == expected


def test_sync_manifest_version_overwrites(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"name": "poppy", "version": "0.0.1"}))

    build_mcpb_mod._sync_manifest_version(tmp_path, "0.9.0")

    assert json.loads(manifest_path.read_text())["version"] == "0.9.0"


def test_build_mcpb_raises_when_cli_missing(tmp_path: Path) -> None:
    """The CLI's absence is the only meaningful failure mode worth surfacing."""
    with mock.patch("poppy.build_mcpb.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="`mcpb` CLI not found"):
            build_mcpb_mod.build_mcpb(repo_root=REPO_ROOT, output_dir=tmp_path)


@pytest.mark.skipif(shutil.which("mcpb") is None, reason="`mcpb` CLI not installed")
def test_build_mcpb_produces_bundle(tmp_path: Path) -> None:
    """End-to-end: produces a non-empty .mcpb whose manifest validates."""
    produced = build_mcpb_mod.build_mcpb(repo_root=REPO_ROOT, output_dir=tmp_path)

    assert produced.exists()
    assert produced.suffix == ".mcpb"
    assert produced.stat().st_size > 50_000  # MCPB is a zip — Poppy source alone clears 50KB easily

    # mcpb validates archive structure during pack; double-check by unpacking.
    extract = tmp_path / "extracted"
    result = subprocess.run(
        ["mcpb", "unpack", str(produced), str(extract)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((extract / "manifest.json").read_text())
    assert manifest["name"] == "poppy-memory"
    assert manifest["server"]["type"] == "uv"
    assert (extract / "server" / "main.py").exists()
    assert (extract / "src" / "poppy" / "__init__.py").exists()
    assert (extract / "pyproject.toml").exists()


def test_shim_resolve_poppy_dir_handles_unexpanded_template(monkeypatch: pytest.MonkeyPatch) -> None:
    """Claude Desktop can pass `${HOME}/.poppy` literally — the shim must not
    create a directory named `${HOME}` in cwd."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("poppy_mcpb_main", REPO_ROOT / "mcpb" / "server" / "main.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    home = Path("/tmp/fake-home").resolve()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("POPPY_DIR", "${HOME}/.poppy")
    # _resolve_poppy_dir() expands $HOME via os.path.expandvars.
    assert module._resolve_poppy_dir() == home / ".poppy"

    # Tilde expansion path.
    monkeypatch.setenv("POPPY_DIR", "~/.poppy")
    assert module._resolve_poppy_dir() == home / ".poppy"

    # Empty / unset falls back to ~/.poppy.
    monkeypatch.setenv("POPPY_DIR", "")
    assert module._resolve_poppy_dir() == home / ".poppy"
    monkeypatch.delenv("POPPY_DIR", raising=False)
    assert module._resolve_poppy_dir() == home / ".poppy"

    # Unresolvable substitution variable falls back rather than creating a literal.
    monkeypatch.setenv("POPPY_DIR", "${UNSET_VAR}/.poppy")
    assert module._resolve_poppy_dir() == home / ".poppy"

    # Real path passes through unchanged.
    monkeypatch.setenv("POPPY_DIR", "/var/data/poppy")
    assert module._resolve_poppy_dir() == Path("/var/data/poppy")


def test_build_mcpb_excludes_pycache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """__pycache__ and ruff/pytest cache dirs must not bloat the bundle."""
    # Drop into a synthetic repo so we can plant __pycache__ deterministically.
    fake_repo = tmp_path / "repo"
    (fake_repo / "src" / "poppy" / "__pycache__").mkdir(parents=True)
    (fake_repo / "src" / "poppy" / "__pycache__" / "junk.pyc").write_bytes(b"x" * 1024)
    (fake_repo / "src" / "poppy" / "__init__.py").write_text("")
    (fake_repo / "src" / "poppy" / "real.py").write_text("# real source\n")

    stage = tmp_path / "stage"
    stage.mkdir()
    build_mcpb_mod._copy_source(fake_repo, stage)

    pkg_dir = stage / "src" / "poppy"
    assert (pkg_dir / "real.py").exists()
    assert not (pkg_dir / "__pycache__").exists()
