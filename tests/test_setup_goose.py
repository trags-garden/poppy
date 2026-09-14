"""Tests for `poppy setup goose` — installs the goose MCP extension + primer."""

from __future__ import annotations

from pathlib import Path

import pytest

from poppy.setup.goose import (
    _extensions_block_span,
    _set_poppy_extension,
    get_goose_config_dir,
    install_for_goose,
    is_goose_installed,
)


@pytest.fixture(autouse=True)
def deterministic_poppy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executable = tmp_path / "resolved-bin" / "poppy"
    monkeypatch.setattr("poppy.setup.claude_code.shutil.which", lambda _name: str(executable))


def expected_poppy_path(tmp_path: Path) -> str:
    return str((tmp_path / "resolved-bin" / "poppy").resolve())


# ---------------------------------------------------------------------------
# install_for_goose — file layout
# ---------------------------------------------------------------------------


def test_install_writes_config_yaml(tmp_path: Path) -> None:
    paths = install_for_goose(goose_config_dir=tmp_path)
    config = (tmp_path / "config.yaml").read_text()
    assert "extensions:" in config
    assert "poppy:" in config
    assert "type: stdio" in config
    assert f"cmd: {expected_poppy_path(tmp_path)}" in config
    assert "      - serve\n      - --source\n      - goose\n" in config
    assert paths["config.yaml"] == tmp_path / "config.yaml"


def test_install_writes_primer(tmp_path: Path) -> None:
    install_for_goose(goose_config_dir=tmp_path)
    hints = tmp_path / ".goosehints"
    assert hints.exists()
    body = hints.read_text()
    assert "<!-- POPPY:BEGIN -->" in body
    assert "<!-- POPPY:END -->" in body
    assert "poppy" in body.lower()


def test_install_is_idempotent(tmp_path: Path) -> None:
    install_for_goose(goose_config_dir=tmp_path)
    snap1 = (tmp_path / "config.yaml").read_text()
    install_for_goose(goose_config_dir=tmp_path)
    snap2 = (tmp_path / "config.yaml").read_text()
    assert snap1 == snap2
    assert is_goose_installed(tmp_path)


def test_install_preserves_other_top_level_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "# Goose config\nGOOSE_PROVIDER: anthropic\nGOOSE_MODEL: claude-haiku-4-5\nexperiments:\n  feature_x: true\n"
    )
    install_for_goose(goose_config_dir=tmp_path)
    body = config_path.read_text()
    assert "# Goose config" in body
    assert "GOOSE_PROVIDER: anthropic" in body
    assert "GOOSE_MODEL: claude-haiku-4-5" in body
    assert "experiments:" in body
    assert "feature_x: true" in body
    assert "extensions:" in body
    assert "poppy:" in body


def test_install_preserves_sibling_extensions(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "extensions:\n"
        "  developer:\n"
        "    type: builtin\n"
        "    name: developer\n"
        "    enabled: true\n"
        "  some-mcp:\n"
        "    type: stdio\n"
        "    name: some-mcp\n"
        "    cmd: npx\n"
        "    args:\n"
        "      - -y\n"
        "      - some-mcp-server\n"
        "    enabled: true\n"
    )
    install_for_goose(goose_config_dir=tmp_path)
    body = config_path.read_text()
    assert "developer:" in body
    assert "type: builtin" in body
    assert "some-mcp:" in body
    assert "some-mcp-server" in body
    assert "poppy:" in body


def test_install_replaces_existing_poppy_entry(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "extensions:\n"
        "  poppy:\n"
        "    type: stdio\n"
        "    name: poppy\n"
        "    cmd: /old/path/to/poppy\n"
        "    args:\n"
        "      - serve\n"
        "      - --legacy\n"
        "    enabled: true\n"
        "  developer:\n"
        "    type: builtin\n"
        "    enabled: true\n"
    )
    install_for_goose(goose_config_dir=tmp_path)
    body = config_path.read_text()
    assert "/old/path/to/poppy" not in body
    assert "--legacy" not in body
    assert f"cmd: {expected_poppy_path(tmp_path)}" in body
    assert "      - serve\n      - --source\n      - goose\n" in body
    # Sibling extension survives.
    assert "developer:" in body
    assert "type: builtin" in body


def test_install_creates_extensions_block_when_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("GOOSE_PROVIDER: anthropic\n")
    install_for_goose(goose_config_dir=tmp_path)
    body = config_path.read_text()
    assert body.startswith("GOOSE_PROVIDER: anthropic\n")
    assert "extensions:" in body
    assert "poppy:" in body


# ---------------------------------------------------------------------------
# is_goose_installed
# ---------------------------------------------------------------------------


def test_is_goose_installed_false_when_no_config(tmp_path: Path) -> None:
    assert is_goose_installed(tmp_path) is False


def test_is_goose_installed_false_when_no_extensions(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("GOOSE_PROVIDER: anthropic\n")
    assert is_goose_installed(tmp_path) is False


def test_is_goose_installed_true_after_install(tmp_path: Path) -> None:
    install_for_goose(goose_config_dir=tmp_path)
    assert is_goose_installed(tmp_path) is True


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_extensions_block_span_finds_block() -> None:
    text = "extensions:\n  foo:\n    bar: baz\nGOOSE_PROVIDER: anthropic\n"
    span = _extensions_block_span(text)
    assert span is not None
    start, end = span
    body = text[start:end]
    assert body == "  foo:\n    bar: baz\n"


def test_extensions_block_span_extends_to_eof() -> None:
    text = "extensions:\n  foo:\n    bar: baz\n"
    span = _extensions_block_span(text)
    assert span is not None
    assert text[span[0] : span[1]] == "  foo:\n    bar: baz\n"


def test_set_poppy_extension_on_empty_file(tmp_path: Path) -> None:
    out = _set_poppy_extension("")
    assert out.startswith("extensions:\n")
    assert "poppy:" in out
    assert f"cmd: {expected_poppy_path(tmp_path)}" in out


def test_get_goose_config_dir_honors_goose_path_root(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "goose-root"
    monkeypatch.delenv("POPPY_GOOSE_CONFIG_DIR", raising=False)
    monkeypatch.setenv("GOOSE_PATH_ROOT", str(root))
    assert get_goose_config_dir() == root / "config"


def test_get_goose_config_dir_prefers_poppy_override(tmp_path: Path, monkeypatch) -> None:
    override = tmp_path / "poppy-override"
    monkeypatch.setenv("POPPY_GOOSE_CONFIG_DIR", str(override))
    monkeypatch.setenv("GOOSE_PATH_ROOT", str(tmp_path / "goose-root"))
    assert get_goose_config_dir() == override
