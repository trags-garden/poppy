"""Tests for `poppy setup hermes-agent` — installs the hermes memory plugin."""

from __future__ import annotations

from pathlib import Path

from poppy.setup.hermes import (
    HERMES_PLUGIN_NAME,
    HERMES_SOUL_BODY,
    _set_memory_provider,
    install_for_hermes,
    is_hermes_installed,
)

POPPY_BEGIN = "<!-- POPPY:BEGIN -->"
POPPY_END = "<!-- POPPY:END -->"

# ---------------------------------------------------------------------------
# install_for_hermes — file layout
# ---------------------------------------------------------------------------


def test_install_writes_plugin_dir(tmp_path: Path) -> None:
    paths = install_for_hermes(hermes_home=tmp_path)

    plugin_dir = tmp_path / "plugins" / HERMES_PLUGIN_NAME
    assert plugin_dir.is_dir()
    assert (plugin_dir / "plugin.yaml").exists()
    assert (plugin_dir / "__init__.py").exists()
    assert (plugin_dir / "README.md").exists()

    assert paths["Plugin dir"] == plugin_dir
    assert paths["plugin.yaml"] == plugin_dir / "plugin.yaml"
    assert paths["__init__.py"] == plugin_dir / "__init__.py"


def test_install_writes_hermes_guidance_to_fresh_soul_md(tmp_path: Path) -> None:
    paths = install_for_hermes(hermes_home=tmp_path)
    soul_md = tmp_path / "SOUL.md"

    assert paths["Memory guidance (SOUL.md)"] == soul_md
    assert soul_md.read_text() == f"{POPPY_BEGIN}\n{HERMES_SOUL_BODY}\n{POPPY_END}\n"
    assert not (tmp_path / "AGENTS.md").exists()
    for tool_name in ("poppy_recall", "poppy_remember", "poppy_forget", "poppy_status"):
        assert tool_name in HERMES_SOUL_BODY


def test_install_writes_config(tmp_path: Path) -> None:
    install_for_hermes(hermes_home=tmp_path)
    config = (tmp_path / "config.yaml").read_text()
    assert "memory:" in config
    assert "provider: poppy" in config


def test_install_is_idempotent(tmp_path: Path) -> None:
    install_for_hermes(hermes_home=tmp_path)
    config_snap1 = (tmp_path / "config.yaml").read_text()
    soul_snap1 = (tmp_path / "SOUL.md").read_text()
    install_for_hermes(hermes_home=tmp_path)
    assert (tmp_path / "config.yaml").read_text() == config_snap1
    assert (tmp_path / "SOUL.md").read_text() == soul_snap1
    assert soul_snap1.count(POPPY_BEGIN) == 1
    assert is_hermes_installed(tmp_path)


def test_install_preserves_other_config_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("# Hermes config\nlogging:\n  level: info\nmodel:\n  name: claude-haiku-4-5\n")
    install_for_hermes(hermes_home=tmp_path)
    body = config_path.read_text()
    assert "# Hermes config" in body
    assert "logging:" in body
    assert "level: info" in body
    assert "model:" in body
    assert "name: claude-haiku-4-5" in body
    assert "provider: poppy" in body


def test_install_replaces_existing_provider(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("memory:\n  provider: honcho\n")
    install_for_hermes(hermes_home=tmp_path)
    body = config_path.read_text()
    assert "provider: poppy" in body
    assert "provider: honcho" not in body


def test_install_preserves_memory_block_subkeys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("memory:\n  scope: profile\n  provider: honcho\n  ttl: 30d\n")
    install_for_hermes(hermes_home=tmp_path)
    body = config_path.read_text()
    assert "scope: profile" in body
    assert "ttl: 30d" in body
    assert "provider: poppy" in body
    assert "provider: honcho" not in body


def test_install_appends_guidance_to_existing_soul_md(tmp_path: Path) -> None:
    soul_md = tmp_path / "SOUL.md"
    user_content = "# My persona\n\nBe thoughtful.\n"
    soul_md.write_text(user_content)

    install_for_hermes(hermes_home=tmp_path)

    assert soul_md.read_text() == f"{user_content}\n{POPPY_BEGIN}\n{HERMES_SOUL_BODY}\n{POPPY_END}\n"


def test_install_replaces_stale_soul_block_in_place(tmp_path: Path) -> None:
    soul_md = tmp_path / "SOUL.md"
    soul_md.write_text(f"prefix\n{POPPY_BEGIN}\nstale guidance\n{POPPY_END}\nsuffix\n")

    install_for_hermes(hermes_home=tmp_path)

    assert soul_md.read_text() == f"prefix\n{POPPY_BEGIN}\n{HERMES_SOUL_BODY}\n{POPPY_END}\nsuffix\n"


def test_install_deletes_legacy_agents_md_containing_only_managed_block(tmp_path: Path) -> None:
    agents_md = tmp_path / "AGENTS.md"
    agents_md.write_text(f"  \n{POPPY_BEGIN}\nlegacy primer\n{POPPY_END}\n")

    install_for_hermes(hermes_home=tmp_path)

    assert not agents_md.exists()


def test_install_removes_legacy_block_and_preserves_agents_md_content(tmp_path: Path) -> None:
    agents_md = tmp_path / "AGENTS.md"
    agents_md.write_text(f"prefix\n{POPPY_BEGIN}\nlegacy primer\n{POPPY_END}\nsuffix\n")

    install_for_hermes(hermes_home=tmp_path)
    first_migration = agents_md.read_text()
    install_for_hermes(hermes_home=tmp_path)

    assert first_migration == "prefix\n\nsuffix\n"
    assert agents_md.read_text() == first_migration


# ---------------------------------------------------------------------------
# _set_memory_provider — unit tests on the YAML hand-merger
# ---------------------------------------------------------------------------


def test_set_provider_empty_input() -> None:
    out = _set_memory_provider("", "poppy")
    assert out == "memory:\n  provider: poppy\n"


def test_set_provider_no_memory_block() -> None:
    text = "logging:\n  level: info\n"
    out = _set_memory_provider(text, "poppy")
    assert "logging:" in out
    assert "level: info" in out
    assert out.endswith("memory:\n  provider: poppy\n")


def test_set_provider_no_trailing_newline() -> None:
    text = "logging:\n  level: info"
    out = _set_memory_provider(text, "poppy")
    assert "memory:\n  provider: poppy\n" in out


def test_set_provider_replaces_inline_comment() -> None:
    text = "memory:\n  provider: honcho  # legacy\n"
    out = _set_memory_provider(text, "poppy")
    assert "provider: poppy" in out
    assert "# legacy" in out  # inline comment preserved


def test_set_provider_preserves_neighbor_keys() -> None:
    text = "memory:\n  scope: profile\n  provider: honcho\n  ttl: 30d\nother:\n  key: value\n"
    out = _set_memory_provider(text, "poppy")
    assert "scope: profile" in out
    assert "ttl: 30d" in out
    assert "other:\n  key: value" in out
    assert "provider: poppy" in out


# ---------------------------------------------------------------------------
# is_hermes_installed
# ---------------------------------------------------------------------------


def test_is_installed_false_before_setup(tmp_path: Path) -> None:
    assert not is_hermes_installed(tmp_path)


def test_is_installed_true_after_setup(tmp_path: Path) -> None:
    install_for_hermes(hermes_home=tmp_path)
    assert is_hermes_installed(tmp_path)


def test_is_installed_false_when_provider_changed(tmp_path: Path) -> None:
    install_for_hermes(hermes_home=tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_path.read_text().replace("provider: poppy", "provider: honcho"))
    assert not is_hermes_installed(tmp_path)


# ---------------------------------------------------------------------------
# Plugin file content is loadable Python
# ---------------------------------------------------------------------------


def test_plugin_init_is_valid_python(tmp_path: Path) -> None:
    """The rendered __init__.py must parse cleanly — would catch escaping bugs."""
    import ast

    install_for_hermes(hermes_home=tmp_path)
    init_src = (tmp_path / "plugins" / "poppy" / "__init__.py").read_text()
    ast.parse(init_src)  # raises SyntaxError on failure


def test_plugin_init_references_memory_provider_abc(tmp_path: Path) -> None:
    install_for_hermes(hermes_home=tmp_path)
    init_src = (tmp_path / "plugins" / "poppy" / "__init__.py").read_text()
    assert "from agent.memory_provider import MemoryProvider" in init_src
    assert "class PoppyMemoryProvider(MemoryProvider)" in init_src
    assert "def register(ctx)" in init_src
    # Sanity: tool schemas the agent will see
    for tool in ("poppy_recall", "poppy_remember", "poppy_forget", "poppy_status"):
        assert tool in init_src
