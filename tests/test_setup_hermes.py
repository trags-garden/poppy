"""Tests for `poppy setup hermes-agent` — installs the hermes memory plugin."""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import types
from importlib import resources
from pathlib import Path

import pytest
from click.testing import CliRunner

from poppy.cli.main import cli
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
    assert out == "memory:\n  provider: poppy  # legacy\n"


def test_set_provider_preserves_neighbor_keys() -> None:
    text = "memory:\n  scope: profile\n  provider: honcho\n  ttl: 30d\nother:\n  key: value\n"
    out = _set_memory_provider(text, "poppy")
    assert "scope: profile" in out
    assert "ttl: 30d" in out
    assert "other:\n  key: value" in out
    assert "provider: poppy" in out


@pytest.mark.parametrize("line", ["\t\t\n", "  key: value\n"])
def test_set_provider_long_block_without_provider_finishes_fast(line: str) -> None:
    # A child process lets the timeout stop a regressed regex that never returns.
    code = (
        f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / 'src')!r})\n"
        "from poppy.setup.hermes import _set_memory_provider\n"
        f"lines = {line!r} * 40\n"
        "assert _set_memory_provider('memory:\\n' + lines, 'poppy') "
        "== 'memory:\\n  provider: poppy\\n' + lines\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=3)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("memory:\n  scope: profile\n", "memory:\n  provider: poppy\n  scope: profile\n"),
        (
            "memory:\n\t\t\n  key: value\n\tprovider:\thoncho  # legacy\nother: unchanged\n",
            "memory:\n\t\t\n  key: value\n\tprovider:\tpoppy  # legacy\nother: unchanged\n",
        ),
        ("memory:\n  provider: honcho", "memory:\n  provider: poppy"),
        ("memory:\n  provider: honcho  \n", "memory:\n  provider: poppy\n"),
        ("memory:\n  provider: honcho#legacy\n", "memory:\n  provider: poppy#legacy\n"),
        ("memory:\n  provider:   # c\n", "memory:\n  provider:   poppy # c\n"),
        ("memory:\n  provider: honcho\t # legacy\n", "memory:\n  provider: poppy\t # legacy\n"),
        (
            "memory:\n  scope: profile\nother:\n  provider: honcho\n",
            "memory:\n  provider: poppy\n  scope: profile\nother:\n  provider: honcho\n",
        ),
        (
            "memory:\n  scope: profile\nmemory:\n  provider: honcho\n",
            "memory:\n  scope: profile\nmemory:\n  provider: poppy\n",
        ),
    ],
)
def test_set_provider_preserves_exact_content(text: str, expected: str) -> None:
    assert _set_memory_provider(text, "poppy") == expected


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


def test_plugin_init_is_the_packaged_file_byte_for_byte(tmp_path: Path) -> None:
    install_for_hermes(hermes_home=tmp_path)
    packaged = resources.files("poppy.setup").joinpath("hermes_plugin.py.txt").read_bytes()
    assert (tmp_path / "plugins" / "poppy" / "__init__.py").read_bytes() == packaged


def test_plugin_init_references_memory_provider_abc(tmp_path: Path) -> None:
    install_for_hermes(hermes_home=tmp_path)
    init_src = (tmp_path / "plugins" / "poppy" / "__init__.py").read_text()
    assert "from agent.memory_provider import MemoryProvider" in init_src
    assert "class PoppyMemoryProvider(MemoryProvider)" in init_src
    assert "def register(ctx)" in init_src
    # Sanity: tool schemas the agent will see
    for tool in ("poppy_recall", "poppy_remember", "poppy_forget", "poppy_status"):
        assert tool in init_src


# ---------------------------------------------------------------------------
# Plugin behaviour: load the installed file against stub Hermes modules
# ---------------------------------------------------------------------------


@pytest.fixture
def plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Import the installed plugin with Hermes' two imports stubbed and the
    poppy subprocess replaced by a recorder. Yields (module, calls)."""

    class MemoryProvider:
        pass

    def tool_error(message: str) -> str:
        return json.dumps({"error": message})

    agent = types.ModuleType("agent")
    memory_provider = types.ModuleType("agent.memory_provider")
    memory_provider.MemoryProvider = MemoryProvider
    tools = types.ModuleType("tools")
    registry = types.ModuleType("tools.registry")
    registry.tool_error = tool_error
    for name, module in {
        "agent": agent,
        "agent.memory_provider": memory_provider,
        "tools": tools,
        "tools.registry": registry,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    install_for_hermes(hermes_home=tmp_path)
    init_path = tmp_path / "plugins" / "poppy" / "__init__.py"
    spec = importlib.util.spec_from_file_location("hermes_poppy_plugin", init_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls: list[list[str]] = []

    def fake_run_poppy(args, timeout=0):
        calls.append(list(args))
        return {"success": True, "output": "ok"}

    monkeypatch.setattr(module, "_run_poppy", fake_run_poppy)
    return module, calls


def test_plugin_recall_tool_runs_poppy_recall(plugin) -> None:
    module, calls = plugin
    provider = module.PoppyMemoryProvider()
    provider.handle_tool_call("poppy_recall", {"query": "auth flow", "project": "web", "limit": 3})
    assert calls == [["recall", "auth flow", "--json", "--project", "web", "--limit", "3"]]


def test_plugin_remember_tool_runs_poppy_remember(plugin) -> None:
    module, calls = plugin
    provider = module.PoppyMemoryProvider()
    result = provider.handle_tool_call(
        "poppy_remember", {"content": "use uv", "memory_type": "decision", "project": "web"}
    )
    assert calls == [["remember", "use uv", "--type", "decision", "--project", "web"]]
    assert json.loads(result) == {"result": "ok"}


def test_plugin_forget_tool_runs_poppy_forget(plugin) -> None:
    module, calls = plugin
    provider = module.PoppyMemoryProvider()
    provider.handle_tool_call("poppy_forget", {"memory_id": "abc123"})
    assert calls == [["forget", "abc123", "--yes"]]


def test_plugin_unknown_tool_returns_error_without_running_poppy(plugin) -> None:
    module, calls = plugin
    provider = module.PoppyMemoryProvider()
    result = provider.handle_tool_call("poppy_nope", {})
    assert json.loads(result) == {"error": "Unknown tool: poppy_nope"}
    assert calls == []


def _manifest_hooks(manifest: str) -> list[str]:
    """Entries of the top-level ``hooks:`` list in a plugin.yaml body."""
    hooks, in_hooks = [], False
    for line in manifest.splitlines():
        if not line.startswith((" ", "-")):
            in_hooks = line.startswith("hooks:")
        elif in_hooks and line.strip().startswith("- "):
            hooks.append(line.strip()[2:].strip())
    return hooks


def test_manifest_hooks_parser_reads_the_hooks_list() -> None:
    manifest = "name: poppy\nhooks:\n  - on_session_end\n  - sync_turn\nprovides:\n  - memory\n"
    assert _manifest_hooks(manifest) == ["on_session_end", "sync_turn"]


def test_plugin_manifest_hooks_are_implemented_by_the_provider(plugin, tmp_path: Path) -> None:
    module, _ = plugin
    manifest = (tmp_path / "plugins" / "poppy" / "plugin.yaml").read_text()
    for hook in _manifest_hooks(manifest):
        assert hook in vars(module.PoppyMemoryProvider), hook


def _plugin_subcommands(source: str) -> list[str]:
    """The first argument of every ``_run_poppy`` call in the plugin: either an
    inline list or a local variable assigned a list in the same function."""
    subcommands = []
    for func in ast.walk(ast.parse(source)):
        if not isinstance(func, ast.FunctionDef):
            continue
        lists = {
            node.targets[0].id: node.value
            for node in ast.walk(func)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.List)
        }
        for node in ast.walk(func):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_run_poppy":
                argv = node.args[0]
                if isinstance(argv, ast.Name):
                    argv = lists[argv.id]
                assert isinstance(argv, ast.List), ast.unparse(node)
                subcommands.append(ast.literal_eval(argv.elts[0]))
    return subcommands


def test_plugin_only_runs_registered_poppy_subcommands(tmp_path: Path) -> None:
    install_for_hermes(hermes_home=tmp_path)
    source = (tmp_path / "plugins" / "poppy" / "__init__.py").read_text()
    subcommands = _plugin_subcommands(source)
    assert {"recall", "remember", "forget", "stats"} <= set(subcommands)
    assert set(subcommands) <= set(cli.commands)


def test_setup_command_promises_only_what_the_plugin_does(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import poppy.cli.main as cli_main
    import poppy.setup.hermes as hermes_setup

    real_install = hermes_setup.install_for_hermes
    monkeypatch.setattr(hermes_setup, "install_for_hermes", lambda: real_install(hermes_home=tmp_path))
    monkeypatch.setattr(cli_main, "_record_agent_setup", lambda client: None)
    result = CliRunner().invoke(cli, ["setup", "hermes-agent"])
    assert result.exit_code == 0, result.output
    assert "recall relevant memories before each turn" in result.output
    assert "consolidate" not in result.output
