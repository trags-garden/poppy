"""Install Poppy as a Hermes Agent memory provider plugin.

Hermes (Nous Research, github.com/NousResearch/hermes-agent) discovers
third-party memory providers under ``$HERMES_HOME/plugins/<name>/`` (default
``~/.hermes/plugins/<name>/``). Each plugin is a directory with:

  plugin.yaml   — metadata (name, version, description, hooks)
  __init__.py   — implements MemoryProvider ABC + ``register(ctx)``
  README.md     — user-facing docs (optional)

Activation lives in ``~/.hermes/config.yaml`` under ``memory.provider``.
Only ONE external provider is active at a time; the built-in
``MEMORY.md``/``USER.md`` writes stay active alongside it.

The plugin we install shells out to the ``poppy`` CLI for recall/remember
so a hermes session shares ``~/.poppy/memories.db`` with Claude Code.
The shape follows Hermes's third-party memory-provider plugin convention.
"""

from __future__ import annotations

import os
import re
from importlib import resources
from pathlib import Path

HERMES_PLUGIN_NAME = "poppy"

_POPPY_BEGIN = "<!-- POPPY:BEGIN -->"
_POPPY_END = "<!-- POPPY:END -->"
HERMES_SOUL_BODY = """\
Poppy memory is active.
- `poppy_recall`: use before acting in unfamiliar areas.
- `poppy_remember`: store non-obvious decisions, preferences, and lessons.
- `poppy_forget`: remove stale entries.
- `poppy_status`: check memory state."""


def get_hermes_home() -> Path:
    """Mirror ``hermes_constants.get_hermes_home()`` — env var, then ``~/.hermes``."""
    val = os.environ.get("HERMES_HOME", "").strip()
    if val:
        return Path(val).expanduser()
    return Path.home() / ".hermes"


# ---------------------------------------------------------------------------
# Plugin files
# ---------------------------------------------------------------------------

_PLUGIN_YAML = """\
name: poppy
version: 1.0.0
description: "Poppy: local-first developer memory shared with Claude Code via the poppy CLI."
external_dependencies:
  - name: poppy
    install: "pipx install poppy-memory"
    check: "poppy --help"
hooks:
  - on_pre_compress
  - on_session_end
"""


# The plugin __init__.py shells out to the poppy CLI installed on PATH. This
# avoids tying the plugin to a particular poppy Python install — hermes runs
# on its own python, poppy on its own python, and the subprocess is the
# stable contract between them. It ships as ``hermes_plugin.py.txt`` so Poppy
# never imports it: it imports Hermes' own ``agent`` and ``tools`` packages.
_PLUGIN_INIT_RESOURCE = "hermes_plugin.py.txt"


def _plugin_init_bytes() -> bytes:
    return resources.files("poppy.setup").joinpath(_PLUGIN_INIT_RESOURCE).read_bytes()


_PLUGIN_README = """\
# Poppy Memory Provider for Hermes

Local-first developer memory shared with Claude Code, Cursor, Codex, Pi,
Copilot CLI and any other tool wired into the same Poppy install.

## Requirements

Install the Poppy CLI:
```bash
pipx install poppy-memory
```

## Setup

```bash
poppy setup hermes-agent     # writes this plugin + flips memory.provider
```

Or manually:
```bash
hermes config set memory.provider poppy
```

## Config

| Env Var | Required | Description |
|---------|----------|-------------|
| `POPPY_DIR` | No | Override the Poppy data directory (default `~/.poppy`) |

## Tools

| Tool | Description |
|------|-------------|
| `poppy_recall` | Search developer memory for relevant context |
| `poppy_remember` | Store a decision / preference / lesson |
| `poppy_forget` | Delete a memory by id |
| `poppy_status` | Engine info + memory count |

Hermes' built-in `MEMORY.md` / `USER.md` writes are mirrored into Poppy via
the `on_memory_write` hook.
"""


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _set_memory_provider(config_text: str, provider: str) -> str:
    """Set ``memory.provider: <provider>`` in a hermes ``config.yaml`` body.

    Hand-rolled to avoid a YAML dependency. Handles three cases:
      1. File has ``memory:`` block with a ``provider:`` line — replace the value.
      2. File has ``memory:`` block but no ``provider:`` — insert below the header.
      3. No ``memory:`` block — append a fresh one.

    Preserves trailing comments on the replaced line and surrounding content.
    """
    # Scan each line once, including indented blank lines, to avoid backtracking.
    insert_at = None
    in_memory = False
    offset = 0
    for line in config_text.split("\n"):
        line_end = offset + len(line)
        if line.startswith("memory:") and line_end < len(config_text):
            in_memory = True
            if insert_at is None:
                insert_at = line_end + 1
        elif line.startswith((" ", "\t")):
            body = line.lstrip(" \t")
            if in_memory and body.startswith("provider:"):
                value = body[len("provider:") :]
                value_start = line_end - len(value.lstrip(" \t"))
                comment_at = line.find("#", value_start - offset)
                if comment_at >= 0:
                    # YAML comments need the whitespace separating them from the value.
                    while comment_at > value_start - offset and line[comment_at - 1].isspace():
                        comment_at -= 1
                trailing = line[comment_at:] if comment_at >= 0 else ""
                if comment_at == value_start - offset:
                    # An empty value leaves no whitespace before the comment.
                    trailing = " " + trailing
                return config_text[:value_start] + provider + trailing + config_text[line_end:]
        else:
            in_memory = False
        offset = line_end + 1

    # Case 2: `memory:` block exists but lacks `provider:`.
    if insert_at is not None:
        new_line = f"  provider: {provider}\n"
        return config_text[:insert_at] + new_line + config_text[insert_at:]

    # Case 3: no memory block at all.
    sep = "" if config_text.endswith("\n") or config_text == "" else "\n"
    return config_text + sep + f"memory:\n  provider: {provider}\n"


def _activate_provider(config_path: Path, provider: str = HERMES_PLUGIN_NAME) -> Path:
    """Set ``memory.provider: poppy`` in ``~/.hermes/config.yaml``."""
    existing = config_path.read_text() if config_path.exists() else ""
    new_text = _set_memory_provider(existing, provider)
    if new_text != existing:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(new_text)
    return config_path


# ---------------------------------------------------------------------------
# Top-level installer used by the CLI
# ---------------------------------------------------------------------------


def _install_soul_block(soul_path: Path) -> Path:
    """Insert or replace Hermes-specific Poppy guidance without changing user content."""
    block = f"{_POPPY_BEGIN}\n{HERMES_SOUL_BODY}\n{_POPPY_END}"
    existing = soul_path.read_text() if soul_path.exists() else ""

    if _POPPY_BEGIN in existing and _POPPY_END in existing:
        start = existing.index(_POPPY_BEGIN)
        end = existing.index(_POPPY_END, start) + len(_POPPY_END)
        new_text = existing[:start] + block + existing[end:]
    elif existing:
        separator = "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
        new_text = existing + separator + block + "\n"
    else:
        new_text = block + "\n"

    soul_path.parent.mkdir(parents=True, exist_ok=True)
    soul_path.write_text(new_text)
    return soul_path


def _remove_legacy_agents_block(agents_path: Path) -> None:
    """Remove the obsolete managed primer from HERMES_HOME/AGENTS.md, if present."""
    if not agents_path.exists():
        return
    existing = agents_path.read_text()
    if _POPPY_BEGIN not in existing or _POPPY_END not in existing:
        return

    start = existing.index(_POPPY_BEGIN)
    end = existing.index(_POPPY_END, start) + len(_POPPY_END)
    new_text = existing[:start] + existing[end:]
    if new_text.strip():
        agents_path.write_text(new_text)
    else:
        agents_path.unlink()


def install_for_hermes(hermes_home: Path | None = None) -> dict[str, Path]:
    """Install Poppy as a Hermes memory plugin. Returns a {label: path} map.

    Writes:
      - ``$HERMES_HOME/plugins/poppy/{plugin.yaml,__init__.py,README.md}``
      - ``$HERMES_HOME/config.yaml`` with ``memory.provider: poppy`` set
      - ``$HERMES_HOME/SOUL.md`` with brief, managed Hermes tool guidance

    Also removes the obsolete managed Poppy block from ``$HERMES_HOME/AGENTS.md``.
    """
    home = hermes_home or get_hermes_home()
    plugin_dir = home / "plugins" / HERMES_PLUGIN_NAME
    plugin_dir.mkdir(parents=True, exist_ok=True)

    plugin_yaml = plugin_dir / "plugin.yaml"
    plugin_yaml.write_text(_PLUGIN_YAML)

    plugin_init = plugin_dir / "__init__.py"
    plugin_init.write_bytes(_plugin_init_bytes())

    plugin_readme = plugin_dir / "README.md"
    plugin_readme.write_text(_PLUGIN_README)

    config_path = _activate_provider(home / "config.yaml")
    primer_path = _install_soul_block(home / "SOUL.md")
    _remove_legacy_agents_block(home / "AGENTS.md")

    return {
        "Plugin dir": plugin_dir,
        "plugin.yaml": plugin_yaml,
        "__init__.py": plugin_init,
        "config.yaml": config_path,
        "Memory guidance (SOUL.md)": primer_path,
    }


def is_hermes_provider_configured(hermes_home: Path | None = None) -> bool:
    """Whether ``~/.hermes/config.yaml`` activates the poppy provider — ONE
    component of a Hermes install, independent of whether the plugin files are on
    disk. `poppy setup hermes-agent` writes this AND the plugin AND SOUL.md, so a
    config-only partial install still counts as a Poppy footprint."""
    home = hermes_home or get_hermes_home()
    config_path = home / "config.yaml"
    if not config_path.exists():
        return False
    return bool(re.search(r"(?m)^[ \t]+provider:[ \t]*poppy[ \t#]*$", config_path.read_text()))


def is_hermes_plugin_present(hermes_home: Path | None = None) -> bool:
    """Whether Poppy's hermes plugin files are on disk (the plugin ``__init__.py``
    exists), independent of whether ``config.yaml`` activates the provider. This
    is ONE component of a Hermes install, so a PARTIAL install (plugin written but
    the provider not yet configured) still counts as a Poppy footprint and doctor
    surfaces exactly what's missing."""
    home = hermes_home or get_hermes_home()
    return (home / "plugins" / HERMES_PLUGIN_NAME / "__init__.py").exists()


def is_hermes_installed(hermes_home: Path | None = None) -> bool:
    """Whether Poppy is wired into hermes (plugin dir + active provider)."""
    home = hermes_home or get_hermes_home()
    return is_hermes_plugin_present(home) and is_hermes_provider_configured(home)
