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
The shape mirrors ``plugins/memory/byterover/`` in the hermes repo.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from poppy.setup.claude_code import install_primer_md_block

HERMES_PLUGIN_NAME = "poppy"


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
description: "Poppy — local-first developer memory shared with Claude Code via the poppy CLI."
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
# stable contract between them.
_PLUGIN_INIT = '''\
"""Poppy memory plugin for Hermes Agent.

Shells out to the ``poppy`` CLI for recall/remember/forget so a hermes session
shares ``~/.poppy/memories.db`` with Claude Code and any other client wired to
the same Poppy install.

Config via environment variables (profile-scoped via each profile's .env):
  POPPY_DIR   — override the Poppy data directory (default ~/.poppy)

Working directory: $HERMES_HOME/poppy/ (profile-scoped sentinel only; the
actual data lives at $POPPY_DIR or ~/.poppy/).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_RECALL_TIMEOUT = 10
_REMEMBER_TIMEOUT = 30
_CONSOLIDATE_TIMEOUT = 180

_MIN_QUERY_LEN = 8
_MIN_OUTPUT_LEN = 20

_poppy_path_lock = threading.Lock()
_cached_poppy_path: Optional[str] = None


def _resolve_poppy_path() -> Optional[str]:
    """Find the poppy binary on PATH or well-known install locations."""
    global _cached_poppy_path
    with _poppy_path_lock:
        if _cached_poppy_path is not None:
            return _cached_poppy_path if _cached_poppy_path != "" else None

    found = shutil.which("poppy")
    if not found:
        home = Path.home()
        candidates = [
            home / ".local" / "bin" / "poppy",
            Path("/usr/local/bin/poppy"),
            Path("/opt/homebrew/bin/poppy"),
        ]
        for c in candidates:
            if c.exists():
                found = str(c)
                break

    with _poppy_path_lock:
        _cached_poppy_path = found or ""
    return found


def _run_poppy(args: List[str], timeout: int = _RECALL_TIMEOUT) -> dict:
    """Run a poppy CLI command. Returns {success, output, error}."""
    poppy_path = _resolve_poppy_path()
    if not poppy_path:
        return {"success": False, "error": "poppy CLI not found. Install: pipx install poppy-memory"}

    cmd = [poppy_path] + args
    env = os.environ.copy()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, env=env,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.returncode == 0:
            return {"success": True, "output": stdout}
        return {"success": False, "error": stderr or stdout or f"poppy exited {result.returncode}"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"poppy timed out after {timeout}s"}
    except FileNotFoundError:
        global _cached_poppy_path
        with _poppy_path_lock:
            _cached_poppy_path = None
        return {"success": False, "error": "poppy CLI not found"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Tool schemas exposed to the agent
# ---------------------------------------------------------------------------

RECALL_SCHEMA = {
    "name": "poppy_recall",
    "description": (
        "Search Poppy's developer memory for relevant context — past decisions, "
        "preferences, lessons learned, project facts. Use BEFORE suggesting an "
        "approach in an unfamiliar area, or when the user references something "
        "from a past session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "project": {"type": "string", "description": "Optional project scope."},
            "limit": {"type": "integer", "description": "Max results (default 10).", "default": 10},
        },
        "required": ["query"],
    },
}

REMEMBER_SCHEMA = {
    "name": "poppy_remember",
    "description": (
        "Store a single memory in Poppy. Use for: user preferences ('we always "
        "use X'), non-obvious decisions (architecture choices, library picks), "
        "or lessons learned (something failed and we now know why). One atomic "
        "fact per call."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to remember."},
            "memory_type": {
                "type": "string",
                "enum": ["fact", "decision", "preference", "lesson"],
                "default": "fact",
            },
            "project": {"type": "string", "description": "Optional project scope."},
        },
        "required": ["content"],
    },
}

FORGET_SCHEMA = {
    "name": "poppy_forget",
    "description": "Delete a Poppy memory by id (from a prior poppy_recall result).",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory id to delete."},
        },
        "required": ["memory_id"],
    },
}

STATUS_SCHEMA = {
    "name": "poppy_status",
    "description": "Check Poppy status — memory count, active engine, install location.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class PoppyMemoryProvider(MemoryProvider):
    """Poppy local-first developer memory via the poppy CLI."""

    def __init__(self):
        self._session_id = ""
        self._sync_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "poppy"

    def is_available(self) -> bool:
        """Check if poppy CLI is installed. No network calls."""
        return _resolve_poppy_path() is not None

    def get_config_schema(self):
        return [
            {
                "key": "poppy_dir",
                "description": "Poppy data directory (default ~/.poppy)",
                "secret": False,
                "env_var": "POPPY_DIR",
            },
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id

    def system_prompt_block(self) -> str:
        if not _resolve_poppy_path():
            return ""
        return (
            "# Poppy Memory\\n"
            "Active. Local-first developer memory shared with Claude Code.\\n"
            "Use poppy_recall BEFORE acting in unfamiliar areas, poppy_remember "
            "for non-obvious decisions/preferences/lessons, poppy_forget to "
            "delete stale entries, poppy_status to check state."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Surface relevant memories before the model is called."""
        if not query or len(query.strip()) < _MIN_QUERY_LEN:
            return ""
        result = _run_poppy(
            ["recall", query.strip()[:1000], "--limit", "5"],
            timeout=_RECALL_TIMEOUT,
        )
        if result["success"] and result.get("output"):
            output = result["output"].strip()
            if len(output) > _MIN_OUTPUT_LEN and "No memories found" not in output:
                return f"## Poppy Memory\\n{output}"
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """No-op: prefetch runs synchronously at turn start."""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Turn-level sync is a no-op for Poppy — consolidation happens at session end."""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Run poppy consolidate over the session transcript."""
        if not messages:
            return

        def _consolidate():
            try:
                parts = []
                for msg in messages[-50:]:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip() and role in ("user", "assistant"):
                        parts.append(f"{role}: {content[:1500]}")
                if not parts:
                    return
                transcript = "\\n\\n".join(parts)
                _run_poppy(
                    ["consolidate", "--source", "hermes-agent", "--text", transcript[:50000]],
                    timeout=_CONSOLIDATE_TIMEOUT,
                )
            except Exception as e:
                logger.debug("Poppy consolidation failed: %s", e)

        t = threading.Thread(target=_consolidate, daemon=True, name="poppy-consolidate")
        t.start()

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract insights before context compression discards turns."""
        if not messages:
            return ""
        self.on_session_end(messages)
        return ""

    def on_memory_write(self, action: str, target: str, content: str, metadata=None) -> None:
        """Mirror hermes' built-in MEMORY.md / USER.md writes into Poppy."""
        if action not in ("add", "replace") or not content:
            return

        def _write():
            try:
                memory_type = "preference" if target == "user" else "fact"
                _run_poppy(
                    ["remember", content[:5000], "--type", memory_type],
                    timeout=_REMEMBER_TIMEOUT,
                )
            except Exception as e:
                logger.debug("Poppy memory mirror failed: %s", e)

        t = threading.Thread(target=_write, daemon=True, name="poppy-memwrite")
        t.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RECALL_SCHEMA, REMEMBER_SCHEMA, FORGET_SCHEMA, STATUS_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name == "poppy_recall":
            return self._tool_recall(args)
        if tool_name == "poppy_remember":
            return self._tool_remember(args)
        if tool_name == "poppy_forget":
            return self._tool_forget(args)
        if tool_name == "poppy_status":
            return self._tool_status()
        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)

    # -- Tool implementations -----------------------------------------------

    def _tool_recall(self, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return tool_error("query is required")
        cmd = ["recall", query.strip()[:2000], "--json"]
        if args.get("project"):
            cmd.extend(["--project", str(args["project"])])
        limit = args.get("limit") or 10
        cmd.extend(["--limit", str(int(limit))])

        result = _run_poppy(cmd, timeout=_RECALL_TIMEOUT)
        if not result["success"]:
            return tool_error(result.get("error", "Recall failed"))

        output = result.get("output", "").strip()
        if not output or "No memories found" in output:
            return json.dumps({"result": "No relevant memories found."})

        try:
            parsed = json.loads(output)
            return json.dumps({"result": parsed})
        except json.JSONDecodeError:
            return json.dumps({"result": output[:8000]})

    def _tool_remember(self, args: dict) -> str:
        content = args.get("content", "")
        if not content:
            return tool_error("content is required")
        memory_type = args.get("memory_type", "fact")
        if memory_type not in ("fact", "decision", "preference", "lesson"):
            memory_type = "fact"
        cmd = ["remember", content[:5000], "--type", memory_type]
        if args.get("project"):
            cmd.extend(["--project", str(args["project"])])

        result = _run_poppy(cmd, timeout=_REMEMBER_TIMEOUT)
        if not result["success"]:
            return tool_error(result.get("error", "Remember failed"))
        return json.dumps({"result": result.get("output", "Memory stored.").strip()})

    def _tool_forget(self, args: dict) -> str:
        memory_id = args.get("memory_id", "")
        if not memory_id:
            return tool_error("memory_id is required")
        result = _run_poppy(["forget", memory_id, "--yes"], timeout=_REMEMBER_TIMEOUT)
        if not result["success"]:
            return tool_error(result.get("error", "Forget failed"))
        return json.dumps({"result": result.get("output", "Forgotten.").strip()})

    def _tool_status(self) -> str:
        result = _run_poppy(["stats"], timeout=15)
        if not result["success"]:
            return tool_error(result.get("error", "Status check failed"))
        return json.dumps({"status": result.get("output", "").strip()})


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register Poppy as a memory provider plugin."""
    ctx.register_memory_provider(PoppyMemoryProvider())
'''


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
the `on_memory_write` hook. Session-end transcripts trigger
`poppy consolidate` for automatic fact extraction.
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
    # Case 1: existing `provider:` under a `memory:` block.
    pattern = re.compile(r"(?ms)^(memory:[^\n]*\n(?:[ \t]+[^\n]*\n)*?[ \t]+provider:[ \t]*)([^\n#]*)([^\n]*)$")
    match = pattern.search(config_text)
    if match:
        before, _old_value, trailing = match.group(1), match.group(2), match.group(3)
        return config_text[: match.start()] + before + provider + trailing + config_text[match.end() :]

    # Case 2: `memory:` block exists but lacks `provider:`.
    header = re.search(r"(?m)^memory:[^\n]*\n", config_text)
    if header:
        insert_at = header.end()
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


def install_for_hermes(hermes_home: Path | None = None) -> dict[str, Path]:
    """Install Poppy as a Hermes memory plugin. Returns a {label: path} map.

    Writes:
      - ``$HERMES_HOME/plugins/poppy/{plugin.yaml,__init__.py,README.md}``
      - ``$HERMES_HOME/config.yaml`` with ``memory.provider: poppy`` set
      - ``$HERMES_HOME/AGENTS.md`` with the managed POPPY:BEGIN/END primer block
    """
    home = hermes_home or get_hermes_home()
    plugin_dir = home / "plugins" / HERMES_PLUGIN_NAME
    plugin_dir.mkdir(parents=True, exist_ok=True)

    plugin_yaml = plugin_dir / "plugin.yaml"
    plugin_yaml.write_text(_PLUGIN_YAML)

    plugin_init = plugin_dir / "__init__.py"
    plugin_init.write_text(_PLUGIN_INIT)

    plugin_readme = plugin_dir / "README.md"
    plugin_readme.write_text(_PLUGIN_README)

    config_path = _activate_provider(home / "config.yaml")
    primer_path = install_primer_md_block(home / "AGENTS.md")

    return {
        "Plugin dir": plugin_dir,
        "plugin.yaml": plugin_yaml,
        "__init__.py": plugin_init,
        "config.yaml": config_path,
        "Primer (AGENTS.md)": primer_path,
    }


def is_hermes_installed(hermes_home: Path | None = None) -> bool:
    """Whether Poppy is wired into hermes (plugin dir + active provider)."""
    home = hermes_home or get_hermes_home()
    plugin_init = home / "plugins" / HERMES_PLUGIN_NAME / "__init__.py"
    if not plugin_init.exists():
        return False
    config_path = home / "config.yaml"
    if not config_path.exists():
        return False
    text = config_path.read_text()
    return bool(re.search(r"(?m)^[ \t]+provider:[ \t]*poppy[ \t#]*$", text))
