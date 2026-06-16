"""Install Poppy into Claude Code (and other MCP clients).

Reference patterns surveyed:
  - omega-memory: --client flag, fast-hook + daemon, managed CLAUDE.md block
  - claude-mem: SessionStart/UserPromptSubmit/PostToolUse/Stop/SessionEnd hooks
  - byterover: MCP / hook / skill modes

Poppy MVP wires:
  - MCP server registration (all clients)
  - SessionStart hook (claude-code) — surfaces project memories
  - Stop hook (claude-code) — silent placeholder, ready for future LLM consolidation
  - Managed CLAUDE.md block (claude-code) — tool discovery for the agent
"""

import json
import os
from pathlib import Path

CLAUDE_MD_BEGIN = "<!-- POPPY:BEGIN -->"
CLAUDE_MD_END = "<!-- POPPY:END -->"

CLAUDE_MD_BODY = """## Poppy memory

You have access to Poppy, a developer memory store. Use it to remember non-obvious
decisions, preferences, and lessons across sessions.

**When to call `remember`:**
- The user states a preference ("we always use X", "don't do Y").
- A non-obvious decision is made (architecture choice, library pick, tradeoff resolved).
- A lesson is learned (something failed and we now know why).

**When to call `recall_index` then `recall_full`:**
- Before suggesting an approach in an unfamiliar area — check if there's a prior decision.
- When the user references something from a past session.
- Use `recall_index` first (cheap, IDs + snippets), then `recall_full` only on the IDs
  that look relevant. Don't call `recall_full` on every result.

**Tool surface:**
- `remember(content, memory_type, project)` — store a single memory.
- `recall_index(query, project, limit)` — IDs + snippets only.
- `recall_full(ids)` — fetch full content for a batch of IDs.
- `recall(query, project, limit)` — convenience: index + full in one call (use sparingly).
- `consolidate(session_summary, facts, project)` — store learnings at session end.
- `context(project, limit)` — most recent memories for a project.
- `forget(id)` — delete a memory.

`memory_type` is one of: `fact`, `decision`, `preference`, `lesson`, `summary`."""


# Backfill prompt — paste into a Claude conversation that has the poppy MCP
# server active. Claude reads its own stored memories and writes each entry
# back through Poppy's remember() tool, deduping via recall_index first.
CLAUDE_IMPORT_PROMPT = """\
Export all of my stored memories and any context you've learned about me from past conversations, AND ingest each entry into Poppy as you go. Preserve my words verbatim where possible, especially for instructions and preferences.

## Categories (output in this order):
1. **Instructions**: Rules I've explicitly asked you to follow going forward — tone, format, style, "always do X", "never do Y", and corrections to your behavior. Only include rules from stored memories, not from conversations.
2. **Identity**: Name, age, location, education, family, relationships, languages, and personal interests.
3. **Career**: Current and past roles, companies, and general skill areas.
4. **Projects**: Projects I meaningfully built or committed to. Ideally ONE entry per project. Include what it does, current status, and any key decisions. Use the project name or a short descriptor as the first words of the entry.
5. **Preferences**: Opinions, tastes, and working-style preferences that apply broadly.

## Poppy ingestion (do this for every entry):
For each entry you produce, call `remember(content, memory_type, project)` with:
- **content**: the verbatim entry text (without the date prefix — store the date inside the content if relevant, e.g. "As of 2026-03-14, ...").
- **memory_type**: map by category:
  - Instructions → `preference`
  - Identity → `fact`
  - Career → `fact`
  - Projects → `decision` if it captures a build/architecture choice; otherwise `fact`. If a project entry contains a clear lesson learned, store it separately as `lesson`.
  - Preferences → `preference`
- **project**: scope to a project slug when the entry is project-specific. Use no project (global) for Identity, broad Preferences, and cross-cutting Instructions.

Before storing, call `recall_index` with a short query derived from the entry to check for duplicates. If a near-duplicate exists, skip the `remember` call and note "[skipped: duplicate of <id>]" next to that line in the export. Do NOT call `recall_full` unless the snippet is ambiguous.

If a single export entry naturally splits into multiple atomic memories (e.g. a project entry containing both a decision and a lesson), store them as separate `remember` calls and list each on its own line in the export.

## Format:
Use section headers for each category. Within each category, list one entry per line, sorted by oldest date first. Format each line as:
[YYYY-MM-DD] - Entry content here. → stored as <memory_type> in <project|global> [id: <returned-id>]
If no date is known, use [unknown] instead.
If skipped as duplicate: [YYYY-MM-DD] - Entry content here. → [skipped: duplicate of <id>]

## Output:
- Wrap the entire export in a single code block for easy copying.
- After the code block, report:
  - Total entries processed, broken down by category and memory_type.
  - Number of duplicates skipped.
  - Whether this is the complete set or if more remain.
  - Any entries you were uncertain how to categorize or scope, so I can review."""


def get_claude_config_dir() -> Path:
    env_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".claude"


def get_poppy_executable() -> str:
    return "poppy"


# ---------------------------------------------------------------------------
# Client config locations
# ---------------------------------------------------------------------------


def get_claude_desktop_config_path() -> Path:
    """Resolve the Claude desktop app's `claude_desktop_config.json`.

    Honors `POPPY_CLAUDE_DESKTOP_CONFIG` for tests and unusual setups. Falls
    back to the platform default — macOS Application Support, Windows %APPDATA%.
    Linux has no first-party desktop app, but we mirror the macOS layout so a
    user with a custom install can point the env var at it.
    """
    override = os.environ.get("POPPY_CLAUDE_DESKTOP_CONFIG")
    if override:
        return Path(override)
    home = Path.home()
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else home / "AppData" / "Roaming"
        return base / "Claude" / "claude_desktop_config.json"
    return home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"


def _client_settings_path(client: str, claude_dir: Path) -> Path:
    """Where the client stores its MCP server registration."""
    home = Path.home()
    if client == "claude-code":
        # Claude Code reads MCP servers from ~/.claude.json (top-level sibling
        # of the ~/.claude directory), NOT from ~/.claude/settings.json. The
        # settings.json file holds hooks and permissions only.
        # Resolve to a sibling so tests passing a tmp_path stay isolated.
        if claude_dir.name == ".claude":
            return claude_dir.parent / ".claude.json"
        return claude_dir / ".claude.json"
    if client == "claude-desktop":
        return get_claude_desktop_config_path()
    if client == "cursor":
        return home / ".cursor" / "mcp.json"
    if client == "windsurf":
        return home / ".codeium" / "windsurf" / "mcp_config.json"
    if client == "codex":
        return home / ".codex" / "config.json"
    if client == "copilot-cli":
        # GitHub Copilot CLI reads MCP servers from ~/.copilot/mcp-config.json.
        return home / ".copilot" / "mcp-config.json"
    if client == "pi":
        # Pi reads MCP servers via the pi-mcp-adapter extension. Precedence per
        # the adapter docs: ~/.config/mcp/mcp.json → ~/.pi/agent/mcp.json →
        # ./.mcp.json → ./.pi/mcp.json. We write to the Pi-global location so
        # the registration travels with the user, not the project.
        return home / ".pi" / "agent" / "mcp.json"
    raise ValueError(f"Unknown client: {client}")


def _read_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# MCP registration
# ---------------------------------------------------------------------------


def _mcp_entry(source: str = "mcp") -> dict:
    # Tag memories from this server with the client name (e.g. "claude-code")
    # rather than the generic "mcp", so the source is meaningful when browsing.
    return {
        "command": get_poppy_executable(),
        "args": ["serve", "--source", source],
        "type": "stdio",
    }


CLAUDE_DESKTOP_BACKUP_SUFFIX = ".pre-poppy.bak"


def _backup_once(path: Path, suffix: str) -> Path | None:
    """Copy `path` to `path + suffix` if the source exists and the backup doesn't.

    Returns the backup path when written, else None. Idempotent — re-running
    setup never clobbers a previous backup.
    """
    if not path.exists():
        return None
    backup = path.with_name(path.name + suffix)
    if backup.exists():
        return None
    backup.write_bytes(path.read_bytes())
    return backup


def install_mcp_config(claude_config_dir: Path | None = None, client: str = "claude-code") -> Path:
    """Register the Poppy MCP server in the chosen client."""
    claude_dir = claude_config_dir or get_claude_config_dir()
    settings_path = _client_settings_path(client, claude_dir)
    if client == "claude-desktop":
        _backup_once(settings_path, CLAUDE_DESKTOP_BACKUP_SUFFIX)
    settings = _read_json(settings_path)

    settings.setdefault("mcpServers", {})
    settings["mcpServers"]["poppy"] = _mcp_entry(source=client)

    _write_json(settings_path, settings)
    return settings_path


def is_mcp_installed(claude_config_dir: Path | None = None, client: str = "claude-code") -> bool:
    claude_dir = claude_config_dir or get_claude_config_dir()
    settings = _read_json(_client_settings_path(client, claude_dir))
    return "poppy" in settings.get("mcpServers", {})


# ---------------------------------------------------------------------------
# Hooks (claude-code only)
# ---------------------------------------------------------------------------

# Hooks definition: event → (matcher, command). Matcher scopes the hook to specific
# tools (PreToolUse/PostToolUse only); empty for events without per-tool dispatch.
_HOOK_DEFS: dict[str, tuple[str, str]] = {
    "SessionStart": ("", "poppy hook session-start"),
    "UserPromptSubmit": ("", "poppy hook user-prompt-submit"),
    "PreToolUse": ("Edit|Write|MultiEdit", "poppy hook pre-tool-use"),
    "SessionEnd": ("", "poppy hook session-end"),
}

# Legacy hooks we used to install but no longer do (still removed during setup
# so an upgrade tidies the user's settings.json without breaking it).
_LEGACY_HOOK_COMMANDS: dict[str, list[str]] = {
    "Stop": ["poppy hook stop"],
}


def _install_hook(claude_dir: Path, event: str) -> Path:
    settings_path = claude_dir / "settings.json"
    settings = _read_json(settings_path)
    settings.setdefault("hooks", {})
    settings["hooks"].setdefault(event, [])

    matcher, command = _HOOK_DEFS[event]
    for group in settings["hooks"][event]:
        for h in group.get("hooks", []):
            if h.get("command") == command:
                return settings_path

    settings["hooks"][event].append(
        {
            "matcher": matcher,
            "hooks": [{"type": "command", "command": command}],
        }
    )
    _write_json(settings_path, settings)
    return settings_path


def install_session_start_hook(claude_config_dir: Path | None = None) -> Path:
    return _install_hook(claude_config_dir or get_claude_config_dir(), "SessionStart")


def install_user_prompt_submit_hook(claude_config_dir: Path | None = None) -> Path:
    return _install_hook(claude_config_dir or get_claude_config_dir(), "UserPromptSubmit")


def install_pre_tool_use_hook(claude_config_dir: Path | None = None) -> Path:
    return _install_hook(claude_config_dir or get_claude_config_dir(), "PreToolUse")


def install_session_end_hook(claude_config_dir: Path | None = None) -> Path:
    return _install_hook(claude_config_dir or get_claude_config_dir(), "SessionEnd")


def remove_legacy_hooks(claude_config_dir: Path | None = None) -> list[str]:
    """Remove poppy hook entries that are no longer installed by current setup.

    Returns a list of (event, command) descriptors that were removed. Idempotent.
    Honors the user's other hook entries — only Poppy-owned commands are touched.
    """
    claude_dir = claude_config_dir or get_claude_config_dir()
    settings_path = claude_dir / "settings.json"
    settings = _read_json(settings_path)
    hooks = settings.get("hooks", {})
    removed: list[str] = []

    for event, legacy_commands in _LEGACY_HOOK_COMMANDS.items():
        if event not in hooks:
            continue
        new_groups: list[dict] = []
        for group in hooks[event]:
            kept = [h for h in group.get("hooks", []) if h.get("command") not in legacy_commands]
            for h in group.get("hooks", []):
                if h.get("command") in legacy_commands:
                    removed.append(f"{event}:{h.get('command')}")
            if kept:
                new_groups.append({**group, "hooks": kept})
        if new_groups:
            hooks[event] = new_groups
        else:
            hooks.pop(event)

    if removed:
        settings["hooks"] = hooks
        _write_json(settings_path, settings)
    return removed


def is_hook_installed(claude_config_dir: Path | None = None, event: str = "SessionEnd") -> bool:
    claude_dir = claude_config_dir or get_claude_config_dir()
    settings = _read_json(claude_dir / "settings.json")
    if event not in _HOOK_DEFS:
        return False
    _, command = _HOOK_DEFS[event]
    for group in settings.get("hooks", {}).get(event, []):
        for h in group.get("hooks", []):
            if h.get("command") == command:
                return True
    return False


# ---------------------------------------------------------------------------
# Managed CLAUDE.md block
# ---------------------------------------------------------------------------


def install_primer_md_block(md_path: Path) -> Path:
    """Insert (or update) a managed Poppy primer block in an instructions file.

    Used for Claude Code (CLAUDE.md), Copilot CLI (AGENTS.md), and Pi (AGENTS.md).
    The block is bracketed with `POPPY:BEGIN`/`POPPY:END` markers so subsequent
    runs replace it cleanly without disturbing surrounding user content.
    """
    block = f"{CLAUDE_MD_BEGIN}\n{CLAUDE_MD_BODY}\n{CLAUDE_MD_END}"

    if md_path.exists():
        text = md_path.read_text()
        if CLAUDE_MD_BEGIN in text and CLAUDE_MD_END in text:
            start = text.index(CLAUDE_MD_BEGIN)
            end = text.index(CLAUDE_MD_END) + len(CLAUDE_MD_END)
            new_text = text[:start] + block + text[end:]
        else:
            sep = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
            new_text = text + sep + block + "\n"
    else:
        new_text = block + "\n"

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(new_text)
    return md_path


def install_claude_md_block(claude_config_dir: Path | None = None) -> Path:
    """Insert (or update) a managed Poppy block in ~/.claude/CLAUDE.md."""
    claude_dir = claude_config_dir or get_claude_config_dir()
    return install_primer_md_block(claude_dir / "CLAUDE.md")


def managed_primer_present(md_path: Path) -> bool:
    if not md_path.exists():
        return False
    return CLAUDE_MD_BEGIN in md_path.read_text()


def managed_claude_md_present(claude_config_dir: Path | None = None) -> bool:
    claude_dir = claude_config_dir or get_claude_config_dir()
    return managed_primer_present(claude_dir / "CLAUDE.md")


def _client_primer_path(client: str) -> Path | None:
    """Return the global instructions file for a client, or None if no convention applies.

    Conventions used:
      - copilot-cli → ~/.copilot/AGENTS.md (Copilot CLI loads AGENTS.md per its
        --no-custom-instructions flag; the project-scoped equivalent is
        .github/copilot-instructions.md)
      - pi → ~/.pi/AGENTS.md (Pi loads AGENTS.md/CLAUDE.md per its
        --no-context-files flag)
    """
    home = Path.home()
    if client == "copilot-cli":
        return home / ".copilot" / "AGENTS.md"
    if client == "pi":
        return home / ".pi" / "AGENTS.md"
    return None


# ---------------------------------------------------------------------------
# Top-level installer used by the CLI
# ---------------------------------------------------------------------------


def install_for_client(
    *,
    client: str = "claude-code",
    claude_config_dir: Path | None = None,
    install_hooks: bool = True,
    install_claude_md: bool = True,
) -> dict[str, Path]:
    """Install Poppy into the given client. Returns a {label: path} map."""
    paths: dict[str, Path] = {}

    if client == "claude-desktop":
        # Surface the backup path before the merge writes the file, so the
        # user sees what was preserved.
        target = _client_settings_path(client, claude_config_dir or get_claude_config_dir())
        backup = _backup_once(target, CLAUDE_DESKTOP_BACKUP_SUFFIX)
        if backup is not None:
            paths["backup"] = backup

    paths["MCP config"] = install_mcp_config(claude_config_dir, client=client)

    primer_path = _client_primer_path(client)
    if primer_path is not None:
        paths["Primer (AGENTS.md)"] = install_primer_md_block(primer_path)

    if client == "claude-code":
        if install_hooks:
            paths["SessionStart hook"] = install_session_start_hook(claude_config_dir)
            paths["UserPromptSubmit hook"] = install_user_prompt_submit_hook(claude_config_dir)
            paths["PreToolUse hook"] = install_pre_tool_use_hook(claude_config_dir)
            paths["SessionEnd hook"] = install_session_end_hook(claude_config_dir)
            # Migrate older installs: Stop hook used to run consolidation, but it
            # fires after every assistant turn — SessionEnd is the right event.
            remove_legacy_hooks(claude_config_dir)
        if install_claude_md:
            paths["CLAUDE.md block"] = install_claude_md_block(claude_config_dir)

    return paths
