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
import shutil
import stat
import sys
from collections.abc import MutableMapping
from pathlib import Path

import tomlkit
from tomlkit.exceptions import TOMLKitError

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
    found = shutil.which("poppy")
    if found:
        return str(Path(found).resolve())

    sibling = Path(sys.executable).parent / "poppy"
    if sibling.is_file():
        return str(sibling.resolve())

    return "poppy"


def _copilot_home() -> Path:
    override = os.environ.get("COPILOT_HOME")
    return Path(override) if override else Path.home() / ".copilot"


def _pi_agent_dir() -> Path:
    override = os.environ.get("PI_CODING_AGENT_DIR")
    return Path(override) if override else Path.home() / ".pi" / "agent"


def get_codex_home() -> Path:
    """Return Codex's config/data directory, honoring ``CODEX_HOME``."""
    override = os.environ.get("CODEX_HOME")
    return Path(override) if override else Path.home() / ".codex"


def get_cursor_home() -> Path:
    """Return Cursor's global config directory, honoring ``CURSOR_HOME``."""
    override = os.environ.get("CURSOR_HOME")
    return Path(override) if override else Path.home() / ".cursor"


# ---------------------------------------------------------------------------
# Client config locations
# ---------------------------------------------------------------------------


def get_claude_desktop_config_path() -> Path:
    """Resolve the Claude desktop app's `claude_desktop_config.json`.

    Honors `POPPY_CLAUDE_DESKTOP_CONFIG` for tests and unusual setups. Falls
    back to the platform default — macOS Application Support, Windows %APPDATA%,
    Linux XDG config. The first-party Linux (apt) build is an Electron app with
    productName "Claude" and no unconditional userData relocation (verified
    against the shipped claude-desktop 1.22209.0 .deb: userData moves only via
    the app's own `CLAUDE_USER_DATA_DIR` escape hatch or managed enterprise
    setup), so its config directory is `$XDG_CONFIG_HOME/Claude/`, default
    `~/.config/Claude/`.
    """
    override = os.environ.get("POPPY_CLAUDE_DESKTOP_CONFIG")
    if override:
        return Path(override)
    home = Path.home()
    if os.name == "nt":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else home / "AppData" / "Roaming"
        return base / "Claude" / "claude_desktop_config.json"
    if sys.platform == "linux":
        xdg = os.environ.get("XDG_CONFIG_HOME")
        base = Path(xdg) if xdg else home / ".config"
        return base / "Claude" / "claude_desktop_config.json"
    return home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"


def get_claude_desktop_msix_config_path() -> Path | None:
    """Return Claude Desktop's MSIX-virtualized config path when installed."""
    local_appdata = os.environ.get("LOCALAPPDATA")
    if not local_appdata:
        return None

    packages_dir = Path(local_appdata) / "Packages"
    for config_dir in sorted(packages_dir.glob("Claude_*/LocalCache/Roaming/Claude")):
        if config_dir.is_dir():
            return config_dir / "claude_desktop_config.json"
    return None


def get_vscode_mcp_config_path() -> Path:
    """Resolve VS Code's user-profile MCP configuration file."""
    override = os.environ.get("POPPY_VSCODE_MCP_CONFIG")
    if override:
        return Path(override)
    home = Path.home()
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else home / "AppData" / "Roaming"
        return base / "Code" / "User" / "mcp.json"
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Code" / "User" / "mcp.json"
    return home / ".config" / "Code" / "User" / "mcp.json"


def _client_settings_path(client: str, claude_dir: Path) -> Path:
    """Return the client's MCP registration file.

    Client-specific directory overrides are honored, including ``COPILOT_HOME``
    for Copilot CLI and ``PI_CODING_AGENT_DIR`` for Pi.
    """
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
        return get_cursor_home() / "mcp.json"
    if client == "vscode":
        return get_vscode_mcp_config_path()
    if client == "windsurf":
        return home / ".codeium" / "windsurf" / "mcp_config.json"
    if client == "codex":
        return get_codex_home() / "config.toml"
    if client == "copilot-cli":
        return _copilot_home() / "mcp-config.json"
    if client == "pi":
        # Pi reads MCP servers via the pi-mcp-adapter extension. Precedence per
        # the adapter docs: ~/.config/mcp/mcp.json → ~/.pi/agent/mcp.json →
        # ./.mcp.json → ./.pi/mcp.json. We write to the Pi-global location so
        # the registration travels with the user, not the project.
        return _pi_agent_dir() / "mcp.json"
    if client == "gemini":
        return home / ".gemini" / "settings.json"
    raise ValueError(f"Unknown client: {client}")


class CorruptConfigError(Exception):
    """A client config file is unparseable or structurally invalid.

    Raised (in strict mode) instead of silently returning `{}`, so a write path
    that would overwrite the whole file aborts rather than truncating a config
    that was merely malformed (PP-02).
    """


def _read_json(path: Path, *, strict: bool = False) -> dict:
    # NOTE: OSError (an unreadable file) deliberately propagates here. This reader
    # backs WRITE paths (`_install_hook` reads then `_write_json`s the result), so
    # swallowing an unreadable file and returning {} would truncate the user's
    # real config on the next write (data loss). Read-only callers
    # that must tolerate an unreadable config (doctor) catch OSError themselves.
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            if strict:
                raise CorruptConfigError(
                    f"Refusing to overwrite unparseable config at {path}. Fix or remove it, then re-run `poppy setup`."
                ) from exc
            return {}
    return {}


def _write_json(path: Path, data: dict) -> None:
    _write_text(path, json.dumps(data, indent=2))


def _write_text(path: Path, content: str) -> None:
    """Atomically replace a text file using a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve client config permissions; new configs may contain a bearer token.
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    # Atomic write: render to a temp file in the same directory, then os.replace
    # onto the target so a crash mid-write can't leave a half-written config.
    # Same-dir tmp keeps the rename on one filesystem (os.replace requirement).
    tmp = path.with_name(f"{path.name}.poppy-tmp-{os.getpid()}")
    try:
        # Create the temp file and chmod it to the final mode BEFORE writing any
        # content, so a bearer token is never briefly readable at the umask
        # default (e.g. 0644) while the write is in flight (Greptile).
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        try:
            os.fchmod(fd, mode)
        except BaseException:
            os.close(fd)
            raise
        with os.fdopen(fd, "w") as fh:
            fh.write(content)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# MCP registration
# ---------------------------------------------------------------------------


def _mcp_entry(source: str = "mcp", *, daemon: bool = False, port: int = 7679, token: str | None = None) -> dict:
    if daemon:
        if token is None:
            raise ValueError("Daemon MCP registration requires a bearer token.")
        url_key = "httpUrl" if source == "gemini" else "url"
        entry = {
            url_key: f"http://127.0.0.1:{port}/mcp",
            "headers": {"Authorization": f"Bearer {token}"},
        }
        if source not in {"cursor", "gemini"}:
            entry = {"type": "http", **entry}
        return entry
    # Tag memories from this server with the client name (e.g. "claude-code")
    # rather than the generic "mcp", so the source is meaningful when browsing.
    entry = {
        "command": get_poppy_executable(),
        "args": ["serve", "--source", source],
    }
    if source != "gemini":
        entry["type"] = "stdio"
    return entry


# One backup suffix for every client. Each client's config path is distinct, so
# appending the same suffix never collides across clients.
CONFIG_BACKUP_SUFFIX = ".pre-poppy.bak"
# Back-compat alias for callers/tests that reference the old desktop-only name.
CLAUDE_DESKTOP_BACKUP_SUFFIX = CONFIG_BACKUP_SUFFIX


def _backup_once(path: Path, suffix: str) -> Path | None:
    """Copy ``path`` to the first free bounded rotating backup slot.

    Slots are ``suffix``, ``suffix-1``, through ``suffix-9``. Once all slots
    exist, the highest slot is reused so a new malformed file never destroys
    the only recoverable copy. Returns the written path, or ``None`` when the
    source does not exist.
    """
    if not path.exists():
        return None
    mode = stat.S_IMODE(path.stat().st_mode)
    backups = [path.with_name(path.name + suffix)]
    backups.extend(path.with_name(f"{path.name}{suffix}-{index}") for index in range(1, 10))
    backup = next((candidate for candidate in backups if not candidate.exists()), backups[-1])
    data = path.read_bytes()
    # Open (fresh or reused rotation slot) and lock the fd to 0600 BEFORE
    # writing any bytes, then widen to the source's own mode after — a reused
    # slot may still carry a wider mode from an earlier rotation, and backups
    # can contain the same credentials as the original config, so the write
    # itself must never happen at a readable-by-others mode (Greptile).
    fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, data)
        os.fchmod(fd, mode)
    finally:
        os.close(fd)
    return backup


def _read_codex_toml(path: Path, *, strict: bool = False):
    if path.exists():
        try:
            return tomlkit.parse(path.read_text())
        except (OSError, TOMLKitError, UnicodeDecodeError) as exc:
            if strict:
                raise CorruptConfigError(
                    f"Refusing to overwrite unparseable config at {path}. Fix or remove it, then re-run `poppy setup`."
                ) from exc
    return tomlkit.document()


def _install_codex_mcp_config(
    *, daemon: bool = False, daemon_port: int = 7679, daemon_token: str | None = None
) -> tuple[Path, Path | None, Path | None]:
    """Install Codex MCP config and remove Poppy's obsolete JSON registration."""
    config_path = _client_settings_path("codex", get_claude_config_dir())
    legacy_path = config_path.with_suffix(".json")

    # Validate every existing user-owned config before modifying either file.
    settings = _read_codex_toml(config_path, strict=True)
    legacy = _read_json(legacy_path, strict=True)

    mcp_servers = settings.get("mcp_servers")
    if mcp_servers is None:
        mcp_servers = tomlkit.table()
        settings["mcp_servers"] = mcp_servers
    if not isinstance(mcp_servers, MutableMapping):
        raise CorruptConfigError(
            f"Refusing to overwrite invalid mcp_servers config at {config_path}. "
            "Fix or remove it, then re-run `poppy setup`."
        )

    poppy = tomlkit.table()
    if daemon:
        if daemon_token is None:
            raise ValueError("Daemon MCP registration requires a bearer token.")
        poppy["url"] = f"http://127.0.0.1:{daemon_port}/mcp"
        headers = tomlkit.inline_table()
        headers["Authorization"] = f"Bearer {daemon_token}"
        poppy["http_headers"] = headers
    else:
        poppy["command"] = get_poppy_executable()
        poppy["args"] = ["serve", "--source", "codex"]
    mcp_servers["poppy"] = poppy

    backup = _backup_once(config_path, CONFIG_BACKUP_SUFFIX)
    _write_text(config_path, tomlkit.dumps(settings))

    migrated = None
    legacy_servers = legacy.get("mcpServers")
    if isinstance(legacy_servers, dict) and "poppy" in legacy_servers:
        del legacy_servers["poppy"]
        _write_json(legacy_path, legacy)
        migrated = legacy_path

    return config_path, backup, migrated


def _install_json_mcp_config(
    settings_path: Path,
    client: str,
    *,
    daemon: bool = False,
    daemon_port: int = 7679,
    daemon_token: str | None = None,
    backup_existing: bool = True,
) -> Path:
    # Read strict first: abort on a corrupt existing config before we touch it,
    # rather than truncating the user's file to a 3-key stub (PP-02).
    settings = _read_json(settings_path, strict=True)
    # Back up any existing config before the first overwrite, for every client.
    if backup_existing:
        _backup_once(settings_path, CONFIG_BACKUP_SUFFIX)

    servers_key = "servers" if client == "vscode" else "mcpServers"
    settings.setdefault(servers_key, {})
    settings[servers_key]["poppy"] = _mcp_entry(
        source=client,
        daemon=daemon,
        port=daemon_port,
        token=daemon_token,
    )

    _write_json(settings_path, settings)
    return settings_path


def install_mcp_config(
    claude_config_dir: Path | None = None,
    client: str = "claude-code",
    *,
    daemon: bool = False,
    daemon_port: int = 7679,
    daemon_token: str | None = None,
) -> Path:
    """Register the Poppy MCP server in the chosen client."""
    if client == "codex":
        config_path, _, _ = _install_codex_mcp_config(
            daemon=daemon,
            daemon_port=daemon_port,
            daemon_token=daemon_token,
        )
        return config_path

    claude_dir = claude_config_dir or get_claude_config_dir()
    return _install_json_mcp_config(
        _client_settings_path(client, claude_dir),
        client,
        daemon=daemon,
        daemon_port=daemon_port,
        daemon_token=daemon_token,
    )


def is_mcp_installed(claude_config_dir: Path | None = None, client: str = "claude-code") -> bool:
    if client == "codex":
        settings = _read_codex_toml(_client_settings_path(client, get_claude_config_dir()))
        mcp_servers = settings.get("mcp_servers", {})
        return isinstance(mcp_servers, MutableMapping) and "poppy" in mcp_servers

    claude_dir = claude_config_dir or get_claude_config_dir()
    settings = _read_json(_client_settings_path(client, claude_dir))
    servers_key = "servers" if client == "vscode" else "mcpServers"
    # Guard non-object top-level JSON (`[]`, `null`, `"x"`): it parses but has no
    # `.get`, and this backs doctor's read path.
    if not isinstance(settings, MutableMapping):
        return False
    servers = settings.get(servers_key, {})
    return isinstance(servers, MutableMapping) and "poppy" in servers


def _authorization_header_from_mcp_entry(entry: MutableMapping) -> str | None:
    """Return the RAW ``Authorization`` header value baked into a client's Poppy
    MCP entry, VERBATIM — no strip/normalize/reconstruct.

    JSON clients nest it under ``headers``; Codex TOML uses ``http_headers``.
    Doctor sends this exact value when probing, so a malformed header (extra
    whitespace like ``"Bearer  secret"``, wrong case, ...) reproduces the SAME
    401 the client gets rather than being silently corrected into a false OK.
    """
    headers = entry.get("headers")
    if not isinstance(headers, MutableMapping):
        headers = entry.get("http_headers")
    if not isinstance(headers, MutableMapping):
        return None
    # HTTP header names are case-insensitive (RFC 7230), so a client that stored
    # `AUTHORIZATION` / `authorization` authenticates fine — match ANY case of the
    # key, else doctor would drop the header, probe with no credential, and WARN a
    # working client as unauthenticated.
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == "authorization":
            return value if isinstance(value, str) else None
    return None


def poppy_daemon_registration(
    client: str = "claude-code", claude_config_dir: Path | None = None
) -> tuple[str, str | None] | None:
    """Return ``(url, authorization_header)`` when Poppy is registered for
    ``client`` via an HTTP(S) daemon URL, else ``None`` (absent, or a local stdio
    command). ``authorization_header`` is that client's RAW ``Authorization``
    header value (verbatim) or ``None`` when the entry has no auth header.

    Daemon-mode setup (`poppy setup <client> --daemon`) points the client at the
    shared authenticated HTTP daemon instead of spawning `poppy serve`. Doctor
    probes each client with its own EXACT header, so it catches a client whose
    stored credential the daemon 401s (stale, or malformed like ``"Bearer
    secret"`` with a double space) — which the current ``daemon.token`` would
    mask — plus a dead/unreachable daemon — otherwise silent dead MCP servers.

    This returns ANY http(s) URL registration; the CALLER decides locality by
    parsing the host exactly. It must NOT substring-match "127.0.0.1"/"localhost"
    here — ``http://127.0.0.1.evil.com/mcp`` or ``https://localhost.attacker.tld/mcp``
    would pass a substring test and be trusted as local while the client ships
    its bearer token off-box (security).
    """
    claude_dir = claude_config_dir or get_claude_config_dir()
    if client == "claude-desktop-msix":
        msix_path = get_claude_desktop_msix_config_path()
        settings = _read_json(msix_path) if msix_path is not None else {}
        servers_key = "mcpServers"
    elif client == "claude-desktop":
        settings = _read_json(get_claude_desktop_config_path())
        servers_key = "mcpServers"
    elif client == "codex":
        settings = _read_codex_toml(_client_settings_path(client, claude_dir))
        servers_key = "mcp_servers"
    else:
        settings = _read_json(_client_settings_path(client, claude_dir))
        servers_key = "servers" if client == "vscode" else "mcpServers"
    # A config whose top-level JSON is not an object (`[]`, `null`, `"x"`) parses
    # fine but has no `.get`; treat it as "no registration" instead of crashing
    # doctor.
    servers = settings.get(servers_key, {}) if isinstance(settings, MutableMapping) else {}
    if not isinstance(servers, MutableMapping):
        return None
    poppy = servers.get("poppy")
    if not isinstance(poppy, MutableMapping):
        return None
    # An entry with a `url` (or `httpUrl`) key IS a daemon-mode registration — the
    # user pointed the client at an HTTP daemon — even if that URL is malformed.
    # Return it RAW (no `http://` prefix requirement) so the caller validates it
    # and WARNs; collapsing a broken URL to None would make doctor treat the entry
    # as a non-daemon client and print a bare OK. Only an entry
    # with a `command` and no url is the stdio (non-daemon) case.
    url = poppy.get("url")
    if url is None:
        url = poppy.get("httpUrl")
    if isinstance(url, str):
        return url.strip(), _authorization_header_from_mcp_entry(poppy)
    return None


def poppy_daemon_url(client: str = "claude-code", claude_config_dir: Path | None = None) -> str | None:
    """Loopback daemon URL Poppy is registered under for ``client``, else ``None``."""
    registration = poppy_daemon_registration(client, claude_config_dir)
    return registration[0] if registration else None


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

# Hooks definition: event → (matcher, command). Matcher scopes the hook to specific
# tools (PreToolUse/PostToolUse only); empty for events without per-tool dispatch.
_HOOK_DEFS: dict[str, tuple[str, str]] = {
    "SessionStart": ("", "poppy hook session-start"),
    "UserPromptSubmit": ("", "poppy hook user-prompt-submit"),
    "PreToolUse": ("Edit|Write|NotebookEdit", "poppy hook pre-tool-use"),
    "SessionEnd": ("", "poppy hook session-end"),
    "PostCompact": ("", "poppy hook post-compact"),
}

# Legacy hooks we used to install but no longer do (still removed during setup
# so an upgrade tidies the user's settings.json without breaking it).
_LEGACY_HOOK_COMMANDS: dict[str, list[str]] = {
    "Stop": ["poppy hook stop"],
}

# Codex 0.144.1 loads the same JSON hook schema as Claude Code, but only this
# lifecycle subset fires. Stop is per turn, so it is a liveness stamp rather
# than a session-end flush. The edit tool is normalized to ``apply_patch`` by
# the hook layer, even when the transcript records a generic code-mode call.
_CODEX_HOOK_DEFS: dict[str, tuple[str, str]] = {
    "SessionStart": ("", "poppy hook session-start"),
    "UserPromptSubmit": ("", "poppy hook user-prompt-submit"),
    "PreToolUse": ("apply_patch", "poppy hook pre-tool-use"),
    "Stop": ("", "poppy hook stop"),
}

# Cursor rejects the entire hooks file when any event key is not recognized.
# Keep this frozen to the documented event names. The
# installer only creates keys from this set; doctor warns about unknown keys in
# user-owned content that the merge deliberately preserves.
CURSOR_DOCUMENTED_HOOK_EVENTS = frozenset(
    {
        "sessionStart",
        "sessionEnd",
        "beforeSubmitPrompt",
        "preToolUse",
        "postToolUse",
        "postToolUseFailure",
        "subagentStart",
        "subagentStop",
        "beforeShellExecution",
        "afterShellExecution",
        "beforeMCPExecution",
        "afterMCPExecution",
        "beforeReadFile",
        "afterFileEdit",
        "preCompact",
        "stop",
        "afterAgentResponse",
        "afterAgentThought",
        "workspaceOpen",
    }
)

# Native Cursor schema: event -> a flat list of command entries. preToolUse is
# included because Poppy uses its Claude Code equivalent for file-specific
# recall before edits. No post-tool event participates in recall or capture.
_CURSOR_HOOK_DEFS: dict[str, str] = {
    "sessionStart": "poppy hook session-start",
    "beforeSubmitPrompt": "poppy hook user-prompt-submit",
    "preToolUse": "poppy hook pre-tool-use",
    "sessionEnd": "poppy hook session-end",
    "preCompact": "poppy hook post-compact",
}

assert _CURSOR_HOOK_DEFS.keys() <= CURSOR_DOCUMENTED_HOOK_EVENTS

_ALL_POPPY_HOOK_COMMANDS = {command for _matcher, command in (*_HOOK_DEFS.values(), *_CODEX_HOOK_DEFS.values())}


def _invalid_hooks_config(path: Path, reason: str) -> CorruptConfigError:
    return CorruptConfigError(
        f"Refusing to overwrite invalid hooks config at {path}: {reason}. Fix or remove it, then re-run `poppy setup`."
    )


def _validate_hooks_object(settings: object, path: Path, *, allow_null: bool = False) -> dict:
    """Check the root and hooks mapping shared by the hook writers."""
    if not isinstance(settings, dict):
        raise _invalid_hooks_config(path, "config root must be an object")
    hooks = settings.get("hooks", {})
    if hooks is None and allow_null:
        return {}
    if not isinstance(hooks, dict):
        raise _invalid_hooks_config(path, '"hooks" must be an object')
    return hooks


def _validate_claude_hooks_config(settings: object, path: Path) -> None:
    hooks = _validate_hooks_object(settings, path)
    for event, groups in hooks.items():
        event_key = f"hooks.{event}"
        if not isinstance(groups, list):
            raise _invalid_hooks_config(path, f'"{event_key}" must be a list')
        for index, group in enumerate(groups):
            group_key = f"{event_key}[{index}]"
            if not isinstance(group, dict):
                raise _invalid_hooks_config(path, f'"{group_key}" must be an object')
            commands = group.get("hooks", [])
            if not isinstance(commands, list):
                raise _invalid_hooks_config(path, f'"{group_key}.hooks" must be a list')
            for command_index, command in enumerate(commands):
                if not isinstance(command, dict):
                    raise _invalid_hooks_config(path, f'"{group_key}.hooks[{command_index}]" must be an object')


def _validate_cursor_hooks_config(settings: object, path: Path) -> str | None:
    """Reject refused shapes; return a reason for Cursor's existing reset cases."""
    if not isinstance(settings, dict):
        return "malformed hooks config (hooks config root is not an object)"
    hooks = _validate_hooks_object(settings, path, allow_null=True)
    invalid_owned_events = sorted(
        event for event in _CURSOR_HOOK_DEFS if event in hooks and not isinstance(hooks[event], list)
    )
    if invalid_owned_events:
        return "non-list values for Poppy-owned hook events: " + ", ".join(invalid_owned_events)
    return None


def _read_cursor_hooks_config(path: Path) -> tuple[dict, str | None]:
    if not path.exists():
        return {}, None
    try:
        settings = json.loads(path.read_text())
    except UnicodeDecodeError as exc:
        raise CorruptConfigError(
            f"Refusing to overwrite unparseable config at {path}. Fix or remove it, then re-run `poppy setup`."
        ) from exc
    except (OSError, ValueError) as exc:
        return {}, f"malformed hooks config ({exc})"
    reset_reason = _validate_cursor_hooks_config(settings, path)
    return ({} if reset_reason else settings), reset_reason


def install_cursor_hooks(cursor_home: Path | None = None) -> Path:
    """Merge Poppy's hooks into Cursor's native global ``hooks.json``."""
    hooks_path = (cursor_home or get_cursor_home()) / "hooks.json"
    settings, reset_reason = _read_cursor_hooks_config(hooks_path)
    if reset_reason is not None:
        backup = _backup_once(hooks_path, ".bak")
        sys.stderr.write(
            f"poppy setup cursor: found {reset_reason}; backed up {hooks_path} to {backup} "
            "and wrote a fresh valid hooks file\n"
        )

    hooks = settings.get("hooks")
    if hooks is None:
        hooks = {}

    # Only touch Poppy command entries under events Poppy owns. Every other
    # event key and entry is retained exactly as parsed, including unknown keys.
    merged: dict[str, object] = dict(hooks)
    poppy_commands = set(_CURSOR_HOOK_DEFS.values())
    for event, command in _CURSOR_HOOK_DEFS.items():
        entries = hooks.get(event)
        if entries is None:
            kept: list[object] = []
        elif isinstance(entries, list):
            kept = [
                entry for entry in entries if not (isinstance(entry, dict) and entry.get("command") in poppy_commands)
            ]
        else:
            sys.stderr.write(
                f"poppy setup cursor: kept unexpected non-list value for hook event {event!r}; "
                "skipped installing the Poppy hook there\n"
            )
            continue
        merged[event] = [*kept, {"command": command}]

    settings["version"] = 1
    settings["hooks"] = merged
    _write_json(hooks_path, settings)
    return hooks_path


def cursor_unknown_hook_events(hooks_path: Path) -> list[str]:
    """Return event keys that Cursor does not document and will reject."""
    settings = _read_json(hooks_path)
    hooks = settings.get("hooks") if isinstance(settings, dict) else None
    if not isinstance(hooks, dict):
        return []
    return sorted(set(hooks) - CURSOR_DOCUMENTED_HOOK_EVENTS)


def is_cursor_hooks_installed(cursor_home: Path | None = None) -> bool:
    """True when every current Cursor Poppy hook is present exactly once."""
    settings = _read_json((cursor_home or get_cursor_home()) / "hooks.json")
    if not isinstance(settings, dict):
        return False
    hooks = settings.get("hooks", {})
    if settings.get("version") != 1 or not isinstance(hooks, dict):
        return False
    for event, command in _CURSOR_HOOK_DEFS.items():
        entries = hooks.get(event, [])
        if not isinstance(entries, list):
            return False
        matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("command") == command]
        if len(matches) != 1:
            return False
    return True


def install_codex_hooks(codex_home: Path | None = None) -> Path:
    """Merge Poppy's supported hooks into Codex's ``hooks.json``.

    Removes stale/duplicate Poppy commands first, including SessionEnd and
    PostCompact entries copied by Codex's settings import, while preserving all
    non-Poppy commands and groups. Re-appending the current definitions gives a
    deterministic, idempotent result.
    """
    hooks_path = (codex_home or get_codex_home()) / "hooks.json"
    settings = _read_json(hooks_path, strict=True)
    hooks = _validate_hooks_object(settings, hooks_path, allow_null=True)

    # Preserve verbatim any shape the merger does not positively recognize as a
    # poppy-command group; only entries it identifies as poppy commands are
    # stripped and re-appended below. This keeps user-authored config intact and
    # never crashes on an odd shape.
    merged: dict[str, object] = {}
    for event, groups in hooks.items():
        if not isinstance(groups, list):
            # A non-list event value is a shape we do not understand — keep it.
            merged[event] = groups
            continue
        kept_groups: list[object] = []
        for group in groups:
            # A group with no recognizable "hooks" list is not a poppy group; keep
            # it exactly (this covers e.g. {"matcher": "custom", "disabled": true}).
            if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
                kept_groups.append(group)
                continue
            commands = group["hooks"]
            kept = [h for h in commands if not isinstance(h, dict) or h.get("command") not in _ALL_POPPY_HOOK_COMMANDS]
            if len(kept) == len(commands):
                kept_groups.append(group)  # no poppy commands here — preserve as authored
            elif kept:
                kept_groups.append({**group, "hooks": kept})  # keep the non-poppy remainder
            # else: a purely-poppy group — drop it; the current def is re-appended below
        if kept_groups:
            merged[event] = kept_groups

    for event, (matcher, command) in _CODEX_HOOK_DEFS.items():
        poppy_group = {"matcher": matcher, "hooks": [{"type": "command", "command": command}]}
        existing = merged.get(event)
        if existing is None:
            merged[event] = [poppy_group]
        elif isinstance(existing, list):
            existing.append(poppy_group)
        else:
            # The user's event value is a non-list shape we preserved verbatim;
            # never overwrite it. Skip installing here (doctor will flag it as not
            # installed) rather than destroy the user's config or crash.
            sys.stderr.write(
                f"poppy setup codex: kept unexpected non-list value for hook event {event!r}; "
                "skipped installing the Poppy hook there\n"
            )

    settings["hooks"] = merged
    _write_json(hooks_path, settings)
    return hooks_path


def is_codex_hooks_installed(codex_home: Path | None = None) -> bool:
    """True when every current Codex Poppy hook is present exactly once."""
    settings = _read_json((codex_home or get_codex_home()) / "hooks.json")
    if not isinstance(settings, MutableMapping):
        return False
    hooks = settings.get("hooks", {})
    if not isinstance(hooks, MutableMapping):
        return False
    for event, (matcher, command) in _CODEX_HOOK_DEFS.items():
        entries = hooks.get(event, [])
        if not isinstance(entries, list):
            return False
        matches = [
            group
            for group in entries
            if isinstance(group, dict)
            and group.get("matcher") == matcher
            and isinstance(group.get("hooks"), list)
            and any(isinstance(h, dict) and h.get("command") == command for h in group["hooks"])
        ]
        if len(matches) != 1:
            return False
    return True


def has_any_cursor_poppy_hook(cursor_home: Path | None = None) -> bool:
    """True if AT LEAST ONE Poppy Cursor hook is present. A PARTIAL install (some
    hooks removed) still counts as configured, so doctor keeps the client's
    footprint and surfaces exactly what's missing instead of going silent."""
    settings = _read_json((cursor_home or get_cursor_home()) / "hooks.json")
    if not isinstance(settings, MutableMapping):
        return False
    hooks = settings.get("hooks", {})
    if not isinstance(hooks, MutableMapping):
        return False
    for event, command in _CURSOR_HOOK_DEFS.items():
        entries = hooks.get(event, [])
        if isinstance(entries, list) and any(
            isinstance(entry, dict) and entry.get("command") == command for entry in entries
        ):
            return True
    return False


def has_any_codex_poppy_hook(codex_home: Path | None = None) -> bool:
    """True if AT LEAST ONE Poppy Codex hook is present (partial install still
    counts as configured)."""
    settings = _read_json((codex_home or get_codex_home()) / "hooks.json")
    if not isinstance(settings, MutableMapping):
        return False
    hooks = settings.get("hooks", {})
    if not isinstance(hooks, MutableMapping):
        return False
    for event, (_matcher, command) in _CODEX_HOOK_DEFS.items():
        entries = hooks.get(event, [])
        if not isinstance(entries, list):
            continue  # a null / non-list event value means "no such hook", not a crash
        for group in entries:
            if (
                isinstance(group, dict)
                and isinstance(group.get("hooks"), list)
                and any(isinstance(h, dict) and h.get("command") == command for h in group["hooks"])
            ):
                return True
    return False


def _install_hook(claude_dir: Path, event: str) -> Path:
    settings_path = claude_dir / "settings.json"
    settings = _read_json(settings_path, strict=True)
    _validate_claude_hooks_config(settings, settings_path)
    settings.setdefault("hooks", {})
    settings["hooks"].setdefault(event, [])

    matcher, command = _HOOK_DEFS[event]
    poppy_hooks = [{"type": "command", "command": command}]

    # Exact single-command groups are Poppy-owned and safe to migrate in place.
    for group in settings["hooks"][event]:
        if group.get("hooks") == poppy_hooks:
            if group.get("matcher") != matcher:
                group["matcher"] = matcher
                _write_json(settings_path, settings)
            return settings_path

    # A mixed group may be user-owned. Treat the command as installed without
    # changing the group or adding a duplicate Poppy hook.
    for group in settings["hooks"][event]:
        for h in group.get("hooks", []):
            if h.get("command") == command:
                return settings_path

    settings["hooks"][event].append(
        {
            "matcher": matcher,
            "hooks": poppy_hooks,
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


def install_post_compact_hook(claude_config_dir: Path | None = None) -> Path:
    return _install_hook(claude_config_dir or get_claude_config_dir(), "PostCompact")


def remove_legacy_hooks(claude_config_dir: Path | None = None) -> list[str]:
    """Remove poppy hook entries that are no longer installed by current setup.

    Returns a list of (event, command) descriptors that were removed. Idempotent.
    Honors the user's other hook entries — only Poppy-owned commands are touched.
    """
    claude_dir = claude_config_dir or get_claude_config_dir()
    settings_path = claude_dir / "settings.json"
    settings = _read_json(settings_path, strict=True)
    _validate_claude_hooks_config(settings, settings_path)
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
    # Guard non-object top-level JSON and non-object nested nodes (`[]`/`null`/a
    # string): they parse but have no `.get`, and this backs doctor's read path.
    if not isinstance(settings, MutableMapping):
        return False
    _, command = _HOOK_DEFS[event]
    hooks = settings.get("hooks", {})
    events = hooks.get(event, []) if isinstance(hooks, MutableMapping) else []
    for group in events if isinstance(events, list) else []:
        if not isinstance(group, MutableMapping):
            continue
        for h in group.get("hooks", []) if isinstance(group.get("hooks"), list) else []:
            if isinstance(h, MutableMapping) and h.get("command") == command:
                return True
    return False


# ---------------------------------------------------------------------------
# Managed CLAUDE.md block
# ---------------------------------------------------------------------------


def install_primer_md_block(md_path: Path) -> Path:
    """Insert (or update) a managed Poppy primer block in an instructions file.

    Used for Claude Code (CLAUDE.md), Codex and Pi (AGENTS.md), and Copilot CLI
    (copilot-instructions.md).
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


def _remove_managed_primer_block(md_path: Path) -> None:
    """Remove only Poppy's managed primer block, deleting an empty result."""
    if not md_path.exists():
        return
    text = md_path.read_text()
    if CLAUDE_MD_BEGIN not in text or CLAUDE_MD_END not in text:
        return
    start = text.index(CLAUDE_MD_BEGIN)
    end_marker = text.find(CLAUDE_MD_END, start)
    if end_marker == -1:
        return
    end = end_marker + len(CLAUDE_MD_END)
    remaining = text[:start] + text[end:]
    if remaining.strip():
        md_path.write_text(remaining)
    else:
        md_path.unlink()


def _legacy_primer_path(client: str) -> Path | None:
    if client == "copilot-cli":
        return _copilot_home() / "AGENTS.md"
    if client == "pi":
        return Path.home() / ".pi" / "AGENTS.md"
    return None


def _client_primer_path(client: str) -> Path | None:
    """Return the client's auto-loaded global instructions file, if supported.

    Copilot CLI uses ``$COPILOT_HOME/copilot-instructions.md`` and Pi uses
    ``$PI_CODING_AGENT_DIR/AGENTS.md``. Their documented environment overrides
    fall back to ``~/.copilot`` and ``~/.pi/agent`` respectively.
    """
    if client == "codex":
        return get_codex_home() / "AGENTS.md"
    if client == "copilot-cli":
        return _copilot_home() / "copilot-instructions.md"
    if client == "pi":
        return _pi_agent_dir() / "AGENTS.md"
    return None


# ---------------------------------------------------------------------------
# Top-level installer used by the CLI
# ---------------------------------------------------------------------------


def validate_client_config(
    *,
    client: str = "claude-code",
    claude_config_dir: Path | None = None,
    install_hooks: bool = True,
) -> None:
    """Reject unparseable or structurally invalid configs before any changes."""
    claude_dir = claude_config_dir or get_claude_config_dir()
    config_path = _client_settings_path(client, claude_dir)
    if client == "codex":
        _read_codex_toml(config_path, strict=True)
        json_paths = [config_path.with_suffix(".json")]
    else:
        json_paths = [config_path]
        if client == "claude-desktop":
            msix_path = get_claude_desktop_msix_config_path()
            if msix_path is not None and msix_path != config_path:
                json_paths.append(msix_path)
    for path in json_paths:
        _read_json(path, strict=True)

    if install_hooks:
        if client == "claude-code":
            path = claude_dir / "settings.json"
            _validate_claude_hooks_config(_read_json(path, strict=True), path)
        elif client == "codex":
            path = get_codex_home() / "hooks.json"
            _validate_hooks_object(_read_json(path, strict=True), path, allow_null=True)
        elif client == "cursor":
            # Validate without performing the writer's intentional backup/reset.
            _read_cursor_hooks_config(get_cursor_home() / "hooks.json")


def install_for_client(
    *,
    client: str = "claude-code",
    claude_config_dir: Path | None = None,
    install_hooks: bool = True,
    install_claude_md: bool = True,
    daemon: bool = False,
    daemon_port: int = 7679,
    daemon_token: str | None = None,
) -> dict[str, Path]:
    """Install Poppy into the given client. Returns a {label: path} map."""
    validate_client_config(client=client, claude_config_dir=claude_config_dir, install_hooks=install_hooks)
    paths: dict[str, Path] = {}

    if client == "claude-desktop":
        # Surface the backup path before the merge writes the file, so the
        # user sees what was preserved.
        target = _client_settings_path(client, claude_config_dir or get_claude_config_dir())
        backup = _backup_once(target, CLAUDE_DESKTOP_BACKUP_SUFFIX)
        if backup is not None:
            paths["backup"] = backup

    if client == "codex":
        config_path, backup, migrated = _install_codex_mcp_config(
            daemon=daemon,
            daemon_port=daemon_port,
            daemon_token=daemon_token,
        )
        paths["MCP config"] = config_path
        if backup is not None:
            paths["backup"] = backup
        if migrated is not None:
            paths["Removed legacy MCP config"] = migrated
        if install_hooks:
            paths["Codex hooks"] = install_codex_hooks()
    else:
        if client == "claude-desktop":
            paths["MCP config"] = _install_json_mcp_config(
                _client_settings_path(client, claude_config_dir or get_claude_config_dir()),
                client,
                daemon=daemon,
                daemon_port=daemon_port,
                daemon_token=daemon_token,
                backup_existing=False,
            )
        else:
            paths["MCP config"] = install_mcp_config(
                claude_config_dir,
                client=client,
                daemon=daemon,
                daemon_port=daemon_port,
                daemon_token=daemon_token,
            )
        if client == "claude-desktop":
            msix_path = get_claude_desktop_msix_config_path()
            if msix_path is not None and msix_path != paths["MCP config"]:
                msix_backup = _backup_once(msix_path, CLAUDE_DESKTOP_BACKUP_SUFFIX)
                if msix_backup is not None:
                    paths["MSIX backup"] = msix_backup
                paths["MSIX MCP config"] = _install_json_mcp_config(msix_path, client, backup_existing=False)

        if client == "cursor" and install_hooks:
            paths["Cursor hooks"] = install_cursor_hooks()

    primer_path = _client_primer_path(client)
    if primer_path is not None:
        legacy_primer = _legacy_primer_path(client)
        if legacy_primer is not None:
            _remove_managed_primer_block(legacy_primer)
        paths[f"Primer ({primer_path.name})"] = install_primer_md_block(primer_path)

    if client == "claude-code":
        if install_hooks:
            paths["SessionStart hook"] = install_session_start_hook(claude_config_dir)
            paths["UserPromptSubmit hook"] = install_user_prompt_submit_hook(claude_config_dir)
            paths["PreToolUse hook"] = install_pre_tool_use_hook(claude_config_dir)
            paths["SessionEnd hook"] = install_session_end_hook(claude_config_dir)
            paths["PostCompact hook"] = install_post_compact_hook(claude_config_dir)
            # Migrate older installs: Stop hook used to run consolidation, but it
            # fires after every assistant turn — SessionEnd is the right event.
            remove_legacy_hooks(claude_config_dir)
        if install_claude_md:
            paths["CLAUDE.md block"] = install_claude_md_block(claude_config_dir)

    return paths
