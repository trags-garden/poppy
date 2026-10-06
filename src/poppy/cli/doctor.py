"""Installation checks for the doctor command."""

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click


@dataclass
class _DaemonContext:
    claude_dir: Path
    state: Any
    clients: dict[str, tuple[str, str | None]]
    probe_cache: dict[tuple, tuple[bool, str | None]] = field(default_factory=dict)
    dead_clients: list[str] = field(default_factory=list)


@dataclass
class _CaptureContext:
    config: Any
    health: Any
    status: Any
    expected: bool


@dataclass
class DoctorContext:
    """Shared read results and the reporter's aggregate success state."""

    ok: bool = True
    poppy_dir: Path = field(init=False)
    daemon: _DaemonContext = field(init=False)
    capture: _CaptureContext = field(init=False)

    def line(self, label: str, status: str, detail: str = "", hint: str = "", separator: str = ", ") -> None:
        if status == "FAIL":
            self.ok = False
        marker = {"OK": "✓", "WARN": "!", "FAIL": "✗"}.get(status, "·")
        bits = [f"  [{marker}] {label}: {status}"]
        if detail:
            bits.append(f"{separator}{detail}")
        if hint and status != "OK":
            # A detail that is already a full sentence ends with its own period;
            # joining that with "<separator>hint" would print a stray ".," right
            # before the hint. Drop that trailing period so the hint reads as a
            # continuation instead.
            if bits and bits[-1].endswith("."):
                bits[-1] = bits[-1][:-1]
            bits.append(f"{separator}{hint}")
        click.echo("".join(bits))


def safe_read(fn, default):
    """Run a read-only config query, treating an unusable config as absent.
    Doctor reads up to eleven client configs it doesn't own; one bad file for
    a client the user never chose must never abort the run (the last command
    the install script prints). Catches the whole read/parse family:
    ``OSError`` (mode 000, TCC denial, EIO) and ``ValueError``, which covers
    both ``UnicodeDecodeError`` (non-UTF-8 bytes) and ``JSONDecodeError``.
    This tolerance is doctor-only, the shared `_read_json`
    still raises so WRITE paths never truncate a config they couldn't read."""
    try:
        return fn()
    except (OSError, ValueError):
        return default


def _is_sendable_header_value(value: str) -> bool:
    """Whether ``value`` can be transmitted as an HTTP header value, no bare
    CR/LF and latin-1 encodable (what http.client requires)."""
    if "\n" in value or "\r" in value:
        return False
    try:
        value.encode("latin-1")
    except UnicodeEncodeError:
        return False
    return True


def _client_daemon_problem(ctx: DoctorContext, client_id: str) -> tuple[str, str] | None:
    from urllib.parse import urlsplit as _urlsplit

    from poppy.mcp_server.lifecycle import probe_status as _probe_status

    poppy_dir = ctx.poppy_dir
    _LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
    _daemon_probe_cache = ctx.daemon.probe_cache
    registration = ctx.daemon.clients.get(client_id)
    if registration is None:
        return None
    url, auth_header = registration
    setup_hint = f"re-run `poppy setup {client_id} --daemon` to rewrite the registration"
    try:
        parsed = _urlsplit(url)
        host = parsed.hostname
        port = parsed.port
        path = parsed.path
    except ValueError:
        return ("registered against the HTTP daemon, but its URL is malformed", setup_hint)
    if not host:
        # No netloc (e.g. `http:/127.0.0.1/mcp`, single slash) → unparseable
        # host; malformed, not "non-loopback".
        return ("registered against the HTTP daemon, but its URL is malformed (no host)", setup_hint)
    if host not in _LOOPBACK_HOSTS:
        return (
            f"registered against a NON-loopback host ({host or 'unknown'}), the client is sending its "
            "bearer token and MCP traffic off-box",
            f"re-run `poppy setup {client_id} --daemon` to point it back at the local daemon",
        )
    if port is None:
        return ("registered against the HTTP daemon, but its URL is malformed", setup_hint)
    # Host+port alone is not health: the daemon speaks plain http and resolves
    # the route from the PATH, so an `https://` scheme or any other path
    # (`/not-mcp`, `/wrong/mcp`, `/mcp;` or `/mcp;broken`) means the client
    # can't reach the endpoint even though a probe of the same host:port
    # /status answers. We use urlsplit (NOT urlparse) so a
    # `;params` segment stays in `.path`, urlparse would peel it off and let
    # `/mcp;` pass. A query (`?x=1`) or fragment (`#frag`) does NOT change the
    # route (the server ignores the query for matching and never sees the
    # fragment), and they're separate urlsplit fields, so those are fine.
    if parsed.scheme != "http" or path.rstrip("/") != "/mcp":
        return (
            f"registered with a URL Poppy's daemon does not serve ({parsed.scheme}://…{path or '/'}); "
            "the daemon speaks http at /mcp",
            setup_hint,
        )
    # A stored header with a newline / non-latin-1 char cannot form a valid
    # HTTP Authorization header, the daemon would never see it, so report it
    # as malformed without probing.
    if auth_header is not None and not _is_sendable_header_value(auth_header):
        return (
            "registered with a bearer token that is not a valid HTTP header "
            "(contains a newline or non-latin-1 character)",
            f"re-run `poppy setup {client_id} --daemon` to refresh the client's token",
        )
    # Probe the EXACT host the client will use (don't rewrite ::1 → 127.0.0.1)
    # with the client's EXACT Authorization header (verbatim, not normalized).
    # `token=None` is explicit: a client probe carries ONLY the client's own
    # credential, including NONE, and must NEVER fall back to the store token
    # (that fallback would send the daemon's real token to a client's listener
    # and mask a no-credential client's 401 as OK). The cache
    # key includes host AND the header. probe_status is TOTAL (any failure →
    # (None, error)), so a broken endpoint never aborts doctor.
    key = (host, port, auth_header)
    if key not in _daemon_probe_cache:
        payload, error = _probe_status(poppy_dir, port, host=host, token=None, auth_header=auth_header)
        _daemon_probe_cache[key] = (payload is not None, error)
    reachable, error = _daemon_probe_cache[key]
    if reachable:
        return None
    if error and "401" in error and auth_header is None:
        return (
            "registered against the HTTP daemon, but the client sends no Authorization "
            "credential and the daemon requires one",
            f"re-run `poppy setup {client_id} --daemon` to write the client's token",
        )
    if error and "401" in error:
        return (
            "registered against the HTTP daemon, but its bearer token is stale (daemon returns 401)",
            f"re-run `poppy setup {client_id} --daemon` to refresh the client's token",
        )
    return (
        "registered against the HTTP daemon, but its port is not reachable",
        "start the daemon (`poppy daemon install && poppy daemon start`) "
        f"or re-run `poppy setup {client_id}` without --daemon",
    )


def _any_present(*checks) -> bool:
    """Footprint = ANY Poppy component present, never "all". A multi-piece
    client (hooks + primer + MCP; plugin + config + guidance) is "set up" the
    moment ONE piece exists, so a PARTIAL install still surfaces exactly
    what's missing instead of going silent. Each check is read independently
    (safe_read) so one unreadable/undecodable piece never masks another.
    One shared rule for every client so it can't regress
    client-by-client."""
    return any(safe_read(check, False) for check in checks)


def _client_components(client_id: str) -> tuple:
    """The independent Poppy components for a client, MCP entry, hooks,
    primer, as zero-arg checks. Any one present ⇒ footprinted."""
    from poppy.setup.claude_code import (
        _client_primer_path,
        has_any_codex_poppy_hook,
        has_any_cursor_poppy_hook,
        is_mcp_installed,
        managed_primer_present,
    )

    checks: list = [lambda: is_mcp_installed(client=client_id)]
    if client_id == "cursor":
        checks.append(has_any_cursor_poppy_hook)
    elif client_id == "codex":
        checks.append(has_any_codex_poppy_hook)
    primer = _client_primer_path(client_id) if client_id in ("codex", "copilot-cli", "pi") else None
    if primer is not None:
        checks.append(lambda: managed_primer_present(primer))
    return tuple(checks)


def _client_has_poppy(client_id: str) -> bool:
    return _any_present(*_client_components(client_id))


def _read_json_or_empty(path: Path) -> dict:
    # Unreadable (OSError), non-UTF-8 (UnicodeDecodeError), malformed
    # (JSONDecodeError, both ValueError), or non-object JSON (`[]`/`null`/`"x"`)
    # → treat as an empty config so a broken Claude Desktop file never aborts
    # doctor.
    try:
        parsed = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _poppy_registered(settings: dict) -> bool:
    # Guard a non-dict `mcpServers` (`{"mcpServers": null}` etc.): `"poppy" in
    # None` would TypeError-crash doctor.
    servers = settings.get("mcpServers")
    return isinstance(servers, dict) and "poppy" in servers


def check_executable(ctx: DoctorContext) -> None:
    line = ctx.line

    poppy_bin = shutil.which("poppy")
    line("poppy executable", "OK" if poppy_bin else "FAIL", poppy_bin or "not on PATH")


def check_storage(ctx: DoctorContext) -> None:
    from poppy.cli import main as _main

    line = ctx.line

    poppy_dir = _main._get_poppy_dir()
    line("storage dir", "OK" if poppy_dir.exists() else "WARN", str(poppy_dir))
    ctx.poppy_dir = poppy_dir


def check_trags_key(ctx: DoctorContext) -> None:
    from poppy.cli import main as _main

    line = ctx.line
    poppy_dir = ctx.poppy_dir

    from poppy.config import CONFIG_FILENAME, TRAGS_API_KEY_ENV, trags_api_key_location

    env_key_set = bool(os.environ.get(TRAGS_API_KEY_ENV))
    key_location = trags_api_key_location(poppy_dir)
    if key_location == "config-file":
        # Plaintext in config.json is the documented fallback, not the intended
        # home: say why the keychain was skipped, so an upgrade that silently
        # fell back is visible here and fixable. Checked ahead of the
        # env override deliberately, POPPY_TRAGS_API_KEY changes which key is
        # USED, not whether a secret sits on disk, so acting on this line's own
        # hint must not silence it while the file copy is still there.
        from poppy import keychain

        config_path = poppy_dir / CONFIG_FILENAME
        # Report the mode actually on disk. save_config writes 0600, but a file
        # predating that, a manual edit or a restore can be looser, and nothing
        # tightens it while the key stays in the file.
        mode = safe_read(lambda: config_path.stat().st_mode & 0o777, None)
        if mode is None:
            mode_note = "file mode unreadable"
        elif mode & 0o077:
            mode_note = f"file mode {mode:04o}, readable beyond the owner"
        else:
            mode_note = f"file mode {mode:04o}"
        # Nothing on disk records WHY the fallback happened, so read the
        # classification this process recorded when it stored the key, and probe
        # the session when there is none. Same classifier `config set` uses, so
        # the two surfaces cannot disagree.
        reason = {
            keychain.PROBE_NO_BACKEND: "no usable OS keychain backend here",
            keychain.PROBE_WRITE_REJECTED: "the OS keychain rejected the write",
            keychain.PROBE_READBACK_FAILED: "the OS keychain could not be read back in this session",
            # Only doctor reaches this one: the fallback happened in an earlier
            # session and this one is healthy, so do not report a failure that
            # is not happening here.
            keychain.PROBE_OK: "the OS keychain is usable in this session, so the key can be moved into it",
        }.get(
            _main._trags_key_keychain_failure(poppy_dir),
            "the OS keychain could not be used for the key in this session",
        )
        detail = f"config.json (plaintext, {mode_note}; {reason})"
        if env_key_set:
            detail += f"; {TRAGS_API_KEY_ENV} is set and takes precedence, but the file copy is still on disk"
        hints = []
        if mode is not None and mode & 0o077:
            hints.append(f"Restrict it with `chmod 600 {config_path}`.")
        if env_key_set:
            # The usual "use the env var" advice is already satisfied here, so
            # the only remaining action is removing the copy it shadows.
            hints.append(
                'Drop the file copy with `poppy config set trags-api-key ""`; '
                f"{TRAGS_API_KEY_ENV} already supplies the key."
            )
        else:
            hints.append(
                f"Prefer {TRAGS_API_KEY_ENV} in the environment, or re-run "
                "`poppy config set trags-api-key <usr_xxxxx>` from an interactive login session."
            )
        line("Trags key", "WARN", detail, hint=" ".join(hints))
    elif env_key_set:
        line("Trags key", "OK", f"{TRAGS_API_KEY_ENV} env")
    elif key_location == "keychain-unreadable":
        line(
            "Trags key",
            "WARN",
            "keychain (unreadable in this session; cannot tell whether a key is stored)",
            hint="Run from an interactive login session, or set POPPY_TRAGS_API_KEY for background jobs.",
        )
    elif key_location == "none":
        line("Trags key", "OK", "none (Trags sync not configured; optional)")
    else:
        line("Trags key", "OK", "keychain")


def check_custom_redaction(ctx: DoctorContext) -> None:
    from poppy.cli import main as _main

    line = ctx.line
    poppy_dir = ctx.poppy_dir

    from poppy.capture.redaction import MIN_CUSTOM_SECRET_LENGTH, load_custom_redaction, valid_env_var_name

    redaction_config = _main.load_config(poppy_dir)
    _patterns, redaction_issues = load_custom_redaction(redaction_config)
    literal_count = sum(
        len(literal.strip()) >= MIN_CUSTOM_SECRET_LENGTH for literal in redaction_config.redaction_literals
    )
    env_var_count = sum(valid_env_var_name(name) for name in redaction_config.redaction_env_vars)
    line(
        "custom redaction",
        "OK",
        f"{literal_count} literal(s), {env_var_count} environment-variable name(s) configured",
    )
    for issue in redaction_issues:
        line("custom redaction entry", "WARN", issue)


def _prepare_daemon(ctx: DoctorContext) -> None:
    from poppy.setup.claude_code import get_claude_config_dir

    poppy_dir = ctx.poppy_dir

    from poppy.mcp_server.lifecycle import inspect_daemon
    from poppy.setup.claude_code import poppy_daemon_registration

    # Which clients are wired to the shared HTTP daemon (`poppy setup <client>
    # --daemon`), and the bearer token baked into each client's own config. A
    # client pointing at a daemon that is gone/unreachable, or holding a stale
    # token the daemon now 401s, is a dead MCP server the plain "poppy in
    # mcpServers" check can't see, so it forces the daemon health checks to run
    # even with no agent, lock, or liveness.
    claude_dir = get_claude_config_dir()
    daemon_clients: dict[str, tuple[str, str | None]] = {}
    for client_id in (
        "claude-code",
        "claude-desktop",
        "claude-desktop-msix",
        "cursor",
        "windsurf",
        "codex",
        "copilot-cli",
        "pi",
        "gemini",
        "vscode",
    ):
        registration = safe_read(lambda cid=client_id: poppy_daemon_registration(cid, claude_dir), None)
        if registration is not None:
            daemon_clients[client_id] = registration

    # Daemon mode is opt-in (`poppy setup <client> --daemon` / `poppy daemon
    # install`); a default (embedded) install never runs one. Report daemon
    # health when a service agent is installed, a daemon is running, OR a client
    # is registered against the daemon, otherwise the happy-path install would
    # show misleading WARNs, while a broken daemon-mode install would
    # go silent. A half-installed daemon still surfaces its WARNs here.
    daemon_state = inspect_daemon(poppy_dir)
    ctx.daemon = _DaemonContext(claude_dir, daemon_state, daemon_clients)
    dead_daemon_clients = [client_id for client_id in daemon_clients if _client_daemon_problem(ctx, client_id)]
    ctx.daemon.dead_clients = dead_daemon_clients


def check_daemon(ctx: DoctorContext) -> None:
    from poppy.cli import main as _main

    line = ctx.line
    _prepare_daemon(ctx)
    daemon_state = ctx.daemon.state
    daemon_clients = ctx.daemon.clients
    dead_daemon_clients = ctx.daemon.dead_clients
    daemon_running = daemon_state.lock_held or daemon_state.reachable

    if daemon_state.installed or daemon_running or daemon_clients:
        # If every registered client reached its daemon (per-client probe on the
        # client's own port+token), the daemon IS running and reachable, even
        # though the summary probe hit the configured/default port, which a client
        # set up on a custom --daemon port won't match. Don't emit a spurious
        # running/reachable WARN that the per-client checks already contradict.
        clients_all_reachable = bool(daemon_clients) and not dead_daemon_clients
        line(
            "daemon installed",
            "OK" if daemon_state.installed else "WARN",
            hint="run `poppy daemon install` to install",
        )
        running_detail = f"PID {daemon_state.pid}" if daemon_state.pid else ""
        line(
            "daemon running",
            "OK" if (daemon_running or clients_all_reachable) else "WARN",
            running_detail,
            hint="run `poppy daemon start` to start",
        )
        line(
            "daemon reachable",
            "OK" if (daemon_state.reachable or clients_all_reachable) else "WARN",
            hint="check `poppy daemon status` and the daemon log",
        )
        if dead_daemon_clients:
            names = ", ".join(sorted(dead_daemon_clients))
            line(
                "daemon (client MCP)",
                "WARN",
                f"{names} registered against the HTTP daemon, but it is not usable (unreachable, "
                "stale token, or malformed URL). MCP is dead there",
                hint="start/reinstall the daemon (`poppy daemon install && poppy daemon start`) "
                "or re-run `poppy setup <client> --daemon` to refresh the registration",
            )
        if daemon_state.status:
            daemon_version = str(daemon_state.status.get("version", "unknown"))
            line(
                "daemon version",
                "OK" if daemon_version == _main.__version__ else "WARN",
                f"daemon {daemon_version}, CLI {_main.__version__}",
                hint="restart the daemon after upgrading Poppy",
            )
            loaded = bool(daemon_state.status.get("models_loaded"))
            deadline = daemon_state.status.get("model_idle_deadline")
            model_detail = f"{daemon_state.status.get('engine', 'unknown')} engine, {'hot' if loaded else 'cold'}"
            if deadline:
                model_detail += f", idle deadline {deadline}"
            line("daemon models", "OK", model_detail)


def check_version(ctx: DoctorContext) -> None:
    from poppy.cli import main as _main

    line = ctx.line
    poppy_dir = ctx.poppy_dir

    version_check = _main.update_check.check(poppy_dir)
    if not version_check.enabled:
        line(
            "version",
            "OK",
            f"{version_check.installed_version} (check is off: {version_check.disabled_reason})",
        )
    elif version_check.update_available:
        line(
            "version",
            "WARN",
            f"{version_check.latest_version} available (installed {version_check.installed_version})",
            hint=_main.update_check.upgrade_command(),
        )
    elif version_check.latest_version is not None:
        line("version", "OK", f"{version_check.installed_version} (latest)")
    else:
        line("version", "WARN", f"{version_check.installed_version} (latest version unavailable)")


def check_encryption(ctx: DoctorContext) -> None:
    line = ctx.line
    poppy_dir = ctx.poppy_dir

    from poppy import encryption

    enc = encryption.status(poppy_dir)
    if enc.inconsistent:
        line("encryption at rest", "FAIL", f"inconsistent state ({enc.state})", hint="run `poppy encrypt repair`")
    elif enc.state == "corrupt":
        line(
            "encryption at rest",
            "FAIL",
            "store file is not plaintext and does not open under the key (corrupt or foreign)",
            hint="restore a good backup; see `poppy encrypt status`",
        )
    elif enc.state == "unreadable":
        line(
            "encryption at rest",
            "FAIL",
            "store could not be read (permission or read-only filesystem), not corruption",
            hint="fix the permissions on the store and its directory",
        )
    elif enc.state == "unknown":
        line(
            "encryption at rest",
            "WARN",
            "store file is not plaintext and cannot be verified here",
            hint="install the encryption extra and provide the key, then `poppy encrypt status`",
        )
    elif not enc.enabled:
        line("encryption at rest", "OK", "off (plaintext local store)")
    elif not enc.deps_installed:
        line(
            "encryption at rest",
            "FAIL",
            "on, but the encryption extra is not installed",
            hint="pipx inject poppy-memory sqlcipher3",
        )
    elif enc.env_key_malformed:
        line(
            "encryption at rest",
            "FAIL",
            "on, but POPPY_DB_KEY is malformed (not 64 hex) and shadows the keychain",
            hint="fix or unset POPPY_DB_KEY",
        )
    elif enc.key_present:
        where = f"{encryption.ENV_KEY} env var" if enc.env_key_active else "OS keychain"
        line("encryption at rest", "OK", f"on, key in {where}")
    else:
        line(
            "encryption at rest",
            "FAIL",
            "on, but no key available here",
            hint="restore the keychain entry or set POPPY_DB_KEY; the store cannot be opened without it",
        )
    if enc.residue:
        has_plain = any(".plain-tmp" in p.name for p in enc.residue)
        detail = "leftover migration temp files" + (" (includes a decrypted copy)" if has_plain else "")
        line("encryption residue", "WARN", detail, hint="run `poppy encrypt repair` to remove them")
    if enc.stray:
        line(
            "encryption sidecars",
            "WARN",
            "quarantined stray sidecars set aside (not replayed)",
            hint="see `poppy encrypt status`; delete the .stray-* files if unneeded",
        )


def check_engine(ctx: DoctorContext) -> None:
    from poppy.cli import main as _main

    line = ctx.line
    poppy_dir = ctx.poppy_dir

    try:
        engine = _main._get_engine()
        s = engine.stats()
        # Surface two drift signals:
        #   1. Configured engine != active engine, get_engine fell back due to
        #      missing deps or an unknown name.
        #   2. memory_embeddings rows tagged with a model_id that doesn't match
        #      the active engine's bi-encoder (or with NULL legacy rows). Those
        #      rows silently degrade to FTS-only recall.
        from poppy.config import load_config as _load_config
        from poppy.engine.migration import stale_stats as _stale_stats

        configured = _load_config(poppy_dir).engine
        engine_msg = f"{s.engine_name} v{s.engine_version} ({s.memory_count} memories)"
        if configured != s.engine_name:
            line(
                "engine",
                "WARN",
                f"configured={configured!r}, active={s.engine_name!r}",
                hint="install the configured engine's deps or `poppy engines use <name>`",
            )
        else:
            line("engine", "OK", engine_msg)
        if engine.model_id is not None:
            drift = _stale_stats(poppy_dir / "memories.db", engine.model_id)
            if drift.needs_migration > 0:
                line(
                    "embedding index",
                    "WARN",
                    f"{drift.needs_migration} memories need re-embedding "
                    f"({drift.missing} never embedded, compatible={drift.compatible})",
                    hint="run `poppy migrate-engine` to re-embed",
                )
    except Exception as exc:
        line("engine", "FAIL", str(exc))


def _prepare_capture(ctx: DoctorContext) -> None:
    poppy_dir = ctx.poppy_dir

    # Resolve auto-capture consent + status once. The per-client "capture
    # liveness" lines below only treat a missing capture as a WARN when the user
    # has actually granted consent, otherwise "no capture yet" is the expected
    # fresh-install default, not a fault. The auto-capture summary line at the
    # end reuses these.
    from poppy.capture import health as _capture_health
    from poppy.capture.policy import ENABLED_STATUSES
    from poppy.capture.policy import evaluate as _evaluate_capture
    from poppy.config import load_config as _load_capture_config

    capture_cfg = _load_capture_config(poppy_dir)
    backend_health = _capture_health.load(poppy_dir)
    cap_status = _evaluate_capture(capture_cfg, backend_broken=backend_health.failing)
    # "Is capture actually expected to run right now", the effective state, not
    # just stored consent. Covers the POPPY_CONSOLIDATE=1 forced-on case (consent
    # may still read pending) so per-client liveness flags a truly dead capture
    # instead of hiding behind "consent pending".
    capture_expected = cap_status in ENABLED_STATUSES
    ctx.capture = _CaptureContext(capture_cfg, backend_health, cap_status, capture_expected)


def check_claude_code(ctx: DoctorContext) -> None:
    from poppy.setup.claude_code import is_hook_installed, is_mcp_installed, managed_claude_md_present

    line = ctx.line
    claude_dir = ctx.daemon.claude_dir

    # Per-client checks run only for clients Poppy is actually set up in, a
    # Cursor-only user must not see Claude Code WARNs they never chose.
    # The gate is Poppy's OWN footprint (MCP entry, a hook, or the CLAUDE.md
    # block), NOT the mere existence of ~/.claude or ~/.claude.json, those exist
    # for every Claude Code user whether or not Poppy was ever installed there.
    # Once any footprint is present, a half-installed setup (e.g. MCP but no
    # hooks) still surfaces the missing pieces as WARNs.
    claude_events = ("SessionStart", "UserPromptSubmit", "PreToolUse", "SessionEnd", "PostCompact")
    claude_mcp_installed = safe_read(lambda: is_mcp_installed(claude_dir), False)
    claude_md_present = safe_read(lambda: managed_claude_md_present(claude_dir), False)
    claude_poppy_present = (
        claude_mcp_installed
        or safe_read(lambda: any(is_hook_installed(claude_dir, event) for event in claude_events), False)
        or claude_md_present
    )
    if claude_poppy_present:
        line("claude config dir", "OK" if claude_dir.exists() else "WARN", str(claude_dir))
        # A daemon-mode registration is only healthy if the daemon answers on the
        # port this client points at AND accepts this client's token; otherwise
        # the "poppy" entry points at a dead server.
        claude_daemon_problem = _client_daemon_problem(ctx, "claude-code") if claude_mcp_installed else None
        if claude_daemon_problem:
            line("MCP server registered", "WARN", claude_daemon_problem[0], hint=claude_daemon_problem[1])
        else:
            line(
                "MCP server registered",
                "OK" if claude_mcp_installed else "WARN",
                hint="run `poppy setup claude-code` to install",
            )
        for event in claude_events:
            line(
                f"{event} hook",
                "OK" if safe_read(lambda e=event: is_hook_installed(claude_dir, e), False) else "WARN",
                hint="run `poppy setup claude-code` to install",
            )
        line(
            "CLAUDE.md block",
            "OK" if claude_md_present else "WARN",
            hint="run `poppy setup claude-code --claude-md` to install",
        )


def check_claude_desktop(ctx: DoctorContext) -> None:
    from poppy.setup.claude_code import get_claude_desktop_config_path, get_claude_desktop_msix_config_path

    line = ctx.line

    # Claude Desktop is MCP-only (no hooks/primer) and, on Windows, has TWO
    # configs: the standard one and the MSIX-virtualized one, which the packaged
    # app actually reads. `poppy setup claude-desktop` writes both, so a user who
    # is a Poppy Claude-Desktop user (registered in EITHER) must have BOTH checked
    # independently, otherwise a broken MSIX registration hides behind a healthy
    # standard config. If Poppy is in neither, the user didn't
    # choose Claude Desktop → stay silent.
    desktop_path = get_claude_desktop_config_path()
    desktop_settings = _read_json_or_empty(desktop_path) if desktop_path.exists() else {}
    desktop_registered = _poppy_registered(desktop_settings)

    msix_desktop_path = get_claude_desktop_msix_config_path()
    # A non-None path means the MSIX Claude app IS installed (its roaming dir was
    # found); the config file inside may still be missing/unreadable.
    msix_present = msix_desktop_path is not None
    msix_exists = msix_present and msix_desktop_path.exists()
    msix_settings = _read_json_or_empty(msix_desktop_path) if msix_exists else {}
    msix_registered = _poppy_registered(msix_settings)

    if desktop_registered or msix_registered:
        # The standard-config line ALWAYS prints once the user is a Poppy Claude
        # Desktop user (registered in standard OR MSIX), don't swallow it when
        # only MSIX registered and the standard file is absent.
        desktop_problem = _client_daemon_problem(ctx, "claude-desktop") if desktop_registered else None
        if desktop_registered and desktop_problem:
            line("Claude desktop MCP", "WARN", f"{desktop_path}: {desktop_problem[0]}", hint=desktop_problem[1])
        elif desktop_registered:
            line("Claude desktop MCP", "OK", str(desktop_path))
        elif desktop_path.exists():
            line(
                "Claude desktop MCP",
                "WARN",
                f"{desktop_path}: Poppy is set up for Claude Desktop but its MCP entry is missing here",
                hint="run `poppy setup claude-desktop` to re-register",
            )
        else:
            line(
                "Claude desktop MCP",
                "WARN",
                f"{desktop_path}: Poppy is not registered in the standard Claude Desktop config",
                hint="run `poppy setup claude-desktop` to register it",
            )
        if msix_present:
            # The MSIX-virtualized config is what the packaged Windows app reads,
            # so it is checked INDEPENDENTLY of the standard config: a missing,
            # unregistered, or dead-daemon MSIX registration must not hide behind
            # a healthy standard config.
            msix_problem = _client_daemon_problem(ctx, "claude-desktop-msix") if msix_registered else None
            if not msix_exists:
                line(
                    "Claude desktop MSIX MCP",
                    "WARN",
                    f"{msix_desktop_path}: MSIX config missing; the packaged app cannot reach Poppy",
                    hint="run `poppy setup claude-desktop` to re-create the MSIX config",
                )
            elif not msix_registered:
                line(
                    "Claude desktop MSIX MCP",
                    "WARN",
                    str(msix_desktop_path),
                    hint="run `poppy setup claude-desktop` to re-register the MSIX config",
                )
            elif msix_problem:
                line("Claude desktop MSIX MCP", "WARN", f"{msix_desktop_path}: {msix_problem[0]}", hint=msix_problem[1])
            else:
                line("Claude desktop MSIX MCP", "OK", str(msix_desktop_path))


def check_client_mcp(ctx: DoctorContext) -> None:
    from poppy.setup.claude_code import is_mcp_installed

    line = ctx.line
    claude_dir = ctx.daemon.claude_dir

    # Other MCP clients. Shown only when Poppy has a footprint in the client
    # (MCP entry or its hooks/primer), not merely that the client's own config
    # file exists, which is true for every user of that tool. A
    # footprinted client whose MCP entry is missing (e.g. deleted) WARNs; the
    # client can't reach Poppy. A daemon-URL registration is healthy only when
    # its specific port answers.
    from poppy.setup.claude_code import _client_settings_path

    for client_id, label, install_cmd in (
        ("cursor", "Cursor MCP", "poppy setup cursor"),
        ("windsurf", "Windsurf MCP", "poppy setup windsurf"),
        ("codex", "Codex MCP", "poppy setup codex"),
        ("copilot-cli", "Copilot CLI MCP", "poppy setup copilot-cli"),
        ("pi", "Pi MCP", "poppy setup pi"),
    ):
        if not _client_has_poppy(client_id):
            continue
        path = _client_settings_path(client_id, claude_dir)
        # Distinguish "MCP entry missing" from "config present but unreadable":
        # a footprinted client (hooks present) with an unreadable MCP config must
        # WARN that Poppy's registration can't be verified, not be treated as a
        # clean re-register.
        mcp_read_failed = False
        registered = False
        try:
            registered = is_mcp_installed(client=client_id)
        except (OSError, ValueError):
            mcp_read_failed = True
        # Codex's TOML reader swallows OSError, so an unreadable config surfaces as
        # registered=False (not an exception). Confirm the existing file is really
        # readable; if not, it's "present but unreadable", not a missing entry, so
        # the fix is permissions/encoding, not `poppy setup`.
        if not registered and not mcp_read_failed and path.exists():
            try:
                path.read_text()
            except (OSError, ValueError):
                mcp_read_failed = True
        daemon_problem = _client_daemon_problem(ctx, client_id) if registered else None
        if daemon_problem:
            line(label, "WARN", f"{path}: {daemon_problem[0]}", hint=daemon_problem[1])
        elif registered:
            line(label, "OK", str(path))
        elif mcp_read_failed:
            line(
                label,
                "WARN",
                f"{path}: config present but unreadable; can't verify Poppy's MCP registration",
                hint="fix the file's permissions/encoding, then re-run `poppy doctor`",
            )
        else:
            # Footprint present (hooks/primer) but the MCP entry is gone, the
            # client can't reach Poppy's tools until it's re-registered.
            line(
                label,
                "WARN",
                f"{path}: Poppy is set up here but its MCP entry is missing",
                hint=f"run `{install_cmd}` to re-register the MCP server",
            )


def check_cursor(ctx: DoctorContext) -> None:
    from poppy.setup.claude_code import get_cursor_home, is_cursor_hooks_installed

    line = ctx.line
    poppy_dir = ctx.poppy_dir
    capture_expected = ctx.capture.expected

    # Cursor rejects the whole file if any hook event name is unknown. Preserve
    # user entries during setup, but make that otherwise silent failure visible.
    # Gated on Poppy being set up in Cursor (not on ~/.cursor merely existing,
    # which is true for every Cursor user), so a Cursor user who chose only
    # Claude Code sees no Cursor WARNs.
    cursor_home = get_cursor_home()
    if _client_has_poppy("cursor"):
        from poppy.capture import journal as _cursor_journal
        from poppy.setup.claude_code import cursor_unknown_hook_events

        cursor_hooks_path = cursor_home / "hooks.json"

        def cursor_line(label: str, status: str, detail: str = "", hint: str = "") -> None:
            line(label, status, detail, hint, separator=", ")

        cursor_hooks_ok = safe_read(is_cursor_hooks_installed, False)
        cursor_hooks_valid = False
        unknown_cursor_events: list[str] = []
        if cursor_hooks_path.exists():
            try:
                cursor_settings = json.loads(cursor_hooks_path.read_text())
                cursor_hooks = cursor_settings.get("hooks") if isinstance(cursor_settings, dict) else None
                cursor_hooks_valid = (
                    isinstance(cursor_settings, dict)
                    and cursor_settings.get("version") == 1
                    and isinstance(cursor_hooks, dict)
                )
                if isinstance(cursor_hooks, dict):
                    unknown_cursor_events = cursor_unknown_hook_events(cursor_hooks_path)
            except (OSError, ValueError):
                # OSError (unreadable) / ValueError (non-UTF-8 or malformed JSON).
                pass

        if not cursor_hooks_path.exists():
            cursor_line(
                "Cursor hooks.json",
                "WARN",
                f"not found at {cursor_hooks_path}",
                hint="run `poppy setup cursor --hooks` to install",
            )
        elif not cursor_hooks_valid:
            cursor_line(
                "Cursor hooks.json",
                "WARN",
                f"invalid JSON or native Cursor hook shape at {cursor_hooks_path}",
                hint="run `poppy setup cursor --hooks` to repair",
            )
        elif unknown_cursor_events:
            cursor_line(
                "Cursor hooks.json",
                "WARN",
                "Cursor will ignore the entire hooks file until these unrecognized event keys "
                f"are removed: {', '.join(unknown_cursor_events)}",
            )
        else:
            cursor_line(
                "Cursor hooks.json",
                "OK" if cursor_hooks_ok else "WARN",
                str(cursor_hooks_path),
                hint="run `poppy setup cursor --hooks` to install missing Poppy hooks",
            )

        cursor_last_capture = _cursor_journal.read_last(poppy_dir, source="cursor")
        if cursor_last_capture is not None:
            cursor_line(
                "Cursor capture liveness",
                "OK",
                f"last capture at {cursor_last_capture.ts}, {cursor_last_capture.count} stored",
            )
        elif not capture_expected:
            # Auto-capture isn't active (consent pending or off): nothing is
            # meant to be captured, so a missing capture is the expected default,
            # not a fault. Informational, never a WARN.
            cursor_line(
                "Cursor capture liveness",
                "INFO",
                "no capture yet (auto-capture not active)",
            )
        elif cursor_hooks_ok:
            cursor_line(
                "Cursor capture liveness",
                "WARN",
                "hooks installed but no capture observed",
                hint="the server-side account flag enable_execute_hook_exec may be off",
            )
        else:
            cursor_line(
                "Cursor capture liveness",
                "WARN",
                "no capture observed",
                hint="install valid Cursor hooks; the account flag enable_execute_hook_exec must also be on",
            )


def check_codex(ctx: DoctorContext) -> None:
    from poppy.setup.claude_code import get_codex_home, is_codex_hooks_installed

    line = ctx.line
    poppy_dir = ctx.poppy_dir
    capture_expected = ctx.capture.expected

    # Codex hooks are trust-gated by a command hash. Presence proves setup wrote
    # the expected lifecycle subset; a journal record proves Codex has actually
    # invoked capture after the user approved the hooks interactively. Gate on
    # Poppy being set up in Codex, NOT on ~/.codex/config.toml existing, which
    # holds `model = ...` for every Codex user and would warn a Cursor-only user
    # about Codex they never chose.
    codex_home = get_codex_home()
    if _client_has_poppy("codex"):
        from poppy.capture import journal as _codex_journal

        codex_hooks_path = codex_home / "hooks.json"
        codex_hooks_ok = safe_read(is_codex_hooks_installed, False)
        line(
            "Codex hooks.json",
            "OK" if codex_hooks_ok else "WARN",
            str(codex_hooks_path) if codex_hooks_path.exists() else f"not found at {codex_hooks_path}",
            hint="run `poppy setup codex --hooks` to install",
        )
        # Only Codex-host records prove Codex capture is alive; a Claude Code
        # capture must not make this line read OK while the Codex hooks are dead.
        codex_last_capture = _codex_journal.read_last(poppy_dir, source="codex")
        if codex_last_capture is not None:
            line(
                "Codex capture liveness",
                "OK",
                f"last capture at {codex_last_capture.ts}, {codex_last_capture.count} stored",
            )
        elif not capture_expected:
            # Auto-capture isn't active: no capture is expected, not a fault.
            line("Codex capture liveness", "INFO", "no capture yet (auto-capture not active)")
        elif codex_hooks_ok:
            line(
                "Codex capture liveness",
                "WARN",
                "hooks installed but no capture observed",
                hint="likely untrusted; run Codex in the TUI once and approve the Poppy hooks",
            )
        else:
            line("Codex capture liveness", "WARN", "no capture observed", hint="install and trust the Codex hooks")


def check_primers(ctx: DoctorContext) -> None:
    from poppy.setup.claude_code import _client_primer_path, managed_primer_present

    line = ctx.line

    # Global primer presence for Codex / Copilot CLI / Pi. Shown only when Poppy
    # is set up in the client (footprint), so a bare ~/.codex/config.toml on a
    # Cursor-only machine produces no primer WARN.
    for client_id, label, install_cmd in (
        ("codex", "Codex primer", "poppy setup codex"),
        ("copilot-cli", "Copilot CLI primer", "poppy setup copilot-cli"),
        ("pi", "Pi primer", "poppy setup pi"),
    ):
        primer = _client_primer_path(client_id)
        if primer is None or not _client_has_poppy(client_id):
            continue
        primer_present = safe_read(lambda p=primer: managed_primer_present(p), False)
        line(
            label,
            "OK" if primer_present else "WARN",
            str(primer),
            hint=f"run `{install_cmd}` to install" if not primer_present else "",
        )


def check_goose(ctx: DoctorContext) -> None:
    from poppy.setup.claude_code import managed_primer_present

    line = ctx.line

    # Goose (Block), YAML-config MCP client. Shown only when Poppy is set up in
    # Goose (its MCP entry or the managed .goosehints primer), not when the goose
    # config dir merely exists, that dir is present for every Goose user.
    from poppy.setup.goose import get_goose_config_dir, is_goose_installed

    goose_dir = get_goose_config_dir()
    goose_hints = goose_dir / ".goosehints"
    goose_installed = safe_read(lambda: is_goose_installed(goose_dir), False)
    goose_primer_present = safe_read(lambda: managed_primer_present(goose_hints), False)
    if goose_installed or goose_primer_present:
        line(
            "Goose MCP",
            "OK" if goose_installed else "WARN",
            str(goose_dir / "config.yaml"),
            hint="run `poppy setup goose` to install" if not goose_installed else "",
        )
        line(
            "Goose primer",
            "OK" if goose_primer_present else "WARN",
            str(goose_hints),
            hint="run `poppy setup goose` to install" if not goose_primer_present else "",
        )


def check_hermes(ctx: DoctorContext) -> None:
    from poppy.setup.claude_code import managed_primer_present

    line = ctx.line

    # Hermes Agent (Nous Research), plugin-based, not MCP. Show status only when
    # Poppy is set up in Hermes (its plugin or the managed SOUL.md guidance), not
    # when ~/.hermes merely exists, that dir is present for every Hermes user.
    from poppy.setup.hermes import (
        HERMES_PLUGIN_NAME,
        get_hermes_home,
        is_hermes_installed,
        is_hermes_plugin_present,
        is_hermes_provider_configured,
    )

    _ = HERMES_PLUGIN_NAME  # pin against ruff auto-strip
    hermes_home = get_hermes_home()
    hermes_primer = hermes_home / "SOUL.md"
    hermes_installed = safe_read(lambda: is_hermes_installed(hermes_home), False)
    hermes_primer_present = safe_read(lambda: managed_primer_present(hermes_primer), False)
    # Footprint = ANY Poppy component `setup hermes-agent` writes: the plugin
    # files on disk, the config.yaml `provider: poppy` setting, OR the managed
    # SOUL.md guidance. Any one present ⇒ footprinted, so a partial install
    # (config-only, plugin-only, ...) still surfaces its missing pieces.
    if _any_present(
        lambda: is_hermes_plugin_present(hermes_home),
        lambda: is_hermes_provider_configured(hermes_home),
        lambda: hermes_primer_present,
    ):
        plugin_dir = hermes_home / "plugins" / "poppy"
        line(
            "Hermes Agent plugin",
            "OK" if hermes_installed else "WARN",
            str(plugin_dir),
            hint="run `poppy setup hermes-agent` to install" if not hermes_installed else "",
        )
        line(
            "Hermes SOUL.md guidance",
            "OK" if hermes_primer_present else "WARN",
            str(hermes_primer),
            hint="run `poppy setup hermes-agent` to install SOUL.md guidance" if not hermes_primer_present else "",
        )


def check_pi_adapter(ctx: DoctorContext) -> None:
    line = ctx.line

    # Pi requires the pi-mcp-adapter extension to consume the MCP config. Shown
    # only when Poppy is set up in Pi, not when ~/.pi/agent merely exists.
    from poppy.setup.claude_code import _pi_agent_dir

    pi_settings_path = _pi_agent_dir() / "settings.json"
    if pi_settings_path.exists() and _client_has_poppy("pi"):
        try:
            pi_settings = json.loads(pi_settings_path.read_text())
        except (OSError, ValueError):
            pi_settings = {}
        if not isinstance(pi_settings, dict):
            pi_settings = {}
        packages = pi_settings.get("packages", [])
        adapter_installed = isinstance(packages, list) and any(
            isinstance(p, str) and "pi-mcp-adapter" in p for p in packages
        )
        line(
            "Pi MCP adapter",
            "OK" if adapter_installed else "WARN",
            hint="run `pi install npm:pi-mcp-adapter` to bridge MCP servers into Pi" if not adapter_installed else "",
        )


def check_auto_capture(ctx: DoctorContext) -> None:
    line = ctx.line
    capture_cfg = ctx.capture.config
    backend_health = ctx.capture.health
    cap_status = ctx.capture.status

    # Auto-capture status: report the granular consent + backend
    # state from the ConsolidationPolicy, not a bare enabled/disabled bool, so
    # "consent pending" and the silent-breakage cases (remote-only, no backend)
    # are distinguishable and each hint points at the real fix. Reuses the
    # cap_status resolved once above.
    from poppy.capture.policy import CaptureStatus, status_message

    capture_doctor = {
        CaptureStatus.ACTIVE: ("OK", ""),
        CaptureStatus.FORCED_ENV: ("OK", ""),
        # Consent pending is the default state of a fresh install, surfaced by
        # the SessionStart notice and `poppy consent`; it is not a problem the
        # doctor should flag as a WARN. Informational; the nudge is already in
        # the status message below, so no separate hint here.
        CaptureStatus.INERT_PENDING: ("INFO", ""),
        # A deliberate off-state is not a problem the doctor should flag.
        CaptureStatus.DISABLED_OPT_OUT: ("OK", ""),
        CaptureStatus.DISABLED_ENV: ("OK", ""),
        CaptureStatus.DISABLED_PROJECT: ("OK", ""),
        # The status message below already spells out the same "install a host
        # CLI" instruction, so no separate hint here.
        CaptureStatus.WARN_REMOTE_ONLY: ("WARN", ""),
        # A host CLI on PATH that fails every extraction: the counter
        # and the last stderr tail come from the recorded backend health.
        CaptureStatus.WARN_BACKEND_BROKEN: (
            "WARN",
            "check the configured endpoint, API key and model"
            if backend_health.cli == "openai-compat"
            else f"check that `{backend_health.cli or 'your host CLI'}` runs and is logged in",
        ),
        CaptureStatus.DISABLED_NO_BACKEND: (
            "WARN",
            "install a supported host CLI (claude/cursor-agent/codex/gemini) for free local capture",
        ),
    }
    status_kind, status_hint = capture_doctor.get(cap_status, ("WARN", ""))
    status_detail = status_message(cap_status)
    if cap_status is CaptureStatus.WARN_BACKEND_BROKEN:
        # The tail is the host CLI's own stderr, and an auth failure can echo back
        # the credential it rejected. Doctor output is what people paste into bug
        # reports, so it goes through the capture redaction pass first.
        from poppy.capture.redaction import load_custom_redaction, redact_secrets

        custom_patterns, _redaction_issues = load_custom_redaction(capture_cfg)
        safe_tail = redact_secrets(backend_health.last_error, custom_patterns)
        status_detail = (
            f"{status_detail} Last error: {backend_health.cli} "
            f"{safe_tail} ({backend_health.consecutive_failures} in a row)."
        )
    line("auto-capture", status_kind, status_detail, hint=status_hint)


def check_remote_capture(ctx: DoctorContext) -> None:
    from poppy.capture import health as _capture_health

    line = ctx.line
    capture_cfg = ctx.capture.config
    backend_health = ctx.capture.health

    remote_health = backend_health.clis.get("openai-compat")
    if remote_health is not None:
        # Surface a configured fallback's failure even without a host CLI or
        # before repeated failures change the overall capture status. One
        # failure can be a dropped connection, so it reads as information and
        # only a run of them warns, matching how a host CLI is reported.
        from poppy.capture.redaction import load_custom_redaction, redact_secrets

        custom_patterns, _redaction_issues = load_custom_redaction(capture_cfg)
        failures = remote_health.consecutive_failures
        broken = failures >= _capture_health.FAILURE_THRESHOLD
        line(
            "openai-compat",
            "WARN" if broken else "INFO",
            f"{redact_secrets(remote_health.last_error, custom_patterns)} ({failures} in a row)",
            hint=(
                "check the configured endpoint, API key and model; retry extraction after fixing the error"
                if broken
                else ""
            ),
        )


def check_last_capture(ctx: DoctorContext) -> None:
    from poppy.capture import journal as _journal
    from poppy.cli import main as _main

    line = ctx.line

    # Last-capture freshness: proof the loop has actually run. Reads the
    # local capture journal; informational, never a failure. Includes the total
    # journal record count so growth over time is visible.
    journal_count = len(_journal.read_all(_main._get_poppy_dir()))
    last_capture = _journal.read_last(_main._get_poppy_dir())
    if last_capture is not None:
        scope = last_capture.project or "all projects"
        line(
            "last capture",
            "OK",
            f"{last_capture.count} stored for {scope} at {last_capture.ts} ({journal_count} journaled total)",
        )
    else:
        line("last capture", "OK", "no captures recorded yet")


def check_capture_state(ctx: DoctorContext) -> None:
    from poppy.cli import main as _main

    line = ctx.line

    # Capture watermark/lock state: per-session progress + captures in flight.
    # Lock files persist between runs; only a held one means a worker is running,
    # and a crashed worker's lock is released by the OS, so none can go stale.
    from poppy.capture import _state as _capture_state
    from poppy.capture.lock import is_held

    tracked_sessions = len(_capture_state.load(_main._get_poppy_dir()))
    running = sum(1 for lock_path in _main._get_poppy_dir().glob("capture-*.flock") if is_held(lock_path))
    in_flight = f", {running} capture(s) in flight" if running else ""
    line("capture state", "OK", f"{tracked_sessions} session(s) tracked{in_flight}")


def run_doctor() -> None:
    """Run the installation checks in their original reporting order."""
    ctx = DoctorContext()
    check_executable(ctx)
    check_storage(ctx)
    check_trags_key(ctx)
    check_custom_redaction(ctx)
    check_daemon(ctx)
    check_version(ctx)
    check_encryption(ctx)
    check_engine(ctx)
    _prepare_capture(ctx)
    check_claude_code(ctx)
    check_claude_desktop(ctx)
    check_client_mcp(ctx)
    check_cursor(ctx)
    check_codex(ctx)
    check_primers(ctx)
    check_goose(ctx)
    check_hermes(ctx)
    check_pi_adapter(ctx)
    check_auto_capture(ctx)
    check_remote_capture(ctx)
    check_last_capture(ctx)
    check_capture_state(ctx)

    if not ctx.ok:
        raise SystemExit(1)
