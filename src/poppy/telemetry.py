"""anonymous CLI telemetry via PostHog.

Fire-and-forget event capture. Never blocks CLI exit, never raises, never
surfaces failures to the user. Off means truly off — no network calls.

Enablement precedence (first match wins):
1. ``POPPY_TELEMETRY_OFF=1`` in the environment → off, always.
2. ``telemetry_enabled`` in ``~/.poppy/config.json`` (set via
   ``poppy telemetry on|off``) → that value.
3. Legacy ``"telemetry": "off"`` in ``~/.poppy/analytics.json`` → off.
4. Default → on.

Design rules:
- One `~/.poppy/analytics.json` per machine: stores `device_id`, a
  `created_at`, and the one-time first-run-notice flag. Generated on first
  invocation.
- Events use `device_id` as the PostHog `distinct_id` until the server
  aliases it to a user_id during `poppy setup trags`.
- Event properties never contain memory content, recall query text, or
  project names — counts, lengths, and fixed enum values only.
- The PostHog Python SDK has its own background consumer; we register
  `atexit` to give it 1 s to flush on process exit and then move on.
"""

from __future__ import annotations

import atexit
import datetime
import json
import logging
import os
import platform
import sys
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from poppy.paths import ensure_poppy_dir, write_text_atomic

# Public project API key — safe to embed (PostHog `phc_*` keys are write-only).
_POSTHOG_KEY = "phc_sL4ZeJng3mDGinNXn8esUpzzVazeomJofqQC6SF52R4F"
_POSTHOG_HOST = "https://eu.i.posthog.com"
# Internal knob for CI/tests only: point the client at a loopback sink so a
# telemetry-ON smoke exercises the real SDK flush path without emitting
# production events. A value that is not a loopback URL disables telemetry
# for the process (fail closed): a mistyped override must never fall back to
# sending real events, and an inherited one can never redirect them elsewhere.
_HOST_ENV = "POPPY_TELEMETRY_HOST"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})
# Wall-clock bound on the exit flush, on every SDK version: shutdown() runs on
# a daemon thread, the process moves on after this many seconds, and the SDK's
# own (unbounded) atexit hook is unregistered at client creation so nothing
# else can hold the exit.
_FLUSH_TIMEOUT_S = 1.0
# One delivery attempt, bounded: telemetry is fire-and-forget, and the SDK's
# default (3 retries with backoff, 15s request timeout) would hold an offline
# user's CLI at exit for seconds per command. flush_interval is the consumer's
# collect window; shutdown() waits it out twice, so the default 0.5s cost every
# healthy-network exit a full second.
_REQUEST_TIMEOUT_S = 2.0
_CLIENT_KWARGS = {"max_retries": 0, "timeout": _REQUEST_TIMEOUT_S, "flush_interval": 0.1}

# Contract: every telemetry emit call site must use a name from this set, and
# the telemetry event table in README.md must list exactly this set.
DOCUMENTED_EVENTS: frozenset[str] = frozenset(
    {
        "agent_setup",
        "cli_install",
        "closed_loop",
        "consent_granted",
        "consent_pending_shown",
        "first_autocapture_stored",
        "memory_write",
        "recall_call",
        "setup_completed",
    }
)

_lock = threading.Lock()
_state: dict[str, Any] = {"client": None, "host": None, "registered_atexit": False}


def _analytics_path(poppy_dir: Path) -> Path:
    return poppy_dir / "analytics.json"


def _load(poppy_dir: Path) -> dict:
    p = _analytics_path(poppy_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save(poppy_dir: Path, data: dict) -> None:
    p = _analytics_path(poppy_dir)
    ensure_poppy_dir(p.parent)
    # Atomic: a torn file reads as empty, which would mint a new device id and
    # replay the first-run notice and milestone events.
    write_text_atomic(p, json.dumps(data, indent=2))


@contextmanager
def _analytics_file_lock(poppy_dir: Path):
    """Serialize analytics.json updates across processes when fcntl is available."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows fallback
        yield
        return

    lock_path = _analytics_path(poppy_dir).with_suffix(".json.lock")
    ensure_poppy_dir(lock_path.parent)
    with lock_path.open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def get_device_id(poppy_dir: Path) -> str | None:
    """Return the stable device_id, or None if telemetry is off / unsupported."""
    if not is_enabled(poppy_dir):
        return None
    return _load(poppy_dir).get("device_id")


def _config_flag(poppy_dir: Path) -> bool | None:
    """Read `telemetry_enabled` from config.json. None when unset or unreadable."""
    try:
        from poppy.config import load_config  # local import: config is the heavier module

        return load_config(poppy_dir).telemetry_enabled
    except Exception:
        return None


def status(poppy_dir: Path) -> tuple[bool, str]:
    """Return (enabled, reason). Precedence: env var, host override, config flag, legacy file, default on."""
    if os.environ.get("POPPY_TELEMETRY_OFF") == "1":
        return False, "POPPY_TELEMETRY_OFF=1 set in the environment"
    # A set-but-unusable host override disables telemetry outright, so status,
    # the first-run notice and the capture path all agree it is off rather than
    # printing "on" while every event is silently dropped.
    if os.environ.get(_HOST_ENV) is not None and _host() is None:
        return False, f"{_HOST_ENV} is set but not a usable loopback URL"
    flag = _config_flag(poppy_dir)
    if flag is not None:
        return flag, f"set in {poppy_dir / 'config.json'}"
    if _load(poppy_dir).get("telemetry") == "off":
        return False, f"legacy opt-out in {_analytics_path(poppy_dir)}"
    return True, "default"


def is_enabled(poppy_dir: Path) -> bool:
    enabled, _ = status(poppy_dir)
    return enabled


def set_enabled(poppy_dir: Path, enabled: bool) -> None:
    """Persist the telemetry choice.

    Writes the first-class `telemetry_enabled` flag to config.json and mirrors
    it into analytics.json for older readers. An explicit choice also counts
    as having seen the first-run notice, so the disclosure never fires after
    the user has already engaged with the switch.
    """
    from poppy.config import load_config, save_config

    cfg = load_config(poppy_dir)
    cfg.telemetry_enabled = enabled
    save_config(cfg)

    data = _load(poppy_dir)
    data["telemetry"] = "on" if enabled else "off"
    data[_NOTICE_KEY] = True
    if "device_id" not in data and enabled:
        data["device_id"] = str(uuid.uuid4())
        data["created_at"] = datetime.datetime.now(datetime.UTC).isoformat()
    _save(poppy_dir, data)


_NOTICE_KEY = "first_run_notice_shown"
_FIRST_RUN_NOTICE = (
    "poppy: anonymous usage telemetry is on (no memory content is ever sent). Disable: poppy telemetry off"
)


def maybe_print_first_run_notice(poppy_dir: Path) -> None:
    """Print the telemetry disclosure to stderr, exactly once ever.

    Never fires when telemetry is off, never prints to stdout, and never
    raises — scripted use (hooks, MCP stdio, pipelines) must not break on a
    notice. The seen-flag persists in analytics.json before printing.
    """
    try:
        if not is_enabled(poppy_dir):
            return
        data = _load(poppy_dir)
        if data.get(_NOTICE_KEY):
            return
        data[_NOTICE_KEY] = True
        _save(poppy_dir, data)
        print(_FIRST_RUN_NOTICE, file=sys.stderr)
    except Exception:
        pass


def _ensure_client(poppy_dir: Path) -> tuple[Any, str, bool] | None:
    """Return (client, device_id, is_first_run) or None when telemetry is off."""
    if not is_enabled(poppy_dir):
        return None

    # Resolve the destination before touching analytics.json. A rejected host
    # override is a full no-op, so it never consumes the first-run state: once
    # the override is removed, the very first delivery still emits cli_install.
    host = _host()
    if host is None:
        return None

    with _lock:
        data = _load(poppy_dir)
        is_first_run = "device_id" not in data
        if is_first_run:
            # Merge, don't replace: analytics.json may already hold the
            # first-run-notice flag from maybe_print_first_run_notice().
            data["device_id"] = str(uuid.uuid4())
            data.setdefault("telemetry", "on")
            data["created_at"] = datetime.datetime.now(datetime.UTC).isoformat()
            _save(poppy_dir, data)

        # Rebuild if the resolved host changed under a cached client (the override
        # was set or cleared mid-process), so a later capture cannot keep hitting
        # the previously resolved host and defeat the loopback control.
        if _state["client"] is not None and _state.get("host") != host:
            _state["client"] = None

        if _state["client"] is None:
            try:
                from posthog import Posthog  # local import: keep import cost off the cold path
            except ImportError:
                return None
            try:
                _silence_sdk_logging()
                client = Posthog(_POSTHOG_KEY, host=host, **_CLIENT_KWARGS)
                _disown_sdk_exit_hook(client)
                if host != _POSTHOG_HOST:
                    _forbid_override_redirects()
                _state["client"] = client
                _state["host"] = host
            except Exception:
                return None
            if not _state["registered_atexit"]:
                atexit.register(_shutdown)
                _state["registered_atexit"] = True

        return _state["client"], data["device_id"], is_first_run


def _host() -> str | None:
    """Production host; a strict loopback override for tests and CI; None (off) for anything else.

    The override must be a bare ``http(s)://<loopback>[:port]`` with no userinfo
    and no backslash. Both guards close a parser-disagreement bypass: a value
    like ``http://evil.example\\@127.0.0.1`` makes ``urlsplit().hostname`` read
    ``127.0.0.1`` while requests would send the payload to ``evil.example``.
    The variable being unset means "no override" (production);
    the variable being present but unusable -- empty, non-loopback, malformed --
    turns telemetry off rather than falling back to production, so a hostile or
    fat-fingered value (including ``POPPY_TELEMETRY_HOST="${SINK:-}"`` with an
    empty ``SINK``) can never redirect events or reach the real project.
    """
    override = os.environ.get(_HOST_ENV)
    if override is None:
        return _POSTHOG_HOST
    if not override or "\\" in override:
        return None
    try:
        parsed = urlsplit(override)
        _ = parsed.port  # raises ValueError for an out-of-range port (e.g. :99999)
    except ValueError:
        return None
    if parsed.scheme not in ("http", "https") or parsed.username or parsed.password:
        return None
    if parsed.hostname in _LOOPBACK_HOSTS:
        return override
    return None


def _silence_sdk_logging() -> None:
    """Keep SDK delivery errors off the user's terminal: telemetry is silent on all failures.

    The SDK logs ``error uploading`` at ERROR level whenever the user is
    offline. With no handler anywhere, Python's last-resort handler prints that
    to stderr; with a root handler configured (any ``logging.basicConfig`` in a
    poppy process), propagation would print it too. Poppy is the application
    that owns these processes and the events are fire-and-forget, so the
    records are never actionable: a NullHandler stops the last-resort print and
    ``propagate = False`` keeps them out of any root handler.
    """
    log = logging.getLogger("posthog")
    if not any(isinstance(h, logging.NullHandler) for h in log.handlers):
        log.addHandler(logging.NullHandler())
    log.propagate = False


def _disown_sdk_exit_hook(client: Any) -> None:
    """Remove the SDK's own atexit hook so :func:`_shutdown` is the only exit flush.

    The SDK registers ``join`` (6.x, 7.0 to 7.4x) or ``_atexit`` (later 7.x) at
    construction. ``join`` waits for the in-flight request with no deadline of
    its own, and the request timeout does not cover DNS resolution, so a stalled
    resolver could hold a CLI exit indefinitely after our bounded flush had
    already returned. The consumer threads are daemons; whatever is still in
    flight when the process exits is dropped, which is the right trade for
    telemetry.
    """
    for name in ("join", "_atexit"):
        hook = getattr(client, name, None)
        if callable(hook):
            try:
                atexit.unregister(hook)
            except Exception:
                pass


def _confine_override_session() -> None:
    """Pin the SDK's HTTP sessions to loopback: no proxy, no redirect.

    ``trust_env = False`` makes requests ignore ``HTTP(S)_PROXY``/``NO_PROXY``
    and netrc, and ``max_redirects = 0`` makes any redirect raise instead of
    being followed. With the override host a loopback IP literal (no DNS), a
    proxy that would route the batch to an external host and a sink that would
    302/307 it off-box are both closed, so the request physically cannot leave
    the machine.
    """
    from posthog import request as _pr  # local import: only when telemetry is on

    for attr in ("_session", "_flags_session"):
        session = getattr(_pr, attr, None)
        if session is not None:
            session.trust_env = False
            session.max_redirects = 0


def _forbid_override_redirects() -> None:
    """Stop the SDK's HTTP session following a redirect off the loopback sink.

    Only used when a ``POPPY_TELEMETRY_HOST`` override is active (tests/CI). URL
    validation cannot cover a live sink that answers a delivery POST with a
    302/307 to an external host, which requests would follow and re-POST the
    body to. Setting ``max_redirects = 0`` makes any redirect
    raise ``TooManyRedirects`` instead, so the payload never leaves loopback.

    The SDK rebuilds its sessions in a fork child (``os.register_at_fork``),
    which would restore the defaults and re-open the vectors, so a fork handler
    re-applies the confinement in the child. The hook is registered on every
    confined build, not once: each new ``Posthog(...)`` registers its own reset
    hook, and child hooks run in registration order, so poppy's must be
    registered after the most recent SDK one to win. Re-runs of
    an idempotent re-confine in the child are harmless. Best-effort and
    version-tolerant: a no-op if the SDK's request module or its sessions are not
    where we expect, and never applied to the production host.
    """
    try:
        _confine_override_session()
        os.register_at_fork(after_in_child=_reconfine_override_session_in_child)
    except Exception:
        pass


def _reconfine_override_session_in_child() -> None:
    """Re-confine the SDK sessions to loopback in a fork child."""
    if not _state.get("client"):
        return
    try:
        _confine_override_session()
    except Exception:
        pass


def _flush_quietly(client: Any) -> None:
    try:
        client.shutdown()
    except Exception:
        pass


def _shutdown() -> None:
    """Flush pending events at exit, within ``_FLUSH_TIMEOUT_S`` on every SDK version.

    posthog-python 6+ dropped the ``timeout`` parameter from ``Client.shutdown``
    and 7.x wraps the method in a decorator that logs a full traceback on any
    error, so a bad keyword surfaced as an 8-line crash after every command.
    ``shutdown()`` itself drains the whole queue with no deadline, so
    it runs on a daemon thread and the process moves on after the bound. The
    SDK's own atexit hook was unregistered at client creation, so a backed-up
    queue or a hanging resolver cannot hold the exit either.

    If the worker thread cannot start -- CPython 3.12.0-3.12.2 raise
    ``RuntimeError`` when a thread is created during interpreter shutdown -- the
    flush is skipped rather than run inline: an inline ``shutdown()`` has no
    wall-clock bound (DNS resolution is not covered by the request timeout) and
    could hold the exit, and the module's contract is that telemetry never
    blocks CLI exit. Dropping a best-effort event on those specific patch
    versions is the right trade.
    """
    client = _state.get("client")
    if client is None:
        return
    try:
        worker = threading.Thread(target=_flush_quietly, args=(client,), name="poppy-telemetry-flush", daemon=True)
        worker.start()
    except Exception:
        return  # never block exit; see the note above
    try:
        worker.join(_FLUSH_TIMEOUT_S)
    except Exception:
        pass


def capture_once(poppy_dir: Path, event: str, properties: dict | None = None) -> bool:
    """Emit a funnel-milestone event at most once per device. Returns True if emitted.

    Latches on a per-event flag in ``analytics.json`` (the ``milestones`` map) so a
    funnel stage — setup, consent, first auto-capture, closed loop —
    records exactly once and a per-session hook can call it every session without
    flooding telemetry. Content-free like :func:`capture`; never raises, and an
    off/erroring telemetry state is a silent no-op that returns ``False``.

    The latch is written before the emit (fire-and-forget, matching this module):
    a milestone marked seen but not delivered is preferable to re-emitting on every
    session when the network is down.
    """
    try:
        if not is_enabled(poppy_dir):
            return False
        with _lock:
            with _analytics_file_lock(poppy_dir):
                data = _load(poppy_dir)
                milestones = data.get("milestones")
                if not isinstance(milestones, dict):
                    milestones = {}
                if milestones.get(event):
                    return False
                milestones[event] = True
                data["milestones"] = milestones
                _save(poppy_dir, data)
    except Exception:
        return False
    # capture() re-acquires _lock and re-reads analytics.json, so it must run
    # outside the lock above — threading.Lock is not reentrant.
    capture(poppy_dir, event, properties)
    return True


def capture(poppy_dir: Path, event: str, properties: dict | None = None) -> None:
    """Best-effort event emit. Silent on all failures."""
    try:
        init = _ensure_client(poppy_dir)
        if init is None:
            return
        client, device_id, is_first_run = init

        if is_first_run:
            try:
                from poppy import __version__ as poppy_version
            except Exception:
                poppy_version = "unknown"
            client.capture(
                "cli_install",
                distinct_id=device_id,
                properties={
                    "version": poppy_version,
                    "python_version": platform.python_version(),
                    "platform": sys.platform,
                },
            )

        client.capture(event, distinct_id=device_id, properties=properties or {})
    except Exception:
        pass
