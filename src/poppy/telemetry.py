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
import os
import platform
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

# Public project API key — safe to embed (PostHog `phc_*` keys are write-only).
_POSTHOG_KEY = "phc_sL4ZeJng3mDGinNXn8esUpzzVazeomJofqQC6SF52R4F"
_POSTHOG_HOST = "https://eu.i.posthog.com"
_FLUSH_TIMEOUT_S = 1.0

_lock = threading.Lock()
_state: dict[str, Any] = {"client": None, "registered_atexit": False}


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
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


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
    """Return (enabled, reason). Precedence: env var, config flag, legacy file, default on."""
    if os.environ.get("POPPY_TELEMETRY_OFF") == "1":
        return False, "POPPY_TELEMETRY_OFF=1 set in the environment"
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

        if _state["client"] is None:
            try:
                from posthog import Posthog  # local import: keep import cost off the cold path
            except ImportError:
                return None
            try:
                _state["client"] = Posthog(_POSTHOG_KEY, host=_POSTHOG_HOST)
            except Exception:
                return None
            if not _state["registered_atexit"]:
                atexit.register(_shutdown)
                _state["registered_atexit"] = True

        return _state["client"], data["device_id"], is_first_run


def _shutdown() -> None:
    client = _state.get("client")
    if client is None:
        return
    try:
        client.shutdown(timeout=_FLUSH_TIMEOUT_S)
    except Exception:
        pass


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
