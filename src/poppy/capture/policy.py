"""ConsolidationPolicy — consent + default-on precedence (ADR-0002).

Auto-capture is **enabled by default but inert until a one-time consent is
recorded.** This module owns the single source of truth for "is capture on right
now, and if not, why" so the SessionEnd/PostCompact backstops, the mid-session
loop, `poppy doctor`, and the SessionStart notice all agree.

Precedence (ADR-0002):

    explicit env ON          -> FORCED_ENV    (operator override; implies consent)
    explicit env OFF         -> DISABLED_ENV
    consent not yet given    -> INERT_PENDING  (prompt/notice shown; nothing captured)
    explicit opt-out         -> DISABLED_OPT_OUT (persisted; survives a default change)
    consent + host-CLI       -> ACTIVE         (default-on, free backend)
    consent + remote-only    -> WARN_REMOTE_ONLY (do NOT auto-spend on a paid model)
    consent + no backend     -> DISABLED_NO_BACKEND

Consent is tri-state and lives in config (``consent``). Legacy installs are
migrated on load (``config.load_config``): an explicit ``consolidate-enabled
true`` is grandfathered as granted, an explicit ``false`` becomes an opt-out, and
unset becomes pending. Both consent and opt-out persist permanently, so a later
default change can never silently re-enable capture someone turned off.
"""

from __future__ import annotations

import os
import shutil
from enum import Enum

from poppy.config import PoppyConfig

# Host CLIs that can run extraction for free on the user's existing login.
_HOST_CLIS = ("claude", "codex", "gemini")

_ENV_VAR = "POPPY_CONSOLIDATE"
_ENV_ON = {"1", "true", "yes", "on"}
_ENV_OFF = {"0", "false", "no", "off"}


class Consent(str, Enum):
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"


class CaptureStatus(str, Enum):
    ACTIVE = "active"
    FORCED_ENV = "forced_env"
    INERT_PENDING = "inert_pending"
    DISABLED_OPT_OUT = "disabled_opt_out"
    DISABLED_ENV = "disabled_env"
    WARN_REMOTE_ONLY = "warn_remote_only"
    DISABLED_NO_BACKEND = "disabled_no_backend"


# Statuses under which capture actually runs.
ENABLED_STATUSES = frozenset({CaptureStatus.ACTIVE, CaptureStatus.FORCED_ENV})


def _env_override() -> str | None:
    """'on' / 'off' / None from the POPPY_CONSOLIDATE escape hatch."""
    raw = os.environ.get(_ENV_VAR, "").strip().lower()
    if raw in _ENV_ON:
        return "on"
    if raw in _ENV_OFF:
        return "off"
    return None


def host_cli_available() -> bool:
    """True if any free host-CLI backend is on PATH."""
    return any(shutil.which(c) for c in _HOST_CLIS)


def remote_backend_configured(cfg: PoppyConfig) -> bool:
    """True if a paid OpenAI-compatible backend is configured (model + key)."""
    model = os.environ.get("POPPY_CONSOLIDATE_MODEL") or cfg.consolidate_model
    api_key = os.environ.get("POPPY_CONSOLIDATE_API_KEY") or cfg.consolidate_api_key or os.environ.get("OPENAI_API_KEY")
    return bool(model and api_key)


def effective_consent(cfg: PoppyConfig) -> Consent:
    """Resolve consent, honouring the legacy ``consolidate_enabled`` bool.

    An explicit opt-out always wins. A directly-set legacy ``consolidate_enabled
    true`` (config not yet migrated) is treated as granted so grandfathered users
    are never re-prompted.
    """
    if cfg.consent == Consent.DENIED.value:
        return Consent.DENIED
    if cfg.consent == Consent.GRANTED.value:
        return Consent.GRANTED
    # pending (or unrecognised): grandfather a legacy explicit-true.
    if cfg.consolidate_enabled:
        return Consent.GRANTED
    return Consent.PENDING


def evaluate(
    cfg: PoppyConfig,
    *,
    host_cli: bool | None = None,
    remote: bool | None = None,
) -> CaptureStatus:
    """Resolve the current capture status from the full precedence (ADR-0002).

    ``host_cli`` / ``remote`` override backend detection for tests; ``None`` means
    auto-detect.
    """
    env = _env_override()
    if env == "on":
        return CaptureStatus.FORCED_ENV
    if env == "off":
        return CaptureStatus.DISABLED_ENV

    consent = effective_consent(cfg)
    if consent is Consent.PENDING:
        return CaptureStatus.INERT_PENDING
    if consent is Consent.DENIED:
        return CaptureStatus.DISABLED_OPT_OUT

    if host_cli is None:
        host_cli = host_cli_available()
    if host_cli:
        return CaptureStatus.ACTIVE
    if remote is None:
        remote = remote_backend_configured(cfg)
    if remote:
        return CaptureStatus.WARN_REMOTE_ONLY
    return CaptureStatus.DISABLED_NO_BACKEND


def is_capture_enabled(
    cfg: PoppyConfig,
    *,
    host_cli: bool | None = None,
    remote: bool | None = None,
) -> bool:
    """Whether auto-capture should run right now."""
    return evaluate(cfg, host_cli=host_cli, remote=remote) in ENABLED_STATUSES


_STATUS_MESSAGES = {
    CaptureStatus.ACTIVE: "Auto-capture is active (extracting locally via your host CLI).",
    CaptureStatus.FORCED_ENV: f"Auto-capture forced on by {_ENV_VAR}.",
    CaptureStatus.INERT_PENDING: (
        "Auto-capture is pending your consent; nothing is captured yet. Run `poppy consent --enable` to turn it on."
    ),
    CaptureStatus.DISABLED_OPT_OUT: "Auto-capture is off (you opted out). Run `poppy consent --enable` to re-enable.",
    CaptureStatus.DISABLED_ENV: f"Auto-capture disabled by {_ENV_VAR}.",
    CaptureStatus.WARN_REMOTE_ONLY: (
        "Auto-capture is INACTIVE: only a paid remote backend is configured, so capture will not "
        "auto-spend. Install a host CLI (claude/codex/gemini) for free local capture."
    ),
    CaptureStatus.DISABLED_NO_BACKEND: (
        "Auto-capture is INACTIVE: no extraction backend found. Install the Claude CLI for free local capture."
    ),
}


def status_message(status: CaptureStatus) -> str:
    return _STATUS_MESSAGES.get(status, str(status.value))
