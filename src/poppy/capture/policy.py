"""ConsolidationPolicy — the consent and default-on precedence rule.

Auto-capture is **enabled by default but inert until a one-time consent is
recorded.** This module owns the single source of truth for "is capture on right
now, and if not, why" so the SessionEnd/PostCompact backstops, the mid-session
loop, `poppy doctor`, and the SessionStart notice all agree.

Consent and default-on precedence:

    explicit env ON          -> FORCED_ENV    (operator override; implies consent)
    explicit env ON + failing CLI -> WARN_BACKEND_BROKEN (still forced on, but broken)
    explicit env OFF         -> DISABLED_ENV
    project on the deny-list -> DISABLED_PROJECT (per-project off switch)
    consent not yet given    -> INERT_PENDING  (prompt/notice shown; nothing captured)
    explicit opt-out         -> DISABLED_OPT_OUT (persisted; survives a default change)
    consent + failing CLI    -> WARN_BACKEND_BROKEN (host CLI present but every run fails)
    consent + host-CLI       -> ACTIVE         (default-on, free backend)
    consent + remote-only    -> WARN_REMOTE_ONLY (do NOT auto-spend on a paid model)
    consent + no backend     -> DISABLED_NO_BACKEND

The per-project deny-list is a scoping control layered on top of the
global consent, not per-project consent: consent stays one-time and global. It
sits just below the ``POPPY_CONSOLIDATE`` env escape hatch — the env override is
the documented operator control and stays the top precedence — and above the
consent branches, so a sensitive repo stays out of capture regardless of consent
state. An allowlist mode was explicitly rejected.

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

from poppy.capture.health import backend_failing
from poppy.config import PoppyConfig, resolved_consolidate_settings

# Host CLIs that can run extraction for free on the user's existing login.
_HOST_CLIS = ("claude", "cursor-agent", "codex", "gemini")

_ENV_VAR = "POPPY_CONSOLIDATE"
_ENV_ON = {"1", "true", "yes", "on"}
_ENV_OFF = {"0", "false", "no", "off"}


class Consent(str, Enum):
    # Consent evidence record (GDPR Art 7(1)); never rename with CLI surface changes.
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"


class CaptureStatus(str, Enum):
    ACTIVE = "active"
    FORCED_ENV = "forced_env"
    INERT_PENDING = "inert_pending"
    DISABLED_OPT_OUT = "disabled_opt_out"
    DISABLED_ENV = "disabled_env"
    DISABLED_PROJECT = "disabled_project"
    WARN_REMOTE_ONLY = "warn_remote_only"
    WARN_BACKEND_BROKEN = "warn_backend_broken"
    DISABLED_NO_BACKEND = "disabled_no_backend"


# Statuses under which capture actually runs. WARN_BACKEND_BROKEN is in the set on
# purpose: the host CLI is failing, not switched off, so capture must keep
# firing — that is what lets a re-login clear the failure record by itself.
ENABLED_STATUSES = frozenset({CaptureStatus.ACTIVE, CaptureStatus.FORCED_ENV, CaptureStatus.WARN_BACKEND_BROKEN})


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
    settings = resolved_consolidate_settings(cfg)
    return bool(settings.model and settings.api_key)


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
    project: str | None = None,
    host_cli: bool | None = None,
    remote: bool | None = None,
    backend_broken: bool | None = None,
) -> CaptureStatus:
    """Resolve capture status from the consent and default-on precedence rule.

    ``project`` is the current project name (from ``project_from_cwd``); when it is
    on the per-project deny-list, capture is off for this repo regardless of
    consent. ``host_cli`` / ``remote`` / ``backend_broken`` override backend
    detection for tests; ``None`` means auto-detect.

    A host CLI on PATH is presence, not health: when the recorded backend health
    says every recent extraction failed, the status is
    ``WARN_BACKEND_BROKEN`` instead of ``ACTIVE`` so the banner and doctor stop
    reporting a dead backend as healthy. That check runs under the forced-on
    override too: ``POPPY_CONSOLIDATE=1`` can force capture to run, but it cannot
    make a logged-out CLI extract anything, and reporting OK there would be the
    original bug behind an env var.
    """
    env = _env_override()
    if env == "on":
        # WARN_BACKEND_BROKEN is an enabled status, so capture still runs exactly
        # as the override asks — the operator just also gets told it is failing.
        if backend_broken is None:
            backend_broken = backend_failing()
        return CaptureStatus.WARN_BACKEND_BROKEN if backend_broken else CaptureStatus.FORCED_ENV
    if env == "off":
        return CaptureStatus.DISABLED_ENV

    # Per-project off switch: a scoping control above the consent
    # branches but below the env escape hatch.
    if cfg.is_project_disabled(project):
        return CaptureStatus.DISABLED_PROJECT

    consent = effective_consent(cfg)
    if consent is Consent.PENDING:
        return CaptureStatus.INERT_PENDING
    if consent is Consent.DENIED:
        return CaptureStatus.DISABLED_OPT_OUT

    if host_cli is None:
        host_cli = host_cli_available()
    if host_cli:
        if backend_broken is None:
            backend_broken = backend_failing()
        if backend_broken:
            return CaptureStatus.WARN_BACKEND_BROKEN
        return CaptureStatus.ACTIVE
    if remote is None:
        remote = remote_backend_configured(cfg)
    if remote:
        return CaptureStatus.WARN_REMOTE_ONLY
    return CaptureStatus.DISABLED_NO_BACKEND


def is_capture_enabled(
    cfg: PoppyConfig,
    *,
    project: str | None = None,
    host_cli: bool | None = None,
    remote: bool | None = None,
    backend_broken: bool | None = None,
) -> bool:
    """Whether auto-capture should run right now (for ``project``, if given)."""
    status = evaluate(cfg, project=project, host_cli=host_cli, remote=remote, backend_broken=backend_broken)
    return status in ENABLED_STATUSES


_STATUS_MESSAGES = {
    CaptureStatus.ACTIVE: "Auto-capture is active (extracting locally via your host CLI).",
    CaptureStatus.FORCED_ENV: f"Auto-capture forced on by {_ENV_VAR}.",
    CaptureStatus.INERT_PENDING: (
        "Auto-capture is pending your consent; nothing is captured yet. "
        "Run `poppy autocapture on --global` to turn it on."
    ),
    CaptureStatus.DISABLED_OPT_OUT: (
        "Auto-capture is off (you opted out). Run `poppy autocapture on --global` to re-enable."
    ),
    CaptureStatus.DISABLED_ENV: f"Auto-capture disabled by {_ENV_VAR}.",
    CaptureStatus.DISABLED_PROJECT: (
        "Auto-capture is off for this project. Run `poppy autocapture on` here to re-enable it."
    ),
    CaptureStatus.WARN_REMOTE_ONLY: (
        "Auto-capture is INACTIVE: only a paid remote backend is configured, so capture will not "
        "auto-spend. Install a host CLI (claude/cursor-agent/codex/gemini) for free local capture."
    ),
    CaptureStatus.WARN_BACKEND_BROKEN: (
        "Auto-capture is INACTIVE: the host CLI is on PATH but every extraction is failing, so "
        "nothing is being captured. Check that the CLI runs and is logged in."
    ),
    CaptureStatus.DISABLED_NO_BACKEND: (
        "Auto-capture is INACTIVE: no extraction backend found. Install a supported host CLI for free local capture."
    ),
}


def status_message(status: CaptureStatus) -> str:
    return _STATUS_MESSAGES.get(status, str(status.value))
