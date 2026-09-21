"""Tests for ConsolidationPolicy (ADR-0002: consent and default-on precedence).

Covers the full precedence matrix: consent-absent inert, env-off beats config,
env-on forces, explicit opt-out persists, host-CLI default-on, remote-only WARN
(no auto-spend), no-backend disabled, and the legacy-bool grandfather.
"""

from __future__ import annotations

import pytest

from poppy.capture.policy import (
    CaptureStatus,
    Consent,
    effective_consent,
    evaluate,
    host_cli_available,
    is_capture_enabled,
    remote_backend_configured,
)
from poppy.config import PoppyConfig


def _cfg(**kw) -> PoppyConfig:
    return PoppyConfig(**kw)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("POPPY_CONSOLIDATE", "POPPY_CONSOLIDATE_MODEL", "POPPY_CONSOLIDATE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_consent_absent_is_inert() -> None:
    assert evaluate(_cfg(), host_cli=True) is CaptureStatus.INERT_PENDING
    assert is_capture_enabled(_cfg(), host_cli=True) is False


def test_env_off_beats_granted_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POPPY_CONSOLIDATE", "off")
    assert evaluate(_cfg(consent="granted"), host_cli=True) is CaptureStatus.DISABLED_ENV
    assert is_capture_enabled(_cfg(consent="granted"), host_cli=True) is False


def test_env_on_forces_even_when_pending(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POPPY_CONSOLIDATE", "1")
    assert evaluate(_cfg(), host_cli=False, remote=False) is CaptureStatus.FORCED_ENV
    assert is_capture_enabled(_cfg(), host_cli=False, remote=False) is True


def test_opt_out_disabled_even_with_backend() -> None:
    assert evaluate(_cfg(consent="denied"), host_cli=True) is CaptureStatus.DISABLED_OPT_OUT
    assert is_capture_enabled(_cfg(consent="denied"), host_cli=True) is False


def test_granted_host_cli_is_active() -> None:
    assert evaluate(_cfg(consent="granted"), host_cli=True) is CaptureStatus.ACTIVE
    assert is_capture_enabled(_cfg(consent="granted"), host_cli=True) is True


def test_cursor_agent_is_a_free_host_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/cursor-agent" if name == "cursor-agent" else None)
    assert host_cli_available() is True


def test_granted_remote_only_warns_no_autospend() -> None:
    assert evaluate(_cfg(consent="granted"), host_cli=False, remote=True) is CaptureStatus.WARN_REMOTE_ONLY
    # WARN-inactive is not "enabled" — never auto-spend on a paid backend.
    assert is_capture_enabled(_cfg(consent="granted"), host_cli=False, remote=True) is False


def test_granted_no_backend_disabled() -> None:
    assert evaluate(_cfg(consent="granted"), host_cli=False, remote=False) is CaptureStatus.DISABLED_NO_BACKEND
    assert is_capture_enabled(_cfg(consent="granted"), host_cli=False, remote=False) is False


def test_legacy_bool_grandfathered_as_granted() -> None:
    assert effective_consent(_cfg(consolidate_enabled=True)) is Consent.GRANTED
    assert evaluate(_cfg(consolidate_enabled=True), host_cli=True) is CaptureStatus.ACTIVE


def test_opt_out_persists_over_legacy_bool() -> None:
    """An explicit opt-out wins even if the legacy enable bool is also set."""
    cfg = _cfg(consent="denied", consolidate_enabled=True)
    assert effective_consent(cfg) is Consent.DENIED
    assert is_capture_enabled(cfg, host_cli=True) is False


def test_remote_backend_detection() -> None:
    assert remote_backend_configured(_cfg(consolidate_model="x", consolidate_api_key="k")) is True
    assert remote_backend_configured(_cfg(consolidate_model="x")) is False
    assert remote_backend_configured(_cfg()) is False


# --- per-project off switch -------------------------------------


def test_disabled_project_beats_granted_consent() -> None:
    """A repo on the deny-list is off even with consent + a working backend."""
    cfg = _cfg(consent="granted", disabled_projects=["client-secret"])
    assert evaluate(cfg, project="client-secret", host_cli=True) is CaptureStatus.DISABLED_PROJECT
    assert is_capture_enabled(cfg, project="client-secret", host_cli=True) is False


def test_other_projects_unaffected_by_a_disabled_one() -> None:
    """The deny-list scopes to its projects only; everything else stays active."""
    cfg = _cfg(consent="granted", disabled_projects=["client-secret"])
    assert evaluate(cfg, project="my-oss", host_cli=True) is CaptureStatus.ACTIVE
    # No project context at all (e.g. not inside a repo) is likewise unaffected.
    assert evaluate(cfg, project=None, host_cli=True) is CaptureStatus.ACTIVE


def test_env_override_beats_project_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The env escape hatch stays the top precedence, above the per-project switch."""
    cfg = _cfg(consent="granted", disabled_projects=["client-secret"])
    monkeypatch.setenv("POPPY_CONSOLIDATE", "off")
    assert evaluate(cfg, project="client-secret", host_cli=True) is CaptureStatus.DISABLED_ENV
    monkeypatch.setenv("POPPY_CONSOLIDATE", "on")
    assert evaluate(cfg, project="client-secret", host_cli=False, remote=False) is CaptureStatus.FORCED_ENV


def test_disabled_project_shown_before_pending_consent() -> None:
    """Even pre-consent, a deny-listed repo reports the project-off state."""
    cfg = _cfg(disabled_projects=["client-secret"])  # consent pending
    assert evaluate(cfg, project="client-secret", host_cli=True) is CaptureStatus.DISABLED_PROJECT
