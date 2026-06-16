"""Tests for ConsolidationPolicy (ADR-0002: consent + default-on precedence).

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
