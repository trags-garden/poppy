"""Unit tests for poppy.telemetry — file mutations, opt-out semantics, notice.

Network is never hit because we never call `capture()` with a working SDK
(the PostHog import is shimmed out per test) and the autouse conftest fixture
sets POPPY_TELEMETRY_OFF=1 unless a test removes it. The tests focus on the
deterministic state of `~/.poppy/analytics.json` and `~/.poppy/config.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import poppy
from poppy import telemetry


@pytest.fixture
def telemetry_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the suite-wide POPPY_TELEMETRY_OFF=1 guard for telemetry-on tests."""
    monkeypatch.delenv("POPPY_TELEMETRY_OFF", raising=False)


class FakeClient:
    """Stand-in for posthog.Posthog matching the real capture() signature."""

    instances: list["FakeClient"] = []

    def __init__(self, *args, **kwargs):
        self.captured: list[tuple[str, str | None, dict | None]] = []
        FakeClient.instances.append(self)

    def capture(self, event, distinct_id=None, properties=None):
        self.captured.append((event, distinct_id, properties))

    def shutdown(self, timeout=None):
        pass


@pytest.fixture
def fresh_client_state():
    """Reset the module-level client cache around a test."""
    FakeClient.instances = []
    old = telemetry._state["client"]
    telemetry._state["client"] = None
    yield
    telemetry._state["client"] = old


def test_first_run_creates_analytics_with_device_id(tmp_path: Path, telemetry_on, fresh_client_state) -> None:
    assert not (tmp_path / "analytics.json").exists()

    with patch("posthog.Posthog", FakeClient):
        telemetry.capture(tmp_path, "memory_write", {"memory_type": "fact"})

    data = json.loads((tmp_path / "analytics.json").read_text())
    assert data["telemetry"] == "on"
    assert len(data["device_id"]) >= 16
    assert "created_at" in data


def test_telemetry_off_writes_nothing(tmp_path: Path, telemetry_on, fresh_client_state) -> None:
    telemetry.set_enabled(tmp_path, False)

    with patch("posthog.Posthog") as fake_ph:
        telemetry.capture(tmp_path, "memory_write", {"memory_type": "fact"})

    # Off should never even import/initialize the client.
    fake_ph.assert_not_called()
    data = json.loads((tmp_path / "analytics.json").read_text())
    assert data["telemetry"] == "off"


def test_get_device_id_returns_none_when_off(tmp_path: Path, telemetry_on) -> None:
    telemetry.set_enabled(tmp_path, False)
    assert telemetry.get_device_id(tmp_path) is None


def test_get_device_id_returns_uuid_when_on(tmp_path: Path, telemetry_on) -> None:
    telemetry.set_enabled(tmp_path, True)
    device_id = telemetry.get_device_id(tmp_path)
    assert device_id is not None
    assert len(device_id) >= 16


def test_capture_never_raises_on_sdk_failure(tmp_path: Path, telemetry_on) -> None:
    telemetry.set_enabled(tmp_path, True)

    class BrokenClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("simulated SDK failure")

    # Reset module-level cache so we re-attempt init under the patched SDK.
    telemetry._state["client"] = None

    with patch("posthog.Posthog", BrokenClient):
        # Must not raise.
        telemetry.capture(tmp_path, "memory_write", {})


def test_set_enabled_preserves_device_id(tmp_path: Path, telemetry_on) -> None:
    telemetry.set_enabled(tmp_path, True)
    first_id = telemetry.get_device_id(tmp_path)
    telemetry.set_enabled(tmp_path, False)
    telemetry.set_enabled(tmp_path, True)
    assert telemetry.get_device_id(tmp_path) == first_id


# ---------------------------------------------------------------------------
# Config.json flag, precedence, status, first-run notice
# ---------------------------------------------------------------------------


def test_default_is_on(tmp_path: Path, telemetry_on) -> None:
    enabled, reason = telemetry.status(tmp_path)
    assert enabled is True
    assert reason == "default"


def test_env_var_always_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POPPY_TELEMETRY_OFF", raising=False)
    telemetry.set_enabled(tmp_path, True)
    assert telemetry.is_enabled(tmp_path) is True

    monkeypatch.setenv("POPPY_TELEMETRY_OFF", "1")
    enabled, reason = telemetry.status(tmp_path)
    assert enabled is False
    assert "POPPY_TELEMETRY_OFF" in reason


def test_set_enabled_persists_flag_in_config_json(tmp_path: Path, telemetry_on) -> None:
    telemetry.set_enabled(tmp_path, False)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["telemetry_enabled"] is False
    assert telemetry.is_enabled(tmp_path) is False

    telemetry.set_enabled(tmp_path, True)
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["telemetry_enabled"] is True
    assert telemetry.is_enabled(tmp_path) is True


def test_legacy_analytics_opt_out_still_respected(tmp_path: Path, telemetry_on) -> None:
    # Older installs persisted only {"telemetry": "off"} in analytics.json.
    (tmp_path / "analytics.json").write_text(json.dumps({"telemetry": "off"}))
    enabled, reason = telemetry.status(tmp_path)
    assert enabled is False
    assert "analytics.json" in reason


def test_corrupt_config_json_falls_back_to_default_on(tmp_path: Path, telemetry_on) -> None:
    (tmp_path / "config.json").write_text("{not json")
    assert telemetry.is_enabled(tmp_path) is True


def test_notice_prints_exactly_once(tmp_path: Path, telemetry_on, capsys: pytest.CaptureFixture) -> None:
    telemetry.maybe_print_first_run_notice(tmp_path)
    captured = capsys.readouterr()
    assert "anonymous usage telemetry is on" in captured.err
    assert "poppy telemetry off" in captured.err
    assert captured.out == ""  # never stdout

    telemetry.maybe_print_first_run_notice(tmp_path)
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""

    data = json.loads((tmp_path / "analytics.json").read_text())
    assert data["first_run_notice_shown"] is True


def test_notice_never_fires_when_off_via_env(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    # conftest sets POPPY_TELEMETRY_OFF=1.
    telemetry.maybe_print_first_run_notice(tmp_path)
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""
    # Off means no analytics.json mutation either.
    assert not (tmp_path / "analytics.json").exists()


def test_notice_never_fires_when_off_via_config(tmp_path: Path, telemetry_on, capsys: pytest.CaptureFixture) -> None:
    telemetry.set_enabled(tmp_path, False)
    telemetry.maybe_print_first_run_notice(tmp_path)
    assert capsys.readouterr().err == ""


def test_notice_never_raises(tmp_path: Path, telemetry_on, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(telemetry, "_save", _boom)
    # Must not raise even when persistence is impossible.
    telemetry.maybe_print_first_run_notice(tmp_path)


def test_notice_flag_survives_first_capture(
    tmp_path: Path, telemetry_on, fresh_client_state, capsys: pytest.CaptureFixture
) -> None:
    """Regression: _ensure_client's first-run write must merge, not replace,
    analytics.json — otherwise the seen-flag is wiped and the notice repeats."""
    telemetry.maybe_print_first_run_notice(tmp_path)
    assert "telemetry is on" in capsys.readouterr().err

    with patch("posthog.Posthog", FakeClient):
        telemetry.capture(tmp_path, "memory_write", {"memory_type": "fact"})

    data = json.loads((tmp_path / "analytics.json").read_text())
    assert data["first_run_notice_shown"] is True
    telemetry.maybe_print_first_run_notice(tmp_path)
    assert capsys.readouterr().err == ""


def test_explicit_choice_counts_as_notice_seen(tmp_path: Path, telemetry_on, capsys: pytest.CaptureFixture) -> None:
    telemetry.set_enabled(tmp_path, True)
    telemetry.maybe_print_first_run_notice(tmp_path)
    assert capsys.readouterr().err == ""


def test_cli_install_reports_resolved_version(tmp_path: Path, telemetry_on, fresh_client_state) -> None:
    with patch("posthog.Posthog", FakeClient):
        telemetry.capture(tmp_path, "recall_call", {"query_length": 3})

    assert len(FakeClient.instances) == 1
    events = FakeClient.instances[0].captured
    install_events = [e for e in events if e[0] == "cli_install"]
    assert len(install_events) == 1
    props = install_events[0][2]
    assert props["version"] == poppy.__version__
