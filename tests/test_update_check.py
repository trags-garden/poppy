from __future__ import annotations

import datetime
import io
import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from poppy import update_check
from poppy.cli.main import cli
from poppy.config import load_config, save_config


def _allow_checks(monkeypatch: pytest.MonkeyPatch, poppy_dir: Path | None = None) -> None:
    monkeypatch.delenv("POPPY_UPDATE_CHECK_OFF", raising=False)
    monkeypatch.delenv("POPPY_TELEMETRY_OFF", raising=False)
    monkeypatch.delenv("POPPY_TELEMETRY_HOST", raising=False)
    if poppy_dir is not None:
        config = load_config(poppy_dir)
        config.telemetry_enabled = True
        save_config(config)


def _payload(version: str) -> dict[str, object]:
    return {"info": {"version": version}}


def test_check_fetches_once_per_day_and_refetches_after_ttl(tmp_path, monkeypatch):
    _allow_checks(monkeypatch, tmp_path)
    monkeypatch.setattr(update_check, "installed_version", lambda: "0.2.2")
    calls: list[int] = []

    def fetcher():
        calls.append(1)
        return _payload("0.2.3")

    start = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    first = update_check.check(tmp_path, fetcher=fetcher, now=start)
    second = update_check.check(tmp_path, fetcher=fetcher, now=start + datetime.timedelta(hours=23))
    third = update_check.check(tmp_path, fetcher=fetcher, now=start + datetime.timedelta(hours=24))

    assert first.refreshed is True
    assert first.latest_version == "0.2.3"
    assert second.refreshed is False
    assert third.refreshed is True
    assert len(calls) == 2
    assert set(json.loads((tmp_path / "update_check.json").read_text())) == {
        "checked_at",
        "latest_version",
        "notified_version",
    }


def test_failed_fetch_is_silent_and_latches_checked_at(tmp_path, monkeypatch, capsys):
    _allow_checks(monkeypatch, tmp_path)
    now = datetime.datetime(2026, 1, 2, tzinfo=datetime.UTC)

    def broken_fetcher():
        raise OSError("offline")

    result = update_check.check(tmp_path, fetcher=broken_fetcher, now=now)

    assert result.latest_version is None
    assert result.refreshed is True
    assert json.loads((tmp_path / "update_check.json").read_text())["checked_at"] == now.isoformat()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_notice_latches_once_for_each_new_version(tmp_path, monkeypatch):
    _allow_checks(monkeypatch, tmp_path)
    monkeypatch.setattr(update_check, "installed_version", lambda: "0.2.2")
    start = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)

    update_check.check(tmp_path, fetcher=lambda: _payload("0.2.3"), now=start)
    assert update_check.consume_update_notice(tmp_path) == (
        "poppy 0.2.3 is available (you have 0.2.2). Upgrade: pip install -U poppy-memory"
    )
    assert update_check.consume_update_notice(tmp_path) is None

    update_check.check(
        tmp_path,
        fetcher=lambda: _payload("0.2.4"),
        now=start + datetime.timedelta(days=1),
    )
    assert update_check.consume_update_notice(tmp_path) == (
        "poppy 0.2.4 is available (you have 0.2.2). Upgrade: pip install -U poppy-memory"
    )


@pytest.mark.parametrize("env_name", ["POPPY_UPDATE_CHECK_OFF", "POPPY_TELEMETRY_OFF"])
def test_environment_gates_prevent_fetch_and_notice(tmp_path, monkeypatch, env_name):
    _allow_checks(monkeypatch, tmp_path)
    monkeypatch.setattr(update_check, "installed_version", lambda: "0.2.2")
    update_check.check(tmp_path, fetcher=lambda: _payload("0.2.3"))
    monkeypatch.setenv(env_name, "1")
    calls: list[int] = []

    result = update_check.check(tmp_path, fetcher=lambda: calls.append(1) or _payload("9.9.9"))

    assert result.enabled is False
    assert env_name in (result.disabled_reason or "")
    assert calls == []
    assert update_check.consume_update_notice(tmp_path) is None


def test_config_gate_prevents_fetch_and_notice(tmp_path, monkeypatch):
    _allow_checks(monkeypatch, tmp_path)
    monkeypatch.setattr(update_check, "installed_version", lambda: "0.2.2")
    update_check.check(tmp_path, fetcher=lambda: _payload("0.2.3"))
    config = load_config(tmp_path)
    config.update_check = False
    save_config(config)
    calls: list[int] = []

    result = update_check.check(tmp_path, fetcher=lambda: calls.append(1) or _payload("9.9.9"))

    assert result.enabled is False
    assert "update_check is off" in (result.disabled_reason or "")
    assert calls == []
    assert update_check.consume_update_notice(tmp_path) is None


@pytest.mark.parametrize("env_value", [None, "0", "false"])
def test_telemetry_config_gate_prevents_http_request(tmp_path, monkeypatch, env_value):
    _allow_checks(monkeypatch, tmp_path)
    if env_value is not None:
        monkeypatch.setenv("POPPY_TELEMETRY_OFF", env_value)
    config = load_config(tmp_path)
    config.telemetry_enabled = False
    config.update_check = True
    save_config(config)

    def unexpected_urlopen(*args, **kwargs):
        pytest.fail("Disabled telemetry must prevent the PyPI request")

    monkeypatch.setattr(update_check.urllib.request, "urlopen", unexpected_urlopen)
    reason = f"telemetry is off: set in {tmp_path / 'config.json'}"
    assert update_check.status(tmp_path) == (False, reason)
    result = update_check.check(tmp_path)
    assert result.enabled is False
    assert result.disabled_reason == reason
    assert result.refreshed is False
    assert not (tmp_path / "update_check.json").exists()


def test_telemetry_enabled_and_update_check_on_reports_enabled(tmp_path, monkeypatch):
    _allow_checks(monkeypatch, tmp_path)
    config = load_config(tmp_path)
    config.telemetry_enabled = True
    config.update_check = True
    save_config(config)

    assert update_check.status(tmp_path) == (True, None)


@pytest.mark.parametrize(
    ("update_off", "telemetry_off", "reason"),
    [("1", "1", "POPPY_UPDATE_CHECK_OFF=1"), (None, "1", "POPPY_TELEMETRY_OFF=1"), (None, None, None)],
)
def test_existing_gates_take_precedence_over_telemetry_config(tmp_path, monkeypatch, update_off, telemetry_off, reason):
    _allow_checks(monkeypatch, tmp_path)
    if update_off is not None:
        monkeypatch.setenv("POPPY_UPDATE_CHECK_OFF", update_off)
    if telemetry_off is not None:
        monkeypatch.setenv("POPPY_TELEMETRY_OFF", telemetry_off)
    config = load_config(tmp_path)
    config.telemetry_enabled = False
    config.update_check = False
    save_config(config)

    assert update_check.status(tmp_path) == (False, reason or f"update_check is off in {tmp_path / 'config.json'}")


def test_unreadable_config_disables_update_check(tmp_path, monkeypatch):
    _allow_checks(monkeypatch)
    (tmp_path / "config.json").mkdir()

    assert update_check.status(tmp_path) == (False, "configuration could not be read")


def test_version_comparison_uses_integer_components():
    assert update_check.is_newer("0.2.10", "0.2.2") is True
    assert update_check.is_newer("0.2.2", "0.2.2") is False
    assert update_check.is_newer("0.2.2.0", "0.2.2") is False
    assert update_check.is_newer("not-a-version", "0.2.2") is False
    assert update_check.is_newer("0.2.3rc1", "0.2.2") is False


def test_unparseable_pypi_version_is_not_cached_as_an_update(tmp_path, monkeypatch):
    _allow_checks(monkeypatch, tmp_path)
    monkeypatch.setattr(update_check, "installed_version", lambda: "0.2.2")

    result = update_check.check(tmp_path, fetcher=lambda: _payload("0.2.3rc1"))

    assert result.latest_version is None
    assert result.update_available is False


def test_injected_fetcher_may_return_the_latest_version_directly(tmp_path, monkeypatch):
    _allow_checks(monkeypatch, tmp_path)
    monkeypatch.setattr(update_check, "installed_version", lambda: "0.2.2")

    result = update_check.check(tmp_path, fetcher=lambda: "0.2.3")

    assert result.latest_version == "0.2.3"
    assert result.update_available is True


def test_upgrade_command_selects_pipx_or_pip(monkeypatch):
    monkeypatch.setattr(sys, "prefix", "/Users/test/.local/pipx/venvs/poppy-memory")
    assert update_check.upgrade_command() == "pipx upgrade poppy-memory"
    monkeypatch.setattr(sys, "prefix", "/Users/test/project/.venv")
    assert update_check.upgrade_command() == "pip install -U poppy-memory"


def test_production_fetcher_sends_only_version_user_agent(monkeypatch):
    captured: dict[str, object] = {}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def fake_urlopen(request, timeout):
        captured["headers"] = dict(request.header_items())
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return Response(b'{"info": {"version": "0.2.3"}}')

    monkeypatch.setattr(update_check.urllib.request, "urlopen", fake_urlopen)
    assert update_check._production_fetcher("0.2.2") == _payload("0.2.3")
    assert captured == {
        "headers": {"User-agent": "poppy-memory/0.2.2 (update-check)"},
        "url": update_check.PYPI_URL,
        "timeout": 2,
    }


@pytest.mark.parametrize(
    ("latest", "expected"),
    [
        ("0.2.2", "version: OK, 0.2.2 (latest)"),
        ("0.2.3", "version: WARN, 0.2.3 available (installed 0.2.2)"),
    ],
)
def test_doctor_reports_version_status(tmp_path, monkeypatch, latest, expected):
    _allow_checks(monkeypatch, tmp_path)
    monkeypatch.setattr(update_check, "installed_version", lambda: "0.2.2")
    update_check.check(tmp_path, fetcher=lambda: _payload(latest))
    env = {
        "POPPY_DIR": str(tmp_path),
        "POPPY_TELEMETRY_OFF": None,
        "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude"),
    }
    runner = CliRunner()
    runner.invoke(cli, ["config", "set", "engine", "seed"], env=env)

    result = runner.invoke(cli, ["doctor"], env=env)

    assert expected in result.output
    if latest == "0.2.3":
        assert "pip install -U poppy-memory" in result.output


def test_doctor_reports_disabled_update_check_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_UPDATE_CHECK_OFF", "1")
    monkeypatch.setattr(update_check, "installed_version", lambda: "0.2.2")
    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={"POPPY_DIR": str(tmp_path), "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude")},
    )
    assert "version: OK, 0.2.2 (check is off: POPPY_UPDATE_CHECK_OFF=1)" in result.output


def test_config_command_sets_update_check_on_and_off(tmp_path, monkeypatch):
    _allow_checks(monkeypatch, tmp_path)
    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path), "POPPY_TELEMETRY_OFF": None}

    off = runner.invoke(cli, ["config", "set", "update_check", "off"], env=env)
    assert off.exit_code == 0
    assert load_config(tmp_path).update_check is False
    on = runner.invoke(cli, ["config", "set", "update_check", "on"], env=env)
    assert on.exit_code == 0
    assert load_config(tmp_path).update_check is True


def test_hook_command_skips_update_epilogue(tmp_path, monkeypatch):
    called: list[Path] = []
    monkeypatch.setattr(update_check, "maybe_print_update_notice", lambda path: called.append(path))

    result = CliRunner().invoke(
        cli,
        ["hook", "stop"],
        input="{}\n",
        env={"POPPY_DIR": str(tmp_path), "POPPY_TELEMETRY_OFF": "1"},
    )

    assert result.exit_code == 0
    assert called == []


def test_non_tty_cli_does_not_print_cached_notice(tmp_path, monkeypatch):
    _allow_checks(monkeypatch, tmp_path)
    monkeypatch.setattr(update_check, "installed_version", lambda: "0.2.2")
    update_check.check(tmp_path, fetcher=lambda: _payload("0.2.3"))

    result = CliRunner().invoke(
        cli,
        ["config", "set", "engine", "seed"],
        env={"POPPY_DIR": str(tmp_path), "POPPY_TELEMETRY_OFF": None},
    )

    assert result.exit_code == 0
    assert "poppy 0.2.3 is available" not in result.stderr
    assert json.loads((tmp_path / "update_check.json").read_text())["notified_version"] is None


def test_unanswered_telemetry_consent_prevents_update_request(tmp_path, monkeypatch):
    _allow_checks(monkeypatch)

    def unexpected_fetch():
        pytest.fail("Unanswered telemetry consent must prevent the PyPI request")

    result = update_check.check(tmp_path, fetcher=unexpected_fetch)
    assert result.enabled is False
    assert result.disabled_reason == "telemetry is off: not answered yet"
    assert not (tmp_path / "update_check.json").exists()
