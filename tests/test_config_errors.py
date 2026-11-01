import sys

import click
import pytest
from click.testing import CliRunner

from poppy.cli import main as main_mod
from poppy.config import load_config


@pytest.mark.parametrize(
    "args",
    [["list"], ["config", "set", "auto-sync", "off"], ["sync", "run"], ["serve", "--no-daemon"]],
)
def test_console_stops_on_torn_config(tmp_path, monkeypatch, args):
    config_path = tmp_path / "config.json"
    payload = '{"auto_sync": "off",'
    config_path.write_text(payload)
    monkeypatch.setattr(sys, "argv", ["poppy", *args])
    result = CliRunner().invoke(click.Command("poppy", callback=main_mod.main), env={"POPPY_DIR": str(tmp_path)})

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert len(result.output.splitlines()) == 1
    assert str(config_path) in result.output
    assert "line 1 column 21 (char 20)" in result.output
    assert "fix the file or move it aside" in result.output
    assert "Traceback" not in result.output
    assert config_path.read_text() == payload
    assert not (tmp_path / "memories.db").exists()


@pytest.mark.parametrize("payload", [b'{"auto_sync": "off",', b"[]", b"null", b"\xff"])
def test_doctor_reports_unreadable_poppy_config(tmp_path, monkeypatch, payload):
    from poppy.cli import doctor as doctor_mod

    config_path = tmp_path / "config.json"
    config_path.write_bytes(payload)
    monkeypatch.setattr(doctor_mod, "check_executable", lambda ctx: None)

    def must_skip(*args, **kwargs):
        pytest.fail("config-dependent checks must be skipped")

    monkeypatch.setattr(doctor_mod, "check_engine", must_skip)
    monkeypatch.setattr(doctor_mod, "_prepare_capture", must_skip)
    monkeypatch.setattr(doctor_mod, "check_version", must_skip)
    result = CliRunner().invoke(main_mod.cli, ["doctor"], env={"POPPY_DIR": str(tmp_path), "HOME": str(tmp_path)})

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert result.output.count(": FAIL") == 1
    assert "config.json: FAIL" in result.output
    assert str(config_path) in result.output
    assert "fix the file or move it aside" in result.output
    assert "remaining checks: SKIP" in result.output
    assert "Traceback" not in result.output
    assert config_path.read_bytes() == payload


@pytest.mark.parametrize(
    "payload, position",
    [
        (b'{"encrypted": true, "telemetry_enabled": false,\n', "line 2 column 1"),
        (b"[]", "expected a JSON object"),
        (b"null", "expected a JSON object"),
        (b'"text"', "expected a JSON object"),
        (b"42", "expected a JSON object"),
        (b"false", "expected a JSON object"),
        (b"\xff", "invalid UTF-8 at byte 0"),
    ],
)
def test_unreadable_config_never_returns_defaults(tmp_path, monkeypatch, payload, position):
    from poppy.errors import ConfigError, PoppyError

    config_path = tmp_path / "config.json"
    config_path.write_bytes(payload)

    def must_not_migrate(*args, **kwargs):
        pytest.fail("unreadable settings must not trigger migrations")

    monkeypatch.setattr("poppy.config._pin_legacy_api_url", must_not_migrate)
    monkeypatch.setattr("poppy.config._migrate_trags_key_to_keychain", must_not_migrate)
    with pytest.raises(ConfigError) as exc:
        load_config(tmp_path)

    assert isinstance(exc.value, PoppyError)
    assert isinstance(exc.value, ValueError)
    assert str(config_path) in str(exc.value)
    assert position in str(exc.value)
    assert "fix the file or move it aside" in str(exc.value)
    assert config_path.read_bytes() == payload


def test_hook_keeps_exiting_zero_on_torn_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"auto_sync": "off",')
    result = CliRunner().invoke(
        main_mod.cli,
        ["hook", "session-start"],
        input="{}",
        env={"POPPY_DIR": str(tmp_path)},
    )

    assert result.exit_code == 0
    assert str(config_path) in result.stderr
    assert "fix the file or move it aside" in result.stderr
    assert "Traceback" not in result.stderr
