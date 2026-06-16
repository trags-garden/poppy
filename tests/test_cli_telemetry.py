"""CLI tests for `poppy telemetry status|on|off` and the first-run notice.

POPPY_DIR always points at a temp dir (autouse conftest fixture plus explicit
env overrides), so the real ~/.poppy is never touched. CliRunner env values of
None remove a variable, which is how telemetry-on cases drop the suite-wide
POPPY_TELEMETRY_OFF=1 guard.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from poppy.cli.main import cli

NOTICE_SNIPPET = "anonymous usage telemetry is on"


def _env_on(tmp_path: Path) -> dict:
    return {"POPPY_DIR": str(tmp_path), "POPPY_TELEMETRY_OFF": None}


def _env_off(tmp_path: Path) -> dict:
    return {"POPPY_DIR": str(tmp_path), "POPPY_TELEMETRY_OFF": "1"}


def test_status_default_on(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["telemetry", "status"], env=_env_on(tmp_path))
    assert result.exit_code == 0
    assert "Telemetry: on (default)" in result.output
    assert "never sent" in result.output


def test_bare_telemetry_shows_status(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["telemetry"], env=_env_on(tmp_path))
    assert result.exit_code == 0
    assert "Telemetry: on (default)" in result.output


def test_off_persists_and_status_reports_off(tmp_path):
    runner = CliRunner()
    env = _env_on(tmp_path)

    result = runner.invoke(cli, ["telemetry", "off"], env=env)
    assert result.exit_code == 0
    assert "Telemetry is off." in result.output

    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["telemetry_enabled"] is False

    result = runner.invoke(cli, ["telemetry", "status"], env=env)
    assert result.exit_code == 0
    assert "Telemetry: off" in result.output
    assert "config.json" in result.output


def test_on_after_off(tmp_path):
    runner = CliRunner()
    env = _env_on(tmp_path)
    runner.invoke(cli, ["telemetry", "off"], env=env)

    result = runner.invoke(cli, ["telemetry", "on"], env=env)
    assert result.exit_code == 0
    assert "Telemetry is on." in result.output

    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["telemetry_enabled"] is True

    result = runner.invoke(cli, ["telemetry", "status"], env=env)
    assert "Telemetry: on" in result.output


def test_env_var_beats_config_flag(tmp_path):
    runner = CliRunner()
    runner.invoke(cli, ["telemetry", "on"], env=_env_on(tmp_path))

    result = runner.invoke(cli, ["telemetry", "status"], env=_env_off(tmp_path))
    assert result.exit_code == 0
    assert "Telemetry: off" in result.output
    assert "POPPY_TELEMETRY_OFF" in result.output


def test_telemetry_on_warns_when_env_var_overrides(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["telemetry", "on"], env=_env_off(tmp_path))
    assert result.exit_code == 0
    assert "POPPY_TELEMETRY_OFF=1 is set" in result.output


def test_notice_fires_exactly_once_on_stderr(tmp_path):
    runner = CliRunner()
    env = _env_on(tmp_path)

    first = runner.invoke(cli, ["list"], env=env)
    assert first.exit_code == 0
    assert NOTICE_SNIPPET in first.stderr
    assert NOTICE_SNIPPET not in first.stdout  # stderr only, stdout stays parseable

    second = runner.invoke(cli, ["list"], env=env)
    assert second.exit_code == 0
    assert NOTICE_SNIPPET not in second.stderr


def test_notice_never_fires_when_env_off(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["list"], env=_env_off(tmp_path))
    assert result.exit_code == 0
    assert NOTICE_SNIPPET not in result.stderr
    assert NOTICE_SNIPPET not in result.stdout


def test_notice_never_fires_after_config_opt_out(tmp_path):
    runner = CliRunner()
    env = _env_on(tmp_path)
    runner.invoke(cli, ["telemetry", "off"], env=env)
    result = runner.invoke(cli, ["list"], env=env)
    assert result.exit_code == 0
    assert NOTICE_SNIPPET not in result.stderr


def test_notice_skipped_for_telemetry_command_itself(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["telemetry", "status"], env=_env_on(tmp_path))
    assert result.exit_code == 0
    assert NOTICE_SNIPPET not in result.stderr


def test_remember_payload_never_contains_content_or_project_name(tmp_path):
    calls: list[tuple[str, dict]] = []

    def _record(poppy_dir, event, properties=None):
        calls.append((event, properties or {}))

    runner = CliRunner()
    with patch("poppy.cli.main.telemetry.capture", side_effect=_record):
        result = runner.invoke(
            cli,
            ["remember", "the launch codes are 0000", "--project", "supersecret-client"],
            env=_env_on(tmp_path),
        )
    assert result.exit_code == 0

    writes = [props for event, props in calls if event == "memory_write"]
    assert len(writes) == 1
    props = writes[0]
    assert props["has_project"] is True
    assert "project" not in props
    blob = json.dumps(props)
    assert "supersecret-client" not in blob
    assert "launch codes" not in blob


def test_recall_payload_contains_query_length_not_text(tmp_path):
    calls: list[tuple[str, dict]] = []

    def _record(poppy_dir, event, properties=None):
        calls.append((event, properties or {}))

    runner = CliRunner()
    with patch("poppy.cli.main.telemetry.capture", side_effect=_record):
        result = runner.invoke(cli, ["recall", "very private query"], env=_env_on(tmp_path))
    assert result.exit_code == 0

    recalls = [props for event, props in calls if event == "recall_call"]
    assert len(recalls) == 1
    props = recalls[0]
    assert props["query_length"] == len("very private query")
    assert "very private query" not in json.dumps(props)
