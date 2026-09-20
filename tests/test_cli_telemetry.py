"""CLI tests for `poppy telemetry status|on|off` and the first-run consent prompt.

POPPY_DIR always points at a temp dir (autouse conftest fixture plus explicit
env overrides), so the real ~/.poppy is never touched. CliRunner env values of
None remove a variable, which is how telemetry-on cases drop the suite-wide
POPPY_TELEMETRY_OFF=1 guard.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from poppy.cli.main import cli

PROMPT_SNIPPET = "Send anonymous usage events?"


def _env_on(tmp_path: Path) -> dict:
    return {"POPPY_DIR": str(tmp_path), "POPPY_TELEMETRY_OFF": None}


def _env_off(tmp_path: Path) -> dict:
    return {"POPPY_DIR": str(tmp_path), "POPPY_TELEMETRY_OFF": "1"}


def test_status_unanswered(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["telemetry", "status"], env=_env_on(tmp_path))
    assert result.exit_code == 0
    assert "Telemetry: off (not answered yet)" in result.output


def test_bare_telemetry_shows_status(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["telemetry"], env=_env_on(tmp_path))
    assert result.exit_code == 0
    assert "Telemetry: off (not answered yet)" in result.output


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


def test_telemetry_on_reports_effective_off_under_env_override(tmp_path):
    """`telemetry on` must match `telemetry status`: an env override keeps it off."""
    runner = CliRunner()
    result = runner.invoke(cli, ["telemetry", "on"], env=_env_off(tmp_path))
    assert result.exit_code == 0
    assert "currently off" in result.output
    assert "POPPY_TELEMETRY_OFF=1" in result.output
    # Config still records the intent, so unsetting the override re-enables it.
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["telemetry_enabled"] is True


def test_telemetry_on_reports_effective_off_under_bad_host_override(tmp_path):
    """A non-loopback POPPY_TELEMETRY_HOST disables telemetry; `on` says so, matching status."""
    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path), "POPPY_TELEMETRY_OFF": None, "POPPY_TELEMETRY_HOST": "https://eu.i.posthog.com"}
    on = runner.invoke(cli, ["telemetry", "on"], env=env)
    assert on.exit_code == 0
    assert "currently off" in on.output and "POPPY_TELEMETRY_HOST" in on.output
    status = runner.invoke(cli, ["telemetry", "status"], env=env)
    assert "Telemetry: off" in status.output and "POPPY_TELEMETRY_HOST" in status.output


@pytest.fixture
def interactive(monkeypatch):
    """Report a terminal to the consent check.

    CliRunner swaps the standard streams for its own buffers during invoke, so
    they cannot be made to look like a terminal from out here; the predicate is
    patched instead. Stubbing the whole `sys` module inside `telemetry` would
    also work, and would silently break every other attribute read off it.
    `test_redirected_stream_prevents_prompt` covers the predicate itself.
    """
    from poppy import telemetry

    monkeypatch.setattr(telemetry, "_streams_are_a_terminal", lambda: True)


@pytest.mark.parametrize(("answer", "enabled"), [("y\n", True), ("n\n", False), ("\n", False)])
def test_prompt_persists_answer_once_on_stderr(tmp_path, interactive, answer, enabled):
    runner = CliRunner()
    env = _env_on(tmp_path)
    with patch("posthog.Posthog") as client:
        first = runner.invoke(cli, ["list"], input=answer, env=env)
        assert first.exit_code == 0
        assert PROMPT_SNIPPET in first.stderr
        assert PROMPT_SNIPPET not in first.stdout
        assert json.loads((tmp_path / "config.json").read_text())["telemetry_enabled"] is enabled
        second = runner.invoke(cli, ["list"], env=env)
        assert second.exit_code == 0
        assert PROMPT_SNIPPET not in second.output
        client.assert_not_called()


def test_non_interactive_run_is_silent_and_sends_nothing(tmp_path):
    with patch("posthog.Posthog") as client, patch("poppy.cli.main._get_engine") as engine:
        engine.return_value.recall.return_value = []
        result = CliRunner().invoke(cli, ["recall", "private query"], env=_env_on(tmp_path))
    assert result.exit_code == 0
    assert result.stderr == ""
    assert PROMPT_SNIPPET not in result.stdout
    assert not (tmp_path / "analytics.json").exists()
    from poppy.config import load_config

    assert load_config(tmp_path).telemetry_enabled is None
    client.assert_not_called()


@pytest.mark.parametrize("stream_name", ["stdin", "stdout", "stderr"])
def test_redirected_stream_prevents_prompt(tmp_path, monkeypatch, stream_name):
    monkeypatch.delenv("POPPY_TELEMETRY_OFF", raising=False)
    from poppy import telemetry

    for name in ("stdin", "stdout", "stderr"):
        monkeypatch.setattr(getattr(sys, name), "isatty", lambda name=name: name != stream_name)
    with patch("click.confirm") as confirm:
        telemetry.maybe_prompt_for_consent(tmp_path)
    confirm.assert_not_called()
    assert not (tmp_path / "config.json").exists()


@pytest.mark.parametrize("command", ["hook", "serve", "daemon", "telemetry"])
def test_background_and_telemetry_commands_skip_prompt_even_on_tty(tmp_path, interactive, monkeypatch, command):
    # Keep the real root dispatch, replacing only the long-running command body.
    monkeypatch.setitem(cli.commands, command, click.Command(command, callback=lambda: None))
    with patch("click.confirm") as confirm:
        result = CliRunner().invoke(cli, [command], env=_env_on(tmp_path))
    assert result.exit_code == 0
    assert result.output == ""
    confirm.assert_not_called()
    assert not (tmp_path / "config.json").exists()


def test_detached_sync_worker_skips_prompt_even_on_tty(tmp_path, interactive):
    with patch("poppy.sync.auto.run_worker"), patch("click.confirm") as confirm:
        result = CliRunner().invoke(cli, ["sync", "_auto-worker"], env=_env_on(tmp_path))
    assert result.exit_code == 0
    assert result.output == ""
    confirm.assert_not_called()


def test_interactive_sync_command_prompts(tmp_path, interactive):
    result = CliRunner().invoke(cli, ["sync", "status"], input="n\n", env=_env_on(tmp_path))
    assert result.exit_code == 0
    assert PROMPT_SNIPPET in result.stderr
    assert json.loads((tmp_path / "config.json").read_text())["telemetry_enabled"] is False


@pytest.mark.parametrize("choice", ["on", "off"])
def test_explicit_choice_prevents_prompt(tmp_path, interactive, choice):
    runner = CliRunner()
    with patch("click.confirm") as confirm:
        assert runner.invoke(cli, ["telemetry", choice], env=_env_on(tmp_path)).exit_code == 0
        assert runner.invoke(cli, ["list"], env=_env_on(tmp_path)).exit_code == 0
    confirm.assert_not_called()


@pytest.mark.parametrize(
    "override", [{"POPPY_TELEMETRY_OFF": "1"}, {"POPPY_TELEMETRY_HOST": "https://example.invalid"}]
)
def test_environment_overrides_prevent_prompt(tmp_path, interactive, override):
    with patch("click.confirm") as confirm:
        result = CliRunner().invoke(cli, ["list"], env={**_env_on(tmp_path), **override})
    assert result.exit_code == 0
    assert result.stderr == ""
    confirm.assert_not_called()


def test_aborted_prompt_leaves_consent_unanswered(tmp_path, interactive):
    result = CliRunner().invoke(cli, ["list"], input="", env=_env_on(tmp_path))
    assert result.exit_code == 0
    assert PROMPT_SNIPPET in result.stderr
    from poppy.config import load_config

    assert load_config(tmp_path).telemetry_enabled is None


def test_legacy_default_on_notice_does_not_count_as_consent(tmp_path, interactive):
    (tmp_path / "analytics.json").write_text(json.dumps({"telemetry": "on", "first_run_notice_shown": True}))
    result = CliRunner().invoke(cli, ["list"], input="n\n", env=_env_on(tmp_path))
    assert result.exit_code == 0
    assert PROMPT_SNIPPET in result.stderr
    assert json.loads((tmp_path / "config.json").read_text())["telemetry_enabled"] is False


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


@pytest.mark.parametrize(("answer", "enabled"), [(True, True), (False, False)])
def test_command_emits_only_after_affirmative_answer(tmp_path, interactive, answer, enabled):
    from poppy import telemetry

    def confirm(*args, **kwargs):
        assert not telemetry.is_enabled(tmp_path)
        client.assert_not_called()
        return answer

    with (
        patch.dict(telemetry._state, {"client": None, "host": None, "registered_atexit": True}),
        patch("posthog.Posthog") as client,
        patch("click.confirm", side_effect=confirm),
        patch("poppy.cli.main._get_engine") as engine,
    ):
        engine.return_value.recall.return_value = []
        result = CliRunner().invoke(cli, ["recall", "private query"], env=_env_on(tmp_path))
        assert result.exit_code == 0
        assert client.called is enabled
        assert client.return_value.capture.called is enabled


def test_legacy_opt_out_prevents_prompt(tmp_path, interactive):
    (tmp_path / "analytics.json").write_text(json.dumps({"telemetry": "off"}))
    with patch("click.confirm") as confirm:
        result = CliRunner().invoke(cli, ["list"], env=_env_on(tmp_path))
    assert result.exit_code == 0
    confirm.assert_not_called()


@pytest.mark.parametrize("flag", ["--help", "-h", "--yes"])
def test_flags_meaning_do_not_ask_me_skip_the_prompt(tmp_path, interactive, monkeypatch, flag):
    """Help output stays scriptable, and `--yes` is an install saying it will not answer.

    Click parses a subcommand's arguments only after the root callback has run,
    so the guard reads argv; the CliRunner arguments stand in for the command
    body that does run.
    """
    monkeypatch.setattr(sys, "argv", ["poppy", "list", flag])
    with patch("click.confirm") as confirm:
        result = CliRunner().invoke(cli, ["list"], env=_env_on(tmp_path))
    assert result.exit_code == 0
    assert result.stderr == ""
    confirm.assert_not_called()
    assert not (tmp_path / "config.json").exists()


def test_unanswered_status_says_it_has_not_been_asked(tmp_path):
    result = CliRunner().invoke(cli, ["telemetry", "status"], env=_env_on(tmp_path))
    assert result.exit_code == 0
    assert "Telemetry: off (not answered yet)" in result.output
    assert "Nothing is sent until you answer" in result.output


def test_declined_status_does_not_claim_it_is_unasked(tmp_path):
    runner = CliRunner()
    assert runner.invoke(cli, ["telemetry", "off"], env=_env_on(tmp_path)).exit_code == 0
    result = runner.invoke(cli, ["telemetry", "status"], env=_env_on(tmp_path))
    assert result.exit_code == 0
    assert "not answered yet" not in result.output
    assert "Nothing is sent until you answer" not in result.output
