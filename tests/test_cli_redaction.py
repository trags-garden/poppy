"""Dedicated CLI coverage for custom capture redaction."""

from __future__ import annotations

import json

from click.testing import CliRunner

from poppy.cli.main import cli
from poppy.config import load_config


def test_redaction_add_list_remove_round_trip(tmp_path) -> None:
    runner = CliRunner()
    env = {
        "POPPY_DIR": str(tmp_path),
        "POPPY_TEST_LIST_TOKEN": "value-that-must-not-be-printed",
    }

    literal_add = runner.invoke(cli, ["redaction", "add", "  bespoke-secret  "], env=env)
    env_add = runner.invoke(cli, ["redaction", "add", "--env", "POPPY_TEST_LIST_TOKEN"], env=env)
    assert literal_add.exit_code == 0, literal_add.output
    assert env_add.exit_code == 0, env_add.output
    # The literal is a secret: add confirms by character count, never by value.
    assert "bespoke-secret" not in literal_add.output
    assert "(14 characters)" in literal_add.output
    # Environment-variable names are not secret, so the name is shown on add.
    assert "POPPY_TEST_LIST_TOKEN" in env_add.output
    assert "value-that-must-not-be-printed" not in env_add.output
    config = load_config(tmp_path)
    assert config.redaction_literals == ["bespoke-secret"]
    assert config.redaction_env_vars == ["POPPY_TEST_LIST_TOKEN"]

    listed = runner.invoke(cli, ["redaction", "list"], env=env)
    assert listed.exit_code == 0, listed.output
    assert "6 secret pattern families, always on" in listed.output
    # The literal is itself a secret: hidden by default, revealed only on --show.
    assert "bespoke-secret" not in listed.output
    assert "(14 characters, hidden; use --show)" in listed.output
    assert "POPPY_TEST_LIST_TOKEN" in listed.output
    assert "value-that-must-not-be-printed" not in listed.output

    shown = runner.invoke(cli, ["redaction", "list", "--show"], env=env)
    assert shown.exit_code == 0, shown.output
    assert "bespoke-secret" in shown.output
    assert "value-that-must-not-be-printed" not in shown.output

    literal_remove = runner.invoke(cli, ["redaction", "remove", "bespoke-secret"], env=env)
    env_remove = runner.invoke(cli, ["redaction", "remove", "--env", "POPPY_TEST_LIST_TOKEN"], env=env)
    missing_remove = runner.invoke(cli, ["redaction", "remove", "not-configured"], env=env)
    assert literal_remove.exit_code == 0, literal_remove.output
    assert env_remove.exit_code == 0, env_remove.output
    assert missing_remove.exit_code == 0, missing_remove.output
    assert "No custom literal redaction found" in missing_remove.output
    # Remove (success and not-found) describes the literal by length, never value.
    assert "bespoke-secret" not in literal_remove.output
    assert "(14 characters)" in literal_remove.output
    assert "not-configured" not in missing_remove.output
    assert "(14 characters)" in missing_remove.output
    config = load_config(tmp_path)
    assert config.redaction_literals == []
    assert config.redaction_env_vars == []


def test_redaction_add_validation_errors(tmp_path) -> None:
    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path)}

    short = runner.invoke(cli, ["redaction", "add", "abc"], env=env)
    invalid_env = runner.invoke(cli, ["redaction", "add", "--env", "NOT-AN-IDENTIFIER"], env=env)
    both = runner.invoke(cli, ["redaction", "add", "literal", "--env", "TOKEN"], env=env)

    assert short.exit_code == 2
    assert "at least 4 characters" in short.output
    assert invalid_env.exit_code == 2
    assert "must match [A-Za-z_][A-Za-z0-9_]*" in invalid_env.output
    assert both.exit_code == 2
    assert "either a literal or --env NAME" in both.output


def test_redaction_duplicate_add_is_friendly_no_op(tmp_path) -> None:
    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path)}

    assert runner.invoke(cli, ["redaction", "add", "repeat-secret"], env=env).exit_code == 0
    duplicate = runner.invoke(cli, ["redaction", "add", "repeat-secret"], env=env)

    assert duplicate.exit_code == 0
    assert "already exists" in duplicate.output
    # A duplicate add must not echo the configured secret literal either.
    assert "repeat-secret" not in duplicate.output
    assert "(13 characters)" in duplicate.output
    assert load_config(tmp_path).redaction_literals == ["repeat-secret"]


def test_doctor_reports_custom_redaction_counts_and_invalid_entries(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "engine": "seed",
                "redaction": {
                    "literals": ["abc", "valid-secret"],
                    "env_vars": ["BAD-NAME", "VALID_TOKEN"],
                },
            }
        )
    )
    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={"POPPY_DIR": str(tmp_path), "CLAUDE_CONFIG_DIR": str(tmp_path / ".claude")},
    )

    assert result.exit_code == 0, result.output
    assert "custom redaction: OK" in result.output
    assert "1 literal(s), 1 environment-variable name(s) configured" in result.output
    assert "custom redaction entry: WARN" in result.output
    # The too-short literal is identified by position and length, never by value.
    assert "Skipped custom literal #1 (3 characters)" in result.output
    assert "'abc'" not in result.output
    assert "Skipped environment variable 'BAD-NAME'" in result.output
