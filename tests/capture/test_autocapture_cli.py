"""Integration tests for the unified automatic-capture command."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from poppy.cli.main import _CONSENT_DISCLOSURE, cli
from poppy.config import load_config, save_config


@pytest.fixture(autouse=True)
def _clear_capture_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POPPY_CONSOLIDATE", raising=False)


def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str = "myrepo") -> Path:
    project = tmp_path / name
    project.mkdir()
    (project / "pyproject.toml").write_text("")
    monkeypatch.chdir(project)
    return project


def test_off_without_scope_noninteractive_is_usage_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _project(tmp_path, monkeypatch)
    result = CliRunner().invoke(cli, ["autocapture", "off"], env={"POPPY_DIR": str(tmp_path)})

    assert result.exit_code == 2
    assert "--global or --project NAME" in result.output


@pytest.mark.parametrize(
    ("choice", "project_disabled", "consent"),
    [
        ("just this project", True, "pending"),
        ("everywhere", False, "denied"),
    ],
)
def test_off_tty_prompts_for_scope_and_acts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    choice: str,
    project_disabled: bool,
    consent: str,
) -> None:
    _project(tmp_path, monkeypatch)
    monkeypatch.setattr("click.testing._NamedTextIOWrapper.isatty", lambda _self: True)

    result = CliRunner().invoke(
        cli,
        ["autocapture", "off"],
        input=f"{choice}\n",
        env={"POPPY_DIR": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    config = load_config(tmp_path)
    assert config.is_project_disabled("myrepo") is project_disabled
    assert config.consent == consent


def test_off_global_outside_project_sets_denied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)

    result = CliRunner().invoke(
        cli,
        ["autocapture", "off", "--global"],
        env={"POPPY_DIR": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    assert load_config(tmp_path).consent == "denied"
    assert "persists across upgrades" in result.output


def test_global_and_project_are_mutually_exclusive(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["autocapture", "on", "--global", "--project", "myrepo"],
        env={"POPPY_DIR": str(tmp_path)},
    )

    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


def test_on_global_pending_discloses_confirms_and_emits_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict | None]] = []
    monkeypatch.setattr(
        "poppy.telemetry.capture_once",
        lambda _dir, event, props=None: calls.append((event, props)),
    )

    result = CliRunner().invoke(
        cli,
        ["autocapture", "on", "--global"],
        input="y\n",
        env={"POPPY_DIR": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    assert _CONSENT_DISCLOSURE.strip() in result.output
    assert load_config(tmp_path).consent == "granted"
    assert calls == [("consent_granted", {"via": "consent_cmd"})]


def test_on_global_denied_uses_distinct_regrant_confirmation(tmp_path: Path) -> None:
    config = load_config(tmp_path)
    config.consent = "denied"
    save_config(config)

    result = CliRunner().invoke(
        cli,
        ["autocapture", "on", "--global"],
        input="y\n",
        env={"POPPY_DIR": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    assert "You previously opted out of automatic capture. Re-enable it everywhere?" in result.output
    assert load_config(tmp_path).consent == "granted"


def test_on_global_yes_still_prints_disclosure(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["autocapture", "on", "--global", "--yes"],
        env={"POPPY_DIR": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    assert _CONSENT_DISCLOSURE.strip() in result.output
    assert "[Y/n]" not in result.output
    assert load_config(tmp_path).consent == "granted"


def test_project_on_is_deterministic_and_points_to_global_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _project(tmp_path, monkeypatch)
    config = load_config(tmp_path)
    config.disable_project("myrepo")
    save_config(config)

    result = CliRunner().invoke(cli, ["autocapture", "on"], env={"POPPY_DIR": str(tmp_path)})

    assert result.exit_code == 0, result.output
    config = load_config(tmp_path)
    assert config.is_project_disabled("myrepo") is False
    assert config.consent == "pending"
    assert "poppy autocapture on --global" in result.output


@pytest.mark.parametrize("args", [["autocapture"], ["autocapture", "status"]])
def test_status_reports_project_mask_and_global_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    _project(tmp_path, monkeypatch)
    config = load_config(tmp_path)
    config.disable_project("myrepo")
    save_config(config)

    result = CliRunner().invoke(cli, args, env={"POPPY_DIR": str(tmp_path)})

    assert result.exit_code == 0, result.output
    assert "Resolved: Auto-capture is off for this project" in result.output
    assert "Project 'myrepo': off" in result.output
    assert "Global consent: pending" in result.output


def test_autocapture_is_visible_and_capture_alias_is_hidden(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli, ["--help"], env={"POPPY_DIR": str(tmp_path)})

    assert result.exit_code == 0, result.output
    assert "autocapture" in result.output
    assert "\n  capture " not in result.output
