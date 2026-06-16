"""Integration: consent flow (command, setup prompt, SessionStart notice).

Covers `poppy consent --enable/--disable/--status`, the TTY setup prompt, the
non-interactive pending path, that a fresh install captures nothing until consent,
and that opt-out persists across a reload (a simulated default change).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from poppy.cli.hooks import hook
from poppy.cli.main import cli
from poppy.config import load_config
from poppy.consolidation import is_enabled


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POPPY_CONSOLIDATE", raising=False)


def test_consent_enable_records_granted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    r = CliRunner().invoke(cli, ["consent", "--enable"])
    assert r.exit_code == 0, r.output
    assert load_config(tmp_path).consent == "granted"


def test_consent_disable_persists_across_reload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    r = CliRunner().invoke(cli, ["consent", "--disable"])
    assert r.exit_code == 0, r.output
    # Opt-out persists — a later default change can never silently re-enable.
    assert load_config(tmp_path).consent == "denied"
    assert load_config(tmp_path).consent == "denied"


def test_consent_status_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    r = CliRunner().invoke(cli, ["consent", "--status"])
    assert r.exit_code == 0, r.output
    assert "pending" in r.output.lower()


def test_fresh_install_inert_until_consent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setattr("poppy.capture.policy.host_cli_available", lambda: True)

    # Fresh install: consent pending → capture inert even with a backend present.
    assert is_enabled(load_config(tmp_path)) is False

    CliRunner().invoke(cli, ["consent", "--enable"])
    # After consent + host CLI → default-on.
    assert is_enabled(load_config(tmp_path)) is True


def test_setup_tty_prompt_records_consent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On a TTY the setup consent helper asks and persists the answer."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    import poppy.cli.main as main

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("click.confirm", lambda *a, **k: True)

    main._maybe_prompt_consent(load_config(tmp_path), assume_yes=False)
    assert load_config(tmp_path).consent == "granted"

    # Declining persists as an opt-out, not pending.
    monkeypatch.setenv("POPPY_DIR", str(tmp_path / "b"))
    monkeypatch.setattr("click.confirm", lambda *a, **k: False)
    main._maybe_prompt_consent(load_config(tmp_path / "b"), assume_yes=False)
    assert load_config(tmp_path / "b").consent == "denied"


def test_setup_assume_yes_grants_without_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    import poppy.cli.main as main

    def _boom(*a, **k):
        raise AssertionError("must not prompt when --yes is given")

    monkeypatch.setattr("click.confirm", _boom)
    main._maybe_prompt_consent(load_config(tmp_path), assume_yes=True)
    assert load_config(tmp_path).consent == "granted"


def test_setup_noninteractive_leaves_consent_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-interactive `poppy setup claude-code` leaves consent pending."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setattr("poppy.setup.claude_code.install_for_client", lambda **kw: {})
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    r = CliRunner().invoke(cli, ["setup", "claude-code", "--no-hooks", "--no-claude-md"])
    assert r.exit_code == 0, r.output
    assert load_config(tmp_path).consent == "pending"
    assert "pending your consent" in r.output


def test_setup_grandfathered_user_not_reprompted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    import poppy.cli.main as main

    def _boom(*a, **k):
        raise AssertionError("a grandfathered user must never be re-prompted")

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("click.confirm", _boom)

    # Legacy explicit-true → grandfathered as granted; helper is a no-op.
    cfg = load_config(tmp_path)
    cfg.consolidate_enabled = True
    main._maybe_prompt_consent(cfg, assume_yes=False)


def test_session_start_shows_pending_banner_every_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """While consent is pending the banner nudges every session — a
    persistent reminder until the user acts, not a one-time notice."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    payload = json.dumps({"cwd": str(tmp_path), "session_id": "s1", "hook_event_name": "SessionStart"})
    runner = CliRunner()

    first = runner.invoke(hook, ["session-start"], input=payload)
    assert first.exit_code == 0
    assert "poppy consent --enable" in first.stdout

    # Still pending on a later session → the nudge persists (no silent one-and-done).
    second = runner.invoke(hook, ["session-start"], input=payload)
    assert second.exit_code == 0
    assert "poppy consent --enable" in second.stdout


def test_session_start_banner_not_pending_after_consent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Once consent is recorded the banner switches off the pending nudge and
    reports the active state instead."""
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setattr("poppy.capture.policy.host_cli_available", lambda: True)
    CliRunner().invoke(cli, ["consent", "--enable"])  # status no longer pending

    payload = json.dumps({"cwd": str(tmp_path), "session_id": "s1", "hook_event_name": "SessionStart"})
    r = CliRunner().invoke(hook, ["session-start"], input=payload)
    assert r.exit_code == 0
    ctx = json.loads(r.stdout)["hookSpecificOutput"]["additionalContext"]
    assert "pending your consent" not in ctx
    assert "active" in ctx.lower()
