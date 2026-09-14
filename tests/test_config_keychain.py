"""Trags API key storage in the OS keychain.

Two groups:

* Keychain-backed behavior runs against the in-memory keyring backend installed
  by the autouse ``_isolate_keychain`` conftest fixture, so nothing here touches
  the real OS keychain. ``keyring`` is a base dependency now, so these tests run
  everywhere; the ``requires_keyring`` marker is retained only to skip cleanly on
  an ancient env that predates keyring being a base dep.
* Headless-fallback behavior forces the keychain unavailable via monkeypatch, so
  it runs and asserts the config.json fallback in every environment, standing in
  for a headless host where keyring is installed but no backend is usable.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from poppy import keychain
from poppy.config import (
    TRAGS_API_KEY_ENV,
    PoppyConfig,
    _trags_key_account,
    load_config,
    resolve_trags_api_key,
    save_config,
    trags_api_key_location,
)

KEY = "usr_secretkey_abc123"

try:
    import keyring  # noqa: F401

    _HAS_KEYRING = True
except ImportError:
    _HAS_KEYRING = False

requires_keyring = pytest.mark.skipif(not _HAS_KEYRING, reason="keyring not installed (pre-base-dep env)")


def _set_key(poppy_dir: Path, value: str) -> PoppyConfig:
    cfg = PoppyConfig(poppy_dir=poppy_dir)
    cfg.set("trags-api-key", value)
    save_config(cfg)
    return cfg


# ---------------- keychain-backed (requires a keyring backend) ----------------


@requires_keyring
def test_set_stores_in_keychain_not_config_file(tmp_path):
    _set_key(tmp_path, KEY)

    # The plaintext key must not be anywhere in config.json.
    text = (tmp_path / "config.json").read_text()
    assert KEY not in text
    assert "trags_api_key" not in text

    # It is in the keychain, under an account distinct from the db-key accounts.
    account = _trags_key_account(tmp_path)
    assert account.startswith("trags-api-key:")
    assert keychain.get_secret(account) == KEY

    # And it resolves back for consumers.
    assert resolve_trags_api_key(load_config(tmp_path)) == KEY
    assert trags_api_key_location(tmp_path) == "keychain"


@requires_keyring
def test_migration_moves_plaintext_key_into_keychain_and_scrubs_file(tmp_path):
    # Simulate an upgrade: a pre-existing config.json with a plaintext key.
    (tmp_path / "config.json").write_text(f'{{"trags_api_key": "{KEY}", "auto_sync": "off"}}')

    cfg = load_config(tmp_path)

    # Transparently relocated: field cleared, file scrubbed, keychain populated.
    assert cfg.trags_api_key is None
    text = (tmp_path / "config.json").read_text()
    assert KEY not in text
    assert "trags_api_key" not in text
    assert keychain.get_secret(_trags_key_account(tmp_path)) == KEY
    # Unrelated settings survive the scrub rewrite.
    assert "auto_sync" in text
    assert resolve_trags_api_key(cfg) == KEY


@requires_keyring
def test_env_override_wins_over_keychain(tmp_path, monkeypatch):
    _set_key(tmp_path, KEY)
    monkeypatch.setenv(TRAGS_API_KEY_ENV, "usr_from_env")
    assert resolve_trags_api_key(load_config(tmp_path)) == "usr_from_env"


@requires_keyring
def test_empty_value_clears_the_keychain_entry(tmp_path):
    _set_key(tmp_path, KEY)
    assert keychain.get_secret(_trags_key_account(tmp_path)) == KEY

    # An explicit clear removes the stored secret and resolves to nothing.
    cfg = PoppyConfig(poppy_dir=tmp_path)
    cfg.set("trags-api-key", "")
    save_config(cfg)

    assert keychain.get_secret(_trags_key_account(tmp_path)) is None
    assert not resolve_trags_api_key(load_config(tmp_path))


@requires_keyring
def test_resolution_is_cached_and_does_not_re_hit_keychain(tmp_path, monkeypatch):
    account = _trags_key_account(tmp_path)
    keychain.set_secret(account, KEY)
    cfg = PoppyConfig(poppy_dir=tmp_path)

    # First read populates the per-process cache.
    assert resolve_trags_api_key(cfg) == KEY

    # Any further keychain read within the process would be a bug on the hot path.
    def _boom(_account):
        raise AssertionError("keychain must not be read again once cached")

    monkeypatch.setattr(keychain, "get_secret", _boom)
    assert resolve_trags_api_key(cfg) == KEY


@requires_keyring
def test_cache_invalidates_across_processes_when_config_changes(tmp_path):
    # Regression for the MCP-server staleness: a long-lived process resolves
    # before onboarding and caches the negative result.
    save_config(PoppyConfig(poppy_dir=tmp_path))  # config.json exists, no key yet
    assert resolve_trags_api_key(load_config(tmp_path)) is None

    # "Another process" runs `poppy setup trags`: the key lands in the keychain
    # and save_config rewrites config.json (a fresh inode), with no in-process
    # cache invalidation reaching the long-lived process.
    keychain.set_secret(_trags_key_account(tmp_path), KEY)
    save_config(PoppyConfig(poppy_dir=tmp_path))

    # The long-lived process re-resolves and must now see the key.
    assert resolve_trags_api_key(load_config(tmp_path)) == KEY


@requires_keyring
def test_unrelated_save_preserves_migrated_key(tmp_path):
    # Key lives in the keychain; a save for an unrelated setting (field is None)
    # must not delete or move it.
    _set_key(tmp_path, KEY)
    cfg = load_config(tmp_path)
    assert cfg.trags_api_key is None
    cfg.set("auto-sync", "off")
    save_config(cfg)
    assert keychain.get_secret(_trags_key_account(tmp_path)) == KEY
    assert resolve_trags_api_key(load_config(tmp_path)) == KEY


@requires_keyring
def test_successful_write_invalidates_cached_miss(tmp_path):
    assert resolve_trags_api_key(PoppyConfig(poppy_dir=tmp_path)) is None
    _set_key(tmp_path, KEY)
    assert resolve_trags_api_key(load_config(tmp_path)) == KEY


@requires_keyring
@pytest.mark.parametrize("already_stored", [False, True])
def test_migration_invalidates_cached_miss_even_if_scrub_fails(tmp_path, monkeypatch, already_stored):
    import poppy.config as config_module

    (tmp_path / "config.json").write_text(json.dumps({"trags_api_key": KEY}))
    assert resolve_trags_api_key(PoppyConfig(poppy_dir=tmp_path)) is None
    if already_stored:
        keychain.set_secret(_trags_key_account(tmp_path), KEY)
    set_secret = Mock(wraps=keychain.set_secret)
    monkeypatch.setattr(keychain, "set_secret", set_secret)

    def fail_save(_config):
        raise OSError("cannot rewrite config")

    monkeypatch.setattr(config_module, "save_config", fail_save)
    cfg = load_config(tmp_path)
    assert resolve_trags_api_key(cfg) == KEY
    assert set_secret.call_count == (0 if already_stored else 1)


@pytest.fixture(params=["raises", "absent", "different"])
def failed_readback(request, monkeypatch):
    """Writes succeed in the isolated keyring, but reads cannot verify them."""
    real_get = keychain.get_secret

    def get_secret(_account):
        if request.param == "raises":
            raise keychain.KeychainUnavailable("interaction not allowed in this session")
        return None if request.param == "absent" else "usr_different_key"

    monkeypatch.setattr(keychain, "get_secret", get_secret)
    return real_get, request.param


@requires_keyring
def test_migration_preserves_plaintext_when_readback_fails(tmp_path, failed_readback):
    path = tmp_path / "config.json"
    original = f'{{"trags_api_key": "{KEY}", "auto_sync": "off"}}\n'
    path.write_text(original)

    cfg = load_config(tmp_path)

    assert path.read_text() == original
    assert cfg.trags_api_key == KEY
    expected = None if failed_readback[1] == "raises" else KEY
    assert failed_readback[0](_trags_key_account(tmp_path)) == expected
    assert trags_api_key_location(tmp_path) == "config-file"
    assert resolve_trags_api_key(cfg) == KEY


@requires_keyring
def test_save_preserves_plaintext_when_readback_fails(tmp_path, failed_readback):
    cfg = _set_key(tmp_path, KEY)

    assert json.loads((tmp_path / "config.json").read_text())["trags_api_key"] == KEY
    expected = None if failed_readback[1] == "raises" else KEY
    assert failed_readback[0](_trags_key_account(tmp_path)) == expected
    assert trags_api_key_location(tmp_path) == "config-file"
    assert resolve_trags_api_key(cfg) == KEY


@requires_keyring
@pytest.mark.parametrize("failed_readback", ["raises"], indirect=True)
def test_repeated_loads_never_write_when_keychain_is_unreadable(tmp_path, monkeypatch, failed_readback):
    (tmp_path / "config.json").write_text(json.dumps({"trags_api_key": KEY}))
    set_secret = Mock(wraps=keychain.set_secret)
    monkeypatch.setattr(keychain, "set_secret", set_secret)

    for _ in range(3):
        assert resolve_trags_api_key(load_config(tmp_path)) == KEY

    set_secret.assert_not_called()
    assert failed_readback[0](_trags_key_account(tmp_path)) is None


# ------------------------- headless fallback (forced) -------------------------


@pytest.fixture
def _no_keychain(monkeypatch):
    """Force the keychain unavailable, as on a headless Linux / CI host."""
    from poppy.errors import KeychainUnavailable

    monkeypatch.setattr(keychain, "available", lambda: False)

    def _unavailable(*_args, **_kwargs):
        raise KeychainUnavailable("no backend in this test")

    monkeypatch.setattr(keychain, "get_secret", _unavailable)
    monkeypatch.setattr(keychain, "set_secret", _unavailable)


def test_fallback_writes_plaintext_when_no_keychain(tmp_path, _no_keychain):
    _set_key(tmp_path, KEY)
    # With no backend, the key falls back to config.json (0600) as documented.
    text = (tmp_path / "config.json").read_text()
    assert KEY in text
    assert resolve_trags_api_key(load_config(tmp_path)) == KEY
    assert trags_api_key_location(tmp_path) == "config-file"


def test_migration_is_noop_when_no_keychain(tmp_path, _no_keychain):
    (tmp_path / "config.json").write_text(f'{{"trags_api_key": "{KEY}"}}')
    cfg = load_config(tmp_path)
    # Left in place (still resolvable), not lost, and no error raised.
    assert cfg.trags_api_key == KEY
    assert KEY in (tmp_path / "config.json").read_text()
    assert resolve_trags_api_key(cfg) == KEY


def test_env_override_wins_without_keychain(tmp_path, _no_keychain, monkeypatch):
    (tmp_path / "config.json").write_text(f'{{"trags_api_key": "{KEY}"}}')
    monkeypatch.setenv(TRAGS_API_KEY_ENV, "usr_from_env")
    assert resolve_trags_api_key(load_config(tmp_path)) == "usr_from_env"


def test_location_reports_none_when_no_backend(tmp_path, _no_keychain):
    # No backend at all (headless host, nothing configured) is "none", not a warning.
    save_config(PoppyConfig(poppy_dir=tmp_path))
    assert trags_api_key_location(tmp_path) == "none"


@requires_keyring
def test_location_reports_unreadable_when_backend_read_raises(tmp_path, monkeypatch):
    # A backend exists but this session cannot read it (macOS Background session, -25308).
    def unreadable(_account):
        raise keychain.KeychainUnavailable("interaction not allowed")

    monkeypatch.setattr(keychain, "get_secret", unreadable)
    save_config(PoppyConfig(poppy_dir=tmp_path))
    assert trags_api_key_location(tmp_path) == "keychain-unreadable"


@requires_keyring
def test_location_reports_absent_key(tmp_path):
    assert trags_api_key_location(tmp_path) == "none"


def test_resolve_reports_no_key_for_an_explicitly_empty_value(tmp_path):
    # `trags_api_key == ""` is a user's clear, not a value. With no env override
    # and nothing in the keychain for this store, resolution has to report "no
    # key" rather than hand back the empty string it was given.
    cfg = PoppyConfig(poppy_dir=tmp_path, trags_api_key="")

    assert resolve_trags_api_key(cfg) is None


# --------------------------------- CLI path ----------------------------------


def test_config_set_never_echoes_the_secret(tmp_path, monkeypatch):
    # Whatever backend is present, the raw key must never reach stdout/scrollback,
    # and it must still resolve afterwards.
    from click.testing import CliRunner

    from poppy.cli.main import cli

    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setenv("POPPY_TELEMETRY_OFF", "1")
    res = CliRunner().invoke(cli, ["config", "set", "trags-api-key", KEY])
    assert res.exit_code == 0, res.output
    assert KEY not in res.output
    assert resolve_trags_api_key(load_config(tmp_path)) == KEY


@requires_keyring
@pytest.mark.parametrize("env_override", [False, True])
def test_config_set_clear_reports_cleared(tmp_path, monkeypatch, env_override):
    from click.testing import CliRunner

    from poppy.cli.main import cli

    _set_key(tmp_path, KEY)
    if env_override:
        monkeypatch.setenv(TRAGS_API_KEY_ENV, "usr_from_env")
    res = CliRunner().invoke(cli, ["config", "set", "trags-api-key", ""], env={"POPPY_DIR": str(tmp_path)})
    assert res.exit_code == 0, res.output
    assert "Cleared trags-api-key." in res.output
    assert "could not be read back" not in res.output
    assert KEY not in res.output
    assert "usr_from_env" not in res.output
    note = "Note: POPPY_TRAGS_API_KEY is set in your environment and overrides the stored key."
    assert (note in res.output) == env_override
    monkeypatch.delenv(TRAGS_API_KEY_ENV, raising=False)
    assert resolve_trags_api_key(load_config(tmp_path)) is None


@requires_keyring
@pytest.mark.parametrize("unreadable", [False, True])
@pytest.mark.parametrize("env_override", [False, True])
def test_config_set_clear_reports_surviving_keychain_copy(tmp_path, monkeypatch, unreadable, env_override):
    from click.testing import CliRunner

    from poppy.cli.main import cli

    _set_key(tmp_path, KEY)
    monkeypatch.setattr(keychain, "delete_secret", lambda _account: None)

    def get_secret(_account):
        if unreadable:
            raise keychain.KeychainUnavailable("interaction not allowed")
        return KEY

    monkeypatch.setattr(keychain, "get_secret", get_secret)
    if env_override:
        monkeypatch.setenv(TRAGS_API_KEY_ENV, "usr_from_env")
    res = CliRunner().invoke(cli, ["config", "set", "trags-api-key", ""], env={"POPPY_DIR": str(tmp_path)})

    if unreadable:
        assert res.exit_code == 0, res.output
        assert (
            "Cleared trags-api-key from config.json. This session cannot read the OS keychain, "
            "so a keychain copy may survive; re-run this command from an interactive login "
            "session to be sure."
        ) in res.output
    else:
        assert res.exit_code != 0, res.output
        assert (
            "Could not remove trags-api-key from the OS keychain; the key is still active. "
            "Re-run from an interactive login session or remove the 'poppy-memory' item "
            "in your keychain manager."
        ) in res.output
        monkeypatch.delenv(TRAGS_API_KEY_ENV, raising=False)
        assert resolve_trags_api_key(load_config(tmp_path)) == KEY
    assert "Cleared trags-api-key." not in res.output
    assert KEY not in res.output
    assert "usr_from_env" not in res.output
    assert not json.loads((tmp_path / "config.json").read_text()).get("trags_api_key")
    note = "Note: POPPY_TRAGS_API_KEY is set in your environment and overrides the stored key."
    assert (note in res.output) == env_override


def _refusing_backend(monkeypatch, method: str, error: Exception):
    """Make the in-memory keyring backend refuse one operation, as a locked
    login keychain does in a background session (-25308)."""
    import keyring

    def refuse(_self, *_args, **_kwargs):
        raise error

    monkeypatch.setattr(type(keyring.get_keyring()), method, refuse)


@requires_keyring
def test_delete_secret_treats_a_missing_entry_as_deleted():
    # The backend raises PasswordDeleteError for an absent entry; that is a
    # success from Poppy's side, not a failure.
    keychain.delete_secret("trags-api-key:never-stored")


@requires_keyring
@pytest.mark.parametrize("readable", [True, False])
def test_delete_secret_raises_when_the_entry_survives(monkeypatch, readable):
    from keyring.errors import PasswordDeleteError

    account = "trags-api-key:/tmp/does-not-matter"
    keychain.set_secret(account, KEY)
    _refusing_backend(monkeypatch, "delete_password", PasswordDeleteError("interaction not allowed"))
    if not readable:
        # Cannot read back either, so Poppy cannot claim the entry is gone.
        _refusing_backend(monkeypatch, "get_password", RuntimeError("interaction not allowed"))

    with pytest.raises(keychain.KeychainUnavailable, match="could not delete from the OS keychain"):
        keychain.delete_secret(account)


@requires_keyring
def test_delete_secret_raises_when_an_accepted_delete_kept_the_entry(monkeypatch):
    # A backend can accept the call and keep the entry (a Secret Service
    # collection that is locked answers rather than raises), so the read-back is
    # what proves the removal, not the delete's own return.
    import keyring

    account = "trags-api-key:/tmp/kept-after-delete"
    keychain.set_secret(account, KEY)
    monkeypatch.setattr(type(keyring.get_keyring()), "delete_password", lambda *_a, **_k: None)

    with pytest.raises(keychain.KeychainUnavailable, match="still readable afterwards"):
        keychain.delete_secret(account)
    assert keychain.get_secret(account) == KEY


def test_config_set_clear_is_clean_without_a_keychain_backend(tmp_path, monkeypatch, _no_keychain):
    # Headless host: the key is in config.json because no backend is usable, so
    # every keychain call fails. Clearing it is a clean clear -- a delete that
    # fails for want of a keychain is not a surviving secret.
    from click.testing import CliRunner

    from poppy.cli.main import cli

    cfg = PoppyConfig(poppy_dir=tmp_path)
    cfg.set("trags-api-key", KEY)
    save_config(cfg)
    assert trags_api_key_location(tmp_path) == "config-file"
    _refusing_backend(monkeypatch, "delete_password", RuntimeError("no backend here"))
    _refusing_backend(monkeypatch, "get_password", RuntimeError("no backend here"))

    res = CliRunner().invoke(cli, ["config", "set", "trags-api-key", ""], env={"POPPY_DIR": str(tmp_path)})

    assert res.exit_code == 0, res.output
    assert "Cleared trags-api-key." in res.output
    assert "still active" not in res.output
    assert KEY not in res.output
    assert resolve_trags_api_key(load_config(tmp_path)) is None


def _rejecting_key_writes(monkeypatch):
    """A backend that refuses to write this store's key entry and nothing else.

    The PR #13 review case: reads of `trags-api-key:<store>` work and the
    throwaway `session-probe:*` entry works, so a probe run after the fact calls
    the session healthy. Only the write that was actually refused knows better.
    """
    import keyring

    backend = type(keyring.get_keyring())
    real_set = backend.set_password

    def set_password(self, service, username, password):
        if username.startswith("trags-api-key:"):
            raise RuntimeError("interaction not allowed for this credential")
        return real_set(self, service, username, password)

    monkeypatch.setattr(backend, "set_password", set_password)


@requires_keyring
@pytest.mark.parametrize("readable", [True, False])
def test_config_set_clear_reports_a_refused_keychain_delete(tmp_path, monkeypatch, readable):
    # A backend that refuses the delete must not report a clean clear.
    # With reads working the surviving entry is proven by re-reading the
    # location; with reads refused too, the refusal the delete itself hit is the
    # only evidence there is, and it still has to reach the user.
    from click.testing import CliRunner
    from keyring.errors import PasswordDeleteError

    from poppy.cli.main import cli

    _set_key(tmp_path, KEY)
    _refusing_backend(monkeypatch, "delete_password", PasswordDeleteError("interaction not allowed"))
    if not readable:
        _refusing_backend(monkeypatch, "get_password", RuntimeError("interaction not allowed"))

    res = CliRunner().invoke(cli, ["config", "set", "trags-api-key", ""], env={"POPPY_DIR": str(tmp_path)})

    assert res.exit_code != 0, res.output
    still = "the key is still active" if readable else "the key may still be active"
    assert (
        f"Could not remove trags-api-key from the OS keychain; {still}. "
        "Re-run from an interactive login session or remove the 'poppy-memory' item "
        "in your keychain manager."
    ) in res.output
    assert "Cleared trags-api-key." not in res.output
    assert KEY not in res.output
    if readable:
        # The warning is true: the surviving entry still resolves.
        assert resolve_trags_api_key(load_config(tmp_path)) == KEY


@requires_keyring
def test_config_set_reports_a_rejected_keychain_write(tmp_path, monkeypatch):
    # Reads work, the write is refused: the message must name the write, not
    # blame read-back.
    from click.testing import CliRunner

    from poppy.cli.main import cli

    _refusing_backend(monkeypatch, "set_password", RuntimeError("interaction not allowed"))

    res = CliRunner().invoke(cli, ["config", "set", "trags-api-key", KEY], env={"POPPY_DIR": str(tmp_path)})

    assert res.exit_code == 0, res.output
    assert "The OS keychain rejected the write." in res.output
    assert "could not be read back" not in res.output
    assert "No OS keychain backend" not in res.output
    assert str(tmp_path / "config.json") in res.output
    assert KEY not in res.output
    assert trags_api_key_location(tmp_path) == "config-file"


@requires_keyring
def test_config_set_names_the_rejected_write_when_a_probe_would_pass(tmp_path, monkeypatch):
    # Only this store's key entry is refused, so a probe run afterwards writes,
    # reads and deletes its own entry happily and would report a healthy
    # session. The reason has to come from the write that failed.
    from click.testing import CliRunner

    from poppy.cli.main import cli

    keychain.set_secret(_trags_key_account(tmp_path), "usr_previous_key")
    _rejecting_key_writes(monkeypatch)
    assert keychain.probe() == keychain.PROBE_OK

    res = CliRunner().invoke(cli, ["config", "set", "trags-api-key", KEY], env={"POPPY_DIR": str(tmp_path)})

    assert res.exit_code == 0, res.output
    assert "The OS keychain rejected the write." in res.output
    assert "could not be read back" not in res.output
    assert "could not be used for the key in this session" not in res.output
    assert str(tmp_path / "config.json") in res.output
    assert KEY not in res.output
    assert trags_api_key_location(tmp_path) == "config-file"


@requires_keyring
def test_config_set_does_not_claim_the_keychain_when_the_location_is_unknown(tmp_path, monkeypatch):
    # A guard for an unreachable combination, not a reachable path: after a
    # successful non-empty set, `keychain-unreadable` cannot happen (a keychain
    # write that passed read-back proves reads work), so the state is
    # synthesized here. The branch it covers replaced one that asserted the key
    # was in the keychain; an unconfirmed location claims nothing.
    from click.testing import CliRunner

    import poppy.config as config_module
    from poppy.cli.main import cli

    monkeypatch.setattr(config_module, "trags_api_key_location", lambda _poppy_dir: "keychain-unreadable")

    res = CliRunner().invoke(cli, ["config", "set", "trags-api-key", KEY], env={"POPPY_DIR": str(tmp_path)})

    assert res.exit_code == 0, res.output
    assert "Set trags-api-key. This session could not confirm where the key was stored" in res.output
    assert "stored in the OS keychain" not in res.output
    assert "plaintext" not in res.output
    assert KEY not in res.output


@requires_keyring
def test_config_set_reports_readback_failure(tmp_path, failed_readback):
    from click.testing import CliRunner

    from poppy.cli.main import cli

    res = CliRunner().invoke(cli, ["config", "set", "trags-api-key", KEY], env={"POPPY_DIR": str(tmp_path)})
    assert res.exit_code == 0, res.output
    # The message must not blame the read for a write the backend
    # refused; it names the step that failed
    # rather than hedging over both. This fixture fails the read back, so that
    # is the step the copy has to name.
    assert "The OS keychain could not be read back in this session." in res.output
    assert "rejected the write" not in res.output
    assert "No OS keychain backend" not in res.output
    assert KEY not in res.output
    assert trags_api_key_location(tmp_path) == "config-file"


def test_config_set_reports_no_backend(tmp_path, _no_keychain):
    from click.testing import CliRunner

    from poppy.cli.main import cli

    res = CliRunner().invoke(cli, ["config", "set", "trags-api-key", KEY], env={"POPPY_DIR": str(tmp_path)})
    assert res.exit_code == 0, res.output
    assert "No OS keychain backend is available here." in res.output
    assert "could not be read back" not in res.output
    assert KEY not in res.output
    assert resolve_trags_api_key(load_config(tmp_path)) == KEY


@pytest.mark.parametrize("unreadable", [False, True])
def test_sync_status_distinguishes_unreadable_from_absent(tmp_path, monkeypatch, unreadable):
    from click.testing import CliRunner

    from poppy.cli.main import cli

    def get_secret(_account):
        if unreadable:
            raise keychain.KeychainUnavailable("interaction not allowed")
        return None

    monkeypatch.setattr(keychain, "get_secret", get_secret)
    res = CliRunner().invoke(cli, ["sync", "status"], env={"POPPY_DIR": str(tmp_path)})
    assert res.exit_code == 0, res.output
    if unreadable:
        assert (
            "This session cannot read the OS keychain, so it cannot tell whether a Trags key is stored there."
            in res.output
        )
        assert "interactive login session" in res.output
        assert "POPPY_TRAGS_API_KEY for background jobs" in res.output
        assert "run `poppy setup trags` if you have not configured Trags yet" in res.output
        assert "Trags not configured" not in res.output
    else:
        assert "Trags not configured. Run `poppy setup trags` to connect this machine" in res.output
        assert "(or `poppy config set trags-api-key <usr_xxxxx>` if you already have a key)." in res.output
        # The sign-in path is named before the raw-key path.
        assert res.output.index("poppy setup trags") < res.output.index("poppy config set")


@requires_keyring
@pytest.mark.parametrize("source", ["env", "keychain", "config-file", "keychain-unreadable", "none"])
def test_doctor_reports_trags_key_source(tmp_path, monkeypatch, source):
    from click.testing import CliRunner

    from poppy.cli.main import cli

    cfg = PoppyConfig(poppy_dir=tmp_path, engine="seed", update_check=False)
    save_config(cfg)
    if source == "keychain":
        keychain.set_secret(_trags_key_account(tmp_path), KEY)
    elif source != "none":

        def unreadable(_account):
            raise keychain.KeychainUnavailable("interaction not allowed")

        monkeypatch.setattr(keychain, "get_secret", unreadable)
        if source == "config-file":
            # A backend exists but cannot be read back, so save_config leaves
            # the key in the file and records the read-back failure, which is
            # the reason doctor reports; pin `available` so it is deterministic.
            monkeypatch.setattr(keychain, "available", lambda: True)
            cfg.trags_api_key = KEY
            save_config(cfg)
        elif source == "env":
            monkeypatch.setenv(TRAGS_API_KEY_ENV, KEY)

    res = CliRunner().invoke(cli, ["doctor"], env={"POPPY_DIR": str(tmp_path), "HOME": str(tmp_path)})
    assert res.exit_code == 0, res.output
    key_line = next(line for line in res.output.splitlines() if "Trags key:" in line)
    if source == "keychain-unreadable":
        assert "Trags key: WARN" in key_line
        assert "keychain (unreadable in this session; cannot tell whether a key is stored)" in key_line
        assert "interactive login session" in key_line
        assert "POPPY_TRAGS_API_KEY for background jobs" in key_line
    elif source == "none":
        assert "Trags key: OK" in key_line
        assert "none (Trags sync not configured; optional)" in key_line
    elif source == "config-file":
        assert "Trags key: WARN" in key_line
        # save_config recorded which step failed, so doctor names it rather than
        # hedging over every way the keychain could have gone wrong.
        reason = "the OS keychain could not be read back in this session"
        assert f"config.json (plaintext, file mode 0600; {reason})" in key_line
        assert "POPPY_TRAGS_API_KEY" in key_line
        assert "interactive login session" in key_line
    else:
        assert "Trags key: OK" in key_line
        detail = {"env": "POPPY_TRAGS_API_KEY env", "keychain": "keychain"}[source]
        assert detail in key_line
    assert KEY not in res.output


@requires_keyring
def test_doctor_offers_to_move_a_plaintext_key_when_the_keychain_works(tmp_path):
    # The state seen from a healthy session: the key fell back to the
    # file in an earlier one (a background job), and this session can use the
    # keychain. Doctor must not claim a failure that is not happening here.
    from click.testing import CliRunner

    from poppy.cli.main import cli

    save_config(PoppyConfig(poppy_dir=tmp_path, engine="seed", update_check=False))
    # Written straight to the file: save_config would migrate it to the working
    # keychain, which is the state this test needs to keep out of.
    data = json.loads((tmp_path / "config.json").read_text())
    data["trags_api_key"] = KEY
    (tmp_path / "config.json").write_text(json.dumps(data))
    assert trags_api_key_location(tmp_path) == "config-file"

    res = CliRunner().invoke(cli, ["doctor"], env={"POPPY_DIR": str(tmp_path), "HOME": str(tmp_path)})

    assert res.exit_code == 0, res.output
    key_line = next(line for line in res.output.splitlines() if "Trags key:" in line)
    assert "Trags key: WARN" in key_line
    assert "the OS keychain is usable in this session, so the key can be moved into it" in key_line
    assert "could not be read back" not in key_line
    assert KEY not in res.output


@requires_keyring
def test_doctor_names_a_rejected_write_like_config_set(tmp_path, monkeypatch):
    # doctor must not blame read-back for a write the backend rejected: it reads
    # the classification `config set` recorded, and falls back to the same probe
    # vocabulary in a fresh process where there is nothing recorded.
    from click.testing import CliRunner

    from poppy.cli.main import cli

    save_config(PoppyConfig(poppy_dir=tmp_path, engine="seed", update_check=False))
    _rejecting_key_writes(monkeypatch)
    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path), "HOME": str(tmp_path)}

    set_res = runner.invoke(cli, ["config", "set", "trags-api-key", KEY], env=env)
    assert set_res.exit_code == 0, set_res.output
    assert "The OS keychain rejected the write." in set_res.output

    res = runner.invoke(cli, ["doctor"], env=env)

    assert res.exit_code == 0, res.output
    key_line = next(line for line in res.output.splitlines() if "Trags key:" in line)
    assert "Trags key: WARN" in key_line
    assert "config.json (plaintext, file mode 0600; the OS keychain rejected the write)" in key_line
    assert "could not be read back" not in key_line
    assert KEY not in res.output


def test_doctor_warns_on_plaintext_key_without_backend(tmp_path, _no_keychain):
    # Headless host: the key fell back to config.json because no backend is
    # usable, and doctor has to name that reason.
    from click.testing import CliRunner

    from poppy.cli.main import cli

    cfg = PoppyConfig(poppy_dir=tmp_path, engine="seed", update_check=False)
    cfg.trags_api_key = KEY
    save_config(cfg)
    assert trags_api_key_location(tmp_path) == "config-file"

    res = CliRunner().invoke(cli, ["doctor"], env={"POPPY_DIR": str(tmp_path), "HOME": str(tmp_path)})
    assert res.exit_code == 0, res.output
    key_line = next(line for line in res.output.splitlines() if "Trags key:" in line)
    assert "Trags key: WARN" in key_line
    assert "config.json (plaintext, file mode 0600; no usable OS keychain backend here)" in key_line
    assert "interactive login session" in key_line
    assert KEY not in res.output


def test_doctor_warns_on_plaintext_key_under_env_override(tmp_path, _no_keychain, monkeypatch):
    # POPPY_TRAGS_API_KEY decides which key is USED; it does not remove the
    # plaintext copy from disk. The warning therefore has to survive the very
    # hint it prints, or following that hint just hides the exposure.
    from click.testing import CliRunner

    from poppy.cli.main import cli

    cfg = PoppyConfig(poppy_dir=tmp_path, engine="seed", update_check=False)
    cfg.trags_api_key = KEY
    save_config(cfg)
    assert trags_api_key_location(tmp_path) == "config-file"
    monkeypatch.setenv(TRAGS_API_KEY_ENV, "usr_from_env")

    res = CliRunner().invoke(cli, ["doctor"], env={"POPPY_DIR": str(tmp_path), "HOME": str(tmp_path)})
    assert res.exit_code == 0, res.output
    key_line = next(line for line in res.output.splitlines() if "Trags key:" in line)
    assert "Trags key: WARN" in key_line
    assert "config.json (plaintext" in key_line
    assert f"{TRAGS_API_KEY_ENV} is set and takes precedence" in key_line
    # The usual "set the env var" advice is already satisfied, so the hint has
    # to name the one action left: remove the copy the env var shadows.
    assert 'poppy config set trags-api-key ""' in key_line
    assert "Prefer POPPY_TRAGS_API_KEY in the environment" not in key_line
    assert KEY not in res.output
    assert "usr_from_env" not in res.output


@pytest.mark.parametrize("mode", [0o600, 0o644])
def test_doctor_reports_the_real_config_file_mode(tmp_path, _no_keychain, mode):
    # The mode used to be asserted in the message rather than checked. Nothing
    # tightens a legacy or hand-edited config.json while the key stays in it, so
    # doctor has to stat the file and say what it finds.
    from click.testing import CliRunner

    from poppy.cli.main import cli

    cfg = PoppyConfig(poppy_dir=tmp_path, engine="seed", update_check=False)
    cfg.trags_api_key = KEY
    save_config(cfg)
    config_path = tmp_path / "config.json"
    config_path.chmod(mode)

    res = CliRunner().invoke(cli, ["doctor"], env={"POPPY_DIR": str(tmp_path), "HOME": str(tmp_path)})
    assert res.exit_code == 0, res.output
    assert config_path.stat().st_mode & 0o777 == mode, "doctor must not rewrite config.json"
    key_line = next(line for line in res.output.splitlines() if "Trags key:" in line)
    assert "Trags key: WARN" in key_line
    assert f"file mode {mode:04o}" in key_line
    if mode == 0o600:
        assert "readable beyond the owner" not in key_line
        assert "chmod 600" not in key_line
    else:
        assert "readable beyond the owner" in key_line
        assert f"chmod 600 {config_path}" in key_line
    assert KEY not in res.output
