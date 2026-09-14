"""Direct unit tests for the OS keychain helpers.

The autouse ``_isolate_keychain`` conftest fixture swaps in an in-memory keyring
backend, so nothing here touches the real OS keychain. ``writable`` is the
interesting one: it is the only honest answer to "may this session use the
keychain", because on macOS a real backend loads in an ssh, launchd or
background session and then refuses every call.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("keyring")

from poppy import keychain  # noqa: E402
from poppy.errors import KeychainUnavailable  # noqa: E402


def _stored_probe() -> str | None:
    """What the backend is still holding under the probe account, if anything."""
    import keyring

    return keyring.get_password(keychain.SERVICE, keychain._PROBE_ACCOUNT)


def test_available_is_false_for_the_failing_backend(monkeypatch):
    """``available`` only classifies the backend; it never touches it."""
    import keyring
    from keyring.backends import fail

    monkeypatch.setattr(keyring, "get_keyring", lambda: fail.Keyring())
    assert keychain.available() is False


def test_writable_round_trips_and_deletes_its_probe():
    assert keychain.writable() is True
    assert _stored_probe() is None


def test_writable_uses_one_fixed_account_so_a_hard_kill_strands_at_most_one(monkeypatch):
    """A per-process account name leaves one dead entry per killed pid; pin the fix.

    Nothing enumerates or sweeps probe entries, so the only bound on the litter
    is that every probe writes the same account and the next one overwrites it.
    """
    accounts: list[str] = []
    real_set = keychain.set_secret

    def _set(account: str, value: str) -> None:
        accounts.append(account)
        real_set(account, value)

    monkeypatch.setattr(keychain, "set_secret", _set)

    assert keychain.writable() is True
    assert accounts == [keychain._PROBE_ACCOUNT]
    assert str(os.getpid()) not in keychain._PROBE_ACCOUNT


def test_writable_is_false_without_a_backend_and_writes_nothing(monkeypatch):
    writes: list[str] = []
    monkeypatch.setattr(keychain, "available", lambda: False)
    monkeypatch.setattr(keychain, "set_secret", lambda account, value: writes.append(account))

    assert keychain.writable() is False
    assert writes == []


def test_writable_is_false_when_the_write_is_refused(monkeypatch):
    """errSecInteractionNotAllowed on the write: a locked login keychain."""

    def _refuse(_account: str, _value: str) -> None:
        raise KeychainUnavailable("could not write to the OS keychain: interaction not allowed (-25308)")

    monkeypatch.setattr(keychain, "set_secret", _refuse)
    assert keychain.writable() is False


def test_writable_is_false_when_the_read_back_is_refused_and_still_cleans_up(monkeypatch):
    """A backend may accept the write and refuse the read; the entry still goes."""

    def _refuse(_account: str) -> str | None:
        raise KeychainUnavailable("could not read from the OS keychain: interaction not allowed (-25308)")

    monkeypatch.setattr(keychain, "get_secret", _refuse)

    assert keychain.writable() is False
    assert _stored_probe() is None


def test_writable_is_false_when_the_value_reads_back_as_something_else(monkeypatch):
    """A store handing back a foreign value is not one to put the key in."""
    monkeypatch.setattr(keychain, "get_secret", lambda _account: "not a poppy probe")
    assert keychain.writable() is False


def test_writable_is_false_when_the_entry_never_persists(monkeypatch):
    """A write that reports success and stores nothing is reported, not retried away."""
    reads: list[None] = []

    def _get(_account: str) -> str | None:
        reads.append(None)
        return None

    monkeypatch.setattr(keychain, "get_secret", _get)

    assert keychain.writable() is False
    assert len(reads) == 2  # the one retry, then an honest no


def test_writable_accepts_another_processes_probe_value(monkeypatch):
    """A second Poppy probing at the same moment is not a locked keychain.

    A session that may not use the keychain is refused with an exception; it
    never gets somebody else's value back. So any probe value proves the
    round-trip this is testing for.
    """
    monkeypatch.setattr(keychain, "get_secret", lambda _account: f"{keychain._PROBE_VALUE_PREFIX}999999")
    assert keychain.writable() is True


def test_writable_retries_once_when_the_entry_is_deleted_underneath(monkeypatch):
    """A concurrent probe's delete landing between our write and read is not a no."""
    real_get = keychain.get_secret
    reads: list[str | None] = [None]

    def _get(account: str) -> str | None:
        return reads.pop(0) if reads else real_get(account)

    monkeypatch.setattr(keychain, "get_secret", _get)

    assert keychain.writable() is True
    assert reads == []
    assert _stored_probe() is None
