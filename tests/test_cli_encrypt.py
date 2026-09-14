"""CLI tests for `poppy encrypt status|enable|disable`.

POPPY_DIR points at a temp dir and the keychain is stubbed in-memory (autouse
conftest fixtures), so neither the real ~/.poppy nor the OS keychain is touched.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

pytest.importorskip("sqlcipher3")
pytest.importorskip("keyring")

from poppy import encryption, keychain  # noqa: E402
from poppy.cli.main import cli  # noqa: E402
from poppy.db import connect  # noqa: E402
from poppy.engine.seed import SeedEngine  # noqa: E402
from poppy.models import Memory, Source  # noqa: E402


def _env(tmp_path: Path) -> dict:
    return {"POPPY_DIR": str(tmp_path), "POPPY_TELEMETRY_OFF": "1"}


def _seed(tmp_path: Path, content: str = "hello") -> None:
    now = datetime.now(timezone.utc)
    engine = SeedEngine(db_path=tmp_path / "memories.db")
    engine.ingest(
        Memory(
            id="m1",
            content=content,
            memory_type="fact",
            source=Source(type="manual", session_id=None, timestamp=now),
            project=None,
            related_to=[],
            created_at=now,
            updated_at=now,
        )
    )
    engine._conn.close()


def test_status_off_by_default(tmp_path):
    result = CliRunner().invoke(cli, ["encrypt", "status"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "Encryption: off" in result.output
    assert "dependencies: installed" in result.output


def test_bare_encrypt_shows_status(tmp_path):
    result = CliRunner().invoke(cli, ["encrypt"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "Encryption: off" in result.output


def test_enable_then_status_on(tmp_path):
    _seed(tmp_path, "migrate me")
    runner = CliRunner()

    result = runner.invoke(cli, ["encrypt", "enable", "--yes"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "Encryption enabled" in result.output

    assert encryption.database_is_encrypted(tmp_path / "memories.db") is True
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["encrypted"] is True

    status = runner.invoke(cli, ["encrypt", "status"], env=_env(tmp_path))
    assert "Encryption: on" in status.output
    assert "encrypted on disk: yes" in status.output


def test_enable_confirmation_prompt_accepts_input(tmp_path):
    _seed(tmp_path)
    result = CliRunner().invoke(cli, ["encrypt", "enable"], env=_env(tmp_path), input="y\n")
    assert result.exit_code == 0, result.output
    assert encryption.database_is_encrypted(tmp_path / "memories.db") is True


def test_enable_abort_leaves_plaintext(tmp_path):
    _seed(tmp_path)
    result = CliRunner().invoke(cli, ["encrypt", "enable"], env=_env(tmp_path), input="n\n")
    assert result.exit_code != 0  # click.confirm(abort=True) exits non-zero
    assert encryption.database_is_encrypted(tmp_path / "memories.db") is False


def test_enable_twice_is_gentle(tmp_path):
    _seed(tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["encrypt", "enable", "--yes"], env=_env(tmp_path))
    result = runner.invoke(cli, ["encrypt", "enable", "--yes"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "already encrypted" in result.output


def test_disable_returns_plaintext(tmp_path):
    _seed(tmp_path, "roundtrip")
    runner = CliRunner()
    runner.invoke(cli, ["encrypt", "enable", "--yes"], env=_env(tmp_path))

    result = runner.invoke(cli, ["encrypt", "disable", "--yes"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "Encryption disabled" in result.output

    db = tmp_path / "memories.db"
    assert encryption.database_is_encrypted(db) is False
    raw = sqlite3.connect(str(db))
    try:
        assert raw.execute("SELECT content FROM memories WHERE id='m1'").fetchone()[0] == "roundtrip"
    finally:
        raw.close()


def test_disable_when_off_is_gentle(tmp_path):
    _seed(tmp_path)
    result = CliRunner().invoke(cli, ["encrypt", "disable", "--yes"], env=_env(tmp_path))
    assert result.exit_code == 0
    assert "not encrypted" in result.output


def test_doctor_reports_encryption(tmp_path):
    _seed(tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["encrypt", "enable", "--yes"], env=_env(tmp_path))
    result = runner.invoke(cli, ["doctor"], env=_env(tmp_path))
    assert "encryption at rest" in result.output
    assert "key in OS keychain" in result.output


def test_doctor_reports_env_key_source(tmp_path):
    # With POPPY_DB_KEY set, doctor must not claim the key is in the keychain.
    env = _env(tmp_path)
    env["POPPY_DB_KEY"] = "ab" * 32
    _seed(tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["encrypt", "enable", "--yes"], env=env)
    result = runner.invoke(cli, ["doctor"], env=env)
    assert "POPPY_DB_KEY env var" in result.output
    assert "key in OS keychain" not in result.output


def test_status_and_repair_reconcile_stale_sentinel(tmp_path):
    _seed(tmp_path)
    (tmp_path / ".poppy-encrypted").touch()  # stale marker, plaintext on disk
    runner = CliRunner()

    status = runner.invoke(cli, ["encrypt", "status"], env=_env(tmp_path))
    assert "inconsistent state" in status.output

    repair = runner.invoke(cli, ["encrypt", "repair"], env=_env(tmp_path))
    assert repair.exit_code == 0
    assert "stale encryption marker" in repair.output
    assert not (tmp_path / ".poppy-encrypted").exists()


def test_status_shows_pipx_hint_message(tmp_path, monkeypatch):
    # Force the deps-missing branch to check the install hint copy.
    from poppy import encryption

    monkeypatch.setattr(encryption, "dependencies_installed", lambda: False)
    result = CliRunner().invoke(cli, ["encrypt", "status"], env=_env(tmp_path))
    assert "pipx inject poppy-memory sqlcipher3" in result.output


def test_status_reports_malformed_env_key(tmp_path):
    _seed(tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["encrypt", "enable", "--yes"], env={**_env(tmp_path), "POPPY_DB_KEY": "ab" * 32})
    # A malformed POPPY_DB_KEY shadows the keychain; status must say so, not
    # advise restoring the keychain.
    result = runner.invoke(cli, ["encrypt", "status"], env={**_env(tmp_path), "POPPY_DB_KEY": "not-hex"})
    assert "MALFORMED" in result.output
    assert "key in keychain" not in result.output


def test_doctor_reports_malformed_env_key(tmp_path):
    _seed(tmp_path)
    runner = CliRunner()
    runner.invoke(cli, ["encrypt", "enable", "--yes"], env={**_env(tmp_path), "POPPY_DB_KEY": "cd" * 32})
    result = runner.invoke(cli, ["doctor"], env={**_env(tmp_path), "POPPY_DB_KEY": "bad-key"})
    assert "POPPY_DB_KEY is malformed" in result.output


# --- The key write is verified by a read-back ----------------------


class _FakeKeyring:
    """An in-memory keyring with scripted reads, recording writes and deletes.

    ``on_read`` is handed this fake and returns what the read answers with, or
    raises to stand in for a backend that refuses the read.

    ``keychain.writable`` round-trips a throwaway probe entry before the key is
    touched. That probe is answered from its own slot and kept out of
    ``writes``, ``deletes`` and ``stored``: these tests script what happens to
    the *key*, and a keychain that round-trips a probe is the precondition they
    are meant to run under.
    """

    def __init__(self, on_read) -> None:
        self._on_read = on_read
        self.stored: dict[tuple[str, str], str] = {}
        self.writes: list[tuple[str, str]] = []
        self.deletes: list[tuple[str, str]] = []
        self._probe: dict[tuple[str, str], str] = {}

    def install(self, monkeypatch) -> "_FakeKeyring":
        import keyring

        monkeypatch.setattr(keyring, "get_password", self.get_password)
        monkeypatch.setattr(keyring, "set_password", self.set_password)
        monkeypatch.setattr(keyring, "delete_password", self.delete_password)
        return self

    @staticmethod
    def _is_probe(username: str) -> bool:
        return username.startswith(keychain._PROBE_ACCOUNT)

    def get_password(self, service: str, username: str) -> str | None:
        if self._is_probe(username):
            return self._probe.get((service, username))
        return self._on_read(self)

    def set_password(self, service: str, username: str, password: str) -> None:
        if self._is_probe(username):
            self._probe[(service, username)] = password
            return
        self.writes.append((service, username))
        self.stored[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        if self._is_probe(username):
            self._probe.pop((service, username), None)
            return
        self.deletes.append((service, username))
        self.stored.pop((service, username), None)


def _refuse_reads_after_the_write(fake: _FakeKeyring) -> str | None:
    """Answer the pre-write probe, then refuse every read, as -25308 does."""
    if fake.writes:
        from keyring.errors import KeyringError

        raise KeyringError("interaction not allowed (-25308)")
    return None


def test_enable_refuses_when_the_key_cannot_be_read_back(tmp_path, monkeypatch):
    """A backend that accepts the write but denies the read-back encrypts nothing.

    Stands in for a macOS Background session, where the keychain item is written
    but reads are refused with errSecInteractionNotAllowed (-25308). The pre-write
    probe answers here (nothing is stored yet), so the refusal can only come from
    the post-write read-back gate.
    """
    backend = _FakeKeyring(_refuse_reads_after_the_write).install(monkeypatch)
    _seed(tmp_path, "stay plaintext")

    result = CliRunner().invoke(cli, ["encrypt", "enable", "--yes"], env=_env(tmp_path))

    account = f"db-key:{tmp_path.resolve()}"
    # The write went through, so the refusal is the read-back gate and nothing earlier.
    assert backend.writes == [(keychain.SERVICE, account)]
    assert result.exit_code != 0
    assert "the write succeeded but reading it back failed" in result.output
    assert "could not be verified in the OS keychain" in result.output
    assert "POPPY_DB_KEY" in result.output
    assert "interactive login session" in result.output
    # No store was created: still plaintext, no marker, no enabled flag.
    assert encryption.database_is_encrypted(tmp_path / "memories.db") is False
    assert encryption.is_enabled(tmp_path) is False
    assert not (tmp_path / "config.json").exists()
    # The key was generated in this run and protects nothing, so it is not left behind.
    assert backend.deletes == [(keychain.SERVICE, account)]
    assert backend.stored == {}


def test_enable_refuses_when_the_key_reads_back_as_nothing(tmp_path, monkeypatch):
    """A backend that swallows the refused read and answers None is caught too."""
    backend = _FakeKeyring(lambda fake: None).install(monkeypatch)
    _seed(tmp_path, "stay plaintext")

    result = CliRunner().invoke(cli, ["encrypt", "enable", "--yes"], env=_env(tmp_path))

    account = f"db-key:{tmp_path.resolve()}"
    assert backend.writes == [(keychain.SERVICE, account)]
    assert result.exit_code != 0
    assert "the write succeeded but the key read back as empty" in result.output
    assert "POPPY_DB_KEY" in result.output
    assert "interactive login session" in result.output
    assert encryption.database_is_encrypted(tmp_path / "memories.db") is False
    assert encryption.is_enabled(tmp_path) is False
    assert backend.deletes == [(keychain.SERVICE, account)]
    assert backend.stored == {}


def _locked_keyring():
    """macOS in a session with no keychain UI: the backend loads, every call fails.

    Stands in for errSecInteractionNotAllowed (-25308), what an ssh, launchd or
    background session gets while the login keychain is locked to it.
    """
    from keyring.backend import KeyringBackend
    from keyring.errors import KeyringError

    class _LockedKeyring(KeyringBackend):
        priority = 1

        def _refuse(self):
            raise KeyringError("Can't store password on keychain: (-25308, 'Unknown Error')")

        def get_password(self, service: str, username: str) -> str | None:
            self._refuse()

        def set_password(self, service: str, username: str, password: str) -> None:
            self._refuse()

        def delete_password(self, service: str, username: str) -> None:
            self._refuse()

    return _LockedKeyring()


def _with_keyring(backend):
    import keyring

    class _Swap:
        def __enter__(self):
            self.previous = keyring.get_keyring()
            keyring.set_keyring(backend)

        def __exit__(self, *exc):
            keyring.set_keyring(self.previous)
            return False

    return _Swap()


def test_enable_refuses_when_the_keychain_denies_writes(tmp_path):
    """A locked keychain fails as one actionable line, not a raw -25308."""
    _seed(tmp_path, "stay plaintext")
    with _with_keyring(_locked_keyring()):
        result = CliRunner().invoke(cli, ["encrypt", "enable", "--yes"], env=_env(tmp_path))

    assert result.exit_code == 1
    assert len(result.output.strip().splitlines()) == 1
    assert "POPPY_DB_KEY" in result.output
    assert "login keychain is locked" in result.output
    assert "interactive login session" in result.output
    assert "-25308" not in result.output
    # Nothing was touched: the store is still plaintext and encryption is off.
    assert encryption.database_is_encrypted(tmp_path / "memories.db") is False
    assert encryption.is_enabled(tmp_path) is False


def test_status_reports_a_refused_keychain_when_the_probe_write_fails(tmp_path):
    """A backend that loads but refuses a probe write is not available.

    Reported as present-and-refused rather than missing: that distinction is what
    sends a user to an interactive login session instead of off to install a
    keyring backend, and it is what keeps this line from contradicting a "key in
    keychain: yes" underneath it in a session that may read but not write.
    """
    with _with_keyring(_locked_keyring()):
        result = CliRunner().invoke(cli, ["encrypt", "status"], env=_env(tmp_path))

    assert result.exit_code == 0
    assert "keychain backend: present, but this session cannot store and read back an entry" in result.output
    assert "keychain backend: available" not in result.output


def test_status_with_the_env_key_prints_no_keychain_line_at_all(tmp_path):
    """POPPY_DB_KEY is the key source, so the keychain is not probed or printed."""
    result = CliRunner().invoke(cli, ["encrypt", "status"], env={**_env(tmp_path), "POPPY_DB_KEY": "ab" * 32})

    assert result.exit_code == 0
    assert "key source: POPPY_DB_KEY" in result.output
    assert "keychain backend:" not in result.output


def test_disable_with_no_key_prints_the_same_line_as_recall(tmp_path):
    """No backend and no POPPY_DB_KEY is "no usable key", not keyrings.alt."""
    from keyring.backends import fail

    _seed(tmp_path, "locked away")
    enabled = CliRunner().invoke(cli, ["encrypt", "enable", "--yes"], env={**_env(tmp_path), "POPPY_DB_KEY": "ab" * 32})
    assert enabled.exit_code == 0, enabled.output

    with _with_keyring(fail.Keyring()):
        result = CliRunner().invoke(cli, ["encrypt", "disable", "--yes"], env=_env(tmp_path))
        # The same store opened the way recall and remember open it.
        with pytest.raises(encryption.EncryptionError) as opened:
            connect(tmp_path / "memories.db")

    assert result.exit_code == 1
    assert "keyrings.alt" not in result.output
    assert "no usable key was found" in result.output
    assert "Expected it in the OS keychain (or the POPPY_DB_KEY environment variable)" in result.output
    assert str(opened.value) in result.output
