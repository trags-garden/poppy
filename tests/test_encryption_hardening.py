"""Regression tests for the PR #4 review findings on local-store encryption.

Each test maps to a numbered finding from the adversarial review. The autouse
conftest fixtures isolate POPPY_DIR and stub the keychain in-memory, so nothing
here touches the real store or OS keychain. Skips cleanly without the extra.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("sqlcipher3")
pytest.importorskip("keyring")

from poppy import encryption, keychain  # noqa: E402
from poppy.db import connect  # noqa: E402
from poppy.engine.seed import SeedEngine  # noqa: E402
from poppy.errors import EncryptionError, KeychainUnavailable  # noqa: E402
from poppy.models import Memory, Source  # noqa: E402


def _memory(mem_id: str, content: str) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        content=content,
        memory_type="fact",
        source=Source(type="manual", session_id=None, timestamp=now),
        project=None,
        related_to=[],
        created_at=now,
        updated_at=now,
    )


def _seed(poppy_dir: Path, *contents: str) -> Path:
    db = poppy_dir / "memories.db"
    engine = SeedEngine(db_path=db)
    for i, c in enumerate(contents):
        engine.ingest(_memory(f"m{i}", c))
    engine._conn.close()
    return db


def _account(poppy_dir: Path) -> str:
    return f"db-key:{poppy_dir.resolve()}"


# --- #1 keychain key normalization / validation -----------------------------


def test_uppercase_keychain_key_still_opens(tmp_path):
    _seed(tmp_path, "hello")
    encryption.enable(tmp_path)
    canonical = keychain.get_secret(_account(tmp_path))
    # Simulate a hand-restored key: uppercased with a trailing newline.
    keychain.set_secret(_account(tmp_path), canonical.upper() + "\n")
    conn = connect(tmp_path / "memories.db")
    try:
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
    finally:
        conn.close()


def test_enable_normalizes_pre_existing_keychain_key(tmp_path):
    # A pre-existing non-canonical key is normalized and re-stored before use.
    key = encryption.generate_key()
    keychain.set_secret(_account(tmp_path), key.upper() + "\n")
    _seed(tmp_path, "x")
    encryption.enable(tmp_path)
    assert keychain.get_secret(_account(tmp_path)) == key  # canonical form persisted
    conn = connect(tmp_path / "memories.db")
    try:
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
    finally:
        conn.close()


def test_malformed_keychain_key_fails_before_encrypting(tmp_path):
    keychain.set_secret(_account(tmp_path), "not-hex-garbage")
    _seed(tmp_path, "x")
    with pytest.raises(EncryptionError, match="malformed"):
        encryption.enable(tmp_path)
    # The store must be left plaintext, never encrypted under a bad key.
    assert encryption.database_is_encrypted(tmp_path / "memories.db") is False


# --- #4 / #12 error taxonomy ------------------------------------------------


def test_keychain_unavailable_is_encryption_error():
    assert issubclass(KeychainUnavailable, EncryptionError)
    assert issubclass(encryption.DependencyMissing, EncryptionError)


def test_keychain_failure_surfaces_as_encryption_error(tmp_path, monkeypatch):
    _seed(tmp_path, "x")
    encryption.enable(tmp_path)

    def _boom(account):
        raise KeychainUnavailable("backend down")

    monkeypatch.setattr(keychain, "get_secret", _boom)
    # Opening the encrypted store with a failing keychain must raise an
    # EncryptionError (caught by the CLI), not a bare RuntimeError.
    with pytest.raises(EncryptionError):
        connect(tmp_path / "memories.db")


# --- #7 wrong key during disable --------------------------------------------


def test_disable_wrong_key_raises_friendly_error(tmp_path, monkeypatch):
    key = encryption.generate_key()
    monkeypatch.setenv("POPPY_DB_KEY", key)
    _seed(tmp_path, "x")
    encryption.enable(tmp_path)
    # Switch to a different (valid-format) key: format check passes, decrypt fails.
    monkeypatch.setenv("POPPY_DB_KEY", encryption.generate_key())
    with pytest.raises(EncryptionError, match="key is wrong|could not open"):
        encryption.disable(tmp_path)
    # Original encrypted store untouched; no decrypted residue left behind.
    assert encryption.database_is_encrypted(tmp_path / "memories.db") is True
    assert encryption.find_residue(tmp_path) == []


# --- #13 disable rotates the key even with an env override ------------------


def test_disable_deletes_keychain_key_even_with_env_set(tmp_path, monkeypatch):
    _seed(tmp_path, "x")
    encryption.enable(tmp_path)  # stores key in the keychain
    stored = keychain.get_secret(_account(tmp_path))
    assert stored is not None
    monkeypatch.setenv("POPPY_DB_KEY", stored)
    encryption.disable(tmp_path)
    assert keychain.get_secret(_account(tmp_path)) is None


# --- #15 verification on a store with no memories table ---------------------


def test_enable_then_disable_empty_store(tmp_path):
    result = encryption.enable(tmp_path)  # created_empty, no tables
    assert result.created_empty is True
    # Must not crash on a missing `memories` table.
    rows = encryption.disable(tmp_path)
    assert rows == 0
    assert encryption.database_is_encrypted(tmp_path / "memories.db") is False


# --- #5 / #10 sentinel / header disagreement --------------------------------


def test_plaintext_with_stale_sentinel_is_inconsistent(tmp_path):
    _seed(tmp_path, "x")  # plaintext
    (tmp_path / ".poppy-encrypted").touch()  # stale marker
    st = encryption.status(tmp_path)
    assert st.inconsistent is True
    assert st.state == "plaintext-stale-sentinel"
    # Neither enable nor disable crashes with a raw driver error.
    with pytest.raises(EncryptionError, match="inconsistent"):
        encryption.enable(tmp_path)
    with pytest.raises(EncryptionError, match="inconsistent"):
        encryption.disable(tmp_path)


def test_repair_removes_stale_sentinel(tmp_path):
    _seed(tmp_path, "x")
    (tmp_path / ".poppy-encrypted").touch()
    actions = encryption.repair(tmp_path)
    assert any("stale encryption marker" in a for a in actions)
    assert encryption.store_state(tmp_path) == "plaintext"


def test_repair_recreates_missing_sentinel(tmp_path):
    _seed(tmp_path, "x")
    encryption.enable(tmp_path)
    (tmp_path / ".poppy-encrypted").unlink()  # encrypted-no-sentinel
    assert encryption.store_state(tmp_path) == "encrypted-no-sentinel"
    actions = encryption.repair(tmp_path)
    assert any("recreated the missing encryption marker" in a for a in actions)
    assert encryption.store_state(tmp_path) == "encrypted"


# --- #6 corrupt plaintext store keeps the native diagnostic -----------------


def test_corrupt_header_without_sentinel_gives_clean_error(tmp_path):
    db = tmp_path / "memories.db"
    db.write_bytes(b"\x00" * 100)  # garbage header, no sentinel, no key
    # No raw sqlite3 traceback: a clean EncryptionError that names both the
    # corrupt and the encrypted-lost-marker possibilities (N6).
    with pytest.raises(EncryptionError, match="not a plaintext SQLite database"):
        connect(db)


# --- #3 / #14 decrypted residue ---------------------------------------------


def test_residue_detected_and_repaired(tmp_path):
    _seed(tmp_path, "secret")
    encryption.enable(tmp_path)
    residue = tmp_path / "memories.db.plain-tmp"
    residue.write_text("a decrypted copy left by a killed disable")
    st = encryption.status(tmp_path)
    assert residue in st.residue
    actions = encryption.repair(tmp_path)
    assert any("plain-tmp" in a for a in actions)
    assert not residue.exists()


# --- the SH/EX gate: constructive exclusion ---------------------------------


def _fast_gate(monkeypatch):
    """Shrink the gate acquire windows so refusal tests do not wait seconds."""
    from poppy import db

    monkeypatch.setattr(db, "_GATE_EXCLUSIVE_ATTEMPTS", 2)
    monkeypatch.setattr(db, "_GATE_EXCLUSIVE_SLEEP_S", 0.01)
    monkeypatch.setattr(db, "_GATE_SHARED_ATTEMPTS", 2)
    monkeypatch.setattr(db, "_GATE_SHARED_SLEEP_S", 0.01)


def test_open_connection_blocks_migration(tmp_path, monkeypatch):
    pytest.importorskip("fcntl")
    _fast_gate(monkeypatch)
    db = _seed(tmp_path, "x")
    conn = connect(db)  # holds the shared gate for its lifetime
    try:
        with pytest.raises(EncryptionError, match="other Poppy processes have the store open"):
            encryption.enable(tmp_path)
    finally:
        conn.close()
    # Once the connection closes the gate is free and the migration proceeds.
    encryption.enable(tmp_path)
    assert encryption.database_is_encrypted(tmp_path / "memories.db") is True


def test_migration_refusal_names_registered_surface(tmp_path, monkeypatch):
    pytest.importorskip("fcntl")
    _fast_gate(monkeypatch)
    from poppy import writers

    db = _seed(tmp_path, "x")
    conn = connect(db)  # the gate SH that actually blocks the migration
    try:
        with writers.registered(tmp_path, "serve"):  # best-effort naming only
            with pytest.raises(EncryptionError, match="the MCP server"):
                encryption.enable(tmp_path)
    finally:
        conn.close()


def test_migration_refusal_gives_daemon_stop_and_restart_story(tmp_path):
    from poppy import db, writers

    with writers.registered(tmp_path, "daemon"):
        message = db._migration_refusal(tmp_path)

    assert "poppy daemon stop" in message
    assert "poppy daemon start" in message
    assert "Stop them and retry." in message


def test_migration_refusal_lists_daemon_and_ui_with_daemon_hint(tmp_path):
    from poppy import db, writers

    with writers.registered(tmp_path, "daemon"), writers.registered(tmp_path, "ui"):
        message = db._migration_refusal(tmp_path)

    assert "the MCP daemon (`poppy daemon`)" in message
    assert "the web UI (`poppy ui`)" in message
    assert "Stop them and retry." in message
    assert "For the daemon: run `poppy daemon stop` first, then `poppy daemon start` after the migration." in message


def test_connect_blocks_while_migration_holds_gate(tmp_path, monkeypatch):
    fcntl = pytest.importorskip("fcntl")
    from poppy import db as dbmod

    _fast_gate(monkeypatch)
    _seed(tmp_path, "x")
    fd = os.open(str(tmp_path / dbmod.GATE_FILENAME), os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)  # a migration is holding the exclusive gate
    try:
        with pytest.raises(EncryptionError, match="migration is in progress"):
            connect(tmp_path / "memories.db")
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_dead_writer_lock_is_reaped(tmp_path):
    pytest.importorskip("fcntl")
    from poppy import writers

    # A leftover lock file whose owner is gone: no live flock, so it is stale.
    stale = tmp_path / "writers"
    stale.mkdir()
    (stale / "serve.999999.lock").touch()
    assert writers.live_writers(tmp_path) == []
    assert not (stale / "serve.999999.lock").exists()  # reaped


# --- N3 / N6 encrypted-no-sentinel is opened + self-healed, not crashed ------


def test_encrypted_no_sentinel_self_heals_on_connect(tmp_path):
    _seed(tmp_path, "healme")
    encryption.enable(tmp_path)
    (tmp_path / ".poppy-encrypted").unlink()  # crash window / lost dotfile
    assert encryption.store_state(tmp_path) == "encrypted-no-sentinel"
    conn = connect(tmp_path / "memories.db")  # must open, not raise
    try:
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
    finally:
        conn.close()
    # Self-healed so the fast path applies next time.
    assert (tmp_path / ".poppy-encrypted").exists()


# --- N4 repair after a crashed disable retires the orphaned key --------------


def test_repair_after_crashed_disable_deletes_key(tmp_path):
    # State a crashed disable leaves: plaintext DB + leftover sentinel + key.
    _seed(tmp_path, "x")
    keychain.set_secret(_account(tmp_path), encryption.generate_key())
    (tmp_path / ".poppy-encrypted").touch()
    assert encryption.store_state(tmp_path) == "plaintext-stale-sentinel"
    encryption.repair(tmp_path)
    assert keychain.get_secret(_account(tmp_path)) is None  # rotation preserved


# --- N5 journal residue ------------------------------------------------------


def test_journal_residue_detected_and_removed(tmp_path):
    _seed(tmp_path, "x")
    journal = tmp_path / "memories.db.plain-tmp-journal"
    journal.write_text("stray rollback journal from a killed export")
    assert journal in encryption.find_residue(tmp_path)
    encryption.cleanup_residue(tmp_path)
    assert not journal.exists()


# --- N8 corrupt file is never marked encrypted -------------------------------


def test_repair_does_not_mark_corrupt_store(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_DB_KEY", encryption.generate_key())  # a key resolves
    (tmp_path / "memories.db").write_bytes(b"\x00" * 200)  # will not open under it
    assert encryption.store_state(tmp_path) == "corrupt"
    actions = encryption.repair(tmp_path)
    assert any("not a valid encrypted" in a for a in actions)
    assert not (tmp_path / ".poppy-encrypted").exists()  # no marker manufactured
    with pytest.raises(EncryptionError, match="not a plaintext SQLite database"):
        encryption.enable(tmp_path)


# --- WAL-replay destruction is prevented ------------------------------------


def _craft_plaintext_wal_beside(db: Path) -> None:
    """Put a populated PLAINTEXT -wal next to ``db`` (the crash-window bomb)."""
    pdb = db.with_name("plaintext-source.db")
    pw = sqlite3.connect(str(pdb))
    try:
        pw.execute("PRAGMA journal_mode=WAL")
        pw.execute("PRAGMA wal_autocheckpoint=0")
        pw.execute("CREATE TABLE t(x)")
        pw.execute("INSERT INTO t VALUES ('BOMB')")
        pw.commit()
        (db.parent / (db.name + "-wal")).write_bytes((pdb.with_name("plaintext-source.db-wal")).read_bytes())
    finally:
        pw.close()


def test_diagnostics_do_not_replay_stray_wal(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_DB_KEY", "ab" * 32)
    _seed(tmp_path, "secret-A", "secret-B")
    encryption.enable(tmp_path)
    db = tmp_path / "memories.db"
    sha0 = hashlib.sha256(db.read_bytes()).hexdigest()

    _craft_plaintext_wal_beside(db)
    (tmp_path / ".poppy-encrypted").unlink()  # ambiguous: encrypted db, no sentinel

    # A read-only diagnostic must not replay the stray wal or mutate the store.
    encryption.status(tmp_path)
    encryption.store_state(tmp_path)
    assert hashlib.sha256(db.read_bytes()).hexdigest() == sha0  # main db untouched


def test_connect_refuses_ambiguous_wal_then_repair_recovers(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_DB_KEY", "ab" * 32)
    _seed(tmp_path, "secret-A", "secret-B")
    encryption.enable(tmp_path)
    db = tmp_path / "memories.db"
    _craft_plaintext_wal_beside(db)  # opposite-cipher stray wal
    (tmp_path / ".poppy-encrypted").unlink()

    # connect must NOT silently quarantine/replay: it refuses (no data touched).
    with pytest.raises(EncryptionError, match="repair"):
        connect(db)

    # repair (under the exclusive gate) sets aside the proven wrong-cipher wal and
    # heals the marker; the store then opens with its real data intact.
    encryption.repair(tmp_path)
    conn = connect(db)
    try:
        rows = {r[0] for r in conn.execute("SELECT content FROM memories")}
    finally:
        conn.close()
    assert rows == {"secret-A", "secret-B"}  # real data, not the BOMB
    assert not db.with_name(db.name + "-wal").exists()  # the wrong-cipher wal is gone


def test_repair_replays_same_cipher_wal_preserving_commits(tmp_path, monkeypatch):
    # A healthy encrypted store loses its sentinel while a LEGIT encrypted -wal
    # holds committed rows (a hard-kill state). repair must REPLAY it, not discard.
    monkeypatch.setenv("POPPY_DB_KEY", "ab" * 32)
    db = _seed(tmp_path, "base")
    encryption.enable(tmp_path)
    wal = db.with_name(db.name + "-wal")
    # Commit a row, then snapshot main+wal BEFORE any checkpoint (a reader pins the
    # wal), and restore that snapshot after close to simulate a hard-kill: the row
    # exists only in the -wal, not yet in the main file.
    reader = connect(db)
    reader.execute("BEGIN")
    reader.execute("SELECT count(*) FROM memories").fetchall()
    w = connect(db)
    _raw_insert(w, "late", "committed-in-wal")
    w.commit()
    main_bytes = db.read_bytes()
    wal_bytes = wal.read_bytes()
    w.close()
    reader.close()
    db.write_bytes(main_bytes)
    wal.write_bytes(wal_bytes)
    (tmp_path / ".poppy-encrypted").unlink()  # encrypted-no-sentinel, wal present

    with pytest.raises(EncryptionError, match="repair"):
        connect(db)
    encryption.repair(tmp_path)  # must replay the same-cipher wal, not discard it
    conn = connect(db)
    try:
        rows = {r[0] for r in conn.execute("SELECT content FROM memories")}
    finally:
        conn.close()
    assert "committed-in-wal" in rows  # the committed row was preserved


def test_migration_leaves_no_sidecar_beside_swapped_store(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_DB_KEY", "cd" * 32)
    _seed(tmp_path, "x")
    encryption.enable(tmp_path)
    db = tmp_path / "memories.db"
    # No plaintext -wal/-shm may sit beside the freshly encrypted file.
    assert not (db.with_name(db.name + "-wal")).exists()
    assert not (db.with_name(db.name + "-shm")).exists()


# --- diagnostics honesty ----------------------------------------------------


def test_open_without_sentinel_survives_torn_config(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_DB_KEY", "cd" * 32)
    _seed(tmp_path, "x")
    encryption.enable(tmp_path)
    (tmp_path / ".poppy-encrypted").unlink()
    (tmp_path / "config.json").write_text('{"engine": "seed"')  # torn write, invalid JSON
    conn = connect(tmp_path / "memories.db")  # must open, not raise JSONDecodeError
    try:
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
    finally:
        conn.close()


def test_stale_env_key_does_not_look_corrupt(tmp_path, monkeypatch):
    _seed(tmp_path, "x")
    encryption.enable(tmp_path)  # keychain key (no env)
    (tmp_path / ".poppy-encrypted").unlink()
    # A wrong POPPY_DB_KEY shadows the keychain; probing must still try the
    # keychain key and find the store healthy, not declare it corrupt.
    monkeypatch.setenv("POPPY_DB_KEY", encryption.generate_key())
    assert encryption.store_state(tmp_path) == "encrypted-no-sentinel"


def test_status_deep_detects_bitrot_behind_sentinel(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_DB_KEY", "ab" * 32)
    _seed(tmp_path, "x")
    encryption.enable(tmp_path)
    db = tmp_path / "memories.db"
    data = bytearray(db.read_bytes())
    for i in range(16, 96):  # corrupt page 1 content, keep the sentinel intact
        data[i] ^= 0xFF
    db.write_bytes(bytes(data))
    assert encryption.status(tmp_path).state == "encrypted"  # shallow trusts the sentinel
    assert encryption.status(tmp_path, deep=True).state in ("corrupt", "unreadable")


def test_readonly_db_file_reports_permission_not_corrupt(tmp_path, monkeypatch):
    if os.geteuid() == 0:  # pragma: no cover
        pytest.skip("root bypasses file permissions")
    monkeypatch.setenv("POPPY_DB_KEY", "ef" * 32)
    _seed(tmp_path, "x")
    encryption.enable(tmp_path)
    db = tmp_path / "memories.db"
    os.chmod(db, 0o000)
    try:
        with pytest.raises(encryption.StorePermissionError):
            connect(db)  # sentinel intact -> encrypted open -> permission, not corruption
    finally:
        os.chmod(db, 0o644)


def test_gate_released_on_close_allows_later_migration(tmp_path):
    pytest.importorskip("fcntl")
    db = _seed(tmp_path, "x")
    conn = connect(db)
    conn.close()  # releasing the gate fd
    encryption.enable(tmp_path)  # would refuse if the gate were still held
    assert encryption.database_is_encrypted(db) is True


# --- round-6: lockout chain + gate robustness -------------------------------


def _raw_insert(conn, mid, content):
    conn.execute(
        "INSERT INTO memories (id, content, memory_type, project, source_type, source_session_id, "
        "source_timestamp, confidence, related_to, created_at, updated_at) VALUES "
        "(?, ?, 'fact', NULL, 'manual', NULL, '2026-01-01T00:00:00+00:00', 1.0, '[]', "
        "'2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')",
        (mid, content),
    )


def test_unreadable_encrypted_file_not_misclassified_plaintext(tmp_path):
    if os.geteuid() == 0:  # pragma: no cover
        pytest.skip("root bypasses file permissions")
    _seed(tmp_path, "x")
    encryption.enable(tmp_path)  # keychain key, sentinel intact
    key = keychain.get_secret(_account(tmp_path))
    assert key is not None
    db = tmp_path / "memories.db"
    os.chmod(db, 0o000)  # unreadable file (a sudo/backup permissions mishap)
    try:
        # Must NOT be called plaintext-stale-sentinel (whose repair deletes the key).
        assert encryption.store_state(tmp_path) == "unreadable"
        encryption.repair(tmp_path)
        assert keychain.get_secret(_account(tmp_path)) == key  # key preserved: no lockout
    finally:
        os.chmod(db, 0o644)


def test_repair_on_readonly_dir_no_traceback(tmp_path):
    if os.geteuid() == 0:  # pragma: no cover
        pytest.skip("root bypasses permissions")
    monkeypatch_key = "ab" * 32
    os.environ["POPPY_DB_KEY"] = monkeypatch_key
    try:
        _seed(tmp_path, "x")
        encryption.enable(tmp_path)
        (tmp_path / ".poppy-encrypted").unlink()  # encrypted-no-sentinel
        os.chmod(tmp_path, 0o555)  # read-only dir
        try:
            actions = encryption.repair(tmp_path)  # must not raise a raw PermissionError
            assert any("could not" in a or "permission" in a.lower() for a in actions)
        finally:
            os.chmod(tmp_path, 0o755)
    finally:
        del os.environ["POPPY_DB_KEY"]


def test_repair_on_torn_config_no_traceback(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_DB_KEY", "ab" * 32)
    _seed(tmp_path, "x")
    encryption.enable(tmp_path)
    (tmp_path / ".poppy-encrypted").unlink()
    (tmp_path / "config.json").write_text('{"engine": "seed"')  # torn JSON
    actions = encryption.repair(tmp_path)  # must not raise JSONDecodeError
    assert actions


def test_readonly_dir_without_gate_gives_permission_error(tmp_path):
    if os.geteuid() == 0:  # pragma: no cover
        pytest.skip("root bypasses permissions")
    db = tmp_path / "memories.db"
    c = sqlite3.connect(str(db))  # plaintext store, and NO db.gate created
    c.execute("CREATE TABLE memories (id text)")
    c.commit()
    c.close()
    os.chmod(tmp_path, 0o555)  # read-only dir: the gate file cannot be created
    try:
        with pytest.raises(encryption.StorePermissionError):
            connect(db)
    finally:
        os.chmod(tmp_path, 0o755)


def test_stale_env_key_recall_still_works_via_keychain(tmp_path, monkeypatch):
    _seed(tmp_path, "x")
    encryption.enable(tmp_path)  # keychain key, sentinel intact
    monkeypatch.setenv("POPPY_DB_KEY", encryption.generate_key())  # wrong, shadows keychain
    for _ in range(2):  # must work repeatedly, not "once then break"
        conn = connect(tmp_path / "memories.db")
        try:
            assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
        finally:
            conn.close()


def test_percent_in_poppy_dir_opens(tmp_path, monkeypatch):
    d = tmp_path / "dir%41pct"
    d.mkdir()
    monkeypatch.setenv("POPPY_DB_KEY", "ab" * 32)
    _seed(d, "x")
    encryption.enable(d)
    (d / ".poppy-encrypted").unlink()  # force the immutable-probe path
    assert encryption.store_state(d) == "encrypted-no-sentinel"  # probe worked despite the %
    conn = connect(d / "memories.db")
    conn.close()


@pytest.mark.parametrize("dirname", ["dir?x", "dir#x", "dir%41x"])
def test_plaintext_probe_escapes_uri_characters_in_path(tmp_path, dirname):
    # Repair deletes a key on the strength of this probe. Unescaped, '?' and '#'
    # truncate the URI path so SQLite creates and opens an empty database beside
    # the store (a corrupt store "passes"), and '%41' decodes to a missing path.
    d = tmp_path / dirname
    d.mkdir()
    good = _seed(d, "x")
    corrupt = d / "corrupt.db"
    corrupt.write_bytes(b"SQLite format 3\x00" + b"\xff" * 4080)

    assert encryption._opens_as_plaintext(good) is True
    assert encryption._opens_as_plaintext(corrupt) is False
    assert [p.name for p in tmp_path.iterdir()] == [dirname]  # no stray database created


def test_leaked_connection_releases_gate(tmp_path):
    pytest.importorskip("fcntl")
    import gc

    db = _seed(tmp_path, "x")
    conn = connect(db)
    del conn  # no close(): the weakref finalizer must release the gate fd
    gc.collect()
    encryption.enable(tmp_path)  # would refuse if the gate were still held
    assert encryption.database_is_encrypted(db) is True


def test_plaintext_stale_sentinel_live_wal_not_lost(tmp_path):
    db = _seed(tmp_path, "base")
    (tmp_path / ".poppy-encrypted").touch()  # plaintext-stale-sentinel
    a = connect(db)  # opens plaintext (no wal yet)
    _raw_insert(a, "a1", "A-committed")
    a.commit()  # now a live plaintext wal exists
    # A second connect must NOT quarantine A's live wal; it refuses instead.
    with pytest.raises(EncryptionError, match="repair"):
        connect(db)
    a.close()  # checkpoints A's committed row into the store
    encryption.repair(tmp_path)  # removes the stale sentinel
    conn = connect(db)
    try:
        rows = {r[0] for r in conn.execute("SELECT content FROM memories")}
    finally:
        conn.close()
    assert "A-committed" in rows  # A's committed write survived


def test_cross_process_open_connection_blocks_migration(tmp_path, monkeypatch):
    """A real second PROCESS holding a connection blocks a migration (the gate is
    kernel-enforced, so this is cross-process, not just same-process fds)."""
    pytest.importorskip("fcntl")
    import subprocess
    import sys
    import textwrap
    import time

    import poppy

    src_dir = str(Path(poppy.__file__).resolve().parent.parent)
    db = _seed(tmp_path, "x")
    ready = tmp_path / "ready.flag"
    code = textwrap.dedent(f"""
        import sys, time, pathlib
        sys.path.insert(0, {src_dir!r})
        from poppy.db import connect
        c = connect({str(db)!r})
        pathlib.Path({str(ready)!r}).write_text("open")
        time.sleep(10)
    """)
    proc = subprocess.Popen([sys.executable, "-c", code], env={"PATH": os.environ.get("PATH", "")})
    try:
        for _ in range(100):
            if ready.exists():
                break
            time.sleep(0.05)
        assert ready.exists(), "child never opened its connection"
        monkeypatch.setattr("poppy.db._GATE_EXCLUSIVE_ATTEMPTS", 3)
        monkeypatch.setattr("poppy.db._GATE_EXCLUSIVE_SLEEP_S", 0.02)
        with pytest.raises(EncryptionError, match="other Poppy processes have the store open"):
            encryption.enable(tmp_path)
    finally:
        proc.terminate()
        proc.wait()
    # Once the child is gone the migration proceeds.
    encryption.enable(tmp_path)
    assert encryption.database_is_encrypted(db) is True


def test_repair_keeps_key_when_store_does_not_open_as_plaintext(tmp_path):
    # A magic-header-but-corrupt file is classified plaintext-stale-sentinel, but
    # repair must NOT delete the key unless the store PROVES it opens as plaintext
    # (never delete a key on a mere magic-looking header -> no lockout).
    db = tmp_path / "memories.db"
    db.write_bytes(b"SQLite format 3\x00" + b"\xff" * 400)  # magic header, garbage body
    (tmp_path / ".poppy-encrypted").touch()
    keychain.set_secret(_account(tmp_path), encryption.generate_key())
    assert encryption.store_state(tmp_path) == "plaintext-stale-sentinel"
    actions = encryption.repair(tmp_path)
    assert keychain.get_secret(_account(tmp_path)) is not None  # key preserved
    assert any("KEPT the keychain key" in a for a in actions)


def test_wrong_cipher_wal_replay_is_destructive_documented(tmp_path, monkeypatch):
    # Documents WHY connect refuses ambiguous-state opens: a matching-mode open of
    # a WRONG-cipher stray wal is replayed on checkpoint and destroys the store
    # (SQLCipher frame validation does NOT reject it). This is the disproof of the
    # "frame validation makes it safe" hypothesis; guards against silent regression.
    from sqlcipher3 import dbapi2 as sq

    key = "ab" * 32
    monkeypatch.setenv("POPPY_DB_KEY", key)
    db = _seed(tmp_path, "s0", "s1", "s2")
    encryption.enable(tmp_path)
    sha0 = hashlib.sha256(db.read_bytes()).hexdigest()
    _craft_plaintext_wal_beside(db)  # opposite-cipher stray wal

    o = sq.connect(str(db))
    o.execute(f"PRAGMA key = \"x'{key}'\"")
    try:
        o.execute("SELECT 1 FROM sqlite_master").fetchall()
    except Exception:
        pass
    o.close()  # the checkpoint on close applies the wrong-cipher wal
    assert hashlib.sha256(db.read_bytes()).hexdigest() != sha0  # store was mutated (destroyed)


# --- round-7: repair never destroys data it cannot prove disposable ---------


def _encrypted_no_sentinel_with_pending_wal(tmp_path, content="committed-in-wal"):
    """Build an encrypted store that lost its sentinel with a committed row that
    lives ONLY in a pending same-cipher -wal (a hard-kill state)."""
    db = _seed(tmp_path, "base")
    encryption.enable(tmp_path)
    wal = db.with_name(db.name + "-wal")
    reader = connect(db)
    reader.execute("BEGIN")
    reader.execute("SELECT count(*) FROM memories").fetchall()
    w = connect(db)
    _raw_insert(w, "late", content)
    w.commit()
    main_bytes = db.read_bytes()
    wal_bytes = wal.read_bytes()
    w.close()
    reader.close()
    db.write_bytes(main_bytes)
    wal.write_bytes(wal_bytes)
    (tmp_path / ".poppy-encrypted").unlink()
    return db, wal


def test_repair_readonly_main_preserves_wal_row(tmp_path, monkeypatch):
    # POC5: read-only main (0o444) + same-cipher pending wal. repair cannot replay
    # into the read-only main, so it must PRESERVE (leave the wal), never quarantine
    # + delete the committed row.
    if os.geteuid() == 0:  # pragma: no cover
        pytest.skip("root bypasses file permissions")
    monkeypatch.setenv("POPPY_DB_KEY", "ab" * 32)
    db, wal = _encrypted_no_sentinel_with_pending_wal(tmp_path)
    os.chmod(db, 0o444)
    try:
        actions = encryption.repair(tmp_path)
        assert wal.exists() and wal.stat().st_size > 0  # the wal (with the row) is intact
        assert not list(tmp_path.glob("*stray*"))  # NOT quarantined
        assert any("preserved" in a.lower() or "left" in a.lower() for a in actions)
    finally:
        os.chmod(db, 0o644)
    # With write access restored, repair replays and the committed row appears.
    encryption.repair(tmp_path)
    conn = connect(db)
    try:
        rows = {r[0] for r in conn.execute("SELECT content FROM memories")}
    finally:
        conn.close()
    assert "committed-in-wal" in rows


def test_repair_copy_failure_is_guarded(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_DB_KEY", "ab" * 32)
    _encrypted_no_sentinel_with_pending_wal(tmp_path)

    def _boom(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(encryption, "_copy_store_for_test", _boom)
    actions = encryption.repair(tmp_path)  # must not raise a raw OSError
    assert any("could not verify" in a for a in actions)


def test_repair_keeps_setaside_stray(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_DB_KEY", "ab" * 32)
    _seed(tmp_path, "secret")
    encryption.enable(tmp_path)
    db = tmp_path / "memories.db"
    _craft_plaintext_wal_beside(db)  # opposite-cipher wal beside the encrypted store
    (tmp_path / ".poppy-encrypted").unlink()
    encryption.repair(tmp_path)
    strays = list(tmp_path.glob("*stray*"))
    assert strays  # the unrecoverable wal was set aside
    encryption.repair(tmp_path)  # a second repair must NOT delete the set-aside stray
    assert list(tmp_path.glob("*stray*")) == strays


def test_exclusive_gate_readonly_dir_permission_error(tmp_path, monkeypatch):
    if os.geteuid() == 0:  # pragma: no cover
        pytest.skip("root bypasses permissions")
    monkeypatch.setenv("POPPY_DB_KEY", "ab" * 32)
    db = tmp_path / "memories.db"
    c = sqlite3.connect(str(db))  # plaintext store, and NO db.gate created
    c.execute("CREATE TABLE memories (id text)")
    c.commit()
    c.close()
    os.chmod(tmp_path, 0o555)  # read-only dir: exclusive_gate can't create db.gate
    try:
        with pytest.raises(encryption.StorePermissionError):
            encryption.enable(tmp_path)
    finally:
        os.chmod(tmp_path, 0o755)


# --- The key write is verified by a read-back ----------------------


def test_enable_refuses_when_key_reads_back_changed(tmp_path, monkeypatch):
    """Write reports success but the key reads back as another value: refuse."""
    _seed(tmp_path, "x")
    reads = [None, encryption.generate_key()]  # no existing entry, then a different read-back
    real_get = keychain.get_secret

    def _get(account: str) -> str | None:
        # Only the store's own key misbehaves; keychain.writable()'s throwaway
        # probe entry still round-trips through the in-memory backend.
        if account.startswith(keychain._PROBE_ACCOUNT):
            return real_get(account)
        return reads.pop(0) if reads else None

    monkeypatch.setattr(keychain, "get_secret", _get)

    with pytest.raises(EncryptionError, match="read back as a different value"):
        encryption.enable(tmp_path)

    assert encryption.database_is_encrypted(tmp_path / "memories.db") is False
    assert encryption.is_enabled(tmp_path) is False


def test_enable_keeps_an_existing_keychain_key_when_the_read_back_fails(tmp_path, monkeypatch):
    """A reused entry may be the only copy of the key, so a failed read-back must not delete it."""
    import keyring

    _seed(tmp_path, "x")
    account = f"db-key:{tmp_path.resolve()}"
    existing = encryption.generate_key()
    keychain.set_secret(account, existing)
    reads = [existing]  # the existing-key lookup finds it; the read-back is refused
    real_get = keychain.get_secret

    def _get(account: str) -> str | None:
        # keychain.writable()'s throwaway probe still round-trips; only reads of
        # the store's own key are scripted here.
        if account.startswith(keychain._PROBE_ACCOUNT):
            return real_get(account)
        if reads:
            return reads.pop(0)
        raise KeychainUnavailable("could not read from the OS keychain: interaction not allowed (-25308)")

    monkeypatch.setattr(keychain, "get_secret", _get)

    with pytest.raises(EncryptionError, match="reading it back failed"):
        encryption.enable(tmp_path)

    assert encryption.database_is_encrypted(tmp_path / "memories.db") is False
    assert encryption.is_enabled(tmp_path) is False
    assert keyring.get_password(keychain.SERVICE, account) == existing


def test_enable_refusal_on_a_failed_key_write_carries_the_same_guidance(tmp_path, monkeypatch):
    """A refused keychain WRITE gets the POPPY_DB_KEY / interactive-login copy the reads carry."""

    real_set = keychain.set_secret

    def _boom(account: str, value: str) -> None:
        # keychain.writable()'s throwaway probe still round-trips; only the write
        # of the store's own key is refused here.
        if account.startswith(keychain._PROBE_ACCOUNT):
            real_set(account, value)
            return
        raise KeychainUnavailable("could not write to the OS keychain: interaction not allowed (-25308)")

    _seed(tmp_path, "x")
    monkeypatch.setattr(keychain, "set_secret", _boom)

    with pytest.raises(EncryptionError, match="could not be verified in the OS keychain") as excinfo:
        encryption.enable(tmp_path)

    message = str(excinfo.value)
    assert "writing it failed" in message
    assert "POPPY_DB_KEY" in message
    assert "interactive login session" in message
    assert encryption.database_is_encrypted(tmp_path / "memories.db") is False
    assert encryption.is_enabled(tmp_path) is False


# --- The keychain is probed only where the answer is used -----------


class _SpyKeyring:
    """Records every call that reaches the credential store, and answers them."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self._store: dict[tuple[str, str], str] = {}

    def install(self, monkeypatch) -> "_SpyKeyring":
        import keyring

        monkeypatch.setattr(keyring, "get_password", self.get_password)
        monkeypatch.setattr(keyring, "set_password", self.set_password)
        monkeypatch.setattr(keyring, "delete_password", self.delete_password)
        return self

    def get_password(self, service: str, username: str) -> str | None:
        self.calls.append(("get", username))
        return self._store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.calls.append(("set", username))
        self._store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self.calls.append(("del", username))
        self._store.pop((service, username), None)


def test_status_touches_the_keychain_not_at_all_when_the_env_key_is_active(tmp_path, monkeypatch):
    """POPPY_DB_KEY is the key source, so status must do no keychain IO at all.

    On a locked Secret Service every probe call is an unlock prompt (four, with
    the delete and the read-back that proves it), on a path that is documented
    for headless and CI use and that never shows the result. `poppy doctor` calls status, so a hung credential
    store would block it.
    """
    _seed(tmp_path, "x")
    monkeypatch.setenv("POPPY_DB_KEY", "ab" * 32)
    spy = _SpyKeyring().install(monkeypatch)

    st = encryption.status(tmp_path)

    assert spy.calls == []
    assert st.env_key_active is True
    assert st.keychain_state == "not-probed"
    assert st.key_present is True


def test_status_probes_the_keychain_when_it_is_the_key_source(tmp_path, monkeypatch):
    """Without the env key the keychain IS the key source, so the probe is paid for."""
    _seed(tmp_path, "x")
    spy = _SpyKeyring().install(monkeypatch)

    st = encryption.status(tmp_path)

    # The trailing read is delete_secret proving the probe entry actually went:
    # keyring reports the same failure for an absent entry and a refused delete,
    # so only a read-back tells them apart.
    assert [call for call in spy.calls if call[1].startswith(keychain._PROBE_ACCOUNT)] == [
        ("set", keychain._PROBE_ACCOUNT),
        ("get", keychain._PROBE_ACCOUNT),
        ("del", keychain._PROBE_ACCOUNT),
        ("get", keychain._PROBE_ACCOUNT),
    ]
    assert st.keychain_state == "usable"


def test_status_reports_a_readable_but_unwritable_keychain_as_refused(tmp_path, monkeypatch):
    """A session that reads but cannot write still has its key: say so, both lines.

    The probe fails, yet resolve_key finds the entry. Reporting that keychain as
    flatly unavailable would contradict the key it just read.
    """
    _seed(tmp_path, "x")
    account = f"db-key:{tmp_path.resolve()}"
    keychain.set_secret(account, encryption.generate_key())
    real_set = keychain.set_secret

    def _refuse_probe_writes(account_name: str, value: str) -> None:
        if account_name.startswith(keychain._PROBE_ACCOUNT):
            raise KeychainUnavailable("could not write to the OS keychain: interaction not allowed (-25308)")
        real_set(account_name, value)

    monkeypatch.setattr(keychain, "set_secret", _refuse_probe_writes)

    st = encryption.status(tmp_path)

    assert st.keychain_state == "refused"
    assert st.key_present is True


def test_status_reports_no_backend_as_unavailable(tmp_path, monkeypatch):
    """No real backend is a different fact from a backend that refuses the probe."""
    _seed(tmp_path, "x")
    monkeypatch.setattr(keychain, "available", lambda: False)

    assert encryption.status(tmp_path).keychain_state == "unavailable"


def test_disable_names_the_keychain_failure_under_the_shared_no_key_line(tmp_path, monkeypatch):
    """A locked keychain must not read as "this store was encrypted elsewhere".

    The first sentence stays the one recall and remember print; what the keychain
    actually said is appended, so a user whose key IS there is not sent looking
    for another machine.
    """
    _seed(tmp_path, "x")
    monkeypatch.setenv("POPPY_DB_KEY", "ab" * 32)
    encryption.enable(tmp_path)
    monkeypatch.delenv("POPPY_DB_KEY")

    def _refuse(_account: str) -> str | None:
        raise KeychainUnavailable("could not read from the OS keychain: interaction not allowed (-25308)")

    monkeypatch.setattr(keychain, "get_secret", _refuse)

    with pytest.raises(EncryptionError) as excinfo:
        encryption.disable(tmp_path)

    message = str(excinfo.value)
    assert message.startswith(encryption._no_key_message(tmp_path / "memories.db"))
    assert "interaction not allowed (-25308)" in message
    assert "only locked" in message
    assert "keyrings.alt" not in message
