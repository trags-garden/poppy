"""Encryption at rest for the local Poppy store.

These exercise the real SQLCipher + keyring path. ``keyring`` is stubbed to an
in-memory backend by the autouse ``_isolate_keychain`` fixture in conftest, so
nothing here touches the OS keychain. The whole module skips if the optional
encryption dependencies are not installed.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip("sqlcipher3")
pytest.importorskip("keyring")

from poppy import encryption, keychain  # noqa: E402
from poppy.db import SQLITE_MAGIC, connect  # noqa: E402
from poppy.engine.seed import SeedEngine  # noqa: E402
from poppy.models import Memory, Source  # noqa: E402


def _memory(mem_id: str, content: str, project: str | None = None) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        content=content,
        memory_type="fact",
        source=Source(type="manual", session_id=None, timestamp=now),
        project=project,
        related_to=[],
        created_at=now,
        updated_at=now,
    )


def _seed_store(poppy_dir: Path, *memories: Memory) -> Path:
    """Create a plaintext store with the given memories, closing all handles."""
    db = poppy_dir / "memories.db"
    engine = SeedEngine(db_path=db)
    for mem in memories:
        engine.ingest(mem)
    engine._conn.close()
    return db


# --- detection --------------------------------------------------------------


def test_plaintext_store_has_sqlite_magic(tmp_path):
    db = _seed_store(tmp_path, _memory("m1", "hello world"))
    assert db.read_bytes()[:16] == SQLITE_MAGIC
    assert encryption.database_is_encrypted(db) is False
    assert encryption.is_enabled(tmp_path) is False


def test_absent_or_empty_db_is_not_encrypted(tmp_path):
    assert encryption.database_is_encrypted(tmp_path / "memories.db") is False
    assert encryption.is_enabled(tmp_path) is False


# --- enable / migrate -------------------------------------------------------


def test_enable_migrates_existing_memories(tmp_path):
    _seed_store(tmp_path, _memory("m1", "alpha"), _memory("m2", "beta"), _memory("m3", "gamma"))

    result = encryption.enable(tmp_path)

    assert result.created_empty is False
    assert result.migrated_rows == 3
    db = tmp_path / "memories.db"
    assert encryption.database_is_encrypted(db) is True
    assert encryption.is_enabled(tmp_path) is True
    # The sentinel and config flag are both set.
    assert (tmp_path / ".poppy-encrypted").exists()
    # Memories survive and are readable through the normal connect chokepoint.
    engine = SeedEngine(db_path=db)
    try:
        ids = {m.id for m in engine.list_all(limit=100)}
        assert ids == {"m1", "m2", "m3"}
    finally:
        engine._conn.close()


def test_enable_stores_key_in_keychain(tmp_path):
    _seed_store(tmp_path, _memory("m1", "hello"))
    assert keychain.get_secret(f"db-key:{tmp_path.resolve()}") is None
    encryption.enable(tmp_path)
    key = keychain.get_secret(f"db-key:{tmp_path.resolve()}")
    assert key is not None
    assert len(key) == 64


def test_enable_empty_store_creates_encrypted(tmp_path):
    # No DB file yet: enable should create an encrypted one that then accepts writes.
    result = encryption.enable(tmp_path)
    assert result.created_empty is True
    db = tmp_path / "memories.db"
    assert encryption.database_is_encrypted(db) is True

    engine = SeedEngine(db_path=db)
    try:
        engine.ingest(_memory("m1", "written after enable"))
        assert engine.get("m1").content == "written after enable"
    finally:
        engine._conn.close()


def test_enable_is_idempotent_guarded(tmp_path):
    _seed_store(tmp_path, _memory("m1", "x"))
    encryption.enable(tmp_path)
    with pytest.raises(encryption.EncryptionError, match="already encrypted"):
        encryption.enable(tmp_path)


# --- on-disk confidentiality ------------------------------------------------


def test_content_not_plaintext_on_disk(tmp_path):
    secret = "SQUIRREL_CANARY_9f3a"
    _seed_store(tmp_path, _memory("m1", f"the {secret} is buried here"))
    db = tmp_path / "memories.db"
    assert secret.encode() in db.read_bytes()  # sanity: plaintext leaks before encryption

    encryption.enable(tmp_path)

    assert secret.encode() not in db.read_bytes()
    assert db.read_bytes()[:16] != SQLITE_MAGIC


def test_fts_index_not_plaintext_on_disk(tmp_path):
    # The FTS5 inverted index is the reason field-level encryption is not enough;
    # whole-DB encryption must cover it too.
    token = "PLATYPUSWORD"
    _seed_store(tmp_path, _memory("m1", f"a sentence containing {token} indexed by fts5"))
    encryption.enable(tmp_path)
    assert token.encode() not in (tmp_path / "memories.db").read_bytes()


def test_wal_sidecar_not_plaintext_on_disk(tmp_path):
    encryption.enable(tmp_path)
    db = tmp_path / "memories.db"
    canary = "WAL_CANARY_beadfeed"
    conn = connect(db)  # routed through SQLCipher
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS probe (v TEXT)")
        conn.execute("INSERT INTO probe (v) VALUES (?)", (canary,))
        conn.commit()  # committed but left in the -wal until checkpoint
        wal = db.with_name(db.name + "-wal")
        if wal.exists() and wal.stat().st_size > 0:
            assert canary.encode() not in wal.read_bytes()
    finally:
        conn.close()


# --- connect routing --------------------------------------------------------


def test_connect_routes_encrypted_and_stdlib_cannot_open(tmp_path):
    _seed_store(tmp_path, _memory("m1", "routed"))
    encryption.enable(tmp_path)
    db = tmp_path / "memories.db"

    conn = connect(db)
    try:
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
    finally:
        conn.close()

    with pytest.raises(sqlite3.DatabaseError):
        raw = sqlite3.connect(str(db))
        try:
            raw.execute("SELECT count(*) FROM memories").fetchone()
        finally:
            raw.close()


def test_embedding_blob_survives_encryption(tmp_path):
    # Proves BLOB columns (embeddings) round-trip through SQLCipher without the
    # ML models needing to load.
    encryption.enable(tmp_path)
    db = tmp_path / "memories.db"
    blob = bytes(range(256)) * 4
    conn = connect(db)
    try:
        conn.execute("CREATE TABLE emb (id TEXT PRIMARY KEY, vec BLOB NOT NULL)")
        conn.execute("INSERT INTO emb VALUES (?, ?)", ("m1", blob))
        conn.commit()
    finally:
        conn.close()
    conn2 = connect(db)
    try:
        got = conn2.execute("SELECT vec FROM emb WHERE id = ?", ("m1",)).fetchone()[0]
        assert bytes(got) == blob
    finally:
        conn2.close()


def test_seed_engine_retrieve_on_encrypted_store(tmp_path):
    encryption.enable(tmp_path)
    db = tmp_path / "memories.db"
    engine = SeedEngine(db_path=db)
    try:
        engine.ingest(_memory("m1", "the capital of France is Paris"))
        engine.ingest(_memory("m2", "the mitochondria is the powerhouse of the cell"))
        results = engine.retrieve("France", limit=5)
        assert results
        assert results[0].memory.id == "m1"
    finally:
        engine._conn.close()


# --- key handling -----------------------------------------------------------


def test_missing_key_raises_clear_error(tmp_path):
    _seed_store(tmp_path, _memory("m1", "x"))
    encryption.enable(tmp_path)
    keychain.delete_secret(f"db-key:{tmp_path.resolve()}")
    with pytest.raises(encryption.EncryptionError, match="no usable key was found"):
        connect(tmp_path / "memories.db")


def test_wrong_key_raises_clear_error(tmp_path):
    _seed_store(tmp_path, _memory("m1", "x"))
    encryption.enable(tmp_path)
    keychain.set_secret(f"db-key:{tmp_path.resolve()}", encryption.generate_key())
    with pytest.raises(encryption.EncryptionError, match="key is wrong|could not open"):
        connect(tmp_path / "memories.db")


def test_wrong_key_error_states_the_problem_once(tmp_path):
    """The formatted cause must not repeat the sentence and the path."""
    _seed_store(tmp_path, _memory("m1", "x"))
    encryption.enable(tmp_path)
    keychain.set_secret(f"db-key:{tmp_path.resolve()}", encryption.generate_key())
    db = tmp_path / "memories.db"
    with pytest.raises(encryption.EncryptionError) as opened:
        connect(db)
    message = str(opened.value)
    assert message.count("the key is wrong or the file is not a SQLCipher database") == 1
    assert message.count(str(db)) == 1


def test_open_encrypted_chains_the_driver_error_the_message_unwraps(tmp_path):
    """_wrong_key_message unwraps one __cause__; pin the link it needs.

    Drop the ``raise ... from exc`` in _open_encrypted and the store path and the
    reason quietly start appearing twice again. The rendered-message test above
    would catch it, but only by accident of what it renders; this pins the chain.
    """
    _seed_store(tmp_path, _memory("m1", "x"))
    encryption.enable(tmp_path)
    db = tmp_path / "memories.db"

    with pytest.raises(encryption.EncryptionError) as raised:
        encryption._open_encrypted(db, encryption.generate_key(), wal=False)

    cause = raised.value.__cause__
    assert cause is not None, "keep `raise ... from exc`: _wrong_key_message unwraps it"
    assert not isinstance(cause, encryption.EncryptionError), "the cause is the driver error, not our own"
    assert encryption._wrong_key_message(db, raised.value).count(str(db)) == 1


def test_env_key_override(tmp_path, monkeypatch):
    key = encryption.generate_key()
    monkeypatch.setenv("POPPY_DB_KEY", key)
    _seed_store(tmp_path, _memory("m1", "env keyed"))
    encryption.enable(tmp_path)
    # With the env override active, nothing is written to the keychain.
    assert keychain.get_secret(f"db-key:{tmp_path.resolve()}") is None
    conn = connect(tmp_path / "memories.db")
    try:
        assert conn.execute("SELECT count(*) FROM memories").fetchone()[0] == 1
    finally:
        conn.close()


def test_invalid_env_key_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("POPPY_DB_KEY", "not-a-valid-key")
    with pytest.raises(encryption.EncryptionError, match="64 hex characters"):
        encryption.enable(tmp_path)


# --- disable ----------------------------------------------------------------


def test_disable_returns_to_plaintext(tmp_path):
    _seed_store(tmp_path, _memory("m1", "alpha"), _memory("m2", "beta"))
    encryption.enable(tmp_path)

    rows = encryption.disable(tmp_path)

    assert rows == 2
    db = tmp_path / "memories.db"
    assert encryption.database_is_encrypted(db) is False
    assert encryption.is_enabled(tmp_path) is False
    assert not (tmp_path / ".poppy-encrypted").exists()
    assert keychain.get_secret(f"db-key:{tmp_path.resolve()}") is None
    # Readable by plain stdlib sqlite3 again.
    raw = sqlite3.connect(str(db))
    try:
        assert raw.execute("SELECT count(*) FROM memories").fetchone()[0] == 2
    finally:
        raw.close()


def test_disable_when_not_encrypted_raises(tmp_path):
    _seed_store(tmp_path, _memory("m1", "x"))
    with pytest.raises(encryption.EncryptionError, match="not encrypted"):
        encryption.disable(tmp_path)


def test_enable_disable_roundtrip_preserves_content(tmp_path):
    original = "exact content with unicode: café, 日本語, 🌸"
    _seed_store(tmp_path, _memory("m1", original))
    encryption.enable(tmp_path)
    encryption.disable(tmp_path)
    engine = SeedEngine(db_path=tmp_path / "memories.db")
    try:
        assert engine.get("m1").content == original
    finally:
        engine._conn.close()


# --- status -----------------------------------------------------------------


def test_status_reflects_lifecycle(tmp_path):
    _seed_store(tmp_path, _memory("m1", "x"))
    before = encryption.status(tmp_path)
    assert before.enabled is False
    assert before.deps_installed is True

    encryption.enable(tmp_path)
    after = encryption.status(tmp_path)
    assert after.enabled is True
    assert after.encrypted_on_disk is True
    assert after.key_present is True
