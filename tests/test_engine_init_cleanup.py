"""Regression tests for connection cleanup when engine construction fails."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import poppy.engine._closet_engine as closet_module
import poppy.engine.bloom as bloom_module
import poppy.engine.seed as seed_module
from poppy.engine._closet_engine import ClosetHybridEngine
from poppy.engine.bloom import BloomEngine


class _InitFailure(RuntimeError):
    pass


def _legacy_db_with_null_enrichment(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(seed_module.SCHEMA)
        conn.execute("ALTER TABLE memories ADD COLUMN enriched_content TEXT")
        conn.execute(
            """INSERT INTO memories (
                   id, content, memory_type, project, source_type, source_timestamp,
                   created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "legacy",
                "legacy content",
                "fact",
                "test",
                "manual",
                "2026-07-17T00:00:00+00:00",
                "2026-07-17T00:00:00+00:00",
                "2026-07-17T00:00:00+00:00",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _assert_write_lock_available(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA busy_timeout = 50")
        conn.execute("BEGIN IMMEDIATE")
        conn.rollback()
    finally:
        conn.close()


def test_migration_failure_rolls_back_and_closes_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "memories.db"
    _legacy_db_with_null_enrichment(db_path)
    opened_connections: list[sqlite3.Connection] = []
    real_connect = closet_module.connect_db

    def tracked_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened_connections.append(conn)
        return conn

    def fail_during_enrichment(conn: sqlite3.Connection) -> None:
        conn.execute("UPDATE memories SET enriched_content = content WHERE enriched_content IS NULL")
        assert conn.in_transaction
        raise _InitFailure("migration exploded")

    monkeypatch.setattr(closet_module, "connect_db", tracked_connect)
    monkeypatch.setattr(seed_module, "_migrate_enriched_content", fail_during_enrichment)

    with pytest.raises(_InitFailure, match="migration exploded"):
        ClosetHybridEngine(db_path)

    assert len(opened_connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened_connections[0].execute("SELECT 1")
    _assert_write_lock_available(db_path)


def test_subclass_model_failure_closes_base_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "memories.db"
    opened_connections: list[sqlite3.Connection] = []
    real_connect = closet_module.connect_db

    def tracked_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened_connections.append(conn)
        return conn

    def fail_model_factory(*args, **kwargs):
        raise _InitFailure("model construction exploded")

    monkeypatch.setattr(closet_module, "connect_db", tracked_connect)
    monkeypatch.setattr(bloom_module, "FastembedModels", fail_model_factory)

    with pytest.raises(_InitFailure, match="model construction exploded"):
        BloomEngine(db_path)

    assert len(opened_connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened_connections[0].execute("SELECT 1")
    _assert_write_lock_available(db_path)


def test_retry_after_partial_migration_rewires_stale_fts_triggers(tmp_path: Path) -> None:
    """A construction retry must repair content-form FTS triggers left by a failed migration.

    ALTER TABLE autocommits, so a prior attempt that died between adding
    enriched_content and rewiring the triggers leaves the column present with
    the legacy triggers live — exactly the state _legacy_db_with_null_enrichment
    builds. The rewrite decision must come from the trigger SQL, not from
    whether this construction added the column.
    """
    db_path = tmp_path / "memories.db"
    _legacy_db_with_null_enrichment(db_path)

    engine = ClosetHybridEngine(db_path)
    try:
        triggers = dict(
            engine._conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'trigger'"
                " AND name IN ('memory_ai', 'memory_ad', 'memory_au')"
            )
        )
        assert set(triggers) == {"memory_ai", "memory_ad", "memory_au"}
        assert "enriched_content" in triggers["memory_ai"]
        assert "enriched_content" in triggers["memory_au"]
    finally:
        engine._conn.close()
