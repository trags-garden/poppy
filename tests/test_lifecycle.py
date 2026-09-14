"""Tests for memory lifecycle: TTL parsing, edit, supersede, expire."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from poppy.engine.seed import SeedEngine
from poppy.lifecycle import (
    edit_memory,
    parse_expires_at,
    parse_since,
    parse_ttl,
    resolve_expiry,
    supersede_memory,
)
from poppy.models import Filters, Memory, Source


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _mem(content: str, mid: str = "m1", expires_at: datetime | None = None) -> Memory:
    n = _now()
    return Memory(
        id=mid,
        content=content,
        memory_type="fact",
        source=Source(type="manual", session_id=None, timestamp=n),
        project=None,
        related_to=[],
        created_at=n,
        updated_at=n,
        expires_at=expires_at,
    )


@pytest.fixture
def engine(tmp_path: Path) -> SeedEngine:
    return SeedEngine(db_path=tmp_path / "mem.db")


# -------- parse_ttl ----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("30d", timedelta(days=30)),
        ("2w", timedelta(weeks=2)),
        ("12h", timedelta(hours=12)),
        ("90m", timedelta(minutes=90)),
        ("60s", timedelta(seconds=60)),
        ("1w3d", timedelta(weeks=1, days=3)),
        ("7", timedelta(days=7)),
    ],
)
def test_parse_ttl_units(text: str, expected: timedelta) -> None:
    assert parse_ttl(text) == expected


@pytest.mark.parametrize("bad", ["", "0d", "-1d", "garbage", "30x", "30d junk", "0"])
def test_parse_ttl_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_ttl(bad)


def test_parse_expires_at_assumes_utc_when_naive() -> None:
    dt = parse_expires_at("2026-06-01T12:00:00")
    assert dt.tzinfo is not None


def test_resolve_expiry_mutually_exclusive() -> None:
    with pytest.raises(ValueError):
        resolve_expiry("30d", "2026-06-01T00:00:00")


def test_resolve_expiry_none_when_unset() -> None:
    assert resolve_expiry(None, None) is None


# -------- parse_since ---------------------------------------------------------


def test_parse_since_iso_date_assumes_utc() -> None:
    dt = parse_since("2026-06-01")
    assert dt == datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)


def test_parse_since_iso_datetime() -> None:
    dt = parse_since("2026-06-01T12:30:00+02:00")
    assert dt == datetime(2026, 6, 1, 12, 30, 0, tzinfo=timezone(timedelta(hours=2)))


def test_parse_since_relative_duration() -> None:
    anchor = datetime(2026, 6, 9, 12, 0, 0, tzinfo=timezone.utc)
    assert parse_since("7d", now=anchor) == anchor - timedelta(days=7)
    assert parse_since("12h", now=anchor) == anchor - timedelta(hours=12)
    assert parse_since("1w3d", now=anchor) == anchor - timedelta(weeks=1, days=3)


@pytest.mark.parametrize("bad", ["", "   ", "not-a-date", "30x", "0d", "-1d"])
def test_parse_since_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_since(bad)


# -------- edit_memory --------------------------------------------------------


def test_edit_preserves_id_created_at_and_source(engine: SeedEngine) -> None:
    m = _mem("original")
    engine.ingest(m)
    result = edit_memory(engine, m.id, content="updated")
    assert result.changed
    fetched = engine.get(m.id)
    assert fetched is not None
    assert fetched.id == m.id
    assert fetched.content == "updated"
    assert fetched.created_at == m.created_at
    assert fetched.source.type == m.source.type
    assert fetched.updated_at >= m.updated_at


def test_edit_unknown_id_raises(engine: SeedEngine) -> None:
    with pytest.raises(KeyError):
        edit_memory(engine, "missing", content="x")


def test_edit_no_change_returns_unchanged(engine: SeedEngine) -> None:
    m = _mem("same")
    engine.ingest(m)
    result = edit_memory(engine, m.id, content="same")
    assert not result.changed


def test_edit_clears_expiry(engine: SeedEngine) -> None:
    future = _now() + timedelta(days=10)
    m = _mem("with-ttl", expires_at=future)
    engine.ingest(m)
    result = edit_memory(engine, m.id, clear_expiry=True)
    assert result.changed
    assert engine.get(m.id).expires_at is None


def test_edit_unsets_project(engine: SeedEngine) -> None:
    n = _now()
    m = Memory(
        id="p1",
        content="x",
        memory_type="fact",
        source=Source("manual", None, n),
        project="poppy",
        related_to=[],
        created_at=n,
        updated_at=n,
    )
    engine.ingest(m)
    edit_memory(engine, m.id, project_unset=True)
    assert engine.get(m.id).project is None


def test_edit_rejects_expires_and_clear(engine: SeedEngine) -> None:
    m = _mem("x")
    engine.ingest(m)
    with pytest.raises(ValueError):
        edit_memory(engine, m.id, expires_at=_now() + timedelta(days=1), clear_expiry=True)


# -------- supersede ----------------------------------------------------------


def test_supersede_tombstones_old_and_links_new(engine: SeedEngine, tmp_path: Path) -> None:
    old = _mem("old fact", mid="old1")
    engine.ingest(old)
    new = _mem("new fact", mid="new1")
    result = supersede_memory(engine, new, old.id, poppy_dir=tmp_path)
    assert result.tombstoned
    assert engine.get(old.id) is None
    fetched = engine.get(new.id)
    assert fetched is not None
    assert old.id in fetched.related_to

    from poppy.ui.tombstones import TombstoneStore

    ts = TombstoneStore(tmp_path / "memories.db")
    assert ts.get(old.id) is not None


def test_supersede_unknown_old_raises(engine: SeedEngine, tmp_path: Path) -> None:
    new = _mem("x", mid="n2")
    with pytest.raises(KeyError):
        supersede_memory(engine, new, "ghost", poppy_dir=tmp_path)


# -------- expiry filtering + purge -------------------------------------------


def test_expired_memory_filtered_from_retrieve(engine: SeedEngine) -> None:
    past = _now() - timedelta(days=1)
    m = _mem("expired beta", mid="e1", expires_at=past)
    engine.ingest(m)
    assert engine.retrieve("beta") == []
    results = engine.retrieve("beta", filters=Filters(include_expired=True))
    assert [r.memory.id for r in results] == ["e1"]


def test_expired_memory_filtered_from_list(engine: SeedEngine) -> None:
    past = _now() - timedelta(days=1)
    future = _now() + timedelta(days=1)
    engine.ingest(_mem("alive", mid="a", expires_at=future))
    engine.ingest(_mem("dead", mid="d", expires_at=past))
    engine.ingest(_mem("forever", mid="f"))
    ids = {m.id for m in engine.list_all()}
    assert ids == {"a", "f"}
    all_ids = {m.id for m in engine.list_all(filters=Filters(include_expired=True))}
    assert all_ids == {"a", "d", "f"}


def test_purge_expired_hard_deletes(engine: SeedEngine) -> None:
    past = _now() - timedelta(days=1)
    engine.ingest(_mem("old", mid="o", expires_at=past))
    engine.ingest(_mem("alive", mid="a"))
    n = engine.purge_expired()
    assert n == 1
    remaining = {m.id for m in engine.list_all(filters=Filters(include_expired=True))}
    assert remaining == {"a"}


# -------- schema migration ---------------------------------------------------


def test_schema_migration_on_pre_lifecycle_db(tmp_path: Path) -> None:
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            project TEXT,
            source_type TEXT NOT NULL,
            source_session_id TEXT,
            source_timestamp TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            related_to TEXT DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE memory_fts USING fts5(id UNINDEXED, content);
        CREATE TRIGGER memory_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memory_fts(id, content) VALUES (new.id, new.content);
        END;
        """
    )
    iso = _now().isoformat()
    conn.execute(
        """INSERT INTO memories (id, content, memory_type, source_type, source_timestamp,
           created_at, updated_at) VALUES ('legacy', 'old', 'fact', 'manual', ?, ?, ?)""",
        (iso, iso, iso),
    )
    conn.commit()
    conn.close()

    engine = SeedEngine(db_path=db)
    mems = engine.list_all()
    assert [m.id for m in mems] == ["legacy"]
    assert mems[0].expires_at is None
    assert [r.memory.id for r in engine.retrieve("old")] == ["legacy"]
