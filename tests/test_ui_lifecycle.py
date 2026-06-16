"""Tests for the UI server lifecycle endpoints (TTL, supersede, include_expired)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from poppy.engine.seed import SeedEngine
from poppy.models import Memory, Source


def _ingest(engine: SeedEngine, mid: str, content: str, expires_at=None) -> Memory:
    n = datetime.now(timezone.utc)
    m = Memory(
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
    engine.ingest(m)
    return m


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Force the UI to use the lightweight SeedEngine for both reader and writer
    # so tests don't pull the SproutEngine's models. UI's `get_fast_engine` already
    # returns the FTS5-only path; we patch `get_engine` (writer) to also be Baseline.
    db_path = tmp_path / "memories.db"
    engine = SeedEngine(db_path=db_path)
    fast_engine = SeedEngine(db_path=db_path)

    from poppy.ui import server as ui_server

    monkeypatch.setattr(ui_server, "get_fast_engine", lambda _d: fast_engine)
    monkeypatch.setattr(ui_server, "get_engine", lambda _d: engine)
    app = ui_server.create_app(poppy_dir=tmp_path)
    return TestClient(app)


def test_memory_out_includes_expires_at(app_client: TestClient, tmp_path: Path) -> None:
    future = datetime.now(timezone.utc) + timedelta(days=10)
    db = SeedEngine(db_path=tmp_path / "memories.db")
    _ingest(db, "x1", "permanent fact")
    _ingest(db, "x2", "ttl fact", expires_at=future)

    items = app_client.get("/api/memories").json()["items"]
    by_id = {i["id"]: i for i in items}
    assert by_id["x1"]["expires_at"] is None
    assert by_id["x2"]["expires_at"] == future.isoformat()
    # The tombstone-specific field is null on active memories.
    assert by_id["x2"]["tombstone_expires_at"] is None


def test_list_excludes_expired_by_default(app_client: TestClient, tmp_path: Path) -> None:
    past = datetime.now(timezone.utc) - timedelta(days=1)
    db = SeedEngine(db_path=tmp_path / "memories.db")
    _ingest(db, "stale", "stale fact", expires_at=past)
    _ingest(db, "fresh", "fresh fact")

    items = app_client.get("/api/memories").json()["items"]
    assert {i["id"] for i in items} == {"fresh"}

    items = app_client.get("/api/memories?include_expired=true").json()["items"]
    assert {i["id"] for i in items} == {"stale", "fresh"}


def test_patch_sets_ttl(app_client: TestClient, tmp_path: Path) -> None:
    db = SeedEngine(db_path=tmp_path / "memories.db")
    _ingest(db, "edit1", "no ttl")

    resp = app_client.patch("/api/memories/edit1", json={"ttl": "30d"})
    assert resp.status_code == 200
    expires_at = resp.json()["expires_at"]
    assert expires_at is not None
    parsed = datetime.fromisoformat(expires_at)
    diff = parsed - datetime.now(timezone.utc)
    assert timedelta(days=29) < diff < timedelta(days=31)


def test_patch_clear_expiry(app_client: TestClient, tmp_path: Path) -> None:
    future = datetime.now(timezone.utc) + timedelta(days=5)
    db = SeedEngine(db_path=tmp_path / "memories.db")
    _ingest(db, "edit2", "with ttl", expires_at=future)

    resp = app_client.patch("/api/memories/edit2", json={"clear_expiry": True})
    assert resp.status_code == 200
    assert resp.json()["expires_at"] is None


def test_patch_preserves_existing_ttl_when_not_specified(app_client: TestClient, tmp_path: Path) -> None:
    future = datetime.now(timezone.utc) + timedelta(days=5)
    db = SeedEngine(db_path=tmp_path / "memories.db")
    _ingest(db, "edit3", "with ttl", expires_at=future)

    resp = app_client.patch("/api/memories/edit3", json={"content": "updated content only"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"] == "updated content only"
    # TTL should NOT be stripped — pre-lifecycle UI bug regression check.
    assert body["expires_at"] is not None


def test_patch_rejects_conflicting_expiry_flags(app_client: TestClient, tmp_path: Path) -> None:
    db = SeedEngine(db_path=tmp_path / "memories.db")
    _ingest(db, "edit4", "x")

    resp = app_client.patch(
        "/api/memories/edit4",
        json={"ttl": "30d", "clear_expiry": True},
    )
    assert resp.status_code == 400


def test_supersede_endpoint(app_client: TestClient, tmp_path: Path) -> None:
    db = SeedEngine(db_path=tmp_path / "memories.db")
    _ingest(db, "old1", "use all-MiniLM")

    resp = app_client.post(
        "/api/memories/old1/supersede",
        json={"content": "use bge-large", "memory_type": "decision"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["supersedes"] == "old1"
    assert body["tombstoned"] is True
    new = body["new"]
    assert "old1" in new["related_to"]

    # Old is gone from the engine, present in tombstones.
    items = app_client.get("/api/memories").json()["items"]
    assert "old1" not in {i["id"] for i in items}
    tombs = app_client.get("/api/memories?scope=tombstoned").json()["items"]
    assert "old1" in {i["id"] for i in tombs}


def test_supersede_unknown_id_404s(app_client: TestClient) -> None:
    resp = app_client.post(
        "/api/memories/ghost/supersede",
        json={"content": "x", "memory_type": "fact"},
    )
    assert resp.status_code == 404


def test_supersede_records_back_pointer(app_client: TestClient, tmp_path: Path) -> None:
    """Tombstoned memory exposes superseded_by pointing at the new id."""
    db = SeedEngine(db_path=tmp_path / "memories.db")
    _ingest(db, "old2", "use all-MiniLM")

    resp = app_client.post(
        "/api/memories/old2/supersede",
        json={"content": "use bge-large", "memory_type": "decision"},
    )
    new_id = resp.json()["new"]["id"]

    # GET-by-id on the tombstoned old returns superseded_by populated.
    old = app_client.get("/api/memories/old2").json()
    assert old["tombstoned"] is True
    assert old["superseded_by"] == new_id
    # And from the tombstone list view as well.
    tombs = app_client.get("/api/memories?scope=tombstoned").json()["items"]
    by_id = {t["id"]: t for t in tombs}
    assert by_id["old2"]["superseded_by"] == new_id


def test_plain_delete_has_null_superseded_by(app_client: TestClient, tmp_path: Path) -> None:
    """A non-supersede tombstone must not have superseded_by populated."""
    db = SeedEngine(db_path=tmp_path / "memories.db")
    _ingest(db, "doomed", "obsolete fact")

    app_client.delete("/api/memories/doomed")
    old = app_client.get("/api/memories/doomed").json()
    assert old["tombstoned"] is True
    assert old["superseded_by"] is None


def test_tombstone_migration_idempotent_on_existing_db(tmp_path: Path) -> None:
    """Existing DBs without the superseded_by column upgrade transparently."""
    import sqlite3

    db_path = tmp_path / "memories.db"
    # Simulate a legacy DB: create the table without the column.
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """CREATE TABLE ui_tombstones (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            project TEXT,
            source_type TEXT NOT NULL,
            source_session_id TEXT,
            source_timestamp TEXT NOT NULL,
            confidence REAL NOT NULL,
            related_to TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            tombstoned_at TEXT NOT NULL
        );"""
    )
    conn.commit()
    conn.close()

    # Now opening the store must add the column without error.
    from poppy.ui.tombstones import TombstoneStore

    store = TombstoneStore(db_path=db_path)
    cols = {row["name"] for row in store._conn.execute("PRAGMA table_info(ui_tombstones)").fetchall()}
    assert "superseded_by" in cols
    # And running it twice must remain idempotent.
    store2 = TombstoneStore(db_path=db_path)
    cols2 = {row["name"] for row in store2._conn.execute("PRAGMA table_info(ui_tombstones)").fetchall()}
    assert cols == cols2
