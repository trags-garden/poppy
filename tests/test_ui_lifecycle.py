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
    # so tests don't pull bloom's models. UI's `get_fast_engine` already returns
    # the FTS5-only path; we patch `get_engine` (writer) to also be Baseline.
    db_path = tmp_path / "memories.db"
    engine = SeedEngine(db_path=db_path)
    fast_engine = SeedEngine(db_path=db_path)

    from poppy.ui import server as ui_server

    monkeypatch.setattr(ui_server, "get_fast_engine", lambda _d: fast_engine)
    monkeypatch.setattr(ui_server, "get_engine", lambda _d: engine)
    app = ui_server.create_app(poppy_dir=tmp_path)
    # TrustedHostMiddleware only accepts loopback hosts; TestClient's default
    # Host is "testserver", so bind the client to a loopback name.
    return TestClient(app, base_url="http://localhost")


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


def test_delete_and_restore_trigger_autosync(
    app_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dashboard delete and restore must queue a background sync, like
    the CLI and MCP paths, so the tombstone (and its later removal) propagate
    cross-device instead of sitting unpushed until some other sync fires."""
    calls: list[Path] = []
    monkeypatch.setattr("poppy.sync.auto.trigger", lambda poppy_dir, **kw: (calls.append(poppy_dir), True)[1])

    db = SeedEngine(db_path=tmp_path / "memories.db")
    _ingest(db, "d1", "delete me")

    assert app_client.delete("/api/memories/d1").status_code == 200
    assert len(calls) == 1  # delete queued a sync

    assert app_client.post("/api/memories/d1/restore").status_code == 200
    assert len(calls) == 2  # restore queued a sync


def test_tombstone_migration_idempotent_on_existing_db(tmp_path: Path) -> None:
    """Existing DBs without the superseded_by/expires_at columns upgrade transparently."""
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
    assert "memory_expires_at" in cols
    # And running it twice must remain idempotent.
    store2 = TombstoneStore(db_path=db_path)
    cols2 = {row["name"] for row in store2._conn.execute("PRAGMA table_info(ui_tombstones)").fetchall()}
    assert cols == cols2


def test_tombstone_migration_adds_memory_expires_at_to_superseded_by_era_db(tmp_path: Path) -> None:
    """A DB created before the memory_expires_at column (but after
    superseded_by) opens fine, gains it, and round-trips a TTL through a tombstone."""
    import sqlite3

    db_path = tmp_path / "memories.db"
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
            tombstoned_at TEXT NOT NULL,
            superseded_by TEXT
        );"""
    )
    conn.commit()
    conn.close()

    from poppy.ui.tombstones import TombstoneStore

    store = TombstoneStore(db_path=db_path)
    cols = {row["name"] for row in store._conn.execute("PRAGMA table_info(ui_tombstones)").fetchall()}
    assert "memory_expires_at" in cols

    ttl = datetime.now(timezone.utc) + timedelta(days=2)
    engine = SeedEngine(db_path=db_path)
    mem = _ingest(engine, "t1", "ttl fact", expires_at=ttl)
    store.add(mem)
    assert store.get("t1").memory.expires_at == ttl


def test_ui_restore_keeps_ttl_and_bumps_updated_at(
    app_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dashboard restore returns a live row with a fresh updated_at
    (so sync's push carries it) and the original expires_at."""
    monkeypatch.setattr("poppy.sync.auto.trigger", lambda poppy_dir, **kw: True)
    ttl = datetime.now(timezone.utc) + timedelta(days=2)
    db = SeedEngine(db_path=tmp_path / "memories.db")
    original = _ingest(db, "r1", "ttl fact", expires_at=ttl)

    deleted = app_client.delete("/api/memories/r1").json()
    restored = app_client.post("/api/memories/r1/restore").json()

    assert restored["expires_at"] == ttl.isoformat()
    assert restored["updated_at"] > deleted["tombstoned_at"]
    assert restored["updated_at"] > original.updated_at.isoformat()
    assert app_client.get("/api/memories/r1").json()["expires_at"] == ttl.isoformat()


def test_tombstone_migration_survives_a_concurrent_opener(tmp_path: Path) -> None:
    """Two processes opening the store at once (UI, CLI, autosync worker) can both
    read the schema before either writes it, so both try to ADD COLUMN. The loser
    used to die with sqlite3.OperationalError: duplicate column name.

    The race is reproduced by handing the second migration the schema snapshot it
    would have taken before the first one ran.
    """
    import sqlite3

    from poppy.ui.tombstones import _migrate_columns

    db_path = tmp_path / "memories.db"
    conn_a = sqlite3.connect(str(db_path))
    conn_a.row_factory = sqlite3.Row
    conn_a.executescript(
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
    conn_a.commit()

    conn_b = sqlite3.connect(str(db_path))
    conn_b.row_factory = sqlite3.Row
    stale_cols = [{"name": r["name"]} for r in conn_b.execute("PRAGMA table_info(ui_tombstones)").fetchall()]

    # Process A migrates and commits while B still holds its pre-migration view.
    _migrate_columns(conn_a)
    conn_a.commit()

    class _Rows:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

    class _StaleSchemaConn:
        """conn_b, answering its first PRAGMA from the snapshot taken above."""

        def __init__(self, conn, stale):
            self._conn = conn
            self._stale = stale

        def execute(self, sql, *params):
            if self._stale is not None and "PRAGMA" in sql:
                rows, self._stale = self._stale, None
                return _Rows(rows)
            return self._conn.execute(sql, *params)

        def commit(self):
            return self._conn.commit()

    _migrate_columns(_StaleSchemaConn(conn_b, stale_cols))  # must not raise
    _migrate_columns(conn_a)  # and re-running on a migrated DB stays a no-op

    cols = {row["name"] for row in conn_b.execute("PRAGMA table_info(ui_tombstones)").fetchall()}
    assert {"superseded_by", "memory_expires_at"} <= cols
    conn_a.close()
    conn_b.close()

    from poppy.ui.tombstones import TombstoneStore

    TombstoneStore(db_path=db_path)  # the store still opens normally


def test_ui_restore_of_an_elapsed_ttl_returns_410(
    app_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A memory whose TTL ran out while it was deleted is not resurrected.
    The dashboard gets 410 Gone and the tombstone is cleared."""
    monkeypatch.setattr("poppy.sync.auto.trigger", lambda poppy_dir, **kw: True)
    from poppy.models import Memory, Source
    from poppy.ui.tombstones import TombstoneStore

    elapsed = datetime.now(timezone.utc) - timedelta(hours=1)
    n = datetime.now(timezone.utc) - timedelta(days=1)
    store = TombstoneStore(tmp_path / "memories.db")
    store.add(
        Memory(
            id="e1",
            content="short-lived fact",
            memory_type="fact",
            source=Source(type="manual", session_id=None, timestamp=n),
            project=None,
            related_to=[],
            created_at=n,
            updated_at=n,
            expires_at=elapsed,
        )
    )

    resp = app_client.post("/api/memories/e1/restore")

    assert resp.status_code == 410
    assert "TTL elapsed" in resp.json()["detail"]
    assert store.get("e1") is not None, "the restore attempt destroyed the last copy"
    assert app_client.post("/api/memories/e1/restore").status_code == 410  # still there to find


def test_tombstone_migration_backfills_a_token_for_existing_rows(tmp_path: Path) -> None:
    """Rows written before the token column still need a usable token, or the
    conditional delete would silently match nothing for them."""
    import sqlite3

    db_path = tmp_path / "memories.db"
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
            tombstoned_at TEXT NOT NULL,
            superseded_by TEXT
        );"""
    )
    stamp = datetime.now(timezone.utc).isoformat()
    for mid in ("old1", "old2"):
        conn.execute(
            "INSERT INTO ui_tombstones (id, content, memory_type, source_type, source_timestamp,"
            " confidence, related_to, created_at, updated_at, tombstoned_at)"
            " VALUES (?, ?, 'fact', 'manual', ?, 1.0, '[]', ?, ?, ?)",
            (mid, f"legacy {mid}", stamp, stamp, stamp, stamp),
        )
    conn.commit()
    conn.close()

    from poppy.ui.tombstones import TombstoneStore

    store = TombstoneStore(db_path=db_path)
    tokens = {t.memory.id: t.token for t in store.list_all()}

    assert all(tokens.values()), "legacy rows must get a token"
    assert len(set(tokens.values())) == 2, "each row needs its own token"
    assert store.remove("old1", token=tokens["old1"]) is True
    assert store.remove("old2", token="not-the-right-token") is False
    assert store.get("old2") is not None


def test_tombstone_migration_guard_covers_the_sqlcipher_driver(tmp_path: Path) -> None:
    """An encrypted store runs on SQLCipher, whose OperationalError is NOT a
    sqlite3.OperationalError. A typed except would let the loser of a two-process
    migration race abort startup on exactly the stores that can least afford it."""
    from poppy.ui.tombstones import _migrate_columns

    class _SqlcipherOperationalError(Exception):
        """Stand-in for sqlcipher3.dbapi2.OperationalError: unrelated to sqlite3."""

    class _Rows:
        @staticmethod
        def fetchall():
            return [{"name": "id"}]  # every late column looks missing

        @staticmethod
        def fetchone():
            return None  # nothing to backfill

    class _Conn:
        def __init__(self, message):
            self.message = message
            self.altered = 0

        def execute(self, sql, *params):
            if "PRAGMA" in sql:
                return _Rows()
            if sql.startswith("ALTER"):
                self.altered += 1
                raise _SqlcipherOperationalError(self.message)
            return _Rows()

        def commit(self):
            return None

    losing = _Conn("duplicate column name: memory_expires_at")
    _migrate_columns(losing)  # must not raise
    assert losing.altered == 3  # swallowed every duplicate

    broken = _Conn("no such table: ui_tombstones")
    with pytest.raises(Exception, match="no such table"):
        _migrate_columns(broken)


def test_ui_restore_reports_success_when_it_raced(
    app_client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raced restore still restored the row, so the dashboard sees a normal 200.
    The stale tombstone beside it is the next sync's problem, not the user's."""
    monkeypatch.setattr("poppy.sync.auto.trigger", lambda poppy_dir, **kw: True)
    db = SeedEngine(db_path=tmp_path / "memories.db")
    original = _ingest(db, "raced1", "contested fact")

    from poppy import write_flow

    monkeypatch.setattr(
        write_flow,
        "restore",
        lambda *a, **kw: write_flow.RestoreResult(found=True, memory=original, raced=True),
    )

    resp = app_client.post("/api/memories/raced1/restore")

    assert resp.status_code == 200
    assert resp.json()["id"] == "raced1"
