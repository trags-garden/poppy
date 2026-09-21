"""Legacy copy cleanup, conservative classification, and caller compatibility."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from poppy.engine._legacy_copies import MARKER_COLUMN, mark_legacy_copies_for_cleanup
from poppy.engine.bloom import BloomEngine
from poppy.engine.seed import SeedEngine
from poppy.models import Memory, Source
from poppy.sync import MAX_CONSECUTIVE_TRANSPORT_FAILURES, pull, push
from poppy.sync.state import SyncState, save
from poppy.ui.tombstones import TombstoneStore
from poppy.write_flow import forget

# The deletion body sent by the 0.3.0 release.
LEGACY_DELETION_BODY = "[poppy: derived per-speaker copy removed]"

SECRET = "zebrafishcadenza"


class _FakeBiEncoder:
    def embed(self, texts):
        for t in texts:
            h = hash(t) & 0xFFFF
            yield np.array([h & 0xF, (h >> 4) & 0xF, (h >> 8) & 0xF, (h >> 12) & 0xF], dtype=np.float32)


class _FakeCrossEncoder:
    def rerank(self, query, docs):
        return [float(len(d)) for d in docs]


def _local_deletion_time(db: Path, memory_id: str) -> datetime | None:
    with sqlite3.connect(db) as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'sync_local_deletions'").fetchone():
            return None
        rows = conn.execute("SELECT deleted_at FROM sync_local_deletions WHERE id = ?", (memory_id,)).fetchall()
    return datetime.fromisoformat(rows[0][0]) if rows else None


def _migrate_only(db: Path, *, had_bloom_schema: bool = True) -> None:
    """Run the one-time marker migration on its own.

    Opening an engine also REMOVES what that migration marks, which is the
    product behaviour. These tests are about the grading itself, so they run it
    directly instead of through an open.
    """
    from poppy.engine._legacy_copies import ensure_legacy_copy_tables

    conn = sqlite3.connect(str(db))
    try:
        ensure_legacy_copy_tables(conn)
        mark_legacy_copies_for_cleanup(conn, had_bloom_schema=had_bloom_schema)
        conn.commit()
    finally:
        conn.close()


def _bloom(db_path: Path) -> BloomEngine:
    return BloomEngine(db_path=db_path, bi_encoder=_FakeBiEncoder(), cross_encoder=_FakeCrossEncoder())


def _memory(mid: str, content: str, *, related_to: list[str] | None = None, project: str = "p1") -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mid,
        content=content,
        memory_type="fact",
        source=Source(type="cli", session_id=None, timestamp=now),
        project=project,
        related_to=list(related_to or []),
        created_at=now,
        updated_at=now,
        confidence=1.0,
    )


def _turns(secret: str = SECRET) -> str:
    return json.dumps(
        [
            {"speaker": "Alice", "dia_id": "D1", "text": secret},
            {"speaker": "Bob", "dia_id": "D2", "text": "noted"},
        ]
    )


def _rows(db_path: Path, sql: str, params: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _all_ids(db_path: Path) -> list[str]:
    return [r[0] for r in _rows(db_path, "SELECT id FROM memories ORDER BY id")]


def _marked_ids(db_path: Path) -> list[str]:
    """Every row the engine treats as a copy, whichever grade adopted it."""
    return [r[0] for r in _rows(db_path, f"SELECT id FROM memories WHERE {MARKER_COLUMN} >= 1 ORDER BY id")]


def _adopted_unverified_ids(db_path: Path) -> list[str]:
    """Copies adopted on inference: marked, but their text is not ours to rewrite."""
    return [r[0] for r in _rows(db_path, f"SELECT id FROM memories WHERE {MARKER_COLUMN} = 2 ORDER BY id")]


class _RecordingClient:
    """TragsClient stand-in that records every wire row and can serve rows back."""

    base_url = "https://trags.test"

    def __init__(self, rows: list[dict] | None = None, *, echo: bool = False) -> None:
        self.upserts: list[dict] = []
        self._rows = rows or []
        self._echo = echo

    def upsert(self, row: dict) -> None:
        self.upserts.append(dict(row))
        if self._echo:
            self._rows = [r for r in self._rows if r["id"] != row["id"]] + [dict(row)]

    def iter_all_since(self, updated_since=None, *, page_size: int = 100):
        yield from self._rows

    def ping(self) -> None:
        """The probe a push with nothing to send makes."""
        return None


def test_a_reingest_leaves_no_stale_fts_row_for_the_edited_memory(tmp_path: Path) -> None:
    """The residue the row-level assertions above cannot see.

    The parent is rewritten with ``INSERT OR REPLACE``, and REPLACE deletes the
    old row without firing the FTS delete trigger. ``memories`` therefore looked
    redacted while ``memory_fts`` still held the pre-edit text as a live row —
    searchable, and able to pull the memory back through RRF. Asserted against
    the index itself rather than through ``retrieve``, which reads content from
    ``memories``.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _turns()))
    engine.ingest(_memory("sess-2026-01", "totally redacted prose"))

    assert _rows(db, "SELECT count(*) FROM memory_fts WHERE content LIKE ?", (f"%{SECRET}%",)) == [(0,)]
    # One index row per surviving memory, holding the new text.
    assert _rows(db, "SELECT id, content FROM memory_fts") == [("sess-2026-01", "totally redacted prose")]


def _real_lookalike(db: Path) -> None:
    """A parent plus a REAL memory whose id sits under the parent's closet prefix."""
    engine = _bloom(db)
    engine.ingest(_memory("mem_customer", _turns()))
    engine.ingest(_memory("mem_customer_closet_notes", "the wardrobe budget for Q3"))


def test_a_real_memory_shaped_like_a_closet_is_listed_and_synced(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    _real_lookalike(db)

    for engine in (_bloom(db), SeedEngine(db_path=db)):
        listed = {m.id for m in engine.list_all()}
        assert listed == {"mem_customer", "mem_customer_closet_notes"}

    client = _RecordingClient()
    push(
        engine=_bloom(db),
        tombstones=TombstoneStore(db),
        client=client,
        state=SyncState(),
        poppy_dir=tmp_path,
    )
    assert {r["id"] for r in client.upserts} == {"mem_customer", "mem_customer_closet_notes"}


def test_forgetting_the_parent_does_not_delete_a_real_lookalike(tmp_path: Path) -> None:
    """The delete is prefix-scoped AND marker-gated; the unmarked neighbour survives."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    _real_lookalike(db)

    assert forget(engine, tmp_path, "mem_customer").deleted is True

    assert _all_ids(db) == ["mem_customer_closet_notes"]
    assert engine.get("mem_customer_closet_notes").content == "the wardrobe budget for Q3"
    assert TombstoneStore(db).list_copy_deletions() == []


def test_a_seed_redaction_does_not_delete_a_real_lookalike(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    _real_lookalike(db)

    seed = SeedEngine(db_path=db)
    seed.ingest(_memory("mem_customer", "redacted"))  # content edit on the parent

    assert _all_ids(db) == ["mem_customer", "mem_customer_closet_notes"]
    assert seed.delete("mem_customer") is True
    assert _all_ids(db) == ["mem_customer_closet_notes"]


def test_a_pulled_memory_shaped_like_a_closet_is_ingested_untouched(tmp_path: Path) -> None:
    """No closet detection at the wire: the marker is local provenance, not a shape."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    now = datetime.now(timezone.utc).isoformat()
    row = {
        "id": "mem_customer_closet_notes",
        "content": "the wardrobe budget for Q3",
        "memory_type": "fact",
        "project": None,
        "source_type": "ui",
        "source_session_id": None,
        "source_timestamp": now,
        "confidence": 1.0,
        "related_to": [],
        "expires_at": None,
        "superseded_by": None,
        "created_at": now,
        "updated_at": now,
        "deleted_at": None,
    }

    result = pull(
        engine=engine,
        tombstones=TombstoneStore(db),
        client=_RecordingClient(rows=[row]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.applied_live == 1
    assert engine.get("mem_customer_closet_notes").content == "the wardrobe budget for Q3"
    assert _marked_ids(db) == []
    assert [m.id for m in engine.list_all()] == ["mem_customer_closet_notes"]


def _strip_marker(db: Path) -> None:
    """Take a store back to its pre-marker shape: copies present, column absent.

    The replacement table is written out in full rather than built with
    ``CREATE TABLE ... AS SELECT``, which does NOT carry the primary key over.
    Without it ``INSERT OR REPLACE`` stops replacing and starts appending, so a
    later edit of a memory leaves the old row sitting beside the new one and
    every read gets whichever comes first — a fixture that quietly does not
    behave like a real store.
    """
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE m2 (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            enriched_content TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            project TEXT,
            source_type TEXT NOT NULL,
            source_session_id TEXT,
            source_timestamp TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            related_to TEXT DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            expires_at TEXT
        );
        INSERT INTO m2 SELECT id, content, enriched_content, memory_type, project,
            source_type, source_session_id, source_timestamp, confidence, related_to,
            created_at, updated_at, expires_at FROM memories;
        DROP TABLE memories;
        ALTER TABLE m2 RENAME TO memories;
        """
    )
    conn.commit()
    conn.close()


def _insert_legacy(db: Path, mid: str, content: str, related_to: list[str]) -> None:
    """Write a row directly, as a pre-marker client would have."""
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO memories (id, content, enriched_content, memory_type, project, source_type,"
        " source_session_id, source_timestamp, confidence, related_to, created_at, updated_at, expires_at)"
        " VALUES (?, ?, ?, 'fact', NULL, 'cli', NULL, ?, 1.0, ?, ?, ?, NULL)",
        (mid, content, content, now, json.dumps(related_to), now, now),
    )
    conn.commit()
    conn.close()


def _tier_b_store(tmp_path: Path) -> Path:
    """A legacy store whose copy text no longer matches what the parent derives.

    This is what a pre-fix fallback edit leaves: the parent was rewritten and the
    copies kept the removed text. Strong evidence, but not proof, so the copy is
    marked and its text is kept.
    """
    db = _legacy_store(tmp_path)
    conn = sqlite3.connect(str(db))
    conn.execute(
        "UPDATE memories SET content = ?, enriched_content = ? WHERE id = ?",
        (_turns("harmlessreplacement"), "x", "sess-2026-01"),
    )
    conn.commit()
    conn.close()
    return db


def _write_legacy(engine, memory: Memory) -> None:
    """Write the fixed two-speaker fixture in the old on-disk format."""
    engine.ingest(memory)
    turns = json.loads(memory.content)
    for speaker, slug in (("Alice", "alice"), ("Bob", "bob")):
        content = json.dumps([turn for turn in turns if turn["speaker"] == speaker])
        with engine._conn:
            engine._conn.execute(
                "INSERT INTO memories (id, content, enriched_content, memory_type, project, source_type, "
                "source_session_id, source_timestamp, confidence, related_to, created_at, updated_at, "
                "expires_at, is_closet) "
                "SELECT ?, ?, ?, memory_type, project, source_type, source_session_id, source_timestamp, "
                "confidence, ?, created_at, updated_at, expires_at, 1 FROM memories WHERE id = ?",
                (memory.id + "_closet_" + slug, content, content, json.dumps([memory.id]), memory.id),
            )
            engine._conn.execute(
                "INSERT INTO memory_embeddings (id, embedding, model_id) "
                "SELECT ?, embedding, model_id FROM memory_embeddings WHERE id = ?",
                (memory.id + "_closet_" + slug, memory.id),
            )


def _legacy_store(tmp_path: Path) -> Path:
    """A store written by a pre-marker client: closets present, column absent."""
    db = tmp_path / "memories.db"
    _write_legacy(_bloom(db), _memory("sess-2026-01", _turns()))
    _strip_marker(db)
    return db


def _legacy_store_open(tmp_path: Path) -> tuple[Path, "BloomEngine"]:
    """A pre-marker store, with an engine already open on it.

    The engine is opened BEFORE the legacy rows exist. Opening one now also
    removes what the marker migration marks, and these tests are about what the
    announcement queue does with that state afterwards, so they build it around
    an engine that is already up rather than through a second open.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    _write_legacy(engine, _memory("sess-2026-01", _turns()))
    _strip_marker(db)
    _migrate_only(db)
    _queue_legacy(db, _marked_ids(db))
    return db, engine


def test_migration_marks_closets_it_can_re_derive_from_a_live_parent(tmp_path: Path) -> None:
    db = _legacy_store(tmp_path)
    assert MARKER_COLUMN not in {r[1] for r in _rows(db, "PRAGMA table_info(memories)")}

    _migrate_only(db)  # the migration alone; an open would also remove what it marks

    assert _marked_ids(db) == ["sess-2026-01_closet_alice", "sess-2026-01_closet_bob"]
    assert [m.id for m in _bloom(db).list_all()] == ["sess-2026-01"]


def test_a_still_derived_cloud_copy_is_not_ingested_after_cleanup(tmp_path: Path) -> None:
    """A device on the older release keeps publishing the copy we just removed.

    Its publication carries a FRESHER stamp than the row this store removed, so
    the deletion record does not cover it. Taken at face value that looks like
    someone reclaiming the id, and the speaker text comes back as an ordinary
    memory, drops the suppression, and is pushed up again.
    """
    db = tmp_path / "memories.db"
    writer = _bloom(db)
    _write_legacy(writer, _memory("sess-2026-01", _turns()))
    copy = writer.get("sess-2026-01_closet_alice")
    assert copy is not None and SECRET in copy.content
    writer._conn.close()

    engine = _bloom(db)  # the upgrade: the copy goes, recorded at its own stamp
    assert engine.get("sess-2026-01_closet_alice") is None
    store = TombstoneStore(db)

    later = copy.updated_at + timedelta(days=1)
    row = _cloud_row("sess-2026-01_closet_alice", copy.content, when=later)
    row["related_to"] = ["sess-2026-01"]
    row["created_at"] = copy.created_at.isoformat()
    pull(engine=engine, tombstones=store, client=_RecordingClient([row]), state=SyncState(), poppy_dir=tmp_path)

    assert engine.get("sess-2026-01_closet_alice") is None  # not resurrected
    assert "sess-2026-01_closet_alice" not in {m.id for m in engine.list_all()}
    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    # The parent is a real memory and still goes up; the copy of it does not.
    assert {r["id"] for r in client.upserts} == {"sess-2026-01"}
    # ... and the record has moved up to the stamp just seen, so the next
    # publication of the same copy is covered without re-grading it.
    assert _local_deletion_time(db, "sess-2026-01_closet_alice") == later


def test_a_genuine_recreation_after_cleanup_still_lands(tmp_path: Path) -> None:
    """The guard above must not swallow a real memory written at a copy's id."""
    db = tmp_path / "memories.db"
    writer = _bloom(db)
    _write_legacy(writer, _memory("sess-2026-01", _turns()))
    copy_updated = writer.get("sess-2026-01_closet_alice").updated_at
    writer._conn.close()

    engine = _bloom(db)
    store = TombstoneStore(db)

    later = copy_updated + timedelta(days=1)
    note = _cloud_row("sess-2026-01_closet_alice", "an independent note of my own", when=later)
    pull(engine=engine, tombstones=store, client=_RecordingClient([note]), state=SyncState(), poppy_dir=tmp_path)

    assert engine.get("sess-2026-01_closet_alice").content == "an independent note of my own"
    assert _local_deletion_time(db, "sess-2026-01_closet_alice") is None  # reclaimed


def test_marking_and_removing_legacy_copies_commit_as_one(tmp_path: Path) -> None:
    """A crash between the two must not leave rows marked and queued.

    The marks and the upload entries the migration writes are only correct
    together with the removal that consumes them. Committed on their own they
    are an upload this version never drains and an older client sharing the
    store does.
    """
    db = _legacy_store(tmp_path)
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TRIGGER refuse_cleanup BEFORE DELETE ON memories BEGIN SELECT RAISE(ABORT, 'cleanup interrupted'); END"
    )
    conn.commit()
    conn.close()

    with pytest.raises(sqlite3.IntegrityError, match="cleanup interrupted"):
        _bloom(db)

    assert MARKER_COLUMN not in {r[1] for r in _rows(db, "PRAGMA table_info(memories)")}
    assert _rows(db, "SELECT id FROM legacy_closet_ids") == []

    conn = sqlite3.connect(str(db))
    conn.execute("DROP TRIGGER refuse_cleanup")
    conn.commit()
    conn.close()

    engine = _bloom(db)  # the retry does both
    assert _marked_ids(db) == []
    assert [m.id for m in engine.list_all()] == ["sess-2026-01"]
    assert _pending_ids(TombstoneStore(db)) == []


def test_a_pre_marker_store_is_cleaned_on_the_first_open(tmp_path: Path) -> None:
    """One open, not two.

    A store written before the marker existed has nothing marked to find, so the
    removal has to run after the migration that identifies the copies. Running
    it first left them stored, recallable and queued for upload for the whole of
    that session.
    """
    db = _legacy_store(tmp_path)

    engine = _bloom(db)

    assert _marked_ids(db) == []
    assert _all_ids(db) == ["sess-2026-01"]
    assert [m.id for m in engine.list_all()] == ["sess-2026-01"]
    assert _pending_ids(TombstoneStore(db)) == []


def test_migration_leaves_a_real_memory_shaped_like_an_orphan_closet_alone(tmp_path: Path) -> None:
    """The orphan purge is conjunctive, so a hand-named memory survives it.

    ``mem_customer_closet_notes`` splits like a closet and has no parent row, but
    it carries prose rather than a single-speaker turn list, so it fails the
    shape test and is left listed, recallable and unmarked.
    """
    db = _legacy_store(tmp_path)
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM memories WHERE id = ?", ("sess-2026-01",))
    conn.commit()
    conn.close()
    # Even a matching back-reference is not enough on its own.
    _insert_legacy(db, "mem_customer_closet_notes", "the wardrobe budget for Q3", ["mem_customer"])

    engine = _bloom(db)

    assert "mem_customer_closet_notes" in _all_ids(db)
    assert "mem_customer_closet_notes" not in _marked_ids(db)
    assert "mem_customer_closet_notes" in {m.id for m in engine.list_all()}


def test_migration_is_idempotent_and_runs_once(tmp_path: Path) -> None:
    db = _legacy_store(tmp_path)
    _bloom(db)
    _bloom(db)  # Remove the derived rows created by the older migration.
    before = _rows(db, "SELECT id, content, is_closet FROM memories ORDER BY id")

    mark_legacy_copies_for_cleanup(sqlite3.connect(str(db)), had_bloom_schema=True)
    _bloom(db)
    SeedEngine(db_path=db)

    assert _rows(db, "SELECT id, content, is_closet FROM memories ORDER BY id") == before


def test_migration_runs_on_a_seed_only_store_without_bloom_columns(tmp_path: Path) -> None:
    """Seed opens the store first on a machine that never loaded the default engine."""
    db = tmp_path / "memories.db"
    seed = SeedEngine(db_path=db)
    seed.ingest(_memory("plain", "just a note"))

    assert MARKER_COLUMN in {r[1] for r in _rows(db, "PRAGMA table_info(memories)")}
    assert [m.id for m in seed.list_all()] == ["plain"]


def test_a_single_speaker_session_gets_no_closet(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    solo = json.dumps([{"speaker": "Alice", "text": "one"}, {"speaker": "Alice", "text": "two"}])
    _bloom(db).ingest(_memory("sess-2026-01", solo))

    assert _all_ids(db) == ["sess-2026-01"]


def test_forgetting_a_closet_id_directly_writes_no_speaker_text(tmp_path: Path) -> None:
    """``retrieve`` surfaces closets, so a closet id is reachable by forget.

    The ordinary path would snapshot the speaker turns into ``ui_tombstones``
    and push them as the body of a soft-delete. It must not.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    _write_legacy(engine, _memory("sess-2026-01", _turns()))

    result = forget(engine, tmp_path, "sess-2026-01_closet_alice")
    assert result.deleted is True

    store = TombstoneStore(db)
    assert store.list_all() == []  # nothing restorable, nothing snapshotted
    assert store.get("sess-2026-01_closet_alice") is None
    assert {ct.id for ct in store.list_copy_deletions()} == {"sess-2026-01_closet_alice"}
    assert engine.get("sess-2026-01_closet_alice") is None

    conn = sqlite3.connect(str(db))
    conn.execute(  # a pre-fix leak
        "INSERT OR IGNORE INTO legacy_closet_ids (id) VALUES ('sess-2026-01_closet_alice')"
    )
    conn.commit()
    conn.close()

    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    wire = json.dumps([r for r in client.upserts if r["id"] == "sess-2026-01_closet_alice"])
    assert wire == "[]"
    assert SECRET not in wire


def test_forgetting_a_real_lookalike_still_tombstones_it_normally(tmp_path: Path) -> None:
    """The closet branch is marker-gated, so a real memory keeps its restore window."""
    db = tmp_path / "memories.db"
    _real_lookalike(db)
    engine = _bloom(db)

    result = forget(engine, tmp_path, "mem_customer_closet_notes")

    assert result.deleted is True
    store = TombstoneStore(db)
    assert [t.memory.id for t in store.list_all()] == ["mem_customer_closet_notes"]
    assert store.get("mem_customer_closet_notes").memory.content == "the wardrobe budget for Q3"
    assert store.list_copy_deletions() == []


def _cloud_row(mid: str, content: str, *, deleted: bool = False, when: datetime | None = None) -> dict:
    iso = (when or datetime.now(timezone.utc)).isoformat()
    return {
        "id": mid,
        "content": content,
        "memory_type": "fact",
        "project": None,
        "source_type": "ui",
        "source_session_id": None,
        "source_timestamp": iso,
        "confidence": 1.0,
        "related_to": [],
        "expires_at": None,
        "superseded_by": None,
        "created_at": iso,
        "updated_at": iso,
        "deleted_at": iso if deleted else None,
    }


def _legacy_deletion_row(mid: str, when: datetime | None = None, *, updated: datetime | None = None) -> dict:
    """Exactly the row the previous release sent to delete a derived copy.

    Every field it pinned to a constant, because that whole shape is what the
    reader recognises: a row carrying a real project or source is a real memory
    and takes the ordinary path however its body reads.
    """
    iso = (when or datetime.now(timezone.utc)).isoformat()
    return {
        "id": mid,
        "content": LEGACY_DELETION_BODY,
        "memory_type": "fact",
        "project": None,
        "source_type": None,
        "source_session_id": None,
        "source_timestamp": iso,
        "confidence": 1.0,
        "related_to": [],
        "expires_at": None,
        "superseded_by": None,
        "created_at": iso,
        "updated_at": (updated or when or datetime.now(timezone.utc)).isoformat(),
        "deleted_at": iso,
    }


def test_pull_never_overwrites_a_marked_closet_with_a_live_cloud_row(tmp_path: Path) -> None:
    """A live cloud row for a marked closet id must not become a real memory.

    Ingesting it would write is_closet=0, so the copy would stop being derived
    data: it would list, it would push live, and forgetting its parent would no
    longer reach it — the secret ends up in the cloud as a live row.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    _write_legacy(engine, _memory("sess-2026-01", _turns()))
    store = TombstoneStore(db)

    result = pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[_cloud_row("sess-2026-01_closet_alice", "cloud copy of Alice's turns")]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.skipped_copies == 1
    assert result.applied_live == 0
    assert _marked_ids(db) == ["sess-2026-01_closet_alice", "sess-2026-01_closet_bob"]
    assert [m.id for m in engine.list_all()] == ["sess-2026-01"]
    assert engine.get("sess-2026-01_closet_alice").content != "cloud copy of Alice's turns"

    # And the parent's redaction still reaches it.
    assert forget(engine, tmp_path, "sess-2026-01").deleted is True
    assert _all_ids(db) == []
    assert not _bloom(db).retrieve(SECRET, limit=10)

    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert [r for r in client.upserts if r["deleted_at"] is None] == []


def test_pull_never_deletes_a_marked_closet_via_a_cloud_tombstone(tmp_path: Path) -> None:
    """A cloud tombstone for a closet id must not remove a copy bloom owns.

    The server-side cleanup of pre-marker leaks sends exactly these
    rows. They refer to the cloud's stale copy, not to the row this device
    holds, and the local row is the one the memory beside it is graded against.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    _write_legacy(engine, _memory("sess-2026-01", _turns()))
    store = TombstoneStore(db)

    result = pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[_cloud_row("sess-2026-01_closet_alice", "leaked copy", deleted=True)]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.skipped_copies == 1
    assert result.applied_tombstones == 0
    assert _marked_ids(db) == ["sess-2026-01_closet_alice", "sess-2026-01_closet_bob"]
    assert store.list_all() == []  # no snapshot of the leaked text either
    # The local row is untouched. It is no longer offered by recall: nothing
    # derives these rows for recall any more, and the memory they were copied
    # from is indexed in full.
    assert engine.get("sess-2026-01_closet_alice") is not None
    assert "sess-2026-01_closet_alice" not in {r.memory.id for r in engine.retrieve(SECRET, limit=10)}


def test_pull_does_not_resurrect_a_closet_this_device_just_forgot(tmp_path: Path) -> None:
    """An older cloud row must not undo a closet deletion inside its window."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    _write_legacy(engine, _memory("sess-2026-01", _turns()))
    store = TombstoneStore(db)
    assert forget(engine, tmp_path, "sess-2026-01_closet_alice", tombstones=store).deleted is True

    stale = _cloud_row(
        "sess-2026-01_closet_alice",
        f'[{{"speaker":"Alice","text":"{SECRET}"}}]',
        when=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    result = pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[stale]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.skipped_copies == 1
    assert result.applied_live == 0
    assert engine.get("sess-2026-01_closet_alice") is None
    assert not [r for r in engine.retrieve(SECRET, limit=10) if r.memory.id.endswith("_closet_alice")]


def test_pull_applies_a_real_closet_shaped_memory_when_a_parent_is_present(tmp_path: Path) -> None:
    """The skip reads the LOCAL marker, never the incoming id's shape.

    A real cloud memory sitting under a live parent's closet prefix is unmarked
    here, so it syncs like any other memory.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("mem_customer", _turns()))
    store = TombstoneStore(db)

    result = pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[_cloud_row("mem_customer_closet_notes", "the wardrobe budget for Q3")]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.applied_live == 1
    assert result.skipped_copies == 0
    assert engine.get("mem_customer_closet_notes").content == "the wardrobe budget for Q3"
    assert sorted(m.id for m in engine.list_all()) == ["mem_customer", "mem_customer_closet_notes"]


def test_a_seed_write_over_a_closet_id_makes_it_a_real_memory(tmp_path: Path) -> None:
    """A note written at a closet's id must stop being derived data.

    Keeping the marker would hide the user's note from every list and then
    destroy it when the unrelated parent memory was deleted.
    """
    db = tmp_path / "memories.db"
    seed = SeedEngine(db_path=db)
    _write_legacy(_bloom(db), _memory("sess-2026-01", _turns()))

    seed.ingest(_memory("sess-2026-01_closet_alice", "my own note about Alice"))

    assert _marked_ids(db) == ["sess-2026-01_closet_bob"]
    assert sorted(m.id for m in seed.list_all()) == ["sess-2026-01", "sess-2026-01_closet_alice"]

    # It survives the unrelated parent's deletion, and syncs as a real memory.
    # The row still marked as a copy goes with that parent.
    assert forget(seed, tmp_path, "sess-2026-01").deleted is True
    assert _all_ids(db) == ["sess-2026-01_closet_alice"]
    assert seed.get("sess-2026-01_closet_alice").content == "my own note about Alice"


def test_migration_leaves_a_real_memory_at_a_derivable_closet_id_alone(tmp_path: Path) -> None:
    """Re-derivability is not enough — the row must also BE closet-shaped.

    `meeting` is multi-speaker with an Alice, so bloom would mint
    `meeting_closet_alice`. A real memory already sitting at that id holds the
    only copy of something the user cares about; marking and rebuilding it would
    overwrite the body with Alice's turns and hide it from every list.
    """
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("meeting", "placeholder"))
    _strip_marker(db)
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE memories SET content = ?, enriched_content = ? WHERE id = ?", (_turns(), "x", "meeting"))
    conn.commit()
    conn.close()
    _insert_legacy(db, "meeting_closet_alice", "ONLY COPY OF CUSTOMER CONTRACT", [])

    engine = _bloom(db)

    assert engine.get("meeting_closet_alice").content == "ONLY COPY OF CUSTOMER CONTRACT"
    assert _marked_ids(db) == []
    assert sorted(m.id for m in engine.list_all()) == ["meeting", "meeting_closet_alice"]


_LIST_SPEAKER = '[{"speaker": ["Alice"], "text": "malformed"}]'


def test_a_malformed_legacy_row_does_not_abort_the_migration(tmp_path: Path) -> None:
    """A non-string speaker must classify as "leave alone", not raise.

    A raise inside the backfill rolls back the ALTER too, so every subsequent
    open retries and fails the same way: the store becomes unopenable.
    """
    db = _legacy_store(tmp_path)
    _insert_legacy(db, "meeting_closet_alice", _LIST_SPEAKER, ["meeting"])
    _insert_legacy(db, "meeting_closet_notes", "just some prose", [])

    _migrate_only(db)  # must not raise
    # The well-formed closets alongside them still got marked, before the open
    # below removes them.
    assert _marked_ids(db) == ["sess-2026-01_closet_alice", "sess-2026-01_closet_bob"]

    engine = _bloom(db)  # must not raise either

    assert MARKER_COLUMN in {r[1] for r in _rows(db, "PRAGMA table_info(memories)")}
    assert engine.get("meeting_closet_alice").content == _LIST_SPEAKER
    assert engine.get("meeting_closet_notes").content == "just some prose"
    listed = {m.id for m in engine.list_all()}
    assert {"meeting_closet_alice", "meeting_closet_notes"} <= listed


def test_a_row_the_classifier_cannot_read_does_not_abort_the_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A grading failure must leave the store openable.

    The grading runs inside the transaction that adds the marker column, so an
    exception rolls that column back too and every later open retries and fails
    the same way. Nothing is graded, so nothing is removed either.
    """
    from poppy.engine import _legacy_copies

    db = _legacy_store(tmp_path)

    def unreadable(*args: object, **kwargs: object) -> tuple[str, str | None]:
        raise RuntimeError("row cannot be graded")

    monkeypatch.setattr(_legacy_copies, "classify_legacy_copy", unreadable)

    engine = _bloom(db)  # must not raise

    assert MARKER_COLUMN in {r[1] for r in _rows(db, "PRAGMA table_info(memories)")}
    assert _marked_ids(db) == []
    assert engine.get("sess-2026-01_closet_alice") is not None


def test_ingesting_malformed_turns_does_not_crash_the_engine(tmp_path: Path) -> None:
    """Full-content enrichment tolerates malformed speaker values."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _LIST_SPEAKER))

    assert _all_ids(db) == ["sess-2026-01"]
    assert SeedEngine(db_path=db).get("sess-2026-01").content == _LIST_SPEAKER


def test_expiry_does_not_announce_closet_deletions(tmp_path: Path) -> None:
    """TTL is not a redaction: the cloud row carries the same expires_at."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    expired = _memory("sess-2026-01", _turns())
    expired.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
    engine.ingest(expired)

    assert engine.purge_expired() == 1
    assert _all_ids(db) == []
    assert TombstoneStore(db).list_copy_deletions() == []


def test_forgetting_a_closet_reports_no_restore_window(tmp_path: Path) -> None:
    """Nothing was snapshotted, so advertising a 7-day window would be a lie."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    _write_legacy(engine, _memory("sess-2026-01", _turns()))

    result = forget(engine, tmp_path, "sess-2026-01_closet_alice")

    assert result.deleted is True
    assert result.tombstoned is True
    assert result.tombstone is None


@pytest.mark.parametrize("engine_name", ["bloom", "seed"])
def test_edit_memory_refuses_a_marked_closet(tmp_path: Path, engine_name: str) -> None:
    """`poppy edit <closet id> --project x` keeps the text and would clear the marker.

    The copy would then list, push live with the secret, and survive the
    parent's redaction. Editing a derived copy is refused; edit the parent.
    """
    from poppy.lifecycle import edit_memory

    db = tmp_path / "memories.db"
    writer = _bloom(db)
    engine = _bloom(db) if engine_name == "bloom" else SeedEngine(db_path=db)
    _write_legacy(writer, _memory("sess-2026-01", _turns()))

    with pytest.raises(KeyError, match="memory not found: sess-2026-01_closet_alice") as exc:
        edit_memory(engine, "sess-2026-01_closet_alice", project="other")

    assert exc.value.args == ("memory not found: sess-2026-01_closet_alice",)
    assert _marked_ids(db) == ["sess-2026-01_closet_alice", "sess-2026-01_closet_bob"]
    assert [m.id for m in engine.list_all()] == ["sess-2026-01"]


def test_a_changed_content_write_still_reclaims_the_id(tmp_path: Path) -> None:
    """The round-1 guarantee survives: a real note over a closet id is a real memory."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    _write_legacy(engine, _memory("sess-2026-01", _turns()))

    engine.ingest(_memory("sess-2026-01_closet_alice", "my own note"))

    assert _marked_ids(db) == ["sess-2026-01_closet_bob"]
    assert sorted(m.id for m in engine.list_all()) == ["sess-2026-01", "sess-2026-01_closet_alice"]


def test_a_reclaimed_closet_id_still_receives_cloud_updates(tmp_path: Path) -> None:
    """Forget a copy, write a real note at that id, then pull a newer version.

    The closet tombstone must not make pull skip it: the watermark advances
    either way, so a skip loses the update permanently.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    _write_legacy(engine, _memory("sess-2026-01", _turns()))
    store = TombstoneStore(db)
    assert forget(engine, tmp_path, "sess-2026-01_closet_alice", tombstones=store).deleted is True
    assert store.has_copy_deletion("sess-2026-01_closet_alice") is True

    # The id is reclaimed by an independent note, which drops the stale record.
    SeedEngine(db_path=db).ingest(_memory("sess-2026-01_closet_alice", "a note of my own"))
    assert store.has_copy_deletion("sess-2026-01_closet_alice") is False

    newer = _cloud_row(
        "sess-2026-01_closet_alice",
        "the newer version from another device",
        when=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    result = pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[newer]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.skipped_copies == 0
    assert result.applied_live == 1
    assert engine.get("sess-2026-01_closet_alice").content == "the newer version from another device"


def test_a_reclaimed_id_syncs_even_if_the_tombstone_survives(tmp_path: Path) -> None:
    """Second line: the skip requires no live local row, not just a cleared record."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    _write_legacy(engine, _memory("sess-2026-01", _turns()))
    store = TombstoneStore(db)
    forget(engine, tmp_path, "sess-2026-01_closet_alice", tombstones=store)
    SeedEngine(db_path=db).ingest(_memory("sess-2026-01_closet_alice", "a note of my own"))
    store.add_copy_deletions(["sess-2026-01_closet_alice"])  # re-record it by hand

    result = pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[_cloud_row("sess-2026-01_closet_alice", "newer still")]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.applied_live == 1
    assert engine.get("sess-2026-01_closet_alice").content == "newer still"


def test_migration_marks_a_closet_that_arrived_through_the_cloud(tmp_path: Path) -> None:
    """The enrichment proves nothing about a closet's provenance.

    Clients at or below 0.3.0 push closets live, and a second device that pulls
    one runs it through the ordinary ingest, which re-enriches it with the
    FULL-session preamble instead of the per-speaker one. That row is a genuine
    legacy closet with a completely different enrichment, so requiring the
    per-speaker preamble left exactly these unmarked, listed, and alive through
    their parent's redaction.
    """
    db = _legacy_store(tmp_path)
    conn = sqlite3.connect(str(db))
    # What HybridEngine.ingest writes for a pulled row: the joint-session
    # preamble, not "Alice's contributions".
    conn.execute(
        "UPDATE memories SET enriched_content = ? WHERE id = ?",
        (
            "Conversation on January 01, 2026 at 12:00 PM between Alice and Bob.\n\nD1 Alice: " + SECRET,
            "sess-2026-01_closet_alice",
        ),
    )
    conn.commit()
    conn.close()

    _migrate_only(db)
    assert _marked_ids(db) == ["sess-2026-01_closet_alice", "sess-2026-01_closet_bob"]

    engine = _bloom(db)

    assert [m.id for m in engine.list_all()] == ["sess-2026-01"]

    assert forget(engine, tmp_path, "sess-2026-01").deleted is True
    assert _all_ids(db) == []
    assert not _bloom(db).retrieve(SECRET, limit=10)


def _stored_updated_at(db: Path, memory_id: str) -> str:
    return _rows(db, "SELECT updated_at FROM memories WHERE id = ?", (memory_id,))[0][0]


def _pending_ids(store: TombstoneStore) -> list[str]:
    """Just the ids from the announcement queue; the timestamp is asserted separately."""
    return [memory_id for memory_id, _ in store.pending_legacy_announcements()]


def _backup_rows(db: Path) -> dict[str, str]:
    return {r[0]: r[1] for r in _rows(db, "SELECT id, action FROM closet_migration_backup ORDER BY id")}


def test_an_inferentially_adopted_copy_keeps_text_and_backup(tmp_path: Path) -> None:
    """Inference preserves the original text and backup without uploading it."""
    db = _tier_b_store(tmp_path)

    _migrate_only(db)
    # Only Alice's text changed, so only her copy is inferential; Bob's still
    # matches what the parent derives and is proven.
    assert _backup_rows(db) == {"sess-2026-01_closet_alice": "adopted"}
    assert "sess-2026-01_closet_alice" in _marked_ids(db)
    # NOT announced: an announcement deletes the cloud row everywhere, which
    # inference does not earn. Proven rows also need no upload.
    assert _pending_ids(TombstoneStore(db)) == []

    engine = _bloom(db)

    kept = engine.get("sess-2026-01_closet_alice")
    assert SECRET in kept.content  # not rewritten, not destroyed
    # The inferred tier survives the open; only the proven one is removed.
    assert engine.get("sess-2026-01_closet_bob") is None

    # Kept, but not shown: it still holds a projection of text the memory
    # beside it holds, so a listing must not offer it as a memory of its own.
    assert [m.id for m in engine.list_all()] == ["sess-2026-01"]
    client = _RecordingClient()
    push(
        engine=engine,
        tombstones=TombstoneStore(db),
        client=client,
        state=SyncState(),
        poppy_dir=tmp_path,
    )
    assert not any(SECRET in json.dumps(r) for r in client.upserts if r["deleted_at"] is None)


def test_the_migration_backup_ages_out_on_the_tombstone_clock(tmp_path: Path) -> None:
    db = _tier_b_store(tmp_path)
    _bloom(db)
    assert len(_backup_rows(db)) == 1

    store = TombstoneStore(db)
    assert store.purge_expired(pushed_through=None) == 0  # nothing old enough yet
    assert len(_backup_rows(db)) == 1

    conn = sqlite3.connect(str(db))
    conn.execute(
        "UPDATE closet_migration_backup SET migrated_at = ?",
        ((datetime.now(timezone.utc) - timedelta(days=8)).isoformat(),),
    )
    conn.commit()
    conn.close()

    TombstoneStore(db).purge_expired(pushed_through=None)
    assert _backup_rows(db) == {}


def test_a_pulled_closet_tombstone_never_becomes_a_restorable_entry(tmp_path: Path) -> None:
    """Another device's closet deletion, on a device that never had that copy.

    Filed as a ui tombstone it would show in Trash with the placeholder
    (`LEGACY_DELETION_BODY`) as its body, and restoring it would create junk
    and push it back.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    store = TombstoneStore(db)
    row = _legacy_deletion_row("sess-2026-01_closet_alice")

    result = pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[row]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.applied_tombstones == 0
    assert result.skipped_echoes == 1
    assert store.list_all() == []  # nothing in Trash
    assert _local_deletion_time(db, "sess-2026-01_closet_alice") is not None


def test_an_ordinary_tombstone_is_still_a_restorable_entry(tmp_path: Path) -> None:
    """The sentinel check must not swallow a real deletion."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    store = TombstoneStore(db)

    result = pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[_cloud_row("mem_real", "a real memory body", deleted=True)]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.applied_tombstones == 1
    assert [t.memory.id for t in store.list_all()] == ["mem_real"]
    assert store.list_copy_deletions() == []


def test_a_real_memory_matching_the_placeholder_still_gets_deleted(tmp_path: Path) -> None:
    """Matching the body is a hint, not proof of provenance.

    A live local row means the id belongs to a real memory, so its deletion runs
    through the ordinary tombstone path however its body reads: it applies, it
    leaves a restorable entry, and it records no permanent suppression.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("mem_odd", LEGACY_DELETION_BODY))
    store = TombstoneStore(db)

    row = _cloud_row("mem_odd", LEGACY_DELETION_BODY, deleted=True)
    result = pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[row]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.applied_tombstones == 1
    assert engine.get("mem_odd") is None  # the deletion actually applied
    assert [t.memory.id for t in store.list_all()] == ["mem_odd"]
    assert _local_deletion_time(db, "mem_odd") is None


def test_supersede_refuses_a_marked_closet(tmp_path: Path) -> None:
    """Superseding a copy snapshots its speaker turns into Trash and onto the wire.

    A later restore then brings them back unmarked, listed, and pushed live.
    """
    from poppy.lifecycle import supersede_memory

    db = tmp_path / "memories.db"
    engine = _bloom(db)
    _write_legacy(engine, _memory("sess-2026-01", _turns()))

    with pytest.raises(KeyError, match="memory not found: sess-2026-01_closet_alice") as exc:
        supersede_memory(engine, _memory("mem_new", "replacement"), "sess-2026-01_closet_alice", poppy_dir=tmp_path)

    assert exc.value.args == ("memory not found: sess-2026-01_closet_alice",)
    store = TombstoneStore(db)
    assert store.list_all() == []  # nothing snapshotted
    assert _marked_ids(db) == ["sess-2026-01_closet_alice", "sess-2026-01_closet_bob"]


def test_remember_with_supersedes_pointing_at_a_closet_is_refused(tmp_path: Path) -> None:
    """The reachable surface: `poppy remember --supersedes <copy id>` and the MCP twin."""
    from poppy.write_flow import remember

    db = tmp_path / "memories.db"
    engine = _bloom(db)
    _write_legacy(engine, _memory("sess-2026-01", _turns()))

    with pytest.raises(KeyError, match="memory not found: sess-2026-01_closet_alice"):
        remember(engine, tmp_path, content="replacement", supersedes="sess-2026-01_closet_alice")

    assert TombstoneStore(db).list_all() == []  # no snapshot of the speaker turns
    assert _marked_ids(db) == ["sess-2026-01_closet_alice", "sess-2026-01_closet_bob"]


def test_auto_supersede_never_picks_a_derived_copy(tmp_path: Path) -> None:
    """No human types an id on this path.

    A copy inherits the project and memory_type of the memory it came from, so
    it passes every filter this path applies. Two things keep it out: recall no
    longer offers a row marked as a copy, and the reconciler refuses one that
    reaches it by any other route.
    """
    from poppy.capture.reconciler import find_candidates

    db = tmp_path / "memories.db"
    engine = _bloom(db)
    _write_legacy(engine, _memory("sess-2026-01", _turns()))

    # The copy is the shorter document, so it would outrank the memory it came
    # from under the test reranker: the ordering that used to pick it.
    ranked = [s.memory.id for s in engine.retrieve(SECRET, limit=10)]
    assert "sess-2026-01_closet_alice" not in ranked

    candidates = find_candidates(engine, _memory("mem_new", SECRET), top_k=10)

    assert [c.memory.id for c in candidates] == ["sess-2026-01"]


def test_restore_refuses_an_id_that_is_now_a_derived_copy(tmp_path: Path) -> None:
    """Defence in depth for a tombstone an older build could have left behind."""
    from poppy.write_flow import restore

    db = tmp_path / "memories.db"
    engine = _bloom(db)
    _write_legacy(engine, _memory("sess-2026-01", _turns()))
    store = TombstoneStore(db)
    # A tombstone naming a copy, as the unguarded supersede path used to write.
    store.add(engine.get("sess-2026-01_closet_alice"))

    assert restore(engine, tmp_path, "sess-2026-01_closet_alice", tombstones=store).found is False

    assert "sess-2026-01_closet_alice" in _marked_ids(db)
    assert [m.id for m in engine.list_all()] == ["sess-2026-01"]


def test_migration_leaves_a_row_a_present_parent_does_not_account_for(tmp_path: Path) -> None:
    """Parent exists but derives nothing, and the row predates it: not a copy.

    A genuine copy always shares its parent's created_at. A hand-crafted turn
    list pointing at a prose memory does not, and must not be purged.
    """
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("meeting", "just some prose, no speakers"))
    _strip_marker(db)
    _insert_legacy(db, "meeting_closet_alice", json.dumps([{"speaker": "Alice", "text": SECRET}]), ["meeting"])
    conn = sqlite3.connect(str(db))
    conn.execute(
        "UPDATE memories SET created_at = ? WHERE id = ?",
        ("2020-01-01T00:00:00+00:00", "meeting_closet_alice"),
    )
    conn.commit()
    conn.close()

    engine = _bloom(db)

    assert sorted(_all_ids(db)) == ["meeting", "meeting_closet_alice"]
    assert "meeting_closet_alice" not in _marked_ids(db)
    assert sorted(m.id for m in engine.list_all()) == ["meeting", "meeting_closet_alice"]


def test_a_newer_cloud_row_at_a_deleted_copys_id_is_ingested(tmp_path: Path) -> None:
    """Device B writes an independent note at an id device A deleted as a copy.

    Skipping it loses the note for good: the watermark advances either way.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    _write_legacy(engine, _memory("sess-2026-01", _turns()))
    store = TombstoneStore(db)
    forget(engine, tmp_path, "sess-2026-01_closet_alice", tombstones=store)
    assert store.has_copy_deletion("sess-2026-01_closet_alice") is True

    newer = _cloud_row(
        "sess-2026-01_closet_alice",
        "an independent note from device B",
        when=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    result = pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[newer]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.skipped_copies == 0
    assert result.applied_live == 1
    assert engine.get("sess-2026-01_closet_alice").content == "an independent note from device B"
    assert _marked_ids(db) == ["sess-2026-01_closet_bob"]
    # Reclaiming the id drops the deletion record, so it cannot suppress again.
    assert store.has_copy_deletion("sess-2026-01_closet_alice") is False


def test_an_older_cloud_copy_at_a_deleted_copys_id_is_still_skipped(tmp_path: Path) -> None:
    """The round-2 guarantee survives the freshness rule."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    _write_legacy(engine, _memory("sess-2026-01", _turns()))
    store = TombstoneStore(db)
    forget(engine, tmp_path, "sess-2026-01_closet_alice", tombstones=store)

    stale = _cloud_row(
        "sess-2026-01_closet_alice",
        f'[{{"speaker":"Alice","text":"{SECRET}"}}]',
        when=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    result = pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[stale]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.skipped_copies == 1
    assert engine.get("sess-2026-01_closet_alice") is None


def test_sync_ages_out_the_local_side_tables(tmp_path: Path, synced_state) -> None:
    """`poppy ui` startup was the only caller, so a CLI-only user kept them forever.

    That includes the migration's plaintext pre-images of removed text.
    """
    from poppy.sync import sync as run_sync

    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _turns()))
    store = TombstoneStore(db)
    store.note_remote_memories({"sess-2026-01"}, _RecordingClient.base_url)
    forget(engine, tmp_path, "sess-2026-01", tombstones=store)
    store.add_copy_deletions(["sess-2026-01_closet_alice", "sess-2026-01_closet_bob"])
    assert len(store.list_copy_deletions()) == 2

    old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE closet_tombstones SET tombstoned_at = ?", (old,))
    conn.execute("UPDATE ui_tombstones SET tombstoned_at = ?", (old,))
    conn.commit()
    conn.close()

    save(tmp_path, synced_state())  # only a SENT deletion ages out
    run_sync(engine=engine, tombstones=store, client=_RecordingClient(), poppy_dir=tmp_path)

    assert store.list_copy_deletions() == []
    assert store.list_all() == []


def test_an_unsent_tombstone_survives_the_purge_and_still_propagates(tmp_path: Path, synced_state) -> None:
    """The laptop-offline case: forget last week, first sync today.

    Purging by age alone drops the tombstone before push can send it, so the
    cloud row stays live and the next pull brings the forgotten memory back.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("mem_secret", SECRET))
    store = TombstoneStore(db)
    store.note_remote_memories({"mem_secret"}, _RecordingClient.base_url)
    forget(engine, tmp_path, "mem_secret", tombstones=store)

    # Backdate the deletion past the restore window, with the DELETION never
    # pushed. The memory itself was seen in this remote, making its deletion
    # sendable even though the push watermark has not reached it.
    old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE ui_tombstones SET tombstoned_at = ?", (old,))
    conn.commit()
    conn.close()

    from poppy.sync import sync as run_sync

    client = _RecordingClient()
    save(tmp_path, synced_state())  # sync() loads the state itself
    run_sync(engine=engine, tombstones=store, client=client, poppy_dir=tmp_path)

    # It was sent as a deletion, and only then dropped.
    sent = [r for r in client.upserts if r["id"] == "mem_secret"]
    assert len(sent) == 1
    assert sent[0]["deleted_at"] is not None
    assert store.get("mem_secret") is None


def test_an_unsent_closet_deletion_is_not_purged_by_age(tmp_path: Path) -> None:
    """Same rule for the copy deletions: unsent means a leaked cloud copy stays."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _turns()))
    store = TombstoneStore(db)
    forget(engine, tmp_path, "sess-2026-01", tombstones=store)
    store.add_copy_deletions(["sess-2026-01_closet_alice", "sess-2026-01_closet_bob"])

    old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE closet_tombstones SET tombstoned_at = ?", (old,))
    conn.commit()
    conn.close()

    # No push watermark: nothing has been sent, so nothing may be dropped.
    assert store.purge_expired(pushed_through=None) == 0
    assert len(store.list_copy_deletions()) == 2

    # Once a push has covered them, they age out.
    store.purge_expired(pushed_through=datetime.now(timezone.utc).isoformat())
    assert store.list_copy_deletions() == []


def test_the_migration_backup_still_ages_out_without_a_watermark(tmp_path: Path) -> None:
    """Pre-images have nothing to push, so the window alone decides."""
    db = _tier_b_store(tmp_path)
    _bloom(db)
    assert len(_backup_rows(db)) == 1

    conn = sqlite3.connect(str(db))
    conn.execute(
        "UPDATE closet_migration_backup SET migrated_at = ?",
        ((datetime.now(timezone.utc) - timedelta(days=8)).isoformat(),),
    )
    conn.commit()
    conn.close()

    TombstoneStore(db).purge_expired(pushed_through=None)  # no watermark
    assert _backup_rows(db) == {}


def test_a_pulled_copy_deletion_is_stamped_with_the_deletion_time(tmp_path: Path) -> None:
    """Receipt time would suppress a recreation that actually came after it.

    A deletes the copy at 12:00:00; B writes an independent note at that id at
    12:00:02; A hears about the deletion at 12:00:03. Stamping 12:00:03 makes
    the deletion look newer than B's note, so A skips it forever.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    store = TombstoneStore(db)
    deleted_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)

    pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[_legacy_deletion_row("sess-1_closet_alice", deleted_at)]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    recorded = _local_deletion_time(db, "sess-1_closet_alice")
    assert recorded is not None
    assert recorded == deleted_at  # the deletion's time, not now

    # B's recreation is two seconds later, so it must land.
    note = _cloud_row("sess-1_closet_alice", "an independent note", when=deleted_at + timedelta(seconds=2))
    result = pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[note]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.applied_live == 1
    assert engine.get("sess-1_closet_alice").content == "an independent note"


def test_re_pulling_a_copy_deletion_does_not_refresh_its_timestamp(tmp_path: Path) -> None:
    """Re-stamping on every pull kept the record alive forever and re-pushed it."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    store = TombstoneStore(db)
    deleted_at = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    row = _legacy_deletion_row("sess-1_closet_alice", deleted_at)

    for _ in range(3):
        pull(
            engine=engine,
            tombstones=store,
            client=_RecordingClient(rows=[row]),
            state=SyncState(),
            poppy_dir=tmp_path,
        )

    assert _local_deletion_time(db, "sess-1_closet_alice") == deleted_at


def test_unmarked_legacy_rows_are_not_reclassified_on_reopen(tmp_path: Path) -> None:
    """Opening a store never reruns inference after the marker migration."""
    from poppy.engine._legacy_copies import TIER_LIKELY, TIER_PROVEN, legacy_copy_grades

    db = tmp_path / "memories.db"
    engine = _bloom(db)
    _write_legacy(engine, _memory("sess-2026-01", _turns()))
    conn = sqlite3.connect(str(db))
    conn.execute(f"UPDATE memories SET {MARKER_COLUMN} = 0")  # what an older client leaves
    conn.commit()
    conn.close()

    assert sum(tier in (TIER_PROVEN, TIER_LIKELY) for _, tier in legacy_copy_grades(engine._conn)) == 2

    # Reopening never repairs them, and never writes.
    for _ in range(3):
        _bloom(db)
    assert _marked_ids(db) == []
    assert _backup_rows(db) == {}
    assert TombstoneStore(db).list_copy_deletions() == []


def test_reopening_a_store_with_a_lookalike_never_writes(tmp_path: Path) -> None:
    """`poppy list` and `recall` both open the engine; neither may take a write lock."""
    import poppy.engine._legacy_copies as marker

    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("mem_customer", "a plain memory"))
    engine.ingest(_memory("mem_customer_closet_notes", "the wardrobe budget for Q3"))

    calls: list[str] = []
    real_apply = marker.legacy_copy_grades
    marker.legacy_copy_grades = lambda conn: (calls.append("apply"), real_apply(conn))[1]
    try:
        reopened = _bloom(db)
    finally:
        marker.legacy_copy_grades = real_apply

    assert calls == []
    assert reopened._conn.in_transaction is False
    assert _marked_ids(db) == []


def test_a_bad_supersede_target_skips_one_capture_not_the_batch(tmp_path: Path) -> None:
    """Auto-supersede can still name an unsupersedable target through other paths.

    An exception mid-loop would drop every remaining capture in the batch.
    """
    import poppy.capture.reconciler as reconciler_mod
    from poppy.capture.reconciler import Action, Decision, reconcile_and_ingest
    from poppy.config import PoppyConfig

    db = tmp_path / "memories.db"
    engine = _bloom(db)
    _write_legacy(engine, _memory("sess-2026-01", _turns()))

    first = _memory("mem_a", "first capture")
    second = _memory("mem_b", "second capture")

    def fake_decide(_engine, mem, **_kwargs):
        if mem.id == "mem_a":
            return Decision(action=Action.SUPERSEDE, memory=mem, target_id="sess-2026-01_closet_alice")
        return Decision(action=Action.ADD, memory=mem)

    original = reconciler_mod.decide
    reconciler_mod.decide = fake_decide
    try:
        summary = reconcile_and_ingest([first, second], engine=engine, cfg=PoppyConfig(), poppy_dir=tmp_path)
    finally:
        reconciler_mod.decide = original

    assert summary.superseded == 0
    assert summary.skipped == 1
    assert summary.added == 1  # the batch kept going
    assert engine.get("mem_b") is not None
    assert "sess-2026-01_closet_alice" in _marked_ids(db)


def test_a_fresh_store_pushes_no_copy_deletions(tmp_path: Path, synced_state) -> None:
    """Copies created after the fix are never synced, so the cloud has nothing
    to delete — and announcing their removal would delete any real cloud memory
    sharing the id, for every device."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("meeting", _turns()))
    store = TombstoneStore(db)
    store.note_remote_memories({"meeting"}, _RecordingClient.base_url)

    forget(engine, tmp_path, "meeting", tombstones=store)

    assert store.list_copy_deletions() == []
    assert _pending_ids(store) == []  # but nothing is announced

    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=synced_state(), poppy_dir=tmp_path)
    assert {r["id"] for r in client.upserts} == {"meeting"}


def test_a_legacy_store_never_uploads_pending_copy_deletions(tmp_path: Path) -> None:
    """A copy the migration marked may have been leaked by a 0.2.4 client."""
    db = _legacy_store(tmp_path)
    _migrate_only(db)
    _queue_legacy(db, _marked_ids(db))  # queue left by an older client
    legacy = {"sess-2026-01_closet_alice", "sess-2026-01_closet_bob"}
    assert {r[0] for r in _rows(db, "SELECT id FROM legacy_closet_ids")} == legacy
    assert set(_pending_ids(TombstoneStore(db))) == legacy

    # Opening the store retires the queue: nothing here drains it, and an older
    # client sharing this store would upload every entry left pending.
    engine = _bloom(db)
    store = TombstoneStore(db)
    assert _pending_ids(store) == []

    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)

    assert not legacy.intersection(r["id"] for r in client.upserts)
    again = _RecordingClient()
    push(engine=engine, tombstones=store, client=again, state=SyncState(), poppy_dir=tmp_path)
    assert not legacy.intersection(r["id"] for r in again.upserts)


def test_a_local_deletion_record_keeps_its_earliest_time(tmp_path: Path) -> None:
    """The same deletion arrives by cascade and then as its own row.

    A later write must not push the recorded time forward, or re-pulling would
    keep suppressing ever-newer recreations of the id.
    """
    db = tmp_path / "memories.db"
    _bloom(db)
    store = TombstoneStore(db)
    early = datetime(2026, 7, 1, tzinfo=timezone.utc)
    late = datetime(2026, 8, 1, tzinfo=timezone.utc)

    store.add_copy_deletions(["x_closet_alice"], now=early)
    store.add_copy_deletions(["x_closet_alice"], now=late)

    assert store.get_copy_deletion("x_closet_alice").tombstoned_at == early


@pytest.mark.parametrize("engine_name", ["bloom", "seed"])
def test_stats_counts_memories_not_derived_copies(tmp_path: Path, engine_name: str) -> None:
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("sess-2026-01", _turns()))
    engine = _bloom(db) if engine_name == "bloom" else SeedEngine(db_path=db)

    assert engine.stats().memory_count == len(engine.list_all(limit=1000))
    assert engine.stats().memory_count == 1


class _DeadHostClient(_RecordingClient):
    """Every request times out, as a black-holing host does."""

    def __init__(self) -> None:
        super().__init__()
        self.attempts = 0

    def upsert(self, row: dict) -> None:
        self.attempts += 1
        raise TimeoutError("no route to host")


def _queue_legacy(db: Path, ids: list[str]) -> None:
    conn = sqlite3.connect(str(db))
    conn.executemany("INSERT OR IGNORE INTO legacy_closet_ids (id) VALUES (?)", [(i,) for i in ids])
    conn.commit()
    conn.close()


def test_a_dead_host_does_not_grind_through_the_whole_announcement_queue(tmp_path: Path) -> None:
    """Each announcement is a full request holding sync.lock, which also blocks
    `poppy encrypt`. A migrated store can queue thousands; a dead host must not
    mean hours per sync round."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("mem_a", "a memory to push"))
    store = TombstoneStore(db)
    _queue_legacy(db, [f"parent_closet_s{i}" for i in range(100)])

    client = _DeadHostClient()
    with pytest.raises(Exception):
        push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)

    # The main loop's breaker fires, and the announcement pass is not attempted
    # at all once the host is known dead.
    assert client.attempts <= MAX_CONSECUTIVE_TRANSPORT_FAILURES
    assert len(_pending_ids(store)) == 100


def test_forgetting_the_parent_leaves_that_real_memory_alone(tmp_path: Path) -> None:
    """The cascade is marked-only, so the row that won the id is never touched."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-1_closet_alice", "MY OWN NOTE, NOT A COPY"))
    engine.ingest(_memory("sess-1", _turns()))
    store = TombstoneStore(db)

    assert forget(engine, tmp_path, "sess-1", tombstones=store).deleted is True

    assert _all_ids(db) == ["sess-1_closet_alice"]
    assert engine.get("sess-1_closet_alice").content == "MY OWN NOTE, NOT A COPY"
    assert [m.id for m in engine.list_all()] == ["sess-1_closet_alice"]
    # Never recorded as a deleted copy, so pull cannot skip its cloud updates.
    assert store.list_copy_deletions() == []
    assert store.get_copy_deletion("sess-1_closet_alice") is None


def test_a_real_memory_at_a_derived_id_still_syncs(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-1_closet_alice", "MY OWN NOTE, NOT A COPY"))
    engine.ingest(_memory("sess-1", _turns()))

    client = _RecordingClient()
    push(
        engine=engine,
        tombstones=TombstoneStore(db),
        client=client,
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    live = {r["id"] for r in client.upserts if r["deleted_at"] is None}
    assert live == {"sess-1", "sess-1_closet_alice"}


class _FreshnessClient(_RecordingClient):
    """A cloud that resolves writes by freshness, as migration 032 does.

    A write at or above the stored `updated_at` applies; anything older is
    ignored and answered 200, which is what makes an announcement safe to
    consume whether or not it took effect.
    """

    def __init__(self, rows: list[dict] | None = None) -> None:
        super().__init__(rows)
        self.state: dict[str, dict] = {r["id"]: dict(r) for r in (rows or [])}
        self.ignored: list[str] = []

    def upsert(self, row: dict) -> None:
        self.upserts.append(dict(row))
        current = self.state.get(row["id"])
        if current is not None and datetime.fromisoformat(row["updated_at"]) < datetime.fromisoformat(
            current["updated_at"]
        ):
            self.ignored.append(row["id"])
            return  # 200, stale_ignored
        self.state[row["id"]] = dict(row)


def test_a_pending_copy_claim_does_not_modify_another_devices_note(tmp_path: Path) -> None:
    """A pending copy claim cannot overwrite an independent cloud memory."""
    db, engine = _legacy_store_open(tmp_path)
    store = TombstoneStore(db)
    assert "sess-2026-01_closet_alice" in _pending_ids(store)

    # B reclaimed the id later and uploaded an independent note.
    bs_note = _cloud_row(
        "sess-2026-01_closet_alice",
        "B'S INDEPENDENT NOTE",
        when=datetime.now(timezone.utc) + timedelta(days=1),
    )
    cloud = _FreshnessClient(rows=[bs_note])

    push(engine=engine, tombstones=store, client=cloud, state=SyncState(), poppy_dir=tmp_path)

    assert not any(r["id"] == bs_note["id"] for r in cloud.upserts)
    assert cloud.state["sess-2026-01_closet_alice"]["content"] == "B'S INDEPENDENT NOTE"
    assert cloud.state["sess-2026-01_closet_alice"]["deleted_at"] is None


def test_legacy_copy_claims_do_not_write_to_the_cloud(tmp_path: Path) -> None:
    """Pending copy claims have no outgoing wire representation."""
    db = _legacy_store(tmp_path)
    # What a 0.2.4 client pushed: the copy row exactly as it stands locally.
    leaked = {
        cid: _cloud_row(cid, "leaked speaker turns", when=datetime.fromisoformat(_stored_updated_at(db, cid)))
        for cid in ("sess-2026-01_closet_alice", "sess-2026-01_closet_bob")
    }
    engine = _bloom(db)
    store = TombstoneStore(db)
    cloud = _FreshnessClient(rows=list(leaked.values()))

    push(engine=engine, tombstones=store, client=cloud, state=SyncState(), poppy_dir=tmp_path)

    assert not any("_closet_" in row["id"] for row in cloud.upserts)


def _leaked_pair(when: datetime) -> tuple[dict, dict]:
    """What a <=0.2.4 client pushed: the parent AND its per-speaker copies, live."""
    parent = _cloud_row("p", _turns(), when=when)
    parent["created_at"] = when.isoformat()
    copy = _cloud_row("p_closet_alice", json.dumps([{"speaker": "Alice", "dia_id": "D1", "text": SECRET}]), when=when)
    copy["created_at"] = when.isoformat()
    copy["related_to"] = ["p"]
    return parent, copy


def test_a_real_lookalike_still_wins_the_id_against_adoption(tmp_path: Path) -> None:
    """Adoption is gated on the same conjunctive test, so prose is never adopted."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("p_closet_alice", "MY OWN NOTE, NOT A COPY"))

    engine.ingest(_memory("p", _turns()))

    assert engine.get("p_closet_alice").content == "MY OWN NOTE, NOT A COPY"
    assert _marked_ids(db) == []
    assert _pending_ids(TombstoneStore(db)) == []
    assert _backup_rows(db) == {}


def test_an_equal_timestamp_tombstone_deletes_the_local_row(tmp_path: Path) -> None:
    """Device B pulled leaked copies as ordinary rows; device A announces the
    cleanup stamped with the copy's own timestamp, which ties with B's row.

    Treating a tie as "local is newer" left the row live on B, and B's next push
    re-uploaded the secret the cleanup had just removed.
    """
    db = tmp_path / "memories.db"
    engine = SeedEngine(db_path=db)  # B is on the fallback engine
    store = TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=1)
    _, copy = _leaked_pair(when)

    pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[copy]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )
    assert engine.get("p_closet_alice") is not None

    # A's cleanup, carrying the copy's own updated_at.
    cleanup = dict(copy)
    cleanup["content"] = LEGACY_DELETION_BODY
    cleanup["deleted_at"] = when.isoformat()
    result = pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[cleanup]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.skipped_stale == 0
    assert engine.get("p_closet_alice") is None
    assert not engine.retrieve(SECRET, limit=10)

    # And B does not re-upload it.
    cloud = _FreshnessClient(rows=[cleanup])
    push(engine=engine, tombstones=store, client=cloud, state=SyncState(), poppy_dir=tmp_path)
    assert not any(r["deleted_at"] is None and r["id"] == "p_closet_alice" for r in cloud.upserts)
    assert cloud.state["p_closet_alice"]["deleted_at"] is not None


def test_a_strictly_newer_local_row_still_beats_a_tombstone(tmp_path: Path) -> None:
    """Live-row precedence is unchanged: only a TIE now goes to the deletion."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    store = TombstoneStore(db)
    engine.ingest(_memory("mem_x", "the local edit, written later"))

    stale = _cloud_row("mem_x", "an older body", deleted=True, when=datetime.now(timezone.utc) - timedelta(days=1))
    result = pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[stale]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.skipped_stale == 1
    assert engine.get("mem_x").content == "the local edit, written later"


def test_tier_proven_marks_without_backup_or_upload(tmp_path: Path) -> None:
    """Text IS what the parent derives: not an inference, so it may be announced."""
    db = _legacy_store(tmp_path)
    _migrate_only(db)

    assert _marked_ids(db) == ["sess-2026-01_closet_alice", "sess-2026-01_closet_bob"]
    assert _backup_rows(db) == {}  # nothing was rewritten, so nothing to snapshot
    assert _pending_ids(TombstoneStore(db)) == []
    assert [m.id for m in _bloom(db).list_all()] == ["sess-2026-01"]


def test_tier_likely_marks_but_never_rewrites_or_announces(tmp_path: Path) -> None:
    """L2's repro: a curated split under a derived id keeps its text.

    An importer can legitimately write one, and from here it is indistinguishable
    from a copy left stale by the pre-fix edit bug. Hidden, so the redaction
    promise holds; kept, so nothing is destroyed if the guess is wrong.
    """
    db = tmp_path / "memories.db"
    _write_legacy(_bloom(db), _memory("p", _turns()))
    _strip_marker(db)
    conn = sqlite3.connect(str(db))
    conn.execute(
        "UPDATE memories SET content = ?, enriched_content = ? WHERE id = ?",
        (json.dumps([{"speaker": "Alice", "text": "ONLY COPY OF SIGNED APPROVAL"}]), "x", "p_closet_alice"),
    )
    conn.commit()
    conn.close()

    engine = _bloom(db)

    kept = engine.get("p_closet_alice")
    assert kept.content == json.dumps([{"speaker": "Alice", "text": "ONLY COPY OF SIGNED APPROVAL"}])
    assert "p_closet_alice" in _marked_ids(db)
    assert _backup_rows(db) == {"p_closet_alice": "adopted"}
    assert "p_closet_alice" not in _pending_ids(TombstoneStore(db))


def test_tier_orphan_is_left_entirely_alone(tmp_path: Path) -> None:
    """No parent to check against, so nothing is confirmed and nothing is done."""
    db = _legacy_store(tmp_path)
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM memories WHERE id = ?", ("sess-2026-01",))
    conn.commit()
    conn.close()

    engine = _bloom(db)

    assert sorted(_all_ids(db)) == ["sess-2026-01_closet_alice", "sess-2026-01_closet_bob"]
    assert _marked_ids(db) == []
    assert _backup_rows(db) == {}
    assert _pending_ids(TombstoneStore(db)) == []
    assert len({m.id for m in engine.list_all()}) == 2


def test_a_seed_only_store_is_never_graded(tmp_path: Path) -> None:
    """A store that never ran the default engine cannot hold a copy.

    L1's repro: `dialog42_closet_alice` with no parent, written by seed alone.
    Opening it with the default engine adds that engine's schema, so by the time
    the migration runs the store LOOKS like one it wrote — the answer has to be
    captured before.
    """
    db = tmp_path / "memories.db"
    seed = SeedEngine(db_path=db)
    seed.ingest(
        _memory("dialog42_closet_alice", json.dumps([{"speaker": "Alice", "text": SECRET}]), related_to=["dialog42"])
    )
    assert MARKER_COLUMN in {r[1] for r in _rows(db, "PRAGMA table_info(memories)")}

    engine = _bloom(db)  # first ever bloom open: adds its schema, then migrates

    assert _all_ids(db) == ["dialog42_closet_alice"]
    assert _marked_ids(db) == []
    assert _backup_rows(db) == {}
    assert _pending_ids(TombstoneStore(db)) == []
    assert [m.id for m in engine.list_all()] == ["dialog42_closet_alice"]


def test_an_independent_record_holding_the_same_turns_is_not_a_copy(tmp_path: Path) -> None:
    """Matching text strengthens the evidence; it never replaces provenance.

    An independent record can hold exactly the turns a memory projects while
    pointing at its own source and
    predating that memory. Graded on text alone it was called PROVEN and its
    cloud row announced for deletion.
    """
    db = tmp_path / "memories.db"
    _write_legacy(_bloom(db), _memory("p", _turns()))
    projected = _rows(db, "SELECT content FROM memories WHERE id = 'p_closet_alice'")[0][0]
    _strip_marker(db)
    conn = sqlite3.connect(str(db))
    conn.execute(
        "UPDATE memories SET related_to = ?, created_at = ?, project = 'elsewhere' WHERE id = ?",
        (json.dumps(["independent-record"]), "2026-01-01T00:00:00+00:00", "p_closet_alice"),
    )
    conn.commit()
    conn.close()

    _migrate_only(db)
    assert _marked_ids(db) == ["p_closet_bob"]
    assert _pending_ids(TombstoneStore(db)) == []

    engine = _bloom(db)

    assert engine.get("p_closet_alice").content == projected  # untouched
    assert "p_closet_alice" in {m.id for m in engine.list_all()}


def test_a_pulled_tombstone_does_not_overwrite_a_note_written_meanwhile(tmp_path: Path, synced_state) -> None:
    """The ordinary tombstone path, same wrong-clock class as the copy lanes.

    Device B holds a legacy copy as an ordinary row. A's deletion of it is
    stamped T1. While B's pull is in flight, C writes an independent note at that
    id at T2. Recorded at receipt time (T3), B's Trash entry looks newer than
    C's note and B's next push deletes it for everyone, leaving only the
    placeholder.
    """
    db = tmp_path / "memories.db"
    engine = SeedEngine(db_path=db)  # B is on the fallback engine
    store = TombstoneStore(db)
    t0 = datetime.now(timezone.utc) - timedelta(hours=3)
    t1 = t0 + timedelta(hours=1)
    t2 = t0 + timedelta(hours=2)

    # B pulled the leaked copy as an ordinary memory.
    leaked = _cloud_row("p_closet_alice", json.dumps([{"speaker": "Alice", "text": SECRET}]), when=t0)
    pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[leaked]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )
    assert engine.get("p_closet_alice") is not None

    # A's deletion of it, stamped T1.
    deletion = _cloud_row("p_closet_alice", "the leaked body", deleted=True, when=t1)
    pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[deletion]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert engine.get("p_closet_alice") is None
    recorded = store.get("p_closet_alice")
    assert recorded is not None
    assert recorded.tombstoned_at == t1  # the deletion's clock, not ours

    # C's independent note at T2, written while B's pull was in flight.
    theirs = _cloud_row("p_closet_alice", "C'S INDEPENDENT NOTE", when=t2)
    cloud = _FreshnessClient(rows=[theirs])
    push(engine=engine, tombstones=store, client=cloud, state=synced_state(), poppy_dir=tmp_path)

    # A learned deletion is not re-announced, even with a fresh watermark.
    assert _RecordingClient.base_url in recorded.sent_remotes
    sent = [r for r in cloud.upserts if r["id"] == "p_closet_alice"]
    assert sent == []
    assert cloud.ignored == []
    assert cloud.state["p_closet_alice"]["content"] == "C'S INDEPENDENT NOTE"
    assert cloud.state["p_closet_alice"]["deleted_at"] is None


def test_a_locally_made_deletion_still_gets_the_current_time(tmp_path: Path) -> None:
    """Only PULLED deletions carry someone else's clock."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("mem_x", "a memory"))
    store = TombstoneStore(db)

    before = datetime.now(timezone.utc)
    forget(engine, tmp_path, "mem_x", tombstones=store)

    recorded = store.get("mem_x")
    assert recorded is not None
    assert recorded.tombstoned_at >= before


def test_a_pulled_tombstone_does_not_downgrade_a_newer_local_one(tmp_path: Path, synced_state) -> None:
    """Forget here at 14:00, then pull an older cloud deletion dated 12:00.

    Replacing the record with 12:00 meant push sent the downgraded deletion, it
    lost to another device's 13:00 recreation, and the next pull brought the
    forgotten content back.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    store = TombstoneStore(db)
    noon = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(hours=6)
    one = noon + timedelta(hours=1)
    two = noon + timedelta(hours=2)

    engine.ingest(_memory("mem_x", SECRET))
    forget(engine, tmp_path, "mem_x", tombstones=store)
    # Date the local deletion 14:00 (two hours after the cloud's).
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE ui_tombstones SET tombstoned_at = ? WHERE id = 'mem_x'", (two.isoformat(),))
    conn.commit()
    conn.close()

    older = _cloud_row("mem_x", SECRET, deleted=True, when=noon)
    pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[older]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert store.get("mem_x").tombstoned_at == two  # not lowered to noon

    # Another device recreated the memory at 13:00, between the two deletions.
    recreated = _cloud_row("mem_x", "A RECREATION", when=one)
    cloud = _FreshnessClient(rows=[recreated])
    push(engine=engine, tombstones=store, client=cloud, state=synced_state(), poppy_dir=tmp_path)

    sent = [r for r in cloud.upserts if r["id"] == "mem_x"]
    assert sent and all(r["updated_at"] == two.isoformat() for r in sent)
    assert cloud.ignored == []  # 14:00 beats the 13:00 recreation
    assert cloud.state["mem_x"]["deleted_at"] is not None
    assert engine.get("mem_x") is None


def test_a_local_copy_deletion_is_a_floor_for_pulled_ones(tmp_path: Path) -> None:
    """Same rule on the copy records, where "earliest wins" pulls the other way.

    That rule stops a re-pull refreshing a record forward. It must not let a
    stranger's older deletion overwrite one this device made.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    store = TombstoneStore(db)
    noon = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(hours=6)
    two = noon + timedelta(hours=2)

    engine.ingest(_memory("sess-1", _turns()))
    forget(engine, tmp_path, "sess-1", tombstones=store)
    store.add_copy_deletions(["sess-1_closet_alice", "sess-1_closet_bob"])
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE closet_tombstones SET tombstoned_at = ?", (two.isoformat(),))
    conn.commit()
    conn.close()

    # An older sighting of the same deletion must not lower it...
    store.add_copy_deletions(["sess-1_closet_alice"], now=noon)
    assert store.get_copy_deletion("sess-1_closet_alice").tombstoned_at == two

    # ... while between two REMOTE sightings the earliest still wins.
    store.add_copy_deletions(["remote_closet_bob"], now=two)
    store.add_copy_deletions(["remote_closet_bob"], now=noon)
    assert store.get_copy_deletion("remote_closet_bob").tombstoned_at == noon


def _pull_leaked_pair(tmp_path: Path, engine, store, *, when: datetime) -> None:
    """What a fresh store receives from an account a <=0.2.4 client synced."""
    parent = _cloud_row("p", _turns(), when=when)
    parent["created_at"] = when.isoformat()
    copy = _cloud_row("p_closet_alice", json.dumps([{"speaker": "Alice", "dia_id": "D1", "text": SECRET}]), when=when)
    copy["created_at"] = when.isoformat()
    copy["related_to"] = ["p"]
    for row in (parent, copy):
        pull(
            engine=engine,
            tombstones=store,
            client=_RecordingClient(rows=[row]),
            state=SyncState(),
            poppy_dir=tmp_path,
        )


def test_an_older_pulled_deletion_never_replaces_a_newer_trash_snapshot(tmp_path: Path) -> None:
    """The copy's id was reclaimed by a real note, then that note was deleted.

    The cloud's deletion of the OLD copy arrives afterwards, older than the
    note's deletion. Trash must keep showing the note, not the copy's speaker
    text, or restoring from Trash brings the secret back as an ordinary memory.
    """
    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    t0 = datetime.now(timezone.utc) - timedelta(days=1)
    parent, copy = _leaked_pair(t0)
    pull(engine=engine, tombstones=store, client=_RecordingClient([parent]), state=SyncState(), poppy_dir=tmp_path)
    assert forget(engine, tmp_path, "p", tombstones=store).deleted is True
    engine.ingest(_memory("p_closet_alice", "INDEPENDENT NOTE"))
    assert forget(engine, tmp_path, "p_closet_alice", tombstones=store).deleted is True
    note_deletion = store.get("p_closet_alice")
    assert note_deletion is not None

    copy["deleted_at"] = copy["updated_at"] = (t0 + timedelta(hours=1)).isoformat()
    pull(engine=engine, tombstones=store, client=_RecordingClient([copy]), state=SyncState(), poppy_dir=tmp_path)

    kept = store.get("p_closet_alice")
    assert kept is not None
    assert kept.memory.content == "INDEPENDENT NOTE"
    assert kept.tombstoned_at == note_deletion.tombstoned_at
    assert kept.token == note_deletion.token

    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert not any(SECRET in json.dumps(r) for r in client.upserts if r["id"] != "p")


@pytest.mark.parametrize("engine_name", ["seed", "bloom"])
def test_a_metadata_edit_cannot_undo_a_forget_that_landed_first(
    tmp_path: Path, engine_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Edit reads the row, another process forgets it, edit writes the row back.

    Ungated, the write-back resurrects the memory and bloom re-derives its
    per-speaker copies from it. Under the gate the edit sees the forget.
    """
    import poppy.db as db_mod
    from poppy.lifecycle import edit_memory

    db = tmp_path / "memories.db"
    writer = _bloom(db)
    writer.ingest(_memory("p", _turns()))
    engine = SeedEngine(db_path=db) if engine_name == "seed" else _bloom(db)
    store = TombstoneStore(db)
    real_gate = db_mod.write_gate
    raced: list[bool] = []

    @contextmanager
    def gate_with_a_forget_ahead_of_us(poppy_dir: Path):
        if not raced:
            raced.append(True)
            assert forget(writer, tmp_path, "p", tombstones=store).deleted is True
        with real_gate(poppy_dir):
            yield

    monkeypatch.setattr(db_mod, "write_gate", gate_with_a_forget_ahead_of_us)
    with pytest.raises(KeyError):
        edit_memory(engine, "p", project="moved", poppy_dir=tmp_path)

    assert raced
    assert engine.get("p") is None
    assert engine.get("p_closet_alice") is None
    assert not engine.retrieve(SECRET, limit=10)


def test_a_supersede_cannot_undo_a_forget_that_landed_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same seam as the edit: the whole supersede runs under one gate."""
    import poppy.db as db_mod
    from poppy.lifecycle import supersede_memory

    db = tmp_path / "memories.db"
    engine, other = _bloom(db), _bloom(db)
    engine.ingest(_memory("p", _turns()))
    store = TombstoneStore(db)
    real_gate = db_mod.write_gate
    raced: list[bool] = []

    @contextmanager
    def gate_with_a_forget_ahead_of_us(poppy_dir: Path):
        if not raced:
            raced.append(True)
            assert forget(other, tmp_path, "p", tombstones=store).deleted is True
        with real_gate(poppy_dir):
            yield

    monkeypatch.setattr(db_mod, "write_gate", gate_with_a_forget_ahead_of_us)
    with pytest.raises(KeyError):
        supersede_memory(engine, _memory("replacement", "safe replacement"), "p", poppy_dir=tmp_path)

    assert raced
    assert engine.get("replacement") is None
    assert engine.get("p_closet_alice") is None
    assert not engine.retrieve(SECRET, limit=10)


@pytest.mark.parametrize("engine_name", ["seed", "bloom"])
def test_a_forget_leaves_an_unmarked_copy_shaped_memory_whose_text_is_not_derived(
    tmp_path: Path, engine_name: str
) -> None:
    """LIKELY evidence at redaction time is not enough to delete an unmarked row.

    The user wrote ``session_closet_alice`` themselves: same link, same creation
    time, single-speaker turn list, but text the parent never derives. On a
    store that never ran bloom nothing adopted it, so it reaches the parent's
    forget unmarked. It must survive, listed and recallable.
    """
    db = tmp_path / "memories.db"
    engine = SeedEngine(db_path=db) if engine_name == "seed" else _bloom(db)
    store = TombstoneStore(db)
    parent = _memory("session", _turns())
    engine.ingest(parent)
    own = _memory(
        "session_closet_alice",
        json.dumps([{"speaker": "Alice", "text": "ONLY COPY OF CONTRACT"}]),
        related_to=["session"],
    )
    own.created_at = parent.created_at
    engine.ingest(own)
    assert engine.get("session_closet_alice").content == own.content

    assert forget(engine, tmp_path, "session", tombstones=store).deleted is True

    kept = engine.get("session_closet_alice")
    assert kept is not None and kept.content == own.content
    assert "session_closet_alice" in {m.id for m in engine.list_all()}
    assert engine.retrieve("CONTRACT", limit=10)
    assert "session_closet_alice" not in _backup_rows(db)
    assert "session_closet_alice" not in _pending_ids(store)


def test_created_at_is_compared_as_an_instant_not_a_string() -> None:
    """``+02:00`` and its UTC spelling are the same moment; a text compare said otherwise."""
    from poppy.engine._legacy_copies import has_copy_provenance

    content = json.dumps([{"speaker": "Alice", "dia_id": "D1", "text": SECRET}])
    common = dict(content=content, related_raw=json.dumps(["p"]))
    assert has_copy_provenance(
        "p", "alice", created_at="2026-07-01T12:00:00+02:00", parent_created_at="2026-07-01T10:00:00+00:00", **common
    )
    assert not has_copy_provenance(
        "p", "alice", created_at="2026-07-01T12:00:00+02:00", parent_created_at="2026-07-01T12:00:00+00:00", **common
    )


def test_a_pulled_copy_deletion_is_dated_from_its_deleted_at(tmp_path: Path) -> None:
    """A cleanup row whose updated_at was bumped after its deleted_at must not bury a recreation in between."""
    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    t0 = datetime.now(timezone.utc) - timedelta(days=2)
    cleanup = _legacy_deletion_row("p_closet_alice", t0)
    cleanup["updated_at"] = (t0 + timedelta(hours=2)).isoformat()
    pull(engine=engine, tombstones=store, client=_RecordingClient([cleanup]), state=SyncState(), poppy_dir=tmp_path)
    assert _local_deletion_time(db, "p_closet_alice") == t0

    recreation = _cloud_row("p_closet_alice", "independent note after deletion", when=t0 + timedelta(hours=1))
    pull(engine=engine, tombstones=store, client=_RecordingClient([recreation]), state=SyncState(), poppy_dir=tmp_path)
    assert engine.get("p_closet_alice").content == recreation["content"]
    assert store.get_copy_deletion("p_closet_alice") is None


def test_purge_boundaries_compare_as_instants(tmp_path: Path) -> None:
    """A deletion at 06:00Z spelled 01:00-05:00, and a watermark 12:00+02:00 (10:00Z)."""
    db = tmp_path / "memories.db"
    _bloom(db)
    store = TombstoneStore(db)
    store.add(
        _memory("ordinary", "independent retained snapshot"),
        tombstoned_at=datetime.fromisoformat("2026-07-01T01:00:00-05:00"),
    )
    store.purge_expired(pushed_through="2026-07-01T05:00:00+00:00")
    assert store.get("ordinary") is not None  # 06:00Z is after 05:00Z

    store.add_copy_deletions(["p_closet_alice"], now=datetime(2026, 7, 1, 11, tzinfo=timezone.utc))
    store.purge_expired(pushed_through="2026-07-01T12:00:00+02:00")
    assert store.get_copy_deletion("p_closet_alice") is not None  # 11:00Z is after 10:00Z


def test_a_local_forget_always_snapshots_even_past_a_skewed_record(tmp_path: Path) -> None:
    """A pulled deletion dated in the future must not stop this device's own forget from snapshotting."""
    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    engine.ingest(_memory("note", "MY NOTE"))
    skewed = _memory("note", "OLDER TEXT")
    store.add(skewed, tombstoned_at=datetime.now(timezone.utc) + timedelta(hours=1))

    assert forget(engine, tmp_path, "note", tombstones=store).deleted is True
    assert store.get("note").memory.content == "MY NOTE"


def test_a_trash_entry_that_is_not_the_parents_text_survives_the_parents_forget(tmp_path: Path) -> None:
    """A real note that once lived at the copy's id keeps its Trash entry."""
    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    engine.ingest(_memory("p", _turns()))
    store.add(_memory("p_closet_alice", "MY OWN NOTE AT THAT ID"))

    assert forget(engine, tmp_path, "p", tombstones=store).deleted is True
    assert store.get("p_closet_alice").memory.content == "MY OWN NOTE AT THAT ID"
    assert "p_closet_alice" not in dict(store.pending_legacy_announcements())


def test_a_trash_entry_with_the_same_text_but_other_provenance_survives(tmp_path: Path) -> None:
    """Matching text alone never condemns a Trash entry: an independent memory can hold the same turns."""
    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    _write_legacy(engine, _memory("p", _turns()))
    same_text = engine.get("p_closet_alice").content
    independent = _memory("p_closet_alice", same_text, related_to=["independent-record"], project="elsewhere")
    independent.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    store.add(independent)

    assert forget(engine, tmp_path, "p", tombstones=store).deleted is True
    kept = store.get("p_closet_alice")
    assert kept is not None and kept.memory.related_to == ["independent-record"]
    assert "p_closet_alice" not in dict(store.pending_legacy_announcements())


@pytest.mark.parametrize("engine_name", ["seed", "bloom"])
def test_forgetting_a_copy_directly_also_clears_its_legacy_trash_entry(tmp_path: Path, engine_name: str) -> None:
    """An older client trashed the copy, the parent was re-ingested and re-derived it; now forget the copy again.

    The content-free path must also take the old snapshot, claim the
    announcement with its deletion time, and leave nothing to restore.
    """
    from poppy.write_flow import restore

    db = tmp_path / "memories.db"
    writer = _bloom(db)
    engine = SeedEngine(db_path=db) if engine_name == "seed" else _bloom(db)
    _write_legacy(writer, _memory("p", _turns()))
    store = TombstoneStore(db)
    legacy = store.add(engine.get("p_closet_alice"))

    result = forget(engine, tmp_path, "p_closet_alice", tombstones=store)
    assert result.deleted is True

    assert engine.get("p_closet_alice") is None
    assert store.get("p_closet_alice") is None
    assert restore(engine, tmp_path, "p_closet_alice", tombstones=store).found is False
    assert store.announced_copy_claim("p_closet_alice") == legacy.tombstoned_at

    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert not any(SECRET in json.dumps(r) for r in client.upserts if r["id"] != "p")


def test_forgetting_a_proven_unmarked_copy_on_a_seed_store_is_content_free(tmp_path: Path) -> None:
    """A store that never ran bloom holds a pulled leaked copy unmarked; the user forgets it by id.

    It must go the content-free way: no Trash snapshot of the speaker text, no
    text on the wire, and no outgoing deletion marker.
    """
    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=1)
    _pull_leaked_pair(tmp_path, seed, store, when=when)
    assert "p_closet_alice" not in _marked_ids(db)

    result = forget(seed, tmp_path, "p_closet_alice", tombstones=store)
    assert result.deleted is True
    assert result.tombstone is None

    assert seed.get("p_closet_alice") is None
    assert store.get("p_closet_alice") is None
    assert store.get_copy_deletion("p_closet_alice") is not None
    assert store.announced_copy_claim("p_closet_alice") == when

    client = _RecordingClient()
    push(engine=seed, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    sent = [r for r in client.upserts if r["id"] == "p_closet_alice"]
    assert sent == []
    assert not any(SECRET in json.dumps(r) for r in client.upserts if r["id"] != "p")


def test_forgetting_a_real_lookalike_still_snapshots_it(tmp_path: Path) -> None:
    """Only full proof takes the content-free path; a real note at a copy-shaped id keeps its Trash entry."""
    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    seed.ingest(_memory("p", _turns()))
    seed.ingest(_memory("p_closet_alice", "MY OWN NOTE", related_to=["p"]))

    result = forget(seed, tmp_path, "p_closet_alice", tombstones=store)
    assert result.deleted is True and result.tombstone is not None
    assert store.get("p_closet_alice").memory.content == "MY OWN NOTE"
    assert "p_closet_alice" not in dict(store.pending_legacy_announcements())


def test_forgetting_a_proven_unmarked_copy_also_clears_its_legacy_trash_entry(tmp_path: Path) -> None:
    """Seed store, pulled leaked pair, a legacy snapshot of the copy in Trash; forget the copy by id."""
    from poppy.write_flow import restore

    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=1)
    _pull_leaked_pair(tmp_path, seed, store, when=when)
    legacy = store.add(seed.get("p_closet_alice"))

    assert forget(seed, tmp_path, "p_closet_alice", tombstones=store).deleted is True

    assert store.get("p_closet_alice") is None
    assert restore(seed, tmp_path, "p_closet_alice", tombstones=store).found is False
    assert store.announced_copy_claim("p_closet_alice") == legacy.tombstoned_at
    client = _RecordingClient()
    push(engine=seed, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert not any(SECRET in json.dumps(r) for r in client.upserts if r["id"] != "p")


def test_a_same_text_write_survives_an_unparseable_stored_creation_time(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("note", "some text"))
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE memories SET created_at = 'not a timestamp' WHERE id = 'note'")
    conn.commit()
    conn.close()
    engine.ingest(_memory("note", "some text", project="moved"))
    assert engine.get("note").project == "moved"


def test_a_cleanup_deletion_landing_on_a_live_unmarked_copy_leaves_no_trash_entry(tmp_path: Path) -> None:
    """Seed store holds the pulled leaked copy live; the server's content-free cleanup arrives at the same stamp.

    The copy goes, and nothing restorable is left: restoring a placeholder body
    would write it back as a real memory and push it over the cloud's cleanup.
    """
    from poppy.write_flow import restore

    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=1)
    _pull_leaked_pair(tmp_path, seed, store, when=when)
    assert seed.get("p_closet_alice") is not None

    cleanup = _legacy_deletion_row("p_closet_alice", when)
    result = pull(
        engine=seed, tombstones=store, client=_RecordingClient([cleanup]), state=SyncState(), poppy_dir=tmp_path
    )

    assert result.skipped_echoes == 1
    assert seed.get("p_closet_alice") is None
    assert store.get("p_closet_alice") is None
    assert _local_deletion_time(db, "p_closet_alice") is not None
    assert restore(seed, tmp_path, "p_closet_alice", tombstones=store).found is False
    client = _RecordingClient()
    push(engine=seed, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert not any(r["id"] == "p_closet_alice" and r["deleted_at"] is None for r in client.upserts)


@pytest.mark.parametrize("verb", ["supersede", "edit"])
def test_a_proven_unmarked_copy_is_refused_to_supersede_and_edit(tmp_path: Path, verb: str) -> None:
    """Seed store, pulled leaked pair: superseding or editing the copy by id is refused like a marked one."""
    from poppy.lifecycle import edit_memory, supersede_memory

    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    _pull_leaked_pair(tmp_path, seed, store, when=datetime.now(timezone.utc) - timedelta(days=1))
    assert "p_closet_alice" not in _marked_ids(db)

    with pytest.raises(ValueError, match="derived per-speaker copy"):
        if verb == "supersede":
            supersede_memory(seed, _memory("replacement", "safe"), "p_closet_alice", poppy_dir=tmp_path)
        else:
            edit_memory(seed, "p_closet_alice", content="edited", poppy_dir=tmp_path)

    assert store.get("p_closet_alice") is None
    assert seed.get("p_closet_alice") is not None
    client = _RecordingClient()
    push(engine=seed, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert not any(r["id"] == "p_closet_alice" and r["deleted_at"] is not None for r in client.upserts)


@pytest.mark.parametrize("engine_name", ["seed", "bloom"])
def test_a_real_memory_at_a_derived_id_survives_the_parents_forget(tmp_path: Path, engine_name: str) -> None:
    """Grading is the same one used everywhere, so prose is never swept."""
    db = tmp_path / "memories.db"
    writer = _bloom(db)
    writer.ingest(_memory("p", _turns()))
    writer.ingest(_memory("p_closet_alice", "MY OWN NOTE, NOT A COPY", related_to=["somewhere-else"]))

    engine = SeedEngine(db_path=db) if engine_name == "seed" else _bloom(db)
    store = TombstoneStore(db)
    assert forget(engine, tmp_path, "p", tombstones=store).deleted is True

    assert engine.get("p_closet_alice").content == "MY OWN NOTE, NOT A COPY"
    assert "p_closet_alice" in {m.id for m in engine.list_all()}
    assert "p_closet_alice" not in _pending_ids(store)


def _trash_entry(db: Path, memory_id: str) -> tuple | None:
    rows = _rows(db, "SELECT id, content FROM ui_tombstones WHERE id = ?", (memory_id,))
    return rows[0] if rows else None


def test_restore_legacy_snapshot_after_parent_forget(tmp_path: Path) -> None:
    """A content-carrying cloud tombstone for a copy id must not become restorable.

    Only an old (0.2.4) client deleting a copy row by hand writes one. Pull had no
    local row to compare it with, so it filed the speaker text in Trash; once the
    parent was forgotten nothing could grade it any more, and restore brought the
    text back as an ordinary memory that push then sent live.
    """
    from poppy.write_flow import restore

    db = tmp_path / "memories.db"
    # A seed store: it never synthesises, so nothing marks the id for pull to skip.
    engine, store = SeedEngine(db_path=db), TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=1)
    parent, copy = _leaked_pair(when)
    pull(engine=engine, tombstones=store, client=_RecordingClient([parent]), state=SyncState(), poppy_dir=tmp_path)

    # The old client's hand deletion of the copy row: a tombstone WITH its text.
    copy["deleted_at"] = copy["updated_at"] = (when + timedelta(hours=1)).isoformat()
    result = pull(
        engine=engine, tombstones=store, client=_RecordingClient([copy]), state=SyncState(), poppy_dir=tmp_path
    )

    assert result.applied_tombstones == 1
    assert _trash_entry(db, "p_closet_alice") is None
    assert store.get_copy_deletion("p_closet_alice") is not None

    assert forget(engine, tmp_path, "p", tombstones=store).deleted is True
    assert restore(engine, tmp_path, "p_closet_alice", tombstones=store).found is False
    assert not engine.retrieve(SECRET, limit=10)


def test_legacy_trash_snapshot_is_sent_after_parent_forget(tmp_path: Path) -> None:
    """The same entry, on the wire: push must never carry the speaker text up.

    Filed as an ordinary Trash entry, the copy's text went back up as the BODY of
    a soft-delete on the very next push — before any redaction of the parent had a
    chance to grade it — and any device pulling that row filed it as restorable in
    turn.
    """
    db = tmp_path / "memories.db"
    engine, store = SeedEngine(db_path=db), TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=1)
    parent, copy = _leaked_pair(when)
    pull(engine=engine, tombstones=store, client=_RecordingClient([parent]), state=SyncState(), poppy_dir=tmp_path)
    copy["deleted_at"] = copy["updated_at"] = (when + timedelta(hours=1)).isoformat()
    pull(engine=engine, tombstones=store, client=_RecordingClient([copy]), state=SyncState(), poppy_dir=tmp_path)

    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert not any(SECRET in json.dumps(r) for r in client.upserts if r["id"] != "p")

    assert forget(engine, tmp_path, "p", tombstones=store).deleted is True

    after = _RecordingClient()
    push(engine=engine, tombstones=store, client=after, state=SyncState(), poppy_dir=tmp_path)
    # The parent's own Trash entry carries the parent's text, as it always did.
    assert not any(SECRET in json.dumps(r) for r in after.upserts if r["id"] != "p")


def test_a_real_lookalike_deletion_is_still_filed_as_a_restorable_entry(tmp_path: Path) -> None:
    """Grading cuts both ways: a real memory at a copy-shaped id keeps its Trash entry."""
    db = tmp_path / "memories.db"
    engine, store = SeedEngine(db_path=db), TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=1)
    parent, _copy = _leaked_pair(when)
    pull(engine=engine, tombstones=store, client=_RecordingClient([parent]), state=SyncState(), poppy_dir=tmp_path)

    theirs = _cloud_row("p_closet_alice", "SOMEONE ELSE'S REAL NOTE", deleted=True, when=when + timedelta(hours=1))
    theirs["related_to"] = ["p"]
    pull(engine=engine, tombstones=store, client=_RecordingClient([theirs]), state=SyncState(), poppy_dir=tmp_path)

    kept = store.get("p_closet_alice")
    assert kept is not None and kept.memory.content == "SOMEONE ELSE'S REAL NOTE"


def test_restore_refuses_a_legacy_trash_copy_whose_parent_is_still_live(tmp_path: Path) -> None:
    """Item 10: the guard read the LIVE row's marker, and there is no live row here.

    On 0.2.4 the user forgot the copy and kept the parent. After upgrading,
    restoring that entry created an unmarked memory holding the speaker text,
    which the next push uploaded live with no cleanup queued.
    """
    from poppy.write_flow import restore

    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    _write_legacy(engine, _memory("p", _turns()))
    legacy = store.add(engine.get("p_closet_alice"))
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM memories WHERE id = 'p_closet_alice'")
    conn.execute("DELETE FROM memory_embeddings WHERE id = 'p_closet_alice'")
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="derived per-speaker copy"):
        restore(engine, tmp_path, "p_closet_alice", tombstones=store)

    assert engine.get("p_closet_alice") is None
    assert store.get("p_closet_alice") is None  # proven, so the entry goes too
    assert store.announced_copy_claim("p_closet_alice") == legacy.tombstoned_at
    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert not any(r["id"] == "p_closet_alice" and r["deleted_at"] is None for r in client.upserts)
    assert not any(SECRET in json.dumps(r) for r in client.upserts if r["id"] != "p")


def test_restore_still_returns_a_real_lookalike_from_trash(tmp_path: Path) -> None:
    """The refusal is bounded by provenance: an ordinary note at a copy id still restores."""
    from poppy.write_flow import restore

    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    engine.ingest(_memory("p", _turns()))
    engine.ingest(_memory("p_closet_notes", "MY OWN NOTE"))
    assert forget(engine, tmp_path, "p_closet_notes", tombstones=store).deleted is True

    result = restore(engine, tmp_path, "p_closet_notes", tombstones=store)
    assert result.memory is not None and result.memory.content == "MY OWN NOTE"


def test_push_still_sends_a_row_that_is_only_a_lookalike(tmp_path: Path) -> None:
    """The re-read asks the same question list_all asked: a real memory at a copy id still syncs."""
    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    engine.ingest(_memory("p", _turns()))
    engine.ingest(_memory("p_closet_notes", "MY OWN NOTE"))

    client = _RecordingClient()
    result = push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)

    assert result.sent_live == 2
    assert {r["id"] for r in client.upserts} == {"p", "p_closet_notes"}


def test_newer_remote_closet_deletion_is_not_retained(tmp_path: Path) -> None:
    """A server cleanup at t3 left the local record still saying t1.

    "Earliest wins" is about not refreshing a record with a later SIGHTING of the
    same deletion. A cleanup tombstone is a different event with its own clock, and
    a record stuck at t1 let the cloud's t2 copy of that id through once the pull
    watermark had been reset.
    """
    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    t1 = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    t2 = t1 + timedelta(days=1)
    t3 = t1 + timedelta(days=2)
    store.add_copy_deletions(["sess-1_closet_alice"], now=t1)

    cleanup = _legacy_deletion_row("sess-1_closet_alice", t3)
    pull(engine=engine, tombstones=store, client=_RecordingClient([cleanup]), state=SyncState(), poppy_dir=tmp_path)

    assert _local_deletion_time(db, "sess-1_closet_alice") == t3

    # ... and the record now suppresses the t2 copy a watermark reset re-offers.
    stale = _cloud_row("sess-1_closet_alice", json.dumps([{"speaker": "Alice", "text": SECRET}]), when=t2)
    result = pull(
        engine=engine, tombstones=store, client=_RecordingClient([stale]), state=SyncState(), poppy_dir=tmp_path
    )

    assert result.skipped_stale == 1
    assert engine.get("sess-1_closet_alice") is None


def test_a_cleanup_deletion_never_lowers_a_record(tmp_path: Path) -> None:
    """Raising only: an older cleanup must not pull a newer record down."""
    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    early = datetime(2026, 7, 1, tzinfo=timezone.utc)
    late = datetime(2026, 8, 1, tzinfo=timezone.utc)
    store.add_copy_deletions(["sess-1_closet_alice"], now=late)

    cleanup = _legacy_deletion_row("sess-1_closet_alice", early)
    pull(engine=engine, tombstones=store, client=_RecordingClient([cleanup]), state=SyncState(), poppy_dir=tmp_path)

    assert store.get_copy_deletion("sess-1_closet_alice").tombstoned_at == late


def test_audit_sync_purge_during_wait_does_not_revive_captured_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pull captured an old copy row; a concurrent push purged the deletion record while it waited.

    The record is what makes pull skip the cloud's stale copy, and it ages out once
    the announcement has landed — while the cloud row lives until the server applies
    that delete. With the record gone the waiting pull ingested the captured copy as
    an ordinary memory. The announcement queue outlives the record and says the same
    thing: this id was PROVEN to be a copy.
    """
    import poppy.sync as sync_mod
    from poppy.db import write_gate as real_gate

    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=30)
    _pull_leaked_pair(tmp_path, seed, store, when=when)
    # The copy is forgotten by id and the PARENT stays: that is what leaves the
    # evidence in place, and the claim is only an existence hint.
    assert forget(seed, tmp_path, "p_closet_alice", tombstones=store).deleted is True
    assert store.announced_copy_claim("p_closet_alice") is not None
    assert "p_closet_alice" in _pending_ids(store)
    # The announcement lands, so the record is free to age out.
    store.mark_legacy_announced(store.pending_legacy_announcements())
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE closet_tombstones SET tombstoned_at = ?", (when.isoformat(),))
    conn.commit()
    conn.close()

    captured = _cloud_row(
        "p_closet_alice", json.dumps([{"speaker": "Alice", "dia_id": "D1", "text": SECRET}]), when=when
    )
    captured["created_at"] = when.isoformat()
    captured["related_to"] = ["p"]

    purged: list[int] = []

    @contextmanager
    def gate_with_a_purge_ahead_of_us(poppy_dir: Path):
        # The cleanup push gets the lock first and ages the record out.
        if not purged:
            purged.append(store.purge_expired(pushed_through=datetime.now(timezone.utc).isoformat()))
            assert store.get_copy_deletion("p_closet_alice") is None
        with real_gate(poppy_dir):
            yield

    monkeypatch.setattr(sync_mod, "write_gate", gate_with_a_purge_ahead_of_us)
    result = pull(
        engine=seed,
        tombstones=store,
        client=_RecordingClient([captured]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert purged
    assert result.applied_live == 0
    assert result.skipped_copies == 1
    assert seed.get("p_closet_alice") is None
    # The parent is still here, so its own turns are recallable — through the parent
    # only. No second row holds them.
    assert [s.memory.id for s in seed.retrieve(SECRET, limit=10)] == ["p"]


def test_a_real_note_at_an_announced_id_is_still_ingested(tmp_path: Path) -> None:
    """A claim suppresses only what it can PROVE is the copy, and reclaiming the id drops it."""
    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=30)
    _pull_leaked_pair(tmp_path, seed, store, when=when)
    assert forget(seed, tmp_path, "p", tombstones=store).deleted is True
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM closet_tombstones")  # only the claim is left
    conn.commit()
    conn.close()

    note = _cloud_row("p_closet_alice", "AN INDEPENDENT NOTE", when=datetime.now(timezone.utc))
    result = pull(engine=seed, tombstones=store, client=_RecordingClient([note]), state=SyncState(), poppy_dir=tmp_path)

    assert result.applied_live == 1
    assert seed.get("p_closet_alice").content == "AN INDEPENDENT NOTE"
    assert "p_closet_alice" not in _pending_ids(store)


@pytest.mark.parametrize("engine_name", ["seed", "bloom"])
def test_a_same_text_write_does_not_invent_a_link_for_a_real_lookalike(tmp_path: Path, engine_name: str) -> None:
    """Only a row the parent PROVES keeps its stored link; a real note takes the caller's."""
    db = tmp_path / "memories.db"
    engine = SeedEngine(db_path=db) if engine_name == "seed" else _bloom(db)
    engine.ingest(_memory("p", _turns()))
    engine.ingest(_memory("p_closet_notes", "MY OWN NOTE", related_to=["somewhere-else"]))

    engine.ingest(_memory("p_closet_notes", "MY OWN NOTE", related_to=[]))

    assert engine.get("p_closet_notes").related_to == []


def test_a_dry_run_on_a_migrated_store_still_runs(tmp_path: Path) -> None:
    """The guard is about the pending migration only; an ordinary dry run is unchanged."""
    from click.testing import CliRunner

    from poppy.cli.main import cli

    poppy_dir = tmp_path / ".poppy"
    poppy_dir.mkdir()
    SeedEngine(db_path=poppy_dir / "memories.db").ingest(_memory("note", "MY NOTE"))
    (poppy_dir / "config.json").write_text(
        json.dumps({"trags_api_key": "usr_test", "trags_api_url": "http://127.0.0.1:9", "engine": "seed"})
    )

    env = {"HOME": str(tmp_path), "POPPY_DIR": str(poppy_dir), "POPPY_TELEMETRY_OFF": "1"}
    result = CliRunner().invoke(cli, ["sync", "push", "--dry-run"], env=env)

    assert "one-time marker migration" not in result.output
    assert "push: 1 live" in result.output


def test_a_pulled_copy_snapshot_stays_local_after_parent_deletion(tmp_path: Path) -> None:
    """A proven copy snapshot stays non-restorable after its parent is deleted."""
    from poppy.write_flow import restore

    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=30)
    parent, copy = _leaked_pair(when)
    copy["deleted_at"] = copy["updated_at"] = (when + timedelta(hours=1)).isoformat()
    pull(engine=seed, tombstones=store, client=_RecordingClient([parent, copy]), state=SyncState(), poppy_dir=tmp_path)

    assert store.announced_copy_claim("p_closet_alice") is not None
    assert "p_closet_alice" in _pending_ids(store)

    assert forget(seed, tmp_path, "p", tombstones=store).deleted is True
    cloud = _FreshnessClient([parent, copy])
    push(engine=seed, tombstones=store, client=cloud, state=SyncState(), poppy_dir=tmp_path)

    assert not any(r["id"] == copy["id"] for r in cloud.upserts)

    store.purge_expired(pushed_through=datetime.now(timezone.utc).isoformat())
    pull(
        engine=seed,
        tombstones=store,
        client=_RecordingClient([cloud.state["p_closet_alice"]]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )
    assert seed.get("p_closet_alice") is None
    assert restore(seed, tmp_path, "p_closet_alice", tombstones=store).found is False
    assert not seed.retrieve(SECRET, limit=10)


def test_a_newer_claim_suppresses_a_copy_an_older_record_would_admit(tmp_path: Path) -> None:
    """Record and claim both count, and the LATER one decides.

    Record at 10:00, a legacy Trash entry for the copy deleted at 12:00 whose
    refusal queues that claim. A cloud copy captured at 11:00 compared only
    against the record, so it was ingested — and the ingest cancelled the pending
    announcement, leaving the speaker text live here and up there.
    """
    from poppy.write_flow import restore

    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    ten = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(hours=6)
    eleven, twelve = ten + timedelta(hours=1), ten + timedelta(hours=2)
    _write_legacy(engine, _memory("p", _turns()))
    copy_text = engine.get("p_closet_alice").content  # exactly what p derives
    engine_copy_updated_at = engine.get("p_closet_alice").updated_at
    legacy = store.add(engine.get("p_closet_alice"), tombstoned_at=twelve)
    assert legacy.tombstoned_at == twelve
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM memories WHERE id = 'p_closet_alice'")
    conn.execute("DELETE FROM memory_embeddings WHERE id = 'p_closet_alice'")
    conn.commit()
    conn.close()
    store.add_copy_deletions(["p_closet_alice"], now=ten)

    with pytest.raises(ValueError, match="derived per-speaker copy"):
        restore(engine, tmp_path, "p_closet_alice", tombstones=store)
    assert store.get_copy_deletion("p_closet_alice").tombstoned_at == ten  # the older record
    # The claim is the LATER of the entry's own two stamps, so the server's freshness
    # gate cannot answer it stale.
    claimed = store.announced_copy_claim("p_closet_alice")
    assert claimed == max(twelve, engine_copy_updated_at)

    captured = _cloud_row("p_closet_alice", copy_text, when=eleven)
    captured["created_at"] = engine.get("p").created_at.isoformat()
    captured["related_to"] = ["p"]
    result = pull(
        engine=engine, tombstones=store, client=_RecordingClient([captured]), state=SyncState(), poppy_dir=tmp_path
    )

    assert result.applied_live == 0
    assert result.skipped_copies == 1
    assert engine.get("p_closet_alice") is None
    assert store.announced_copy_claim("p_closet_alice") is not None
    assert "p_closet_alice" in _pending_ids(store)


def test_a_future_stamped_claim_does_not_hide_a_later_cloud_row(tmp_path: Path) -> None:
    """A claim is an existence hint, never a comparator.

    Earlier clients advanced a claim from an incoming row's own updated_at.
    Used as a floor, even clamped to now, a claim dated ten days ahead hid every honestly stamped row
    written at that id, and since the queue is never purged it hid them for good,
    with the pull watermark moving past them. The row's own EVIDENCE decides instead.
    """
    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    seed.ingest(_memory("p", _turns()))
    far = datetime.now(timezone.utc) + timedelta(days=10)
    store.claim_leaked_copy("p_closet_alice", far)
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM closet_tombstones")  # only the future claim is left
    conn.commit()
    conn.close()

    note = _cloud_row("p_closet_alice", "AN INDEPENDENT NOTE", when=datetime.now(timezone.utc) - timedelta(hours=1))
    result = pull(engine=seed, tombstones=store, client=_RecordingClient([note]), state=SyncState(), poppy_dir=tmp_path)

    assert result.applied_live == 1
    assert seed.get("p_closet_alice").content == "AN INDEPENDENT NOTE"

    # Ingesting a real memory there reclaims the id, so the claim goes with it.
    assert store.announced_copy_claim("p_closet_alice") is None


def test_restore_returns_a_likely_trash_snapshot_to_the_user(tmp_path: Path) -> None:
    """Only PROVEN is refused. A LIKELY entry is as likely a memory the user wrote.

    An importer-written per-speaker split: right shape, the parent's creation
    instant, text the parent does not derive. Refusing it for good contradicted
    the frozen provenance guards, which leave
    anything short of proof to the user. The restored row keeps its stored
    related_to, so a later redaction of the parent still grades it.
    """
    from poppy.write_flow import restore

    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    parent = _memory("p", _turns())
    seed.ingest(parent)
    split = _memory(
        "p_closet_alice",
        json.dumps([{"speaker": "Alice", "dia_id": "D9", "text": "A CURATED SPLIT, NOT A COPY"}]),
        related_to=["p"],
    )
    split.created_at = parent.created_at
    seed.ingest(split)
    assert forget(seed, tmp_path, "p_closet_alice", tombstones=store).deleted is True
    assert store.get("p_closet_alice") is not None

    result = restore(seed, tmp_path, "p_closet_alice", tombstones=store)

    assert result.memory is not None and "CURATED SPLIT" in result.memory.content
    assert seed.get("p_closet_alice").related_to == ["p"]


def test_forgetting_an_adopted_copy_keeps_a_pre_image_of_its_trash_entry(tmp_path: Path) -> None:
    """The row-text rule is reached by a DIRECT forget too, which backed nothing up."""
    db = _legacy_store(tmp_path)
    TombstoneStore(db)  # the 0.2.4 client's Trash table
    conn = sqlite3.connect(str(db))
    stale = _rows(db, "SELECT content, created_at, updated_at FROM memories WHERE id = 'sess-2026-01_closet_alice'")[0]
    conn.execute(
        "INSERT INTO ui_tombstones (id, content, memory_type, project, source_type, source_session_id,"
        " source_timestamp, confidence, related_to, created_at, updated_at, tombstoned_at, token)"
        " VALUES (?, ?, 'fact', NULL, 'cli', NULL, ?, 1.0, ?, ?, ?, ?, 'tok')",
        ("sess-2026-01_closet_alice", stale[0], stale[1], json.dumps(["sess-2026-01"]), stale[1], stale[2], stale[2]),
    )
    conn.execute(
        "UPDATE memories SET content = ?, enriched_content = ? WHERE id = 'sess-2026-01'",
        (_turns("harmlessreplacement"), "x"),
    )
    conn.commit()
    conn.close()
    engine, store = _bloom(db), TombstoneStore(db)  # adopts the stale copy as LIKELY
    assert "sess-2026-01_closet_alice" in _adopted_unverified_ids(db)
    # The adoption's own pre-image ages out after the seven-day window, so it is not
    # what keeps this text: with it gone, the entry went with nothing behind it.
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM closet_migration_backup")
    conn.commit()
    conn.close()
    assert _backup_rows(db) == {}

    assert forget(engine, tmp_path, "sess-2026-01_closet_alice", tombstones=store).deleted is True

    assert store.get("sess-2026-01_closet_alice") is None
    assert _backup_rows(db) == {"sess-2026-01_closet_alice": "cleared"}
    kept = _rows(db, "SELECT content FROM closet_migration_backup WHERE id = ?", ("sess-2026-01_closet_alice",))
    assert SECRET in kept[0][0]


def test_a_failed_liveness_read_is_recorded_like_any_other_push_error(tmp_path: Path) -> None:
    """The re-read touches SQLite, so it belongs inside the per-item try.

    Raised out of the loop it left push before it persisted anything: no
    watermark, no error stamp, and every pending announcement skipped.
    """
    from poppy.sync import remote_state_for

    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    seed.ingest(_memory("note", "MY NOTE"))
    store.claim_leaked_copy("legacy_closet_alice", datetime.now(timezone.utc) - timedelta(days=1))
    client = _RecordingClient()

    def get_explodes(_memory_id):
        raise sqlite3.OperationalError("database is locked")

    seed.get = get_explodes  # type: ignore[method-assign]
    result = push(engine=seed, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)

    # Recorded as a LOCAL failure, not a network one: no raw sqlite error at the CLI,
    # the watermark frozen, and no upload attempted.
    assert result.errors == 1 and result.sent_live == 0
    state = remote_state_for(tmp_path, client.base_url)
    assert "local read failed" in state.errors["push"]
    assert state.last_pushed_at is None
    assert client.upserts == []
    assert "legacy_closet_alice" in _pending_ids(store)


def test_an_authoritative_deletion_raises_a_local_floor(tmp_path: Path) -> None:
    """The docstring claims it raises over a LOCAL record too, so prove it — up to now."""
    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    engine.ingest(_memory("sess-1", _turns()))
    assert forget(engine, tmp_path, "sess-1", tombstones=store).deleted is True
    store.add_copy_deletions(["sess-1_closet_alice", "sess-1_closet_bob"])
    assert _rows(db, "SELECT is_local FROM closet_tombstones WHERE id = 'sess-1_closet_alice'") == [(1,)]
    old = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(days=3)
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE closet_tombstones SET tombstoned_at = ?", (old.isoformat(),))
    conn.commit()
    conn.close()

    later = old + timedelta(days=1)
    cleanup = _legacy_deletion_row("sess-1_closet_alice", later)
    pull(engine=engine, tombstones=store, client=_RecordingClient([cleanup]), state=SyncState(), poppy_dir=tmp_path)

    assert _local_deletion_time(db, "sess-1_closet_alice") == later
    # ... and an older cleanup still cannot lower it.
    older = _legacy_deletion_row("sess-1_closet_alice", old)
    pull(engine=engine, tombstones=store, client=_RecordingClient([older]), state=SyncState(), poppy_dir=tmp_path)
    assert _local_deletion_time(db, "sess-1_closet_alice") == later

    # A future deletion follows the same last-writer-wins ordering.
    ahead = _legacy_deletion_row("sess-1_closet_alice", datetime.now(timezone.utc) + timedelta(days=365))
    pull(engine=engine, tombstones=store, client=_RecordingClient([ahead]), state=SyncState(), poppy_dir=tmp_path)
    raised = _local_deletion_time(db, "sess-1_closet_alice")
    assert raised == datetime.fromisoformat(ahead["deleted_at"])


def test_a_dry_run_without_a_remote_still_reports_the_missing_key(tmp_path: Path) -> None:
    """The migration guard runs after the remote check: an unconfigured remote comes first."""
    from click.testing import CliRunner

    from poppy.cli.main import cli

    poppy_dir = tmp_path / ".poppy"
    poppy_dir.mkdir()
    _legacy_store(poppy_dir)
    (poppy_dir / "config.json").write_text(json.dumps({"engine": "seed"}))

    env = {"HOME": str(tmp_path), "POPPY_DIR": str(poppy_dir), "POPPY_TELEMETRY_OFF": "1"}
    result = CliRunner().invoke(cli, ["sync", "run", "--dry-run"], env=env)

    assert result.exit_code != 0
    assert "Trags API key not configured" in result.output
    assert "one-time marker migration" not in result.output


def test_restore_leaves_an_independent_live_note_at_a_copys_id_alone(tmp_path: Path) -> None:
    """Grading the Trash entry must not happen while something is LIVE at that id.

    A PROVEN legacy snapshot for the copy, and an independent note the user (or an
    importer) wrote at the same id, stamped BEFORE that deletion. Grading first
    deleted the entry and queued a cleanup carrying the entry's own time — against
    the note's id — so push overwrote the cloud note and pulling the cleanup deleted
    the local one, leaving only the placeholder in Trash.
    """
    from poppy.write_flow import restore

    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    eleven = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(hours=2)
    twelve = eleven + timedelta(hours=1)
    _write_legacy(engine, _memory("p", _turns()))
    store.add(engine.get("p_closet_alice"), tombstoned_at=twelve)  # the 0.2.4 entry
    note = _memory("p_closet_alice", "AN INDEPENDENT NOTE")
    note.updated_at = note.created_at = eleven
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM memories WHERE id = 'p_closet_alice'")
    conn.execute("DELETE FROM memory_embeddings WHERE id = 'p_closet_alice'")
    conn.commit()
    conn.close()
    engine.ingest(note)

    result = restore(engine, tmp_path, "p_closet_alice", tombstones=store)

    assert result.already_live is True
    assert engine.get("p_closet_alice").content == "AN INDEPENDENT NOTE"
    assert store.get("p_closet_alice") is None  # the stale entry is cleared, as before
    assert "p_closet_alice" not in _pending_ids(store)

    cloud = _FreshnessClient([_cloud_row("p_closet_alice", "AN INDEPENDENT NOTE", when=eleven)])
    push(engine=engine, tombstones=store, client=cloud, state=SyncState(), poppy_dir=tmp_path)
    assert cloud.state["p_closet_alice"]["content"] == "AN INDEPENDENT NOTE"
    assert cloud.state["p_closet_alice"]["deleted_at"] is None


def test_a_claim_dated_ahead_does_not_suppress_a_recreation(tmp_path: Path) -> None:
    """A claim is existence, not a clock — not even clamped to now.

    A migrated claim dated ten days ahead, a local copy deletion from yesterday, and
    an independent cloud note from an hour ago: the note is newer than the deletion,
    so it must be ingested. Compared against the claim it was skipped, and the pull
    watermark moved past it, losing it on this device for good.
    """
    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    seed.ingest(_memory("p", _turns()))
    store.add_copy_deletions(["p_closet_alice"], now=datetime.now(timezone.utc) - timedelta(days=1))
    store.claim_leaked_copy("p_closet_alice", datetime.now(timezone.utc) + timedelta(days=10))

    note = _cloud_row("p_closet_alice", "AN INDEPENDENT NOTE", when=datetime.now(timezone.utc) - timedelta(hours=1))
    result = pull(engine=seed, tombstones=store, client=_RecordingClient([note]), state=SyncState(), poppy_dir=tmp_path)

    assert result.skipped_copies == 0
    assert result.applied_live == 1
    assert seed.get("p_closet_alice").content == "AN INDEPENDENT NOTE"


def test_the_leaked_copy_itself_is_still_suppressed_under_a_claim(tmp_path: Path) -> None:
    """The other half of the evidence rule: PROVEN is suppressed however it is stamped."""
    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=30)
    _pull_leaked_pair(tmp_path, seed, store, when=when)
    leaked = _cloud_row("p_closet_alice", seed.get("p_closet_alice").content, when=when)
    leaked["created_at"] = seed.get("p").created_at.isoformat()
    leaked["related_to"] = ["p"]
    assert forget(seed, tmp_path, "p_closet_alice", tombstones=store).deleted is True
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM closet_tombstones")  # only the claim is left
    conn.commit()
    conn.close()

    # Stamped in the future, so no comparison would have saved us.
    leaked["updated_at"] = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
    result = pull(
        engine=seed, tombstones=store, client=_RecordingClient([leaked]), state=SyncState(), poppy_dir=tmp_path
    )

    assert result.skipped_copies == 1
    assert seed.get("p_closet_alice") is None


def test_push_sends_the_freshly_read_row_not_the_scanned_one(tmp_path: Path) -> None:
    """An edit landing between the candidate scan and the upload is what goes up."""
    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    seed.ingest(_memory("note", "THE SECRET IS " + SECRET))

    real_list_all = seed.list_all
    edited: list[bool] = []

    def list_all_then_an_edit_lands(*args, **kwargs):
        rows = real_list_all(*args, **kwargs)
        if not edited:
            edited.append(True)
            seed.ingest(_memory("note", "redacted"))
        return rows

    seed.list_all = list_all_then_an_edit_lands  # type: ignore[method-assign]
    client = _RecordingClient()
    push(engine=seed, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)

    assert edited
    sent = [r for r in client.upserts if r["id"] == "note"]
    assert sent and all(r["content"] == "redacted" for r in sent)
    assert not any(SECRET in json.dumps(r) for r in client.upserts)


def test_a_republished_copy_stays_suppressed_without_upload(tmp_path: Path) -> None:
    """Local evidence suppresses a stale copy without an outgoing deletion marker."""
    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=30)
    _pull_leaked_pair(tmp_path, seed, store, when=when)
    copy_text = seed.get("p_closet_alice").content
    assert forget(seed, tmp_path, "p_closet_alice", tombstones=store).deleted is True
    store.mark_legacy_announced(store.pending_legacy_announcements())  # the cloud accepted it
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM closet_tombstones")  # the record has aged out
    conn.commit()
    conn.close()

    republished = _cloud_row("p_closet_alice", copy_text, when=datetime.now(timezone.utc) - timedelta(hours=1))
    republished["created_at"] = seed.get("p").created_at.isoformat()
    republished["related_to"] = ["p"]
    result = pull(
        engine=seed, tombstones=store, client=_RecordingClient([republished]), state=SyncState(), poppy_dir=tmp_path
    )

    assert result.skipped_copies == 1
    assert seed.get("p_closet_alice") is None
    assert store.announced_copy_claim("p_closet_alice") is not None
    assert "p_closet_alice" in _pending_ids(store)

    assert forget(seed, tmp_path, "p", tombstones=store).deleted is True
    cloud = _FreshnessClient([republished])
    push(engine=seed, tombstones=store, client=cloud, state=SyncState(), poppy_dir=tmp_path)

    assert not any("_closet_" in row["id"] for row in cloud.upserts)


def test_a_proven_copy_newer_than_the_record_is_still_suppressed(tmp_path: Path) -> None:
    """PROVEN means PROVEN, whatever the row is stamped — a record present does not soften it."""
    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=30)
    _pull_leaked_pair(tmp_path, seed, store, when=when)
    copy_text = seed.get("p_closet_alice").content
    assert forget(seed, tmp_path, "p_closet_alice", tombstones=store).deleted is True
    record = store.get_copy_deletion("p_closet_alice")
    assert record is not None

    ahead = _cloud_row("p_closet_alice", copy_text, when=datetime.now(timezone.utc) + timedelta(days=3))
    ahead["created_at"] = seed.get("p").created_at.isoformat()
    ahead["related_to"] = ["p"]
    result = pull(
        engine=seed, tombstones=store, client=_RecordingClient([ahead]), state=SyncState(), poppy_dir=tmp_path
    )

    assert result.skipped_copies == 1
    assert result.applied_live == 0
    assert seed.get("p_closet_alice") is None
    claimed = store.announced_copy_claim("p_closet_alice")
    assert claimed == datetime.fromisoformat(ahead["updated_at"])  # raised to what we saw


def test_a_pulled_copy_snapshot_replaces_an_older_entry_at_that_id(tmp_path: Path) -> None:
    """Recording content-free must not leave an older content-carrying entry beside it.

    ``tombstones.add`` — what ran here before the grading branch — would have replaced
    it; recording beside it left the speaker text restorable and pushable.
    """
    from poppy.write_flow import restore

    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=30)
    parent, copy = _leaked_pair(when)
    pull(engine=seed, tombstones=store, client=_RecordingClient([parent]), state=SyncState(), poppy_dir=tmp_path)
    older = _memory("p_closet_alice", "AN EARLIER LIFE OF THAT ID")
    store.add(older, tombstoned_at=when)
    copy["deleted_at"] = copy["updated_at"] = (when + timedelta(hours=1)).isoformat()

    pull(engine=seed, tombstones=store, client=_RecordingClient([copy]), state=SyncState(), poppy_dir=tmp_path)

    assert store.get("p_closet_alice") is None
    assert restore(seed, tmp_path, "p_closet_alice", tombstones=store).found is False
    assert store.get_copy_deletion("p_closet_alice") is not None


def test_a_newer_entry_at_that_id_survives_a_pulled_copy_snapshot(tmp_path: Path) -> None:
    """The one case add would have KEPT: a note deleted at that id after the copy was."""
    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=30)
    parent, copy = _leaked_pair(when)
    pull(engine=seed, tombstones=store, client=_RecordingClient([parent]), state=SyncState(), poppy_dir=tmp_path)
    note = _memory("p_closet_alice", "A NOTE DELETED LATER")
    store.add(note, tombstoned_at=when + timedelta(days=2))
    copy["deleted_at"] = copy["updated_at"] = (when + timedelta(hours=1)).isoformat()

    pull(engine=seed, tombstones=store, client=_RecordingClient([copy]), state=SyncState(), poppy_dir=tmp_path)

    kept = store.get("p_closet_alice")
    assert kept is not None and kept.memory.content == "A NOTE DELETED LATER"


@pytest.mark.parametrize("engine_kind", ["bloom", "seed"])
def test_a_copy_kept_on_inference_is_kept_out_of_list_stats_and_recall(tmp_path: Path, engine_kind: str) -> None:
    """Kept in the store, and out of the three ways of reading it that search.

    Its text is not rewritten, because it may be a split someone curated rather
    than a stale copy. It is still a projection of text the memory beside it
    holds, so a listing, a count and a recall must not surface it.

    Scope: list, stats and recall only. A read by exact id still returns the
    row, which is what the cleanup and the redaction path rely on.
    """
    db = _tier_b_store(tmp_path)
    copy_id = "sess-2026-01_closet_alice"

    engine = SeedEngine(db) if engine_kind == "seed" else _bloom(db)

    assert SECRET in engine.get(copy_id).content
    assert [m.id for m in engine.list_all()] == ["sess-2026-01"]
    assert engine.stats().memory_count == 1
    assert copy_id not in [scored.memory.id for scored in engine.retrieve(SECRET, limit=10)]


@pytest.mark.parametrize("engine_kind", ["bloom", "seed"])
def test_forgetting_a_memory_takes_the_copy_kept_beside_it(tmp_path: Path, engine_kind: str) -> None:
    """A copy must not outlive the text it was copied from.

    Nothing regenerates these rows, so the memory being deleted is the only
    thing that kept this one explainable.
    """
    db = _tier_b_store(tmp_path)
    copy_id = "sess-2026-01_closet_alice"
    engine = SeedEngine(db) if engine_kind == "seed" else _bloom(db)
    store = TombstoneStore(db)
    assert engine.get(copy_id) is not None

    assert forget(engine, tmp_path, "sess-2026-01", tombstones=store).deleted is True

    assert engine.get(copy_id) is None
    assert _all_ids(db) == []
    # Content-free on the way out: no Trash entry carries the speaker text.
    assert all(SECRET not in (entry.memory.content or "") for entry in store.list_all())


@pytest.mark.parametrize("engine_kind", ["bloom", "seed"])
def test_editing_the_text_takes_a_copy_of_what_it_replaced(tmp_path: Path, engine_kind: str) -> None:
    """Editing a secret out of a memory must not leave a copy holding it."""
    db = _tier_b_store(tmp_path)
    copy_id = "sess-2026-01_closet_alice"
    engine = SeedEngine(db) if engine_kind == "seed" else _bloom(db)

    engine.ingest(_memory("sess-2026-01", "a plain sentence with nothing sensitive in it"))

    assert engine.get(copy_id) is None
    assert _all_ids(db) == ["sess-2026-01"]


def test_a_field_not_stored_as_text_earns_no_grade(tmp_path: Path) -> None:
    """Storage class decides what may be graded, not what the bytes spell.

    A value stored as a blob is not one any release that wrote copies produced,
    so it is nobody's copy however exactly it matches. Decoding it and grading
    on the text alone made a row that spells the projection removable, and the
    removal keeps no pre-image.
    """
    db = _legacy_store(tmp_path)
    copy_id = "sess-2026-01_closet_alice"
    projected = _rows(db, "SELECT content FROM memories WHERE id = ?", (copy_id,))[0][0]
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE memories SET content = CAST(? AS BLOB) WHERE id = ?", (projected, copy_id))
    conn.commit()
    conn.close()
    assert _rows(db, "SELECT typeof(content) FROM memories WHERE id = ?", (copy_id,)) == [("blob",)]

    engine = _bloom(db)

    assert engine.get(copy_id) is not None
    assert _rows(db, "SELECT is_closet FROM memories WHERE id = ?", (copy_id,)) == [(0,)]
    assert copy_id in [m.id for m in engine.list_all()]
    assert _backup_rows(db) == {}
    # Its sibling is still stored as text, so it is graded and cleaned as before.
    assert engine.get("sess-2026-01_closet_bob") is None


def test_a_row_that_is_not_text_leaves_the_store_openable(tmp_path: Path) -> None:
    """Bytes that are not text in a candidate column must not brick the store.

    The database driver raises while BUILDING the result set, before any per-row
    guard can run, and that rolls back the marker column with it, so every later
    open retried and failed the same way.
    """
    db = _legacy_store(tmp_path)
    stamp = "2020-01-01T00:00:00+00:00"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO memories (id, content, enriched_content, memory_type, project, source_type,"
        " source_session_id, source_timestamp, confidence, related_to, created_at, updated_at,"
        " expires_at) VALUES ('meeting_closet_bytes', CAST(? AS TEXT), 'x', 'fact', NULL, 'cli',"
        " NULL, ?, 1.0, '[]', ?, ?, NULL)",
        (b"\xff\xfe not text", stamp, stamp, stamp),
    )
    conn.commit()
    conn.close()

    _bloom(db)  # must not raise
    engine = _bloom(db)  # and the store stays openable afterwards

    assert MARKER_COLUMN in {r[1] for r in _rows(db, "PRAGMA table_info(memories)")}
    # The rows it COULD read were still graded and cleaned.
    assert engine.get("sess-2026-01_closet_alice") is None
    # The one it could not is left exactly as it was.
    assert _rows(db, "SELECT is_closet FROM memories WHERE id = 'meeting_closet_bytes'") == [(0,)]


@pytest.mark.parametrize("engine_kind", ["seed", "bloom"])
@pytest.mark.parametrize("changed_field", [None, "related_to", "created_at", "content"])
def test_first_open_requires_all_copy_evidence(tmp_path: Path, engine_kind: str, changed_field: str | None) -> None:
    db = _legacy_store(tmp_path)
    copy_id = "sess-2026-01_closet_alice"
    edits = {
        "related_to": json.dumps(["independent-record"]),
        "created_at": "2020-01-01T00:00:00+00:00",
        "content": json.dumps([{"speaker": "Alice", "text": "An independently edited split"}]),
    }
    with sqlite3.connect(db) as conn:
        if changed_field:
            conn.execute(f"UPDATE memories SET {changed_field} = ? WHERE id = ?", (edits[changed_field], copy_id))
        before = conn.execute("SELECT content FROM memories WHERE id = ?", (copy_id,)).fetchone()[0]
    engine = SeedEngine(db) if engine_kind == "seed" else _bloom(db)
    if changed_field is None:
        assert engine.get(copy_id) is None
        assert _local_deletion_time(db, copy_id) is not None
        assert _backup_rows(db) == {}
    else:
        assert engine.get(copy_id).content == before
        assert _local_deletion_time(db, copy_id) is None
        expected_marker = 2 if changed_field == "content" else 0
        assert _rows(db, "SELECT is_closet FROM memories WHERE id = ?", (copy_id,)) == [(expected_marker,)]
        assert _backup_rows(db) == ({copy_id: "adopted"} if expected_marker else {})
    assert engine.get("sess-2026-01_closet_bob") is None
    assert _pending_ids(TombstoneStore(db)) == []


@pytest.mark.parametrize("engine_kind", ["seed", "bloom"])
def test_pre_marker_cleanup_keeps_the_copys_own_deletion_time(tmp_path: Path, engine_kind: str) -> None:
    db = _legacy_store(tmp_path)
    stamp = "2020-06-01T10:00:00+02:00"
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE memories SET updated_at = ? WHERE id = 'sess-2026-01_closet_alice'", (stamp,))
    engine = SeedEngine(db) if engine_kind == "seed" else _bloom(db)
    assert engine.get("sess-2026-01_closet_alice") is None
    assert _local_deletion_time(db, "sess-2026-01_closet_alice") == datetime.fromisoformat(stamp)
    assert _backup_rows(db) == {}


def test_legacy_projection_handles_colliding_slugs_and_nested_parent_ids(tmp_path: Path) -> None:
    from poppy.engine._legacy_copies import TIER_NONE, TIER_PROVEN, classify_legacy_copy

    db = tmp_path / "memories.db"
    engine = SeedEngine(db)
    parent = _memory(
        "session_closet_archive",
        json.dumps(
            [
                {"speaker": "O'Brien", "text": "first"},
                {"speaker": "O Brien", "text": "second"},
                {"speaker": "O Brien 2", "text": "third"},
            ]
        ),
    )
    engine.ingest(parent)
    for slug, turn in zip(("o_brien", "o_brien_2", "o_brien_2_2"), json.loads(parent.content)):
        tier, owner = classify_legacy_copy(
            engine._conn,
            parent.id + "_closet_" + slug,
            content=json.dumps([turn]),
            related_raw=json.dumps([parent.id]),
            created_at=parent.created_at.isoformat(),
        )
        assert (tier, owner) == (TIER_PROVEN, parent.id)
    tier, _ = classify_legacy_copy(
        engine._conn,
        parent.id + "_closet_wrong",
        content=json.dumps([json.loads(parent.content)[0]]),
        related_raw=json.dumps([parent.id]),
        created_at=parent.created_at.isoformat(),
    )
    assert tier == TIER_NONE


def test_reclaiming_an_id_cancels_its_pending_announcement(tmp_path: Path) -> None:
    """Forget the parent, write an independent note at a copy's id, then sync.

    Left pending, push sent the note live and then a NEWER deletion for the same
    id, and the next pull removed the note the user had just written.
    """
    db, engine = _legacy_store_open(tmp_path)
    store = TombstoneStore(db)
    assert "sess-2026-01_closet_alice" in _pending_ids(store)

    forget(engine, tmp_path, "sess-2026-01", tombstones=store)
    engine.ingest(_memory("sess-2026-01_closet_alice", "an independent note of my own"))

    assert _pending_ids(store) == ["sess-2026-01_closet_bob"]
    assert store.get_copy_deletion("sess-2026-01_closet_alice") is None

    client = _RecordingClient(echo=True)
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)

    for_id = [r for r in client.upserts if r["id"] == "sess-2026-01_closet_alice"]
    assert len(for_id) == 1
    assert for_id[0]["deleted_at"] is None  # live, not a deletion
    assert for_id[0]["content"] == "an independent note of my own"

    # The independent note survives the round trip.
    result = pull(
        engine=engine,
        tombstones=store,
        client=client,
        state=SyncState(),
        poppy_dir=tmp_path,
    )
    assert result.skipped_copies == 0
    assert _pending_ids(store) == ["sess-2026-01_closet_bob"]
    assert engine.get("sess-2026-01_closet_alice").content == "an independent note of my own"


def test_announcement_claims_compare_as_instants_across_offsets(tmp_path: Path) -> None:
    """A claim stamped ``12:00+02:00`` (10:00Z) must be advanced by a proven re-push at ``11:00Z``."""
    from poppy.engine._legacy_copies import COPY_CLAIM_TABLE, rearm_legacy_announcement

    db = tmp_path / "memories.db"
    _bloom(db)
    conn = sqlite3.connect(str(db))
    rearm_legacy_announcement(conn, "x_closet_alice", "2026-07-01T12:00:00+02:00")
    rearm_legacy_announcement(conn, "x_closet_alice", "2026-07-01T11:00:00+00:00")
    conn.commit()
    sql = f"SELECT legacy_updated_at FROM {COPY_CLAIM_TABLE} WHERE id = ?"
    (stored,) = conn.execute(sql, ("x_closet_alice",)).fetchone()
    conn.close()
    assert datetime.fromisoformat(stored) == datetime(2026, 7, 1, 11, 0, tzinfo=timezone.utc)


def test_a_copy_deletion_record_outlives_the_window_while_its_announcement_is_pending(tmp_path: Path) -> None:
    """The push watermark is not an acknowledgement of the announcement queue.

    Offline eight days, then a sync where ordinary pushes succeed and every
    announcement fails: the watermark advances, and housekeeping used to purge
    the record that makes pull skip the cloud's stale copy. The next pull then
    re-ingested the leaked copy and that ingest cancelled the announcement.
    """
    from poppy.engine._legacy_copies import mark_legacy_announced, rearm_legacy_announcement

    db = tmp_path / "memories.db"
    _bloom(db)
    store = TombstoneStore(db)
    eight_days_ago = datetime.now(timezone.utc) - timedelta(days=8)
    store.add_copy_deletions(["x_closet_alice", "y_closet_bob"], now=eight_days_ago)
    conn = sqlite3.connect(str(db))
    rearm_legacy_announcement(conn, "x_closet_alice", eight_days_ago.isoformat())
    conn.commit()

    store.purge_expired(pushed_through=datetime.now(timezone.utc).isoformat())
    assert store.get_copy_deletion("x_closet_alice") is not None  # announcement still pending
    assert store.get_copy_deletion("y_closet_bob") is None  # nothing pending: window + watermark apply

    mark_legacy_announced(conn, ["x_closet_alice"])
    conn.commit()
    conn.close()
    store.purge_expired(pushed_through=datetime.now(timezone.utc).isoformat())
    assert store.get_copy_deletion("x_closet_alice") is None


@pytest.mark.parametrize("engine_kind", ["seed", "bloom"])
def test_public_accessor_hides_an_inferred_copy(tmp_path, engine_kind):
    db = _tier_b_store(tmp_path)
    engine = SeedEngine(db_path=db) if engine_kind == "seed" else _bloom(db)
    copy_id = "sess-2026-01_closet_alice"
    assert engine.get(copy_id) is not None
    assert engine.get_public(copy_id) is None
    assert engine.get_public("missing") is None
    assert engine.get_public("sess-2026-01") == engine.get("sess-2026-01")


@pytest.mark.parametrize("engine_kind", ["seed", "bloom"])
@pytest.mark.parametrize("operation", ["forget", "edit", "expire"])
def test_parent_removal_does_not_expose_a_hidden_copys_old_snapshot(tmp_path, engine_kind, operation):
    from dataclasses import replace

    db = _tier_b_store(tmp_path)
    engine = SeedEngine(db_path=db) if engine_kind == "seed" else _bloom(db)
    copy_id = "sess-2026-01_closet_alice"
    copy = engine.get(copy_id)
    store = TombstoneStore(db)
    store.add(copy)
    parent = engine.get("sess-2026-01")

    # While the copy and its snapshot are still stored, both must already read
    # as missing. Asserting this only after the parent is gone would prove
    # nothing about the accessors: by then the rows are deleted, so an
    # unfiltered read answers None too.
    assert engine.get(copy_id) is not None
    assert store.get(copy_id) is not None
    assert engine.get_public(copy_id) is None
    assert store.get_public(copy_id) is None
    assert copy.content not in json.dumps([t.memory.content for t in store.list_public()])

    if operation == "forget":
        forget(engine, tmp_path, parent.id, tombstones=store)
    elif operation == "edit":
        engine.ingest(replace(parent, content="replacement"))
    else:
        engine.ingest(replace(parent, expires_at=datetime.now(timezone.utc) - timedelta(days=1)))
        engine.purge_expired()
    assert engine.get(copy_id) is None
    assert store.get(copy_id) is None
    assert store.get_public(copy_id) is None
    assert copy.content not in json.dumps([t.memory.content for t in store.list_all()])


def test_push_rechecks_a_snapshot_cleared_by_parent_redaction(tmp_path, monkeypatch):
    db = _tier_b_store(tmp_path)
    engine = _bloom(db)
    copy = engine.get("sess-2026-01_closet_alice")
    store = TombstoneStore(db)
    store.add(copy)
    store.note_remote_memories([copy.id], "https://trags.test")
    original = store.list_all

    def scan_then_redact():
        snapshots = original()
        engine.delete("sess-2026-01")
        return snapshots

    monkeypatch.setattr(store, "list_all", scan_then_redact)
    client = _RecordingClient()
    result = push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert result.sent_live == 0
    assert result.sent_tombstones == 0
    assert result.errors == 0
    assert client.upserts == []


@pytest.mark.parametrize("command", ["push", "pull", "run"])
def test_sync_preview_uses_open_time_cleanup(tmp_path, monkeypatch, command):
    import importlib

    from click.testing import CliRunner

    cli_module = importlib.import_module("poppy.cli.main")
    db = _legacy_store(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"engine": "seed"}))

    class Client(_RecordingClient):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    client = Client()
    monkeypatch.setattr(cli_module, "_sync_client", lambda: (client, client.base_url))
    result = CliRunner().invoke(
        cli_module.cli, ["sync", command, "--dry-run"], env={"POPPY_DIR": str(tmp_path), "HOME": str(tmp_path)}
    )
    assert result.exit_code == 0, result.output
    assert MARKER_COLUMN in {r[1] for r in _rows(db, "PRAGMA table_info(memories)")}
    assert _all_ids(db) == ["sess-2026-01"]
    assert client.upserts == []
    assert "one-time marker migration" not in result.output
    assert not (tmp_path / "sync_state.json").exists()
