"""Closet redaction: identity is a provenance marker, not an id heuristic.

The default engine expands a multi-speaker memory into one synthetic copy per
speaker. Those copies duplicate the parent's text, so a redaction that misses
them leaves the "forgotten" text recallable — and, before this fix, pushes it to
the cloud as a live row.

Three inferential identity tests shipped and broke the same way (a ``_closet_``
substring, a ``related_to`` back-reference, a ``mem_`` id prefix): each one
misclassifies an input that merely shares the shape, and ids of EXTERNAL origin
(sync pull, the web API, importers, research harnesses) never go through the
local id generators at all. Identity is now the ``memories.is_closet`` column,
written where bloom creates the row.

These tests are written against the three ways the fix could be wrong:

  a. a real memory shaped like a closet gets hidden or deleted;
  b. closet text stays recallable after a forget whose parent id is NOT ``mem_``;
  c. speaker text leaks into a tombstone or onto the sync wire.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from poppy.engine._closet_marker import MARKER_COLUMN, derive_closets, migrate_closet_marker
from poppy.engine.bloom import BloomEngine
from poppy.engine.seed import SeedEngine
from poppy.models import Filters, Memory, Source
from poppy.sync import MAX_CONSECUTIVE_TRANSPORT_FAILURES, MAX_LEGACY_ANNOUNCEMENTS_PER_SYNC, pull, push
from poppy.sync.serializer import CLOSET_TOMBSTONE_CONTENT
from poppy.sync.state import SyncState, save
from poppy.ui.tombstones import TombstoneStore
from poppy.write_flow import forget

SECRET = "zebrafishcadenza"


class _FakeBiEncoder:
    def embed(self, texts):
        for t in texts:
            h = hash(t) & 0xFFFF
            yield np.array([h & 0xF, (h >> 4) & 0xF, (h >> 8) & 0xF, (h >> 12) & 0xF], dtype=np.float32)


class _FakeCrossEncoder:
    def rerank(self, query, docs):
        return [float(len(d)) for d in docs]


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


# --- the marker itself ----------------------------------------------------


def test_bloom_marks_the_closets_it_creates_and_only_those(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("sess-2026-01", _turns()))

    assert _all_ids(db) == ["sess-2026-01", "sess-2026-01_closet_alice", "sess-2026-01_closet_bob"]
    assert _marked_ids(db) == ["sess-2026-01_closet_alice", "sess-2026-01_closet_bob"]


def test_a_reingest_that_drops_a_speaker_clears_the_stale_marker(tmp_path: Path) -> None:
    """A parent re-ingested as prose must not leave a marked row behind."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _turns()))
    engine.ingest(_memory("sess-2026-01", "redacted"))

    assert _all_ids(db) == ["sess-2026-01"]
    assert _marked_ids(db) == []
    assert not [r for r in engine.retrieve(SECRET, limit=10) if SECRET in r.memory.content]


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


# --- (b) redaction works for a parent id of ANY origin --------------------


@pytest.mark.parametrize(
    "parent_id",
    [
        "mem_deadbeef1234",  # minted locally by write_flow.make_memory_id
        "sess-2026-01",  # external: a research harness / importer id
        "7f3a2b1c-0000-4000-8000-000000000000",  # external: a cloud/web-API uuid
    ],
)
def test_forget_removes_the_speaker_copies_for_any_parent_id_origin(
    tmp_path: Path, parent_id: str, synced_state
) -> None:
    """The proving case: an external id must redact exactly like a local one.

    Every previous identity test keyed off the id, so a parent minted outside
    Poppy produced closets the redaction did not recognise: the text stayed in
    ``memories``, stayed recallable, and pushed to the cloud as a live row.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory(parent_id, _turns()))
    TombstoneStore(db).note_remote_memories({parent_id}, _RecordingClient.base_url)
    assert len(_marked_ids(db)) == 2

    result = forget(engine, tmp_path, parent_id)
    assert result.deleted is True

    # Nothing left in any table that could carry the text back.
    assert _all_ids(db) == []
    assert _rows(db, "SELECT id FROM memory_embeddings") == []
    assert not _bloom(db).retrieve(SECRET, limit=10)
    assert not SeedEngine(db_path=db).retrieve(SECRET, limit=10)

    # And nothing closet-shaped is offered to sync as a live row.
    client = _RecordingClient()
    push(
        engine=_bloom(db),
        tombstones=TombstoneStore(db),
        client=client,
        state=synced_state(),
        poppy_dir=tmp_path,
    )
    # Nothing live at all, and — on a store created after the marker fix — no
    # copy deletions either: those copies were never pushed, so the cloud has
    # nothing to delete. (The PARENT's tombstone does snapshot its content; that
    # is the 7-day restore window, unchanged by this fix.)
    assert [r for r in client.upserts if r["deleted_at"] is None] == []
    assert {r["id"] for r in client.upserts} == {parent_id}
    assert TombstoneStore(db).pending_legacy_announcements() == []


def test_forget_on_the_fallback_engine_removes_the_speaker_copies(tmp_path: Path) -> None:
    """Same guarantee when the store is bloom's but the write engine is seed."""
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("sess-2026-01", _turns()))

    seed = SeedEngine(db_path=db)
    assert forget(seed, tmp_path, "sess-2026-01").deleted is True

    assert _all_ids(db) == []
    assert not _bloom(db).retrieve(SECRET, limit=10)


# --- (c) tombstones carry no speaker text ---------------------------------


def test_closet_tombstones_are_content_free_locally_and_on_the_wire(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _turns()))
    forget(engine, tmp_path, "sess-2026-01")

    store = TombstoneStore(db)
    closets = {ct.id for ct in store.list_closets()}
    assert closets == {"sess-2026-01_closet_alice", "sess-2026-01_closet_bob"}

    # The sidecar the UI reads holds only the parent. No speaker row is
    # snapshotted anywhere, so nothing can restore one.
    assert [t.memory.id for t in store.list_all()] == ["sess-2026-01"]
    assert not any(SECRET in str(r) for r in _rows(db, "SELECT * FROM closet_tombstones"))

    # Queue the ids as leaked — only a copy that predates the marker migration
    # has a cloud row to delete, so only those are ever announced.
    conn = sqlite3.connect(str(db))
    conn.executemany("INSERT OR IGNORE INTO legacy_closet_ids (id) VALUES (?)", [(c,) for c in sorted(closets)])
    conn.commit()
    conn.close()

    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)

    sent = {r["id"]: r for r in client.upserts}
    assert closets <= set(sent)
    for cid in closets:
        row = sent[cid]
        assert row["deleted_at"] is not None
        assert row["content"] == CLOSET_TOMBSTONE_CONTENT
        assert row["related_to"] == []
        assert row["project"] is None
        assert row["source_type"] is None
    # Every field of a closet row is the id, the time, or a constant: neither the
    # redacted text nor a speaker NAME reaches the wire through one.
    closet_wire = json.dumps([sent[cid] for cid in closets])
    assert not any(term in closet_wire for term in (SECRET, "Alice", "Bob"))


def test_supersede_tombstones_the_old_memorys_speaker_copies(tmp_path: Path) -> None:
    from poppy.lifecycle import supersede_memory

    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _turns()))

    supersede_memory(engine, _memory("sess-2026-02", "the replacement"), "sess-2026-01", poppy_dir=tmp_path)

    assert _all_ids(db) == ["sess-2026-02"]
    assert {ct.id for ct in TombstoneStore(db).list_closets()} == {
        "sess-2026-01_closet_alice",
        "sess-2026-01_closet_bob",
    }


# --- (a) a real memory shaped like a closet is never touched --------------


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
    _real_lookalike(db)
    engine = _bloom(db)

    assert forget(engine, tmp_path, "mem_customer").deleted is True

    assert _all_ids(db) == ["mem_customer_closet_notes"]
    assert engine.get("mem_customer_closet_notes").content == "the wardrobe budget for Q3"
    assert {ct.id for ct in TombstoneStore(db).list_closets()} == {
        "mem_customer_closet_alice",
        "mem_customer_closet_bob",
    }


def test_a_seed_redaction_does_not_delete_a_real_lookalike(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    _real_lookalike(db)

    seed = SeedEngine(db_path=db)
    seed.ingest(_memory("mem_customer", "redacted"))  # content edit on the parent

    assert _all_ids(db) == ["mem_customer", "mem_customer_closet_notes"]
    assert seed.delete("mem_customer") is True
    assert _all_ids(db) == ["mem_customer_closet_notes"]


# --- the sync wire is never pattern-matched -------------------------------


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


# --- the one-time migration -----------------------------------------------


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


def _legacy_store(tmp_path: Path) -> Path:
    """A store written by a pre-marker client: closets present, column absent."""
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("sess-2026-01", _turns()))
    _strip_marker(db)
    return db


def test_migration_marks_closets_it_can_re_derive_from_a_live_parent(tmp_path: Path) -> None:
    db = _legacy_store(tmp_path)
    assert MARKER_COLUMN not in {r[1] for r in _rows(db, "PRAGMA table_info(memories)")}

    _bloom(db)  # opening the store runs the migration

    assert _marked_ids(db) == ["sess-2026-01_closet_alice", "sess-2026-01_closet_bob"]
    assert [m.id for m in _bloom(db).list_all()] == ["sess-2026-01"]


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
    before = _rows(db, "SELECT id, content, is_closet FROM memories ORDER BY id")

    migrate_closet_marker(sqlite3.connect(str(db)), had_bloom_schema=True)
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


# --- derivation is shared, so ingest and migration cannot drift -----------


def test_derive_closets_is_what_ingest_writes(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    memory = _memory("sess-2026-01", _turns())
    engine = _bloom(db)
    engine.ingest(memory)

    derived = derive_closets(memory.content, memory.source.timestamp.isoformat())
    assert [slug for slug, _, _ in derived] == ["alice", "bob"]
    for slug, raw, enriched in derived:
        row = _rows(
            db,
            "SELECT content, enriched_content FROM memories WHERE id = ?",
            (f"sess-2026-01_closet_{slug}",),
        )
        assert row == [(raw, enriched)]


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
    engine.ingest(_memory("sess-2026-01", _turns()))

    result = forget(engine, tmp_path, "sess-2026-01_closet_alice")
    assert result.deleted is True

    store = TombstoneStore(db)
    assert store.list_all() == []  # nothing restorable, nothing snapshotted
    assert store.get("sess-2026-01_closet_alice") is None
    assert {ct.id for ct in store.list_closets()} == {"sess-2026-01_closet_alice"}
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
    assert CLOSET_TOMBSTONE_CONTENT in wire
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
    assert store.list_closets() == []


# --- round 2: sync must never touch a marked closet in either direction ----


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


def test_pull_never_overwrites_a_marked_closet_with_a_live_cloud_row(tmp_path: Path) -> None:
    """A live cloud row for a marked closet id must not become a real memory.

    Ingesting it would write is_closet=0, so the copy would stop being derived
    data: it would list, it would push live, and forgetting its parent would no
    longer reach it — the secret ends up in the cloud as a live row.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _turns()))
    store = TombstoneStore(db)

    result = pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[_cloud_row("sess-2026-01_closet_alice", "cloud copy of Alice's turns")]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.skipped_closets == 1
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
    rows. They refer to the cloud's stale copy, not to the one this device
    derives locally, so applying them would silently break local recall.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _turns()))
    store = TombstoneStore(db)

    result = pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[_cloud_row("sess-2026-01_closet_alice", "leaked copy", deleted=True)]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.skipped_closets == 1
    assert result.applied_tombstones == 0
    assert _marked_ids(db) == ["sess-2026-01_closet_alice", "sess-2026-01_closet_bob"]
    assert store.list_all() == []  # no snapshot of the leaked text either
    assert "sess-2026-01_closet_alice" in {r.memory.id for r in engine.retrieve(SECRET, limit=10)}


def test_pull_does_not_resurrect_a_closet_this_device_just_forgot(tmp_path: Path) -> None:
    """An older cloud row must not undo a closet deletion inside its window."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _turns()))
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

    assert result.skipped_closets == 1
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
    assert result.skipped_closets == 0
    assert engine.get("mem_customer_closet_notes").content == "the wardrobe budget for Q3"
    assert sorted(m.id for m in engine.list_all()) == ["mem_customer", "mem_customer_closet_notes"]


# --- round 2: only bloom's closet synthesis sets the marker ---------------


def test_a_seed_write_over_a_closet_id_makes_it_a_real_memory(tmp_path: Path) -> None:
    """A note written at a closet's id must stop being derived data.

    Keeping the marker would hide the user's note from every list and then
    destroy it when the unrelated parent memory was deleted.
    """
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("sess-2026-01", _turns()))
    seed = SeedEngine(db_path=db)

    seed.ingest(_memory("sess-2026-01_closet_alice", "my own note about Alice"))

    assert _marked_ids(db) == ["sess-2026-01_closet_bob"]
    assert sorted(m.id for m in seed.list_all()) == ["sess-2026-01", "sess-2026-01_closet_alice"]

    # It survives the unrelated parent's deletion, and syncs as a real memory.
    assert forget(seed, tmp_path, "sess-2026-01").deleted is True
    assert _all_ids(db) == ["sess-2026-01_closet_alice"]
    assert seed.get("sess-2026-01_closet_alice").content == "my own note about Alice"


# --- round 2: the migration never rewrites a memory on the strength of an id


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


# --- round 2: malformed legacy content must not brick the store ----------


_LIST_SPEAKER = '[{"speaker": ["Alice"], "text": "malformed"}]'


def test_a_malformed_legacy_row_does_not_abort_the_migration(tmp_path: Path) -> None:
    """A non-string speaker must classify as "leave alone", not raise.

    A raise inside the backfill rolls back the ALTER too, so every subsequent
    open retries and fails the same way: the store becomes unopenable.
    """
    db = _legacy_store(tmp_path)
    _insert_legacy(db, "meeting_closet_alice", _LIST_SPEAKER, ["meeting"])
    _insert_legacy(db, "meeting_closet_notes", "just some prose", [])

    engine = _bloom(db)  # must not raise

    assert MARKER_COLUMN in {r[1] for r in _rows(db, "PRAGMA table_info(memories)")}
    assert engine.get("meeting_closet_alice").content == _LIST_SPEAKER
    assert engine.get("meeting_closet_notes").content == "just some prose"
    listed = {m.id for m in engine.list_all()}
    assert {"meeting_closet_alice", "meeting_closet_notes"} <= listed
    # The well-formed closets alongside them still got marked.
    assert _marked_ids(db) == ["sess-2026-01_closet_alice", "sess-2026-01_closet_bob"]


def test_ingesting_malformed_turns_does_not_crash_the_engine(tmp_path: Path) -> None:
    """derive_closets treats a non-string speaker as "not a turn", never an error."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _LIST_SPEAKER))

    assert derive_closets(_LIST_SPEAKER, None) == []
    assert _all_ids(db) == ["sess-2026-01"]
    assert SeedEngine(db_path=db).get("sess-2026-01").content == _LIST_SPEAKER


# --- round 2: every local removal of a closet reaches the cloud ----------


def test_an_edit_that_drops_a_speaker_tombstones_that_closet(tmp_path: Path) -> None:
    """An edit used as a redaction must not leave a cloud copy live."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _turns()))

    without_alice = json.dumps([{"speaker": "Bob", "text": "just us now"}, {"speaker": "Carol", "text": "agreed"}])
    engine.ingest(_memory("sess-2026-01", without_alice))

    store = TombstoneStore(db)
    # Alice's copy disappeared, so its deletion is announced. Bob's was rewritten
    # rather than removed, and Carol's is new, so neither is.
    assert {ct.id for ct in store.list_closets()} == {"sess-2026-01_closet_alice"}
    assert _marked_ids(db) == ["sess-2026-01_closet_bob", "sess-2026-01_closet_carol"]
    assert not [r for r in engine.retrieve(SECRET, limit=10) if SECRET in r.memory.content]


def test_an_unchanged_reingest_announces_no_closet_deletions(tmp_path: Path) -> None:
    """Re-ingesting the same memory must not push a burst of pointless deletes."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    memory = _memory("sess-2026-01", _turns())
    engine.ingest(memory)
    engine.ingest(memory)

    assert TombstoneStore(db).list_closets() == []
    assert len(_marked_ids(db)) == 2


def test_a_seed_content_edit_tombstones_the_copies_it_clears(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("sess-2026-01", _turns()))

    SeedEngine(db_path=db).ingest(_memory("sess-2026-01", "redacted"))

    assert {ct.id for ct in TombstoneStore(db).list_closets()} == {
        "sess-2026-01_closet_alice",
        "sess-2026-01_closet_bob",
    }


def test_expiry_does_not_announce_closet_deletions(tmp_path: Path) -> None:
    """TTL is not a redaction: the cloud row carries the same expires_at."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    expired = _memory("sess-2026-01", _turns())
    expired.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
    engine.ingest(expired)

    assert engine.purge_expired() == 1
    assert _all_ids(db) == []
    assert TombstoneStore(db).list_closets() == []


def test_forgetting_a_closet_reports_no_restore_window(tmp_path: Path) -> None:
    """Nothing was snapshotted, so advertising a 7-day window would be a lie."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _turns()))

    result = forget(engine, tmp_path, "sess-2026-01_closet_alice")

    assert result.deleted is True
    assert result.tombstoned is True
    assert result.tombstone is None


# --- round 3: a metadata edit must not launder a closet into a real memory --


@pytest.mark.parametrize("engine_name", ["bloom", "seed"])
def test_edit_memory_refuses_a_marked_closet(tmp_path: Path, engine_name: str) -> None:
    """`poppy edit <closet id> --project x` keeps the text and would clear the marker.

    The copy would then list, push live with the secret, and survive the
    parent's redaction. Editing a derived copy is refused; edit the parent.
    """
    from poppy.lifecycle import edit_memory

    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("sess-2026-01", _turns()))
    engine = _bloom(db) if engine_name == "bloom" else SeedEngine(db_path=db)

    with pytest.raises(ValueError) as exc:
        edit_memory(engine, "sess-2026-01_closet_alice", project="other")

    assert "sess-2026-01" in str(exc.value)  # points at the parent
    assert _marked_ids(db) == ["sess-2026-01_closet_alice", "sess-2026-01_closet_bob"]
    assert [m.id for m in engine.list_all()] == ["sess-2026-01"]


@pytest.mark.parametrize("engine_name", ["bloom", "seed"])
def test_an_unchanged_content_write_preserves_the_marker(tmp_path: Path, engine_name: str) -> None:
    """Defence in depth at the write itself, below lifecycle's refusal.

    Only a CHANGE of the text makes a row a real memory. A re-ingest carrying the
    same body with different metadata leaves the marker where it is.
    """
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("sess-2026-01", _turns()))
    engine = _bloom(db) if engine_name == "bloom" else SeedEngine(db_path=db)
    closet = engine.get("sess-2026-01_closet_alice")

    engine.ingest(_memory("sess-2026-01_closet_alice", closet.content, related_to=["sess-2026-01"]))

    assert "sess-2026-01_closet_alice" in _marked_ids(db)
    assert [m.id for m in engine.list_all()] == ["sess-2026-01"]

    # And the parent's redaction still reaches it.
    assert forget(engine, tmp_path, "sess-2026-01").deleted is True
    assert _all_ids(db) == []


def test_a_changed_content_write_still_reclaims_the_id(tmp_path: Path) -> None:
    """The round-1 guarantee survives: a real note over a closet id is a real memory."""
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("sess-2026-01", _turns()))
    engine = _bloom(db)

    engine.ingest(_memory("sess-2026-01_closet_alice", "my own note"))

    assert _marked_ids(db) == ["sess-2026-01_closet_bob"]
    assert sorted(m.id for m in engine.list_all()) == ["sess-2026-01", "sess-2026-01_closet_alice"]


# --- round 3: a reclaimed id must keep syncing ----------------------------


def test_a_reclaimed_closet_id_still_receives_cloud_updates(tmp_path: Path) -> None:
    """Forget a copy, write a real note at that id, then pull a newer version.

    The closet tombstone must not make pull skip it: the watermark advances
    either way, so a skip loses the update permanently.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _turns()))
    store = TombstoneStore(db)
    assert forget(engine, tmp_path, "sess-2026-01_closet_alice", tombstones=store).deleted is True
    assert store.has_closet_tombstone("sess-2026-01_closet_alice") is True

    # The id is reclaimed by an independent note, which drops the stale record.
    SeedEngine(db_path=db).ingest(_memory("sess-2026-01_closet_alice", "a note of my own"))
    assert store.has_closet_tombstone("sess-2026-01_closet_alice") is False

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

    assert result.skipped_closets == 0
    assert result.applied_live == 1
    assert engine.get("sess-2026-01_closet_alice").content == "the newer version from another device"


def test_a_reclaimed_id_syncs_even_if_the_tombstone_survives(tmp_path: Path) -> None:
    """Second line: the skip requires no live local row, not just a cleared record."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _turns()))
    store = TombstoneStore(db)
    forget(engine, tmp_path, "sess-2026-01_closet_alice", tombstones=store)
    SeedEngine(db_path=db).ingest(_memory("sess-2026-01_closet_alice", "a note of my own"))
    store.add_closets(["sess-2026-01_closet_alice"])  # re-record it by hand

    result = pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[_cloud_row("sess-2026-01_closet_alice", "newer still")]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.applied_live == 1
    assert engine.get("sess-2026-01_closet_alice").content == "newer still"


# --- round 3: the fallback engine's expiry cascades ------------------------


def test_seed_purge_expired_cascades_to_the_speaker_copies(tmp_path: Path) -> None:
    """An orphan left by an expiring parent stays recallable while list_all hides it."""
    db = tmp_path / "memories.db"
    parent = _memory("sess-2026-01", _turns())
    parent.expires_at = datetime.now(timezone.utc) - timedelta(days=1)
    _bloom(db).ingest(parent)
    # Push the copies' expiry into the future, the state a pre-fix fallback edit
    # of the parent's TTL leaves behind.
    conn = sqlite3.connect(str(db))
    conn.execute(
        f"UPDATE memories SET expires_at = ? WHERE {MARKER_COLUMN} = 1",
        ((datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),),
    )
    conn.commit()
    conn.close()

    seed = SeedEngine(db_path=db)
    assert seed.purge_expired() == 1

    assert _all_ids(db) == []
    assert not seed.retrieve(SECRET, limit=10)


# --- round 3: the hardened shape test -------------------------------------


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
    # What ClosetHybridEngine.ingest writes for a pulled row: the joint-session
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

    engine = _bloom(db)

    assert _marked_ids(db) == ["sess-2026-01_closet_alice", "sess-2026-01_closet_bob"]
    assert [m.id for m in engine.list_all()] == ["sess-2026-01"]

    assert forget(engine, tmp_path, "sess-2026-01").deleted is True
    assert _all_ids(db) == []
    assert not _bloom(db).retrieve(SECRET, limit=10)


# --- round 4: the migration keeps a recoverable pre-image ------------------


def _stored_updated_at(db: Path, memory_id: str) -> str:
    return _rows(db, "SELECT updated_at FROM memories WHERE id = ?", (memory_id,))[0][0]


def _pending_ids(store: TombstoneStore) -> list[str]:
    """Just the ids from the announcement queue; the timestamp is asserted separately."""
    return [memory_id for memory_id, _ in store.pending_legacy_announcements()]


def _backup_rows(db: Path) -> dict[str, str]:
    return {r[0]: r[1] for r in _rows(db, "SELECT id, action FROM closet_migration_backup ORDER BY id")}


def test_an_inferentially_adopted_copy_is_hidden_but_intact(tmp_path: Path) -> None:
    """The likely tier: marked, so hidden and covered by the parent's redaction,
    but its text is kept and a pre-image is stored.

    Keeping the text is the point. The evidence is strong but not proof, and a
    curated split an importer wrote looks exactly like a stale copy from here —
    so it is hidden rather than rewritten, and stays recoverable either way.
    """
    db = _tier_b_store(tmp_path)

    engine = _bloom(db)
    # Only Alice's text changed, so only her copy is inferential; Bob's still
    # matches what the parent derives and is proven.
    assert _backup_rows(db) == {"sess-2026-01_closet_alice": "adopted"}
    assert "sess-2026-01_closet_alice" in _marked_ids(db)

    kept = engine.get("sess-2026-01_closet_alice")
    assert SECRET in kept.content  # not rewritten, not destroyed

    # NOT announced: an announcement deletes the cloud row everywhere, which
    # inference does not earn. Bob's proven copy is.
    assert _pending_ids(TombstoneStore(db)) == ["sess-2026-01_closet_bob"]

    # Hidden from every list, and never pushed.
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

    # The redaction promise still holds: forgetting the parent clears it.
    assert forget(engine, tmp_path, "sess-2026-01").deleted is True
    assert _all_ids(db) == []
    assert not _bloom(db).retrieve(SECRET, limit=10)


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


# --- round 4: no statement may exceed SQLite's parameter limit -------------


def test_delete_marked_closets_chunks(tmp_path: Path, monkeypatch) -> None:
    from poppy.engine import _closet_marker

    db = tmp_path / "memories.db"
    turns = json.dumps([{"speaker": f"S{i}", "text": f"line {i}"} for i in range(6)])
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", turns))
    assert len(_marked_ids(db)) == 6

    monkeypatch.setattr(_closet_marker, "SQL_PARAM_CHUNK", 2)
    assert engine.delete("sess-2026-01") is True
    assert _all_ids(db) == []


# --- round 4: a pulled closet tombstone is not a Trash entry ---------------


def test_a_pulled_closet_tombstone_never_becomes_a_restorable_entry(tmp_path: Path) -> None:
    """Another device's closet deletion, on a device that never had that copy.

    Filed as a ui tombstone it would show in Trash with the placeholder
    (`CLOSET_TOMBSTONE_CONTENT`) as its body, and restoring it would create junk
    and push it back.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    store = TombstoneStore(db)
    row = _cloud_row("sess-2026-01_closet_alice", CLOSET_TOMBSTONE_CONTENT, deleted=True)
    row["memory_type"] = "fact"

    result = pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[row]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.applied_tombstones == 0
    assert result.skipped_closets == 1
    assert store.list_all() == []  # nothing in Trash
    assert {ct.id for ct in store.list_closets()} == {"sess-2026-01_closet_alice"}


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
    assert store.list_closets() == []


def test_a_real_memory_matching_the_placeholder_still_gets_deleted(tmp_path: Path) -> None:
    """Content equality is a hint, not proof of provenance.

    A live local row means the id belongs to a real memory, so its deletion runs
    through the ordinary tombstone path however its body reads.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("mem_odd", CLOSET_TOMBSTONE_CONTENT))
    store = TombstoneStore(db)

    row = _cloud_row("mem_odd", CLOSET_TOMBSTONE_CONTENT, deleted=True)
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


# --- round 5: supersede is the twin of forget, and must be guarded too -----


def test_supersede_refuses_a_marked_closet(tmp_path: Path) -> None:
    """Superseding a copy snapshots its speaker turns into Trash and onto the wire.

    A later restore then brings them back unmarked, listed, and pushed live.
    """
    from poppy.lifecycle import supersede_memory

    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _turns()))

    with pytest.raises(ValueError) as exc:
        supersede_memory(engine, _memory("mem_new", "replacement"), "sess-2026-01_closet_alice", poppy_dir=tmp_path)

    assert "sess-2026-01" in str(exc.value)
    store = TombstoneStore(db)
    assert store.list_all() == []  # nothing snapshotted
    assert _marked_ids(db) == ["sess-2026-01_closet_alice", "sess-2026-01_closet_bob"]


def test_remember_with_supersedes_pointing_at_a_closet_is_refused(tmp_path: Path) -> None:
    """The reachable surface: `poppy remember --supersedes <copy id>` and the MCP twin."""
    from poppy.write_flow import remember

    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _turns()))

    with pytest.raises(ValueError):
        remember(engine, tmp_path, content="replacement", supersedes="sess-2026-01_closet_alice")

    assert TombstoneStore(db).list_all() == []  # no snapshot of the speaker turns
    assert _marked_ids(db) == ["sess-2026-01_closet_alice", "sess-2026-01_closet_bob"]


def test_auto_supersede_never_picks_a_derived_copy(tmp_path: Path) -> None:
    """No human types an id on this path.

    Candidates come from retrieve(), which still returns copies, and a copy
    carries the parent's project and memory_type so it passes the filters.
    """
    from poppy.capture.reconciler import find_candidates

    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _turns()))

    # The copy outranks the parent under the test reranker (shorter document),
    # so this is the ordering that would have picked it.
    ranked = [s.memory.id for s in engine.retrieve(SECRET, limit=10)]
    assert "sess-2026-01_closet_alice" in ranked

    candidates = find_candidates(engine, _memory("mem_new", SECRET), top_k=10)

    assert [c.memory.id for c in candidates] == ["sess-2026-01"]


def test_restore_refuses_an_id_that_is_now_a_derived_copy(tmp_path: Path) -> None:
    """Defence in depth for a tombstone an older build could have left behind."""
    from poppy.write_flow import restore

    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _turns()))
    store = TombstoneStore(db)
    # A tombstone naming a copy, as the unguarded supersede path used to write.
    store.add(engine.get("sess-2026-01_closet_alice"))

    with pytest.raises(ValueError):
        restore(engine, tmp_path, "sess-2026-01_closet_alice", tombstones=store)

    assert "sess-2026-01_closet_alice" in _marked_ids(db)
    assert [m.id for m in engine.list_all()] == ["sess-2026-01"]


# --- round 5: a present parent's created_at gates the purge ---------------


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


# --- round 5: a copy deletion must not strand a newer recreation ----------


def test_a_newer_cloud_row_at_a_deleted_copys_id_is_ingested(tmp_path: Path) -> None:
    """Device B writes an independent note at an id device A deleted as a copy.

    Skipping it loses the note for good: the watermark advances either way.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _turns()))
    store = TombstoneStore(db)
    forget(engine, tmp_path, "sess-2026-01_closet_alice", tombstones=store)
    assert store.has_closet_tombstone("sess-2026-01_closet_alice") is True

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

    assert result.skipped_closets == 0
    assert result.applied_live == 1
    assert engine.get("sess-2026-01_closet_alice").content == "an independent note from device B"
    assert _marked_ids(db) == ["sess-2026-01_closet_bob"]
    # Reclaiming the id drops the deletion record, so it cannot suppress again.
    assert store.has_closet_tombstone("sess-2026-01_closet_alice") is False


def test_an_older_cloud_copy_at_a_deleted_copys_id_is_still_skipped(tmp_path: Path) -> None:
    """The round-2 guarantee survives the freshness rule."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _turns()))
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

    assert result.skipped_closets == 1
    assert engine.get("sess-2026-01_closet_alice") is None


# --- round 5: the local side tables age out without a dashboard visit -----


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
    assert len(store.list_closets()) == 2

    old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE closet_tombstones SET tombstoned_at = ?", (old,))
    conn.execute("UPDATE ui_tombstones SET tombstoned_at = ?", (old,))
    conn.commit()
    conn.close()

    save(tmp_path, synced_state())  # only a SENT deletion ages out
    run_sync(engine=engine, tombstones=store, client=_RecordingClient(), poppy_dir=tmp_path)

    assert store.list_closets() == []
    assert store.list_all() == []


# --- round 6: a deletion must reach the cloud before it is purged ----------


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

    old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE closet_tombstones SET tombstoned_at = ?", (old,))
    conn.commit()
    conn.close()

    # No push watermark: nothing has been sent, so nothing may be dropped.
    assert store.purge_expired(pushed_through=None) == 0
    assert len(store.list_closets()) == 2

    # Once a push has covered them, they age out.
    store.purge_expired(pushed_through=datetime.now(timezone.utc).isoformat())
    assert store.list_closets() == []


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


# --- round 6: a pulled deletion keeps its own timestamp -------------------


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
        client=_RecordingClient(
            rows=[_cloud_row("sess-1_closet_alice", CLOSET_TOMBSTONE_CONTENT, deleted=True, when=deleted_at)]
        ),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    recorded = store.get_closet("sess-1_closet_alice")
    assert recorded is not None
    assert recorded.tombstoned_at == deleted_at  # the deletion's time, not now

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
    row = _cloud_row("sess-1_closet_alice", CLOSET_TOMBSTONE_CONTENT, deleted=True, when=deleted_at)

    for _ in range(3):
        pull(
            engine=engine,
            tombstones=store,
            client=_RecordingClient(rows=[row]),
            state=SyncState(),
            poppy_dir=tmp_path,
        )

    assert store.get_closet("sess-1_closet_alice").tombstoned_at == deleted_at


# --- round 6: an older client can still write unmarked copies -------------


def test_copies_written_by_an_older_client_are_reported_not_repaired(tmp_path: Path) -> None:
    """Mixed installs are diagnosed, never silently reclassified.

    Marking them means running the migration's inference outside the migration,
    and every version of that has destroyed rows that merely arrived from the
    cloud. `poppy doctor` counts them instead; the repair is a separate,
    explicit command.
    """
    from poppy.cli.main import _unmarked_copies_count

    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _turns()))
    conn = sqlite3.connect(str(db))
    conn.execute(f"UPDATE memories SET {MARKER_COLUMN} = 0")  # what an older client leaves
    conn.commit()
    conn.close()

    assert _unmarked_copies_count(tmp_path) == 2

    # Reopening never repairs them, and never writes.
    for _ in range(3):
        _bloom(db)
    assert _marked_ids(db) == []
    assert _backup_rows(db) == {}
    assert TombstoneStore(db).list_closets() == []


def test_the_unmarked_count_ignores_a_real_lookalike(tmp_path: Path) -> None:
    """Only rows a live parent re-derives are counted, so nothing real is reported."""
    from poppy.cli.main import _unmarked_copies_count

    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("mem_customer", "a plain memory"))
    engine.ingest(_memory("mem_customer_closet_notes", "the wardrobe budget for Q3"))

    assert _unmarked_copies_count(tmp_path) == 0


def test_reopening_a_store_with_a_lookalike_never_writes(tmp_path: Path) -> None:
    """`poppy list` and `recall` both open the engine; neither may take a write lock."""
    import poppy.engine._closet_marker as marker

    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("mem_customer", "a plain memory"))
    engine.ingest(_memory("mem_customer_closet_notes", "the wardrobe budget for Q3"))

    calls: list[str] = []
    real_apply = marker._apply_backfill
    marker._apply_backfill = lambda conn, plan: (calls.append("apply"), real_apply(conn, plan))[1]
    try:
        reopened = _bloom(db)
    finally:
        marker._apply_backfill = real_apply

    assert calls == []
    assert reopened._conn.in_transaction is False
    assert _marked_ids(db) == []


# --- round 7: a cascaded copy deletion carries the parent deletion's time --


def test_a_cascaded_copy_deletion_uses_the_parent_deletions_timestamp(tmp_path: Path) -> None:
    """A July 3 note at a copy's id must survive a July 2 parent deletion.

    Stamping the cascade with receipt time (September) made the deletion look
    newer than the note, so the note was skipped and a deletion pushed for it.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    july2 = datetime(2026, 7, 2, 12, 0, 0, tzinfo=timezone.utc)
    # Older than the incoming deletion, or pull would keep the local row as the
    # fresher state and never cascade at all.
    parent = _memory("sess-1", _turns())
    parent.created_at = parent.updated_at = july2 - timedelta(days=1)
    parent.source.timestamp = july2 - timedelta(days=1)
    engine.ingest(parent)
    store = TombstoneStore(db)

    parent_deletion = _cloud_row("sess-1", "the parent body", deleted=True, when=july2)
    pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[parent_deletion]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    recorded = store.get_closet("sess-1_closet_alice")
    assert recorded is not None
    assert recorded.tombstoned_at == july2  # not "now"

    note = _cloud_row("sess-1_closet_alice", "an independent note", when=july2 + timedelta(days=1))
    result = pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[note]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.skipped_closets == 0
    assert result.applied_live == 1
    assert engine.get("sess-1_closet_alice").content == "an independent note"


# --- round 7: one bad supersede target must not drop a whole capture batch -


def test_a_bad_supersede_target_skips_one_capture_not_the_batch(tmp_path: Path) -> None:
    """Auto-supersede can still name an unsupersedable target through other paths.

    An exception mid-loop would drop every remaining capture in the batch.
    """
    import poppy.capture.reconciler as reconciler_mod
    from poppy.capture.reconciler import Action, Decision, reconcile_and_ingest
    from poppy.config import PoppyConfig

    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _turns()))

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


# --- round 8: the doctor lines actually render ----------------------------


def test_doctor_reports_the_migration_backup(tmp_path: Path) -> None:
    """The line crashed on every store that had backup rows: `import datetime`
    in the CLI is the MODULE, so `datetime.fromisoformat` was an AttributeError
    that `safe_read` did not catch."""
    from poppy.cli.main import _closet_backup_status

    db = _tier_b_store(tmp_path)
    _bloom(db)  # the migration snapshots the rows it adopts on inference
    assert _backup_rows(db)

    rows, deadline = _closet_backup_status(tmp_path)

    assert rows == 1
    assert deadline is not None and len(deadline) == 10  # an ISO date


def test_doctor_reads_an_encrypted_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Through poppy.db.connect, not raw sqlite3.

    Raw sqlite3 cannot read a SQLCipher file, so both doctor lines silently
    reported nothing on exactly the installs that most need them.
    """
    pytest.importorskip("sqlcipher3")
    from poppy import encryption
    from poppy.cli.main import _closet_backup_status

    # Built inline and closed as we go: encryption refuses to migrate while any
    # connection still holds the store's gate lock.
    db = tmp_path / "memories.db"
    writer = _bloom(db)
    writer.ingest(_memory("sess-2026-01", _turns()))
    writer._conn.close()
    _strip_marker(db)
    # Drift the parent so one copy lands on the inferential tier, which is the
    # only one that snapshots.
    conn = sqlite3.connect(str(db))
    conn.execute(
        "UPDATE memories SET content = ?, enriched_content = ? WHERE id = ?",
        (_turns("harmlessreplacement"), "x", "sess-2026-01"),
    )
    conn.commit()
    conn.close()
    migrated = _bloom(db)  # the migration snapshots the row it adopts on inference
    migrated._conn.close()

    monkeypatch.setenv("POPPY_DB_KEY", encryption.generate_key())
    encryption.enable(tmp_path)

    # A raw connection cannot see the table at all — this is the bug.
    raw = sqlite3.connect(str(db))
    try:
        with pytest.raises(sqlite3.DatabaseError):
            raw.execute("SELECT COUNT(*) FROM closet_migration_backup").fetchone()
    finally:
        raw.close()

    assert _closet_backup_status(tmp_path)[0] == 1


def test_doctor_helpers_are_quiet_on_a_store_that_does_not_exist(tmp_path: Path) -> None:
    from poppy.cli.main import _closet_backup_status, _unmarked_copies_count

    assert _closet_backup_status(tmp_path) == (0, None)
    assert _unmarked_copies_count(tmp_path) == 0


# --- round 8: only pre-fix copies are announced to the cloud --------------


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

    assert len(store.list_closets()) == 2  # recorded locally, for the pull skip
    assert _pending_ids(store) == []  # but nothing is announced

    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=synced_state(), poppy_dir=tmp_path)
    assert {r["id"] for r in client.upserts} == {"meeting"}


def test_a_legacy_store_announces_exactly_the_pre_fix_ids_once(tmp_path: Path) -> None:
    """A copy the migration marked may have been leaked by a 0.2.4 client."""
    db = _legacy_store(tmp_path)
    engine = _bloom(db)  # the one-time migration records the legacy ids
    legacy = {"sess-2026-01_closet_alice", "sess-2026-01_closet_bob"}
    assert {r[0] for r in _rows(db, "SELECT id FROM legacy_closet_ids")} == legacy

    store = TombstoneStore(db)
    assert set(_pending_ids(store)) == legacy

    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)

    announced = {r["id"] for r in client.upserts if r["id"] in legacy}
    assert announced == legacy
    assert all(r["deleted_at"] is not None for r in client.upserts if r["id"] in legacy)
    assert not any(SECRET in json.dumps(r) for r in client.upserts if r["id"] in legacy)

    # Sent once: the pending flag is cleared only on a 2xx, and a second push
    # after that sends nothing more.
    assert _pending_ids(store) == []
    again = _RecordingClient()
    push(engine=engine, tombstones=store, client=again, state=SyncState(), poppy_dir=tmp_path)
    assert {r["id"] for r in again.upserts} & legacy == set()


# --- round 8: a rebuild restores the whole row, not just the body ---------


def test_a_rebuild_restores_the_parents_project_not_just_its_text(tmp_path: Path) -> None:
    """A copy inherits its parent's project at creation.

    A pre-fix fallback edit that moved the parent to a private project left the
    copy under the old public one, so refreshing only the body would keep the
    parent's private turns retrievable under the stale public filter.
    """
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("meeting", _turns(), project="public"))
    _strip_marker(db)
    # What the pre-fix fallback edit leaves: parent moved to private, copies not.
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE memories SET project = 'private', memory_type = 'note' WHERE id = 'meeting'")
    conn.commit()
    conn.close()

    engine = _bloom(db)

    copy = engine.get("meeting_closet_alice")
    assert copy.project == "private"
    assert copy.memory_type == "note"
    assert engine.retrieve(SECRET, filters=Filters(project="public"), limit=10) == []
    assert {r.memory.id for r in engine.retrieve(SECRET, filters=Filters(project="private"), limit=10)} >= {
        "meeting_closet_alice"
    }


# --- round 9: the announcement lane is independent of the local record -----


def test_a_pulled_parent_deletion_still_announces_both_leaked_copies(tmp_path: Path) -> None:
    """Device B, migrated from a legacy store, pulls A's deletion of the parent.

    The announcement used to ride on the local deletion record: the cascade wrote
    it, then the re-stamp rewrote it, and the leaked flag was lost — so the
    copies were never announced and stayed live in the cloud.
    """
    db = _legacy_store(tmp_path)
    engine = _bloom(db)  # the migration queues both ids
    legacy = {"sess-2026-01_closet_alice", "sess-2026-01_closet_bob"}
    store = TombstoneStore(db)
    assert set(_pending_ids(store)) == legacy

    # A's deletion of the parent arrives, newer than the local row.
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE memories SET updated_at = ? WHERE id = 'sess-2026-01'", ("2026-07-01T00:00:00+00:00",))
    conn.commit()
    conn.close()
    engine = _bloom(db)
    july2 = datetime(2026, 7, 2, 12, 0, 0, tzinfo=timezone.utc)
    pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[_cloud_row("sess-2026-01", "the parent body", deleted=True, when=july2)]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )
    assert _all_ids(db) == []

    # The local records keep event time for the pull skip...
    assert store.get_closet("sess-2026-01_closet_alice").tombstoned_at == july2
    # ... and the announcements are still pending, on their own lane.
    assert set(_pending_ids(store)) == legacy

    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert legacy <= {r["id"] for r in client.upserts}
    assert _pending_ids(store) == []


def test_an_announcement_survives_a_purge_of_the_local_record(tmp_path: Path) -> None:
    """A cascaded local record is stamped with the remote deletion time, which can
    be old enough to be purged on the first sync. The announcement must not go
    with it."""
    db = _legacy_store(tmp_path)
    _bloom(db)
    store = TombstoneStore(db)
    legacy = set(_pending_ids(store))
    assert len(legacy) == 2

    old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE closet_tombstones SET tombstoned_at = ?", (old,))
    conn.commit()
    conn.close()
    store.purge_expired(pushed_through=datetime.now(timezone.utc).isoformat())

    assert store.list_closets() == []  # local records aged out
    assert set(_pending_ids(store)) == legacy  # the queue did not


def test_an_edit_that_keeps_both_speakers_still_announces_the_leaked_copies(tmp_path: Path) -> None:
    """Redacting Alice's line while keeping both speakers rebuilds both copies.

    No local deletion happens, so nothing on the deletion path would ever have
    announced them — and the cloud kept the pre-fix copy of the old secret.
    """
    db = _legacy_store(tmp_path)
    engine = _bloom(db)
    store = TombstoneStore(db)
    legacy = set(_pending_ids(store))
    assert len(legacy) == 2

    scrubbed = json.dumps(
        [{"speaker": "Alice", "dia_id": "D1", "text": "redacted"}, {"speaker": "Bob", "dia_id": "D2", "text": "noted"}]
    )
    engine.ingest(_memory("sess-2026-01", scrubbed))
    assert not [r for r in engine.retrieve(SECRET, limit=10) if SECRET in r.memory.content]

    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert legacy <= {r["id"] for r in client.upserts}


def test_a_failed_announcement_stays_pending(tmp_path: Path) -> None:
    """Cleared only on a 2xx, so an offline or rejected sync retries it."""

    class _FailingClient(_RecordingClient):
        def upsert(self, row: dict) -> None:
            if row["id"].endswith("_closet_alice"):
                raise RuntimeError("boom")
            super().upsert(row)

    db = _legacy_store(tmp_path)
    engine = _bloom(db)
    store = TombstoneStore(db)

    push(engine=engine, tombstones=store, client=_FailingClient(), state=SyncState(), poppy_dir=tmp_path)

    assert _pending_ids(store) == ["sess-2026-01_closet_alice"]

    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert "sess-2026-01_closet_alice" in {r["id"] for r in client.upserts}
    assert _pending_ids(store) == []


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

    store.add_closets(["x_closet_alice"], now=early)
    store.add_closets(["x_closet_alice"], now=late)

    assert store.get_closet("x_closet_alice").tombstoned_at == early


# --- round 9: derived copies are not memories the user has ----------------


@pytest.mark.parametrize("engine_name", ["bloom", "seed"])
def test_stats_counts_memories_not_derived_copies(tmp_path: Path, engine_name: str) -> None:
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("sess-2026-01", _turns()))
    engine = _bloom(db) if engine_name == "bloom" else SeedEngine(db_path=db)

    assert engine.stats().memory_count == len(engine.list_all(limit=1000))
    assert engine.stats().memory_count == 1


def test_an_unchanged_reingest_keeps_the_copys_back_reference(tmp_path: Path) -> None:
    """The caller's Memory usually has an empty related_to; the stored link stays."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-2026-01", _turns()))
    copy = engine.get("sess-2026-01_closet_alice")
    assert copy.related_to == ["sess-2026-01"]

    engine.ingest(_memory("sess-2026-01_closet_alice", copy.content))  # related_to=[]

    assert engine.get("sess-2026-01_closet_alice").related_to == ["sess-2026-01"]
    assert "sess-2026-01_closet_alice" in _marked_ids(db)


# --- round 10: the announcement lane is bounded ---------------------------


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


def test_the_announcement_loop_has_its_own_breaker(tmp_path: Path) -> None:
    """Reached when the main loop had nothing to send, so it never tripped."""

    class _LiveThenDead(_RecordingClient):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        def upsert(self, row: dict) -> None:
            self.attempts += 1
            raise TimeoutError("no route to host")

    db = tmp_path / "memories.db"
    engine = _bloom(db)  # nothing local to push
    store = TombstoneStore(db)
    _queue_legacy(db, [f"parent_closet_s{i}" for i in range(50)])

    client = _LiveThenDead()
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)

    assert client.attempts == MAX_CONSECUTIVE_TRANSPORT_FAILURES
    assert len(_pending_ids(store)) == 50


def test_a_large_backlog_drains_over_rounds(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    store = TombstoneStore(db)
    total = MAX_LEGACY_ANNOUNCEMENTS_PER_SYNC + 300
    _queue_legacy(db, [f"parent_closet_s{i:04d}" for i in range(total)])

    first = _RecordingClient()
    push(engine=engine, tombstones=store, client=first, state=SyncState(), poppy_dir=tmp_path)

    assert len(first.upserts) == MAX_LEGACY_ANNOUNCEMENTS_PER_SYNC
    assert len(_pending_ids(store)) == 300

    # The cap applies EVERY round, so a 500-id backlog takes three.
    second = _RecordingClient()
    push(engine=engine, tombstones=store, client=second, state=SyncState(), poppy_dir=tmp_path)
    assert len(second.upserts) == MAX_LEGACY_ANNOUNCEMENTS_PER_SYNC
    assert len(_pending_ids(store)) == 100

    third = _RecordingClient()
    push(engine=engine, tombstones=store, client=third, state=SyncState(), poppy_dir=tmp_path)
    assert len(third.upserts) == 100
    assert _pending_ids(store) == []


def test_a_refused_announcement_does_not_stop_the_rest(tmp_path: Path) -> None:
    """A server response means this id is refused, not that the host is gone."""
    from poppy.sync.client import TragsError

    class _RefusesOne(_RecordingClient):
        def upsert(self, row: dict) -> None:
            if row["id"] == "parent_closet_s1":
                raise TragsError("409 conflict")
            super().upsert(row)

    db = tmp_path / "memories.db"
    engine = _bloom(db)
    store = TombstoneStore(db)
    _queue_legacy(db, [f"parent_closet_s{i}" for i in range(4)])

    push(engine=engine, tombstones=store, client=_RefusesOne(), state=SyncState(), poppy_dir=tmp_path)

    assert _pending_ids(store) == ["parent_closet_s1"]


# --- round 10: reclaiming an id cancels its announcement ------------------


def test_reclaiming_an_id_cancels_its_pending_announcement(tmp_path: Path) -> None:
    """Forget the parent, write an independent note at a copy's id, then sync.

    Left pending, push sent the note live and then a NEWER deletion for the same
    id, and the next pull removed the note the user had just written.
    """
    db = _legacy_store(tmp_path)
    engine = _bloom(db)
    store = TombstoneStore(db)
    assert "sess-2026-01_closet_alice" in _pending_ids(store)

    forget(engine, tmp_path, "sess-2026-01", tombstones=store)
    engine.ingest(_memory("sess-2026-01_closet_alice", "an independent note of my own"))

    assert _pending_ids(store) == ["sess-2026-01_closet_bob"]
    assert store.get_closet("sess-2026-01_closet_alice") is None

    client = _RecordingClient(echo=True)
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)

    for_id = [r for r in client.upserts if r["id"] == "sess-2026-01_closet_alice"]
    assert len(for_id) == 1
    assert for_id[0]["deleted_at"] is None  # live, not a deletion
    assert for_id[0]["content"] == "an independent note of my own"

    # And it survives the round trip. (Bob's announcement comes back too and is
    # recognised as one of ours, which is why skipped_closets is 1 and not 0.)
    result = pull(
        engine=engine,
        tombstones=store,
        client=client,
        state=SyncState(),
        poppy_dir=tmp_path,
    )
    assert result.skipped_closets == 1
    assert "sess-2026-01_closet_bob" in {ct.id for ct in store.list_closets()}
    assert engine.get("sess-2026-01_closet_alice").content == "an independent note of my own"


# --- round 11: synthesis never overwrites a real memory --------------------


def test_synthesis_leaves_a_real_memory_holding_a_derived_id(tmp_path: Path) -> None:
    """Nothing reserves the derived id space, and ids arrive from outside Poppy.

    Overwriting here replaced the user's memory with a speaker projection,
    marked it, hid it from every list, and let a later redaction of the parent
    delete it with no tombstone and no way back.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    # An external id, as sync pull / the web API / an importer can supply.
    engine.ingest(_memory("sess-1_closet_alice", "MY OWN NOTE, NOT A COPY"))

    engine.ingest(_memory("sess-1", _turns()))  # would derive that exact id

    survivor = engine.get("sess-1_closet_alice")
    assert survivor.content == "MY OWN NOTE, NOT A COPY"
    assert _marked_ids(db) == ["sess-1_closet_bob"]  # only the free id was derived
    assert "sess-1_closet_alice" in {m.id for m in engine.list_all()}


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
    assert {ct.id for ct in store.list_closets()} == {"sess-1_closet_bob"}
    assert store.get_closet("sess-1_closet_alice") is None


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


# --- round 11: regenerating a copy drops its stale deletion record ---------


def test_a_restore_clears_the_earlier_copy_deletion_record(tmp_path: Path) -> None:
    """forget 12:00, restore 12:01, forget 12:02.

    MIN kept 12:00, so a leaked cloud copy stamped 12:01 was NEWER than the
    recorded deletion and passed the freshness skip — ingested live, with the
    secret, after the user had forgotten the memory twice.
    """
    from poppy.write_flow import restore

    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("sess-1", _turns()))
    store = TombstoneStore(db)

    noon = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    forget(engine, tmp_path, "sess-1", tombstones=store)
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE closet_tombstones SET tombstoned_at = ?", (noon.isoformat(),))
    conn.commit()
    conn.close()

    # 12:01 — restored, so the copies are derived again and the record is stale.
    restore(engine, tmp_path, "sess-1", tombstones=store)
    assert store.get_closet("sess-1_closet_alice") is None
    assert "sess-1_closet_alice" in _marked_ids(db)

    # 12:02 — forgotten again; the record now carries the LATER deletion.
    forget(engine, tmp_path, "sess-1", tombstones=store)
    recorded = store.get_closet("sess-1_closet_alice")
    assert recorded is not None and recorded.tombstoned_at > noon

    # A leaked cloud copy stamped 12:01 is older than that, so it is skipped.
    leaked = _cloud_row(
        "sess-1_closet_alice",
        f'[{{"speaker":"Alice","text":"{SECRET}"}}]',
        when=noon + timedelta(minutes=1),
    )
    result = pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[leaked]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.skipped_closets == 1
    assert engine.get("sess-1_closet_alice") is None
    assert not _bloom(db).retrieve(SECRET, limit=10)


def test_regenerating_a_copy_keeps_its_pending_announcement(tmp_path: Path) -> None:
    """A restore does not un-leak what an older client already pushed."""
    db = _legacy_store(tmp_path)
    engine = _bloom(db)
    store = TombstoneStore(db)
    legacy = set(_pending_ids(store))
    assert len(legacy) == 2

    engine.ingest(_memory("sess-2026-01", _turns()))  # re-derives both copies

    assert set(_pending_ids(store)) == legacy
    assert store.list_closets() == []  # but the local records are gone


# --- round 12: the announcement claims the row it actually leaked ----------


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


def test_an_announcement_does_not_destroy_a_note_another_device_wrote(tmp_path: Path) -> None:
    """A holds a genuine legacy copy; B reclaims that id and uploads a note.

    Local migration provenance says the id was a copy HERE. It does not say we
    still own the row up there. Stamped now, A's cleanup beat B's newer note and
    deleted it for everyone.
    """
    db = _legacy_store(tmp_path)
    engine = _bloom(db)  # migration captures the copy's own updated_at
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

    # The announcement was sent and answered, so it is consumed and never retried.
    assert "sess-2026-01_closet_alice" in {r["id"] for r in cloud.upserts}
    assert _pending_ids(store) == []
    # But it lost the freshness comparison, so B's note is untouched.
    assert "sess-2026-01_closet_alice" in cloud.ignored
    assert cloud.state["sess-2026-01_closet_alice"]["content"] == "B'S INDEPENDENT NOTE"
    assert cloud.state["sess-2026-01_closet_alice"]["deleted_at"] is None


def test_an_untouched_leaked_copy_is_deleted_in_the_cloud(tmp_path: Path) -> None:
    """The case the announcement exists for: nobody touched the leaked row."""
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

    assert cloud.ignored == []
    for cid in leaked:
        assert cloud.state[cid]["deleted_at"] is not None
        assert cloud.state[cid]["content"] == CLOSET_TOMBSTONE_CONTENT
    assert _pending_ids(store) == []


def test_the_announcement_carries_the_copys_pre_migration_timestamp(tmp_path: Path) -> None:
    """Captured BEFORE the rebuild, which overwrites it with the parent's."""
    db = _legacy_store(tmp_path)
    # A pre-fix fallback edit moved the PARENT's updated_at and left the copies
    # alone — which is exactly when capturing late would read the wrong value.
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE memories SET updated_at = ? WHERE id = 'sess-2026-01'", ("2027-01-01T00:00:00+00:00",))
    conn.commit()
    conn.close()
    before = _stored_updated_at(db, "sess-2026-01_closet_alice")
    assert before != "2027-01-01T00:00:00+00:00"

    engine = _bloom(db)
    # The rebuild wrote the parent's current updated_at onto the copy.
    assert _stored_updated_at(db, "sess-2026-01_closet_alice") == "2027-01-01T00:00:00+00:00"

    store = TombstoneStore(db)
    queued = dict(store.pending_legacy_announcements())
    assert queued["sess-2026-01_closet_alice"] == before

    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    sent = next(r for r in client.upserts if r["id"] == "sess-2026-01_closet_alice")
    assert sent["updated_at"] == before
    assert sent["deleted_at"] == before


# --- round 13: a fresh install pulling a 0.2.4 account --------------------


def _leaked_pair(when: datetime) -> tuple[dict, dict]:
    """What a <=0.2.4 client pushed: the parent AND its per-speaker copies, live."""
    parent = _cloud_row("p", _turns(), when=when)
    parent["created_at"] = when.isoformat()
    copy = _cloud_row("p_closet_alice", json.dumps([{"speaker": "Alice", "dia_id": "D1", "text": SECRET}]), when=when)
    copy["created_at"] = when.isoformat()
    copy["related_to"] = ["p"]
    return parent, copy


@pytest.mark.parametrize("copy_first", [True, False])
def test_a_pulled_leaked_copy_is_adopted_in_either_arrival_order(tmp_path: Path, copy_first: bool) -> None:
    """A fresh 0.3.x install pulling an account a 0.2.4 client synced.

    The store is created with the marker column, so the one-time migration never
    runs and the copy arrives unmarked. If it lands BEFORE its parent, refusing
    to derive at that id left it unmarked for ever: listed, synced, and untouched
    by any redaction of the parent.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    store = TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=1)
    parent, copy = _leaked_pair(when)
    rows = [copy, parent] if copy_first else [parent, copy]

    for row in rows:  # separate pulls, so the order is really the variable
        pull(
            engine=engine,
            tombstones=store,
            client=_RecordingClient(rows=[row]),
            state=SyncState(),
            poppy_dir=tmp_path,
        )

    # Either way: marked, out of every list, and queued as a leak the cloud owes.
    assert "p_closet_alice" in _marked_ids(db)
    assert [m.id for m in engine.list_all()] == ["p"]
    assert "p_closet_alice" in _pending_ids(store)
    # Nothing is snapshotted either way: the pulled copy's text is exactly what
    # the parent derives, so adopting it changes no memory content.
    assert _backup_rows(db) == {}

    # And the parent's redaction now reaches it.
    assert forget(engine, tmp_path, "p", tombstones=store).deleted is True
    assert _all_ids(db) == []
    assert not _bloom(db).retrieve(SECRET, limit=10)

    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert [r for r in client.upserts if r["deleted_at"] is None] == []
    # The copy's own rows carry nothing. (The PARENT's tombstone does snapshot
    # its content for the 7-day restore window, which this fix does not change.)
    assert not any(SECRET in json.dumps(r) for r in client.upserts if r["id"] != "p")


def test_an_adopted_copy_is_announced_once_with_its_arrival_timestamp(tmp_path: Path) -> None:
    """It came down from the cloud, so the cloud still holds it and owes a delete."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    store = TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=1)
    parent, copy = _leaked_pair(when)
    pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[copy, parent]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert dict(store.pending_legacy_announcements())["p_closet_alice"] == when.isoformat()

    cloud = _FreshnessClient(rows=[copy])
    push(engine=engine, tombstones=store, client=cloud, state=SyncState(), poppy_dir=tmp_path)

    assert cloud.ignored == []  # equal timestamp, so the delete applies
    assert cloud.state["p_closet_alice"]["deleted_at"] is not None
    assert _pending_ids(store) == []


def test_a_real_lookalike_still_wins_the_id_against_adoption(tmp_path: Path) -> None:
    """Adoption is gated on the same conjunctive test, so prose is never adopted."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("p_closet_alice", "MY OWN NOTE, NOT A COPY"))

    engine.ingest(_memory("p", _turns()))

    assert engine.get("p_closet_alice").content == "MY OWN NOTE, NOT A COPY"
    assert _marked_ids(db) == ["p_closet_bob"]
    assert _pending_ids(TombstoneStore(db)) == []
    assert _backup_rows(db) == {}


# --- round 13: an equal-timestamp deletion wins ---------------------------


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
    cleanup["content"] = CLOSET_TOMBSTONE_CONTENT
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


# --- round 14: evidence is graded, and each grade earns only what it proves --


def test_tier_proven_marks_aligns_and_announces(tmp_path: Path) -> None:
    """Text IS what the parent derives: not an inference, so it may be announced."""
    db = _legacy_store(tmp_path)
    engine = _bloom(db)

    assert _marked_ids(db) == ["sess-2026-01_closet_alice", "sess-2026-01_closet_bob"]
    assert _backup_rows(db) == {}  # nothing was rewritten, so nothing to snapshot
    assert set(_pending_ids(TombstoneStore(db))) == {
        "sess-2026-01_closet_alice",
        "sess-2026-01_closet_bob",
    }
    assert [m.id for m in engine.list_all()] == ["sess-2026-01"]


def test_tier_likely_marks_but_never_rewrites_or_announces(tmp_path: Path) -> None:
    """L2's repro: a curated split under a derived id keeps its text.

    An importer can legitimately write one, and from here it is indistinguishable
    from a copy left stale by the pre-fix edit bug. Hidden, so the redaction
    promise holds; kept, so nothing is destroyed if the guess is wrong.
    """
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("p", _turns()))
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


def test_doctor_reports_the_two_tiers_that_need_a_human(tmp_path: Path) -> None:
    from poppy.cli.main import _closet_repair_counts

    db = _tier_b_store(tmp_path)
    _bloom(db)
    adopted, orphans = _closet_repair_counts(tmp_path)
    assert (adopted, orphans) == (1, 0)

    other = tmp_path / "orphan"
    other.mkdir()
    odb = _legacy_store(other)
    conn = sqlite3.connect(str(odb))
    conn.execute("DELETE FROM memories WHERE id = ?", ("sess-2026-01",))
    conn.commit()
    conn.close()
    _bloom(odb)
    assert _closet_repair_counts(other) == (0, 2)


# --- round 14: a re-pushed leak refreshes, a reclaimed id cancels ----------


def test_a_newer_sighting_of_our_leak_refreshes_the_announcement(tmp_path: Path) -> None:
    """An old client re-pushing the same copy moves the cloud row's timestamp.

    A queued announcement stamped with the older sighting loses the freshness
    comparison, is answered `stale_ignored`, and the queue clears with the text
    still live up there.
    """
    db = _legacy_store(tmp_path)
    engine = _bloom(db)
    store = TombstoneStore(db)
    first = dict(store.pending_legacy_announcements())["sess-2026-01_closet_alice"]

    later = datetime.now(timezone.utc) + timedelta(days=1)
    revision = _cloud_row(
        "sess-2026-01_closet_alice",
        engine.get("sess-2026-01_closet_alice").content,  # still our derived text
        when=later,
    )
    revision["related_to"] = ["sess-2026-01"]
    revision["created_at"] = _stored_updated_at(db, "sess-2026-01_closet_alice")
    conn = sqlite3.connect(str(db))
    revision["created_at"] = conn.execute(
        "SELECT created_at FROM memories WHERE id = 'sess-2026-01_closet_alice'"
    ).fetchone()[0]
    conn.close()

    pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[revision]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    refreshed = dict(store.pending_legacy_announcements())["sess-2026-01_closet_alice"]
    assert refreshed == later.isoformat()
    assert refreshed > first


def test_a_reclaimed_id_seen_on_pull_cancels_the_announcement(tmp_path: Path) -> None:
    """Someone else owns the id now, so announcing would delete THEIR memory."""
    db = _legacy_store(tmp_path)
    engine = _bloom(db)
    store = TombstoneStore(db)
    assert "sess-2026-01_closet_alice" in _pending_ids(store)

    theirs = _cloud_row(
        "sess-2026-01_closet_alice",
        "SOMEONE ELSE'S REAL MEMORY",
        when=datetime.now(timezone.utc) + timedelta(days=1),
    )
    pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[theirs]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert _pending_ids(store) == ["sess-2026-01_closet_bob"]
    # The local copy is untouched; this was only ever about the cloud.
    assert "sess-2026-01_closet_alice" in _marked_ids(db)


# --- round 15: proof needs provenance, not just matching text -------------


def test_an_independent_record_holding_the_same_turns_is_not_a_copy(tmp_path: Path) -> None:
    """Matching text strengthens the evidence; it never replaces provenance.

    An independent record can hold exactly the turns a memory projects — an
    importer split, a research fixture — while pointing at its own source and
    predating that memory. Graded on text alone it was called PROVEN and its
    cloud row announced for deletion.
    """
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("p", _turns()))
    projected = _rows(db, "SELECT content FROM memories WHERE id = 'p_closet_alice'")[0][0]
    _strip_marker(db)
    conn = sqlite3.connect(str(db))
    conn.execute(
        "UPDATE memories SET related_to = ?, created_at = ?, project = 'elsewhere' WHERE id = ?",
        (json.dumps(["independent-record"]), "2026-01-01T00:00:00+00:00", "p_closet_alice"),
    )
    conn.commit()
    conn.close()

    engine = _bloom(db)

    assert engine.get("p_closet_alice").content == projected  # untouched
    assert _marked_ids(db) == ["p_closet_bob"]
    assert "p_closet_alice" in {m.id for m in engine.list_all()}
    assert _pending_ids(TombstoneStore(db)) == ["p_closet_bob"]


# --- round 15: a stale leak is still our leak -----------------------------


def test_an_edited_parent_still_deletes_the_untouched_cloud_copy(tmp_path: Path) -> None:
    """Editing a memory while keeping every speaker leaves the cloud copy stale.

    It holds exactly what the edit was meant to remove. The announcement queued
    at migration carries that copy's own timestamp, so as long as nobody has
    touched the cloud row the two tie and the delete lands.
    """
    db = _legacy_store(tmp_path)
    engine = _bloom(db)
    store = TombstoneStore(db)
    old_cloud_text = engine.get("sess-2026-01_closet_alice").content
    leaked_at = dict(store.pending_legacy_announcements())["sess-2026-01_closet_alice"]
    assert SECRET in old_cloud_text

    # Redact Alice's line, both speakers retained, so the copy is re-derived.
    # Through lifecycle.edit_memory, which is what a real edit uses: it preserves
    # created_at, and that field is part of the provenance the grading reads.
    from poppy.lifecycle import edit_memory

    edit_memory(engine, "sess-2026-01", content=_turns("harmlessreplacement"))
    assert SECRET not in engine.get("sess-2026-01_closet_alice").content

    # The cloud still holds the pre-edit copy, untouched since it was leaked.
    untouched = _cloud_row("sess-2026-01_closet_alice", old_cloud_text)
    untouched["related_to"] = ["sess-2026-01"]
    untouched["created_at"] = _rows(db, "SELECT created_at FROM memories WHERE id = 'sess-2026-01_closet_alice'")[0][0]
    untouched["updated_at"] = leaked_at

    pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[untouched]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    # Seeing it does not move the claim — its text is no longer our derivation,
    # so nothing here PROVES the cloud row is our leak rather than someone
    # else's edit. The claim queued at migration is what deletes it.
    queued = dict(store.pending_legacy_announcements())
    assert queued["sess-2026-01_closet_alice"] == leaked_at

    cloud = _FreshnessClient(rows=[untouched])
    push(engine=engine, tombstones=store, client=cloud, state=SyncState(), poppy_dir=tmp_path)
    assert cloud.ignored == []
    assert cloud.state["sess-2026-01_closet_alice"]["deleted_at"] is not None


def test_an_edited_split_on_another_device_is_not_overwritten(tmp_path: Path) -> None:
    """The other half: device B owns that id and edits it.

    B's row passes the shape test — it shares the parent's created_at and points
    at it — so it grades LIKELY here. Copying B's timestamp into our claim would
    have made the announcement tie and replace B's text with the placeholder.
    """
    db = _legacy_store(tmp_path)
    engine = _bloom(db)
    store = TombstoneStore(db)
    leaked_at = dict(store.pending_legacy_announcements())["sess-2026-01_closet_alice"]

    later = datetime.now(timezone.utc) + timedelta(days=1)
    theirs = _cloud_row(
        "sess-2026-01_closet_alice",
        json.dumps([{"speaker": "Alice", "dia_id": "D1", "text": "B'S SIGNED APPROVAL"}]),
        when=later,
    )
    theirs["related_to"] = ["sess-2026-01"]
    theirs["created_at"] = _rows(db, "SELECT created_at FROM memories WHERE id = 'sess-2026-01_closet_alice'")[0][0]

    pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[theirs]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    # The claim is untouched, so it stays older than B's row.
    assert dict(store.pending_legacy_announcements())["sess-2026-01_closet_alice"] == leaked_at

    cloud = _FreshnessClient(rows=[theirs])
    push(engine=engine, tombstones=store, client=cloud, state=SyncState(), poppy_dir=tmp_path)

    assert "sess-2026-01_closet_alice" in cloud.ignored
    assert cloud.state["sess-2026-01_closet_alice"]["content"] == theirs["content"]
    assert cloud.state["sess-2026-01_closet_alice"]["deleted_at"] is None


# --- round 15: an adopted copy survives the parent's next write -----------


def test_a_metadata_edit_keeps_an_adopted_copy_but_a_content_edit_clears_it(tmp_path: Path) -> None:
    """Which parent writes may destroy an adopted copy, and which may not.

    A metadata-only edit replays the same body: nothing about the memory's text
    has moved, so the adopted text is kept. A content edit is a redaction of the
    old text, so every copy of it goes — otherwise editing a secret out of a
    memory leaves an adopted copy holding it, still returned by recall.
    """
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("p", _turns()))
    _strip_marker(db)
    curated = json.dumps([{"speaker": "Alice", "text": "ONLY COPY OF SIGNED APPROVAL"}])
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE memories SET content = ?, enriched_content = 'x' WHERE id = ?", (curated, "p_closet_alice"))
    conn.commit()
    conn.close()

    engine = _bloom(db)
    assert _adopted_unverified_ids(db) == ["p_closet_alice"]

    # A metadata-only edit replays the same body, so nothing about the memory's
    # text has moved and the adopted copy is left alone.
    engine.ingest(_memory("p", _turns(), project="p2"))
    assert engine.get("p_closet_alice").content == curated
    assert _adopted_unverified_ids(db) == ["p_closet_alice"]
    assert _backup_rows(db) == {"p_closet_alice": "adopted"}

    # A CONTENT edit is a redaction of the old text, so every copy of it goes —
    # adopted ones included, or editing a secret out would leave one holding it.
    engine.ingest(_memory("p", _turns("harmlessreplacement")))
    assert _adopted_unverified_ids(db) == []
    # The id is free again, so it is re-derived from the new text.
    assert "p_closet_alice" in _marked_ids(db)
    assert SECRET not in engine.get("p_closet_alice").content
    assert not [r for r in engine.retrieve(SECRET, limit=10) if SECRET in r.memory.content]
    # It might have been a curated split, so the pre-image is the way back.
    assert _backup_rows(db) == {"p_closet_alice": "cleared"}


def test_an_adopted_copy_stays_hidden_and_unsynced(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("p", _turns()))
    _strip_marker(db)
    conn = sqlite3.connect(str(db))
    conn.execute(
        "UPDATE memories SET content = ?, enriched_content = 'x' WHERE id = ?",
        (json.dumps([{"speaker": "Alice", "text": "CURATED"}]), "p_closet_alice"),
    )
    conn.commit()
    conn.close()

    engine = _bloom(db)

    assert [m.id for m in engine.list_all()] == ["p"]
    assert [m.id for m in SeedEngine(db_path=db).list_all()] == ["p"]
    client = _RecordingClient()
    push(
        engine=engine,
        tombstones=TombstoneStore(db),
        client=client,
        state=SyncState(),
        poppy_dir=tmp_path,
    )
    assert not any(r["id"] == "p_closet_alice" and r["deleted_at"] is None for r in client.upserts)


# --- round 16: adoption at ingest carries the same state as at migration --


def test_an_ingest_adopted_copy_survives_a_parent_edit(tmp_path: Path) -> None:
    """The ingest path's twin of the migration's likely tier.

    It wrote the DERIVED state, so the very next parent write — a metadata-only
    edit is enough — deleted and regenerated the row, destroying the text the
    grade exists to keep. Doctor never saw it either, since it counts the
    adopted state.
    """
    from poppy.cli.main import _closet_repair_counts

    db = tmp_path / "memories.db"
    engine = _bloom(db)
    store = TombstoneStore(db)
    curated = json.dumps([{"speaker": "Alice", "dia_id": "D1", "text": "ONLY COPY OF SIGNED APPROVAL"}])
    when = datetime.now(timezone.utc) - timedelta(days=1)

    # The copy arrives first, as it does from an account an older client synced.
    copy = _cloud_row("p_closet_alice", curated, when=when)
    copy["created_at"] = when.isoformat()
    copy["related_to"] = ["p"]
    parent = _cloud_row("p", _turns(), when=when)
    parent["created_at"] = when.isoformat()
    for row in (copy, parent):
        pull(
            engine=engine,
            tombstones=store,
            client=_RecordingClient(rows=[row]),
            state=SyncState(),
            poppy_dir=tmp_path,
        )

    assert _adopted_unverified_ids(db) == ["p_closet_alice"]
    assert _closet_repair_counts(tmp_path) == (1, 0)
    assert "p_closet_alice" not in _pending_ids(store)  # inference never announces

    engine.ingest(_memory("p", _turns(), project="p2"))  # metadata only: kept
    assert engine.get("p_closet_alice").content == curated
    assert _adopted_unverified_ids(db) == ["p_closet_alice"]

    engine.ingest(_memory("p", _turns("harmlessreplacement")))  # content edit: cleared
    assert _adopted_unverified_ids(db) == []
    assert SECRET not in engine.get("p_closet_alice").content
    assert _backup_rows(db)["p_closet_alice"] == "cleared"

    assert forget(engine, tmp_path, "p", tombstones=store).deleted is True
    assert _all_ids(db) == []


# --- round 16: a republished leak re-arms a completed announcement --------


def test_a_republished_leak_re_arms_the_announcement(tmp_path: Path) -> None:
    """An older client can push the copy again after we have cleaned it up.

    The queue row survives with `announce_pending = 0`, so an INSERT that
    ignores conflicts left it done for ever and the text lived on in the cloud.
    """
    db = _legacy_store(tmp_path)
    engine = _bloom(db)
    store = TombstoneStore(db)

    first = _RecordingClient()
    push(engine=engine, tombstones=store, client=first, state=SyncState(), poppy_dir=tmp_path)
    assert _pending_ids(store) == []  # announced and confirmed

    # The old client republishes the same copy, so the cloud row is live again
    # with a newer timestamp.
    later = datetime.now(timezone.utc) + timedelta(days=1)
    again = _cloud_row("sess-2026-01_closet_alice", engine.get("sess-2026-01_closet_alice").content, when=later)
    again["related_to"] = ["sess-2026-01"]
    again["created_at"] = _rows(db, "SELECT created_at FROM memories WHERE id = 'sess-2026-01_closet_alice'")[0][0]

    pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[again]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    # Re-armed, and the claim advanced to the republished row's timestamp so it
    # is not answered `stale_ignored`.
    assert _pending_ids(store) == ["sess-2026-01_closet_alice"]
    assert dict(store.pending_legacy_announcements())["sess-2026-01_closet_alice"] == later.isoformat()

    cloud = _FreshnessClient(rows=[again])
    push(engine=engine, tombstones=store, client=cloud, state=SyncState(), poppy_dir=tmp_path)
    assert cloud.ignored == []
    assert cloud.state["sess-2026-01_closet_alice"]["deleted_at"] is not None


def test_the_fallback_engine_also_clears_an_adopted_copy_on_a_content_edit(tmp_path: Path) -> None:
    """Seed has no speaker expansion, so a content edit there just clears them.

    Both engines have to agree: an adopted copy of redacted text must not
    survive on one and not the other.
    """
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("p", _turns()))
    _strip_marker(db)
    curated = json.dumps([{"speaker": "Alice", "dia_id": "D1", "text": "ONLY COPY OF SIGNED APPROVAL"}])
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE memories SET content = ?, enriched_content = 'x' WHERE id = ?", (curated, "p_closet_alice"))
    conn.commit()
    conn.close()
    _bloom(db)
    assert _adopted_unverified_ids(db) == ["p_closet_alice"]

    seed = SeedEngine(db_path=db)
    same = seed.get("p").content
    seed.ingest(_memory("p", same, project="p2"))  # metadata only: kept
    assert _adopted_unverified_ids(db) == ["p_closet_alice"]

    seed.ingest(_memory("p", "redacted prose"))  # content edit: cleared
    assert _all_ids(db) == ["p"]
    assert not seed.retrieve("APPROVAL", limit=10)
    assert _backup_rows(db)["p_closet_alice"] == "cleared"


# --- round 18: a pulled deletion keeps its own clock ----------------------


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


# --- round 19: a deletion record never moves backwards --------------------


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
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE closet_tombstones SET tombstoned_at = ?", (two.isoformat(),))
    conn.commit()
    conn.close()

    # An older sighting of the same deletion must not lower it...
    store.add_closets(["sess-1_closet_alice"], now=noon)
    assert store.get_closet("sess-1_closet_alice").tombstoned_at == two

    # ... while between two REMOTE sightings the earliest still wins.
    store.add_closets(["remote_closet_bob"], now=two)
    store.add_closets(["remote_closet_bob"], now=noon)
    assert store.get_closet("remote_closet_bob").tombstoned_at == noon


# --- round 20: a pulled EDIT dates its cascade from the edit --------------


@pytest.mark.parametrize("engine_name", ["bloom", "seed"])
def test_a_pulled_edit_does_not_bury_a_later_note_at_a_copys_id(tmp_path: Path, engine_name: str) -> None:
    """July 1 multi-speaker memory, July 2 edit to prose, July 3 note.

    The edit clears the copies of the July 1 text. Dated at receipt time
    (September) that deletion looks newer than the July 3 note, so the note is
    skipped and the watermark advances past it for good.
    """
    db = tmp_path / "memories.db"
    jul1 = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    jul2 = jul1 + timedelta(days=1)
    jul3 = jul1 + timedelta(days=2)

    writer = _bloom(db)
    original = _memory("sess-1", _turns())
    original.created_at = original.updated_at = jul1
    original.source.timestamp = jul1
    writer.ingest(original)
    assert "sess-1_closet_alice" in _marked_ids(db)

    engine = _bloom(db) if engine_name == "bloom" else SeedEngine(db_path=db)
    store = TombstoneStore(db)

    # July 2: another device replaced the text with prose.
    edited = _cloud_row("sess-1", "redacted prose", when=jul2)
    edited["created_at"] = jul1.isoformat()
    pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[edited]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )
    assert engine.get("sess-1").content == "redacted prose"
    recorded = store.get_closet("sess-1_closet_alice")
    assert recorded is not None
    assert recorded.tombstoned_at == jul2  # the edit's clock, not today's

    # July 3: an independent note written at that id, after the edit.
    note = _cloud_row("sess-1_closet_alice", "AN INDEPENDENT NOTE", when=jul3)
    result = pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[note]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.skipped_closets == 0
    assert result.applied_live == 1
    assert engine.get("sess-1_closet_alice").content == "AN INDEPENDENT NOTE"


def test_a_locally_made_edit_still_dates_its_cascade_now(tmp_path: Path) -> None:
    """Only edits that ARRIVED carry someone else's clock."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    store = TombstoneStore(db)
    engine.ingest(_memory("sess-1", _turns()))

    before = datetime.now(timezone.utc)
    engine.ingest(_memory("sess-1", "redacted prose"))  # local content edit

    recorded = store.get_closet("sess-1_closet_alice")
    assert recorded is not None and recorded.tombstoned_at >= before


# --- round 21: unmarked copies are graded at redaction time ---------------


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


@pytest.mark.parametrize("engine_name", ["seed", "bloom"])
def test_forget_clears_an_unmarked_legacy_copy(tmp_path: Path, engine_name: str) -> None:
    """A fallback-only store never synthesises, so it never adopts at ingest.

    Nothing marked the pulled copy, so the redaction path had nothing to clear:
    forgetting the memory left the copy recallable and the next push sent the
    text live. Grading at redaction time makes that path self-sufficient.
    """
    db = tmp_path / "memories.db"
    engine = SeedEngine(db_path=db) if engine_name == "seed" else _bloom(db)
    store = TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=1)
    _pull_leaked_pair(tmp_path, engine, store, when=when)
    assert engine.get("p_closet_alice") is not None

    assert forget(engine, tmp_path, "p", tombstones=store).deleted is True

    assert _all_ids(db) == []
    assert not engine.retrieve(SECRET, limit=10)
    assert not _bloom(db).retrieve(SECRET, limit=10)

    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert [r for r in client.upserts if r["deleted_at"] is None] == []
    assert not any(SECRET in json.dumps(r) for r in client.upserts if r["id"] != "p")

    # Its text was ours, so the cloud copy is announced exactly once.
    assert "p_closet_alice" in {r["id"] for r in client.upserts}


@pytest.mark.parametrize("engine_name", ["seed", "bloom"])
def test_a_forget_landing_while_pull_waits_for_the_gate_is_honoured(
    tmp_path: Path, engine_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pull decides what a row is INSIDE the write gate, not before it.

    The store holds a parent and its unmarked legacy copy. The cloud still has
    the copy. While pull is waiting for the gate, another process forgets the
    parent, which deletes the copy and records its deletion. A pull that had
    already decided "ordinary live row" would now ingest the stale cloud text
    over that deletion, clear the record, and push the secret back up.
    """
    import poppy.sync as sync_mod
    from poppy.db import write_gate as real_gate

    db = tmp_path / "memories.db"
    engine = SeedEngine(db_path=db) if engine_name == "seed" else _bloom(db)
    store = TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=1)
    _pull_leaked_pair(tmp_path, engine, store, when=when)
    assert engine.get("p_closet_alice") is not None

    stale = _cloud_row("p_closet_alice", json.dumps([{"speaker": "Alice", "dia_id": "D1", "text": SECRET}]), when=when)
    stale["created_at"] = when.isoformat()
    stale["related_to"] = ["p"]

    raced: list[bool] = []

    @contextmanager
    def gate_with_a_forget_ahead_of_us(poppy_dir: Path):
        # The other process holds the gate first and forgets the parent.
        if not raced:
            raced.append(True)
            assert forget(engine, tmp_path, "p", tombstones=store).deleted is True
        with real_gate(poppy_dir):
            yield

    monkeypatch.setattr(sync_mod, "write_gate", gate_with_a_forget_ahead_of_us)
    result = pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[stale]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert raced
    assert result.applied_live == 0
    assert result.skipped_closets == 1
    assert engine.get("p_closet_alice") is None
    assert store.get_closet("p_closet_alice") is not None
    assert not engine.retrieve(SECRET, limit=10)

    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    # The parent's own tombstone carries its text (cloud Trash); the copy must not.
    assert not any(SECRET in json.dumps(r) for r in client.upserts if r["id"] != "p")


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


def test_a_redaction_holds_the_write_lock_while_it_enumerates_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Between listing a parent's copies and deleting them, no other connection may write.

    Otherwise a real memory restored at one of those ids in that window is
    deleted by id along with the copies.
    """
    import poppy.engine._closet_marker as marker

    db = tmp_path / "memories.db"
    engine = _bloom(db)
    engine.ingest(_memory("p", _turns()))
    original = marker.marked_closet_ids
    seen: list[type] = []

    def enumerate_then_try_to_write(conn, parent_id, **kwargs):
        ids = original(conn, parent_id, **kwargs)
        if parent_id == "p" and not seen:
            other = sqlite3.connect(str(db), timeout=0)
            try:
                other.execute("UPDATE memories SET project = 'x' WHERE id = 'p_closet_alice'")
                other.commit()
                seen.append(type(None))
            except sqlite3.OperationalError as exc:
                seen.append(type(exc))
            finally:
                other.close()
        return ids

    monkeypatch.setattr(marker, "marked_closet_ids", enumerate_then_try_to_write)
    assert forget(engine, tmp_path, "p").deleted is True

    assert seen == [sqlite3.OperationalError]
    assert engine.get("p_closet_alice") is None


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
    from poppy.engine._closet_marker import is_closet_shaped

    content = json.dumps([{"speaker": "Alice", "dia_id": "D1", "text": SECRET}])
    common = dict(content=content, related_raw=json.dumps(["p"]))
    assert is_closet_shaped(
        "p", "alice", created_at="2026-07-01T12:00:00+02:00", parent_created_at="2026-07-01T10:00:00+00:00", **common
    )
    assert not is_closet_shaped(
        "p", "alice", created_at="2026-07-01T12:00:00+02:00", parent_created_at="2026-07-01T12:00:00+00:00", **common
    )


def test_a_forget_takes_only_its_own_copies_not_a_prefix_siblings(tmp_path: Path) -> None:
    """``p`` and ``p_closet_notes`` are both multi-speaker parents.

    The copies of the second sit under the first's ``_closet_`` prefix. Forgetting
    ``p`` must clear ``p_closet_alice`` and leave ``p_closet_notes_closet_alice``
    marked and recallable, with no deletion recorded for it.
    """
    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    engine.ingest(_memory("p", _turns()))
    engine.ingest(_memory("p_closet_notes", _turns("siblingsecret")))
    assert engine.get("p_closet_alice") is not None
    assert engine.get("p_closet_notes_closet_alice") is not None

    assert forget(engine, tmp_path, "p", tombstones=store).deleted is True

    assert engine.get("p_closet_alice") is None
    sibling = engine.get("p_closet_notes_closet_alice")
    assert sibling is not None and "siblingsecret" in sibling.content
    assert engine.is_closet_row("p_closet_notes_closet_alice")
    assert store.get_closet("p_closet_notes_closet_alice") is None
    assert store.get_closet("p_closet_alice") is not None
    assert [r.memory.id for r in engine.retrieve("siblingsecret", limit=10) if r.memory.id.endswith("_closet_alice")]


def test_announcement_claims_compare_as_instants_across_offsets(tmp_path: Path) -> None:
    """A claim stamped ``12:00+02:00`` (10:00Z) must be advanced by a proven re-push at ``11:00Z``."""
    from poppy.engine._closet_marker import LEGACY_CLOSET_TABLE, rearm_legacy_announcement, record_legacy_closet_ids

    db = tmp_path / "memories.db"
    _bloom(db)
    conn = sqlite3.connect(str(db))
    record_legacy_closet_ids(conn, [("x_closet_alice", "2026-07-01T12:00:00+02:00")])
    rearm_legacy_announcement(conn, "x_closet_alice", "2026-07-01T11:00:00+00:00")
    conn.commit()
    sql = f"SELECT legacy_updated_at FROM {LEGACY_CLOSET_TABLE} WHERE id = ?"
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
    from poppy.engine._closet_marker import mark_legacy_announced, record_legacy_closet_ids

    db = tmp_path / "memories.db"
    _bloom(db)
    store = TombstoneStore(db)
    eight_days_ago = datetime.now(timezone.utc) - timedelta(days=8)
    store.add_closets(["x_closet_alice", "y_closet_bob"], now=eight_days_ago)
    conn = sqlite3.connect(str(db))
    record_legacy_closet_ids(conn, [("x_closet_alice", eight_days_ago.isoformat())])
    conn.commit()

    store.purge_expired(pushed_through=datetime.now(timezone.utc).isoformat())
    assert store.get_closet("x_closet_alice") is not None  # announcement still pending
    assert store.get_closet("y_closet_bob") is None  # nothing pending: window + watermark apply

    mark_legacy_announced(conn, ["x_closet_alice"])
    conn.commit()
    conn.close()
    store.purge_expired(pushed_through=datetime.now(timezone.utc).isoformat())
    assert store.get_closet("x_closet_alice") is None


def test_a_same_text_seed_write_onto_a_copy_keeps_its_parent_link(tmp_path: Path) -> None:
    """Re-ingesting a marked copy with identical text and an empty related_to.

    The marker is kept, so the back-reference must be kept too, or the parent's
    forget no longer finds the copy and the speaker text stays recallable.
    """
    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("p", _turns()))
    seed = SeedEngine(db_path=db)
    store = TombstoneStore(db)
    copy = seed.get("p_closet_alice")
    assert copy is not None and copy.related_to == ["p"]
    seed.ingest(_memory("p_closet_alice", copy.content))  # default related_to=[]

    assert seed.is_closet_row("p_closet_alice")
    assert seed.get("p_closet_alice").related_to == ["p"]
    assert forget(seed, tmp_path, "p", tombstones=store).deleted is True
    assert seed.get("p_closet_alice") is None
    assert not seed.retrieve(SECRET, limit=10)


def test_a_same_text_parent_reingest_keeps_its_creation_time_and_its_copys_proof(tmp_path: Path) -> None:
    """Seed pulled the pair; bloom re-ingests the parent body with a new created_at and project.

    The copy is proven by sharing the parent's creation time. Replacing that
    time on a same-text write stranded the copy past the parent's forget.
    """
    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    _pull_leaked_pair(tmp_path, seed, store, when=datetime.now(timezone.utc) - timedelta(days=1))
    original = seed.get("p").created_at
    engine = _bloom(db)
    engine.ingest(_memory("p", _turns(), project="new-project"))
    assert engine.get("p").created_at == original
    assert engine.get("p").project == "new-project"

    assert forget(engine, tmp_path, "p", tombstones=store).deleted is True
    assert engine.get("p_closet_alice") is None
    assert not seed.retrieve(SECRET, limit=10)


def test_an_acknowledgement_clears_only_the_claim_that_was_sent(tmp_path: Path) -> None:
    """While push's announcement is in flight, a pull sees the copy republished and re-arms it.

    The old claim is answered stale_ignored; clearing by id alone would consume
    the new one with the text still live.
    """
    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    t0 = datetime.now(timezone.utc) - timedelta(days=2)
    parent, copy = _leaked_pair(t0)
    pull(
        engine=engine, tombstones=store, client=_RecordingClient([parent, copy]), state=SyncState(), poppy_dir=tmp_path
    )
    newer = dict(copy, updated_at=(t0 + timedelta(days=1)).isoformat())

    class RacingCloud(_FreshnessClient):
        raced = False

        def upsert(self, row: dict) -> None:
            if row["id"] == copy["id"] and row["deleted_at"] is not None and not self.raced:
                self.raced = True
                self.state[copy["id"]] = dict(newer)
                pull(
                    engine=engine,
                    tombstones=store,
                    client=_RecordingClient([newer]),
                    state=SyncState(),
                    poppy_dir=tmp_path,
                )
            super().upsert(row)

    cloud = RacingCloud([copy])
    push(engine=engine, tombstones=store, client=cloud, state=SyncState(), poppy_dir=tmp_path)
    assert cloud.raced
    assert cloud.state[copy["id"]]["deleted_at"] is None  # the old claim lost
    assert dict(store.pending_legacy_announcements()).get(copy["id"]) == newer["updated_at"]  # the new one is kept


def test_a_pulled_copy_deletion_is_dated_from_its_deleted_at(tmp_path: Path) -> None:
    """A cleanup row whose updated_at was bumped after its deleted_at must not bury a recreation in between."""
    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    t0 = datetime.now(timezone.utc) - timedelta(days=2)
    cleanup = _cloud_row("p_closet_alice", CLOSET_TOMBSTONE_CONTENT, deleted=True, when=t0)
    cleanup["updated_at"] = (t0 + timedelta(hours=2)).isoformat()
    pull(engine=engine, tombstones=store, client=_RecordingClient([cleanup]), state=SyncState(), poppy_dir=tmp_path)
    assert store.get_closet("p_closet_alice").tombstoned_at == t0

    recreation = _cloud_row("p_closet_alice", "independent note after deletion", when=t0 + timedelta(hours=1))
    pull(engine=engine, tombstones=store, client=_RecordingClient([recreation]), state=SyncState(), poppy_dir=tmp_path)
    assert engine.get("p_closet_alice").content == recreation["content"]
    assert store.get_closet("p_closet_alice") is None


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

    store.add_closets(["p_closet_alice"], now=datetime(2026, 7, 1, 11, tzinfo=timezone.utc))
    store.purge_expired(pushed_through="2026-07-01T12:00:00+02:00")
    assert store.get_closet("p_closet_alice") is not None  # 11:00Z is after 10:00Z


def test_a_redaction_that_cannot_get_the_write_lock_aborts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Another connection holds the write lock while it rewrites a copy's id as a real note.

    The redaction must fail loudly. Enumerating without the lock and deleting
    afterwards, by id, would take the note the other writer commits in between.
    """
    import poppy.engine._closet_marker as marker

    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    engine.ingest(_memory("p", _turns()))
    engine._conn.execute("PRAGMA busy_timeout = 50")
    store._conn.execute("PRAGMA busy_timeout = 50")

    other = sqlite3.connect(str(db), timeout=0)
    real_delete = engine.delete

    def other_takes_the_lock_then_delete(memory_id, **kwargs):
        # Forget has snapshotted the parent; the other writer now holds the
        # write lock, mid-way through turning the copy's id into a real note.
        other.execute("BEGIN IMMEDIATE")
        other.execute("UPDATE memories SET content = 'INDEPENDENT NOTE', is_closet = 0 WHERE id = 'p_closet_alice'")
        return real_delete(memory_id, **kwargs)

    original = marker.marked_closet_ids

    def enumerate_then_the_other_writer_commits(conn, parent_id, **kwargs):
        ids = original(conn, parent_id, **kwargs)
        if other.in_transaction:
            other.commit()  # the window: the lock was never ours
        return ids

    monkeypatch.setattr(engine, "delete", other_takes_the_lock_then_delete)
    monkeypatch.setattr(marker, "marked_closet_ids", enumerate_then_the_other_writer_commits)
    with pytest.raises(sqlite3.OperationalError):
        forget(engine, tmp_path, "p", tombstones=store)
    if other.in_transaction:
        other.commit()
    other.close()

    assert engine.get("p") is not None
    assert engine.get("p_closet_alice").content == "INDEPENDENT NOTE"


def test_a_local_forget_always_snapshots_even_past_a_skewed_record(tmp_path: Path) -> None:
    """A pulled deletion dated in the future must not stop this device's own forget from snapshotting."""
    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    engine.ingest(_memory("note", "MY NOTE"))
    skewed = _memory("note", "OLDER TEXT")
    store.add(skewed, tombstoned_at=datetime.now(timezone.utc) + timedelta(hours=1))

    assert forget(engine, tmp_path, "note", tombstones=store).deleted is True
    assert store.get("note").memory.content == "MY NOTE"


@pytest.mark.parametrize("engine_name", ["seed", "bloom"])
def test_a_parents_forget_also_clears_a_legacy_trash_entry_of_its_copy(tmp_path: Path, engine_name: str) -> None:
    """An older client let the user send a copy to Trash; the copy was re-derived and the snapshot stayed.

    Forgetting the parent must take that snapshot too, claim the announcement
    with the snapshot's own deletion time, and leave nothing to restore.
    """
    from poppy.write_flow import restore

    db = tmp_path / "memories.db"
    _bloom(db).ingest(_memory("p", _turns()))
    engine = SeedEngine(db_path=db) if engine_name == "seed" else _bloom(db)
    store = TombstoneStore(db)
    legacy = store.add(engine.get("p_closet_alice"))  # what a 0.2.4 forget of the copy left behind
    assert SECRET in store.get("p_closet_alice").memory.content

    assert forget(engine, tmp_path, "p", tombstones=store).deleted is True

    assert store.get("p_closet_alice") is None
    assert engine.get("p_closet_alice") is None
    assert restore(engine, tmp_path, "p_closet_alice", tombstones=store).found is False
    pending = dict(store.pending_legacy_announcements())
    assert datetime.fromisoformat(pending["p_closet_alice"]) == legacy.tombstoned_at

    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert not any(SECRET in json.dumps(r) for r in client.upserts if r["id"] != "p")
    assert "p_closet_alice" in {r["id"] for r in client.upserts}


def test_a_trash_entry_that_is_not_the_parents_text_survives_the_parents_forget(tmp_path: Path) -> None:
    """A real note that once lived at the copy's id keeps its Trash entry."""
    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    engine.ingest(_memory("p", _turns()))
    store.add(_memory("p_closet_alice", "MY OWN NOTE AT THAT ID"))

    assert forget(engine, tmp_path, "p", tombstones=store).deleted is True
    assert store.get("p_closet_alice").memory.content == "MY OWN NOTE AT THAT ID"
    assert "p_closet_alice" not in dict(store.pending_legacy_announcements())


def test_a_content_edit_also_clears_a_legacy_trash_entry_of_its_copy(tmp_path: Path) -> None:
    """The bloom edit path passes tombstone=False; the snapshot of the OLD text still goes."""
    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    engine.ingest(_memory("p", _turns()))
    legacy = store.add(engine.get("p_closet_alice"))

    engine.ingest(_memory("p", "plain prose, no speakers"))

    assert store.get("p_closet_alice") is None
    assert engine.get("p_closet_alice") is None
    pending = dict(store.pending_legacy_announcements())
    assert datetime.fromisoformat(pending["p_closet_alice"]) == legacy.tombstoned_at


def test_a_forget_clears_a_legacy_trash_entry_even_with_no_live_copy_left(tmp_path: Path) -> None:
    """An older client forgot both copies: snapshots kept, rows gone. The parent's forget must still reach them."""
    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    engine.ingest(_memory("p", _turns()))
    store.add(engine.get("p_closet_alice"))
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM memories WHERE id LIKE 'p\\_closet\\_%' ESCAPE '\\'")
    conn.execute("DELETE FROM memory_embeddings WHERE id LIKE 'p\\_closet\\_%' ESCAPE '\\'")
    conn.commit()
    conn.close()
    assert engine.get("p_closet_alice") is None
    assert store.get("p_closet_alice") is not None

    assert forget(engine, tmp_path, "p", tombstones=store).deleted is True
    assert store.get("p_closet_alice") is None
    assert "p_closet_alice" in dict(store.pending_legacy_announcements())


def test_a_trash_entry_with_the_same_text_but_other_provenance_survives(tmp_path: Path) -> None:
    """Matching text alone never condemns a Trash entry: an independent memory can hold the same turns."""
    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    engine.ingest(_memory("p", _turns()))
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
    _bloom(db).ingest(_memory("p", _turns()))
    engine = SeedEngine(db_path=db) if engine_name == "seed" else _bloom(db)
    store = TombstoneStore(db)
    legacy = store.add(engine.get("p_closet_alice"))

    result = forget(engine, tmp_path, "p_closet_alice", tombstones=store)
    assert result.deleted is True

    assert engine.get("p_closet_alice") is None
    assert store.get("p_closet_alice") is None
    assert restore(engine, tmp_path, "p_closet_alice", tombstones=store).found is False
    assert datetime.fromisoformat(dict(store.pending_legacy_announcements())["p_closet_alice"]) == legacy.tombstoned_at

    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert not any(SECRET in json.dumps(r) for r in client.upserts if r["id"] != "p")


def test_forgetting_a_proven_unmarked_copy_on_a_seed_store_is_content_free(tmp_path: Path) -> None:
    """A store that never ran bloom holds a pulled leaked copy unmarked; the user forgets it by id.

    It must go the content-free way: no Trash snapshot of the speaker text, no
    text on the wire, and the cloud row announced with its own timestamp.
    """
    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=1)
    _pull_leaked_pair(tmp_path, seed, store, when=when)
    assert not seed.is_closet_row("p_closet_alice")

    result = forget(seed, tmp_path, "p_closet_alice", tombstones=store)
    assert result.deleted is True
    assert result.tombstone is None

    assert seed.get("p_closet_alice") is None
    assert store.get("p_closet_alice") is None
    assert store.get_closet("p_closet_alice") is not None
    assert datetime.fromisoformat(dict(store.pending_legacy_announcements())["p_closet_alice"]) == when

    client = _RecordingClient()
    push(engine=seed, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    sent = [r for r in client.upserts if r["id"] == "p_closet_alice"]
    assert sent and all(r["content"] == CLOSET_TOMBSTONE_CONTENT for r in sent)
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
    assert datetime.fromisoformat(dict(store.pending_legacy_announcements())["p_closet_alice"]) == legacy.tombstoned_at
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

    cleanup = _cloud_row("p_closet_alice", CLOSET_TOMBSTONE_CONTENT, deleted=True, when=when)
    result = pull(
        engine=seed, tombstones=store, client=_RecordingClient([cleanup]), state=SyncState(), poppy_dir=tmp_path
    )

    assert result.applied_tombstones == 1
    assert seed.get("p_closet_alice") is None
    assert store.get("p_closet_alice") is None
    assert store.get_closet("p_closet_alice") is not None
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
    assert not seed.is_closet_row("p_closet_alice")

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
    conn = sqlite3.connect(str(db))
    conn.execute(f"UPDATE memories SET {MARKER_COLUMN} = 0")  # nothing is marked
    conn.execute(
        "UPDATE memories SET content = ?, related_to = ? WHERE id = ?",
        ("MY OWN NOTE, NOT A COPY", json.dumps(["somewhere-else"]), "p_closet_alice"),
    )
    conn.commit()
    conn.close()

    engine = SeedEngine(db_path=db) if engine_name == "seed" else _bloom(db)
    store = TombstoneStore(db)
    assert forget(engine, tmp_path, "p", tombstones=store).deleted is True

    assert engine.get("p_closet_alice").content == "MY OWN NOTE, NOT A COPY"
    assert "p_closet_alice" in {m.id for m in engine.list_all()}
    assert "p_closet_alice" not in _pending_ids(store)


def test_a_seed_content_edit_clears_an_unmarked_legacy_copy(tmp_path: Path) -> None:
    """The same gap on the edit path rather than the delete path."""
    db = tmp_path / "memories.db"
    engine = SeedEngine(db_path=db)
    store = TombstoneStore(db)
    _pull_leaked_pair(tmp_path, engine, store, when=datetime.now(timezone.utc) - timedelta(days=1))

    engine.ingest(_memory("p", "redacted prose"))

    assert _all_ids(db) == ["p"]
    assert not engine.retrieve(SECRET, limit=10)
    assert _backup_rows(db)["p_closet_alice"] == "cleared"


# --- round 22: one mechanism dates every record a remote write clears -----


@pytest.mark.parametrize("engine_name", ["seed", "bloom"])
@pytest.mark.parametrize("via", ["deletion", "edit"])
def test_a_remote_write_dates_an_unmarked_copy_it_clears(tmp_path: Path, engine_name: str, via: str) -> None:
    """Sync used to correct these afterwards, and only for the MARKED ids it
    could enumerate. An unmarked copy graded and cleared during the same call
    kept receipt time and became a floor nothing lowers, so a later independent
    row at that id was skipped for ever."""
    db = tmp_path / "memories.db"
    engine = SeedEngine(db_path=db) if engine_name == "seed" else _bloom(db)
    store = TombstoneStore(db)
    jul1 = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    jul2 = jul1 + timedelta(days=1)
    jul3 = jul1 + timedelta(days=2)
    _pull_leaked_pair(tmp_path, engine, store, when=jul1)
    # The default engine adopts it while deriving the parent's copies; the
    # fallback engine never synthesises, so it leaves it unmarked. The clearing
    # has to date its record from the remote event either way, and it is the
    # UNMARKED case that sync's old post-hoc correction could not reach.
    assert ("p_closet_alice" in _marked_ids(db)) is (engine_name == "bloom")

    if via == "deletion":
        remote = _cloud_row("p", _turns(), deleted=True, when=jul2)
    else:
        remote = _cloud_row("p", "redacted prose", when=jul2)
        remote["created_at"] = jul1.isoformat()
    pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[remote]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert engine.get("p_closet_alice") is None
    recorded = store.get_closet("p_closet_alice")
    assert recorded is not None
    assert recorded.tombstoned_at == jul2  # the remote event's clock

    # July 3: an independent row written at that id after the remote write.
    note = _cloud_row("p_closet_alice", "AN INDEPENDENT NOTE", when=jul3)
    result = pull(
        engine=engine,
        tombstones=store,
        client=_RecordingClient(rows=[note]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.skipped_closets == 0
    assert engine.get("p_closet_alice").content == "AN INDEPENDENT NOTE"


def test_a_local_redaction_still_dates_a_graded_copy_now(tmp_path: Path) -> None:
    """The threading is scoped to writes sync is applying for someone else."""
    db = tmp_path / "memories.db"
    engine = SeedEngine(db_path=db)
    store = TombstoneStore(db)
    _pull_leaked_pair(tmp_path, engine, store, when=datetime.now(timezone.utc) - timedelta(days=2))

    before = datetime.now(timezone.utc)
    forget(engine, tmp_path, "p", tombstones=store)

    recorded = store.get_closet("p_closet_alice")
    assert recorded is not None and recorded.tombstoned_at >= before


# --- round 22: a proven copy is announced on every redaction path ---------


def test_a_content_edit_announces_a_proven_unmarked_copy(tmp_path: Path) -> None:
    """The default engine's edit path passes tombstone=False, which decides
    whether a LOCAL deletion record is written. Whether the cloud still holds
    our leaked text is a different question, and gating it there left the
    cloud's copy of the secret alive indefinitely."""
    db = tmp_path / "memories.db"
    seed = SeedEngine(db_path=db)
    store = TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=1)
    _pull_leaked_pair(tmp_path, seed, store, when=when)
    assert "p_closet_alice" not in _pending_ids(store)

    engine = _bloom(db)
    engine.ingest(_memory("p", "redacted prose"))  # local content edit

    assert engine.get("p_closet_alice") is None
    assert "p_closet_alice" in _pending_ids(store)

    leaked = _cloud_row("p_closet_alice", json.dumps([{"speaker": "Alice", "text": SECRET}]), when=when)
    cloud = _FreshnessClient(rows=[leaked])
    push(engine=engine, tombstones=store, client=cloud, state=SyncState(), poppy_dir=tmp_path)

    assert cloud.ignored == []
    assert cloud.state["p_closet_alice"]["deleted_at"] is not None
    assert cloud.state["p_closet_alice"]["content"] == CLOSET_TOMBSTONE_CONTENT


# --- round 23: a Trash snapshot is graded like everything else --


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
    assert store.get_closet("p_closet_alice") is not None

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


def test_audit_lifecycle_unmarked_copy_trash_survives_parent_forget(tmp_path: Path) -> None:
    """Forgetting an unmarked copy by id must never snapshot its speaker text.

    The whole chain: a seed store pulled the leaked pair, an external writer
    replayed the copy with no related_to, and the row stopped being provable. The
    direct forget then took the ordinary path and put the speaker turns in Trash;
    the parent's later forget could not grade that entry (nothing named the
    parent), and restoring it revived the text as an ordinary, pushable memory.
    """
    from poppy.write_flow import restore

    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    _pull_leaked_pair(tmp_path, seed, store, when=datetime.now(timezone.utc) - timedelta(days=1))
    copy = seed.get("p_closet_alice")
    seed.ingest(_memory("p_closet_alice", copy.content, related_to=[]))  # same text, link dropped

    assert forget(seed, tmp_path, "p_closet_alice", tombstones=store).deleted is True
    assert store.get("p_closet_alice") is None  # content-free, so nothing to revive
    assert store.get_closet("p_closet_alice") is not None

    assert forget(seed, tmp_path, "p", tombstones=store).deleted is True

    assert restore(seed, tmp_path, "p_closet_alice", tombstones=store).found is False
    assert not seed.retrieve(SECRET, limit=10)
    client = _RecordingClient()
    push(engine=seed, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert not any(SECRET in json.dumps(r) for r in client.upserts if r["id"] != "p")


def test_audit_lifecycle_legacy_trash_copy_survives_parent_edit(tmp_path: Path) -> None:
    """A snapshot of an ADOPTED copy, whose text the parent no longer derives.

    A 0.2.4 client trashed the copy and then edited the parent under the fallback
    engine, so the upgrade re-derives a different body and adopts the stale row as
    LIKELY. Comparing the snapshot with the parent's CURRENT derivation could not
    see it, and the parent's redaction left it restorable and pushable.
    """
    from poppy.write_flow import restore

    db = _legacy_store(tmp_path)  # parent + its two pre-marker copies, no column
    store = TombstoneStore(db)
    conn = sqlite3.connect(str(db))
    stale = _rows(db, "SELECT content, created_at, updated_at FROM memories WHERE id = 'sess-2026-01_closet_alice'")[0]
    # The old client's Trash entry for the copy, holding the copy's own text.
    conn.execute(
        "INSERT INTO ui_tombstones (id, content, memory_type, project, source_type, source_session_id,"
        " source_timestamp, confidence, related_to, created_at, updated_at, tombstoned_at, token)"
        " VALUES (?, ?, 'fact', NULL, 'cli', NULL, ?, 1.0, ?, ?, ?, ?, 'tok')",
        (
            "sess-2026-01_closet_alice",
            stale[0],
            stale[1],
            json.dumps(["sess-2026-01"]),
            stale[1],
            stale[2],
            stale[2],
        ),
    )
    # ... and its pre-fix edit of the parent, which left the copies untouched.
    conn.execute(
        "UPDATE memories SET content = ?, enriched_content = ? WHERE id = 'sess-2026-01'",
        (_turns("harmlessreplacement"), "x"),
    )
    conn.commit()
    conn.close()

    engine = _bloom(db)  # the upgrade: adopts the stale copy as LIKELY
    store = TombstoneStore(db)
    assert "sess-2026-01_closet_alice" in _adopted_unverified_ids(db)

    assert forget(engine, tmp_path, "sess-2026-01", tombstones=store).deleted is True

    assert store.get("sess-2026-01_closet_alice") is None
    assert restore(engine, tmp_path, "sess-2026-01_closet_alice", tombstones=store).found is False
    # Never announced: the grade that identified the row beside it was inference,
    # and an announcement deletes the cloud's row for every device.
    assert "sess-2026-01_closet_alice" not in _pending_ids(store)
    # The text is still recoverable from the redaction's own pre-image.
    assert _backup_rows(db)["sess-2026-01_closet_alice"] in {"adopted", "cleared"}
    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert not any(SECRET in json.dumps(r) for r in client.upserts if r["id"] != "sess-2026-01")


def test_restore_refuses_a_legacy_trash_copy_whose_parent_is_still_live(tmp_path: Path) -> None:
    """Item 10: the guard read the LIVE row's marker, and there is no live row here.

    On 0.2.4 the user forgot the copy and kept the parent. After upgrading,
    restoring that entry created an unmarked memory holding the speaker text,
    which the next push uploaded live with no cleanup queued.
    """
    from poppy.write_flow import restore

    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    engine.ingest(_memory("p", _turns()))
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
    assert datetime.fromisoformat(dict(store.pending_legacy_announcements())["p_closet_alice"]) == legacy.tombstoned_at
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


# --- round 23: push asks again before it uploads ----------------


def test_push_snapshot_sends_copy_after_forget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synced_state) -> None:
    """Push materialises every candidate and then uploads; a forget landing between the two.

    The forget's own tombstone is newer, so the cloud converges on the next cycle
    — but a legacy unmarked copy's text was uploaded once MORE after the user had
    deleted its parent. Push re-reads each row immediately before its upsert.
    """
    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    _pull_leaked_pair(tmp_path, seed, store, when=datetime.now(timezone.utc) - timedelta(days=1))
    assert seed.get("p_closet_alice") is not None
    assert "p_closet_alice" in {m.id for m in seed.list_all()}  # unmarked, so push sees it

    real_list_all = seed.list_all
    landed: list[bool] = []

    def list_all_then_a_forget_lands(*args, **kwargs):
        rows = real_list_all(*args, **kwargs)
        if not landed:
            landed.append(True)
            assert forget(seed, tmp_path, "p", tombstones=store).deleted is True
        return rows

    monkeypatch.setattr(seed, "list_all", list_all_then_a_forget_lands)
    client = _RecordingClient()
    push(engine=seed, tombstones=store, client=client, state=synced_state(), poppy_dir=tmp_path)

    assert landed
    assert not any(r["id"] == "p_closet_alice" and r["deleted_at"] is None for r in client.upserts)
    assert not any(SECRET in json.dumps(r) for r in client.upserts if r["id"] != "p")
    # The parent went the same way, and its deletion still travels as a tombstone.
    assert not any(r["id"] == "p" and r["deleted_at"] is None for r in client.upserts)
    assert any(r["id"] == "p" and r["deleted_at"] is not None for r in client.upserts)


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


# --- round 23: a copy-deletion record may be raised, never lowered


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
    store.add_closets(["sess-1_closet_alice"], now=t1)

    cleanup = _cloud_row("sess-1_closet_alice", CLOSET_TOMBSTONE_CONTENT, deleted=True, when=t3)
    pull(engine=engine, tombstones=store, client=_RecordingClient([cleanup]), state=SyncState(), poppy_dir=tmp_path)

    assert store.get_closet("sess-1_closet_alice").tombstoned_at == t3

    # ... and the record now suppresses the t2 copy a watermark reset re-offers.
    stale = _cloud_row("sess-1_closet_alice", json.dumps([{"speaker": "Alice", "text": SECRET}]), when=t2)
    result = pull(
        engine=engine, tombstones=store, client=_RecordingClient([stale]), state=SyncState(), poppy_dir=tmp_path
    )

    assert result.skipped_closets == 1
    assert engine.get("sess-1_closet_alice") is None


def test_a_cleanup_deletion_never_lowers_a_record(tmp_path: Path) -> None:
    """Raising only: an older cleanup must not pull a newer record down."""
    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    early = datetime(2026, 7, 1, tzinfo=timezone.utc)
    late = datetime(2026, 8, 1, tzinfo=timezone.utc)
    store.add_closets(["sess-1_closet_alice"], now=late)

    cleanup = _cloud_row("sess-1_closet_alice", CLOSET_TOMBSTONE_CONTENT, deleted=True, when=early)
    pull(engine=engine, tombstones=store, client=_RecordingClient([cleanup]), state=SyncState(), poppy_dir=tmp_path)

    assert store.get_closet("sess-1_closet_alice").tombstoned_at == late


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
            assert store.get_closet("p_closet_alice") is None
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
    assert result.skipped_closets == 1
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


# --- round 23: a same-text write keeps a copy's ownership -------


@pytest.mark.parametrize("engine_name", ["seed", "bloom"])
def test_audit_engine_unmarked_copy_same_text_related_edit(tmp_path: Path, engine_name: str) -> None:
    """An external writer replays the pulled copy with the same text and no related_to.

    On a store that never ran bloom nothing adopted that copy, so its
    back-reference is the only thing identifying it. Erased, the parent's forget
    graded its own copy a stranger and left the speaker text live and pushable.
    """
    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=1)
    _pull_leaked_pair(tmp_path, seed, store, when=when)
    copy = seed.get("p_closet_alice")
    assert copy is not None and copy.related_to == ["p"]
    assert "p_closet_alice" not in _marked_ids(db)

    engine = seed if engine_name == "seed" else _bloom(db)
    replay = _memory("p_closet_alice", copy.content, related_to=[])  # no link, same text
    engine.ingest(replay)

    assert engine.get("p_closet_alice").related_to == ["p"]

    assert forget(engine, tmp_path, "p", tombstones=store).deleted is True
    assert engine.get("p_closet_alice") is None
    assert not engine.retrieve(SECRET, limit=10)
    assert "p_closet_alice" in _pending_ids(store)
    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert not any(SECRET in json.dumps(r) for r in client.upserts if r["id"] != "p")


@pytest.mark.parametrize("engine_name", ["seed", "bloom"])
def test_a_same_text_write_does_not_invent_a_link_for_a_real_lookalike(tmp_path: Path, engine_name: str) -> None:
    """Only a row the parent PROVES keeps its stored link; a real note takes the caller's."""
    db = tmp_path / "memories.db"
    engine = SeedEngine(db_path=db) if engine_name == "seed" else _bloom(db)
    engine.ingest(_memory("p", _turns()))
    engine.ingest(_memory("p_closet_notes", "MY OWN NOTE", related_to=["somewhere-else"]))

    engine.ingest(_memory("p_closet_notes", "MY OWN NOTE", related_to=[]))

    assert engine.get("p_closet_notes").related_to == []


# --- round 23: a dry run reads, it does not migrate -------------


@pytest.mark.parametrize("command", [["sync", "run"], ["sync", "push"], ["sync", "pull"]])
def test_audit_sync_cli_dry_run_does_not_migrate_legacy_store(tmp_path: Path, command: list[str]) -> None:
    """Constructing an engine is what runs the one-time migration, and a dry run constructs one.

    It classified every copy-shaped row, rewrote the proven ones and queued cloud
    cleanup announcements before ``dry_run`` was ever looked at. Lossless and
    unsent, but a dry run has to be read-only.
    """
    from click.testing import CliRunner

    from poppy.cli.main import cli

    poppy_dir = tmp_path / ".poppy"
    poppy_dir.mkdir()
    db = _legacy_store(poppy_dir)
    assert MARKER_COLUMN not in {r[1] for r in _rows(db, "PRAGMA table_info(memories)")}
    # A dead local url, so a regression fails on the assertions rather than
    # reaching a real host.
    (poppy_dir / "config.json").write_text(
        json.dumps({"trags_api_key": "usr_test", "trags_api_url": "http://127.0.0.1:9", "engine": "seed"})
    )

    env = {"HOME": str(tmp_path), "POPPY_DIR": str(poppy_dir), "POPPY_TELEMETRY_OFF": "1"}
    result = CliRunner().invoke(cli, [*command, "--dry-run"], env=env)

    assert result.exit_code == 0, result.output
    assert "one-time marker migration" in result.output
    assert MARKER_COLUMN not in {r[1] for r in _rows(db, "PRAGMA table_info(memories)")}
    assert _rows(db, "SELECT id FROM legacy_closet_ids") == []  # nothing queued for the cloud
    assert _backup_rows(db) == {}
    assert _all_ids(db) == ["sess-2026-01", "sess-2026-01_closet_alice", "sess-2026-01_closet_bob"]


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


# --- round 24: review round 1 of PR #85 -------------------------


def test_a_pulled_copy_snapshot_also_announces_the_cloud_cleanup(tmp_path: Path) -> None:
    """Recording the deletion locally is half the job: the cloud row still holds the text.

    The remote row is soft-deleted but its BODY is the speaker turns. On main the
    entry went to Trash and the parent's own forget graded it and queued the
    delete; filing it content-free and stopping there left that text up there for
    good, and a device pulling it after the parent was gone graded it ORPHAN and
    offered it as restorable.
    """
    from poppy.write_flow import restore

    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=30)
    parent, copy = _leaked_pair(when)
    copy["deleted_at"] = copy["updated_at"] = (when + timedelta(hours=1)).isoformat()
    pull(engine=seed, tombstones=store, client=_RecordingClient([parent, copy]), state=SyncState(), poppy_dir=tmp_path)

    assert "p_closet_alice" in _pending_ids(store)

    assert forget(seed, tmp_path, "p", tombstones=store).deleted is True
    cloud = _FreshnessClient([parent, copy])
    push(engine=seed, tombstones=store, client=cloud, state=SyncState(), poppy_dir=tmp_path)

    # The cloud's copy of the speaker text is replaced by the placeholder.
    assert cloud.ignored == []
    assert SECRET not in cloud.state["p_closet_alice"]["content"]
    assert cloud.state["p_closet_alice"]["deleted_at"] is not None

    # And the cleaned row, re-pulled after the local record has aged out, is not
    # restorable: the claim outlives the record and still names the id.
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
    engine.ingest(_memory("p", _turns()))
    copy_text = engine.get("p_closet_alice").content  # exactly what p derives
    engine_copy_updated_at = engine.get("p_closet_alice").updated_at
    legacy = store.add(engine.get("p_closet_alice"), tombstoned_at=twelve)
    assert legacy.tombstoned_at == twelve
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM memories WHERE id = 'p_closet_alice'")
    conn.execute("DELETE FROM memory_embeddings WHERE id = 'p_closet_alice'")
    conn.commit()
    conn.close()
    store.add_closets(["p_closet_alice"], now=ten)

    with pytest.raises(ValueError, match="derived per-speaker copy"):
        restore(engine, tmp_path, "p_closet_alice", tombstones=store)
    assert store.get_closet("p_closet_alice").tombstoned_at == ten  # the older record
    # The claim is the LATER of the entry's own two stamps, so the server's freshness
    # gate cannot answer it stale.
    claimed = datetime.fromisoformat(dict(store.pending_legacy_announcements())["p_closet_alice"])
    assert claimed == max(twelve, engine_copy_updated_at)

    captured = _cloud_row("p_closet_alice", copy_text, when=eleven)
    captured["created_at"] = engine.get("p").created_at.isoformat()
    captured["related_to"] = ["p"]
    result = pull(
        engine=engine, tombstones=store, client=_RecordingClient([captured]), state=SyncState(), poppy_dir=tmp_path
    )

    assert result.applied_live == 0
    assert result.skipped_closets == 1
    assert engine.get("p_closet_alice") is None
    assert "p_closet_alice" in _pending_ids(store)  # the announcement survives


def test_a_future_stamped_claim_does_not_hide_a_later_cloud_row(tmp_path: Path) -> None:
    """A claim is an existence hint, never a comparator.

    ``note_leaked_cloud_copy`` advances a claim from an incoming row's own
    updated_at, and a 0.2.4 client could put any stamp there. Used as a floor — even
    clamped to now — a claim dated ten days ahead hid every honestly stamped row
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
    ``refuse_if_derived_copy`` and ``_clear_unmarked_copies``, which both leave
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
    # the watermark frozen, and the announcement queue still drained this cycle.
    assert result.errors == 1 and result.sent_live == 0
    state = remote_state_for(tmp_path, client.base_url)
    assert "local read failed" in state.errors["push"]
    assert state.last_pushed_at is None
    assert [r["id"] for r in client.upserts] == ["legacy_closet_alice"]
    assert "legacy_closet_alice" not in _pending_ids(store)


def test_an_authoritative_deletion_raises_a_local_floor(tmp_path: Path) -> None:
    """The docstring claims it raises over a LOCAL record too, so prove it — up to now."""
    db = tmp_path / "memories.db"
    engine, store = _bloom(db), TombstoneStore(db)
    engine.ingest(_memory("sess-1", _turns()))
    assert forget(engine, tmp_path, "sess-1", tombstones=store).deleted is True
    assert _rows(db, "SELECT is_local FROM closet_tombstones WHERE id = 'sess-1_closet_alice'") == [(1,)]
    old = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(days=3)
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE closet_tombstones SET tombstoned_at = ?", (old.isoformat(),))
    conn.commit()
    conn.close()

    later = old + timedelta(days=1)
    cleanup = _cloud_row("sess-1_closet_alice", CLOSET_TOMBSTONE_CONTENT, deleted=True, when=later)
    pull(engine=engine, tombstones=store, client=_RecordingClient([cleanup]), state=SyncState(), poppy_dir=tmp_path)

    assert store.get_closet("sess-1_closet_alice").tombstoned_at == later
    # ... and an older cleanup still cannot lower it.
    older = _cloud_row("sess-1_closet_alice", CLOSET_TOMBSTONE_CONTENT, deleted=True, when=old)
    pull(engine=engine, tombstones=store, client=_RecordingClient([older]), state=SyncState(), poppy_dir=tmp_path)
    assert store.get_closet("sess-1_closet_alice").tombstoned_at == later

    # ... and a placeholder stamped years ahead raises it no further than now: the
    # timestamp is whatever the announcing client wrote.
    ahead = _cloud_row(
        "sess-1_closet_alice",
        CLOSET_TOMBSTONE_CONTENT,
        deleted=True,
        when=datetime.now(timezone.utc) + timedelta(days=365),
    )
    before = datetime.now(timezone.utc)
    pull(engine=engine, tombstones=store, client=_RecordingClient([ahead]), state=SyncState(), poppy_dir=tmp_path)
    raised = store.get_closet("sess-1_closet_alice").tombstoned_at
    assert before <= raised <= datetime.now(timezone.utc)


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


# --- round 25: review round 2 of PR #85 -------------------------


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
    engine.ingest(_memory("p", _turns()))
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
    store.add_closets(["p_closet_alice"], now=datetime.now(timezone.utc) - timedelta(days=1))
    store.claim_leaked_copy("p_closet_alice", datetime.now(timezone.utc) + timedelta(days=10))

    note = _cloud_row("p_closet_alice", "AN INDEPENDENT NOTE", when=datetime.now(timezone.utc) - timedelta(hours=1))
    result = pull(engine=seed, tombstones=store, client=_RecordingClient([note]), state=SyncState(), poppy_dir=tmp_path)

    assert result.skipped_closets == 0
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

    assert result.skipped_closets == 1
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


# --- round 26: review round 3 of PR #85 -------------------------


def test_a_republished_leak_seen_after_the_record_expired_is_announced_again(tmp_path: Path) -> None:
    """Suppressing the copy is half the job: the cloud is still holding it.

    A 0.2.4 client republishes the copy after the local deletion record has aged out.
    Pull skipped it on the strength of the claim and re-armed nothing, so the cloud row
    stayed live with the speaker text for good — where main, having no claim to skip
    on, ingested it and let the parent's forget announce it.
    """
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

    assert result.skipped_closets == 1
    assert seed.get("p_closet_alice") is None
    assert "p_closet_alice" in _pending_ids(store)  # re-armed at the republished stamp

    assert forget(seed, tmp_path, "p", tombstones=store).deleted is True
    cloud = _FreshnessClient([republished])
    push(engine=seed, tombstones=store, client=cloud, state=SyncState(), poppy_dir=tmp_path)

    assert cloud.ignored == []
    assert cloud.state["p_closet_alice"]["content"] == CLOSET_TOMBSTONE_CONTENT
    assert cloud.state["p_closet_alice"]["deleted_at"] is not None


def test_a_proven_copy_newer_than_the_record_is_still_suppressed(tmp_path: Path) -> None:
    """PROVEN means PROVEN, whatever the row is stamped — a record present does not soften it."""
    db = tmp_path / "memories.db"
    seed, store = SeedEngine(db_path=db), TombstoneStore(db)
    when = datetime.now(timezone.utc) - timedelta(days=30)
    _pull_leaked_pair(tmp_path, seed, store, when=when)
    copy_text = seed.get("p_closet_alice").content
    assert forget(seed, tmp_path, "p_closet_alice", tombstones=store).deleted is True
    record = store.get_closet("p_closet_alice")
    assert record is not None

    ahead = _cloud_row("p_closet_alice", copy_text, when=datetime.now(timezone.utc) + timedelta(days=3))
    ahead["created_at"] = seed.get("p").created_at.isoformat()
    ahead["related_to"] = ["p"]
    result = pull(
        engine=seed, tombstones=store, client=_RecordingClient([ahead]), state=SyncState(), poppy_dir=tmp_path
    )

    assert result.skipped_closets == 1
    assert result.applied_live == 0
    assert seed.get("p_closet_alice") is None
    claimed = datetime.fromisoformat(dict(store.pending_legacy_announcements())["p_closet_alice"])
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
    assert store.get_closet("p_closet_alice") is not None


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
