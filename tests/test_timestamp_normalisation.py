"""Timestamps are stored in ONE spelling, so text comparisons mean time.

Poppy keeps ISO-8601 timestamps as TEXT and compares that text inside SQLite in
places where the answer is about time: the push watermark filter, the TTL expiry
purge on both engines, Trash's window. A text comparison of two ISO strings
compares WALL CLOCK, so it is only a comparison of instants when every value
carries the same offset — and values with other offsets reach the store for real,
from importers, research harnesses, hand-built rows and other clients.

Two probes come straight from the issue, one per broken site:

  a. a parent pushed with ``updated_at`` at ``12:00+02:00`` (10:00Z), then
     forgotten at ``11:00Z``: the tombstone's ``11:00+00:00`` sorted BELOW the
     watermark as text, so push skipped the newer deletion and the cloud kept the
     redacted parent live;
  b. a memory expiring an hour from now, expressed at ``-12:00``: the stored
     string sorted below ``now`` in UTC, so the gated purge hard-deleted a live
     memory (and its per-speaker copies) on the spot.

The rest cover the fix itself: writes normalise, the one-off rewrite brings
existing rows into line exactly once, and the two compare sites still answer
correctly for a row that reached the store without passing through a write path.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from poppy.engine._timestamps import (
    MIGRATIONS_TABLE,
    REPUSH_MARKER,
    TIMESTAMP_MIGRATION,
    migration_applied,
    normalise_stored_timestamps,
)
from poppy.engine.bloom import BloomEngine
from poppy.engine.seed import SCHEMA as SEED_SCHEMA
from poppy.engine.seed import SeedEngine
from poppy.models import Filters, Memory, Source
from poppy.sync import push
from poppy.sync.client import TragsError
from poppy.sync.state import PUSH_WATERMARK_VERSION, SyncState, load
from poppy.ui.tombstones import TombstoneStore

PLUS_TWO = timezone(timedelta(hours=2))
MINUS_TWELVE = timezone(timedelta(hours=-12))


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


def _seed(db_path: Path) -> SeedEngine:
    return SeedEngine(db_path=db_path)


ENGINES = [pytest.param(_seed, id="seed"), pytest.param(_bloom, id="bloom")]


class _RecordingClient:
    base_url = "https://trags.test"

    def __init__(self) -> None:
        self.upserts: list[dict] = []

    def upsert(self, row: dict) -> None:
        self.upserts.append(dict(row))

    def iter_all_since(self, updated_since=None, *, page_size: int = 100):
        return iter(())

    def ping(self) -> None:
        """The probe a push with nothing to send makes."""
        return None


def _memory(
    mid: str,
    *,
    when: datetime,
    expires_at: datetime | None = None,
    content: str = "the fact itself",
) -> Memory:
    return Memory(
        id=mid,
        content=content,
        memory_type="fact",
        source=Source(type="cli", session_id="s1", timestamp=when),
        project="p1",
        related_to=[],
        created_at=when,
        updated_at=when,
        confidence=1.0,
        expires_at=expires_at,
    )


def _turns() -> str:
    return json.dumps(
        [
            {"speaker": "Alice", "dia_id": "D1", "text": "alice said a thing"},
            {"speaker": "Bob", "dia_id": "D2", "text": "bob said another"},
        ]
    )


def _rows(db: Path, sql: str, params: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _write(db: Path, sql: str, params: tuple = ()) -> None:
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


# --- (a) the push watermark ------------------------------------------------


def test_audit_sync_offset_watermark_does_not_drop_parent_forget(tmp_path: Path) -> None:
    """A deletion newer than the last push must be sent, whatever the parent's offset.

    The parent is written at ``12:00+02:00`` — 10:00Z — and pushed, so the
    watermark is that instant. It is forgotten an hour later at ``11:00Z``. As raw
    text ``2026-07-01T11:00:00+00:00`` is LESS than ``2026-07-01T12:00:00+02:00``,
    so the tombstone fell under push's ``iso <= watermark`` filter and was never
    sent: the cloud kept the memory the user forgot, live and recallable on every
    other device.
    """
    db = tmp_path / "memories.db"
    engine = SeedEngine(db_path=db)
    wrote_at = datetime(2026, 7, 1, 12, 0, tzinfo=PLUS_TWO)
    memory = _memory("sess-2026-01", when=wrote_at)
    engine.ingest(memory)

    first = _RecordingClient()
    push(engine=engine, tombstones=TombstoneStore(db), client=first, state=SyncState(), poppy_dir=tmp_path)
    assert [r["id"] for r in first.upserts] == ["sess-2026-01"]
    # The watermark is the INSTANT that was pushed, in the canonical spelling.
    assert load(tmp_path).remotes["https://trags.test"].last_pushed_at == "2026-07-01T10:00:00+00:00"

    # Forget it an hour after the write (what `write_flow.forget` does: tombstone
    # first so the deletion is restorable and pushable, then drop the row).
    store = TombstoneStore(db)
    store.add(memory, tombstoned_at=datetime(2026, 7, 1, 11, 0, tzinfo=timezone.utc))
    assert engine.delete("sess-2026-01") is True

    second = _RecordingClient()
    result = push(engine=engine, tombstones=store, client=second, state=load(tmp_path), poppy_dir=tmp_path)

    assert result.sent_tombstones == 1
    assert result.skipped == 0
    sent = {r["id"]: r for r in second.upserts}
    assert sent["sess-2026-01"]["deleted_at"] is not None
    # Tombstones never move the live watermark.
    assert load(tmp_path).remotes["https://trags.test"].last_pushed_at == "2026-07-01T10:00:00+00:00"


def test_push_still_skips_everything_at_or_below_the_watermark(tmp_path: Path) -> None:
    """The spelling changed; the watermark semantics did not.

    A second push with nothing new sends nothing and re-sends nothing — the
    ``iso <= watermark`` filter still excludes the boundary row, which is what
    keeps push from paying for one write per local write for ever.
    """
    db = tmp_path / "memories.db"
    engine = SeedEngine(db_path=db)
    engine.ingest(_memory("m1", when=datetime(2026, 7, 1, 12, 0, tzinfo=PLUS_TWO)))
    engine.ingest(_memory("m2", when=datetime(2026, 7, 1, 9, 0, tzinfo=timezone.utc)))

    store = TombstoneStore(db)
    first = _RecordingClient()
    push(engine=engine, tombstones=store, client=first, state=SyncState(), poppy_dir=tmp_path)
    # Oldest instant first: m2 at 09:00Z, then m1 at 10:00Z — the ORDER is by
    # instant, which is what it always claimed to be.
    assert [r["id"] for r in first.upserts] == ["m2", "m1"]

    second = _RecordingClient()
    result = push(engine=engine, tombstones=store, client=second, state=load(tmp_path), poppy_dir=tmp_path)
    assert second.upserts == []
    assert result.sent_live == 0
    assert result.skipped == 2


def test_push_watermark_written_before_the_fix_is_read_as_an_instant(tmp_path: Path) -> None:
    """An existing state file can hold a watermark with a non-UTC offset.

    The one-off store rewrite cannot reach ``sync_state.json``, so push normalises
    the watermark it reads. Without that, the first push after upgrading compares
    canonical row text against a ``+02:00`` watermark and skips the deletion all
    over again.
    """
    db = tmp_path / "memories.db"
    engine = SeedEngine(db_path=db)
    memory = _memory("sess-2026-01", when=datetime(2026, 7, 1, 12, 0, tzinfo=PLUS_TWO))
    engine.ingest(memory)
    store = TombstoneStore(db)
    store.note_remote_memories({memory.id}, "https://trags.test")  # This ID already exists remotely.
    store.add(memory, tombstoned_at=datetime(2026, 7, 1, 11, 0, tzinfo=timezone.utc))
    engine.delete("sess-2026-01")

    # Exactly what the buggy version persisted: the row's own spelling.
    (tmp_path / "sync_state.json").write_text(
        json.dumps({"remotes": {"https://trags.test": {"last_pushed_at": "2026-07-01T12:00:00+02:00"}}})
    )

    client = _RecordingClient()
    result = push(engine=engine, tombstones=store, client=client, state=load(tmp_path), poppy_dir=tmp_path)
    assert result.sent_tombstones == 1
    assert [r["id"] for r in client.upserts] == ["sess-2026-01"]


# --- (b) the TTL expiry purge ---------------------------------------------


@pytest.mark.parametrize("make_engine", ENGINES)
def test_audit_purge_expired_keeps_a_future_expiry_written_in_another_offset(tmp_path: Path, make_engine) -> None:
    """A memory expiring an hour from now must survive the purge, offset or not.

    ``(now + 1h)`` expressed at ``-12:00`` has a wall clock eleven hours BEHIND
    now, so the stored string sorted below ``now`` in UTC and the gated purge
    hard-deleted a live memory immediately.
    """
    db = tmp_path / "memories.db"
    engine = make_engine(db)
    now = datetime.now(timezone.utc)
    expires = (now + timedelta(hours=1)).astimezone(MINUS_TWELVE)
    engine.ingest(_memory("sess-2026-01", when=now, expires_at=expires))

    assert engine.purge_expired() == 0
    survivor = engine.get("sess-2026-01")
    assert survivor is not None
    # The instant is preserved exactly; only the spelling is ours.
    assert survivor.expires_at == expires
    assert _rows(db, "SELECT expires_at FROM memories WHERE id = 'sess-2026-01'")[0][0].endswith("+00:00")


@pytest.mark.parametrize("make_engine", ENGINES)
def test_purge_expired_parses_a_stamp_that_never_went_through_a_write(tmp_path: Path, make_engine) -> None:
    """Belt-and-braces: the purge decides by parsing, not by the stored spelling.

    Writes normalise, so this row is planted directly with ``sqlite3`` — the shape
    of a 0.2.4 client sharing ``~/.poppy``, or any tool writing the store itself.
    """
    db = tmp_path / "memories.db"
    engine = make_engine(db)
    now = datetime.now(timezone.utc)
    engine.ingest(_memory("sess-2026-01", when=now, expires_at=now + timedelta(hours=1)))
    future_elsewhere = (now + timedelta(hours=1)).astimezone(MINUS_TWELVE).isoformat()
    _write(db, "UPDATE memories SET expires_at = ? WHERE id = 'sess-2026-01'", (future_elsewhere,))

    assert make_engine(db).purge_expired() == 0
    assert make_engine(db).get("sess-2026-01") is not None

    # And one that really has expired still goes, so the purge has not simply
    # stopped working.
    past_elsewhere = (now - timedelta(hours=1)).astimezone(MINUS_TWELVE).isoformat()
    _write(db, "UPDATE memories SET expires_at = ? WHERE id = 'sess-2026-01'", (past_elsewhere,))
    assert make_engine(db).purge_expired() == 1
    assert make_engine(db).get("sess-2026-01") is None


def test_purge_expired_keeps_the_speaker_copies_of_a_surviving_parent(tmp_path: Path) -> None:
    """The cascade follows the parent: a parent that is not expired keeps its copies."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    now = datetime.now(timezone.utc)
    expires = (now + timedelta(hours=1)).astimezone(MINUS_TWELVE)
    engine.ingest(_memory("sess-2026-01", when=now, expires_at=expires, content=_turns()))
    copies = [r[0] for r in _rows(db, "SELECT id FROM memories WHERE is_closet >= 1")]
    assert len(copies) == 2

    assert engine.purge_expired() == 0
    assert [r[0] for r in _rows(db, "SELECT id FROM memories WHERE is_closet >= 1")] == copies


def test_purge_expired_leaves_an_unparseable_expiry_alone(tmp_path: Path) -> None:
    """A stamp no clock wrote is not evidence that a memory's life is over.

    A purge is irreversible, so the row stays. (Reading it back through the engine
    raises on the unparseable stamp, as it did before this change, so the row is
    asserted in the store rather than through ``get``.)
    """
    db = tmp_path / "memories.db"
    engine = SeedEngine(db_path=db)
    engine.ingest(_memory("m1", when=datetime.now(timezone.utc), expires_at=datetime.now(timezone.utc)))
    _write(db, "UPDATE memories SET expires_at = 'whenever' WHERE id = 'm1'")

    assert SeedEngine(db_path=db).purge_expired() == 0
    assert _rows(db, "SELECT id FROM memories") == [("m1",)]


# --- write time -----------------------------------------------------------


@pytest.mark.parametrize("make_engine", ENGINES)
def test_every_stored_timestamp_is_written_as_utc_text(tmp_path: Path, make_engine) -> None:
    db = tmp_path / "memories.db"
    engine = make_engine(db)
    when = datetime(2026, 7, 1, 12, 0, tzinfo=PLUS_TWO)
    engine.ingest(_memory("sess-2026-01", when=when, expires_at=datetime(2027, 1, 1, 6, 0, tzinfo=MINUS_TWELVE)))

    stored = _rows(
        db,
        "SELECT source_timestamp, created_at, updated_at, expires_at FROM memories WHERE id = 'sess-2026-01'",
    )[0]
    assert stored == (
        "2026-07-01T10:00:00+00:00",
        "2026-07-01T10:00:00+00:00",
        "2026-07-01T10:00:00+00:00",
        "2027-01-01T18:00:00+00:00",
    )
    # Same instants, read back through the engine.
    round_tripped = engine.get("sess-2026-01")
    assert round_tripped is not None
    assert round_tripped.updated_at == when


def test_trash_snapshot_timestamps_are_written_as_utc_text(tmp_path: Path) -> None:
    """`restore` writes the snapshot back into ``memories``, so it is normalised too."""
    db = tmp_path / "memories.db"
    SeedEngine(db_path=db)
    memory = _memory(
        "sess-2026-01",
        when=datetime(2026, 7, 1, 12, 0, tzinfo=PLUS_TWO),
        expires_at=datetime(2027, 1, 1, 6, 0, tzinfo=MINUS_TWELVE),
    )
    TombstoneStore(db).add(memory, tombstoned_at=datetime(2026, 7, 1, 11, 0, tzinfo=timezone.utc))

    stored = _rows(
        db,
        "SELECT source_timestamp, created_at, updated_at, tombstoned_at, memory_expires_at FROM ui_tombstones",
    )[0]
    assert stored == (
        "2026-07-01T10:00:00+00:00",
        "2026-07-01T10:00:00+00:00",
        "2026-07-01T10:00:00+00:00",
        "2026-07-01T11:00:00+00:00",
        "2027-01-01T18:00:00+00:00",
    )


# --- the one-off rewrite --------------------------------------------------


def _legacy_store(db: Path) -> None:
    """A store whose rows predate the normalisation, with no engine ever opened.

    ``TombstoneStore`` first so ``ui_tombstones`` and the closet side tables
    exist, then the memories schema by hand: opening an engine is the thing under
    test, and doing it here would run the migration before the deviant rows land.
    """
    TombstoneStore(db)
    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(SEED_SCHEMA)
        conn.execute(
            "INSERT INTO memories (id, content, memory_type, project, source_type, source_session_id,"
            " source_timestamp, confidence, related_to, created_at, updated_at, expires_at)"
            " VALUES ('m1', 'imported note', 'fact', 'p1', 'import', NULL,"
            " '2026-07-01T12:00:00+02:00', 1.0, '[]',"
            " '2026-07-01T12:00:00+02:00', '2026-07-01T12:00:00+02:00', '2027-01-01T06:00:00-12:00')"
        )
        # Already canonical, and a naive stamp that means UTC by convention.
        conn.execute(
            "INSERT INTO memories (id, content, memory_type, project, source_type, source_session_id,"
            " source_timestamp, confidence, related_to, created_at, updated_at, expires_at)"
            " VALUES ('m2', 'local note', 'fact', NULL, 'cli', NULL,"
            " '2026-07-01T09:00:00+00:00', 1.0, '[]',"
            " '2026-07-01T09:00:00+00:00', '2026-07-01T09:00:00+00:00', NULL)"
        )
        conn.execute(
            "INSERT INTO memories (id, content, memory_type, project, source_type, source_session_id,"
            " source_timestamp, confidence, related_to, created_at, updated_at, expires_at)"
            " VALUES ('m3', 'naive note', 'fact', NULL, 'cli', NULL,"
            " '2026-07-01T08:00:00', 1.0, '[]', '2026-07-01T08:00:00', '2026-07-01T08:00:00', NULL)"
        )
        conn.execute(
            "INSERT INTO ui_tombstones (id, content, memory_type, project, source_type, source_session_id,"
            " source_timestamp, confidence, related_to, created_at, updated_at, tombstoned_at,"
            " superseded_by, memory_expires_at, token)"
            " VALUES ('gone', 'deleted note', 'fact', NULL, 'cli', NULL,"
            " '2026-06-01T12:00:00+02:00', 1.0, '[]', '2026-06-01T12:00:00+02:00',"
            " '2026-06-01T12:00:00+02:00', '2026-06-01T13:00:00+02:00', NULL, NULL, 'tok')"
        )
        conn.execute(
            "INSERT INTO closet_tombstones (id, tombstoned_at, is_local)"
            " VALUES ('m1_closet_alice', '2026-06-01T13:00:00+02:00', 1)"
        )
        conn.commit()
    finally:
        conn.close()


def test_the_one_off_rewrite_normalises_existing_rows_and_records_itself(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    _legacy_store(db)

    SeedEngine(db_path=db)

    assert _rows(db, "SELECT source_timestamp, created_at, updated_at, expires_at FROM memories ORDER BY id") == [
        (
            "2026-07-01T10:00:00+00:00",
            "2026-07-01T10:00:00+00:00",
            "2026-07-01T10:00:00+00:00",
            "2027-01-01T18:00:00+00:00",
        ),
        (
            "2026-07-01T09:00:00+00:00",
            "2026-07-01T09:00:00+00:00",
            "2026-07-01T09:00:00+00:00",
            None,
        ),
        (
            "2026-07-01T08:00:00+00:00",
            "2026-07-01T08:00:00+00:00",
            "2026-07-01T08:00:00+00:00",
            None,
        ),
    ]
    assert _rows(db, "SELECT tombstoned_at, updated_at FROM ui_tombstones") == [
        ("2026-06-01T11:00:00+00:00", "2026-06-01T10:00:00+00:00")
    ]
    assert _rows(db, "SELECT tombstoned_at FROM closet_tombstones") == [("2026-06-01T11:00:00+00:00",)]

    # Recorded by name, so it runs once per store rather than once per schema
    # shape: the marker column's presence says nothing about timestamps. The
    # second row is the re-push request, since rows push reads were rewritten.
    assert sorted(r[0] for r in _rows(db, f"SELECT name FROM {MIGRATIONS_TABLE}")) == sorted(
        [TIMESTAMP_MIGRATION, REPUSH_MARKER]
    )


def test_the_one_off_rewrite_runs_exactly_once_and_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    _legacy_store(db)

    conn = sqlite3.connect(str(db))
    try:
        assert migration_applied(conn, TIMESTAMP_MIGRATION) is False
        # Two memories, one Trash snapshot, one closet deletion record. The
        # memory already stored in canonical spelling is not rewritten.
        assert normalise_stored_timestamps(conn) == 4
        assert migration_applied(conn, TIMESTAMP_MIGRATION) is True
        applied_at = conn.execute(
            f"SELECT applied_at FROM {MIGRATIONS_TABLE} WHERE name = ?", (TIMESTAMP_MIGRATION,)
        ).fetchone()[0]
        # Recorded: a second call is a no-op, and so is a later engine open.
        assert normalise_stored_timestamps(conn) == 0
    finally:
        conn.close()

    SeedEngine(db_path=db)
    assert _rows(db, f"SELECT applied_at FROM {MIGRATIONS_TABLE} WHERE name = '{TIMESTAMP_MIGRATION}'") == [
        (applied_at,)
    ]
    assert applied_at.endswith("+00:00")


def test_the_one_off_rewrite_leaves_an_unparseable_stamp_verbatim(tmp_path: Path) -> None:
    """A value no clock wrote is left exactly as it is rather than mangled."""
    db = tmp_path / "memories.db"
    _legacy_store(db)
    _write(db, "UPDATE memories SET updated_at = 'not a time' WHERE id = 'm1'")

    SeedEngine(db_path=db)

    assert _rows(db, "SELECT updated_at, created_at FROM memories WHERE id = 'm1'") == [
        ("not a time", "2026-07-01T10:00:00+00:00")
    ]


def test_the_one_off_rewrite_survives_a_store_with_no_trash_table(tmp_path: Path) -> None:
    """A store the UI has never opened has no ``ui_tombstones``; that is not an error."""
    db = tmp_path / "memories.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(SEED_SCHEMA)
        conn.execute(
            "INSERT INTO memories (id, content, memory_type, project, source_type, source_session_id,"
            " source_timestamp, confidence, related_to, created_at, updated_at, expires_at)"
            " VALUES ('m1', 'note', 'fact', NULL, 'cli', NULL, '2026-07-01T12:00:00+02:00', 1.0, '[]',"
            " '2026-07-01T12:00:00+02:00', '2026-07-01T12:00:00+02:00', NULL)"
        )
        conn.commit()
    finally:
        conn.close()

    SeedEngine(db_path=db)

    assert _rows(db, "SELECT updated_at FROM memories") == [("2026-07-01T10:00:00+00:00",)]


class _FailsOnRecord:
    """A connection that refuses the bookkeeping INSERT, after the rewrites ran."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def __getattr__(self, name: str):
        return getattr(self._conn, name)

    def execute(self, sql: str, *args):
        if MIGRATIONS_TABLE in sql and "INSERT" in sql:
            raise sqlite3.OperationalError("disk I/O error")
        return self._conn.execute(sql, *args)


def test_the_rewrite_and_its_record_land_together(tmp_path: Path) -> None:
    """A failure part way leaves no record, so the next open retries.

    The transaction is what guarantees it: without one, a store could be marked
    migrated while half its rows still carried the old spelling, and nothing
    would ever revisit them.
    """
    db = tmp_path / "memories.db"
    _legacy_store(db)

    conn = sqlite3.connect(str(db))
    try:
        with pytest.raises(sqlite3.OperationalError):
            normalise_stored_timestamps(_FailsOnRecord(conn))  # type: ignore[arg-type]
    finally:
        conn.close()

    # Nothing recorded, and nothing rewritten either.
    assert _rows(db, f"SELECT name FROM sqlite_master WHERE type='table' AND name='{MIGRATIONS_TABLE}'") == []
    assert _rows(db, "SELECT updated_at FROM memories WHERE id = 'm1'") == [("2026-07-01T12:00:00+02:00",)]

    SeedEngine(db_path=db)
    assert _rows(db, "SELECT updated_at FROM memories WHERE id = 'm1'") == [("2026-07-01T10:00:00+00:00",)]


# --- out-of-range stamps: normalise must never raise (review round 1) -----


OUT_OF_RANGE = (
    # Accepted by `fromisoformat`, but year 10000 / year 0 once shifted to UTC.
    "9999-12-31T23:59:59-05:00",
    "0001-01-01T00:00:00+02:00",
)


@pytest.mark.parametrize("stamp", OUT_OF_RANGE)
@pytest.mark.parametrize("make_engine", ENGINES)
def test_an_out_of_range_stamp_does_not_brick_the_store(tmp_path: Path, make_engine, stamp: str) -> None:
    """A stamp whose UTC form is unrepresentable is left verbatim, not raised on.

    ``datetime.astimezone`` raises OverflowError for these, and the rewrite runs
    inside the engine constructor: an escaping exception rolls the migration back
    and re-raises, so EVERY later open of that store fails and the CLI is dead.
    ``parse_expires_at`` refuses such a value at the write boundary; a row that is
    already in the store has to be survivable.
    """
    db = tmp_path / "memories.db"
    _legacy_store(db)
    _write(db, "UPDATE memories SET expires_at = ? WHERE id = 'm1'", (stamp,))

    engine = make_engine(db)  # must not raise

    assert _rows(db, "SELECT expires_at FROM memories WHERE id = 'm1'") == [(stamp,)]
    # Still openable afterwards, and the purge does not choke on it either.
    assert make_engine(db).purge_expired() in (0, 1)
    probe = sqlite3.connect(str(db))
    try:
        assert migration_applied(probe, TIMESTAMP_MIGRATION) is True
    finally:
        probe.close()
    del engine


@pytest.mark.parametrize("stamp", OUT_OF_RANGE)
def test_an_out_of_range_stamp_survives_ingest_and_trash(tmp_path: Path, stamp: str) -> None:
    """The two other paths through ``utc_iso``: an engine write and a Trash snapshot."""
    db = tmp_path / "memories.db"
    engine = SeedEngine(db_path=db)
    edge = datetime.fromisoformat(stamp)
    memory = _memory("m1", when=datetime.now(timezone.utc), expires_at=edge)

    engine.ingest(memory)  # must not raise
    TombstoneStore(db).add(memory, tombstoned_at=datetime(2026, 7, 1, 11, 0, tzinfo=timezone.utc))

    assert _rows(db, "SELECT expires_at FROM memories WHERE id = 'm1'") == [(stamp,)]
    assert _rows(db, "SELECT memory_expires_at FROM ui_tombstones WHERE id = 'm1'") == [(stamp,)]


@pytest.mark.parametrize("stamp", OUT_OF_RANGE)
def test_the_write_boundary_refuses_an_out_of_range_expiry(tmp_path: Path, stamp: str) -> None:
    """`remember --expires-at` is rejected with a readable error, not stored."""
    from click.testing import CliRunner

    from poppy.cli.main import cli
    from poppy.lifecycle import parse_expires_at

    with pytest.raises(ValueError, match="outside the range Poppy can store"):
        parse_expires_at(stamp)

    result = CliRunner().invoke(
        cli,
        ["remember", "a note", "--expires-at", stamp],
        env={"HOME": str(tmp_path), "POPPY_DIR": str(tmp_path / ".poppy"), "POPPY_TELEMETRY_OFF": "1"},
    )
    assert result.exit_code != 0
    assert "outside the range Poppy can store" in result.output


def test_the_write_boundary_refuses_a_ttl_past_the_end_of_the_calendar(tmp_path: Path) -> None:
    from poppy.lifecycle import resolve_expiry

    with pytest.raises(ValueError, match="lands after year 9999"):
        resolve_expiry("9999999d", None)


# --- the re-push after a rewrite (review round 1) --------------------------


def _store_with_two_rows(db: Path, *, b_updated: str) -> None:
    """A legacy store holding A (already canonical) and B (spelled by the caller)."""
    TombstoneStore(db)
    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(SEED_SCHEMA)
        for mid, updated in (("A", "2026-07-01T10:00:00+00:00"), ("B", b_updated)):
            conn.execute(
                "INSERT INTO memories (id, content, memory_type, project, source_type, source_session_id,"
                " source_timestamp, confidence, related_to, created_at, updated_at, expires_at)"
                " VALUES (?, ?, 'fact', NULL, 'cli', NULL, ?, 1.0, '[]', ?, ?, NULL)",
                (mid, f"body of {mid}", updated, updated, updated),
            )
        conn.commit()
    finally:
        conn.close()


def _state(tmp_path: Path, *, url: str = "https://trags.test", **values) -> None:
    (tmp_path / "sync_state.json").write_text(json.dumps({"remotes": {url: values}}))


def _remote(tmp_path: Path, url: str = "https://trags.test"):
    return load(tmp_path).remotes[url]


def test_a_rewrite_that_moves_a_push_candidate_forces_one_full_repush(tmp_path: Path) -> None:
    """The upgrade must not abandon a row that was still waiting to be uploaded.

    The old client pushed A at ``10:00+00:00`` (so the watermark is that string)
    and FAILED on B at ``11:00+02:00`` — 09:00Z, but larger as text, which is
    exactly why the old ordering kept retrying it. Re-spelled to ``09:00+00:00``,
    B falls below the watermark and would be skipped for ever, with push reporting
    no errors. The store therefore records that the rewrite moved a push candidate,
    and each remote makes one pass for that event.
    """
    db = tmp_path / "memories.db"
    _store_with_two_rows(db, b_updated="2026-07-01T11:00:00+02:00")
    _state(tmp_path, last_pushed_at="2026-07-01T10:00:00+00:00")

    engine = SeedEngine(db_path=db)  # runs the rewrite, records the event
    store = TombstoneStore(db)
    stamp = store.repush_stamp()
    assert stamp is not None
    assert _rows(db, "SELECT updated_at FROM memories WHERE id = 'B'") == [("2026-07-01T09:00:00+00:00",)]

    client = _RecordingClient()
    result = push(engine=engine, tombstones=store, client=client, state=load(tmp_path), poppy_dir=tmp_path)

    # B is on the server, which is the whole point; A rides along, which is free
    # (the upsert is idempotent and the server's freshness gate settles it).
    assert {r["id"] for r in client.upserts} == {"A", "B"}
    assert result.skipped == 0
    assert result.errors == 0
    # Recorded against THIS remote, by the same save as the watermark.
    assert _remote(tmp_path).repush_done_for == stamp
    assert _remote(tmp_path).last_pushed_at == "2026-07-01T10:00:00+00:00"
    # The store's record is a fact, not a token: it is still there afterwards.
    assert TombstoneStore(db).repush_stamp() == stamp

    # And the pass is not repeated: the next push is an ordinary delta.
    again = _RecordingClient()
    second = push(engine=engine, tombstones=store, client=again, state=load(tmp_path), poppy_dir=tmp_path)
    assert again.upserts == []
    assert second.skipped == 2


def test_a_state_save_that_never_lands_does_not_consume_the_repush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recovery and the watermark are one write, so a crash between them cannot exist.

    A consumable flag had to be cleared by its own write to the database, which
    could never be atomic with the watermark save in ``sync_state.json``: a Ctrl-C,
    a crash or a full disk in between left the flag cleared and the row unsent, for
    ever. Here the marker IS part of that save.
    """
    db = tmp_path / "memories.db"
    _store_with_two_rows(db, b_updated="2026-07-01T11:00:00+02:00")
    _state(tmp_path, last_pushed_at="2026-07-01T10:00:00+00:00")
    engine = SeedEngine(db_path=db)
    store = TombstoneStore(db)
    stamp = store.repush_stamp()

    def explode(*_args, **_kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr("poppy.sync.state.save", explode)
    client = _RecordingClient()
    with pytest.raises(OSError):
        push(engine=engine, tombstones=store, client=client, state=load(tmp_path), poppy_dir=tmp_path)
    monkeypatch.undo()

    # Nothing was recorded, so the next push owes the same pass — B is not lost.
    assert _remote(tmp_path).repush_done_for is None
    again = _RecordingClient()
    push(engine=engine, tombstones=store, client=again, state=load(tmp_path), poppy_dir=tmp_path)
    assert {r["id"] for r in again.upserts} == {"A", "B"}
    assert _remote(tmp_path).repush_done_for == stamp


def test_a_row_failing_during_the_repush_retries_without_a_second_full_pass(tmp_path: Path) -> None:
    """The frozen watermark and the marker are saved together, then ordinary retry.

    A fails in the middle of the recovery pass. B, below it, is proven sent, so the
    watermark freezes there and the marker is recorded with it: the next push is a
    plain delta that retries A alone.
    """
    db = tmp_path / "memories.db"
    _store_with_two_rows(db, b_updated="2026-07-01T11:00:00+02:00")
    _state(tmp_path, last_pushed_at="2026-07-01T10:00:00+00:00")
    engine = SeedEngine(db_path=db)
    store = TombstoneStore(db)
    stamp = store.repush_stamp()

    class _RefusesA(_RecordingClient):
        def upsert(self, row: dict) -> None:
            if row["id"] == "A":
                raise TragsError("500 from the server")
            super().upsert(row)

    failing = _RefusesA()
    result = push(engine=engine, tombstones=store, client=failing, state=load(tmp_path), poppy_dir=tmp_path)
    assert [r["id"] for r in failing.upserts] == ["B"]
    assert result.errors == 1
    # Progress recorded at B's instant, with the marker, in one save.
    assert _remote(tmp_path).last_pushed_at == "2026-07-01T09:00:00+00:00"
    assert _remote(tmp_path).repush_done_for == stamp

    working = _RecordingClient()
    second = push(engine=engine, tombstones=store, client=working, state=load(tmp_path), poppy_dir=tmp_path)
    assert [r["id"] for r in working.upserts] == ["A"]  # A alone, not a second full pass
    assert second.skipped == 1
    assert _remote(tmp_path).last_pushed_at == "2026-07-01T10:00:00+00:00"


def test_every_remote_makes_its_own_recovery_pass(tmp_path: Path) -> None:
    """The watermark is per remote, so the recovery has to be too.

    A single consumable flag in the store let the first remote to sync consume it
    while every other remote kept its stale watermark and skipped B for ever.
    """
    db = tmp_path / "memories.db"
    _store_with_two_rows(db, b_updated="2026-07-01T11:00:00+02:00")
    (tmp_path / "sync_state.json").write_text(
        json.dumps(
            {
                "remotes": {
                    "https://trags.test": {"last_pushed_at": "2026-07-01T10:00:00+00:00"},
                    "https://other.test": {"last_pushed_at": "2026-07-01T10:00:00+00:00"},
                }
            }
        )
    )
    engine = SeedEngine(db_path=db)
    store = TombstoneStore(db)
    stamp = store.repush_stamp()

    first = _RecordingClient()
    push(engine=engine, tombstones=store, client=first, state=load(tmp_path), poppy_dir=tmp_path)
    assert {r["id"] for r in first.upserts} == {"A", "B"}

    second_client = _RecordingClient()
    second_client.base_url = "https://other.test"
    push(engine=engine, tombstones=store, client=second_client, state=load(tmp_path), poppy_dir=tmp_path)

    assert {r["id"] for r in second_client.upserts} == {"A", "B"}
    assert _remote(tmp_path, "https://trags.test").repush_done_for == stamp
    assert _remote(tmp_path, "https://other.test").repush_done_for == stamp


def test_a_rewrite_that_changed_nothing_does_not_force_a_repush(tmp_path: Path) -> None:
    """A store already spelled in UTC pays nothing for the migration."""
    db = tmp_path / "memories.db"
    _store_with_two_rows(db, b_updated="2026-07-01T09:00:00+00:00")
    # Written by a client of the current watermark generation, so the ONE-TIME pass
    # that generation buys is not owed either: this isolates the rewrite.
    _state(tmp_path, last_pushed_at="2026-07-01T10:00:00+00:00", push_watermark_v=PUSH_WATERMARK_VERSION)

    engine = SeedEngine(db_path=db)
    store = TombstoneStore(db)
    assert store.repush_stamp() is None

    client = _RecordingClient()
    result = push(engine=engine, tombstones=store, client=client, state=load(tmp_path), poppy_dir=tmp_path)
    assert client.upserts == []
    assert result.skipped == 2
    assert _remote(tmp_path).repush_done_for is None


def test_a_repush_is_not_requested_for_a_rewrite_confined_to_the_side_tables(tmp_path: Path) -> None:
    """Push reads ``memories`` and ``ui_tombstones``; the closet tables are not candidates."""
    db = tmp_path / "memories.db"
    TombstoneStore(db)
    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(SEED_SCHEMA)
        conn.execute(
            "INSERT INTO closet_tombstones (id, tombstoned_at, is_local)"
            " VALUES ('x_closet_alice', '2026-06-01T13:00:00+02:00', 1)"
        )
        conn.commit()
    finally:
        conn.close()

    SeedEngine(db_path=db)

    assert _rows(db, "SELECT tombstoned_at FROM closet_tombstones") == [("2026-06-01T11:00:00+00:00",)]
    assert TombstoneStore(db).repush_stamp() is None


def test_a_dry_run_push_reports_the_repush_without_recording_it(tmp_path: Path) -> None:
    """`sync --dry-run` must describe the pass that is coming and change nothing."""
    db = tmp_path / "memories.db"
    _store_with_two_rows(db, b_updated="2026-07-01T11:00:00+02:00")
    _state(tmp_path, last_pushed_at="2026-07-01T10:00:00+00:00")
    engine = SeedEngine(db_path=db)
    store = TombstoneStore(db)

    client = _RecordingClient()
    result = push(
        engine=engine, tombstones=store, client=client, state=load(tmp_path), poppy_dir=tmp_path, dry_run=True
    )

    assert client.upserts == []
    assert result.sent_live == 2  # the full pass, not the delta of one
    assert result.skipped == 0
    assert _remote(tmp_path).repush_done_for is None
    assert _remote(tmp_path).last_pushed_at == "2026-07-01T10:00:00+00:00"


# --- a watermark whose own spelling hides rows (review round 2) ------------


def test_a_watermark_with_a_negative_offset_is_not_trusted_either(tmp_path: Path) -> None:
    """Normalising a NEGATIVE-offset watermark moves it forward and can hide a row.

    ``2026-07-02T10:00:00-05:00`` is 15:00Z. Under the old text ordering a pending
    canonical row at ``11:00+00:00`` sorted ABOVE it (``11`` > ``10``) and was being
    retried; normalised, the watermark reads ``15:00+00:00`` and that row falls
    under it — so the spelling fix would bury a row no rewrite had touched. Such a
    watermark buys one full pass too, after which the saved one is canonical and
    this never fires for the remote again.
    """
    db = tmp_path / "memories.db"
    # Already canonical, so nothing is rewritten and no repush event is recorded:
    # this isolates the watermark's own spelling.
    _store_with_two_rows(db, b_updated="2026-07-02T11:00:00+00:00")
    _state(tmp_path, last_pushed_at="2026-07-02T10:00:00-05:00")
    engine = SeedEngine(db_path=db)
    store = TombstoneStore(db)
    assert store.repush_stamp() is None

    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=load(tmp_path), poppy_dir=tmp_path)

    assert "B" in {r["id"] for r in client.upserts}
    saved = _remote(tmp_path).last_pushed_at
    assert saved == "2026-07-02T11:00:00+00:00"  # canonical from here on

    # Fires once: the next push is an ordinary delta.
    again = _RecordingClient()
    second = push(engine=engine, tombstones=store, client=again, state=load(tmp_path), poppy_dir=tmp_path)
    assert again.upserts == []
    assert second.skipped == 2


def test_a_naive_watermark_also_buys_one_pass_and_then_is_canonical(tmp_path: Path) -> None:
    """The same rule covers a legacy naive watermark, which means UTC but is spelled otherwise."""
    db = tmp_path / "memories.db"
    _store_with_two_rows(db, b_updated="2026-07-01T09:00:00+00:00")
    _state(tmp_path, last_pushed_at="2026-07-01T10:00:00")

    engine = SeedEngine(db_path=db)
    store = TombstoneStore(db)
    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=load(tmp_path), poppy_dir=tmp_path)

    assert {r["id"] for r in client.upserts} == {"A", "B"}
    assert _remote(tmp_path).last_pushed_at == "2026-07-01T10:00:00+00:00"

    again = _RecordingClient()
    push(engine=engine, tombstones=store, client=again, state=load(tmp_path), poppy_dir=tmp_path)
    assert again.upserts == []


# --- the `since` filter shares the one spelling ---------------------------


@pytest.mark.parametrize("make_engine", ENGINES)
def test_the_since_filter_takes_a_bound_in_any_offset(tmp_path: Path, make_engine) -> None:
    """``Filters.since`` is compared with stored text, so it goes through the same helper."""
    db = tmp_path / "memories.db"
    engine = make_engine(db)
    engine.ingest(_memory("old", when=datetime(2026, 7, 1, 8, 0, tzinfo=timezone.utc)))
    engine.ingest(_memory("new", when=datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)))

    # 12:00+02:00 is 10:00Z: `old` is below it, `new` is above. Compared as raw
    # text the bound would read 12:00 and hide `new` as well.
    bound = datetime(2026, 7, 1, 12, 0, tzinfo=PLUS_TWO)
    listed = {m.id for m in make_engine(db).list_all(filters=Filters(since=bound), limit=50)}
    assert listed == {"new"}


# --- the purge takes the write lock before it reads (review round 1) -------


class _RecordsTransactionState:
    """Connection proxy that notes whether a transaction was open per statement."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.seen: list[tuple[str, bool]] = []

    def __getattr__(self, name: str):
        return getattr(self._conn, name)

    def execute(self, sql: str, *args):
        self.seen.append((sql, self._conn.in_transaction))
        return self._conn.execute(sql, *args)


def test_bloom_purge_reads_the_expiring_rows_under_the_write_lock(tmp_path: Path) -> None:
    """The SELECT and the DELETEs are one atomic decision.

    The purge decides in Python now, so the two statements are separate: a
    ``poppy edit --ttl`` landing in between would push a memory's expiry out and
    still have it deleted, on the strength of the expiry it just replaced. The
    lock must therefore be held before the rows are read, which both engines now
    get from the shared ``poppy.db.write_txn`` helper.
    """
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    now = datetime.now(timezone.utc)
    engine.ingest(_memory("m1", when=now, expires_at=now - timedelta(hours=1)))

    proxy = _RecordsTransactionState(engine._conn)
    engine._conn = proxy  # type: ignore[assignment]
    assert engine.purge_expired() == 1

    selects = [(sql, in_txn) for sql, in_txn in proxy.seen if sql.startswith("SELECT id,")]
    assert selects, "the purge no longer reads the expiring rows with this statement"
    assert all(in_txn for _sql, in_txn in selects), "the expiring rows were read outside the write lock"
    assert _rows(db, "SELECT id FROM memories") == []


def test_bloom_purge_releases_the_write_lock_when_a_delete_fails(tmp_path: Path) -> None:
    """A raising purge must not leave every other connection blocked on the lock."""
    db = tmp_path / "memories.db"
    engine = _bloom(db)
    now = datetime.now(timezone.utc)
    engine.ingest(_memory("m1", when=now, expires_at=now - timedelta(hours=1)))

    class _FailsOnDelete(_RecordsTransactionState):
        def execute(self, sql: str, *args):
            if sql.startswith("DELETE FROM memories"):
                raise sqlite3.OperationalError("disk I/O error")
            return super().execute(sql, *args)

    real = engine._conn
    engine._conn = _FailsOnDelete(real)  # type: ignore[assignment]
    with pytest.raises(sqlite3.OperationalError):
        engine.purge_expired()

    assert real.in_transaction is False
    assert _rows(db, "SELECT id FROM memories") == [("m1",)]


# --- colliding state keys must not launder the watermark (review round 3) --


def _duplicate_key_state(tmp_path: Path, *, first: dict, second: dict) -> None:
    """A pre-v0.2.2 state file holding the same remote under both spellings of its URL."""
    (tmp_path / "sync_state.json").write_text(
        json.dumps({"remotes": {"https://trags.test": first, "https://trags.test/": second}})
    )


def test_merging_duplicate_keys_keeps_the_watermark_spelling_verbatim(tmp_path: Path) -> None:
    """The merge must not launder a non-canonical watermark into a canonical one.

    Two keys collapse into one on load. Picking the later INSTANT is right; storing
    it CANONICALISED is not, because push decides it owes a recovery pass by seeing
    that the stored spelling is not canonical. A stored ``10:00-05:00`` (15:00Z)
    came back as ``15:00+00:00``, so the check saw nothing to do and a pending
    tombstone at ``11:00+00:00`` stayed buried under the watermark until Trash aged
    it out seven days later.
    """
    db = tmp_path / "memories.db"
    engine = SeedEngine(db_path=db)
    memory = _memory("sess-2026-01", when=datetime(2026, 7, 2, 8, 0, tzinfo=timezone.utc))
    engine.ingest(memory)
    store = TombstoneStore(db)
    store.note_remote_memories({memory.id}, "https://trags.test")  # This ID already exists remotely.
    store.add(memory, tombstoned_at=datetime(2026, 7, 2, 11, 0, tzinfo=timezone.utc))
    engine.delete("sess-2026-01")
    # Nothing was rewritten, so there is no re-push marker: the watermark's own
    # spelling still requires the recovery pass; per-ID evidence permits sending.
    assert store.repush_stamp() is None

    _duplicate_key_state(
        tmp_path,
        first={"last_pushed_at": "2026-07-02T10:00:00-05:00", "pushed_count": 3},
        second={"last_pushed_at": None},
    )
    merged = load(tmp_path).remotes["https://trags.test"]
    assert merged.last_pushed_at == "2026-07-02T10:00:00-05:00"  # verbatim, not 15:00+00:00

    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=load(tmp_path), poppy_dir=tmp_path)

    sent = {r["id"]: r for r in client.upserts}
    assert "sess-2026-01" in sent
    assert sent["sess-2026-01"]["deleted_at"] is not None
    # The spelling migration resets the live mark; a tombstone cannot carry it.
    assert load(tmp_path).remotes["https://trags.test"].last_pushed_at is None


def test_merging_duplicate_keys_keeps_the_lower_watermark_generation(tmp_path: Path) -> None:
    """The merged watermark can be the OLDER client's, so the pass is still owed.

    The merge takes the later INSTANT across the two keys. When that value came
    from the key written before the fix, a merged record claiming the newer
    generation would carry exactly the watermark the pass exists to distrust — and
    would retire the pass in the same breath.
    """
    db = tmp_path / "memories.db"
    # Already canonical, so no rewrite is recorded: the generation is the only
    # thing that can force a pass here.
    _store_with_two_rows(db, b_updated="2026-07-01T09:00:00+00:00")
    engine = SeedEngine(db_path=db)
    store = TombstoneStore(db)
    assert store.repush_stamp() is None

    _duplicate_key_state(
        tmp_path,
        first={"last_pushed_at": "2026-07-01T10:00:00+00:00"},  # pre-fix: no generation
        second={"last_pushed_at": "2026-07-01T08:00:00+00:00", "push_watermark_v": PUSH_WATERMARK_VERSION},
    )
    merged = load(tmp_path).remotes["https://trags.test"]
    assert merged.last_pushed_at == "2026-07-01T10:00:00+00:00"  # the pre-fix key's
    assert merged.push_watermark_v == 0  # so the merged remote still owes the pass

    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=load(tmp_path), poppy_dir=tmp_path)

    assert {r["id"] for r in client.upserts} == {"A", "B"}
    assert load(tmp_path).remotes["https://trags.test"].push_watermark_v == PUSH_WATERMARK_VERSION


def test_merging_duplicate_keys_keeps_repush_done_only_when_both_agree(tmp_path: Path) -> None:
    """One key's finished pass says nothing about the other key's watermark.

    The two records carry independent watermarks. A key that holds the stamp with NO
    watermark — its recovery pass failed at the very first row — merged with a key
    holding an old watermark would claim the pass was done and skip every row under
    that watermark for ever.
    """
    db = tmp_path / "memories.db"
    _store_with_two_rows(db, b_updated="2026-07-01T11:00:00+02:00")
    engine = SeedEngine(db_path=db)
    store = TombstoneStore(db)
    stamp = store.repush_stamp()
    assert stamp is not None

    _duplicate_key_state(
        tmp_path,
        first={"last_pushed_at": None, "repush_done_for": stamp},
        second={"last_pushed_at": "2026-07-01T10:00:00+00:00"},
    )
    merged = load(tmp_path).remotes["https://trags.test"]
    assert merged.last_pushed_at == "2026-07-01T10:00:00+00:00"
    assert merged.repush_done_for is None  # the two disagree, so the pass is still owed

    client = _RecordingClient()
    push(engine=engine, tombstones=store, client=client, state=load(tmp_path), poppy_dir=tmp_path)

    assert {r["id"] for r in client.upserts} == {"A", "B"}
    assert load(tmp_path).remotes["https://trags.test"].repush_done_for == stamp


def test_merging_duplicate_keys_keeps_repush_done_when_both_carry_it(tmp_path: Path) -> None:
    """Agreement is kept, so a merged remote does not repeat a pass it has made."""
    db = tmp_path / "memories.db"
    _store_with_two_rows(db, b_updated="2026-07-01T11:00:00+02:00")
    engine = SeedEngine(db_path=db)
    store = TombstoneStore(db)
    stamp = store.repush_stamp()

    _duplicate_key_state(
        tmp_path,
        # Both keys also carry the current watermark generation, so the pass THAT
        # buys is not what a full re-push here would prove.
        first={
            "last_pushed_at": "2026-07-01T10:00:00+00:00",
            "repush_done_for": stamp,
            "push_watermark_v": PUSH_WATERMARK_VERSION,
        },
        second={
            "last_pushed_at": "2026-07-01T09:00:00+00:00",
            "repush_done_for": stamp,
            "push_watermark_v": PUSH_WATERMARK_VERSION,
        },
    )
    assert load(tmp_path).remotes["https://trags.test"].repush_done_for == stamp

    client = _RecordingClient()
    result = push(engine=engine, tombstones=store, client=client, state=load(tmp_path), poppy_dir=tmp_path)
    assert client.upserts == []
    assert result.skipped == 2
