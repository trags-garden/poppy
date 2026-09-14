"""Every bloom write is all-or-nothing, and the encoder runs outside it.

The default engine's ingest is a read-decide-write sequence: it clears the
per-speaker copies of the memory it is about to rewrite, writes the parent, then
re-derives the copies. Embedding sits on that path, and it is the one step that
routinely fails for reasons that have nothing to do with the database — the
fastembed models load lazily, unload when idle, and raise
``ModelUnavailableError`` on a cold cache with no network.

Before the fix the sequence ran with no transaction guard: a failure left the
write lock held on a connection that a long-lived ``poppy serve`` never closes
(every other process then got ``database is locked``), and the NEXT successful
write on that connection committed whatever the failed one had already deleted —
a failed content edit reported as failed, whose speaker copies vanished minutes
later while the parent kept its old text.

These tests are written against the observable contract rather than the
implementation: after any failed write the connection holds no transaction,
another process can still write the store, and the rows are exactly as they were.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from poppy.engine import _closet_engine
from poppy.engine.bloom import BloomEngine
from poppy.errors import ModelUnavailableError
from poppy.models import Memory, Source

OLD_TEXT = "the pumpkin cadenza"
NEW_TEXT = "the rhubarb cadenza"


class _FakeBiEncoder:
    """fastembed TextEmbedding stand-in: .embed(iterable) -> generator of vectors.

    ``crc32`` rather than ``hash`` so the same text gives the same vector in
    every run: ``hash`` is salted per process and can land on the all-zero
    vector.
    """

    def embed(self, texts):
        for t in texts:
            h = zlib.crc32(t.encode()) & 0xFFFF
            yield np.array([h & 0xF, (h >> 4) & 0xF, (h >> 8) & 0xF, (h >> 12) & 0xF], dtype=np.float32)


class _FakeCrossEncoder:
    def rerank(self, query, docs):
        return [float(len(d)) for d in docs]


def _bloom(db_path: Path) -> BloomEngine:
    return BloomEngine(db_path=db_path, bi_encoder=_FakeBiEncoder(), cross_encoder=_FakeCrossEncoder())


def _turns(text: str) -> str:
    """A two-speaker session, which bloom expands into one copy per speaker."""
    return json.dumps(
        [
            {"speaker": "Alice", "dia_id": "D1", "text": text},
            {"speaker": "Bob", "dia_id": "D2", "text": "noted"},
        ]
    )


def _memory(mid: str, content: str) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mid,
        content=content,
        memory_type="fact",
        source=Source(type="cli", session_id=None, timestamp=now),
        project="p1",
        related_to=[],
        created_at=now,
        updated_at=now,
        confidence=1.0,
    )


def _rows(db_path: Path, sql: str, params: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _family(db_path: Path, parent_id: str) -> list[tuple]:
    """The parent row and every per-speaker copy of it, id + text + marker."""
    return _rows(
        db_path,
        "SELECT id, content, enriched_content, is_closet FROM memories WHERE id = ? OR id LIKE ? ORDER BY id",
        (parent_id, parent_id + "_closet_%"),
    )


def _embedding_ids(db_path: Path, parent_id: str) -> list[tuple]:
    """The stored vectors for a memory and its copies, by id."""
    return _rows(
        db_path,
        "SELECT id FROM memory_embeddings WHERE id = ? OR id LIKE ? ORDER BY id",
        (parent_id, parent_id + "_closet_%"),
    )


def _assert_another_process_can_write(db_path: Path) -> None:
    """A SECOND connection must be able to take the write lock within a second.

    This is the user-visible half of the bug: an abandoned open transaction on
    the engine's connection blocks every other Poppy process on the machine
    (CLI, UI, sync worker) with ``database is locked`` until the holder exits.
    The probe row is removed again so it cannot affect any later assertion.
    """
    conn = sqlite3.connect(str(db_path), timeout=1)
    try:
        conn.execute(
            """INSERT INTO memories
               (id, content, enriched_content, memory_type, source_type, source_timestamp,
                created_at, updated_at)
               VALUES ('probe_lock', 'probe', 'probe', 'fact', 'cli', '2026-01-01T00:00:00+00:00',
                       '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')"""
        )
        conn.commit()
        conn.execute("DELETE FROM memories WHERE id = 'probe_lock'")
        conn.commit()
    except sqlite3.OperationalError as exc:
        pytest.fail(f"another connection could not write the store: {exc}")
    finally:
        conn.close()


def _fail_embed(monkeypatch: pytest.MonkeyPatch, engine: BloomEngine) -> None:
    """The real failure mode: the encoder is unavailable when a write needs it."""

    def boom(text: str):
        raise ModelUnavailableError("model cache is cold and the network is unavailable")

    monkeypatch.setattr(engine, "_embed", boom)


def test_ingest_that_cannot_embed_leaves_the_store_writable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed fresh ingest commits nothing and holds no lock."""
    db = tmp_path / "poppy.db"
    engine = _bloom(db)
    engine.ingest(_memory("mem_keep", _turns(OLD_TEXT)))
    before = _family(db, "mem_keep")
    assert len(before) == 3  # parent + Alice + Bob

    _fail_embed(monkeypatch, engine)
    with pytest.raises(ModelUnavailableError):
        engine.ingest(_memory("mem_new", "a plain fact"))

    assert engine._conn.in_transaction is False
    _assert_another_process_can_write(db)
    assert _rows(db, "SELECT id FROM memories WHERE id = 'mem_new'") == []
    assert _family(db, "mem_keep") == before
    engine._conn.close()


def test_failed_content_edit_keeps_both_speaker_copies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The reported case: an edit that cannot embed must not disturb the copies.

    A content edit replays through ingest, which clears the copies before writing
    the new text. If the embed failure happens after that clearing, the user is
    told the edit failed while the copies are already gone.
    """
    db = tmp_path / "poppy.db"
    engine = _bloom(db)
    engine.ingest(_memory("mem_edit", _turns(OLD_TEXT)))
    before = _family(db, "mem_edit")
    assert len(before) == 3
    # The parent holds both speakers' turns; only Alice's copy repeats her text.
    assert sum(OLD_TEXT in row[2] for row in before) == 2

    _fail_embed(monkeypatch, engine)
    with pytest.raises(ModelUnavailableError):
        engine.ingest(_memory("mem_edit", _turns(NEW_TEXT)))

    assert engine._conn.in_transaction is False
    _assert_another_process_can_write(db)
    after = _family(db, "mem_edit")
    assert after == before
    assert len(after) == 3
    assert all(NEW_TEXT not in row[1] for row in after)
    engine._conn.close()


def test_failure_inside_the_write_rolls_the_deletion_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure AFTER the copies are deleted still leaves them standing.

    Moving the embed out of the transaction is not enough on its own: the write
    sequence makes several statements and any of them can fail. Here the failure
    is raised from the tombstone bookkeeping at the end of ingest, long after the
    old copies have been deleted and the new rows written, so only the rollback
    can restore them.
    """
    db = tmp_path / "poppy.db"
    engine = _bloom(db)
    engine.ingest(_memory("mem_edit", _turns(OLD_TEXT)))
    before = _family(db, "mem_edit")

    def boom(*args, **kwargs):
        raise RuntimeError("bookkeeping failed mid-write")

    monkeypatch.setattr(_closet_engine, "record_closet_tombstones", boom)
    with pytest.raises(RuntimeError):
        engine.ingest(_memory("mem_edit", _turns(NEW_TEXT)))

    assert engine._conn.in_transaction is False
    _assert_another_process_can_write(db)
    assert _family(db, "mem_edit") == before
    # The FTS index is written by triggers inside the same transaction, so it
    # rolls back with the rows: one entry per surviving row, none for the text
    # that was never committed.
    assert _rows(db, "SELECT count(*) FROM memory_fts WHERE content LIKE ?", (f"%{NEW_TEXT}%",)) == [(0,)]
    engine._conn.close()


def test_next_successful_ingest_commits_nothing_from_the_failed_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The delayed-damage half of the bug: a later write must not adopt the leftovers.

    With the transaction left open, the deletions from the failed edit sat
    uncommitted on the engine's connection and were committed by the next
    unrelated ``remember`` — copies gone, parent still holding its old text.
    """
    db = tmp_path / "poppy.db"
    engine = _bloom(db)
    engine.ingest(_memory("mem_edit", _turns(OLD_TEXT)))
    before = _family(db, "mem_edit")

    def boom(*args, **kwargs):
        raise RuntimeError("bookkeeping failed mid-write")

    monkeypatch.setattr(_closet_engine, "record_closet_tombstones", boom)
    with pytest.raises(RuntimeError):
        engine.ingest(_memory("mem_edit", _turns(NEW_TEXT)))
    monkeypatch.undo()

    # An ordinary, unrelated write on the same engine and the same connection.
    engine.ingest(_memory("mem_later", "an unrelated fact"))

    assert _rows(db, "SELECT content FROM memories WHERE id = 'mem_later'") == [("an unrelated fact",)]
    assert _family(db, "mem_edit") == before
    assert _rows(db, "SELECT count(*) FROM memories WHERE content LIKE ?", (f"%{NEW_TEXT}%",)) == [(0,)]
    engine._conn.close()


def test_failed_delete_leaves_the_memory_and_its_copies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``delete`` is the other redaction path and gets the same guard.

    The failure is injected AFTER the per-speaker copies have been deleted — the
    only point at which a missing rollback is observable — so both their rows and
    their embedding rows have to come back.
    """
    db = tmp_path / "poppy.db"
    engine = _bloom(db)
    engine.ingest(_memory("mem_del", _turns(OLD_TEXT)))
    before = _family(db, "mem_del")
    before_vectors = _embedding_ids(db, "mem_del")
    assert len(before) == 3
    assert len(before_vectors) == 3

    real_sweep = _closet_engine.delete_marked_closets

    def sweep_then_fail(*args, **kwargs):
        removed = real_sweep(*args, **kwargs)
        assert removed, "the copies should have been deleted before the failure"
        raise RuntimeError("bookkeeping failed after the copy sweep")

    monkeypatch.setattr(_closet_engine, "delete_marked_closets", sweep_then_fail)
    with pytest.raises(RuntimeError):
        engine.delete("mem_del")

    assert engine._conn.in_transaction is False
    _assert_another_process_can_write(db)
    assert _family(db, "mem_del") == before
    assert _embedding_ids(db, "mem_del") == before_vectors
    engine._conn.close()


def test_a_failed_write_cannot_erase_another_threads_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two threads on one engine: a rollback may only undo its own thread's write.

    The engine's connection is usable from any thread, so without the engine
    lock thread B's ingest sees A's transaction already open, treats itself as a
    nested caller, writes, commits nothing and returns the id as a success —
    then A fails and its rollback takes B's row with it. The lock makes B wait,
    so B's write is its own transaction and survives.
    """
    db = tmp_path / "poppy.db"
    engine = _bloom(db)
    engine.ingest(_memory("mem_a", _turns(OLD_TEXT)))
    before = _family(db, "mem_a")

    a_inside = threading.Event()
    b_done = threading.Event()
    real_tombstones = _closet_engine.record_closet_tombstones

    def stall_then_fail(*args, **kwargs):
        if threading.current_thread().name != "writer-a":
            return real_tombstones(*args, **kwargs)
        a_inside.set()
        # Hold A's transaction open and give B every chance to slip into it.
        # Under the lock B cannot, so this wait simply times out.
        b_done.wait(timeout=1.0)
        raise RuntimeError("bookkeeping failed mid-write")

    monkeypatch.setattr(_closet_engine, "record_closet_tombstones", stall_then_fail)

    unexpected: list[BaseException] = []

    def writer_a() -> None:
        try:
            engine.ingest(_memory("mem_a", _turns(NEW_TEXT)))
        except RuntimeError:
            pass  # the injected failure
        except BaseException as exc:  # pragma: no cover - surfaced by the assert below
            unexpected.append(exc)

    def writer_b() -> None:
        try:
            a_inside.wait(timeout=10)
            engine.ingest(_memory("mem_b", "an unrelated fact"))
        except BaseException as exc:  # pragma: no cover - surfaced by the assert below
            unexpected.append(exc)
        finally:
            b_done.set()

    thread_a = threading.Thread(target=writer_a, name="writer-a")
    thread_b = threading.Thread(target=writer_b, name="writer-b")
    thread_a.start()
    thread_b.start()
    thread_a.join(timeout=30)
    thread_b.join(timeout=30)
    assert not thread_a.is_alive() and not thread_b.is_alive()
    assert unexpected == []

    # B reported success, so B's row must be in the store.
    assert _rows(db, "SELECT content FROM memories WHERE id = 'mem_b'") == [("an unrelated fact",)]
    # A failed, so nothing of A's edit landed.
    assert _family(db, "mem_a") == before
    assert engine._conn.in_transaction is False
    engine._conn.close()
