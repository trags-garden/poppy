"""Tests for the shared Remember/Forget core flows.

The whole point of the extraction is that the flow is exercised once here,
through the core interface with a tmp poppy_dir and the model-free SeedEngine,
instead of re-tested per surface. CLI/MCP/UI adapter tests only need to check
their own presentation and error mapping.
"""

import dataclasses
import datetime as dt

import pytest

from poppy.engine.seed import SeedEngine
from poppy.ui.tombstones import TombstoneStore
from poppy.write_flow import ForgetResult, RememberResult, forget, make_memory_id, remember, restore


@pytest.fixture
def engine(tmp_path):
    return SeedEngine(db_path=tmp_path / "memories.db")


@pytest.fixture(autouse=True)
def _no_autosync(monkeypatch):
    """Record autosync triggers without spawning the real detached worker."""
    calls: list = []
    monkeypatch.setattr("poppy.sync.auto.trigger", lambda poppy_dir, **kw: (calls.append(poppy_dir), True)[1])
    return calls


def _conflict_llm(monkeypatch, target_id, *, confidence):
    monkeypatch.setattr(
        "poppy.consolidation.call_llm",
        lambda prompt, *, transcript_path, cfg, **kwargs: [
            {"id": target_id, "confidence": confidence, "reason": "replaces"}
        ],
    )


# ---------- make_memory_id ----------


def test_make_memory_id_shape():
    mid = make_memory_id()
    assert mid.startswith("mem_")
    assert len(mid) == len("mem_") + 12
    assert mid != make_memory_id()


# ---------- remember ----------


def test_remember_ingests(engine, tmp_path, _no_autosync):
    result = remember(engine, tmp_path, content="use ruff", memory_type="preference")
    assert isinstance(result, RememberResult)
    assert result.wrote is True
    assert result.mode == "off"
    assert result.superseded_id is None
    stored = engine.get(result.memory.id)
    assert stored is not None
    assert stored.content == "use ruff"
    assert stored.memory_type == "preference"
    assert _no_autosync == [tmp_path], "a successful write triggers autosync"


def test_remember_stamps_source(engine, tmp_path):
    result = remember(engine, tmp_path, content="x", source="cursor")
    assert engine.get(result.memory.id).source.type == "cursor"


def test_remember_default_source_is_manual(engine, tmp_path):
    result = remember(engine, tmp_path, content="x")
    assert engine.get(result.memory.id).source.type == "manual"


def test_remember_resolves_ttl(engine, tmp_path):
    result = remember(engine, tmp_path, content="temp", ttl="30d")
    assert result.memory.expires_at is not None
    delta = result.memory.expires_at - result.memory.created_at
    assert abs(delta - dt.timedelta(days=30)) < dt.timedelta(seconds=2)


def test_remember_invalid_ttl_raises_valueerror(engine, tmp_path):
    with pytest.raises(ValueError):
        remember(engine, tmp_path, content="temp", ttl="not-a-duration")


def test_remember_ttl_and_expires_at_mutually_exclusive(engine, tmp_path):
    with pytest.raises(ValueError):
        remember(engine, tmp_path, content="x", ttl="1d", expires_at="2027-01-01")


def test_remember_explicit_supersedes(engine, tmp_path, _no_autosync):
    old = remember(engine, tmp_path, content="use all-MiniLM", memory_type="decision")
    result = remember(engine, tmp_path, content="use bge-large", memory_type="decision", supersedes=old.memory.id)
    assert result.superseded_id == old.memory.id
    assert engine.get(old.memory.id) is None, "old memory removed from the engine"
    assert TombstoneStore(tmp_path / "memories.db").get(old.memory.id) is not None
    assert old.memory.id in engine.get(result.memory.id).related_to


def test_remember_supersedes_unknown_id_raises_keyerror(engine, tmp_path):
    with pytest.raises(KeyError):
        remember(engine, tmp_path, content="x", supersedes="mem_doesnotexist")


def test_remember_check_mode_is_dry_run(engine, tmp_path, monkeypatch, _no_autosync):
    first = remember(engine, tmp_path, content="use all-MiniLM", memory_type="decision", project="poppy")
    _conflict_llm(monkeypatch, first.memory.id, confidence=0.91)

    before = len(engine.list_all(limit=10))
    result = remember(
        engine, tmp_path, content="use bge-large", memory_type="decision", project="poppy", check_conflicts=True
    )
    assert result.mode == "check"
    assert result.wrote is False
    assert result.memory is None
    assert [c.memory.id for c in result.conflicts] == [first.memory.id]
    assert len(engine.list_all(limit=10)) == before, "check mode never writes"


def test_remember_check_mode_with_supersedes_is_dry_run(engine, tmp_path, _no_autosync):
    """check + an explicit supersedes must NOT tombstone the target, and must
    report wrote=False. Regression for the misreport where check-mode's early
    return sat inside the `not supersedes` guard and fell through to a
    destructive supersede."""
    target = remember(engine, tmp_path, content="old fact", memory_type="fact").memory
    _no_autosync.clear()

    result = remember(
        engine, tmp_path, content="new fact", memory_type="fact", supersedes=target.id, check_conflicts=True
    )
    assert result.mode == "check"
    assert result.wrote is False
    assert result.memory is None
    assert result.superseded_id is None
    assert engine.get(target.id) is not None, "check mode must not tombstone the supersede target"
    assert TombstoneStore(tmp_path / "memories.db").get(target.id) is None
    assert _no_autosync == [], "check mode never triggers autosync"


def test_remember_auto_supersede(engine, tmp_path, monkeypatch, _no_autosync):
    first = remember(engine, tmp_path, content="use all-MiniLM", memory_type="decision", project="poppy")
    _conflict_llm(monkeypatch, first.memory.id, confidence=0.91)

    result = remember(
        engine, tmp_path, content="use bge-large", memory_type="decision", project="poppy", auto_supersede=True
    )
    assert result.superseded_id == first.memory.id
    assert engine.get(first.memory.id) is None


def test_remember_conflict_detection_error_still_writes(engine, tmp_path, monkeypatch, _no_autosync):
    def boom(*a, **k):
        raise RuntimeError("llm down")

    monkeypatch.setattr("poppy.capture.reconciler.detect_conflicts", boom)
    result = remember(engine, tmp_path, content="x", memory_type="fact", project="poppy", auto_supersede=True)
    assert result.wrote is True
    assert result.conflict_error is not None
    assert "llm down" in result.conflict_error
    assert engine.get(result.memory.id) is not None


def test_remember_emits_memory_write(engine, tmp_path, monkeypatch):
    """Every surface emits the same content-free memory_write event: has_project
    (a boolean), never the project name itself (privacy)."""
    events: list = []
    monkeypatch.setattr(
        "poppy.telemetry.capture",
        lambda poppy_dir, event, properties=None: events.append((event, properties or {})),
    )
    remember(engine, tmp_path, content="use ruff", memory_type="preference", project="poppy", source="cursor")
    assert ("memory_write", {"memory_type": "preference", "has_project": True, "source": "cursor"}) in events


# ---------- forget ----------


def test_forget_tombstones_and_deletes(engine, tmp_path, _no_autosync):
    mem = remember(engine, tmp_path, content="ephemeral").memory
    _no_autosync.clear()

    result = forget(engine, tmp_path, mem.id)
    assert isinstance(result, ForgetResult)
    assert result.deleted is True
    assert result.tombstoned is True
    assert engine.get(mem.id) is None
    assert TombstoneStore(tmp_path / "memories.db").get(mem.id) is not None
    assert _no_autosync == [tmp_path], "a successful delete triggers autosync"


def test_forget_nonexistent(engine, tmp_path, _no_autosync):
    result = forget(engine, tmp_path, "mem_nope")
    assert result.deleted is False
    assert result.tombstoned is False
    assert result.memory is None
    assert result.already_tombstoned is False
    assert _no_autosync == [], "nothing to delete means no sync"


def test_forget_is_idempotent_when_already_tombstoned(engine, tmp_path):
    mem = remember(engine, tmp_path, content="gone once").memory
    forget(engine, tmp_path, mem.id)

    result = forget(engine, tmp_path, mem.id)
    assert result.deleted is False
    assert result.memory is None
    assert result.already_tombstoned is True
    assert result.tombstone is not None


def test_forget_uses_reader_for_existence(tmp_path):
    """A separate reader engine can answer existence while the writer deletes."""
    reader = SeedEngine(db_path=tmp_path / "memories.db")
    writer = SeedEngine(db_path=tmp_path / "memories.db")
    mem = remember(writer, tmp_path, content="via reader").memory

    result = forget(writer, tmp_path, mem.id, reader=reader)
    assert result.deleted is True
    assert reader.get(mem.id) is None


# ---------- restore ----------


def test_restore_returns_live_row_with_ttl_and_fresh_updated_at(engine, tmp_path, _no_autosync):
    """Restore re-ingests with updated_at = now (so push sees
    it above the watermark) and keeps the memory's own expires_at."""
    ttl = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=2)
    store = TombstoneStore(tmp_path / "memories.db")
    res = remember(engine, tmp_path, content="ttl fact", memory_type="fact", expires_at=ttl.isoformat())
    mem_id = res.memory.id

    fr = forget(engine, tmp_path, mem_id, tombstones=store)
    assert fr.tombstoned

    restored = restore(engine, tmp_path, mem_id, tombstones=store)

    assert restored.found and not restored.expired
    assert restored.memory.expires_at == ttl
    assert restored.memory.updated_at > fr.tombstone.tombstoned_at
    assert store.get(mem_id) is None
    live = engine.get(mem_id)
    assert live is not None
    assert live.expires_at == ttl
    assert _no_autosync[-1] == tmp_path  # restore queues a sync like forget does


def test_restore_unknown_id_is_not_found(engine, tmp_path):
    res = restore(engine, tmp_path, "mem_nope")
    assert not res.found
    assert res.memory is None


def test_restore_stamps_after_the_tombstone_when_the_clock_steps_back(engine, tmp_path):
    """An NTP step or VM resume can put now() behind tombstoned_at; the restored
    row must still sort after the tombstone or push skips it again."""
    store = TombstoneStore(tmp_path / "memories.db")
    res = remember(engine, tmp_path, content="clock fact", memory_type="fact")
    fr = forget(engine, tmp_path, res.memory.id, tombstones=store)
    stepped_back = fr.tombstone.tombstoned_at - dt.timedelta(hours=1)

    restored = restore(engine, tmp_path, res.memory.id, tombstones=store, now=stepped_back)

    assert restored.memory.updated_at == fr.tombstone.tombstoned_at + dt.timedelta(microseconds=1)


def test_restore_does_not_clobber_a_live_row(engine, tmp_path):
    """A stale tombstone plus a live row (re-created or concurrently restored):
    the live row wins untouched and the tombstone is cleared."""
    store = TombstoneStore(tmp_path / "memories.db")
    res = remember(engine, tmp_path, content="original", memory_type="fact")
    mem_id = res.memory.id
    fr = forget(engine, tmp_path, mem_id, tombstones=store)
    recreated = dataclasses.replace(fr.memory, content="re-created", updated_at=dt.datetime.now(dt.timezone.utc))
    engine.ingest(recreated)

    out = restore(engine, tmp_path, mem_id, tombstones=store)

    assert out.already_live
    assert out.memory.content == "re-created"
    assert engine.get(mem_id).content == "re-created"
    assert store.get(mem_id) is None


def test_restore_keeps_the_row_when_it_loses_the_tombstone_claim(engine, tmp_path, monkeypatch):
    """Two restores of one memory: the first clears the tombstone, the second finds
    its conditional delete already lost. The loser must keep the row it ingested.

    Deleting it to "undo" would be unrecoverable — on the unlocked path (Windows,
    or after the write gate times out) the loser would wipe the winner's row and
    leave neither a memory nor a tombstone."""
    store = TombstoneStore(tmp_path / "memories.db")
    res = remember(engine, tmp_path, content="contested", memory_type="fact")
    mem_id = res.memory.id
    forget(engine, tmp_path, mem_id, tombstones=store)

    # The other restore got to the tombstone first: our conditional delete misses.
    monkeypatch.setattr(store, "remove", lambda *a, **kw: False)

    out = restore(engine, tmp_path, mem_id, tombstones=store)

    assert out.raced
    assert out.memory is not None
    live = engine.get(mem_id)
    assert live is not None, "the loser of the claim deleted the restored row"
    assert live.content == "contested"


def test_restore_keeps_the_row_when_a_delete_replaces_the_tombstone_mid_write(engine, tmp_path):
    """A forget landing during the ingest leaves its own tombstone. The restored
    row stays: a live row beside a stale tombstone is recoverable (pull drops the
    tombstone once a newer live row exists), losing the row is not."""
    store = TombstoneStore(tmp_path / "memories.db")
    res = remember(engine, tmp_path, content="original", memory_type="fact")
    mem_id = res.memory.id
    fr = forget(engine, tmp_path, mem_id, tombstones=store)

    real_ingest = engine.ingest
    replacement = {}

    def racing_ingest(memory):
        real_ingest(memory)
        replacement["t1"] = store.add(memory)

    engine.ingest = racing_ingest  # type: ignore[method-assign]

    out = restore(engine, tmp_path, mem_id, tombstones=store)

    assert out.raced
    assert engine.get(mem_id) is not None, "the restored row was deleted"
    surviving = store.get(mem_id)
    assert surviving is not None and surviving.token == replacement["t1"].token
    assert surviving.token != fr.tombstone.token


def test_restore_keeps_the_tombstone_when_the_ingest_fails(engine, tmp_path):
    """Crash safety: the tombstone is the only copy of a deleted memory, so it is
    cleared only after the row is safely back. A failing ingest (a model load on a
    bloom store, a full disk, SQLITE_BUSY) must leave the memory restorable."""
    store = TombstoneStore(tmp_path / "memories.db")
    res = remember(engine, tmp_path, content="precious", memory_type="fact")
    mem_id = res.memory.id
    forget(engine, tmp_path, mem_id, tombstones=store)

    real_ingest = engine.ingest

    def failing_ingest(memory):
        raise RuntimeError("engine unavailable")

    engine.ingest = failing_ingest  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        restore(engine, tmp_path, mem_id, tombstones=store)

    survivor = store.get(mem_id)
    assert survivor is not None, "the memory would have been lost outright"
    assert survivor.memory.content == "precious"

    # And once the engine works again the memory comes back.
    engine.ingest = real_ingest  # type: ignore[method-assign]
    out = restore(engine, tmp_path, mem_id, tombstones=store)
    assert out.memory.content == "precious"
    assert engine.get(mem_id) is not None
    assert store.get(mem_id) is None


def test_write_gate_serializes_two_holders(tmp_path):
    """The gate is what keeps a pull in the sync worker from interleaving with a
    restore in the UI: while one holder is inside, the other waits."""
    import threading

    from poppy.db import write_gate

    inside = threading.Event()
    release = threading.Event()
    acquired_second = threading.Event()

    def first():
        with write_gate(tmp_path):
            inside.set()
            release.wait(5)

    def second():
        with write_gate(tmp_path):
            acquired_second.set()

    t1 = threading.Thread(target=first)
    t1.start()
    assert inside.wait(5)

    t2 = threading.Thread(target=second)
    t2.start()
    assert not acquired_second.wait(0.5), "the second holder got in while the first held the gate"

    release.set()
    t1.join(5)
    assert acquired_second.wait(5), "the gate never let the waiter through"
    t2.join(5)


def test_forget_and_restore_cannot_interleave(engine, tmp_path):
    """A delete and a restore of the same memory must not overlap.

    Unserialized, they destroy it: forget reads the live row and writes its
    tombstone, restore sees both and clears that tombstone as stale, then forget
    deletes the row — leaving no memory and no tombstone. forget holds the same
    write gate restore does, so the restore waits its turn and the memory
    survives either ordering."""
    import threading
    import time

    store = TombstoneStore(tmp_path / "memories.db")
    res = remember(engine, tmp_path, content="contended", memory_type="fact")
    mem_id = res.memory.id

    real_add = store.add
    tombstoned = threading.Event()

    def slow_add(memory, **kwargs):
        ts = real_add(memory, **kwargs)
        tombstoned.set()
        # Still inside forget's gate, before its engine.delete: the exact window
        # a restore used to slip into.
        time.sleep(0.3)
        return ts

    store.add = slow_add  # type: ignore[method-assign]

    deleter = threading.Thread(target=lambda: forget(engine, tmp_path, mem_id, tombstones=store))
    deleter.start()
    assert tombstoned.wait(5)

    out = restore(engine, tmp_path, mem_id, tombstones=store)
    deleter.join(5)
    assert not deleter.is_alive()

    live = engine.get(mem_id)
    assert live is not None or store.get(mem_id) is not None, "the memory was lost outright"
    # The restore waited for the delete to finish, so it wins cleanly.
    assert live is not None
    assert live.content == "contended"
    assert out.memory is not None
