"""Tests for the Trags sync layer's error handling.

The `sync/` package shipped with no tests. These cover the exception hierarchy
and the stop-on-auth / guard-the-network-loop behavior added to push, pull, and
the auto-sync worker so a revoked key surfaces cleanly instead of a raw
traceback or a silent 32x retry storm.
"""

from __future__ import annotations

import errno
import json
import os
import re
from datetime import datetime, timedelta, timezone

import pytest

from poppy.engine.seed import SeedEngine
from poppy.models import Memory, Source
from poppy.sync import pull, push
from poppy.sync.client import TragsAuthError, TragsConflictError, TragsError, TragsQuotaError
from poppy.sync.serializer import is_tombstone, memory_to_wire, tombstone_to_wire
from poppy.sync.state import PUSH_WATERMARK_VERSION, RemoteState, SyncState, clear_error, load, record_error, save
from poppy.ui.tombstones import TombstoneStore

_NOW = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)


def _memory(mid: str, *, updated: datetime, expires_at: datetime | None = None) -> Memory:
    return Memory(
        id=mid,
        content=f"content for {mid}",
        memory_type="fact",
        source=Source(type="test", session_id="s1", timestamp=_NOW),
        project="proj",
        related_to=[],
        created_at=_NOW,
        updated_at=updated,
        confidence=1.0,
        expires_at=expires_at,
    )


class _FakeClient:
    """Stand-in for TragsClient: configurable failures, records calls."""

    def __init__(
        self,
        *,
        base_url: str = "https://trags.test",
        rows: list[dict] | None = None,
        upsert_error: Exception | None = None,
        error_on_tombstones: bool = True,
        fail_after: int = 0,
        synced_ids: set[str] | None = None,
        iter_error: Exception | None = None,
        iter_error_after: int = 0,
        ping_error: Exception | None = None,
        echo: bool = False,
    ) -> None:
        self.base_url = base_url
        self._rows = rows or []
        self._upsert_error = upsert_error
        # When False, `upsert_error` fires only for LIVE rows — tombstones (which
        # consume no server quota) succeed. Lets a test model a 402 that starves
        # live upserts while deletions still go through.
        self._error_on_tombstones = error_on_tombstones
        # Let the first `fail_after` upserts succeed before `upsert_error` kicks
        # in — models one free quota slot before the cap bites.
        self._fail_after = fail_after
        # Server-faithful quota gating: when set, only LIVE rows whose id is NOT
        # already live in the cloud (a CREATE) raise `upsert_error`; updates to
        # ids in this set — and tombstones — succeed, mirroring the server's
        # `v_creates_new_live` gate.
        self._synced_ids = synced_ids
        self._iter_error = iter_error
        self._iter_error_after = iter_error_after
        # The no-op probe push makes when it has nothing above the watermark.
        self._ping_error = ping_error
        self.ping_calls = 0
        self.upsert_attempts = 0
        # Server-like mode: remember every upserted row keyed by id and serve it
        # back from iter_all_since, so a push/pull round trip is realistic. Mirrors
        # `upsert_memory_with_quota` (migration 027), whose ON CONFLICT DO UPDATE
        # writes deleted_at = EXCLUDED.deleted_at — a live upsert un-deletes a
        # soft-deleted row.
        self._echo = echo
        self.upserts: list[dict] = []
        self.rows_by_id: dict[str, dict] = {}

    def upsert(self, memory: dict):
        self.upsert_attempts += 1
        if self._should_fail(memory):
            raise self._upsert_error
        self.upserts.append(dict(memory))
        if self._echo:
            self.rows_by_id[memory["id"]] = dict(memory)
        return memory, True

    def _should_fail(self, memory: dict) -> bool:
        if self._upsert_error is None or self.upsert_attempts <= self._fail_after:
            return False
        if is_tombstone(memory):
            return self._error_on_tombstones
        # A live row. With server-faithful gating, only creates (id not already
        # live in the cloud) fail; updates to synced ids succeed.
        if self._synced_ids is not None:
            return memory["id"] not in self._synced_ids
        return True

    def ping(self) -> None:
        self.ping_calls += 1
        if self._ping_error is not None:
            raise self._ping_error

    def iter_all_since(self, updated_since=None, *, page_size: int = 100):
        if self._echo:
            since = datetime.fromisoformat(updated_since) if updated_since else None
            served = [
                r for r in self.rows_by_id.values() if since is None or datetime.fromisoformat(r["updated_at"]) >= since
            ]
            # The real API returns newest-first; pull re-sorts before applying.
            for row in sorted(served, key=lambda r: datetime.fromisoformat(r["updated_at"]), reverse=True):
                yield row
            return
        for i, row in enumerate(self._rows):
            if self._iter_error is not None and i >= self._iter_error_after:
                raise self._iter_error
            yield row
        # Also raise when the configured failure index is at/after the last row
        # (e.g. yield every row, then fail fetching the next page).
        if self._iter_error is not None and len(self._rows) <= self._iter_error_after:
            raise self._iter_error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def _engine_and_tombstones(tmp_path):
    db = tmp_path / "memories.db"
    return SeedEngine(db_path=db), TombstoneStore(db)


def test_exception_hierarchy():
    assert issubclass(TragsAuthError, TragsError)
    assert issubclass(TragsConflictError, TragsError)


def test_push_soft_error_freezes_watermark(tmp_path):
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))
    engine.ingest(_memory("m2", updated=_NOW + timedelta(seconds=1)))
    client = _FakeClient(upsert_error=TragsError("500 boom"))
    state = SyncState()

    res = push(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)

    # A non-auth server error is a per-item soft failure: every item errors,
    # nothing is sent, and the watermark stays frozen so they all retry later.
    assert res.errors == 2
    assert res.sent_live == 0
    assert state.remotes[client.base_url].last_pushed_at is None


def test_push_stops_and_raises_on_auth(tmp_path):
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))
    engine.ingest(_memory("m2", updated=_NOW + timedelta(seconds=1)))
    client = _FakeClient(upsert_error=TragsAuthError("401"))
    state = SyncState()

    with pytest.raises(TragsAuthError):
        push(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)

    # Stopped after the first 401 rather than hammering the server once per item.
    assert client.upsert_attempts == 1


def test_pull_applies_live_rows(tmp_path):
    engine, tombstones = _engine_and_tombstones(tmp_path)
    rows = [
        memory_to_wire(_memory("r1", updated=_NOW)),
        memory_to_wire(_memory("r2", updated=_NOW + timedelta(seconds=1))),
    ]
    client = _FakeClient(rows=rows)
    state = SyncState()

    res = pull(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)

    assert res.applied_live == 2
    assert {m.id for m in engine.list_all(limit=100)} == {"r1", "r2"}
    assert load(tmp_path).remotes[client.base_url].last_pulled_at is not None


def test_pull_tombstone_blocks_reingest(tmp_path):
    """A local tombstone at least as new as the incoming live row blocks
    the ingest, so a memory forgotten on this device cannot resurrect on the next
    pull (the tombstone hasn't been pushed yet — sync is pull-then-push)."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    # Local forget: tombstone recorded (tombstoned_at = now, after _NOW), live
    # row removed from the engine, cloud row still live and no newer than ours.
    tombstones.add(_memory("m1", updated=_NOW))
    rows = [memory_to_wire(_memory("m1", updated=_NOW))]
    client = _FakeClient(rows=rows)
    state = SyncState()

    res = pull(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)

    assert res.applied_live == 0
    assert res.skipped_stale == 1
    assert engine.get("m1") is None  # stayed deleted, did not resurrect


def test_pull_newer_live_row_wins_over_tombstone(tmp_path):
    """Re-creating after deletion still works — a genuinely newer live row
    beats an older tombstone and is ingested."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    tombstones.add(_memory("m1", updated=_NOW))  # tombstoned_at = now
    # Incoming live row updated after the local tombstone (re-created elsewhere).
    future = datetime.now(timezone.utc) + timedelta(days=1)
    rows = [memory_to_wire(_memory("m1", updated=future))]
    client = _FakeClient(rows=rows)
    state = SyncState()

    res = pull(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)

    assert res.applied_live == 1
    assert res.skipped_stale == 0
    assert engine.get("m1") is not None  # re-created


def test_pull_guards_transient_error_midpagination(tmp_path):
    engine, tombstones = _engine_and_tombstones(tmp_path)
    rows = [memory_to_wire(_memory("r1", updated=_NOW))]
    # Yield r1, then a network/server error before the next page.
    client = _FakeClient(rows=rows, iter_error=TragsError("503"), iter_error_after=1)
    state = SyncState()

    # Must not escape as a raw traceback; the collected row still applies.
    res = pull(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)

    assert res.applied_live == 1
    assert res.errors >= 1
    # Watermark frozen so the unseen tail retries next sync.
    assert state.remotes[client.base_url].last_pulled_at is None


def test_pull_raises_on_auth(tmp_path):
    engine, tombstones = _engine_and_tombstones(tmp_path)
    client = _FakeClient(rows=[], iter_error=TragsAuthError("401"), iter_error_after=0)
    state = SyncState()

    with pytest.raises(TragsAuthError):
        pull(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)


def test_state_record_and_clear_error(tmp_path):
    url = "https://trags.test"
    record_error(tmp_path, url, "pull failed: 503", source="pull")
    rs = load(tmp_path).remotes[url]
    assert rs.errors["pull"] == "pull failed: 503"
    assert rs.last_error == "pull failed: 503"  # back-compat read helper

    clear_error(tmp_path, url)
    assert load(tmp_path).remotes[url].errors == {}


def test_state_load_ignores_unknown_keys(tmp_path):
    # A state file written by a newer Poppy (extra field) must still load.
    (tmp_path / "sync_state.json").write_text(
        '{"remotes": {"https://trags.test": {"last_pushed_at": "x", "future_field": 1}}}'
    )
    rs = load(tmp_path).remotes["https://trags.test"]
    assert rs.last_pushed_at == "x"


def test_run_worker_stops_on_auth_error(tmp_path, monkeypatch):
    """A revoked key must stop the worker, not spin it up to max_rounds with
    backoff, invisibly, on every write (the retry-32x regression)."""
    from poppy.sync import auto

    monkeypatch.setattr(auto, "DEBOUNCE_S", 0)

    calls = {"n": 0}

    def fake_do_sync(poppy_dir):
        calls["n"] += 1
        raise TragsAuthError("401 Unauthorized")

    monkeypatch.setattr(auto, "_do_sync", fake_do_sync)

    auto._touch_pending(tmp_path)
    auto.run_worker(tmp_path)

    assert calls["n"] == 1  # ran once, then stopped
    assert not (tmp_path / auto.PENDING_FILENAME).exists()  # not re-armed


def test_do_sync_records_auth_error_then_clears(tmp_path, monkeypatch):
    """_do_sync records the auth failure (so `sync status` can show it) and a
    later clean sync clears it."""
    from poppy.config import PoppyConfig, save_config
    from poppy.sync import auto, state

    save_config(PoppyConfig(poppy_dir=tmp_path, engine="seed"))
    monkeypatch.setenv("POPPY_TRAGS_API_KEY", "usr_test")
    url = PoppyConfig().trags_api_url

    def boom(**kwargs):
        raise TragsAuthError("401 Unauthorized")

    monkeypatch.setattr("poppy.sync.sync", boom)
    with pytest.raises(TragsAuthError):
        auto._do_sync(tmp_path)
    rs = state.load(tmp_path).remotes[url]
    assert rs.last_error is not None and "auth failed" in rs.last_error

    # A subsequent clean sync clears the recorded error.
    monkeypatch.setattr("poppy.sync.sync", lambda **kwargs: _fake_sync_result())
    auto._do_sync(tmp_path)
    assert state.load(tmp_path).remotes[url].last_error is None


def _fake_sync_result():
    from poppy.sync import PullResult, PushResult, SyncResult

    return SyncResult(
        push=PushResult(sent_live=0, sent_tombstones=0, skipped=0, errors=0),
        pull=PullResult(applied_live=0, applied_tombstones=0, skipped_stale=0, errors=0),
    )


def test_tombstone_wire_round_trips_expires_at(tmp_path):
    """A TTL'd memory's expires_at must survive the tombstone wire hop.

    tombstone_to_wire pushed expires_at: null because the tombstone table never
    stored it, so a delete silently stripped the TTL cloud-side too."""
    from poppy.sync.serializer import is_tombstone, tombstone_to_wire, wire_to_memory

    _, tombstones = _engine_and_tombstones(tmp_path)
    ttl = datetime.now(timezone.utc) + timedelta(days=2)

    ts = tombstones.add(_memory("m1", updated=_NOW, expires_at=ttl))
    assert ts.memory.expires_at == ttl

    row = tombstone_to_wire(tombstones.get("m1"))
    assert is_tombstone(row)
    assert row["expires_at"] == ttl.isoformat()

    # And the pull side rebuilds a tombstone that still carries the TTL.
    incoming = wire_to_memory(row)
    assert incoming.expires_at == ttl
    tombstones.add(incoming)
    assert tombstones.get("m1").memory.expires_at == ttl


def test_restore_reaches_cloud_and_survives_the_next_pull(tmp_path, monkeypatch):
    """Forget → push → restore → sync must leave the memory
    live locally *and* live on the server, with its TTL intact.

    Restore used to re-ingest with the memory's original updated_at, below the
    push watermark, so push skipped it: the server row stayed soft-deleted and
    the next pull re-applied that tombstone, deleting the memory again."""
    monkeypatch.setattr("poppy.sync.auto.trigger", lambda poppy_dir, **kw: True)
    from poppy.write_flow import forget, restore

    engine, tombstones = _engine_and_tombstones(tmp_path)
    ttl = datetime.now(timezone.utc) + timedelta(days=2)
    engine.ingest(_memory("m1", updated=_NOW, expires_at=ttl))

    client = _FakeClient(echo=True)
    state = SyncState()

    def _push():
        return push(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)

    def _pull():
        return pull(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)

    # 1. Initial push: the server holds a live row with the TTL.
    assert _push().sent_live == 1
    assert client.rows_by_id["m1"]["deleted_at"] is None
    assert client.rows_by_id["m1"]["expires_at"] == ttl.isoformat()

    # 2. Forget, then push the tombstone: the server row is soft-deleted.
    fr = forget(engine, tmp_path, "m1", tombstones=tombstones)
    tombstoned_at = fr.tombstone.tombstoned_at
    assert _push().sent_tombstones == 1
    assert client.rows_by_id["m1"]["deleted_at"] == tombstoned_at.isoformat()

    # 3. Restore from the tombstone.
    res = restore(engine, tmp_path, "m1", tombstones=tombstones)
    assert res.found and not res.expired
    assert res.memory.expires_at == ttl
    assert res.memory.updated_at > tombstoned_at

    # 4. Two full sync rounds. The first push must carry the restore; neither
    #    pull may re-apply the (older) server tombstone.
    for _ in range(2):
        _pull()
        _push()

    live = engine.get("m1")
    assert live is not None, "restore was undone by the next pull"
    assert live.expires_at == ttl
    assert tombstones.get("m1") is None

    final = client.rows_by_id["m1"]
    assert final["deleted_at"] is None, "the cloud row is still soft-deleted"
    assert final["expires_at"] == ttl.isoformat()
    assert final["updated_at"] > tombstoned_at.isoformat()
    assert client.upserts[-1]["id"] == "m1"
    assert client.upserts[-1]["deleted_at"] is None


def test_pull_applies_a_server_side_restore(tmp_path):
    """The same promise from the other direction: a live row newer than our local
    tombstone wins, and the stale tombstone is dropped so the UI stops showing the
    memory as deleted and push cannot re-upload the delete."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    tombstones.add(_memory("m1", updated=_NOW))  # tombstoned_at = now
    restored_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    rows = [memory_to_wire(_memory("m1", updated=restored_at))]
    client = _FakeClient(rows=rows)
    state = SyncState()

    res = pull(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)

    assert res.applied_live == 1
    assert engine.get("m1") is not None
    assert tombstones.get("m1") is None


def test_pull_does_not_clear_a_tombstone_written_during_the_pull(tmp_path):
    """The stale-tombstone cleanup must not swallow a delete made while we ran.

    Race: pull reads tombstone T0, ingests the newer cloud row, and before it
    clears T0 a local delete tombstones that row as T1 and removes it. Deleting
    by id alone would drop T1, leaving neither a live row nor a tombstone — the
    user's deletion silently lost, invisible to the next push."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    t0 = tombstones.add(_memory("m1", updated=_NOW))
    restored_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    client = _FakeClient(rows=[memory_to_wire(_memory("m1", updated=restored_at))])
    state = SyncState()

    real_ingest = engine.ingest

    def racing_ingest(memory, **kwargs):
        # **kwargs: sync passes `remote_event_ts` to an engine that takes it, and
        # this stands in for the real method on a real engine.
        real_ingest(memory, **kwargs)
        # Concurrent UI delete lands between the ingest and the cleanup.
        tombstones.add(memory)
        engine.delete(memory.id)

    engine.ingest = racing_ingest  # type: ignore[method-assign]

    pull(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)

    t1 = tombstones.get("m1")
    assert t1 is not None, "a delete made during the pull was silently dropped"
    assert t1.tombstoned_at > t0.tombstoned_at
    assert engine.get("m1") is None


def test_restore_of_an_elapsed_ttl_does_not_resurrect_the_memory(tmp_path, monkeypatch, synced_state):
    """A TTL that ran out while the memory was tombstoned ends the memory.

    Bringing it back live would either hide it anyway (expired rows are filtered
    from list, retrieve and push) or, if the expiry were dropped, keep data the
    user scheduled to disappear. The tombstone is cleared, nothing is ingested,
    and the caller is told why. Nothing is pushed: the cloud row stays deleted."""
    monkeypatch.setattr("poppy.sync.auto.trigger", lambda poppy_dir, **kw: True)
    from poppy.write_flow import restore

    engine, tombstones = _engine_and_tombstones(tmp_path)
    tombstones.note_remote_memories({"m1"}, "https://trags.test")
    elapsed = datetime.now(timezone.utc) - timedelta(days=1)
    tombstones.add(_memory("m1", updated=_NOW, expires_at=elapsed))

    res = restore(engine, tmp_path, "m1", tombstones=tombstones)

    assert res.found and res.expired
    assert res.memory is None
    assert engine.get("m1") is None
    # The tombstone is left standing: a recovery action must not be what destroys
    # the last copy. It ages out of the restore window on its own.
    assert tombstones.get("m1") is not None
    assert restore(engine, tmp_path, "m1", tombstones=tombstones).expired

    # And the tombstone still propagates, so the cloud row is soft-deleted rather
    # than left live with an expiry in the past.
    client = _FakeClient(echo=True)
    push_res = push(engine=engine, tombstones=tombstones, client=client, state=synced_state(), poppy_dir=tmp_path)

    assert push_res.sent_live == 0
    assert push_res.sent_tombstones == 1
    assert client.rows_by_id["m1"]["deleted_at"] is not None


def test_pull_cleanup_keeps_a_same_tick_replacement_tombstone(tmp_path, monkeypatch):
    """The cleanup must key on the tombstone's token, not its timestamp.

    ``tombstoned_at`` is a wall clock, so a delete that replaces the tombstone
    within the same tick carries the same timestamp. A conditional delete keyed
    on the timestamp would still remove that replacement and lose the user's
    fresh deletion, which is the whole bug the condition exists to prevent."""

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ARG003 - signature matches datetime.now
            return datetime(2026, 8, 1, 9, 0, 0, tzinfo=timezone.utc)

    monkeypatch.setattr("poppy.ui.tombstones.datetime", _FrozenDatetime)

    engine, tombstones = _engine_and_tombstones(tmp_path)
    t0 = tombstones.add(_memory("m1", updated=_NOW))
    client = _FakeClient(rows=[memory_to_wire(_memory("m1", updated=datetime.now(timezone.utc)))])
    state = SyncState()

    real_ingest = engine.ingest

    def racing_ingest(memory, **kwargs):
        # **kwargs: sync passes `remote_event_ts` to an engine that takes it, and
        # this stands in for the real method on a real engine.
        real_ingest(memory, **kwargs)
        # Concurrent local delete, landing on the same frozen tombstoned_at.
        tombstones.add(memory)
        engine.delete(memory.id)

    engine.ingest = racing_ingest  # type: ignore[method-assign]

    pull(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)

    t1 = tombstones.get("m1")
    assert t1 is not None, "the same-tick replacement tombstone was deleted"
    assert t1.tombstoned_at == t0.tombstoned_at  # identical timestamps
    assert t1.token != t0.token  # only the token tells them apart
    assert engine.get("m1") is None


# --- 402 free-plan quota cap ---------------------------------------

_QUOTA_MSG = "Free plan memory limit reached. Upgrade to Trags Pro to remove the cap."


def test_client_raises_quota_error_on_402():
    """A 402 with a `quota_exceeded` body maps to TragsQuotaError carrying the
    server's detail message verbatim (not a generic TragsError)."""
    import httpx

    from poppy.sync.client import TragsClient

    resp = httpx.Response(402, json={"detail": _QUOTA_MSG, "code": "quota_exceeded"})
    with pytest.raises(TragsQuotaError) as ei:
        TragsClient._raise_for_status(resp)
    assert str(ei.value) == _QUOTA_MSG


def test_client_402_without_quota_code_is_generic_error():
    """A 402 that isn't the quota cap (or an unparseable body) stays a generic
    TragsError so it isn't mistaken for the upgrade prompt."""
    import httpx

    from poppy.sync.client import TragsClient

    other = httpx.Response(402, json={"detail": "nope", "code": "payment_required"})
    with pytest.raises(TragsError) as ei:
        TragsClient._raise_for_status(other)
    assert not isinstance(ei.value, TragsQuotaError)

    unparseable = httpx.Response(402, text="not json")
    with pytest.raises(TragsError) as ei2:
        TragsClient._raise_for_status(unparseable)
    assert not isinstance(ei2.value, TragsQuotaError)


def test_push_records_quota_and_freezes(tmp_path):
    """A 402 quota cap freezes the watermark, records the message in the sync
    state, and re-raises so the caller surfaces the upgrade prompt once."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))
    engine.ingest(_memory("m2", updated=_NOW + timedelta(seconds=1)))
    client = _FakeClient(upsert_error=TragsQuotaError(_QUOTA_MSG))
    state = SyncState()

    with pytest.raises(TragsQuotaError) as ei:
        push(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)

    assert str(ei.value) == _QUOTA_MSG

    rs = load(tmp_path).remotes[client.base_url]
    # Watermark frozen so the rows retry later; error recorded for `sync status`.
    assert rs.last_pushed_at is None
    assert rs.errors["push"] == _QUOTA_MSG


def test_push_success_clears_stale_quota_error(tmp_path):
    """A later clean push that sends something clears a stale push-origin quota
    banner from a prior failure."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))
    client = _FakeClient()  # upserts succeed
    record_error(tmp_path, client.base_url, _QUOTA_MSG, source="push")

    res = push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)

    assert res.errors == 0
    rs = load(tmp_path).remotes[client.base_url]
    assert rs.errors == {}  # the observed, unchanged push banner was cleared


def test_run_worker_stops_on_quota_error(tmp_path, monkeypatch):
    """A quota cap stops the worker (no retry storm) and logs the message rather
    than `sync ok`."""
    from poppy.sync import auto

    monkeypatch.setattr(auto, "DEBOUNCE_S", 0)
    calls = {"n": 0}

    def fake_do_sync(poppy_dir):
        calls["n"] += 1
        raise TragsQuotaError(_QUOTA_MSG)

    monkeypatch.setattr(auto, "_do_sync", fake_do_sync)

    auto._touch_pending(tmp_path)
    auto.run_worker(tmp_path)

    assert calls["n"] == 1  # ran once, then stopped
    assert not (tmp_path / auto.PENDING_FILENAME).exists()  # not re-armed
    log = (tmp_path / auto.LOG_FILENAME).read_text()
    assert "quota exceeded" in log
    assert "sync ok" not in log


def test_cli_sync_push_quota_exits_nonzero_and_status_shows(tmp_path, monkeypatch):
    """End-to-end: `poppy sync push` prints the server message once, exits
    non-zero, and `poppy sync status` then shows a `last error:` line."""
    from click.testing import CliRunner

    from poppy.cli.main import cli

    (tmp_path / "config.json").write_text('{"engine": "seed"}')  # url defaults to trags.ai
    engine, _ = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))

    fake = _FakeClient(base_url="https://trags.ai", upsert_error=TragsQuotaError(_QUOTA_MSG))
    monkeypatch.setattr("poppy.cli.main._sync_client", lambda: (fake, "https://trags.ai"))

    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path), "POPPY_TRAGS_API_KEY": "usr_test", "POPPY_TELEMETRY_OFF": "1"}

    result = runner.invoke(cli, ["sync", "push"], env=env)
    assert result.exit_code != 0
    assert result.output.count(_QUOTA_MSG) == 1

    status = runner.invoke(cli, ["sync", "status"], env=env)
    assert status.exit_code == 0
    assert "last error (push):" in status.output
    assert _QUOTA_MSG in status.output


def test_push_quota_still_sends_tombstones(tmp_path):
    """A 402 on a live upsert must not starve tombstones. A pending deletion
    consumes no quota and must still reach the cloud, or a `forget` is lost once
    the tombstone retention window lapses. The live row defers via the frozen
    watermark and the quota error still surfaces."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    # The cap 402s live upserts but not tombstones (they cost no quota).
    client = _FakeClient(
        rows=[memory_to_wire(_memory("t1", updated=_NOW))],
        upsert_error=TragsQuotaError(_QUOTA_MSG),
        error_on_tombstones=False,
    )
    state = SyncState()
    # Pull the cloud's live row before forgetting it: this deletion frees quota.
    assert pull(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path).applied_live == 1
    tombstones.add(engine.get("t1"))
    engine.delete("t1")
    engine.ingest(_memory("m1", updated=_NOW))  # live row — will 402

    with pytest.raises(TragsQuotaError):
        push(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)

    # The tombstone completed; only tombstones got through (the live row was skipped).
    assert client.upserts, "the pending tombstone was never sent"
    assert all(is_tombstone(r) for r in client.upserts)
    assert [r["id"] for r in client.upserts] == ["t1"]
    rs = load(tmp_path).remotes[client.base_url]
    assert rs.last_pushed_at is None  # frozen so the live row retries later
    assert rs.last_error == _QUOTA_MSG


def test_push_failure_does_not_clear_prior_error(tmp_path):
    """A failed push (errors > 0) must not clear the recorded error and pretend
    the remote is healthy: `sync status` has to keep showing a failure."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))
    state = SyncState()

    # Push A: hits the cap, records the quota error.
    quota_client = _FakeClient(upsert_error=TragsQuotaError(_QUOTA_MSG))
    with pytest.raises(TragsQuotaError):
        push(engine=engine, tombstones=tombstones, client=quota_client, state=state, poppy_dir=tmp_path)
    assert load(tmp_path).remotes[quota_client.base_url].last_error == _QUOTA_MSG

    # Push B: every row 500s. Must NOT clear the banner — nothing synced.
    err_client = _FakeClient(upsert_error=TragsError("500 boom"))
    res = push(engine=engine, tombstones=tombstones, client=err_client, state=state, poppy_dir=tmp_path)
    assert res.errors > 0
    rs = load(tmp_path).remotes[err_client.base_url]
    assert rs.errors.get("push") is not None  # a failure is still visible, not cleared


def test_push_equal_timestamp_row_not_skipped_after_failure(tmp_path):
    """Two memories share a timestamp T; the first succeeds (advancing toward T),
    the second 402s. The frozen watermark must sit STRICTLY BELOW T, or the
    second row is skipped forever by the `iso <= watermark` candidate filter."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))
    engine.ingest(_memory("m2", updated=_NOW))  # identical timestamp
    # One free slot: the first upsert succeeds, the next 402s.
    client = _FakeClient(upsert_error=TragsQuotaError(_QUOTA_MSG), fail_after=1)
    state = SyncState()

    with pytest.raises(TragsQuotaError):
        push(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)

    rs = load(tmp_path).remotes[client.base_url]
    # Frozen below T (here nothing precedes T, so None) — never AT T.
    assert rs.last_pushed_at is None or rs.last_pushed_at < _NOW.isoformat()

    # A later push still re-attempts the rows at T rather than skipping them.
    client2 = _FakeClient(upsert_error=TragsQuotaError(_QUOTA_MSG))
    with pytest.raises(TragsQuotaError):
        push(engine=engine, tombstones=tombstones, client=client2, state=state, poppy_dir=tmp_path)
    assert client2.upsert_attempts >= 1  # the T-row was retried, not skipped


def test_push_freezes_below_failure_keeping_earlier_success(tmp_path):
    """A success strictly before the failure timestamp stays pushed (its row is
    <= the frozen watermark), while the failed row and everything at/after it
    retries."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("early", updated=_NOW))  # succeeds
    engine.ingest(_memory("late", updated=_NOW + timedelta(seconds=5)))  # 402s
    client = _FakeClient(upsert_error=TragsQuotaError(_QUOTA_MSG), fail_after=1)
    state = SyncState()

    with pytest.raises(TragsQuotaError):
        push(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)

    rs = load(tmp_path).remotes[client.base_url]
    # Watermark advanced to the early success but stays below the failed row.
    assert rs.last_pushed_at == _NOW.isoformat()
    assert rs.last_pushed_at < (_NOW + timedelta(seconds=5)).isoformat()


# --- a stamp the clock has not reached ---------------------------
#
# `_NOW` is a fixed instant in the PAST, so `_NOW + 1461 days` is reliably ahead
# of any clock these tests run under and `_NOW + minutes` is reliably behind it.
# Nothing here reads the wall clock.

_SKEWED = _NOW + timedelta(days=1461)  # ~2030: a bad import, or a clock four years fast


def test_push_future_dated_row_does_not_block_later_writes(tmp_path):
    """A row stamped in the future must not carry the watermark with it.

    A skewed clock or a bad import writes `updated_at` in 2030. Pushing it used to
    advance `last_pushed_at` past every honest `now`, so every later write fell
    under the `iso <= watermark` filter and was skipped silently, for ever. The
    future row is still sent; it just does not move the mark.
    """
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("skewed", updated=_SKEWED))
    client = _FakeClient()

    first = push(engine=engine, tombstones=tombstones, client=client, state=SyncState(), poppy_dir=tmp_path)

    assert first.sent_live == 1  # the odd row is still uploaded
    assert load(tmp_path).remotes[client.base_url].last_pushed_at is None  # mark unmoved

    # A normal local write after it still reaches the cloud.
    engine.ingest(_memory("normal", updated=_NOW))
    after = _FakeClient()
    second = push(engine=engine, tombstones=tombstones, client=after, state=load(tmp_path), poppy_dir=tmp_path)

    assert "normal" in {u["id"] for u in after.upserts}
    assert second.errors == 0

    # And the mark landed ON that write rather than on the 2030 row: the next push
    # skips `normal` and re-sends only the row no clock has reached.
    third = _FakeClient()
    push(engine=engine, tombstones=tombstones, client=third, state=load(tmp_path), poppy_dir=tmp_path)
    assert {u["id"] for u in third.upserts} == {"skewed"}


def test_a_row_that_must_be_re_sent_does_not_drift_the_pushed_total(tmp_path):
    """`poppy sync status` counts delivered work, and a re-send is not new work.

    The 2030 row goes up on every push because nothing on disk can record it as
    done: the watermark is this device's whole ledger of what is finished, and that
    row is never in it. Counting each attempt walked the cumulative total upward on
    a store where nothing at all was happening.
    """
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("skewed", updated=_SKEWED))
    client = _FakeClient()

    push(engine=engine, tombstones=tombstones, client=client, state=SyncState(), poppy_dir=tmp_path)
    after_first = load(tmp_path).remotes[client.base_url].pushed_count

    again = _FakeClient()
    push(engine=engine, tombstones=tombstones, client=again, state=load(tmp_path), poppy_dir=tmp_path)

    assert again.upserts  # it really was sent again
    assert load(tmp_path).remotes[client.base_url].pushed_count == after_first


def test_push_does_not_trust_a_watermark_that_cannot_be_true(tmp_path):
    """A mark on an instant that has not happened is refused, whoever wrote it.

    The one-time pass below retires the marks older clients left, but a downgrade
    and up again, or a merge taking the max over a pre-fix duplicate key, can put
    one there afterwards. Such a mark hides every row underneath it, so it is not
    reused even when the state says an up-to-date client wrote it.
    """
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))
    client = _FakeClient()
    state = SyncState(
        remotes={
            client.base_url: RemoteState(
                last_pushed_at=_SKEWED.isoformat(),
                # Up to date, so the one-time migration is NOT what saves this row.
                push_watermark_v=PUSH_WATERMARK_VERSION,
            )
        }
    )

    res = push(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)

    assert res.sent_live == 1
    assert {u["id"] for u in client.upserts} == {"m1"}

    # Recovered, not stuck in a re-push loop: the next push is an ordinary delta.
    again = _FakeClient()
    second = push(engine=engine, tombstones=tombstones, client=again, state=load(tmp_path), poppy_dir=tmp_path)
    assert again.upserts == []
    assert second.skipped == 1


def test_push_recovers_a_write_stranded_by_a_watermark_that_has_since_elapsed(tmp_path):
    """The recovery cannot depend on the bad mark still LOOKING bad.

    At 12:00 an old client pushed a row dated 12:05 and let it carry the mark; a
    real write at 12:01 then fell under it and was skipped. By 12:06 the mark is an
    ordinary past timestamp — a poisoned `12:05` and an honest one are one string —
    so no reading of the value can still find the damage, and the 12:01 write would
    stay stranded for ever with push reporting no errors. What the state records is
    therefore WHICH CLIENT wrote the mark: a file with no `push_watermark_v` buys
    one full pass, and only after it are marks trusted again.
    """
    engine, tombstones = _engine_and_tombstones(tmp_path)
    # Written at 12:01 by the old client and never sent: it sits under the mark.
    engine.ingest(_memory("stranded", updated=_NOW + timedelta(minutes=1)))
    # What the old client left, carried by a row dated 12:05. `_NOW` is itself in
    # the past, so this stamp has ALREADY ELAPSED: it looks entirely ordinary now,
    # which is the whole point.
    (tmp_path / "sync_state.json").write_text(
        json.dumps(
            {
                "remotes": {
                    "https://trags.test": {
                        "last_pushed_at": (_NOW + timedelta(minutes=5)).isoformat(),
                        "pushed_count": 1,
                    }
                }
            }
        )
    )

    client = _FakeClient()
    first = push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)

    assert {u["id"] for u in client.upserts} == {"stranded"}  # no longer skipped
    assert first.errors == 0
    assert load(tmp_path).remotes[client.base_url].push_watermark_v == PUSH_WATERMARK_VERSION

    # Once per remote, not on every push: the next one is an ordinary delta.
    engine.ingest(_memory("later", updated=_NOW + timedelta(minutes=10)))
    again = _FakeClient()
    second = push(engine=engine, tombstones=tombstones, client=again, state=load(tmp_path), poppy_dir=tmp_path)

    assert {u["id"] for u in again.upserts} == {"later"}
    assert second.skipped == 1  # `stranded` is under the mark now, and stays there


def test_a_state_save_that_never_lands_leaves_the_watermark_pass_owed(tmp_path, monkeypatch):
    """The marker and the watermark are ONE write, so a crash between them cannot exist.

    A marker recorded by its own write could land while the watermark's did not,
    retiring the recovery for a device that never made it — and the stranded rows
    would have no second chance.
    """
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("stranded", updated=_NOW + timedelta(minutes=1)))
    (tmp_path / "sync_state.json").write_text(
        json.dumps({"remotes": {"https://trags.test": {"last_pushed_at": (_NOW + timedelta(minutes=5)).isoformat()}}})
    )

    def explode(*_args, **_kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr("poppy.sync.state.save", explode)
    with pytest.raises(OSError):
        push(engine=engine, tombstones=tombstones, client=_FakeClient(), state=load(tmp_path), poppy_dir=tmp_path)
    monkeypatch.undo()

    # Nothing was recorded, so the next push owes the same pass: the row is not lost.
    assert load(tmp_path).remotes["https://trags.test"].push_watermark_v == 0
    again = _FakeClient()
    push(engine=engine, tombstones=tombstones, client=again, state=load(tmp_path), poppy_dir=tmp_path)
    assert {u["id"] for u in again.upserts} == {"stranded"}


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_an_interrupted_state_save_keeps_the_previous_state(tmp_path, monkeypatch, failure):
    """A save that dies part way must leave the last good state readable.

    The lenient ``load`` reads a torn file as "no state", so a save that
    truncated in place would silently reset every remote's watermarks and force
    the recovery passes to run again on the next sync.
    """
    before = SyncState(
        remotes={"https://trags.test": RemoteState(last_pushed_at="2026-07-01T12:00:00+00:00", pushed_count=3)}
    )
    save(tmp_path, before)
    after = SyncState(
        remotes={"https://trags.test": RemoteState(last_pushed_at="2026-07-02T12:00:00+00:00", pushed_count=9)}
    )

    if failure == "write":
        real_write = os.write

        def disk_fills(fd, data):
            real_write(fd, data[: len(data) // 2])
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(os, "write", disk_fills)
    else:

        def killed_before_rename(*_args, **_kwargs):
            raise OSError("interrupted")

        monkeypatch.setattr(os, "replace", killed_before_rename)

    with pytest.raises(OSError):
        save(tmp_path, after)
    monkeypatch.undo()

    assert load(tmp_path, strict=True) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sync_state.json"]


def test_do_sync_keeps_error_when_cycle_has_errors(tmp_path, monkeypatch):
    """The auto-sync path must not clear a recorded failure when the cycle ended
    with errors (a 409/500 on a row) — otherwise `sync status` looks healthy
    while the row stays unsynced."""
    from poppy.config import PoppyConfig, save_config
    from poppy.sync import auto, state

    save_config(PoppyConfig(poppy_dir=tmp_path, engine="seed"))
    monkeypatch.setenv("POPPY_TRAGS_API_KEY", "usr_test")
    url = PoppyConfig().trags_api_url

    # A prior push recorded a soft failure.
    state.record_error(tmp_path, url, "push failed: 1 error(s)")

    # A cycle that still reports errors>0 must leave the banner in place.
    def erroring_sync(**kwargs):
        from poppy.sync import PullResult, PushResult, SyncResult

        return SyncResult(
            push=PushResult(sent_live=0, sent_tombstones=0, skipped=0, errors=1),
            pull=PullResult(applied_live=0, applied_tombstones=0, skipped_stale=0, errors=0),
        )

    monkeypatch.setattr("poppy.sync.sync", erroring_sync)
    summary = auto._do_sync(tmp_path)
    assert summary["errors"] == 1
    assert state.load(tmp_path).remotes[url].last_error is not None

    # And a fully clean cycle does clear it.
    monkeypatch.setattr("poppy.sync.sync", lambda **kwargs: _fake_sync_result())
    auto._do_sync(tmp_path)
    assert state.load(tmp_path).remotes[url].last_error is None


def test_run_worker_logs_incomplete_not_ok_on_errors(tmp_path, monkeypatch):
    """A cycle with errors must not be logged as `sync ok`."""
    from poppy.sync import auto

    monkeypatch.setattr(auto, "DEBOUNCE_S", 0)
    monkeypatch.setattr(auto, "_do_sync", lambda poppy_dir: {"errors": 1, "push_live": 0})

    auto._touch_pending(tmp_path)
    auto.run_worker(tmp_path)

    log = (tmp_path / auto.LOG_FILENAME).read_text()
    assert "sync incomplete" in log
    assert "sync ok" not in log


def test_push_quota_still_syncs_edits_to_existing_memories(tmp_path):
    """At the cap the server only quota-gates NEW live rows. An edit to an
    already-synced memory is a plain UPDATE that consumes no quota and must still
    reach the cloud — dropping it would strand content edits (e.g. removing
    sensitive text) forever while capped."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("new1", updated=_NOW))  # brand-new create -> 402
    engine.ingest(_memory("old1", updated=_NOW + timedelta(seconds=1)))  # edit to synced -> ok
    # Server-faithful: only creates (id not already live) are quota-gated.
    client = _FakeClient(upsert_error=TragsQuotaError(_QUOTA_MSG), synced_ids={"old1"})
    state = SyncState()

    with pytest.raises(TragsQuotaError):
        push(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)

    sent_ids = {r["id"] for r in client.upserts}
    assert "old1" in sent_ids  # the edit reached the cloud despite the cap
    assert "new1" not in sent_ids  # the create 402'd
    rs = load(tmp_path).remotes[client.base_url]
    assert rs.last_error == _QUOTA_MSG  # the cap is still surfaced


def test_push_noop_does_not_clear_pull_error(tmp_path):
    """A push that sends nothing (no candidates) must not clear a pull-origin
    banner — it resolved nothing."""
    engine, tombstones = _engine_and_tombstones(tmp_path)  # empty store -> no candidates
    client = _FakeClient()
    record_error(tmp_path, client.base_url, "pull failed: 503", source="pull")

    res = push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)

    assert res.sent_live == 0 and res.sent_tombstones == 0
    rs = load(tmp_path).remotes[client.base_url]
    assert rs.errors == {"pull": "pull failed: 503"}  # untouched


def test_push_clean_clears_auth_but_not_pull_data(tmp_path):
    """A successful authenticated push proves the key is valid again, so it
    clears an AUTH-origin banner — but it must still leave a pull-DATA failure
    (source="pull") it didn't address."""
    # Auth banner: a clean push resolves it (the write authenticated).
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))
    client = _FakeClient()  # succeeds
    record_error(tmp_path, client.base_url, "auth failed (check the Trags API key)", source="auth")

    push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)
    assert load(tmp_path).remotes[client.base_url].errors == {}  # auth cleared

    # Pull-DATA banner (a 503): a clean push does NOT resolve it.
    engine2, tombstones2 = _engine_and_tombstones(tmp_path / "b")
    engine2.ingest(_memory("m2", updated=_NOW))
    client2 = _FakeClient()
    record_error(tmp_path / "b", client2.base_url, "pull failed: 503", source="pull")

    push(engine=engine2, tombstones=tombstones2, client=client2, state=load(tmp_path / "b"), poppy_dir=tmp_path / "b")
    rs2 = load(tmp_path / "b").remotes[client2.base_url]
    assert rs2.errors == {"pull": "pull failed: 503"}  # pull-data banner survives


def test_cli_sync_push_noop_surfaces_pull_error_and_exits_nonzero(tmp_path, monkeypatch):
    """`poppy sync push` with a prior pull-origin error and nothing to send must
    leave the banner intact AND exit non-zero, not look healthy."""
    from click.testing import CliRunner

    from poppy.cli.main import cli
    from poppy.sync import state as sync_state

    (tmp_path / "config.json").write_text('{"engine": "seed"}')  # url defaults to trags.ai
    url = "https://trags.ai"
    sync_state.record_error(tmp_path, url, "auth failed (check the Trags API key)", source="pull")

    fake = _FakeClient(base_url=url)  # nothing to push -> zero requests
    monkeypatch.setattr("poppy.cli.main._sync_client", lambda: (fake, url))

    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path), "POPPY_TRAGS_API_KEY": "usr_test", "POPPY_TELEMETRY_OFF": "1"}
    result = runner.invoke(cli, ["sync", "push"], env=env)

    assert result.exit_code != 0  # surfaced the unresolved pull error
    rs = sync_state.load(tmp_path).remotes[url]
    assert rs.last_error is not None  # not cleared by the no-op push
    assert rs.last_error_source == "pull"


def test_pull_records_error_on_transient_failure(tmp_path):
    """A pull that 503s mid-pagination must persist the failure (source="pull")
    so `sync status` and the CLI exit code reflect the incomplete pull."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    client = _FakeClient(rows=[], iter_error=TragsError("503 boom"), iter_error_after=0)
    state = SyncState()

    res = pull(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)

    assert res.errors == 1
    rs = load(tmp_path).remotes[client.base_url]
    assert rs.last_error is not None
    assert rs.last_error_source == "pull"
    assert "503" in rs.last_error


def test_pull_clean_clears_pull_and_auth_banner(tmp_path):
    """A clean pull authenticated and completed, so it clears a pull- or
    auth-origin banner — but not a push-origin one it didn't address."""
    for source, cleared in (("pull", True), ("auth", True), ("push", False)):
        sub = tmp_path / source
        engine, tombstones = _engine_and_tombstones(sub)
        client = _FakeClient(rows=[])  # clean, no rows
        record_error(sub, client.base_url, f"{source} banner", source=source)

        pull(engine=engine, tombstones=tombstones, client=client, state=load(sub), poppy_dir=sub)

        rs = load(sub).remotes[client.base_url]
        if cleared:
            assert source not in rs.errors, f"{source} should be cleared by a clean pull"
        else:
            assert rs.errors.get(source) == f"{source} banner", "push-origin banner must survive a clean pull"


def test_sync_clean_cycle_clears_legacy_untagged_error(tmp_path):
    """A legacy state file whose `last_error` has no source (old single-slot
    schema) must be cleared by the first fully-clean cycle — otherwise it is
    unclearable and the exit code stays non-zero forever."""
    from poppy.sync import sync as do_sync

    # Old-schema state file: single `last_error` field, no `errors` dict.
    (tmp_path / "sync_state.json").write_text(
        '{"remotes": {"https://trags.test": {"last_error": "legacy failure", "last_error_at": "2026-01-01T00:00:00"}}}'
    )
    # It loads (migrated) as an outstanding error.
    assert load(tmp_path).remotes["https://trags.test"].errors

    engine, tombstones = _engine_and_tombstones(tmp_path)
    client = _FakeClient(echo=True)  # clean pull + clean push
    do_sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path)

    rs = load(tmp_path).remotes["https://trags.test"]
    assert rs.errors == {}


def test_cli_sync_push_clears_auth_after_key_fixed_and_exits_zero(tmp_path, monkeypatch):
    """An auth 401 recorded earlier; the user fixes the key; the next
    `poppy sync push` sends the rows, clears the banner, and exits 0."""
    from click.testing import CliRunner

    from poppy.cli.main import cli
    from poppy.sync import state as sync_state

    (tmp_path / "config.json").write_text('{"engine": "seed"}')
    url = "https://trags.ai"
    sync_state.record_error(tmp_path, url, "auth failed (check the Trags API key)", source="auth")

    engine, _ = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))
    fake = _FakeClient(base_url=url)  # key valid now -> upserts succeed
    monkeypatch.setattr("poppy.cli.main._sync_client", lambda: (fake, url))

    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path), "POPPY_TRAGS_API_KEY": "usr_test", "POPPY_TELEMETRY_OFF": "1"}
    result = runner.invoke(cli, ["sync", "push"], env=env)

    assert result.exit_code == 0  # banner resolved, no leftover error
    assert sync_state.load(tmp_path).remotes[url].last_error is None


def test_cli_sync_run_pull_error_exits_nonzero(tmp_path, monkeypatch):
    """`poppy sync run` where the pull 503s but the push is clean must persist a
    pull banner and exit non-zero, not look healthy."""
    from click.testing import CliRunner

    from poppy.cli.main import cli
    from poppy.sync import state as sync_state

    (tmp_path / "config.json").write_text('{"engine": "seed"}')
    url = "https://trags.ai"
    fake = _FakeClient(base_url=url, rows=[], iter_error=TragsError("503 boom"), iter_error_after=0)
    monkeypatch.setattr("poppy.cli.main._sync_client", lambda: (fake, url))

    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path), "POPPY_TRAGS_API_KEY": "usr_test", "POPPY_TELEMETRY_OFF": "1"}
    result = runner.invoke(cli, ["sync", "run"], env=env)

    assert result.exit_code != 0
    rs = sync_state.load(tmp_path).remotes[url]
    assert rs.last_error is not None
    assert rs.last_error_source == "pull"


def test_push_unchanged_store_resends_nothing(tmp_path):
    """After a full push, subsequent pushes of an UNCHANGED store must re-send
    NOTHING — the boundary row at exactly `last_pushed_at` is not re-upserted
    (`iso <= watermark`), so `pushed_count` stays stable and no paid write is
    wasted per local write."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))
    engine.ingest(_memory("m2", updated=_NOW + timedelta(seconds=1)))
    client = _FakeClient()
    state = SyncState()

    first = push(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)
    assert first.sent_live == 2
    after_first = load(tmp_path).remotes[client.base_url].pushed_count
    attempts_after_first = client.upsert_attempts

    # Pushes #2 and #3 with no local changes send nothing.
    for _ in range(2):
        res = push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)
        assert res.sent_live == 0 and res.sent_tombstones == 0
    assert client.upsert_attempts == attempts_after_first  # no boundary re-upsert
    assert load(tmp_path).remotes[client.base_url].pushed_count == after_first  # stable


def test_push_noop_keeps_the_push_banner_and_clears_what_the_probe_proved(tmp_path):
    """A no-op push (nothing above the watermark) sends no memory, so it must not
    clear a push banner — an unresolved row failure is still unresolved. Its
    probe DID reach the server, so it clears the two banners that fact settles:
    the auth banner, and a probe failure from an earlier idle run."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))
    client = _FakeClient()
    state = SyncState()

    # First push syncs m1 and advances the watermark past it.
    push(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)

    # Record a push+auth banner on disk, then push again with nothing new to send.
    record_error(tmp_path, client.base_url, "stale push error", source="push")
    record_error(tmp_path, client.base_url, "stale auth error", source="auth")
    record_error(tmp_path, client.base_url, "stale probe error", source="probe")
    res = push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)

    assert res.sent_live == 0 and res.sent_tombstones == 0  # nothing sent
    assert client.ping_calls == 1  # but the key was still checked
    rs = load(tmp_path).remotes[client.base_url]
    assert rs.errors == {"push": "stale push error"}  # the row failure is untouched
    assert "auth" not in rs.errors  # the probe proved the key works
    assert "probe" not in rs.errors  # ... and that the host answers


def test_pull_and_push_errors_held_in_separate_slots(tmp_path):
    """A run with a pull failure AND a push failure records BOTH (one slot each);
    a later clean push clears only the push slot, leaving the pull failure."""
    from poppy.sync import sync as do_sync

    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))
    client = _FakeClient(
        rows=[],
        iter_error=TragsError("pull 503"),
        iter_error_after=0,
        upsert_error=TragsError("push 500"),
    )
    do_sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path)

    rs = load(tmp_path).remotes[client.base_url]
    assert rs.errors.get("pull") == "pull 503"
    assert rs.errors.get("push") == "push 500"  # neither overwrote the other

    # A later clean push clears ONLY the push slot; the pull failure stays.
    good = _FakeClient()
    push(engine=engine, tombstones=tombstones, client=good, state=load(tmp_path), poppy_dir=tmp_path)
    rs2 = load(tmp_path).remotes[good.base_url]
    assert "push" not in rs2.errors
    assert rs2.errors.get("pull") == "pull 503"


def test_state_url_key_normalized(tmp_path):
    """State written under the stripped host (as TragsClient keys it) must be read
    under the trailing-slash config variant, or a failure silently vanishes."""
    from poppy.sync import remote_state_for
    from poppy.sync.state import record_error

    record_error(tmp_path, "https://trags.ai", "push 500", source="push")
    rs = remote_state_for(tmp_path, "https://trags.ai/")
    assert rs.errors.get("push") == "push 500"


def test_cli_sync_push_trailing_slash_url_surfaces_failure(tmp_path, monkeypatch):
    """With a trailing-slash configured URL, a recorded push failure must still be
    found by `_fail_if_unresolved_error` (non-zero exit) and `sync status`."""
    from click.testing import CliRunner

    from poppy.cli.main import cli

    (tmp_path / "config.json").write_text('{"engine": "seed", "trags_api_url": "https://trags.ai/"}')
    engine, _ = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))
    # Client strips the slash (base_url="https://trags.ai"); CLI passes the RAW
    # configured url with the slash. Both must resolve the same state key.
    fake = _FakeClient(base_url="https://trags.ai", upsert_error=TragsError("500 boom"))
    monkeypatch.setattr("poppy.cli.main._sync_client", lambda: (fake, "https://trags.ai/"))

    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path), "POPPY_TRAGS_API_KEY": "usr_test", "POPPY_TELEMETRY_OFF": "1"}
    result = runner.invoke(cli, ["sync", "push"], env=env)
    assert result.exit_code != 0  # recorded push failure found via the normalized key

    status = runner.invoke(cli, ["sync", "status"], env=env)
    assert "500 boom" in status.output


def test_run_worker_redrains_forget_queued_during_quota_cycle(tmp_path, monkeypatch):
    """A `forget` queued mid-cycle at the cap must flush in the SAME worker run,
    EVEN when the pass made zero progress (its only work was quota-blocked
    creates). Tombstones aren't quota-gated, so the worker re-drains purely on
    `pending` being set and sends the deletion instead of stranding it."""
    from poppy.config import PoppyConfig, save_config
    from poppy.sync import auto

    monkeypatch.setattr(auto, "DEBOUNCE_S", 0)
    save_config(PoppyConfig(poppy_dir=tmp_path, engine="seed"))
    monkeypatch.setenv("POPPY_TRAGS_API_KEY", "usr_test")

    calls = {"n": 0}
    sent = {"tombstone": False}

    def fake_do_sync(poppy_dir):
        calls["n"] += 1
        if calls["n"] == 1:
            # No progress: the only work was a quota-blocked create. A `forget`
            # sets pending AFTER this pass's snapshot.
            auto._touch_pending(poppy_dir)
            raise TragsQuotaError("cap")
        if calls["n"] == 2:
            # Re-drain: the queued tombstone (not gated) flushes; cap still hit.
            sent["tombstone"] = True
            raise TragsQuotaError("cap")
        raise AssertionError("no third drain — nothing pending after pass 2")

    monkeypatch.setattr(auto, "_do_sync", fake_do_sync)
    auto._touch_pending(tmp_path)
    auto.run_worker(tmp_path)

    assert sent["tombstone"], "the mid-cycle forget's tombstone was stranded"
    assert calls["n"] == 2  # re-drained exactly once, no spin
    assert not (tmp_path / auto.PENDING_FILENAME).exists()  # fully drained


def test_run_worker_does_not_spin_on_quota_without_new_work(tmp_path, monkeypatch):
    """A quota-blocked create with no NEW work queued must NOT re-drain: the pass
    leaves `pending` unset, so the loop stops and leaves it for the next
    trigger."""
    from poppy.config import PoppyConfig, save_config
    from poppy.sync import auto

    monkeypatch.setattr(auto, "DEBOUNCE_S", 0)
    save_config(PoppyConfig(poppy_dir=tmp_path, engine="seed"))
    monkeypatch.setenv("POPPY_TRAGS_API_KEY", "usr_test")

    calls = {"n": 0}

    def fake_do_sync(poppy_dir):
        calls["n"] += 1
        raise TragsQuotaError("cap")  # never any progress

    monkeypatch.setattr(auto, "_do_sync", fake_do_sync)
    auto._touch_pending(tmp_path)
    auto.run_worker(tmp_path)

    assert calls["n"] == 1  # stopped after one pass, no spin


def test_load_merges_colliding_normalized_urls(tmp_path):
    """A legacy state file stored under BOTH `https://trags.ai` (populated) and
    `https://trags.ai/` (empty) must MERGE on load (most-advanced wins), not let
    the empty record reset the populated one and force a full replay."""
    import json

    (tmp_path / "sync_state.json").write_text(
        json.dumps(
            {
                "remotes": {
                    "https://trags.ai": {
                        "last_pushed_at": "2026-07-01T00:00:00",
                        "last_pulled_at": "2026-07-02T00:00:00",
                        "pushed_count": 10000,
                        "pulled_count": 5000,
                    },
                    "https://trags.ai/": {"last_pushed_at": None, "pushed_count": 0},
                }
            }
        )
    )

    state = load(tmp_path)
    assert list(state.remotes) == ["https://trags.ai"]  # collapsed to one key
    rs = state.remotes["https://trags.ai"]
    # Populated watermarks kept, VERBATIM: the merge picks the later INSTANT rather
    # than the larger string, but stores the winner's own spelling. Push decides it
    # owes a recovery pass by seeing that a stored watermark is not canonical, so a
    # merge that canonicalised it would hide that and skip pending rows.
    assert rs.last_pushed_at == "2026-07-01T00:00:00"
    assert rs.last_pulled_at == "2026-07-02T00:00:00"
    assert rs.pushed_count == 10000  # counters not reset to 0
    assert rs.pulled_count == 5000


def test_load_merges_error_slots_across_colliding_urls(tmp_path):
    """Merging colliding URL records keeps error slots from both."""
    import json

    (tmp_path / "sync_state.json").write_text(
        json.dumps(
            {
                "remotes": {
                    "https://trags.ai": {"errors": {"push": "push 500"}},
                    "https://trags.ai/": {"errors": {"pull": "pull 503"}},
                }
            }
        )
    )

    rs = load(tmp_path).remotes["https://trags.ai"]
    assert rs.errors.get("push") == "push 500"
    assert rs.errors.get("pull") == "pull 503"


def test_push_clean_does_not_clear_concurrent_error(tmp_path):
    """The exact race: sync A starts with NO push error and completes a clean
    push; a concurrent worker records `errors["push"]` AFTER A's snapshot. A's
    compare-and-clear must NOT wipe that newer error — it clears only what it
    observed at cycle start."""
    from poppy.sync.state import record_error

    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))
    state = SyncState()  # A's cycle-start snapshot: no errors

    class _RacingClient(_FakeClient):
        def upsert(self, memory):
            # A concurrent sync records a NEW push failure mid-cycle — after A
            # captured its snapshot, before A persists.
            record_error(tmp_path, self.base_url, "quota 402 from worker", source="push")
            return super().upsert(memory)

    client = _RacingClient()
    push(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)

    rs = load(tmp_path).remotes[client.base_url]
    assert rs.errors.get("push") == "quota 402 from worker"  # survived A's clean push


def test_push_clean_clears_only_unchanged_observed_error(tmp_path):
    """A clean push clears a push error it OBSERVED and that is unchanged, but
    leaves one that a concurrent sync overwrote with a newer message."""
    from poppy.sync.state import get_remote, record_error, save

    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))

    # Cycle A starts having observed an OLD push error on disk.
    state = SyncState()
    get_remote(state, "https://trags.test").errors["push"] = "old push error"
    save(tmp_path, state)
    a_state = load(tmp_path)  # A's snapshot: {"push": "old push error"}

    class _RacingClient(_FakeClient):
        def upsert(self, memory):
            # Concurrent sync overwrites the push slot with a NEWER message.
            record_error(tmp_path, self.base_url, "newer push error", source="push")
            return super().upsert(memory)

    client = _RacingClient()
    push(engine=engine, tombstones=tombstones, client=client, state=a_state, poppy_dir=tmp_path)

    rs = load(tmp_path).remotes[client.base_url]
    assert rs.errors.get("push") == "newer push error"  # changed slot not cleared


def test_push_clean_does_not_clear_aba_same_text_error(tmp_path):
    """ABA: compare-and-clear must match on GENERATION, not message text. Cycle A
    observes a quota error Q (generation N); a concurrent sync records a NEW
    quota error with IDENTICAL text Q (generation N+1) mid-cycle; A's clean push
    must NOT clear it — a different failure that merely looks the same."""
    from poppy.sync.state import record_error

    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))
    # A's cycle observes an existing quota error Q (generation N).
    record_error(tmp_path, "https://trags.test", _QUOTA_MSG, source="push")
    a_state = load(tmp_path)  # snapshot: push at generation N

    class _RacingClient(_FakeClient):
        def upsert(self, memory):
            # A concurrent sync records a NEW push failure with IDENTICAL text
            # (generation N+1) — a distinct failure, not the one A observed.
            record_error(tmp_path, self.base_url, _QUOTA_MSG, source="push")
            return super().upsert(memory)

    client = _RacingClient()
    push(engine=engine, tombstones=tombstones, client=client, state=a_state, poppy_dir=tmp_path)

    rs = load(tmp_path).remotes[client.base_url]
    assert rs.errors.get("push") == _QUOTA_MSG  # the newer same-text error survived


class _ReadTimeout(Exception):
    """Stand-in for a transport exception (e.g. httpx.ReadTimeout) — NOT a TragsError."""


def test_push_persists_quota_when_later_row_transport_errors(tmp_path):
    """A 402 on row A followed by a transport exception on row B must NOT discard
    the quota error: it is persisted (push slot) and re-raised with precedence,
    so `sync status` keeps the upgrade prompt and the worker stops on quota
    instead of a generic retry/backoff storm."""

    class _FlakyClient(_FakeClient):
        def upsert(self, memory):
            self.upsert_attempts += 1
            if memory["id"] == "A":
                raise TragsQuotaError(_QUOTA_MSG)  # 402 quota_exceeded
            raise _ReadTimeout("read timeout")  # transport error on the next POST

    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("A", updated=_NOW))
    engine.ingest(_memory("B", updated=_NOW + timedelta(seconds=1)))
    client = _FlakyClient()

    with pytest.raises(TragsQuotaError):  # quota wins precedence, not the timeout
        push(engine=engine, tombstones=tombstones, client=client, state=SyncState(), poppy_dir=tmp_path)

    rs = load(tmp_path).remotes[client.base_url]
    assert rs.errors.get("push") == _QUOTA_MSG  # persisted despite the later throw


def test_push_transport_error_recorded_and_raised(tmp_path):
    """With no quota, a transport exception is recorded as a push failure (so
    `sync status` and the exit code reflect it) AND surfaces for retry."""

    class _FlakyClient(_FakeClient):
        def upsert(self, memory):
            self.upsert_attempts += 1
            raise _ReadTimeout("read timeout")

    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("A", updated=_NOW))
    client = _FlakyClient()

    with pytest.raises(_ReadTimeout):  # surfaced for retry
        push(engine=engine, tombstones=tombstones, client=client, state=SyncState(), poppy_dir=tmp_path)

    rs = load(tmp_path).remotes[client.base_url]
    assert "read timeout" in (rs.errors.get("push") or "")  # recorded


def test_cli_sync_push_quota_survives_later_transport_error(tmp_path, monkeypatch):
    """End-to-end: quota on row A + a transport throw on row B still exits
    non-zero, prints the upgrade prompt, and shows the cap in `sync status`."""
    from click.testing import CliRunner

    from poppy.cli.main import cli

    class _FlakyClient(_FakeClient):
        def upsert(self, memory):
            self.upsert_attempts += 1
            if memory["id"] == "A":
                raise TragsQuotaError(_QUOTA_MSG)
            raise _ReadTimeout("read timeout")

    (tmp_path / "config.json").write_text('{"engine": "seed"}')
    engine, _ = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("A", updated=_NOW))
    engine.ingest(_memory("B", updated=_NOW + timedelta(seconds=1)))
    fake = _FlakyClient(base_url="https://trags.ai")
    monkeypatch.setattr("poppy.cli.main._sync_client", lambda: (fake, "https://trags.ai"))

    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path), "POPPY_TRAGS_API_KEY": "usr_test", "POPPY_TELEMETRY_OFF": "1"}
    result = runner.invoke(cli, ["sync", "push"], env=env)
    assert result.exit_code != 0
    assert _QUOTA_MSG in result.output  # upgrade prompt fired

    status = runner.invoke(cli, ["sync", "status"], env=env)
    assert _QUOTA_MSG in status.output  # cap still surfaced


def test_push_transport_error_does_not_strand_later_tombstone(tmp_path, synced_state):
    """A per-row transport error must NOT abandon the rest of the queue: a later
    TOMBSTONE (a deletion, quota-exempt) still gets its attempt in the same push.
    Row A 402s, row B times out, row C is a tombstone -> C's deletion is sent,
    the quota error is persisted+surfaced (precedence), and no new writes are
    needed to flush C."""

    class _FlakyClient(_FakeClient):
        def upsert(self, memory):
            self.upsert_attempts += 1
            if is_tombstone(memory):
                self.upserts.append(dict(memory))  # deletion is quota-exempt
                return memory, True
            if memory["id"] == "A":
                raise TragsQuotaError(_QUOTA_MSG)
            raise _ReadTimeout("read timeout")  # transport error on live row B

    engine, tombstones = _engine_and_tombstones(tmp_path)
    tombstones.note_remote_memories({"C"}, "https://trags.test")
    engine.ingest(_memory("A", updated=_NOW))  # 402 quota_exceeded
    engine.ingest(_memory("B", updated=_NOW + timedelta(seconds=1)))  # transport timeout
    tombstones.add(_memory("C", updated=_NOW + timedelta(seconds=2)))  # deletion
    client = _FlakyClient()

    with pytest.raises(TragsQuotaError):  # quota keeps precedence
        push(engine=engine, tombstones=tombstones, client=client, state=synced_state(), poppy_dir=tmp_path)

    assert any(is_tombstone(r) and r["id"] == "C" for r in client.upserts), "tombstone C was stranded"
    rs = load(tmp_path).remotes[client.base_url]
    assert rs.errors.get("push") == _QUOTA_MSG  # quota persisted + surfaced


# --- Push failure-handling: circuit breaker + independent quota/transport retry ---


def test_push_circuit_breaker_stops_on_dead_remote(tmp_path):
    """(b) A dead/black-holing remote must stop the push after
    MAX_CONSECUTIVE_TRANSPORT_FAILURES, not grind the whole backlog (which would
    hold the worker lock for ~N x timeout)."""
    from poppy.sync import MAX_CONSECUTIVE_TRANSPORT_FAILURES

    class _DeadClient(_FakeClient):
        def upsert(self, memory):
            self.upsert_attempts += 1
            raise _ReadTimeout("connection timed out")

    engine, tombstones = _engine_and_tombstones(tmp_path)
    for i in range(20):
        engine.ingest(_memory(f"m{i}", updated=_NOW + timedelta(seconds=i)))
    client = _DeadClient()

    with pytest.raises(_ReadTimeout):
        push(engine=engine, tombstones=tombstones, client=client, state=SyncState(), poppy_dir=tmp_path)

    assert client.upsert_attempts == MAX_CONSECUTIVE_TRANSPORT_FAILURES  # stopped early


def test_push_quota_and_transport_are_independent_signals(tmp_path, synced_state):
    """(c) A 402 on a create + a transport-failed TOMBSTONE + another tombstone:
    both tombstones are attempted, the second's deletion lands, the quota is
    surfaced (precedence), AND the transport failure is flagged for retry so the
    failed deletion isn't lost."""

    class _FlakyClient(_FakeClient):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.attempted: list[str] = []

        def upsert(self, memory):
            self.upsert_attempts += 1
            self.attempted.append(memory["id"])
            if is_tombstone(memory):
                if memory["id"] == "B":
                    raise _ReadTimeout("read timeout")  # deletion transport-fails
                self.upserts.append(dict(memory))
                return memory, True
            raise TragsQuotaError(_QUOTA_MSG)  # the create A 402s

    engine, tombstones = _engine_and_tombstones(tmp_path)
    tombstones.note_remote_memories({"B"}, "https://trags.test")
    tombstones.note_remote_memories({"C"}, "https://trags.test")
    engine.ingest(_memory("A", updated=_NOW))
    tombstones.add(_memory("B", updated=_NOW + timedelta(seconds=1)))
    tombstones.add(_memory("C", updated=_NOW + timedelta(seconds=2)))
    client = _FlakyClient()

    with pytest.raises(TragsQuotaError) as ei:
        push(engine=engine, tombstones=tombstones, client=client, state=synced_state(), poppy_dir=tmp_path)

    assert "B" in client.attempted and "C" in client.attempted  # both attempted
    assert any(r["id"] == "C" for r in client.upserts)  # C's deletion landed
    assert getattr(ei.value, "transport_retry", False) is True  # B scheduled for retry
    assert load(tmp_path).remotes[client.base_url].errors.get("push") == _QUOTA_MSG  # quota surfaced


def test_run_worker_retries_transport_failed_deletion_after_quota(tmp_path, monkeypatch):
    """(c, worker): when a cycle stops on quota but a row transport-failed, the
    worker re-arms and re-drains so the failed deletion is retried in the same
    run — quota and transport-retry are independent."""
    from poppy.config import PoppyConfig, save_config
    from poppy.sync import auto

    monkeypatch.setattr(auto, "DEBOUNCE_S", 0)
    save_config(PoppyConfig(poppy_dir=tmp_path, engine="seed"))
    monkeypatch.setenv("POPPY_TRAGS_API_KEY", "usr_test")

    calls = {"n": 0}
    deletion_retried = {"v": False}

    def fake_do_sync(poppy_dir):
        calls["n"] += 1
        if calls["n"] == 1:
            exc = TragsQuotaError("cap")
            exc.transport_retry = True  # a tombstone transport-failed this cycle
            raise exc
        if calls["n"] == 2:
            deletion_retried["v"] = True  # the re-drain retries the deletion
            return {"errors": 0}
        raise AssertionError("no third pass expected")

    monkeypatch.setattr(auto, "_do_sync", fake_do_sync)
    auto._touch_pending(tmp_path)
    auto.run_worker(tmp_path)

    assert deletion_retried["v"], "the transport-failed deletion was not retried"
    assert calls["n"] == 2


# --- Quota self-convergence (pure revert: no per-id memo, no backoff) ---


def test_push_converges_when_cap_lifts(tmp_path):
    """Convergence with NO per-id state: after the cap lifts, the very next push
    attempts the pending create, it 201s, syncs, and the quota banner clears —
    just the normal candidate scan, no manual-only requirement."""

    class _CapClient(_FakeClient):
        def upsert(self, memory):
            self.upsert_attempts += 1
            raise TragsQuotaError(_QUOTA_MSG)

    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("X", updated=_NOW))
    with pytest.raises(TragsQuotaError):
        push(engine=engine, tombstones=tombstones, client=_CapClient(), state=load(tmp_path), poppy_dir=tmp_path)
    assert load(tmp_path).remotes["https://trags.test"].errors.get("push") == _QUOTA_MSG

    # Cap lifts (delete a remote memory / upgrade): the next push just works.
    ok = _FakeClient()  # every upsert succeeds now
    push(engine=engine, tombstones=tombstones, client=ok, state=load(tmp_path), poppy_dir=tmp_path)
    assert "X" in {r["id"] for r in ok.upserts}  # X synced
    assert load(tmp_path).remotes["https://trags.test"].errors.get("push") is None  # banner cleared


def test_forget_a_capped_memory_clears_banner_on_clean_push(tmp_path):
    """Forgetting a previously-402'd memory: its tombstone (quota-exempt) sends
    and the banner clears on the next clean push — no permanent wedge."""

    class _CapClient(_FakeClient):
        def upsert(self, memory):
            self.upsert_attempts += 1
            raise TragsQuotaError(_QUOTA_MSG)

    engine, tombstones = _engine_and_tombstones(tmp_path)
    # X already occupies cloud quota; pull it before the capped push and forget.
    remote = _FakeClient(rows=[memory_to_wire(_memory("X", updated=_NOW))])
    assert (
        pull(engine=engine, tombstones=tombstones, client=remote, state=load(tmp_path), poppy_dir=tmp_path).applied_live
        == 1
    )
    with pytest.raises(TragsQuotaError):
        push(engine=engine, tombstones=tombstones, client=_CapClient(), state=load(tmp_path), poppy_dir=tmp_path)
    assert load(tmp_path).remotes["https://trags.test"].errors.get("push") == _QUOTA_MSG

    # Forget X: the live row goes, a tombstone takes its place (not quota-gated).
    engine.delete("X")
    tombstones.add(_memory("X", updated=_NOW))
    ok = _FakeClient()
    push(engine=engine, tombstones=tombstones, client=ok, state=load(tmp_path), poppy_dir=tmp_path)

    assert any(is_tombstone(r) and r["id"] == "X" for r in ok.upserts)  # deletion sent
    assert [r["id"] for r in ok.upserts if is_tombstone(r)] == ["X"]
    assert load(tmp_path).remotes["https://trags.test"].errors.get("push") is None


def test_capped_forget_lands_without_a_backoff_window(tmp_path, monkeypatch):
    """The pure revert has NO worker backoff, so a `forget` queued while the
    account is capped is processed on the very next run — the worker never skips
    a fresh pass and drops the quota-exempt tombstone."""
    from poppy.sync import auto

    monkeypatch.setattr(auto, "DEBOUNCE_S", 0)
    calls = {"n": 0}
    tombstone_sent = {"v": False}

    def fake_do_sync(poppy_dir):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TragsQuotaError("cap")  # first auto push is at the cap
        tombstone_sent["v"] = True  # the forget's tombstone lands next run
        return {"errors": 0}

    monkeypatch.setattr(auto, "_do_sync", fake_do_sync)

    # First write-triggered run hits the cap.
    auto._touch_pending(tmp_path)
    auto.run_worker(tmp_path)
    assert calls["n"] == 1

    # A `poppy forget X` right after re-triggers — the worker runs immediately
    # (no backoff window to skip it) and the deletion lands.
    auto._touch_pending(tmp_path)
    auto.run_worker(tmp_path)
    assert tombstone_sent["v"], "the capped user's forget was dropped by a window"
    assert calls["n"] == 2


# --- The manual CLI path reports failures the way the worker does ---


def test_cli_sync_push_401_then_status_shows_the_auth_error(tmp_path, monkeypatch):
    """A 401 on `poppy sync push` must land in the auth slot, so the following
    `poppy sync status` shows a `last error:` line naming it, the same as the
    auto-worker path. Previously, the CLI printed the 401 and forgot it."""
    from click.testing import CliRunner

    from poppy.cli.main import cli

    (tmp_path / "config.json").write_text('{"engine": "seed"}')
    url = "https://trags.ai"
    engine, _ = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))
    # The message Trags actually sends on a revoked key: a body, with no status
    # code in it. The recorded error has to supply the 401 itself.
    fake = _FakeClient(base_url=url, upsert_error=TragsAuthError('{"detail":"Unauthorized"}'))
    monkeypatch.setattr("poppy.cli.main._sync_client", lambda: (fake, url))

    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path), "POPPY_TRAGS_API_KEY": "usr_test", "POPPY_TELEMETRY_OFF": "1"}
    pushed = runner.invoke(cli, ["sync", "push"], env=env)
    status = runner.invoke(cli, ["sync", "status"], env=env)

    assert pushed.exit_code != 0
    assert "last error (auth)" in status.output
    assert "401" in status.output
    assert load(tmp_path).remotes[url].last_error_source == "auth"


def test_cli_sync_pull_dry_run_401_records_nothing(tmp_path, monkeypatch):
    """A `--dry-run` that hits a 401 still reports it, but writes no state."""
    from click.testing import CliRunner

    from poppy.cli.main import cli

    (tmp_path / "config.json").write_text('{"engine": "seed"}')
    url = "https://trags.ai"
    _engine_and_tombstones(tmp_path)
    fake = _FakeClient(base_url=url, rows=[], iter_error=TragsAuthError("401 Unauthorized"), iter_error_after=0)
    monkeypatch.setattr("poppy.cli.main._sync_client", lambda: (fake, url))

    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path), "POPPY_TRAGS_API_KEY": "usr_test", "POPPY_TELEMETRY_OFF": "1"}
    result = runner.invoke(cli, ["sync", "pull", "--dry-run"], env=env)

    assert result.exit_code != 0
    assert url not in load(tmp_path).remotes  # nothing written by a dry run


def test_cli_sync_push_dry_run_asks_the_server_nothing(tmp_path, monkeypatch):
    """`sync push --dry-run` uploads nothing and probes nothing, so even a
    revoked key cannot be observed: it reports what WOULD be sent, exits 0 and
    writes no state. (The test that carried this name invoked `sync pull`, so
    the push dry run had no coverage at all.)"""
    from click.testing import CliRunner

    from poppy.cli.main import cli

    (tmp_path / "config.json").write_text('{"engine": "seed"}')
    url = "https://trags.ai"
    engine, _ = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))  # something to send, if it were sending
    revoked = TragsAuthError('{"detail":"Unauthorized"}')
    fake = _FakeClient(base_url=url, upsert_error=revoked, ping_error=revoked)
    monkeypatch.setattr("poppy.cli.main._sync_client", lambda: (fake, url))

    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path), "POPPY_TRAGS_API_KEY": "usr_test", "POPPY_TELEMETRY_OFF": "1"}
    result = runner.invoke(cli, ["sync", "push", "--dry-run"], env=env)

    assert result.exit_code == 0
    assert "1 live" in result.output  # it still says what it would have sent
    assert fake.upsert_attempts == 0 and fake.ping_calls == 0
    assert url not in load(tmp_path).remotes  # nothing written by a dry run


def test_push_with_nothing_to_send_still_checks_the_key(tmp_path):
    """A push whose rows are all below the watermark must make ONE authenticated
    request, so a revoked key (deleted account) is reported instead of a clean
    `0 live, 0 tombstones, N skipped, 0 errors`."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))
    first = _FakeClient()
    push(engine=engine, tombstones=tombstones, client=first, state=SyncState(), poppy_dir=tmp_path)

    # The account is deleted: nothing new to send, and the key is now rejected.
    dead = _FakeClient(ping_error=TragsAuthError("401 Unauthorized"))
    with pytest.raises(TragsAuthError):
        push(engine=engine, tombstones=tombstones, client=dead, state=load(tmp_path), poppy_dir=tmp_path)

    assert dead.ping_calls == 1
    assert dead.upsert_attempts == 0  # the probe is not a write


def test_push_noop_probe_records_a_server_failure_in_its_own_slot(tmp_path):
    """A non-auth failure on the probe (a 500, a dead host) is recorded, so
    `sync status` and the exit code stop reading as healthy — but in the PROBE
    slot, not push's: the probe uploaded nothing, so it has no business writing
    the slot that only an upload can clear."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    client = _FakeClient(ping_error=TragsError("500 boom"))  # empty store -> no candidates

    res = push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)

    assert res.errors == 1
    rs = load(tmp_path).remotes[client.base_url]
    assert rs.errors == {"probe": "500 boom"}


def test_push_probe_failure_clears_on_the_next_healthy_idle_push(tmp_path):
    """The banner must not outlive the failure. Recorded in the push slot it
    could only be cleared by an upload, and an idle store never makes one — so a
    single transient 500 left `poppy sync push` failing for ever."""
    engine, tombstones = _engine_and_tombstones(tmp_path)  # empty: nothing to send, ever
    sick = _FakeClient(ping_error=TragsError("500 boom"))
    push(engine=engine, tombstones=tombstones, client=sick, state=load(tmp_path), poppy_dir=tmp_path)
    assert load(tmp_path).remotes[sick.base_url].errors == {"probe": "500 boom"}

    healthy = _FakeClient()
    res = push(engine=engine, tombstones=tombstones, client=healthy, state=load(tmp_path), poppy_dir=tmp_path)

    assert res.errors == 0
    assert healthy.ping_calls == 1
    assert load(tmp_path).remotes[healthy.base_url].errors == {}


def test_a_real_row_failure_survives_a_later_successful_probe(tmp_path):
    """The other half of the same rule: a probe proves the host answers, which
    says NOTHING about the row whose upload failed. Only an upload clears push."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))
    failing = _FakeClient(upsert_error=TragsError("500 row boom"))
    push(engine=engine, tombstones=tombstones, client=failing, state=load(tmp_path), poppy_dir=tmp_path)
    assert load(tmp_path).remotes[failing.base_url].errors == {"push": "500 row boom"}

    # The row is gone locally, so the next push has nothing to send and probes.
    engine.delete("m1")
    healthy = _FakeClient()
    push(engine=engine, tombstones=tombstones, client=healthy, state=load(tmp_path), poppy_dir=tmp_path)

    assert healthy.ping_calls == 1
    assert load(tmp_path).remotes[healthy.base_url].errors == {"push": "500 row boom"}


def test_push_probe_needs_a_client_that_can_ping(tmp_path):
    """A client object with no `ping` is a programming error, not a server
    failure. It used to be swallowed into a recorded `probe failed: 'X' object
    has no attribute 'ping'`, which reads as if Trags had said it."""
    engine, tombstones = _engine_and_tombstones(tmp_path)

    class _PinglessClient:
        base_url = "https://trags.test"

        def upsert(self, memory):
            return memory, True

    with pytest.raises(AttributeError):
        push(
            engine=engine,
            tombstones=tombstones,
            client=_PinglessClient(),
            state=load(tmp_path),
            poppy_dir=tmp_path,
        )


def test_push_probe_quota_takes_the_quota_path(tmp_path):
    """A 402 on a read is not expected, but if the server sends one the caller
    must get the upgrade prompt, exactly as the upsert loop gives it — not a
    generic soft failure swallowed as `except TragsError`."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    client = _FakeClient(ping_error=TragsQuotaError("Free plan memory limit reached."))

    with pytest.raises(TragsQuotaError):
        push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)

    assert load(tmp_path).remotes[client.base_url].errors == {"probe": "Free plan memory limit reached."}


def test_sync_cycle_reuses_the_pulls_request_instead_of_probing(tmp_path):
    """An idle `sync run` must not pay for two identical GETs. Pull has just
    authenticated with this same client, so push takes that as the proof its
    probe would have bought — including for clearing the banners."""
    from poppy.sync import sync as do_sync

    engine, tombstones = _engine_and_tombstones(tmp_path)
    client = _FakeClient()
    record_error(tmp_path, client.base_url, "500 boom", source="probe")
    record_error(tmp_path, client.base_url, "stale auth error", source="auth")

    res = do_sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path)

    assert client.ping_calls == 0  # the pull's request was enough
    assert res.pull.errors == 0 and res.push.errors == 0
    assert load(tmp_path).remotes[client.base_url].errors == {}


def test_push_dry_run_makes_no_probe(tmp_path):
    """`--dry-run` sends nothing and asks nothing: the probe is a request."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    client = _FakeClient()

    push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path, dry_run=True)

    assert client.ping_calls == 0
    assert load(tmp_path).remotes == {}


def test_cli_sync_push_noop_against_dead_account_exits_nonzero(tmp_path, monkeypatch):
    """End to end: `poppy sync push` with everything already pushed and a revoked
    key exits non-zero and leaves a `last error:` line behind for `sync status`."""
    from click.testing import CliRunner

    from poppy.cli.main import cli

    (tmp_path / "config.json").write_text('{"engine": "seed"}')
    url = "https://trags.ai"
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))
    push(engine=engine, tombstones=tombstones, client=_FakeClient(base_url=url), state=SyncState(), poppy_dir=tmp_path)

    dead = _FakeClient(base_url=url, ping_error=TragsAuthError('{"detail":"Unauthorized"}'))
    monkeypatch.setattr("poppy.cli.main._sync_client", lambda: (dead, url))

    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path), "POPPY_TRAGS_API_KEY": "usr_test", "POPPY_TELEMETRY_OFF": "1"}
    pushed = runner.invoke(cli, ["sync", "push"], env=env)
    status = runner.invoke(cli, ["sync", "status"], env=env)

    assert pushed.exit_code != 0
    assert "401" in pushed.output
    assert "last error (auth)" in status.output
    assert "401" in status.output  # named by us: the server's body carries no code


def test_cli_sync_push_recovers_after_a_transient_probe_failure(tmp_path, monkeypatch):
    """End to end, the blocker this fix exists for: one 5xx (or offline laptop)
    on an idle `poppy sync push` must not wedge the CLI. The failed run exits
    non-zero, the next healthy run exits 0 and `sync status` comes back clean."""
    from click.testing import CliRunner

    from poppy.cli.main import cli

    (tmp_path / "config.json").write_text('{"engine": "seed"}')
    url = "https://trags.ai"
    _engine_and_tombstones(tmp_path)  # empty store: every push is a no-op probe
    clients = [
        _FakeClient(base_url=url, ping_error=TragsError("500 boom")),
        _FakeClient(base_url=url),
        _FakeClient(base_url=url),
    ]
    monkeypatch.setattr("poppy.cli.main._sync_client", lambda: (clients.pop(0), url))

    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path), "POPPY_TRAGS_API_KEY": "usr_test", "POPPY_TELEMETRY_OFF": "1"}
    failed = runner.invoke(cli, ["sync", "push"], env=env)

    assert failed.exit_code != 0
    assert "probe: 500 boom" in failed.output
    assert load(tmp_path).remotes[url].errors == {"probe": "500 boom"}

    healthy = runner.invoke(cli, ["sync", "push"], env=env)
    status = runner.invoke(cli, ["sync", "status"], env=env)

    assert healthy.exit_code == 0
    assert "sync incomplete" not in healthy.output
    assert "last error" not in status.output
    assert load(tmp_path).remotes[url].errors == {}


# --- A tombstone converges instead of bouncing forever --------------


def _wire_counts(output: str, prefix: str) -> tuple[int, int]:
    """``(live, tombstones)`` off one report line of `poppy sync run`."""
    line = next(ln for ln in output.splitlines() if ln.strip().startswith(prefix))
    match = re.search(r"(\d+) live, (\d+) tombstones", line)
    assert match is not None, f"unparseable report line: {line}"
    return int(match.group(1)), int(match.group(2))


def test_three_sync_runs_after_a_forget_move_the_tombstone_once(tmp_path, monkeypatch):
    """`poppy forget --yes X` then three `poppy sync run` must settle.

    Reported against 0.3.0: every run showed `pull: 0 live, 1 tombstones | push:
    0 live, 1 tombstones` with `tombstoned_at` advancing, so N deletions cost N
    POSTs on every future sync and the seven-day prune never elapsed.

    Two halves, and only the second still bit. The stamp is preserved on both
    sides now, so the tombstone rises above the push watermark exactly once. The
    pull watermark is inclusive, though, so the deletion this device pushed is
    served back by every later pull — and re-recording a deletion already held
    rewrote the row and reported a tombstone applied on a store that had long
    since converged.
    """
    from click.testing import CliRunner

    from poppy.cli.main import cli

    monkeypatch.setattr("poppy.sync.auto.trigger", lambda poppy_dir, **kw: True)
    (tmp_path / "config.json").write_text('{"engine": "seed"}')
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))

    client = _FakeClient(echo=True)
    monkeypatch.setattr("poppy.cli.main._sync_client", lambda: (client, client.base_url))
    runner = CliRunner()
    env = {"POPPY_DIR": str(tmp_path), "POPPY_TELEMETRY_OFF": "1"}

    # The memory is live on the server before anything is deleted.
    assert runner.invoke(cli, ["sync", "run"], env=env).exit_code == 0
    assert client.rows_by_id["m1"]["deleted_at"] is None

    assert runner.invoke(cli, ["forget", "m1", "--yes"], env=env).exit_code == 0
    recorded = tombstones.get("m1")
    assert recorded is not None

    reports = []
    for _ in range(3):
        result = runner.invoke(cli, ["sync", "run"], env=env)
        assert result.exit_code == 0, result.output
        reports.append(result.output)
        again = tombstones.get("m1")
        # The deletion keeps its own clock AND its own row: a rewrite with the
        # same stamp still churns the token, which is what tells two deletions
        # of one id apart.
        assert again.tombstoned_at == recorded.tombstoned_at
        assert again.token == recorded.token

    # Pushed once, on the first run, and never again.
    assert [_wire_counts(o, "push:")[1] for o in reports] == [1, 0, 0]
    deletes = [u for u in client.upserts if u["id"] == "m1" and u["deleted_at"] is not None]
    assert len(deletes) == 1, "the tombstone was re-sent on a later run"
    assert deletes[0]["deleted_at"] == recorded.tombstoned_at.isoformat()

    # And applied on pull only if it carries something this device does not
    # already hold, which its own echo never does.
    assert [_wire_counts(o, "pull:")[1] for o in reports] == [0, 0, 0]
    assert client.rows_by_id["m1"]["deleted_at"] == recorded.tombstoned_at.isoformat()


def test_pull_records_the_servers_deleted_at_as_tombstoned_at(tmp_path):
    """A mirrored incoming soft-delete keeps the deletion's clock, not ours.

    Receipt time would make a deletion that happened two days ago look brand new:
    push then sends it back up as newer than anything written since, and the
    record never reaches the far side of the seven-day window, so the prune never
    elapses.
    """
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))
    deleted_at = datetime.now(timezone.utc) - timedelta(days=2)
    # `updated_at` is bumped past `deleted_at` on the way in, as a server-side
    # cleanup leaves it: the deletion's own field is the one that counts.
    row = memory_to_wire(_memory("m1", updated=deleted_at + timedelta(seconds=30)))
    row["deleted_at"] = deleted_at.isoformat()
    client = _FakeClient(rows=[row])
    state = SyncState()

    res = pull(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)

    assert res.applied_tombstones == 1
    assert engine.get("m1") is None
    recorded = tombstones.get("m1")
    assert recorded is not None
    assert recorded.tombstoned_at == deleted_at

    # A learned deletion is already sent to its source, even with no watermark.
    assert client.base_url in recorded.sent_remotes
    push(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)
    assert client.upserts == []
    assert load(tmp_path).remotes[client.base_url].last_pushed_at is None


def test_pull_applies_a_deletion_whose_snapshot_changed(tmp_path):
    """An equal `deleted_at` carrying different text is news, not this device's echo.

    Trash holds "old draft", deleted at T. The cloud serves that same deletion
    back with the body the memory ended up with and an `updated_at` after T — the
    row was corrected AFTER it was deleted, which is a state change this record
    has never seen. Reading it as an echo on the strength of the matching
    timestamp kept the superseded draft and handed it back on restore.
    """
    from dataclasses import replace

    from poppy.write_flow import restore

    engine, tombstones = _engine_and_tombstones(tmp_path)
    deleted_at = _NOW + timedelta(hours=1)
    draft = replace(_memory("m1", updated=_NOW), content="old draft")
    tombstones.add(draft, tombstoned_at=deleted_at)

    corrected = replace(draft, content="corrected final text", updated_at=deleted_at + timedelta(seconds=30))
    row = memory_to_wire(corrected)
    row["deleted_at"] = deleted_at.isoformat()
    client = _FakeClient(rows=[row])

    res = pull(engine=engine, tombstones=tombstones, client=client, state=SyncState(), poppy_dir=tmp_path)

    assert (res.applied_tombstones, res.skipped_echoes) == (1, 0)
    recorded = tombstones.get("m1")
    assert recorded.memory.content == "corrected final text"
    # The correction moved the snapshot, never the deletion's own clock.
    assert recorded.tombstoned_at == deleted_at

    # And once filed, THAT snapshot is the one the store holds: the next pull of
    # the same row says nothing new and goes quiet, token intact.
    res = pull(engine=engine, tombstones=tombstones, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert (res.applied_tombstones, res.skipped_echoes) == (0, 1)
    assert tombstones.get("m1").token == recorded.token

    assert restore(engine, tmp_path, "m1", tombstones=tombstones).memory.content == "corrected final text"


def test_dry_run_reports_an_echoed_deletion_the_way_the_real_run_does(tmp_path):
    """`--dry-run` must not promise a tombstone the real run then declines to apply.

    The row is this store's own deletion as push serialized it, `updated_at`
    bumped to the deletion's clock. Both runs have to call it what it is.
    """
    engine, tombstones = _engine_and_tombstones(tmp_path)
    tombstones.add(_memory("m1", updated=_NOW), tombstoned_at=_NOW + timedelta(hours=1))
    client = _FakeClient(rows=[tombstone_to_wire(tombstones.get("m1"))])

    dry = pull(engine=engine, tombstones=tombstones, client=client, state=SyncState(), poppy_dir=tmp_path, dry_run=True)
    wet = pull(engine=engine, tombstones=tombstones, client=client, state=SyncState(), poppy_dir=tmp_path)

    assert (dry.applied_tombstones, dry.skipped_echoes) == (0, 1)
    assert (wet.applied_tombstones, wet.skipped_echoes) == (0, 1)


def test_a_legacy_naive_tombstoned_at_is_read_as_utc(tmp_path):
    """A record written before the column was normalised carries no offset.

    Comparing it with the incoming `deleted_at` raises TypeError, and from inside
    the pull loop that ends the whole cycle rather than one row. Naive means UTC
    here: it is the only thing the column has ever held.
    """
    import sqlite3

    engine, tombstones = _engine_and_tombstones(tmp_path)
    deleted_at = _NOW + timedelta(hours=1)
    tombstones.add(_memory("m1", updated=_NOW), tombstoned_at=deleted_at)
    row = tombstone_to_wire(tombstones.get("m1"))
    conn = sqlite3.connect(str(tmp_path / "memories.db"))
    conn.execute(
        "UPDATE ui_tombstones SET tombstoned_at = ? WHERE id = ?",
        (deleted_at.replace(tzinfo=None).isoformat(), "m1"),
    )
    conn.commit()
    conn.close()

    res = pull(
        engine=engine, tombstones=tombstones, client=_FakeClient(rows=[row]), state=SyncState(), poppy_dir=tmp_path
    )

    assert (res.applied_tombstones, res.skipped_echoes, res.errors) == (0, 1, 0)


# --- An unreachable host is one offline line, not a traceback ------
#
# Every client below is a REAL TragsClient over an httpx.MockTransport, never a
# `_FakeClient`: the wrapping under test lives in `TragsClient._send`, so a fake
# that stubs `upsert`/`list_since` would never execute the code these cover.


def _wired_client(handler, url: str = "https://trags.test"):
    """A real TragsClient whose transport runs `handler` for every request."""
    import httpx

    from poppy.sync.client import TragsClient

    return TragsClient(
        base_url=url,
        api_key="usr_test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _offline_client(url: str = "https://trags.test", *, calls: list | None = None, error=None):
    """A real TragsClient whose transport refuses every connection.

    `calls` collects the attempted methods, so a test can count requests without
    reaching past the public surface into `client._send`.
    """
    import httpx

    def _refuse(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request.method)
        raise (error or httpx.ConnectError)("[Errno 61] Connection refused", request=request)

    return _wired_client(_refuse, url)


def _tombstone_row(mid: str, *, updated):
    """A wire row that deletes `mid` — the pull that costs a local memory."""
    row = memory_to_wire(_memory(mid, updated=updated))
    row["deleted_at"] = row["updated_at"]
    return row


def _cli_env(tmp_path) -> dict:
    return {"POPPY_DIR": str(tmp_path), "POPPY_TRAGS_API_KEY": "usr_test", "POPPY_TELEMETRY_OFF": "1"}


def _run_cli(tmp_path, monkeypatch, client, args: list[str]):
    from click.testing import CliRunner

    from poppy.cli.main import cli

    (tmp_path / "config.json").write_text('{"engine": "seed"}')
    monkeypatch.setattr("poppy.cli.main._sync_client", lambda: (client, client.base_url))
    return CliRunner().invoke(cli, args, env=_cli_env(tmp_path))


def test_client_wraps_transport_error_as_trags_error():
    """An httpx failure from the transport surfaces as a TragsError, so the sync
    loops' single `except TragsError` catches it. It never reaches
    `_raise_for_status`, which only sees requests that got an answer."""
    import httpx

    from poppy.sync.client import TragsTransportError

    with _offline_client() as client:
        with pytest.raises(TragsTransportError) as ei:
            client.list_since()
        with pytest.raises(TragsTransportError):
            client.upsert(memory_to_wire(_memory("m1", updated=_NOW)))

    assert isinstance(ei.value, TragsError)
    assert "https://trags.test" in str(ei.value)
    assert isinstance(ei.value.__cause__, httpx.ConnectError)  # cause kept for logs


def test_client_wraps_request_errors_outside_the_transport_subset():
    """`RequestError`, not its `TransportError` subset: a redirect loop from a
    misconfigured `trags-api-url` is the same "no usable response" to a caller
    and must not escape as the raw traceback this wrapping exists to remove."""
    import httpx

    from poppy.sync.client import TragsTransportError

    assert not issubclass(httpx.TooManyRedirects, httpx.TransportError)  # the gap being closed

    def _redirect_loop(request: httpx.Request) -> httpx.Response:
        raise httpx.TooManyRedirects("too many redirects", request=request)

    with _wired_client(_redirect_loop) as client:
        with pytest.raises(TragsTransportError):
            client.list_since()


def test_connect_refused_is_known_not_sent_but_a_timeout_is_not():
    """A refused connection proves the request never landed; a read timeout
    proves nothing, because the server may have applied it and lost the reply."""
    import httpx

    from poppy.sync.client import TragsTransportError

    with _offline_client() as client:
        with pytest.raises(TragsTransportError) as refused:
            client.list_since()
    with _offline_client(error=httpx.ReadTimeout) as client:
        with pytest.raises(TragsTransportError) as timed_out:
            client.list_since()

    assert refused.value.outcome_unknown is False
    assert timed_out.value.outcome_unknown is True


def test_push_offline_stops_after_three_rows(tmp_path):
    """A dead host is not a per-row refusal: push stops after
    MAX_CONSECUTIVE_TRANSPORT_FAILURES instead of grinding the whole backlog at
    one full timeout each, and re-raises so the worker backs off."""
    from poppy.sync import MAX_CONSECUTIVE_TRANSPORT_FAILURES
    from poppy.sync.client import TragsTransportError

    engine, tombstones = _engine_and_tombstones(tmp_path)
    for i in range(6):
        engine.ingest(_memory(f"m{i}", updated=_NOW + timedelta(seconds=i)))
    attempts: list[str] = []
    client = _offline_client(calls=attempts)

    with pytest.raises(TragsTransportError):
        push(engine=engine, tombstones=tombstones, client=client, state=SyncState(), poppy_dir=tmp_path)

    assert len(attempts) == MAX_CONSECUTIVE_TRANSPORT_FAILURES
    rs = load(tmp_path).remotes[client.base_url]
    assert rs.last_pushed_at is None  # watermark frozen, every row retries
    assert "Cannot reach Trags" in (rs.errors.get("push") or "")


def test_pull_reraises_transport_error_after_persisting(tmp_path):
    """`pull` must re-raise like `push` does, not record-and-return. The worker's
    retry, the dry-run exit code and the CLI's summary all key off the
    exception; swallowing it made an offline pull look like a clean sync."""
    from poppy.sync.client import TragsTransportError

    engine, tombstones = _engine_and_tombstones(tmp_path)
    client = _offline_client()

    with pytest.raises(TragsTransportError):
        pull(engine=engine, tombstones=tombstones, client=client, state=SyncState(), poppy_dir=tmp_path)

    rs = load(tmp_path).remotes[client.base_url]
    assert rs.last_pulled_at is None  # watermark frozen
    assert "Cannot reach Trags" in (rs.errors.get("pull") or "")  # still recorded too


def test_cli_sync_run_offline_prints_one_line_and_exits_nonzero(tmp_path, monkeypatch):
    """Done when: `poppy sync run` against an unreachable
    `trags-api-url` prints a single offline message with no traceback and exits
    non-zero. Asserted as the COMPLETE output: the one offline line, then
    click's own `Aborted!` marker, which the auth and quota handlers print too."""
    import httpx

    url = "http://127.0.0.1:9"
    engine, _ = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))
    result = _run_cli(tmp_path, monkeypatch, _offline_client(url), ["sync", "run"])

    assert result.exit_code != 0
    # Aborted by the CLI, not an httpx exception escaping to the top level.
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert not isinstance(result.exception, httpx.HTTPError)
    assert result.output == (
        f"Cannot reach Trags at {url}: [Errno 61] Connection refused. "
        "Nothing was sent and nothing changed locally. The next sync retries.\n"
        "Aborted!\n"
    )


def test_cli_sync_run_offline_reports_the_pull_it_already_applied(tmp_path, monkeypatch):
    """The pull deleted a local memory and the push then found the host gone.
    Claiming "your local memories are unchanged" here is false, and it is the
    dangerous direction: the user has no reason to go looking for `m0` before
    its tombstone ages out."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            row = _tombstone_row("m0", updated=_NOW + timedelta(seconds=9))
            return httpx.Response(200, json={"items": [row], "next_cursor": None})
        raise httpx.ConnectError("[Errno 61] Connection refused", request=request)

    engine, _ = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m0", updated=_NOW))
    engine.ingest(_memory("m1", updated=_NOW + timedelta(seconds=1)))
    result = _run_cli(tmp_path, monkeypatch, _wired_client(handler), ["sync", "run"])

    assert result.exit_code != 0
    assert engine.get("m0") is None  # the pull really did delete a local memory
    assert engine.get("m1") is not None  # and only that one
    assert "This run applied 1 pulled change before the host stopped answering." in result.output
    assert "unchanged" not in result.output
    assert "Nothing was sent" not in result.output


def test_cli_sync_run_offline_reports_the_rows_it_already_sent(tmp_path, monkeypatch):
    """Two rows reached the cloud before the host died. "Nothing was sent" is
    false, and a user who reads it will not know two memories are now up there."""
    import httpx

    posts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"items": [], "next_cursor": None})
        posts["n"] += 1
        if posts["n"] <= 2:
            return httpx.Response(201, json=json.loads(request.content.decode()))
        raise httpx.ConnectError("[Errno 61] Connection refused", request=request)

    engine, _ = _engine_and_tombstones(tmp_path)
    for i in range(8):
        engine.ingest(_memory(f"m{i}", updated=_NOW + timedelta(seconds=i)))
    result = _run_cli(tmp_path, monkeypatch, _wired_client(handler), ["sync", "run"])

    assert result.exit_code != 0
    assert posts["n"] == 5  # 2 accepted, then the breaker stops it after 3 failures
    assert "This run sent 2 rows before the host stopped answering." in result.output
    assert "Nothing was sent" not in result.output


def test_cli_sync_run_offline_read_timeout_reports_an_unknown_outcome(tmp_path, monkeypatch):
    """A timeout cannot establish that the server rejected or rolled back the
    write — the request was on the wire and the reply was lost. Say so instead
    of guessing in either direction."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"items": [], "next_cursor": None})
        raise httpx.ReadTimeout("timed out", request=request)

    engine, _ = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m1", updated=_NOW))
    result = _run_cli(tmp_path, monkeypatch, _wired_client(handler), ["sync", "run"])

    assert result.exit_code != 0
    assert "The last request may still have completed." in result.output
    assert "Nothing was sent" not in result.output
    assert "unchanged" not in result.output


def test_cli_sync_pull_dry_run_offline_exits_nonzero(tmp_path, monkeypatch):
    """A dry run persists no error, so the offline case can only reach the user
    through the exception. Exiting 0 with `1 errors` in a counter line and no
    explanation is the worst of both."""
    _engine_and_tombstones(tmp_path)
    result = _run_cli(tmp_path, monkeypatch, _offline_client(), ["sync", "pull", "--dry-run"])

    assert result.exit_code != 0
    assert "Cannot reach Trags at https://trags.test" in result.output
    assert "1 errors" not in result.output  # no bare counter line standing in for the reason


def test_cli_sync_run_dry_run_offline_exits_nonzero(tmp_path, monkeypatch):
    _engine_and_tombstones(tmp_path)
    result = _run_cli(tmp_path, monkeypatch, _offline_client(), ["sync", "run", "--dry-run"])

    assert result.exit_code != 0
    assert "Cannot reach Trags at https://trags.test" in result.output
    assert "1 errors" not in result.output


def test_worker_retries_an_offline_pull_and_keeps_the_pending_flag(tmp_path, monkeypatch):
    """With nothing to push, the pull is the whole cycle. Wrapping its transport
    failure must not cost the worker its retry: it has to reach the worker's
    `except Exception`, re-arm `sync.pending` and back off, or incoming cloud
    rows wait for an unrelated local write to trigger the next attempt."""
    from poppy.config import PoppyConfig, load_config, save_config
    from poppy.sync import auto

    monkeypatch.setattr(auto, "DEBOUNCE_S", 0)
    save_config(PoppyConfig(poppy_dir=tmp_path, engine="seed"))
    monkeypatch.setenv("POPPY_TRAGS_API_KEY", "usr_test")
    url = load_config(tmp_path).trags_api_url

    attempts: list[str] = []
    # A fresh client per round: `_do_sync` closes the one it is handed.
    monkeypatch.setattr("poppy.sync.TragsClient", lambda **kw: _offline_client(url, calls=attempts))
    sleeps: list[float] = []
    monkeypatch.setattr(auto.time, "sleep", lambda s: sleeps.append(s))

    auto._touch_pending(tmp_path)
    auto.run_worker(tmp_path, max_rounds=2)  # an empty store: nothing to push

    assert attempts == ["GET", "GET"]  # retried, not abandoned after one attempt
    assert sleeps == [1, 2]  # with backoff between rounds
    assert (tmp_path / auto.PENDING_FILENAME).exists()  # still armed for the next trigger


def test_cli_dry_run_offline_does_not_report_simulated_work_as_done(tmp_path, monkeypatch):
    """Page one holds a tombstone, page two dies mid-pagination. A dry run wrote
    nothing, so its counts are what WOULD have happened; reporting them as
    completed work is the same falsehood as the "nothing happened" line this
    issue started from, pointing the other way."""
    import httpx

    pages = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        pages["n"] += 1
        if pages["n"] == 1:
            row = _tombstone_row("m0", updated=_NOW + timedelta(seconds=9))
            return httpx.Response(200, json={"items": [row], "next_cursor": "page-2"})
        raise httpx.ConnectError("[Errno 61] Connection refused", request=request)

    engine, _ = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("m0", updated=_NOW))

    for args in (["sync", "pull", "--dry-run"], ["sync", "run", "--dry-run"]):
        pages["n"] = 0
        result = _run_cli(tmp_path, monkeypatch, _wired_client(handler), args)

        assert result.exit_code != 0, args
        assert "This run would have applied 1 pulled change" in result.output, args
        assert "This run applied" not in result.output, args
        assert engine.get("m0") is not None, args  # the dry run really did touch nothing


# --- Push: a deletion must not create the row it deletes ---------
#
# A tombstone on the wire is the memory's whole body plus a `deleted_at`, so an
# upsert meaning "remove this" CREATES the row when the remote holds nothing at
# that id. Each tombstone therefore needs proof for its own ID and remote.


@pytest.fixture
def forget_memory(monkeypatch):
    """Use the real forget flow without spawning an autosync worker."""
    monkeypatch.setattr("poppy.sync.auto.trigger", lambda poppy_dir, **kw: True)
    from poppy.write_flow import forget

    def apply(engine, tombstones, poppy_dir, memory_id):
        res = forget(engine, poppy_dir, memory_id, tombstones=tombstones)
        assert res.tombstoned
        return res

    return apply


def test_push_sends_no_deletion_to_a_remote_that_has_taken_nothing(tmp_path, forget_memory):
    """The Done-when case: a store whose sync state has no push watermark for
    this remote — exactly what `poppy setup trags` leaves behind — sends zero
    tombstones, so a memory that only ever lived here does not reach the cloud
    for the first time because the user deleted it."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("secret", updated=_NOW))
    forget_memory(engine, tombstones, tmp_path, "secret")

    client = _FakeClient(echo=True)
    state = SyncState()
    assert state.remotes.get(client.base_url) is None, "the premise: nothing has been pushed here"

    res = push(engine=engine, tombstones=tombstones, client=client, state=state, poppy_dir=tmp_path)

    assert res.sent_tombstones == 0
    assert client.upserts == []
    assert "secret" not in client.rows_by_id, "the deletion created the row it meant to remove"
    # And the push that sent nothing left the watermark alone. One that moved it
    # would put the deletion under the `iso <= watermark` candidate filter and
    # hand `purge_expired` a bound claiming the server had accepted it.
    assert load(tmp_path).remotes[client.base_url].last_pushed_at is None


def test_a_no_op_push_writes_no_watermark_and_ages_out_unknown_deletions(tmp_path):
    """An unknown ID has no pending deletion; retention does not move the mark."""
    from poppy.sync import sync as run_sync

    engine, tombstones = _engine_and_tombstones(tmp_path)
    long_ago = datetime.now(timezone.utc) - timedelta(days=30)
    tombstones.add(_memory("secret", updated=long_ago - timedelta(days=1)), tombstoned_at=long_ago)

    client = _FakeClient()
    res = run_sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path)

    assert res.push.sent_tombstones == 0
    assert client.upserts == []
    assert load(tmp_path).remotes[client.base_url].last_pushed_at is None
    assert tombstones.get("secret") is None


class _PartialFirstPushClient(_FakeClient):
    fail: str | None = "A"

    def upsert(self, memory):
        if memory["id"] == self.fail:
            raise TragsError("500 temporary failure")
        return super().upsert(memory)


def test_partial_first_push_deletion_survives_a_newer_write(tmp_path, monkeypatch):
    """A fails/B lands: B's deletion travels even while A holds the watermark."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("A", updated=_NOW))
    engine.ingest(_memory("B", updated=_NOW + timedelta(seconds=1)))
    client = _PartialFirstPushClient(echo=True)

    def do_push():
        return push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)

    first = do_push()
    assert (first.sent_live, first.errors) == (1, 1)
    assert client.rows_by_id["B"]["deleted_at"] is None
    state = load(tmp_path).remotes[client.base_url]
    assert state.last_pushed_at is None

    deleted_at = _NOW + timedelta(seconds=2)
    tombstones.add(engine.get("B"), tombstoned_at=deleted_at)
    engine.delete("B")
    second = do_push()
    assert (second.sent_tombstones, second.errors) == (1, 1)
    assert client.rows_by_id["B"]["deleted_at"] == deleted_at.isoformat()
    assert load(tmp_path).remotes[client.base_url].last_pushed_at is None

    created_at = _NOW + timedelta(seconds=3)
    engine.ingest(_memory("C", updated=created_at))
    client.fail = None
    third = do_push()
    assert (third.sent_live, third.sent_tombstones, third.errors) == (2, 0, 0)
    # B was acknowledged even though A kept the live watermark frozen.
    pull(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)
    assert do_push().sent_tombstones == 0
    assert load(tmp_path).remotes[client.base_url].last_pushed_at == created_at.isoformat()

    for _ in range(2):
        assert do_push().sent_tombstones == 0
        assert client.rows_by_id["B"]["deleted_at"] == deleted_at.isoformat()

    class AfterRetention(datetime):
        @classmethod
        def now(cls, tz=None):
            return deleted_at + timedelta(days=8)

    monkeypatch.setattr("poppy.ui.tombstones.datetime", AfterRetention)
    tombstones.purge_expired(pushed_through=created_at.isoformat(), require_sent=True)
    assert tombstones.get("B") is None
    assert client.rows_by_id["B"]["deleted_at"] == deleted_at.isoformat()


def test_forget_everything_after_partial_first_push_sends_deletions(tmp_path):
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("A", updated=_NOW))
    engine.ingest(_memory("B", updated=_NOW + timedelta(seconds=1)))
    client = _PartialFirstPushClient(echo=True)
    first = push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)
    assert (first.sent_live, first.errors) == (1, 1)
    assert load(tmp_path).remotes[client.base_url].last_pushed_at is None

    for i, mid in enumerate(("A", "B"), start=2):
        tombstones.add(engine.get(mid), tombstoned_at=_NOW + timedelta(seconds=i))
        engine.delete(mid)
    client.fail = None
    second = push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)
    assert (second.sent_live, second.sent_tombstones, second.errors) == (0, 1, 0)
    pull(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)
    third = push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)
    assert third.sent_tombstones == 0
    assert is_tombstone(client.rows_by_id["B"])
    assert "A" not in client.rows_by_id
    assert load(tmp_path).remotes[client.base_url].last_pushed_at is None


def test_a_memory_forgotten_between_two_pushes_stays_local(tmp_path, forget_memory):
    """An unrelated successful push cannot authorize a private deletion."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("first", updated=_NOW))
    client = _FakeClient(echo=True)

    push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)
    assert load(tmp_path).remotes[client.base_url].last_pushed_at == _NOW.isoformat()

    # Written and forgotten inside the window between two pushes.
    engine.ingest(_memory("secret", updated=_NOW + timedelta(seconds=1)))
    forget_memory(engine, tombstones, tmp_path, "secret")

    # Unrelated later uploads (and reopening the store) still cannot authorize X.
    for n in range(2, 5):
        engine.ingest(_memory(f"unrelated-{n}", updated=_NOW + timedelta(seconds=n)))
        tombstones = TombstoneStore(tmp_path / "memories.db")
        second = push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)
        assert second.sent_tombstones == 0
        assert "secret" not in client.rows_by_id
        assert tombstones.get("secret") is not None


@pytest.mark.parametrize("live_offset", [-1, 0, 1])
def test_first_live_success_never_sends_a_declined_tombstone(tmp_path, live_offset):
    """An unseen deletion is never sent and ages out once below the watermark."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    deleted_at = _NOW
    tombstones.add(_memory("secret", updated=_NOW - timedelta(days=1)), tombstoned_at=deleted_at)
    live_at = deleted_at + timedelta(seconds=live_offset)
    engine.ingest(_memory("live", updated=live_at))
    client = _FakeClient(echo=True)

    first = push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)
    assert (first.sent_live, first.sent_tombstones, first.skipped) == (1, 0, 1)
    assert "secret" not in client.rows_by_id
    remote = load(tmp_path).remotes[client.base_url]
    assert remote.last_pushed_at == live_at.isoformat()
    tombstones.purge_expired(pushed_through=remote.last_pushed_at, require_sent=True)
    assert (tombstones.get("secret") is None) == (live_offset >= 0)

    from poppy.sync import sync

    for _ in range(5):
        second = sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path)
        assert second.push.sent_tombstones == 0
        assert second.push.sent_live == 0
        assert "secret" not in client.rows_by_id


def test_first_candidate_402_does_not_block_deleting_a_pulled_row(tmp_path):
    engine, tombstones = _engine_and_tombstones(tmp_path)
    client = _FakeClient(
        echo=True,
        upsert_error=TragsQuotaError(_QUOTA_MSG),
        error_on_tombstones=False,
        synced_ids={"B", "D"},
    )
    client.rows_by_id = {mid: memory_to_wire(_memory(mid, updated=_NOW + timedelta(seconds=1))) for mid in ("B", "D")}
    assert (
        pull(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path).applied_live
        == 2
    )
    remote = load(tmp_path).remotes[client.base_url]
    assert remote.last_pushed_at is None

    engine.ingest(_memory("A", updated=_NOW))  # Oldest candidate is a capped create.
    deleted_at = _NOW + timedelta(seconds=2)
    tombstones.add(engine.get("B"), tombstoned_at=deleted_at)
    engine.delete("B")
    engine.ingest(_memory("D", updated=_NOW + timedelta(seconds=3)))
    with pytest.raises(TragsQuotaError):
        push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)
    assert client.upsert_attempts == 3
    assert [r["id"] for r in client.upserts] == ["B", "D"]
    assert client.rows_by_id["B"]["deleted_at"] == deleted_at.isoformat()
    remote = load(tmp_path).remotes[client.base_url]
    assert remote.last_pushed_at is None


def test_402_without_any_accepted_or_pulled_row_does_not_unlock_deletions(tmp_path):
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("create", updated=_NOW))
    tombstones.add(_memory("secret", updated=_NOW), tombstoned_at=_NOW + timedelta(seconds=1))
    capped = _FakeClient(upsert_error=TragsQuotaError(_QUOTA_MSG), error_on_tombstones=False)
    with pytest.raises(TragsQuotaError):
        push(engine=engine, tombstones=tombstones, client=capped, state=load(tmp_path), poppy_dir=tmp_path)
    assert capped.upserts == []

    tombstones.add(engine.get("create"), tombstoned_at=_NOW + timedelta(seconds=2))
    engine.delete("create")
    client = _FakeClient()
    res = push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)
    assert res.sent_tombstones == 0
    assert client.upserts == []


def test_evidence_is_scoped_to_the_remote(tmp_path, forget_memory):
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("B", updated=_NOW))
    first = _FakeClient(echo=True)
    push(engine=engine, tombstones=tombstones, client=first, state=load(tmp_path), poppy_dir=tmp_path)
    ts = forget_memory(engine, tombstones, tmp_path, "B").tombstone
    assert ts.memory.id in tombstones.known_ids(first.base_url)
    assert first.base_url not in ts.sent_remotes

    other = _FakeClient(base_url="https://other.test", echo=True)
    engine.ingest(_memory("unrelated", updated=_NOW + timedelta(seconds=1)))
    for _ in range(3):
        res = push(engine=engine, tombstones=tombstones, client=other, state=load(tmp_path), poppy_dir=tmp_path)
        assert res.sent_tombstones == 0
        assert "B" not in other.rows_by_id
    assert (
        push(
            engine=engine, tombstones=tombstones, client=first, state=load(tmp_path), poppy_dir=tmp_path
        ).sent_tombstones
        == 1
    )


def test_lost_first_upload_response_is_healed_by_next_sync_pull(tmp_path, forget_memory):
    from poppy.sync import sync

    class LostResponse(_FakeClient):
        lose_response = True

        def upsert(self, row):
            super().upsert(row)
            if self.lose_response:
                self.lose_response = False
                raise TimeoutError("server stored it; reply lost")

    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("X", updated=_NOW))
    client = LostResponse(echo=True)
    with pytest.raises(TimeoutError):
        sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path)
    assert load(tmp_path).remotes[client.base_url].last_pushed_at is None
    ts = forget_memory(engine, tombstones, tmp_path, "X").tombstone
    assert ts.memory.id not in tombstones.known_ids(client.base_url)
    assert client.rows_by_id["X"]["deleted_at"] is None

    result = sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path)
    assert result.pull.skipped_stale == 1
    assert result.push.sent_tombstones == 1
    assert client.rows_by_id["X"]["deleted_at"] == ts.tombstoned_at.isoformat()
    assert engine.get("X") is None
    assert sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path).push.sent_tombstones == 0


@pytest.mark.parametrize("cloud_id, expected", [("unrelated", 0), ("X", 1)])
def test_dry_run_tombstone_count_matches_pull_then_push(tmp_path, cloud_id, expected):
    from poppy.sync import sync

    engine, tombstones = _engine_and_tombstones(tmp_path)
    ts = tombstones.add(_memory("X", updated=_NOW))
    client = _FakeClient(echo=True)
    client.rows_by_id[cloud_id] = memory_to_wire(_memory(cloud_id, updated=_NOW))
    dry = sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path, dry_run=True)
    assert dry.push.sent_tombstones == expected
    assert tombstones.get("X") == ts
    assert tombstones.known_ids(client.base_url) == set()
    assert not (tmp_path / "sync_state.json").exists()
    assert client.upserts == []
    real = sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path)
    assert real.push.sent_tombstones == expected
    assert ("X" in client.rows_by_id) == bool(expected)


def test_pending_healed_deletion_survives_restart_failure_and_purge(tmp_path):
    engine, tombstones = _engine_and_tombstones(tmp_path)
    tombstones.add(_memory("X", updated=_NOW), tombstoned_at=_NOW + timedelta(seconds=1))
    engine.ingest(_memory("newer", updated=_NOW + timedelta(seconds=2)))
    client = _FakeClient(echo=True)
    push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)
    mark = load(tmp_path).remotes[client.base_url].last_pushed_at
    assert mark == (_NOW + timedelta(seconds=2)).isoformat()
    client.rows_by_id["X"] = memory_to_wire(_memory("X", updated=_NOW))
    pull(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)
    tombstones.purge_expired(pushed_through=mark, require_sent=True)
    assert tombstones.get("X") is not None
    reopened = TombstoneStore(tmp_path / "memories.db")
    assert "X" in reopened.known_ids(client.base_url)
    assert client.base_url not in reopened.get("X").sent_remotes
    failing = _FakeClient(upsert_error=TragsError("500 retry"))
    assert (
        push(engine=engine, tombstones=reopened, client=failing, state=load(tmp_path), poppy_dir=tmp_path).errors == 1
    )
    assert "X" in reopened.known_ids(client.base_url)
    assert client.base_url not in reopened.get("X").sent_remotes
    assert load(tmp_path).remotes[client.base_url].last_pushed_at == mark
    assert (
        push(
            engine=engine, tombstones=reopened, client=client, state=load(tmp_path), poppy_dir=tmp_path
        ).sent_tombstones
        == 1
    )
    assert is_tombstone(client.rows_by_id["X"])
    assert load(tmp_path).remotes[client.base_url].last_pushed_at == mark


def test_stale_row_on_partial_pull_records_per_id_evidence(tmp_path, forget_memory):
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("B", updated=_NOW + timedelta(seconds=1)))
    client = _FakeClient(
        rows=[memory_to_wire(_memory("B", updated=_NOW))],
        iter_error=TragsError("500 next page failed"),
        iter_error_after=1,
    )
    res = pull(engine=engine, tombstones=tombstones, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert (res.skipped_stale, res.errors) == (1, 1)
    assert load(tmp_path).remotes[client.base_url].last_pulled_at is None
    ts = forget_memory(engine, tombstones, tmp_path, "B").tombstone
    assert ts.memory.id in tombstones.known_ids(client.base_url)
    assert client.base_url not in ts.sent_remotes


@pytest.mark.parametrize("mark_offset", [-1, 0, 1, 2, 3])
def test_legacy_tombstone_resends_once_regardless_of_watermark(tmp_path, mark_offset):
    from poppy.sync.state import save

    engine, tombstones = _engine_and_tombstones(tmp_path)
    tombstones.add(_memory("X", updated=_NOW), tombstoned_at=_NOW + timedelta(seconds=2))
    tombstones._conn.execute("ALTER TABLE ui_tombstones DROP COLUMN sent_remotes")
    tombstones._conn.commit()
    client = _FakeClient(echo=True)
    save(
        tmp_path,
        SyncState(
            remotes={
                client.base_url: RemoteState(
                    last_pushed_at=(_NOW + timedelta(seconds=mark_offset)).isoformat(),
                    push_watermark_v=PUSH_WATERMARK_VERSION,
                )
            }
        ),
    )
    tombstones = TombstoneStore(tmp_path / "memories.db")
    migrated = tombstones.get("X")
    assert migrated.sent_remotes == set()
    dry = push(
        engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path, dry_run=True
    )
    assert dry.sent_tombstones == 1
    assert tombstones.get("X") == migrated
    real = push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)
    assert real.sent_tombstones == 1
    assert "sync_evidence" not in {r["name"] for r in tombstones._conn.execute("PRAGMA table_info(ui_tombstones)")}
    engine.ingest(_memory("later", updated=_NOW + timedelta(seconds=4)))
    for _ in range(3):
        assert (
            push(
                engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path
            ).sent_tombstones
            == 0
        )
    assert is_tombstone(client.rows_by_id["X"])


def test_removed_remote_acceptance_field_is_ignored(tmp_path):
    from poppy.sync.state import save

    path = tmp_path / "sync_state.json"
    path.write_text(json.dumps({"remotes": {"https://trags.test/": {"has_accepted": True}}}))
    state = load(tmp_path)
    save(tmp_path, state)
    assert "has_accepted" not in path.read_text()
    assert "https://trags.test" in state.remotes


def test_pending_legacy_announcement_does_not_upload_private_deletion(tmp_path):
    engine, tombstones = _engine_and_tombstones(tmp_path)
    tombstones.claim_leaked_copy("legacy", _NOW)
    tombstones.add(_memory("secret", updated=_NOW))
    client = _FakeClient(echo=True)
    assert (
        push(
            engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path
        ).sent_tombstones
        == 0
    )
    for _ in range(3):
        assert (
            push(
                engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path
            ).sent_tombstones
            == 0
        )
        assert "secret" not in client.rows_by_id


def test_tombstone_sent_schema_migrates_existing_rows(tmp_path):
    import sqlite3

    from poppy.ui.tombstones import SCHEMA

    db = tmp_path / "memories.db"
    # Open a pre-evidence schema, with a genuine snapshot and no new column.
    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA)
    conn.execute(
        """INSERT INTO ui_tombstones (
            id, content, memory_type, source_type, source_timestamp, confidence,
            related_to, created_at, updated_at, tombstoned_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "X",
            "private",
            "fact",
            "manual",
            _NOW.isoformat(),
            1.0,
            "[]",
            _NOW.isoformat(),
            _NOW.isoformat(),
            (_NOW + timedelta(seconds=1)).isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    store = TombstoneStore(db)
    ts = store.get("X")
    assert ts.memory.content == "private"
    assert ts.memory.updated_at == _NOW
    assert ts.sent_remotes == set()
    assert ts.token is not None
    assert TombstoneStore(db).get("X") == ts


def test_pull_evidence_survives_reopen_and_normalizes_remote_url(tmp_path, forget_memory):
    engine, tombstones = _engine_and_tombstones(tmp_path)
    client = _FakeClient(base_url="https://trags.test/", rows=[memory_to_wire(_memory("X", updated=_NOW))])
    pull(engine=engine, tombstones=tombstones, client=client, state=SyncState(), poppy_dir=tmp_path)
    reopened = TombstoneStore(tmp_path / "memories.db")
    ts = forget_memory(engine, reopened, tmp_path, "X").tombstone
    assert reopened.known_ids("https://trags.test/") == {"X"}
    assert ts.sent_remotes == set()
    assert ts.memory.source.type == "test"  # Authorship remains intact.
    assert load(tmp_path).remotes["https://trags.test"].last_pushed_at is None
    assert (
        push(
            engine=engine, tombstones=reopened, client=client, state=load(tmp_path), poppy_dir=tmp_path
        ).sent_tombstones
        == 1
    )


def test_tombstone_acknowledgement_cannot_clear_a_same_tick_replacement(tmp_path):
    engine, tombstones = _engine_and_tombstones(tmp_path)
    tombstones.note_remote_memories({"X"}, "https://trags.test")
    original = tombstones.add(_memory("X", updated=_NOW), tombstoned_at=_NOW + timedelta(seconds=1))

    class ReplacingClient(_FakeClient):
        def upsert(self, row):
            super().upsert(row)
            tombstones.add(original.memory, tombstoned_at=original.tombstoned_at)

    client = ReplacingClient()
    push(engine=engine, tombstones=tombstones, client=client, state=SyncState(), poppy_dir=tmp_path)
    replacement = tombstones.get("X")
    assert replacement.token != original.token
    assert client.base_url not in replacement.sent_remotes
    # Same timestamp is already covered, but the replacement still gets sent.
    assert (
        push(
            engine=engine, tombstones=tombstones, client=_FakeClient(), state=load(tmp_path), poppy_dir=tmp_path
        ).sent_tombstones
        == 1
    )


@pytest.mark.parametrize(
    ("known", "sent", "expected_purged"),
    [
        (set(), set(), 1),
        ({"https://trags.test"}, set(), 0),
        ({"https://trags.test"}, {"https://trags.test"}, 1),
        ({"https://trags.test", "https://other.test"}, {"https://trags.test"}, 0),
    ],
)
def test_purge_expired_requires_no_pending_deletions(tmp_path, known, sent, expected_purged):
    _, tombstones = _engine_and_tombstones(tmp_path)
    now = datetime.now(timezone.utc)
    deleted_at = now - timedelta(days=8)
    ts = tombstones.add(_memory("secret", updated=deleted_at), tombstoned_at=deleted_at)
    for url in known:
        tombstones.note_remote_memories({"secret"}, url)
    for url in sent:
        tombstones.mark_sent([ts], url)

    assert tombstones.purge_expired(pushed_through=now.isoformat(), require_sent=True) == expected_purged
    assert (tombstones.get("secret") is None) == bool(expected_purged)


def test_purge_keeps_a_deletion_pending_for_another_remote(tmp_path):
    engine, tombstones = _engine_and_tombstones(tmp_path)
    for url in ("https://trags.test", "https://other.test"):
        tombstones.note_remote_memories({"X"}, url)
    tombstones.add(_memory("X", updated=_NOW), tombstoned_at=_NOW + timedelta(seconds=1))
    first = _FakeClient()
    push(engine=engine, tombstones=tombstones, client=first, state=SyncState(), poppy_dir=tmp_path)
    tombstones.purge_expired(pushed_through=(_NOW + timedelta(seconds=2)).isoformat(), require_sent=True)
    assert tombstones.get("X") is not None
    other = _FakeClient(base_url="https://other.test")
    assert (
        push(
            engine=engine, tombstones=tombstones, client=other, state=load(tmp_path), poppy_dir=tmp_path
        ).sent_tombstones
        == 1
    )
    assert tombstones.purge_expired(pushed_through=(_NOW + timedelta(seconds=2)).isoformat(), require_sent=True) == 1


def test_dry_run_does_not_send_a_tombstone_replaced_by_pulled_restore(tmp_path):
    from poppy.sync import sync

    engine, tombstones = _engine_and_tombstones(tmp_path)
    tombstones.add(_memory("X", updated=_NOW), tombstoned_at=_NOW + timedelta(seconds=1))
    client = _FakeClient(echo=True)
    client.rows_by_id["X"] = memory_to_wire(_memory("X", updated=_NOW + timedelta(seconds=2)))
    assert (
        sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path, dry_run=True).push.sent_tombstones
        == 0
    )
    assert tombstones.get("X") is not None
    assert sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path).push.sent_tombstones == 0
    assert tombstones.get("X") is None


@pytest.mark.parametrize("upgrade", [False, True])
def test_edit_then_forget_of_synced_memory_sends_deletion(tmp_path, forget_memory, upgrade):
    from poppy.sync.state import save

    engine = SeedEngine(db_path=tmp_path / "memories.db")
    engine.ingest(_memory("X", updated=_NOW))
    client = _FakeClient(echo=True)
    if upgrade:
        # A real pre-provenance store: watermark and live row already on disk
        # when TombstoneStore first opens. No pull can rediscover this old row.
        client.rows_by_id["X"] = memory_to_wire(engine.get("X"))
        save(
            tmp_path,
            SyncState(
                remotes={
                    client.base_url: RemoteState(
                        last_pushed_at=_NOW.isoformat(),
                        last_pulled_at=(_NOW + timedelta(days=1)).isoformat(),
                        push_watermark_v=PUSH_WATERMARK_VERSION,
                    )
                }
            ),
        )
        tombstones = TombstoneStore(tmp_path / "memories.db")
    else:
        tombstones = TombstoneStore(tmp_path / "memories.db")
        push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)
    assert tombstones.known_ids(client.base_url) == {"X"}
    engine.ingest(_memory("X", updated=_NOW + timedelta(seconds=1)))
    forget_memory(engine, tombstones, tmp_path, "X")
    reopened = TombstoneStore(tmp_path / "memories.db")
    result = push(engine=engine, tombstones=reopened, client=client, state=load(tmp_path), poppy_dir=tmp_path)
    assert result.sent_tombstones == 1
    assert is_tombstone(client.rows_by_id["X"])
    assert load(tmp_path).remotes[client.base_url].last_pushed_at == _NOW.isoformat()


def test_expired_then_forgotten_row_under_watermark_is_not_sent(tmp_path, forget_memory):
    # The memory is created AFTER upgrade. Expiry excludes it from every push,
    # so an unrelated successful upload must never make its deletion sendable.
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("private", updated=_NOW, expires_at=_NOW + timedelta(seconds=1)))
    later = _NOW + timedelta(seconds=2)
    engine.ingest(_memory("public", updated=later))
    client = _FakeClient(echo=True)
    push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)
    assert tombstones.known_ids(client.base_url) == {"public"}
    forget_memory(engine, tombstones, tmp_path, "private")
    result = push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)
    assert result.sent_tombstones == 0
    assert "private" not in client.rows_by_id
    assert load(tmp_path).remotes[client.base_url].last_pushed_at == later.isoformat()


@pytest.mark.parametrize("expires_at", [_NOW, _NOW + timedelta(days=365)])
def test_migration_backfills_all_tombstone_snapshots(tmp_path, expires_at):
    from poppy.sync.state import save

    engine, tombstones = _engine_and_tombstones(tmp_path)
    tombstones.add(_memory("X", updated=_NOW, expires_at=expires_at), tombstoned_at=_NOW + timedelta(seconds=2))
    with tombstones._conn:
        tombstones._conn.execute("ALTER TABLE ui_tombstones DROP COLUMN sent_remotes")
    client = _FakeClient()
    save(
        tmp_path,
        SyncState(
            remotes={
                client.base_url: RemoteState(
                    last_pushed_at=(_NOW + timedelta(seconds=1)).isoformat(),
                    push_watermark_v=PUSH_WATERMARK_VERSION,
                )
            }
        ),
    )
    reopened = TombstoneStore(tmp_path / "memories.db")
    assert reopened.known_ids(client.base_url) == {"X"}
    assert reopened.get("X").sent_remotes == set()
    assert (
        push(
            engine=engine, tombstones=reopened, client=client, state=load(tmp_path), poppy_dir=tmp_path
        ).sent_tombstones
        == 1
    )


def test_migration_is_once_only_and_scoped_per_remote(tmp_path):
    from poppy.sync.state import save

    engine = SeedEngine(db_path=tmp_path / "memories.db")
    engine.ingest(_memory("before", updated=_NOW))
    engine.ingest(_memory("after", updated=_NOW + timedelta(microseconds=1)))
    engine.ingest(_memory("expired", updated=_NOW, expires_at=_NOW))
    engine.ingest(_memory("closet", updated=_NOW))
    with engine._conn:
        engine._conn.execute("UPDATE memories SET is_closet = 1 WHERE id = 'closet'")
    client = _FakeClient()
    other = "https://other.test"
    save(
        tmp_path,
        SyncState(
            remotes={
                client.base_url: RemoteState(last_pushed_at=_NOW.isoformat(), push_watermark_v=PUSH_WATERMARK_VERSION),
                other: RemoteState(last_pushed_at=(_NOW - timedelta(seconds=1)).isoformat()),
            }
        ),
    )
    store = TombstoneStore(tmp_path / "memories.db")
    assert store.known_ids(client.base_url) == {"before", "after", "expired"}
    assert store.known_ids(other) == {"before", "after", "expired"}
    assert store.known_ids("https://new.test") == set()
    engine.ingest(_memory("backdated-import", updated=_NOW - timedelta(seconds=1)))
    reopened = TombstoneStore(tmp_path / "memories.db")
    assert reopened.known_ids(client.base_url) == {"before", "after", "expired"}
    assert reopened.known_ids(other) == {"before", "after", "expired"}
    ts = reopened.add(engine.get("backdated-import"))
    engine.delete(ts.memory.id)
    assert (
        push(
            engine=engine, tombstones=reopened, client=client, state=load(tmp_path), poppy_dir=tmp_path
        ).sent_tombstones
        == 0
    )


@pytest.mark.parametrize("deleted_offset", [-1, 0, 1])
@pytest.mark.parametrize("failure", [TragsError("500 retry"), TragsAuthError("401 retry"), _ReadTimeout("retry")])
def test_failed_tombstone_send_never_changes_live_watermark(tmp_path, deleted_offset, failure):
    from poppy.sync.state import save

    engine, tombstones = _engine_and_tombstones(tmp_path)
    client = _FakeClient(upsert_error=failure)
    tombstones.note_remote_memories({"X"}, client.base_url)
    tombstones.add(
        _memory("X", updated=_NOW - timedelta(seconds=2)), tombstoned_at=_NOW + timedelta(seconds=deleted_offset)
    )
    save(
        tmp_path,
        SyncState(
            remotes={
                client.base_url: RemoteState(
                    last_pushed_at=_NOW.isoformat(),
                    push_watermark_v=PUSH_WATERMARK_VERSION,
                )
            }
        ),
    )
    try:
        result = push(engine=engine, tombstones=tombstones, client=client, state=load(tmp_path), poppy_dir=tmp_path)
        assert result.errors == 1
    except (TragsAuthError, _ReadTimeout):
        pass
    assert client.upsert_attempts == 1
    assert load(tmp_path).remotes[client.base_url].last_pushed_at == _NOW.isoformat()
    assert tombstones.get("X").sent_remotes == set()


def test_failed_tombstone_does_not_freeze_later_live_upload(tmp_path):
    engine, tombstones = _engine_and_tombstones(tmp_path)
    tombstones.note_remote_memories({"A"}, "https://trags.test")
    tombstones.add(_memory("A", updated=_NOW), tombstoned_at=_NOW)
    engine.ingest(_memory("B", updated=_NOW + timedelta(seconds=1)))
    client = _PartialFirstPushClient()
    result = push(engine=engine, tombstones=tombstones, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert (result.sent_live, result.errors) == (1, 1)
    assert load(tmp_path).remotes[client.base_url].last_pushed_at == (_NOW + timedelta(seconds=1)).isoformat()


@pytest.mark.parametrize("holds_live", [False, True])
def test_pulled_deletion_is_sent_at_creation_and_dry_run_agrees(tmp_path, holds_live):
    from poppy.sync import sync

    engine, tombstones = _engine_and_tombstones(tmp_path)
    wrote_at = datetime.now(timezone.utc) - timedelta(hours=1)
    memory = _memory("X", updated=wrote_at)
    if holds_live:
        engine.ingest(memory)
        tombstones.note_remote_memories({"X"}, "https://trags.test")
    row = memory_to_wire(_memory("X", updated=wrote_at + timedelta(seconds=1)))
    row["deleted_at"] = row["updated_at"]
    client = _FakeClient(rows=[row])
    dry = sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path, dry_run=True)
    assert (dry.push.sent_live, dry.push.sent_tombstones) == (0, 0)
    assert tombstones.get("X") is None
    real = sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path)
    assert real.push.sent_tombstones == 0
    assert tombstones.get("X").sent_remotes == {client.base_url}
    assert client.upserts == []
    assert load(tmp_path).remotes[client.base_url].last_pushed_at is None


def test_push_batches_provenance_before_watermark_persist(tmp_path, monkeypatch):
    import poppy.sync as sync_module

    engine, tombstones = _engine_and_tombstones(tmp_path)
    for n in range(20):
        engine.ingest(_memory(str(n), updated=_NOW + timedelta(seconds=n)))
    client = _FakeClient()
    commits = []
    tombstones._conn.set_trace_callback(lambda sql: commits.append(sql) if sql == "COMMIT" else None)

    def crash_before_state_save(*args):
        reopened = TombstoneStore(tmp_path / "memories.db")
        assert reopened.known_ids(client.base_url) == {str(n) for n in range(20)}
        assert len(commits) == 1
        raise RuntimeError("interrupted before state save")

    monkeypatch.setattr(sync_module, "mutate_remote", crash_before_state_save)
    with pytest.raises(RuntimeError, match="interrupted"):
        push(engine=engine, tombstones=tombstones, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert not (tmp_path / "sync_state.json").exists()


def test_pull_commits_known_ids_once_per_batch(tmp_path):
    engine, tombstones = _engine_and_tombstones(tmp_path)
    client = _FakeClient(rows=[memory_to_wire(_memory(str(n), updated=_NOW)) for n in range(20)])
    commits = []
    tombstones._conn.set_trace_callback(lambda sql: commits.append(sql) if sql == "COMMIT" else None)
    result = pull(engine=engine, tombstones=tombstones, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert result.applied_live == 20
    assert tombstones.known_ids(client.base_url) == {str(n) for n in range(20)}
    assert len(commits) == 1


def test_migration_preserves_sent_marks_and_discards_snapshot_guesses(tmp_path):
    _, tombstones = _engine_and_tombstones(tmp_path)
    for mid in ("sent", "pending", "unseen"):
        tombstones.add(_memory(mid, updated=_NOW))
    with tombstones._conn:
        tombstones._conn.execute("ALTER TABLE ui_tombstones DROP COLUMN sent_remotes")
        tombstones._conn.execute("ALTER TABLE ui_tombstones ADD COLUMN sync_evidence TEXT")
        for mid in ("sent", "pending", "unseen"):
            tombstones._conn.execute(
                "UPDATE ui_tombstones SET sync_evidence = ? WHERE id = ?",
                (json.dumps({"https://trags.test": mid}), mid),
            )
    reopened = TombstoneStore(tmp_path / "memories.db")
    assert reopened.known_ids("https://trags.test") == {"sent"}
    assert reopened.get("sent").sent_remotes == {"https://trags.test"}
    assert reopened.get("pending").sent_remotes == set()
    assert reopened.get("unseen").sent_remotes == set()
    assert "sync_evidence" not in {r["name"] for r in reopened._conn.execute("PRAGMA table_info(ui_tombstones)")}


def test_pulled_deletion_echo_from_another_remote_is_not_reannounced(tmp_path):
    from poppy.sync import sync

    engine, tombstones = _engine_and_tombstones(tmp_path)
    memory = _memory("X", updated=datetime.now(timezone.utc) - timedelta(hours=1))
    deletion = tombstones.add(memory, sent_to="https://first.test")
    client = _FakeClient(rows=[tombstone_to_wire(deletion)])
    dry = sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path, dry_run=True)
    assert (dry.pull.skipped_echoes, dry.push.sent_tombstones) == (1, 0)
    assert tombstones.get("X").sent_remotes == {"https://first.test"}
    real = sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path)
    assert (real.pull.skipped_echoes, real.push.sent_tombstones) == (1, 0)
    assert tombstones.get("X").sent_remotes == {"https://first.test", client.base_url}
    assert client.upserts == []


@pytest.mark.parametrize("watermark", [None, "garbage"])
def test_migration_backfills_without_a_valid_watermark(tmp_path, watermark):
    from poppy.sync.state import save

    engine = SeedEngine(db_path=tmp_path / "memories.db")
    engine.ingest(_memory("X", updated=_NOW))
    save(tmp_path, SyncState(remotes={"https://trags.test": RemoteState(last_pushed_at=watermark)}))
    tombstones = TombstoneStore(tmp_path / "memories.db")
    assert tombstones.known_ids("https://trags.test") == {"X"}


def test_upgrade_forget_pulled_row_above_push_watermark_sends_deletion(tmp_path, forget_memory):
    from poppy.sync import sync
    from poppy.sync.state import save

    db = tmp_path / "memories.db"
    engine = SeedEngine(db_path=db)
    theirs = _memory("theirs", updated=_NOW + timedelta(hours=1))
    engine.ingest(theirs)
    client = _FakeClient(echo=True)
    client.rows_by_id[theirs.id] = memory_to_wire(theirs)
    save(
        tmp_path,
        SyncState(
            remotes={
                client.base_url: RemoteState(
                    last_pushed_at=_NOW.isoformat(),
                    last_pulled_at=(_NOW + timedelta(hours=2)).isoformat(),
                    push_watermark_v=PUSH_WATERMARK_VERSION,
                )
            }
        ),
    )
    tombstones = TombstoneStore(db)  # Upgrade after the old client's pull.
    forget_memory(engine, tombstones, tmp_path, theirs.id)
    result = sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path)
    assert result.pull.applied_live == 0  # The cursor cannot rediscover theirs.
    assert result.push.sent_tombstones == 1
    assert is_tombstone(client.rows_by_id[theirs.id])


def test_upgrade_forget_offline_edit_above_push_watermark_sends_deletion(tmp_path, forget_memory):
    from poppy.sync import sync
    from poppy.sync.state import save

    db = tmp_path / "memories.db"
    engine = SeedEngine(db_path=db)
    client = _FakeClient(echo=True)
    # Old client uploaded X at 12:00, then pulled another row at 13:00.
    original = _memory("X", updated=_NOW)
    other = _memory("other", updated=_NOW + timedelta(hours=1))
    client.upsert(memory_to_wire(original))
    client.rows_by_id[other.id] = memory_to_wire(other)
    engine.ingest(original)
    engine.ingest(other)
    save(
        tmp_path,
        SyncState(
            remotes={
                client.base_url: RemoteState(
                    last_pushed_at=_NOW.isoformat(),
                    last_pulled_at=other.updated_at.isoformat(),
                    push_watermark_v=PUSH_WATERMARK_VERSION,
                )
            }
        ),
    )
    edited = _memory("X", updated=_NOW + timedelta(hours=2))
    edited.content = "offline edit never pushed"
    engine.ingest(edited)  # 14:00 edit BEFORE upgrade.
    tombstones = TombstoneStore(db)
    forget_memory(engine, tombstones, tmp_path, "X")
    client.upserts.clear()
    result = sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path)
    assert result.push.sent_tombstones == 1
    assert is_tombstone(client.rows_by_id["X"])
    assert client.rows_by_id["X"]["content"] == edited.content


def test_upgrade_future_old_watermark_does_not_acknowledge_unsent_deletion(tmp_path):
    from poppy.sync.state import save

    engine, tombstones = _engine_and_tombstones(tmp_path)
    client = _FakeClient(echo=True)
    original = _memory("X", updated=_NOW)
    client.upsert(memory_to_wire(original))  # Uploaded at 12:00.
    tombstones.add(original, tombstoned_at=_NOW + timedelta(hours=2))  # Forgotten at 14:00, still unsent.
    with tombstones._conn:
        tombstones._conn.execute("ALTER TABLE ui_tombstones DROP COLUMN sent_remotes")
    save(
        tmp_path,
        SyncState(
            remotes={
                client.base_url: RemoteState(
                    last_pushed_at=(_NOW + timedelta(hours=3)).isoformat(),
                    last_pulled_at=(_NOW + timedelta(hours=1)).isoformat(),
                    push_watermark_v=0,
                )
            }
        ),
    )
    reopened = TombstoneStore(tmp_path / "memories.db")
    assert reopened.get("X").sent_remotes == set()
    result = push(engine=engine, tombstones=reopened, client=client, state=load(tmp_path), poppy_dir=tmp_path)
    assert result.sent_tombstones == 1
    assert is_tombstone(client.rows_by_id["X"])
    assert reopened.get("X").sent_remotes == {client.base_url}
    assert (
        push(
            engine=engine, tombstones=reopened, client=client, state=load(tmp_path), poppy_dir=tmp_path
        ).sent_tombstones
        == 0
    )


@pytest.mark.parametrize("state_file", ["absent", "empty"])
def test_upgrade_without_remote_then_setup_never_uploads_existing_deletions(tmp_path, monkeypatch, state_file):
    import httpx
    from click.testing import CliRunner

    from poppy.cli.main import cli
    from poppy.setup import trags as setup_trags
    from poppy.sync import sync
    from poppy.sync.state import save

    engine, tombstones = _engine_and_tombstones(tmp_path)
    tombstones.add(_memory("secret", updated=_NOW))
    engine.ingest(_memory("pre-upgrade-live", updated=_NOW))
    with tombstones._conn:
        tombstones._conn.execute("ALTER TABLE ui_tombstones DROP COLUMN sent_remotes")
    if state_file == "empty":
        save(tmp_path, SyncState())
    reopened = TombstoneStore(tmp_path / "memories.db")
    assert reopened.known_ids("https://trags.test") == set()
    assert "sent_remotes" in {r["name"] for r in reopened._conn.execute("PRAGMA table_info(ui_tombstones)")}

    # Run the real setup command, substituting only external authorization and
    # keychain interactions. Setup saves config, with no sync history yet.
    monkeypatch.setenv("POPPY_DIR", str(tmp_path))
    monkeypatch.setattr("poppy.keychain.available", lambda: False)
    monkeypatch.setattr(setup_trags.webbrowser, "open", lambda *_a, **_kw: True)
    monkeypatch.setattr(setup_trags, "_decrypt_api_key", lambda *_a: "usr_test_setup")
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *_a, **_kw: httpx.Response(
            201, json={"code": "TEST", "setup_url": "https://trags.test/setup", "poll_interval_seconds": 0.001}
        ),
    )
    monkeypatch.setattr(httpx, "get", lambda *_a, **_kw: httpx.Response(200, json={"api_key_encrypted": "test"}))
    result = CliRunner().invoke(cli, ["setup", "trags", "--api-url", "https://trags.test"])
    assert result.exit_code == 0, result.output
    assert load(tmp_path).remotes == {}
    client = _FakeClient(echo=True)
    reopened = TombstoneStore(tmp_path / "memories.db")
    assert reopened.known_ids(client.base_url) == set()
    first = sync(engine=engine, tombstones=reopened, client=client, poppy_dir=tmp_path)
    assert (first.push.sent_live, first.push.sent_tombstones) == (1, 0)
    reopened = TombstoneStore(tmp_path / "memories.db")
    second = sync(engine=engine, tombstones=reopened, client=client, poppy_dir=tmp_path)
    assert second.push.sent_tombstones == 0
    assert "secret" not in client.rows_by_id


@pytest.mark.parametrize(
    "bad_state",
    ["{", "[]", '{"remotes": []}', '{"remotes": {"https://trags.test": []}}', "invalid-utf8", "unreadable"],
)
def test_migration_invalid_state_leaves_marker_absent_and_retries(tmp_path, monkeypatch, bad_state):
    from pathlib import Path

    from poppy.sync.state import save

    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("X", updated=_NOW))
    tombstones.add(_memory("deleted", updated=_NOW))
    with tombstones._conn:
        tombstones._conn.execute("ALTER TABLE ui_tombstones DROP COLUMN sent_remotes")
    state_path = tmp_path / "sync_state.json"
    state_path.write_bytes(b"\xff" if bad_state == "invalid-utf8" else bad_state.encode())
    read_text = Path.read_text

    def unreadable(path, *args, **kwargs):
        if path == state_path:
            raise PermissionError("unreadable test state")
        return read_text(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        if bad_state == "unreadable":
            scoped.setattr(Path, "read_text", unreadable)
        with pytest.raises(RuntimeError, match="Cannot migrate sync provenance.*sync_state.json.*Repair"):
            TombstoneStore(tmp_path / "memories.db")
    assert "sent_remotes" not in {r["name"] for r in tombstones._conn.execute("PRAGMA table_info(ui_tombstones)")}
    assert tombstones.known_ids("https://trags.test") == set()
    save(tmp_path, SyncState(remotes={"https://trags.test": RemoteState()}))
    reopened = TombstoneStore(tmp_path / "memories.db")
    assert reopened.known_ids("https://trags.test") == {"X", "deleted"}
    assert reopened.get("deleted").sent_remotes == set()


def test_migration_relative_paths_load_state_beside_resolved_db(tmp_path, monkeypatch):
    from pathlib import Path

    from poppy.runtime import get_poppy_dir
    from poppy.sync.state import save

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POPPY_DIR", ".poppy")
    poppy_dir = get_poppy_dir()
    assert poppy_dir == Path(".poppy")
    poppy_dir.mkdir()
    engine, tombstones = _engine_and_tombstones(poppy_dir)
    engine.ingest(_memory("X", updated=_NOW))
    tombstones.add(_memory("deleted", updated=_NOW))
    with tombstones._conn:
        tombstones._conn.execute("ALTER TABLE ui_tombstones DROP COLUMN sent_remotes")
    save(poppy_dir, SyncState(remotes={"https://trags.test": RemoteState()}))
    (tmp_path / "sync_state.json").write_text("unreadable JSON")
    loaded_paths = []

    def track_load(path, **kwargs):
        loaded_paths.append(path)
        return load(path, **kwargs)

    monkeypatch.setattr("poppy.sync.state.load", track_load)
    reopened = TombstoneStore(poppy_dir / "memories.db")
    assert loaded_paths == [poppy_dir.resolve()]
    assert reopened.known_ids("https://trags.test") == {"X", "deleted"}
    assert reopened.get("deleted").sent_remotes == set()


def test_migration_in_memory_does_not_load_cwd_sync_state(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.chdir(tmp_path)
    (tmp_path / "sync_state.json").write_text("unreadable JSON")

    def unexpected_load(*args, **kwargs):
        pytest.fail("in-memory database must not load sync state")

    monkeypatch.setattr("poppy.sync.state.load", unexpected_load)
    tombstones = TombstoneStore(Path(":memory:"))
    assert tombstones.known_ids("https://trags.test") == set()
    assert "sent_remotes" in {r["name"] for r in tombstones._conn.execute("PRAGMA table_info(ui_tombstones)")}


@pytest.mark.parametrize("engine_kind", ["seed", "bloom"])
def test_open_removes_existing_derived_rows_without_upload(tmp_path, engine_kind):
    from poppy.engine._closet_engine import ClosetHybridEngine
    from poppy.sync import sync

    engine, tombstones = _engine_and_tombstones(tmp_path)
    for mid in ("parent", "derived", "lookalike_closet_alice", "unverified"):
        engine.ingest(_memory(mid, updated=_NOW))
    with engine._conn:
        engine._conn.execute("UPDATE memories SET is_closet = 1 WHERE id = 'derived'")
        engine._conn.execute("UPDATE memories SET is_closet = 2 WHERE id = 'unverified'")
        engine._conn.execute("CREATE TABLE memory_embeddings (id TEXT PRIMARY KEY, embedding BLOB)")
        engine._conn.execute("INSERT INTO memory_embeddings VALUES ('derived', X'00')")
        engine._conn.execute(
            "INSERT INTO legacy_closet_ids (id, legacy_updated_at) VALUES ('derived', ?)", (_NOW.isoformat(),)
        )
    tombstones.add(engine.get("derived"))
    tombstones.note_remote_memories({"derived"}, "https://trags.test")
    engine._conn.close()
    factory = SeedEngine if engine_kind == "seed" else ClosetHybridEngine
    engine = factory(tmp_path / "memories.db")
    assert engine.get("derived") is None
    assert engine.get("parent") is not None
    assert engine.get("lookalike_closet_alice") is not None
    assert engine.get("unverified") is not None
    assert not engine._conn.execute("SELECT 1 FROM memory_fts WHERE id = 'derived'").fetchone()
    assert not engine._conn.execute("SELECT 1 FROM memory_embeddings WHERE id = 'derived'").fetchone()
    assert tombstones.get("derived") is None
    client = _FakeClient(rows=[memory_to_wire(_memory("derived", updated=_NOW))])
    for _ in range(2):
        result = sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path)
        assert result.pull.errors == result.push.errors == 0
        assert engine.get("derived") is None
    assert all(row["id"] != "derived" for row in client.upserts)
    engine._conn.close()
    engine = factory(tmp_path / "memories.db")
    assert engine.get("derived") is None


def test_existing_cloud_cleanup_tombstone_does_not_become_restorable(tmp_path):
    from poppy.sync import sync

    engine, tombstones = _engine_and_tombstones(tmp_path)
    row = _release_030_redacted_row("removed-copy")
    client = _FakeClient(rows=[row])
    for _ in range(2):
        result = sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path)
        assert result.pull.errors == result.push.errors == 0
        assert engine.get(row["id"]) is None
        assert tombstones.get(row["id"]) is None
        state = load(tmp_path).remotes[client.base_url]
        assert state.last_pulled_at == row["updated_at"]
        assert state.pulled_count == state.pushed_count == 0
    assert client.upserts == []


def test_cleanup_deletion_outlives_trash_but_allows_newer_recreation(tmp_path):
    from poppy.sync import sync

    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("derived", updated=_NOW))
    with engine._conn:
        engine._conn.execute("UPDATE memories SET is_closet = 1")
    engine._conn.close()
    engine = SeedEngine(tmp_path / "memories.db")
    deleted_at = _NOW + timedelta(days=1)
    with engine._conn:
        engine._conn.execute("UPDATE sync_local_deletions SET deleted_at = ?", (deleted_at.isoformat(),))
    tombstones.purge_expired(pushed_through=datetime.now(timezone.utc).isoformat())
    stale = memory_to_wire(_memory("derived", updated=_NOW))
    client = _FakeClient(rows=[stale])
    result = sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path)
    assert result.pull.skipped_stale == 1
    assert result.pull.errors == result.push.errors == 0
    assert engine.get("derived") is None
    assert client.upserts == []

    newer = _memory("derived", updated=deleted_at + timedelta(seconds=1))
    newer.content = "An independent replacement"
    client = _FakeClient(rows=[memory_to_wire(newer)])
    result = sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path)
    assert result.pull.applied_live == 1
    assert result.pull.errors == result.push.errors == 0
    assert engine.get("derived").content == newer.content


def test_cleanup_failure_rolls_back_rows_vectors_and_announcements(tmp_path):
    import sqlite3

    engine, _ = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("derived", updated=_NOW))
    with engine._conn:
        engine._conn.execute("UPDATE memories SET is_closet = 1")
        engine._conn.execute("INSERT INTO legacy_closet_ids (id) VALUES ('derived')")
        engine._conn.execute("CREATE TABLE memory_embeddings (id TEXT PRIMARY KEY, embedding BLOB)")
        engine._conn.execute("INSERT INTO memory_embeddings VALUES ('derived', X'00')")
        engine._conn.execute(
            "CREATE TRIGGER refuse_cleanup BEFORE DELETE ON memories "
            "BEGIN SELECT RAISE(ABORT, 'cleanup interrupted'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="cleanup interrupted"):
        SeedEngine(tmp_path / "memories.db")
    assert engine.get("derived") is not None
    assert engine._conn.execute("SELECT 1 FROM memory_embeddings WHERE id = 'derived'").fetchone()
    assert engine._conn.execute("SELECT 1 FROM legacy_closet_ids WHERE id = 'derived'").fetchone()
    assert not engine._conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'sync_local_deletions'").fetchone()
    with engine._conn:
        engine._conn.execute("DROP TRIGGER refuse_cleanup")
    reopened = SeedEngine(tmp_path / "memories.db")
    assert reopened.get("derived") is None


def _release_030_redacted_row(memory_id):
    # This is the shape the 0.3.0 release sends.
    return {
        "id": memory_id,
        "content": "[poppy: derived per-speaker copy removed]",
        "memory_type": "fact",
        "project": None,
        "source_type": None,
        "source_session_id": None,
        "source_timestamp": _NOW.isoformat(),
        "confidence": 1.0,
        "related_to": [],
        "expires_at": None,
        "superseded_by": None,
        "created_at": _NOW.isoformat(),
        "updated_at": _NOW.isoformat(),
        "deleted_at": _NOW.isoformat(),
    }


@pytest.mark.parametrize("local_version", [None, "older", "equal", "newer"])
def test_release_030_redacted_deletion_round_trip(tmp_path, local_version):
    from poppy.sync import sync
    from poppy.write_flow import restore

    engine, tombstones = _engine_and_tombstones(tmp_path)
    row = _release_030_redacted_row("deleted")
    if local_version is not None:
        seconds = {"older": -1, "equal": 0, "newer": 1}[local_version]
        engine.ingest(_memory(row["id"], updated=_NOW + timedelta(seconds=seconds)))
    client = _FakeClient(rows=[row])
    for _ in range(3):
        result = sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path)
        assert result.pull.errors == result.push.errors == 0
        assert load(tmp_path).remotes[client.base_url].last_pulled_at == row["updated_at"]
        assert tombstones.get(row["id"]) is None
        if local_version == "newer":
            assert engine.get(row["id"]).content == "content for deleted"
        else:
            assert engine.get(row["id"]) is None
            assert not restore(engine, tmp_path, row["id"], tombstones=tombstones).found
    assert all(r["content"] != row["content"] for r in client.upserts)
    if local_version != "newer":
        assert client.upserts == []
        # A reset pull cursor and expired Trash must not admit a stale live copy.
        tombstones.purge_expired(pushed_through=datetime.now(timezone.utc).isoformat())
        engine._conn.close()
        engine = SeedEngine(tmp_path / "memories.db")
        stale_client = _FakeClient(rows=[memory_to_wire(_memory(row["id"], updated=_NOW))])
        result = pull(engine=engine, tombstones=tombstones, client=stale_client, state=SyncState(), poppy_dir=tmp_path)
        assert result.skipped_stale == 1
        assert engine.get(row["id"]) is None


def test_release_030_redacted_deletion_dry_run_does_not_write(tmp_path):
    from poppy.sync import sync

    engine, tombstones = _engine_and_tombstones(tmp_path)
    client = _FakeClient(rows=[_release_030_redacted_row("deleted")])
    result = sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path, dry_run=True)
    assert result.pull.errors == result.push.errors == 0
    assert not engine._conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'sync_local_deletions'").fetchone()
    assert tombstones.list_all() == []
    assert client.upserts == []


def _release_030_decode(row):
    """The 0.3.0 reader's field mapping, independent of the current serializer."""

    def parse(value):
        return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None

    created = parse(row.get("created_at"))
    if created is None:
        raise ValueError("missing created_at")
    return Memory(
        id=row["id"],
        content=row["content"],
        memory_type=row["memory_type"],
        source=Source(
            type=row.get("source_type") or "trags-sync",
            session_id=row.get("source_session_id"),
            timestamp=parse(row.get("source_timestamp")) or created,
        ),
        project=row.get("project"),
        related_to=list(row.get("related_to") or []),
        created_at=created,
        updated_at=parse(row.get("updated_at")) or created,
        confidence=float(row.get("confidence") or 1.0),
        expires_at=parse(row.get("expires_at")),
    )


class _Release030Peer:
    """0.3.0's ordinary-row apply and push-watermark rules for a protocol test.

    Repeated pull boundaries can reapply a deletion, but push only sends versions
    strictly above its watermark. Deletion snapshots retain the event timestamp.
    """

    def __init__(self):
        self.live = {}
        self.deleted = {}
        self.last_pulled = None
        self.last_pushed = None

    def exchange(self, rows):
        for row in sorted(rows, key=lambda r: r["updated_at"]):
            memory = _release_030_decode(row)
            existing = self.live.get(memory.id)
            if row.get("deleted_at") is not None:
                if existing is None or existing.updated_at <= memory.updated_at:
                    self.live.pop(memory.id, None)
                    self.deleted[memory.id] = dict(row)
            else:
                deletion = self.deleted.get(memory.id)
                if deletion is not None and datetime.fromisoformat(deletion["deleted_at"]) >= memory.updated_at:
                    continue
                if existing is None or existing.updated_at <= memory.updated_at:
                    self.live[memory.id] = memory
                    self.deleted.pop(memory.id, None)
            self.last_pulled = max(self.last_pulled or row["updated_at"], row["updated_at"])
        # The release writes the same ordinary shape it reads. These tests use
        # UTC timestamps, matching the release's normalized watermark ordering.
        candidates = [dict(row) for row in rows if row["id"] in self.live] + list(self.deleted.values())
        outgoing = [row for row in candidates if self.last_pushed is None or row["updated_at"] > self.last_pushed]
        if outgoing:
            self.last_pushed = max(row["updated_at"] for row in outgoing)
        return outgoing


@pytest.mark.parametrize("kind", ["live", "deleted", "superseded"])
def test_new_wire_shapes_round_trip_through_release_030_peer(tmp_path, kind):
    from poppy.sync import sync

    engine, tombstones = _engine_and_tombstones(tmp_path)
    memory = _memory("shared", updated=_NOW)
    client = _FakeClient(echo=True)
    engine.ingest(memory)
    if kind != "live":
        engine.delete(memory.id)
        tombstones.note_remote_memories({memory.id}, client.base_url)
        tombstones.add(
            memory,
            tombstoned_at=_NOW + timedelta(seconds=1),
            superseded_by="replacement" if kind == "superseded" else None,
        )
    result = sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path)
    assert result.pull.errors == result.push.errors == 0
    assert len(client.upserts) == 1
    row = client.upserts[0]
    # This is the ordinary row shape the 0.3.0 release sends.
    assert row == {
        "id": "shared",
        "content": "content for shared",
        "memory_type": "fact",
        "project": "proj",
        "source_type": "test",
        "source_session_id": "s1",
        "source_timestamp": _NOW.isoformat(),
        "confidence": 1.0,
        "related_to": [],
        "expires_at": None,
        "superseded_by": "replacement" if kind == "superseded" else None,
        "created_at": _NOW.isoformat(),
        "updated_at": (_NOW if kind == "live" else _NOW + timedelta(seconds=1)).isoformat(),
        "deleted_at": None if kind == "live" else (_NOW + timedelta(seconds=1)).isoformat(),
    }
    peer = _Release030Peer()
    if kind != "live":
        peer.live[memory.id] = memory
    outgoing = peer.exchange([row])
    assert outgoing == [row]
    for _ in range(3):
        echoed = _FakeClient(rows=outgoing)
        result = sync(engine=engine, tombstones=tombstones, client=echoed, poppy_dir=tmp_path)
        assert result.pull.errors == result.push.errors == 0
        assert echoed.upserts == []
        assert peer.exchange([row]) == []
        assert peer.last_pulled == peer.last_pushed == row["updated_at"]
        if kind == "live":
            assert engine.get(memory.id).content == peer.live[memory.id].content
        else:
            assert engine.get(memory.id) is None
            assert memory.id not in peer.live
            assert peer.deleted[memory.id]["superseded_by"] == row["superseded_by"]
    if kind != "live":
        assert peer.exchange([memory_to_wire(memory)]) == []
        assert memory.id not in peer.live


def test_cleanup_matches_snapshot_timestamps_as_instants(tmp_path):
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("derived", updated=_NOW))
    tombstones.add(engine.get("derived"))
    tombstones.note_remote_memories({"derived"}, "https://trags.test")
    with engine._conn:
        engine._conn.execute(
            "UPDATE memories SET is_closet = 1, created_at = ? WHERE id = 'derived'",
            (_NOW.astimezone(timezone(timedelta(hours=2))).isoformat(),),
        )
    reopened = SeedEngine(tmp_path / "memories.db")
    assert reopened.get("derived") is None
    assert tombstones.get("derived") is None
    client = _FakeClient()
    push(engine=reopened, tombstones=tombstones, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert client.upserts == []


def test_cleanup_records_the_rows_own_time_not_the_upgrade_clock(tmp_path):
    """A derived row removed on upgrade must not hide a version written since.

    Stamped with the upgrade's clock the record sits above every version of the
    id written before it, so a real memory another device wrote at that id
    months ago is refused on the next pull and the watermark moves past it: the
    remote version is lost here for good.
    """
    from poppy.sync import sync

    engine, tombstones = _engine_and_tombstones(tmp_path)
    written = _NOW - timedelta(days=200)
    engine.ingest(_memory("derived", updated=written))
    with engine._conn:
        engine._conn.execute("UPDATE memories SET is_closet = 1")
    engine._conn.close()

    engine = SeedEngine(tmp_path / "memories.db")
    stamp = engine._conn.execute("SELECT deleted_at FROM sync_local_deletions").fetchone()[0]
    assert datetime.fromisoformat(stamp) == written

    # Another device reclaimed the id between that write and this upgrade.
    reclaimed = _memory("derived", updated=written + timedelta(days=1))
    reclaimed.content = "An independent note another device wrote"
    client = _FakeClient(rows=[memory_to_wire(reclaimed)])
    result = sync(engine=engine, tombstones=tombstones, client=client, poppy_dir=tmp_path)

    assert result.pull.errors == 0
    assert engine.get("derived").content == "An independent note another device wrote"


def test_an_unreadable_stamp_falls_back_to_the_past_not_to_now(tmp_path):
    """A record it cannot date must not silently outrank every earlier version.

    Dated now, the record sits above every version of the id written before the
    upgrade, and a legitimate older recreation is refused for good. Dated from
    what the row does carry, or from the earliest instant when it carries
    nothing readable, the record can only let something through, and what it
    would let through is caught by the grading on the pull side.
    """
    engine, tombstones = _engine_and_tombstones(tmp_path)
    created = _NOW - timedelta(days=300)
    engine.ingest(_memory("derived", updated=_NOW - timedelta(days=299)))
    with engine._conn:
        engine._conn.execute(
            "UPDATE memories SET is_closet = 1, updated_at = 'not-a-timestamp', created_at = ?",
            (created.isoformat(),),
        )
    engine._conn.close()

    engine = SeedEngine(tmp_path / "memories.db")
    stamp = engine._conn.execute("SELECT deleted_at FROM sync_local_deletions").fetchone()[0]
    assert datetime.fromisoformat(stamp) == created

    # An older, genuinely different memory at that id still lands.
    older = _memory("derived", updated=created + timedelta(seconds=1))
    older.content = "An independent note written before the upgrade"
    result = pull(
        engine=engine,
        tombstones=tombstones,
        client=_FakeClient(rows=[memory_to_wire(older)]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert result.errors == 0
    assert engine.get("derived").content == "An independent note written before the upgrade"


def test_cleanup_retires_the_upload_queue_including_orphans(tmp_path):
    """Nothing here drains that queue, but an older client sharing the store would."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("real", updated=_NOW))
    with engine._conn:
        # An id that is no longer in `memories`: left by an earlier redaction.
        engine._conn.execute(
            "INSERT INTO legacy_closet_ids (id, announce_pending, legacy_updated_at) VALUES (?, 1, ?)",
            ("gone_closet_alice", _NOW.isoformat()),
        )
    engine._conn.close()

    # No marked rows at all, so the cleanup must still reach the queue.
    SeedEngine(tmp_path / "memories.db")._conn.close()

    assert TombstoneStore(tmp_path / "memories.db").pending_legacy_announcements() == []


def test_a_reclaimed_id_drops_its_deletion_record(tmp_path):
    """The record must not outlive the thing it described."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("derived", updated=_NOW - timedelta(days=2)))
    with engine._conn:
        engine._conn.execute("UPDATE memories SET is_closet = 1")
    engine._conn.close()
    engine = SeedEngine(tmp_path / "memories.db")
    assert engine._conn.execute("SELECT 1 FROM sync_local_deletions WHERE id = 'derived'").fetchone()

    newer = _memory("derived", updated=_NOW)
    newer.content = "A real memory at that id"
    pull(
        engine=engine,
        tombstones=tombstones,
        client=_FakeClient(rows=[memory_to_wire(newer)]),
        state=SyncState(),
        poppy_dir=tmp_path,
    )

    assert engine.get("derived").content == "A real memory at that id"
    assert not engine._conn.execute("SELECT 1 FROM sync_local_deletions WHERE id = 'derived'").fetchone()


def test_cleanup_preserves_an_independent_snapshot_at_the_same_id(tmp_path):
    engine, tombstones = _engine_and_tombstones(tmp_path)
    previous = _memory("derived", updated=_NOW - timedelta(days=1))
    previous.content = "An independent note deleted before this ID was reused"
    tombstones.add(previous)
    engine.ingest(_memory("derived", updated=_NOW))
    with engine._conn:
        engine._conn.execute("UPDATE memories SET is_closet = 1")
    reopened = SeedEngine(tmp_path / "memories.db")
    assert reopened.get("derived") is None
    assert tombstones.get("derived").memory.content == previous.content


def test_cleanup_suppresses_an_old_content_carrying_cloud_deletion(tmp_path):
    from poppy.sync import sync

    engine, tombstones = _engine_and_tombstones(tmp_path)
    memory = _memory("derived", updated=_NOW)
    engine.ingest(memory)
    with engine._conn:
        engine._conn.execute("UPDATE memories SET is_closet = 1")
    reopened = SeedEngine(tmp_path / "memories.db")
    row = memory_to_wire(memory)
    row["deleted_at"] = _NOW.isoformat()
    client = _FakeClient(rows=[row])
    result = sync(engine=reopened, tombstones=tombstones, client=client, poppy_dir=tmp_path)
    assert result.pull.errors == result.push.errors == 0
    assert tombstones.get(memory.id) is None
    assert client.upserts == []


def test_redacted_deletion_records_suppression_before_it_touches_trash(tmp_path, monkeypatch):
    """The record is persisted first, so an interrupted pull retries safely."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    # A Trash entry at the id, no live row: clearing it is the step that can fail.
    tombstones.add(_memory("deleted", updated=_NOW - timedelta(days=2)), tombstoned_at=_NOW - timedelta(days=1))
    client = _FakeClient(rows=[_release_030_redacted_row("deleted")])
    original_remove = tombstones.remove

    def interrupted(*args, **kwargs):
        raise OSError("interrupted deletion")

    monkeypatch.setattr(tombstones, "remove", interrupted)
    with pytest.raises(OSError, match="interrupted deletion"):
        pull(engine=engine, tombstones=tombstones, client=client, state=SyncState(), poppy_dir=tmp_path)
    # Already durable, even though the pull did not finish.
    assert engine._conn.execute("SELECT 1 FROM sync_local_deletions WHERE id = 'deleted'").fetchone()

    monkeypatch.setattr(tombstones, "remove", original_remove)
    result = pull(engine=engine, tombstones=tombstones, client=client, state=SyncState(), poppy_dir=tmp_path)
    assert result.errors == 0
    assert engine.get("deleted") is None
    assert tombstones.get("deleted") is None


def test_a_real_memory_shaped_like_a_retired_deletion_keeps_its_trash_entry(tmp_path):
    """The retired shape never destroys a live row it cannot prove is a copy."""
    engine, tombstones = _engine_and_tombstones(tmp_path)
    engine.ingest(_memory("deleted", updated=_NOW - timedelta(days=1)))
    row = _release_030_redacted_row("deleted")
    # A real memory that happens to carry the retired body, with its own source.
    row["source_type"] = "cli"
    row["project"] = "work"

    result = pull(
        engine=engine, tombstones=tombstones, client=_FakeClient(rows=[row]), state=SyncState(), poppy_dir=tmp_path
    )

    assert result.applied_tombstones == 1
    assert engine.get("deleted") is None  # the deletion still applies
    assert tombstones.get("deleted") is not None  # but it stays restorable
    assert not engine._conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'sync_local_deletions'").fetchone()
