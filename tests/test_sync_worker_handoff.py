"""Deterministic worker handoffs using real file locks and stubbed sync calls."""

import builtins
import fcntl
from contextlib import contextmanager
from pathlib import Path

import pytest

from poppy import writers
from poppy.sync import auto
from poppy.sync.client import TragsAuthError, TragsQuotaError


@pytest.fixture(autouse=True)
def no_delays(monkeypatch):
    monkeypatch.setattr(auto, "DEBOUNCE_S", 0)
    sleeps = []
    monkeypatch.setattr(auto.time, "sleep", sleeps.append)
    return sleeps


@pytest.fixture
def lock_handles(tmp_path, monkeypatch):
    handles = []

    def tracking_open(path, *args, **kwargs):
        handle = builtins.open(path, *args, **kwargs)
        if Path(path) == tmp_path / auto.LOCK_FILENAME:
            handles.append(handle)
        return handle

    monkeypatch.setattr(auto, "open", tracking_open, raising=False)
    return handles


@pytest.fixture
def registration_events(monkeypatch):
    events = []
    registered = writers.registered

    @contextmanager
    def tracking_registration(poppy_dir, surface):
        with registered(poppy_dir, surface):
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")

    monkeypatch.setattr(writers, "registered", tracking_registration)
    return events


@pytest.fixture
def stub_sync(monkeypatch):
    guard_errors = []

    def install(action=None):
        calls = []

        def guarded_sync(poppy_dir):
            calls.append(poppy_dir)
            try:
                # Separate opens contend even within this process, exercising
                # the same kernel flock as a competing worker.
                with open(poppy_dir / auto.LOCK_FILENAME) as probe:
                    with pytest.raises(BlockingIOError):
                        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                assert writers.live_writers(poppy_dir) == [writers.SURFACE_LABELS["sync"]]
            except BaseException as exc:
                # run_worker catches generic exceptions, so retain guard failures
                # for an assertion outside its retry loop as well.
                guard_errors.append(exc)
                raise
            return action(poppy_dir) if action else {"errors": 0}

        monkeypatch.setattr(auto, "_do_sync", guarded_sync)
        return calls

    yield install
    assert not guard_errors


@pytest.mark.parametrize("arrivals", [1, 2])
def test_pending_after_absence_is_synced(tmp_path, monkeypatch, stub_sync, lock_handles, registration_events, arrivals):
    pending = tmp_path / auto.PENDING_FILENAME
    calls = stub_sync()
    exists = Path.exists
    injected = []
    handoff_checks = []

    def inject_after_absence(path):
        present = exists(path)
        if path == pending:
            if lock_handles[-1].closed:
                assert writers.live_writers(tmp_path) == []
                handoff_checks.append(present)
            elif not present and len(injected) < arrivals:
                injected.append(True)
                pending.touch()
                # The newly spawned worker loses the lock while A is finishing.
                auto.run_worker(tmp_path)
        return present

    monkeypatch.setattr(Path, "exists", inject_after_absence)
    pending.touch()

    auto.run_worker(tmp_path)

    assert len(calls) == arrivals + 1
    assert not exists(pending)
    assert handoff_checks == [True] * arrivals + [False]
    assert registration_events == ["enter", "exit"] * (arrivals + 1)
    assert all(handle.closed for handle in lock_handles)


def test_competing_worker_owns_pending_at_handoff(tmp_path, monkeypatch, stub_sync, lock_handles):
    pending = tmp_path / auto.PENDING_FILENAME
    calls = stub_sync()
    exists = Path.exists
    competitor = None

    def take_lock_after_release(path):
        nonlocal competitor
        present = exists(path)
        if path == pending:
            if lock_handles[-1].closed and competitor is None:
                assert writers.live_writers(tmp_path) == []
                competitor = open(tmp_path / auto.LOCK_FILENAME)
                fcntl.flock(competitor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif not present:
                pending.touch()
        # The first absence is observed under the lock, even though a write
        # immediately sets pending. The post-release check sees that write.
        return present

    monkeypatch.setattr(Path, "exists", take_lock_after_release)
    pending.touch()
    try:
        auto.run_worker(tmp_path)
        assert len(calls) == 1
        assert exists(pending)
        assert competitor is not None
        assert len(lock_handles) == 2
        assert all(handle.closed for handle in lock_handles)
    finally:
        if competitor is not None:
            competitor.close()


def test_auth_stop_does_not_handoff_pending(tmp_path, stub_sync, lock_handles):
    pending = tmp_path / auto.PENDING_FILENAME

    def fail_auth(poppy_dir):
        pending.touch()
        raise TragsAuthError("Unauthorized")

    calls = stub_sync(fail_auth)
    pending.touch()

    auto.run_worker(tmp_path)

    assert len(calls) == 1
    assert pending.exists()
    assert len(lock_handles) == 1


@pytest.mark.parametrize("max_rounds", [0, 1, 4, 7])
def test_generic_failures_share_budget(tmp_path, stub_sync, lock_handles, no_delays, max_rounds):
    def fail(poppy_dir):
        raise RuntimeError("Server unavailable")

    calls = stub_sync(fail)
    auto._touch_pending(tmp_path)

    auto.run_worker(tmp_path, max_rounds=max_rounds)

    assert len(calls) == max_rounds
    assert (tmp_path / auto.PENDING_FILENAME).exists()
    assert len(lock_handles) == 1
    assert no_delays == [min(2**round_number, 30) for round_number in range(max_rounds)]


def test_handoffs_share_budget(tmp_path, monkeypatch, stub_sync, lock_handles, registration_events):
    pending = tmp_path / auto.PENDING_FILENAME
    calls = stub_sync()
    exists = Path.exists
    absence_checks = []

    def always_queue_after_absence(path):
        present = exists(path)
        if path == pending and not present:
            absence_checks.append(lock_handles[-1].closed)
            pending.touch()
        return present

    monkeypatch.setattr(Path, "exists", always_queue_after_absence)
    pending.touch()

    auto.run_worker(tmp_path, max_rounds=3)

    assert len(calls) == 3
    assert absence_checks == [False, False]
    assert registration_events == ["enter", "exit"] * 3
    assert len(lock_handles) == 3


def test_budget_exhaustion_does_not_check_pending_after_release(tmp_path, monkeypatch, stub_sync, lock_handles):
    pending = tmp_path / auto.PENDING_FILENAME
    exists = Path.exists

    def queue_more(poppy_dir):
        pending.touch()
        return {"errors": 0}

    def forbid_handoff_check(path):
        if path == pending:
            assert not lock_handles[-1].closed
        return exists(path)

    calls = stub_sync(queue_more)
    monkeypatch.setattr(Path, "exists", forbid_handoff_check)
    pending.touch()

    auto.run_worker(tmp_path, max_rounds=1)

    assert len(calls) == 1
    assert exists(pending)
    assert len(lock_handles) == 1


def test_quota_stop_does_not_handoff(tmp_path, monkeypatch, stub_sync, lock_handles):
    pending = tmp_path / auto.PENDING_FILENAME
    exists = Path.exists

    def fail_quota(poppy_dir):
        raise TragsQuotaError("Memory limit reached")

    def queue_after_quota_absence(path):
        present = exists(path)
        if path == pending:
            assert not lock_handles[-1].closed
            if not present:
                pending.touch()
        return present

    calls = stub_sync(fail_quota)
    monkeypatch.setattr(Path, "exists", queue_after_quota_absence)
    pending.touch()

    auto.run_worker(tmp_path)

    assert len(calls) == 1
    assert exists(pending)
    assert len(lock_handles) == 1


def test_rearmed_failure_does_not_handoff(tmp_path, monkeypatch, stub_sync, lock_handles):
    pending = tmp_path / auto.PENDING_FILENAME
    exists = Path.exists

    def fail(poppy_dir):
        raise RuntimeError("Server unavailable")

    def consume_retry_during_backoff(delay):
        pending.unlink()

    def queue_after_failure_absence(path):
        present = exists(path)
        if path == pending:
            assert not lock_handles[-1].closed
            if not present:
                pending.touch()
        return present

    calls = stub_sync(fail)
    monkeypatch.setattr(auto.time, "sleep", consume_retry_during_backoff)
    monkeypatch.setattr(Path, "exists", queue_after_failure_absence)
    pending.touch()

    auto.run_worker(tmp_path)

    assert len(calls) == 1
    assert exists(pending)
    assert len(lock_handles) == 1
