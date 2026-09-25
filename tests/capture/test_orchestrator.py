"""CaptureOrchestrator — the shared capture tail, tested through the injected
interface.

The whole point of the reshape: exercise extract → build → ingest → advance
watermark → journal end-to-end with a fake LLM + fake reconcile and a tmp
poppy_dir — no module-global monkeypatching — and pin the
watermark-after-success invariant in one place.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from poppy.capture import journal
from poppy.capture.cadence import capture_count
from poppy.capture.orchestrator import (
    CaptureOrchestrator,
    CaptureOutcome,
    CapturePlan,
    build_capture_memories,
    run_capture_worker,
)
from poppy.capture.reconciler import ReconcileSummary
from poppy.capture.watermark import get_watermark


class _RecordingReconcile:
    """Fake reconcile step: stores nothing, just records that it ran and how many."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, candidates, *, engine, cfg, poppy_dir) -> ReconcileSummary:
        self.calls.append(len(candidates))
        return ReconcileSummary(added=len(candidates))


def _llm(items):
    def _call(prompt, *, transcript_path, cfg):
        return list(items)

    return _call


def _orch(tmp_path: Path, *, llm, reconcile) -> CaptureOrchestrator:
    return CaptureOrchestrator(engine=object(), cfg=object(), poppy_dir=tmp_path, llm=llm, reconcile=reconcile)


def _plan(**overrides) -> CapturePlan:
    base = dict(session_id="sess", prompt="p", source_type="claude-code", project="proj")
    base.update(overrides)
    return CapturePlan(**base)


# ---------- build_capture_memories ----------


def test_build_capture_memories_stamps_provenance():
    mems = build_capture_memories(
        [{"type": "decision", "content": "use rebase merges"}],
        source_type="cursor",
        session_id="s1",
        project="poppy",
    )
    assert len(mems) == 1
    m = mems[0]
    assert m.memory_type == "decision"
    assert m.source.type == "cursor"
    assert m.source.session_id == "s1"
    assert m.project == "poppy"
    assert m.confidence == 0.7  # captured candidates ingest below manual (1.0)
    assert m.id.startswith("mem_")


# ---------- the happy path ----------


def test_run_extracts_builds_ingests_and_records(tmp_path: Path):
    reconcile = _RecordingReconcile()
    orch = _orch(tmp_path, llm=_llm([{"type": "fact", "content": "blue-green deploys"}]), reconcile=reconcile)

    outcome = orch.run(_plan(advance_watermark_to=6, record_soft_cap=True, journal=True))

    assert isinstance(outcome, CaptureOutcome)
    assert outcome.stored == 1
    assert [c.content for c in outcome.candidates] == ["blue-green deploys"]
    assert reconcile.calls == [1]
    assert get_watermark(tmp_path, "sess") == 6
    assert capture_count(tmp_path, "sess") == 1
    rec = journal.read_last(tmp_path)
    assert rec is not None and rec.session_id == "sess" and rec.count == 1


def test_max_items_caps_candidates(tmp_path: Path):
    reconcile = _RecordingReconcile()
    items = [{"type": "fact", "content": f"f{i}"} for i in range(10)]
    orch = _orch(tmp_path, llm=_llm(items), reconcile=reconcile)

    outcome = orch.run(_plan(max_items=3))

    assert len(outcome.candidates) == 3
    assert reconcile.calls == [3]


# ---------- watermark-after-success: watermark advances only after a successful ingest ----------


def test_empty_extraction_writes_nothing_and_leaves_watermark(tmp_path: Path):
    reconcile = _RecordingReconcile()
    orch = _orch(tmp_path, llm=_llm([]), reconcile=reconcile)

    outcome = orch.run(_plan(advance_watermark_to=9, record_soft_cap=True, journal=True))

    assert outcome.stored == 0
    assert reconcile.calls == [], "no extraction → no ingest"
    assert get_watermark(tmp_path, "sess") == 0, "watermark must not advance without an ingest"
    assert capture_count(tmp_path, "sess") == 0
    assert journal.read_last(tmp_path) is None


def test_watermark_not_advanced_when_reconcile_raises(tmp_path: Path):
    def boom(candidates, *, engine, cfg, poppy_dir):
        raise RuntimeError("ingest failed")

    orch = _orch(tmp_path, llm=_llm([{"type": "fact", "content": "x"}]), reconcile=boom)

    with pytest.raises(RuntimeError):
        orch.run(_plan(advance_watermark_to=5, journal=True))

    # The watermark step sits after the ingest, so a failed ingest leaves the
    # turns for the next fire / backstop to re-cover.
    assert get_watermark(tmp_path, "sess") == 0
    assert journal.read_last(tmp_path) is None


# ---------- the per-entry post-ingest flags ----------


def test_none_watermark_skips_watermark(tmp_path: Path):
    """PostCompact keys idempotency off the composite event id, not a watermark —
    advance_watermark_to=None must not touch the state file."""
    orch = _orch(tmp_path, llm=_llm([{"type": "fact", "content": "x"}]), reconcile=_RecordingReconcile())

    orch.run(_plan(advance_watermark_to=None))

    assert get_watermark(tmp_path, "sess") == 0


def test_journal_false_skips_journal(tmp_path: Path):
    orch = _orch(tmp_path, llm=_llm([{"type": "fact", "content": "x"}]), reconcile=_RecordingReconcile())

    orch.run(_plan(journal=False))

    assert journal.read_last(tmp_path) is None


def test_soft_cap_only_counted_when_requested(tmp_path: Path):
    orch = _orch(tmp_path, llm=_llm([{"type": "fact", "content": "x"}]), reconcile=_RecordingReconcile())

    orch.run(_plan(record_soft_cap=False))

    assert capture_count(tmp_path, "sess") == 0


# ---------- the default LLM resolves lazily (no import cycle, honors patch) ----------


def test_default_llm_resolves_consolidation_call_llm(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "poppy.consolidation.call_llm",
        lambda prompt, *, transcript_path, cfg: [{"type": "fact", "content": "via default llm"}],
    )
    orch = CaptureOrchestrator(engine=object(), cfg=object(), poppy_dir=tmp_path, reconcile=_RecordingReconcile())

    outcome = orch.run(_plan())

    assert [c.content for c in outcome.candidates] == ["via default llm"]


# ---------- run_capture_worker: "stored > 0 → autosync + funnel emit" ----------


def test_run_capture_worker_triggers_sync_and_milestone_when_stored(tmp_path: Path):
    triggered: list = []
    milestones: list = []
    n = run_capture_worker(
        lambda payload: 3,
        {"session_id": "s"},
        milestone="mid_session",
        poppy_dir=tmp_path,
        trigger=triggered.append,
        emit_milestone=lambda d, m: milestones.append((d, m)),
    )
    assert n == 3
    assert triggered == [tmp_path]
    assert milestones == [(tmp_path, "mid_session")]


def test_run_capture_worker_no_sync_no_milestone_when_nothing_stored(tmp_path: Path):
    triggered: list = []
    milestones: list = []
    n = run_capture_worker(
        lambda payload: 0,
        {"session_id": "s"},
        milestone="session_end",
        poppy_dir=tmp_path,
        trigger=triggered.append,
        emit_milestone=lambda d, m: milestones.append((d, m)),
    )
    assert n == 0
    assert triggered == []
    assert milestones == []


def test_run_capture_worker_default_milestone_emit_latches_funnel_event(tmp_path: Path, monkeypatch):
    """The default emit path records first_autocapture_stored via telemetry.capture_once."""
    events: list = []
    monkeypatch.setattr(
        "poppy.telemetry.capture_once",
        lambda poppy_dir, event, properties=None: events.append((event, properties)) or True,
    )

    run_capture_worker(lambda payload: 1, {}, milestone="post_compact", poppy_dir=tmp_path, trigger=lambda d: None)

    assert events == [("first_autocapture_stored", {"trigger": "post_compact"})]
