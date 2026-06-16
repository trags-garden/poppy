"""Tests for CaptureReconciler (ADR-0003: autonomous reconciliation).

The reconciler decides ADD / SUPERSEDE / SKIP for each captured candidate before
ingest. The cheap lexical prefilter resolves clear duplicates (SKIP) and clearly
novel candidates (ADD) without an LLM call; only the ambiguous band consults the
LLM verdict, and an uncertain verdict biases to ADD. A SUPERSEDE is reversible.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from poppy.capture.reconciler import (
    Action,
    decide,
    reconcile_and_ingest,
)
from poppy.config import PoppyConfig
from poppy.engine.seed import SeedEngine
from poppy.models import Filters, Memory, Source


def _mk(mid: str, content: str, *, project: str | None = "poppy", memory_type: str = "decision") -> Memory:
    n = datetime.now(timezone.utc)
    return Memory(
        id=mid,
        content=content,
        memory_type=memory_type,
        source=Source(type="claude-code", session_id="sess", timestamp=n),
        project=project,
        related_to=[],
        created_at=n,
        updated_at=n,
        confidence=0.7,
    )


@pytest.fixture
def engine(tmp_path: Path) -> SeedEngine:
    return SeedEngine(db_path=tmp_path / "memories.db")


def _ban_llm(monkeypatch: pytest.MonkeyPatch) -> list:
    """Make any LLM call a test failure; returns the (empty) call log."""
    called: list = []

    def boom(prompt, *, transcript_path, cfg):
        called.append(prompt)
        return []

    monkeypatch.setattr("poppy.consolidation.call_llm", boom)
    return called


def test_clear_duplicate_skips_without_llm(engine: SeedEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    engine.ingest(_mk("a", "The team uses ruff for linting and formatting."))
    called = _ban_llm(monkeypatch)

    dup = _mk("dup", "The team uses ruff for linting and formatting.")
    decision = decide(engine, dup, cfg=PoppyConfig())

    assert decision.action is Action.SKIP
    assert decision.target_id == "a"
    assert called == [], "a clear duplicate must not spend an LLM call"


def test_clearly_new_adds_without_llm(engine: SeedEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    engine.ingest(_mk("a", "The team uses ruff for linting and formatting."))
    called = _ban_llm(monkeypatch)

    novel = _mk("new", "Deployments run on Kubernetes via Helm charts in us-east-1.")
    decision = decide(engine, novel, cfg=PoppyConfig())

    assert decision.action is Action.ADD
    assert called == [], "a clearly novel candidate must not spend an LLM call"


def test_contradiction_supersedes_above_threshold(
    engine: SeedEngine, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine.ingest(_mk("existing", "Use all-MiniLM for embeddings."))

    monkeypatch.setattr(
        "poppy.consolidation.call_llm",
        lambda prompt, *, transcript_path, cfg: [{"id": "existing", "confidence": 0.95, "reason": "replaces"}],
    )

    replacement = _mk("new", "Use bge-large for embeddings instead.")
    summary = reconcile_and_ingest([replacement], engine=engine, cfg=PoppyConfig(), poppy_dir=tmp_path)

    assert summary.superseded == 1
    assert summary.added == 0
    # Old is tombstoned (gone from the engine), new points back at it.
    assert engine.get("existing") is None
    survivors = engine.list_all(filters=Filters(), limit=50)
    assert len(survivors) == 1
    assert "existing" in survivors[0].related_to


def test_uncertain_verdict_biases_to_add(engine: SeedEngine, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    engine.ingest(_mk("existing", "Use all-MiniLM for embeddings."))

    called: list = []

    def low_confidence(prompt, *, transcript_path, cfg):
        called.append(prompt)
        return [{"id": "existing", "confidence": 0.5, "reason": "maybe related"}]

    monkeypatch.setattr("poppy.consolidation.call_llm", low_confidence)

    candidate = _mk("new", "Use bge-large for embeddings instead.")
    summary = reconcile_and_ingest([candidate], engine=engine, cfg=PoppyConfig(), poppy_dir=tmp_path)

    assert called, "an ambiguous candidate must consult the LLM"
    assert summary.added == 1
    assert summary.superseded == 0
    # Both kept — adding never loses information.
    assert engine.get("existing") is not None
    assert {m.id for m in engine.list_all(filters=Filters(), limit=50)} == {"existing", "new"}


def test_same_content_twice_does_not_accumulate(
    engine: SeedEngine, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: re-consolidating the same content must not pile up near-dups."""
    _ban_llm(monkeypatch)
    cfg = PoppyConfig()

    first = reconcile_and_ingest(
        [_mk("c1", "Prefer asyncpg over psycopg for Postgres access.")],
        engine=engine,
        cfg=cfg,
        poppy_dir=tmp_path,
    )
    second = reconcile_and_ingest(
        [_mk("c2", "Prefer asyncpg over psycopg for Postgres access.")],
        engine=engine,
        cfg=cfg,
        poppy_dir=tmp_path,
    )

    assert first.added == 1
    assert second.skipped == 1
    assert second.added == 0
    assert len(engine.list_all(filters=Filters(), limit=50)) == 1


def test_intra_batch_dedup(engine: SeedEngine, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Two near-identical candidates in one batch store only once."""
    _ban_llm(monkeypatch)
    batch = [
        _mk("c1", "CI runs on GitHub Actions with Node 24."),
        _mk("c2", "CI runs on GitHub Actions with Node 24."),
    ]
    summary = reconcile_and_ingest(batch, engine=engine, cfg=PoppyConfig(), poppy_dir=tmp_path)
    assert summary.added == 1
    assert summary.skipped == 1


def test_supersede_is_reversible_tombstone_written(
    engine: SeedEngine, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from poppy.ui.tombstones import TombstoneStore

    engine.ingest(_mk("existing", "Use all-MiniLM for embeddings."))
    monkeypatch.setattr(
        "poppy.consolidation.call_llm",
        lambda prompt, *, transcript_path, cfg: [{"id": "existing", "confidence": 0.95, "reason": "replaces"}],
    )

    new = _mk("new", "Use bge-large for embeddings instead.")
    reconcile_and_ingest([new], engine=engine, cfg=PoppyConfig(), poppy_dir=tmp_path)

    tomb = TombstoneStore(tmp_path / "memories.db").get("existing")
    assert tomb is not None, "superseded memory must be restorable from a tombstone"
    assert tomb.superseded_by == "new"
