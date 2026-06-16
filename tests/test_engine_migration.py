"""Tests for the engine migration helpers.

Covers the three signals the rest of Poppy reads: ``stale_stats``,
``count_targets``, and ``migrate``. The fixtures avoid loading sentence-
transformers by stubbing the engine's ``_embed`` method with a deterministic
fake — these tests only verify migration plumbing, not embedding quality.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from poppy.engine.migration import (
    MigrateFilters,
    count_targets,
    list_targets,
    migrate,
    stale_stats,
    sweep_orphans,
)
from poppy.engine.seed import SeedEngine, _migrate_embedding_model_id
from poppy.models import Memory, Source


@dataclass
class _FakeEngine:
    """Minimal stand-in for a RetrievalEngine that tags embeddings."""

    model_id: str = "fake-bi-encoder-v1"

    def _embed(self, text: str) -> np.ndarray:
        # Deterministic 4-dim vector keyed off the content's hash so swaps are
        # detectable without loading real models.
        h = hash(text) & 0xFFFF
        return np.array([h & 0xF, (h >> 4) & 0xF, (h >> 8) & 0xF, (h >> 12) & 0xF], dtype=np.float32)


def _make_memory(
    mid: str, content: str, *, project: str = "p1", memory_type: str = "fact", created_days_ago: int = 0
) -> Memory:
    now = datetime.now(timezone.utc) - timedelta(days=created_days_ago)
    return Memory(
        id=mid,
        content=content,
        memory_type=memory_type,
        source=Source(type="cli", session_id=None, timestamp=now),
        project=project,
        related_to=[],
        created_at=now,
        updated_at=now,
        confidence=1.0,
    )


def _seed_with_baseline_and_attach_embeddings(
    db_path: Path,
    rows: list[tuple[Memory, str | None, np.ndarray]],
) -> None:
    """Use SeedEngine for the memories row, then manually attach embeddings.

    SeedEngine creates memories + memory_fts but never touches
    memory_embeddings, so we attach embeddings ourselves with arbitrary
    model_id values (None included) to simulate every migration state.
    """
    engine = SeedEngine(db_path=db_path)
    for memory, _model_id, _vec in rows:
        engine.ingest(memory)
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE IF NOT EXISTS memory_embeddings (id TEXT PRIMARY KEY, embedding BLOB NOT NULL)")
    _migrate_embedding_model_id(conn)
    for memory, model_id, vec in rows:
        conn.execute(
            "INSERT OR REPLACE INTO memory_embeddings (id, embedding, model_id) VALUES (?, ?, ?)",
            (memory.id, vec.tobytes(), model_id),
        )
    conn.commit()
    conn.close()


def test_stale_stats_segregates_by_model_id(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    rows = [
        (_make_memory("m1", "alpha"), "fake-bi-encoder-v1", np.array([1.0], dtype=np.float32)),
        (_make_memory("m2", "beta"), "other-model", np.array([2.0], dtype=np.float32)),
        (_make_memory("m3", "gamma"), None, np.array([3.0], dtype=np.float32)),
    ]
    _seed_with_baseline_and_attach_embeddings(db, rows)

    stats = stale_stats(db, "fake-bi-encoder-v1")
    assert stats.compatible == 1
    assert stats.stale == 1
    assert stats.unknown == 1
    assert stats.needs_migration == 2


def test_stale_stats_returns_zero_for_no_embedding_engine(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    SeedEngine(db_path=db).ingest(_make_memory("m1", "alpha"))
    # SeedEngine.model_id is None — caller signals "no bi-encoder", we
    # short-circuit to zeros rather than reading a (possibly nonexistent)
    # memory_embeddings table.
    assert stale_stats(db, None).needs_migration == 0


def test_stale_stats_handles_missing_db(tmp_path: Path) -> None:
    assert stale_stats(tmp_path / "nope.db", "any").needs_migration == 0


def test_count_targets_respects_filters(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    rows = [
        # mem-id, content, project, type, created_days_ago, embedding model_id
        ("m1", "old fact in p1", "p1", "fact", 30, "other"),
        ("m2", "recent fact in p1", "p1", "fact", 1, None),
        ("m3", "recent decision in p2", "p2", "decision", 1, "other"),
        ("m4", "recent fact in p2", "p2", "fact", 1, "fake-bi-encoder-v1"),  # already current
    ]
    seed = [
        (
            _make_memory(mid, content, project=proj, memory_type=mt, created_days_ago=days),
            model,
            np.array([float(i)], dtype=np.float32),
        )
        for i, (mid, content, proj, mt, days, model) in enumerate(rows)
    ]
    _seed_with_baseline_and_attach_embeddings(db, seed)

    # Default: every row that isn't already on the active model_id.
    assert count_targets(db, "fake-bi-encoder-v1", MigrateFilters()) == 3

    # Scoped by project: only p1 ⇒ m1 + m2.
    assert count_targets(db, "fake-bi-encoder-v1", MigrateFilters(project="p1")) == 2

    # Scoped by memory_type=decision: only m3.
    assert count_targets(db, "fake-bi-encoder-v1", MigrateFilters(memory_type="decision")) == 1

    # Scoped by --since 7d: m1 is 30 days old, excluded ⇒ m2 + m3.
    assert count_targets(db, "fake-bi-encoder-v1", MigrateFilters(since_days=7)) == 2

    # --all: every row even the already-compatible one.
    assert count_targets(db, "fake-bi-encoder-v1", MigrateFilters(include_compatible=True)) == 4


def test_migrate_re_embeds_stale_rows_only(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    seed = [
        (_make_memory("m1", "alpha"), "other", np.array([9.0], dtype=np.float32)),
        (_make_memory("m2", "beta"), None, np.array([9.0], dtype=np.float32)),
        (
            _make_memory("m3", "gamma"),
            "fake-bi-encoder-v1",
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        ),  # already current
    ]
    _seed_with_baseline_and_attach_embeddings(db, seed)

    progress_calls: list[tuple[int, int]] = []
    done = migrate(_FakeEngine(), db, MigrateFilters(), on_progress=lambda d, t: progress_calls.append((d, t)))
    assert done == 2
    # Progress reports once per row, both with total=2.
    assert progress_calls == [(1, 2), (2, 2)]

    final = stale_stats(db, "fake-bi-encoder-v1")
    assert final.compatible == 3
    assert final.stale == 0
    assert final.unknown == 0


def test_migrate_is_idempotent_on_clean_db(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    seed = [
        (_make_memory("m1", "alpha"), "fake-bi-encoder-v1", np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)),
    ]
    _seed_with_baseline_and_attach_embeddings(db, seed)

    assert migrate(_FakeEngine(), db, MigrateFilters()) == 0


def test_migrate_rejects_engine_without_model_id(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    SeedEngine(db_path=db).ingest(_make_memory("m1", "alpha"))

    # FTS-only engine: dataclass field overrides via constructor, not class-attr
    # shadowing (which dataclass inheritance ignores).
    fts_only = _FakeEngine(model_id=None)
    with pytest.raises(ValueError, match="does not use embeddings"):
        migrate(fts_only, db, MigrateFilters())


def test_stale_stats_separates_orphans_from_live_drift(tmp_path: Path) -> None:
    """An embedding whose ``memories`` row was deleted is an orphan, not stale.

    Pre-fix, ``stale_stats`` counted the embeddings table directly so an
    orphan inflated ``needs_migration`` forever — ``migrate-engine`` JOINs to
    ``memories`` and can't act on it. The user saw a WARN they could never
    clear.
    """
    db = tmp_path / "memories.db"
    rows = [
        (_make_memory("live1", "alpha"), "other", np.array([1.0], dtype=np.float32)),
        (_make_memory("live2", "beta"), "fake-bi-encoder-v1", np.array([2.0], dtype=np.float32)),
        (_make_memory("ghost", "will be deleted"), "other", np.array([3.0], dtype=np.float32)),
    ]
    _seed_with_baseline_and_attach_embeddings(db, rows)
    # Simulate the deleted-memory-without-cascade case.
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM memories WHERE id = 'ghost'")
    conn.commit()
    conn.close()

    stats = stale_stats(db, "fake-bi-encoder-v1")
    assert stats.compatible == 1  # live2
    assert stats.stale == 1  # live1 (other model_id)
    assert stats.unknown == 0
    assert stats.orphans == 1  # ghost
    # The user-facing migration count excludes the orphan since it can't be acted on.
    assert stats.needs_migration == 1


def test_sweep_orphans_removes_dangling_embeddings(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    rows = [
        (_make_memory("live", "alpha"), "fake-bi-encoder-v1", np.array([1.0], dtype=np.float32)),
        (_make_memory("ghost", "deleted"), "other", np.array([2.0], dtype=np.float32)),
    ]
    _seed_with_baseline_and_attach_embeddings(db, rows)
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM memories WHERE id = 'ghost'")
    conn.commit()
    conn.close()

    assert sweep_orphans(db) == 1
    # Second call is a no-op — sweep is idempotent.
    assert sweep_orphans(db) == 0
    # And the doctor count goes clean.
    final = stale_stats(db, "fake-bi-encoder-v1")
    assert final.orphans == 0
    assert final.needs_migration == 0


def test_sweep_orphans_handles_missing_table(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    SeedEngine(db_path=db).ingest(_make_memory("m1", "alpha"))
    # No memory_embeddings table on a baseline-only DB — sweep should no-op.
    assert sweep_orphans(db) == 0


def test_list_targets_orders_by_created_at(tmp_path: Path) -> None:
    """Resumable migration depends on a stable order so partial runs make linear progress."""
    db = tmp_path / "memories.db"
    rows = [
        (_make_memory("old", "x", created_days_ago=10), "other", np.array([0.0], dtype=np.float32)),
        (_make_memory("mid", "y", created_days_ago=5), "other", np.array([0.0], dtype=np.float32)),
        (_make_memory("new", "z", created_days_ago=1), "other", np.array([0.0], dtype=np.float32)),
    ]
    _seed_with_baseline_and_attach_embeddings(db, rows)

    ids = [mid for mid, _ in list_targets(db, "fake-bi-encoder-v1", MigrateFilters())]
    assert ids == ["old", "mid", "new"]
