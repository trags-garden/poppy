"""Expired rows are invisible the same way to every reader of the store.

A memory whose TTL has run out stays on disk until ``purge_expired`` removes it.
Until then ``list_all`` hides it, so ``stats`` has to hide it too: the dashboard
prints the total from ``stats`` beside the rows from ``list_all``, and ``poppy
stats`` is the number a user checks that list against. A total that counted rows
nothing can show reads as memories going missing.

Encoders are injected fakes: no ONNX, no model downloads.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from poppy.engine.bloom import BloomEngine
from poppy.engine.interface import RetrievalEngine
from poppy.engine.seed import SeedEngine
from poppy.models import Filters, Memory, Source


class _FakeBiEncoder:
    def embed(self, texts):
        for t in texts:
            h = hash(t) & 0xFFFF
            yield np.array([h & 0xF, (h >> 4) & 0xF, (h >> 8) & 0xF, (h >> 12) & 0xF], dtype=np.float32)


class _FakeCrossEncoder:
    def rerank(self, query, docs):
        return [float(len(d)) for d in docs]


def _seed(db_path: Path) -> SeedEngine:
    return SeedEngine(db_path=db_path)


def _bloom(db_path: Path) -> BloomEngine:
    return BloomEngine(db_path=db_path, bi_encoder=_FakeBiEncoder(), cross_encoder=_FakeCrossEncoder())


def _memory(mid: str, *, expires_at: datetime | None = None) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mid,
        content=f"memory {mid}",
        memory_type="fact",
        source=Source(type="cli", session_id=None, timestamp=now),
        project="p1",
        related_to=[],
        created_at=now,
        updated_at=now,
        confidence=1.0,
        expires_at=expires_at,
    )


@pytest.fixture(params=["seed", "bloom"])
def engine(request: pytest.FixtureRequest, tmp_path: Path) -> RetrievalEngine:
    factory = {"seed": _seed, "bloom": _bloom}[request.param]
    return factory(tmp_path / "memories.db")


def test_stats_counts_only_what_list_all_shows(engine: RetrievalEngine) -> None:
    """Six rows, one of them already expired: both readers report five."""
    past = datetime.now(timezone.utc) - timedelta(days=1)
    for i in range(5):
        engine.ingest(_memory(f"mem_{i:03d}"))
    engine.ingest(_memory("mem_expired", expires_at=past))

    visible = engine.list_all(limit=100)

    assert len(visible) == 5
    assert engine.stats().memory_count == len(visible)


def test_expired_row_is_still_there_until_purged(engine: RetrievalEngine) -> None:
    """Hiding it from the total is not deleting it: the purge still finds it."""
    past = datetime.now(timezone.utc) - timedelta(days=1)
    engine.ingest(_memory("mem_live"))
    engine.ingest(_memory("mem_expired", expires_at=past))

    assert len(engine.list_all(filters=Filters(include_expired=True), limit=100)) == 2
    assert engine.purge_expired() == 1
    assert engine.stats().memory_count == 1
