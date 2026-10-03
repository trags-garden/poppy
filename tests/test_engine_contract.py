from datetime import datetime, timezone

import pytest
from test_engine_bloom import _FakeBiEncoder, _FakeCrossEncoder

from poppy.engine.bloom import BloomEngine
from poppy.engine.seed import SeedEngine
from poppy.models import Memory, Source


def _memory(memory_id: str, content: str) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=memory_id,
        content=content,
        memory_type="fact",
        source=Source(type="cli", session_id=None, timestamp=now),
        project=None,
        related_to=[],
        created_at=now,
        updated_at=now,
        confidence=1.0,
    )


@pytest.mark.parametrize("engine_kind", ["seed", "bloom"])
def test_retrieve_scores_rise_with_relevance(tmp_path, engine_kind):
    db_path = tmp_path / "memories.db"
    engine = (
        SeedEngine(db_path=db_path)
        if engine_kind == "seed"
        else BloomEngine(db_path=db_path, bi_encoder=_FakeBiEncoder(), cross_encoder=_FakeCrossEncoder())
    )
    for i, repetitions in enumerate((8, 4, 1)):
        engine.ingest(_memory(f"m{i}", " ".join(["python"] * repetitions)))
    # A larger corpus keeps full-text ranks far enough from zero to expose direction.
    for i in range(30):
        engine.ingest(_memory(f"filler{i}", "other"))

    results = engine.retrieve("python", limit=3)

    assert len(results) == 3
    assert results[0].memory.id == "m0"
    scores = [result.score for result in results]
    assert scores[0] > scores[1] > scores[2], scores
    if engine_kind == "seed":
        assert all(0.0 <= score <= 1.0 for score in scores)
