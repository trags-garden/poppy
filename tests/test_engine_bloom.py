"""Tests for the BloomEngine — the default ONNX/fastembed engine.

Covers two things:

* the schema-upgrade contract: a DB created by an older engine (``baseline``
  schema, no ``enriched_content`` column) must open transparently under
  BloomEngine and remain queryable. No silent fallback to baseline if the
  upgrade fails — surface the error.
* the model_id contract: rows tagged by a different bi-encoder go FTS-only
  until ``poppy migrate-engine`` re-embeds them, so two vector spaces never
  mix inside RRF.

fastembed / ONNX is stubbed (injected encoders) so these stay fast and offline;
we exercise the storage + schema plumbing, not retrieval quality. The real ONNX
path is exercised by the loader tests.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from poppy.engine.bloom import BloomEngine
from poppy.engine.migration import MigrateFilters, stale_stats
from poppy.engine.migration import migrate as run_migrate
from poppy.engine.seed import SeedEngine
from poppy.models import Memory, Source

# The bi-encoder the torch engines used before 0.3.0. Stores written by those
# versions still carry it, so it stands in for "another engine's vectors".
LEGACY_TORCH_MODEL_ID = "BAAI/bge-small-en-v1.5"


def _det_vec(text: str) -> np.ndarray:
    """Deterministic 4-dim vector keyed off content hash."""
    h = hash(text) & 0xFFFF
    return np.array([h & 0xF, (h >> 4) & 0xF, (h >> 8) & 0xF, (h >> 12) & 0xF], dtype=np.float32)


class _FakeBiEncoder:
    """fastembed TextEmbedding stand-in: .embed(iterable) -> generator of vectors."""

    def embed(self, texts):
        for t in texts:
            yield _det_vec(t)


class _FakeCrossEncoder:
    """fastembed TextCrossEncoder stand-in: .rerank(query, docs) -> scores."""

    def rerank(self, query, docs):
        return [float(len(d)) for d in docs]


def _make_engine(db_path: Path) -> BloomEngine:
    return BloomEngine(
        db_path=db_path,
        bi_encoder=_FakeBiEncoder(),
        cross_encoder=_FakeCrossEncoder(),
    )


def _memory(
    mid: str,
    content: str,
    *,
    project: str = "p1",
    memory_type: str = "fact",
    expires_at: datetime | None = None,
) -> Memory:
    now = datetime.now(timezone.utc)
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
        expires_at=expires_at,
    )


def _embedding_model_ids(db_path: Path) -> list[str | None]:
    conn = sqlite3.connect(str(db_path))
    try:
        return [r[0] for r in conn.execute("SELECT model_id FROM memory_embeddings ORDER BY id")]
    finally:
        conn.close()


def test_fresh_db_round_trip(tmp_path: Path) -> None:
    engine = _make_engine(tmp_path / "t.db")
    engine.ingest(_memory("m1", "user prefers vim over emacs"))
    got = engine.get("m1")
    assert got is not None
    assert got.content == "user prefers vim over emacs"


@pytest.mark.parametrize("engine_kind", ["bloom", "seed"])
def test_multi_speaker_ingest_stores_one_unmarked_row(tmp_path: Path, engine_kind: str) -> None:
    db = tmp_path / "memories.db"
    engine = _make_engine(db) if engine_kind == "bloom" else SeedEngine(db)
    content = '[{"speaker":"Alice","text":"hello"},{"speaker":"Bob","text":"world"}]'
    memory = _memory("conversation", content)
    for _ in range(2):
        engine.ingest(memory)
        with sqlite3.connect(db) as conn:
            assert "is_closet" in {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
            assert conn.execute("SELECT id, content, is_closet FROM memories").fetchall() == [(memory.id, content, 0)]
            assert conn.execute("SELECT COUNT(*) FROM memory_fts").fetchone()[0] == 1
            if engine_kind == "bloom":
                assert conn.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0] == 1


def test_model_id_is_the_onnx_fingerprint(tmp_path: Path) -> None:
    """The ONNX model_id must not drift: stores indexed by earlier releases
    depend on it to keep their vectors without re-embedding.
    """
    assert BloomEngine.model_id == "BAAI/bge-small-en-v1.5-onnx"
    assert BloomEngine.model_id != LEGACY_TORCH_MODEL_ID

    engine = _make_engine(tmp_path / "t.db")
    engine.ingest(_memory("m1", "alpha"))
    assert _embedding_model_ids(tmp_path / "t.db") == [BloomEngine.model_id]


def test_stats_reports_bloom(tmp_path: Path) -> None:
    engine = _make_engine(tmp_path / "t.db")
    engine.ingest(_memory("m1", "alpha"))
    s = engine.stats()
    assert s.engine_name == "bloom"
    assert s.memory_count == 1


def test_opens_legacy_baseline_db_without_error(tmp_path: Path) -> None:
    """Existing baseline DB must upgrade to enriched_content schema on open.

    Pre-fix, INSERT INTO memories (..., enriched_content, ...) errored with
    'no such column: enriched_content' the first time bloom wrote.
    The migration helper ALTERs the column in and rewires the FTS triggers.
    """
    db = tmp_path / "memories.db"
    base = SeedEngine(db_path=db)
    base.ingest(_memory("legacy1", "legacy memory body"))

    engine = _make_engine(db)
    # Legacy row still readable.
    legacy = engine.get("legacy1")
    assert legacy is not None
    assert legacy.content == "legacy memory body"

    # New writes succeed — this is the regression case.
    engine.ingest(_memory("new1", "new memory body"))
    assert engine.get("new1").content == "new memory body"


def test_legacy_rows_backfilled_into_enriched_content(tmp_path: Path) -> None:
    db = tmp_path / "memories.db"
    SeedEngine(db_path=db).ingest(_memory("legacy", "body text"))
    _make_engine(db)  # triggers migration

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT enriched_content FROM memories WHERE id='legacy'").fetchone()
    conn.close()
    # Backfill: enriched_content := content for pre-migration rows.
    assert row["enriched_content"] == "body text"


def test_fts_triggers_rewired_to_enriched_content_after_migration(tmp_path: Path) -> None:
    """After bloom opens a legacy DB, new ingests must FTS-index the
    enriched column, not the raw content. Otherwise the bench-grade enrichment
    never reaches the index for newly-written rows.
    """
    db = tmp_path / "memories.db"
    SeedEngine(db_path=db).ingest(_memory("legacy", "alpha"))

    engine = _make_engine(db)
    # JSON-shape content triggers the speaker enrichment path.
    turns_json = '[{"speaker":"Alice","dia_id":"D1","text":"hello"},{"speaker":"Bob","dia_id":"D2","text":"world"}]'
    engine.ingest(_memory("new", turns_json))

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    # The 'content' column inside memory_fts holds the indexed text. After
    # rewire it should be the enriched preamble + per-turn lines, not the raw
    # JSON string.
    indexed = conn.execute("SELECT content FROM memory_fts WHERE id='new'").fetchone()
    conn.close()
    assert indexed is not None
    assert "Conversation" in indexed["content"]
    assert "Alice" in indexed["content"]
    assert '"speaker"' not in indexed["content"]  # raw JSON should NOT be indexed


def test_expires_at_filtered_on_retrieve_and_list(tmp_path: Path) -> None:
    engine = _make_engine(tmp_path / "t.db")
    past = datetime.now(timezone.utc) - timedelta(days=1)
    future = datetime.now(timezone.utc) + timedelta(days=1)
    engine.ingest(_memory("expired", "expired body", expires_at=past))
    engine.ingest(_memory("live", "live body", expires_at=future))

    listed_ids = {m.id for m in engine.list_all()}
    assert listed_ids == {"live"}

    # Expired body is also excluded from retrieve.
    results = engine.retrieve("body", limit=10)
    assert {r.memory.id for r in results}.isdisjoint({"expired"})


def test_purge_expired_drops_memory_and_embedding(tmp_path: Path) -> None:
    """purge_expired must also clean up synthetic closet rows."""
    engine = _make_engine(tmp_path / "t.db")
    past = datetime.now(timezone.utc) - timedelta(days=1)
    turns_json = '[{"speaker":"Alice","dia_id":"D1","text":"hi"},{"speaker":"Bob","dia_id":"D2","text":"yo"}]'
    engine.ingest(_memory("session1", turns_json, expires_at=past))

    conn = sqlite3.connect(str(engine._db_path))
    pre = conn.execute("SELECT COUNT(*) FROM memories WHERE id LIKE 'session1%'").fetchone()[0]
    conn.close()
    assert pre == 1

    purged = engine.purge_expired()
    assert purged == 1  # only the parent counts; closet rows are cascade

    conn = sqlite3.connect(str(engine._db_path))
    post_mem = conn.execute("SELECT COUNT(*) FROM memories WHERE id LIKE 'session1%'").fetchone()[0]
    post_emb = conn.execute("SELECT COUNT(*) FROM memory_embeddings WHERE id LIKE 'session1%'").fetchone()[0]
    conn.close()
    assert post_mem == 0
    assert post_emb == 0


def test_migration_helper_is_idempotent(tmp_path: Path) -> None:
    """Reopening the same DB must not double-rewire triggers or duplicate FTS rows."""
    db = tmp_path / "memories.db"
    SeedEngine(db_path=db).ingest(_memory("legacy", "alpha"))
    _make_engine(db)
    _make_engine(db)  # second open

    conn = sqlite3.connect(str(db))
    triggers = conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()
    fts_count = conn.execute("SELECT COUNT(*) FROM memory_fts").fetchone()[0]
    conn.close()
    assert len(triggers) == 3
    assert fts_count == 1


def test_schema_migration_failure_raises(tmp_path: Path, monkeypatch: Any) -> None:
    """If the schema migration can't complete, raise loud rather than silently
    falling back. Users must see broken-DB conditions, not get degraded recall.
    """
    db = tmp_path / "memories.db"
    SeedEngine(db_path=db).ingest(_memory("legacy", "alpha"))

    # Simulate an immutable DB: make memories table read-only by removing
    # write permission via ATTACH+detach is awkward; instead, monkeypatch the
    # migration helper to raise.
    from poppy.engine import seed as baseline_mod

    def boom(_conn: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(baseline_mod, "_migrate_enriched_content", boom)
    with pytest.raises(sqlite3.OperationalError, match="disk full"):
        _make_engine(db)


def test_injected_encoders_stay_resident_and_skip_loading(tmp_path: Path) -> None:
    """Injected encoders must not lazily load or unload (FastembedModels._injected)."""
    engine = _make_engine(tmp_path / "t.db")
    assert engine._models._injected is True
    engine.ingest(_memory("m1", "content"))
    # A retrieve exercises both embed + rerank against the injected fakes.
    results = engine.retrieve("content", limit=5)
    assert any(r.memory.id == "m1" for r in results)


def test_mixed_model_id_db_self_heals_then_migrates(tmp_path: Path) -> None:
    """A row tagged by another engine goes FTS-only until re-embedded."""
    db = tmp_path / "memories.db"
    engine = _make_engine(db)
    engine.ingest(_memory("m1", "kubernetes ingress controller config"))
    engine.ingest(_memory("m2", "postgres connection pooling settings"))

    # Simulate a row left behind by a different bi-encoder.
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE memory_embeddings SET model_id = ? WHERE id = 'm1'", (LEGACY_TORCH_MODEL_ID,))
    conn.commit()
    conn.close()

    stats = stale_stats(db, BloomEngine.model_id)
    assert stats.compatible == 1
    assert stats.stale == 1
    assert stats.needs_migration == 1

    # m1 still retrievable via the FTS channel (degraded, not lost).
    ids = {r.memory.id for r in engine.retrieve("kubernetes ingress", limit=10)}
    assert "m1" in ids

    # migrate-engine re-embeds the stale row under bloom's model_id.
    migrated = run_migrate(engine, db, MigrateFilters())
    assert migrated == 1
    healed = stale_stats(db, BloomEngine.model_id)
    assert healed.needs_migration == 0
    assert _embedding_model_ids(db) == [BloomEngine.model_id, BloomEngine.model_id]


def test_legacy_torch_tagged_db_migrates_to_onnx(tmp_path: Path) -> None:
    """The documented 0.3.0 upgrade path for a store written by the torch engines.

    Those engines are gone, so their rows arrive tagged with the torch model_id:
    keyword-only until ``poppy migrate-engine`` re-embeds them, then whole.
    """
    db = tmp_path / "memories.db"
    engine = _make_engine(db)
    engine.ingest(_memory("m1", "always validate FastAPI request bodies with Pydantic"))
    engine.ingest(_memory("m2", "prefer PostgreSQL over MySQL for complex joins"))

    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE memory_embeddings SET model_id = ?", (LEGACY_TORCH_MODEL_ID,))
    conn.commit()
    conn.close()

    assert stale_stats(db, BloomEngine.model_id).needs_migration == 2
    assert run_migrate(engine, db, MigrateFilters()) == 2
    assert set(_embedding_model_ids(db)) == {BloomEngine.model_id}
    assert stale_stats(db, BloomEngine.model_id).needs_migration == 0


_TWO_SPEAKER_TURNS = '[{"speaker":"Alice","dia_id":"D1","text":"hi"},{"speaker":"Bob","dia_id":"D2","text":"yo"}]'


@pytest.mark.parametrize("engine_kind", ["bloom", "seed"])
def test_ingest_never_calls_legacy_classification(tmp_path: Path, monkeypatch, engine_kind: str) -> None:
    from poppy.engine import _legacy_copies

    engine = _make_engine(tmp_path / "memories.db") if engine_kind == "bloom" else SeedEngine(tmp_path / "memories.db")

    def unexpected(*args, **kwargs):
        pytest.fail("ingest reached legacy copy cleanup")

    monkeypatch.setattr(_legacy_copies, "classify_legacy_copy", unexpected)
    monkeypatch.setattr(_legacy_copies, "_projected_texts", unexpected)
    memory = _memory("conversation", _TWO_SPEAKER_TURNS)
    engine.ingest(memory)
    for marker in (1, 2):
        with engine._conn:
            engine._conn.execute("UPDATE memories SET is_closet = ? WHERE id = ?", (marker, memory.id))
        engine.ingest(memory)
        assert [tuple(row) for row in engine._conn.execute("SELECT id, is_closet FROM memories")] == [(memory.id, 0)]
