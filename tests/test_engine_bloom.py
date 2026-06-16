"""Tests for the BloomEngine — the local champion engine.

Covers the schema-upgrade contract: a DB created by an older engine
(``baseline``/``best`` schema, no ``enriched_content`` column) must open
transparently under BloomEngine and remain queryable. No silent
fallback to baseline if the upgrade fails — surface the error.

Sentence-transformers and the cross-encoder are stubbed so these tests stay
fast and offline; we exercise the storage + schema plumbing, not retrieval
quality. End-to-end retrieval quality is covered by the benchmark suite.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from poppy.engine.bloom import BloomEngine
from poppy.engine.seed import SeedEngine
from poppy.models import Memory, Source


class _FakeBiEncoder:
    """Deterministic 4-dim vector keyed off content hash."""

    def encode(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        h = hash(text) & 0xFFFF
        return np.array([h & 0xF, (h >> 4) & 0xF, (h >> 8) & 0xF, (h >> 12) & 0xF], dtype=np.float32)


class _FakeCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(len(b)) for _, b in pairs]


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


def test_fresh_db_round_trip(tmp_path: Path) -> None:
    engine = _make_engine(tmp_path / "t.db")
    engine.ingest(_memory("m1", "user prefers vim over emacs"))
    got = engine.get("m1")
    assert got is not None
    assert got.content == "user prefers vim over emacs"


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


def test_purge_expired_drops_parent_and_closets(tmp_path: Path) -> None:
    """purge_expired must also clean up synthetic closet rows."""
    engine = _make_engine(tmp_path / "t.db")
    past = datetime.now(timezone.utc) - timedelta(days=1)
    turns_json = '[{"speaker":"Alice","dia_id":"D1","text":"hi"},{"speaker":"Bob","dia_id":"D2","text":"yo"}]'
    engine.ingest(_memory("session1", turns_json, expires_at=past))

    conn = sqlite3.connect(str(engine._db_path))
    pre = conn.execute("SELECT COUNT(*) FROM memories WHERE id LIKE 'session1%'").fetchone()[0]
    conn.close()
    assert pre >= 2  # parent + at least one closet

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


def test_init_loads_models_via_offline_safe_loader(tmp_path: Path, monkeypatch: Any) -> None:
    """When encoders are not injected, __init__ goes through the
    offline-safe loader (local_files_only + ModelUnavailableError) and announces
    the first-run download, instead of calling the bare constructors."""
    import poppy.engine.bloom as bloom_mod

    loaded: list[tuple[str, str]] = []
    announced: list[tuple[str, ...]] = []

    def fake_load(kind: str, repo_id: str) -> Any:
        loaded.append((kind, repo_id))
        return _FakeBiEncoder() if kind == "bi" else _FakeCrossEncoder()

    monkeypatch.setattr(bloom_mod, "load_st_model", fake_load)
    monkeypatch.setattr(bloom_mod, "announce_first_run_download", lambda repos: announced.append(tuple(repos)))

    BloomEngine(db_path=tmp_path / "t.db")

    assert loaded == [("bi", bloom_mod.BI_ENCODER), ("cross", bloom_mod.CROSS_ENCODER)]
    assert announced == [(bloom_mod.BI_ENCODER, bloom_mod.CROSS_ENCODER)]


def test_init_with_injected_encoders_skips_loader(tmp_path: Path, monkeypatch: Any) -> None:
    """Injected encoders (tests, benchmarks) must not trigger model loading or notices."""
    import poppy.engine.bloom as bloom_mod

    def boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("loader must not be called")

    monkeypatch.setattr(bloom_mod, "load_st_model", boom)
    monkeypatch.setattr(bloom_mod, "announce_first_run_download", boom)

    engine = _make_engine(tmp_path / "t.db")  # injects fakes
    engine.ingest(_memory("m1", "still works"))
    assert engine.get("m1") is not None
