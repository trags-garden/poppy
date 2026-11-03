"""Unavailable models defer embeddings without interrupting writes or recall."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_daemon_backfill import CONVERSATION, _full_row
from test_engine_bloom import _FakeBiEncoder, _FakeCrossEncoder
from test_engine_migration import _make_memory

from poppy.config import load_config, save_config
from poppy.db import connect
from poppy.engine import _fastembed_loader as fe
from poppy.engine.bloom import BloomEngine
from poppy.engine.seed import SEED_INVALIDATED_MODEL_ID, SeedEngine
from poppy.errors import ModelUnavailableError
from poppy.mcp_server import daemon
from poppy.mcp_server.server import PoppyMcpServer
from poppy.runtime import get_engine
from poppy.tombstones import TombstoneStore


@pytest.fixture
def offline_models(tmp_path, monkeypatch):
    state = SimpleNamespace(unavailable={"bi", "cross"}, loads=[], reranks=[])

    class CrossEncoder(_FakeCrossEncoder):
        def rerank(self, query, docs, **kwargs):
            state.reranks.append((query, docs))
            return super().rerank(query, docs, **kwargs)

    def load(model_name, **kwargs):
        kind = "bi" if model_name == fe.BI_ENCODER else "cross"
        state.loads.append(kind)
        if kind in state.unavailable:
            raise OSError("connection refused")
        return _FakeBiEncoder() if kind == "bi" else CrossEncoder()

    monkeypatch.setenv("POPPY_FASTEMBED_CACHE", str(tmp_path / "models"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(fe, "is_fastembed_model_cached", lambda *args: False)
    monkeypatch.setattr("fastembed.TextEmbedding", load)
    monkeypatch.setattr("fastembed.rerank.cross_encoder.TextCrossEncoder", load)
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize("supersede", [False, True])
async def test_offline_remember_and_recall_recover_without_restart(
    tmp_path, offline_models, supersede, real_model_health, capsys
):
    db_path = tmp_path / "memories.db"
    if supersede:
        initial = BloomEngine(db_path, bi_encoder=_FakeBiEncoder(), cross_encoder=_FakeCrossEncoder())
        initial.ingest(_make_memory("old", "old memory"))
        initial._conn.close()

    errors = []
    engine = get_engine(tmp_path, error_sink=errors.append)
    server = PoppyMcpServer(tmp_path, engine=engine)
    conn = engine._conn
    assert isinstance(engine, BloomEngine)
    assert offline_models.loads == errors == []

    saved = await server.handle_remember(CONVERSATION, project="p1", supersedes="old" if supersede else None)
    memory = engine.get(saved["id"])
    assert memory is not None and memory.content == CONVERSATION
    assert _full_row(db_path, memory.id) is None
    with connect(db_path) as stored:
        assert stored.execute("SELECT enriched_content FROM memories WHERE id = ?", (memory.id,)).fetchone() == (
            CONVERSATION,
        )
    if supersede:
        assert engine.get("old") is None
        assert _full_row(db_path, "old") is None
        assert TombstoneStore(db_path).get("old").memory.content == "old memory"

    cfg = load_config(tmp_path)
    cfg.recall_min_score = 0.99
    save_config(cfg)
    recalled = await server.handle_recall("hi", project="p1")
    index = await server.handle_recall_index("hi", project="p1")
    assert [m["id"] for m in recalled["memories"]] == [memory.id]
    assert [m["id"] for m in index["results"]] == [memory.id]
    assert len(errors) == 3
    assert all("ModelUnavailableError" in error for error in errors)
    assert offline_models.reranks == []
    assert "keyword-only" not in capsys.readouterr().err

    # Health checks and backfill must still see the missing model and retry later.
    with pytest.raises(ModelUnavailableError, match="one-time model download"):
        engine.check_models()
    with pytest.raises(ModelUnavailableError):
        daemon._backfill_once(engine, tmp_path)
    assert len(errors) == 3

    offline_models.unavailable.clear()
    recovered = await server.handle_recall("hi", project="p1")
    assert [m["id"] for m in recovered["memories"]] == [memory.id]
    assert offline_models.reranks == [("hi", [CONVERSATION])]
    assert engine._conn is conn and server._engine is engine
    assert len(errors) == 3

    assert daemon._backfill_once(engine, tmp_path) == 1
    fresh = BloomEngine(tmp_path / "fresh.db", bi_encoder=_FakeBiEncoder(), cross_encoder=_FakeCrossEncoder())
    fresh.ingest(memory)
    assert _full_row(db_path, memory.id) == _full_row(fresh._db_path, memory.id)
    assert daemon._backfill_once(engine, tmp_path) == 0


def test_offline_reingest_invalidates_embedding_and_backfills_like_fresh(tmp_path, offline_models):
    db_path = tmp_path / "memories.db"
    memory = _make_memory("edited", "obsolete text")
    fresh = BloomEngine(tmp_path / "fresh.db", bi_encoder=_FakeBiEncoder(), cross_encoder=_FakeCrossEncoder())
    initial = BloomEngine(db_path, bi_encoder=_FakeBiEncoder(), cross_encoder=_FakeCrossEncoder())
    initial.ingest(memory)
    initial._conn.close()
    engine = get_engine(tmp_path)
    updated = replace(memory, content=CONVERSATION)

    assert engine.ingest(updated) == memory.id
    assert _full_row(db_path, memory.id)[2] == SEED_INVALIDATED_MODEL_ID
    assert [m.memory.id for m in engine.retrieve("hi")] == [memory.id]
    assert engine.retrieve("obsolete") == []

    offline_models.unavailable.clear()
    assert daemon._backfill_once(engine, tmp_path) == 1
    fresh.ingest(updated)
    assert _full_row(db_path, memory.id) == _full_row(fresh._db_path, memory.id)


def test_missing_reranker_falls_back_only_for_that_search(tmp_path, offline_models):
    offline_models.unavailable = {"cross"}
    engine = get_engine(tmp_path)
    memory = _make_memory("stored", "keyword memory")
    engine.ingest(memory)

    matches = engine.retrieve("keyword")
    assert [m.memory.id for m in matches] == [memory.id]
    assert all(m.keyword_only for m in matches)
    assert _full_row(engine._db_path, memory.id)[2] == engine.model_id

    offline_models.unavailable.clear()
    matches = engine.retrieve("unrelated")
    assert [m.memory.id for m in matches] == [memory.id]
    assert all(not m.keyword_only for m in matches)
    assert offline_models.loads == ["bi", "cross", "cross"]


@pytest.mark.parametrize("failure", [ValueError("bad input"), RuntimeError("corrupt model")])
def test_non_model_ingest_error_propagates(tmp_path, monkeypatch, failure):
    errors = []
    engine = get_engine(tmp_path, error_sink=errors.append)

    def fail(text):
        raise failure

    monkeypatch.setattr(engine, "_embed", fail)
    with pytest.raises(type(failure), match=str(failure)):
        engine.ingest(_make_memory("failed", "not stored"))
    assert engine.get("failed") is None
    assert errors == []


def test_offline_supersede_rolls_back_archive_and_delete_on_write_error(tmp_path, monkeypatch, offline_models):
    engine = get_engine(tmp_path)
    old = _make_memory("old", "keep this memory")
    engine.ingest(old)
    with connect(engine._db_path) as conn:
        conn.execute("CREATE TABLE archived (id TEXT)")

    def archive(conn):
        conn.execute("INSERT INTO archived VALUES (?)", (old.id,))

    def fail(self, memory):
        raise OSError("disk failure")

    monkeypatch.setattr(SeedEngine, "ingest", fail)
    with pytest.raises(OSError, match="disk failure"):
        engine.supersede(old.id, _make_memory("new", "replacement"), before_delete=archive)
    assert engine.get(old.id) == old
    assert engine.get("new") is None
    with connect(engine._db_path) as conn:
        assert conn.execute("SELECT * FROM archived").fetchall() == []
