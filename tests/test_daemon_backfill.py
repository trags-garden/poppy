"""Bounded background restoration of enriched text, vectors, and the keyword index."""

import threading

import anyio
import pytest
from starlette.applications import Starlette
from test_engine_migration import _FakeEngine, _make_memory

from poppy.db import connect
from poppy.engine._timestamps import migration_applied
from poppy.engine.bloom import BloomEngine
from poppy.engine.migration import BACKFILL_LIMIT, ENRICHMENT_REPAIR, backfill
from poppy.engine.seed import SeedEngine
from poppy.mcp_server import daemon

CONVERSATION = '[{"speaker": "A", "text": "hi"}, {"speaker": "B", "text": "yo"}]'


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(BloomEngine, "_embed", _FakeEngine._embed)
    return BloomEngine(tmp_path / "memories.db")


def _full_row(db_path, memory_id):
    with connect(db_path) as conn:
        return conn.execute(
            "SELECT m.enriched_content, e.embedding, e.model_id FROM memories m "
            "JOIN memory_embeddings e ON m.id = e.id WHERE m.id = ?",
            (memory_id,),
        ).fetchone()


def test_daemon_backfill_matches_fresh_ingest_and_updates_fts(tmp_path, engine):
    memory = _make_memory("conversation", CONVERSATION)
    fresh = BloomEngine(tmp_path / "fresh.db")
    fresh.ingest(memory)
    SeedEngine(engine._db_path).ingest(memory)

    assert daemon._backfill_once(engine, tmp_path) == 1
    assert _full_row(engine._db_path, memory.id) == _full_row(fresh._db_path, memory.id)
    with connect(engine._db_path) as conn:
        assert conn.execute("SELECT id FROM memory_fts WHERE memory_fts MATCH 'Conversation'").fetchall() == [
            (memory.id,)
        ]


@pytest.mark.parametrize("model_id", [None, "old-model", "seed-invalidated"])
def test_daemon_backfills_every_default_migration_target(tmp_path, engine, model_id):
    memory = _make_memory("conversation", CONVERSATION)
    engine.ingest(memory)
    expected = _full_row(engine._db_path, memory.id)
    with connect(engine._db_path) as conn:
        conn.execute("UPDATE memory_embeddings SET model_id = ?", (model_id,))
        conn.execute("UPDATE memories SET enriched_content = content")

    assert daemon._backfill_once(engine, tmp_path) == 1
    assert _full_row(engine._db_path, memory.id) == expected


def test_current_model_raw_conversation_is_repaired_only_once(tmp_path, engine, monkeypatch):
    memory = _make_memory("conversation", CONVERSATION)
    engine.ingest(memory)
    engine.ingest(_make_memory("plain", "plain string"))
    expected = _full_row(engine._db_path, memory.id)
    plain = _full_row(engine._db_path, "plain")
    with connect(engine._db_path) as conn:
        conn.execute("UPDATE memories SET enriched_content = content WHERE id = ?", (memory.id,))
        conn.execute(
            "UPDATE memory_embeddings SET embedding = ? WHERE id = ?",
            (_FakeEngine()._embed(memory.content).tobytes(), memory.id),
        )

    assert daemon._backfill_once(engine, tmp_path) == 1
    assert _full_row(engine._db_path, memory.id) == expected
    assert _full_row(engine._db_path, "plain") == plain
    with connect(engine._db_path) as conn:
        assert migration_applied(conn, ENRICHMENT_REPAIR)

    def unexpected_scan(*args):
        pytest.fail("completed enrichment repair must not scan memories again")

    monkeypatch.setattr(engine, "embed_text_for", unexpected_scan)
    assert daemon._backfill_once(engine, tmp_path) == 0


def test_daemon_backfill_bounds_scan_and_embeddings_per_pass(tmp_path, engine, monkeypatch):
    seed = SeedEngine(engine._db_path)
    for i in range(BACKFILL_LIMIT + 3):
        seed.ingest(_make_memory(f"m{i}", CONVERSATION))
    renders = []
    render = engine.embed_text_for

    def record_render(*args):
        renders.append(args)
        return render(*args)

    monkeypatch.setattr(engine, "embed_text_for", record_render)
    assert daemon._backfill_once(engine, tmp_path) == BACKFILL_LIMIT
    # One rendering per inspected row and one per embedding, both bounded.
    assert len(renders) == 2 * BACKFILL_LIMIT
    with connect(engine._db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0] == BACKFILL_LIMIT
        assert not migration_applied(conn, ENRICHMENT_REPAIR)

    renders.clear()
    assert daemon._backfill_once(engine, tmp_path) == 3
    assert len(renders) == 6
    with connect(engine._db_path) as conn:
        assert migration_applied(conn, ENRICHMENT_REPAIR)


def test_repair_scan_resumes_and_failed_embedding_remains_a_target(engine, monkeypatch):
    for i in range(3):
        engine.ingest(_make_memory(f"m{i}", CONVERSATION))
    with connect(engine._db_path) as conn:
        conn.execute("UPDATE memories SET enriched_content = content")

    assert backfill(engine, engine._db_path, limit=2) == 2
    embed = engine._embed

    def unavailable(text):
        raise OSError("model temporarily unavailable")

    monkeypatch.setattr(engine, "_embed", unavailable)
    with pytest.raises(OSError, match="temporarily unavailable"):
        backfill(engine, engine._db_path, limit=2)
    with connect(engine._db_path) as conn:
        assert migration_applied(conn, ENRICHMENT_REPAIR)

    monkeypatch.setattr(engine, "_embed", embed)
    assert backfill(engine, engine._db_path, limit=2) == 1
    assert backfill(engine, engine._db_path, limit=2) == 0


@pytest.mark.asyncio
async def test_daemon_lifespan_runs_backfill_on_a_worker_thread(tmp_path, engine, monkeypatch):
    ran = threading.Event()
    threads = []

    def run_pass(active, poppy_dir):
        assert active is engine
        assert poppy_dir == tmp_path
        threads.append(threading.get_ident())
        ran.set()

    monkeypatch.setattr(daemon, "_backfill_once", run_pass)
    app = daemon.create_http_app(Starlette(), engine=engine, poppy_dir=tmp_path, port=7679)
    async with app.router.lifespan_context(app):
        assert await anyio.to_thread.run_sync(ran.wait, 5)

    assert threads and threading.get_ident() not in threads


@pytest.mark.asyncio
async def test_daemon_retries_offline_start_and_reuses_recovered_models(tmp_path, engine, monkeypatch):
    seed = SeedEngine(engine._db_path)
    seed.ingest(_make_memory("offline", CONVERSATION))
    resolutions = []

    def resolve(poppy_dir):
        resolutions.append(poppy_dir)
        return engine

    class Finished(Exception):
        pass

    pauses = []

    async def pause(seconds):
        pauses.append(seconds)
        if len(pauses) == 2:
            raise Finished

    monkeypatch.setattr("poppy.runtime.get_engine", resolve)
    monkeypatch.setattr(daemon.anyio, "sleep", pause)
    with pytest.raises(Finished):
        await daemon._backfill_loop(seed, tmp_path)

    assert resolutions == [tmp_path]
    assert _full_row(engine._db_path, "offline")[2] == engine.model_id
