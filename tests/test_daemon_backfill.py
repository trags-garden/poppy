"""Bounded background restoration of enriched text, vectors, and the keyword index."""

import threading

import anyio
import pytest
from click.testing import CliRunner
from starlette.applications import Starlette
from test_engine_bloom import _FakeBiEncoder, _FakeCrossEncoder
from test_engine_migration import _FakeEngine, _make_memory

from poppy.cli.main import cli
from poppy.db import connect
from poppy.engine._fastembed_loader import FastembedModels
from poppy.engine._timestamps import migration_applied
from poppy.engine.bloom import BloomEngine
from poppy.engine.migration import BACKFILL_LIMIT, ENRICHMENT_REPAIR, backfill
from poppy.engine.seed import SeedEngine
from poppy.errors import ModelUnavailableError
from poppy.mcp_server import daemon
from poppy.mcp_server.server import PoppyMcpServer
from poppy.runtime import get_engine

CONVERSATION = '[{"speaker": "A", "text": "hi"}, {"speaker": "B", "text": "yo"}]'
TEXT_LIST = '["Never deploy on Friday"]'
RULE_LIST = '[{"rule": "Never deploy on Friday"}]'
MESSAGE_LIST = '[{"role": "user", "content": "Never deploy on Friday"}]'
MIXED_LIST = (
    '[{"speaker": "A", "text": "hello"}, "Never deploy on Friday", 42, null, ["nested text"], '
    '{"rule": "Keep café text"}, {"role": "user", "content": "Keep message text"}, '
    '{"text": "", "rule": "Keep empty turn text"}, {"text": 42, "rule": "Keep invalid turn text"}]'
)


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


@pytest.mark.parametrize("write_path", ["fresh", "backfill"])
@pytest.mark.parametrize(
    "content",
    [TEXT_LIST, RULE_LIST, MESSAGE_LIST, MIXED_LIST],
)
def test_json_list_text_stays_searchable_by_keyword(tmp_path, engine, write_path, content):
    memory = _make_memory("list", content)
    seed = SeedEngine(engine._db_path)
    if write_path == "fresh":
        engine.ingest(memory)
    else:
        seed.ingest(memory)
        assert daemon._backfill_once(engine, tmp_path) == 1

    assert [match.memory.id for match in seed.retrieve("Friday")] == [memory.id]
    indexed = _full_row(engine._db_path, memory.id)[0]
    if content != MIXED_LIST:
        assert indexed == content
    else:
        assert indexed.startswith("Conversation on ")
        assert (
            'A: hello\nNever deploy on Friday\n42\nnull\n["nested text"]\n'
            '{"rule": "Keep café text"}\n{"role": "user", "content": "Keep message text"}\n'
            '{"text": "", "rule": "Keep empty turn text"}\n{"text": 42, "rule": "Keep invalid turn text"}'
        ) in indexed


@pytest.mark.parametrize("content", [TEXT_LIST, RULE_LIST, MESSAGE_LIST, CONVERSATION])
@pytest.mark.parametrize("write_path", ["backfill", "manual"])
def test_current_model_damaged_enrichment_is_repaired(tmp_path, engine, content, write_path):
    memory = _make_memory("damaged", content)
    engine.ingest(memory)
    expected = _full_row(engine._db_path, memory.id)
    damaged = content if content == CONVERSATION else "Conversation on July 01, 2026."
    with connect(engine._db_path) as conn:
        conn.execute("UPDATE memories SET enriched_content = ?", (damaged,))
        conn.execute("UPDATE memory_embeddings SET embedding = ?", (_FakeEngine()._embed(damaged).tobytes(),))
    keyword = "Conversation" if content == CONVERSATION else "Friday"
    assert SeedEngine(engine._db_path).retrieve(keyword) == []

    if write_path == "manual":
        env = {"POPPY_DIR": str(tmp_path)}
        before = _full_row(engine._db_path, memory.id)
        dry_run = CliRunner().invoke(cli, ["migrate-engine", "--dry-run"], env=env)
        assert dry_run.exit_code == 0, dry_run.output
        assert "Target: 1 memories" in dry_run.output
        assert _full_row(engine._db_path, memory.id) == before
        with connect(engine._db_path) as conn:
            assert not migration_applied(conn, ENRICHMENT_REPAIR)
        result = CliRunner().invoke(cli, ["migrate-engine"], env=env)
        assert result.exit_code == 0, result.output
        assert "Target: 1 memories" in result.output
        assert "Re-embedded 1 memories." in result.output
    else:
        assert daemon._backfill_once(engine, tmp_path) == 1

    assert _full_row(engine._db_path, memory.id) == expected
    assert [match.memory.id for match in SeedEngine(engine._db_path).retrieve(keyword)] == [memory.id]
    with connect(engine._db_path) as conn:
        assert migration_applied(conn, ENRICHMENT_REPAIR)
    assert daemon._backfill_once(engine, tmp_path) == 0


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
        raise ModelUnavailableError("model temporarily unavailable")

    monkeypatch.setattr(engine, "_embed", unavailable)
    with pytest.raises(ModelUnavailableError, match="temporarily unavailable"):
        backfill(engine, engine._db_path, limit=2)
    with connect(engine._db_path) as conn:
        assert migration_applied(conn, ENRICHMENT_REPAIR)

    monkeypatch.setattr(engine, "_embed", embed)
    assert backfill(engine, engine._db_path, limit=2) == 1
    assert backfill(engine, engine._db_path, limit=2) == 0


def test_backfill_skips_bad_row_and_logs_only_id_and_exception_type(engine, monkeypatch, caplog):
    seed = SeedEngine(engine._db_path)
    seed.ingest(_make_memory("bad", "private memory text", created_days_ago=1))
    seed.ingest(_make_memory("good", "searchable memory"))
    embed = engine._embed
    attempts = []

    def fail_one(text):
        attempts.append(text)
        if text == "private memory text":
            raise ValueError("failed to embed private memory text")
        return embed(text)

    monkeypatch.setattr(engine, "_embed", fail_one)
    assert backfill(engine, engine._db_path, limit=2) == 1
    assert attempts == ["private memory text", "searchable memory"]
    assert _full_row(engine._db_path, "bad") is None
    assert _full_row(engine._db_path, "good")[2] == engine.model_id
    assert [record.getMessage() for record in caplog.records] == ["Memory bad: ValueError"]
    assert all(record.exc_info is None for record in caplog.records)

    monkeypatch.setattr(engine, "_embed", embed)
    assert backfill(engine, engine._db_path, limit=2) == 1


def test_manual_migration_finishes_a_partially_scanned_enrichment_repair(tmp_path, engine):
    for i in range(BACKFILL_LIMIT + 3):
        engine.ingest(_make_memory(f"m{i}", CONVERSATION))
    with connect(engine._db_path) as conn:
        conn.execute("UPDATE memories SET enriched_content = content")
    assert backfill(engine, engine._db_path, limit=2) == 2

    result = CliRunner().invoke(cli, ["migrate-engine"], env={"POPPY_DIR": str(tmp_path)})

    assert result.exit_code == 0, result.output
    assert f"Re-embedded {BACKFILL_LIMIT + 1} memories." in result.output
    with connect(engine._db_path) as conn:
        assert migration_applied(conn, ENRICHMENT_REPAIR)
        assert conn.execute("SELECT COUNT(*) FROM memories WHERE enriched_content = content").fetchone()[0] == 0


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
async def test_daemon_retries_offline_start_and_serves_full_pipeline_with_same_engine(tmp_path, monkeypatch, capsys):
    seed = SeedEngine(tmp_path / "memories.db")
    seed.ingest(_make_memory("offline", CONVERSATION))
    available = False
    loads = []
    reranks = []

    class CrossEncoder(_FakeCrossEncoder):
        def rerank(self, query, docs, **kwargs):
            reranks.append((query, docs))
            return super().rerank(query, docs, **kwargs)

    def load(self, kind):
        loads.append(kind)
        if not available:
            raise ModelUnavailableError("models unavailable")
        return _FakeBiEncoder() if kind == "bi" else CrossEncoder()

    monkeypatch.setattr(FastembedModels, "_load_model", load)
    engine = get_engine(tmp_path)
    server = PoppyMcpServer(poppy_dir=tmp_path, engine=engine)
    assert loads == []

    def unexpected_construction(*args, **kwargs):
        pytest.fail("backfill must use the engine serving requests")

    monkeypatch.setattr("poppy.runtime.get_engine", unexpected_construction)

    class Finished(Exception):
        pass

    pauses = []

    async def pause(seconds):
        nonlocal available
        pauses.append(seconds)
        assert seconds == daemon.BACKFILL_INTERVAL_S
        available = True
        if len(pauses) == 2:
            raise Finished

    monkeypatch.setattr(daemon.anyio, "sleep", pause)
    with pytest.raises(Finished):
        await daemon._backfill_loop(engine, tmp_path)

    assert _full_row(engine._db_path, "offline")[2] == engine.model_id
    result = await server.handle_recall(query="unrelated query")
    assert [memory["id"] for memory in result["memories"]] == ["offline"]
    assert server._engine is engine
    assert loads == ["bi", "bi", "cross"]
    assert reranks == [("unrelated query", [_full_row(engine._db_path, "offline")[0]])]
    assert "Daemon backfill failed" not in capsys.readouterr().err


@pytest.mark.asyncio
async def test_daemon_backfill_quietly_skips_model_free_engine(tmp_path, monkeypatch, capsys):
    seed = SeedEngine(tmp_path / "memories.db")
    seed.ingest(_make_memory("keyword", "keyword memory"))

    def unexpected_construction(*args, **kwargs):
        pytest.fail("a model-free engine must not be replaced during backfill")

    class Finished(Exception):
        pass

    async def pause(seconds):
        raise Finished

    monkeypatch.setattr("poppy.runtime.get_engine", unexpected_construction)
    monkeypatch.setattr(daemon.anyio, "sleep", pause)
    with pytest.raises(Finished):
        await daemon._backfill_loop(seed, tmp_path)

    assert seed.stats().memory_count == 1
    assert capsys.readouterr().err == ""
