import datetime

import pytest

from poppy.models import Filters, Memory, Source

try:
    from poppy.engine.sprout import SproutEngine

    HAS_ML = True
except ImportError:
    HAS_ML = False

pytestmark = pytest.mark.skipif(not HAS_ML, reason="sentence-transformers not installed")


def _make_memory(id: str, content: str, memory_type: str = "fact", project: str | None = None) -> Memory:
    return Memory(
        id=id,
        content=content,
        memory_type=memory_type,
        source=Source(type="manual", session_id=None, timestamp=datetime.datetime.now(datetime.UTC)),
        project=project,
        related_to=[],
        created_at=datetime.datetime.now(datetime.UTC),
        updated_at=datetime.datetime.now(datetime.UTC),
        confidence=1.0,
    )


@pytest.fixture(scope="module")
def shared_engine(tmp_path_factory):
    """Share a single engine across tests to avoid reloading ML models per test."""
    tmp_dir = tmp_path_factory.mktemp("best_engine")
    engine = SproutEngine(db_path=tmp_dir / "test.db")
    engine.ingest(_make_memory("mem_001", "always use Pydantic validation on FastAPI endpoints"))
    engine.ingest(_make_memory("mem_002", "prefer PostgreSQL over MySQL for complex queries"))
    engine.ingest(_make_memory("mem_003", "use black formatter with 120 line length"))
    engine.ingest(_make_memory("mem_004", "chose React over Vue for the frontend", memory_type="decision"))
    engine.ingest(_make_memory("mem_005", "deploy to AWS using CDK", project="trags"))
    return engine


def test_ingest_and_get(shared_engine):
    retrieved = shared_engine.get("mem_001")
    assert retrieved is not None
    assert retrieved.content == "always use Pydantic validation on FastAPI endpoints"


def test_retrieve_semantic(shared_engine):
    """Semantic search should find relevant results even without exact keyword match."""
    results = shared_engine.retrieve("input validation in Python web APIs")
    assert len(results) > 0
    # Pydantic/FastAPI memory should rank high for this semantic query
    ids = [r.memory.id for r in results]
    assert "mem_001" in ids


def test_retrieve_with_project_filter(shared_engine):
    results = shared_engine.retrieve("deployment", filters=Filters(project="trags"))
    assert all(r.memory.project == "trags" for r in results)


def test_retrieve_with_type_filter(shared_engine):
    results = shared_engine.retrieve("frontend framework", filters=Filters(memory_type="decision"))
    assert all(r.memory.memory_type == "decision" for r in results)


def test_retrieve_limit(shared_engine):
    results = shared_engine.retrieve("code formatting", limit=2)
    assert len(results) <= 2


def test_delete(tmp_path):
    engine = SproutEngine(db_path=tmp_path / "test_del.db")
    engine.ingest(_make_memory("mem_del", "temporary memory"))
    assert engine.delete("mem_del") is True
    assert engine.get("mem_del") is None
    assert engine.delete("mem_del") is False


def test_ingest_preserves_incoming_updated_at_on_update(tmp_path):
    """Sync pull writes a Memory with the remote's updated_at. The UPDATE
    path must honor that timestamp, not overwrite it with `datetime.now()`,
    or pull→push will ping-pong rows forever."""
    import datetime as _dt

    engine = SproutEngine(db_path=tmp_path / "test_pres.db")
    mem = _make_memory("mem_pres", "content v1")
    engine.ingest(mem)

    fixed = _dt.datetime(2026, 1, 1, 12, 0, tzinfo=_dt.timezone.utc)
    mem.content = "content v2"
    mem.updated_at = fixed
    engine.ingest(mem)

    retrieved = engine.get("mem_pres")
    assert retrieved.updated_at == fixed


def test_list_all(shared_engine):
    memories = shared_engine.list_all()
    assert len(memories) == 5


def test_list_all_with_filters(shared_engine):
    results = shared_engine.list_all(filters=Filters(memory_type="decision"))
    assert len(results) == 1
    assert results[0].id == "mem_004"


def test_stats(shared_engine):
    s = shared_engine.stats()
    assert s.memory_count == 5
    assert s.engine_name == "sprout"
    assert s.storage_bytes > 0


def test_ingest_duplicate_updates(tmp_path):
    engine = SproutEngine(db_path=tmp_path / "test_dup.db")
    engine.ingest(_make_memory("mem_dup", "original content"))
    engine.ingest(_make_memory("mem_dup", "updated content"))
    retrieved = engine.get("mem_dup")
    assert retrieved.content == "updated content"


def test_init_loads_models_via_offline_safe_loader(tmp_path, monkeypatch):
    """When encoders are not injected, __init__ goes through the
    offline-safe loader (local_files_only + ModelUnavailableError) and announces
    the first-run download, instead of calling the bare constructors."""
    import numpy as np

    import poppy.engine.sprout as sprout_mod

    class _FakeBi:
        def encode(self, text, normalize_embeddings=True):
            return np.zeros(4, dtype=np.float32)

    class _FakeCross:
        def predict(self, pairs):
            return [0.0 for _ in pairs]

    loaded: list[tuple[str, str]] = []
    announced: list[tuple[str, ...]] = []

    def fake_load(kind, repo_id):
        loaded.append((kind, repo_id))
        return _FakeBi() if kind == "bi" else _FakeCross()

    monkeypatch.setattr(sprout_mod, "load_st_model", fake_load)
    monkeypatch.setattr(sprout_mod, "announce_first_run_download", lambda repos: announced.append(tuple(repos)))

    SproutEngine(db_path=tmp_path / "t.db")

    assert loaded == [("bi", sprout_mod.BI_ENCODER), ("cross", sprout_mod.CROSS_ENCODER)]
    assert announced == [(sprout_mod.BI_ENCODER, sprout_mod.CROSS_ENCODER)]
