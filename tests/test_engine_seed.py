import datetime

from poppy.engine.seed import SeedEngine
from poppy.models import Filters, Memory, Source


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


def test_ingest_and_get(tmp_path):
    engine = SeedEngine(db_path=tmp_path / "test.db")
    mem = _make_memory("mem_001", "use Pydantic for validation")
    result_id = engine.ingest(mem)
    assert result_id == "mem_001"
    retrieved = engine.get("mem_001")
    assert retrieved is not None
    assert retrieved.content == "use Pydantic for validation"


def test_ingest_duplicate_updates(tmp_path):
    engine = SeedEngine(db_path=tmp_path / "test.db")
    mem = _make_memory("mem_001", "original content")
    engine.ingest(mem)
    mem.content = "updated content"
    engine.ingest(mem)
    retrieved = engine.get("mem_001")
    assert retrieved.content == "updated content"


def test_ingest_preserves_incoming_updated_at_on_update(tmp_path):
    """Sync pull writes a Memory with the remote's updated_at. The UPDATE
    path must honor that timestamp, not overwrite it with `datetime.now()`,
    or pull→push will ping-pong rows forever."""
    import datetime as _dt

    engine = SeedEngine(db_path=tmp_path / "test.db")
    mem = _make_memory("mem_001", "content v1")
    engine.ingest(mem)

    fixed = _dt.datetime(2026, 1, 1, 12, 0, tzinfo=_dt.timezone.utc)
    mem.content = "content v2"
    mem.updated_at = fixed
    engine.ingest(mem)

    retrieved = engine.get("mem_001")
    assert retrieved.updated_at == fixed


def test_retrieve_fts(tmp_path):
    engine = SeedEngine(db_path=tmp_path / "test.db")
    engine.ingest(_make_memory("mem_001", "always use Pydantic validation on FastAPI endpoints"))
    engine.ingest(_make_memory("mem_002", "prefer PostgreSQL over MySQL for complex queries"))
    engine.ingest(_make_memory("mem_003", "use black formatter with 120 line length"))

    results = engine.retrieve("Pydantic validation")
    assert len(results) > 0
    assert results[0].memory.id == "mem_001"


def test_retrieve_no_results(tmp_path):
    engine = SeedEngine(db_path=tmp_path / "test.db")
    engine.ingest(_make_memory("mem_001", "use black formatter"))
    results = engine.retrieve("kubernetes deployment")
    assert len(results) == 0


def test_retrieve_with_project_filter(tmp_path):
    engine = SeedEngine(db_path=tmp_path / "test.db")
    engine.ingest(_make_memory("mem_001", "use FastAPI", project="trags"))
    engine.ingest(_make_memory("mem_002", "use Flask", project="other"))

    results = engine.retrieve("web framework", filters=Filters(project="trags"))
    assert all(r.memory.project == "trags" for r in results)


def test_retrieve_with_type_filter(tmp_path):
    engine = SeedEngine(db_path=tmp_path / "test.db")
    engine.ingest(_make_memory("mem_001", "chose FastAPI over Flask", memory_type="decision"))
    engine.ingest(_make_memory("mem_002", "always validate inputs", memory_type="preference"))

    results = engine.retrieve("FastAPI", filters=Filters(memory_type="decision"))
    assert all(r.memory.memory_type == "decision" for r in results)


def test_delete(tmp_path):
    engine = SeedEngine(db_path=tmp_path / "test.db")
    engine.ingest(_make_memory("mem_001", "test memory"))
    assert engine.delete("mem_001") is True
    assert engine.get("mem_001") is None
    assert engine.delete("mem_001") is False


def test_list_all(tmp_path):
    engine = SeedEngine(db_path=tmp_path / "test.db")
    engine.ingest(_make_memory("mem_001", "first memory"))
    engine.ingest(_make_memory("mem_002", "second memory"))
    engine.ingest(_make_memory("mem_003", "third memory"))

    memories = engine.list_all()
    assert len(memories) == 3


def test_list_all_with_filters(tmp_path):
    engine = SeedEngine(db_path=tmp_path / "test.db")
    engine.ingest(_make_memory("mem_001", "fact one", memory_type="fact", project="poppy"))
    engine.ingest(_make_memory("mem_002", "decision one", memory_type="decision", project="poppy"))
    engine.ingest(_make_memory("mem_003", "fact two", memory_type="fact", project="trags"))

    results = engine.list_all(filters=Filters(project="poppy"))
    assert len(results) == 2

    results = engine.list_all(filters=Filters(memory_type="fact"))
    assert len(results) == 2


def test_stats(tmp_path):
    engine = SeedEngine(db_path=tmp_path / "test.db")
    engine.ingest(_make_memory("mem_001", "test memory"))
    s = engine.stats()
    assert s.memory_count == 1
    assert s.engine_name == "seed"
    assert s.storage_bytes > 0


def test_list_all_with_limit(tmp_path):
    engine = SeedEngine(db_path=tmp_path / "test.db")
    for i in range(10):
        engine.ingest(_make_memory(f"mem_{i:03d}", f"memory number {i}"))
    results = engine.list_all(limit=5)
    assert len(results) == 5
