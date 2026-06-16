import datetime

from poppy.models import Filters, Memory, ScoredMemory, Source


def test_memory_creation():
    source = Source(type="manual", session_id=None, timestamp=datetime.datetime.now(datetime.UTC))
    memory = Memory(
        id="mem_001",
        content="always use Pydantic validation on FastAPI endpoints",
        memory_type="preference",
        source=source,
        project="trags-apps",
        related_to=[],
        created_at=datetime.datetime.now(datetime.UTC),
        updated_at=datetime.datetime.now(datetime.UTC),
        confidence=1.0,
    )
    assert memory.id == "mem_001"
    assert memory.memory_type == "preference"
    assert memory.project == "trags-apps"


def test_memory_creation_no_project():
    source = Source(type="manual", session_id=None, timestamp=datetime.datetime.now(datetime.UTC))
    memory = Memory(
        id="mem_002",
        content="prefer list comprehensions over map/filter",
        memory_type="preference",
        source=source,
        project=None,
        related_to=[],
        created_at=datetime.datetime.now(datetime.UTC),
        updated_at=datetime.datetime.now(datetime.UTC),
        confidence=1.0,
    )
    assert memory.project is None


def test_filters_defaults():
    f = Filters()
    assert f.project is None
    assert f.since is None
    assert f.memory_type is None
    assert f.min_confidence is None


def test_filters_with_values():
    f = Filters(project="poppy", memory_type="decision")
    assert f.project == "poppy"
    assert f.memory_type == "decision"


def test_scored_memory():
    source = Source(type="manual", session_id=None, timestamp=datetime.datetime.now(datetime.UTC))
    memory = Memory(
        id="mem_003",
        content="test content",
        memory_type="fact",
        source=source,
        project=None,
        related_to=[],
        created_at=datetime.datetime.now(datetime.UTC),
        updated_at=datetime.datetime.now(datetime.UTC),
        confidence=1.0,
    )
    scored = ScoredMemory(memory=memory, score=0.95)
    assert scored.score == 0.95
    assert scored.memory.id == "mem_003"
