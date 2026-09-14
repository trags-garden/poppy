from dataclasses import dataclass
from datetime import datetime

# Canonical memory types in use across the codebase: the CLI's four writable
# types (fact/decision/preference/lesson), the LLM-consolidation "summary", and
# the UI/session "context" pin. `normalize_memory_type` coerces anything else to
# the safe default so a local write boundary (MCP `remember`/`edit`) can't stamp
# a novel type string onto surfaces that render or key off it. Note: cloud sync
# pull deliberately does NOT normalize — freeform types are by design on the
# Trags side; the UI's render-time escaping is the XSS guard there.
MEMORY_TYPES: frozenset[str] = frozenset({"fact", "decision", "preference", "lesson", "summary", "context"})
DEFAULT_MEMORY_TYPE = "fact"


def normalize_memory_type(value: str | None) -> str:
    """Return a canonical memory type, coercing unknown/empty values to the default."""
    if value is None:
        return DEFAULT_MEMORY_TYPE
    candidate = value.strip().lower()
    return candidate if candidate in MEMORY_TYPES else DEFAULT_MEMORY_TYPE


@dataclass
class Source:
    type: str  # claude-code | cursor | manual | obsidian
    session_id: str | None
    timestamp: datetime


@dataclass
class Memory:
    id: str
    content: str
    memory_type: str  # fact | decision | preference | lesson | context
    source: Source
    project: str | None
    related_to: list[str]
    created_at: datetime
    updated_at: datetime
    confidence: float = 1.0
    expires_at: datetime | None = None


@dataclass
class Filters:
    project: str | None = None
    since: datetime | None = None
    memory_type: str | None = None
    min_confidence: float | None = None
    include_expired: bool = False


@dataclass
class ScoredMemory:
    memory: Memory
    score: float
