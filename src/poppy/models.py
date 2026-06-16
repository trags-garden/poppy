from dataclasses import dataclass
from datetime import datetime


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
