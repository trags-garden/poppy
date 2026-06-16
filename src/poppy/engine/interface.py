from abc import ABC, abstractmethod
from dataclasses import dataclass

from poppy.models import Filters, Memory, ScoredMemory


@dataclass
class EngineStats:
    memory_count: int
    storage_bytes: int
    engine_name: str
    engine_version: str


@dataclass
class ConsolidationResult:
    merged: int
    removed: int
    updated: int


class RetrievalEngine(ABC):
    # Embedding model fingerprint. Engines that use a bi-encoder return a stable
    # identifier (e.g. "all-MiniLM-L6-v2"). FTS-only engines like the baseline
    # return None — they neither produce nor consume embeddings, so a stored
    # row's model_id never matters to them.
    #
    # The shared `memory_embeddings` table tags each BLOB with the model_id
    # that produced it. On read, engines filter to rows matching this
    # value; mismatching rows degrade to FTS-only quality for that document
    # rather than feeding meaningless cross-model cosine scores into RRF.
    model_id: str | None = None

    @abstractmethod
    def ingest(self, memory: Memory) -> str:
        """Store a memory, return its ID."""

    @abstractmethod
    def retrieve(self, query: str, filters: Filters | None = None, limit: int = 10) -> list[ScoredMemory]:
        """Search memories, return ranked results."""

    @abstractmethod
    def get(self, memory_id: str) -> Memory | None:
        """Get a single memory by ID."""

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID. Returns True if found and deleted."""

    @abstractmethod
    def list_all(self, filters: Filters | None = None, limit: int = 50) -> list[Memory]:
        """List memories, optionally filtered."""

    @abstractmethod
    def consolidate(self) -> ConsolidationResult:
        """Periodic maintenance -- merge, summarize, reorganize."""

    @abstractmethod
    def stats(self) -> EngineStats:
        """Memory count, storage size, index health."""
