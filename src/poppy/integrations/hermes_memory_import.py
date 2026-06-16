"""Import curated memory files from ~/.hermes/memories/.

Hermes Agent (Nous Research) stores agent-curated facts in
``~/.hermes/memories/MEMORY.md`` and user-profile facts in
``~/.hermes/memories/USER.md``. Each file is ``§``-delimited paragraphs.

We map:
  MEMORY.md → memory_type=fact          (agent-curated observations)
  USER.md   → memory_type=preference    (user-profile facts)

IDs are content-hash-based so re-running the import is idempotent —
edits in hermes show up as new memories on the next sync, unedited
paragraphs are skipped.
"""

from __future__ import annotations

import datetime
import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from poppy.engine.interface import RetrievalEngine
from poppy.models import Memory, Source

_DELIMITER = "§"


@dataclass
class ParsedHermesMemory:
    body: str
    source_file: str  # "MEMORY.md" | "USER.md"


@dataclass
class HermesImportResult:
    imported: int
    skipped: int
    failed: int
    paths_imported: list[str]  # content snippets for dry-run reporting


def default_hermes_memories_dir() -> Path:
    """Mirror hermes_constants.get_hermes_home(): env var, then ~/.hermes."""
    val = os.environ.get("HERMES_HOME", "").strip()
    if val:
        return Path(val).expanduser() / "memories"
    return Path.home() / ".hermes" / "memories"


def iter_paragraphs(path: Path) -> Iterator[str]:
    if not path.is_file():
        return
    text = path.read_text()
    for chunk in text.split(_DELIMITER):
        para = chunk.strip()
        if para:
            yield para


def stable_id(source_file: str, body: str) -> str:
    digest = hashlib.sha256(f"{source_file}::{body}".encode()).hexdigest()[:12]
    return f"mem_hermem_{digest}"


def map_memory_type(source_file: str) -> str:
    if source_file == "USER.md":
        return "preference"
    return "fact"


def import_hermes_memories(
    engine: RetrievalEngine,
    memories_dir: Path | None = None,
    *,
    dry_run: bool = False,
) -> HermesImportResult:
    memories_dir = memories_dir or default_hermes_memories_dir()
    if not memories_dir.is_dir():
        return HermesImportResult(imported=0, skipped=0, failed=0, paths_imported=[])

    imported: list[str] = []
    skipped = 0
    failed = 0

    for filename in ("MEMORY.md", "USER.md"):
        path = memories_dir / filename
        for body in iter_paragraphs(path):
            try:
                memory_id = stable_id(filename, body)
                if engine.get(memory_id) is not None:
                    skipped += 1
                    continue
                if dry_run:
                    imported.append(f"{filename}: {body[:80]}")
                    continue
                now = datetime.datetime.now(datetime.UTC)
                memory = Memory(
                    id=memory_id,
                    content=body,
                    memory_type=map_memory_type(filename),
                    source=Source(type="hermes-memory", session_id=filename, timestamp=now),
                    project=None,
                    related_to=[],
                    created_at=now,
                    updated_at=now,
                    confidence=1.0,
                )
                engine.ingest(memory)
                imported.append(f"{filename}: {body[:80]}")
            except Exception:
                failed += 1

    return HermesImportResult(
        imported=len(imported) if not dry_run else 0,
        skipped=skipped,
        failed=failed,
        paths_imported=imported,
    )
