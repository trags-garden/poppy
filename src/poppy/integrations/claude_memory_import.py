"""Import curated auto-memory files from ~/.claude/projects/<slug>/memory/."""

import datetime
import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from poppy.engine.interface import RetrievalEngine
from poppy.models import Memory, Source

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


@dataclass
class ParsedMemory:
    name: str
    description: str
    type: str
    body: str
    source_path: Path


@dataclass
class ImportResult:
    imported: int
    skipped: int
    failed: int
    paths_imported: list[Path]


def default_claude_projects_dir() -> Path:
    return Path.home() / ".claude" / "projects"


def iter_memory_files(projects_dir: Path) -> Iterator[Path]:
    for project_dir in sorted(projects_dir.iterdir()):
        memory_dir = project_dir / "memory"
        if not memory_dir.is_dir():
            continue
        for path in sorted(memory_dir.glob("*.md")):
            if path.name == "MEMORY.md":
                continue
            yield path


def parse_memory_file(path: Path) -> ParsedMemory | None:
    text = path.read_text()
    match = FRONTMATTER_RE.match(text)
    if not match:
        return None
    fm_block, body = match.groups()
    fm: dict[str, str] = {}
    for line in fm_block.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        fm[key.strip()] = val.strip()
    return ParsedMemory(
        name=fm.get("name", path.stem),
        description=fm.get("description", ""),
        type=fm.get("type", "user"),
        body=body.strip(),
        source_path=path,
    )


def project_name_from_slug(slug: str) -> str:
    """Heuristic: take last '-'-separated segment.

    Slug encoding is lossy (path separators and dots both collapse into '-'),
    so we accept that 'karpathy-guidelines' ends up as 'guidelines'. The user
    can rename via stored memories if needed.
    """
    return slug.lstrip("-").split("-")[-1] or slug


def map_memory_type(auto_type: str) -> str:
    """Map auto-memory type to Poppy memory_type."""
    if auto_type == "feedback":
        return "preference"
    return "fact"


def stable_id(slug: str, filename_stem: str) -> str:
    digest = hashlib.sha256(f"{slug}/{filename_stem}".encode()).hexdigest()[:12]
    return f"mem_clmem_{digest}"


def build_content(parsed: ParsedMemory) -> str:
    header = parsed.name
    if parsed.description:
        header = f"{parsed.name} — {parsed.description}"
    if parsed.body:
        return f"{header}\n\n{parsed.body}"
    return header


def import_claude_memories(
    engine: RetrievalEngine,
    projects_dir: Path | None = None,
    *,
    dry_run: bool = False,
) -> ImportResult:
    projects_dir = projects_dir or default_claude_projects_dir()
    if not projects_dir.is_dir():
        return ImportResult(imported=0, skipped=0, failed=0, paths_imported=[])

    imported: list[Path] = []
    skipped = 0
    failed = 0

    for path in iter_memory_files(projects_dir):
        slug = path.parent.parent.name
        parsed = parse_memory_file(path)
        if parsed is None:
            failed += 1
            continue

        memory_id = stable_id(slug, path.stem)
        if engine.get(memory_id) is not None:
            skipped += 1
            continue

        if dry_run:
            imported.append(path)
            continue

        now = datetime.datetime.now(datetime.UTC)
        memory = Memory(
            id=memory_id,
            content=build_content(parsed),
            memory_type=map_memory_type(parsed.type),
            source=Source(type="claude-memory", session_id=f"{slug}/{path.stem}", timestamp=now),
            project=project_name_from_slug(slug),
            related_to=[],
            created_at=now,
            updated_at=now,
            confidence=1.0,
        )
        engine.ingest(memory)
        imported.append(path)

    return ImportResult(imported=len(imported), skipped=skipped, failed=failed, paths_imported=imported)
