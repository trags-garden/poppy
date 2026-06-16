"""CaptureJournal — local record of what each capture stored.

One local source of truth for the SessionStart banner, ``poppy doctor``, and the
dashboard: each capture appends a record with the session, project, count, and a
short preview of each stored item. It is a per-device cache of already-synced
data, so it stays local (it is NOT synced — see the sync boundary).

Banner/doctor surfaces that read this are out of scope for the capture slices;
this module exists so the capture worker has one place to record provenance the
later surfaces will read.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

JOURNAL_FILENAME = "capture_journal.jsonl"
PREVIEW_CHARS = 80
MAX_ENTRIES = 200


@dataclass
class CaptureRecord:
    ts: str
    session_id: str
    project: str | None
    count: int
    items: list[dict[str, str]] = field(default_factory=list)


def _journal_path(poppy_dir: Path) -> Path:
    return poppy_dir / JOURNAL_FILENAME


def _preview(text: str) -> str:
    text = " ".join((text or "").split())
    return text[:PREVIEW_CHARS]


def record(
    poppy_dir: Path,
    *,
    session_id: str,
    project: str | None,
    count: int,
    items: list[Any],
    max_entries: int = MAX_ENTRIES,
) -> CaptureRecord:
    """Append a capture record. ``items`` are objects with ``.memory_type`` and
    ``.content`` (Memory) — only the type and a short preview are stored."""
    rec = CaptureRecord(
        ts=datetime.datetime.now(datetime.UTC).isoformat(),
        session_id=session_id,
        project=project,
        count=count,
        items=[
            {"type": getattr(it, "memory_type", "fact"), "preview": _preview(getattr(it, "content", ""))}
            for it in items
        ],
    )

    poppy_dir.mkdir(parents=True, exist_ok=True)
    path = _journal_path(poppy_dir)
    existing: list[str] = []
    if path.exists():
        existing = [ln for ln in path.read_text().splitlines() if ln.strip()]
    existing.append(json.dumps(rec.__dict__))
    existing = existing[-max_entries:]
    path.write_text("\n".join(existing) + "\n")
    return rec


def read_all(poppy_dir: Path) -> list[dict[str, Any]]:
    """Every journal record as raw dicts, oldest first. Skips unparseable lines.

    Public accessor for surfaces that need the whole journal (e.g. the doctor's
    journal-count line); most callers want :func:`read_last` instead.
    """
    path = _journal_path(poppy_dir)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def read_last(poppy_dir: Path) -> CaptureRecord | None:
    """The most recent capture record, or None if nothing captured yet."""
    rows = read_all(poppy_dir)
    if not rows:
        return None
    data = rows[-1]
    return CaptureRecord(
        ts=data.get("ts", ""),
        session_id=data.get("session_id", ""),
        project=data.get("project"),
        count=int(data.get("count", 0)),
        items=data.get("items", []),
    )


def last_session_count(poppy_dir: Path) -> int | None:
    """Total memories captured in the most recent session that captured anything.

    A session can produce several capture records (each mid-session fire plus the
    SessionEnd backstop), so the banner's "M captured last session" sums the counts
    of every record sharing the latest record's ``session_id``. Returns ``None``
    when nothing has been captured yet, so the banner can omit the clause rather
    than print a misleading "0 captured".
    """
    rows = read_all(poppy_dir)
    if not rows:
        return None
    last_session = rows[-1].get("session_id")
    return sum(int(r.get("count", 0)) for r in rows if r.get("session_id") == last_session)
