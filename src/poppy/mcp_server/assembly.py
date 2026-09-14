"""Context assembly for the MCP `recall` tool.

Turns the flat ranked list `handle_recall` produces into reasoning-ready
context: per-memory date anchor, relevance score, and an explicit
superseded / expired / expiring marker, grouped by project and
deterministically ordered (score desc). This is the calling agent's only
"answer pipeline" — Poppy ships no answerer — so packaging is load-bearing.

Formatting/assembly only: no engine or retrieval changes. The supersession
signal lives in the UI/lifecycle tombstone sidecar, not on the live Memory,
so it is read here read-only and defensively (see `lookup_superseded`).
"""

from __future__ import annotations

import datetime
from pathlib import Path

# Window sizing. `recall` already caps the result count via its `limit`
# argument; this bounds how much of each memory's content spends the context
# budget. A memory longer than this is truncated with an ellipsis.
PER_MEMORY_CHAR_BUDGET = 500

# How soon an expiry is worth flagging as "expiring" rather than left silent.
EXPIRING_SOON = datetime.timedelta(days=7)

_NO_PROJECT = "(no project)"


def lifecycle_markers(
    *,
    expires_at: datetime.datetime | None,
    superseded: bool,
    now: datetime.datetime,
) -> list[str]:
    """Staleness markers for one memory, in display order.

    - ``superseded``: a newer memory has replaced this one (tombstone sidecar).
    - ``expired``: past its ``expires_at`` (only reachable when the caller
      passes ``include_expired``).
    - ``expires <date>``: expires within :data:`EXPIRING_SOON`.
    """
    markers: list[str] = []
    if superseded:
        markers.append("superseded")
    if expires_at is not None:
        if expires_at <= now:
            markers.append("expired")
        elif expires_at <= now + EXPIRING_SOON:
            markers.append(f"expires {expires_at.date().isoformat()}")
    return markers


def lookup_superseded(poppy_dir: Path, ids: list[str]) -> set[str]:
    """IDs among ``ids`` the tombstone sidecar records as superseded.

    Read-only and defensive: if the sidecar table has never been created (no
    supersede or `poppy ui` has run) or the DB can't be opened, returns an
    empty set so recall never fails on the marker lookup. Deliberately does not
    create or migrate ``ui_tombstones`` — that stays owned by the UI/lifecycle
    layer.
    """
    if not ids:
        return set()
    db_path = poppy_dir / "memories.db"
    if not db_path.exists():
        return set()

    from poppy.db import apply_row_factory
    from poppy.db import connect as connect_db

    conn = None
    try:
        conn = connect_db(db_path)
        apply_row_factory(conn)
        table = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ui_tombstones'").fetchone()
        if not table:
            return set()
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"SELECT id FROM ui_tombstones WHERE superseded_by IS NOT NULL AND id IN ({placeholders})",
            tuple(ids),
        ).fetchall()
        return {row["id"] for row in rows}
    except Exception:
        # A marker lookup must never take down recall itself.
        return set()
    finally:
        if conn is not None:
            conn.close()


def assemble_recall_context(memories: list[dict], *, char_budget: int = PER_MEMORY_CHAR_BUDGET) -> str:
    """Assemble enriched, grouped, reasoning-ready recall context.

    ``memories`` are `handle_recall` dicts, already deterministically ordered
    (score desc). Groups by project — preserving that score-desc order both
    across groups and within each — and renders one line per memory with its
    date anchor, relevance score, any staleness markers, budget-truncated
    content, and id.
    """
    if not memories:
        return "No memories found."

    groups: dict[str, list[dict]] = {}
    for m in memories:
        key = m.get("project") or _NO_PROJECT
        groups.setdefault(key, []).append(m)

    blocks: list[str] = []
    for project, mems in groups.items():
        lines = [f"## {project}"]
        lines.extend(_format_memory_line(m, char_budget=char_budget) for m in mems)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_memory_line(m: dict, *, char_budget: int) -> str:
    date = _date_only(m.get("created_at"))
    score = m.get("score")
    score_str = f"score {score:.3f}" if isinstance(score, (int, float)) else "score n/a"
    markers = m.get("markers") or []
    marker_str = f" ⚠ {', '.join(markers)}" if markers else ""
    content = _truncate(m.get("content", ""), char_budget)
    return f"- [{m.get('memory_type')}]{marker_str} ({date}, {score_str}) {content} (id: {m.get('id')})"


def _date_only(iso: str | None) -> str:
    if not iso:
        return "unknown date"
    return iso[:10]


def _truncate(text: str, budget: int) -> str:
    if budget > 0 and len(text) > budget:
        return text[: budget - 1].rstrip() + "…"
    return text
