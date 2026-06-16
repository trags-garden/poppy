"""TranscriptWindow — incremental ``(watermark, now]`` reader (ADR-0001).

The shared transcript parser for both the SessionEnd backstop and the mid-session
capture worker. It tail-reads the Claude Code JSONL, skips compact-summary
entries, keeps user/assistant turns, flattens text blocks, drops tool-only and
sub-threshold turns, and truncates each message.

A "turn" is one qualifying (text-bearing) user/assistant message. The watermark
indexes these qualifying turns, so the window ``(watermark, now]`` is exactly the
turns added since the last capture. The reader **clamps** to the turns actually
present: if compaction shrank the transcript below the watermark, the window is
empty and the new watermark resets down to what remains — compaction can never
desync the watermark.

This generalizes ``consolidation.read_transcript_messages`` (which reads only the
last 60 messages); the 60-message cap silently dropped the early turns of long
sessions, which this fixes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Each message is truncated to this many characters before extraction.
DEFAULT_PER_MESSAGE_CHARS = 1500

# Turns with fewer than this many characters of text are dropped as sub-threshold
# noise (e.g. a bare "ok"). Tool-only turns flatten to empty text and are dropped
# regardless.
MIN_TURN_CHARS = 2


@dataclass
class Window:
    """The new turns in ``(watermark, now]`` plus the watermark to persist next."""

    messages: list[dict[str, str]] = field(default_factory=list)
    new_watermark: int = 0
    total_turns: int = 0

    def __bool__(self) -> bool:
        return bool(self.messages)


def _extract_text(content: Any) -> str:
    """Flatten Anthropic message content (string or block list) to plain text.

    Keeps only ``text`` blocks; thinking / tool_use / tool_result are dropped so
    a tool-only turn flattens to empty and is filtered out by the caller.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(p for p in parts if p)


def _is_compact_summary(row: dict) -> bool:
    """True for an entry that is a compaction summary, not a real turn."""
    if row.get("isCompactSummary") or row.get("isCompactBoundary"):
        return True
    if row.get("subtype") in ("compact_boundary", "compact_summary"):
        return True
    msg = row.get("message")
    return isinstance(msg, dict) and bool(msg.get("isCompactSummary"))


def read_turns(
    path: Path,
    *,
    per_message_chars: int = DEFAULT_PER_MESSAGE_CHARS,
    min_chars: int = MIN_TURN_CHARS,
) -> list[dict[str, str]]:
    """All qualifying turns in the transcript, oldest first.

    Skips compact-summary entries and non-user/assistant rows, flattens text,
    drops tool-only and sub-threshold turns, and truncates each message.
    """
    if not path.exists():
        return []

    turns: list[dict[str, str]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if row.get("type") not in ("user", "assistant"):
                continue
            if _is_compact_summary(row):
                continue
            msg = row.get("message") or {}
            role = msg.get("role")
            if role not in ("user", "assistant"):
                continue
            text = _extract_text(msg.get("content")).strip()
            if len(text) < min_chars:  # tool-only (empty) and sub-threshold turns
                continue
            if len(text) > per_message_chars:
                text = text[:per_message_chars].rstrip() + "…"
            turns.append({"role": role, "text": text})
    return turns


def read_window(
    path: Path,
    *,
    watermark: int = 0,
    per_message_chars: int = DEFAULT_PER_MESSAGE_CHARS,
    min_chars: int = MIN_TURN_CHARS,
) -> Window:
    """The capture window ``(watermark, now]`` over the transcript's turns.

    ``new_watermark`` is the total qualifying-turn count — persist it after a
    successful ingest. If ``watermark`` is at or beyond the turns present (a
    re-fire, or post-compaction shrink), the window is empty and ``new_watermark``
    clamps to what remains.
    """
    turns = read_turns(path, per_message_chars=per_message_chars, min_chars=min_chars)
    total = len(turns)
    start = max(0, watermark)
    if start >= total:
        return Window(messages=[], new_watermark=total, total_turns=total)
    return Window(messages=turns[start:], new_watermark=total, total_turns=total)
