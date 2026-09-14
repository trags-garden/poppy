"""TranscriptWindow — incremental ``(watermark, now]`` windowing (ADR-0001).

The ADR-0001 capture-window machinery, decoupled from transcript parsing:
it takes the qualifying turns a reader produced (see ``capture.transcript``) and
returns the window ``(watermark, now]`` plus the watermark to persist next. JSONL
format knowledge lives in ``capture.transcript``; this file owns only the
windowing/clamping, so the two are testable apart.

A "turn" is one qualifying (text-bearing) user/assistant message. The watermark
indexes these qualifying turns, so the window ``(watermark, now]`` is exactly the
turns added since the last capture. The reader **clamps** to the turns actually
present: if compaction shrank the transcript below the watermark, the window is
empty and the new watermark resets down to what remains — compaction can never
desync the watermark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from poppy.capture.transcript import (
    DEFAULT_PER_MESSAGE_CHARS,
    MIN_TURN_CHARS,
    Turn,
    read_transcript_turns,
)


@dataclass
class Window:
    """The new turns in ``(watermark, now]`` plus the watermark to persist next."""

    messages: list[Turn] = field(default_factory=list)
    new_watermark: int = 0
    total_turns: int = 0

    def __bool__(self) -> bool:
        return bool(self.messages)


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
    # Adapter dispatch uses the transcript's record shape: Codex has a leading
    # session_meta row, while Claude Code uses top-level user/assistant rows.
    # This supports CODEX_HOME overrides without guessing from directory names.
    turns = read_transcript_turns(path, per_message_chars=per_message_chars, min_chars=min_chars)
    total = len(turns)
    start = max(0, watermark)
    if start >= total:
        return Window(messages=[], new_watermark=total, total_turns=total)
    return Window(messages=turns[start:], new_watermark=total, total_turns=total)
