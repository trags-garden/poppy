"""Per-session capture watermark (ADR-0001).

The capture watermark is the index of the last transcript turn already captured
for a session. Every capture — each mid-session fire and the SessionEnd backstop
— reads only the window ``(watermark, now]`` and advances the watermark on a
successful ingest, so no turn is ever extracted twice.

It is per-session (keyed by session id) and resets at SessionStart, so cadence
and coverage are predictable per session. Distinct from the per-device **sync**
watermark in ``poppy.sync`` — different layer, different lifetime.
"""

from __future__ import annotations

from pathlib import Path

from poppy.capture import _state


def get_watermark(poppy_dir: Path, session_id: str) -> int:
    """The last captured turn index for a session (0 if never captured)."""
    entry = _state.load(poppy_dir).get(session_id, {})
    try:
        return max(0, int(entry.get("watermark", 0)))
    except (TypeError, ValueError):
        return 0


def set_watermark(poppy_dir: Path, session_id: str, value: int) -> None:
    """Advance (or clamp) the watermark for a session."""
    data = _state.load(poppy_dir)
    entry = data.setdefault(session_id, {})
    entry["watermark"] = max(0, int(value))
    _state.save(poppy_dir, data)


def reset_session(poppy_dir: Path, session_id: str) -> None:
    """Reset a session's capture state at SessionStart (watermark + cadence)."""
    if not session_id:
        return
    data = _state.load(poppy_dir)
    data[session_id] = {"watermark": 0, "turns": 0, "captures": 0}
    _state.save(poppy_dir, data)
