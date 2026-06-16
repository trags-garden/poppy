"""TurnCadence — per-session turn counter + cadence gate (ADR-0001).

The mid-session loop fires a capture every Nth user turn. This module owns the
per-session turn counter and the cadence / soft-cap gates. State lives in the
shared per-session file (``_state``); the counters reset at SessionStart via
``watermark.reset_session`` so cadence and coverage are predictable per session
and concurrent sessions never interfere.

Defaults (N / K) are sane starting points; ADR-0003's autoresearch pass tunes
them later.
"""

from __future__ import annotations

from pathlib import Path

from poppy.capture import _state

# Fire a capture every Nth user turn.
DEFAULT_CADENCE_N = 3

# Per-session soft cap: after K mid-session captures, stop firing and let the
# SessionEnd backstop flush the remainder (defers, never drops).
DEFAULT_SOFT_CAP_K = 20


def register_turn(poppy_dir: Path, session_id: str) -> int:
    """Increment and return this session's turn count."""
    data = _state.load(poppy_dir)
    entry = data.setdefault(session_id, {})
    try:
        turns = int(entry.get("turns", 0))
    except (TypeError, ValueError):
        turns = 0
    turns += 1
    entry["turns"] = turns
    _state.save(poppy_dir, data)
    return turns


def should_capture(count: int, *, n: int = DEFAULT_CADENCE_N) -> bool:
    """True on every Nth turn (3, 6, 9, ... for n=3)."""
    return count > 0 and n > 0 and count % n == 0


def capture_count(poppy_dir: Path, session_id: str) -> int:
    """How many mid-session captures have fired for this session."""
    entry = _state.load(poppy_dir).get(session_id, {})
    try:
        return int(entry.get("captures", 0))
    except (TypeError, ValueError):
        return 0


def soft_cap_reached(poppy_dir: Path, session_id: str, *, k: int = DEFAULT_SOFT_CAP_K) -> bool:
    """True once the per-session soft cap of mid-session captures is hit."""
    return capture_count(poppy_dir, session_id) >= k


def record_capture(poppy_dir: Path, session_id: str) -> int:
    """Count one completed mid-session capture; returns the new total."""
    data = _state.load(poppy_dir)
    entry = data.setdefault(session_id, {})
    try:
        captures = int(entry.get("captures", 0))
    except (TypeError, ValueError):
        captures = 0
    captures += 1
    entry["captures"] = captures
    _state.save(poppy_dir, data)
    return captures
