"""Transcript reader seam: session ref in → turns out, one adapter per source app.

The **Source app** is the coding agent a memory came from (the real client:
``claude-code``, ``cursor``, ``codex``, …), never the transport.
This module is the seam behind that term for *reading* a session transcript into
turns. Adding a new client's transcript support is one adapter here, not a hunt
across ``window.py`` + ``detect_host_cli`` + hardcoded source literals — the
direct enabler for the non-Claude-Code fast-follows.

A **turn** is one qualifying (text-bearing) user/assistant message,
``{"role", "text"}``. The publish tree ships three adapters:

* :func:`read_claude_code_turns` — the Claude Code JSONL transcript (the live
  capture path). The incremental ``(watermark, now]`` windowing in
  ``capture.window`` is layered on top of these turns, so JSONL parsing and the
  ADR-0001 watermark-after-success window machinery is now testable apart.

* :func:`read_codex_turns` — Codex rollout JSONL, identified by its leading
  ``session_meta`` record.

* :func:`read_cursor_turns` — Cursor agent transcript JSONL, identified by its
  top-level role and nested message content.

:func:`extract_text` (Anthropic string-or-block content → plain text) is shared,
replacing the two copies that used to live in ``window.py`` and ``consolidation.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# A parsed transcript turn: {"role": "user"|"assistant", "text": <plain text>}.
Turn = dict[str, str]

# Each Claude-Code turn is truncated to this many characters at parse time.
DEFAULT_PER_MESSAGE_CHARS = 1500

# Turns with fewer than this many characters of text are dropped as sub-threshold
# noise (e.g. a bare "ok"). Tool-only turns flatten to empty text and are dropped
# regardless.
MIN_TURN_CHARS = 2


def extract_text(content: Any) -> str:
    """Flatten Anthropic message content (string or block list) to plain text.

    Keeps only ``text`` blocks; thinking / tool_use / tool_result are dropped so a
    tool-only turn flattens to empty and is filtered out by the caller.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Adapter: Claude Code JSONL (the live capture transcript)
# ---------------------------------------------------------------------------


def _is_compact_summary(row: dict) -> bool:
    """True for an entry that is a compaction summary, not a real turn."""
    if row.get("isCompactSummary") or row.get("isCompactBoundary"):
        return True
    if row.get("subtype") in ("compact_boundary", "compact_summary"):
        return True
    msg = row.get("message")
    return isinstance(msg, dict) and bool(msg.get("isCompactSummary"))


def read_claude_code_turns(
    path: Path,
    *,
    per_message_chars: int = DEFAULT_PER_MESSAGE_CHARS,
    min_chars: int = MIN_TURN_CHARS,
) -> list[Turn]:
    """All qualifying turns in a Claude Code JSONL transcript, oldest first.

    Skips compact-summary entries and non-user/assistant rows, flattens text,
    drops tool-only and sub-threshold turns, and truncates each message. The
    ``(watermark, now]`` windowing lives in ``capture.window`` on top of this.
    """
    if not path.exists():
        return []

    turns: list[Turn] = []
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
            text = extract_text(msg.get("content")).strip()
            if len(text) < min_chars:  # tool-only (empty) and sub-threshold turns
                continue
            if len(text) > per_message_chars:
                text = text[:per_message_chars].rstrip() + "…"
            turns.append({"role": role, "text": text})
    return turns


# ---------------------------------------------------------------------------
# Adapter: Codex rollout JSONL
# ---------------------------------------------------------------------------

_CODEX_SYNTHETIC_TAGS: dict[str, tuple[str, ...]] = {
    "developer": ("permissions", "multi_agent_mode", "apps_instructions", "skills_instructions"),
    "user": ("environment_context", "recommended_plugins", "current_date", "cwd", "shell", "timezone"),
}


def _codex_block_texts(content: Any) -> list[str]:
    """The text of each ``input_text`` / ``output_text`` block, in order.

    Returned per block (not pre-joined) so synthetic-context blocks can be
    dropped individually — concatenating first would let one synthetic block
    poison a whole multi-block message, or a real block mask a synthetic one.
    """
    if not isinstance(content, list):
        return []
    return [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") in ("input_text", "output_text")
    ]


def _is_codex_synthetic(role: str, text: str) -> bool:
    stripped = text.lstrip()
    return any(stripped.startswith(f"<{tag}>") for tag in _CODEX_SYNTHETIC_TAGS.get(role, ()))


def detect_transcript_host(path: Path) -> str | None:
    """Identify a supported transcript by its records, never its directory name.

    Codex rollouts start with a ``session_meta`` envelope. Claude Code rows use
    top-level ``user`` / ``assistant`` types. Cursor rows instead carry a
    top-level role and a nested message with content. Unknown and malformed rows
    are skipped so future metadata additions do not break dispatch.
    """
    if not path.exists():
        return None
    with path.open() as fh:
        for line in fh:
            # A UTF-8 BOM on the first record would make json.loads fail and the
            # codex session_meta go unrecognized, silently misrouting the whole
            # rollout to the Claude reader (zero capture). Strip it before shape
            # detection; detection re-runs on every read, so a partially flushed
            # session_meta still recovers on the next fire.
            line = line.lstrip("\ufeff")
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if row.get("type") == "session_meta":
                return "codex"
            if row.get("type") in ("user", "assistant"):
                return "claude-code"
            if row.get("role") in ("user", "assistant"):
                message = row.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), list):
                    return "cursor"
    return None


def read_codex_turns(
    path: Path,
    *,
    per_message_chars: int = DEFAULT_PER_MESSAGE_CHARS,
    min_chars: int = MIN_TURN_CHARS,
) -> list[Turn]:
    """All qualifying message response items in a Codex rollout, oldest first."""
    if not path.exists():
        return []

    turns: list[Turn] = []
    with path.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or row.get("type") != "response_item":
                continue
            payload = row.get("payload")
            if not isinstance(payload, dict) or payload.get("type") != "message":
                continue
            role = payload.get("role")
            # Restrict to user/assistant, matching the Claude Code reader — a
            # "developer" turn is injected instruction content, never a memory.
            if role not in ("user", "assistant"):
                continue
            # Drop synthetic-context blocks individually and keep the rest in
            # order; a message left with no real blocks flattens to empty and is
            # filtered out below.
            kept = [b for b in _codex_block_texts(payload.get("content")) if b and not _is_codex_synthetic(role, b)]
            text = "\n".join(kept).strip()
            if len(text) < min_chars:
                continue
            if len(text) > per_message_chars:
                text = text[:per_message_chars].rstrip() + "…"
            turns.append({"role": role, "text": text})
    return turns


# ---------------------------------------------------------------------------
# Adapter: Cursor agent transcript JSONL
# ---------------------------------------------------------------------------

_CURSOR_USER_QUERY = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.DOTALL)
_CURSOR_TIMESTAMP = re.compile(r"<timestamp>.*?</timestamp>\s*", re.DOTALL)


def _cursor_text_blocks(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []
    return [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
    ]


def read_cursor_turns(
    path: Path,
    *,
    per_message_chars: int = DEFAULT_PER_MESSAGE_CHARS,
    min_chars: int = MIN_TURN_CHARS,
) -> list[Turn]:
    """All qualifying user and assistant turns in a Cursor transcript."""
    if not path.exists():
        return []

    turns: list[Turn] = []
    with path.open() as fh:
        for line in fh:
            line = line.lstrip("\ufeff").strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            role = row.get("role")
            if role not in ("user", "assistant"):
                continue
            message = row.get("message")
            if not isinstance(message, dict):
                continue

            blocks = _cursor_text_blocks(message.get("content"))
            if role == "user":
                cleaned: list[str] = []
                for block in blocks:
                    match = _CURSOR_USER_QUERY.search(block)
                    text = match.group(1) if match else _CURSOR_TIMESTAMP.sub("", block)
                    if text.strip():
                        cleaned.append(text.strip())
            else:
                # Cursor uses [REDACTED] as a stripped-thinking marker. Only a
                # marker-only block is synthetic; the literal may be real answer text.
                cleaned = [block.strip() for block in blocks if block.strip() != "[REDACTED]"]

            text = "\n".join(cleaned).strip()
            if len(text) < min_chars:
                continue
            if len(text) > per_message_chars:
                text = text[:per_message_chars].rstrip() + "…"
            turns.append({"role": role, "text": text})
    return turns


def read_transcript_turns(
    path: Path,
    *,
    per_message_chars: int = DEFAULT_PER_MESSAGE_CHARS,
    min_chars: int = MIN_TURN_CHARS,
) -> list[Turn]:
    """Dispatch to the adapter selected from the transcript's record shape."""
    host = detect_transcript_host(path)
    if host == "codex":
        return read_codex_turns(path, per_message_chars=per_message_chars, min_chars=min_chars)
    if host == "cursor":
        return read_cursor_turns(path, per_message_chars=per_message_chars, min_chars=min_chars)
    return read_claude_code_turns(path, per_message_chars=per_message_chars, min_chars=min_chars)
