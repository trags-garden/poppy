"""End-of-session consolidation: read transcript → LLM extracts durable memories.

Two backends, tried in order:

1. **Host CLI subprocess** (default, zero-config): shells out to the same
   coding-agent CLI that ran the session — `claude -p` for Claude Code,
   `codex exec` for Codex, `gemini -p` for Gemini CLI. Uses the user's
   existing login. No API key needed; cost lands on their existing
   subscription.

2. **OpenAI-compatible HTTP** (fallback): pulls model + base URL + api_key
   from ~/.poppy/config.json (preferred) or POPPY_CONSOLIDATE_* env vars
   (override). Useful when the user prefers a specific cheap model
   (Ollama Cloud kimi-k2.6, glm-5.1, etc.) or runs somewhere without
   a host CLI on PATH.

Either way, nothing runs until the user records a one-time consent
(`poppy consent --enable`, or the prompt during `poppy setup claude-code`).
The full precedence lives in ``poppy.capture.policy`` (ADR-0002);
``POPPY_CONSOLIDATE`` remains an explicit on/off override.
"""

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from poppy.capture import journal
from poppy.capture.cadence import record_capture, soft_cap_reached
from poppy.capture.lock import single_flight
from poppy.capture.policy import is_capture_enabled
from poppy.capture.reconciler import reconcile_and_ingest
from poppy.capture.watermark import get_watermark, set_watermark
from poppy.capture.window import read_window
from poppy.config import PoppyConfig, load_config
from poppy.models import Filters, Memory, Source
from poppy.runtime import get_engine, get_poppy_dir

ALLOWED_TYPES = {"fact", "decision", "preference", "lesson"}

# Minimum qualifying turns in a SessionEnd backstop window before a consolidation
# pass is worthwhile. Preserves the pre-watermark gate; the mid-session loop adds
# its own minimum-content gate.
MIN_BACKSTOP_TURNS = 4

# Minimum characters of substantive new content in a mid-session capture window
# before a fire is worthwhile (minimum-content gate). A sane default,
# tuned later by autoresearch (ADR-0003).
MIN_CAPTURE_CHARS = 200

CONSOLIDATION_PROMPT = """You are a developer-memory consolidator. The transcript below is one coding-agent session.

Extract durable memories that would help in a future, unrelated session in this project. Each item is one sentence, self-contained, and would still be true in a week.

Output strict JSON: an array of objects with two keys:
  - "type": one of "fact", "decision", "preference", "lesson"
  - "content": one sentence

Skip:
  - Anything specific to today's task that won't matter tomorrow.
  - Things obvious from the codebase, git history, or already-stored memories.
  - Routine status updates ("ran tests", "committed").

If nothing durable: return [].
Cap: {max_items} items.

Respond with ONLY the JSON array, no prose, no fences.

Transcript:
{transcript}
"""


def _make_id() -> str:
    return f"mem_{uuid.uuid4().hex[:12]}"


def _build_memories(
    items: list[dict[str, str]],
    *,
    source_type: str,
    session_id: str | None,
    project: str | None,
) -> list[Memory]:
    """Turn extracted {type, content} dicts into Memory candidates with provenance.

    Provenance (source app + session id) is set here so every captured memory is
    auditable. The candidates are not written directly — they go through
    the CaptureReconciler (ADR-0003) before ingest.
    """
    now = datetime.datetime.now(datetime.UTC)
    return [
        Memory(
            id=_make_id(),
            content=item["content"],
            memory_type=item["type"],
            source=Source(type=source_type, session_id=session_id, timestamp=now),
            project=project,
            related_to=[],
            created_at=now,
            updated_at=now,
            confidence=0.7,
        )
        for item in items
    ]


def _project_from_cwd(cwd: str | None) -> str | None:
    if not cwd:
        return None
    return Path(cwd).name or None


def is_enabled(cfg: PoppyConfig | None = None) -> bool:
    """Whether auto-capture should run, per the ConsolidationPolicy.

    Delegates to ``capture.policy``: default-on once consent is recorded and a
    free host-CLI backend is present; inert until consent; never auto-spends on a
    remote-only backend. ``POPPY_CONSOLIDATE`` remains an explicit on/off override.
    """
    cfg = cfg if cfg is not None else load_config(get_poppy_dir())
    return is_capture_enabled(cfg)


# ---------------------------------------------------------------------------
# Transcript I/O
# ---------------------------------------------------------------------------


def read_transcript_messages(path: Path, max_messages: int = 60) -> list[dict[str, str]]:
    """Read a Claude Code transcript JSONL into a list of {role, text} pairs."""
    if not path.exists():
        return []

    messages: list[dict[str, str]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") not in ("user", "assistant"):
                continue
            msg = row.get("message") or {}
            role = msg.get("role")
            text = _extract_text(msg.get("content"))
            if not text:
                continue
            messages.append({"role": role, "text": text})
    return messages[-max_messages:]


def _extract_text(content: Any) -> str:
    """Pull plain text out of Anthropic message content (string or block list)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        # skip thinking, tool_use, tool_result — too noisy for consolidation
    return "\n".join(p for p in parts if p)


def format_transcript(messages: list[dict[str, str]], char_budget: int = 16000) -> str:
    """Render messages into a compact transcript respecting a char budget."""
    lines: list[str] = []
    used = 0
    for msg in messages:
        role = msg["role"].upper()
        text = msg["text"].strip()
        if len(text) > 1500:
            text = text[:1500].rstrip() + "…"
        block = f"{role}: {text}"
        if used + len(block) > char_budget:
            break
        lines.append(block)
        used += len(block) + 1
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Backend 1: Host CLI subprocess
# ---------------------------------------------------------------------------


def detect_host_cli(transcript_path: str | None) -> str | None:
    """Pick the host coding-agent CLI based on the transcript path.

    Returns the executable name if it exists on PATH, else None.
    """
    if not transcript_path:
        return None
    # Claude Code stores transcripts under ~/.claude/projects/<slug>/
    if "/.claude/projects/" in transcript_path and shutil.which("claude"):
        return "claude"
    # Codex stores transcripts under ~/.codex/...
    if "/.codex/" in transcript_path and shutil.which("codex"):
        return "codex"
    # Gemini CLI stores transcripts under ~/.gemini/...
    if "/.gemini/" in transcript_path and shutil.which("gemini"):
        return "gemini"
    # Last resort — if claude is on PATH, use it.
    if shutil.which("claude"):
        return "claude"
    return None


def call_host_cli(prompt: str, *, cli: str, timeout_s: int = 120) -> str | None:
    """Invoke a coding-agent CLI in headless mode. Returns the model's text."""
    if cli == "claude":
        # claude -p reads the prompt from stdin or as a positional arg.
        # --output-format text gives us the raw text response (no JSON envelope).
        cmd = ["claude", "-p", "--output-format", "text"]
    elif cli == "codex":
        # codex exec takes a prompt arg; output is plain text.
        cmd = ["codex", "exec", prompt]
    elif cli == "gemini":
        cmd = ["gemini", "-p"]
    else:
        sys.stderr.write(f"poppy consolidate: unknown cli {cli!r}\n")
        return None

    try:
        if cli == "codex":
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        else:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"poppy consolidate: {cli} timed out after {timeout_s}s\n")
        return None
    except OSError as exc:
        sys.stderr.write(f"poppy consolidate: {cli} spawn failed: {exc}\n")
        return None
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip().splitlines()[-3:]
        sys.stderr.write(
            f"poppy consolidate: {cli} exited rc={proc.returncode} stderr_tail={' | '.join(stderr_tail)!r}\n"
        )
        return None
    return proc.stdout or None


# ---------------------------------------------------------------------------
# Backend 2: OpenAI-compatible HTTP
# ---------------------------------------------------------------------------


def call_openai_compat(
    prompt: str,
    *,
    model: str,
    base_url: str | None,
    api_key: str,
    max_tokens: int = 800,
) -> str | None:
    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(base_url=base_url, api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens,
        )
    except Exception:
        return None
    return resp.choices[0].message.content


def _resolved_openai_settings(cfg: PoppyConfig) -> tuple[str | None, str | None, str | None]:
    """(model, base_url, api_key) — env vars override config when set."""
    model = os.environ.get("POPPY_CONSOLIDATE_MODEL") or cfg.consolidate_model
    base_url = os.environ.get("POPPY_CONSOLIDATE_BASE_URL") or cfg.consolidate_base_url
    api_key = os.environ.get("POPPY_CONSOLIDATE_API_KEY") or cfg.consolidate_api_key or os.environ.get("OPENAI_API_KEY")
    return model, base_url, api_key


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_json_array(text: str) -> list[dict[str, str]]:
    """Tolerant JSON parser. Strips ```json fences, finds the first [...] array."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        arr = json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(arr, list):
        return []
    out: list[dict[str, str]] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        mtype = item.get("type", "fact")
        content = (item.get("content") or "").strip()
        if not content:
            continue
        if mtype not in ALLOWED_TYPES:
            mtype = "fact"
        out.append({"type": mtype, "content": content})
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def call_llm(prompt: str, *, transcript_path: str | None, cfg: PoppyConfig) -> list[dict[str, str]]:
    """Pick a backend, run it, parse the JSON array.

    Tries host CLI first (free, no key). Falls back to OpenAI-compat using
    config / env. Logs a one-line classification of every empty result so
    silent zeros are diagnosable.
    """
    cli = detect_host_cli(transcript_path)
    if cli:
        text = call_host_cli(prompt, cli=cli)
        if text is None:
            # call_host_cli already logged the failure mode (timeout / rc / OSError).
            pass
        else:
            parsed = parse_json_array(text)
            if parsed:
                return parsed
            stripped = text.strip()
            if stripped == "[]":
                sys.stderr.write(f"poppy consolidate: {cli} returned empty array (nothing durable)\n")
            else:
                snippet = stripped[:200].replace("\n", " ")
                sys.stderr.write(
                    f"poppy consolidate: {cli} output unparseable as JSON array "
                    f"(len={len(text)}, first 200: {snippet!r})\n"
                )

    model, base_url, api_key = _resolved_openai_settings(cfg)
    if model and api_key:
        text = call_openai_compat(prompt, model=model, base_url=base_url, api_key=api_key)
        if text:
            parsed = parse_json_array(text)
            if not parsed:
                snippet = text.strip()[:200].replace("\n", " ")
                sys.stderr.write(f"poppy consolidate: openai-compat output yielded 0 items (first 200: {snippet!r})\n")
            return parsed
        sys.stderr.write("poppy consolidate: openai-compat returned no text\n")
    elif not cli:
        sys.stderr.write("poppy consolidate: no host cli detected and no openai-compat fallback configured\n")
    return []


def consolidate_stop_event(payload: dict) -> int:
    """Run consolidation for a Stop hook payload. Returns the number of memories stored.

    No-op (returns 0) if disabled, transcript missing, or session already consolidated.
    """
    cfg = load_config(get_poppy_dir())
    if not is_enabled(cfg):
        return 0

    session_id = payload.get("session_id")
    transcript_path = payload.get("transcript_path")
    if not session_id or not transcript_path:
        return 0

    project = _project_from_cwd(payload.get("cwd"))
    engine = get_engine(get_poppy_dir())
    poppy_dir = get_poppy_dir()

    # Incremental capture (ADR-0001): read only the window (watermark, end]. The
    # watermark is the idempotency key — a re-fired hook finds an empty window and
    # no-ops — and it lets the backstop flush only the tail the mid-session loop
    # has not already captured, instead of re-reading the whole session.
    watermark = get_watermark(poppy_dir, session_id)
    window = read_window(Path(transcript_path), watermark=watermark)
    if len(window.messages) < MIN_BACKSTOP_TURNS:
        return 0

    transcript = format_transcript(window.messages)
    if not transcript.strip():
        return 0

    max_items = int(os.environ.get("POPPY_CONSOLIDATE_MAX_ITEMS", "5"))
    prompt = CONSOLIDATION_PROMPT.format(max_items=max_items, transcript=transcript)
    extracted = call_llm(prompt, transcript_path=transcript_path, cfg=cfg)
    if not extracted:
        return 0

    candidates = _build_memories(
        extracted[:max_items], source_type="claude-code", session_id=session_id, project=project
    )
    summary = reconcile_and_ingest(candidates, engine=engine, cfg=cfg, poppy_dir=poppy_dir)
    # Advance the watermark only after a successful extract+ingest pass (ADR-0001);
    # a failed extraction leaves the turns for the next fire / backstop to re-cover.
    set_watermark(poppy_dir, session_id, window.new_watermark)
    # Journal the backstop capture so the SessionStart banner / `poppy doctor`
    # see it too — the mid-session loop already journals; the backstop must as
    # well or a short session's only capture is invisible.
    journal.record(poppy_dir, session_id=session_id, project=project, count=summary.stored, items=candidates)
    return summary.stored


def _window_substance(messages: list[dict[str, str]]) -> int:
    return sum(len(m.get("text", "")) for m in messages)


def consolidate_capture_event(payload: dict) -> int:
    """Mid-session capture (ADR-0001). Returns the number of memories stored.

    Fired every Nth turn by the UserPromptSubmit hook via a detached worker. It
    processes only the window ``(watermark, now]`` under a per-session
    single-flight lock, skips thin windows (minimum-content gate), and stops once
    the per-session soft cap is hit — the SessionEnd backstop flushes the rest.
    No-op (returns 0) when consolidation is disabled or no backend/transcript is
    available, so it never blocks the developer's prompt.
    """
    cfg = load_config(get_poppy_dir())
    if not is_enabled(cfg):
        return 0

    session_id = payload.get("session_id")
    transcript_path = payload.get("transcript_path")
    if not session_id or not transcript_path:
        return 0

    poppy_dir = get_poppy_dir()
    with single_flight(poppy_dir, session_id) as acquired:
        if not acquired:
            return 0  # another capture worker is already running for this session
        if soft_cap_reached(poppy_dir, session_id):
            return 0  # defer the remainder to the SessionEnd backstop

        watermark = get_watermark(poppy_dir, session_id)
        window = read_window(Path(transcript_path), watermark=watermark)
        if _window_substance(window.messages) < MIN_CAPTURE_CHARS:
            return 0  # minimum-content gate — too little new content to be worth a fire

        transcript = format_transcript(window.messages)
        if not transcript.strip():
            return 0

        project = _project_from_cwd(payload.get("cwd"))
        engine = get_engine(poppy_dir)
        max_items = int(os.environ.get("POPPY_CONSOLIDATE_MAX_ITEMS", "5"))
        prompt = CONSOLIDATION_PROMPT.format(max_items=max_items, transcript=transcript)
        extracted = call_llm(prompt, transcript_path=transcript_path, cfg=cfg)
        if not extracted:
            return 0

        candidates = _build_memories(
            extracted[:max_items], source_type="claude-code", session_id=session_id, project=project
        )
        summary = reconcile_and_ingest(candidates, engine=engine, cfg=cfg, poppy_dir=poppy_dir)
        # Advance the watermark so no turn is captured twice, count the capture
        # against the soft cap, and journal what was stored (provenance/banner).
        set_watermark(poppy_dir, session_id, window.new_watermark)
        record_capture(poppy_dir, session_id)
        journal.record(poppy_dir, session_id=session_id, project=project, count=summary.stored, items=candidates)
        return summary.stored


COMPACT_CONSOLIDATION_PROMPT = """You are a developer-memory consolidator. The text below is Claude Code's auto-generated summary of a long coding session that just hit context-compaction.

Extract durable memories that would help in a future, unrelated session in this project. Each item is one sentence, self-contained, and would still be true in a week.

Output strict JSON: an array of objects with two keys:
  - "type": one of "fact", "decision", "preference", "lesson"
  - "content": one sentence

Skip:
  - Anything specific to today's task that won't matter tomorrow.
  - Things obvious from the codebase, git history, or already-stored memories.
  - Routine status updates ("ran tests", "committed").

If nothing durable: return [].
Cap: {max_items} items.

Respond with ONLY the JSON array, no prose, no fences.

Compact summary:
{summary}
"""


def _compact_event_id(session_id: str, summary: str) -> str:
    """Composite session_id for a single PostCompact fire.

    A session can compact multiple times. Tagging stored memories with just
    `session_id` would only let us consolidate once. We hash the summary so
    each distinct compact event gets its own idempotency key, but a re-fire
    of the *same* event (rare, but possible) collapses cleanly.
    """
    digest = hashlib.sha1(summary.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{session_id}:compact:{digest}"


def consolidate_compact_event(payload: dict) -> int:
    """Run consolidation for a PostCompact hook payload. Returns memories stored.

    PostCompact hands us a pre-distilled `compact_summary` written by the
    host LLM itself — far cleaner than the raw transcript. We feed that
    summary to the consolidator instead of replaying the JSONL.

    Idempotency keys off `session_id + sha1(summary)` so multiple compacts
    in the same session all consolidate, but a re-fire of the same event
    is a no-op.
    """
    cfg = load_config(get_poppy_dir())
    if not is_enabled(cfg):
        return 0

    session_id = payload.get("session_id")
    summary = (payload.get("compact_summary") or payload.get("summary") or "").strip()
    if not session_id or not summary:
        return 0

    event_id = _compact_event_id(session_id, summary)
    project = _project_from_cwd(payload.get("cwd"))
    engine = get_engine(get_poppy_dir())

    # Idempotency: bail if this exact compact event was already consolidated.
    existing = engine.list_all(filters=Filters(project=project), limit=200)
    if any(m.source.session_id == event_id for m in existing):
        return 0

    max_items = int(os.environ.get("POPPY_CONSOLIDATE_MAX_ITEMS", "5"))
    # Cap summary input — compact_summary can run several KB; keep prompt small.
    capped = summary[:12000]
    prompt = COMPACT_CONSOLIDATION_PROMPT.format(max_items=max_items, summary=capped)
    extracted = call_llm(prompt, transcript_path=payload.get("transcript_path"), cfg=cfg)
    if not extracted:
        return 0

    candidates = _build_memories(extracted[:max_items], source_type="claude-code", session_id=event_id, project=project)
    poppy_dir = get_poppy_dir()
    reconciled = reconcile_and_ingest(candidates, engine=engine, cfg=cfg, poppy_dir=poppy_dir)
    journal.record(poppy_dir, session_id=event_id, project=project, count=reconciled.stored, items=candidates)
    return reconciled.stored
