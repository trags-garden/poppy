"""Claude Code, Cursor, and Codex hook entrypoints.

Each hook reads JSON from stdin and writes
either nothing (silent exit 0) or a JSON envelope on stdout. Hooks must be
fast and fail open — never block the agent on a Poppy error.
"""

import json
import os
import re
import sys
from pathlib import Path

import click

from poppy.engine.interface import RetrievalEngine
from poppy.models import Filters, ScoredMemory
from poppy.paths import ensure_poppy_dir
from poppy.project import project_from_cwd
from poppy.runtime import get_fast_engine, get_poppy_dir
from poppy.sources import IMPORT_SOURCE_TYPES

# Tiny stopword set — just enough to skip the worst noise without over-aggressive
# filtering. The engine ranks anyway; this only keeps us from issuing useless
# queries like "the" or "what".
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "what",
        "when",
        "how",
        "but",
        "are",
        "was",
        "were",
        "you",
        "have",
        "has",
        "had",
        "should",
        "could",
        "would",
        "from",
        "your",
        "use",
        "using",
        "make",
        "made",
        "any",
        "all",
        "not",
        "yes",
        "did",
        "does",
        "not",
        "now",
        "here",
        "there",
        "into",
        "out",
        "off",
        "yet",
    }
)


def _read_hook_input() -> dict:
    raw = sys.stdin.read()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return _normalize_hook_payload(payload)


def _normalize_hook_payload(payload: dict) -> dict:
    """Normalize Cursor's hook envelope to Poppy's existing hook fields."""
    if "cursor_version" not in payload:
        return payload

    normalized = dict(payload)
    normalized["_poppy_host"] = "cursor"
    if not normalized.get("cwd"):
        roots = normalized.get("workspace_roots")
        if isinstance(roots, list) and roots and isinstance(roots[0], str):
            normalized["cwd"] = roots[0]
    if not normalized.get("transcript_path"):
        transcript_path = os.environ.get("CURSOR_TRANSCRIPT_PATH")
        if transcript_path:
            normalized["transcript_path"] = transcript_path
    return normalized


def _payload_host(payload: dict) -> str | None:
    """Detect a hook host from reliable payload fields, then transcript shape."""
    if payload.get("_poppy_host") == "cursor" or "cursor_version" in payload:
        return "cursor"
    transcript_path = payload.get("transcript_path")
    if transcript_path:
        try:
            from poppy.capture.transcript import detect_transcript_host

            return detect_transcript_host(Path(transcript_path))
        except Exception:
            pass
    return None


def _truncate(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[:n].rstrip() + "…"


# SessionStart injects a snapshot of the project's memories. Unlike the per-turn
# hooks there is no prompt to rank against, so selection is deliberately by
# recency: list_all returns created_at DESC, so the newest memories lead. Two
# caps keep the block bounded no matter how large the store or how long an
# individual memory is — without them a single long memory blew the block up to
# ~17 KB. Each memory is truncated to _SESSION_START_MEMORY_CHARS, and
# the memory lines together are held under _SESSION_START_BLOCK_BUDGET bytes;
# when memories are dropped to fit, one short omission marker is appended past
# that, so the true ceiling is the budget plus a single ~90-byte marker line.
# The budget is sized so a normal run of short memories all fit and only
# pathological ones trigger truncation.
_SESSION_START_MAX_MEMORIES = 15
_SESSION_START_MEMORY_CHARS = 300
_SESSION_START_BLOCK_BUDGET = 2048  # bytes (~500 tokens)


def _render_session_memories(memories: list, *, project: str | None) -> str | None:
    """Render the SessionStart memory block under a fixed byte budget.

    Memories arrive recency-first. Each line is truncated to
    _SESSION_START_MEMORY_CHARS; memory lines are added until one more would push
    them past _SESSION_START_BLOCK_BUDGET bytes. When any are dropped, a single
    omission marker is appended past the budget (so the block's true ceiling is
    the budget plus that one short line) so the reader knows content was dropped.
    At least one memory is always shown (the first is emitted even if it alone
    exceeds the budget, since it is already per-memory truncated). Returns None
    when there are no memories.
    """
    if not memories:
        return None
    header = "## Poppy memories for this project:" if project else "## Poppy memories:"
    lines = [header]
    used = len(header.encode("utf-8"))
    shown = 0
    for m in memories:
        line = f"- [{m.memory_type}] {_truncate(m.content, _SESSION_START_MEMORY_CHARS)}"
        cost = len(line.encode("utf-8")) + 1  # +1 for the joining newline
        if shown > 0 and used + cost > _SESSION_START_BLOCK_BUDGET:
            break
        lines.append(line)
        used += cost
        shown += 1
    omitted = len(memories) - shown
    if omitted > 0:
        lines.append(f"- …{omitted} more memories not shown (context budget)")
    return "\n".join(lines)


def _tokens(text: str, max_tokens: int = 6) -> list[str]:
    """Pull content tokens out of free-form text.

    Skips short tokens, stopwords, and dedupes. Used to issue multiple
    single-word FTS5 queries against SeedEngine, whose retrieve()
    phrase-wraps the input — multi-word queries never match otherwise.
    """
    raw = re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text.lower())
    out: list[str] = []
    seen: set[str] = set()
    for tok in raw:
        if tok in seen or tok in _STOPWORDS:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= max_tokens:
            break
    return out


def _multi_token_retrieve(
    engine: RetrievalEngine,
    queries: list[str],
    filters: Filters,
    limit: int,
    per_query_limit: int = 5,
) -> list[ScoredMemory]:
    """Run several single-token queries; round-robin-merge by engine rank.

    SeedEngine's `score` field is the inverse of BloomEngine's (lower means
    more relevant — it's 1/(1+|bm25_rank|)). Sorting on `score` is therefore
    not portable. Instead we treat each engine.retrieve()'s native ordering
    as authoritative and round-robin across the per-query result lists,
    deduping by memory id and stopping at ``limit``. Each query's top hit
    surfaces before any query's second hit.
    """
    per_query_results = [engine.retrieve(q, filters=filters, limit=per_query_limit) for q in queries]
    out: list[ScoredMemory] = []
    seen: set[str] = set()
    max_rank = max((len(r) for r in per_query_results), default=0)
    for rank in range(max_rank):
        for results in per_query_results:
            if rank >= len(results):
                continue
            r = results[rank]
            if r.memory.id in seen:
                continue
            seen.add(r.memory.id)
            out.append(r)
            if len(out) >= limit:
                return out
    return out


def _is_prior_session_autocapture(memory, current_session_id: str) -> bool:
    """Whether ``memory`` was auto-captured in a session other than the current one.

    Auto-captured memories carry the capturing session in ``source.session_id`` —
    the raw session id for mid-session/SessionEnd captures, or
    ``"<session>:compact:<hash>"`` for PostCompact captures. Manual writes
    (``remember`` and the MCP path) leave it ``None``. Bulk imports may carry a
    synthetic session id for provenance and are explicitly excluded. A memory
    whose token is the current session (or is prefixed by it, i.e. a compact
    capture from this same session) is not from a *prior* session, so it does not
    count toward the closed loop.
    """
    if memory.source.type in IMPORT_SOURCE_TYPES:
        return False
    sid = getattr(memory.source, "session_id", None)
    if not sid or not current_session_id:
        return False
    return sid != current_session_id and not sid.startswith(f"{current_session_id}:")


def _maybe_emit_closed_loop(poppy_dir: Path, session_id: str | None, memories: list, *, channel: str) -> None:
    """Emit the ``closed_loop`` activation event when the loop closes.

    The closed loop is the first time ≥1 injected memory is auto-captured AND from
    a prior session — the agent knew something the user never typed into a memory
    tool. Properties are content-free (counts + channel enum) and the event is
    latched once per device. Never raises — telemetry must not break a hook.
    """
    try:
        if not session_id or not memories:
            return
        loop_count = sum(1 for m in memories if _is_prior_session_autocapture(m, session_id))
        if loop_count == 0:
            return
        from poppy import telemetry

        telemetry.capture_once(
            poppy_dir,
            "closed_loop",
            {"channel": channel, "injected_count": len(memories), "loop_count": loop_count},
        )
    except Exception as exc:
        sys.stderr.write(f"poppy closed-loop telemetry error: {exc}\n")


@click.group()
def hook():
    """Claude Code hook entrypoints. Invoked by Poppy-installed hooks."""
    if os.environ.get("POPPY_SUPPRESS_HOOKS"):
        raise click.exceptions.Exit(0)


@hook.command("session-start")
def session_start():
    """SessionStart hook: status banner + a recency-ranked snapshot of this project's memories.

    Prepends a one-line trust banner: "Poppy active, N memories, M
    captured last session" when capture is running, a loud INACTIVE line when the
    install is partially broken (engine failed, no/remote-only backend), or a
    consent nudge when capture is pending. Deliberate off-states (opt-out /
    env-off) stay silent. The banner makes invisible background capture
    observable.

    The memory list below it is the newest ``_SESSION_START_MAX_MEMORIES``
    memories (there is no prompt to rank against at session start, so recency is
    the selection policy), rendered by ``_render_session_memories`` under a fixed
    per-memory and total byte budget so the block can never blow up the context.
    """
    try:
        payload = _read_hook_input()
        host = _payload_host(payload)
        cwd = payload.get("cwd")
        project = project_from_cwd(cwd)
        poppy_dir = get_poppy_dir()

        # Watermark reset is host-aware. Claude Code and Cursor reset on every SessionStart
        # (unchanged from origin/main). Codex appends to one rollout across
        # resumes under the same session id, so resetting mid-stream would
        # re-extract the whole appended tail — for Codex, only a fresh "startup"
        # resets; a resume, or a fire with no source, preserves the watermark
        # (a genuinely new Codex session has fresh state anyway, so preserving is
        # the safe default). The host is read from the transcript shape, never the
        # directory name; an unrecognized/empty transcript defaults to Claude Code.
        session_id = payload.get("session_id")
        if session_id:
            reset = host != "codex" or payload.get("source") == "startup"
            if reset:
                try:
                    from poppy.capture.watermark import reset_session

                    reset_session(poppy_dir, session_id)
                except Exception:
                    pass

        # Memory count + the rows to surface come from one query; the engine
        # failing to load is itself an INACTIVE signal (recall is down), so we
        # render a loud banner rather than failing open silently.
        engine_ok = True
        memory_count = 0
        memories: list = []
        try:
            engine = get_fast_engine(poppy_dir)
            filters = Filters(project=project) if project else Filters()
            scoped = engine.list_all(filters=filters, limit=10_000)
            memory_count = len(scoped)
            memories = scoped[:_SESSION_START_MAX_MEMORIES]
        except Exception as exc:
            engine_ok = False
            sys.stderr.write(f"poppy session-start engine error: {exc}\n")

        sections: list[str] = []
        banner, banner_user_facing, banner_status = _session_banner(
            poppy_dir, project=project, memory_count=memory_count, engine_ok=engine_ok
        )
        if banner:
            sections.append(banner)

        # Activation funnel: the consent-pending nudge is now human-visible,
        # so record that the developer was actually shown it — the first
        # funnel bottleneck. Once per device.
        try:
            from poppy.capture.policy import CaptureStatus

            if banner_status is CaptureStatus.INERT_PENDING:
                from poppy import telemetry

                telemetry.capture_once(poppy_dir, "consent_pending_shown", {"channel": "session_start"})
        except Exception as exc:
            sys.stderr.write(f"poppy session-start telemetry error: {exc}\n")

        # Relevance-floor gate: this dump is pure recency —
        # there is no query and no score to floor against — so when a SessionStart
        # floor is configured we abstain from injecting the unscored list rather
        # than pad the session with possibly-irrelevant memories. The banner still
        # shows. Default floor 0.0 keeps today's behaviour (inject the dump).
        inject_recency = True
        try:
            from poppy.config import load_config

            inject_recency = load_config(poppy_dir).session_start_min_score <= 0.0
        except Exception:
            inject_recency = True

        # When we do inject, the block is capped per-memory and under a total
        # byte budget so a large store can never blow up the context.
        if inject_recency:
            block = _render_session_memories(memories, project=project)
            if block:
                sections.append(block)
                # Activation: the closed loop is the magic moment — an
                # auto-captured memory from a PRIOR session is injected into this
                # one. Only fires when the snapshot is actually injected.
                if host != "cursor":
                    _maybe_emit_closed_loop(poppy_dir, session_id, memories, channel="session_start")

        if not sections:
            sys.exit(0)

        # Cursor has no probe-verified native SessionStart context channel. Do
        # not emit Claude Code's compatibility-gated nested envelope there.
        if host == "cursor":
            if banner and banner_user_facing:
                sys.stderr.write(f"{banner}\n")
            sys.exit(0)

        envelope: dict = {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": "\n\n".join(sections),
            }
        }
        # The consent-pending nudge and INACTIVE warnings are worthless if
        # they only reach the model context — the human never sees stdout, so a
        # founder sat in consent-pending for weeks. Surface those banners through the
        # hook `systemMessage` channel so Claude Code renders them to the developer;
        # additionalContext stays as the secondary, agent-facing copy.
        if banner and banner_user_facing:
            envelope["systemMessage"] = banner
        sys.stdout.write(json.dumps(envelope))
    except Exception as exc:
        sys.stderr.write(f"poppy session-start hook error: {exc}\n")
    sys.exit(0)


def _session_banner(
    poppy_dir: Path, *, project: str | None, memory_count: int, engine_ok: bool
) -> tuple[str | None, bool, object | None]:
    """Render the SessionStart status banner, whether it must reach the human, and the status.

    Returns ``(banner_text, user_facing, status)``. ``user_facing`` is True for the
    states the developer needs to actually *see* — consent-pending and the INACTIVE
    breakage cases — so the caller can also emit them via the hook
    ``systemMessage`` channel; the reassuring ACTIVE line is agent-context only. A
    ``None`` banner is never user-facing. ``status`` is the resolved
    ``CaptureStatus`` (or ``None`` on error) so the caller can attach funnel
    telemetry without re-evaluating the policy.

    Reads the capture status (ConsolidationPolicy) and the last-session capture
    count (CaptureJournal). Never raises — a banner failure must not break the
    session, so any error falls through to no banner.
    """
    try:
        from poppy.capture import health, journal
        from poppy.capture.banner import is_user_facing_status, render_banner
        from poppy.capture.policy import evaluate
        from poppy.config import load_config

        backend = health.load(poppy_dir)
        status = evaluate(load_config(poppy_dir), project=project, backend_broken=backend.failing)
        banner = render_banner(
            status,
            project=project,
            memory_count=memory_count,
            last_session_count=journal.last_session_count(poppy_dir),
            engine_ok=engine_ok,
            backend_cli=backend.cli,
        )
        # A failed engine renders a loud INACTIVE banner regardless of status, so
        # it is user-facing on its own; otherwise defer to the status predicate.
        user_facing = banner is not None and (not engine_ok or is_user_facing_status(status))
        return banner, user_facing, status
    except Exception as exc:
        sys.stderr.write(f"poppy session-start banner error: {exc}\n")
        return None, False, None


@hook.command("user-prompt-submit")
def user_prompt_submit():
    """UserPromptSubmit hook: deterministic per-turn semantic recall.

    Runs an FTS5 search against the user's prompt and injects the top hits
    as additionalContext. Uses SeedEngine to keep per-turn latency low.
    """
    try:
        payload = _read_hook_input()
        # Mid-session capture trigger: fire a detached background capture
        # every Nth turn. Fast + detached — it never blocks the prompt or recall.
        try:
            _maybe_fire_capture(payload)
        except Exception as exc:
            sys.stderr.write(f"poppy capture trigger error: {exc}\n")
        # Cursor beforeSubmitPrompt supports cadence side effects but no context
        # injection output. Keep stdout empty for its native schema.
        if _payload_host(payload) == "cursor":
            sys.exit(0)
        prompt = (payload.get("prompt") or "").strip()
        # Skip trivially short prompts — rarely useful, just adds noise.
        if len(prompt) < 8:
            sys.exit(0)

        project = project_from_cwd(payload.get("cwd"))
        engine = get_fast_engine(get_poppy_dir())
        tokens = _tokens(prompt)
        if not tokens:
            sys.exit(0)
        # Try project-scoped first; fall back to cross-project on miss.
        # When the user is in a project subdir we want scoped results, but
        # when the project filter drops everything we'd rather surface
        # globally-relevant memories than nothing.
        results: list[ScoredMemory] = []
        if project:
            results = _multi_token_retrieve(engine, tokens, Filters(project=project), limit=3)
        if not results:
            results = _multi_token_retrieve(engine, tokens, Filters(), limit=3)
        if not results:
            sys.exit(0)

        lines = ["## Poppy memories possibly relevant to this message:"]
        for r in results:
            content = _truncate(r.memory.content, 240)
            lines.append(f"- [{r.memory.memory_type}] {content}")
        envelope = {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": "\n".join(lines),
            }
        }
        sys.stdout.write(json.dumps(envelope))
        _maybe_emit_closed_loop(
            get_poppy_dir(), payload.get("session_id"), [r.memory for r in results], channel="recall_prompt"
        )
    except Exception as exc:
        sys.stderr.write(f"poppy user-prompt-submit hook error: {exc}\n")
    sys.exit(0)


@hook.command("pre-tool-use")
def pre_tool_use():
    """PreToolUse hook: surface memories about files about to be edited.

    Searches FTS5 by file path/basename and injects the top hits. The point
    is surgical recall before a code change ("have we made decisions about
    this file before?"), without dumping irrelevant project-wide memories.
    """
    try:
        payload = _read_hook_input()
        tool_input = payload.get("tool_input") or {}
        cursor_host = _payload_host(payload) == "cursor"
        if cursor_host and (payload.get("tool_name") != "Write" or not tool_input.get("file_path")):
            sys.exit(0)
        file_paths = _tool_file_paths(payload.get("tool_name"), tool_input)
        if not file_paths:
            sys.exit(0)

        # Search by basename + stem + parent dir name for every patch target.
        paths = [Path(file_path) for file_path in file_paths]
        candidates: list[str] = []
        for path in paths:
            for raw in (path.stem, path.name, path.parent.name):
                if not raw:
                    continue
                candidates.extend(_tokens(raw))
        if not candidates:
            sys.exit(0)
        # Dedupe preserving order
        tokens = list(dict.fromkeys(candidates))

        project = project_from_cwd(payload.get("cwd"))
        engine = get_fast_engine(get_poppy_dir())
        results: list[ScoredMemory] = []
        if project:
            results = _multi_token_retrieve(engine, tokens, Filters(project=project), limit=3)
        if not results:
            results = _multi_token_retrieve(engine, tokens, Filters(), limit=3)
        if not results:
            sys.exit(0)

        names = ", ".join(f"`{path.name}`" for path in paths)
        lines = [f"## Poppy memories touching {names}:"]
        for r in results:
            content = _truncate(r.memory.content, 240)
            lines.append(f"- [{r.memory.memory_type}] {content}")
        context = "\n".join(lines)
        if cursor_host:
            # Hard safety contract: recall hooks must only allow and exit 0.
            # Cursor treats "deny" or exit code 2 as a blocked user tool call.
            envelope = {"permission": "allow", "additional_context": context}
        else:
            envelope = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "additionalContext": context,
                }
            }
        sys.stdout.write(json.dumps(envelope))
        _maybe_emit_closed_loop(
            get_poppy_dir(), payload.get("session_id"), [r.memory for r in results], channel="recall_file"
        )
    except Exception as exc:
        sys.stderr.write(f"poppy pre-tool-use hook error: {exc}\n")
    sys.exit(0)


_PATCH_FILE_HEADER = re.compile(r"^\*\*\* (?:Update|Add|Delete) File: (.+)$", re.MULTILINE)


def _tool_file_paths(tool_name: str | None, tool_input: dict) -> list[str]:
    """Extract direct file arguments or Codex apply_patch header targets."""
    direct = tool_input.get("file_path") or tool_input.get("path")
    if direct:
        return [str(direct)]
    if tool_name != "apply_patch":
        return []
    patch = tool_input.get("command") or tool_input.get("patch") or ""
    return list(dict.fromkeys(match.strip() for match in _PATCH_FILE_HEADER.findall(str(patch)) if match.strip()))


@hook.command("stop")
def stop():
    """Stop hook: stamp session liveness without triggering capture.

    Codex fires Stop after every assistant turn and has no SessionEnd event.
    The timestamp is the state seam for a future idle sweeper; cadence capture
    remains exclusively on UserPromptSubmit.
    """
    try:
        _stamp_last_seen(_read_hook_input())
    except Exception:
        pass
    sys.exit(0)


def _stamp_last_seen(payload: dict) -> None:
    """Consent-gated liveness stamp shared by Codex Stop and session end."""
    session_id = payload.get("session_id")
    if not session_id:
        return
    from poppy.config import load_config
    from poppy.consolidation import is_enabled

    project = project_from_cwd(payload.get("cwd"))
    if is_enabled(load_config(get_poppy_dir()), project=project):
        from poppy.capture._state import stamp_last_seen

        stamp_last_seen(get_poppy_dir(), session_id)


# Keys of the raw hook payload that the replay commands actually re-feed to
# consolidation. Everything else in the payload is dropped before it is written
# to disk so the debug logs never persist the whole unbounded blob.
_COMPACT_REPLAY_KEYS = (
    "session_id",
    "transcript_path",
    "cwd",
    "compact_summary",
    "summary",
    "trigger",
    "compact_trigger",
    "cursor_version",
    "hook_event_name",
)
_SESSION_END_REPLAY_KEYS = ("session_id", "transcript_path", "cwd", "reason", "cursor_version", "hook_event_name")


def _replay_payload(payload: dict, keys: tuple[str, ...]) -> dict:
    """Trim a hook payload to just the fields a replay command needs.

    The debug logs exist so `poppy hook replay-*` can re-run a real event, but
    the raw payload can carry the entire session's content and arbitrary extra
    Claude Code fields. Persisting only the replay-relevant keys keeps the log
    small and predictable; 0600 (see `_append_debug_log`) protects what remains.
    """
    return {k: payload[k] for k in keys if k in payload}


def _append_debug_log(log_name: str, snapshot: dict, max_entries: int = 50) -> None:
    """Append snapshot to ~/.poppy/<log_name>, keeping at most `max_entries`.

    These logs can hold session-derived content (a PostCompact summary), so the
    file is tightened to 0600 — owner-only, never readable by other local
    accounts on a shared host.
    """
    ensure_poppy_dir(get_poppy_dir())
    log_path = get_poppy_dir() / log_name
    existing: list[dict] = []
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                existing.append(json.loads(line))
            except Exception:
                continue
    existing.append(snapshot)
    existing = existing[-max_entries:]
    log_path.write_text("\n".join(json.dumps(e) for e in existing) + "\n")
    try:
        log_path.chmod(0o600)
    except OSError:
        pass  # Best-effort; a chmod failure must not break the hook.


def _log_compact_payload(payload: dict) -> None:
    import datetime as _dt

    summary_text = payload.get("compact_summary") or payload.get("summary") or ""
    _append_debug_log(
        "postcompact-debug.log",
        {
            "ts": _dt.datetime.now(_dt.UTC).isoformat(),
            "keys": sorted(payload.keys()),
            "session_id": payload.get("session_id"),
            "trigger": payload.get("trigger") or payload.get("compact_trigger"),
            "summary_len": len(summary_text),
            "summary_head": summary_text[:240],
            "payload": _replay_payload(payload, _COMPACT_REPLAY_KEYS),
        },
    )


def _log_session_end_payload(payload: dict) -> None:
    import datetime as _dt

    transcript_path = payload.get("transcript_path") or ""
    transcript_size = 0
    try:
        if transcript_path:
            transcript_size = Path(transcript_path).stat().st_size
    except Exception:
        pass
    _append_debug_log(
        "sessionend-debug.log",
        {
            "ts": _dt.datetime.now(_dt.UTC).isoformat(),
            "keys": sorted(payload.keys()),
            "session_id": payload.get("session_id"),
            "reason": payload.get("reason"),
            "cwd": payload.get("cwd"),
            "transcript_path": transcript_path,
            "transcript_bytes": transcript_size,
            "payload": _replay_payload(payload, _SESSION_END_REPLAY_KEYS),
        },
    )


def _spawn_detached_worker(subcommand: str, payload: dict, worker_log_name: str) -> None:
    """Spawn `poppy hook <subcommand>` detached, piping payload JSON to its stdin.

    Used by hooks that need to run past Claude Code's hook-timeout window
    (PostCompact at 60s, others at the user's configured timeout). The child
    starts in a new session so it survives the parent hook's exit.
    """
    import os
    import subprocess

    log_path = get_poppy_dir() / worker_log_name
    log_fd = open(log_path, "ab")
    poppy_bin = sys.argv[0] if sys.argv and sys.argv[0] else "poppy"
    proc = subprocess.Popen(
        [poppy_bin, "hook", subcommand],
        stdin=subprocess.PIPE,
        stdout=log_fd,
        stderr=log_fd,
        start_new_session=True,
        close_fds=True,
        env={**os.environ},
    )
    try:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(payload).encode("utf-8"))
        proc.stdin.close()
    except Exception:
        pass
    log_fd.close()


def _maybe_fire_capture(payload: dict) -> None:
    """Mid-session capture trigger using ADR-0001's watermark-after-success rule.

    Every Nth turn, while consolidation is enabled, spawn a detached capture
    worker for the ``(watermark, now]`` window. All gating that needs the
    transcript or the store (single-flight, minimum-content, host-specific soft
    cap) lives in the worker; this stays fast so it never adds latency to the
    prompt, and is a no-op when consent/consolidation is absent.
    """
    session_id = payload.get("session_id")
    transcript_path = payload.get("transcript_path")
    if not session_id or not transcript_path:
        return

    from poppy.config import load_config
    from poppy.consolidation import is_enabled

    # Respect the per-project capture off switch as well as global
    # consent — a disabled repo must not even spawn the capture worker.
    project = project_from_cwd(payload.get("cwd"))
    if not is_enabled(load_config(get_poppy_dir()), project=project):
        return

    from poppy.capture.cadence import register_turn, should_capture, soft_cap_reached

    poppy_dir = get_poppy_dir()
    count = register_turn(poppy_dir, session_id)
    if not should_capture(count):
        return
    codex_session = _payload_host(payload) == "codex"
    if not codex_session and soft_cap_reached(poppy_dir, session_id):
        return  # the SessionEnd backstop will flush the remainder

    _spawn_detached_worker("_capture-worker", payload, "capture-worker.log")


@hook.command("_capture-worker", hidden=True)
def _capture_worker():
    """Internal: run a mid-session capture synchronously (spawned detached).

    Reads the same payload JSON from stdin so the LLM call can outlive the
    UserPromptSubmit hook's timeout window.
    """
    try:
        from poppy import writers

        # Register as a live writer so `poppy encrypt` refuses to migrate while
        # this detached worker is extracting + writing.
        with writers.registered(get_poppy_dir(), "capture"):
            payload = _read_hook_input()
            from poppy.capture.orchestrator import run_capture_worker
            from poppy.consolidation import consolidate_capture_event

            n = run_capture_worker(consolidate_capture_event, payload, milestone="mid_session")
            sys.stderr.write(f"poppy capture worker: stored {n} memories\n")
    except Exception as exc:
        sys.stderr.write(f"poppy capture worker error: {exc}\n")
    sys.exit(0)


@hook.command("post-compact")
def post_compact():
    """PostCompact hook: extract durable memories from the compact summary.

    The LLM extraction can take longer than Claude Code's 60s hook timeout
    (we got "Hook cancelled" on real fires). So this entry does only fast
    work synchronously (log the payload, then spawn a detached background
    worker to run the actual consolidation) and exits within ~100ms.

    The worker is `poppy hook _post-compact-worker`, invoked with the same
    payload re-piped to its stdin. It runs in a new process group, with
    stdin closed and stdout/stderr appended to ~/.poppy/postcompact-worker.log,
    so it survives this hook's exit.
    """
    try:
        payload = _read_hook_input()
    except Exception as exc:
        sys.stderr.write(f"poppy post-compact hook error: {exc}\n")
        sys.exit(0)

    try:
        _log_compact_payload(payload)
    except Exception as exc:
        sys.stderr.write(f"poppy post-compact log error: {exc}\n")

    try:
        _spawn_detached_worker("_post-compact-worker", payload, "postcompact-worker.log")
    except Exception as exc:
        sys.stderr.write(f"poppy post-compact spawn error: {exc}\n")
    sys.exit(0)


@hook.command("_post-compact-worker", hidden=True)
def _post_compact_worker():
    """Internal: run consolidate_compact_event synchronously.

    Spawned detached by the post-compact hook so the LLM call can run
    past Claude Code's 60s hook timeout. Reads the same payload JSON
    from stdin.
    """
    try:
        from poppy import writers

        with writers.registered(get_poppy_dir(), "post-compact"):
            payload = _read_hook_input()
            from poppy.capture.orchestrator import run_capture_worker
            from poppy.consolidation import consolidate_compact_event, consolidate_stop_event

            consolidate = consolidate_stop_event if _payload_host(payload) == "cursor" else consolidate_compact_event
            n = run_capture_worker(consolidate, payload, milestone="post_compact")
            sys.stderr.write(f"poppy post-compact worker: stored {n} memories\n")
    except Exception as exc:
        sys.stderr.write(f"poppy post-compact worker error: {exc}\n")
    sys.exit(0)


@hook.command("replay-compact")
@click.option(
    "--last",
    "n",
    default=1,
    show_default=True,
    help="Replay the Nth-from-last entry in postcompact-debug.log (1 = most recent).",
)
def replay_compact(n: int):
    """Replay a captured PostCompact payload synchronously, for testing.

    Reads ~/.poppy/postcompact-debug.log and re-runs consolidate_compact_event
    against the chosen entry. Idempotent: re-running on the same payload
    is a no-op once memories are stored.
    """
    log_path = get_poppy_dir() / "postcompact-debug.log"
    if not log_path.exists():
        click.echo(f"no debug log at {log_path}", err=True)
        sys.exit(1)
    entries = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except Exception:
            continue
    if not entries:
        click.echo("debug log is empty", err=True)
        sys.exit(1)
    if n < 1 or n > len(entries):
        click.echo(f"--last {n} out of range (have {len(entries)} entries)", err=True)
        sys.exit(1)

    entry = entries[-n]
    payload = entry.get("payload") or {}
    click.echo(
        f"replaying entry from {entry.get('ts')}: "
        f"session={entry.get('session_id')} summary_len={entry.get('summary_len')}"
    )
    try:
        from poppy.consolidation import consolidate_compact_event, consolidate_stop_event

        consolidate = consolidate_stop_event if _payload_host(payload) == "cursor" else consolidate_compact_event
        stored = consolidate(payload)
        click.echo(f"stored {stored} memories")
    except Exception as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)


@hook.command("session-end")
def session_end():
    """SessionEnd hook: end-of-session consolidation.

    Fires when the user closes the session (`/exit`, `/clear`, ctrl+c×2,
    window close, prompt-submitted-while-busy). Logs the payload to
    ~/.poppy/sessionend-debug.log for audit, then spawns a detached
    worker so the LLM call can outlive Claude Code's hook timeout.

    Default: silent no-op when consolidation isn't enabled (the worker
    short-circuits inside consolidate_stop_event).
    """
    try:
        payload = _read_hook_input()
    except Exception as exc:
        sys.stderr.write(f"poppy session-end hook error: {exc}\n")
        sys.exit(0)

    try:
        _stamp_last_seen(payload)
    except Exception:
        pass

    try:
        _log_session_end_payload(payload)
    except Exception as exc:
        sys.stderr.write(f"poppy session-end log error: {exc}\n")

    try:
        _spawn_detached_worker("_session-end-worker", payload, "sessionend-worker.log")
    except Exception as exc:
        sys.stderr.write(f"poppy session-end spawn error: {exc}\n")
    sys.exit(0)


@hook.command("_session-end-worker", hidden=True)
def _session_end_worker():
    """Internal: run consolidate_stop_event synchronously.

    Spawned detached by the session-end hook so the LLM call can run
    past Claude Code's hook timeout. Reads the same payload JSON
    from stdin.
    """
    try:
        from poppy import writers

        with writers.registered(get_poppy_dir(), "session-end"):
            payload = _read_hook_input()
            from poppy.capture.orchestrator import run_capture_worker
            from poppy.consolidation import consolidate_stop_event

            n = run_capture_worker(consolidate_stop_event, payload, milestone="session_end")
            sys.stderr.write(f"poppy session-end worker: stored {n} memories\n")
    except Exception as exc:
        sys.stderr.write(f"poppy session-end worker error: {exc}\n")
    sys.exit(0)


@hook.command("replay-session-end")
@click.option(
    "--last",
    "n",
    default=1,
    show_default=True,
    help="Replay the Nth-from-last entry in sessionend-debug.log (1 = most recent).",
)
def replay_session_end(n: int):
    """Replay a captured SessionEnd payload synchronously, for testing."""
    log_path = get_poppy_dir() / "sessionend-debug.log"
    if not log_path.exists():
        click.echo(f"no debug log at {log_path}", err=True)
        sys.exit(1)
    entries = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except Exception:
            continue
    if not entries:
        click.echo("debug log is empty", err=True)
        sys.exit(1)
    if n < 1 or n > len(entries):
        click.echo(f"--last {n} out of range (have {len(entries)} entries)", err=True)
        sys.exit(1)

    entry = entries[-n]
    payload = entry.get("payload") or {}
    click.echo(
        f"replaying entry from {entry.get('ts')}: "
        f"session={entry.get('session_id')} transcript_bytes={entry.get('transcript_bytes')}"
    )
    try:
        from poppy.consolidation import consolidate_stop_event

        stored = consolidate_stop_event(payload)
        click.echo(f"stored {stored} memories")
    except Exception as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)
