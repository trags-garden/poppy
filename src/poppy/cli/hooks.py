"""Claude Code hook entrypoints.

Each hook reads JSON from stdin (Claude Code hook input format) and writes
either nothing (silent exit 0) or a JSON envelope on stdout. Hooks must be
fast and fail open — never block the agent on a Poppy error.
"""

import json
import re
import sys
from pathlib import Path

import click

from poppy.engine.interface import RetrievalEngine
from poppy.models import Filters, ScoredMemory
from poppy.runtime import get_fast_engine, get_poppy_dir

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
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


# Markers that indicate a project root, in priority order. CLAUDE.md /
# AGENTS.md beat package files because the user authored them deliberately;
# package files beat .git because monorepo .git can sit far above the actual
# project. Scanning stops at the first match walking up from cwd.
_PROJECT_MARKERS: tuple[str, ...] = (
    "CLAUDE.md",
    "AGENTS.md",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "build.gradle",
    "build.gradle.kts",
    "Gemfile",
    ".git",
)


def _project_from_cwd(cwd: str | None, max_depth: int = 6) -> str | None:
    """Resolve a project name from cwd by walking up looking for project markers.

    Returns the basename of the directory containing the first marker found.
    Returns None if cwd is unset or no marker found within ``max_depth`` parents.
    Falling back to ``Path(cwd).name`` would mis-tag cases like ``~/code/personal``
    where the basename ("personal") is not actually a project — better to
    return None and let the hook search across all projects.
    """
    if not cwd:
        return None
    start = Path(cwd)
    if not start.is_absolute():
        return start.name or None
    current = start
    for _ in range(max_depth):
        for marker in _PROJECT_MARKERS:
            if (current / marker).exists():
                return current.name or None
        if current.parent == current:
            break
        current = current.parent
    return None


def _truncate(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[:n].rstrip() + "…"


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


@click.group()
def hook():
    """Claude Code hook entrypoints. Invoked by Poppy-installed hooks."""


@hook.command("session-start")
def session_start():
    """SessionStart hook: status banner + the most relevant memories for this project.

    Prepends a one-line trust banner: "Poppy active, N memories, M
    captured last session" when capture is running, a loud INACTIVE line when the
    install is partially broken (engine failed, no/remote-only backend), or a
    consent nudge when capture is pending. Deliberate off-states (opt-out /
    env-off) stay silent. The banner makes invisible background capture
    observable; the memory list below it is unchanged.
    """
    try:
        payload = _read_hook_input()
        cwd = payload.get("cwd")
        project = _project_from_cwd(cwd)
        poppy_dir = get_poppy_dir()

        # Reset this session's capture watermark + cadence so coverage is
        # predictable per session (ADR-0001).
        session_id = payload.get("session_id")
        if session_id:
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
            memories = scoped[:15]
        except Exception as exc:
            engine_ok = False
            sys.stderr.write(f"poppy session-start engine error: {exc}\n")

        sections: list[str] = []
        banner = _session_banner(poppy_dir, project=project, memory_count=memory_count, engine_ok=engine_ok)
        if banner:
            sections.append(banner)

        if memories:
            lines = ["## Poppy memories for this project:" if project else "## Poppy memories:"]
            for m in memories:
                lines.append(f"- [{m.memory_type}] {m.content}")
            sections.append("\n".join(lines))

        if not sections:
            sys.exit(0)

        envelope = {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": "\n\n".join(sections),
            }
        }
        sys.stdout.write(json.dumps(envelope))
    except Exception as exc:
        sys.stderr.write(f"poppy session-start hook error: {exc}\n")
    sys.exit(0)


def _session_banner(poppy_dir: Path, *, project: str | None, memory_count: int, engine_ok: bool) -> str | None:
    """Render the SessionStart status banner, or None when nothing shows.

    Reads the capture status (ConsolidationPolicy) and the last-session capture
    count (CaptureJournal). Never raises — a banner failure must not break the
    session, so any error falls through to no banner.
    """
    try:
        from poppy.capture import journal
        from poppy.capture.banner import render_banner
        from poppy.capture.policy import evaluate
        from poppy.config import load_config

        status = evaluate(load_config(poppy_dir))
        return render_banner(
            status,
            project=project,
            memory_count=memory_count,
            last_session_count=journal.last_session_count(poppy_dir),
            engine_ok=engine_ok,
        )
    except Exception as exc:
        sys.stderr.write(f"poppy session-start banner error: {exc}\n")
        return None


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
        prompt = (payload.get("prompt") or "").strip()
        # Skip trivially short prompts — rarely useful, just adds noise.
        if len(prompt) < 8:
            sys.exit(0)

        project = _project_from_cwd(payload.get("cwd"))
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
    except Exception as exc:
        sys.stderr.write(f"poppy user-prompt-submit hook error: {exc}\n")
    sys.exit(0)


@hook.command("pre-tool-use")
def pre_tool_use():
    """PreToolUse hook (Edit|Write|MultiEdit): surface memories about the file.

    Searches FTS5 by file path/basename and injects the top hits. The point
    is surgical recall before a code change — "have we made decisions about
    this file before?" — without dumping irrelevant project-wide memories.
    """
    try:
        payload = _read_hook_input()
        tool_input = payload.get("tool_input") or {}
        file_path = tool_input.get("file_path") or tool_input.get("path") or ""
        if not file_path:
            sys.exit(0)

        # Search by basename + stem + parent dir name. Tokenized so each runs
        # as a single-word FTS5 phrase match (SeedEngine wraps in quotes).
        path = Path(file_path)
        candidates: list[str] = []
        for raw in (path.stem, path.name, path.parent.name):
            if not raw:
                continue
            candidates.extend(_tokens(raw))
        if not candidates:
            sys.exit(0)
        # Dedupe preserving order
        tokens = list(dict.fromkeys(candidates))

        project = _project_from_cwd(payload.get("cwd"))
        engine = get_fast_engine(get_poppy_dir())
        results: list[ScoredMemory] = []
        if project:
            results = _multi_token_retrieve(engine, tokens, Filters(project=project), limit=3)
        if not results:
            results = _multi_token_retrieve(engine, tokens, Filters(), limit=3)
        if not results:
            sys.exit(0)

        lines = [f"## Poppy memories touching `{path.name}`:"]
        for r in results:
            content = _truncate(r.memory.content, 240)
            lines.append(f"- [{r.memory.memory_type}] {content}")
        envelope = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": "\n".join(lines),
            }
        }
        sys.stdout.write(json.dumps(envelope))
    except Exception as exc:
        sys.stderr.write(f"poppy pre-tool-use hook error: {exc}\n")
    sys.exit(0)


@hook.command("stop")
def stop():
    """Stop hook: deprecated, no-op.

    Claude Code fires Stop after every assistant turn, not at session close.
    Consolidation moved to the SessionEnd hook (`poppy hook session-end`).
    Existing installs that still call `poppy hook stop` get a fast no-op
    rather than running the LLM on partial transcripts each turn.
    """
    try:
        _read_hook_input()
    except Exception:
        pass
    sys.exit(0)


def _append_debug_log(log_name: str, snapshot: dict, max_entries: int = 50) -> None:
    """Append snapshot to ~/.poppy/<log_name>, keeping at most `max_entries`."""
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
            "payload": payload,
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
            "payload": payload,
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
    """Mid-session capture trigger (ADR-0001).

    Every Nth turn, while consolidation is enabled, spawn a detached capture
    worker for the ``(watermark, now]`` window. All gating that needs the
    transcript or the store (single-flight, minimum-content, soft cap) lives in
    the worker; this stays fast so it never adds latency to the prompt, and is a
    no-op when consent/consolidation is absent.
    """
    session_id = payload.get("session_id")
    transcript_path = payload.get("transcript_path")
    if not session_id or not transcript_path:
        return

    from poppy.config import load_config
    from poppy.consolidation import is_enabled

    if not is_enabled(load_config(get_poppy_dir())):
        return

    from poppy.capture.cadence import register_turn, should_capture, soft_cap_reached

    poppy_dir = get_poppy_dir()
    count = register_turn(poppy_dir, session_id)
    if not should_capture(count):
        return
    if soft_cap_reached(poppy_dir, session_id):
        return  # the SessionEnd backstop will flush the remainder

    _spawn_detached_worker("_capture-worker", payload, "capture-worker.log")


@hook.command("_capture-worker", hidden=True)
def _capture_worker():
    """Internal: run a mid-session capture synchronously (spawned detached).

    Reads the same payload JSON from stdin so the LLM call can outlive the
    UserPromptSubmit hook's timeout window.
    """
    try:
        payload = _read_hook_input()
        from poppy.consolidation import consolidate_capture_event

        n = consolidate_capture_event(payload)
        sys.stderr.write(f"poppy capture worker: stored {n} memories\n")
        if n > 0:
            from poppy.sync.auto import trigger as _trigger_autosync

            _trigger_autosync(get_poppy_dir())
    except Exception as exc:
        sys.stderr.write(f"poppy capture worker error: {exc}\n")
    sys.exit(0)


@hook.command("post-compact")
def post_compact():
    """PostCompact hook: extract durable memories from the compact summary.

    The LLM extraction can take longer than Claude Code's 60s hook timeout
    (we got "Hook cancelled" on real fires). So this entry does only fast
    work synchronously — log the payload, then spawn a detached background
    worker to run the actual consolidation — and exits within ~100ms.

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
        payload = _read_hook_input()
        from poppy.consolidation import consolidate_compact_event

        n = consolidate_compact_event(payload)
        sys.stderr.write(f"poppy post-compact worker: stored {n} memories\n")
        if n > 0:
            from poppy.sync.auto import trigger as _trigger_autosync

            _trigger_autosync(get_poppy_dir())
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
    against the chosen entry. Idempotent — re-running on the same payload
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
        from poppy.consolidation import consolidate_compact_event

        stored = consolidate_compact_event(payload)
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
        payload = _read_hook_input()
        from poppy.consolidation import consolidate_stop_event

        n = consolidate_stop_event(payload)
        sys.stderr.write(f"poppy session-end worker: stored {n} memories\n")
        if n > 0:
            from poppy.sync.auto import trigger as _trigger_autosync

            _trigger_autosync(get_poppy_dir())
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
