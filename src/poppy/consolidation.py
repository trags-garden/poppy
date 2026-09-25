"""End-of-session consolidation: read transcript → LLM extracts durable memories.

Two backends, tried in order:

1. **Host CLI subprocess** (default, zero-config): shells out to the same
   coding-agent CLI that ran the session: `claude -p` for Claude Code,
   `cursor-agent -p` for Cursor, `codex exec` for Codex, or `gemini -p` for Gemini CLI. Uses the user's
   existing login. No API key needed; cost lands on their existing
   subscription.

2. **OpenAI-compatible HTTP** (fallback): pulls model + base URL + api_key
   from ~/.poppy/config.json (preferred) or POPPY_CONSOLIDATE_* env vars
   (override). Useful when the user prefers a specific cheap model
   (Ollama Cloud kimi-k2.6, glm-5.1, etc.) or runs somewhere without
   a host CLI on PATH.

Either way, nothing runs until the user records a one-time consent
(`poppy autocapture on --global`, or the prompt during `poppy setup claude-code`).
The full consent and default-on precedence lives in ``poppy.capture.policy``;
``POPPY_CONSOLIDATE`` remains an explicit on/off override.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from poppy.capture import health
from poppy.capture.cadence import soft_cap_reached
from poppy.capture.lock import single_flight
from poppy.capture.orchestrator import CaptureOrchestrator, CapturePlan
from poppy.capture.policy import is_capture_enabled
from poppy.capture.transcript import detect_transcript_host
from poppy.capture.watermark import get_watermark
from poppy.capture.window import read_window
from poppy.config import PoppyConfig, load_config, resolved_consolidate_settings
from poppy.models import Filters
from poppy.project import project_from_cwd
from poppy.runtime import get_engine, get_poppy_dir
from poppy.sources import source_from_capture_host, source_from_host_cli

ALLOWED_TYPES = {"fact", "decision", "preference", "lesson"}

# Minimum qualifying turns in a SessionEnd backstop window before a consolidation
# pass is worthwhile. Preserves the pre-watermark gate; the mid-session loop adds
# its own minimum-content gate.
MIN_BACKSTOP_TURNS = 4

# Minimum characters of substantive new content in a mid-session capture window
# before a fire is worthwhile (minimum-content gate). A sane default,
# tuned later by the automatic reconciliation tuning pass.
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


def is_enabled(cfg: PoppyConfig | None = None, *, project: str | None = None) -> bool:
    """Whether auto-capture should run, per the ConsolidationPolicy.

    Delegates to ``capture.policy``: default-on once consent is recorded and a
    free host-CLI backend is present; inert until consent; never auto-spends on a
    remote-only backend. ``POPPY_CONSOLIDATE`` remains an explicit on/off override.
    ``project`` (when given) is checked against the per-project deny-list
    so capture stays off for a repo the developer disabled.
    """
    cfg = cfg if cfg is not None else load_config(get_poppy_dir())
    return is_capture_enabled(cfg, project=project)


# ---------------------------------------------------------------------------
# Transcript I/O
# ---------------------------------------------------------------------------


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

# Order to try host CLIs in when there is no transcript to identify the host
# (the conflict verdict, which judges one memory rather than a session). Same
# four CLIs the transcript-based branch below can pick, so a no-transcript call
# can never reach a backend a normal capture would not have used.
HOST_CLI_PREFERENCE = ("claude", "cursor-agent", "codex", "gemini")

# Seconds a host CLI gets to answer. Sized for the big job: a whole transcript
# window to extract memories from. Callers with a small prompt pass a smaller
# budget.
HOST_CLI_TIMEOUT_S = 120


def detect_host_cli(transcript_path: str | None) -> str | None:
    """Pick the host coding-agent CLI based on the transcript path.

    Without a transcript, fall back to the first CLI on PATH in
    ``HOST_CLI_PREFERENCE`` order.
    Returns the executable name if it exists on PATH, else None.
    """
    if not transcript_path:
        return next((cli for cli in HOST_CLI_PREFERENCE if shutil.which(cli)), None)
    transcript_host = detect_transcript_host(Path(transcript_path))
    if transcript_host == "cursor" and shutil.which("cursor-agent"):
        return "cursor-agent"
    if transcript_host == "codex" and shutil.which("codex"):
        return "codex"
    if transcript_host == "claude-code" and shutil.which("claude"):
        return "claude"
    # Claude Code stores transcripts under ~/.claude/projects/<slug>/
    if "/.claude/projects/" in transcript_path and shutil.which("claude"):
        return "claude"
    # Codex stores transcripts under ~/.codex/...
    if "/.codex/" in transcript_path and shutil.which("codex"):
        return "codex"
    if "/.cursor/" in transcript_path and shutil.which("cursor-agent"):
        return "cursor-agent"
    # Gemini CLI stores transcripts under ~/.gemini/...
    if "/.gemini/" in transcript_path and shutil.which("gemini"):
        return "gemini"
    # Last resort — if claude is on PATH, use it.
    if shutil.which("claude"):
        return "claude"
    return None


def _capture_source(transcript_path: str) -> str:
    host = detect_transcript_host(Path(transcript_path))
    return source_from_capture_host(host, fallback_cli=detect_host_cli(transcript_path))


def call_host_cli(
    prompt: str, *, cli: str, timeout_s: int = HOST_CLI_TIMEOUT_S, record_health: bool = True
) -> str | None:
    """Invoke a coding-agent CLI in headless mode. Returns the model's text.

    Every hard failure (non-zero exit, timeout, spawn error) is recorded as
    backend health so a host CLI that is on PATH but broken — the
    logged-out ``claude`` case — degrades the SessionStart banner and
    ``poppy doctor`` instead of failing silently. Clearing that record is left to
    ``call_llm``, which is where the output is parsed: exiting 0 is not proof the
    backend worked, since a CLI can exit 0 while printing an error page.

    ``record_health=False`` still logs a failure but keeps it out of that record.
    It is for callers on a tighter timeout than an extraction, whose result would
    otherwise misreport the backend in both directions: their timeout is not
    evidence the CLI is broken, and their success is not evidence it is healthy.
    """
    if cli == "claude":
        # claude -p reads the prompt from stdin or as a positional arg.
        # --output-format text gives us the raw text response (no JSON envelope).
        cmd = ["claude", "-p", "--output-format", "text"]
    elif cli == "cursor-agent":
        cmd = ["cursor-agent", "-p"]
    elif cli == "codex":
        # codex exec takes a prompt arg; output is plain text.
        cmd = ["codex", "exec", prompt]
    elif cli == "gemini":
        cmd = ["gemini", "-p"]
    else:
        sys.stderr.write(f"poppy consolidate: unknown cli {cli!r}\n")
        return None

    child_env = {**os.environ, "POPPY_SUPPRESS_HOOKS": "1"}
    try:
        if cli == "codex":
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=child_env)
        else:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout_s, env=child_env)
    except subprocess.TimeoutExpired:
        if record_health:
            health.record_failure(cli, f"timed out after {timeout_s}s")
        sys.stderr.write(f"poppy consolidate: {cli} timed out after {timeout_s}s\n")
        return None
    except OSError as exc:
        if record_health:
            health.record_failure(cli, f"spawn failed: {exc}")
        sys.stderr.write(f"poppy consolidate: {cli} spawn failed: {exc}\n")
        return None
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip().splitlines()[-3:]
        detail = f"exited rc={proc.returncode} stderr_tail={' | '.join(stderr_tail)!r}"
        if record_health:
            health.record_failure(cli, detail)
        sys.stderr.write(f"poppy consolidate: {cli} {detail}\n")
        return None
    return proc.stdout or None


# ---------------------------------------------------------------------------
# Backend 2: OpenAI-compatible HTTP
# ---------------------------------------------------------------------------


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

# A remote call needs a few seconds to be worth starting. Below this the
# fallback is skipped and said so, rather than started and cut off instantly.
MIN_FALLBACK_TIMEOUT_S = 2.0

# A chat completion is a few kilobytes. A body past this is a server streaming
# without end, which would fill memory and outlast the budget.
MAX_RESPONSE_BYTES = 8 * 1024 * 1024

# Reaching a working endpoint is quick even over a slow link, so the connect
# and pool waits are capped well under the budget, leaving it for the answer.
CONNECT_TIMEOUT_S = 5.0

# Shorter configured keys are a local server's placeholder ("ollama", "none"),
# not a secret, so seeing one in an answer says nothing and must not throw the
# answer away.
MIN_ECHOED_KEY_LEN = 12


class OpenAICompatError(Exception):
    """A fallback failure safe to display in worker logs and backend health."""


def _is_loopback(host: str) -> bool:
    """Whether a hostname names this machine, judged on the literal form alone.

    A fully qualified name may carry a trailing root dot, and the unspecified
    addresses reach this machine too. Deliberately no DNS: a lookup here would
    be a network call on the path that decides whether the network is trusted.
    """
    host = host.lower().removesuffix(".")
    if host in ("localhost", "0.0.0.0", "::", "::1") or host.endswith(".localhost"):
        return True
    return host.startswith("127.")


def _endpoint(base_url: str | None) -> tuple[str, bool]:
    """The chat-completions URL, and whether it is served from this machine.

    A configured URL is user input, and a malformed one makes ``urlsplit``
    raise, so this reports a named failure rather than letting a traceback
    abort a capture pass. A local model server is the normal way to run an
    OpenAI-compatible endpoint and needs no TLS, so only plain http to another
    host is worth warning about.
    """
    raw = (base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/")
    try:
        parsed = urlsplit(raw)
        host = (parsed.hostname or "").lower()
    except ValueError as exc:
        raise OpenAICompatError("configured endpoint is not a valid URL") from exc
    if parsed.scheme not in ("http", "https") or not host:
        raise OpenAICompatError("configured endpoint is not an http or https URL")
    loopback = _is_loopback(host)
    if parsed.scheme == "http" and not loopback:
        sys.stderr.write(
            f"poppy consolidate: warning, http://{host} is not encrypted, so the API key is sent in clear text\n"
        )
    return f"{raw}/chat/completions", loopback


def _http_client(*, timeout_s: float, trust_env: bool, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """The client every fallback request goes through.

    ``transport`` substitutes the socket layer only, so a test exercises the
    real timeouts, redirect policy and proxy decision made here.

    ``follow_redirects=False`` keeps the Authorization header from being
    replayed at a location the endpoint picks. ``trust_env`` is off for a
    server on this machine, where an environment proxy is both wrong and a way
    to capture the key, and on for a remote endpoint, where a corporate proxy
    is often how the request gets out at all.

    Connecting and waiting for a pool slot get a short cap of their own, since
    each httpx timeout is per operation: an endpoint that stalls the connect
    would otherwise spend the whole budget before a single byte is read.
    """
    handshake_s = min(CONNECT_TIMEOUT_S, timeout_s)
    timeout = httpx.Timeout(timeout_s, connect=handshake_s, pool=handshake_s)
    kwargs = {"transport": transport} if transport is not None else {}
    return httpx.Client(timeout=timeout, trust_env=trust_env, follow_redirects=False, **kwargs)


def _read_bounded(resp: httpx.Response, *, deadline: float) -> bytes:
    """Read a streamed body under a wall-clock deadline and a size cap.

    An httpx timeout applies per socket operation, so a server that sends one
    byte just inside it holds the connection open for as long as it likes. A
    capture pass runs while a lock another worker will steal on a fixed TTL, so
    the read needs a real deadline and a bound on how much it will hold.
    """
    chunks: list[bytes] = []
    total = 0
    for chunk in resp.iter_bytes():
        if time.monotonic() >= deadline:
            raise OpenAICompatError("timed out reading the response")
        total += len(chunk)
        if total > MAX_RESPONSE_BYTES:
            raise OpenAICompatError(f"response larger than {MAX_RESPONSE_BYTES // (1024 * 1024)}MB")
        chunks.append(chunk)
    return b"".join(chunks)


def _reject_if_key_echoed(text: str, api_key: str) -> None:
    """Discard a whole answer that quotes the configured credential back.

    Response text is parsed into stored memories, which sync, and its opening
    is logged, so a key in it would persist. Rewriting the key out of the text
    would be worse than dropping it: a key that happens to read like ordinary
    words would silently corrupt real memories. An endpoint echoing the
    credential is not one to trust the rest of the answer from either.
    """
    if len(api_key) >= MIN_ECHOED_KEY_LEN and api_key in text:
        raise OpenAICompatError("response echoed the configured API key and was discarded")


def call_openai_compat(
    prompt: str,
    *,
    model: str,
    base_url: str | None,
    api_key: str,
    max_tokens: int = 800,
    timeout_s: float = HOST_CLI_TIMEOUT_S,
) -> str:
    """Request a chat completion, raising a specific, credential-free failure."""
    deadline = time.monotonic() + timeout_s
    url, loopback = _endpoint(base_url)
    try:
        with _http_client(timeout_s=timeout_s, trust_env=not loopback) as client:
            with client.stream(
                "POST",
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                    "max_tokens": max_tokens,
                },
            ) as resp:
                resp.raise_for_status()
                body = _read_bounded(resp, deadline=deadline)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        unmapped = "server error" if status >= 500 else httpx.codes.get_reason_phrase(status) or "unexpected status"
        reason = {
            401: "authentication failed",
            403: "access denied",
            429: "rate limit exceeded",
        }.get(status, "redirected, which is not followed" if 300 <= status < 400 else unmapped)
        raise OpenAICompatError(f"HTTP {status}: {reason}") from exc
    except httpx.TimeoutException as exc:
        raise OpenAICompatError(f"timed out after {timeout_s:.0f}s ({type(exc).__name__})") from exc
    except httpx.RequestError as exc:
        # URLs, exception messages and response bodies can contain credentials
        # or prompt text. Report the transport category without persisting them.
        kind = "connection failed" if isinstance(exc, httpx.NetworkError) else "request failed"
        raise OpenAICompatError(f"{kind} ({type(exc).__name__})") from exc
    except (httpx.InvalidURL, ValueError) as exc:
        raise OpenAICompatError("invalid endpoint URL or request configuration") from exc

    try:
        # Bytes, so a body that is not UTF-8 lands here as a decode error
        # rather than anywhere further down.
        payload = json.loads(body)
    except ValueError as exc:
        raise OpenAICompatError("response is not valid JSON") from exc
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OpenAICompatError("response has no text in choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise OpenAICompatError("response has no text in choices[0].message.content")
    # Before returning, so no caller can parse, store or log the text.
    _reject_if_key_echoed(content, api_key)
    return content


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _find_json_array(text: str) -> list[object] | None:
    """The first JSON array in ``text``, or None when there is none.

    Split out of ``parse_json_array`` so a caller can tell "the backend answered
    with an array holding nothing durable" apart from "the output was not a JSON
    array at all" -- the two look identical once normalised to ``[]``.
    """
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        arr = json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return None
    return arr if isinstance(arr, list) else None


def parse_json_array(text: str) -> list[dict[str, str]]:
    """Tolerant JSON parser. Strips ```json fences, finds the first [...] array."""
    arr = _find_json_array(text)
    if arr is None:
        return []
    out: list[dict[str, str]] = []
    for item in arr:
        if not isinstance(item, dict):
            continue
        mtype = item.get("type", "fact")
        raw_content = item.get("content")
        # A model can put a number or an object where text belongs. Drop the
        # item rather than letting one malformed entry end the capture pass.
        if not isinstance(raw_content, str):
            continue
        content = raw_content.strip()
        if not content:
            continue
        if not isinstance(mtype, str) or mtype not in ALLOWED_TYPES:
            mtype = "fact"
        out.append({"type": mtype, "content": content})
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def call_llm(
    prompt: str,
    *,
    transcript_path: str | None,
    cfg: PoppyConfig,
    parser: Callable[[str], list[dict]] = parse_json_array,
    host_timeout_s: int = HOST_CLI_TIMEOUT_S,
    record_health: bool = True,
) -> list[dict]:
    """Pick a backend, run it, parse the JSON array.

    Tries host CLI first (free, no key). Falls back to OpenAI-compat using
    config / env. Logs a one-line classification of every empty result so
    silent zeros are diagnosable. The default parser extracts durable memories;
    callers requesting another response shape can supply their own parser, and
    callers with a small prompt can shorten ``host_timeout_s`` for both backends.

    ``record_health=False`` leaves the extraction-backend health record
    untouched, for a caller whose timeout is too short to judge the backend by.

    ``host_timeout_s`` budgets the whole call, not each backend in turn. A
    capture pass runs one extraction plus a verdict per candidate while holding
    the per-session lock, and that budget is sized on one timeout per call; a
    fallback free to spend a second full timeout after a slow host CLI would
    let the lock go stale under a worker that is still running.
    """
    deadline = time.monotonic() + host_timeout_s
    cli = detect_host_cli(transcript_path)
    if cli:
        text = call_host_cli(prompt, cli=cli, timeout_s=host_timeout_s, record_health=record_health)
        if text is None:
            # call_host_cli already logged the failure mode (timeout / rc / OSError).
            pass
        else:
            parsed = parser(text)
            arr = _find_json_array(text)
            empty_array = arr is not None and not arr
            if (parsed or empty_array) and record_health:
                # Exit 0 *and* an array we could parse is the only evidence the
                # backend really works; an empty array is a real answer ("nothing
                # durable here"). An exit-0 error page falls through and leaves
                # the failure record standing.
                health.record_success(cli)
            if parsed:
                return parsed
            if empty_array:
                # An empty array is the host CLI answering "nothing
                # durable here", not a backend failure. Stop on it, so a capture
                # window with nothing to keep never re-asks a paid remote model.
                # A non-empty array whose items are unusable is a broken answer,
                # so it keeps the fallback below.
                sys.stderr.write(f"poppy consolidate: {cli} returned empty array (nothing durable)\n")
                return []
            snippet = text.strip()[:200].replace("\n", " ")
            detail = f"len={len(text)}, first 200: {snippet!r}"
            sys.stderr.write(f"poppy consolidate: {cli} output unparseable as JSON array ({detail})\n")

    settings = resolved_consolidate_settings(cfg)
    model, base_url, api_key = settings.model, settings.base_url, settings.api_key
    if model and api_key:
        remaining = deadline - time.monotonic()
        if remaining < MIN_FALLBACK_TIMEOUT_S:
            sys.stderr.write(
                f"poppy consolidate: openai-compat skipped, {cli or 'the host cli'} used the {host_timeout_s}s budget\n"
            )
            return []
        try:
            text = call_openai_compat(prompt, model=model, base_url=base_url, api_key=api_key, timeout_s=remaining)
        except OpenAICompatError as exc:
            if record_health:
                health.record_failure("openai-compat", str(exc))
            sys.stderr.write(f"poppy consolidate: openai-compat {exc}\n")
            return []
        parsed = parser(text)
        arr = _find_json_array(text)
        if (parsed or (arr is not None and not arr)) and record_health:
            # An answer we could parse is the only evidence the fallback really
            # works, so extraction as a whole is healthy even if the host CLI
            # just failed. Text we cannot parse leaves the record standing.
            health.record_success()
        if not parsed:
            snippet = text.strip()[:200].replace("\n", " ")
            sys.stderr.write(f"poppy consolidate: openai-compat output yielded 0 items (first 200: {snippet!r})\n")
        return parsed
    elif not cli:
        sys.stderr.write("poppy consolidate: no host cli detected and no openai-compat fallback configured\n")
    return []


def _orchestrator(cfg: PoppyConfig, poppy_dir: Path, *, engine=None) -> CaptureOrchestrator:
    """Build the capture orchestrator that owns the shared extract→ingest tail.

    ``engine`` is passed by the entry point that already built one for an
    idempotency scan; the window-based entries let it default to ``get_engine``.
    """
    return CaptureOrchestrator(
        engine=engine if engine is not None else get_engine(poppy_dir, delegate_models_to_daemon=True),
        cfg=cfg,
        poppy_dir=poppy_dir,
    )


def consolidate_stop_event(payload: dict) -> int:
    """Run consolidation for a Stop hook payload. Returns the number of memories stored.

    No-op (returns 0) if disabled, transcript missing, or session already consolidated.
    """
    cfg = load_config(get_poppy_dir())
    project = project_from_cwd(payload.get("cwd"))
    if not is_enabled(cfg, project=project):
        return 0

    session_id = payload.get("session_id")
    transcript_path = payload.get("transcript_path")
    if not session_id or not transcript_path:
        return 0

    poppy_dir = get_poppy_dir()

    with single_flight(poppy_dir, session_id) as acquired:
        if not acquired:
            return 0

        # Incremental capture: read only the window (watermark, end].
        # The lock keeps cadence, preCompact, and sessionEnd from extracting the
        # same window concurrently. read_window also clamps an oversized watermark.
        watermark = get_watermark(poppy_dir, session_id)
        window = read_window(Path(transcript_path), watermark=watermark)
        if len(window.messages) < MIN_BACKSTOP_TURNS:
            return 0

        transcript = format_transcript(window.messages)
        if not transcript.strip():
            return 0

        max_items = resolved_consolidate_settings(cfg).max_items
        plan = CapturePlan(
            session_id=session_id,
            prompt=CONSOLIDATION_PROMPT.format(max_items=max_items, transcript=transcript),
            source_type=_capture_source(transcript_path),
            project=project,
            transcript_path=transcript_path,
            max_items=max_items,
            # Watermark-after-success rule: advance the watermark only after a successful ingest.
            # Journal the backstop capture so a short session's only capture is
            # visible to the banner and `poppy doctor`.
            advance_watermark_to=window.new_watermark,
            journal=True,
        )
        return _orchestrator(cfg, poppy_dir).run(plan).stored


def _window_substance(messages: list[dict[str, str]]) -> int:
    return sum(len(m.get("text", "")) for m in messages)


def consolidate_capture_event(payload: dict) -> int:
    """Watermark-after-success mid-session capture. Returns the
    number of memories stored.

    Fired every Nth turn by the UserPromptSubmit hook via a detached worker. It
    processes only the window ``(watermark, now]`` under a per-session
    single-flight lock, and skips thin windows (minimum-content gate). Claude
    Code stops at the per-session soft cap and lets SessionEnd flush the rest;
    Codex disables that cap because it has no session-end event.
    No-op (returns 0) when consolidation is disabled or no backend/transcript is
    available, so it never blocks the developer's prompt.
    """
    cfg = load_config(get_poppy_dir())
    project = project_from_cwd(payload.get("cwd"))
    if not is_enabled(cfg, project=project):
        return 0

    session_id = payload.get("session_id")
    transcript_path = payload.get("transcript_path")
    if not session_id or not transcript_path:
        return 0

    poppy_dir = get_poppy_dir()
    with single_flight(poppy_dir, session_id) as acquired:
        if not acquired:
            return 0  # another capture worker is already running for this session
        codex_session = detect_transcript_host(Path(transcript_path)) == "codex"
        if not codex_session and soft_cap_reached(poppy_dir, session_id):
            return 0  # defer the remainder to the SessionEnd backstop

        watermark = get_watermark(poppy_dir, session_id)
        window = read_window(Path(transcript_path), watermark=watermark)
        if _window_substance(window.messages) < MIN_CAPTURE_CHARS:
            return 0  # minimum-content gate — too little new content to be worth a fire

        transcript = format_transcript(window.messages)
        if not transcript.strip():
            return 0

        max_items = resolved_consolidate_settings(cfg).max_items
        plan = CapturePlan(
            session_id=session_id,
            prompt=CONSOLIDATION_PROMPT.format(max_items=max_items, transcript=transcript),
            source_type=_capture_source(transcript_path),
            project=project,
            transcript_path=transcript_path,
            max_items=max_items,
            # Advance the watermark so no turn is captured twice and journal what
            # was stored. Only hosts with a session-end backstop count the soft cap.
            advance_watermark_to=window.new_watermark,
            record_soft_cap=not codex_session,
            journal=True,
        )
        # The ingest + watermark advance + soft-cap count must all happen while we
        # still hold the single-flight lock, so run inside the `with` block.
        return _orchestrator(cfg, poppy_dir).run(plan).stored


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
    project = project_from_cwd(payload.get("cwd"))
    if not is_enabled(cfg, project=project):
        return 0

    session_id = payload.get("session_id")
    summary = (payload.get("compact_summary") or payload.get("summary") or "").strip()
    if not session_id or not summary:
        return 0

    event_id = _compact_event_id(session_id, summary)
    poppy_dir = get_poppy_dir()
    engine = get_engine(poppy_dir, delegate_models_to_daemon=True)

    # Idempotency: bail if this exact compact event was already consolidated.
    # (Keyed off the composite event id, not the watermark — a session can compact
    # more than once, so this entry point does not use the capture watermark.)
    existing = engine.list_all(filters=Filters(project=project), limit=200)
    if any(m.source.session_id == event_id for m in existing):
        return 0

    max_items = resolved_consolidate_settings(cfg).max_items
    transcript_path = payload.get("transcript_path")
    plan = CapturePlan(
        session_id=event_id,
        # Cap summary input — compact_summary can run several KB; keep prompt small.
        prompt=COMPACT_CONSOLIDATION_PROMPT.format(max_items=max_items, summary=summary[:12000]),
        source_type=_capture_source(transcript_path) if transcript_path else source_from_host_cli(None),
        project=project,
        transcript_path=transcript_path,
        max_items=max_items,
        journal=True,
    )
    return _orchestrator(cfg, poppy_dir, engine=engine).run(plan).stored
