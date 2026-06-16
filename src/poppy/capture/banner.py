"""SessionStart status banner.

A one-line banner prepended to the SessionStart context so the developer can see
whether auto-capture is working. It reads the capture status (ConsolidationPolicy,
ADR-0002), the project memory count (engine), and the last-session capture count
(CaptureJournal) — the three signals that make silent background capture
observable rather than invisible.

The render is a pure function so the active / INACTIVE / consent-pending wording is
testable without a live agent session. Three rules drive it:

* **Active** (capture is running): a reassuring line with the memory count and how
  many memories the last session captured — proof the loop runs.
* **INACTIVE** (consented but broken — no backend, remote-only, or the engine
  failed to load): a *loud* line, because this is the silent-breakage case the
  banner exists to catch (a partially-installed Poppy that quietly captures
  nothing for weeks).
* **Consent pending**: a nudge pointing at ``poppy consent --enable`` — never a
  misleading "0 captured", since nothing is captured until consent is recorded.

A deliberate off-state (explicit opt-out or a ``POPPY_CONSOLIDATE=0`` override) is
silent: the developer made that choice, so the banner does not nag every session.
"""

from __future__ import annotations

from poppy.capture.policy import CaptureStatus

# Statuses that mean "the developer turned this off on purpose" — no banner.
_SILENT_STATUSES = frozenset({CaptureStatus.DISABLED_OPT_OUT, CaptureStatus.DISABLED_ENV})


def _scope(project: str | None) -> str:
    return "this project" if project else "all projects"


def _memory_clause(memory_count: int, project: str | None) -> str:
    noun = "memory" if memory_count == 1 else "memories"
    return f"{memory_count} {noun} for {_scope(project)}"


def render_banner(
    status: CaptureStatus,
    *,
    project: str | None,
    memory_count: int,
    last_session_count: int | None,
    engine_ok: bool = True,
) -> str | None:
    """Render the SessionStart banner line, or ``None`` when nothing should show.

    ``last_session_count`` is the journal's total for the most recent session that
    captured anything (``None`` when nothing has ever been captured).
    """
    # A broken engine means recall itself is down — the loudest INACTIVE case,
    # independent of capture consent/backend.
    if not engine_ok:
        return (
            "## Poppy: INACTIVE\n"
            "Poppy could not load its memory engine, so recall and capture are both off. "
            "Run `poppy doctor` to diagnose."
        )

    if status in _SILENT_STATUSES:
        return None

    if status is CaptureStatus.INERT_PENDING:
        mem = _memory_clause(memory_count, project)
        return (
            "## Poppy\n"
            f"Automatic capture is pending your consent; nothing is captured yet ({mem}). "
            "Run `poppy consent --enable` to turn it on, or `poppy consent --disable` to dismiss."
        )

    if status in (CaptureStatus.WARN_REMOTE_ONLY, CaptureStatus.DISABLED_NO_BACKEND):
        if status is CaptureStatus.WARN_REMOTE_ONLY:
            why = "only a paid remote backend is configured, so capture will not auto-spend"
        else:
            why = "no extraction backend was found"
        return (
            "## Poppy: INACTIVE\n"
            f"Auto-capture is on but {why}, so nothing is being captured. "
            "Install a host CLI (claude / codex / gemini) for free local capture, then `poppy doctor` to verify."
        )

    # ACTIVE / FORCED_ENV — capture is running.
    line = f"## Poppy: active\nRemembering as you work · {_memory_clause(memory_count, project)}"
    if last_session_count is not None:
        noun = "memory" if last_session_count == 1 else "memories"
        line += f" · {last_session_count} {noun} captured last session"
    line += "."
    return line
