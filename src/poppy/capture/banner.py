"""SessionStart status banner.

A one-line banner prepended to the SessionStart context so the developer can see
whether auto-capture is working. It reads the capture status (ConsolidationPolicy,
with the consent and default-on precedence rule), the project memory count
(engine), and the last-session capture count
(CaptureJournal) — the three signals that make silent background capture
observable rather than invisible.

The render is a pure function so the active / INACTIVE / consent-pending wording is
testable without a live agent session. Three rules drive it:

* **Active** (capture is running): a reassuring line with the memory count and how
  many memories the last session captured — proof the loop runs.
* **INACTIVE** (consented but broken — no backend, remote-only, a host CLI whose
  every extraction fails, or the engine failed to load): a *loud* line, because
  this is the silent-breakage case the banner exists to catch (a
  partially-installed Poppy that quietly captures nothing for weeks).
* **Consent pending**: a nudge pointing at ``poppy autocapture on --global`` — never a
  misleading "0 captured", since nothing is captured until consent is recorded.

A deliberate off-state (explicit opt-out or a ``POPPY_CONSOLIDATE=0`` override) is
silent: the developer made that choice, so the banner does not nag every session.
"""

from __future__ import annotations

from poppy.capture.policy import CaptureStatus

# Statuses that mean "the developer turned this off on purpose" — no banner.
_SILENT_STATUSES = frozenset({CaptureStatus.DISABLED_OPT_OUT, CaptureStatus.DISABLED_ENV})

# Statuses whose banner the developer must actually *see*, not just have injected
# into agent context: the consent-pending nudge and the two INACTIVE
# breakage cases. These are surfaced through the hook `systemMessage` channel so
# Claude Code renders them to the human — the ACTIVE line is reassurance only and
# stays in agent context, and the deliberate off-states show nothing at all. A
# failed engine is user-facing too, but that path is decided by the caller
# (`engine_ok`) since it does not correspond to a CaptureStatus.
_USER_FACING_STATUSES = frozenset(
    {
        CaptureStatus.INERT_PENDING,
        CaptureStatus.WARN_REMOTE_ONLY,
        CaptureStatus.WARN_BACKEND_BROKEN,
        CaptureStatus.DISABLED_NO_BACKEND,
    }
)


def is_user_facing_status(status: CaptureStatus) -> bool:
    """Whether this status's banner must reach the human via ``systemMessage``."""
    return status in _USER_FACING_STATUSES


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
    backend_cli: str | None = None,
) -> str | None:
    """Render the SessionStart banner line, or ``None`` when nothing should show.

    ``last_session_count`` is the journal's total for the most recent session that
    captured anything (``None`` when nothing has ever been captured).
    ``backend_cli`` names the failing extraction backend for the
    ``WARN_BACKEND_BROKEN`` line.
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

    if status is CaptureStatus.DISABLED_PROJECT:
        # A deliberate per-project off switch. Shown so the state is
        # visible, but not user-facing (no systemMessage nag): the developer chose
        # this, exactly like the opt-out, so it stays in agent context only.
        return (
            "## Poppy\n"
            f"Automatic capture is off for {_scope(project)}. Recall still works. "
            "Run `poppy autocapture on` here to re-enable capturing in this project."
        )

    if status is CaptureStatus.INERT_PENDING:
        mem = _memory_clause(memory_count, project)
        return (
            "## Poppy\n"
            f"Automatic capture is pending your consent; nothing is captured yet ({mem}). "
            "Run `poppy autocapture on --global` to turn it on, "
            "or `poppy autocapture off --global` to dismiss."
        )

    if status is CaptureStatus.WARN_BACKEND_BROKEN:
        # The host CLI exists but every extraction fails (a logged-out CLI, say):
        # the exact "quietly captures nothing for weeks" case the banner is for.
        backend = f"`{backend_cli}`" if backend_cli else "your host CLI"
        return (
            "## Poppy: INACTIVE\n"
            f"Auto-capture is on but every extraction through {backend} is failing, "
            "so nothing is being captured. Check that the CLI runs and is logged in, "
            "then `poppy doctor` for the error."
        )

    if status in (CaptureStatus.WARN_REMOTE_ONLY, CaptureStatus.DISABLED_NO_BACKEND):
        if status is CaptureStatus.WARN_REMOTE_ONLY:
            why = "only a paid remote backend is configured, so capture will not auto-spend"
        else:
            why = "no extraction backend was found"
        return (
            "## Poppy: INACTIVE\n"
            f"Auto-capture is on but {why}, so nothing is being captured. "
            "Install a host CLI (claude / cursor-agent / codex / gemini) for free local capture, "
            "then `poppy doctor` to verify."
        )

    # ACTIVE / FORCED_ENV — capture is running.
    line = f"## Poppy: active\nRemembering as you work · {_memory_clause(memory_count, project)}"
    if last_session_count is not None:
        noun = "memory" if last_session_count == 1 else "memories"
        line += f" · {last_session_count} {noun} captured last session"
    line += "."
    return line
