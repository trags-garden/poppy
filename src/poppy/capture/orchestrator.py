"""CaptureOrchestrator — the one place the capture pipeline's tail lives (ADR-0001).

The three capture entry points (mid-session, SessionEnd backstop, PostCompact)
used to each copy-paste the same orchestration tail:

    LLM extract → build candidates → reconcile & ingest
      → (on success) advance the watermark → count the soft cap → journal

ADR-0001's core invariant — *the watermark advances only after a successful
ingest* — was a call-ordering convention hand-replicated at each site, i.e.
several separate chances to reorder it. This module owns that sequence once:

* ``CaptureOrchestrator.run(plan)`` runs the tail in the one guarded order.
* Collaborators (engine, LLM, reconcile) are constructor parameters, the
  injection style the engine layer already uses — so the pipeline is testable
  end-to-end with a fake transcript + fake LLM, no module-global monkeypatching.

Each entry point keeps its own *gating* (single-flight lock, watermark window,
soft cap, min-content, idempotency scan) — those are genuinely per-entry — then
resolves a ``CapturePlan`` and hands it here.

Anything a plan does not enable is skipped: ``advance_watermark_to=None`` means
"no watermark" (PostCompact keys idempotency off the composite event id
instead), ``record_soft_cap=False`` means "not a mid-session fire", and
``journal=False`` means "don't journal".
"""

from __future__ import annotations

import datetime
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from poppy.capture import journal as _journal
from poppy.capture.cadence import record_capture as _record_capture
from poppy.capture.reconciler import ReconcileSummary, reconcile_and_ingest
from poppy.capture.redaction import load_custom_redaction, redact_secrets
from poppy.capture.watermark import set_watermark as _set_watermark
from poppy.models import Memory, Source
from poppy.write_flow import make_memory_id

if TYPE_CHECKING:
    from poppy.config import PoppyConfig
    from poppy.engine.interface import RetrievalEngine

# Confidence stamped on auto-captured candidates before reconciliation. Lower than
# a manual write (1.0) — these are machine-extracted and go through the
# CaptureReconciler (ADR-0003) before ingest.
CAPTURE_CONFIDENCE = 0.7

# Signature of the LLM extraction backend (poppy.consolidation.call_llm).
LlmFn = Callable[..., list[dict[str, str]]]
# Signature of the reconcile-and-ingest step.
ReconcileFn = Callable[..., ReconcileSummary]


def build_capture_memories(
    items: list[dict[str, str]],
    *,
    source_type: str,
    session_id: str | None,
    project: str | None,
    config: "PoppyConfig | None" = None,
) -> list[Memory]:
    """Turn extracted ``{type, content}`` dicts into Memory candidates with provenance.

    Provenance (source app + session id) is stamped here so every captured memory
    is auditable, in exactly one place. The candidates are not written directly —
    they go through the CaptureReconciler (ADR-0003) before ingest.

    Secret redaction happens here, at construction, so a captured
    credential never exists as a Memory in raw form. This is the single, earliest
    seam: every downstream consumer — the reconciler/store, sync, and the local
    capture journal (which records an 80-char preview of each candidate, in
    plaintext, *before* the reconciler runs) — therefore only ever sees the masked
    content. Redacting later (e.g. only inside the reconciler) would leave the
    journal preview holding the raw secret.
    """
    # Custom redaction config is an optional safety layer, never a capture
    # dependency. The orchestrator threads its already-loaded config; direct
    # callers fall back to a load from runtime.get_poppy_dir(), including its
    # POPPY_DIR override.
    try:
        if config is None:
            from poppy.config import load_config  # noqa: PLC0415
            from poppy.runtime import get_poppy_dir  # noqa: PLC0415

            config = load_config(get_poppy_dir())
        extra_patterns, _issues = load_custom_redaction(config)
    except Exception:
        extra_patterns = ()

    now = datetime.datetime.now(datetime.UTC)
    return [
        Memory(
            id=make_memory_id(),
            content=redact_secrets(item["content"], extra_patterns),
            memory_type=item["type"],
            source=Source(type=source_type, session_id=session_id, timestamp=now),
            project=project,
            related_to=[],
            created_at=now,
            updated_at=now,
            confidence=CAPTURE_CONFIDENCE,
        )
        for item in items
    ]


@dataclass
class CapturePlan:
    """A fully-resolved capture, ready to extract → ingest → record.

    Built by an entry point after its own gating; consumed by
    ``CaptureOrchestrator.run``. The three post-ingest flags map onto the parts
    of the tail that differ between entry points.
    """

    session_id: str
    prompt: str
    source_type: str
    project: str | None = None
    transcript_path: str | None = None  # host-CLI backend detection for the LLM call
    max_items: int = 5
    # ADR-0001: the new watermark to advance to *after* a successful ingest, or
    # None for entry points that do not use the watermark (PostCompact).
    advance_watermark_to: int | None = None
    # Count this capture against the per-session mid-session soft cap.
    record_soft_cap: bool = False
    # Journal the capture (SessionStart banner / `poppy doctor` visibility).
    journal: bool = True


@dataclass
class CaptureOutcome:
    """Result of a capture run: what was stored and the candidates built."""

    stored: int = 0
    candidates: list[Memory] = field(default_factory=list)


class CaptureOrchestrator:
    """Runs the shared capture tail with the ADR-0001 ordering in one place.

    Collaborators are injected so the pipeline is testable without patching
    module globals. ``llm`` and ``reconcile`` default to the production
    implementations, resolved lazily so importing this module never drags in
    ``poppy.consolidation`` (which imports this one).
    """

    def __init__(
        self,
        *,
        engine: "RetrievalEngine",
        cfg: "PoppyConfig",
        poppy_dir: Path,
        llm: LlmFn | None = None,
        reconcile: ReconcileFn | None = None,
    ) -> None:
        self._engine = engine
        self._cfg = cfg
        self._poppy_dir = poppy_dir
        self._llm = llm
        self._reconcile = reconcile if reconcile is not None else reconcile_and_ingest

    def _extract(self, plan: CapturePlan) -> list[dict[str, str]]:
        llm = self._llm
        if llm is None:
            # Lazy default: resolve the attribute at call time so tests that patch
            # `poppy.consolidation.call_llm` are still honored.
            from poppy import consolidation

            llm = consolidation.call_llm
        return llm(plan.prompt, transcript_path=plan.transcript_path, cfg=self._cfg)

    def run(self, plan: CapturePlan) -> CaptureOutcome:
        """Extract → build → ingest, then advance the watermark / soft cap /
        journal — the watermark step only ever after a successful ingest."""
        extracted = self._extract(plan)
        if not extracted:
            return CaptureOutcome(stored=0, candidates=[])

        candidates = build_capture_memories(
            extracted[: plan.max_items],
            source_type=plan.source_type,
            session_id=plan.session_id,
            project=plan.project,
            config=self._cfg,
        )
        summary = self._reconcile(candidates, engine=self._engine, cfg=self._cfg, poppy_dir=self._poppy_dir)

        # --- ADR-0001 post-ingest sequence, in exactly one place ---
        # Advancing the watermark only here guarantees a failed extract/ingest
        # (which returned above, or raised before this line) leaves the turns for
        # the next fire / backstop to re-cover.
        if plan.advance_watermark_to is not None:
            _set_watermark(self._poppy_dir, plan.session_id, plan.advance_watermark_to)
        if plan.record_soft_cap:
            _record_capture(self._poppy_dir, plan.session_id)
        if plan.journal:
            _journal.record(
                self._poppy_dir,
                session_id=plan.session_id,
                project=plan.project,
                count=summary.stored,
                items=candidates,
                source=plan.source_type,
            )
        return CaptureOutcome(stored=summary.stored, candidates=candidates)


def _emit_first_autocapture(poppy_dir: Path, trigger: str) -> None:
    """Record the ``first_autocapture_stored`` funnel milestone.

    Fired by a capture worker after it stores at least one memory. Latched once per
    device (the first auto-capture is the funnel stage that matters); ``trigger``
    is the content-free capture path enum. Never raises.
    """
    try:
        from poppy import telemetry

        telemetry.capture_once(poppy_dir, "first_autocapture_stored", {"trigger": trigger})
    except Exception as exc:
        sys.stderr.write(f"poppy first-autocapture telemetry error: {exc}\n")


def run_capture_worker(
    consolidate_fn: Callable[[dict], int],
    payload: dict,
    *,
    milestone: str,
    poppy_dir: Path | None = None,
    trigger: Callable[[Path], None] | None = None,
    emit_milestone: Callable[[Path, str], None] | None = None,
) -> int:
    """Run a capture entry point, then trigger autosync + the funnel milestone
    iff it stored anything.

    The three detached hook workers (mid-session / PostCompact / SessionEnd) all
    share this shape: consolidate → if anything was stored, push it and record
    the ``first_autocapture_stored`` milestone. ``milestone`` is the
    worker's content-free trigger enum (``mid_session`` / ``post_compact`` /
    ``session_end``). ``trigger`` and ``emit_milestone`` are injectable so the
    "stored > 0 → autosync + funnel emit" branch is reachable in tests without
    spawning the real detached sync worker or touching telemetry.
    """
    n = consolidate_fn(payload)
    if n > 0:
        if poppy_dir is None:
            from poppy.runtime import get_poppy_dir

            poppy_dir = get_poppy_dir()
        if trigger is None:
            from poppy.sync.auto import trigger as _default_trigger

            trigger = _default_trigger
        trigger(poppy_dir)
        (emit_milestone or _emit_first_autocapture)(poppy_dir, milestone)
    return n
