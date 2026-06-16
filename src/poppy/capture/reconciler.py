"""CaptureReconciler — dedup-on-capture before ingest (ADR-0003).

Background capture produces many candidate memories with no human in the loop.
Before any candidate is written, the reconciler decides — fully autonomously —
whether to ADD it, SUPERSEDE an existing memory, or SKIP it as a duplicate.

Three tiers, cheapest first (ADR-0003):

1. **Cheap prefilter** (no LLM): a normalized lexical-similarity score against
   the same-project / same-type candidates already in the store.
   - ``sim >= DUPLICATE_THRESHOLD``  → SKIP (clear duplicate).
   - no candidate ``sim >= AMBIGUOUS_MIN_SIM`` → ADD (clearly new).
2. **LLM verdict** (the shipped ``detect_conflicts`` routine) resolves only the
   ambiguous middle band.
   - a single candidate clears ``AUTO_SUPERSEDE_THRESHOLD`` → SUPERSEDE.
   - otherwise → ADD.
3. **Bias to ADD**: whenever the verdict is uncertain we ADD rather than
   supersede — adding never loses information.

The safety net for a wrong SUPERSEDE is reversibility, not review: supersede
tombstones the old memory through the same 7-day window as a manual delete
(``poppy.lifecycle.supersede_memory``), so a wrong merge is recoverable.

The lexical prefilter is deliberately engine-agnostic (it works on the baseline
FTS engine with no embeddings). A future autoresearch pass tunes the thresholds
against a labelled dedup-quality set (ADR-0003); until then they are
conservative — bias to ADD.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path

from poppy.config import PoppyConfig
from poppy.conflict_detection import detect_conflicts, find_candidates, pick_auto_supersede
from poppy.engine.interface import RetrievalEngine
from poppy.lifecycle import supersede_memory
from poppy.models import Memory

# A candidate at or above this lexical similarity to an existing same-type memory
# is a clear duplicate — SKIP without spending an LLM call.
DUPLICATE_THRESHOLD = 0.97

# Below this, the candidate shares too little with anything in the store to be a
# conflict — ADD without spending an LLM call. The band in between
# (AMBIGUOUS_MIN_SIM .. DUPLICATE_THRESHOLD) is where the LLM verdict runs.
AMBIGUOUS_MIN_SIM = 0.50

# How many same-project / same-type neighbours the prefilter inspects.
DEFAULT_TOP_K = 5


class Action(str, Enum):
    ADD = "add"
    SUPERSEDE = "supersede"
    SKIP = "skip"


@dataclass
class Decision:
    """What the reconciler chose for a single candidate memory."""

    action: Action
    memory: Memory
    # For SUPERSEDE: the existing memory id replaced. For SKIP: the duplicate hit.
    target_id: str | None = None
    reason: str = ""


@dataclass
class ReconcileSummary:
    """Aggregate outcome of reconciling a batch of candidates."""

    added: int = 0
    superseded: int = 0
    skipped: int = 0
    decisions: list[Decision] = field(default_factory=list)

    @property
    def stored(self) -> int:
        """Memories newly written to the store (ADD + SUPERSEDE)."""
        return self.added + self.superseded


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _similarity(a: str, b: str) -> float:
    """Cheap, deterministic, engine-agnostic lexical similarity in [0, 1]."""
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def decide(
    engine: RetrievalEngine,
    candidate: Memory,
    *,
    cfg: PoppyConfig,
    top_k: int = DEFAULT_TOP_K,
) -> Decision:
    """Decide ADD / SUPERSEDE / SKIP for one candidate. Does not write."""
    neighbours = find_candidates(engine, candidate, top_k=top_k, min_score=0.0)

    best_sim = 0.0
    best_id: str | None = None
    for n in neighbours:
        sim = _similarity(candidate.content, n.memory.content)
        if sim > best_sim:
            best_sim, best_id = sim, n.memory.id

    # Tier 1a: clear duplicate — skip without an LLM call.
    if best_sim >= DUPLICATE_THRESHOLD:
        return Decision(Action.SKIP, candidate, target_id=best_id, reason="duplicate")

    # Tier 1b: clearly new — nothing close enough to conflict, add without an LLM call.
    if best_sim < AMBIGUOUS_MIN_SIM:
        return Decision(Action.ADD, candidate, reason="novel")

    # Tier 2: ambiguous band — let the LLM judge. Only a single high-confidence
    # contradiction/replacement supersedes; everything else biases to ADD.
    conflicts = detect_conflicts(engine, candidate, cfg=cfg, top_k=top_k)
    pick = pick_auto_supersede(conflicts)
    if pick is not None:
        return Decision(
            Action.SUPERSEDE,
            candidate,
            target_id=pick.memory.id,
            reason=pick.reason or "supersedes prior memory",
        )
    return Decision(Action.ADD, candidate, reason="uncertain — bias to add")


def reconcile_and_ingest(
    memories: list[Memory],
    *,
    engine: RetrievalEngine,
    cfg: PoppyConfig,
    poppy_dir: Path,
    top_k: int = DEFAULT_TOP_K,
) -> ReconcileSummary:
    """Reconcile each candidate against the store, then apply the decision.

    Candidates are processed in order and written as they are decided, so a
    later candidate in the same batch can dedup against an earlier one that was
    just added (overlapping capture windows must not store the same decision
    twice). Returns a summary; callers use ``summary.stored`` for the count.
    """
    summary = ReconcileSummary()
    for mem in memories:
        decision = decide(engine, mem, cfg=cfg, top_k=top_k)
        summary.decisions.append(decision)
        if decision.action is Action.ADD:
            engine.ingest(mem)
            summary.added += 1
        elif decision.action is Action.SUPERSEDE and decision.target_id:
            supersede_memory(engine, mem, decision.target_id, poppy_dir=poppy_dir)
            summary.superseded += 1
        else:  # SKIP
            summary.skipped += 1
    return summary
