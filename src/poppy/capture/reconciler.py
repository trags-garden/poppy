"""CaptureReconciler — ADR-0003's automatic reconciliation decision brain in
one seam.

Background capture produces many candidate memories with no human in the loop.
Before any candidate is written, the reconciler decides — fully autonomously —
whether to ADD it, SUPERSEDE an existing memory, or SKIP it as a duplicate.

This module owns the whole decision brain behind the `reconcile_and_ingest`
seam: the two-tier decision, *all* of its tunable thresholds, the LLM verdict
(prompt + parse), candidate retrieval, and apply/tombstone. ADR-0003 designates
the supersede-confidence threshold for automatic-reconciliation tuning, so it lives here with
the rest — one module, one set of tunables to point the tuning pass at.

ADR-0003's three automatic-reconciliation tiers, cheapest first:

1. **Cheap lexical prefilter** (no LLM): a normalized `difflib` similarity score
   against the same-project / same-type candidates already in the store.
   - ``sim >= DUPLICATE_THRESHOLD``  → SKIP (clear duplicate).
   - no candidate ``sim >= AMBIGUOUS_MIN_SIM`` → ADD (clearly new).
2. **LLM verdict** resolves only the ambiguous middle band, reusing the neighbours
   the prefilter already fetched (one retrieval pass per candidate).
   - a single candidate clears ``AUTO_SUPERSEDE_THRESHOLD`` → SUPERSEDE.
   - otherwise → ADD.
3. **Bias to ADD**: whenever the verdict is uncertain we ADD rather than
   supersede — adding never loses information.

The safety net for a wrong SUPERSEDE is reversibility, not review: supersede
tombstones the old memory through the same 7-day window as a manual delete
(``poppy.lifecycle.supersede_memory``), so a wrong merge is recoverable.

The interactive remember path (the ``poppy remember`` CLI and the MCP ``remember``
tool) consumes the same verdict interface (``detect_conflicts`` /
``pick_auto_supersede``) on purpose, rather than sharing internals by accident.

The lexical prefilter is deliberately engine-agnostic (it works on the baseline
FTS engine with no embeddings); embeddings only enter at the neighbour-fetch
(`find_candidates`, via the engine's `retrieve`). A future tuning pass will use
a labelled dedup-quality set under ADR-0003; until then
they are conservative — bias to ADD.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path

from poppy.config import PoppyConfig
from poppy.engine.interface import RetrievalEngine
from poppy.lifecycle import supersede_memory
from poppy.models import Filters, Memory

log = logging.getLogger(__name__)

# --- Tier-1 lexical prefilter thresholds ---
# A candidate at or above this lexical similarity to an existing same-type memory
# is a clear duplicate — SKIP without spending an LLM call.
DUPLICATE_THRESHOLD = 0.97

# Below this, the candidate shares too little with anything in the store to be a
# conflict — ADD without spending an LLM call. The band in between
# (AMBIGUOUS_MIN_SIM .. DUPLICATE_THRESHOLD) is where the LLM verdict runs.
AMBIGUOUS_MIN_SIM = 0.50

# --- Tier-2 LLM verdict thresholds ---
# Minimum retrieval score for a neighbour to be worth showing the LLM verdict.
CANDIDATE_MIN_SCORE = 0.30

# In the ambiguous band, exactly one candidate must clear this confidence for an
# auto-supersede; otherwise we bias to ADD. Intentionally conservative — the
# ADR-0003's automatic-reconciliation tuning pass points here.
AUTO_SUPERSEDE_THRESHOLD = 0.85

# Seconds a verdict gets from the backend, well under the whole-transcript
# default. A verdict judges one memory against at most DEFAULT_TOP_K neighbours,
# so it is a small prompt and a slow answer is not worth waiting for. The budget
# has to stay small: a background capture pass runs one verdict per extracted
# candidate while it holds the per-session capture lock, and that lock is treated
# as abandoned after capture.lock.LOCK_TTL_S (300s), at which point a second
# worker steals it from the one still running. One extraction (120s) plus a
# verdict for every candidate in a default batch of five stays inside that
# window, because a budget covers a whole call_llm rather than each backend it
# tries. It also keeps `remember --check-conflicts` from parking a terminal.
CONFLICT_LLM_TIMEOUT_S = 20

# How many same-project / same-type neighbours the prefilter / verdict inspect.
DEFAULT_TOP_K = 5


CONFLICT_PROMPT = """\
You evaluate whether a NEW memory replaces or contradicts EXISTING memories.

NEW memory ({memory_type}{project_clause}):
"{new_content}"

EXISTING memories (same type and project, ranked by similarity):
{candidates_block}

Return a strict JSON array (no prose, no markdown). One entry per existing
memory that is replaced or directly contradicted by NEW. Skip memories that
merely cover the same topic but state a different fact about a different thing.

Each entry: {{"id": "<memory id>", "confidence": <0.0-1.0>, "reason": "<short>"}}

If no existing memory is replaced/contradicted, return [].
"""


class Action(str, Enum):
    ADD = "add"
    SUPERSEDE = "supersede"
    SKIP = "skip"


@dataclass
class Candidate:
    memory: Memory
    score: float


@dataclass
class Conflict:
    memory: Memory
    confidence: float
    reason: str


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


# ---------------------------------------------------------------------------
# Tier-2 verdict: candidate retrieval + LLM judgement
# ---------------------------------------------------------------------------


def find_candidates(
    engine: RetrievalEngine,
    new_memory: Memory,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = CANDIDATE_MIN_SCORE,
) -> list[Candidate]:
    """Same project + type candidates: retrieve()-ranked first, recent-fallback after.

    No LLM cost. The retrieve() pre-rank narrows the LLM's job; if the engine's
    text matcher (e.g. baseline FTS5 phrase-only) returns nothing, we top up
    with the most recent same-project / same-type memories so the LLM still
    gets a chance to spot a conflict. Filters out the new memory itself.
    """
    filters = Filters(project=new_memory.project, memory_type=new_memory.memory_type)
    seen: set[str] = {new_memory.id}
    out: list[Candidate] = []

    try:
        scored = engine.retrieve(new_memory.content, filters=filters, limit=top_k * 2)
    except Exception:
        scored = []
    for s in scored:
        if s.memory.id in seen:
            continue
        if s.score is not None and s.score < min_score:
            continue
        out.append(Candidate(memory=s.memory, score=s.score if s.score is not None else 0.0))
        seen.add(s.memory.id)
        if len(out) >= top_k:
            return out

    # Top up from list_all when retrieve() didn't yield enough; happens on
    # phrase-only engines (baseline) where the new content shares no contiguous
    # span with existing entries.
    try:
        recent = engine.list_all(filters=filters, limit=top_k * 4)
    except Exception:
        recent = []
    for m in recent:
        if m.id in seen:
            continue
        out.append(Candidate(memory=m, score=0.0))
        seen.add(m.id)
        if len(out) >= top_k:
            break
    return out


def _verdict_from_candidates(
    new_memory: Memory,
    candidates: list[Candidate],
    *,
    cfg: PoppyConfig,
) -> list[Conflict]:
    """Run pre-fetched candidates through the consolidation LLM and parse verdicts.

    Split out from ``detect_conflicts`` so the reconciler can reuse the neighbours
    it already fetched for the lexical prefilter — one ``engine.retrieve`` pass
    per candidate instead of two. Returns [] if there are no candidates or the
    LLM returns nothing parseable. Does not write.
    """
    if not candidates:
        return []

    prompt = _build_prompt(new_memory, candidates)
    try:
        from poppy.consolidation import call_llm
    except Exception:
        log.warning("conflict detection: poppy.consolidation unavailable, skipping LLM call")
        return []

    # A verdict never speaks for the extraction backend's health. On this much
    # shorter timeout a slow-but-working CLI would look broken, and three of
    # those in one capture pass is enough to put a false "every extraction is
    # failing" line in front of someone whose capture is fine. The reverse is
    # just as wrong: a fast verdict would clear a genuinely broken CLI's record.
    # The failure is still logged, just not counted.
    raw = call_llm(
        prompt,
        transcript_path=None,
        cfg=cfg,
        parser=parse_llm_response,
        host_timeout_s=CONFLICT_LLM_TIMEOUT_S,
        record_health=False,
    )
    if not raw:
        return []

    by_id = {c.memory.id: c.memory for c in candidates}
    conflicts: list[Conflict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        mid = str(entry.get("id", "")).strip()
        if mid not in by_id:
            continue
        try:
            confidence = float(entry.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        confidence = max(0.0, min(1.0, confidence))
        reason = str(entry.get("reason", "")).strip()
        conflicts.append(Conflict(memory=by_id[mid], confidence=confidence, reason=reason))

    conflicts.sort(key=lambda c: c.confidence, reverse=True)
    return conflicts


def detect_conflicts(
    engine: RetrievalEngine,
    new_memory: Memory,
    *,
    cfg: PoppyConfig,
    top_k: int = DEFAULT_TOP_K,
) -> list[Conflict]:
    """Fetch neighbours and run them through the LLM verdict.

    The interface the interactive remember path (the ``poppy remember`` CLI and
    the MCP ``remember`` tool) consumes for its suggest / auto-supersede modes.
    The reconciler's own ``decide`` does not call this — it reuses the neighbours
    it already fetched (see ``_verdict_from_candidates``) to avoid a second
    retrieval.
    """
    candidates = find_candidates(engine, new_memory, top_k=top_k)
    return _verdict_from_candidates(new_memory, candidates, cfg=cfg)


def pick_auto_supersede(conflicts: list[Conflict]) -> Conflict | None:
    """Return the single conflict to auto-supersede, or None if ambiguous.

    Auto-supersede is intentionally strict: exactly one candidate must clear the
    threshold, otherwise we downgrade to suggest / bias-to-ADD behavior so a
    human (interactive path) or the ADD default (reconcile path) wins.
    """
    high = [c for c in conflicts if c.confidence >= AUTO_SUPERSEDE_THRESHOLD]
    if len(high) == 1:
        return high[0]
    return None


def _build_prompt(new_memory: Memory, candidates: list[Candidate]) -> str:
    project_clause = f", project={new_memory.project}" if new_memory.project else ""
    candidates_block = "\n".join(
        f'- id={c.memory.id} score={c.score:.2f} content="{_oneline(c.memory.content)}"' for c in candidates
    )
    return CONFLICT_PROMPT.format(
        memory_type=new_memory.memory_type,
        project_clause=project_clause,
        new_content=_oneline(new_memory.content),
        candidates_block=candidates_block,
    )


def _oneline(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().replace('"', "'")[:600]


def parse_llm_response(text: str) -> list[dict]:
    """Resilient parse of an LLM JSON-array response. Public for testing."""
    text = text.strip()
    if text.startswith("```"):
        # Strip markdown fences if the LLM wrapped them.
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    return data if isinstance(data, list) else []


# ---------------------------------------------------------------------------
# Tier-1 lexical prefilter + the ADD / SUPERSEDE / SKIP decision
# ---------------------------------------------------------------------------


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
    # One retrieval pass: fetch neighbours at min_score=0.0 so the lexical
    # prefilter can catch a near-identical duplicate the engine happened to score
    # low, and reuse the same neighbours for the LLM verdict below.
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

    # Tier 2: ambiguous band — let the LLM judge, reusing the neighbours already
    # fetched (no second engine.retrieve). Only a single high-confidence
    # contradiction/replacement supersedes; everything else biases to ADD.
    conflicts = _verdict_from_candidates(candidate, neighbours, cfg=cfg)
    pick = pick_auto_supersede(conflicts)
    if pick is not None:
        return Decision(
            Action.SUPERSEDE,
            candidate,
            target_id=pick.memory.id,
            reason=pick.reason or "supersedes prior memory",
        )
    return Decision(Action.ADD, candidate, reason="uncertain, bias to add")


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
        # Candidates arrive already secret-redacted from orchestrator.build_capture_memories
        # — redaction happens at construction, the single earliest seam, so
        # both the store and the local capture journal only ever see masked content.
        decision = decide(engine, mem, cfg=cfg, top_k=top_k)
        summary.decisions.append(decision)
        if decision.action is Action.ADD:
            engine.ingest(mem)
            summary.added += 1
        elif decision.action is Action.SUPERSEDE and decision.target_id:
            try:
                supersede_memory(engine, mem, decision.target_id, poppy_dir=poppy_dir)
            except (ValueError, KeyError) as exc:
                # The target turned out not to be supersedable — a row
                # deleted since it was picked, say. Skip
                # THIS item and keep going: aborting mid-loop would drop every
                # remaining capture in the batch on the floor. Nothing is
                # swallowed silently, and nothing is written for this one.
                log.warning("supersede skipped for %s: %s", decision.target_id, exc)
                summary.skipped += 1
                continue
            summary.superseded += 1
        else:  # SKIP
            summary.skipped += 1
    return summary
