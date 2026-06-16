"""LLM-assisted conflict detection on remember writes (conflict-detection feature).

Two-stage pipeline:
  1. find_candidates() — semantic retrieval against same-project / same-type memories.
     No LLM cost. If nothing scores above the threshold, conflict detection short-circuits.
  2. detect_conflicts() — pass the new memory + candidates to the consolidation LLM
     and parse a JSON array of {id, confidence, reason}.

The CLI (and MCP) wraps these into three modes via `auto-supersede` config:
  off:     no detection (default, zero LLM cost)
  suggest: detect, write normally, surface candidates as a hint
  auto:    detect, supersede the single high-confidence candidate if any

`AUTO_SUPERSEDE_THRESHOLD` is intentionally conservative — auto-mode should be
silent on ambiguous cases and downgrade to suggest behavior.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from poppy.config import PoppyConfig
from poppy.engine.interface import RetrievalEngine
from poppy.models import Filters, Memory

log = logging.getLogger(__name__)

CANDIDATE_MIN_SCORE = 0.30
AUTO_SUPERSEDE_THRESHOLD = 0.85
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


@dataclass
class Candidate:
    memory: Memory
    score: float


@dataclass
class Conflict:
    memory: Memory
    confidence: float
    reason: str


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


def detect_conflicts(
    engine: RetrievalEngine,
    new_memory: Memory,
    *,
    cfg: PoppyConfig,
    top_k: int = DEFAULT_TOP_K,
) -> list[Conflict]:
    """Run candidates through the consolidation LLM and parse the verdicts.

    Returns [] if no candidates pass the score threshold or the LLM returns
    nothing parseable. The caller decides what to do with non-empty results
    (suggest hint, auto-supersede, etc.) — this function does not write.
    """
    candidates = find_candidates(engine, new_memory, top_k=top_k)
    if not candidates:
        return []

    prompt = _build_prompt(new_memory, candidates)
    try:
        from poppy.consolidation import call_llm
    except Exception:
        log.warning("conflict detection: poppy.consolidation unavailable, skipping LLM call")
        return []

    raw = call_llm(prompt, transcript_path=None, cfg=cfg)
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


def pick_auto_supersede(conflicts: list[Conflict]) -> Conflict | None:
    """Return the single conflict to auto-supersede, or None if ambiguous.

    Auto-mode is intentionally strict: exactly one candidate must clear the
    threshold, otherwise we downgrade to suggest behavior so the human picks.
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
