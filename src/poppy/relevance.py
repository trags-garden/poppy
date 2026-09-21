"""Cross-encoder relevance-floor helpers.

Poppy's recall surface ranks with a cross-encoder (the default bloom engine),
whose score is a raw ms-marco logit — unbounded and length-sensitive,
so it must not be floored against a magic constant directly. We map the logit to
a bounded relevance *probability* with a logistic (sigmoid) and floor on that.

The floor is default-off (``0.0``): a sigmoid output is always > 0, so a 0.0
floor keeps every candidate — byte-for-byte today's behaviour. A positive floor
drops candidates the cross-encoder scores as only weakly related. This is a
post-retrieve filter: it changes nothing about engine ranking.

Scope note: the fast hooks (UserPromptSubmit / PreToolUse) run on SeedEngine,
whose score is an FTS rank-reciprocal, not a cross-encoder logit — a CE
probability floor is not meaningful there, so this helper is applied on the
CE-scored recall handlers, and the SessionStart recency dump (which has no score
at all) is gated separately. See the PR / issue for the full rationale.
"""

from __future__ import annotations

import math

# Provisional recommended production floor, expressed as a cross-encoder
# relevance probability. NOT yet calibrated: the LongMemEval `_abs` sweep that
# sets the real default is pending an evaluation that has not been published
# yet. 0.2 is deliberately conservative — it corresponds to a CE logit of
# about -1.4, so it drops only clearly-unrelated matches while leaving genuine
# hits (which the cross-encoder scores well above 0.5) untouched. Ship
# default-off (0.0); this is the value to recommend once a store wants
# abstention before calibration.
RECOMMENDED_MIN_SCORE = 0.2


def normalize_ce_score(raw_score: float) -> float:
    """Map a raw cross-encoder logit to a relevance probability in (0, 1)."""
    # Split by sign to keep math.exp away from overflow on large-magnitude logits.
    if raw_score >= 0:
        return 1.0 / (1.0 + math.exp(-raw_score))
    exp_raw = math.exp(raw_score)
    return exp_raw / (1.0 + exp_raw)


def clears_floor(raw_score: float, min_score: float) -> bool:
    """Whether a raw CE logit clears the probability floor ``min_score``.

    A floor <= 0 disables filtering entirely (always True), so the default of
    0.0 is an exact no-op regardless of engine or score scale.
    """
    if min_score <= 0.0:
        return True
    return normalize_ce_score(raw_score) >= min_score
