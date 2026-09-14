"""Unit tests for the cross-encoder relevance-floor helpers."""

import math

from poppy.relevance import RECOMMENDED_MIN_SCORE, clears_floor, normalize_ce_score


def test_normalize_maps_logit_to_probability():
    assert normalize_ce_score(0.0) == 0.5
    # Large positive/negative logits saturate toward 1 / 0 without overflow.
    assert normalize_ce_score(20.0) > 0.999
    assert normalize_ce_score(-20.0) < 0.001
    # Matches the logistic function.
    assert math.isclose(normalize_ce_score(2.0), 1 / (1 + math.exp(-2.0)))


def test_floor_off_is_a_no_op_even_for_very_negative_scores():
    # A floor of 0 (default) keeps everything, regardless of score scale/sign.
    assert clears_floor(-1000.0, 0.0) is True
    assert clears_floor(0.0, 0.0) is True
    # Negative floors are also treated as off.
    assert clears_floor(-5.0, -0.1) is True


def test_floor_suppresses_weak_and_keeps_strong():
    # sigmoid(5) ~= 0.993 clears a 0.9 floor; sigmoid(-5) ~= 0.0067 does not.
    assert clears_floor(5.0, 0.9) is True
    assert clears_floor(-5.0, 0.5) is False
    # Boundary: sigmoid(0) == 0.5 exactly clears a 0.5 floor.
    assert clears_floor(0.0, 0.5) is True


def test_recommended_default_is_a_conservative_probability():
    assert 0.0 < RECOMMENDED_MIN_SCORE < 0.5
