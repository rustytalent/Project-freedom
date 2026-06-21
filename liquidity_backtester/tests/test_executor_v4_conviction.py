"""Tests for the continuous conviction score (signal strength)."""
from __future__ import annotations

import json

import pytest

from liqpool.research.belief.executor_v4 import (
    ConvictionConfig,
    compute_conviction,
)
from liqpool.research.belief.executor_v4.conviction import (
    CONVICTION_NEUTRAL,
    CONVICTION_STRONG_BULL,
    CONVICTION_STRONG_BEAR,
    CONVICTION_WEAK_BULL,
)


def test_strong_bull_when_all_inputs_high():
    c = compute_conviction(
        direction=+1, confidence=0.9,
        web_directional_consensus=0.8,
        mtf_alignment_score=0.85, antithesis_score=0.05,
    )
    assert c.label == CONVICTION_STRONG_BULL
    assert c.signed_value > 0.7
    assert c.size_multiplier > 0.9


def test_weak_bull_when_inputs_modest():
    c = compute_conviction(
        direction=+1, confidence=0.45,
        web_directional_consensus=0.10,
        mtf_alignment_score=0.20, antithesis_score=0.50,
    )
    assert c.label in (CONVICTION_WEAK_BULL, CONVICTION_NEUTRAL)
    assert c.size_multiplier < 0.7


def test_strong_bear_signed_negative():
    c = compute_conviction(
        direction=-1, confidence=0.9,
        web_directional_consensus=-0.8,
        mtf_alignment_score=0.85, antithesis_score=0.05,
    )
    assert c.label == CONVICTION_STRONG_BEAR
    assert c.signed_value < -0.7


def test_web_disagreement_penalizes_magnitude():
    """A bull trade gets NO web credit if consensus is bearish."""
    agree = compute_conviction(
        direction=+1, confidence=0.8,
        web_directional_consensus=+0.7,
        mtf_alignment_score=0.6, antithesis_score=0.1,
    )
    disagree = compute_conviction(
        direction=+1, confidence=0.8,
        web_directional_consensus=-0.7,    # web says bearish
        mtf_alignment_score=0.6, antithesis_score=0.1,
    )
    assert disagree.magnitude < agree.magnitude
    assert any("DISAGREE" in n for n in disagree.notes)


def test_strong_signal_sizes_bigger_than_weak():
    strong = compute_conviction(
        direction=+1, confidence=0.9, web_directional_consensus=0.8,
        mtf_alignment_score=0.85, antithesis_score=0.05)
    weak = compute_conviction(
        direction=+1, confidence=0.4, web_directional_consensus=0.1,
        mtf_alignment_score=0.2, antithesis_score=0.5)
    # The founder's exact ask: strong signal → bigger size.
    assert strong.size_multiplier > weak.size_multiplier


def test_direction_sign_preserved_unchanged():
    """Conviction must NOT alter the discrete direction contract."""
    c = compute_conviction(direction=+1, confidence=0.5)
    assert c.direction == +1
    c2 = compute_conviction(direction=-1, confidence=0.5)
    assert c2.direction == -1


def test_neutral_when_direction_zero():
    c = compute_conviction(direction=0, confidence=0.9,
                            web_directional_consensus=0.0)
    assert c.label == CONVICTION_NEUTRAL


def test_conviction_serializable():
    c = compute_conviction(direction=+1, confidence=0.7,
                            web_directional_consensus=0.5,
                            mtf_alignment_score=0.6, antithesis_score=0.2)
    json.dumps(c.to_dict())


def test_conviction_config_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        ConvictionConfig(w_confidence=0.9, w_web_consensus=0.9,
                          w_mtf_alignment=0.9, w_anti_skepticism=0.9)


def test_size_multiplier_bounded():
    # Even max inputs cap at 1.0; min inputs floor at 0.25.
    hi = compute_conviction(direction=+1, confidence=1.0,
                             web_directional_consensus=1.0,
                             mtf_alignment_score=1.0, antithesis_score=0.0)
    lo = compute_conviction(direction=+1, confidence=0.0,
                             web_directional_consensus=0.0,
                             mtf_alignment_score=0.0, antithesis_score=1.0)
    assert hi.size_multiplier <= 1.0
    assert lo.size_multiplier >= 0.25
