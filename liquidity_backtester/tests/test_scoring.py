"""Tests for the opaque, non-invertible public scoring layer.

These are the IP tripwire: they prove G/D are rank-based and opaque, that the
public record exposes only allow-listed neutral fields, and that no internal
estimate or revealing name can leak.
"""
from __future__ import annotations

import json

import pytest

from liqpool import scoring
from liqpool.scoring import (
    COMPLIANCE_TAG,
    COMPLIANCE_TEXT_KEYS,
    PUBLIC_KEYS,
    FORBIDDEN_PUBLIC_TOKENS,
    InternalLevel,
    g_scores,
    feature_states,
    public_record,
    score_levels,
)


def _levels(n=12, side="above"):
    out = []
    for i in range(n):
        frac = (i + 1) / (n + 1)
        mid = 100.0 + i
        out.append(InternalLevel(
            symbol="TEST.NS", side=side,
            level_low=mid - 0.5, level_high=mid + 0.5, level_mid=mid,
            p_touch=0.2 + 0.7 * frac, p_up=0.5 + 0.4 * (frac - 0.5),
            q=0.3 + 0.6 * frac, as_of="2026-05-29T15:30:00+05:30",
        ))
    return out


def test_g_is_integer_in_range():
    gs = g_scores(_levels(), "cust-a", "2026-05-29")
    assert all(isinstance(g, int) for g in gs)
    assert all(0 <= g <= 100 for g in gs)


def test_g_is_monotone_in_internal_signal_within_customer():
    # Levels are constructed with strictly increasing internal strength.
    levels = _levels(n=15)
    gs = g_scores(levels, "cust-a", "2026-05-29")
    # Rank should be (weakly) increasing with the constructed strength order.
    # Allow the +-1 canary dither: no inversion larger than the dither.
    for i in range(1, len(gs)):
        assert gs[i] >= gs[i - 1] - 2


def test_top_level_outranks_bottom_level():
    levels = _levels(n=20)
    gs = g_scores(levels, "cust-a", "2026-05-29")
    assert gs[-1] > gs[0]


def test_watermark_differs_per_customer_but_preserves_top():
    levels = _levels(n=20)
    ga = g_scores(levels, "cust-a", "2026-05-29")
    gb = g_scores(levels, "cust-b", "2026-05-29")
    # Different customers get numerically different feeds (watermark)...
    assert ga != gb
    # ...but both rank the strongest level at/near the top.
    assert ga.index(max(ga)) == gb.index(max(gb)) == len(levels) - 1


def test_feature_states_are_from_alphabet():
    levels = _levels(n=10)
    ss = feature_states(levels, "cust-a", "2026-05-29")
    assert set(ss).issubset(set(scoring.FEATURE_STATES))


def test_feature_states_are_non_directional_labels():
    # No customer-facing state label may carry directional meaning.
    for label in scoring.FEATURE_STATES:
        low = label.lower()
        for banned in ("up", "down", "long", "short", "bull", "bear",
                       "+", "-", "buy", "sell"):
            assert banned not in low


def test_feature_state_neutral_when_direction_missing():
    lvl = InternalLevel("X.NS", "above", 99.5, 100.5, 100.0,
                        p_touch=0.6, p_up=None, q=0.5, as_of="2026-05-29")
    ss = feature_states([lvl], "cust-a", "2026-05-29")
    assert ss == [scoring.FEATURE_STATES[1]]


def test_public_record_only_allowlisted_keys():
    lvl = _levels(1)[0]
    rec = public_record(lvl, 73, "state_1")
    assert set(rec.keys()) == set(PUBLIC_KEYS)


def test_public_record_does_not_leak_internal_values():
    # Distinctive internal values that cannot collide with public geometry.
    lvl = InternalLevel(
        symbol="TEST.NS", side="above",
        level_low=99.5, level_high=100.5, level_mid=100.0,
        p_touch=0.72391, p_up=0.61373, q=0.81247,
        as_of="2026-05-29T15:30:00+05:30",
    )
    rec = public_record(lvl, 73, "state_1")
    blob = json.dumps(rec)
    for forbidden_val in ("0.72391", "0.61373", "0.81247"):
        assert forbidden_val not in blob


def test_public_record_keys_have_no_forbidden_tokens():
    lvl = _levels(1)[0]
    rec = public_record(lvl, 50, "state_2")
    for key in rec:
        for forbidden in FORBIDDEN_PUBLIC_TOKENS:
            assert forbidden not in key.lower()


def test_public_record_values_have_no_forbidden_tokens_except_compliance_text():
    lvl = _levels(1)[0]
    rec = public_record(lvl, 50, "state_2")
    for key, value in rec.items():
        if key in COMPLIANCE_TEXT_KEYS:
            continue  # compliance text intentionally negates banned words
        text = json.dumps(value).lower()
        for forbidden in FORBIDDEN_PUBLIC_TOKENS:
            assert forbidden not in text, f"{forbidden!r} leaked via {key}"


def test_record_carries_compliance_tag_not_full_disclaimer():
    lvl = _levels(1)[0]
    rec = public_record(lvl, 1, "state_3")
    assert rec["compliance_tag"] == COMPLIANCE_TAG
    # The bulky full disclaimer must NOT be repeated per observation.
    assert "interpretation_note" not in rec
    assert "Users are solely responsible" not in json.dumps(rec)


def test_score_levels_end_to_end():
    levels = _levels(8) + _levels(8, side="below")
    recs = score_levels(levels, "cust-a", "2026-05-29")
    assert len(recs) == len(levels)
    assert all(set(r.keys()) == set(PUBLIC_KEYS) for r in recs)


def test_empty_input():
    assert score_levels([], "cust-a", "2026-05-29") == []
    assert g_scores([], "cust-a", "2026-05-29") == []
