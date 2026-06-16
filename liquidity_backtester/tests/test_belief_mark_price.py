"""Tests for the Belief Engine mark-price layer.

Pins the founder's #1 correction: LTP is never the primary mark; a clean
quote-derived microprice/mid is, with explicit quality grading and a
fallback chain for dirty quotes.
"""
from __future__ import annotations

import math

import pytest

from liqpool.research.belief.mark_price import (
    MarkPrice,
    MarkPriceConfig,
    MarkPriceTracker,
    Quote,
    compute_mark,
)


def test_microprice_used_on_good_quote_and_leans_to_heavy_side():
    # Heavy bid (800) vs thin ask (200) → microprice should sit above mid.
    q = Quote(bid=100.0, ask=100.4, bid_qty=800, ask_qty=200, ltp=100.2, ltp_age_s=0.2)
    m = compute_mark(q)
    assert m.source == "microprice"
    assert m.quality_label == "good"
    assert m.mid == pytest.approx(100.2)
    assert m.price > m.mid, "heavy bid must push microprice toward the ask"
    assert m.tradable


def test_microprice_leans_down_on_heavy_ask():
    q = Quote(bid=100.0, ask=100.4, bid_qty=200, ask_qty=800)
    m = compute_mark(q)
    assert m.price < m.mid, "heavy ask must push microprice toward the bid"


def test_wide_spread_falls_back_to_mid():
    q = Quote(bid=10.0, ask=12.0, bid_qty=50, ask_qty=50, ltp=11.0, ltp_age_s=0.2)
    m = compute_mark(q)
    assert m.source == "mid"
    assert "wide_spread" in m.flags
    assert m.price == pytest.approx(11.0)


def test_thin_depth_blocks_microprice():
    # Tight spread but almost no depth → microprice not trusted, use mid.
    q = Quote(bid=100.0, ask=100.1, bid_qty=0.0, ask_qty=0.0)
    m = compute_mark(q)
    assert m.source == "mid"
    assert "thin_depth" in m.flags


def test_crossed_quote_falls_back_to_last_valid():
    q = Quote(bid=101.0, ask=100.0, bid_qty=10, ask_qty=10, ltp=100.5, ltp_age_s=0.2)
    m = compute_mark(q, last_valid_price=100.3)
    assert m.is_crossed
    assert m.source == "last_valid"
    assert m.price == pytest.approx(100.3)
    assert m.quality_label == "poor"
    assert not m.tradable


def test_crossed_without_fallback_uses_fresh_ltp_then_invalid():
    fresh = Quote(bid=101.0, ask=100.0, ltp=100.5, ltp_age_s=0.2)
    m_fresh = compute_mark(fresh)
    assert m_fresh.source == "ltp"
    assert "ltp_fallback" in m_fresh.flags

    stale = Quote(bid=101.0, ask=100.0, ltp=100.5, ltp_age_s=99.0)
    m_stale = compute_mark(stale)
    assert m_stale.source == "invalid"
    assert math.isnan(m_stale.price)


def test_locked_quote_is_flagged_but_mid_usable():
    q = Quote(bid=100.0, ask=100.0, bid_qty=100, ask_qty=100, ltp=100.0, ltp_age_s=0.2)
    m = compute_mark(q)
    assert m.is_locked
    assert "locked" in m.flags
    # Locked spread = 0 → spread_score high; mid == bid == ask.
    assert m.price == pytest.approx(100.0)


def test_ltp_confirms_only_when_fresh_and_inside_book():
    inside_fresh = Quote(bid=100.0, ask=100.4, bid_qty=300, ask_qty=300,
                         ltp=100.2, ltp_age_s=0.5)
    assert compute_mark(inside_fresh).ltp_confirms

    outside = Quote(bid=100.0, ask=100.4, bid_qty=300, ask_qty=300,
                    ltp=105.0, ltp_age_s=0.5)
    m_out = compute_mark(outside)
    assert not m_out.ltp_confirms
    assert "ltp_outside" in m_out.flags

    stale = Quote(bid=100.0, ask=100.4, bid_qty=300, ask_qty=300,
                  ltp=100.2, ltp_age_s=99.0)
    m_stale = compute_mark(stale)
    assert not m_stale.ltp_confirms
    assert "stale_ltp" in m_stale.flags


def test_nonpositive_quote_is_invalid():
    q = Quote(bid=0.0, ask=0.0)
    m = compute_mark(q)
    assert m.source == "invalid"
    assert "nonpositive_quote" in m.flags


def test_quality_decreases_with_spread():
    tight = compute_mark(Quote(bid=100.0, ask=100.1, bid_qty=500, ask_qty=500))
    loose = compute_mark(Quote(bid=100.0, ask=101.5, bid_qty=500, ask_qty=500))
    assert tight.quality > loose.quality


def test_tracker_remembers_last_good_and_falls_back():
    tr = MarkPriceTracker()
    good = tr.update("K", Quote(bid=50.0, ask=50.2, bid_qty=300, ask_qty=300))
    assert good.source in ("microprice", "mid")
    assert tr.last_good("K") == pytest.approx(good.price)
    # Now a crossed quote — should fall back to the remembered mark.
    crossed = tr.update("K", Quote(bid=51.0, ask=50.0, bid_qty=10, ask_qty=10))
    assert crossed.source == "last_valid"
    assert crossed.price == pytest.approx(good.price)


def test_tracker_does_not_store_dirty_marks():
    tr = MarkPriceTracker()
    tr.update("K", Quote(bid=50.0, ask=50.2, bid_qty=300, ask_qty=300))
    first = tr.last_good("K")
    # A crossed quote must not overwrite the remembered good mark.
    tr.update("K", Quote(bid=60.0, ask=50.0))
    assert tr.last_good("K") == pytest.approx(first)


def test_config_validation():
    with pytest.raises(ValueError, match="spread_weight"):
        MarkPriceConfig(spread_weight=1.5)
    with pytest.raises(ValueError, match="usable"):
        MarkPriceConfig(max_spread_pct_good=0.10, max_spread_pct_usable=0.05)


def test_mark_to_dict_is_json_serializable():
    m = compute_mark(Quote(bid=100.0, ask=100.2, bid_qty=300, ask_qty=300))
    import json
    json.dumps(m.to_dict())
