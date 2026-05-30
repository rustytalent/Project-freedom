"""Tests for the pocket-sensitivity aggregation and filter logic.

The script's verdict hinges on:
  1. Filters correctly slice pools by session × sector × side × factor.
  2. _summarise_trades correctly computes per-cell metrics, including a
     statistically defensible 95% CI on net_R.
  3. _verdict only flags cells that meet the sample threshold AND have a
     CI lower bound > 0.

These tests pin all three with hand-built inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.pocket_sensitivity import (
    DEFAULT_POCKETS,
    PocketRow,
    _summarise_trades,
    _verdict,
    filter_pools,
)
from liqpool.pools import Pool
from liqpool.tester import PoolResult


def _ts(utc: str) -> pd.Timestamp:
    """UTC-naive timestamp (matches the parquet convention nse_session expects).
    e.g. "04:30" UTC -> 10:00 IST after +5:30."""
    return pd.Timestamp(f"2026-05-26 {utc}")


def _pool(asset: str, side: str) -> Pool:
    return Pool(
        side=side, price_low=100.0, price_high=101.0,
        formed_at=_ts("04:00"), available_at=_ts("04:00"),
        contributors=[], score=1.0, tfs=["base"], asset=asset,
    )


def _result_touched(touched_at_utc: str) -> PoolResult:
    return PoolResult(
        pool_idx=0, side="low", formed_at=_ts("04:00"),
        price_low=100.0, price_high=101.0, score=1.0,
        outcome="respected_strong",
        touched_at=_ts(touched_at_utc),
    )


def _result_untouched() -> PoolResult:
    return PoolResult(
        pool_idx=0, side="low", formed_at=_ts("04:00"),
        price_low=100.0, price_high=101.0, score=1.0,
        outcome="untouched",
    )


# ---------------------------------------------------------------------------
# Filter tests
# ---------------------------------------------------------------------------

def test_filter_empty_pocket_keeps_everything():
    pools = [_pool("HDFCBANK", "low"), _pool("TCS", "high")]
    results = [_result_touched("04:45"), _result_untouched()]
    fp, fr = filter_pools(pools, results, {})
    assert len(fp) == 2 and len(fr) == 2


def test_filter_by_session_drops_untouched():
    # Untouched pool has no session attribution, so it must be dropped when
    # a sessions filter is active. 04:45 UTC = 10:15 IST = nse_session "morning"
    # (the boundary; nse_session's "morning" is [10:15, 12:00) IST).
    pools = [_pool("HDFCBANK", "low"), _pool("TCS", "low")]
    results = [_result_touched("04:45"), _result_untouched()]
    fp, _ = filter_pools(pools, results, {"sessions": ("morning",)})
    assert len(fp) == 1
    assert fp[0].asset == "HDFCBANK"


def test_filter_by_session_midday_vs_afternoon():
    # 06:30 UTC = 12:00 IST = "midday" boundary; nse_session returns "midday".
    # 09:00 UTC = 14:30 IST = "closing".
    pools = [_pool("A", "low"), _pool("B", "low"), _pool("C", "low")]
    results = [
        _result_touched("04:45"),    # 10:15 IST = "morning"
        _result_touched("07:00"),    # 12:30 IST = "midday"
        _result_touched("09:30"),    # 15:00 IST = "closing"
    ]
    fp, _ = filter_pools(pools, results, {"sessions": ("morning", "midday")})
    assert {p.asset for p in fp} == {"A", "B"}

    fp, _ = filter_pools(pools, results, {"sessions": ("closing",)})
    assert {p.asset for p in fp} == {"C"}


def test_filter_by_sector():
    pools = [
        _pool("HDFCBANK", "low"),    # BANKING
        _pool("TCS", "low"),          # IT
        _pool("MARUTI", "low"),       # AUTO
    ]
    results = [_result_touched("04:45")] * 3
    fp, _ = filter_pools(pools, results, {"sectors": ("AUTO",)})
    assert len(fp) == 1 and fp[0].asset == "MARUTI"

    fp, _ = filter_pools(pools, results, {"sectors": ("AUTO", "BANKING")})
    assert {p.asset for p in fp} == {"HDFCBANK", "MARUTI"}


def test_filter_by_side():
    pools = [_pool("A", "low"), _pool("B", "high")]
    results = [_result_touched("04:45")] * 2
    fp, _ = filter_pools(pools, results, {"sides": ("low",)})
    assert {p.asset for p in fp} == {"A"}


def test_filter_combined_session_sector_side():
    # The "winning combo" — must AND all three filters.
    pools = [
        _pool("MARUTI",   "low"),   # AUTO, long, morning -> KEEP
        _pool("MARUTI",   "high"),  # AUTO, short -> DROP (wrong side)
        _pool("HDFCBANK", "low"),   # BANKING (wrong sector) -> DROP
        _pool("TCS",      "low"),   # IT (wrong sector) -> DROP
    ]
    results = [_result_touched("04:45")] * 4   # 10:15 IST = "morning"
    fp, _ = filter_pools(pools, results, {
        "sessions": ("morning", "midday"),
        "sectors":  ("AUTO", "FMCG", "PHARMA"),
        "sides":    ("low",),
    })
    assert len(fp) == 1 and fp[0].asset == "MARUTI" and fp[0].side == "low"


# ---------------------------------------------------------------------------
# Summary aggregation tests
# ---------------------------------------------------------------------------

def _trade(mode: str, gross_pnl: float, total_cost: float,
           risk_inr: float = 10.0, entry: float = 100.0, qty: int = 1):
    net_pnl = gross_pnl - total_cost
    return {
        "mode": mode,
        "gross_pnl": gross_pnl, "total_cost": total_cost,
        "net_pnl": net_pnl, "risk_inr": risk_inr,
        "net_r": net_pnl / risk_inr, "entry": entry, "quantity": qty,
    }


def test_summarise_empty_returns_no_rows():
    assert _summarise_trades("p", "qty=1", pd.DataFrame()) == []


def test_summarise_basic_arithmetic():
    # 4 winners, 1 loser. risk=10. cost=1 per trade.
    df = pd.DataFrame([
        _trade("touch_confirmed", +5.0, 1.0),    # net_R = (5-1)/10 = +0.4
        _trade("touch_confirmed", +3.0, 1.0),    # +0.2
        _trade("touch_confirmed", +2.0, 1.0),    # +0.1
        _trade("touch_confirmed", +4.0, 1.0),    # +0.3
        _trade("touch_confirmed", -3.0, 1.0),    # net_R = -0.4
    ])
    rows = _summarise_trades("morning", "qty=1", df)
    assert len(rows) == 1
    r = rows[0]
    assert r.trades == 5
    assert abs(r.win - 4 / 5) < 1e-9
    # mean net_R = (0.4 + 0.2 + 0.1 + 0.3 - 0.4) / 5 = 0.12
    assert abs(r.net_R - 0.12) < 1e-9
    # mean gross_R = (5+3+2+4-3)/(5*10) = 0.22
    assert abs(r.gross_R - 0.22) < 1e-9
    # cost_R = 0.22 - 0.12 = 0.10
    assert abs(r.cost_R - 0.10) < 1e-9
    # PF = wins(4+2+1+3) / losses(4) = 10/4 = 2.5
    assert abs(r.PF - 2.5) < 1e-9


def test_summarise_includes_ci95():
    # Two trades, both +0.5R, zero variance -> CI collapses to point.
    df = pd.DataFrame([
        _trade("blind_limit", +5.0, 0.0),       # net_R +0.5
        _trade("blind_limit", +5.0, 0.0),       # net_R +0.5
    ])
    rows = _summarise_trades("p", "qty=1", df)
    r = rows[0]
    assert abs(r.net_R - 0.5) < 1e-9
    assert r.se_net_R < 1e-9
    assert abs(r.ci95_lo - 0.5) < 1e-6
    assert abs(r.ci95_hi - 0.5) < 1e-6


def test_summarise_ci95_widens_with_variance():
    rng = np.random.default_rng(0)
    # Symmetric returns around 0 -> mean ~0, wide CI.
    df = pd.DataFrame([
        _trade("touch_confirmed", float(x), 0.0)
        for x in rng.normal(loc=0.0, scale=5.0, size=200)
    ])
    rows = _summarise_trades("p", "qty=1", df)
    r = rows[0]
    assert r.se_net_R > 0
    assert r.ci95_lo < r.net_R < r.ci95_hi


# ---------------------------------------------------------------------------
# Verdict tests
# ---------------------------------------------------------------------------

def _mk_row(pocket: str, mode: str, n: int,
            net_R: float, ci_lo: float, ci_hi: float) -> PocketRow:
    return PocketRow(
        pocket=pocket, sizing="₹100,000", mode=mode,
        trades=n, win=0.5, gross_R=net_R + 0.3, net_R=net_R,
        cost_R=0.3, PF=1.5, se_net_R=(ci_hi - ci_lo) / (2 * 1.96),
        ci95_lo=ci_lo, ci95_hi=ci_hi,
        avg_cost_inr=60.0, avg_notional_inr=100_000.0,
    )


def test_verdict_flags_only_cells_with_lower_bound_positive():
    rows = [
        _mk_row("winning_combo", "displacement_confirmed", 200,
                net_R=+0.40, ci_lo=+0.15, ci_hi=+0.65),    # statistically positive
        _mk_row("morning_midday", "touch_confirmed", 300,
                net_R=+0.05, ci_lo=-0.10, ci_hi=+0.20),    # borderline
        _mk_row("kitchen_sink", "blind_limit", 1000,
                net_R=-0.50, ci_lo=-0.60, ci_hi=-0.40),    # negative
    ]
    lines = _verdict(rows, min_trades=50)
    joined = "\n".join(lines)
    assert "Statistically positive cells (CI lower bound > 0): 1" in joined
    assert "displacement_confirmed" in joined
    assert "Borderline" in joined
    assert "touch_confirmed" in joined


def test_verdict_respects_min_trades():
    # A "positive" cell with only 10 trades must NOT be flagged.
    rows = [
        _mk_row("tiny_pocket", "touch_confirmed", 10,
                net_R=+0.80, ci_lo=+0.10, ci_hi=+1.50),    # too few trades
    ]
    lines = _verdict(rows, min_trades=50)
    joined = "\n".join(lines)
    assert "Statistically positive cells (CI lower bound > 0): 0" in joined
    assert "tiny_pocket" not in joined


def test_verdict_no_positive_cells_at_all():
    rows = [
        _mk_row("kitchen_sink", "blind_limit", 1000,
                net_R=-0.50, ci_lo=-0.60, ci_hi=-0.40),
    ]
    lines = _verdict(rows, min_trades=50)
    joined = "\n".join(lines)
    assert "(no positive cells at any sizing or pocket" in joined


# ---------------------------------------------------------------------------
# Pocket-definition smoke test (catches typos in DEFAULT_POCKETS)
# ---------------------------------------------------------------------------

def test_default_pockets_have_valid_keys():
    valid_keys = {"sessions", "sectors", "sides", "factors"}
    for name, pocket in DEFAULT_POCKETS.items():
        for key in pocket:
            assert key in valid_keys, f"pocket {name!r} has invalid key {key!r}"
        # Optional values are tuples/lists of strings only.
        for key, vals in pocket.items():
            for v in vals:
                assert isinstance(v, str), f"pocket {name!r}.{key} has non-string {v!r}"
