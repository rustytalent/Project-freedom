"""Tests for the pre-touch pocket sensitivity analyzer.

The script's value is the CI-aware verdict layer over the existing Track A
sweep parquet. These tests pin:
  1. filter_trades AND-s the time-bucket / sector / direction criteria.
  2. _summarise computes net_R, gross_R, cost_R and a 95% CI on net_R.
  3. _verdict only flags pockets with n >= min_trades AND ci95_lo > 0.

No simulator dependency — pure pandas + arithmetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.pretouch_pocket_sensitivity import (
    DEFAULT_POCKETS,
    PocketRow,
    _summarise,
    _verdict,
    filter_trades,
)


def _trade_row(*, time_bucket="morning", sector="AUTO", direction="UP",
               factor="EQHL", pool_quality=0.5,
               gross_pnl=10.0, total_cost=2.0, risk_inr=10.0):
    net_pnl = gross_pnl - total_cost
    return {
        "time_bucket": time_bucket,
        "sector": sector,
        "direction": direction,
        "factor": factor,
        "pool_quality": pool_quality,
        "gross_pnl": gross_pnl,
        "total_cost": total_cost,
        "net_pnl": net_pnl,
        "risk_inr": risk_inr,
        "net_r": net_pnl / risk_inr,
    }


# ---------------------------------------------------------------------------
# filter_trades
# ---------------------------------------------------------------------------

def test_empty_pocket_keeps_everything():
    trades = pd.DataFrame([
        _trade_row(time_bucket="morning"),
        _trade_row(time_bucket="afternoon", sector="IT"),
    ])
    assert len(filter_trades(trades, {})) == 2


def test_time_bucket_filter():
    trades = pd.DataFrame([
        _trade_row(time_bucket="morning"),
        _trade_row(time_bucket="midday"),
        _trade_row(time_bucket="afternoon"),
    ])
    out = filter_trades(trades, {"time_buckets": ("morning", "midday")})
    assert set(out["time_bucket"]) == {"morning", "midday"}


def test_sector_filter():
    trades = pd.DataFrame([
        _trade_row(sector="AUTO"),
        _trade_row(sector="IT"),
        _trade_row(sector="BANKING"),
    ])
    out = filter_trades(trades, {"sectors": ("IT", "BANKING")})
    assert set(out["sector"]) == {"IT", "BANKING"}


def test_direction_filter():
    trades = pd.DataFrame([
        _trade_row(direction="UP"),
        _trade_row(direction="DOWN"),
    ])
    out = filter_trades(trades, {"directions": ("UP",)})
    assert list(out["direction"]) == ["UP"]


def test_winning_combo_ands_all_three_criteria():
    trades = pd.DataFrame([
        # Match: morning + AUTO + UP
        _trade_row(time_bucket="morning", sector="AUTO", direction="UP"),
        # Reject: morning + IT
        _trade_row(time_bucket="morning", sector="IT", direction="UP"),
        # Reject: midday + AUTO + DOWN
        _trade_row(time_bucket="midday", sector="AUTO", direction="DOWN"),
        # Match: midday + FMCG + UP
        _trade_row(time_bucket="midday", sector="FMCG", direction="UP"),
        # Reject: afternoon + AUTO
        _trade_row(time_bucket="afternoon", sector="AUTO", direction="UP"),
    ])
    out = filter_trades(trades, DEFAULT_POCKETS["winning_combo"])
    assert len(out) == 2


def test_q_cohort_filter_keeps_top_fraction():
    trades = pd.DataFrame([_trade_row(pool_quality=q) for q in
                            np.linspace(0.10, 1.00, 100)])
    out = filter_trades(trades, {"q_cohort": (0.25,)})    # top 25%
    # 100 trades * 0.25 = 25 should remain; allow small rounding from quantile.
    assert 20 <= len(out) <= 30
    assert out["pool_quality"].min() >= trades["pool_quality"].quantile(0.74)


# ---------------------------------------------------------------------------
# _summarise — net_R, gross_R, cost_R, CI95
# ---------------------------------------------------------------------------

def test_summarise_returns_none_on_empty():
    assert _summarise("p", pd.DataFrame()) is None


def test_summarise_basic_metrics():
    # 4 winners (+0.8R after cost), 1 loser (-0.5R after cost)
    # gross: 4 wins of +1.0R + 1 loss of -0.3R; per-trade cost 0.2R
    # mean net_R = (4*0.8 + (-0.5)) / 5 = 2.7/5 = 0.54
    trades = pd.DataFrame([
        _trade_row(gross_pnl=10.0, total_cost=2.0),       # net_r = 0.8
        _trade_row(gross_pnl=10.0, total_cost=2.0),
        _trade_row(gross_pnl=10.0, total_cost=2.0),
        _trade_row(gross_pnl=10.0, total_cost=2.0),
        _trade_row(gross_pnl=-3.0, total_cost=2.0),       # net_r = -0.5
    ])
    row = _summarise("test", trades)
    assert row is not None
    assert row.trades == 5
    assert abs(row.win - 0.8) < 1e-9
    assert abs(row.net_R - 0.54) < 1e-9
    # gross_R = (4*10 + (-3)) / (5*10) = 37/50 = 0.74
    assert abs(row.gross_R - 0.74) < 1e-9
    # cost_R = 0.74 - 0.54 = 0.20
    assert abs(row.cost_R - 0.20) < 1e-9


def test_ci95_collapses_to_point_with_zero_variance():
    trades = pd.DataFrame([
        _trade_row(gross_pnl=10.0, total_cost=2.0),    # all identical -> net_r=0.8
        _trade_row(gross_pnl=10.0, total_cost=2.0),
        _trade_row(gross_pnl=10.0, total_cost=2.0),
    ])
    row = _summarise("p", trades)
    assert abs(row.se_net_R) < 1e-9
    assert abs(row.ci95_lo - row.net_R) < 1e-9
    assert abs(row.ci95_hi - row.net_R) < 1e-9


def test_ci95_brackets_mean_with_variance():
    rng = np.random.default_rng(42)
    trades = pd.DataFrame([
        _trade_row(gross_pnl=float(g), total_cost=0.0)
        for g in rng.normal(loc=0.0, scale=5.0, size=200)
    ])
    row = _summarise("p", trades)
    assert row.se_net_R > 0
    assert row.ci95_lo < row.net_R < row.ci95_hi


# ---------------------------------------------------------------------------
# _verdict — only flag positive cells with sufficient sample
# ---------------------------------------------------------------------------

def _mk_row(pocket: str, n: int, net_R: float,
            ci_lo: float, ci_hi: float) -> PocketRow:
    return PocketRow(
        pocket=pocket, trades=n, win=0.5,
        gross_R=net_R + 0.2, net_R=net_R, cost_R=0.2,
        PF=1.5, se_net_R=(ci_hi - ci_lo) / (2 * 1.96),
        ci95_lo=ci_lo, ci95_hi=ci_hi, avg_cost_inr=60.0,
    )


def test_verdict_flags_only_statistically_positive():
    rows = [
        _mk_row("winning_combo", 200, +0.40, +0.15, +0.65),    # statistically positive
        _mk_row("morning",      300, +0.05, -0.10, +0.20),    # borderline
        _mk_row("afternoon",   1000, -0.50, -0.60, -0.40),    # negative
    ]
    lines = _verdict(rows, min_trades=50)
    joined = "\n".join(lines)
    assert "Statistically positive pockets (CI lower bound > 0): 1" in joined
    assert "winning_combo" in joined
    assert "Borderline" in joined and "morning" in joined


def test_verdict_excludes_small_samples():
    rows = [
        _mk_row("tiny", 10, +0.80, +0.10, +1.50),    # too few trades
    ]
    lines = _verdict(rows, min_trades=50)
    joined = "\n".join(lines)
    assert "Statistically positive pockets (CI lower bound > 0): 0" in joined
    assert "tiny" not in joined


def test_verdict_handles_no_positives_at_all():
    rows = [
        _mk_row("kitchen_sink", 1000, -0.50, -0.60, -0.40),
    ]
    lines = _verdict(rows, min_trades=50)
    joined = "\n".join(lines)
    assert "(no positive pockets at any size" in joined


def test_default_pockets_well_formed():
    valid_keys = {"time_buckets", "sectors", "directions", "factors", "q_cohort"}
    for name, pocket in DEFAULT_POCKETS.items():
        for k in pocket:
            assert k in valid_keys, f"pocket {name!r} has invalid key {k!r}"
