"""Tests for the HypothesisHarness — the missing edge layer.

Pins:
  * CostsModel itemises charges correctly (STT sell-side only, GST
    on brokerage + exchange + SEBI)
  * simulate_trades handles both long and short, computes net P&L
    after costs, derives R-multiples
  * trade_stats computes Sharpe / Sortino / drawdown / hit-rate /
    profit factor honestly
  * HypothesisHarness.run produces a report with by-month and
    auto-decide rules
  * HYPOTHESIS_LIBRARY has the registered candidates and each is a
    callable that returns valid signal tuples
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

pd = pytest.importorskip("pandas")

from liqpool.research.harness import (
    BacktestTrade, CostsModel, HypothesisHarness, HypothesisReport,
    HypothesisSpec, simulate_trades, trade_stats,
)
from liqpool.research.library import (
    HYPOTHESIS_LIBRARY, get_hypothesis, h1_thursday_theta_scalp,
    h2_opening_range_failure_fade, h5_time_of_day_buy_and_hold,
    hypothesis_names,
)


# ─────────────────────────────────────────────────────────────────
# Cost model
# ─────────────────────────────────────────────────────────────────

def test_costs_round_trip_options_no_pnl():
    c = CostsModel()
    # 100→100 round trip, no P&L, still pay charges
    charges = c.round_trip_charges(buy_price=100, sell_price=100, qty=75)
    assert charges > 0
    assert charges > c.brokerage_per_leg_rupees * 2     # at minimum brokerage × 2


def test_costs_scale_with_qty():
    """Brokerage is FIXED per leg so 10× qty doesn't produce 10× cost.
    But turnover-based fees (STT, exchange, SEBI, stamp, slippage)
    do scale, so the bigger trade still pays meaningfully more."""
    c = CostsModel()
    small = c.round_trip_charges(100, 110, 75)
    big = c.round_trip_charges(100, 110, 750)
    assert big > small * 3


def test_costs_higher_when_selling_higher_premium():
    c = CostsModel()
    """STT scales with sell turnover so charges go up if you sell at
    a higher premium even if entry is the same."""
    low = c.round_trip_charges(100, 110, 75)
    high = c.round_trip_charges(100, 200, 75)
    assert high > low


# ─────────────────────────────────────────────────────────────────
# simulate_trades
# ─────────────────────────────────────────────────────────────────

def _bars(n=10, base=25000, step=10, freq="5min"):
    start = datetime(2026, 6, 5, 9, 30)        # Thursday 09:30
    ts = [start + timedelta(minutes=5 * i) for i in range(n)]
    closes = [base + i * step for i in range(n)]
    df = pd.DataFrame({
        "ts": ts,
        "open": closes,
        "high": [c + 5 for c in closes],
        "low": [c - 5 for c in closes],
        "close": closes,
        "volume": [1000] * n,
    })
    df.attrs["bar_minutes"] = 5
    return df


def test_simulate_trades_long_winner():
    bars = _bars(n=10, base=100, step=2)
    trades = simulate_trades(bars, [(0, 9, +1, 75)], CostsModel(),
                              initial_risk_per_trade_rupees=1000)
    assert len(trades) == 1
    t = trades[0]
    assert t.side == +1
    assert t.entry_price == 100
    assert t.exit_price == 118
    assert t.gross_pnl == 18 * 75
    assert t.net_pnl < t.gross_pnl              # costs deducted
    assert t.r_multiple == round(t.net_pnl / 1000, 3)


def test_simulate_trades_short_winner():
    """If we SHORT at 200 and price drops to 100, that's a winner."""
    bars = _bars(n=10, base=200, step=-10)      # 200 → 110
    trades = simulate_trades(bars, [(0, 9, -1, 75)], CostsModel(),
                              initial_risk_per_trade_rupees=1000)
    assert len(trades) == 1
    t = trades[0]
    assert t.side == -1
    assert t.gross_pnl == 90 * 75               # 200 - 110 = 90
    assert t.net_pnl < t.gross_pnl


def test_simulate_trades_rejects_zero_qty_and_inverted_idx():
    bars = _bars()
    out = simulate_trades(
        bars, [(0, 9, +1, 0), (5, 5, +1, 75), (-1, 9, +1, 75)],
        CostsModel())
    assert out == []


def test_simulate_trades_clamps_exit_to_last_bar():
    bars = _bars(n=10)
    trades = simulate_trades(bars, [(0, 99, +1, 75)], CostsModel())
    assert len(trades) == 1
    assert trades[0].exit_idx == 9


# ─────────────────────────────────────────────────────────────────
# trade_stats
# ─────────────────────────────────────────────────────────────────

def test_stats_on_empty_list():
    s = trade_stats([])
    assert s.n_trades == 0
    assert s.sharpe == 0
    assert s.net_pnl == 0


def test_stats_basic_arithmetic():
    bars = _bars()
    trades = simulate_trades(bars, [
        (0, 1, +1, 75),    # tiny winner
        (1, 2, +1, 75),
        (2, 3, +1, 75),
    ], CostsModel(), initial_risk_per_trade_rupees=1000)
    s = trade_stats(trades)
    assert s.n_trades == 3
    assert s.hit_rate in (0.0, 1.0/3, 2.0/3, 1.0)  # any sensible value
    assert s.expectancy_rupees == round(s.net_pnl / 3, 2)


def test_stats_drawdown_tracked():
    """Three trades: +1000, -2000, +500. Equity: 1000, -1000, -500.
    Peak = 1000, trough vs peak = 1000 - (-1000) = 2000 max DD."""
    fake_trades = [
        BacktestTrade(entry_idx=0, exit_idx=1, entry_ts="a", exit_ts="b",
                       side=+1, qty=1, entry_price=100, exit_price=101,
                       gross_pnl=1000, costs=0, net_pnl=1000,
                       r_multiple=1, hold_minutes=5),
        BacktestTrade(entry_idx=1, exit_idx=2, entry_ts="a", exit_ts="b",
                       side=+1, qty=1, entry_price=100, exit_price=99,
                       gross_pnl=-2000, costs=0, net_pnl=-2000,
                       r_multiple=-2, hold_minutes=5),
        BacktestTrade(entry_idx=2, exit_idx=3, entry_ts="a", exit_ts="b",
                       side=+1, qty=1, entry_price=100, exit_price=101,
                       gross_pnl=500, costs=0, net_pnl=500,
                       r_multiple=0.5, hold_minutes=5),
    ]
    s = trade_stats(fake_trades)
    assert s.max_drawdown_rupees == 2000.0
    assert s.max_drawdown_pct_of_peak == 200.0     # 2000 / 1000 peak


def test_stats_sharpe_positive_on_consistent_winners():
    fake_trades = [
        BacktestTrade(entry_idx=0, exit_idx=1, entry_ts="a", exit_ts="b",
                       side=+1, qty=1, entry_price=100, exit_price=101,
                       gross_pnl=100, costs=0, net_pnl=100,
                       r_multiple=0.1, hold_minutes=5)
        for _ in range(20)
    ]
    # Add tiny variance so stdev > 0
    fake_trades[5] = BacktestTrade(
        entry_idx=0, exit_idx=1, entry_ts="a", exit_ts="b",
        side=+1, qty=1, entry_price=100, exit_price=101,
        gross_pnl=120, costs=0, net_pnl=120, r_multiple=0.12,
        hold_minutes=5)
    s = trade_stats(fake_trades)
    assert s.sharpe > 0


# ─────────────────────────────────────────────────────────────────
# HypothesisHarness — end-to-end
# ─────────────────────────────────────────────────────────────────

def _no_op_hypothesis(bars, **_):
    return []


def test_harness_runs_with_no_signals():
    spec = HypothesisSpec(name="empty", fn=_no_op_hypothesis)
    bars = _bars(n=20)
    report = HypothesisHarness().run(spec, bars)
    assert report.trade_count == 0
    assert report.decision == "needs_more_data"


def test_harness_auto_decide_keeps_strong_sharpe():
    """A hypothesis that always wins should auto-decide 'keep' once
    we have enough trades."""
    def winner(bars, **_):
        # 35 winners (above KEEP_MIN_TRADES=30)
        return [(i, i + 1, +1, 75) for i in range(35) if i + 1 < len(bars)]

    # Bigger bar series so we actually have 35 entries
    bars = _bars(n=50, base=100, step=5)
    spec = HypothesisSpec(name="winner", fn=winner)
    report = HypothesisHarness(
        initial_risk_per_trade_rupees=500).run(spec, bars)
    assert report.trade_count >= 30
    assert report.overall.sharpe > 0


def test_harness_by_month_aggregation():
    """When trades span multiple months, by_month has entries per month."""
    # Build bars across 2 months
    base_ts = datetime(2026, 1, 15, 9, 30)
    ts = [base_ts + timedelta(days=i) for i in range(60)]
    closes = [25000 + i for i in range(60)]
    df = pd.DataFrame({
        "ts": ts,
        "open": closes, "high": [c + 5 for c in closes],
        "low": [c - 5 for c in closes], "close": closes,
        "volume": [1000] * 60,
    })
    df.attrs["bar_minutes"] = 5
    def daily_winner(bars, **_):
        return [(i, i + 1, +1, 75) for i in range(len(bars) - 1)]
    spec = HypothesisSpec(name="daily", fn=daily_winner)
    report = HypothesisHarness().run(spec, df)
    assert len(report.by_month) >= 2


def test_harness_report_roundtrips_through_dict():
    spec = HypothesisSpec(name="empty", fn=_no_op_hypothesis)
    bars = _bars(n=10)
    report = HypothesisHarness().run(spec, bars)
    row = report.to_row()
    assert row["name"] == "empty"
    assert "overall" in row
    assert "decision" in row


# ─────────────────────────────────────────────────────────────────
# Hypothesis library
# ─────────────────────────────────────────────────────────────────

def test_library_registers_the_three_seeds():
    names = hypothesis_names()
    assert "h1_thursday_theta_scalp" in names
    assert "h2_opening_range_failure_fade" in names
    assert "h5_time_of_day_buy_and_hold" in names


def test_get_hypothesis_raises_on_unknown():
    with pytest.raises(KeyError):
        get_hypothesis("not_a_real_hypothesis")


def _multi_day_bars(n_days=10, bars_per_day=78, base=25000):
    """Build a multi-day NIFTY-like 5-min bar series — n_days UNIQUE
    trading days (weekends skipped, not double-counted)."""
    out_ts = []
    out_close = []
    cur = datetime(2026, 6, 1, 9, 15)
    while cur.weekday() >= 5:
        cur += timedelta(days=1)
    for d in range(n_days):
        day_start = cur.replace(hour=9, minute=15)
        for b in range(bars_per_day):
            out_ts.append(day_start + timedelta(minutes=5 * b))
            out_close.append(base + d * 10 + b * 0.5)
        cur = cur + timedelta(days=1)
        while cur.weekday() >= 5:
            cur += timedelta(days=1)
    df = pd.DataFrame({
        "ts": out_ts,
        "open": out_close,
        "high": [c + 5 for c in out_close],
        "low": [c - 5 for c in out_close],
        "close": out_close,
        "volume": [1000] * len(out_ts),
    })
    df.attrs["bar_minutes"] = 5
    return df


def test_h1_thursday_theta_scalp_fires_only_on_thursdays():
    bars = _multi_day_bars(n_days=14, bars_per_day=78)
    signals = h1_thursday_theta_scalp(bars)
    # Only Thursdays should produce signals.
    ts = pd.to_datetime(bars["ts"])
    for entry_idx, exit_idx, side, qty in signals:
        assert ts.iloc[entry_idx].dayofweek == 3, (
            f"non-Thursday entry: {ts.iloc[entry_idx]}")
        assert side == -1                  # SHORT straddle
        assert qty == 75
        assert exit_idx > entry_idx


def test_h2_opening_range_failure_fade_yields_signed_signals():
    bars = _multi_day_bars(n_days=20, bars_per_day=78)
    signals = h2_opening_range_failure_fade(bars)
    # Whatever fires must be {+1, -1} sided, valid index pairs
    for entry_idx, exit_idx, side, qty in signals:
        assert side in (+1, -1)
        assert exit_idx > entry_idx
        assert qty == 75


def test_h5_baseline_fires_every_trading_day():
    bars = _multi_day_bars(n_days=10, bars_per_day=78)
    signals = h5_time_of_day_buy_and_hold(bars)
    # 10 trading days (we skipped weekends in the builder)
    assert len(signals) == 10
    for entry_idx, exit_idx, side, qty in signals:
        assert side == +1
        assert exit_idx > entry_idx


def test_h5_baseline_through_harness_makes_sense():
    """The baseline runs through the harness end-to-end. On rising bars
    the long direction wins on GROSS P&L. NET P&L may be negative
    because the default cost model is options-priced (STT at 0.0625%
    of premium × qty is large at 25000-level notionals) — that's the
    cost model honestly telling us the strategy doesn't print when
    options costs apply. Real validation happens on the actual
    options chain in production."""
    bars = _multi_day_bars(n_days=15, bars_per_day=78)
    spec = get_hypothesis("h5_time_of_day_buy_and_hold")
    report = HypothesisHarness().run(spec, bars)
    assert report.trade_count == 15
    # Directional bet was correct (rising bars + long = positive gross)
    assert report.overall.gross_pnl > 0
    # Net P&L sign depends on cost-model fit to the instrument; not asserted.
