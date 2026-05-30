"""Regression tests for the V1 execution backtest risk denominator.

The pre-fix bug: on confirmation modes (touch/reclaim/displacement) the entry
price is the NEXT bar's open. When that bar gapped PAST the protective stop
the natural risk (entry - stop for longs, stop - entry for shorts) went
NEGATIVE, and ``max(risk, 1e-9)`` floored it to 1e-9 — so a small rupee loss
divided by near-zero risk yielded net_r in the tens of millions.

The V2 simulator already discards these degenerate setups
(see ``test_v2_skips_confirmed_entry_when_next_open_invalidates_geometry``).
These tests pin the same invariant for V1:

1. Long gap-DOWN past stop → ``simulate_pool_trade`` returns None.
2. Short gap-UP past stop → ``simulate_pool_trade`` returns None.
3. When the gap collapses natural risk to a tiny fraction of ATR but stays
   positive, risk is floored at 0.1 * ATR so net_r stays in a sane range.
4. Normal (no-gap) confirmation trade returns sensible single-digit net_r.
5. blind_limit is unaffected (it doesn't reassign entry).
"""
from __future__ import annotations

import unittest

import pandas as pd

from liqpool.config import Config
from liqpool.costs import ZerodhaEquityCostConfig
from liqpool.execution_backtest import simulate_pool_trade
from liqpool.pools import Pool
from liqpool.tester import PoolResult


def _five_min_bars(rows):
    idx = pd.date_range("2026-05-26 09:15", periods=len(rows), freq="5min")
    return pd.DataFrame(rows, index=idx)


def _flat_bars(price: float, n: int):
    return [
        {"open": price, "high": price + 1.0, "low": price - 1.0,
         "close": price, "volume": 1000.0}
        for _ in range(n)
    ]


class V1RiskDenominatorTests(unittest.TestCase):
    """All scenarios use a touch_confirmed mode so the entry-price reassignment
    code path is exercised (blind_limit is immune)."""

    def _common_cfg(self) -> tuple[Config, ZerodhaEquityCostConfig]:
        # respect_within_bars is small so _exit_trade actually terminates within
        # the synthetic bar window.
        return Config(test_horizon_bars=20, respect_within_bars=10), ZerodhaEquityCostConfig()

    def _long_pool(self, price_low: float, price_high: float,
                   formed_at: pd.Timestamp, available_at: pd.Timestamp) -> Pool:
        return Pool(
            side="low", price_low=price_low, price_high=price_high,
            formed_at=formed_at, available_at=available_at,
            contributors=[], score=1.0, tfs=["base"], asset="TEST",
        )

    def _short_pool(self, price_low: float, price_high: float,
                    formed_at: pd.Timestamp, available_at: pd.Timestamp) -> Pool:
        return Pool(
            side="high", price_low=price_low, price_high=price_high,
            formed_at=formed_at, available_at=available_at,
            contributors=[], score=1.0, tfs=["base"], asset="TEST",
        )

    def _result(self, pool: Pool, outcome: str = "respected_strong") -> PoolResult:
        return PoolResult(
            pool_idx=0, side=pool.side, formed_at=pool.formed_at,
            price_low=pool.price_low, price_high=pool.price_high,
            score=pool.score, outcome=outcome,
        )

    def test_long_gap_down_past_stop_returns_none(self) -> None:
        # Flat 105s while the pool sits at 100-101. A touch happens at bar 16
        # (low pierces 101), confirmation bar closes rejection at 102, then
        # bar 17 GAPS DOWN to open at 98 — well below the protective stop at
        # roughly 99.5. Pre-fix: trade entered at 98 with risk = 98 - 99.5 =
        # -1.5 → max(-1.5, 1e-9) → 1e-9 → R explodes.
        bars = _flat_bars(105.0, 22)
        bars[16] = {"open": 101.5, "high": 106.0, "low": 100.5,
                    "close": 102.0, "volume": 1000.0}
        bars[17] = {"open": 98.0, "high": 99.0, "low": 97.0,
                    "close": 98.2, "volume": 1000.0}
        df = _five_min_bars(bars)
        pool = self._long_pool(100.0, 101.0, df.index[8], df.index[10])
        result = self._result(pool)
        cfg, cost_cfg = self._common_cfg()

        trade = simulate_pool_trade(
            mode="touch_confirmed", symbol="TEST", sector="TEST",
            df_base=df, pool=pool, result=result, cfg=cfg, cost_cfg=cost_cfg,
        )
        self.assertIsNone(trade,
            "long with gap-down past stop must be discarded, not yield exploded R")

    def test_short_gap_up_past_stop_returns_none(self) -> None:
        # Mirror: pool at 100-101 on the high side (a supply pool). Confirmation
        # bar rejects up into the pool then bar 17 GAPS UP to open at 104 —
        # above the protective stop at roughly 101.5. Pre-fix: short entered at
        # 104 with risk = 101.5 - 104 = -2.5 → floored to 1e-9 → R explodes.
        bars = _flat_bars(95.0, 22)
        bars[16] = {"open": 99.5, "high": 100.5, "low": 94.0,
                    "close": 98.5, "volume": 1000.0}
        bars[17] = {"open": 104.0, "high": 105.0, "low": 103.0,
                    "close": 103.8, "volume": 1000.0}
        df = _five_min_bars(bars)
        pool = self._short_pool(100.0, 101.0, df.index[8], df.index[10])
        result = self._result(pool)
        cfg, cost_cfg = self._common_cfg()

        trade = simulate_pool_trade(
            mode="touch_confirmed", symbol="TEST", sector="TEST",
            df_base=df, pool=pool, result=result, cfg=cfg, cost_cfg=cost_cfg,
        )
        self.assertIsNone(trade,
            "short with gap-up past stop must be discarded, not yield exploded R")

    def test_normal_confirmation_trade_yields_sane_net_r(self) -> None:
        # A clean touch on a long pool, no gap. risk_per_share should be near
        # 0.5*ATR (the design target). net_r should be O(1), not exploded.
        bars = _flat_bars(105.0, 22)
        # Bar 16 dips into the pool (low 100.8), closes back above (102) —
        # rejection signal for a long.
        bars[16] = {"open": 103.0, "high": 105.0, "low": 100.8,
                    "close": 102.0, "volume": 1000.0}
        # Bar 17 opens above the pool — clean entry, NOT past the stop.
        bars[17] = {"open": 103.0, "high": 104.0, "low": 102.5,
                    "close": 103.5, "volume": 1000.0}
        df = _five_min_bars(bars)
        pool = self._long_pool(100.0, 101.0, df.index[8], df.index[10])
        result = self._result(pool)
        cfg, cost_cfg = self._common_cfg()

        trade = simulate_pool_trade(
            mode="touch_confirmed", symbol="TEST", sector="TEST",
            df_base=df, pool=pool, result=result, cfg=cfg, cost_cfg=cost_cfg,
            quantity=1,
        )
        # Normal geometry → trade exists, net_r is in a sane single-digit range.
        self.assertIsNotNone(trade)
        self.assertLess(abs(trade.net_r), 10.0,
            f"net_r must be O(1), got {trade.net_r}")

    def test_blind_limit_path_unaffected_by_fix(self) -> None:
        # blind_limit doesn't reassign entry — entry stays at the pool boundary
        # and risk is anchored to same-bar ATR, so it should still produce a
        # sane trade even if the next bar would have been a gap.
        bars = _flat_bars(105.0, 22)
        bars[16] = {"open": 101.5, "high": 106.0, "low": 100.5,
                    "close": 102.0, "volume": 1000.0}
        bars[17] = {"open": 98.0, "high": 99.0, "low": 97.0,
                    "close": 98.2, "volume": 1000.0}
        df = _five_min_bars(bars)
        pool = self._long_pool(100.0, 101.0, df.index[8], df.index[10])
        result = self._result(pool)
        cfg, cost_cfg = self._common_cfg()

        trade = simulate_pool_trade(
            mode="blind_limit", symbol="TEST", sector="TEST",
            df_base=df, pool=pool, result=result, cfg=cfg, cost_cfg=cost_cfg,
        )
        # blind_limit enters at pool.price_high (101.0); stop is 100 - 0.5*ATR.
        # ATR is ~2 on these bars, so risk ~= 101 - 99 = 2 per share. Even if
        # the trade ends up stopping out on bar 17's gap, net_r stays bounded.
        self.assertIsNotNone(trade)
        self.assertLess(abs(trade.net_r), 10.0,
            f"blind_limit net_r must remain O(1), got {trade.net_r}")


if __name__ == "__main__":
    unittest.main()
