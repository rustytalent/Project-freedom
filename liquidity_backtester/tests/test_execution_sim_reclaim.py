from __future__ import annotations

import unittest

import pandas as pd

from liqpool.config import Config, DetectionParams
from liqpool.execution_simulator_v2 import ExecutionV2Config, simulate_pool_trade_v2
from liqpool.pools import Pool
from liqpool.tester import PoolResult


def _five_min_bars_from_ist(rows, start_ist: str = "10:00") -> pd.DataFrame:
    h, m = start_ist.split(":")
    ist_min = int(h) * 60 + int(m)
    utc_min = ist_min - (5 * 60 + 30)
    if utc_min < 0:
        utc_min += 24 * 60
    utc_h, utc_m = divmod(utc_min, 60)
    idx = pd.date_range(f"2026-05-26 {utc_h:02d}:{utc_m:02d}",
                        periods=len(rows), freq="5min")
    return pd.DataFrame(rows, index=idx)


def _cfg() -> Config:
    return Config(
        test_horizon_bars=24,
        respect_within_bars=12,
        reclaim_within_bars=4,
        strong_break_atr=0.10,
        detect=DetectionParams(atr_period=2),
    )


def _pool(side: str, idx: pd.DatetimeIndex) -> Pool:
    return Pool(
        side=side,
        price_low=100.0,
        price_high=101.0,
        formed_at=idx[0],
        available_at=idx[0],
        contributors=[],
        score=1.0,
        tfs=["base"],
        asset="TEST",
    )


def _result(pool: Pool) -> PoolResult:
    return PoolResult(
        pool_idx=0,
        side=pool.side,
        formed_at=pool.formed_at,
        price_low=pool.price_low,
        price_high=pool.price_high,
        score=pool.score,
        outcome="swept_and_reclaimed",
        tfs=list(pool.tfs),
    )


class SweepReclaimModeTests(unittest.TestCase):
    def test_entry_only_fires_after_confirmed_reclaim_bar(self) -> None:
        rows = [
            {"open": 105.0, "high": 106.0, "low": 104.0, "close": 105.0, "volume": 1000},
            {"open": 105.0, "high": 106.0, "low": 104.0, "close": 105.0, "volume": 1000},
            # Strong sweep through the low-side pool, but no reclaim yet.
            {"open": 101.2, "high": 101.5, "low": 97.5, "close": 98.0, "volume": 1000},
            {"open": 98.0, "high": 99.5, "low": 97.8, "close": 99.0, "volume": 1000},
            # Reclaim closes back inside the pool; entry must be next bar open.
            {"open": 99.0, "high": 100.8, "low": 98.8, "close": 100.5, "volume": 1000},
            {"open": 100.8, "high": 102.8, "low": 100.6, "close": 102.2, "volume": 1000},
            {"open": 102.2, "high": 103.5, "low": 102.0, "close": 103.0, "volume": 1000},
        ]
        df = _five_min_bars_from_ist(rows)
        pool = _pool("low", df.index)

        trade = simulate_pool_trade_v2(
            mode="sweep_reclaim",
            symbol="TEST",
            sector="TEST",
            df_base=df,
            pool=pool,
            result=_result(pool),
            cfg=_cfg(),
            v2_cfg=ExecutionV2Config(fill_policy="generous", slippage_model="flat",
                                     base_slippage_bps=0.0),
        )

        self.assertIsNotNone(trade)
        self.assertEqual(trade.entry_at, str(df.index[5]))
        self.assertEqual(trade.entry_reason, "sweep_reclaim_close:next_bar_confirmation")
        self.assertEqual(trade.direction, "UP")

    def test_no_lookahead_stop_uses_extreme_before_entry_only(self) -> None:
        rows = [
            {"open": 105.0, "high": 106.0, "low": 104.0, "close": 105.0, "volume": 1000},
            {"open": 101.2, "high": 101.5, "low": 97.5, "close": 98.0, "volume": 1000},
            {"open": 98.0, "high": 100.8, "low": 98.0, "close": 100.5, "volume": 1000},
            # This extreme happens after the reclaim bar. It may stop the trade,
            # but it must not be used to PLACE the initial stop.
            {"open": 100.7, "high": 101.0, "low": 90.0, "close": 91.0, "volume": 1000},
            {"open": 91.0, "high": 92.0, "low": 90.5, "close": 91.5, "volume": 1000},
        ]
        df = _five_min_bars_from_ist(rows)
        pool = _pool("low", df.index)

        trade = simulate_pool_trade_v2(
            mode="sweep_reclaim",
            symbol="TEST",
            sector="TEST",
            df_base=df,
            pool=pool,
            result=_result(pool),
            cfg=_cfg(),
            v2_cfg=ExecutionV2Config(fill_policy="generous", slippage_model="flat",
                                     base_slippage_bps=0.0),
        )

        self.assertIsNotNone(trade)
        self.assertEqual(trade.stop, 97.5)
        self.assertNotEqual(trade.stop, 90.0)

    def test_short_side_stop_and_target_placement(self) -> None:
        rows = [
            {"open": 95.0, "high": 96.0, "low": 94.0, "close": 95.0, "volume": 1000},
            {"open": 95.0, "high": 96.0, "low": 94.0, "close": 95.0, "volume": 1000},
            # Strong sweep upward through a high-side pool.
            {"open": 99.8, "high": 103.5, "low": 99.5, "close": 103.0, "volume": 1000},
            # Reclaim closes back inside the zone.
            {"open": 103.0, "high": 103.2, "low": 100.2, "close": 100.5, "volume": 1000},
            {"open": 100.2, "high": 100.4, "low": 98.0, "close": 98.5, "volume": 1000},
            {"open": 98.5, "high": 99.0, "low": 97.0, "close": 97.5, "volume": 1000},
        ]
        df = _five_min_bars_from_ist(rows)
        pool = _pool("high", df.index)

        trade = simulate_pool_trade_v2(
            mode="sweep_reclaim",
            symbol="TEST",
            sector="TEST",
            df_base=df,
            pool=pool,
            result=_result(pool),
            cfg=_cfg(),
            v2_cfg=ExecutionV2Config(fill_policy="generous", slippage_model="flat",
                                     base_slippage_bps=0.0,
                                     reclaim_target_atr_mult=0.5),
        )

        self.assertIsNotNone(trade)
        self.assertEqual(trade.direction, "DOWN")
        self.assertEqual(trade.stop, 103.5)
        self.assertLess(trade.target, pool.price_low)


if __name__ == "__main__":
    unittest.main()
