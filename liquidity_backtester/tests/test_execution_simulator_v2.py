from __future__ import annotations

import unittest

import pandas as pd

from liqpool.config import Config
from liqpool.execution_simulator_v2 import (
    ExecutionV2Config,
    PreTouchDirectionalSimulator,
    RupeeTargetExecutionConfig,
    compute_slippage_bps,
    compute_zerodha_intraday_costs,
    resolve_intrabar_path,
    simulate_pool_trade_v2,
)
from liqpool.pools import Pool
from liqpool.tester import PoolResult


def _one_minute_bars(rows):
    idx = pd.date_range("2026-05-26 09:15", periods=len(rows), freq="1min")
    return pd.DataFrame(rows, index=idx)


class ExecutionSimulatorV2Tests(unittest.TestCase):
    def test_zerodha_intraday_costs_are_itemized(self) -> None:
        costs = compute_zerodha_intraday_costs(100.0, 102.0, 10, exchange="NSE")

        self.assertAlmostEqual(costs["brokerage_buy"], 0.30, places=6)
        self.assertAlmostEqual(costs["brokerage_sell"], 0.306, places=6)
        self.assertAlmostEqual(costs["stt"], 0.255, places=6)
        self.assertAlmostEqual(costs["exchange_txn"], 0.062014, places=6)
        self.assertAlmostEqual(costs["sebi"], 0.00202, places=6)
        self.assertAlmostEqual(costs["stamp"], 0.03, places=6)
        self.assertGreater(costs["gst"], 0.0)
        self.assertGreater(costs["total_cost_inr"], 1.0)

    def test_intrabar_path_resolves_target_before_later_stop(self) -> None:
        bars = _one_minute_bars([
            {"open": 100.0, "high": 101.2, "low": 100.0, "close": 101.0, "volume": 1},
            {"open": 101.0, "high": 101.0, "low": 98.7, "close": 99.0, "volume": 1},
        ])

        out = resolve_intrabar_path(
            bars.index[0], 100.0, 99.0, 101.0, "TEST", "UP", bars,
        )

        self.assertEqual(out["resolution"], "target_hit")
        self.assertEqual(out["price"], 101.0)

    def test_intrabar_path_uses_stop_first_when_same_bar_ambiguous(self) -> None:
        bars = _one_minute_bars([
            {"open": 100.0, "high": 101.5, "low": 98.5, "close": 100.0, "volume": 1},
        ])

        out = resolve_intrabar_path(
            bars.index[0], 100.0, 99.0, 101.0, "TEST", "UP", bars,
        )

        self.assertEqual(out["resolution"], "stop_hit")
        self.assertEqual(out["price"], 99.0)

    def test_state_dependent_slippage_increases_near_level_and_open(self) -> None:
        mid_far = compute_slippage_bps(0.002, 5.0, "mid", base_bps=2.0)
        open_near = compute_slippage_bps(0.002, 0.5, "open", base_bps=2.0)

        self.assertGreater(open_near, mid_far)

    def test_pretouch_directional_simulator_can_target_pool_journey(self) -> None:
        bars_1m = _one_minute_bars([
            {"open": 100.0, "high": 101.0, "low": 99.8, "close": 100.8, "volume": 1},
            {"open": 100.8, "high": 103.0, "low": 100.7, "close": 102.8, "volume": 1},
            {"open": 102.8, "high": 105.4, "low": 102.6, "close": 105.1, "volume": 1},
        ])
        bars_5m = pd.DataFrame({
            "open": [100.0],
            "high": [105.4],
            "low": [99.8],
            "close": [105.1],
            "volume": [3],
        }, index=[bars_1m.index[0]])

        result = PreTouchDirectionalSimulator().simulate(
            pool_zone=(110.0, 111.0),
            pool_side="above",
            entry_timestamp=bars_1m.index[0],
            entry_price=100.0,
            bars_5m=bars_5m,
            bars_1m=bars_1m,
            atr_at_entry=1.0,
            target_fraction=0.5,
            stop_atr_mult=2.0,
            quantity=1,
            slippage_bps=0.0,
        )

        self.assertEqual(result["direction"], "UP")
        self.assertEqual(result["exit_reason"], "target")
        self.assertGreater(result["gross_pnl"], 0.0)

    def test_v2_skips_confirmed_entry_when_next_open_invalidates_geometry(self) -> None:
        idx = pd.date_range("2026-05-26 09:15", periods=22, freq="5min")
        df = pd.DataFrame({
            "open": [105.0] * 22,
            "high": [106.0] * 22,
            "low": [104.0] * 22,
            "close": [105.0] * 22,
            "volume": [1000.0] * 22,
        }, index=idx)
        df.loc[idx[16], ["open", "high", "low", "close"]] = [101.5, 106.0, 100.5, 102.0]
        df.loc[idx[17], ["open", "high", "low", "close"]] = [98.0, 99.0, 97.0, 98.2]
        pool = Pool(
            side="low",
            price_low=100.0,
            price_high=101.0,
            formed_at=idx[8],
            available_at=idx[10],
            contributors=[],
            score=1.0,
            tfs=["base"],
            asset="TEST",
        )
        result = PoolResult(
            pool_idx=0,
            side="low",
            formed_at=pool.formed_at,
            price_low=pool.price_low,
            price_high=pool.price_high,
            score=pool.score,
            outcome="respected_strong",
        )

        trade = simulate_pool_trade_v2(
            mode="touch_confirmed",
            symbol="TEST",
            sector="TEST",
            df_base=df,
            pool=pool,
            result=result,
            cfg=Config(test_horizon_bars=20),
            v2_cfg=ExecutionV2Config(fill_policy="neutral", use_1m_resolution=False),
        )

        self.assertIsNone(trade)

    def test_v2_refuses_new_entries_after_mis_cutoff(self) -> None:
        # Timestamps are UTC-naive in production. 09:05 UTC = 14:35 IST, after
        # the 14:30 no-new-MIS-entry cutoff.
        idx = pd.date_range("2026-05-26 08:50", periods=8, freq="5min")
        df = pd.DataFrame({
            "open": [105.0] * 8,
            "high": [106.0] * 8,
            "low": [103.0] * 8,
            "close": [105.0] * 8,
            "volume": [1000.0] * 8,
        }, index=idx)
        df.loc[idx[3], ["open", "high", "low", "close"]] = [105.0, 106.0, 99.5, 102.0]
        pool = Pool(
            side="low",
            price_low=100.0,
            price_high=101.0,
            formed_at=idx[0],
            available_at=idx[0],
            contributors=[],
            score=1.0,
            tfs=["base"],
            asset="TEST",
        )
        result = PoolResult(
            pool_idx=0,
            side="low",
            formed_at=pool.formed_at,
            price_low=pool.price_low,
            price_high=pool.price_high,
            score=pool.score,
            outcome="respected_strong",
        )

        trade = simulate_pool_trade_v2(
            mode="blind_limit",
            symbol="TEST",
            sector="TEST",
            df_base=df,
            pool=pool,
            result=result,
            cfg=Config(test_horizon_bars=20),
            v2_cfg=ExecutionV2Config(fill_policy="generous", use_1m_resolution=False),
        )

        self.assertIsNone(trade)

    def test_v2_caps_time_exit_at_same_day_mis_squareoff(self) -> None:
        # 08:30-09:45 UTC = 14:00-15:15 IST, followed by next session.
        same_day = list(pd.date_range("2026-05-26 08:30", periods=16, freq="5min"))
        idx = pd.DatetimeIndex(same_day + [pd.Timestamp("2026-05-27 03:45")])
        df = pd.DataFrame({
            "open": [103.0] * len(idx),
            "high": [104.0] * len(idx),
            "low": [102.0] * len(idx),
            "close": [103.0] * len(idx),
            "volume": [1000.0] * len(idx),
        }, index=idx)
        df.loc[idx[1], ["open", "high", "low", "close"]] = [103.0, 104.0, 100.5, 102.0]
        df.loc[idx[-1], ["open", "high", "low", "close"]] = [120.0, 121.0, 119.0, 120.0]
        pool = Pool(
            side="low",
            price_low=100.0,
            price_high=101.0,
            formed_at=idx[0],
            available_at=idx[0],
            contributors=[],
            score=1.0,
            tfs=["base"],
            asset="TEST",
        )
        result = PoolResult(
            pool_idx=0,
            side="low",
            formed_at=pool.formed_at,
            price_low=pool.price_low,
            price_high=pool.price_high,
            score=pool.score,
            outcome="respected_strong",
        )

        trade = simulate_pool_trade_v2(
            mode="blind_limit",
            symbol="TEST",
            sector="TEST",
            df_base=df,
            pool=pool,
            result=result,
            cfg=Config(test_horizon_bars=200, respect_within_bars=200),
            v2_cfg=ExecutionV2Config(fill_policy="generous", use_1m_resolution=False),
        )

        self.assertIsNotNone(trade)
        self.assertEqual(trade.exit_reason, "time_exit")
        self.assertEqual(trade.exit_at, "2026-05-26 09:45:00")
        self.assertLess(trade.exit_reference, 120.0)

    def test_v2_can_size_from_target_notional(self) -> None:
        idx = pd.date_range("2026-05-26 04:30", periods=22, freq="5min")
        df = pd.DataFrame({
            "open": [105.0] * 22,
            "high": [106.0] * 22,
            "low": [104.0] * 22,
            "close": [105.0] * 22,
            "volume": [1000.0] * 22,
        }, index=idx)
        df.loc[idx[3], ["open", "high", "low", "close"]] = [1002.0, 1006.0, 999.5, 1002.0]
        df.loc[idx[4], ["open", "high", "low", "close"]] = [1001.0, 1004.0, 1000.5, 1002.0]
        pool = Pool(
            side="low",
            price_low=999.0,
            price_high=1000.0,
            formed_at=idx[0],
            available_at=idx[0],
            contributors=[],
            score=1.0,
            tfs=["base"],
            asset="TEST",
        )
        result = PoolResult(
            pool_idx=0,
            side="low",
            formed_at=pool.formed_at,
            price_low=pool.price_low,
            price_high=pool.price_high,
            score=pool.score,
            outcome="respected_strong",
        )

        trade = simulate_pool_trade_v2(
            mode="blind_limit",
            symbol="TEST",
            sector="TEST",
            df_base=df,
            pool=pool,
            result=result,
            cfg=Config(test_horizon_bars=20),
            v2_cfg=ExecutionV2Config(fill_policy="generous",
                                     use_1m_resolution=False,
                                     slippage_model="flat",
                                     base_slippage_bps=0.0,
                                     quantity=1,
                                     notional_inr=100000.0),
        )

        self.assertIsNotNone(trade)
        self.assertEqual(trade.quantity, 100)

    def test_v2_rupee_target_sizes_to_reward_floor(self) -> None:
        idx = pd.date_range("2026-05-26 04:30", periods=8, freq="5min")
        df = pd.DataFrame({
            "open": [1004.0] * 8,
            "high": [1005.0] * 8,
            "low": [1003.0] * 8,
            "close": [1004.0] * 8,
            "volume": [1000.0] * 8,
        }, index=idx)
        df.loc[idx[2], ["open", "high", "low", "close"]] = [1002.0, 1003.0, 999.5, 1001.0]
        df.loc[idx[3], ["open", "high", "low", "close"]] = [1001.0, 1007.0, 1000.0, 1006.5]
        pool = Pool(
            side="low",
            price_low=999.0,
            price_high=1000.0,
            formed_at=idx[0],
            available_at=idx[0],
            contributors=[],
            score=1.0,
            tfs=["base"],
            asset="TEST",
        )
        result = PoolResult(
            pool_idx=0,
            side="low",
            formed_at=pool.formed_at,
            price_low=pool.price_low,
            price_high=pool.price_high,
            score=pool.score,
            outcome="respected_strong",
        )

        trade = simulate_pool_trade_v2(
            mode="blind_limit",
            symbol="TEST",
            sector="TEST",
            df_base=df,
            pool=pool,
            result=result,
            cfg=Config(test_horizon_bars=20),
            v2_cfg=ExecutionV2Config(
                fill_policy="generous",
                use_1m_resolution=False,
                slippage_model="flat",
                base_slippage_bps=0.0,
                rupee_target=RupeeTargetExecutionConfig(),
            ),
        )

        self.assertIsNotNone(trade)
        self.assertEqual(trade.sizing_mode, "rupee_target")
        self.assertEqual(trade.quantity, 100)
        self.assertAlmostEqual(trade.target_move_inr, 6.0)
        self.assertAlmostEqual(trade.target_reward_inr, 600.0)
        self.assertAlmostEqual(trade.entry_notional_inr, 100000.0)
        self.assertAlmostEqual(trade.stop, 997.0)
        self.assertAlmostEqual(trade.target, 1006.0)
        self.assertEqual(trade.exit_reason, "target")

    def test_v2_rupee_target_widens_move_when_max_notional_binds(self) -> None:
        idx = pd.date_range("2026-05-26 04:30", periods=8, freq="5min")
        df = pd.DataFrame({
            "open": [3004.0] * 8,
            "high": [3005.0] * 8,
            "low": [3003.0] * 8,
            "close": [3004.0] * 8,
            "volume": [1000.0] * 8,
        }, index=idx)
        df.loc[idx[2], ["open", "high", "low", "close"]] = [3002.0, 3003.0, 2999.5, 3001.0]
        df.loc[idx[3], ["open", "high", "low", "close"]] = [3001.0, 3010.0, 3000.0, 3009.5]
        pool = Pool(
            side="low",
            price_low=2999.0,
            price_high=3000.0,
            formed_at=idx[0],
            available_at=idx[0],
            contributors=[],
            score=1.0,
            tfs=["base"],
            asset="TEST",
        )
        result = PoolResult(
            pool_idx=0,
            side="low",
            formed_at=pool.formed_at,
            price_low=pool.price_low,
            price_high=pool.price_high,
            score=pool.score,
            outcome="respected_strong",
        )

        trade = simulate_pool_trade_v2(
            mode="blind_limit",
            symbol="TEST",
            sector="TEST",
            df_base=df,
            pool=pool,
            result=result,
            cfg=Config(test_horizon_bars=20),
            v2_cfg=ExecutionV2Config(
                fill_policy="generous",
                use_1m_resolution=False,
                slippage_model="flat",
                base_slippage_bps=0.0,
                rupee_target=RupeeTargetExecutionConfig(),
            ),
        )

        self.assertIsNotNone(trade)
        self.assertEqual(trade.quantity, 66)
        self.assertLessEqual(trade.entry_notional_inr, 200000.0)
        self.assertAlmostEqual(trade.target_move_inr, 600.0 / 66.0)
        self.assertAlmostEqual(trade.target_reward_inr, 600.0)
        self.assertAlmostEqual(trade.target, 3000.0 + 600.0 / 66.0)

    def test_v2_rupee_target_skips_when_min_notional_cannot_fit(self) -> None:
        idx = pd.date_range("2026-05-26 04:30", periods=8, freq="5min")
        df = pd.DataFrame({
            "open": [250004.0] * 8,
            "high": [250005.0] * 8,
            "low": [250003.0] * 8,
            "close": [250004.0] * 8,
            "volume": [1000.0] * 8,
        }, index=idx)
        df.loc[idx[2], ["open", "high", "low", "close"]] = [250002.0, 250003.0, 249999.5, 250001.0]
        pool = Pool(
            side="low",
            price_low=249999.0,
            price_high=250000.0,
            formed_at=idx[0],
            available_at=idx[0],
            contributors=[],
            score=1.0,
            tfs=["base"],
            asset="TEST",
        )
        result = PoolResult(
            pool_idx=0,
            side="low",
            formed_at=pool.formed_at,
            price_low=pool.price_low,
            price_high=pool.price_high,
            score=pool.score,
            outcome="respected_strong",
        )

        trade = simulate_pool_trade_v2(
            mode="blind_limit",
            symbol="TEST",
            sector="TEST",
            df_base=df,
            pool=pool,
            result=result,
            cfg=Config(test_horizon_bars=20),
            v2_cfg=ExecutionV2Config(
                fill_policy="generous",
                use_1m_resolution=False,
                slippage_model="flat",
                base_slippage_bps=0.0,
                rupee_target=RupeeTargetExecutionConfig(),
            ),
        )

        self.assertIsNone(trade)


if __name__ == "__main__":
    unittest.main()
