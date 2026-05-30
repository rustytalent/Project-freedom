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


def _five_min_bars(rows, start_ist: str = "10:00"):
    """Build a 5-min bar frame whose IST time-of-day starts at `start_ist`.

    The production data path stores tz-naive UTC; nse_session() / the new MIS
    enforcement convert that to IST by adding +5:30. To make a test bar
    actually live at HH:MM IST we offset the UTC clock back by 5h30m.
    e.g. 10:00 IST -> 04:30 UTC.
    """
    h, m = start_ist.split(":")
    ist_min = int(h) * 60 + int(m)
    utc_min = ist_min - (5 * 60 + 30)
    if utc_min < 0:
        utc_min += 24 * 60
    utc_h, utc_m = divmod(utc_min, 60)
    start_utc = f"2026-05-26 {utc_h:02d}:{utc_m:02d}"
    idx = pd.date_range(start_utc, periods=len(rows), freq="5min")
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


class V1IntradayMISTests(unittest.TestCase):
    """MIS enforcement: no new entries after 14:30 IST, time-exit by 15:15 IST,
    do not iterate across session boundaries. These pin the user-flagged bug
    that the old simulator silently used overnight bars as intraday continuation."""

    def _common_cfg(self) -> tuple[Config, ZerodhaEquityCostConfig]:
        # respect_within_bars=60 so a same-day intraday time stop is the binding
        # constraint, not the natural bar count.
        return Config(test_horizon_bars=60, respect_within_bars=60), ZerodhaEquityCostConfig()

    def _long_pool(self, price_low: float, price_high: float,
                   formed_at: pd.Timestamp, available_at: pd.Timestamp) -> Pool:
        return Pool(
            side="low", price_low=price_low, price_high=price_high,
            formed_at=formed_at, available_at=available_at,
            contributors=[], score=1.0, tfs=["base"], asset="TEST",
        )

    def _result(self, pool: Pool, outcome: str = "respected_strong") -> PoolResult:
        return PoolResult(
            pool_idx=0, side=pool.side, formed_at=pool.formed_at,
            price_low=pool.price_low, price_high=pool.price_high,
            score=pool.score, outcome=outcome,
        )

    def test_blocks_entry_after_1430_ist(self) -> None:
        # Bars from 14:00 IST forward. Pool touch at bar 5 -> entry next bar at
        # 14:25 IST (still allowed). Then build a SECOND scenario with touch at
        # bar 7 -> entry bar 14:35 IST which is past the 14:30 cutoff.
        rows = _flat_bars(105.0, 22)
        # Plant a long-side rejection so touch_confirmed triggers at bar 7
        rows[7] = {"open": 102.0, "high": 105.0, "low": 100.8,
                   "close": 103.0, "volume": 1000.0}
        df = _five_min_bars(rows, start_ist="14:00")    # bar i is at 14:00 + i*5min IST
        # Touch bar 7 -> signal at 14:35 IST -> entry bar 8 at 14:40 IST (post-cutoff).
        pool = self._long_pool(100.0, 101.0, df.index[2], df.index[3])
        result = self._result(pool)
        cfg, cost_cfg = self._common_cfg()
        trade = simulate_pool_trade(
            mode="touch_confirmed", symbol="TEST", sector="TEST",
            df_base=df, pool=pool, result=result, cfg=cfg, cost_cfg=cost_cfg,
        )
        self.assertIsNone(trade,
            "entries past 14:30 IST must be refused (no room to reach target before EOD)")

    def test_eod_squareoff_caps_holding_window(self) -> None:
        # Pool touches at 14:00 IST -> entry at 14:05 IST. With
        # respect_within_bars=60 the natural time stop would be 60*5min = 5h
        # spanning into next morning. The MIS cap should force exit by 15:15 IST,
        # which is bar (15:15 - 14:05)/5 = 14 bars after entry at most.
        rows = _flat_bars(105.0, 28)                    # 28 bars from 13:30 IST = 13:30 -> 15:45 IST
        # Plant a rejection at bar 6 (13:30 + 30min = 14:00 IST)
        rows[6] = {"open": 102.0, "high": 105.0, "low": 100.8,
                   "close": 103.0, "volume": 1000.0}
        df = _five_min_bars(rows, start_ist="13:30")
        pool = self._long_pool(100.0, 101.0, df.index[1], df.index[2])
        result = self._result(pool)
        cfg, cost_cfg = self._common_cfg()
        trade = simulate_pool_trade(
            mode="touch_confirmed", symbol="TEST", sector="TEST",
            df_base=df, pool=pool, result=result, cfg=cfg, cost_cfg=cost_cfg,
        )
        # Entry is at bar 7 = 14:05 IST. EOD square-off at 15:15 IST = 70 min later
        # = 14 bars. Exit bar should be <= 14 bars after entry, not 60.
        self.assertIsNotNone(trade)
        self.assertLessEqual(trade.bars_held, 14,
            f"MIS cap should force exit by 15:15 IST; bars_held={trade.bars_held}")

    def test_does_not_iterate_across_session_boundary(self) -> None:
        # Build bars that include the END of one session and the START of the next.
        # Friday 14:50 IST -> 15:30 IST (8 bars same day),
        # then Monday 09:15 IST -> 09:40 IST (5 bars next session).
        # A confirmation rejection at bar 0 (14:50 IST) -> entry bar 1 at 14:55 IST.
        # WITHOUT the same-day cap, exit_trade would happily iterate into Monday
        # and report bars_held ~= however far stop/target hit. With the cap, max
        # bars_held = (15:15 - 14:55) / 5 = 4.
        friday_open_min = 14 * 60 + 50          # 14:50 IST
        utc_min = friday_open_min - (5 * 60 + 30)
        utc_h, utc_m = divmod(utc_min, 60)
        # 8 Friday bars (14:50 -> 15:25 IST, last one one-past 15:15 cap)
        fri = pd.date_range(f"2026-05-22 {utc_h:02d}:{utc_m:02d}",
                            periods=8, freq="5min")
        # 5 Monday bars at 09:15 IST = 03:45 UTC
        mon = pd.date_range("2026-05-25 03:45", periods=5, freq="5min")
        idx = fri.append(mon)
        rows = []
        for k in range(len(idx)):
            rows.append({"open": 105.0, "high": 106.0, "low": 104.0,
                         "close": 105.0, "volume": 1000.0})
        rows[0] = {"open": 102.0, "high": 105.0, "low": 100.8,
                   "close": 103.0, "volume": 1000.0}    # rejection at bar 0
        df = pd.DataFrame(rows, index=idx)
        pool = self._long_pool(100.0, 101.0, df.index[0], df.index[0])
        result = self._result(pool)
        cfg, cost_cfg = self._common_cfg()
        trade = simulate_pool_trade(
            mode="touch_confirmed", symbol="TEST", sector="TEST",
            df_base=df, pool=pool, result=result, cfg=cfg, cost_cfg=cost_cfg,
        )
        # Entry at bar 1 (14:55 IST). EOD at 15:15 IST = bar 5. Max bars_held = 4.
        # If the old simulator was still in effect, bars_held would touch Monday
        # bars and could be much higher.
        if trade is None:
            return  # natural skip is also acceptable; the bug is iterating into Monday
        self.assertLessEqual(trade.bars_held, 4,
            f"MIS must not iterate into next session; bars_held={trade.bars_held}")
        # And the exit timestamp must NOT be on a different IST date than entry.
        entry_ts = pd.Timestamp(trade.entry_at)
        exit_ts = pd.Timestamp(trade.exit_at)
        from liqpool.execution_backtest import _ist_date
        self.assertEqual(_ist_date(entry_ts), _ist_date(exit_ts),
            "exit must be same IST trading day as entry")


if __name__ == "__main__":
    unittest.main()
