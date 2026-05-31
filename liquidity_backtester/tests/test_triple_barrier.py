"""Tests for the triple-barrier label function.

The label is the foundation the force model trains on. These tests pin:
  * Target hits: long target hit → +target_atr/stop_atr R; short symmetric.
  * Stop hits: monotone -1R, worse if gap-through.
  * Time exits: continuous close-to-entry distance in stop units.
  * MIS cap: a trade entered late in the session gets EOD-squareoff, not
    overnight horizon.
  * Same-bar ambiguity: when bar range covers both barriers, stop is
    chosen (conservative).
  * Edge cases: zero atr, empty frame, out-of-bounds entry_idx, sides not
    in {long, short}.
"""
from __future__ import annotations

import unittest

import pandas as pd

from liqpool.triple_barrier import triple_barrier_label


def _bars(rows, start_ist: str = "10:00"):
    """Build a 5-min OHLCV frame whose IST time-of-day starts at start_ist.

    Matches the convention used by tests/test_execution_backtest_risk.py:
    tz-naive timestamps are treated as UTC; IST = UTC + 5:30. So to get
    10:00 IST we offset the UTC clock to 04:30.
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


class TripleBarrierLabelTests(unittest.TestCase):

    def test_long_target_hit_returns_rr_ratio(self) -> None:
        # entry at 100.0, atr=1.0, stop_atr=0.5 -> stop@99.5, target_atr=2.0 -> target@102.0
        # bar 1 hits high=102.5 (>= 102.0) and low=99.8 (above stop). target hit.
        # R = (102.0 - 100.0) / (0.5 * 1.0) = 4.0
        rows = [{"open": 100.0, "high": 100.5, "low": 99.8, "close": 100.0, "volume": 0}]
        rows += [{"open": 100.2, "high": 102.5, "low": 99.8, "close": 102.0, "volume": 0}]
        rows += [{"open": 102.0, "high": 102.5, "low": 101.5, "close": 102.0, "volume": 0}]
        df = _bars(rows, start_ist="10:00")
        r, reason, exit_idx = triple_barrier_label(
            df, entry_idx=0, side="long", entry_price=100.0,
            stop_atr=0.5, target_atr=2.0, horizon_bars=10, atr_value=1.0,
        )
        self.assertEqual(reason, "target")
        self.assertAlmostEqual(r, 4.0, places=6)
        self.assertEqual(exit_idx, 1)

    def test_short_target_hit_symmetric(self) -> None:
        # short at 100.0, atr=1.0, stop_atr=0.5 -> stop@100.5, target_atr=2.0 -> target@98.0
        rows = [{"open": 100.0, "high": 100.3, "low": 99.5, "close": 100.0, "volume": 0}]
        rows += [{"open": 99.8, "high": 100.2, "low": 97.5, "close": 98.0, "volume": 0}]
        rows += [{"open": 98.0, "high": 98.5, "low": 97.5, "close": 98.0, "volume": 0}]
        df = _bars(rows, start_ist="10:00")
        r, reason, _ = triple_barrier_label(
            df, entry_idx=0, side="short", entry_price=100.0,
            stop_atr=0.5, target_atr=2.0, horizon_bars=10, atr_value=1.0,
        )
        self.assertEqual(reason, "target")
        self.assertAlmostEqual(r, 4.0, places=6)

    def test_long_stop_hit_returns_minus_one(self) -> None:
        # entry 100, stop@99.5, target@102.0. Bar 1 low=99.0 (gap-through? No, open=99.8 > stop).
        # fill = min(open=99.8, stop=99.5) = 99.5. R = (99.5 - 100) / 0.5 = -1.0
        rows = [{"open": 100.0, "high": 100.5, "low": 99.8, "close": 100.0, "volume": 0}]
        rows += [{"open": 99.8, "high": 100.0, "low": 99.0, "close": 99.2, "volume": 0}]
        df = _bars(rows, start_ist="10:00")
        r, reason, _ = triple_barrier_label(
            df, entry_idx=0, side="long", entry_price=100.0,
            stop_atr=0.5, target_atr=2.0, horizon_bars=10, atr_value=1.0,
        )
        self.assertEqual(reason, "stop")
        self.assertAlmostEqual(r, -1.0, places=6)

    def test_long_stop_gap_through_fills_at_open(self) -> None:
        # entry 100, stop@99.5. Bar 1 GAPS DOWN: open=99.0 (already past stop), low=98.5.
        # fill = min(open=99.0, stop=99.5) = 99.0. R = (99.0 - 100) / 0.5 = -2.0  (WORSE than -1).
        rows = [{"open": 100.0, "high": 100.5, "low": 99.8, "close": 100.0, "volume": 0}]
        rows += [{"open": 99.0, "high": 99.0, "low": 98.5, "close": 98.7, "volume": 0}]
        df = _bars(rows, start_ist="10:00")
        r, reason, _ = triple_barrier_label(
            df, entry_idx=0, side="long", entry_price=100.0,
            stop_atr=0.5, target_atr=2.0, horizon_bars=10, atr_value=1.0,
        )
        self.assertEqual(reason, "stop")
        self.assertAlmostEqual(r, -2.0, places=6)

    def test_same_bar_ambiguity_resolves_to_stop(self) -> None:
        # entry 100, stop@99.5, target@102.0. Bar 1 has both: low=99.0 AND high=103.0.
        # Conservative: stop hits first (mirrors V1 simulator).
        rows = [{"open": 100.0, "high": 100.5, "low": 99.8, "close": 100.0, "volume": 0}]
        rows += [{"open": 100.2, "high": 103.0, "low": 99.0, "close": 102.5, "volume": 0}]
        df = _bars(rows, start_ist="10:00")
        r, reason, _ = triple_barrier_label(
            df, entry_idx=0, side="long", entry_price=100.0,
            stop_atr=0.5, target_atr=2.0, horizon_bars=10, atr_value=1.0,
        )
        self.assertEqual(reason, "stop")
        self.assertLess(r, 0.0)

    def test_time_exit_returns_continuous_realized_R(self) -> None:
        # No stop, no target hit. Final close = 100.4. stop_dist = 0.5.
        # R = (100.4 - 100.0) / 0.5 = +0.8
        rows = [{"open": 100.0, "high": 100.4, "low": 99.7, "close": 100.0, "volume": 0}]
        rows += [{"open": 100.0, "high": 100.4, "low": 99.7, "close": 100.2, "volume": 0}]
        rows += [{"open": 100.2, "high": 100.4, "low": 99.8, "close": 100.4, "volume": 0}]
        df = _bars(rows, start_ist="10:00")
        r, reason, exit_idx = triple_barrier_label(
            df, entry_idx=0, side="long", entry_price=100.0,
            stop_atr=0.5, target_atr=10.0, horizon_bars=2, atr_value=1.0,
        )
        self.assertEqual(reason, "time_exit")
        self.assertAlmostEqual(r, 0.8, places=6)
        self.assertEqual(exit_idx, 2)

    def test_eod_squareoff_caps_horizon_within_session(self) -> None:
        # Entry at 15:00 IST. horizon_bars = 60 would otherwise extend to next day,
        # but EOD square-off at 15:15 IST means at most 3 more bars are usable
        # (15:05, 15:10, 15:15). Reason should be "eod_squareoff".
        rows = [
            {"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0, "volume": 0}
            for _ in range(20)
        ]
        df = _bars(rows, start_ist="15:00")
        r, reason, exit_idx = triple_barrier_label(
            df, entry_idx=0, side="long", entry_price=100.0,
            stop_atr=0.5, target_atr=10.0, horizon_bars=60, atr_value=1.0,
        )
        self.assertEqual(reason, "eod_squareoff")
        self.assertLessEqual(exit_idx, 3)

    def test_no_room_when_entry_is_at_end_of_session(self) -> None:
        # Entry at 15:20 IST, only EOD bar at 15:25 IST. last_intraday_bar_idx
        # stops at the bar with minute <= 15:15 IST -> no room.
        rows = [
            {"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0, "volume": 0}
            for _ in range(5)
        ]
        df = _bars(rows, start_ist="15:20")
        r, reason, _ = triple_barrier_label(
            df, entry_idx=0, side="long", entry_price=100.0,
            stop_atr=0.5, target_atr=2.0, horizon_bars=60, atr_value=1.0,
        )
        self.assertEqual(reason, "no_room")
        self.assertEqual(r, 0.0)

    def test_invalid_side_raises(self) -> None:
        df = _bars([{"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 0}])
        with self.assertRaises(ValueError):
            triple_barrier_label(
                df, entry_idx=0, side="up", entry_price=100.0,
                stop_atr=0.5, target_atr=2.0, horizon_bars=5, atr_value=1.0,
            )

    def test_empty_or_out_of_bounds_returns_zero(self) -> None:
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        r, reason, _ = triple_barrier_label(
            empty, entry_idx=0, side="long", entry_price=100.0,
            stop_atr=0.5, target_atr=2.0, horizon_bars=5, atr_value=1.0,
        )
        self.assertEqual((r, reason), (0.0, "no_room"))

        df = _bars([{"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 0}])
        r, reason, _ = triple_barrier_label(
            df, entry_idx=99, side="long", entry_price=100.0,
            stop_atr=0.5, target_atr=2.0, horizon_bars=5, atr_value=1.0,
        )
        self.assertEqual((r, reason), (0.0, "no_room"))

    def test_zero_atr_falls_back_safely(self) -> None:
        # atr=0 -> stop_dist=0. Should not divide by zero; return no_room or
        # graceful zero. Defensive guard: max(atr, 1e-9) so stop_dist becomes
        # tiny; any nonzero move is huge R. We just check no crash.
        df = _bars([{"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 0}])
        r, reason, _ = triple_barrier_label(
            df, entry_idx=0, side="long", entry_price=100.0,
            stop_atr=0.5, target_atr=2.0, horizon_bars=5, atr_value=0.0,
        )
        # Single-bar df, no following bars -> no_room
        self.assertEqual(reason, "no_room")


if __name__ == "__main__":
    unittest.main()
