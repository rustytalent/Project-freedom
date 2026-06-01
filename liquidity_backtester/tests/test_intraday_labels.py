"""Same-session MIS-aware label generation tests.

Pre-fix bug (documented in docs/CODEX_HANDOFF.md): the upstream models
(quality, direction, proximity, reaction) were trained on labels that
iterated bars across overnight gaps. Concretely:

  * tester.test_pools used a 200-bar window (~2.7 calendar days) per pool;
    a pool's "respect" or "break" could be credited to next-session price
    action.
  * timing.generate_snapshots built future_max_high / future_min_low arrays
    that included overnight bars; direction_label and pool_touch_labels
    inherited that.

Under intraday MIS rules these labels are wrong: a trader cannot hold to
the next day. The fix caps both label windows at the same IST trading day
EOD (minute <= 15:15 IST), gated by ``cfg.intraday_session_only`` for
tester and the ``intraday_session_only`` kwarg for ``generate_snapshots``.

These tests pin the new contract.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from liqpool.config import Config
from liqpool.intraday import EOD_SQUAREOFF_IST_MIN, ist_date, ist_minute_of_day
from liqpool.pools import Pool
from liqpool.tester import test_pools as run_test_pools     # rename so pytest
                                                            # doesn't collect it
from liqpool.timing import StateFeaturizer, generate_snapshots


# ---------------------------------------------------------------------------
# Helpers — build bars whose IST clock starts at the requested HH:MM IST.
# ---------------------------------------------------------------------------

def _bars_ist(n_bars: int, start_ist: str, base_price: float = 100.0,
              freq_min: int = 5) -> pd.DataFrame:
    """Build OHLCV bars whose IST clock starts at start_ist (HH:MM).

    The codebase's convention is tz-naive == UTC. To get bars that actually
    live at IST time-of-day HH:MM we offset the UTC clock back by 5h30m.
    """
    h, m = (int(x) for x in start_ist.split(":"))
    ist_min = h * 60 + m
    utc_min = ist_min - (5 * 60 + 30)
    if utc_min < 0:
        utc_min += 24 * 60
    utc_h, utc_m = divmod(utc_min, 60)
    start_utc = f"2026-05-26 {utc_h:02d}:{utc_m:02d}"
    idx = pd.date_range(start_utc, periods=n_bars, freq=f"{freq_min}min")
    rng = np.random.default_rng(0)
    close = base_price + np.cumsum(rng.normal(0.0, 0.05, n_bars))
    return pd.DataFrame({
        "open":   close,
        "high":   close + 0.5,
        "low":    close - 0.5,
        "close":  close,
        "volume": np.full(n_bars, 1000.0),
    }, index=idx)


def _bars_three_sessions(n_per_session: int = 75) -> pd.DataFrame:
    """Three back-to-back NSE sessions of n_per_session bars each.

    Three sessions gives us enough total bars (>=225) that snapshot indices
    past the j>=80 generate_snapshots warm-up land on session 2 or 3.
    Session 1 base price 100, session 2 base 200, session 3 base 150 —
    the price discontinuities make cross-session label leakage easy to detect.
    """
    # Friday 03:45 UTC = Friday 09:15 IST
    sess1_idx = pd.date_range("2026-05-22 03:45", periods=n_per_session, freq="5min")
    # Monday 03:45 UTC = Monday 09:15 IST
    sess2_idx = pd.date_range("2026-05-25 03:45", periods=n_per_session, freq="5min")
    # Tuesday 03:45 UTC = Tuesday 09:15 IST
    sess3_idx = pd.date_range("2026-05-26 03:45", periods=n_per_session, freq="5min")

    def _ohlc(idx, base, seed):
        rng = np.random.default_rng(seed)
        close = base + np.cumsum(rng.normal(0.0, 0.05, len(idx)))
        return pd.DataFrame({
            "open":   close, "high":   close + 0.5, "low":    close - 0.5,
            "close":  close, "volume": np.full(len(idx), 1000.0),
        }, index=idx)

    return pd.concat([
        _ohlc(sess1_idx, 100.0, 0),
        _ohlc(sess2_idx, 200.0, 1),
        _ohlc(sess3_idx, 150.0, 2),
    ])


def _pool(price_low: float, price_high: float, formed_at, available_at,
          side: str = "low") -> Pool:
    return Pool(
        side=side, price_low=price_low, price_high=price_high,
        formed_at=formed_at, available_at=available_at,
        contributors=[], score=1.0, tfs=["base"], asset="TEST",
    )


# ---------------------------------------------------------------------------
# tester.test_pools — intraday cap
# ---------------------------------------------------------------------------

class IntradayTesterTests(unittest.TestCase):
    """Pin that the per-pool label window is capped at same-session EOD when
    ``cfg.intraday_session_only`` is True."""

    def _cfg(self, intraday: bool = True) -> Config:
        cfg = Config()
        cfg.test_horizon_bars = 200          # the buggy default
        cfg.respect_within_bars = 12         # small enough that morning pools test
        cfg.intraday_session_only = intraday
        return cfg

    def test_intraday_caps_horizon_within_session(self) -> None:
        # 75 morning bars; pool becomes available at bar 10 (~10:05 IST).
        # Without intraday cap, end = 10 + 200 = 210 -> hits len(df)=75 anyway,
        # so the bug only bites when df extends past EOD. Use the two-session
        # fixture below for the real test; this one verifies the legacy path
        # behaves identically when there are no overnight bars to cross.
        df = _bars_ist(75, start_ist="09:15")     # 09:15 -> 15:25 IST
        p = _pool(price_low=80.0, price_high=81.0,
                  formed_at=df.index[5], available_at=df.index[10])
        cfg = self._cfg(intraday=True)
        results = run_test_pools(df, [p], cfg)
        self.assertEqual(len(results), 1)
        # The forward window is the same in both modes when df ends mid-session.
        results_legacy = run_test_pools(df, [p], self._cfg(intraday=False))
        self.assertEqual(results[0].outcome, results_legacy[0].outcome)

    def test_intraday_blocks_label_from_reading_next_session(self) -> None:
        # Three-session fixture. Pool available late on session 1 (bar 70 of
        # session 1 = 15:05 IST). Pool zone is set so price touches it on
        # session 2 morning (where the base price jumps to 200.0). Legacy
        # mode reaches across the overnight gap and credits a touch; intraday
        # mode must NOT.
        df = _bars_three_sessions(n_per_session=75)
        # Bar 70 of session 1 = ~15:05 IST.
        p = _pool(price_low=200.0, price_high=201.0,
                  formed_at=df.index[65], available_at=df.index[70],
                  side="high")    # supply zone

        intraday_results = run_test_pools(df, [p], self._cfg(intraday=True))
        legacy_results = run_test_pools(df, [p], self._cfg(intraday=False))
        self.assertEqual(intraday_results[0].touched_at, None,
            "intraday MIS labels must NOT read across the overnight gap")
        self.assertIsNotNone(legacy_results[0].touched_at,
            "legacy mode should still see the cross-session touch — confirms "
            "the fixture actually exposes the bug")

    def test_intraday_horizon_insufficient_for_late_session_pool(self) -> None:
        # Pool available very late in the session — few same-session bars
        # remain. With min_horizon ~= respect_within_bars+1 = 13, this pool
        # may be marked "horizon_insufficient" under intraday mode.
        df = _bars_ist(75, start_ist="09:15")
        # Bar 70 ~ 15:05 IST; only ~3 bars left to 15:15 IST EOD cap.
        p = _pool(price_low=80.0, price_high=81.0,
                  formed_at=df.index[68], available_at=df.index[70])
        cfg = self._cfg(intraday=True)
        cfg.respect_within_bars = 20         # bumps min_horizon to 21
        results = run_test_pools(df, [p], cfg)
        self.assertEqual(results[0].outcome, "horizon_insufficient")


# ---------------------------------------------------------------------------
# timing.generate_snapshots — intraday future window
# ---------------------------------------------------------------------------

class IntradaySnapshotTests(unittest.TestCase):
    """Pin that direction/proximity label windows in ``generate_snapshots``
    are capped at same-session EOD when ``intraday_session_only=True``."""

    def test_future_window_capped_at_session_end(self) -> None:
        # 3-session df. Pick a snapshot in session 2 at 14:00 IST. With 75
        # session-1 bars + ~57 bars into session 2 (09:15 -> 14:00), the
        # snapshot lands at bar ~132 — past the generate_snapshots j>=80
        # warm-up. Under legacy mode the future window of 60 bars extends
        # into session 3; under intraday mode it caps at session 2 EOD
        # (15:15 IST = ~15 bars after the snapshot).
        df = _bars_three_sessions(n_per_session=75)
        # Find the session-2 bar at 14:00 IST (must be past bar 80).
        snapshot_idx = None
        for i in range(80, len(df)):
            if ist_minute_of_day(df.index[i]) == 14 * 60:
                snapshot_idx = i
                break
        self.assertIsNotNone(snapshot_idx,
            "expected a 14:00 IST bar after warm-up in the 3-session fixture")

        # An "active" demand pool below price (untouched).
        p = _pool(price_low=50.0, price_high=51.0,
                  formed_at=df.index[0], available_at=df.index[0])
        from liqpool.tester import PoolResult
        r = PoolResult(pool_idx=0, side="low",
                       formed_at=df.index[0],
                       price_low=p.price_low, price_high=p.price_high,
                       score=p.score, outcome="untouched")

        sf = StateFeaturizer(df)
        intraday_snaps = generate_snapshots(
            df, [p], [r], sf,
            window_start=df.index[snapshot_idx],
            window_end=df.index[snapshot_idx],
            sample_every=1, max_horizon=60,
            intraday_session_only=True,
        )
        legacy_snaps = generate_snapshots(
            df, [p], [r], sf,
            window_start=df.index[snapshot_idx],
            window_end=df.index[snapshot_idx],
            sample_every=1, max_horizon=60,
            intraday_session_only=False,
        )
        self.assertEqual(len(intraday_snaps), 1)
        self.assertEqual(len(legacy_snaps), 1)

        # Same-session cap: 14:00 IST -> 15:15 IST = 75 IST minutes
        # = 15 5-min bars. So n_future <= 15.
        self.assertLessEqual(intraday_snaps[0].n_future_bars, 15)
        # Legacy mode crosses into Monday and uses full max_horizon=60.
        self.assertGreater(legacy_snaps[0].n_future_bars,
                           intraday_snaps[0].n_future_bars)
        # Every future bar that the intraday snapshot consumed must be on
        # the same IST date as the snapshot itself.
        anchor_date = ist_date(df.index[snapshot_idx])
        bars_used = intraday_snaps[0].n_future_bars
        for k in range(1, bars_used + 1):
            future_ts = df.index[snapshot_idx + k]
            self.assertEqual(ist_date(future_ts), anchor_date,
                f"future bar at +{k} crossed session boundary "
                f"(date {ist_date(future_ts)} vs anchor {anchor_date})")
            self.assertLessEqual(ist_minute_of_day(future_ts),
                                  EOD_SQUAREOFF_IST_MIN,
                f"future bar at +{k} past EOD square-off")

    def test_direction_label_reflects_only_same_session_excursion(self) -> None:
        # Same 3-session fixture; snapshot on session 2 at 13:30 IST.
        # Force session-2-afternoon DOWN and session-3 UP. Intraday
        # direction_label sees only session 2 afternoon (DOWN); legacy sees
        # session 3 (UP).
        df = _bars_three_sessions(n_per_session=75)
        snapshot_idx = None
        for i in range(80, len(df)):
            if ist_minute_of_day(df.index[i]) == 13 * 60 + 30:
                snapshot_idx = i
                break
        self.assertIsNotNone(snapshot_idx)
        df = df.copy()
        sess2_end = 75 + 75            # last bar idx of session 2 = 149 (0-indexed: 149)
        # Force session-2 afternoon down (snapshot+1 .. end-of-session-2).
        for i in range(snapshot_idx + 1, sess2_end):
            df.iloc[i, df.columns.get_loc("close")] -= 5.0
            df.iloc[i, df.columns.get_loc("high")] -= 5.0
            df.iloc[i, df.columns.get_loc("low")] -= 5.0
        # Force session-3 way up (bars 150 onward).
        for i in range(sess2_end, len(df)):
            df.iloc[i, df.columns.get_loc("close")] += 100.0
            df.iloc[i, df.columns.get_loc("high")] += 100.0
            df.iloc[i, df.columns.get_loc("low")] += 100.0

        p = _pool(price_low=50.0, price_high=51.0,
                  formed_at=df.index[0], available_at=df.index[0])
        from liqpool.tester import PoolResult
        r = PoolResult(pool_idx=0, side="low", formed_at=df.index[0],
                       price_low=p.price_low, price_high=p.price_high,
                       score=p.score, outcome="untouched")
        sf = StateFeaturizer(df)
        intraday_snap = generate_snapshots(
            df, [p], [r], sf,
            window_start=df.index[snapshot_idx],
            window_end=df.index[snapshot_idx],
            sample_every=1, max_horizon=60,
            intraday_session_only=True,
        )[0]
        legacy_snap = generate_snapshots(
            df, [p], [r], sf,
            window_start=df.index[snapshot_idx],
            window_end=df.index[snapshot_idx],
            sample_every=1, max_horizon=60,
            intraday_session_only=False,
        )[0]
        # Intraday direction reflects Friday afternoon -> down (0).
        # Legacy direction sees Monday up -> 1.
        intraday_dir = intraday_snap.direction_label(
            min(intraday_snap.n_future_bars, 12))
        legacy_dir = legacy_snap.direction_label(60)
        # We constructed a regime where these MUST disagree.
        self.assertEqual(intraday_dir, 0,
                          "intraday direction must see Friday afternoon down-move")
        self.assertEqual(legacy_dir, 1,
                          "legacy direction crosses to Monday and sees up-move")


if __name__ == "__main__":
    unittest.main()
