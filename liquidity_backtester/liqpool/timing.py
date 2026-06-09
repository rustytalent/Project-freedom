"""Track 3: direction and timing/proximity layer with multi-horizon support.

Two LightGBM models trained on time-series snapshots:

  DirectionModel:   P(up | state) over a single primary horizon (default 78 bars = 1 NSE session).
                    Target = (max_up_atr >= max_dn_atr) over horizon.

  ProximityModel:   P(pool touched within H bars | state, pool). Walkforward trains THREE of
                    these at H = 78 (1 day), 156 (2 days), 312 (4 days). Each next-pool entry
                    in the run output shows all three so the user can see "when".

A Snapshot stores enough information (cumulative future max-high / min-low arrays and per-pool
bars-to-touch) to derive labels for ANY horizon <= the max_horizon used at generation time.
That lets us train multiple horizon-specific models from one snapshot pass.

All snapshot generation is causal — features at T only use bars with timestamp <= T, and labels
are derived from bars in (T, T+H].
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

from .pools import Pool
from .tester import PoolResult
from .regime import compute_regime_series, nse_session, SESSION_LABELS
from .indicators import atr
from .distance_calibration import DistanceCalibrator
from .intraday import last_intraday_bar_idx


def distance_bucket(distance_atr: float) -> str:
    if distance_atr < 1.0:
        return "0-1 ATR"
    if distance_atr < 3.0:
        return "1-3 ATR"
    if distance_atr < 5.0:
        return "3-5 ATR"
    if distance_atr < 10.0:
        return "5-10 ATR"
    return "10+ ATR"


def _bucket_binary_metrics(y: np.ndarray, p: np.ndarray) -> Dict:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    pred = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    actual = np.asarray(y, dtype=int)
    auc = float(roc_auc_score(actual, pred)) if len(set(actual)) > 1 else None
    return {
        "n": int(len(actual)),
        "base_rate": float(actual.mean()) if len(actual) else 0.0,
        "auc": auc,
        "brier": float(brier_score_loss(actual, pred)) if len(actual) else 0.0,
        "logloss": float(log_loss(actual, pred)) if len(actual) else 0.0,
        "calibration_error": float(pred.mean() - actual.mean()) if len(actual) else 0.0,
    }


# ---------------------------------------------------------------------------
# State features
# ---------------------------------------------------------------------------

_STATE_NUMERIC = [
    "ret_1", "ret_6", "ret_24", "ret_78",
    "mom_12_atr", "zscore_close_50", "range_6_atr",
    "adx_14", "vol_ratio",
    "n_above", "n_below",
    "nearest_above_atr", "nearest_below_atr",
    "pull_above", "pull_below", "pull_ratio",
    "minutes_since_session_open",
    # MTF today-relative session context. These are computed from the 5-min
    # base bars by collapsing each IST trading day into running stats: they
    # tell the model "where are we within today's session?" without needing
    # a separate higher-TF data feed. Cheap and intraday-MIS-honest.
    "htf_today_range_atr",
    "htf_today_pos_in_range",
    "htf_today_open_to_now_atr",
    "htf_session_volume_ratio",
    "htf_overnight_gap_atr",
    # Anchored VWAP + FRVP: volume-weighted features that the pure
    # price-structure detector ignores. AVWAP tells the model "is price
    # above/below the volume-weighted fair value since {today,this-week}-open?
    # And is that fair value rising or falling?". FRVP (fixed-range volume
    # profile) tells the model "where did most of today's trading actually
    # happen?" — the Point of Control (POC), Value Area High/Low (VAH/VAL).
    # Both are causal: at bar j we only read bars i<=j of the same session
    # (AVWAP today / FRVP today) or of the same IST week (AVWAP week).
    "avwap_today_dist_atr",
    "avwap_today_slope_5_atr",
    "avwap_today_dev_sigmas",
    "avwap_week_dist_atr",
    "avwap_week_slope_5_atr",
    "poc_today_dist_atr",
    "vah_today_dist_atr",
    "val_today_dist_atr",
    "in_value_area_today",
    # NSE F&O expiry-cycle event context. Indian equity volatility is
    # heavily driven by the weekly (Thursday) and monthly (last-Thursday)
    # options-expiry rhythm. Even cash-equity intraday traders feel the
    # tape change on expiry days. These features are pure calendar math
    # off the bar's own IST date — no extra data feed.
    "days_to_monthly_expiry",
    "is_weekly_expiry_day",
    "is_morning_after_expiry",
    "is_monthly_expiry_week",
    # Path / context features. Same-instant snapshots (z-score, dist-to-pool)
    # cannot distinguish a bull-trap from a genuine breakdown — both have
    # identical state at the trap-fade moment but opposite futures. These
    # features encode the SHAPE of how price got here, not just the level:
    # opening-range character, trendiness-vs-choppiness, the gap-then-fade
    # pattern an institutional liquidity grab leaves on the tape, and the
    # current vol regime. All causal from the same 5-min bars.
    "first_15min_range_atr",
    "first_15min_direction_atr",
    "path_efficiency_30",
    "direction_changes_30",
    "is_gap_up_trap_fade",
    "is_gap_down_reversal",
    "vol_regime_zscore_20d",
]
_STATE_SESSION = [f"st_session_{s}" for s in SESSION_LABELS]
STATE_FEATURE_NAMES = _STATE_NUMERIC + _STATE_SESSION


class StateFeaturizer:
    """Pre-computes per-bar series; provides cheap snapshot-time feature lookup."""

    def __init__(self, df_base: pd.DataFrame):
        self.df = df_base
        self.atr_14 = atr(df_base, 14).bfill()
        self.atr_60 = atr(df_base, 60).bfill()
        self.regime = compute_regime_series(df_base)
        self.close = df_base["close"].values
        self.high = df_base["high"].values
        self.low = df_base["low"].values
        self.open_ = df_base["open"].values
        self.volume = (df_base["volume"].values
                       if "volume" in df_base.columns
                       else np.zeros(len(df_base), dtype=float))
        self.idx = df_base.index
        # Causal warmup: the first 9 bars are NaN under min_periods=10. A
        # naive .bfill() copies bar 10's mean+std backward into bars 0-9,
        # which is a small but real look-ahead. Instead, fill warmup with
        # the CAUSAL expanding mean/std (uses only bars i <= j), and only
        # fall back to NaN -> 0 when even the expanding window is empty.
        _close = df_base["close"]
        _r50_mean = _close.rolling(50, min_periods=10).mean()
        _r50_std = _close.rolling(50, min_periods=10).std()
        self.roll_mean_50 = _r50_mean.fillna(_close.expanding(min_periods=1).mean()).values
        self.roll_std_50 = _r50_std.fillna(_close.expanding(min_periods=1).std()).fillna(0.0).values
        self._precompute_session_relative_arrays()
        self._precompute_avwap_arrays()
        self._precompute_frvp_arrays()
        self._precompute_expiry_arrays()
        self._precompute_path_context_arrays()

    def _precompute_session_relative_arrays(self) -> None:
        """Build per-bar running stats over the bar's own IST trading day.

        Each IST date forms one session. For every bar j we record the running
        high/low/cumulative-volume from that day's first bar up to j, today's
        open price, and (for context features) the previous day's last close
        and a 20-day average session volume. All in O(n) time over the bars.
        """
        n = len(self.idx)
        if n == 0:
            self._today_open_idx = np.zeros(0, dtype=int)
            self._today_open_price = np.zeros(0)
            self._today_low_run = np.zeros(0)
            self._today_high_run = np.zeros(0)
            self._today_cum_vol = np.zeros(0)
            self._prev_day_last_close = np.full(0, np.nan)
            self._prev_20d_avg_vol = np.zeros(0)
            return

        ist_dt = pd.DatetimeIndex(self.idx) + pd.Timedelta(hours=5, minutes=30)
        ist_dates = ist_dt.normalize().values
        new_day = np.concatenate(([True], ist_dates[1:] != ist_dates[:-1]))

        today_open_idx = np.empty(n, dtype=int)
        today_open_price = np.empty(n, dtype=float)
        today_low_run = np.empty(n, dtype=float)
        today_high_run = np.empty(n, dtype=float)
        today_cum_vol = np.empty(n, dtype=float)

        cur_open_idx = 0
        cur_open_price = float(self.open_[0])
        cur_low = float("inf")
        cur_high = float("-inf")
        cur_vol = 0.0
        for j in range(n):
            if new_day[j]:
                cur_open_idx = j
                cur_open_price = float(self.open_[j])
                cur_low = float(self.low[j])
                cur_high = float(self.high[j])
                cur_vol = float(self.volume[j])
            else:
                cur_low = min(cur_low, float(self.low[j]))
                cur_high = max(cur_high, float(self.high[j]))
                cur_vol += float(self.volume[j])
            today_open_idx[j] = cur_open_idx
            today_open_price[j] = cur_open_price
            today_low_run[j] = cur_low
            today_high_run[j] = cur_high
            today_cum_vol[j] = cur_vol

        self._today_open_idx = today_open_idx
        self._today_open_price = today_open_price
        self._today_low_run = today_low_run
        self._today_high_run = today_high_run
        self._today_cum_vol = today_cum_vol

        # Per-day rollups: total volume and last close on each IST date.
        day_change_idx = np.where(new_day)[0]
        end_idx = np.concatenate((day_change_idx[1:] - 1, [n - 1]))
        day_total_vol = today_cum_vol[end_idx]
        day_last_close = self.close[end_idx]

        # day index for each bar.
        day_index = np.searchsorted(day_change_idx, np.arange(n), side="right") - 1

        # prev_day_last_close per bar: NaN on bars of the very first IST day.
        prev_close_per_day = np.concatenate(([np.nan], day_last_close[:-1]))
        self._prev_day_last_close = prev_close_per_day[day_index]

        # 20-day trailing average of session volume, excluding today.
        n_days = len(day_total_vol)
        cumsum = np.concatenate(([0.0], np.cumsum(day_total_vol)))
        prev_avg_per_day = np.empty(n_days, dtype=float)
        for k in range(n_days):
            lo = max(0, k - 20)
            if k == lo:
                prev_avg_per_day[k] = 0.0
            else:
                prev_avg_per_day[k] = (cumsum[k] - cumsum[lo]) / float(k - lo)
        self._prev_20d_avg_vol = prev_avg_per_day[day_index]

    def _precompute_avwap_arrays(self) -> None:
        """Anchored VWAP from today's open and from this IST week's open.

        Per-bar value = cum(typical_price * volume) / cum(volume), where the
        running sums reset at each anchor. Standard-deviation band tracks
        the population variance of typical_price about the running AVWAP,
        which the predict-time code uses to convert price-vs-AVWAP into a
        sigma deviation feature. All O(n) over the bars.
        """
        n = len(self.idx)
        if n == 0:
            self._avwap_today = np.zeros(0)
            self._avwap_week = np.zeros(0)
            self._avwap_today_sigma = np.zeros(0)
            return

        ist_dt = pd.DatetimeIndex(self.idx) + pd.Timedelta(hours=5, minutes=30)
        ist_dates = ist_dt.normalize().values
        new_day = np.concatenate(([True], ist_dates[1:] != ist_dates[:-1]))
        # Week anchor: weekday-of-IST-date drops (Mon=0 after Fri=4) OR a
        # gap > 2 calendar days (covers Fri->Mon and longer holidays).
        ist_weekday = ist_dt.weekday.values
        ist_day_diff = np.diff(ist_dates).astype("timedelta64[D]").astype(int)
        new_week = np.concatenate((
            [True],
            (ist_weekday[1:] < ist_weekday[:-1]) | (ist_day_diff > 2),
        ))

        # Typical price = (high + low + close) / 3 — standard VWAP input.
        tp = (self.high + self.low + self.close) / 3.0
        vol = self.volume.astype(float)

        avwap_today = np.empty(n, dtype=float)
        avwap_week = np.empty(n, dtype=float)
        avwap_today_sigma = np.empty(n, dtype=float)
        # Running accumulators reset at each anchor.
        d_cum_pv = 0.0
        d_cum_v = 0.0
        d_cum_pv2 = 0.0    # second moment of price for sigma band
        w_cum_pv = 0.0
        w_cum_v = 0.0
        for j in range(n):
            if new_day[j]:
                d_cum_pv = 0.0
                d_cum_v = 0.0
                d_cum_pv2 = 0.0
            if new_week[j]:
                w_cum_pv = 0.0
                w_cum_v = 0.0
            v_j = vol[j]
            tp_j = tp[j]
            d_cum_pv += tp_j * v_j
            d_cum_v += v_j
            d_cum_pv2 += (tp_j ** 2) * v_j
            w_cum_pv += tp_j * v_j
            w_cum_v += v_j
            # Guard against zero-volume bars at the start of a session.
            d_v_eff = max(d_cum_v, 1e-9)
            w_v_eff = max(w_cum_v, 1e-9)
            avwap_today[j] = d_cum_pv / d_v_eff
            avwap_week[j] = w_cum_pv / w_v_eff
            mean_t = avwap_today[j]
            var_t = max(0.0, (d_cum_pv2 / d_v_eff) - mean_t * mean_t)
            avwap_today_sigma[j] = float(np.sqrt(var_t))

        self._avwap_today = avwap_today
        self._avwap_week = avwap_week
        self._avwap_today_sigma = avwap_today_sigma

    def _precompute_frvp_arrays(self) -> None:
        """Fixed-range volume profile per IST session: POC, VAH, VAL per bar.

        For each session we bin the typical-price stream by a fixed width
        (anchored at the session-open ATR(60), so bins are stable across the
        day even as range expands). Per bar j we report:

          * POC[j] — price (bin midpoint) with the highest accumulated volume
                     from session-open to j.
          * VAH[j], VAL[j] — high and low edges of the price range whose
                     accumulated volume is the smallest cluster summing to
                     >= 70% of session-so-far total volume, ranked from the
                     POC outward. The "developing 70% value area" standard.

        All causal (only reads bars <= j of the same session). O(n * bins);
        bins ~= 30, so trivial.
        """
        n = len(self.idx)
        if n == 0:
            self._poc_today = np.zeros(0)
            self._vah_today = np.zeros(0)
            self._val_today = np.zeros(0)
            return

        ist_dt = pd.DatetimeIndex(self.idx) + pd.Timedelta(hours=5, minutes=30)
        ist_dates = ist_dt.normalize().values
        new_day = np.concatenate(([True], ist_dates[1:] != ist_dates[:-1]))

        tp = (self.high + self.low + self.close) / 3.0
        vol = self.volume.astype(float)
        atr60_vals = self.atr_60.values

        poc_arr = np.empty(n, dtype=float)
        vah_arr = np.empty(n, dtype=float)
        val_arr = np.empty(n, dtype=float)

        # Per-session state.
        bin_volume: Dict[int, float] = {}
        origin = 0.0
        bin_width = 1.0
        total_vol = 0.0

        for j in range(n):
            if new_day[j]:
                bin_volume = {}
                origin = float(self.open_[j])
                bin_width = max(0.10 * float(atr60_vals[j]), 1e-6)
                total_vol = 0.0
            bin_idx = int(np.floor((tp[j] - origin) / bin_width))
            bin_volume[bin_idx] = bin_volume.get(bin_idx, 0.0) + vol[j]
            total_vol += vol[j]
            # POC: bin with max volume.
            if total_vol <= 1e-9:
                poc_arr[j] = tp[j]
                vah_arr[j] = tp[j]
                val_arr[j] = tp[j]
                continue
            poc_bin = max(bin_volume.items(), key=lambda kv: kv[1])[0]
            poc_price = origin + (poc_bin + 0.5) * bin_width
            poc_arr[j] = poc_price
            # VAH/VAL: expand outward from the POC through VISITED bins
            # (not just adjacent indices — a random-walk session leaves
            # gaps that should be skipped over, not treated as terminators).
            visited = sorted(bin_volume.keys())
            poc_pos = visited.index(poc_bin)
            upper_pos = poc_pos
            lower_pos = poc_pos
            target = 0.70 * total_vol
            cum = bin_volume[poc_bin]
            while cum < target:
                has_up = upper_pos + 1 < len(visited)
                has_dn = lower_pos > 0
                if not has_up and not has_dn:
                    break
                if not has_dn:
                    upper_pos += 1
                    cum += bin_volume[visited[upper_pos]]
                elif not has_up:
                    lower_pos -= 1
                    cum += bin_volume[visited[lower_pos]]
                else:
                    next_up_vol = bin_volume[visited[upper_pos + 1]]
                    next_dn_vol = bin_volume[visited[lower_pos - 1]]
                    if next_up_vol >= next_dn_vol:
                        upper_pos += 1
                        cum += next_up_vol
                    else:
                        lower_pos -= 1
                        cum += next_dn_vol
            high_bin = visited[upper_pos]
            low_bin = visited[lower_pos]
            vah_arr[j] = origin + (high_bin + 1) * bin_width
            val_arr[j] = origin + low_bin * bin_width

        self._poc_today = poc_arr
        self._vah_today = vah_arr
        self._val_today = val_arr

    def _precompute_expiry_arrays(self) -> None:
        """NSE F&O expiry-cycle event features per bar.

        Convention used here (true for NSE as of 2024-2026):
          * Weekly expiry for Nifty / Bank Nifty options falls on Thursday.
          * Monthly expiry (last Thursday of the month) applies to single-
            stock futures + options and to index futures.
          * Caveat: when Thursday is an NSE holiday, expiry shifts to the
            preceding Wednesday. This featurizer does NOT carry an NSE
            holiday calendar, so on those <5% of months the
            ``days_to_monthly_expiry`` value will be off by one. Documented
            limitation; sufficient signal granularity for the GBM.
        """
        n = len(self.idx)
        if n == 0:
            self._days_to_monthly_expiry = np.zeros(0, dtype=float)
            self._is_weekly_expiry_day = np.zeros(0, dtype=float)
            self._is_morning_after_expiry = np.zeros(0, dtype=float)
            self._is_monthly_expiry_week = np.zeros(0, dtype=float)
            return

        ist_dt = pd.DatetimeIndex(self.idx) + pd.Timedelta(hours=5, minutes=30)
        ist_dates = ist_dt.normalize()
        # Thursday == 3 in pandas / Python (Monday=0). ``weekday`` on a
        # DatetimeIndex returns a numpy array directly in modern pandas.
        is_thursday = np.asarray(ist_dt.weekday == 3, dtype=float)

        # Last Thursday of each bar's IST month.
        last_thu_of_month = np.empty(n, dtype="datetime64[ns]")
        for k in range(n):
            d = ist_dates[k]
            # First day of the NEXT month, minus one day = last day of d's month.
            if d.month == 12:
                next_first = pd.Timestamp(year=d.year + 1, month=1, day=1)
            else:
                next_first = pd.Timestamp(year=d.year, month=d.month + 1, day=1)
            last_day = next_first - pd.Timedelta(days=1)
            # Step back from last_day to the most recent Thursday.
            shift = (last_day.weekday() - 3) % 7
            last_thu_of_month[k] = (last_day - pd.Timedelta(days=int(shift))).to_datetime64()

        days_to_monthly_expiry = (
            (last_thu_of_month - ist_dates.values).astype("timedelta64[D]")
            .astype(float)
        )
        # When the bar's date is AFTER its month's last Thursday (the
        # leftover few days), the expiry has already happened: report
        # 0 instead of a negative number so the GBM sees a clean monotone.
        days_to_monthly_expiry = np.maximum(days_to_monthly_expiry, 0.0)

        # "Morning after expiry" = the bar's IST date is the FIRST trading
        # day after a Thursday. Detect via: previous bar's IST date != this
        # bar's, AND the previous IST date was a Thursday. Edge case: very
        # first bar of the dataset can't have a "previous" — set 0.
        # "Morning after expiry" applies to EVERY bar of any IST trading day
        # whose immediately-previous trading day (in the data) was a Thursday,
        # not just the first bar of that day. Build per-unique-IST-date
        # lookup, then map back per-bar.
        ist_date_vals = ist_dates.values
        new_day_mask = np.concatenate(([True], ist_date_vals[1:] != ist_date_vals[:-1]))
        unique_idx_starts = np.where(new_day_mask)[0]
        unique_dates = ist_date_vals[unique_idx_starts]
        unique_weekdays = pd.DatetimeIndex(unique_dates).weekday
        prev_day_was_thu_per_unique = np.concatenate((
            [False],
            np.asarray(unique_weekdays[:-1] == 3, dtype=bool),
        ))
        # day_index: which unique-date index does each bar belong to?
        day_index = np.searchsorted(unique_idx_starts, np.arange(n),
                                     side="right") - 1
        prev_was_thu = prev_day_was_thu_per_unique[day_index].astype(float)

        # "Monthly expiry week" — within 7 calendar days of the last Thursday.
        is_monthly_expiry_week = (days_to_monthly_expiry <= 7).astype(float)

        self._days_to_monthly_expiry = days_to_monthly_expiry
        self._is_weekly_expiry_day = is_thursday
        self._is_morning_after_expiry = prev_was_thu
        self._is_monthly_expiry_week = is_monthly_expiry_week

    def _precompute_path_context_arrays(self) -> None:
        """Path / context features encoding the SHAPE of recent bars.

        Per-bar arrays of length n:
          * _first_15min_range_atr[j] — range of bars [open_idx, open_idx+2]
                                        of j's session, in ATR(14) units.
                                        0.0 before bar 3 of the session.
          * _first_15min_direction_atr[j] — close[open_idx+2] - open[open_idx],
                                        in ATR(14) units. Sign-bearing.
          * _path_efficiency_30[j] — |close[j] - close[j-29]| /
                                        sum_{k=j-28..j} |close[k] - close[k-1]|.
                                        Range 0..1; 1.0 = perfectly linear,
                                        0.0 = pure noise.
          * _direction_changes_30[j] — count of sign flips in
                                        diff(close[j-29..j]) over 30 bars.
          * _is_gap_up_trap_fade[j] — 1.0 when overnight gap > 0.5 ATR AND
                                        first-15-min direction < -0.3 ATR.
                                        The classic institutional bull-trap.
          * _is_gap_down_reversal[j] — 1.0 when overnight gap < -0.5 ATR AND
                                        first-15-min direction > +0.3 ATR.
                                        The classic morning panic + reversal.
          * _vol_regime_zscore_20d[j] — z-score of session-end ATR(14)
                                        against the trailing 20 IST trading
                                        days. Captures vol expansion /
                                        contraction regime.
        """
        n = len(self.idx)
        if n == 0:
            self._first_15min_range_atr = np.zeros(0, dtype=float)
            self._first_15min_direction_atr = np.zeros(0, dtype=float)
            self._path_efficiency_30 = np.zeros(0, dtype=float)
            self._direction_changes_30 = np.zeros(0, dtype=float)
            self._is_gap_up_trap_fade = np.zeros(0, dtype=float)
            self._is_gap_down_reversal = np.zeros(0, dtype=float)
            self._vol_regime_zscore_20d = np.zeros(0, dtype=float)
            return

        atr14 = self.atr_14.values
        c = self.close
        # 1) First-15-min features — derived from session-open bar offsets.
        first_15min_range_atr = np.zeros(n, dtype=float)
        first_15min_dir_atr = np.zeros(n, dtype=float)
        open_idx_arr = self._today_open_idx
        for j in range(n):
            oi = int(open_idx_arr[j])
            bars_in = j - oi
            if bars_in < 2:
                continue                              # not yet 3 bars in
            third_bar = oi + 2
            r = float(self.high[oi:third_bar + 1].max()
                      - self.low[oi:third_bar + 1].min())
            d = float(self.close[third_bar] - self.open_[oi])
            denom = max(float(atr14[third_bar]), 1e-9)
            first_15min_range_atr[j] = r / denom
            first_15min_dir_atr[j] = d / denom
        self._first_15min_range_atr = first_15min_range_atr
        self._first_15min_direction_atr = first_15min_dir_atr

        # 2) Path efficiency over 30 bars (Kaufman efficiency ratio).
        path_eff = np.zeros(n, dtype=float)
        dir_changes = np.zeros(n, dtype=float)
        for j in range(n):
            lo = max(0, j - 29)
            if j - lo < 5:                            # need at least 5 moves
                continue
            window = c[lo:j + 1]
            net = abs(window[-1] - window[0])
            moves = np.abs(np.diff(window))
            total = float(moves.sum())
            if total > 1e-9:
                path_eff[j] = float(net / total)
            else:
                path_eff[j] = 0.0
            # Direction changes: count sign flips in diff(window).
            diffs = np.diff(window)
            signs = np.sign(diffs)
            # Only count flips between non-zero signs.
            nz = signs[signs != 0]
            if len(nz) >= 2:
                dir_changes[j] = float((np.diff(nz) != 0).sum())
        self._path_efficiency_30 = path_eff
        self._direction_changes_30 = dir_changes

        # 3) Trap-fade / reversal flags, using overnight_gap and first_15min_dir.
        gap_atr = np.where(
            np.isnan(self._prev_day_last_close), 0.0,
            (self._today_open_price - np.where(
                np.isnan(self._prev_day_last_close), self._today_open_price,
                self._prev_day_last_close)) / np.maximum(atr14, 1e-9),
        )
        trap_up = ((gap_atr > 0.5) & (first_15min_dir_atr < -0.3)).astype(float)
        trap_dn = ((gap_atr < -0.5) & (first_15min_dir_atr > 0.3)).astype(float)
        self._is_gap_up_trap_fade = trap_up
        self._is_gap_down_reversal = trap_dn

        # 4) Vol regime z-score: per-IST-day session-end ATR, z-scored vs
        #    the trailing 20 IST trading days. Causal: today's day's z-score
        #    uses prior 20 days' stats, NOT including today.
        ist_dt = pd.DatetimeIndex(self.idx) + pd.Timedelta(hours=5, minutes=30)
        ist_dates = ist_dt.normalize().values
        new_day_mask = np.concatenate(([True], ist_dates[1:] != ist_dates[:-1]))
        day_change_idx = np.where(new_day_mask)[0]
        end_idx = np.concatenate((day_change_idx[1:] - 1, [n - 1]))
        day_end_atr = atr14[end_idx]
        n_days = len(day_end_atr)
        per_day_z = np.zeros(n_days, dtype=float)
        for k in range(n_days):
            lo = max(0, k - 20)
            if k - lo < 5:                            # need 5+ days history
                per_day_z[k] = 0.0
                continue
            window = day_end_atr[lo:k]                # exclude today
            mu = float(np.mean(window))
            sd = float(np.std(window, ddof=1)) if len(window) > 1 else 0.0
            if sd > 1e-9:
                per_day_z[k] = float((day_end_atr[k] - mu) / sd)
            else:
                per_day_z[k] = 0.0
        day_index = np.searchsorted(day_change_idx, np.arange(n),
                                     side="right") - 1
        self._vol_regime_zscore_20d = per_day_z[day_index]

    def features_at(self, j: int, active_pools: List[Pool]) -> Dict[str, float]:
        n = len(self.idx)
        if j < 1 or j >= n:
            return {k: 0.0 for k in STATE_FEATURE_NAMES}

        c = self.close
        ts = self.idx[j]
        a = max(float(self.atr_14.iloc[j]), 1e-9)

        def log_ret(k):
            i0 = max(0, j - k)
            if c[i0] <= 0:
                return 0.0
            return float(np.log(c[j] / c[i0]))

        # Momentum over last 12 bars: sum of bar-over-bar moves.
        # np.diff handles boundary cases cleanly (n-1 moves over n bars). The
        # old form `(c[i0:j+1] - c[max(0, i0-1):j])` had a shape mismatch when
        # i0 = 0 (early bars), because both slices ended up length j+1 and
        # length j respectively. np.diff sidesteps that entirely.
        slice_start = max(0, j - 11)
        moves = np.diff(c[slice_start:j + 1])
        mom_12 = float(np.sum(moves) / a) if len(moves) else 0.0

        mean50 = self.roll_mean_50[j]
        std50 = self.roll_std_50[j]
        z = float((c[j] - mean50) / std50) if std50 > 0 else 0.0

        i0 = max(0, j - 5)
        rng6 = float((self.high[i0:j + 1].max() - self.low[i0:j + 1].min()) / a)

        adx_14 = float(self.regime["adx_14"].iloc[j])
        vol_r = float(self.regime["vol_ratio"].iloc[j])
        session = nse_session(ts)
        ist = ts + pd.Timedelta(hours=5, minutes=30)
        ist_min = ist.hour * 60 + ist.minute
        m_since_open = max(0, ist_min - (9 * 60 + 15))

        close_T = c[j]
        n_above = n_below = 0
        nearest_above_d = float("inf")
        nearest_below_d = float("inf")
        pull_above = pull_below = 0.0
        for p in active_pools:
            if p.price_low > close_T:
                d = (p.mid - close_T) / a
                n_above += 1
                nearest_above_d = min(nearest_above_d, d)
                pull_above += 1.0 / max(d, 1.0)
            elif p.price_high < close_T:
                d = (close_T - p.mid) / a
                n_below += 1
                nearest_below_d = min(nearest_below_d, d)
                pull_below += 1.0 / max(d, 1.0)
        nearest_above_d = nearest_above_d if nearest_above_d != float("inf") else 100.0
        nearest_below_d = nearest_below_d if nearest_below_d != float("inf") else 100.0
        pull_ratio = (pull_above - pull_below) / max(pull_above + pull_below, 1e-9)

        today_open_price = float(self._today_open_price[j])
        today_low = float(self._today_low_run[j])
        today_high = float(self._today_high_run[j])
        today_range = today_high - today_low
        htf_today_range_atr = today_range / a
        # Mid-range when the day hasn't traded a non-zero range yet (first bar
        # of day, or pathological flat run). 0.5 is the neutral "no signal" code.
        if today_range > 1e-9:
            htf_today_pos_in_range = (c[j] - today_low) / today_range
        else:
            htf_today_pos_in_range = 0.5
        htf_today_open_to_now_atr = (c[j] - today_open_price) / a

        prev_avg_vol = float(self._prev_20d_avg_vol[j])
        cum_v = float(self._today_cum_vol[j])
        if prev_avg_vol > 1e-9:
            # Ratio of cumulative session volume so far to the trailing 20-day
            # average TOTAL session volume. The model already has
            # ``minutes_since_session_open`` so it can learn the joint
            # "ratio=X at minute Y means heavy/light tape today".
            htf_session_volume_ratio = cum_v / prev_avg_vol
        else:
            htf_session_volume_ratio = 1.0

        prev_close = self._prev_day_last_close[j]
        if not np.isnan(prev_close):
            htf_overnight_gap_atr = (today_open_price - float(prev_close)) / a
        else:
            htf_overnight_gap_atr = 0.0

        # AVWAP-relative features. Slopes use lag=5 so they capture a
        # ~25-min window on 5-min bars; before bar 5 of a session we just
        # report zero (no slope to read).
        avwap_today = float(self._avwap_today[j])
        avwap_week = float(self._avwap_week[j])
        avwap_today_dist_atr = (c[j] - avwap_today) / a
        avwap_week_dist_atr = (c[j] - avwap_week) / a
        if j >= 5:
            avwap_today_slope_5_atr = (avwap_today
                                       - float(self._avwap_today[j - 5])) / a
            avwap_week_slope_5_atr = (avwap_week
                                      - float(self._avwap_week[j - 5])) / a
        else:
            avwap_today_slope_5_atr = 0.0
            avwap_week_slope_5_atr = 0.0
        sigma_t = float(self._avwap_today_sigma[j])
        if sigma_t > 1e-9:
            avwap_today_dev_sigmas = (c[j] - avwap_today) / sigma_t
        else:
            avwap_today_dev_sigmas = 0.0

        # FRVP-relative features. The "in_value_area" flag is a soft 0/1
        # so the GBM can split on it. dist_to_poc preserves sign so the
        # model can distinguish above-POC (overbought relative to today's
        # volume centre) from below-POC.
        poc_t = float(self._poc_today[j])
        vah_t = float(self._vah_today[j])
        val_t = float(self._val_today[j])
        poc_today_dist_atr = (c[j] - poc_t) / a
        vah_today_dist_atr = (c[j] - vah_t) / a
        val_today_dist_atr = (c[j] - val_t) / a
        in_value_area_today = 1.0 if (val_t <= c[j] <= vah_t) else 0.0

        feats: Dict[str, float] = {
            "ret_1": log_ret(1),
            "ret_6": log_ret(6),
            "ret_24": log_ret(24),
            "ret_78": log_ret(78),
            "mom_12_atr": mom_12,
            "zscore_close_50": z,
            "range_6_atr": rng6,
            "adx_14": adx_14,
            "vol_ratio": vol_r,
            "n_above": float(n_above),
            "n_below": float(n_below),
            "nearest_above_atr": float(nearest_above_d),
            "nearest_below_atr": float(nearest_below_d),
            "pull_above": float(pull_above),
            "pull_below": float(pull_below),
            "pull_ratio": float(pull_ratio),
            "minutes_since_session_open": float(m_since_open),
            "htf_today_range_atr": float(htf_today_range_atr),
            "htf_today_pos_in_range": float(htf_today_pos_in_range),
            "htf_today_open_to_now_atr": float(htf_today_open_to_now_atr),
            "htf_session_volume_ratio": float(htf_session_volume_ratio),
            "htf_overnight_gap_atr": float(htf_overnight_gap_atr),
            "avwap_today_dist_atr": float(avwap_today_dist_atr),
            "avwap_today_slope_5_atr": float(avwap_today_slope_5_atr),
            "avwap_today_dev_sigmas": float(avwap_today_dev_sigmas),
            "avwap_week_dist_atr": float(avwap_week_dist_atr),
            "avwap_week_slope_5_atr": float(avwap_week_slope_5_atr),
            "poc_today_dist_atr": float(poc_today_dist_atr),
            "vah_today_dist_atr": float(vah_today_dist_atr),
            "val_today_dist_atr": float(val_today_dist_atr),
            "in_value_area_today": float(in_value_area_today),
            "days_to_monthly_expiry": float(self._days_to_monthly_expiry[j]),
            "is_weekly_expiry_day": float(self._is_weekly_expiry_day[j]),
            "is_morning_after_expiry": float(self._is_morning_after_expiry[j]),
            "is_monthly_expiry_week": float(self._is_monthly_expiry_week[j]),
            "first_15min_range_atr": float(self._first_15min_range_atr[j]),
            "first_15min_direction_atr": float(self._first_15min_direction_atr[j]),
            "path_efficiency_30": float(self._path_efficiency_30[j]),
            "direction_changes_30": float(self._direction_changes_30[j]),
            "is_gap_up_trap_fade": float(self._is_gap_up_trap_fade[j]),
            "is_gap_down_reversal": float(self._is_gap_down_reversal[j]),
            "vol_regime_zscore_20d": float(self._vol_regime_zscore_20d[j]),
        }
        for s in SESSION_LABELS:
            feats[f"st_session_{s}"] = 1.0 if session == s else 0.0
        return feats


# ---------------------------------------------------------------------------
# Multi-horizon Snapshot
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    bar_idx: int
    ts: pd.Timestamp
    close: float
    atr_val: float
    state: Dict[str, float]
    # Cumulative running max-high / min-low over future bars (T+1 .. T+n_future_bars).
    # future_max_high[i-1] = max(high[j+1..j+i]) — i.e. max-high seen so far i bars into the future.
    future_max_high: List[float] = field(default_factory=list)
    future_min_low: List[float] = field(default_factory=list)
    # Per-pool: (pool_idx, bars_to_touch_or_None, distance_atr_at_snapshot, "above"/"below")
    pool_touches: List[Tuple[int, Optional[int], float, str]] = field(default_factory=list)
    # How many future bars we actually observed (could be < max_horizon near end of data).
    n_future_bars: int = 0

    def direction_label(self, horizon: int) -> Optional[int]:
        """1 if up-excursion exceeds down-excursion over `horizon` future bars, 0 if down wins.
        None if not enough data OR the move is an exact tie (e.g. a flat window) — labelling a
        tie as 'up' (the old behaviour) injected a systematic upward bias from flat snapshots."""
        h = min(horizon, self.n_future_bars)
        if h < 5:
            return None
        max_h = self.future_max_high[h - 1]
        min_l = self.future_min_low[h - 1]
        max_up = max_h - self.close
        max_dn = self.close - min_l
        if max_up == max_dn:
            return None
        return 1 if max_up > max_dn else 0

    def max_up_atr(self, horizon: int) -> float:
        h = min(horizon, self.n_future_bars)
        if h == 0:
            return 0.0
        return (self.future_max_high[h - 1] - self.close) / max(self.atr_val, 1e-9)

    def max_dn_atr(self, horizon: int) -> float:
        h = min(horizon, self.n_future_bars)
        if h == 0:
            return 0.0
        return (self.close - self.future_min_low[h - 1]) / max(self.atr_val, 1e-9)

    def pool_touch_labels(self, horizon: int) -> List[Tuple[int, int, float, str]]:
        """Returns (pool_idx, touched_within_horizon, distance_atr, side) for each pool that was
        active at the snapshot. Caller is responsible for filtering by self.n_future_bars >=
        horizon if it needs a definitive 0-label."""
        return [(pi,
                 1 if (bt is not None and bt <= horizon) else 0,
                 d, s)
                for (pi, bt, d, s) in self.pool_touches]


def generate_snapshots(df_base: pd.DataFrame, pools: List[Pool], results: List[PoolResult],
                       featurizer: StateFeaturizer,
                       window_start: pd.Timestamp, window_end: pd.Timestamp,
                       sample_every: int, max_horizon: int,
                       clip_future_to_window: bool = False,
                       intraday_session_only: bool = True) -> List[Snapshot]:
    """Build snapshots at every `sample_every` bar in [window_start, window_end].

    `window_end` bounds WHERE we sample snapshots (the last decision bar). By default a
    snapshot's future trajectory is allowed to extend past `window_end`, up to `max_horizon`
    bars or the data end. This is the correct behaviour for OOS *evaluation* (the future
    outcome is genuinely observable after the decision) and for single-bar causality probes.

    Pass `clip_future_to_window=True` when generating TRAINING snapshots in a walk-forward:
    it caps each snapshot's future trajectory at `window_end` so a train-window snapshot near
    the boundary cannot derive its label from bars that fall inside the OOS test window
    (which would leak test-period price action into training and inflate OOS metrics). Models
    filter `n_future_bars >= horizon` at fit time, so boundary snapshots are purged rather
    than trained on truncated labels.
    """
    idx = df_base.index
    n = len(idx)
    h_arr = df_base["high"].values
    l_arr = df_base["low"].values
    c_arr = df_base["close"].values

    j_start = int(np.searchsorted(idx.values, np.datetime64(window_start), side="left"))
    we_pos = int(np.searchsorted(idx.values, np.datetime64(window_end), side="right")) - 1
    j_end = min(we_pos, n - 5)

    snaps: List[Snapshot] = []
    for j in range(max(j_start, 80), j_end + 1, sample_every):
        T = idx[j]
        close_T = c_arr[j]
        a_T = max(float(featurizer.atr_14.iloc[j]), 1e-9)

        n_future = min(max_horizon, n - 1 - j)
        if clip_future_to_window:
            n_future = min(n_future, we_pos - j)
        if intraday_session_only:
            # Same-session cap: the future window cannot extend past 15:15 IST
            # of the snapshot's own trading day. Without this, direction_label
            # and pool_touch_labels read across overnight gaps, training the
            # models on multi-day moves we cannot actually capture under MIS.
            session_end_idx = last_intraday_bar_idx(idx, j, j + n_future)
            n_future = min(n_future, max(0, session_end_idx - j))
        if n_future < 5:
            continue

        # Cumulative running max-high / min-low for the future window
        f_max_high: List[float] = []
        f_min_low: List[float] = []
        cur_max = h_arr[j + 1]
        cur_min = l_arr[j + 1]
        f_max_high.append(cur_max)
        f_min_low.append(cur_min)
        for i in range(2, n_future + 1):
            cur_max = max(cur_max, float(h_arr[j + i]))
            cur_min = min(cur_min, float(l_arr[j + i]))
            f_max_high.append(cur_max)
            f_min_low.append(cur_min)

        # Active pools at T (available_at <= T, not yet touched/broken)
        active_with_idx: List[Tuple[int, Pool]] = []
        for pi, p in enumerate(pools):
            if p.available_at > T:
                continue
            r = results[pi]
            if r.touched_at is not None and r.touched_at <= T:
                continue
            if r.broken_at is not None and r.broken_at <= T:
                continue
            active_with_idx.append((pi, p))
        active_pools = [p for _, p in active_with_idx]
        state = featurizer.features_at(j, active_pools)

        # Per-pool bars-to-touch (within max_horizon)
        pool_touches: List[Tuple[int, Optional[int], float, str]] = []
        for pi, p in active_with_idx:
            ta = results[pi].touched_at
            bars_to_touch: Optional[int] = None
            if ta is not None and ta > T:
                ta_idx = int(idx.searchsorted(ta, side="left"))
                if j < ta_idx <= j + n_future:
                    bars_to_touch = ta_idx - j
            if p.price_low > close_T:
                dist = (p.mid - close_T) / a_T
                side = "above"
            elif p.price_high < close_T:
                dist = (close_T - p.mid) / a_T
                side = "below"
            else:
                continue   # already inside the zone
            pool_touches.append((pi, bars_to_touch, float(dist), side))

        snaps.append(Snapshot(
            bar_idx=j, ts=T, close=float(close_T), atr_val=a_T,
            state=state,
            future_max_high=f_max_high, future_min_low=f_min_low,
            pool_touches=pool_touches, n_future_bars=int(n_future),
        ))
    return snaps


# ---------------------------------------------------------------------------
# Direction model — single horizon
# ---------------------------------------------------------------------------

@dataclass
class DirectionModel:
    horizon: int = 78                                # 1 NSE trading session
    feature_names: List[str] = field(default_factory=list)
    train_n: int = 0
    val_n: int = 0
    val_brier: float = 0.0
    val_logloss: float = 0.0
    val_auc: float = 0.0
    base_rate: float = 0.0

    def fit(self, snapshots: List[Snapshot], val_frac: float = 0.25, seed: int = 19
            ) -> "DirectionModel":
        import lightgbm as lgb
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

        usable = [(s.state, s.direction_label(self.horizon)) for s in snapshots
                  if s.n_future_bars >= self.horizon and s.direction_label(self.horizon) is not None]
        if len(usable) < 30:
            raise ValueError(f"DirectionModel needs >= 30 valid snapshots at horizon "
                              f"{self.horizon}, got {len(usable)}")
        X_df = pd.DataFrame([u[0] for u in usable], columns=STATE_FEATURE_NAMES).fillna(0.0)
        y = np.array([u[1] for u in usable], dtype=int)
        self.feature_names = list(X_df.columns)
        self.base_rate = float(y.mean())

        rng = np.random.default_rng(seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        rng.shuffle(pos); rng.shuffle(neg)
        n_vp = max(1, int(round(len(pos) * val_frac)))
        n_vn = max(1, int(round(len(neg) * val_frac)))
        val_idx = np.concatenate([pos[:n_vp], neg[:n_vn]])
        tr_idx = np.concatenate([pos[n_vp:], neg[n_vn:]])
        rng.shuffle(val_idx); rng.shuffle(tr_idx)

        X_tr, y_tr = X_df.iloc[tr_idx].values, y[tr_idx]
        X_v, y_v = X_df.iloc[val_idx].values, y[val_idx]

        params = dict(objective="binary", metric="binary_logloss",
                       learning_rate=0.05, num_leaves=15, max_depth=4,
                       min_data_in_leaf=8, feature_fraction=0.85,
                       bagging_fraction=0.85, bagging_freq=5,
                       lambda_l2=2.0, verbose=-1, seed=seed)
        dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=self.feature_names)
        dval = lgb.Dataset(X_v, label=y_v, reference=dtr, feature_name=self.feature_names)
        self._gbm = lgb.train(params, dtr, num_boost_round=300, valid_sets=[dval],
                               valid_names=["val"],
                               callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                                          lgb.log_evaluation(0)])
        val_raw = self._gbm.predict(X_v, num_iteration=self._gbm.best_iteration)
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(val_raw, y_v)
        val_cal = self._iso.transform(val_raw)
        self.train_n = int(len(tr_idx)); self.val_n = int(len(val_idx))
        self.val_brier = float(brier_score_loss(y_v, val_cal))
        self.val_logloss = float(log_loss(y_v, np.clip(val_cal, 1e-6, 1 - 1e-6)))
        self.val_auc = float(roc_auc_score(y_v, val_cal)) if len(set(y_v)) > 1 else 0.0
        return self

    def fit_frame(self, frame: pd.DataFrame, val_frac: float = 0.25,
                  seed: int = 19) -> "DirectionModel":
        """Fit from persisted direction feature rows instead of in-memory Snapshot objects."""
        if frame is None or frame.empty:
            raise ValueError("DirectionModel needs non-empty direction feature frame")
        if "direction_label" not in frame.columns:
            raise ValueError("direction feature frame missing direction_label")
        X_df = frame.reindex(columns=STATE_FEATURE_NAMES).fillna(0.0)
        y = frame["direction_label"].astype(int).to_numpy()
        if len(y) < 30:
            raise ValueError(f"DirectionModel needs >= 30 valid rows, got {len(y)}")
        if len(np.unique(y)) < 2:
            raise ValueError("DirectionModel needs both classes")
        self.feature_names = list(X_df.columns)
        self.base_rate = float(y.mean())

        import lightgbm as lgb
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

        rng = np.random.default_rng(seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        rng.shuffle(pos); rng.shuffle(neg)
        n_vp = max(1, int(round(len(pos) * val_frac)))
        n_vn = max(1, int(round(len(neg) * val_frac)))
        val_idx = np.concatenate([pos[:n_vp], neg[:n_vn]])
        tr_idx = np.concatenate([pos[n_vp:], neg[n_vn:]])
        rng.shuffle(val_idx); rng.shuffle(tr_idx)

        X_tr, y_tr = X_df.iloc[tr_idx].values, y[tr_idx]
        X_v, y_v = X_df.iloc[val_idx].values, y[val_idx]
        params = dict(objective="binary", metric="binary_logloss",
                       learning_rate=0.05, num_leaves=15, max_depth=4,
                       min_data_in_leaf=8, feature_fraction=0.85,
                       bagging_fraction=0.85, bagging_freq=5,
                       lambda_l2=2.0, verbose=-1, seed=seed)
        dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=self.feature_names)
        dval = lgb.Dataset(X_v, label=y_v, reference=dtr, feature_name=self.feature_names)
        self._gbm = lgb.train(params, dtr, num_boost_round=300, valid_sets=[dval],
                               valid_names=["val"],
                               callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False),
                                          lgb.log_evaluation(0)])
        val_raw = self._gbm.predict(X_v, num_iteration=self._gbm.best_iteration)
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(val_raw, y_v)
        val_cal = self._iso.transform(val_raw)
        self.train_n = int(len(tr_idx)); self.val_n = int(len(val_idx))
        self.val_brier = float(brier_score_loss(y_v, val_cal))
        self.val_logloss = float(log_loss(y_v, np.clip(val_cal, 1e-6, 1 - 1e-6)))
        self.val_auc = float(roc_auc_score(y_v, val_cal)) if len(set(y_v)) > 1 else 0.0
        return self

    def predict_state(self, state: Dict[str, float]) -> float:
        X = pd.DataFrame([state], columns=self.feature_names).fillna(0.0).values
        raw = self._gbm.predict(X, num_iteration=self._gbm.best_iteration)
        cal = float(self._iso.transform(raw)[0])
        # Clip to [0.05, 0.95]. Isotonic on small validation sets pushes some predictions to
        # exactly 0 or 1, producing nonsensical "P(up) = 100%" outputs. Clipping caps the
        # confidence at a realistic level — top-quartile-confidence accuracy on this model
        # is ~79%, so claiming 100% certainty was always overconfident.
        return float(np.clip(cal, 0.05, 0.95))

    def predict_batch(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self.feature_names).fillna(0.0).values
        raw = self._gbm.predict(X, num_iteration=self._gbm.best_iteration)
        return np.clip(self._iso.transform(raw), 0.05, 0.95)

    def feature_importance(self, top_k: int = 10) -> List[Tuple[str, int]]:
        if not hasattr(self, "_gbm"):
            return []
        gains = self._gbm.feature_importance(importance_type="gain")
        return sorted(zip(self.feature_names, gains), key=lambda x: -x[1])[:top_k]

    def explain_state(self, state: Dict[str, float], top_k: int = 5) -> Dict:
        """Per-prediction feature contribution explanation. See PoolRespectModel.explain_prediction."""
        if not hasattr(self, "_gbm"):
            return {}
        X = pd.DataFrame([state], columns=self.feature_names).fillna(0.0).values
        contribs = self._gbm.predict(X, pred_contrib=True,
                                       num_iteration=self._gbm.best_iteration)[0]
        base = float(contribs[-1])
        feat = list(zip(self.feature_names, [float(v) for v in contribs[:-1]]))
        feat.sort(key=lambda kv: -abs(kv[1]))
        return {
            "base_logit": base,
            "raw_prediction_logit": float(contribs.sum()),
            "top_features": feat[:top_k],
        }


# ---------------------------------------------------------------------------
# Proximity model — horizon-specific
# ---------------------------------------------------------------------------

# pool_quality (P_respect) is intentionally NOT a proximity model feature: it is an
# in-sample prediction for train pools (the Q model trained on them), which leaks label
# information into P_touch training/eval. It is also conceptually weak — whether price will
# TOUCH a pool shouldn't depend on its post-touch respect quality, and distance dominates
# importance anyway. The column is still emitted by _pool_features_for_snapshot for the
# Q×T joint sanity-check diagnostic, just not consumed as a model input.
_POOL_FEATURE_NAMES = ["distance_atr", "side_above",
                       "pool_score", "pool_width_atr",
                       "pool_n_tfs", "pool_n_contributors",
                       "pool_age_at_avail_bars"]


def _pool_features_for_snapshot(pool: Pool, dist_atr: float, side: str,
                                 quality_pred: float, base_period_seconds: float = 300.0,
                                 atr_val: float = 1.0) -> Dict[str, float]:
    earliest = min((c.ts for c in pool.contributors), default=pool.formed_at)
    age_bars = (pool.available_at - earliest).total_seconds() / base_period_seconds \
               if pool.available_at >= earliest else 0.0
    return {
        "distance_atr": float(dist_atr),
        "side_above": 1.0 if side == "above" else 0.0,
        "pool_quality": float(quality_pred),
        "pool_score": float(pool.score),
        # ATR-normalised so width is comparable across assets of different price levels
        # (a raw price width confounds the feature with the stock's price scale).
        "pool_width_atr": float(pool.width / max(atr_val, 1e-9)),
        "pool_n_tfs": float(len(set(pool.tfs))),
        "pool_n_contributors": float(len(pool.contributors)),
        "pool_age_at_avail_bars": float(age_bars),
    }


@dataclass
class ProximityModel:
    horizon: int = 78
    base_period_seconds: float = 300.0      # inferred and overridden by walkforward at fit time
    feature_names: List[str] = field(default_factory=list)
    train_n: int = 0
    val_n: int = 0
    val_brier: float = 0.0
    val_logloss: float = 0.0
    val_auc: float = 0.0
    val_decile_lift: float = 0.0
    base_rate: float = 0.0

    def fit(self, snapshots: List[Snapshot], pools: List[Pool],
            quality_preds: np.ndarray,
            val_frac: float = 0.25, seed: int = 21) -> "ProximityModel":
        import lightgbm as lgb
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

        rows, labels = [], []
        for s in snapshots:
            if s.n_future_bars < self.horizon:
                continue
            for (pi, touched, dist, side) in s.pool_touch_labels(self.horizon):
                pool = pools[pi]
                pool_feat = _pool_features_for_snapshot(
                    pool, dist, side, float(quality_preds[pi]),
                    base_period_seconds=self.base_period_seconds, atr_val=s.atr_val,
                )
                rows.append({**s.state, **pool_feat})
                labels.append(touched)
        if len(rows) < 100:
            raise ValueError(f"ProximityModel at horizon {self.horizon} needs >= 100 rows, "
                              f"got {len(rows)}")

        feature_cols = STATE_FEATURE_NAMES + _POOL_FEATURE_NAMES
        X_df = pd.DataFrame(rows, columns=feature_cols).fillna(0.0)
        y = np.array(labels, dtype=int)
        self.feature_names = list(X_df.columns)
        self.base_rate = float(y.mean())

        rng = np.random.default_rng(seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        rng.shuffle(pos); rng.shuffle(neg)
        n_vp = max(1, int(round(len(pos) * val_frac)))
        n_vn = max(1, int(round(len(neg) * val_frac)))
        val_idx = np.concatenate([pos[:n_vp], neg[:n_vn]])
        tr_idx = np.concatenate([pos[n_vp:], neg[n_vn:]])
        rng.shuffle(val_idx); rng.shuffle(tr_idx)
        X_tr, y_tr = X_df.iloc[tr_idx].values, y[tr_idx]
        X_v, y_v = X_df.iloc[val_idx].values, y[val_idx]

        params = dict(objective="binary", metric="binary_logloss",
                       learning_rate=0.05, num_leaves=20, max_depth=5,
                       min_data_in_leaf=20, feature_fraction=0.85,
                       bagging_fraction=0.85, bagging_freq=5,
                       lambda_l2=2.0, verbose=-1, seed=seed)
        dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=self.feature_names)
        dval = lgb.Dataset(X_v, label=y_v, reference=dtr, feature_name=self.feature_names)
        self._gbm = lgb.train(params, dtr, num_boost_round=500, valid_sets=[dval],
                               valid_names=["val"],
                               callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False),
                                          lgb.log_evaluation(0)])
        val_raw = self._gbm.predict(X_v, num_iteration=self._gbm.best_iteration)
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(val_raw, y_v)
        val_cal = self._iso.transform(val_raw)
        self.train_n = int(len(tr_idx)); self.val_n = int(len(val_idx))
        self.val_brier = float(brier_score_loss(y_v, val_cal))
        self.val_logloss = float(log_loss(y_v, np.clip(val_cal, 1e-6, 1 - 1e-6)))
        self.val_auc = float(roc_auc_score(y_v, val_cal)) if len(set(y_v)) > 1 else 0.0
        # Decile lift on validation
        if len(val_cal) >= 20:
            order = np.argsort(val_cal)
            nv = len(val_cal)
            bot = order[: nv // 10]; top = order[-nv // 10:]
            br = float(y_v[bot].mean()) if len(bot) else 0.0
            tr = float(y_v[top].mean()) if len(top) else 0.0
            self.val_decile_lift = (tr / br) if br > 0 else float("inf")
        return self

    def fit_frame(self, frame: pd.DataFrame, val_frac: float = 0.25,
                  seed: int = 21) -> "ProximityModel":
        """Fit from persisted proximity rows.

        This is the Mac-safe path used by the feature store: each row already contains state
        features, pool features, and the horizon-specific touch label.
        """
        import lightgbm as lgb
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

        if frame is None or frame.empty:
            raise ValueError(f"ProximityModel at horizon {self.horizon} needs non-empty frame")
        if "touch_label" not in frame.columns:
            raise ValueError("proximity feature frame missing touch_label")
        feature_cols = STATE_FEATURE_NAMES + _POOL_FEATURE_NAMES
        X_df = frame.reindex(columns=feature_cols).fillna(0.0)
        y = frame["touch_label"].astype(int).to_numpy()
        if len(y) < 100:
            raise ValueError(f"ProximityModel at horizon {self.horizon} needs >= 100 rows, "
                              f"got {len(y)}")
        if len(np.unique(y)) < 2:
            raise ValueError(f"ProximityModel at horizon {self.horizon} needs both classes")
        self.feature_names = list(X_df.columns)
        self.base_rate = float(y.mean())

        rng = np.random.default_rng(seed)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        rng.shuffle(pos); rng.shuffle(neg)
        n_vp = max(1, int(round(len(pos) * val_frac)))
        n_vn = max(1, int(round(len(neg) * val_frac)))
        val_idx = np.concatenate([pos[:n_vp], neg[:n_vn]])
        tr_idx = np.concatenate([pos[n_vp:], neg[n_vn:]])
        rng.shuffle(val_idx); rng.shuffle(tr_idx)
        X_tr, y_tr = X_df.iloc[tr_idx].values, y[tr_idx]
        X_v, y_v = X_df.iloc[val_idx].values, y[val_idx]

        params = dict(objective="binary", metric="binary_logloss",
                       learning_rate=0.05, num_leaves=20, max_depth=5,
                       min_data_in_leaf=20, feature_fraction=0.85,
                       bagging_fraction=0.85, bagging_freq=5,
                       lambda_l2=2.0, verbose=-1, seed=seed)
        dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=self.feature_names)
        dval = lgb.Dataset(X_v, label=y_v, reference=dtr, feature_name=self.feature_names)
        self._gbm = lgb.train(params, dtr, num_boost_round=500, valid_sets=[dval],
                               valid_names=["val"],
                               callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False),
                                          lgb.log_evaluation(0)])
        val_raw = self._gbm.predict(X_v, num_iteration=self._gbm.best_iteration)
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(val_raw, y_v)
        val_cal = self._iso.transform(val_raw)
        self.train_n = int(len(tr_idx)); self.val_n = int(len(val_idx))
        self.val_brier = float(brier_score_loss(y_v, val_cal))
        self.val_logloss = float(log_loss(y_v, np.clip(val_cal, 1e-6, 1 - 1e-6)))
        self.val_auc = float(roc_auc_score(y_v, val_cal)) if len(set(y_v)) > 1 else 0.0
        if len(val_cal) >= 20:
            order = np.argsort(val_cal)
            nv = len(val_cal)
            bot = order[: nv // 10]; top = order[-nv // 10:]
            br = float(y_v[bot].mean()) if len(bot) else 0.0
            tr = float(y_v[top].mean()) if len(top) else 0.0
            self.val_decile_lift = (tr / br) if br > 0 else float("inf")
        return self

    def fit_distance_calib(self, frame: pd.DataFrame,
                           min_bucket_n: int = 300) -> Optional[DistanceCalibrator]:
        """Stage-C: fit a per-distance-bucket isotonic layer on labelled OOS rows.

        Uses the post-global-iso predictions as input, so this layer composes
        on top of the existing calibration rather than replacing it. Buckets
        with fewer than ``min_bucket_n`` rows fall back to identity and can
        never make calibration worse than the global model.
        """
        if frame is None or frame.empty:
            return None
        if "touch_label" not in frame.columns or "distance_atr" not in frame.columns:
            return None
        sub = frame[frame["touch_label"].notna()]
        if sub.empty:
            return None
        X = sub.reindex(columns=self.feature_names).fillna(0.0).values
        raw = self._gbm.predict(X, num_iteration=self._gbm.best_iteration)
        raw_global = self._iso.transform(raw)
        calib = DistanceCalibrator(min_bucket_n=min_bucket_n).fit(
            raw_p=raw_global,
            distance_atr=sub["distance_atr"].astype(float).to_numpy(),
            y_true=sub["touch_label"].astype(int).to_numpy(),
        )
        self._distance_calib = calib
        return calib

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        X = frame.reindex(columns=self.feature_names).fillna(0.0).values
        raw = self._gbm.predict(X, num_iteration=self._gbm.best_iteration)
        cal = self._iso.transform(raw)
        dist_calib = getattr(self, "_distance_calib", None)
        if (dist_calib is not None and dist_calib.is_fitted
                and "distance_atr" in frame.columns):
            cal = dist_calib.transform(
                cal, frame["distance_atr"].astype(float).to_numpy())
        return np.clip(cal, 0.0, 1.0)

    def predict_one(self, pool: Pool, dist_atr: float, side: str,
                    state: Dict[str, float], quality_pred: float,
                    atr_val: float = 1.0) -> float:
        pool_feat = _pool_features_for_snapshot(
            pool, dist_atr, side, quality_pred,
            base_period_seconds=self.base_period_seconds, atr_val=atr_val,
        )
        merged = {**state, **pool_feat}
        X = pd.DataFrame([merged], columns=self.feature_names).fillna(0.0).values
        raw = self._gbm.predict(X, num_iteration=self._gbm.best_iteration)
        cal = float(self._iso.transform(raw)[0])
        dist_calib = getattr(self, "_distance_calib", None)
        if dist_calib is not None and dist_calib.is_fitted:
            cal = float(dist_calib.transform(
                np.array([cal]), np.array([float(dist_atr)]))[0])
        return float(np.clip(cal, 0.0, 1.0))

    def feature_importance(self, top_k: int = 10) -> List[Tuple[str, int]]:
        if not hasattr(self, "_gbm"):
            return []
        gains = self._gbm.feature_importance(importance_type="gain")
        return sorted(zip(self.feature_names, gains), key=lambda x: -x[1])[:top_k]


# ---------------------------------------------------------------------------
# Evaluation (multi-horizon proximity, plus joint score sanity check)
# ---------------------------------------------------------------------------

@dataclass
class HorizonProxStats:
    horizon: int
    n: int
    auc: float
    brier: float
    decile_lift: float
    base_rate: float
    logloss: float = 0.0
    distance_bucket_metrics: List[Dict] = field(default_factory=list)


@dataclass
class TimingReport:
    direction_horizon: int = 0
    direction_n: int = 0
    direction_brier: float = 0.0
    direction_logloss: float = 0.0
    direction_auc: float = 0.0
    direction_top_quartile_acc: float = 0.0

    proximity_per_horizon: List[HorizonProxStats] = field(default_factory=list)

    # Joint sanity check using primary (shortest) proximity horizon
    primary_prox_horizon: int = 0
    n_joint_evaluated: int = 0
    joint_top_quartile_respect: float = 0.0
    joint_bottom_quartile_respect: float = 0.0
    joint_lift: float = 0.0


def evaluate_timing(snapshots: List[Snapshot], pools: List[Pool], results: List[PoolResult],
                     direction_model: DirectionModel,
                     proximity_models: Dict[int, ProximityModel],
                     quality_preds: np.ndarray) -> TimingReport:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    rpt = TimingReport()

    # Direction OOS evaluation at the direction model's horizon
    if direction_model is not None and snapshots:
        usable = [s for s in snapshots
                  if s.n_future_bars >= direction_model.horizon
                  and s.direction_label(direction_model.horizon) is not None]
        if usable:
            X = pd.DataFrame([s.state for s in usable], columns=STATE_FEATURE_NAMES).fillna(0.0)
            y = np.array([s.direction_label(direction_model.horizon) for s in usable], dtype=int)
            p = direction_model.predict_batch(X)
            rpt.direction_horizon = direction_model.horizon
            rpt.direction_n = int(len(y))
            rpt.direction_brier = float(brier_score_loss(y, p))
            rpt.direction_logloss = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
            if len(set(y)) > 1:
                rpt.direction_auc = float(roc_auc_score(y, p))
            conf = np.abs(p - 0.5)
            if len(p):
                thr = float(np.quantile(conf, 0.75))
                m = conf >= thr
                if m.sum() > 0:
                    pred_dir = (p[m] >= 0.5).astype(int)
                    rpt.direction_top_quartile_acc = float((pred_dir == y[m]).mean())

    # Proximity OOS evaluation per horizon
    primary_horizon = None
    primary_p_touch = None
    primary_pi_array = None
    primary_touched_array = None
    for h in sorted(proximity_models.keys()):
        pm = proximity_models[h]
        rows, labels, pis, touched_lst, distances = [], [], [], [], []
        for s in snapshots:
            if s.n_future_bars < h:
                continue
            for (pi, touched, dist, side) in s.pool_touch_labels(h):
                pool = pools[pi]
                pf = _pool_features_for_snapshot(
                    pool, dist, side, float(quality_preds[pi]),
                    base_period_seconds=pm.base_period_seconds, atr_val=s.atr_val,
                )
                rows.append({**s.state, **pf})
                labels.append(touched)
                pis.append(pi)
                touched_lst.append(touched)
                distances.append(float(dist))
        if not rows:
            continue
        X = pd.DataFrame(rows, columns=pm.feature_names).fillna(0.0)
        y = np.array(labels, dtype=int)
        raw = pm._gbm.predict(X.values, num_iteration=pm._gbm.best_iteration)
        p = np.clip(pm._iso.transform(raw), 0.0, 1.0)
        if len(set(y)) > 1:
            auc = float(roc_auc_score(y, p))
        else:
            auc = 0.0
        br = float(brier_score_loss(y, p))
        ll = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
        bucket_rows = []
        dist_arr = np.asarray(distances, dtype=float)
        for bucket in ("0-1 ATR", "1-3 ATR", "3-5 ATR", "5-10 ATR", "10+ ATR"):
            mask = np.array([distance_bucket(d) == bucket for d in dist_arr], dtype=bool)
            if not mask.any():
                continue
            row = _bucket_binary_metrics(y[mask], p[mask])
            row["bucket"] = bucket
            bucket_rows.append(row)
        # Decile lift
        decile_lift = 0.0
        if len(p) >= 20:
            order = np.argsort(p)
            nv = len(p)
            bot = order[: nv // 10]; top = order[-nv // 10:]
            bot_rate = float(y[bot].mean()) if len(bot) else 0.0
            top_rate = float(y[top].mean()) if len(top) else 0.0
            decile_lift = (top_rate / bot_rate) if bot_rate > 0 else float("inf")
        rpt.proximity_per_horizon.append(HorizonProxStats(
            horizon=h, n=int(len(y)), auc=auc, brier=br,
            decile_lift=decile_lift, base_rate=float(y.mean()),
            logloss=ll, distance_bucket_metrics=bucket_rows,
        ))
        if primary_horizon is None:
            primary_horizon = h
            primary_p_touch = p
            primary_pi_array = pis
            primary_touched_array = touched_lst

    # Joint sanity check at the primary (shortest) proximity horizon
    if primary_horizon is not None:
        rpt.primary_prox_horizon = primary_horizon
        joint = []
        for k, (pi, touched) in enumerate(zip(primary_pi_array, primary_touched_array)):
            if touched != 1:
                continue
            r = results[pi]
            if r.is_respect:
                label = 1
            elif r.is_break:
                label = 0
            else:
                continue
            score = float(quality_preds[pi]) * float(primary_p_touch[k])
            joint.append((score, label))
        if len(joint) >= 20:
            joint.sort(key=lambda x: x[0])
            m = len(joint)
            bot = joint[: m // 4]
            top = joint[-m // 4:]
            rpt.joint_bottom_quartile_respect = float(np.mean([r for _, r in bot]))
            rpt.joint_top_quartile_respect = float(np.mean([r for _, r in top]))
            rpt.joint_lift = (rpt.joint_top_quartile_respect /
                              rpt.joint_bottom_quartile_respect
                              if rpt.joint_bottom_quartile_respect > 0 else float("inf"))
            rpt.n_joint_evaluated = m

    return rpt


def evaluate_timing_frames(direction_frame: pd.DataFrame,
                           proximity_frames: Dict[int, pd.DataFrame],
                           direction_model: Optional[DirectionModel],
                           proximity_models: Dict[int, ProximityModel]) -> TimingReport:
    """Evaluate timing models from persisted feature-store rows."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    rpt = TimingReport()
    if direction_model is not None and direction_frame is not None and not direction_frame.empty:
        if "direction_label" in direction_frame.columns:
            X = direction_frame.reindex(columns=direction_model.feature_names).fillna(0.0)
            y = direction_frame["direction_label"].astype(int).to_numpy()
            if len(y):
                p = direction_model.predict_batch(X)
                rpt.direction_horizon = direction_model.horizon
                rpt.direction_n = int(len(y))
                rpt.direction_brier = float(brier_score_loss(y, p))
                rpt.direction_logloss = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
                if len(set(y)) > 1:
                    rpt.direction_auc = float(roc_auc_score(y, p))
                conf = np.abs(p - 0.5)
                thr = float(np.quantile(conf, 0.75)) if len(conf) else 1.0
                m = conf >= thr
                if m.sum() > 0:
                    rpt.direction_top_quartile_acc = float(((p[m] >= 0.5).astype(int) == y[m]).mean())

    primary_horizon = None
    primary_scores = None
    primary_respect = None
    for h in sorted(proximity_models.keys()):
        frame = proximity_frames.get(h)
        if frame is None or frame.empty or "touch_label" not in frame.columns:
            continue
        pm = proximity_models[h]
        y = frame["touch_label"].astype(int).to_numpy()
        p = pm.predict_frame(frame)
        auc = float(roc_auc_score(y, p)) if len(set(y)) > 1 else 0.0
        bucket_rows = []
        dist_arr = frame["distance_atr"].astype(float).to_numpy() if "distance_atr" in frame else np.zeros(len(y))
        for bucket in ("0-1 ATR", "1-3 ATR", "3-5 ATR", "5-10 ATR", "10+ ATR"):
            mask = np.array([distance_bucket(d) == bucket for d in dist_arr], dtype=bool)
            if not mask.any():
                continue
            row = _bucket_binary_metrics(y[mask], p[mask])
            row["bucket"] = bucket
            bucket_rows.append(row)
        decile_lift = 0.0
        if len(p) >= 20:
            order = np.argsort(p)
            nv = len(p)
            bot = order[: nv // 10]; top = order[-nv // 10:]
            bot_rate = float(y[bot].mean()) if len(bot) else 0.0
            top_rate = float(y[top].mean()) if len(top) else 0.0
            decile_lift = (top_rate / bot_rate) if bot_rate > 0 else float("inf")
        rpt.proximity_per_horizon.append(HorizonProxStats(
            horizon=h,
            n=int(len(y)),
            auc=auc,
            brier=float(brier_score_loss(y, p)),
            decile_lift=decile_lift,
            base_rate=float(y.mean()),
            logloss=float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
            distance_bucket_metrics=bucket_rows,
        ))
        if primary_horizon is None:
            primary_horizon = h
            q = frame["pool_quality"].astype(float).to_numpy() if "pool_quality" in frame else np.ones(len(p))
            primary_scores = q * p
            primary_respect = frame["respect_label"].to_numpy() if "respect_label" in frame else None

    if primary_horizon is not None:
        rpt.primary_prox_horizon = int(primary_horizon)
    if primary_scores is not None and primary_respect is not None:
        valid = ~pd.isna(primary_respect)
        if valid.sum() >= 20:
            items = sorted(zip(primary_scores[valid], primary_respect[valid]), key=lambda x: x[0])
            m = len(items)
            bot = items[: m // 4]
            top = items[-m // 4:]
            rpt.joint_bottom_quartile_respect = float(np.mean([r for _, r in bot]))
            rpt.joint_top_quartile_respect = float(np.mean([r for _, r in top]))
            rpt.joint_lift = (
                rpt.joint_top_quartile_respect / rpt.joint_bottom_quartile_respect
                if rpt.joint_bottom_quartile_respect > 0 else float("inf")
            )
            rpt.n_joint_evaluated = int(m)
    return rpt


def print_timing_report(rpt: TimingReport, file=None) -> None:
    print("\n=== Track 3: Direction + Timing Models (OOS) ===", file=file)
    print(f"\n[direction model]   horizon = {rpt.direction_horizon} bars  "
          f"(~{rpt.direction_horizon * 5 / 60:.1f}h on 5m base)", file=file)
    print(f"  OOS samples:                    {rpt.direction_n}", file=file)
    print(f"  Brier:                          {rpt.direction_brier:.4f}", file=file)
    print(f"  Log loss:                       {rpt.direction_logloss:.4f}", file=file)
    print(f"  AUC-ROC:                        {rpt.direction_auc:.3f}", file=file)
    print(f"  Top-quartile-confidence acc:    {rpt.direction_top_quartile_acc:.1%}  "
          f"(accuracy when model is most certain)", file=file)

    if rpt.proximity_per_horizon:
        print(f"\n[proximity models per horizon]   target = pool touched within H bars",
              file=file)
        print(f"  {'horizon':<8} {'~hours':<8} {'n_oos':>7} {'AUC':>6} {'Brier':>7} "
              f"{'base_rate':>10} {'decile_lift':>12}", file=file)
        for s in rpt.proximity_per_horizon:
            hours = s.horizon * 5 / 60.0
            lift_str = "inf" if s.decile_lift == float("inf") else f"{s.decile_lift:.1f}x"
            print(f"  {s.horizon:<8} {hours:<7.1f}h {s.n:>7} {s.auc:>6.3f} {s.brier:>7.4f} "
                  f"{s.base_rate:>9.1%} {lift_str:>12}", file=file)
            if s.distance_bucket_metrics:
                print(f"    {'bucket':<9} {'n':>6} {'base':>8} {'AUC':>6} {'Brier':>8} "
                      f"{'logloss':>8} {'cal_err':>8}", file=file)
                for row in s.distance_bucket_metrics:
                    auc_s = f"{row['auc']:.3f}" if row["auc"] is not None else "n/a"
                    print(f"    {row['bucket']:<9} {row['n']:>6} {row['base_rate']:>7.1%} "
                          f"{auc_s:>6} {row['brier']:>8.4f} {row['logloss']:>8.4f} "
                          f"{row['calibration_error']:>+7.1%}", file=file)

    print(f"\n[joint score sanity check]  Q × T_h{rpt.primary_prox_horizon} on touched pools",
          file=file)
    print(f"  n decisive:             {rpt.n_joint_evaluated}", file=file)
    print(f"  Top quartile respect:   {rpt.joint_top_quartile_respect:.1%}", file=file)
    print(f"  Bottom quartile:        {rpt.joint_bottom_quartile_respect:.1%}", file=file)
    print(f"  Joint lift:             {rpt.joint_lift:.2f}x", file=file)
    if rpt.joint_lift < 1.2:
        print(f"  → joint score does NOT rank touched pools well. Use T as filter "
              f"(tradeable today), Q for ranking among tradeable.", file=file)
