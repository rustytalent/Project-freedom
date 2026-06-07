"""Per-strike options featurizer (Stream D.1).

Produces an ML-ready ``DataFrame`` keyed by ``(trading_date_ist, bar_ts,
underlying, strike, side)`` carrying the L1 feature set defined in
``docs/options_strategy_methodology.md §3``.

Design contracts (every one of these is pinned by a test):

  * **Causality.** Every feature's information timestamp is ≤ the bar
    timestamp it sits on. The daily-Greeks join goes through
    :func:`liqpool.warehouse.align_daily_to_intraday` which lags daily
    features by exactly one trading day. ATR uses Wilder's recursion
    with no back-fill.

  * **No look-ahead under truncation.** Truncating the underlying frame
    to the prediction bar produces identical features for that bar to
    running on the full frame. The test pins this on every causal
    column.

  * **Determinism.** Given the same inputs (underlying, Greeks frame,
    strike list, params), the output frame is bit-identical across
    runs. No timestamps from ``now()``; no random seeds.

  * **NaN, not zero, for missing.** When Greeks data is missing for a
    strike (illiquid, delisted, gap day), the corresponding feature
    columns are ``NaN``. Callers downstream must handle imputation —
    we never lie about what we know.

  * **No labels.** The featurizer is label-free. Labels live in
    Stream D.2 (`liqpool/options/labels.py`). This keeps train and
    predict on the same feature contract.

The frame is the input to the LightGBM ``OptionsExpectedReturnModel``
(Stream D.3) AND to the layered-conviction executor (Stream D.5). It
is NOT itself the model; it is the data scaffold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..indicators import atr
from ..warehouse import align_daily_to_intraday, assert_sorted_by_time


# ---------------------------------------------------------------------------
# Public column contract
# ---------------------------------------------------------------------------

# These columns appear in the order below in every emitted frame. New
# columns get appended at the end so downstream consumers (LightGBM
# trainers, executor) can read by name without index assumptions.
OPTIONS_FEATURE_COLUMNS: tuple[str, ...] = (
    # Identification (not features; key columns the model joins on)
    "trading_date_ist", "bar_ts", "underlying", "strike", "side",
    # Per-bar instrument structural features
    "spot", "atr_underlying", "dist_strike_to_spot_atr", "abs_dist_pct",
    "moneyness_bucket",                # ATM / 1OTM / 2OTM / 3OTM
    "dte_trading_days", "weekly_expiry_flag",
    "tod_minutes_since_open", "tod_minutes_to_eod",
    "tod_bucket",                       # open / mid / pre_close / close
    # Per-bar volatility regime (intraday, from underlying)
    "underlying_5m_return", "underlying_30m_return",
    "underlying_realized_vol_30m",
    # Lagged daily Greeks / IV (from align_daily_to_intraday)
    "delta_yday", "gamma_yday", "theta_per_day_pct_yday",
    "vega_per_volpoint_pct_yday",
    "iv_yday", "iv_dod_change_bps", "iv_percentile_60d", "iv_rank_60d",
    "pcr_oi_yday",
    # Lagged daily macro
    "india_vix_close_yday", "india_vix_dod_change",
    "usdinr_dod_change_bps",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OptionsFeaturizerParams:
    """Parameters for the per-strike featurizer.

    Most defaults map directly to choices in
    ``docs/options_strategy_methodology.md``. Knobs are kept minimal
    on purpose — every parameter that exists is a question the test
    suite has to answer.
    """
    atr_period: int = 14
    realized_vol_window_bars: int = 6   # ~30 min on 5m bars
    moneyness_atm_window_pct: float = 0.0025  # ±0.25% = ATM bucket
    moneyness_step_pct: float = 0.0050        # 0.50% per OTM step
    weekly_expiry_dow_by_underlying: dict[str, int] = field(
        default_factory=lambda: {"NIFTY50": 3, "BANKNIFTY": 2}
    )  # Mon=0, ..., Sun=6 → NIFTY Thu, BANKNIFTY Wed
    # ToD buckets, IST minutes since 09:15 open. End is exclusive.
    tod_buckets: tuple[tuple[str, int, int], ...] = (
        ("open",      0,   75),   # 09:15 – 10:30
        ("mid",       75,  225),  # 10:30 – 13:00
        ("pre_close", 225, 330),  # 13:00 – 14:45
        ("close",     330, 360),  # 14:45 – 15:15
    )
    # When the Greeks frame doesn't carry IV percentile / rank already,
    # the featurizer computes them over this window.
    iv_percentile_window_days: int = 60


# ---------------------------------------------------------------------------
# Helpers — pure functions, each tested in isolation
# ---------------------------------------------------------------------------

def _ist_minutes_since_open(ts: pd.Series) -> pd.Series:
    """Minutes since 09:15 IST on each bar timestamp.

    Input ``ts`` is interpreted as IST already (warehouse convention).
    Bars before 09:15 IST clamp to 0; bars after 15:15 clamp to 360.
    """
    h = pd.to_datetime(ts).dt.hour
    m = pd.to_datetime(ts).dt.minute
    minutes = (h - 9) * 60 + (m - 15)
    return minutes.clip(lower=0, upper=360)


def _tod_bucket(minutes_since_open: pd.Series,
                buckets: Sequence[tuple[str, int, int]]) -> pd.Series:
    """Label each bar with its time-of-day bucket per the params."""
    out = pd.Series(["mid"] * len(minutes_since_open),
                    index=minutes_since_open.index, dtype=object)
    # Walk in reverse so the assignment lands in the first matching range.
    for label, lo, hi in reversed(buckets):
        mask = (minutes_since_open >= lo) & (minutes_since_open < hi)
        out.loc[mask] = label
    return out


def _moneyness_bucket(dist_atr: pd.Series,
                      spot: pd.Series, strike: pd.Series,
                      atm_window_pct: float,
                      step_pct: float) -> pd.Series:
    """Bucket the strike's moneyness as ATM / 1OTM / 2OTM / 3OTM_plus.

    Uses absolute percent distance from spot to strike, not ATR-distance,
    so the bucket is stable across vol regimes. ATR-distance is still
    a separate feature column (`dist_strike_to_spot_atr`).
    """
    pct = (strike - spot).abs() / spot.replace(0, np.nan)
    out = pd.Series(["3OTM_plus"] * len(pct), index=pct.index, dtype=object)
    out.loc[pct >= 3 * step_pct + atm_window_pct] = "3OTM_plus"
    out.loc[(pct < 3 * step_pct + atm_window_pct)
            & (pct >= 2 * step_pct + atm_window_pct)] = "2OTM"
    out.loc[(pct < 2 * step_pct + atm_window_pct)
            & (pct >= step_pct + atm_window_pct)] = "1OTM"
    out.loc[pct < atm_window_pct + step_pct] = "ATM_or_near"
    out.loc[pct < atm_window_pct] = "ATM"
    return out


def _trading_days_to_expiry(bar_dates: pd.Series,
                            expiry_dates: pd.Series) -> pd.Series:
    """Trading-day count between bar date and contract expiry date.

    Uses ``numpy.busday_count`` which handles the standard Mon-Fri
    business calendar. NSE-holiday-adjusted DTE would need a holiday
    list (out of scope at v1; the bias is small for ATM weekly DTE).
    """
    bar_d = pd.to_datetime(bar_dates).dt.normalize().values.astype("datetime64[D]")
    exp_d = pd.to_datetime(expiry_dates).dt.normalize().values.astype("datetime64[D]")
    # busday_count is end-exclusive; +1 to include expiry day in DTE
    counts = np.busday_count(bar_d, exp_d) + 1
    return pd.Series(counts, index=bar_dates.index, dtype="float64")


def _percentile_rank_within_window(series: pd.Series, window: int) -> pd.Series:
    """Rolling causal percentile-rank of the current value within the
    last ``window`` observations. Result in [0, 1]. NaN for the
    warmup. Uses strict-less-than count so the same value across the
    window doesn't always return 1.0."""
    arr = series.values.astype(float)
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window + 1)
        slab = arr[lo:i + 1]
        slab = slab[~np.isnan(slab)]
        if len(slab) < max(5, window // 2):  # require some warmup
            continue
        current = arr[i]
        if np.isnan(current):
            continue
        rank = (slab < current).sum()
        out[i] = rank / max(len(slab) - 1, 1)
    return pd.Series(out, index=series.index)


# ---------------------------------------------------------------------------
# Greeks-frame normalisation
# ---------------------------------------------------------------------------

_REQUIRED_GREEKS_COLS = (
    "trading_date", "underlying", "strike",
    "delta", "gamma", "theta", "vega", "iv", "premium_close",
)


def _normalise_greeks(greeks: pd.DataFrame,
                      params: OptionsFeaturizerParams) -> pd.DataFrame:
    """Sanity-check and enrich the daily Greeks frame.

    Adds derived columns:
      - ``theta_per_day_pct = theta / premium_close``
      - ``vega_per_volpoint_pct = vega / premium_close``
      - ``iv_dod_change_bps = (iv - iv_prev) * 10000``
      - ``iv_percentile_60d``, ``iv_rank_60d`` (per-strike rolling)

    All computations are CAUSAL: ``shift(1)`` produces the lagged
    yesterday value; no future bar can contaminate today's row.
    """
    missing = [c for c in _REQUIRED_GREEKS_COLS if c not in greeks.columns]
    if missing:
        raise ValueError(f"greeks frame missing required columns: {missing}")
    g = greeks.copy()
    g["trading_date"] = pd.to_datetime(g["trading_date"]).dt.normalize()
    g = g.sort_values(["underlying", "strike", "trading_date"]).reset_index(
        drop=True)

    # Per-strike derived columns. Use groupby so each strike's rolling
    # computation does not leak into others.
    g["theta_per_day_pct"] = (
        g["theta"] / g["premium_close"].replace(0, np.nan)
    )
    g["vega_per_volpoint_pct"] = (
        g["vega"] / g["premium_close"].replace(0, np.nan)
    )
    g["iv_dod_change_bps"] = (
        g.groupby(["underlying", "strike"])["iv"].diff() * 10_000
    )

    # Rolling percentile + rank of IV within the per-strike window.
    pct_window = params.iv_percentile_window_days
    g["iv_percentile_60d"] = (
        g.groupby(["underlying", "strike"])["iv"]
         .transform(lambda s: _percentile_rank_within_window(s, pct_window))
    )

    def _rank_window(s: pd.Series) -> pd.Series:
        # Min-max rank: where today's IV sits inside the last `window` days'
        # observed [min, max] range. Returns NaN during warmup.
        arr = s.values.astype(float)
        n = len(arr)
        out = np.full(n, np.nan)
        for i in range(n):
            lo = max(0, i - pct_window + 1)
            slab = arr[lo:i + 1]
            slab = slab[~np.isnan(slab)]
            if len(slab) < max(5, pct_window // 2):
                continue
            lo_v, hi_v = float(np.min(slab)), float(np.max(slab))
            if hi_v <= lo_v:
                out[i] = 0.5
                continue
            cur = arr[i]
            if np.isnan(cur):
                continue
            out[i] = (cur - lo_v) / (hi_v - lo_v)
        return pd.Series(out, index=s.index)

    g["iv_rank_60d"] = (
        g.groupby(["underlying", "strike"])["iv"].transform(_rank_window)
    )
    return g


# ---------------------------------------------------------------------------
# Top-level featurizer
# ---------------------------------------------------------------------------

def build_options_feature_frame(
    underlying_bars: pd.DataFrame,
    greeks_daily: pd.DataFrame,
    macro_daily: Optional[pd.DataFrame],
    strikes: Iterable[dict],
    *,
    underlying: str,
    params: Optional[OptionsFeaturizerParams] = None,
) -> pd.DataFrame:
    """Build the ML-ready options feature frame.

    Parameters
    ----------
    underlying_bars
        Intraday 5-minute bars of the underlying index, with columns
        ``timestamp, open, high, low, close, volume``. Index can be
        anything; ``timestamp`` column is the authoritative IST time.
    greeks_daily
        EOD Greeks per (underlying, strike, trading_date). Must carry
        ``trading_date, underlying, strike, delta, gamma, theta, vega,
        iv, premium_close``. Optional ``pcr_oi`` column passes through.
    macro_daily
        Daily macro frame with ``trading_date, india_vix_close,
        usdinr_close``. May be ``None`` — the corresponding feature
        columns will be all-NaN, which downstream code must handle.
    strikes
        Iterable of dicts ``{strike: float, side: "buy"|"sell",
        expiry_date: 'YYYY-MM-DD'}``. The featurizer emits one row
        per (bar × strike × side) combination.
    underlying
        Underlying symbol name — written into the ``underlying``
        column for routing.
    params
        Featurizer parameters; defaults are the v1 narrow-scope values.

    Returns
    -------
    pandas.DataFrame
        Columns per :data:`OPTIONS_FEATURE_COLUMNS`, in that order.
        Index is RangeIndex (no datetime; the time keys are columns).
    """
    p = params or OptionsFeaturizerParams()
    if "timestamp" not in underlying_bars.columns:
        underlying_bars = underlying_bars.copy()
        underlying_bars["timestamp"] = underlying_bars.index
    assert_sorted_by_time(underlying_bars, ts_col="timestamp")

    # ---- Underlying-level features (computed once; broadcast to each strike) -
    bars = underlying_bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    bars["trading_date_ist"] = bars["timestamp"].dt.normalize()
    bars["atr_underlying"] = atr(bars, p.atr_period).bfill().values
    bars["tod_minutes_since_open"] = _ist_minutes_since_open(bars["timestamp"])
    bars["tod_minutes_to_eod"] = (360 - bars["tod_minutes_since_open"]).clip(
        lower=0, upper=360)
    bars["tod_bucket"] = _tod_bucket(bars["tod_minutes_since_open"],
                                     p.tod_buckets)
    bars["underlying_5m_return"] = bars["close"].pct_change(fill_method=None)
    bars["underlying_30m_return"] = bars["close"].pct_change(
        periods=p.realized_vol_window_bars, fill_method=None)
    # 30-min realized vol = stdev of 5m returns over the window. Causal:
    # the rolling window ends at the current bar, never includes future.
    bars["underlying_realized_vol_30m"] = bars["underlying_5m_return"].rolling(
        p.realized_vol_window_bars, min_periods=2).std()

    # ---- Daily Greeks features (lagged via align_daily_to_intraday) ---------
    greeks = _normalise_greeks(greeks_daily, p)

    # ---- Macro merge (lagged via align_daily_to_intraday) ------------------
    if macro_daily is not None:
        macro = macro_daily.copy()
        macro["trading_date"] = pd.to_datetime(macro["trading_date"]).dt.normalize()
        macro = macro.sort_values("trading_date").reset_index(drop=True)
        macro["india_vix_dod_change"] = macro["india_vix_close"].diff()
        macro["usdinr_dod_change_bps"] = macro["usdinr_close"].pct_change() * 10_000
        macro_cols = [
            "india_vix_close", "india_vix_dod_change", "usdinr_dod_change_bps",
        ]
    else:
        macro = None
        macro_cols = []

    # ---- Per-strike row emission -------------------------------------------
    out_rows: list[pd.DataFrame] = []
    for s in strikes:
        strike = float(s["strike"])
        side = str(s["side"])
        expiry = pd.to_datetime(s["expiry_date"]).normalize()
        weekly_dow = p.weekly_expiry_dow_by_underlying.get(underlying)

        strike_bars = bars.copy()
        strike_bars["underlying"] = underlying
        strike_bars["strike"] = strike
        strike_bars["side"] = side
        strike_bars["bar_ts"] = strike_bars["timestamp"]
        strike_bars["spot"] = strike_bars["close"]

        # Structural features keyed to this strike.
        strike_bars["dist_strike_to_spot_atr"] = (
            (strike - strike_bars["spot"])
            / strike_bars["atr_underlying"].replace(0, np.nan)
        )
        strike_bars["abs_dist_pct"] = (
            (strike - strike_bars["spot"]).abs()
            / strike_bars["spot"].replace(0, np.nan)
        )
        strike_bars["moneyness_bucket"] = _moneyness_bucket(
            strike_bars["dist_strike_to_spot_atr"],
            strike_bars["spot"], pd.Series([strike] * len(strike_bars),
                                            index=strike_bars.index),
            p.moneyness_atm_window_pct, p.moneyness_step_pct,
        )
        strike_bars["dte_trading_days"] = _trading_days_to_expiry(
            strike_bars["trading_date_ist"],
            pd.Series([expiry] * len(strike_bars),
                      index=strike_bars.index),
        )
        if weekly_dow is None:
            strike_bars["weekly_expiry_flag"] = pd.NA
        else:
            strike_bars["weekly_expiry_flag"] = (
                strike_bars["trading_date_ist"].dt.dayofweek == weekly_dow
            )

        # Daily-Greeks join: lag by 1 trading day via the warehouse helper.
        g_strike = greeks[
            (greeks["underlying"] == underlying)
            & (np.isclose(greeks["strike"], strike))
        ].copy()
        if not g_strike.empty:
            keep_cols = [
                "trading_date",
                "delta", "gamma", "theta_per_day_pct",
                "vega_per_volpoint_pct", "iv", "iv_dod_change_bps",
                "iv_percentile_60d", "iv_rank_60d",
            ]
            if "pcr_oi" in g_strike.columns:
                keep_cols.append("pcr_oi")
            g_lean = g_strike[keep_cols].rename(columns={
                "delta":                  "delta_yday",
                "gamma":                  "gamma_yday",
                "theta_per_day_pct":      "theta_per_day_pct_yday",
                "vega_per_volpoint_pct":  "vega_per_volpoint_pct_yday",
                "iv":                     "iv_yday",
                "pcr_oi":                 "pcr_oi_yday",
            })
            joined = align_daily_to_intraday(
                strike_bars, g_lean,
                intraday_ts="timestamp", daily_date="trading_date",
            )
        else:
            joined = strike_bars
            for col in (
                "delta_yday", "gamma_yday", "theta_per_day_pct_yday",
                "vega_per_volpoint_pct_yday", "iv_yday",
                "iv_dod_change_bps", "iv_percentile_60d", "iv_rank_60d",
                "pcr_oi_yday",
            ):
                joined[col] = np.nan

        # Macro merge (also lagged by one day via warehouse helper).
        if macro is not None:
            joined = align_daily_to_intraday(
                joined, macro[["trading_date"] + macro_cols],
                intraday_ts="timestamp", daily_date="trading_date",
            )
            # Standardise the renamed columns to the final names.
            joined = joined.rename(columns={
                "india_vix_close": "india_vix_close_yday",
            })
        else:
            joined["india_vix_close_yday"] = np.nan
            joined["india_vix_dod_change"] = np.nan
            joined["usdinr_dod_change_bps"] = np.nan

        out_rows.append(joined)

    if not out_rows:
        # Empty strike list → empty frame with the right schema.
        out = pd.DataFrame(columns=list(OPTIONS_FEATURE_COLUMNS))
        return out
    final = pd.concat(out_rows, ignore_index=True)

    # Ensure every contract column exists (NaN-fill when missing).
    for col in OPTIONS_FEATURE_COLUMNS:
        if col not in final.columns:
            final[col] = np.nan

    # `align_daily_to_intraday` internally sorts by `_date` which is
    # the normalised day. Bars on the same day can land in unstable
    # order across calls. Lock the output ordering by the keys the
    # caller actually consumes by, so truncated inputs produce
    # truncated outputs (not reordered ones).
    final = final.sort_values(
        ["underlying", "strike", "side", "bar_ts"]
    ).reset_index(drop=True)
    return final[list(OPTIONS_FEATURE_COLUMNS)]
