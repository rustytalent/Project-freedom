"""Market regime / context features. Strictly causal — every value at time T is computable from
data with timestamp <= T. Used by featurize.py to enrich the per-pool ML feature vector with
"what was the market doing when this pool became known".

  - ADX(14): trend strength on the base TF. High ADX = trending, low = ranging.
  - Volatility regime: short-ATR / long-ATR ratio. > 1 = vol expanding, < 1 = contracting.
  - NSE session label: opening / morning / midday / closing / off_session, IST-mapped.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

from .indicators import atr


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Standard Wilder ADX(14). Higher = stronger directional move."""
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    dn = -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    prev_c = c.shift()
    tr = pd.concat([(h - l).abs(),
                    (h - prev_c).abs(),
                    (l - prev_c).abs()], axis=1).max(axis=1)
    alpha = 1.0 / period
    atr_n = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * (pd.Series(plus_dm, index=df.index)
                       .ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr_n)
    minus_di = 100.0 * (pd.Series(minus_dm, index=df.index)
                        .ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr_n)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean().fillna(0.0)


def vol_regime(df: pd.DataFrame, fast_period: int = 14, slow_period: int = 60) -> pd.Series:
    """ATR(fast) / ATR(slow). Above 1 = recent vol > long-run vol (expansion)."""
    fast = atr(df, fast_period).bfill()
    slow = atr(df, slow_period).bfill()
    return (fast / slow.replace(0, np.nan)).fillna(1.0)


# NSE session boundaries in IST minutes since midnight.
#   pre-open (handled as off_session for our purposes; we don't trade pre-open)
#   opening_hour : 09:15 - 10:15 IST
#   morning      : 10:15 - 12:00 IST
#   midday       : 12:00 - 13:30 IST
#   closing      : 13:30 - 15:30 IST
def nse_session(ts: pd.Timestamp) -> str:
    """Return NSE session label for a UTC timestamp. Robust to tz-naive (assumed UTC)."""
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    ist = ts + pd.Timedelta(hours=5, minutes=30)
    minute_of_day = ist.hour * 60 + ist.minute
    if minute_of_day < 9 * 60 + 15 or minute_of_day > 15 * 60 + 30:
        return "off_session"
    if minute_of_day < 10 * 60 + 15:
        return "opening_hour"
    if minute_of_day < 12 * 60:
        return "morning"
    if minute_of_day < 13 * 60 + 30:
        return "midday"
    return "closing"


# Stable enumeration so featurize.py can produce consistent one-hot columns across folds.
SESSION_LABELS = ("opening_hour", "morning", "midday", "closing", "off_session")


def compute_regime_series(df: pd.DataFrame) -> pd.DataFrame:
    """Build a DataFrame indexed by df.index with columns adx_14, vol_ratio, session."""
    out = pd.DataFrame(index=df.index)
    out["adx_14"] = adx(df, 14)
    out["vol_ratio"] = vol_regime(df, 14, 60)
    out["session"] = [nse_session(t) for t in df.index]
    return out


def lookup_regime(regime_df: pd.DataFrame, ts: pd.Timestamp) -> dict:
    """Causal lookup: find the latest regime row whose timestamp is <= ts."""
    idx = regime_df.index
    pos = idx.searchsorted(ts, side="right") - 1
    if pos < 0:
        return {"adx_14": float("nan"), "vol_ratio": float("nan"), "session": "off_session"}
    row = regime_df.iloc[pos]
    return {"adx_14": float(row["adx_14"]),
            "vol_ratio": float(row["vol_ratio"]),
            "session": str(row["session"])}
