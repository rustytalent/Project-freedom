"""Pure-numpy indicators used by feature detectors. Vectorised, no pandas overhead in hot paths."""
from __future__ import annotations
import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    s = pd.Series(tr, index=df.index)
    wilder = s.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    # Warm-up (first `period-1` bars) is NaN under min_periods=period. Fill it with the CAUSAL
    # expanding mean of TR rather than back-filling from the first formed value — back-filling
    # would copy a future ATR into the earliest bars (a look-ahead leak). Callers that .bfill()
    # this series then become no-ops, which is the intent.
    return wilder.fillna(s.expanding(min_periods=1).mean())


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def momentum(series: pd.Series, period: int = 10) -> pd.Series:
    return series.diff(period)


def candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar features used both directly and by the in-candle imbalance detector.

    - body_ratio: |c-o| / (h-l)
    - upper_wick_ratio: (h - max(o,c)) / (h-l)
    - lower_wick_ratio: (min(o,c) - l) / (h-l)
    - direction: +1 bull, -1 bear, 0 doji
    - displacement: (c-o) / ATR -- normalised body
    """
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    upper = (h - pd.concat([o, c], axis=1).max(axis=1))
    lower = (pd.concat([o, c], axis=1).min(axis=1) - l)
    out = pd.DataFrame(index=df.index)
    out["body_ratio"] = (body / rng).fillna(0)
    out["upper_wick_ratio"] = (upper / rng).fillna(0)
    out["lower_wick_ratio"] = (lower / rng).fillna(0)
    out["direction"] = np.sign(c - o).astype(int)
    out["range"] = (h - l).values
    a = atr(df, 14)
    out["displacement_atr"] = ((c - o) / a).fillna(0)
    return out
