"""Data fetching and multi-timeframe resampling.

Primary source: yfinance. Caches to local parquet to avoid hammering the API and to keep
backtests reproducible. Falls back to a synthetic OHLCV generator if yfinance is unavailable
or returns nothing (useful for offline development / CI)."""
from __future__ import annotations
from pathlib import Path
from typing import Dict
import hashlib
import pandas as pd
import numpy as np


CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(symbol: str, interval: str, period: str | None, start: str | None, end: str | None) -> Path:
    key = f"{symbol}|{interval}|{period}|{start}|{end}"
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    safe_sym = symbol.replace("/", "_").replace("=", "")
    return CACHE_DIR / f"{safe_sym}_{interval}_{h}.parquet"


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance sometimes returns MultiIndex columns or capitalized names. Normalize to lower-case flat."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.index.name = "ts"
    return df.sort_index()


def _synthetic(symbol: str, interval: str, n_bars: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Geometric-Brownian-ish OHLCV with realistic wicks and occasional jumps."""
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32) ^ seed)
    freq_map = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "60m": "60min",
                "1h": "60min", "1d": "1D", "1D": "1D"}
    freq = freq_map.get(interval, "5min")
    idx = pd.date_range("2024-01-02 09:30", periods=n_bars, freq=freq)
    rets = rng.normal(0, 0.0015, n_bars)
    jump_mask = rng.random(n_bars) < 0.01
    rets[jump_mask] += rng.normal(0, 0.01, jump_mask.sum())
    close = 150 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.001, n_bars)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.001, n_bars)))
    vol = rng.integers(1_000, 50_000, n_bars).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx).rename_axis("ts")


def fetch(symbol: str, interval: str = "5m", period: str | None = "60d",
          start: str | None = None, end: str | None = None, use_cache: bool = True,
          allow_synthetic: bool = True) -> pd.DataFrame:
    fp = _cache_path(symbol, interval, period, start, end)
    if use_cache and fp.exists():
        return pd.read_parquet(fp)

    df: pd.DataFrame | None = None
    try:
        import yfinance as yf
        if start or end:
            raw = yf.download(symbol, interval=interval, start=start, end=end,
                              progress=False, auto_adjust=False, threads=False)
        else:
            raw = yf.download(symbol, interval=interval, period=period,
                              progress=False, auto_adjust=False, threads=False)
        if raw is not None and not raw.empty:
            df = _normalize_ohlcv(raw)
    except Exception as e:
        print(f"[data] yfinance fetch failed: {e}")

    if (df is None or df.empty) and allow_synthetic:
        print(f"[data] using synthetic data for {symbol} {interval} (yfinance unavailable or empty)")
        df = _synthetic(symbol, interval)

    if df is None or df.empty:
        raise RuntimeError(f"No data for {symbol} @ {interval}")

    df.to_parquet(fp)
    return df


# Map of higher-TF strings → pandas resample rule.
_TF_RULE = {
    "15m": "15min", "15min": "15min",
    "30m": "30min", "30min": "30min",
    "60m": "60min", "1h": "60min", "60min": "60min",
    "240m": "240min", "4h": "240min", "240min": "240min",
    "1D": "1D", "1d": "1D", "D": "1D",
    "1W": "1W", "1w": "1W", "W": "1W",
}


def resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = _TF_RULE.get(tf, tf)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(rule, label="left", closed="left").agg(agg).dropna(how="any")
    return out


def multi_timeframe(df_base: pd.DataFrame, tfs: list[str]) -> Dict[str, pd.DataFrame]:
    """Returns {tf: df} including the base TF under key 'base'."""
    out: Dict[str, pd.DataFrame] = {"base": df_base}
    for tf in tfs:
        try:
            out[tf] = resample(df_base, tf)
        except Exception as e:
            print(f"[data] resample {tf} failed: {e}")
    return out
