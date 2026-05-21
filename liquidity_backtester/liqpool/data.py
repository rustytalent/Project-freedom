"""Data fetching and multi-timeframe resampling.

Primary source: yfinance. Caches to local parquet to avoid hammering the API and to keep
backtests reproducible. Falls back to a synthetic OHLCV generator if yfinance is unavailable
or returns nothing (useful for offline development / CI).

Research/production source: local Zerodha/Kite parquet via :class:`ParquetProvider`.
Parquet mode never calls yfinance and never creates synthetic fallback data.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Optional
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


def _normalise_symbol_key(symbol: str) -> str:
    return symbol.upper().replace(".NS", "").replace(".BO", "")


def _normalise_parquet_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize local parquet bars to the internal OHLCV format indexed by UTC-naive ts."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out = out.rename(columns={c: str(c).lower() for c in out.columns})
    if "date" in out.columns and "ts" not in out.columns:
        out = out.rename(columns={"date": "ts"})
    if "datetime" in out.columns and "ts" not in out.columns:
        out = out.rename(columns={"datetime": "ts"})
    if "timestamp" in out.columns and "ts" not in out.columns:
        out = out.rename(columns={"timestamp": "ts"})
    if "ts" in out.columns:
        out["ts"] = pd.to_datetime(out["ts"])
        out = out.set_index("ts")
    else:
        out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    missing = [c for c in ("open", "high", "low", "close", "volume") if c not in out.columns]
    if missing:
        raise ValueError(f"{symbol}: missing parquet OHLCV columns: {missing}")
    out = out[["open", "high", "low", "close", "volume"]].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(how="any").sort_index()
    out.index.name = "ts"
    return _validate_ohlcv(out, symbol)


def _synthetic(symbol: str, interval: str, n_bars: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Geometric-Brownian-ish OHLCV with realistic wicks and occasional jumps.

    Starting price is picked by a tiny symbol→price heuristic so the chart looks plausible for
    that asset (e.g. HDFCBANK ≈ ₹1800). This is only used when no real data source is reachable.
    """
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32) ^ seed)
    freq_map = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "60m": "60min",
                "1h": "60min", "1d": "1D", "1D": "1D"}
    freq = freq_map.get(interval, "5min")
    idx = pd.date_range("2024-01-02 09:30", periods=n_bars, freq=freq)
    sym_u = symbol.upper()
    if "HDFCBANK" in sym_u:
        start_price, vol_scale = 1800.0, 1.0
    elif sym_u.endswith(".NS") or sym_u.endswith(".BO"):
        start_price, vol_scale = 1200.0, 1.0
    elif "BTC" in sym_u:
        start_price, vol_scale = 65000.0, 2.0
    else:
        start_price, vol_scale = 150.0, 1.0
    rets = rng.normal(0, 0.0015 * vol_scale, n_bars)
    jump_mask = rng.random(n_bars) < 0.01
    rets[jump_mask] += rng.normal(0, 0.01 * vol_scale, jump_mask.sum())
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.001 * vol_scale, n_bars)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.001 * vol_scale, n_bars)))
    vol = rng.integers(1_000, 50_000, n_bars).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx).rename_axis("ts")


def fetch(symbol: str, interval: str = "5m", period: str | None = "60d",
          start: str | None = None, end: str | None = None, use_cache: bool = True,
          allow_synthetic: bool = True) -> pd.DataFrame:
    fp = _cache_path(symbol, interval, period, start, end)
    if use_cache and fp.exists():
        return pd.read_parquet(fp)

    def _yf_download(sym: str) -> pd.DataFrame | None:
        try:
            import yfinance as yf
            if start or end:
                raw = yf.download(sym, interval=interval, start=start, end=end,
                                  progress=False, auto_adjust=False, threads=False)
            else:
                raw = yf.download(sym, interval=interval, period=period,
                                  progress=False, auto_adjust=False, threads=False)
            if raw is not None and not raw.empty:
                return _normalize_ohlcv(raw)
        except Exception as e:
            print(f"[data] yfinance fetch failed for {sym}: {e}")
        return None

    df = _yf_download(symbol)

    # Fallback: many NSE symbols (e.g. TATAMOTORS) sometimes 404 on .NS but resolve on .BO
    # (BSE listing). The price data is essentially the same for liquid names. Try once.
    if (df is None or df.empty) and symbol.endswith(".NS"):
        bo_symbol = symbol.replace(".NS", ".BO")
        print(f"[data] {symbol} not available on NSE; trying {bo_symbol} (BSE) ...")
        df = _yf_download(bo_symbol)
        if df is not None and not df.empty:
            print(f"[data] using {bo_symbol} (BSE) data in place of {symbol}")

    if (df is None or df.empty) and allow_synthetic:
        print(f"[data] using synthetic data for {symbol} {interval} (yfinance unavailable or empty)")
        df = _synthetic(symbol, interval)

    if df is None or df.empty:
        raise RuntimeError(f"No data for {symbol} @ {interval}")

    # Data quality validation — log + drop bars with broken OHLC, mark short series.
    df = _validate_ohlcv(df, symbol)

    df.to_parquet(fp)
    return df


class DataProvider:
    """Abstract source of normalized OHLCV bars."""

    def load_base(self, symbol: str, interval: str, period: Optional[str] = None,
                  start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        raise NotImplementedError

    def load_timeframes(self, symbol: str, base_tf: str,
                        higher_tfs: Iterable[str],
                        period: Optional[str] = None,
                        start: Optional[str] = None,
                        end: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        base = self.load_base(symbol, base_tf, period=period, start=start, end=end)
        return multi_timeframe(base, list(higher_tfs))


class YFinanceProvider(DataProvider):
    """Current yfinance provider. Synthetic fallback is opt-in and disabled by parquet mode."""

    def __init__(self, use_cache: bool = True, allow_synthetic: bool = False):
        self.use_cache = use_cache
        self.allow_synthetic = allow_synthetic

    def load_base(self, symbol: str, interval: str, period: Optional[str] = None,
                  start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        return fetch(symbol, interval=interval, period=period, start=start, end=end,
                     use_cache=self.use_cache, allow_synthetic=self.allow_synthetic)


class ParquetProvider(DataProvider):
    """Read local all-symbol parquet bars produced from Zerodha/Kite 1m history.

    Expected files under ``data_dir``:
      all_5m.parquet, all_15m.parquet, all_60m.parquet, all_180m.parquet,
      all_1D.parquet, all_1W.parquet

    Files may contain either a ``date``/``ts`` column or a datetime index. If a ``symbol`` column
    exists, only the requested symbol is selected. Symbols are matched both exactly and without
    exchange suffixes, so ``TCS`` and ``TCS.NS`` resolve to the same stored rows.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir).expanduser()

    @staticmethod
    def _file_key(tf: str) -> str:
        key = tf.replace("min", "m")
        return {
            "base": "5m",
            "5m": "5m",
            "15m": "15m",
            "60m": "60m",
            "1h": "60m",
            "180m": "180m",
            "3h": "180m",
            "1d": "1D",
            "1D": "1D",
            "1w": "1W",
            "1W": "1W",
        }.get(key, key)

    def _path_for_tf(self, tf: str) -> Path:
        key = self._file_key(tf)
        candidates = [
            self.data_dir / f"all_{key}.parquet",
            self.data_dir / f"{key}.parquet",
            self.data_dir / key / "all.parquet",
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(f"missing parquet for timeframe {tf}; tried: "
                                + ", ".join(str(p) for p in candidates))

    def _read_symbol_tf(self, symbol: str, tf: str) -> pd.DataFrame:
        path = self._path_for_tf(tf)
        raw = pd.read_parquet(path)
        raw = raw.rename(columns={c: str(c).lower() for c in raw.columns})
        if "symbol" in raw.columns:
            wanted = _normalise_symbol_key(symbol)
            keys = raw["symbol"].astype(str).map(_normalise_symbol_key)
            raw = raw.loc[keys == wanted].copy()
        if raw.empty:
            raise FileNotFoundError(f"{symbol}: no rows in {path.name}")
        return _normalise_parquet_ohlcv(raw, symbol)

    def load_base(self, symbol: str, interval: str, period: Optional[str] = None,
                  start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        df = self._read_symbol_tf(symbol, interval)
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        if df.empty:
            raise FileNotFoundError(f"{symbol}: no parquet bars after date filters")
        return df

    def load_timeframes(self, symbol: str, base_tf: str,
                        higher_tfs: Iterable[str],
                        period: Optional[str] = None,
                        start: Optional[str] = None,
                        end: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {
            "base": self.load_base(symbol, base_tf, period=period, start=start, end=end)
        }
        for tf in higher_tfs:
            try:
                df = self._read_symbol_tf(symbol, tf)
                if start:
                    df = df[df.index >= pd.Timestamp(start)]
                if end:
                    df = df[df.index <= pd.Timestamp(end)]
                out[tf] = df
            except FileNotFoundError:
                # If a higher TF is not materialized yet, derive it from base locally.
                out[tf] = resample(out["base"], tf)
        return out


def _validate_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Sanity-check bars before they hit pool detection. Drops rows that violate OHLC
    invariants (high >= max(o,c), low <= min(o,c)) and reports counts. Doesn't drop volume=0
    rows because some sessions legitimately have low volume on illiquid times — we just
    log if too many are zero."""
    n_in = len(df)
    if n_in == 0:
        return df

    # OHLC invariants
    bad_high = (df["high"] < df[["open", "close"]].max(axis=1)) | (df["high"] < df["low"])
    bad_low = (df["low"] > df[["open", "close"]].min(axis=1))
    bad_mask = bad_high | bad_low
    n_bad = int(bad_mask.sum())
    if n_bad > 0:
        print(f"[data] {symbol}: dropping {n_bad} bars with broken OHLC invariants")
        df = df.loc[~bad_mask]

    # Volume sanity (warn only)
    if "volume" in df.columns:
        zero_vol = int((df["volume"] <= 0).sum())
        if zero_vol > 0 and zero_vol > 0.05 * len(df):
            print(f"[data] {symbol}: warning — {zero_vol} / {len(df)} bars have zero volume")

    # Minimum bar count
    if len(df) < 200:
        print(f"[data] {symbol}: warning — only {len(df)} bars after cleaning "
              f"(was {n_in}); models may be unstable")

    return df


# Map of higher-TF strings → pandas resample rule.
_TF_RULE = {
    "15m": "15min", "15min": "15min",
    "30m": "30min", "30min": "30min",
    "60m": "60min", "1h": "60min", "60min": "60min",
    "180m": "180min", "3h": "180min", "180min": "180min",
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
