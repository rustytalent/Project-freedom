"""Top-constituent state builder for index manipulation research.

This module turns equity/index bars into the constituent-level rows
expected by :mod:`liqpool.research.manipulation_atlas`.

It is deliberately causal:

* session fields are computed inside each IST trading date,
* previous-session AVWAP is shifted from the last completed session,
* historical position uses prior bars only,
* forward returns are attached only as research labels after the state
  has already been classified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from liqpool.warehouse import WarehouseFileMissing, WarehouseReader

from .state_dataset import (
    StateDatasetConfig,
    build_manipulation_state_frame,
    summarize_state_frame,
)


DEFAULT_FORWARD_BARS = (3, 6, 12, 36, 60)


@dataclass(frozen=True)
class IndexStateBuildConfig:
    """Config for building top-constituent manipulation-state datasets."""

    symbols: tuple[str, ...]
    weights: Mapping[str, float] = field(default_factory=dict)
    index: str = "NIFTY50"
    tf: str = "5m"
    atr_window: int = 14
    historical_window_bars: int = 1500
    min_history_bars: int = 100
    forward_bars: tuple[int, ...] = DEFAULT_FORWARD_BARS
    min_constituents: int = 5


def parse_weights_text(text: str | None) -> dict[str, float]:
    """Parse ``SYMBOL=weight,SYMBOL2=weight`` into a mapping."""

    if not text:
        return {}
    out: dict[str, float] = {}
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"bad weight entry {piece!r}; expected SYMBOL=weight")
        symbol, value = piece.split("=", 1)
        out[symbol.strip().upper()] = float(value.strip())
    return out


def read_weights_csv(path: str | None) -> dict[str, float]:
    """Read a CSV with ``symbol`` and ``weight`` columns."""

    if not path:
        return {}
    df = pd.read_csv(path)
    missing = {"symbol", "weight"} - set(df.columns)
    if missing:
        raise ValueError(f"weights CSV missing columns: {sorted(missing)}")
    return {
        str(row["symbol"]).strip().upper(): float(row["weight"])
        for _, row in df.iterrows()
    }


def build_index_manipulation_dataset(
    reader: WarehouseReader,
    cfg: IndexStateBuildConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Return ``(constituent_rows, state_rows, summary)``.

    ``constituent_rows`` is one row per symbol per timestamp.
    ``state_rows`` is one row per timestamp with the atlas label and
    forward index outcomes attached.
    """

    if not cfg.symbols:
        raise ValueError("at least one symbol is required")

    index_frame = _load_index_frame(reader, cfg)
    index_features = _prepare_index_features(index_frame)

    parts: list[pd.DataFrame] = []
    missing_symbols: list[str] = []
    weights = _normalise_weights(cfg.symbols, cfg.weights)

    for symbol in cfg.symbols:
        try:
            bars, _ = reader.load_resampled(symbol, cfg.tf, normalize_utc=True)
        except WarehouseFileMissing:
            missing_symbols.append(symbol)
            continue
        prepared = _prepare_constituent_frame(
            bars,
            symbol=symbol,
            weight=weights[symbol],
            atr_window=cfg.atr_window,
            historical_window_bars=cfg.historical_window_bars,
            min_history_bars=cfg.min_history_bars,
        )
        parts.append(prepared)

    if not parts:
        raise ValueError("no constituent bars could be loaded")

    constituent_rows = pd.concat(parts, ignore_index=True).sort_values(
        ["timestamp", "symbol"]
    )
    constituent_rows = _attach_index_context(constituent_rows, index_features)
    constituent_rows = _attach_breadth(constituent_rows)

    states = build_manipulation_state_frame(
        constituent_rows,
        StateDatasetConfig(group_cols=("ts",), min_constituents=cfg.min_constituents),
    )
    states = _attach_forward_returns(states, index_features, cfg.forward_bars)

    summary = summarize_state_frame(states)
    summary.update(
        {
            "symbols_requested": list(cfg.symbols),
            "symbols_loaded": sorted(constituent_rows["symbol"].unique().tolist()),
            "symbols_missing": missing_symbols,
            "index": cfg.index,
            "tf": cfg.tf,
            "forward_bars": list(cfg.forward_bars),
        }
    )
    return constituent_rows, states, summary


def _normalise_weights(symbols: Sequence[str], raw: Mapping[str, float]) -> dict[str, float]:
    upper_symbols = [s.upper() for s in symbols]
    if not raw:
        equal = 1.0 / len(upper_symbols)
        return {symbol: equal for symbol in upper_symbols}

    weights = {str(k).strip().upper(): float(v) for k, v in raw.items()}
    missing = [symbol for symbol in upper_symbols if symbol not in weights]
    if missing:
        positive_total = sum(v for v in weights.values() if v > 0)
        fallback = positive_total / max(len(upper_symbols), 1)
        for symbol in missing:
            weights[symbol] = fallback
    total = sum(max(0.0, weights[symbol]) for symbol in upper_symbols)
    if total <= 0:
        equal = 1.0 / len(upper_symbols)
        return {symbol: equal for symbol in upper_symbols}
    return {symbol: max(0.0, weights[symbol]) / total for symbol in upper_symbols}


def _load_index_frame(reader: WarehouseReader, cfg: IndexStateBuildConfig) -> pd.DataFrame:
    try:
        index_frame, _ = reader.load_spot(cfg.index, "5min", normalize_utc=True)
    except WarehouseFileMissing:
        return pd.DataFrame()
    return index_frame


def _to_ist_timestamp(ts: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(ts)
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize("UTC")
    else:
        parsed = parsed.dt.tz_convert("UTC")
    return parsed.dt.tz_convert("Asia/Kolkata")


def _add_session_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    ist = _to_ist_timestamp(out["timestamp"])
    out["timestamp_ist"] = ist.dt.tz_localize(None)
    out["trading_date_ist"] = ist.dt.strftime("%Y-%m-%d")
    out["minute_of_day_ist"] = ist.dt.hour * 60 + ist.dt.minute
    return out


def _prepare_constituent_frame(
    bars: pd.DataFrame,
    *,
    symbol: str,
    weight: float,
    atr_window: int,
    historical_window_bars: int,
    min_history_bars: int,
) -> pd.DataFrame:
    df = _add_session_columns(bars).sort_values("timestamp").reset_index(drop=True)
    df["ts"] = df["timestamp"]
    df["symbol"] = symbol.upper()
    df["weight"] = float(weight)
    df["atr"] = _atr(df, atr_window)

    session = df.groupby("trading_date_ist", sort=True)
    session_open = session["open"].transform("first")
    prev_close_by_date = session["close"].last().shift(1)
    df["prev_session_close"] = df["trading_date_ist"].map(prev_close_by_date)
    df["return_pct"] = _pct_change(df["close"], session_open)
    df["gap_pct"] = _pct_change(session_open, df["prev_session_close"])

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
    df["_pv"] = typical * volume
    cum_pv = df.groupby("trading_date_ist")["_pv"].cumsum()
    cum_vol = volume.groupby(df["trading_date_ist"]).cumsum()
    df["today_avwap"] = np.where(cum_vol > 0, cum_pv / cum_vol, df["close"])
    session_vwap_last = df.groupby("trading_date_ist")["today_avwap"].last().shift(1)
    df["prev_session_avwap"] = df["trading_date_ist"].map(session_vwap_last)

    atr = df["atr"].replace(0, np.nan)
    df["today_avwap_dist_atr"] = ((df["close"] - df["today_avwap"]) / atr).replace(
        [np.inf, -np.inf], np.nan
    )
    df["prev_session_avwap_dist_atr"] = (
        (df["close"] - df["prev_session_avwap"]) / atr
    ).replace([np.inf, -np.inf], np.nan)
    df["historical_position_pct"] = _historical_range_position(
        df["close"],
        window=historical_window_bars,
        min_periods=min_history_bars,
    )

    cols = [
        "timestamp",
        "ts",
        "timestamp_ist",
        "trading_date_ist",
        "minute_of_day_ist",
        "symbol",
        "weight",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "atr",
        "return_pct",
        "gap_pct",
        "today_avwap",
        "prev_session_avwap",
        "today_avwap_dist_atr",
        "prev_session_avwap_dist_atr",
        "historical_position_pct",
    ]
    return df[cols]


def _prepare_index_features(index_frame: pd.DataFrame) -> pd.DataFrame:
    if index_frame.empty:
        return pd.DataFrame(columns=["timestamp", "index_close", "index_return_pct"])
    df = _add_session_columns(index_frame).sort_values("timestamp").reset_index(drop=True)
    session_open = df.groupby("trading_date_ist")["open"].transform("first")
    df["index_return_pct"] = _pct_change(df["close"], session_open)
    return df[["timestamp", "timestamp_ist", "trading_date_ist", "index_return_pct", "close"]].rename(
        columns={"close": "index_close"}
    )


def _attach_index_context(rows: pd.DataFrame, index_features: pd.DataFrame) -> pd.DataFrame:
    if index_features.empty:
        rows = rows.copy()
        rows["index_return_pct"] = np.nan
        rows["index_close"] = np.nan
        return rows
    merged = pd.merge_asof(
        rows.sort_values("timestamp"),
        index_features[["timestamp", "index_return_pct", "index_close"]].sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=7),
    )
    return merged


def _attach_breadth(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    breadth = (
        out.assign(_positive=out["return_pct"] > 0)
        .groupby("timestamp")["_positive"]
        .mean()
    )
    weighted_return = (
        out.assign(_weighted=out["return_pct"] * out["weight"])
        .groupby("timestamp")["_weighted"]
        .sum()
    )
    out["breadth_positive_frac"] = out["timestamp"].map(breadth)
    out["weighted_constituent_return_pct"] = out["timestamp"].map(weighted_return)
    out["index_return_pct"] = out["index_return_pct"].fillna(
        out["weighted_constituent_return_pct"]
    )
    return out


def _attach_forward_returns(
    states: pd.DataFrame,
    index_features: pd.DataFrame,
    horizons: Sequence[int],
) -> pd.DataFrame:
    if states.empty or index_features.empty:
        return states
    out = pd.merge_asof(
        states.sort_values("ts"),
        index_features[["timestamp", "index_close"]].rename(columns={"timestamp": "ts"}).sort_values("ts"),
        on="ts",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=7),
    )
    index_by_ts = index_features.sort_values("timestamp").reset_index(drop=True)
    closes = pd.to_numeric(index_by_ts["index_close"], errors="coerce")
    for horizon in horizons:
        future = closes.shift(-int(horizon))
        labels = pd.DataFrame(
            {
                "ts": index_by_ts["timestamp"],
                f"forward_index_return_{horizon}b_pct": _pct_change(future, closes),
            }
        )
        out = pd.merge_asof(
            out.sort_values("ts"),
            labels.sort_values("ts"),
            on="ts",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=7),
        )
    return out


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    fallback = close.abs() * 0.001
    atr = tr.rolling(window, min_periods=1).mean()
    atr = atr.fillna(fallback)
    return atr.mask(atr <= 0, fallback)


def _pct_change(new: pd.Series, old: pd.Series) -> pd.Series:
    old = pd.to_numeric(old, errors="coerce").replace(0, np.nan)
    new = pd.to_numeric(new, errors="coerce")
    return ((new / old) - 1.0) * 100.0


def _historical_range_position(
    close: pd.Series,
    *,
    window: int,
    min_periods: int,
) -> pd.Series:
    prior = pd.to_numeric(close, errors="coerce").shift(1)
    rolling_min = prior.rolling(window, min_periods=min_periods).min()
    rolling_max = prior.rolling(window, min_periods=min_periods).max()
    span = (rolling_max - rolling_min).replace(0, np.nan)
    pos = ((pd.to_numeric(close, errors="coerce") - rolling_min) / span).clip(0, 1)
    return pos.fillna(0.5)


__all__ = [
    "DEFAULT_FORWARD_BARS",
    "IndexStateBuildConfig",
    "build_index_manipulation_dataset",
    "parse_weights_text",
    "read_weights_csv",
]
