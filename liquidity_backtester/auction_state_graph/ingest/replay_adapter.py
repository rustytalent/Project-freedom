"""Replay data loading.

Supported inputs:
- tick CSV/parquet with timestamp + price/ltp/close columns
- OHLC CSV/parquet, which is expanded into deterministic intra-bar path ticks

The adapter is intentionally local-file only. Kite WebSocket is a future adapter,
not part of the replay-first MVP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..core.models import Tick

TIME_COLUMNS = ("ts", "timestamp", "datetime", "date", "time")
PRICE_COLUMNS = ("price", "ltp", "last_price", "close")


def load_replay_ticks(path: str | Path, symbol: str = "", max_rows: int | None = None) -> list[Tick]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = _read_frame(path)
    if max_rows is not None and max_rows > 0:
        df = df.tail(max_rows).copy()
    return list(_frame_to_ticks(df, symbol=symbol or path.stem))


def demo_ticks(symbol: str = "DEMO", periods: int = 1500, seed: int = 7) -> list[Tick]:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp.now(tz="Asia/Kolkata").floor("s") - pd.Timedelta(seconds=periods)
    price = 24000.0
    ce = 110.0
    pe = 108.0
    ticks: list[Tick] = []
    for i in range(periods):
        phase = np.sin(i / 65.0) * 0.9 + np.sin(i / 17.0) * 0.35
        shock = 0.0
        if i in {280, 620, 980, 1240}:
            shock = rng.normal(0, 6.0)
        step = phase + rng.normal(0, 1.15) + shock
        price += step
        ce = max(1.0, ce + max(step, 0) * 0.55 - max(-step, 0) * 0.20 + rng.normal(0, 0.45))
        pe = max(1.0, pe + max(-step, 0) * 0.55 - max(step, 0) * 0.20 + rng.normal(0, 0.45))
        spread = max(0.05, abs(rng.normal(0.12, 0.04)))
        ticks.append(
            Tick(
                ts=start + pd.Timedelta(seconds=i),
                price=float(price),
                volume=float(max(1.0, rng.normal(40, 15))),
                bid=float(price - spread),
                ask=float(price + spread),
                bid_qty=float(max(1.0, rng.normal(900, 250))),
                ask_qty=float(max(1.0, rng.normal(900, 250))),
                ce_price=float(ce),
                pe_price=float(pe),
                symbol=symbol,
                source_kind="demo_tick_replay",
            )
        )
    return ticks


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported replay file type: {path.suffix}")


def _frame_to_ticks(df: pd.DataFrame, symbol: str) -> Iterable[Tick]:
    if df.empty:
        return []
    df = df.copy()
    lower = {str(col).lower(): col for col in df.columns}
    ts_col = next((lower[name] for name in TIME_COLUMNS if name in lower), None)
    if ts_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            df["_ts"] = df.index
            ts_col = "_ts"
        else:
            raise ValueError("replay data needs ts/timestamp/datetime/date/time column or DatetimeIndex")
    df["_ts"] = pd.to_datetime(df[ts_col])

    has_ohlc = all(col in lower for col in ("open", "high", "low", "close"))
    if has_ohlc:
        return _ohlc_to_path_ticks(df, lower, symbol)

    price_col = next((lower[name] for name in PRICE_COLUMNS if name in lower), None)
    if price_col is None:
        raise ValueError("tick replay needs price/ltp/last_price/close column")
    return _tick_rows(df, lower, price_col, symbol)


def _tick_rows(df: pd.DataFrame, lower: dict[str, str], price_col: str, symbol: str) -> list[Tick]:
    rows: list[Tick] = []
    for data in df.to_dict("records"):
        rows.append(
            Tick(
                ts=pd.Timestamp(data["_ts"]),
                price=float(data[price_col]),
                volume=_maybe_float(data, lower, "volume", 0.0),
                bid=_maybe_float(data, lower, "bid"),
                ask=_maybe_float(data, lower, "ask"),
                bid_qty=_maybe_float(data, lower, "bid_qty"),
                ask_qty=_maybe_float(data, lower, "ask_qty"),
                ce_price=_maybe_float(data, lower, "ce_price"),
                pe_price=_maybe_float(data, lower, "pe_price"),
                symbol=symbol,
                source_kind="tick_replay",
            )
        )
    return rows


def _ohlc_to_path_ticks(df: pd.DataFrame, lower: dict[str, str], symbol: str) -> list[Tick]:
    rows: list[Tick] = []
    for record in df.to_dict("records"):
        ts = pd.Timestamp(record["_ts"])
        open_px = float(record[lower["open"]])
        high_px = float(record[lower["high"]])
        low_px = float(record[lower["low"]])
        close_px = float(record[lower["close"]])
        volume = float(record.get(lower.get("volume", ""), 0.0) or 0.0)
        path = _ohlc_path(open_px, high_px, low_px, close_px)
        step = pd.Timedelta(seconds=12)
        for i, px in enumerate(path):
            rows.append(
                Tick(
                    ts=ts + i * step,
                    price=float(px),
                    volume=volume / max(len(path), 1),
                    symbol=symbol,
                    source_kind="ohlc_path_replay",
                )
            )
    return rows


def _ohlc_path(open_px: float, high_px: float, low_px: float, close_px: float) -> list[float]:
    # A deterministic approximation. If close is above open, test low before high;
    # otherwise test high before low. This preserves OHLC without inventing intent.
    if close_px >= open_px:
        anchors = [open_px, low_px, high_px, close_px]
    else:
        anchors = [open_px, high_px, low_px, close_px]
    path: list[float] = []
    for a, b in zip(anchors, anchors[1:]):
        path.extend(np.linspace(a, b, 5, endpoint=False).tolist())
    path.append(close_px)
    return path


def _maybe_float(data: dict, lower: dict[str, str], key: str, default: float | None = None) -> float | None:
    col = lower.get(key)
    if col is None:
        return default
    value = data.get(col)
    if value is None or pd.isna(value):
        return default
    return float(value)
