"""Phase 4 execution simulator v2.

The v1 execution backtest is intentionally simple and 5-minute-bar based.  This
module keeps the same policy names but adds the pieces needed before strategy
research can be trusted:

- optional 1-minute stop/target path resolution,
- adverse-selection-aware limit fills,
- state-dependent slippage,
- itemised Zerodha intraday cost fields,
- and a pre-touch directional simulator hook for Track A.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .data import _normalise_parquet_ohlcv, _normalise_symbol_key
from .execution_backtest import (
    EXECUTION_MODES,
    _entry_for_mode,
    _headline_factor,
    _levels,
    _reaction_label,
    summarise_execution_trades,
)
from .indicators import atr
from .pools import Pool
from .sectors import sector_of
from .tester import PoolResult


FILL_POLICIES: Dict[str, Dict[str, float | bool | str]] = {
    "generous": {
        "description": "fills at the reference level when touched",
        "fill_buffer_atr": 0.0,
        "use_midpoint_fill": False,
    },
    "neutral": {
        "description": "requires price to trade through the level by 0.10 ATR",
        "fill_buffer_atr": 0.10,
        "use_midpoint_fill": False,
    },
    "conservative": {
        "description": "requires 0.15 ATR through-trade and worsens close-confirmed fills",
        "fill_buffer_atr": 0.15,
        "use_midpoint_fill": True,
    },
}


@dataclass(frozen=True)
class ExecutionV2Config:
    fill_policy: str = "neutral"
    use_1m_resolution: bool = False
    slippage_model: str = "state_dependent"
    base_slippage_bps: float = 2.0
    exchange: str = "NSE"
    quantity: int = 1

    def __post_init__(self) -> None:
        if self.fill_policy not in FILL_POLICIES:
            raise ValueError(f"unknown fill policy {self.fill_policy!r}")
        if self.slippage_model not in ("flat", "state_dependent"):
            raise ValueError(f"unknown slippage model {self.slippage_model!r}")


@dataclass
class ExecutionTradeV2:
    mode: str
    symbol: str
    sector: str
    side: str
    direction: str
    direction_sign: int
    pool_idx: int
    entry_at: str
    exit_at: str
    entry_reference: float
    exit_reference: float
    entry: float
    exit: float
    stop: float
    target: float
    quantity: int
    exit_reason: str
    bars_held: int
    gross_pnl: float
    statutory_cost: float
    slippage_cost: float
    total_cost: float
    net_pnl: float
    risk_inr: float
    net_r: float
    directional_net_r: float
    outcome: str
    reaction_label: str
    mae_atr: float
    mfe_atr: float
    score: float
    tf_count: int
    factor: str
    fill_policy: str
    slippage_model: str
    entry_slippage_bps: float
    exit_slippage_bps: float
    resolution_source: str
    entry_reason: str
    brokerage_buy: float
    brokerage_sell: float
    brokerage_total: float
    stt: float
    exchange_txn: float
    sebi: float
    stamp: float
    gst: float
    total_cost_bps: float
    breakeven_pct_move: float

    def to_dict(self) -> Dict:
        return asdict(self)


def _index_ts(df: pd.DataFrame) -> pd.DatetimeIndex:
    if "timestamp" in df.columns:
        return pd.DatetimeIndex(pd.to_datetime(df["timestamp"]))
    if "ts" in df.columns:
        return pd.DatetimeIndex(pd.to_datetime(df["ts"]))
    return pd.DatetimeIndex(df.index)


def _with_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" not in out.columns:
        out["timestamp"] = _index_ts(out)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out.sort_values("timestamp")


def session_phase(ts: pd.Timestamp) -> str:
    t = pd.Timestamp(ts).time()
    if time(9, 15) <= t < time(10, 0):
        return "open"
    if time(14, 30) <= t <= time(15, 30):
        return "close"
    return "mid"


def compute_slippage_bps(
    realized_vol_5m: float,
    distance_to_level_atr: float,
    session: str,
    base_bps: float = 2.0,
    k1_vol: float = 10.0,
    k2_distance: float = 1.0,
    k3_session: Optional[Dict[str, float]] = None,
) -> float:
    """State-dependent slippage in bps.

    ``realized_vol_5m`` should be ATR/price or another small fractional
    volatility estimate.  The distance term deliberately gets expensive near a
    level, where liquidity-event fills are most adverse-selected.
    """
    k3 = k3_session or {"open": 2.0, "mid": 1.0, "close": 1.5}
    dist = max(float(distance_to_level_atr), 0.05)
    slip = float(base_bps)
    slip += float(k1_vol) * max(float(realized_vol_5m), 0.0)
    slip += float(k2_distance) / dist
    slip *= float(k3.get(session, 1.0))
    return max(float(slip), 0.0)


def compute_zerodha_intraday_costs(
    buy_price: float,
    sell_price: float,
    quantity: int,
    exchange: str = "NSE",
) -> Dict[str, float]:
    """Itemised Zerodha equity intraday charges.

    Rates match Zerodha's charges page checked on 2026-05-27: intraday
    brokerage is min(0.03%, Rs 20) per order, STT is 0.025% on sell side,
    transaction charges are NSE 0.00307% / BSE 0.00375%, GST is 18% on
    brokerage + transaction + SEBI, SEBI is Rs 10 per crore, and stamp duty is
    0.003% on buy side.
    """
    qty = max(int(quantity), 1)
    buy_turnover = abs(float(buy_price)) * qty
    sell_turnover = abs(float(sell_price)) * qty
    total_turnover = buy_turnover + sell_turnover
    exch = str(exchange).upper()
    exchange_rate = 0.0000375 if exch == "BSE" else 0.0000307

    brokerage_buy = min(0.0003 * buy_turnover, 20.0)
    brokerage_sell = min(0.0003 * sell_turnover, 20.0)
    brokerage_total = brokerage_buy + brokerage_sell
    stt = 0.00025 * sell_turnover
    exchange_txn = exchange_rate * total_turnover
    sebi = (10.0 / 10_000_000.0) * total_turnover
    stamp = 0.00003 * buy_turnover
    gst = 0.18 * (brokerage_total + exchange_txn + sebi)
    total_cost = brokerage_total + stt + exchange_txn + sebi + stamp + gst

    return {
        "brokerage_buy": float(brokerage_buy),
        "brokerage_sell": float(brokerage_sell),
        "brokerage_total": float(brokerage_total),
        "stt": float(stt),
        "exchange_txn": float(exchange_txn),
        "sebi": float(sebi),
        "stamp": float(stamp),
        "gst": float(gst),
        "total_cost_inr": float(total_cost),
        "total_cost_bps": float(total_cost / max(total_turnover, 1e-9) * 10000.0),
        "breakeven_pct_move": float(total_cost / max(buy_turnover, 1e-9) * 100.0),
    }


def _slipped_price(price: float, direction: str, leg: str, bps: float) -> float:
    mult = float(bps) / 10000.0
    if direction == "UP":
        return float(price) * (1.0 + mult) if leg == "entry" else float(price) * (1.0 - mult)
    return float(price) * (1.0 - mult) if leg == "entry" else float(price) * (1.0 + mult)


def _buy_sell_prices(direction: str, entry: float, exit_: float) -> Tuple[float, float]:
    if direction == "UP":
        return float(entry), float(exit_)
    return float(exit_), float(entry)


def resolve_intrabar_path(
    entry_timestamp: pd.Timestamp,
    entry_price: float,
    stop_price: float,
    target_price: float,
    symbol: str,
    direction: str,
    intrabar_data_1m: pd.DataFrame,
    exit_deadline: Optional[pd.Timestamp] = None,
    conservative_same_bar: bool = True,
) -> Dict:
    """Resolve stop/target order with 1-minute bars.

    If stop and target both sit inside the same 1-minute candle, the default is
    conservative and records the stop first.
    """
    if intrabar_data_1m is None or intrabar_data_1m.empty:
        return {
            "resolution": "no_data",
            "timestamp": None,
            "price": None,
            "intrabar_path": [],
            "symbol": symbol,
        }
    bars = _with_timestamp_column(intrabar_data_1m)
    start_ts = pd.Timestamp(entry_timestamp)
    mask = bars["timestamp"] >= start_ts
    if exit_deadline is not None:
        mask &= bars["timestamp"] <= pd.Timestamp(exit_deadline)
    bars = bars.loc[mask]
    path = bars["close"].astype(float).tolist() if "close" in bars.columns else []

    for _, row in bars.iterrows():
        low = float(row["low"])
        high = float(row["high"])
        ts = row["timestamp"]
        if direction == "UP":
            stop_hit = low <= float(stop_price)
            target_hit = high >= float(target_price)
        else:
            stop_hit = high >= float(stop_price)
            target_hit = low <= float(target_price)
        if stop_hit and target_hit and conservative_same_bar:
            return {"resolution": "stop_hit", "timestamp": ts, "price": float(stop_price),
                    "intrabar_path": path, "symbol": symbol}
        if stop_hit:
            return {"resolution": "stop_hit", "timestamp": ts, "price": float(stop_price),
                    "intrabar_path": path, "symbol": symbol}
        if target_hit:
            return {"resolution": "target_hit", "timestamp": ts, "price": float(target_price),
                    "intrabar_path": path, "symbol": symbol}

    return {
        "resolution": "no_hit",
        "timestamp": None,
        "price": None,
        "intrabar_path": path,
        "symbol": symbol,
    }


def _load_symbol_1m(symbol: str, raw_1m_dir: str | Path | None) -> Optional[pd.DataFrame]:
    if not raw_1m_dir:
        return None
    root = Path(raw_1m_dir).expanduser()
    if not root.exists():
        return None
    key = _normalise_symbol_key(symbol)
    candidates = [
        root / f"{key}_1m.parquet",
        root / f"{key}.parquet",
        root / f"{symbol}_1m.parquet",
        root / f"{symbol}.parquet",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return None
    raw = pd.read_parquet(path)
    if "symbol" not in raw.columns:
        raw["symbol"] = key
    return _normalise_parquet_ohlcv(raw, symbol, timeframe="1m")


def _entry_with_fill_policy(
    mode: str,
    pool: Pool,
    df: pd.DataFrame,
    entry_idx: int,
    entry_reference: float,
    atr_value: float,
    fill_policy: str,
) -> Optional[Tuple[float, str]]:
    if mode != "blind_limit":
        return float(entry_reference), "next_bar_confirmation"

    policy = FILL_POLICIES[fill_policy]
    buffer_atr = float(policy["fill_buffer_atr"])
    row = df.iloc[entry_idx]
    buffer = buffer_atr * max(float(atr_value), 1e-9)

    if pool.side == "low":
        through_ok = float(row["low"]) <= float(entry_reference) - buffer
    else:
        through_ok = float(row["high"]) >= float(entry_reference) + buffer
    if not through_ok:
        return None

    entry_price = float(entry_reference)
    reason = "touch"
    if bool(policy.get("use_midpoint_fill")):
        close = float(row["close"])
        midpoint = (entry_price + close) / 2.0
        if pool.side == "low":
            entry_price = max(entry_price, midpoint)
        else:
            entry_price = min(entry_price, midpoint)
        reason += "_midpoint"
    return entry_price, reason


def _exit_5m_fallback(
    pool: Pool,
    df: pd.DataFrame,
    entry_idx: int,
    end: int,
    stop: float,
    target: float,
    time_stop_bars: int,
) -> Tuple[int, float, str]:
    last_idx = min(end - 1, entry_idx + max(1, int(time_stop_bars)))
    for j in range(entry_idx + 1, last_idx + 1):
        row = df.iloc[j]
        low = float(row["low"])
        high = float(row["high"])
        if pool.side == "low":
            if low <= stop:
                return j, float(stop), "stop"
            if high >= target:
                return j, float(target), "target"
        else:
            if high >= stop:
                return j, float(stop), "stop"
            if low <= target:
                return j, float(target), "target"
    return last_idx, float(df["close"].iloc[last_idx]), "time_exit"


def simulate_pool_trade_v2(
    *,
    mode: str,
    symbol: str,
    sector: str,
    df_base: pd.DataFrame,
    pool: Pool,
    result: PoolResult,
    cfg: Config,
    v2_cfg: ExecutionV2Config,
    intrabar_1m: Optional[pd.DataFrame] = None,
    time_stop_bars: Optional[int] = None,
) -> Optional[ExecutionTradeV2]:
    if mode not in EXECUTION_MODES:
        raise ValueError(f"unknown execution mode {mode!r}")
    if df_base.empty or result.outcome == "horizon_insufficient":
        return None

    idx = pd.DatetimeIndex(df_base.index)
    start = int(np.searchsorted(idx.values, np.datetime64(pool.available_at), side="left"))
    end = min(start + cfg.test_horizon_bars, len(df_base))
    if end <= start + 1:
        return None

    atr_values = atr(df_base, cfg.detect.atr_period).bfill()
    entry_signal = _entry_for_mode(mode, pool, df_base, atr_values, start, end, cfg)
    if entry_signal is None:
        return None
    signal_idx, entry_reference, entry_reason = entry_signal

    entry_idx = signal_idx
    if mode != "blind_limit":
        entry_idx = signal_idx + 1
        if entry_idx >= end:
            return None
        entry_reference = float(df_base["open"].iloc[entry_idx])
    atr_at_entry = float(atr_values.iloc[entry_idx])
    fill = _entry_with_fill_policy(
        mode, pool, df_base, entry_idx, float(entry_reference), atr_at_entry, v2_cfg.fill_policy,
    )
    if fill is None:
        return None
    entry_price_ref, fill_reason = fill
    if entry_reason and entry_reason != "touch":
        fill_reason = f"{entry_reason}:{fill_reason}"

    stop, target = _levels(pool, atr_at_entry)
    max_hold = time_stop_bars or cfg.respect_within_bars
    last_idx = min(end - 1, entry_idx + max(1, int(max_hold)))
    direction = "UP" if pool.side == "low" else "DOWN"
    direction_sign = 1 if direction == "UP" else -1
    if direction == "UP" and not (stop < entry_price_ref < target):
        return None
    if direction == "DOWN" and not (target < entry_price_ref < stop):
        return None

    resolution_source = "5m"
    if v2_cfg.use_1m_resolution and intrabar_1m is not None and not intrabar_1m.empty:
        deadline = idx[last_idx]
        resolved = resolve_intrabar_path(
            idx[entry_idx], entry_price_ref, stop, target, symbol, direction, intrabar_1m,
            exit_deadline=deadline,
        )
        if resolved["resolution"] in ("stop_hit", "target_hit"):
            exit_price_ref = float(resolved["price"])
            exit_ts = pd.Timestamp(resolved["timestamp"])
            exit_idx = int(np.searchsorted(idx.values, np.datetime64(exit_ts), side="right") - 1)
            exit_idx = min(max(exit_idx, entry_idx), last_idx)
            exit_reason = "stop" if resolved["resolution"] == "stop_hit" else "target"
            resolution_source = "1m"
        else:
            exit_idx = last_idx
            exit_price_ref = float(df_base["close"].iloc[last_idx])
            exit_reason = "time_exit"
            resolution_source = "1m_no_hit"
    else:
        exit_idx, exit_price_ref, exit_reason = _exit_5m_fallback(
            pool, df_base, entry_idx, end, stop, target, int(max_hold),
        )

    dist_atr = abs(pool.mid - entry_price_ref) / max(atr_at_entry, 1e-9)
    vol_frac = atr_at_entry / max(abs(entry_price_ref), 1e-9)
    if v2_cfg.slippage_model == "flat":
        entry_slip = exit_slip = max(float(v2_cfg.base_slippage_bps), 0.0)
    else:
        entry_slip = compute_slippage_bps(
            vol_frac, dist_atr, session_phase(idx[entry_idx]), base_bps=v2_cfg.base_slippage_bps,
        )
        exit_slip = compute_slippage_bps(
            vol_frac, dist_atr, session_phase(idx[exit_idx]), base_bps=v2_cfg.base_slippage_bps,
        )

    entry_price = _slipped_price(entry_price_ref, direction, "entry", entry_slip)
    exit_price = _slipped_price(exit_price_ref, direction, "exit", exit_slip)
    qty = max(int(v2_cfg.quantity), 1)

    if direction == "UP":
        gross_per_share = float(exit_price_ref - entry_price_ref)
        slippage_per_share = max(0.0, (entry_price - entry_price_ref) + (exit_price_ref - exit_price))
        risk_per_share = max(float(entry_price - stop), 1e-9)
        side = "buy"
    else:
        gross_per_share = float(entry_price_ref - exit_price_ref)
        slippage_per_share = max(0.0, (entry_price_ref - entry_price) + (exit_price - exit_price_ref))
        risk_per_share = max(float(stop - entry_price), 1e-9)
        side = "sell"
    gross = gross_per_share * qty
    buy_price, sell_price = _buy_sell_prices(direction, entry_price, exit_price)
    costs = compute_zerodha_intraday_costs(buy_price, sell_price, qty, exchange=v2_cfg.exchange)
    statutory = float(costs["total_cost_inr"])
    slippage_cost = float(slippage_per_share * qty)
    total_cost = statutory + slippage_cost
    net = gross - total_cost
    risk = risk_per_share * qty
    net_r = float(net / max(risk, 1e-9))
    ref_turnover = (abs(entry_price_ref) + abs(exit_price_ref)) * qty
    ref_buy = entry_price_ref if direction == "UP" else exit_price_ref

    return ExecutionTradeV2(
        mode=mode,
        symbol=symbol,
        sector=sector,
        side=side,
        direction=direction,
        direction_sign=direction_sign,
        pool_idx=int(result.pool_idx),
        entry_at=str(idx[entry_idx]),
        exit_at=str(idx[exit_idx]),
        entry_reference=float(entry_price_ref),
        exit_reference=float(exit_price_ref),
        entry=float(entry_price),
        exit=float(exit_price),
        stop=float(stop),
        target=float(target),
        quantity=qty,
        exit_reason=exit_reason,
        bars_held=int(exit_idx - entry_idx),
        gross_pnl=float(gross),
        statutory_cost=statutory,
        slippage_cost=float(slippage_cost),
        total_cost=total_cost,
        net_pnl=float(net),
        risk_inr=float(risk),
        net_r=net_r,
        directional_net_r=float(abs(net_r) * direction_sign),
        outcome=result.outcome,
        reaction_label=_reaction_label(result.outcome),
        mae_atr=float(result.max_excursion_through),
        mfe_atr=float(result.reaction_atr),
        score=float(pool.score),
        tf_count=int(len(set(pool.tfs))),
        factor=_headline_factor(pool),
        fill_policy=v2_cfg.fill_policy,
        slippage_model=v2_cfg.slippage_model,
        entry_slippage_bps=float(entry_slip),
        exit_slippage_bps=float(exit_slip),
        resolution_source=resolution_source,
        entry_reason=fill_reason,
        brokerage_buy=float(costs["brokerage_buy"]),
        brokerage_sell=float(costs["brokerage_sell"]),
        brokerage_total=float(costs["brokerage_total"]),
        stt=float(costs["stt"]),
        exchange_txn=float(costs["exchange_txn"]),
        sebi=float(costs["sebi"]),
        stamp=float(costs["stamp"]),
        gst=float(costs["gst"]),
        total_cost_bps=float(total_cost / max(ref_turnover, 1e-9) * 10000.0),
        breakeven_pct_move=float(total_cost / max(abs(ref_buy) * qty, 1e-9) * 100.0),
    )


def simulate_execution_modes_v2(
    *,
    df_base: pd.DataFrame,
    pools: Sequence[Pool],
    results: Sequence[PoolResult],
    symbol: str,
    sector: str,
    cfg: Config,
    v2_cfg: ExecutionV2Config,
    modes: Iterable[str] = EXECUTION_MODES,
    intrabar_1m: Optional[pd.DataFrame] = None,
    time_stop_bars: Optional[int] = None,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for mode in modes:
        for pool, result in zip(pools, results):
            trade = simulate_pool_trade_v2(
                mode=mode,
                symbol=symbol,
                sector=sector,
                df_base=df_base,
                pool=pool,
                result=result,
                cfg=cfg,
                v2_cfg=v2_cfg,
                intrabar_1m=intrabar_1m,
                time_stop_bars=time_stop_bars,
            )
            if trade is not None:
                rows.append(trade.to_dict())
    return pd.DataFrame(rows)


def build_execution_backtest_v2_for_report(
    report,
    cfg: Config,
    v2_cfg: ExecutionV2Config,
    raw_1m_dir: str | Path | None = None,
    modes: Iterable[str] = EXECUTION_MODES,
    split: str = "oos",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for symbol, ad in report.assets.items():
        if split == "train":
            pools = ad.walkforward.train_pools_for_ml
            results = ad.walkforward.train_results_for_ml
        else:
            pools = ad.walkforward.oos_pools
            results = ad.walkforward.oos_results
        asset_cfg = getattr(ad, "final_cfg", None) or cfg
        intrabar = _load_symbol_1m(symbol, raw_1m_dir) if v2_cfg.use_1m_resolution else None
        frames.append(simulate_execution_modes_v2(
            df_base=ad.base_df,
            pools=pools,
            results=results,
            symbol=symbol,
            sector=sector_of(symbol),
            cfg=asset_cfg,
            v2_cfg=v2_cfg,
            modes=modes,
            intrabar_1m=intrabar,
        ))
    trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return trades, summarise_execution_trades(trades)


def compare_execution_summaries_v1_v2(v1: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    if v1 is None or v1.empty or v2 is None or v2.empty:
        return pd.DataFrame(columns=[
            "mode", "v1_trades", "v2_trades", "delta_trades",
            "v1_net_expectancy_r", "v2_net_expectancy_r", "delta_net_expectancy_r",
            "v1_profit_factor", "v2_profit_factor", "delta_profit_factor",
        ])
    left = v1.add_prefix("v1_").rename(columns={"v1_mode": "mode"})
    right = v2.add_prefix("v2_").rename(columns={"v2_mode": "mode"})
    merged = left.merge(right, on="mode", how="outer")
    merged["delta_trades"] = merged.get("v2_trades", 0).fillna(0) - merged.get("v1_trades", 0).fillna(0)
    merged["delta_net_expectancy_r"] = (
        merged.get("v2_net_expectancy_r", 0).fillna(0)
        - merged.get("v1_net_expectancy_r", 0).fillna(0)
    )
    merged["delta_profit_factor"] = (
        merged.get("v2_profit_factor", 0).fillna(0)
        - merged.get("v1_profit_factor", 0).fillna(0)
    )
    cols = [
        "mode", "v1_trades", "v2_trades", "delta_trades",
        "v1_net_expectancy_r", "v2_net_expectancy_r", "delta_net_expectancy_r",
        "v1_profit_factor", "v2_profit_factor", "delta_profit_factor",
    ]
    return merged[[c for c in cols if c in merged.columns]]


class PreTouchDirectionalSimulator:
    """Simulates entering before touch and targeting the pool approach move."""

    def simulate(
        self,
        *,
        pool_zone: Tuple[float, float],
        pool_side: str,
        entry_timestamp: pd.Timestamp,
        entry_price: float,
        bars_5m: pd.DataFrame,
        bars_1m: Optional[pd.DataFrame],
        atr_at_entry: float,
        stop_atr_mult: float = 1.5,
        target_fraction: float = 0.8,
        max_hold_bars: int = 78,
        quantity: int = 1,
        exchange: str = "NSE",
        slippage_bps: float = 2.0,
    ) -> Dict:
        low, high = float(pool_zone[0]), float(pool_zone[1])
        entry = float(entry_price)
        atr_val = max(float(atr_at_entry), 1e-9)
        if pool_side == "above":
            direction = "UP"
            target = entry + float(target_fraction) * max(low - entry, 0.0)
            stop = entry - float(stop_atr_mult) * atr_val
        else:
            direction = "DOWN"
            target = entry - float(target_fraction) * max(entry - high, 0.0)
            stop = entry + float(stop_atr_mult) * atr_val

        bars = bars_5m.copy()
        deadline = pd.Timestamp(entry_timestamp) + pd.Timedelta(minutes=5 * max_hold_bars)
        if bars_1m is not None and not bars_1m.empty:
            resolved = resolve_intrabar_path(
                entry_timestamp, entry, stop, target, "PRETOUCH", direction, bars_1m,
                exit_deadline=deadline,
            )
            if resolved["resolution"] in ("stop_hit", "target_hit"):
                exit_ref = float(resolved["price"])
                exit_reason = "stop" if resolved["resolution"] == "stop_hit" else "target"
                exit_ts = pd.Timestamp(resolved["timestamp"])
            else:
                after = bars.loc[(pd.DatetimeIndex(bars.index) >= pd.Timestamp(entry_timestamp))
                                 & (pd.DatetimeIndex(bars.index) <= deadline)]
                if after.empty:
                    after = bars.tail(1)
                exit_ref = float(after["close"].iloc[-1])
                exit_reason = "time_exit"
                exit_ts = pd.Timestamp(after.index[-1])
        else:
            after = bars.loc[(pd.DatetimeIndex(bars.index) >= pd.Timestamp(entry_timestamp))
                             & (pd.DatetimeIndex(bars.index) <= deadline)]
            if after.empty:
                after = bars.tail(1)
            exit_ref = float(after["close"].iloc[-1])
            exit_reason = "time_exit"
            exit_ts = pd.Timestamp(after.index[-1])

        entry_exec = _slipped_price(entry, direction, "entry", slippage_bps)
        exit_exec = _slipped_price(exit_ref, direction, "exit", slippage_bps)
        buy_price, sell_price = _buy_sell_prices(direction, entry_exec, exit_exec)
        costs = compute_zerodha_intraday_costs(buy_price, sell_price, quantity, exchange=exchange)
        if direction == "UP":
            gross = (exit_ref - entry) * quantity
            slippage_cost = max(0.0, (entry_exec - entry) + (exit_ref - exit_exec)) * quantity
        else:
            gross = (entry - exit_ref) * quantity
            slippage_cost = max(0.0, (entry - entry_exec) + (exit_exec - exit_ref)) * quantity
        risk = (entry_exec - stop) * quantity if direction == "UP" else (stop - entry_exec) * quantity
        total_cost = costs["total_cost_inr"] + slippage_cost
        net = gross - total_cost
        return {
            "direction": direction,
            "entry_at": str(pd.Timestamp(entry_timestamp)),
            "exit_at": str(exit_ts),
            "entry": float(entry_exec),
            "exit": float(exit_exec),
            "stop": float(stop),
            "target": float(target),
            "exit_reason": exit_reason,
            "gross_pnl": float(gross),
            "statutory_cost": float(costs["total_cost_inr"]),
            "slippage_cost": float(slippage_cost),
            "total_cost": float(total_cost),
            "net_pnl": float(net),
            "risk_inr": float(risk),
            "net_r": float(net / max(risk, 1e-9)),
            **costs,
        }
