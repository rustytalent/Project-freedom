"""Execution-mode backtests for post-touch liquidity-pool trading.

This module answers a different question than the pool tester:

    "If a pool existed, how would a concrete entry/exit rule have performed
     after slippage, brokerage, and taxes?"

The tester can say a pool was respected or swept/reclaimed. The execution
backtest decides whether a strategy could have captured that behavior with a
realistic entry, stop, target, time exit, and cost model.

R semantics are explicit:
- net_r is profitability R: positive = made money, negative = lost money.
- directional_net_r is direction-coded R: positive = long/up trade, negative =
  short/down trade; magnitude is abs(net_r). Use this for directional maps, not
  profitability.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import Config
from .costs import ZerodhaEquityCostConfig, estimate_round_trip_charges
from .indicators import atr
from .pools import Pool
from .sectors import sector_of
from .tester import PoolResult


EXECUTION_MODES = (
    "blind_limit",
    "touch_confirmed",
    "reclaim_confirmed",
    "displacement_confirmed",
)


@dataclass
class ExecutionTrade:
    mode: str
    symbol: str
    sector: str
    side: str
    direction: str
    direction_sign: int
    pool_idx: int
    entry_at: str
    exit_at: str
    entry: float
    exit: float
    stop: float
    target: float
    quantity: int
    exit_reason: str
    bars_held: int
    gross_pnl: float
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

    def to_dict(self) -> Dict:
        return asdict(self)


def _touch(low: float, high: float, pool_low: float, pool_high: float) -> bool:
    return not (high < pool_low or low > pool_high)


def _close_through_atr(pool: Pool, close: float, atr_val: float) -> float:
    a = max(float(atr_val), 1e-9)
    if pool.side == "low" and close < pool.price_low:
        return (pool.price_low - close) / a
    if pool.side == "high" and close > pool.price_high:
        return (close - pool.price_high) / a
    return 0.0


def _inside(pool: Pool, close: float) -> bool:
    return pool.price_low <= close <= pool.price_high


def _entry_for_mode(
    mode: str,
    pool: Pool,
    df: pd.DataFrame,
    atr_values: pd.Series,
    start: int,
    end: int,
    cfg: Config,
) -> Optional[tuple[int, float, str]]:
    outside_seen = False
    first_close_through_idx: Optional[int] = None
    first_touch_idx: Optional[int] = None

    for j in range(start, end):
        row = df.iloc[j]
        in_zone = _touch(
            float(row["low"]), float(row["high"]), pool.price_low, pool.price_high,
        )
        if not in_zone:
            outside_seen = True
        if not in_zone or not outside_seen:
            continue

        if first_touch_idx is None:
            first_touch_idx = j

        if mode == "blind_limit":
            entry = pool.price_high if pool.side == "low" else pool.price_low
            return j, float(entry), "touch"

        close = float(row["close"])
        atr_val = float(atr_values.iloc[j])

        if mode == "touch_confirmed":
            # Require the touch candle to reject back outside the zone.
            if pool.side == "low" and close > pool.price_high:
                return j, close, "rejection_close"
            if pool.side == "high" and close < pool.price_low:
                return j, close, "rejection_close"

        if mode == "reclaim_confirmed":
            close_through = _close_through_atr(pool, close, atr_val)
            if close_through >= cfg.weak_break_atr and first_close_through_idx is None:
                first_close_through_idx = j
            if first_close_through_idx is not None:
                bars_since_break = j - first_close_through_idx
                if 0 < bars_since_break <= cfg.reclaim_within_bars and _inside(pool, close):
                    return j, close, "reclaim_close"
                if bars_since_break > cfg.reclaim_within_bars:
                    return None

        if mode == "displacement_confirmed":
            close_through = _close_through_atr(pool, close, atr_val)
            if close_through >= cfg.weak_break_atr and first_close_through_idx is None:
                first_close_through_idx = j
            if first_close_through_idx is not None:
                bars_since_break = j - first_close_through_idx
                if 0 < bars_since_break <= cfg.reclaim_within_bars:
                    if pool.side == "low" and close > pool.price_high:
                        return j, close, "displacement_reclaim_close"
                    if pool.side == "high" and close < pool.price_low:
                        return j, close, "displacement_reclaim_close"
                if bars_since_break > cfg.reclaim_within_bars:
                    return None

        if first_touch_idx is not None and (j - first_touch_idx) >= cfg.respect_within_bars:
            return None

    return None


def _levels(pool: Pool, atr_value: float) -> tuple[float, float]:
    a = max(float(atr_value), 1e-9)
    if pool.side == "low":
        stop = pool.price_low - 0.5 * a
        target = pool.price_high + 2.0 * a
    else:
        stop = pool.price_high + 0.5 * a
        target = pool.price_low - 2.0 * a
    return float(stop), float(target)


def _exit_trade(
    pool: Pool,
    df: pd.DataFrame,
    entry_idx: int,
    end: int,
    stop: float,
    target: float,
    time_stop_bars: int,
) -> tuple[int, float, str]:
    last_idx = min(end - 1, entry_idx + max(1, int(time_stop_bars)))
    for j in range(entry_idx + 1, last_idx + 1):
        row = df.iloc[j]
        low = float(row["low"])
        high = float(row["high"])
        open_ = float(row["open"])
        if pool.side == "low":
            stop_hit = low <= stop
            target_hit = high >= target
            if stop_hit:
                # A stop is a market order: if the bar gapped open below the stop, it fills at
                # the (worse) open, not the exact stop. Targets are limit fills → capped at target.
                return j, float(min(open_, stop)), "stop"
            if target_hit:
                return j, float(target), "target"
        else:
            stop_hit = high >= stop
            target_hit = low <= target
            if stop_hit:
                return j, float(max(open_, stop)), "stop"
            if target_hit:
                return j, float(target), "target"
    return last_idx, float(df["close"].iloc[last_idx]), "time_exit"


def _reaction_label(outcome: str) -> str:
    if outcome == "respected_strong":
        return "HARD_REJECT"
    if outcome == "swept_and_reclaimed":
        return "SWEEP_RECLAIM"
    if outcome == "respected_weak":
        return "ABSORPTION"
    if outcome in ("broken_strong", "broken_weak"):
        return "FAIL_CONTINUE"
    if outcome == "touched_no_signal":
        return "NO_SIGNAL"
    return "LIQUIDITY_VACUUM"


def _headline_factor(pool: Pool) -> str:
    priority = ("OB", "FVG", "EQHL", "REJ", "ORB", "HVN", "SWING", "PD", "PW", "PM")
    sources = {c.source.split("@", 1)[0] for c in pool.contributors}
    families = set()
    for s in sources:
        if s.startswith("OB_"):
            families.add("OB")
        elif s.startswith("FVG_"):
            families.add("FVG")
        elif s in ("EQH", "EQL"):
            families.add("EQHL")
        elif s.startswith("REJ_"):
            families.add("REJ")
        elif s.startswith("ORB_"):
            families.add("ORB")
        elif s == "HVN":
            families.add("HVN")
        elif s.startswith("SWING_"):
            families.add("SWING")
        elif s.startswith("PD"):
            families.add("PD")
        elif s.startswith("PW"):
            families.add("PW")
        elif s.startswith("PM"):
            families.add("PM")
    for item in priority:
        if item in families:
            return item
    return "OTHER"


def simulate_pool_trade(
    *,
    mode: str,
    symbol: str,
    sector: str,
    df_base: pd.DataFrame,
    pool: Pool,
    result: PoolResult,
    cfg: Config,
    cost_cfg: ZerodhaEquityCostConfig,
    quantity: int = 1,
    time_stop_bars: Optional[int] = None,
) -> Optional[ExecutionTrade]:
    if mode not in EXECUTION_MODES:
        raise ValueError(f"unknown execution mode {mode!r}")
    if df_base.empty or result.outcome == "horizon_insufficient":
        return None

    idx = df_base.index
    start = int(np.searchsorted(idx.values, np.datetime64(pool.available_at), side="left"))
    end = min(start + cfg.test_horizon_bars, len(df_base))
    if end <= start + 1:
        return None

    atr_values = atr(df_base, cfg.detect.atr_period).bfill()
    entry = _entry_for_mode(mode, pool, df_base, atr_values, start, end, cfg)
    if entry is None:
        return None
    signal_idx, entry_price, _entry_reason = entry
    # A confirmation signal is only known at the signal bar's CLOSE, so we cannot fill at that
    # same close (look-ahead). Enter at the NEXT bar's open instead. blind_limit is a resting
    # limit order that fills intrabar at its limit price on the touch bar, so it keeps signal_idx.
    entry_idx = signal_idx
    if mode != "blind_limit":
        entry_idx = signal_idx + 1
        if entry_idx >= end:
            return None
        entry_price = float(df_base["open"].iloc[entry_idx])
    stop, target = _levels(pool, float(atr_values.iloc[entry_idx]))
    exit_idx, exit_price, exit_reason = _exit_trade(
        pool, df_base, entry_idx, end, stop, target,
        time_stop_bars or cfg.respect_within_bars,
    )

    qty = max(int(quantity), 1)
    # Risk integrity: on confirmation modes the entry price is the NEXT bar's
    # open. If that bar gaps PAST the protective stop, the trade would have
    # been stopped out at open before it began — it is not a real trade
    # outcome, it is a slippage event. Returning None here filters those
    # degenerate setups out cleanly. (Blind_limit is immune because its
    # entry is the pool boundary and `_levels()` anchors the stop to the
    # same bar's ATR, so risk is always +0.5*ATR by construction.)
    #
    # Defense in depth: even when the geometry is technically positive, if
    # the gap collapses risk to a tiny fraction of ATR (e.g. entry within
    # 0.1 ATR of stop), the resulting net_r explodes — a few rupees divided
    # by near-zero risk yields millions of R. Floor risk at 0.1*ATR so
    # an edge case can never produce an absurd R again.
    atr_at_entry = max(float(atr_values.iloc[entry_idx]), 1e-9)
    MIN_RISK_FRACTION_OF_ATR = 0.10
    min_risk_per_share = MIN_RISK_FRACTION_OF_ATR * atr_at_entry
    if pool.side == "low":
        natural_risk_per_share = float(entry_price - stop)
        if natural_risk_per_share <= 0:
            return None    # gap-down past stop -> stopped at open, not a real trade
        risk_per_share = max(natural_risk_per_share, min_risk_per_share, 1e-9)
        gross_per_share = float(exit_price - entry_price)
        side = "buy"
        direction = "UP"
        direction_sign = 1
    else:
        natural_risk_per_share = float(stop - entry_price)
        if natural_risk_per_share <= 0:
            return None    # gap-up past stop -> stopped at open, not a real trade
        risk_per_share = max(natural_risk_per_share, min_risk_per_share, 1e-9)
        gross_per_share = float(entry_price - exit_price)
        side = "sell"
        direction = "DOWN"
        direction_sign = -1
    gross = gross_per_share * qty
    # side matters for STT (sell-leg) vs stamp (buy-leg): a short sells at entry, buys at exit.
    charges = estimate_round_trip_charges(
        entry_price, exit_price, qty, cost_cfg,
        side="short" if pool.side == "high" else "long",
    )
    total_cost = float(charges["total_cost"])
    net = gross - total_cost
    risk = risk_per_share * qty
    net_r = float(net / risk)

    return ExecutionTrade(
        mode=mode,
        symbol=symbol,
        sector=sector,
        side=side,
        direction=direction,
        direction_sign=direction_sign,
        pool_idx=int(result.pool_idx),
        entry_at=str(idx[entry_idx]),
        exit_at=str(idx[exit_idx]),
        entry=float(entry_price),
        exit=float(exit_price),
        stop=float(stop),
        target=float(target),
        quantity=qty,
        exit_reason=exit_reason,
        bars_held=int(exit_idx - entry_idx),
        gross_pnl=float(gross),
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
    )


def simulate_execution_modes(
    *,
    df_base: pd.DataFrame,
    pools: Sequence[Pool],
    results: Sequence[PoolResult],
    symbol: str,
    sector: str,
    cfg: Config,
    cost_cfg: ZerodhaEquityCostConfig,
    modes: Iterable[str] = EXECUTION_MODES,
    quantity: int = 1,
    time_stop_bars: Optional[int] = None,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for mode in modes:
        for pool, result in zip(pools, results):
            trade = simulate_pool_trade(
                mode=mode,
                symbol=symbol,
                sector=sector,
                df_base=df_base,
                pool=pool,
                result=result,
                cfg=cfg,
                cost_cfg=cost_cfg,
                quantity=quantity,
                time_stop_bars=time_stop_bars,
            )
            if trade is not None:
                rows.append(trade.to_dict())
    return pd.DataFrame(rows)


def summarise_execution_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(columns=[
            "mode", "trades", "win_rate", "avg_win", "avg_loss",
            "net_expectancy", "net_expectancy_r", "profit_factor",
            "max_drawdown", "up_trades", "down_trades",
            "up_net_expectancy_r", "down_net_expectancy_r",
            "directional_net_expectancy_r", "trades_per_symbol", "avg_bars_held",
        ])

    rows = []
    for mode, df in trades.groupby("mode"):
        net = df["net_pnl"].astype(float)
        wins = net[net > 0]
        losses = net[net < 0]
        equity = net.cumsum()
        drawdown = equity - equity.cummax()
        gross_win = float(wins.sum())
        gross_loss = float(-losses.sum())
        up = df[df["direction"] == "UP"]
        down = df[df["direction"] == "DOWN"]
        rows.append({
            "mode": mode,
            "trades": int(len(df)),
            "win_rate": float((net > 0).mean()) if len(df) else 0.0,
            "avg_win": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) else 0.0,
            "net_expectancy": float(net.mean()) if len(df) else 0.0,
            "net_expectancy_r": float(df["net_r"].mean()) if len(df) else 0.0,
            "directional_net_expectancy_r": float(df["directional_net_r"].mean())
            if "directional_net_r" in df.columns and len(df) else 0.0,
            "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
            "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
            "up_trades": int(len(up)),
            "down_trades": int(len(down)),
            "up_net_expectancy_r": float(up["net_r"].mean()) if len(up) else 0.0,
            "down_net_expectancy_r": float(down["net_r"].mean()) if len(down) else 0.0,
            "trades_per_symbol": float(len(df) / max(df["symbol"].nunique(), 1)),
            "avg_bars_held": float(df["bars_held"].mean()) if len(df) else 0.0,
            "target_exit_rate": float((df["exit_reason"] == "target").mean()),
            "stop_exit_rate": float((df["exit_reason"] == "stop").mean()),
            "time_exit_rate": float((df["exit_reason"] == "time_exit").mean()),
        })
    return pd.DataFrame(rows).sort_values("net_expectancy_r", ascending=False)


def summarise_execution_by_direction(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(columns=[
            "mode", "direction", "trades", "win_rate", "net_expectancy",
            "net_expectancy_r", "profit_factor", "max_drawdown",
        ])

    rows = []
    for (mode, direction), df in trades.groupby(["mode", "direction"]):
        net = df["net_pnl"].astype(float)
        wins = net[net > 0]
        losses = net[net < 0]
        equity = net.cumsum()
        drawdown = equity - equity.cummax()
        gross_win = float(wins.sum())
        gross_loss = float(-losses.sum())
        rows.append({
            "mode": mode,
            "direction": direction,
            "direction_sign": int(df["direction_sign"].iloc[0]),
            "trades": int(len(df)),
            "win_rate": float((net > 0).mean()) if len(df) else 0.0,
            "net_expectancy": float(net.mean()) if len(df) else 0.0,
            "net_expectancy_r": float(df["net_r"].mean()) if len(df) else 0.0,
            "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
            "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
            "target_exit_rate": float((df["exit_reason"] == "target").mean()),
            "stop_exit_rate": float((df["exit_reason"] == "stop").mean()),
            "time_exit_rate": float((df["exit_reason"] == "time_exit").mean()),
        })
    return pd.DataFrame(rows).sort_values(["mode", "direction_sign"])


def build_execution_backtest_for_report(
    report,
    cfg: Config,
    cost_cfg: ZerodhaEquityCostConfig,
    quantity: int = 1,
    modes: Iterable[str] = EXECUTION_MODES,
    split: str = "oos",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for symbol, ad in report.assets.items():
        if split == "train":
            pools = ad.walkforward.train_pools_for_ml
            results = ad.walkforward.train_results_for_ml
        else:
            pools = ad.walkforward.oos_pools
            results = ad.walkforward.oos_results
        asset_cfg = getattr(ad, "final_cfg", None) or cfg
        frames.append(simulate_execution_modes(
            df_base=ad.base_df,
            pools=pools,
            results=results,
            symbol=symbol,
            sector=sector_of(symbol),
            cfg=asset_cfg,
            cost_cfg=cost_cfg,
            modes=modes,
            quantity=quantity,
        ))
    trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return trades, summarise_execution_trades(trades)
