"""Execution-policy labels for Phase 3C.

Pool reaction taxonomies are useful diagnostics, but supervision should also be
defined under an explicit execution policy. This module emits one row per
pool-policy pair:

- generated_trade = 1 when the policy produced an entry
- policy_return_r = realised net R after costs
- policy_label = 1/0 for profitable/non-profitable trades
- no-trade rows are preserved so gates do not hide selection bias
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .costs import ZerodhaEquityCostConfig
from .execution_backtest import (
    EXECUTION_MODES,
    _headline_factor,
    _reaction_label,
    simulate_pool_trade,
)
from .execution_simulator_v2 import (
    ExecutionV2Config,
    _load_symbol_1m,
    simulate_pool_trade_v2,
)
from .pools import Pool
from .sectors import sector_of
from .tester import PoolResult


def _direction_for_pool(pool: Pool) -> Tuple[str, str, int]:
    if pool.side == "low":
        return "buy", "UP", 1
    return "sell", "DOWN", -1


def _no_trade_reason(result: PoolResult) -> str:
    if result.outcome == "horizon_insufficient":
        return "horizon_insufficient"
    if result.touched_at is None:
        return "not_touched"
    return "entry_policy_not_satisfied"


def _base_row(symbol: str, sector: str, mode: str, pool_idx: int,
              pool: Pool, result: PoolResult, split: str) -> Dict:
    side, direction, direction_sign = _direction_for_pool(pool)
    return {
        "mode": mode,
        "split": split,
        "symbol": symbol,
        "sector": sector,
        "pool_idx": int(pool_idx),
        "side": side,
        "direction": direction,
        "direction_sign": int(direction_sign),
        "pool_low": float(pool.price_low),
        "pool_high": float(pool.price_high),
        "pool_mid": float(pool.mid),
        "available_at": str(pool.available_at),
        "formed_at": str(pool.formed_at),
        "touched_at": str(result.touched_at) if result.touched_at is not None else None,
        "bars_to_touch": result.bars_to_touch,
        "outcome": result.outcome,
        "reaction_label": _reaction_label(result.outcome),
        "mae_atr": float(result.max_excursion_through),
        "mfe_atr": float(result.reaction_atr),
        "score": float(pool.score),
        "tf_count": int(len(set(pool.tfs))),
        "factor": _headline_factor(pool),
    }


def policy_label_rows_for_asset(
    *,
    symbol: str,
    df_base: pd.DataFrame,
    pools: Sequence[Pool],
    results: Sequence[PoolResult],
    cfg: Config,
    cost_cfg: ZerodhaEquityCostConfig,
    modes: Iterable[str] = EXECUTION_MODES,
    quantity: int = 1,
    execution_version: str = "v2",
    v2_cfg: Optional[ExecutionV2Config] = None,
    intrabar_1m: Optional[pd.DataFrame] = None,
    split: str = "oos",
) -> List[Dict]:
    if execution_version not in {"v1", "v2"}:
        raise ValueError("execution_version must be 'v1' or 'v2'")
    if execution_version == "v2" and v2_cfg is None:
        v2_cfg = ExecutionV2Config(notional_inr=100_000.0)
    sector = sector_of(symbol)
    rows: List[Dict] = []
    for mode in modes:
        for pool_idx, (pool, result) in enumerate(zip(pools, results)):
            row = _base_row(symbol, sector, mode, pool_idx, pool, result, split)
            row.update({
                "execution_version": execution_version,
                "fill_policy": getattr(v2_cfg, "fill_policy", None) if v2_cfg else None,
                "slippage_model": getattr(v2_cfg, "slippage_model", None) if v2_cfg else None,
                "notional_inr": getattr(v2_cfg, "notional_inr", None) if v2_cfg else None,
            })
            if execution_version == "v2":
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
                )
            else:
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
                )
            if trade is None:
                row.update({
                    "generated_trade": 0,
                    "label_status": "NO_TRADE",
                    "policy_label": np.nan,
                    "policy_return_r": np.nan,
                    "policy_target_return_r": np.nan,
                    "policy_target_win": np.nan,
                    "policy_target_trade_generated": 0,
                    "policy_target_censored": 1,
                    "policy_return_inr": np.nan,
                    "gross_pnl": np.nan,
                    "total_cost": np.nan,
                    "risk_inr": np.nan,
                    "entry_at": None,
                    "exit_at": None,
                    "entry": np.nan,
                    "exit": np.nan,
                    "stop": np.nan,
                    "target": np.nan,
                    "exit_reason": "no_entry",
                    "barrier_label": "no_entry",
                    "bars_held": np.nan,
                    "time_barrier": 0,
                    "censored": 1,
                    "no_trade_reason": _no_trade_reason(result),
                })
            else:
                d = trade.to_dict()
                net_r = float(d["net_r"])
                win = 1 if d["net_pnl"] > 0 else 0
                row.update({
                    "generated_trade": 1,
                    "label_status": "LABELLED",
                    "policy_label": win,
                    "policy_return_r": net_r,
                    "policy_target_return_r": net_r,
                    "policy_target_win": win,
                    "policy_target_trade_generated": 1,
                    "policy_target_censored": 0,
                    "policy_return_inr": float(d["net_pnl"]),
                    "gross_pnl": float(d["gross_pnl"]),
                    "total_cost": float(d["total_cost"]),
                    "statutory_cost": float(d.get("statutory_cost", d["total_cost"])),
                    "slippage_cost": float(d.get("slippage_cost", 0.0)),
                    "risk_inr": float(d["risk_inr"]),
                    "entry_at": d["entry_at"],
                    "exit_at": d["exit_at"],
                    "entry": float(d["entry"]),
                    "exit": float(d["exit"]),
                    "stop": float(d["stop"]),
                    "target": float(d["target"]),
                    "exit_reason": d["exit_reason"],
                    "barrier_label": d["exit_reason"],
                    "bars_held": int(d["bars_held"]),
                    "time_barrier": 1 if d["exit_reason"] == "time_exit" else 0,
                    "censored": 0,
                    "no_trade_reason": "",
                    "resolution_source": d.get("resolution_source", "5m"),
                    "entry_reason": d.get("entry_reason", ""),
                })
            rows.append(row)
    return rows


def build_policy_labels_for_report(
    report,
    cfg: Config,
    cost_cfg: ZerodhaEquityCostConfig,
    quantity: int = 1,
    execution_version: str = "v2",
    v2_cfg: Optional[ExecutionV2Config] = None,
    raw_1m_dir: str | Path | None = None,
    modes: Iterable[str] = EXECUTION_MODES,
    split: str = "oos",
) -> pd.DataFrame:
    if execution_version not in {"v1", "v2"}:
        raise ValueError("execution_version must be 'v1' or 'v2'")
    if execution_version == "v2" and v2_cfg is None:
        v2_cfg = ExecutionV2Config(notional_inr=100_000.0)
    rows: List[Dict] = []
    for symbol, ad in report.assets.items():
        if split == "train":
            pools = ad.walkforward.train_pools_for_ml
            results = ad.walkforward.train_results_for_ml
        elif split == "final":
            pools = ad.final_pools
            results = ad.final_results
        else:
            pools = ad.walkforward.oos_pools
            results = ad.walkforward.oos_results
        asset_cfg = getattr(ad, "final_cfg", None) or cfg
        intrabar_1m = None
        if execution_version == "v2" and v2_cfg and v2_cfg.use_1m_resolution:
            intrabar_1m = _load_symbol_1m(symbol, raw_1m_dir)
        rows.extend(policy_label_rows_for_asset(
            symbol=symbol,
            df_base=ad.base_df,
            pools=pools,
            results=results,
            cfg=asset_cfg,
            cost_cfg=cost_cfg,
            modes=modes,
            quantity=quantity,
            execution_version=execution_version,
            v2_cfg=v2_cfg,
            intrabar_1m=intrabar_1m,
            split=split,
        ))
    return pd.DataFrame(rows)


def summarise_policy_labels(labels: pd.DataFrame) -> pd.DataFrame:
    if labels is None or labels.empty:
        return pd.DataFrame(columns=[
            "mode", "pool_policy_rows", "generated_trades", "trade_rate",
            "win_rate", "mean_return_r", "median_return_r", "profit_factor",
            "target_rate", "stop_rate", "time_barrier_rate", "no_trade_rate",
        ])

    rows = []
    for mode, df in labels.groupby("mode"):
        trades = df[df["generated_trade"] == 1].copy()
        wins = trades[trades["policy_return_inr"] > 0]
        losses = trades[trades["policy_return_inr"] < 0]
        gross_win = float(wins["policy_return_inr"].sum()) if len(wins) else 0.0
        gross_loss = float(-losses["policy_return_inr"].sum()) if len(losses) else 0.0
        rows.append({
            "mode": mode,
            "pool_policy_rows": int(len(df)),
            "generated_trades": int(len(trades)),
            "trade_rate": float(len(trades) / max(len(df), 1)),
            "win_rate": float((trades["policy_label"] == 1).mean()) if len(trades) else 0.0,
            "mean_return_r": float(trades["policy_return_r"].mean()) if len(trades) else 0.0,
            "median_return_r": float(trades["policy_return_r"].median()) if len(trades) else 0.0,
            "profit_factor": (
                float(gross_win / gross_loss) if gross_loss > 0 else float("inf")
            ),
            "target_rate": float((trades["exit_reason"] == "target").mean())
            if len(trades) else 0.0,
            "stop_rate": float((trades["exit_reason"] == "stop").mean())
            if len(trades) else 0.0,
            "time_barrier_rate": float((trades["exit_reason"] == "time_exit").mean())
            if len(trades) else 0.0,
            "no_trade_rate": float((df["generated_trade"] == 0).mean()) if len(df) else 0.0,
        })
    return pd.DataFrame(rows).sort_values("mean_return_r", ascending=False)
