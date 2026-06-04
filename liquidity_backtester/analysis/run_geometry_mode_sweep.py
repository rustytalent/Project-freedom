"""Execution-layer geometry and mode sweep for the saved 710362b bundle.

This script does not retrain models and does not add alpha features. It replays
the existing OOS touched-pool population through the V2 simulator to answer:

1. Does wider post-touch geometry cross the cost wall?
2. Do pool factors prefer respect, break-continuation, or sweep-reclaim modes?
3. Is the new sweep_reclaim mode tradeable or just another negative wrapper?
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from liqpool.config import Config
from liqpool.execution_simulator_v2 import (
    ExecutionV2Config,
    _load_symbol_1m,
    simulate_execution_modes_v2,
)
from liqpool.sectors import sector_of


GEOMETRY_GRID: Tuple[Tuple[float, float], ...] = (
    (0.50, 2.00),
    (1.00, 2.00),
    (1.50, 2.00),
    (1.50, 2.50),
    (2.00, 2.50),
    (2.50, 2.50),
    (1.50, 3.00),
)

MODE_LABELS = {
    "touch_confirmed": "respect_mode",
    "break_confirmed": "break_mode",
    "sweep_reclaim": "reclaim_mode",
}

_WORKER_REPORT = None
_WORKER_CFG = None


def _normal_p_gt_zero(mean_r: float, se_r: float) -> float:
    if not np.isfinite(mean_r) or not np.isfinite(se_r) or se_r <= 0:
        return 1.0 if mean_r <= 0 else 0.0
    z = mean_r / se_r
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def _load_bundle(path: Path):
    with path.expanduser().open("rb") as f:
        return pickle.load(f)


def _gross_r(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    return frame["gross_pnl"].astype(float) / frame["risk_inr"].astype(float).clip(lower=1e-9)


def _cost_r(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    return frame["total_cost"].astype(float) / frame["risk_inr"].astype(float).clip(lower=1e-9)


def _profit_factor(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    pnl = frame["net_pnl"].astype(float)
    gains = float(pnl[pnl > 0].sum())
    losses = abs(float(pnl[pnl < 0].sum()))
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def _max_drawdown_inr(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    if "entry_at" in frame.columns:
        ordered = frame.sort_values("entry_at")
    else:
        ordered = frame
    equity = ordered["net_pnl"].astype(float).cumsum()
    dd = equity - equity.cummax()
    return float(dd.min()) if len(dd) else 0.0


def _summarise(frame: pd.DataFrame) -> Dict:
    if frame.empty:
        return {
            "n": 0,
            "trades": 0,
            "win_rate": 0.0,
            "mean_R": 0.0,
            "mean_gross_R": 0.0,
            "mean_cost_R": 0.0,
            "PF": 0.0,
            "max_drawdown_inr": 0.0,
            "se_R": 0.0,
            "p_value": 1.0,
            "positive_ev_p05": False,
        }
    net_r = frame["net_r"].astype(float)
    mean_r = float(net_r.mean())
    se_r = float(net_r.std(ddof=1) / math.sqrt(len(net_r))) if len(net_r) > 1 else 0.0
    p = _normal_p_gt_zero(mean_r, se_r)
    return {
        "n": int(len(frame)),
        "trades": int(len(frame)),
        "win_rate": float((net_r > 0).mean()),
        "mean_R": mean_r,
        "mean_gross_R": float(_gross_r(frame).mean()),
        "mean_cost_R": float(_cost_r(frame).mean()),
        "PF": float(_profit_factor(frame)),
        "max_drawdown_inr": _max_drawdown_inr(frame),
        "se_R": se_r,
        "p_value": p,
        "positive_ev_p05": bool(mean_r > 0 and p < 0.05),
    }


def _init_worker(report, cfg: Config) -> None:
    global _WORKER_REPORT, _WORKER_CFG
    _WORKER_REPORT = report
    _WORKER_CFG = cfg


def _asset_task(payload: Tuple[str, Tuple[str, ...], Dict, Optional[str], str]) -> pd.DataFrame:
    symbol, modes, v2_kwargs, raw_1m_dir, split = payload
    report = _WORKER_REPORT
    cfg = _WORKER_CFG
    ad = report.assets[symbol]
    if split == "train":
        pools = ad.walkforward.train_pools_for_ml
        results = ad.walkforward.train_results_for_ml
    else:
        pools = ad.walkforward.oos_pools
        results = ad.walkforward.oos_results
    asset_cfg = getattr(ad, "final_cfg", None) or cfg
    v2_cfg = ExecutionV2Config(**v2_kwargs)
    intrabar = _load_symbol_1m(symbol, raw_1m_dir) if v2_cfg.use_1m_resolution else None
    return simulate_execution_modes_v2(
        df_base=ad.base_df,
        pools=pools,
        results=results,
        symbol=symbol,
        sector=sector_of(symbol),
        cfg=asset_cfg,
        v2_cfg=v2_cfg,
        modes=modes,
        intrabar_1m=intrabar,
    )


def _simulate_report(
    report,
    cfg: Config,
    *,
    modes: Sequence[str],
    v2_cfg: ExecutionV2Config,
    raw_1m_dir: Optional[str],
    split: str,
    workers: int,
) -> pd.DataFrame:
    symbols = sorted(report.assets.keys())
    tasks = [
        (symbol, tuple(modes), v2_cfg.__dict__.copy(), raw_1m_dir, split)
        for symbol in symbols
    ]
    if workers <= 1:
        _init_worker(report, cfg)
        frames = [_asset_task(t) for t in tasks]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=int(workers), initializer=_init_worker,
                      initargs=(report, cfg)) as pool:
            frames = list(pool.map(_asset_task, tasks))
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _reclaim_population(report, split: str) -> Dict:
    touched = 0
    swept = 0
    outcomes: Dict[str, int] = {}
    for ad in report.assets.values():
        results = (
            ad.walkforward.train_results_for_ml
            if split == "train" else ad.walkforward.oos_results
        )
        for r in results:
            outcome = getattr(r, "outcome", "")
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            if outcome not in ("untouched", "horizon_insufficient"):
                touched += 1
            if outcome == "swept_and_reclaimed":
                swept += 1
    return {
        "n_total": int(touched),
        "n_swept_and_reclaimed_labels": int(swept),
        "outcome_counts": outcomes,
    }


def _geometry_sweep(report, cfg: Config, args) -> Tuple[List[Dict], Dict[str, pd.DataFrame]]:
    rows: List[Dict] = []
    trade_frames: Dict[str, pd.DataFrame] = {}
    for stop_mult, target_mult in GEOMETRY_GRID:
        label = f"stop_{stop_mult:g}_target_{target_mult:g}"
        print(f"[geometry] replaying {label}")
        v2_cfg = ExecutionV2Config(
            fill_policy=args.fill_policy,
            use_1m_resolution=args.use_1m_resolution,
            slippage_model=args.slippage_model,
            base_slippage_bps=args.base_slippage_bps,
            exchange=args.exchange,
            quantity=args.quantity,
            notional_inr=args.notional_inr,
            stop_atr_mult=stop_mult,
            target_atr_mult=target_mult,
        )
        trades = _simulate_report(
            report, cfg, modes=("touch_confirmed",), v2_cfg=v2_cfg,
            raw_1m_dir=args.raw_1m_dir, split=args.split, workers=args.workers,
        )
        trade_frames[label] = trades
        summary = _summarise(trades)
        summary.update({
            "geometry": label,
            "stop_atr": float(stop_mult),
            "target_atr": float(target_mult),
        })
        rows.append(summary)
    rows.sort(key=lambda r: r["mean_R"], reverse=True)
    return rows, trade_frames


def _mode_matrix(report, cfg: Config, args) -> Tuple[Dict, pd.DataFrame]:
    print("[modes] replaying respect/break/reclaim matrix")
    v2_cfg = ExecutionV2Config(
        fill_policy=args.fill_policy,
        use_1m_resolution=args.use_1m_resolution,
        slippage_model=args.slippage_model,
        base_slippage_bps=args.base_slippage_bps,
        exchange=args.exchange,
        quantity=args.quantity,
        notional_inr=args.notional_inr,
    )
    trades = _simulate_report(
        report, cfg,
        modes=("touch_confirmed", "break_confirmed", "sweep_reclaim"),
        v2_cfg=v2_cfg,
        raw_1m_dir=args.raw_1m_dir,
        split=args.split,
        workers=args.workers,
    )
    matrix: Dict[str, Dict[str, Dict]] = {}
    if trades.empty:
        return matrix, trades
    for (factor, mode), group in trades.groupby(["factor", "mode"], dropna=False):
        factor_key = str(factor or "UNKNOWN")
        mode_key = MODE_LABELS.get(str(mode), str(mode))
        matrix.setdefault(factor_key, {})[mode_key] = _summarise(group)
    return matrix, trades


def _best_factor_pairs(matrix: Dict[str, Dict[str, Dict]]) -> List[Dict]:
    pairs = []
    for factor, modes in matrix.items():
        for mode, summary in modes.items():
            if summary.get("n", 0) >= 50 and summary.get("mean_R", 0.0) > 0:
                row = {"factor": factor, "mode": mode}
                row.update(summary)
                pairs.append(row)
    pairs.sort(key=lambda r: r["mean_R"], reverse=True)
    return pairs


def _print_geometry(rows: List[Dict]) -> None:
    print("\n=== GEOMETRY SWEEP (sorted by mean_R) ===")
    print(f"{'geometry':<20} {'trades':>7} {'win':>7} {'mean_R':>8} "
          f"{'gross_R':>8} {'cost_R':>7} {'PF':>7} {'p':>9} {'maxDD':>10}")
    for r in rows:
        pf = r["PF"]
        pf_s = "inf" if not np.isfinite(pf) else f"{pf:.2f}"
        star = " *" if r["positive_ev_p05"] else ""
        print(f"{r['geometry']:<20} {r['trades']:>7} {r['win_rate']:>6.1%} "
              f"{r['mean_R']:>+8.3f} {r['mean_gross_R']:>+8.3f} "
              f"{r['mean_cost_R']:>7.3f} {pf_s:>7} {r['p_value']:>9.4f} "
              f"₹{r['max_drawdown_inr']:>9.0f}{star}")


def _print_matrix(matrix: Dict[str, Dict[str, Dict]]) -> None:
    print("\n=== PER-FACTOR MODE MATRIX (mean_R) ===")
    print(f"{'factor':<10} {'respect_mode':>14} {'break_mode':>14} {'reclaim_mode':>14}")
    for factor in sorted(matrix):
        modes = matrix[factor]
        vals = []
        for mode in ("respect_mode", "break_mode", "reclaim_mode"):
            s = modes.get(mode)
            vals.append("n/a" if s is None else f"{s['mean_R']:+.3f} ({s['n']})")
        print(f"{factor:<10} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14}")


def run(args) -> Dict:
    bundle = Path(args.bundle).expanduser()
    if not bundle.exists():
        raise SystemExit(f"missing bundle: {bundle}")
    report = _load_bundle(bundle)
    cfg = getattr(report, "config", None) or Config()

    geometry_rows, _ = _geometry_sweep(report, cfg, args)
    matrix, mode_trades = _mode_matrix(report, cfg, args)
    reclaim_pop = _reclaim_population(report, args.split)
    reclaim_trades = (
        mode_trades[mode_trades["mode"] == "sweep_reclaim"]
        if not mode_trades.empty and "mode" in mode_trades.columns
        else pd.DataFrame()
    )
    reclaim_pop["n_qualifying_reclaims"] = int(len(reclaim_trades))
    reclaim_pop["reclaim_mode_summary"] = _summarise(reclaim_trades)

    best_geometry = geometry_rows[0] if geometry_rows else {}
    best_pairs = _best_factor_pairs(matrix)
    headline = {
        "best_geometry": best_geometry.get("geometry"),
        "best_geometry_mean_R": best_geometry.get("mean_R", 0.0),
        "best_factor_mode_pairs": [
            {
                "factor": p["factor"],
                "mode": p["mode"],
                "n": p["n"],
                "mean_R": p["mean_R"],
                "p_value": p["p_value"],
            }
            for p in best_pairs[:10]
        ],
        "reclaim_mode_mean_R": reclaim_pop["reclaim_mode_summary"]["mean_R"],
        "any_positive_EV_cell": bool(
            any(r.get("positive_ev_p05") for r in geometry_rows)
            or any(p.get("p_value", 1.0) < 0.05 for p in best_pairs)
            or reclaim_pop["reclaim_mode_summary"].get("positive_ev_p05", False)
        ),
    }
    out = {
        "geometry_sweep_results": geometry_rows,
        "per_factor_mode_matrix": matrix,
        "reclaim_mode_population": reclaim_pop,
        "headline_findings": headline,
    }

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    _print_geometry(geometry_rows)
    _print_matrix(matrix)
    print("\n=== HEADLINE FINDINGS ===")
    print(json.dumps(headline, indent=2, default=str))
    print(f"\nwrote {out_path}")
    return out


def parse_args(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True,
                    help="Path to multi_asset_report.pkl")
    ap.add_argument("--raw-1m-dir", default=None)
    ap.add_argument("--out", default="output_core25_head_alpha_710362b/geometry_mode_sweep.json")
    ap.add_argument("--split", default="oos", choices=("oos", "train"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--quantity", type=int, default=1)
    ap.add_argument("--notional-inr", type=float, default=100000.0,
                    help="Target per-trade notional. Set 0 to use --quantity instead.")
    ap.add_argument("--exchange", default="NSE")
    ap.add_argument("--fill-policy", default="neutral",
                    choices=("generous", "neutral", "conservative"))
    ap.add_argument("--use-1m-resolution", action="store_true")
    ap.add_argument("--slippage-model", default="state_dependent",
                    choices=("flat", "state_dependent"))
    ap.add_argument("--base-slippage-bps", type=float, default=2.0)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.notional_inr is not None and args.notional_inr <= 0:
        args.notional_inr = None
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
