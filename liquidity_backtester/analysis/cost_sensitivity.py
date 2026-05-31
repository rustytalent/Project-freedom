"""Cost / position-size sensitivity — does the negative R come from costs or the strategy?

Loads a SAVED model bundle (no retrain) and re-runs the execution backtest at
several position sizes, reporting per mode:

  * gross_R   — mean realized R BEFORE costs (the strategy's cost-free ceiling)
  * net_R     — mean realized R after Zerodha costs + slippage
  * cost_R    — the drag (gross_R - net_R)
  * win / PF  — sanity

Why this matters: the Zerodha cost model here is almost entirely
turnover-proportional (STT, exchange, SEBI, stamp, GST, slippage are all
bps × turnover); only the ₹20 brokerage cap is size-dependent. Both gross_pnl
and risk scale with quantity, so net_R is *nearly* scale-invariant. This script
proves whether bumping position size from 1 share to a ₹50k notional actually
moves net_R, and — more importantly — whether gross_R (the ceiling) is even
positive. If gross_R <= 0, no amount of sizing or gating can produce a winning
strategy; the edge problem is structural.

Sizing schemes tested:
  * qty=1                 — the train run's default (tiny notional, fixed-ish brokerage)
  * notional ₹50k/₹100k/₹200k — qty = floor(notional / last_close) per symbol

Usage (on the Mac where the bundle lives):
  PYTHONPATH=. .venv/bin/python analysis/cost_sensitivity.py \
      --model-dir output_models/core25_latest \
      --notionals 50000,100000,200000 \
      --out output_audit
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import multiprocessing as mp
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from liqpool.config import Config
from liqpool.costs import ZerodhaEquityCostConfig
from liqpool.execution_backtest import EXECUTION_MODES, simulate_execution_modes
from liqpool.sectors import sector_of


_WORKER_REPORT = None
_WORKER_CFG = None
_WORKER_COST_CFG = None
_WORKER_SPLIT = "oos"


def _load_bundle(model_dir: Path):
    p = model_dir / "multi_asset_report.pkl"
    if not p.exists():
        raise SystemExit(f"missing bundle: {p}. Run --mode train first.")
    with p.open("rb") as f:
        return pickle.load(f)


def _representative_price(ad) -> float:
    """Last close on the base timeframe — the sizing reference per symbol."""
    try:
        return float(ad.base_df["close"].iloc[-1])
    except Exception:
        return float("nan")


def _trades_for_sizing(report, cfg: Config, cost_cfg: ZerodhaEquityCostConfig,
                       split: str, notional: Optional[float]) -> pd.DataFrame:
    """Build the execution trade frame. notional=None -> fixed quantity=1;
    otherwise quantity = floor(notional / last_close) per symbol (min 1)."""
    frames: List[pd.DataFrame] = []
    for symbol, ad in report.assets.items():
        if split == "train":
            pools = ad.walkforward.train_pools_for_ml
            results = ad.walkforward.train_results_for_ml
        else:
            pools = ad.walkforward.oos_pools
            results = ad.walkforward.oos_results
        asset_cfg = getattr(ad, "final_cfg", None) or cfg
        if notional is None:
            qty = 1
        else:
            price = _representative_price(ad)
            qty = max(1, int(math.floor(notional / price))) if price > 0 else 1
        frames.append(simulate_execution_modes(
            df_base=ad.base_df, pools=pools, results=results,
            symbol=symbol, sector=sector_of(symbol),
            cfg=asset_cfg, cost_cfg=cost_cfg,
            modes=EXECUTION_MODES, quantity=qty,
        ))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _init_worker(report, cfg: Config, cost_cfg: ZerodhaEquityCostConfig, split: str) -> None:
    """Initialise forked workers with the already-loaded bundle.

    On Linux this uses copy-on-write memory, so the 600MB+ bundle is not
    physically copied into every worker unless a worker mutates it.
    """
    global _WORKER_REPORT, _WORKER_CFG, _WORKER_COST_CFG, _WORKER_SPLIT
    _WORKER_REPORT = report
    _WORKER_CFG = cfg
    _WORKER_COST_CFG = cost_cfg
    _WORKER_SPLIT = split


def _asset_sizing_task(task: Tuple[str, Optional[float], str]) -> Tuple[str, pd.DataFrame]:
    label, notional, symbol = task
    report = _WORKER_REPORT
    ad = report.assets[symbol]
    if _WORKER_SPLIT == "train":
        pools = ad.walkforward.train_pools_for_ml
        results = ad.walkforward.train_results_for_ml
    else:
        pools = ad.walkforward.oos_pools
        results = ad.walkforward.oos_results
    asset_cfg = getattr(ad, "final_cfg", None) or _WORKER_CFG
    if notional is None:
        qty = 1
    else:
        price = _representative_price(ad)
        qty = max(1, int(math.floor(notional / price))) if price > 0 else 1
    trades = simulate_execution_modes(
        df_base=ad.base_df, pools=pools, results=results,
        symbol=symbol, sector=sector_of(symbol),
        cfg=asset_cfg, cost_cfg=_WORKER_COST_CFG,
        modes=EXECUTION_MODES, quantity=qty,
    )
    return label, trades


def _mp_context():
    try:
        return mp.get_context("fork")
    except ValueError:
        return mp.get_context()


def _trades_for_schemes_parallel(
    report,
    cfg: Config,
    cost_cfg: ZerodhaEquityCostConfig,
    split: str,
    schemes: List[Tuple[str, Optional[float]]],
    workers: int,
) -> Dict[str, pd.DataFrame]:
    if workers <= 1:
        return {
            label: _trades_for_sizing(report, cfg, cost_cfg, split, notional)
            for label, notional in schemes
        }

    labels = [label for label, _ in schemes]
    frames_by_label: Dict[str, List[pd.DataFrame]] = {label: [] for label in labels}
    tasks = [
        (label, notional, symbol)
        for label, notional in schemes
        for symbol in report.assets.keys()
    ]
    max_workers = max(1, min(int(workers), len(tasks)))
    print(f"[cost-sens] parallel asset/sizing tasks: {len(tasks)}  workers={max_workers}")
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=_mp_context(),
        initializer=_init_worker,
        initargs=(report, cfg, cost_cfg, split),
    ) as ex:
        futures = [ex.submit(_asset_sizing_task, task) for task in tasks]
        for fut in as_completed(futures):
            label, trades = fut.result()
            if not trades.empty:
                frames_by_label[label].append(trades)
    return {
        label: (
            pd.concat(frames_by_label[label], ignore_index=True)
            if frames_by_label[label]
            else pd.DataFrame()
        )
        for label in labels
    }


def _summarise(trades: pd.DataFrame) -> pd.DataFrame:
    """Per-mode gross_R / net_R / cost_R / win / PF / avg cost."""
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for mode, df in trades.groupby("mode"):
        risk = df["risk_inr"].replace(0.0, np.nan)
        gross_r = (df["gross_pnl"] / risk)
        net_r = df["net_r"]
        wins = df[df["net_pnl"] > 0]["net_pnl"].sum()
        losses = -df[df["net_pnl"] < 0]["net_pnl"].sum()
        rows.append({
            "mode": mode,
            "trades": int(len(df)),
            "win": float((df["net_pnl"] > 0).mean()),
            "gross_R": float(gross_r.mean()),
            "net_R": float(net_r.mean()),
            "cost_R": float(gross_r.mean() - net_r.mean()),
            "PF": float(wins / losses) if losses > 0 else float("inf"),
            "avg_cost_inr": float(df["total_cost"].mean()),
            "avg_notional_inr": float((df["entry"] * df["quantity"]).mean()),
        })
    return pd.DataFrame(rows).sort_values("mode").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="output_models/core25_latest")
    ap.add_argument("--split", default="oos", choices=("oos", "train"))
    ap.add_argument("--notionals", default="50000,100000,200000",
                    help="comma-separated rupee notionals to test (plus qty=1 baseline)")
    ap.add_argument("--out", default="output_audit")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel worker processes for asset/sizing replay; use 8-12 on a 16-core VPS")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = _load_bundle(model_dir)
    cfg = Config()
    cost_cfg = ZerodhaEquityCostConfig()
    notionals = [float(x) for x in args.notionals.split(",") if x.strip()]

    schemes: List[tuple[str, Optional[float]]] = [("qty=1", None)]
    schemes += [(f"₹{int(n):,}", n) for n in notionals]

    all_rows = []
    print(f"[cost-sens] bundle: {model_dir/'multi_asset_report.pkl'}  split={args.split}")
    trades_by_label = _trades_for_schemes_parallel(
        report, cfg, cost_cfg, args.split, schemes, workers=args.workers
    )
    for label, _notional in schemes:
        trades = trades_by_label[label]
        summ = _summarise(trades)
        if summ.empty:
            print(f"\n[{label}] (no trades)")
            continue
        summ.insert(0, "sizing", label)
        all_rows.append(summ)
        print(f"\n================ SIZING: {label} ================")
        print(f"  {'mode':<22} {'trades':>7} {'win':>6} {'gross_R':>8} "
              f"{'net_R':>7} {'cost_R':>7} {'PF':>6} {'avg_cost':>9} {'avg_notional':>13}")
        for _, r in summ.iterrows():
            pf = "inf" if not np.isfinite(r["PF"]) else f"{r['PF']:.2f}"
            print(f"  {r['mode']:<22} {r['trades']:>7} {r['win']:>5.1%} "
                  f"{r['gross_R']:>+7.2f} {r['net_R']:>+6.2f} {r['cost_R']:>6.2f} "
                  f"{pf:>6} ₹{r['avg_cost_inr']:>7.2f} ₹{r['avg_notional_inr']:>11,.0f}")

    if all_rows:
        out = pd.concat(all_rows, ignore_index=True)
        path = out_dir / "cost_sensitivity.csv"
        out.to_csv(path, index=False)
        print(f"\n[cost-sens] wrote {path}")

        # Verdict: compare net_R at qty=1 vs the largest notional, and report the
        # gross ceiling — the number that decides if the strategy can ever win.
        base = out[out["sizing"] == "qty=1"].set_index("mode")
        big_label = schemes[-1][0]
        big = out[out["sizing"] == big_label].set_index("mode")
        print("\n================ VERDICT ================")
        worst_gross = float(out["gross_R"].max())
        print(f"  Best gross_R (cost-free ceiling) across all modes/sizings: {worst_gross:+.2f}R")
        for mode in base.index:
            if mode in big.index:
                d = big.loc[mode, "net_R"] - base.loc[mode, "net_R"]
                print(f"  {mode:<22} net_R {base.loc[mode,'net_R']:+.2f} (qty=1) "
                      f"-> {big.loc[mode,'net_R']:+.2f} ({big_label})  Δ={d:+.2f}R  "
                      f"gross_R={big.loc[mode,'gross_R']:+.2f}")
        if worst_gross <= 0:
            print("  => Even the cost-free ceiling is <= 0R. Sizing cannot help; "
                  "the strategy has no positive-R region. Structural change needed (R3/R4).")
        else:
            print("  => A positive gross_R region EXISTS. If net_R improves materially "
                  "with size, costs are the blocker and notional sizing is the fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
