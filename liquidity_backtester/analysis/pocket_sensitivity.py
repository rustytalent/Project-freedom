"""Pocket sensitivity — does the alpha exist on the *documented winning subsets*?

Background: after fixing intraday MIS enforcement in the v1 simulator, kitchen-
sink gross_R lifted from -0.13/-0.35 to -0.06/-0.26 across modes — borderline
positive on displacement_confirmed but still negative on the universe-wide
average. The `reports/phase4_track_a_focused_diagnostics.md` and
`phase4_multi_pocket_slice_analysis.md` already documented that the edge is
**concentrated in pockets**, not uniform:

  morning + 5-8 ATR        : n=3348  R=+0.668  PF=2.60
  midday + 5-8 ATR         : n=576   R=+0.722  PF=2.77
  AUTO sector              : n=194   R=+0.572  PF=2.88
  long-only ex BANKING/IT  : n=440   R=+0.360  PF=1.89

vs. the inverse:
  afternoon                : R=-0.000
  IT sector                : R=-0.155
  BANKING sector           : R=-0.162

This script loads the saved bundle (no retrain), filters the OOS pool
population by session × sector × side × factor combinations, and re-runs the
execution backtest with MIS active per pocket. Output answers a single
question: which (pocket, mode) combinations have a positive realized net_R
under honest MIS arithmetic AND realistic position sizing? Those are the
candidates for the next execution-gating decision.

Reuses :func:`liqpool.execution_backtest.simulate_execution_modes` so the MIS
fix flows through automatically. No new simulator logic.

Usage:
    PYTHONPATH=. .venv/bin/python analysis/pocket_sensitivity.py \\
        --model-dir output_models/core25_latest \\
        --notionals 50000,100000,200000 \\
        --out output_audit
"""
from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from liqpool.config import Config
from liqpool.costs import ZerodhaEquityCostConfig
from liqpool.execution_backtest import (
    EXECUTION_MODES,
    _headline_factor,
    simulate_execution_modes,
)
from liqpool.pools import Pool
from liqpool.regime import nse_session
from liqpool.sectors import sector_of
from liqpool.tester import PoolResult


# ---------------------------------------------------------------------------
# Pocket definitions
#
# Each pocket is a filter dict consumed by :func:`filter_pools`. Untouched
# pools are dropped when a "sessions" filter is active (no touch time -> no
# session attribution); pools without a touch are skipped by the simulator
# anyway, so this is a no-op when session filtering is off.
#
# Keep this list small and meaningful. Each pocket is a HYPOTHESIS about
# where the edge lives, drawn from the existing phase4 reports.
# ---------------------------------------------------------------------------
DEFAULT_POCKETS: Dict[str, Dict[str, Sequence[str]]] = {
    # Baseline — universe-wide, matches cost_sensitivity.
    "kitchen_sink": {},

    # Session-only slices.
    "morning":         {"sessions": ("morning",)},
    "midday":          {"sessions": ("midday",)},
    "afternoon":       {"sessions": ("closing",)},          # documented dead pocket
    "morning_midday":  {"sessions": ("morning", "midday")},

    # Sector-only slices.
    "auto_only":          {"sectors": ("AUTO",)},
    "favorable_sectors":  {"sectors": ("AUTO", "FMCG", "PHARMA")},
    "unfavorable_sectors": {"sectors": ("IT", "BANKING")},   # documented worst

    # Side-only. Pool.side encodes which side of price the zone sits on at
    # formation (low = demand below price, high = supply above price).
    "long_only":  {"sides": ("low",)},
    "short_only": {"sides": ("high",)},

    # The headline winning combo from the phase4 reports.
    "winning_combo": {
        "sessions": ("morning", "midday"),
        "sectors":  ("AUTO", "FMCG", "PHARMA"),
        "sides":    ("low",),
    },

    # The headline losing combo — sanity check; should be deeply negative.
    "losing_combo": {
        "sessions": ("closing",),
        "sectors":  ("IT", "BANKING"),
    },

    # Factor-conditional slices (subset, not exhaustive).
    "eqhl_only": {"factors": ("EQHL",)},
    "ob_only":   {"factors": ("OB",)},
}


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _pool_session(result: PoolResult) -> Optional[str]:
    """Session label at the pool's first-touch timestamp. None for untouched."""
    if result.touched_at is None:
        return None
    return nse_session(pd.Timestamp(result.touched_at))


def filter_pools(pools: Sequence[Pool], results: Sequence[PoolResult],
                 pocket: Dict[str, Sequence[str]]
                 ) -> Tuple[List[Pool], List[PoolResult]]:
    """Apply pocket criteria, return (filtered pools, filtered results).

    All criteria are AND-ed: a pool must match every key in `pocket`. A pool
    missing the data needed for a criterion (e.g. untouched pool when sessions
    filter active) is dropped.
    """
    sessions = set(pocket.get("sessions", ()))
    sectors  = set(pocket.get("sectors",  ()))
    sides    = set(pocket.get("sides",    ()))
    factors  = set(pocket.get("factors",  ()))

    keep_p: List[Pool] = []
    keep_r: List[PoolResult] = []
    for pool, result in zip(pools, results):
        if sessions:
            s = _pool_session(result)
            if s is None or s not in sessions:
                continue
        if sectors:
            if sector_of(pool.asset) not in sectors:
                continue
        if sides:
            if pool.side not in sides:
                continue
        if factors:
            if _headline_factor(pool) not in factors:
                continue
        keep_p.append(pool)
        keep_r.append(result)
    return keep_p, keep_r


# ---------------------------------------------------------------------------
# Sizing + simulation
# ---------------------------------------------------------------------------

def _per_asset_quantity(ad, notional: Optional[float]) -> int:
    """qty = floor(notional / last_close); fallback to 1."""
    if notional is None or notional <= 0:
        return 1
    try:
        price = float(ad.base_df["close"].iloc[-1])
    except Exception:
        return 1
    if price <= 0:
        return 1
    return max(1, int(math.floor(notional / price)))


def simulate_pocket(report, cfg: Config, cost_cfg: ZerodhaEquityCostConfig,
                    pocket: Dict[str, Sequence[str]],
                    notional: Optional[float],
                    split: str = "oos") -> pd.DataFrame:
    """Run simulate_execution_modes on each asset's filtered pool subset.

    MIS enforcement is inherited from simulate_pool_trade — no flag needed.
    Returns one trade per (mode, pool) that produced an entry.
    """
    frames: List[pd.DataFrame] = []
    for symbol, ad in report.assets.items():
        if split == "train":
            pools = ad.walkforward.train_pools_for_ml
            results = ad.walkforward.train_results_for_ml
        else:
            pools = ad.walkforward.oos_pools
            results = ad.walkforward.oos_results
        if not pools:
            continue
        f_pools, f_results = filter_pools(pools, results, pocket)
        if not f_pools:
            continue
        asset_cfg = getattr(ad, "final_cfg", None) or cfg
        qty = _per_asset_quantity(ad, notional)
        frames.append(simulate_execution_modes(
            df_base=ad.base_df, pools=f_pools, results=f_results,
            symbol=symbol, sector=sector_of(symbol),
            cfg=asset_cfg, cost_cfg=cost_cfg,
            modes=EXECUTION_MODES, quantity=qty,
        ))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class PocketRow:
    pocket: str
    sizing: str
    mode: str
    trades: int
    win: float
    gross_R: float
    net_R: float
    cost_R: float
    PF: float
    se_net_R: float
    ci95_lo: float
    ci95_hi: float
    avg_cost_inr: float
    avg_notional_inr: float


def _summarise_trades(pocket: str, sizing: str, trades: pd.DataFrame
                       ) -> List[PocketRow]:
    if trades.empty:
        return []
    rows: List[PocketRow] = []
    for mode, df in trades.groupby("mode"):
        risk = df["risk_inr"].replace(0.0, np.nan)
        gross_r_series = df["gross_pnl"] / risk
        net_r_series = df["net_r"]
        wins_sum = float(df.loc[df["net_pnl"] > 0, "net_pnl"].sum())
        losses_sum = float(-df.loc[df["net_pnl"] < 0, "net_pnl"].sum())
        n = int(len(df))
        net_r_arr = net_r_series.to_numpy(dtype=float)
        net_r_arr = net_r_arr[np.isfinite(net_r_arr)]
        se = float(net_r_arr.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        mean_net = float(net_r_arr.mean()) if len(net_r_arr) else 0.0
        rows.append(PocketRow(
            pocket=pocket, sizing=sizing, mode=mode,
            trades=n,
            win=float((df["net_pnl"] > 0).mean()),
            gross_R=float(gross_r_series.mean()),
            net_R=mean_net,
            cost_R=float(gross_r_series.mean() - mean_net),
            PF=float(wins_sum / losses_sum) if losses_sum > 0 else float("inf"),
            se_net_R=se,
            ci95_lo=mean_net - 1.96 * se,
            ci95_hi=mean_net + 1.96 * se,
            avg_cost_inr=float(df["total_cost"].mean()),
            avg_notional_inr=float((df["entry"] * df["quantity"]).mean()),
        ))
    return rows


def summarise_rows_to_frame(rows: List[PocketRow]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in rows])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_bundle(model_dir: Path):
    p = model_dir / "multi_asset_report.pkl"
    if not p.exists():
        raise SystemExit(f"missing bundle: {p}. Run --mode train first.")
    with p.open("rb") as f:
        return pickle.load(f)


def _print_pocket_block(label: str, rows: List[PocketRow]) -> None:
    if not rows:
        print(f"\n[{label}] (no trades)")
        return
    print(f"\n============== POCKET: {label} ==============")
    print(f"  {'mode':<22} {'sizing':<10} {'trades':>7} {'win':>6} "
          f"{'gross_R':>8} {'net_R':>7} {'95% CI net_R':>16} {'PF':>6}")
    for r in rows:
        pf_s = "inf" if not np.isfinite(r.PF) else f"{r.PF:.2f}"
        ci = f"[{r.ci95_lo:+.2f},{r.ci95_hi:+.2f}]"
        print(f"  {r.mode:<22} {r.sizing:<10} {r.trades:>7} {r.win:>5.1%} "
              f"{r.gross_R:>+7.2f} {r.net_R:>+6.2f} {ci:>16} {pf_s:>6}")


def _verdict(all_rows: List[PocketRow], min_trades: int = 50) -> List[str]:
    """Plain-language summary of which (pocket, mode, sizing) cells are
    positive net_R AND meet the minimum sample threshold."""
    lines: List[str] = []
    positives = [r for r in all_rows
                 if r.trades >= min_trades
                 and r.net_R > 0
                 and r.ci95_lo > 0]                       # CI lower bound > 0
    borderline = [r for r in all_rows
                  if r.trades >= min_trades
                  and r.net_R > 0
                  and r.ci95_lo <= 0 < r.ci95_hi]
    lines.append(f"Minimum sample threshold: n >= {min_trades}")
    lines.append(f"Statistically positive cells (CI lower bound > 0): {len(positives)}")
    for r in positives:
        lines.append(f"  + {r.pocket:<22} {r.mode:<22} {r.sizing:<10} "
                     f"n={r.trades} net_R={r.net_R:+.2f} CI=[{r.ci95_lo:+.2f},{r.ci95_hi:+.2f}] PF={r.PF:.2f}")
    lines.append(f"Borderline (positive mean but CI crosses zero): {len(borderline)}")
    for r in borderline[:10]:
        lines.append(f"  ? {r.pocket:<22} {r.mode:<22} {r.sizing:<10} "
                     f"n={r.trades} net_R={r.net_R:+.2f} CI=[{r.ci95_lo:+.2f},{r.ci95_hi:+.2f}] PF={r.PF:.2f}")
    if not positives and not borderline:
        lines.append("  (no positive cells at any sizing or pocket — even with MIS, "
                     "the strategy has no positive-net-R subset on this OOS data)")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default="output_models/core25_latest")
    ap.add_argument("--split", default="oos", choices=("oos", "train"))
    ap.add_argument("--notionals", default="50000,100000,200000",
                    help="comma-separated rupee notionals to test (plus qty=1 baseline)")
    ap.add_argument("--min-trades", type=int, default=50,
                    help="ignore cells with fewer trades than this in the verdict")
    ap.add_argument("--out", default="output_audit")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = _load_bundle(model_dir)
    cfg = Config()
    cost_cfg = ZerodhaEquityCostConfig()
    notionals: List[Optional[float]] = [None]            # qty=1 baseline
    notionals += [float(x) for x in args.notionals.split(",") if x.strip()]

    all_rows: List[PocketRow] = []
    print(f"[pockets] bundle: {model_dir/'multi_asset_report.pkl'}  split={args.split}")
    for pocket_name, pocket_def in DEFAULT_POCKETS.items():
        for notional in notionals:
            label = "qty=1" if notional is None else f"₹{int(notional):,}"
            trades = simulate_pocket(report, cfg, cost_cfg, pocket_def,
                                      notional=notional, split=args.split)
            rows = _summarise_trades(pocket_name, label, trades)
            all_rows.extend(rows)
        # Per-pocket print, aggregated across all sizings for compactness.
        pocket_rows = [r for r in all_rows if r.pocket == pocket_name]
        _print_pocket_block(pocket_name, pocket_rows)

    out_df = summarise_rows_to_frame(all_rows)
    out_csv = out_dir / "pocket_sensitivity.csv"
    out_df.to_csv(out_csv, index=False)

    print("\n================ VERDICT ================")
    for line in _verdict(all_rows, min_trades=args.min_trades):
        print(line)
    print(f"\n[pockets] wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
