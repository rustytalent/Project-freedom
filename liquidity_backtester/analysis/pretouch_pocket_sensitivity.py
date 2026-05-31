"""Pre-touch (Track A) pocket sensitivity — CI-aware verdict on the existing sweep.

The Track A sweep ``analysis/run_phase4_track_a_pretouch_sweep.py`` already
produces a parquet of pre-touch directional trades with intraday session-end
exits at 15:10 IST and v2 cost/slippage realism. ``focused_diagnostics.py``
already reports per-pocket point estimates + DSR. **What is missing is a 95%
confidence interval on ``net_r`` per pocket so the user can read a pocket as
statistically positive, borderline, or dead.**

This script fills that gap. It consumes the existing
``pretouch_sweep_trades.parquet`` (no simulation), applies the named pockets
the prior reports identified, and prints per-pocket ``trades / win / mean R /
CI95 / PF / avg cost`` plus an explicit verdict. Mirrors the Track B verdict
script ``analysis/pocket_sensitivity.py`` so the two are directly comparable.

Run order:

1. Build the trade parquet (one heavy backtest, MIS-correct by construction):

       PYTHONPATH=. .venv/bin/python analysis/run_phase4_track_a_pretouch_sweep.py \\
           --model-report output_models/core25_latest/multi_asset_report.pkl \\
           --feature-store output_feature_store/core25 \\
           --out-dir output_phase4_track_a_pretouch_sweep \\
           --session-exit-time 15:10

2. Read the verdict (this script, fast):

       PYTHONPATH=. .venv/bin/python analysis/pretouch_pocket_sensitivity.py \\
           --trades output_phase4_track_a_pretouch_sweep/pretouch_sweep_trades.parquet \\
           --out output_audit

The trades parquet already has ``net_r`` per trade. Costs are baked in.
Filtering and aggregating is pure pandas — no simulator dependency.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pockets — the same hypotheses Track B was tested against, mapped onto the
# Track A trade columns. The sweep already filtered to 3-8 ATR; we don't
# re-filter that here.
# ---------------------------------------------------------------------------
DEFAULT_POCKETS: Dict[str, Dict[str, Sequence[str]]] = {
    "kitchen_sink": {},                                          # all sweep trades

    # Time-bucket only (sweep emits "morning"/"midday"/"afternoon").
    "morning":         {"time_buckets": ("morning",)},
    "midday":          {"time_buckets": ("midday",)},
    "afternoon":       {"time_buckets": ("afternoon",)},
    "morning_midday":  {"time_buckets": ("morning", "midday")},

    # Sector only.
    "auto_only":           {"sectors": ("AUTO",)},
    "favorable_sectors":   {"sectors": ("AUTO", "FMCG", "PHARMA")},
    "unfavorable_sectors": {"sectors": ("IT", "BANKING")},

    # Direction only (Track A is long-biased; UP = pool above price).
    "long_only":  {"directions": ("UP",)},
    "short_only": {"directions": ("DOWN",)},

    # The headline winning combo from phase 4 reports.
    "winning_combo": {
        "time_buckets": ("morning", "midday"),
        "sectors":      ("AUTO", "FMCG", "PHARMA"),
        "directions":   ("UP",),
    },

    # Sanity-check losing combo.
    "losing_combo": {
        "time_buckets": ("afternoon",),
        "sectors":      ("IT", "BANKING"),
    },
}


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_trades(trades: pd.DataFrame, pocket: Dict[str, Sequence[str]]
                  ) -> pd.DataFrame:
    """AND together the pocket criteria. Returns a filtered copy."""
    df = trades
    if "time_buckets" in pocket and "time_bucket" in df.columns:
        df = df[df["time_bucket"].isin(pocket["time_buckets"])]
    if "sectors" in pocket and "sector" in df.columns:
        df = df[df["sector"].isin(pocket["sectors"])]
    if "directions" in pocket and "direction" in df.columns:
        df = df[df["direction"].isin(pocket["directions"])]
    if "factors" in pocket and "factor" in df.columns:
        df = df[df["factor"].isin(pocket["factors"])]
    if "q_cohort" in pocket and "pool_quality" in df.columns:
        # Top-K percentile cohort on pool_quality (within the filtered subset).
        frac = float(pocket["q_cohort"][0])           # e.g. ("top_25pct",) -> 0.25
        cutoff = df["pool_quality"].quantile(1.0 - frac)
        df = df[df["pool_quality"] >= cutoff]
    return df


# ---------------------------------------------------------------------------
# Aggregation (CI-aware; same shape as Track B verdict script).
# ---------------------------------------------------------------------------

@dataclass
class PocketRow:
    pocket: str
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


def _summarise(pocket_name: str, df: pd.DataFrame) -> Optional[PocketRow]:
    if df.empty:
        return None
    risk = df["risk_inr"].replace(0.0, np.nan)
    gross_r = df["gross_pnl"] / risk
    net_r = df["net_r"]
    wins_sum = float(df.loc[df["net_pnl"] > 0, "net_pnl"].sum())
    losses_sum = float(-df.loc[df["net_pnl"] < 0, "net_pnl"].sum())
    n = int(len(df))
    arr = net_r.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    mean_net = float(arr.mean()) if len(arr) else 0.0
    se = float(arr.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    return PocketRow(
        pocket=pocket_name,
        trades=n,
        win=float((df["net_pnl"] > 0).mean()),
        gross_R=float(gross_r.mean()),
        net_R=mean_net,
        cost_R=float(gross_r.mean() - mean_net),
        PF=float(wins_sum / losses_sum) if losses_sum > 0 else float("inf"),
        se_net_R=se,
        ci95_lo=mean_net - 1.96 * se,
        ci95_hi=mean_net + 1.96 * se,
        avg_cost_inr=float(df["total_cost"].mean()),
    )


def _print_pocket(row: PocketRow) -> None:
    pf_s = "inf" if not np.isfinite(row.PF) else f"{row.PF:.2f}"
    ci = f"[{row.ci95_lo:+.2f},{row.ci95_hi:+.2f}]"
    print(f"  {row.pocket:<22} {row.trades:>7} {row.win:>5.1%} "
          f"{row.gross_R:>+7.2f} {row.net_R:>+6.2f} {ci:>16} {pf_s:>6} "
          f"₹{row.avg_cost_inr:>7.2f}")


def _verdict(rows: List[PocketRow], min_trades: int) -> List[str]:
    lines: List[str] = []
    positives = [r for r in rows
                 if r.trades >= min_trades and r.net_R > 0 and r.ci95_lo > 0]
    borderline = [r for r in rows
                  if r.trades >= min_trades and r.net_R > 0 and r.ci95_lo <= 0 < r.ci95_hi]
    lines.append(f"Minimum sample threshold: n >= {min_trades}")
    lines.append(f"Statistically positive pockets (CI lower bound > 0): {len(positives)}")
    for r in positives:
        lines.append(f"  + {r.pocket:<22} n={r.trades} net_R={r.net_R:+.2f} "
                     f"CI=[{r.ci95_lo:+.2f},{r.ci95_hi:+.2f}] PF={r.PF:.2f}")
    lines.append(f"Borderline (positive mean, CI crosses zero): {len(borderline)}")
    for r in borderline:
        lines.append(f"  ? {r.pocket:<22} n={r.trades} net_R={r.net_R:+.2f} "
                     f"CI=[{r.ci95_lo:+.2f},{r.ci95_hi:+.2f}] PF={r.PF:.2f}")
    if not positives and not borderline:
        lines.append("  (no positive pockets at any size — Track A pre-touch does not "
                     "show a statistically defensible edge on the current sweep parquet)")
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades",
        default="output_phase4_track_a_pretouch_sweep/pretouch_sweep_trades.parquet",
        help="path to the pretouch_sweep_trades.parquet produced by run_phase4_track_a_pretouch_sweep.py")
    ap.add_argument("--min-trades", type=int, default=50,
        help="exclude pockets with fewer trades from the verdict")
    ap.add_argument("--out", default="output_audit")
    args = ap.parse_args()

    trades_path = Path(args.trades).expanduser()
    if not trades_path.exists():
        raise SystemExit(
            f"missing trade parquet: {trades_path}\n"
            "Run the sweep first:\n"
            "  PYTHONPATH=. .venv/bin/python analysis/run_phase4_track_a_pretouch_sweep.py \\\n"
            "    --model-report output_models/core25_latest/multi_asset_report.pkl \\\n"
            "    --feature-store output_feature_store/core25 \\\n"
            "    --session-exit-time 15:10"
        )

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = pd.read_parquet(trades_path)
    print(f"[pretouch-pockets] loaded {len(trades):,} trades from {trades_path}")
    required_cols = {"net_r", "gross_pnl", "net_pnl", "risk_inr", "total_cost",
                     "time_bucket", "sector", "direction"}
    missing = required_cols - set(trades.columns)
    if missing:
        raise SystemExit(f"sweep parquet missing required columns: {sorted(missing)}")

    # Header
    print(f"\n{'pocket':<22} {'trades':>7} {'win':>6} "
          f"{'gross_R':>8} {'net_R':>7} {'95% CI net_R':>16} {'PF':>6} {'avg_cost':>9}")

    rows: List[PocketRow] = []
    for name, pocket in DEFAULT_POCKETS.items():
        sub = filter_trades(trades, pocket)
        row = _summarise(name, sub)
        if row is not None:
            rows.append(row)
            _print_pocket(row)
        else:
            print(f"  {name:<22} (no trades after filter)")

    out_df = pd.DataFrame([r.__dict__ for r in rows])
    out_csv = out_dir / "pretouch_pocket_sensitivity.csv"
    out_df.to_csv(out_csv, index=False)

    print("\n================ VERDICT ================")
    for line in _verdict(rows, min_trades=args.min_trades):
        print(line)
    print(f"\n[pretouch-pockets] wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
