"""Run the first profitability hypothesis miner.

This runner is intentionally generic: point it at any CSV/Parquet frame
that already contains feature columns plus a forward return column. It
samples transparent AND-rule hypotheses, ranks them, and writes the
candidate table. These are not live alphas; they are research candidates
for the next validation layer.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from liqpool.research.hypothesis_miner import rank_random_hypotheses


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise SystemExit(f"unsupported input format: {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV/Parquet feature frame")
    ap.add_argument("--forward-col", required=True, help="forward net/gross R column to score")
    ap.add_argument("--cost-col", default=None, help="optional per-row cost in R units")
    ap.add_argument("--features", default="", help="comma-separated feature columns; default=all numeric")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frame = _read_frame(Path(args.input))
    features = [x.strip() for x in args.features.split(",") if x.strip()] or None
    ranked = rank_random_hypotheses(
        frame,
        forward_return_col=args.forward_col,
        cost_r_col=args.cost_col,
        n=args.samples,
        seed=args.seed,
        feature_columns=features,
        min_trades=args.min_trades,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ranked.head(args.top).to_csv(out, index=False)
    print(f"[profitability-miner] input rows={len(frame):,} sampled={args.samples:,}")
    print(f"[profitability-miner] wrote {out} rows={min(len(ranked), args.top):,}")
    if not ranked.empty:
        print(ranked.head(min(12, args.top)).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

