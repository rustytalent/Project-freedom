#!/usr/bin/env python3
"""Classify top-constituent market state with the Manipulation Atlas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from liqpool.research import (
    StateDatasetConfig,
    build_manipulation_state_frame,
    summarize_state_frame,
)


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported input format: {path}")


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
        return
    if suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
        return
    raise ValueError(f"unsupported output format: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert constituent-level rows into Manipulation Atlas states. "
            "Input should contain one row per constituent per group."
        )
    )
    parser.add_argument("--input", required=True, help="CSV/parquet constituent state input")
    parser.add_argument("--out", required=True, help="CSV/parquet classified state output")
    parser.add_argument("--summary-out", help="JSON summary output")
    parser.add_argument(
        "--group-cols",
        default="ts",
        help="comma-separated grouping columns, e.g. ts or trading_date,bar_time",
    )
    parser.add_argument("--symbol-col", default="symbol")
    parser.add_argument("--weight-col", default="weight")
    parser.add_argument("--return-col", default="return_pct")
    parser.add_argument("--today-avwap-col", default="today_avwap_dist_atr")
    parser.add_argument("--prev-avwap-col", default="prev_session_avwap_dist_atr")
    parser.add_argument("--historical-position-col", default="historical_position_pct")
    parser.add_argument("--gap-col", default="gap_pct")
    parser.add_argument("--index-return-col", default="index_return_pct")
    parser.add_argument("--breadth-col", default="breadth_positive_frac")
    parser.add_argument("--min-constituents", type=int, default=5)
    args = parser.parse_args()

    cfg = StateDatasetConfig(
        group_cols=tuple(x.strip() for x in args.group_cols.split(",") if x.strip()),
        symbol_col=args.symbol_col,
        weight_col=args.weight_col,
        return_col=args.return_col,
        today_avwap_col=args.today_avwap_col,
        prev_avwap_col=args.prev_avwap_col,
        historical_position_col=args.historical_position_col,
        gap_col=args.gap_col,
        index_return_col=args.index_return_col or None,
        breadth_col=args.breadth_col or None,
        min_constituents=args.min_constituents,
    )

    source = Path(args.input)
    out_path = Path(args.out)
    frame = _read_frame(source)
    states = build_manipulation_state_frame(frame, cfg)
    _write_frame(states, out_path)
    summary = summarize_state_frame(states)

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"wrote {out_path} rows={len(states)}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
