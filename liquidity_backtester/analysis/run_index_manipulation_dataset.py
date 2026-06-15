#!/usr/bin/env python3
"""Build a top-constituent Manipulation Atlas research dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from liqpool.research import (
    IndexStateBuildConfig,
    build_index_manipulation_dataset,
    parse_weights_text,
    read_weights_csv,
)
from liqpool.universe import symbols_for_universe
from liqpool.warehouse import WarehouseReader


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    else:
        raise ValueError(f"unsupported output format: {path}")


def _symbols_from_args(args) -> tuple[str, ...]:
    if args.symbols:
        return tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    if args.universe:
        return tuple(symbols_for_universe(args.universe))
    raise ValueError("provide --symbols or --universe")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build constituent rows and Manipulation Atlas states for index "
            "context research. Use this to test heavyweight masking, AVWAP "
            "defense, breadth, and trap-state ideas against forward returns."
        )
    )
    parser.add_argument("--warehouse-root", default=None)
    parser.add_argument("--symbols", help="comma-separated constituent symbols")
    parser.add_argument("--universe", default=None, help="named universe fallback, e.g. core25")
    parser.add_argument("--weights", default=None, help="SYMBOL=weight,SYMBOL2=weight")
    parser.add_argument("--weights-csv", default=None, help="CSV with symbol,weight columns")
    parser.add_argument("--index", default="NIFTY50")
    parser.add_argument("--tf", default="5m")
    parser.add_argument("--atr-window", type=int, default=14)
    parser.add_argument("--historical-window-bars", type=int, default=1500)
    parser.add_argument("--min-history-bars", type=int, default=100)
    parser.add_argument("--forward-bars", default="3,6,12,36,60")
    parser.add_argument("--min-constituents", type=int, default=5)
    parser.add_argument("--constituents-out", required=True)
    parser.add_argument("--states-out", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()

    symbols = _symbols_from_args(args)
    weights = {}
    weights.update(read_weights_csv(args.weights_csv))
    weights.update(parse_weights_text(args.weights))
    forward_bars = tuple(int(x.strip()) for x in args.forward_bars.split(",") if x.strip())

    cfg = IndexStateBuildConfig(
        symbols=symbols,
        weights=weights,
        index=args.index,
        tf=args.tf,
        atr_window=args.atr_window,
        historical_window_bars=args.historical_window_bars,
        min_history_bars=args.min_history_bars,
        forward_bars=forward_bars,
        min_constituents=args.min_constituents,
    )
    reader = WarehouseReader(root=args.warehouse_root)
    constituents, states, summary = build_index_manipulation_dataset(reader, cfg)

    _write_frame(constituents, Path(args.constituents_out))
    _write_frame(states, Path(args.states_out))
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"wrote constituents: {args.constituents_out} rows={len(constituents)}")
    print(f"wrote states:       {args.states_out} rows={len(states)}")
    print(f"wrote summary:      {args.summary_out}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
