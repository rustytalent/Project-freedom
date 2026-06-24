#!/usr/bin/env python3
"""Run the standalone Auction State Graph replay dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from auction_state_graph import AuctionGraphConfig, run_auction_state_graph


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="run a synthetic replay with CE/PE and depth columns")
    mode.add_argument("--replay", help="CSV/parquet tick or OHLC replay file")
    parser.add_argument("--symbol", default="DEMO")
    parser.add_argument("--timeframe", default="1m", choices=["1m", "5m", "15m"])
    parser.add_argument("--max-candles", type=int, default=300)
    parser.add_argument("--bin-count", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--out", default="output_auction_state_graph")
    args = parser.parse_args()

    config = AuctionGraphConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        max_candles=args.max_candles,
        bin_count=args.bin_count,
        max_replay_rows=args.max_rows,
        dashboard_title=f"Auction State Graph - {args.symbol}",
    )
    summary = run_auction_state_graph(
        config=config,
        out_dir=Path(args.out),
        replay_path=args.replay,
        demo=args.demo,
    )
    print(json.dumps(summary["artifacts"], indent=2))
    print(f"dashboard: {summary['artifacts']['dashboard_html']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
