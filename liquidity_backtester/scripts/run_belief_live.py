"""Run the Premium Belief Engine live from Kite quotes.

This is a standalone shadow runner. It does not place orders. It writes
Sentinel-compatible JSONL rows, so Sentinel can tail the file through
``LiveSignalsTail`` while the belief engine remains independently
operable.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from liqpool.research.belief.live_runner import (
    BeliefLiveConfig,
    DemoBeliefLiveRunner,
    runner_from_env,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--underlying", default="NIFTY")
    p.add_argument("--spot-key", default="NSE:NIFTY 50")
    p.add_argument("--exchange", default="NFO")
    p.add_argument("--strike-step", type=float, default=50.0)
    p.add_argument("--levels", type=int, default=5,
                   help="ATM +/- levels; 5 means 22 contracts")
    p.add_argument("--poll-seconds", type=float, default=1.0)
    p.add_argument("--min-quote-gap-seconds", type=float, default=1.05,
                   help="local guard for Kite quote rate limits")
    p.add_argument("--warmup-bars", type=int, default=80)
    p.add_argument("--refresh-contracts-every", type=int, default=30)
    p.add_argument("--out-jsonl", type=Path,
                   default=Path("/var/lib/sentinel/liqpool_live_signals.jsonl"))
    p.add_argument("--executor-out-jsonl", type=Path, default=None,
                   help="optional append-only JSONL with executor intents only")
    p.add_argument("--max-ticks", type=int, default=None,
                   help="test mode: stop after this many polling ticks")
    p.add_argument("--terminal", action="store_true",
                   help="render a live operator screen in this terminal")
    p.add_argument("--no-clear-terminal", action="store_true",
                   help="append terminal frames instead of clearing the screen")
    p.add_argument("--demo", action="store_true",
                   help="market-closed rehearsal mode with synthetic option quotes")
    p.add_argument("--demo-start-spot", type=float, default=23500.0)
    p.add_argument("--no-streaming-divergence", action="store_true")
    p.add_argument("--disable-executor", action="store_true",
                   help="emit raw Premium Belief signals without the shadow execution governor")
    p.add_argument("--executor-min-entry-confidence", type=float, default=0.66)
    p.add_argument("--executor-min-scalp-confidence", type=float, default=0.72)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = BeliefLiveConfig(
        underlying=args.underlying,
        spot_key=args.spot_key,
        exchange=args.exchange,
        strike_step=args.strike_step,
        levels=args.levels,
        poll_seconds=args.poll_seconds,
        min_quote_gap_seconds=args.min_quote_gap_seconds,
        warmup_bars=args.warmup_bars,
        refresh_contracts_every=args.refresh_contracts_every,
        output_jsonl=args.out_jsonl,
        executor_output_jsonl=args.executor_out_jsonl,
        max_ticks=args.max_ticks,
        include_streaming_divergence=not args.no_streaming_divergence,
        enable_executor=not args.disable_executor,
        executor_min_entry_confidence=args.executor_min_entry_confidence,
        executor_min_scalp_confidence=args.executor_min_scalp_confidence,
        terminal=args.terminal,
        clear_terminal=not args.no_clear_terminal,
        demo_start_spot=args.demo_start_spot,
    )
    try:
        runner = DemoBeliefLiveRunner(cfg=cfg) if args.demo else runner_from_env(cfg)
        runner.run_forever()
    except KeyboardInterrupt:
        return 130
    except Exception:
        logging.getLogger(__name__).exception("belief live runner failed")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
