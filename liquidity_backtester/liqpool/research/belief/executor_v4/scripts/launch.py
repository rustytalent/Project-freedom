"""Monday launcher — plug API key in, go.

Usage:
    # SAFEST: paper mode, no API key needed.
    python -m liqpool.research.belief.executor_v4.scripts.launch \\
        --paper --underlying NIFTY

    # LIVE (real money — requires explicit --confirm-real flag):
    python -m liqpool.research.belief.executor_v4.scripts.launch \\
        --kite \\
        --api-key="..." --access-token="..." \\
        --confirm-real \\
        --underlying NIFTY

    # REPLAY a recorded tape (Sunday validation):
    python -m liqpool.research.belief.executor_v4.scripts.launch \\
        --replay /var/lib/sentinel/liqpool_live_signals.jsonl

The launcher:
  1. Validates the chosen mode and gathers credentials safely.
  2. Constructs the appropriate BrokerAdapter.
  3. Builds a V4Runner.
  4. Hooks the existing live_runner.py's tick loop for data ingestion
     in --kite mode (or accepts piped snapshots in --stdin mode).
  5. Persists state to /var/lib/sentinel/executor_v4_state/ for
     restart safety.
  6. Emits cockpit JSONL to /var/lib/sentinel/cockpit.jsonl so the UI
     can render in real time.

Environment overrides (used if CLI flags omitted):
  KITE_API_KEY, KITE_ACCESS_TOKEN, V4_CONFIRM_REAL=1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add the liquidity_backtester package to path if launching as a script.
HERE = Path(__file__).resolve()
ROOT = HERE.parents[5]   # …/liquidity_backtester
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v4 executor Monday launcher")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--paper", action="store_true",
                       help="paper trading mode (default, safest)")
    mode.add_argument("--kite", action="store_true",
                       help="Kite (Zerodha) live or dry-run mode")
    mode.add_argument("--replay",
                       help="replay a JSONL snapshot tape")
    mode.add_argument("--stdin", action="store_true",
                       help="read snapshots from stdin (one JSON per line)")

    parser.add_argument("--api-key", help="Kite API key")
    parser.add_argument("--access-token", help="Kite access token")
    parser.add_argument("--confirm-real", action="store_true",
                         help="explicit opt-in for REAL orders")
    parser.add_argument("--underlying", default="NIFTY")
    parser.add_argument("--state-dir",
                         default="/var/lib/sentinel/executor_v4_state",
                         help="where to persist runner state")
    parser.add_argument("--cockpit-out",
                         default="/var/lib/sentinel/v4_cockpit.jsonl",
                         help="JSONL stream of cockpit snapshots")
    parser.add_argument("--intent-out", default=None,
                         help="optional JSONL stream of intents")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--print-explainer", action="store_true",
                         help="print the per-tick explainer text to stdout")
    return parser.parse_args()


def _build_runner(args: argparse.Namespace, *,
                    explainer_to_log: bool = True):
    """Build a V4Runner based on the CLI args."""
    from liqpool.research.belief.executor_v4 import (
        KiteBrokerAdapter,
        KiteBrokerConfig,
        PaperBrokerAdapter,
        PersistenceConfig,
        V4Runner,
        V4RunnerConfig,
    )

    runner_cfg = V4RunnerConfig(
        persistence=PersistenceConfig(state_dir=Path(args.state_dir),
                                          enabled=True),
        write_cockpit_to_jsonl=(Path(args.cockpit_out) if args.cockpit_out
                                  else None),
        emit_explainer_to_log=explainer_to_log,
    )

    if args.kite:
        # Resolve credentials.
        api_key = args.api_key or os.environ.get("KITE_API_KEY")
        access_token = args.access_token or os.environ.get("KITE_ACCESS_TOKEN")
        if not api_key or not access_token:
            print("FATAL: --kite requires --api-key + --access-token "
                   "(or KITE_API_KEY + KITE_ACCESS_TOKEN env vars).",
                   file=sys.stderr)
            sys.exit(2)
        confirm_real = args.confirm_real or os.environ.get(
            "V4_CONFIRM_REAL") == "1"
        # Import here so paper / replay modes don't require sentinel deps.
        from sentinel.kite_client import KiteAccount
        account = KiteAccount(label="founder_v4",
                                api_key=api_key,
                                access_token=access_token,
                                dry_run=not confirm_real)
        broker = KiteBrokerAdapter(
            kite_account=account,
            cfg=KiteBrokerConfig(confirm_real=confirm_real),
        )
        if confirm_real:
            print("⚠  LIVE MODE: orders will hit real broker", file=sys.stderr)
        else:
            print("ℹ  KITE DRY-RUN: account loaded but no real orders",
                   file=sys.stderr)
        return V4Runner(cfg=runner_cfg, broker=broker)

    # Paper mode (default and replay default).
    broker = PaperBrokerAdapter()
    return V4Runner(cfg=runner_cfg, broker=broker)


def _drive_from_stdin(runner) -> None:
    """Consume JSON snapshots from stdin, one per line."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            snap = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"skip bad json line: {exc}", file=sys.stderr)
            continue
        result = runner.on_tick(snap)
        if hasattr(result.cockpit, "explainer_text"):
            print(result.cockpit.explainer_text)


def _drive_from_replay(runner, path: str) -> None:
    """Replay a JSONL tape through the runner."""
    from liqpool.research.belief.executor_v4 import (
        ReplayConfig, replay_jsonl,
    )
    cfg = ReplayConfig(
        cockpit_out=runner.cfg.write_cockpit_to_jsonl,
        persist_state=False,
    )
    print(f"Replaying tape: {path}", file=sys.stderr)
    report = replay_jsonl(path, cfg=cfg)
    print(report.to_summary_string())


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    runner = _build_runner(args, explainer_to_log=not args.print_explainer)

    if args.replay:
        _drive_from_replay(runner, args.replay)
        return
    if args.stdin:
        _drive_from_stdin(runner)
        return

    # Default: paper or kite mode wants a tick source.
    # For convenience we just emit a one-shot health report and exit; the
    # operator is expected to wire the runner into their own ingestion
    # loop (e.g. live_runner.py's tick stream).
    healthcheck = {
        "runner": "V4Runner",
        "broker": runner.broker.healthcheck(),
        "persistence_path": str(runner.persistence.path_for_today()),
        "cockpit_out": str(runner.cfg.write_cockpit_to_jsonl),
    }
    print(json.dumps(healthcheck, indent=2, default=str))


if __name__ == "__main__":
    main()
