"""Trailing-stop bot — terminal CLI.

The human in the loop. You open positions in the Zerodha app; when one
runs and you feel the greediness, you tap a command here and the bot
takes over watching it.

Usage:

    # Start the bot (long-running; leave it running in tmux/screen)
    python -m scripts.trailing_stop_cli serve

    # In another shell — arm a trail on a position you've opened
    python -m scripts.trailing_stop_cli arm \\
        --symbol NIFTY26JUN24500CE --side long --qty 75 \\
        --cushion 30 --premium 195

    # List active trails
    python -m scripts.trailing_stop_cli list

    # Cancel a trail (you want to ride further)
    python -m scripts.trailing_stop_cli cancel --trail-id TRAIL_NIFTY26JUN24500CE_1717920000123

    # Tighten / loosen the cushion mid-flight
    python -m scripts.trailing_stop_cli update --trail-id <id> --cushion 50

Safety:
  * The CLI defaults to DRY_RUN. Real orders go out ONLY when
    --confirm-real-orders is passed AND KITE_API_KEY +
    KITE_ACCESS_TOKEN are present in the environment.
  * Active trails are journalled to ~/.liqpool/trailing_stops.jsonl
    (override with --journal). A crash + restart resumes them.

The arm/list/cancel/update commands talk to the running `serve`
process through the journal file (the serve loop polls it on every
tick). v2 will use a unix socket; v1 keeps it filesystem-simple.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

from liqpool.broker_zerodha import (
    ZerodhaBroker,
    ZerodhaConfig,
    broker_from_env,
)
from liqpool.trading.trailing_stop_bot import (
    DEFAULT_POLL_SECONDS,
    TrailingStopBot,
    TrailJournal,
)


DEFAULT_JOURNAL = Path.home() / ".liqpool" / "trailing_stops.jsonl"


def _make_broker(dry_run: bool) -> ZerodhaBroker:
    broker = broker_from_env(dry_run=dry_run)
    if broker is not None:
        return broker
    # Fall back to a fully-simulated broker so the CLI still works
    # without credentials configured (useful for testing the flow).
    return ZerodhaBroker(ZerodhaConfig(
        api_key="DRY", access_token="DRY", dry_run=True,
    ))


def cmd_serve(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    broker = _make_broker(dry_run=not args.confirm_real_orders)
    bot = TrailingStopBot(
        broker=broker,
        journal_path=args.journal,
        poll_seconds=args.poll_seconds,
        confirm_real_orders=args.confirm_real_orders,
    )
    bot.start()
    print(f"trailing-stop bot running. journal={args.journal} "
          f"poll={args.poll_seconds}s confirm={args.confirm_real_orders}")
    print("Press Ctrl-C to stop (active trails persist in the journal).")
    try:
        while True:
            time.sleep(5)
            active = bot.list_active()
            if active:
                print(f"  [{pd.Timestamp.now('UTC').strftime('%H:%M:%S')} UTC] "
                      f"{len(active)} active: " +
                      ", ".join(f"{s.tradingsymbol}(peak={s.peak_premium:.2f},"
                                f"last={s.last_premium:.2f})"
                                for s in active))
    except KeyboardInterrupt:
        print("\nshutting down...")
        bot.stop()
    return 0


def cmd_arm(args: argparse.Namespace) -> int:
    # arm() writes to the journal; a separately-running `serve`
    # process picks it up on the next tick (or this same process
    # if the user is running both in one shell with --inline).
    broker = _make_broker(dry_run=not args.confirm_real_orders)
    bot = TrailingStopBot(
        broker=broker,
        journal_path=args.journal,
        confirm_real_orders=args.confirm_real_orders,
    )
    state = bot.arm(
        tradingsymbol=args.symbol,
        side=args.side,
        quantity=args.qty,
        cushion_rupees=args.cushion,
        current_premium=args.premium,
        exchange=args.exchange,
        note=args.note or "",
    )
    print(f"armed trail {state.trail_id}")
    print(f"  symbol={state.tradingsymbol} side={state.side} qty={state.quantity}")
    print(f"  cushion=Rs{state.cushion_rupees:.2f}  "
          f"activated@{state.activated_at_premium:.2f}")
    print("Make sure the `serve` loop is running, or this trail won't be watched.")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    journal = TrailJournal(args.journal)
    active = journal.replay_active()
    if not active:
        print("(no active trails)")
        return 0
    for s in active.values():
        print(f"{s.trail_id}")
        print(f"  {s.tradingsymbol} side={s.side} qty={s.quantity} "
              f"state={s.state}")
        print(f"  cushion=Rs{s.cushion_rupees:.2f}  "
              f"peak={s.peak_premium:.2f}  last={s.last_premium:.2f}")
        if s.note:
            print(f"  note: {s.note}")
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    broker = _make_broker(dry_run=True)
    bot = TrailingStopBot(broker=broker, journal_path=args.journal)
    # Pull the active set off disk so cancel() can act on it.
    bot._active = bot.journal.replay_active()
    ok = bot.cancel(args.trail_id)
    print("cancelled" if ok else "could not cancel (not found or already terminal)")
    return 0 if ok else 1


def cmd_update(args: argparse.Namespace) -> int:
    broker = _make_broker(dry_run=True)
    bot = TrailingStopBot(broker=broker, journal_path=args.journal)
    bot._active = bot.journal.replay_active()
    ok = bot.update_cushion(args.trail_id, args.cushion)
    print(f"cushion updated to Rs{args.cushion:.2f}" if ok
          else "could not update (not found or not in ARMED state)")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL)
    p.add_argument("--confirm-real-orders", action="store_true",
                   help="actually send orders (default: dry-run only)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="run the poll loop")
    s.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    s.set_defaults(func=cmd_serve)

    a = sub.add_parser("arm", help="start a trail on a position")
    a.add_argument("--symbol", required=True,
                   help="Zerodha tradingsymbol, e.g. NIFTY26JUN24500CE")
    a.add_argument("--side", required=True, choices=("long", "short"))
    a.add_argument("--qty", type=int, required=True)
    a.add_argument("--cushion", type=float, required=True,
                   help="rupee give-back from peak premium that fires the exit")
    a.add_argument("--premium", type=float, required=True,
                   help="the premium AT THE MOMENT YOU ARM THIS")
    a.add_argument("--exchange", default="NFO")
    a.add_argument("--note", default="")
    a.set_defaults(func=cmd_arm)

    sub.add_parser("list", help="list active trails").set_defaults(func=cmd_list)

    c = sub.add_parser("cancel", help="cancel an active trail")
    c.add_argument("--trail-id", required=True)
    c.set_defaults(func=cmd_cancel)

    u = sub.add_parser("update", help="update the cushion on a trail")
    u.add_argument("--trail-id", required=True)
    u.add_argument("--cushion", type=float, required=True)
    u.set_defaults(func=cmd_update)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
