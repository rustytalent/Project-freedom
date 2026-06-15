"""Run the live Kite WebSocket feed.

Wires KiteTicker (kiteconnect package) → LiveFeed → JSONL sink that
Sentinel's cockpit tails on every quote cycle.

Usage (Codex's launch command):

    python -m scripts.run_live_feed \\
        --api-key $KITE_API_KEY \\
        --access-token $KITE_ACCESS_TOKEN \\
        --bundle /var/lib/liqpool/multi_asset_report.pkl \\
        --instruments 256265 \\
        --asset NIFTY \\
        --jsonl /var/lib/sentinel/liqpool_live_signals.jsonl

Where:
  --instruments are Kite numeric instrument tokens (256265 = NIFTY 50
    spot; look up others via instruments dump). Multiple values
    space-separated.
  --jsonl is the path Sentinel's LiveSignalsTail reads. Default
    matches the cockpit's default journal location.

The process runs as a daemon: on KiteTicker disconnect, marks
reconnect, and Kite's client auto-reconnects. SIGINT triggers a clean
shutdown that flushes any open bar.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

from liqpool.live_feed import LiveFeed
from liqpool.live_inference import LiveInferenceBundle

LOG = logging.getLogger("scripts.run_live_feed")


def _audit_to_stderr(event: str, payload: dict) -> None:
    print(json.dumps({"event": event, "payload": payload}), file=sys.stderr,
           flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--api-key", required=True)
    p.add_argument("--access-token", required=True)
    p.add_argument("--bundle", type=Path, required=True,
                   help="trained MultiAssetReport pickle")
    p.add_argument("--instruments", nargs="+", type=int, required=True,
                   help="numeric Kite instrument tokens, space-separated")
    p.add_argument("--asset", default="NIFTY",
                   help="asset label for the published ModelSignals")
    p.add_argument("--jsonl", type=Path, required=True,
                   help="output JSONL; Sentinel tails this")
    p.add_argument("--timeframe-minutes", type=int, default=5)
    p.add_argument("--publish-floor", type=float, default=0.30)
    p.add_argument("--no-mis", action="store_true",
                   help="disable MIS session-window enforcement "
                        "(for replay/back-test against live ticks)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(name)s %(message)s")

    LOG.info("loading bundle from %s", args.bundle)
    bundle = LiveInferenceBundle.from_pickle(args.bundle)
    LOG.info("bundle loaded: version=%s, heads=%s",
             bundle.bundle_version, bundle.head_names())

    feed = LiveFeed(
        bundle=bundle,
        publisher_path=args.jsonl,
        asset=args.asset,
        timeframe_minutes=args.timeframe_minutes,
        publish_floor=args.publish_floor,
        enforce_mis=not args.no_mis,
        audit_sink=_audit_to_stderr,
    )

    # Lazy import so the module remains testable without kiteconnect
    try:
        from kiteconnect import KiteTicker
    except ImportError:
        LOG.error("kiteconnect not installed; pip install kiteconnect")
        return 2

    kt = KiteTicker(args.api_key, args.access_token)
    instrument_tokens = list(args.instruments)

    def on_connect(ws, _resp):
        LOG.info("ws connected; subscribing to %s", instrument_tokens)
        ws.subscribe(instrument_tokens)
        ws.set_mode(ws.MODE_FULL, instrument_tokens)
        feed.mark_back_running()

    def on_ticks(_ws, ticks):
        feed.on_ticks(ticks)

    def on_close(_ws, code, reason):
        LOG.warning("ws closed: code=%s reason=%s", code, reason)
        feed.mark_reconnect(f"code={code} reason={reason}")

    def on_error(_ws, code, reason):
        LOG.error("ws error: code=%s reason=%s", code, reason)
        feed.mark_reconnect(f"error code={code} reason={reason}")

    kt.on_connect = on_connect
    kt.on_ticks = on_ticks
    kt.on_close = on_close
    kt.on_error = on_error

    feed.start()
    LOG.info("starting KiteTicker (threaded=True); SIGINT to stop")
    kt.connect(threaded=True)

    stop = {"flag": False}
    def _shutdown(_signum, _frame):
        LOG.info("signal received; shutting down")
        stop["flag"] = True
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while not stop["flag"]:
            time.sleep(1)
            # print a heartbeat every 60s for ops visibility
            hb = feed.heartbeat
            if hb.n_ticks > 0 and hb.n_ticks % 1000 == 0:
                LOG.info("heartbeat: %s", hb.to_row())
    finally:
        try:
            kt.close()
        except Exception:
            pass
        feed.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
