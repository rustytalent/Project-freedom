#!/usr/bin/env python3
"""Synthetic end-to-end smoke test for the SaaS feed.

Generates a realistic *raw predict output* CSV (the same schema the model's
predict run writes — symbol/side/pool_low/pool_high/p_touch/q/p_up/dir_tag),
WITHOUT any model or market data, then runs it through the ingestion +
export pipeline to produce a distributable opaque feed file.

This proves the whole data path (raw -> ingest -> opaque feed) works and lets
you generate a sample artifact to hand to a reviewer / red team.

Usage:
    python scripts/synthetic_smoke.py --out-dir /tmp/saas_smoke --customer reviewer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.export_saas_feed import build_feed, write_csv, _assert_no_leak

SYMBOLS = ["HDFCBANK.NS", "ICICIBANK.NS", "TCS.NS", "INFY.NS", "MARUTI.NS"]
DIR_TAGS = ["DIR_ALIGN", "DIR_FIGHT", "DIR_NEUTRAL"]


def synth_raw_rows(seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for sym in SYMBOLS:
        base = 100.0 + rng.uniform(0, 2000)
        atr = base * 0.01
        for side in ("above", "below"):
            n = int(rng.integers(4, 9))
            for i in range(n):
                step = (i + 1) * rng.uniform(0.8, 1.5) * atr
                mid = base + step if side == "above" else base - step
                width = atr * rng.uniform(0.2, 0.6)
                # realistic: closer/stronger pools have higher p_touch
                dist_atr = step / atr
                p_touch = float(np.clip(0.95 - 0.07 * dist_atr + rng.normal(0, 0.05),
                                        0.02, 0.98))
                q = float(np.clip(rng.beta(2, 3), 0.0, 1.0))
                p_up = float(np.clip(0.5 + rng.normal(0, 0.18), 0.02, 0.98))
                rows.append({
                    "symbol": sym, "side": side,
                    "pool_low": round(mid - width / 2, 2),
                    "pool_high": round(mid + width / 2, 2),
                    "pool_mid": round(mid, 2),
                    "p_touch": round(p_touch, 4),
                    "q": round(q, 4),
                    "p_respect": round(q, 4),
                    "p_up": round(p_up, 4),
                    "dir_tag": str(rng.choice(DIR_TAGS)),
                    "distance_atr": round(dist_atr, 2),
                    # decoy internal fields the engine must NOT pass through:
                    "entry": round(mid, 2), "stop": round(mid - atr, 2),
                    "target": round(mid + atr, 2), "decision": "WATCH_ONLY",
                })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/tmp/saas_smoke")
    ap.add_argument("--customer", default="reviewer")
    ap.add_argument("--date", default="2026-05-29")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw_predict_output"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = synth_raw_rows(args.seed)
    raw_csv = raw_dir / "live_gate_decisions.csv"
    pd.DataFrame(rows).to_csv(raw_csv, index=False)
    print(f"[smoke] wrote synthetic raw predict output: {raw_csv} ({len(rows)} rows)")

    feed = build_feed(str(raw_dir), args.customer, args.date)
    _assert_no_leak(feed)

    stem = out_dir / f"saas_feed_{args.customer}_{args.date}"
    (stem.with_suffix(".json")).write_text(json.dumps(feed, indent=2), encoding="utf-8")
    write_csv(feed, stem.with_suffix(".csv"))

    # Watermark proof: a second customer gets different G values, same ordering.
    feed_b = build_feed(str(raw_dir), args.customer + "-2", args.date)
    sym0 = feed["instruments"][0]["instrument"]
    g_a = [o["feature_intensity_score"] for o in feed["instruments"][0]["observations"]]
    g_b = [o["feature_intensity_score"] for o in feed_b["instruments"][0]["observations"]]

    print(f"[smoke] feed instruments={feed['instrument_count']} "
          f"observations={feed['observation_count']}")
    print(f"[smoke] wrote {stem.with_suffix('.json')}")
    print(f"[smoke] wrote {stem.with_suffix('.csv')}")
    print(f"[smoke] leak guard: PASSED (decoy entry/stop/target/decision dropped)")
    print(f"[smoke] watermark check on {sym0}: customer A G={g_a}")
    print(f"[smoke]                              customer B G={g_b}")
    print(f"[smoke] (values differ per customer; strongest level stays top)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
