#!/usr/bin/env python3
"""Export a distributable SaaS data file from raw predict outputs.

This is the offline counterpart to ``service/app.py``: instead of serving the
opaque feed over HTTP, it reads a predict-output directory (or a single raw
file) and writes the opaque, non-invertible feed to disk for distribution to a
customer. It reuses the ingestion engine (:mod:`liqpool.ingest`) and the opaque
scoring layer (:mod:`liqpool.scoring`) — it never touches model code.

The output is per-customer: ``G``/``D`` are watermarked with the ``--customer``
id and the feed ``--date`` so a leaked file is attributable and two customers
cannot cross-difference to recover the underlying analytics.

Usage:
    python scripts/export_saas_feed.py \
        --raw output_core25_predict_latest_tracka \
        --customer acme-capital \
        --date 2026-05-29 \
        --out dist/feed_acme_2026-05-29

Writes ``<out>.json`` (full envelope, grouped by instrument) and ``<out>.csv``
(flat one-row-per-observation), plus prints a short summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from liqpool.ingest import RawFeedScorer, read_raw_levels, _first, _SYMBOL_ALIASES
from liqpool.scoring import (
    ANALYTICS_TYPE,
    FEED_VERSION,
    INTERPRETATION_NOTE,
    PUBLIC_KEYS,
)
from liqpool.serving import handle_levels_request


def _symbols_in(source: str | Path) -> list[str]:
    rows = read_raw_levels(source)
    seen: list[str] = []
    for r in rows:
        sym = _first(r, _SYMBOL_ALIASES)
        if sym and str(sym) not in seen:
            seen.append(str(sym))
    return seen


def build_feed(raw: str, customer: str, date: str, scope: str = "s0") -> dict:
    scorer = RawFeedScorer(raw, as_of=date, scope=scope)
    symbols = _symbols_in(raw)
    instruments = []
    total = 0
    for sym in symbols:
        env = handle_levels_request(scorer, symbol=sym, date=date,
                                    customer_id=customer)
        instruments.append({
            "instrument": sym,
            "as_of": env["as_of"],
            "observation_count": env["observation_count"],
            "observations": env["observations"],
        })
        total += env["observation_count"]
    return {
        "analytics_type": ANALYTICS_TYPE,
        "feed_version": FEED_VERSION,
        "feed_date": date,
        "customer_id": customer,
        "instrument_count": len(instruments),
        "observation_count": total,
        "instruments": instruments,
        "interpretation_note": INTERPRETATION_NOTE,
    }


def write_csv(feed: dict, path: Path) -> None:
    import csv
    cols = ["instrument", "G", "D", "zone_low", "zone_high", "zone_mid",
            "scope", "as_of", "feed_version"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for inst in feed["instruments"]:
            for o in inst["observations"]:
                z = o["level_zone"]
                w.writerow([o["instrument"], o["G"], o["D"], z["low"], z["high"],
                            z["mid"], o["scope"], o["as_of"], o["feed_version"]])
        # Carry the disclaimer as a trailing comment row.
        w.writerow([])
        w.writerow(["# " + INTERPRETATION_NOTE])


def _assert_no_leak(feed: dict) -> None:
    """Hard guard: every observation must carry only the allow-listed keys."""
    allow = set(PUBLIC_KEYS)
    for inst in feed["instruments"]:
        for o in inst["observations"]:
            extra = set(o.keys()) - allow
            if extra:
                raise SystemExit(f"LEAK GUARD failed: unexpected keys {extra}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True,
                    help="predict-output directory or a single raw CSV/JSON file")
    ap.add_argument("--customer", required=True, help="customer id (watermark key)")
    ap.add_argument("--date", required=True, help="feed date YYYY-MM-DD (watermark + as_of)")
    ap.add_argument("--scope", default="s0", help="opaque scope code")
    ap.add_argument("--out", required=True, help="output path stem (no extension)")
    args = ap.parse_args()

    feed = build_feed(args.raw, args.customer, args.date, args.scope)
    _assert_no_leak(feed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json")
    csv_path = out.with_suffix(".csv")
    json_path.write_text(json.dumps(feed, indent=2), encoding="utf-8")
    write_csv(feed, csv_path)

    print(f"[export] customer={args.customer} date={args.date}")
    print(f"[export] instruments={feed['instrument_count']} "
          f"observations={feed['observation_count']}")
    print(f"[export] wrote {json_path}")
    print(f"[export] wrote {csv_path}")
    print("[export] leak guard: PASSED (only opaque allow-listed fields emitted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
