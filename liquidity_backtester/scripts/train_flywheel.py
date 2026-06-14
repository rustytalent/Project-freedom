"""Train every flywheel organ from accumulated artifacts and save the hub.

The nightly metabolism step: digest the bundle + shadow log + outcome
log into training frames (extractors), feed every organ
(hub.fit_from_artifacts), persist the hub, and report which organs ate
vs starved.

Usage:
    python -m scripts.train_flywheel \\
        --bundle path/to/multi_asset_report.pkl \\
        --shadow-root data/shadow_log \\
        --outcome-root data/outcome_log \\
        --bundle-fit-date 2026-06-01 \\
        --out models/flywheel

Every input is optional — organs whose food is missing simply stay
unfit (and the serving sites degrade to today's behaviour).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from liqpool.flywheel.hub import FlywheelHub


def _load_bundle(path: Optional[Path]):
    if path is None or not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_shadow_joined(
    root: Optional[Path],
    sentinel_journal: Optional[Path] = None,
    sentinel_export: Optional[Path] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Read liqpool's own shadow_log AND Sentinel's exported decisions
    (Wave 15 cross-codebase integration). Either side may be missing;
    the trainer's "safe-unfit" pattern handles None gracefully."""
    parts: list = []
    if root is not None and root.exists():
        from liqpool.products.shadow_log import join_events_to_resolutions
        own = join_events_to_resolutions(root)
        if own is not None and not own.empty:
            parts.append(own)
    if sentinel_journal is not None or sentinel_export is not None:
        from liqpool.sentinel_adapter import load_sentinel_as_shadow_frame
        sent = load_sentinel_as_shadow_frame(
            journal_dir=sentinel_journal, export_dir=sentinel_export,
            since=since, until=until)
        if sent is not None and not sent.empty:
            parts.append(sent)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def _load_outcome_joined(root: Optional[Path]) -> Optional[pd.DataFrame]:
    """All-dates join: walk every prediction partition, join each
    date via the writer's per-date reader, concat."""
    if root is None or not root.exists():
        return None
    from liqpool.products.outcome_log import OutcomeLogWriter
    pred_root = root / "predictions"
    if not pred_root.exists():
        return None
    writer = OutcomeLogWriter(root)
    parts = []
    for d in sorted(pred_root.glob("trading_date_ist=*")):
        date = d.name.split("=", 1)[1]
        try:
            j = writer.read_joined(date)
            if not j.empty:
                parts.append(j)
        except Exception:
            continue
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bundle", type=Path, default=None)
    p.add_argument("--shadow-root", type=Path, default=None)
    p.add_argument("--outcome-root", type=Path, default=None)
    p.add_argument("--sentinel-journal", type=Path, default=None,
                   help="Sentinel <journal>/ledger_<date>.jsonl directory — "
                        "the live decision stream becomes training food too")
    p.add_argument("--sentinel-export", type=Path, default=None,
                   help="Sentinel exported decision_events directory "
                        "(<root>/sentinel_live/session=<date>/...)")
    p.add_argument("--since", default=None,
                   help="YYYY-MM-DD lower bound for sentinel rows")
    p.add_argument("--until", default=None,
                   help="YYYY-MM-DD upper bound for sentinel rows")
    p.add_argument("--drift-history", type=Path, default=None,
                   help="CSV of chronological run metrics + drift_fired")
    p.add_argument("--bundle-fit-date", default=None,
                   help="YYYY-MM-DD the serving bundle was trained "
                        "(enables the bucket-aging organ)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    drift_history = None
    if args.drift_history is not None and args.drift_history.exists():
        drift_history = pd.read_csv(args.drift_history)

    hub = FlywheelHub()
    fitted = hub.fit_from_artifacts(
        report=_load_bundle(args.bundle),
        shadow_joined=_load_shadow_joined(
            args.shadow_root,
            sentinel_journal=args.sentinel_journal,
            sentinel_export=args.sentinel_export,
            since=args.since, until=args.until),
        outcome_joined=_load_outcome_joined(args.outcome_root),
        drift_history=drift_history,
        bundle_fit_date=args.bundle_fit_date,
    )
    hub.save(args.out)
    print(json.dumps({"saved_to": str(args.out), "organs": fitted}, indent=2))
    starved = [k for k, v in fitted.items() if not v]
    if starved:
        print(f"starved organs (insufficient food, safe-unfit): {starved}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
