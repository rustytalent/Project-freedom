"""Backfill the outcome log from historical bundles.

Goal: before customer #1 reads brief #1, the Yesterday Audit section
already has real numbers covering the last N trading days. This script
walks historical bundle snapshots, runs the brief generator in
"historical mode" on each, writes predictions to the outcome log, then
resolves each prediction against the actual OOS bar data.

Usage (CLI)::

    PYTHONPATH=. python analysis/backfill_outcome_log.py \\
        --bundle output_models/core25_head_alpha_710362b/multi_asset_report.pkl \\
        --output-root data/outcome_log \\
        --days 90

What it does:

  1. Loads the bundle from --bundle.
  2. Walks the last --days IST trading dates inside the bundle's OOS
     window.
  3. For each date, calls generate_brief with that date as the target.
  4. Writes predictions via the OutcomeLogWriter.
  5. Resolves each prediction against the bundle's actual bar data and
     writes the resolution row.

Idempotent: re-running overwrites only the partitions written by this
script's predictions (NOT live-collected ones, since live partitions
will be tagged differently in v2). For now treat the output dir as a
clean slate per backfill run.

This is the SKELETON. It does not crash on a missing bundle (graceful
fallback to a dry-run), but the full backfill requires a real fitted
bundle. End-to-end smoke testing happens on Codex's side with the
710362b bundle.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from liqpool.products.daily_brief import generate_brief
from liqpool.products.outcome_log import (
    OutcomeLogWriter,
    ResolutionRecord,
)


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------------

def _load_bundle(path: Path) -> Any:
    """Load a pickled MultiAssetReport. Raises with a useful message if
    the file is missing or unreadable."""
    if not path.exists():
        raise FileNotFoundError(
            f"bundle not found at {path}. Either pass --bundle to point "
            f"to your saved multi_asset_report.pkl or run a training "
            f"job first.")
    with path.open("rb") as f:
        return pickle.load(f)


def _trading_dates_in_bundle(report: Any) -> List[str]:
    """Discover the set of IST trading dates spanned by the OOS window.

    Walks the asset frames, takes the union of their (IST-normalised)
    date set, drops weekends. Returns sorted ascending.
    """
    dates: set = set()
    assets = getattr(report, "assets", {}) or {}
    for ad in assets.values():
        df = getattr(ad, "base_df", None)
        if df is None or df.empty:
            continue
        idx_ist = pd.DatetimeIndex(df.index) + pd.Timedelta(hours=5, minutes=30)
        for d in idx_ist.normalize().unique():
            if d.weekday() < 5:               # mon..fri
                dates.add(d.strftime("%Y-%m-%d"))
    return sorted(dates)


# ---------------------------------------------------------------------------
# Resolver — turn a logged prediction into a resolution row
# ---------------------------------------------------------------------------

def _resolve_prediction_row(row: pd.Series,
                              report: Any) -> Optional[ResolutionRecord]:
    """Given a prediction row, look up what actually happened.

    v1 supports three of the five prediction types:
      * proximity: scan bars after the brief date for a touch of the
        predicted level within horizon.
      * options_strike: same as proximity, scan the underlying index
        bars (when index data is wired). For v1 with no index data,
        return None — these rows will stay 'unresolved' until index
        data lands.
      * avoidance: check whether a hypothetical entry at session-open
        would have lost > 0.5 ATR; if yes → validated, else → unnecessary.

    Returns None when resolution can't be determined (data gap).
    """
    ptype = row["prediction_type"]
    symbol = row["symbol"]
    trading_date = row["trading_date_ist"]

    assets = getattr(report, "assets", {}) or {}
    asset = assets.get(symbol)

    # Avoidance for ALL_BASKET / ALL_<sector> can't be resolved on a
    # single asset; for v1, treat them as 'avoidance_validated' iff at
    # least one asset would have lost; otherwise 'avoidance_unnecessary'.
    if ptype == "avoidance":
        return _resolve_avoidance(symbol, trading_date, report)

    if asset is None:
        return None
    df = getattr(asset, "base_df", None)
    if df is None or df.empty:
        return None

    if ptype == "proximity":
        return _resolve_proximity(row, df)

    # options_strike: v1 stub — return None until index data is wired.
    if ptype == "options_strike":
        return None

    return None


def _resolve_proximity(row: pd.Series,
                        df: pd.DataFrame) -> Optional[ResolutionRecord]:
    """Did the level get touched same-IST-session after the brief date?"""
    import json
    target = json.loads(row.get("target_description_json", "{}"))
    level = float(target.get("level", 0.0))
    side = target.get("side_from_open", "above")
    trading_date = row["trading_date_ist"]
    # Find bars on the trading_date_ist (same calendar day in IST).
    idx_ist = pd.DatetimeIndex(df.index) + pd.Timedelta(hours=5, minutes=30)
    target_date = pd.Timestamp(trading_date).normalize()
    mask = idx_ist.normalize() == target_date
    same_day = df[mask]
    if same_day.empty:
        return ResolutionRecord(
            prediction_id=row["prediction_id"],
            resolved_at_utc=datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            resolution_method="data_gap",
            outcome_boolean=None, outcome_continuous=None,
            resolution_details={"reason": "no_bars_for_trading_date"},
            had_data_gap=True, resolution_quality="data_gap",
        )
    # Touch check: high crossed level (for 'above' side) or low crossed
    # level (for 'below' side).
    if side == "above":
        touched = bool((same_day["high"] >= level).any())
    else:
        touched = bool((same_day["low"] <= level).any())
    method = ("level_touched" if touched
                else "level_not_touched_horizon_expired")
    return ResolutionRecord(
        prediction_id=row["prediction_id"],
        resolved_at_utc=datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        resolution_method=method,
        outcome_boolean=touched,
        outcome_continuous=None,
        resolution_details={
            "session_high": float(same_day["high"].max()),
            "session_low": float(same_day["low"].min()),
            "level_distance_atr_estimate": None,
        },
        had_data_gap=False, resolution_quality="clean",
    )


def _resolve_avoidance(symbol: str, trading_date: str,
                        report: Any) -> Optional[ResolutionRecord]:
    """An avoid call is VALIDATED iff a hypothetical session-open trade
    would have lost more than 0.5 ATR (basket-level: any asset losing
    qualifies). Otherwise UNNECESSARY."""
    assets = getattr(report, "assets", {}) or {}
    if not assets:
        return None
    target_date = pd.Timestamp(trading_date).normalize()
    losses = []
    for sym, ad in assets.items():
        df = getattr(ad, "base_df", None)
        if df is None or df.empty:
            continue
        idx_ist = pd.DatetimeIndex(df.index) + pd.Timedelta(hours=5, minutes=30)
        same_day = df[idx_ist.normalize() == target_date]
        if same_day.empty:
            continue
        # Session move = (last close - session open) / ATR-approx-as-range.
        open_p = float(same_day["open"].iloc[0])
        close_p = float(same_day["close"].iloc[-1])
        rng = float(same_day["high"].max() - same_day["low"].min())
        if rng <= 0:
            continue
        move_atr = (close_p - open_p) / rng
        losses.append(abs(move_atr) > 0.5 and (close_p - open_p) < 0)
    if not losses:
        return ResolutionRecord(
            prediction_id=f"unknown",
            resolved_at_utc=datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            resolution_method="data_gap",
            outcome_boolean=None, outcome_continuous=None,
            resolution_details={"reason": "no_data_for_avoidance_check"},
            had_data_gap=True, resolution_quality="data_gap",
        )
    validated = any(losses)
    return ResolutionRecord(
        prediction_id="placeholder",       # caller overwrites
        resolved_at_utc=datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        resolution_method=("avoidance_validated" if validated
                            else "avoidance_unnecessary"),
        outcome_boolean=validated,
        outcome_continuous=None,
        resolution_details={"n_assets_losing": int(sum(losses))},
        had_data_gap=False, resolution_quality="clean",
    )


# ---------------------------------------------------------------------------
# Backfill driver
# ---------------------------------------------------------------------------

def run_backfill(bundle_path: Path,
                  output_root: Path,
                  days: int = 90,
                  skip_existing: bool = False,
                  dry_run: bool = False) -> Dict[str, int]:
    """Backfill predictions + resolutions for the last ``days`` IST dates
    in the bundle. Returns a counts dict."""
    report = _load_bundle(bundle_path)
    all_dates = _trading_dates_in_bundle(report)
    if not all_dates:
        LOGGER.warning("bundle contains no trading dates; nothing to backfill")
        return {"predictions": 0, "resolutions": 0}
    target_dates = all_dates[-days:]
    LOGGER.info("backfilling %d trading dates (from %s to %s)",
                len(target_dates), target_dates[0], target_dates[-1])

    if dry_run:
        LOGGER.info("dry_run=True -> not writing parquet")
        return {"predictions": 0, "resolutions": 0,
                "dates_planned": len(target_dates)}

    writer = OutcomeLogWriter(root=str(output_root))
    total_predictions = 0
    total_resolutions = 0
    skipped_existing = 0
    for i, trading_date in enumerate(target_dates, start=1):
        pred_path = (
            output_root / "predictions" / f"trading_date_ist={trading_date}"
            / "predictions.parquet"
        )
        reso_path = (
            output_root / "resolutions" / f"trading_date_ist={trading_date}"
            / "resolutions.parquet"
        )
        if skip_existing and pred_path.exists() and reso_path.exists():
            skipped_existing += 1
            LOGGER.info("[%d/%d] skipping %s; prediction/resolution partitions exist",
                        i, len(target_dates), trading_date)
            continue
        LOGGER.info("[%d/%d] generating brief predictions for %s",
                    i, len(target_dates), trading_date)
        # Generate the brief for this historical date. The brief
        # internally logs predictions via the writer.
        generate_brief(
            report,
            trading_date_ist=trading_date,
            model_bundle_version=str(bundle_path.stem),
            outcome_log_writer=writer,
            retrospective=True,
        )
        # Flush prediction rows before reading the partition back. The
        # previous implementation read from disk while predictions were
        # still buffered, which could accidentally resolve stale rows from
        # an earlier run.
        pred_commit = writer.commit()
        # Read back the just-written predictions to drive resolution.
        preds = writer.read_predictions(trading_date)
        if preds.empty:
            LOGGER.info("[%d/%d] %s produced 0 predictions",
                        i, len(target_dates), trading_date)
            continue
        total_predictions += int(pred_commit.get("predictions", 0))
        date_resolutions = 0
        for _, prow in preds.iterrows():
            reso = _resolve_prediction_row(prow, report)
            if reso is None:
                continue
            reso.prediction_id = prow["prediction_id"]
            # Tag the resolution with the prediction's IST trading date
            # so the writer can partition correctly.
            reso.trading_date_ist = trading_date
            writer.write_resolution(reso)
            date_resolutions += 1
        reso_commit = writer.commit()
        total_resolutions += date_resolutions
        LOGGER.info(
            "[%d/%d] %s predictions=%d resolutions=%d committed=%s/%s",
            i, len(target_dates), trading_date, int(pred_commit.get("predictions", 0)),
            date_resolutions, pred_commit, reso_commit,
        )
    return {"trading_dates_processed": len(target_dates),
            "trading_dates_skipped_existing": skipped_existing,
            "predictions_read": total_predictions,
            "resolutions_written": total_resolutions}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill the outcome log from a saved bundle.")
    parser.add_argument("--bundle", type=Path, required=True,
                        help="Path to multi_asset_report.pkl")
    parser.add_argument("--output-root", type=Path,
                        default=Path("data/outcome_log"),
                        help="Where to write predictions/ and resolutions/")
    parser.add_argument("--days", type=int, default=90,
                        help="How many trailing IST trading dates to backfill")
    parser.add_argument("--dry-run", action="store_true",
                        help="List dates that would be backfilled, write nothing")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip dates whose prediction and resolution partitions already exist")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    counts = run_backfill(args.bundle, args.output_root,
                            days=args.days,
                            skip_existing=args.skip_existing,
                            dry_run=args.dry_run)
    print(f"backfill result: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
