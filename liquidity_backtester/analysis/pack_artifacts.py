"""Pack a training-run output directory into 3-4 consolidated files.

A normal multi_asset_run emits ~20 files (multi_asset_summary.json,
research_summary.json, live_plan.json, per-phase CSVs, leakage_audit,
policy_return tables, reaction_alerts, daily_brief.json/.txt, etc).

That's too many for an individual to ship to a chat tool or a customer
in one go. This script reads the standard output directory and emits
three rich consolidated JSON files plus passes the daily-brief
artifacts through unchanged.

Output files:

  * ``consolidated_summary.json`` — top-level metrics + verdict + brief
    metadata. The file you read FIRST.
  * ``consolidated_models.json`` — per-model OOS metrics, calibration,
    feature importance, audit tables.
  * ``consolidated_backtest.json`` — execution v1/v2, arsenal, policy_R,
    cost wall, reaction alerts.
  * (passes through) ``daily_brief.json`` + ``daily_brief.txt`` if
    present — these are the customer-facing product and stay separate.

Source files NOT modified. The script is additive: re-running is safe.

Usage::

    PYTHONPATH=. python analysis/pack_artifacts.py \\
        --input output_core25_head_alpha_710362b/ \\
        --output packed/core25_head_alpha_710362b/

If --output is omitted, the consolidated files are written into the
input directory itself (``packed_*.json``).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source -> bundle mapping
# ---------------------------------------------------------------------------

# Files routed to consolidated_summary.json
SUMMARY_FILES = {
    "multi_asset_summary":      "multi_asset_summary.json",
    "research_summary":         "research_summary.json",
    "live_plan":                "live_plan.json",
    "sector_data":              "sector_data.json",
    "leakage_audit":            "leakage_audit.json",
}

# Files routed to consolidated_models.json
MODEL_FILES = {
    "phase3_oos_prediction_audit":    "phase3_oos_prediction_audit.csv",
    "phase3_sector_calibration":      "phase3_sector_calibration.csv",
    "reaction_model_report":          "reaction_model_report.csv",
    "reaction_model_calibration":     "reaction_model_calibration.csv",
    "reaction_feature_importance":    "reaction_feature_importance.csv",
    "policy_return_model_report":     "policy_return_model_report.csv",
    "policy_return_model_calibration":"policy_return_model_calibration.csv",
    "policy_return_model_feature_importance":
        "policy_return_model_feature_importance.csv",
    "policy_return_model_oos_predictions":
        "policy_return_model_oos_predictions.csv",
    "policy_label_summary":           "policy_label_summary.csv",
    "leakage_issues":                 "leakage_issues.csv",
}

# Files routed to consolidated_backtest.json
BACKTEST_FILES = {
    "execution_backtest_summary":     "execution_backtest_summary.csv",
    "execution_backtest_by_direction":"execution_backtest_by_direction.csv",
    "execution_backtest_v2_summary":  "execution_backtest_v2_summary.csv",
    "execution_backtest_v2_vs_v1_delta":
        "execution_backtest_v2_vs_v1_delta.csv",
    "reaction_alerts":                "reaction_alerts.csv",
    "track_a_pretouch_setups":        "track_a_pretouch_setups.csv",
}

# Per-row caps so a single CSV doesn't bloat the bundle. Top-200 by
# default for trade-row files (which can have millions of rows).
MAX_ROWS_PER_FILE = {
    "execution_backtest_trades":            500,
    "execution_backtest_v2_trades":         500,
    "reaction_alerts":                      300,
    "policy_return_model_oos_predictions":  300,
    "leakage_issues":                       200,
}

# Files copied through (not consolidated).
PASS_THROUGH = ("daily_brief.json", "daily_brief.txt")


# ---------------------------------------------------------------------------
# Loaders — graceful, return None on missing / unreadable
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        LOGGER.warning("failed to load %s: %s", path, exc)
        return None


def _load_csv(path: Path, key: str) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        cap = MAX_ROWS_PER_FILE.get(key)
        if cap is not None and len(df) > cap:
            df = df.head(cap)
        return df.to_dict(orient="records")
    except Exception as exc:
        LOGGER.warning("failed to load %s: %s", path, exc)
        return None


def _build_bundle(input_dir: Path,
                  file_map: Dict[str, str]) -> Dict[str, Any]:
    bundle: Dict[str, Any] = {}
    for key, filename in file_map.items():
        path = input_dir / filename
        if filename.endswith(".json"):
            bundle[key] = _load_json(path)
        elif filename.endswith(".csv"):
            bundle[key] = _load_csv(path, key)
        else:
            bundle[key] = None
    return bundle


# ---------------------------------------------------------------------------
# Pack driver
# ---------------------------------------------------------------------------

def pack(input_dir: Path, output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Pack the given training-output directory into 3 consolidated
    JSON files + pass through the daily-brief artifacts.

    Returns a dict of {bundle_name: output_path} for all files written.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    if output_dir is None:
        output_dir = input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}

    for bundle_name, file_map in (
        ("consolidated_summary", SUMMARY_FILES),
        ("consolidated_models", MODEL_FILES),
        ("consolidated_backtest", BACKTEST_FILES),
    ):
        bundle = _build_bundle(input_dir, file_map)
        out_path = output_dir / f"{bundle_name}.json"
        with out_path.open("w") as f:
            json.dump(bundle, f, default=str, indent=2)
        written[bundle_name] = out_path
        n_keys = sum(1 for v in bundle.values() if v is not None)
        total_keys = len(file_map)
        LOGGER.info("wrote %s — %d/%d source files present",
                    out_path.name, n_keys, total_keys)

    # Pass-through copies for the daily-brief artifacts.
    for name in PASS_THROUGH:
        src = input_dir / name
        if src.exists() and src.resolve() != (output_dir / name).resolve():
            shutil.copy2(src, output_dir / name)
            written[name] = output_dir / name
            LOGGER.info("passed through %s", name)

    return written


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pack a training-output directory into 3 consolidated files.")
    p.add_argument("--input", type=Path, required=True,
                   help="Training output directory (contains "
                         "multi_asset_summary.json etc.)")
    p.add_argument("--output", type=Path, default=None,
                   help="Where to write the consolidated files. "
                         "Default: same as --input.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    written = pack(args.input, args.output)
    print(f"packed {len(written)} files into {args.output or args.input}:")
    for key, path in written.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
