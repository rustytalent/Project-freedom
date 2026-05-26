"""Analyze the output distribution of the quality model.

Phase 4 Workstream 0 asks whether Q is genuinely weak or merely compressed by
regularization/calibration. This module produces the distribution evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from q_audit_common import DEFAULT_AUDIT_CSV, available_q_columns, load_audit_frame


PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]


def summarize_series(q: pd.Series) -> Dict:
    q = q.dropna().astype(float)
    percentiles = {str(p): float(q.quantile(p / 100.0)) for p in PERCENTILES}
    return {
        "n": int(len(q)),
        "min": float(q.min()),
        "max": float(q.max()),
        "mean": float(q.mean()),
        "std": float(q.std()),
        "range": float(q.max() - q.min()),
        "compression_ratio": float(q.max() - q.min()),
        "percentiles": percentiles,
    }


def analyze_q_distribution(audit_csv_path: Path = DEFAULT_AUDIT_CSV) -> Dict:
    df = load_audit_frame(audit_csv_path)

    overall = {}
    for q_col in available_q_columns(df):
        overall[q_col] = summarize_series(df[q_col])

    asset_q_stats = (
        df.groupby("asset")["blended_q"]
        .agg(["min", "max", "mean", "std"])
        .reset_index()
    )
    asset_q_stats["range"] = asset_q_stats["max"] - asset_q_stats["min"]

    sector_q_stats = (
        df.groupby("sector")["blended_q"]
        .agg(["min", "max", "mean", "std"])
        .reset_index()
    )
    sector_q_stats["range"] = sector_q_stats["max"] - sector_q_stats["min"]

    return {
        "overall": overall,
        "per_asset": asset_q_stats,
        "per_sector": sector_q_stats,
    }


if __name__ == "__main__":
    result = analyze_q_distribution()
    for col, stats in result["overall"].items():
        print(f"\n{col}")
        for key in ["n", "min", "max", "mean", "std", "range", "compression_ratio"]:
            print(f"  {key}: {stats[key]}")
        print("  percentiles:")
        for p, value in stats["percentiles"].items():
            print(f"    p{p}: {value}")
