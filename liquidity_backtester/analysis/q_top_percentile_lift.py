"""Compute exact top-percentile lift for compressed Q predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from q_audit_common import (
    DEFAULT_AUDIT_CSV,
    decisive_rows,
    exact_top_fraction,
    load_audit_frame,
)


def analyze_top_percentile_lift(
    audit_csv_path: Path = DEFAULT_AUDIT_CSV,
    q_col: str = "blended_q",
) -> Dict:
    df = decisive_rows(load_audit_frame(audit_csv_path))
    if q_col not in df.columns:
        raise ValueError(f"Missing Q column: {q_col}")
    df = df[df[q_col].notna()].copy()
    base_rate = float(df["actual"].mean())

    results = {
        "q_col": q_col,
        "n": int(len(df)),
        "base_rate": base_rate,
        "slices": {},
    }
    for label, fraction in [
        ("top_50pct", 0.50),
        ("top_25pct", 0.25),
        ("top_10pct", 0.10),
        ("top_5pct", 0.05),
        ("top_1pct", 0.01),
        ("top_0_1pct", 0.001),
    ]:
        top = exact_top_fraction(df, q_col, fraction)
        actual = float(top["actual"].mean())
        results["slices"][label] = {
            "fraction": fraction,
            "n": int(len(top)),
            "min_q": float(top[q_col].min()),
            "mean_q": float(top[q_col].mean()),
            "max_q": float(top[q_col].max()),
            "actual_respect": actual,
            "lift_over_base": actual - base_rate,
            "lift_pct_relative": (actual / base_rate - 1.0) * 100.0,
        }
    return results


if __name__ == "__main__":
    result = analyze_top_percentile_lift()
    print(f"base_rate: {result['base_rate']}")
    for label, stats in result["slices"].items():
        print(label, stats)
