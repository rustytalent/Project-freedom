"""Measure realized strict-respect rates by Q decile."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from q_audit_common import DEFAULT_AUDIT_CSV, decisive_rows, load_audit_frame


def analyze_q_decile_performance(
    audit_csv_path: Path = DEFAULT_AUDIT_CSV,
    q_col: str = "blended_q",
) -> pd.DataFrame:
    df = decisive_rows(load_audit_frame(audit_csv_path))
    if q_col not in df.columns:
        raise ValueError(f"Missing Q column: {q_col}")

    df = df[df[q_col].notna()].copy()
    df["q_decile"] = pd.qcut(df[q_col], 10, labels=False, duplicates="drop")
    base_rate = float(df["actual"].mean())

    decile_stats = (
        df.groupby("q_decile", dropna=True)
        .agg(
            n=("actual", "size"),
            mean_q_predicted=(q_col, "mean"),
            min_q=(q_col, "min"),
            max_q=(q_col, "max"),
            actual_strict_respect=("actual", "mean"),
        )
        .reset_index()
    )
    decile_stats["lift_over_base"] = decile_stats["actual_strict_respect"] - base_rate
    decile_stats["lift_pct_relative"] = (
        decile_stats["actual_strict_respect"] / base_rate - 1.0
    ) * 100.0
    return decile_stats


if __name__ == "__main__":
    print(analyze_q_decile_performance().to_string(index=False))
