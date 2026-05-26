"""Audit whether absolute live Q gates are reachable."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from q_audit_common import DEFAULT_AUDIT_CSV, DEFAULT_LIVE_PLAN_JSON, load_audit_frame, load_live_gate


def analyze_gate_reachability(
    audit_csv_path: Path = DEFAULT_AUDIT_CSV,
    live_plan_json: Path = DEFAULT_LIVE_PLAN_JSON,
    q_col: str = "blended_q",
) -> Dict:
    df = load_audit_frame(audit_csv_path)
    if q_col not in df.columns:
        raise ValueError(f"Missing Q column: {q_col}")
    q = df[q_col].dropna().astype(float)
    gate = load_live_gate(live_plan_json)
    current_q_gate = float(gate.get("min_q", 0.70))

    max_q = float(q.max())
    pct_above_gate = float((q >= current_q_gate).mean() * 100.0)
    gate_status = "REACHABLE" if max_q >= current_q_gate else "UNREACHABLE"
    gate_percentile = float((q < current_q_gate).mean() * 100.0) if gate_status == "REACHABLE" else None

    return {
        "q_col": q_col,
        "current_absolute_gate": current_q_gate,
        "max_q_observed": max_q,
        "pct_setups_passing_gate": pct_above_gate,
        "gate_status": gate_status,
        "gate_percentile_equivalent": gate_percentile,
        "suggested_percentile_thresholds": {
            "top_25pct_threshold": float(q.quantile(0.75)),
            "top_10pct_threshold": float(q.quantile(0.90)),
            "top_5pct_threshold": float(q.quantile(0.95)),
            "top_1pct_threshold": float(q.quantile(0.99)),
        },
        "live_gate": gate,
    }


if __name__ == "__main__":
    print(analyze_gate_reachability())
