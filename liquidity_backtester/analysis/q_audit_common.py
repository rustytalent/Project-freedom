"""Shared helpers for Phase 4 Workstream 0 Q-scale audits.

These scripts intentionally read existing artifacts only. They do not retrain
models or mutate production configuration.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


DEFAULT_RUN_DIR = Path("output_core25_phase3d_train_fresh_may25")
DEFAULT_AUDIT_CSV = DEFAULT_RUN_DIR / "phase3_oos_prediction_audit.csv"
DEFAULT_LIVE_PLAN_JSON = DEFAULT_RUN_DIR / "live_plan.json"

Q_COLUMNS = ["global_q", "sector_q", "blended_q"]
LEGACY_Q_ALIASES = {
    "q_global": "global_q",
    "q_sector_expert": "sector_q",
    "q_blended": "blended_q",
}


def load_audit_frame(path: Path = DEFAULT_AUDIT_CSV) -> pd.DataFrame:
    """Load and normalize the OOS quality audit artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Q audit CSV not found: {path}")
    df = pd.read_csv(path)
    rename = {old: new for old, new in LEGACY_Q_ALIASES.items() if old in df.columns}
    if rename:
        df = df.rename(columns=rename)

    missing = [col for col in ["asset", "sector", "actual", "blended_q"] if col not in df.columns]
    if missing:
        raise ValueError(f"Q audit CSV missing required columns: {missing}")

    for col in Q_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    return df


def decisive_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where strict-respect label is available."""
    return df[df["actual"].notna()].copy()


def load_live_gate(path: Path = DEFAULT_LIVE_PLAN_JSON) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Live plan JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    gate = data.get("gate") or data.get("gate_config") or {}
    return gate


def available_q_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in Q_COLUMNS if col in df.columns and df[col].notna().any()]


def exact_top_fraction(df: pd.DataFrame, q_col: str, fraction: float) -> pd.DataFrame:
    """Return the exact top N rows by prediction, avoiding threshold-tie inflation."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    n = max(1, int(math.ceil(len(df) * fraction)))
    return df.nlargest(n, q_col)


def pct(x: Optional[float], digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{100.0 * float(x):.{digits}f}%"


def pp(x: Optional[float], digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{100.0 * float(x):+.{digits}f}pp"


def fnum(x: Optional[float], digits: int = 4) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{float(x):.{digits}f}"


def markdown_table(rows: Iterable[Dict], columns: List[str]) -> str:
    rows = list(rows)
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body])
