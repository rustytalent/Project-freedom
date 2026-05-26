"""Percentile rescaling utilities for compressed quality-model outputs.

This is research plumbing only. It does not change the production live gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QScaleMetadata:
    q_col: str
    n: int
    min_q: float
    max_q: float
    mean_q: float
    std_q: float
    quantiles: Dict[str, float]


def fit_q_scale(values: Iterable[float], q_col: str = "blended_q") -> QScaleMetadata:
    q = pd.Series(list(values), dtype="float64").dropna()
    if q.empty:
        raise ValueError("Cannot fit Q scale from empty values")
    quantile_points = [0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
    return QScaleMetadata(
        q_col=q_col,
        n=int(len(q)),
        min_q=float(q.min()),
        max_q=float(q.max()),
        mean_q=float(q.mean()),
        std_q=float(q.std()),
        quantiles={f"p{int(p * 1000):03d}": float(q.quantile(p)) for p in quantile_points},
    )


def add_q_percentile(df: pd.DataFrame, q_col: str = "blended_q") -> pd.DataFrame:
    """Add ascending percentile where 1.0 is the highest observed Q."""
    if q_col not in df.columns:
        raise ValueError(f"Missing Q column: {q_col}")
    out = df.copy()
    out["q_percentile"] = out[q_col].rank(method="average", pct=True)
    out["q_rank_desc"] = out[q_col].rank(method="first", ascending=False)
    return out


def threshold_for_top_fraction(values: Iterable[float], fraction: float) -> float:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    q = pd.Series(list(values), dtype="float64").dropna()
    if q.empty:
        raise ValueError("Cannot compute threshold from empty values")
    return float(q.quantile(1.0 - fraction))


def exact_top_mask(df: pd.DataFrame, q_col: str, fraction: float) -> pd.Series:
    """Boolean mask for exact top N rows, avoiding threshold ties."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    n = max(1, int(np.ceil(len(df) * fraction)))
    top_index = df.nlargest(n, q_col).index
    return df.index.isin(top_index)


def metadata_as_dict(meta: QScaleMetadata) -> Dict:
    return {
        "q_col": meta.q_col,
        "n": meta.n,
        "min_q": meta.min_q,
        "max_q": meta.max_q,
        "mean_q": meta.mean_q,
        "std_q": meta.std_q,
        "quantiles": meta.quantiles,
    }
