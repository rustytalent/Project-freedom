"""Fast leakage probe helpers.

These are intentionally small, deterministic checks that can run in CI without
Core25 runtime artifacts. Full replay/retrain probes still belong in local or
nightly jobs that have access to the parquet warehouse and saved feature store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .pools import Pool
from .timing import Snapshot


@dataclass(frozen=True)
class LabelShuffleProbeResult:
    observed_auc: float
    shuffled_mean_auc: float
    shuffled_min_auc: float
    shuffled_max_auc: float
    repeats: int
    passed: bool


@dataclass(frozen=True)
class ProbeIssue:
    check: str
    message: str
    detail: Mapping[str, object]


def _auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    if len(np.unique(y)) < 2:
        raise ValueError("AUC requires both classes")
    return float(roc_auc_score(y, score))


def label_shuffle_auc_probe(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    seed: int = 42,
    repeats: int = 64,
    lower: float = 0.45,
    upper: float = 0.55,
) -> LabelShuffleProbeResult:
    """Integrity check on a persisted (score, label) pair.

    NOTE: this is NOT, by itself, a leakage detector. Permuting the labels of a
    *fixed* score vector drives AUC to ~0.5 by construction regardless of whether
    the score was computed with leakage, so the shuffled mean alone proves nothing
    about leakage. True leakage detection requires RETRAINING on shuffled labels and
    checking the retrained model still can't separate the holdout — that heavier
    probe belongs in the nightly job where feature matrices are available.

    What this fast probe DOES verify:
      1. the score/label pair has genuine association (`observed_auc` sits clearly
         above the shuffled band — catches a broken/misaligned pipeline that emits
         scores uncorrelated with labels), and
      2. the AUC estimator behaves (shuffled mean ≈ 0.5).
    `passed` requires BOTH, so a degenerate score (observed ≈ shuffled) now fails
    instead of silently passing.
    """
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    if len(y) != len(score):
        raise ValueError("y_true and y_score must have equal length")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    observed = _auc(y, score)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(repeats):
        shuffled = rng.permutation(y)
        aucs.append(_auc(shuffled, score))
    arr = np.asarray(aucs, dtype=float)
    mean_auc = float(arr.mean())
    shuffled_max = float(arr.max())
    shuffle_neutral = bool(lower <= mean_auc <= upper)
    # Guard against an AUC < 0.5 score (label sense flipped): use distance from 0.5.
    observed_has_signal = bool(abs(observed - 0.5) > (shuffled_max - 0.5))
    return LabelShuffleProbeResult(
        observed_auc=observed,
        shuffled_mean_auc=mean_auc,
        shuffled_min_auc=float(arr.min()),
        shuffled_max_auc=shuffled_max,
        repeats=int(repeats),
        passed=bool(shuffle_neutral and observed_has_signal),
    )


def feature_timestamp_issues(rows: Iterable[Mapping[str, object]]) -> List[ProbeIssue]:
    """Return rows where feature as_of_ts is after decision_ts."""
    issues: List[ProbeIssue] = []
    for i, row in enumerate(rows):
        name = str(row.get("feature", row.get("feature_name", f"feature_{i}")))
        decision_ts = pd.Timestamp(row.get("decision_ts"))
        as_of_ts = pd.Timestamp(row.get("as_of_ts"))
        if pd.isna(decision_ts) or pd.isna(as_of_ts):
            issues.append(ProbeIssue(
                check="future_mask",
                message="feature timestamp metadata is missing or invalid",
                detail={"row": i, "feature": name},
            ))
            continue
        if as_of_ts > decision_ts:
            issues.append(ProbeIssue(
                check="future_mask",
                message="feature as_of_ts is after decision_ts",
                detail={
                    "row": i,
                    "feature": name,
                    "as_of_ts": str(as_of_ts),
                    "decision_ts": str(decision_ts),
                },
            ))
    return issues


def assert_no_future_features(rows: Iterable[Mapping[str, object]]) -> None:
    issues = feature_timestamp_issues(rows)
    if issues:
        first = issues[0]
        raise AssertionError(f"{first.check}: {first.message} {dict(first.detail)}")


def snapshot_distance_issues(
    snapshot: Snapshot,
    pools: Sequence[Pool],
    *,
    tolerance: float = 1e-9,
) -> List[ProbeIssue]:
    """Validate stored snapshot distance_atr values from snapshot close and ATR only."""
    issues: List[ProbeIssue] = []
    pool_by_idx = {i: p for i, p in enumerate(pools)}
    for pool_idx, _touched, distance_atr, side in snapshot.pool_touches:
        pool = pool_by_idx.get(pool_idx)
        if pool is None:
            issues.append(ProbeIssue(
                check="distance_atr",
                message="snapshot references a missing pool index",
                detail={"pool_idx": int(pool_idx)},
            ))
            continue
        if side == "above":
            expected = (float(pool.mid) - float(snapshot.close)) / max(float(snapshot.atr_val), 1e-9)
        elif side == "below":
            expected = (float(snapshot.close) - float(pool.mid)) / max(float(snapshot.atr_val), 1e-9)
        else:
            issues.append(ProbeIssue(
                check="distance_atr",
                message="snapshot has invalid pool side",
                detail={"pool_idx": int(pool_idx), "side": side},
            ))
            continue
        if abs(float(distance_atr) - expected) > tolerance:
            issues.append(ProbeIssue(
                check="distance_atr",
                message="stored distance_atr does not match decision-time close and ATR",
                detail={
                    "pool_idx": int(pool_idx),
                    "side": side,
                    "stored": float(distance_atr),
                    "expected": float(expected),
                    "snapshot_ts": str(snapshot.ts),
                },
            ))
    return issues


def assert_snapshot_distance_causal(
    snapshot: Snapshot,
    pools: Sequence[Pool],
    *,
    tolerance: float = 1e-9,
) -> None:
    issues = snapshot_distance_issues(snapshot, pools, tolerance=tolerance)
    if issues:
        first = issues[0]
        raise AssertionError(f"{first.check}: {first.message} {dict(first.detail)}")
