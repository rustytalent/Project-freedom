"""M.4 — SKIP / rejection regret estimator.

Trains on the shadow log: every declined decision (options SKIP,
Q-below-threshold drop, below-target-to-cost rejection) joined with
its resolved counterfactual outcome. The target is binary regret —
"would taking the declined action have paid?" — exactly the verdict
labels the Stream L resolvers emit:

    rejection_regret / avoidance_regret      -> 1
    rejection_vindicated / avoidance_vindicated -> 0

Output: P(regret | decision context). Use cases:

  * Gate tuning: if P(regret) for a gate's marginal declines trends
    above ~0.5, the gate threshold is too tight — the engine is
    declining trades it should take.
  * Adaptive SKIP: the options executor can consult the regret model
    at decision time to soften/harden the bootstrap LCS threshold.

Data shape (one row per resolved shadow event):
    produced by  shadow_log.join_events_to_resolutions(root)
    decision_context — JSON string (the gate's recorded payload)
    counterfactual_outcome — JSON string with a "verdict" field
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..execution._base import LGBMConfig, LGBMRegressorWrapper

# Numeric context keys we lift out of decision_context when present.
# Unknown keys are ignored; missing keys become NaN (LightGBM-native).
CONTEXT_FEATURES: List[str] = [
    "lcs", "macro_score", "regime_score", "options_score",
    "micro_organic_score", "pool_holding_strength", "manipulation_score",
    "q_pred", "p_long", "p_short", "dist_atr_at_last_bar",
    "target_to_cost_ratio", "target_reward_inr", "round_trip_cost_inr",
]

FEATURE_COLUMNS: List[str] = CONTEXT_FEATURES + [
    "kind_skip_options", "kind_q_below", "kind_below_cost", "kind_avoidance",
]

FALLBACK_REGRET: float = 0.5   # max-entropy: unfit model claims nothing
PROB_FLOOR, PROB_CEIL = 0.02, 0.98

_REGRET_VERDICTS = {"rejection_regret", "avoidance_regret"}
_VINDICATED_VERDICTS = {"rejection_vindicated", "avoidance_vindicated"}


def _parse_json(s: Any) -> Dict[str, Any]:
    if isinstance(s, dict):
        return s
    try:
        out = json.loads(s)
        return out if isinstance(out, dict) else {}
    except (TypeError, ValueError):
        return {}


def featurize_shadow_rows(joined: pd.DataFrame) -> pd.DataFrame:
    """Lift numeric features out of the JSON decision_context plus
    one-hot the event kind. Works on the exact frame
    ``join_events_to_resolutions`` returns."""
    rows = []
    for _, r in joined.iterrows():
        ctx = _parse_json(r.get("decision_context"))
        row: Dict[str, float] = {}
        for k in CONTEXT_FEATURES:
            v = ctx.get(k)
            try:
                f = float(v)
                row[k] = f if math.isfinite(f) else np.nan
            except (TypeError, ValueError):
                row[k] = np.nan
        kind = str(r.get("event_kind", ""))
        row["kind_skip_options"] = float(kind == "skip_options_executor")
        row["kind_q_below"] = float(kind == "q_below_threshold")
        row["kind_below_cost"] = float(kind == "below_target_to_cost_ratio")
        row["kind_avoidance"] = float(kind == "avoidance_flag")
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def extract_regret_labels(joined: pd.DataFrame) -> np.ndarray:
    """Binary regret target from the resolver verdicts. Rows whose
    outcome has no recognisable verdict get NaN (the trainer drops
    them)."""
    labels = []
    for _, r in joined.iterrows():
        outcome = _parse_json(r.get("counterfactual_outcome"))
        verdict = str(outcome.get("verdict", ""))
        if verdict in _REGRET_VERDICTS:
            labels.append(1.0)
        elif verdict in _VINDICATED_VERDICTS:
            labels.append(0.0)
        else:
            labels.append(np.nan)
    return np.asarray(labels, dtype=float)


class RegretEstimator:
    """P(regret) for declined decisions. fit() on the joined shadow
    frame; predict on a decision-context dict at gate time."""

    def __init__(self, config: Optional[LGBMConfig] = None) -> None:
        self._wrapper = LGBMRegressorWrapper(
            feature_names=FEATURE_COLUMNS,
            fallback=FALLBACK_REGRET,
            config=config,
        )

    def fit(self, joined: pd.DataFrame) -> "RegretEstimator":
        if joined.empty:
            return self
        X = featurize_shadow_rows(joined)
        y = extract_regret_labels(joined)
        mask = np.isfinite(y)
        if mask.sum() == 0:
            return self
        self._wrapper.fit(X.loc[mask].reset_index(drop=True), y[mask])
        return self

    def predict_regret(self, event_kind: str,
                       decision_context: Dict[str, Any]) -> float:
        df = pd.DataFrame([{
            "event_kind": event_kind,
            "decision_context": decision_context,
        }])
        X = featurize_shadow_rows(df)
        raw = float(self._wrapper.predict(X)[0])
        if not math.isfinite(raw):
            raw = FALLBACK_REGRET
        return max(PROB_FLOOR, min(raw, PROB_CEIL))

    @property
    def is_fitted(self) -> bool:
        return self._wrapper.is_fitted

    @property
    def trained_n(self) -> int:
        return self._wrapper.trained_n

    def save(self, path) -> None:
        self._wrapper.save(path)

    @classmethod
    def load(cls, path) -> "RegretEstimator":
        obj = cls()
        obj._wrapper = LGBMRegressorWrapper.load(path)
        return obj
