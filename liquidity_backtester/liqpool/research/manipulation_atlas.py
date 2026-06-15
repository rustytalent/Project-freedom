"""Manipulation Atlas primitives.

The atlas is not a trading strategy. It is a state classifier that
summarises whether an index move is broad, concentrated, trapped,
absorbed, or unsupported. Scientists and hypothesis miners can then
use the state as input without re-implementing the same market-context
logic in every alpha.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional

import math


@dataclass(frozen=True)
class ConstituentState:
    """One heavyweight constituent's live relationship to value."""

    symbol: str
    weight: float
    return_pct: float = 0.0
    today_avwap_dist_atr: Optional[float] = None
    prev_session_avwap_dist_atr: Optional[float] = None
    historical_position_pct: Optional[float] = None
    gap_pct: Optional[float] = None
    above_today_avwap: Optional[bool] = None
    above_prev_avwap: Optional[bool] = None

    @classmethod
    def from_mapping(cls, row: Mapping) -> "ConstituentState":
        today = _optional_float(row.get("today_avwap_dist_atr"))
        prev = _optional_float(row.get("prev_session_avwap_dist_atr"))
        return cls(
            symbol=str(row.get("symbol", "")),
            weight=max(0.0, _optional_float(row.get("weight"), 0.0) or 0.0),
            return_pct=_optional_float(row.get("return_pct"), 0.0) or 0.0,
            today_avwap_dist_atr=today,
            prev_session_avwap_dist_atr=prev,
            historical_position_pct=_optional_float(
                row.get("historical_position_pct")
            ),
            gap_pct=_optional_float(row.get("gap_pct")),
            above_today_avwap=_optional_bool(row.get("above_today_avwap"), today),
            above_prev_avwap=_optional_bool(row.get("above_prev_avwap"), prev),
        )


@dataclass(frozen=True)
class ManipulationState:
    """Index-level state emitted by the atlas."""

    label: str
    confidence: float
    action_bias: str
    reason_codes: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "confidence": float(self.confidence),
            "action_bias": self.action_bias,
            "reason_codes": list(self.reason_codes),
            "metrics": dict(self.metrics),
        }


def classify_index_manipulation(
    constituents: Iterable[ConstituentState | Mapping],
    *,
    index_return_pct: float = 0.0,
    breadth_positive_frac: Optional[float] = None,
    concentration_threshold: float = 0.55,
) -> ManipulationState:
    """Classify the index state from heavyweight constituents.

    Inputs can be :class:`ConstituentState` instances or dict-like rows
    with equivalent keys. The function is deliberately deterministic and
    transparent because these labels become features for later alphas.
    """
    states = [
        c if isinstance(c, ConstituentState) else ConstituentState.from_mapping(c)
        for c in constituents
    ]
    states = [s for s in states if s.symbol and s.weight > 0]
    if not states:
        return ManipulationState(
            label="unknown",
            confidence=0.0,
            action_bias="wait",
            reason_codes=["no_constituent_state"],
            metrics={},
        )

    total_weight = sum(s.weight for s in states)
    weights = [s.weight / total_weight for s in states]
    weighted_return = sum(w * s.return_pct for w, s in zip(weights, states))
    positive_weight = sum(w for w, s in zip(weights, states) if s.return_pct > 0)
    negative_weight = sum(w for w, s in zip(weights, states) if s.return_pct < 0)
    top3_contribution = _top_abs_contribution_share(states, weights)
    today_support = _weighted_bool(states, weights, "above_today_avwap")
    prev_support = _weighted_bool(states, weights, "above_prev_avwap")
    today_avwap_pressure = _weighted_optional(
        states, weights, "today_avwap_dist_atr"
    )
    prev_avwap_pressure = _weighted_optional(
        states, weights, "prev_session_avwap_dist_atr"
    )
    historical_position = _weighted_optional(
        states, weights, "historical_position_pct"
    )
    breadth = (
        float(breadth_positive_frac)
        if breadth_positive_frac is not None and math.isfinite(float(breadth_positive_frac))
        else positive_weight
    )

    metrics = {
        "index_return_pct": float(index_return_pct),
        "weighted_constituent_return_pct": float(weighted_return),
        "positive_weight_frac": float(positive_weight),
        "negative_weight_frac": float(negative_weight),
        "breadth_positive_frac": float(breadth),
        "top3_contribution_share": float(top3_contribution),
        "today_avwap_support_frac": float(today_support),
        "prev_avwap_support_frac": float(prev_support),
        "today_avwap_pressure_atr": float(today_avwap_pressure),
        "prev_session_avwap_pressure_atr": float(prev_avwap_pressure),
        "historical_position_pct": float(historical_position),
    }

    reasons: List[str] = []

    if index_return_pct > 0 and breadth < 0.45 and top3_contribution >= concentration_threshold:
        reasons += ["index_green_but_breadth_weak", "top3_heavyweight_concentration"]
        if today_support < 0.5:
            reasons.append("heavyweights_not_broadly_above_today_avwap")
        return ManipulationState(
            label="index_masking",
            confidence=_clip01(0.45 + top3_contribution * 0.35 + (0.45 - breadth)),
            action_bias="avoid_chasing_calls",
            reason_codes=reasons,
            metrics=metrics,
        )

    if index_return_pct > 0 and breadth >= 0.58 and today_support >= 0.6 and prev_support >= 0.55:
        return ManipulationState(
            label="broad_sponsorship",
            confidence=_clip01(0.35 + breadth * 0.35 + today_support * 0.2),
            action_bias="directional_setups_allowed",
            reason_codes=["breadth_supports_move", "heavyweights_above_avwap"],
            metrics=metrics,
        )

    if abs(index_return_pct) >= 0.35 and _sign(index_return_pct) != _sign(weighted_return):
        return ManipulationState(
            label="gap_distribution_trap",
            confidence=_clip01(0.5 + abs(index_return_pct - weighted_return) * 0.3),
            action_bias="wait_or_fade_late_chase",
            reason_codes=["index_gap_fights_weighted_constituents"],
            metrics=metrics,
        )

    if index_return_pct < 0 and weighted_return > index_return_pct and prev_support >= 0.5:
        return ManipulationState(
            label="dii_cushion_absorption",
            confidence=_clip01(0.4 + prev_support * 0.35 + max(0.0, weighted_return - index_return_pct) * 0.2),
            action_bias="watch_reclaim_not_breakdown_chase",
            reason_codes=["sell_pressure_absorbed_near_value", "prev_avwap_support_present"],
            metrics=metrics,
        )

    return ManipulationState(
        label="neutral_or_unclassified",
        confidence=0.25,
        action_bias="wait_for_clearer_state",
        reason_codes=["no_high_confidence_manipulation_state"],
        metrics=metrics,
    )


def _optional_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _optional_bool(value, fallback_float: Optional[float]) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is not None:
        s = str(value).strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
    if fallback_float is None:
        return None
    return fallback_float > 0


def _weighted_optional(states: List[ConstituentState], weights: List[float], attr: str) -> float:
    vals = []
    ws = []
    for s, w in zip(states, weights):
        v = getattr(s, attr)
        if v is not None and math.isfinite(float(v)):
            vals.append(float(v))
            ws.append(float(w))
    if not vals:
        return 0.0
    scale = sum(ws)
    return sum(w * v for w, v in zip(ws, vals)) / max(scale, 1e-9)


def _weighted_bool(states: List[ConstituentState], weights: List[float], attr: str) -> float:
    vals = []
    ws = []
    for s, w in zip(states, weights):
        v = getattr(s, attr)
        if v is not None:
            vals.append(1.0 if v else 0.0)
            ws.append(float(w))
    if not vals:
        return 0.0
    scale = sum(ws)
    return sum(w * v for w, v in zip(ws, vals)) / max(scale, 1e-9)


def _top_abs_contribution_share(states: List[ConstituentState], weights: List[float]) -> float:
    contribs = sorted((abs(w * s.return_pct) for w, s in zip(weights, states)), reverse=True)
    total = sum(contribs)
    if total <= 1e-12:
        return 0.0
    return sum(contribs[:3]) / total


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0

