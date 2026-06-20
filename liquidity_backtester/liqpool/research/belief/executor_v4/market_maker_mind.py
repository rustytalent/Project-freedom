"""Market-maker mind — Bayesian posterior over MM intent.

Aggregates the active manipulation patterns plus the latest flow event
into a probability distribution over what the dominant market-maker is
*currently doing*:

  * pinning_near_expiry
  * accumulating
  * distributing
  * hunting_stops
  * faking_direction
  * stepping_back
  * neutral_inventory

The output ``MMPosterior`` carries the full distribution plus a single
``implied_bias`` (the founder asks "what is the MM intending in this
moment?") and an operator-readable ``operator_guidance`` string.

Design choice: TRANSPARENT Bayesian update. Each detected pattern is a
piece of evidence with a fixed likelihood ratio per hypothesis; the
posterior multiplies these ratios against a flat prior, renormalizes,
and returns.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .manipulation_patterns import (
    PATTERN_ACCUMULATION,
    PATTERN_DISTRIBUTION,
    PATTERN_FAKE_BREAKOUT,
    PATTERN_ICEBERG_BUY,
    PATTERN_LIQUIDITY_VACUUM,
    PATTERN_MM_INVENTORY_FLIP,
    PATTERN_PIN_NEAR_EXPIRY,
    PATTERN_SQUEEZE_SETUP,
    PATTERN_STOP_HUNT_LONG,
    PATTERN_STOP_HUNT_SHORT,
    PatternMatch,
)


# Intent labels (kept in sync with manipulation_patterns).
INTENT_PINNING = "pinning_near_expiry"
INTENT_ACCUMULATING = "accumulating"
INTENT_DISTRIBUTING = "distributing"
INTENT_HUNTING_STOPS = "hunting_stops"
INTENT_FAKING_DIRECTION = "faking_direction"
INTENT_STEPPING_BACK = "stepping_back"
INTENT_NEUTRAL = "neutral_inventory"

ALL_INTENTS = (
    INTENT_PINNING, INTENT_ACCUMULATING, INTENT_DISTRIBUTING,
    INTENT_HUNTING_STOPS, INTENT_FAKING_DIRECTION,
    INTENT_STEPPING_BACK, INTENT_NEUTRAL,
)


@dataclass
class MMPosterior:
    """Bayesian posterior over MM intent."""
    intent_distribution: Dict[str, float]  # sums to 1.0
    active_patterns: List[str]
    implied_bias: int                       # +1 / -1 / 0
    implied_volatility_view: str            # "expansion" / "crush" / "neutral"
    dominant_intent: str
    dominant_probability: float
    confidence: float                       # 0..1, scales with evidence count
    operator_guidance: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_distribution": {k: round(v, 4)
                                     for k, v in self.intent_distribution.items()},
            "active_patterns": list(self.active_patterns),
            "implied_bias": self.implied_bias,
            "implied_volatility_view": self.implied_volatility_view,
            "dominant_intent": self.dominant_intent,
            "dominant_probability": round(self.dominant_probability, 4),
            "confidence": round(self.confidence, 3),
            "operator_guidance": self.operator_guidance,
            "notes": list(self.notes),
        }


# Likelihood ratios for each pattern → intent. Values > 1.0 boost
# that intent; values < 1.0 dampen. Hand-calibrated.
_LIKELIHOODS: Dict[str, Dict[str, float]] = {
    PATTERN_STOP_HUNT_SHORT: {
        INTENT_HUNTING_STOPS: 3.0, INTENT_FAKING_DIRECTION: 2.0,
        INTENT_DISTRIBUTING: 1.3,
    },
    PATTERN_STOP_HUNT_LONG: {
        INTENT_HUNTING_STOPS: 3.0, INTENT_FAKING_DIRECTION: 2.0,
        INTENT_ACCUMULATING: 1.3,
    },
    PATTERN_LIQUIDITY_VACUUM: {
        INTENT_STEPPING_BACK: 3.5, INTENT_NEUTRAL: 0.5,
    },
    PATTERN_PIN_NEAR_EXPIRY: {
        INTENT_PINNING: 4.0, INTENT_NEUTRAL: 1.2,
        INTENT_HUNTING_STOPS: 0.7,
    },
    PATTERN_ACCUMULATION: {
        INTENT_ACCUMULATING: 3.5, INTENT_NEUTRAL: 1.1,
        INTENT_DISTRIBUTING: 0.3,
    },
    PATTERN_DISTRIBUTION: {
        INTENT_DISTRIBUTING: 3.5, INTENT_NEUTRAL: 1.1,
        INTENT_ACCUMULATING: 0.3,
    },
    PATTERN_SQUEEZE_SETUP: {
        INTENT_NEUTRAL: 2.0, INTENT_PINNING: 1.5,
    },
    PATTERN_FAKE_BREAKOUT: {
        INTENT_FAKING_DIRECTION: 3.5, INTENT_HUNTING_STOPS: 1.5,
    },
    PATTERN_ICEBERG_BUY: {
        INTENT_ACCUMULATING: 3.5, INTENT_DISTRIBUTING: 0.4,
    },
    PATTERN_MM_INVENTORY_FLIP: {
        INTENT_NEUTRAL: 2.5, INTENT_STEPPING_BACK: 1.5,
    },
}

# Per-intent volatility view.
_INTENT_VOL_VIEW: Dict[str, str] = {
    INTENT_PINNING: "crush",
    INTENT_ACCUMULATING: "expansion",
    INTENT_DISTRIBUTING: "expansion",
    INTENT_HUNTING_STOPS: "expansion",
    INTENT_FAKING_DIRECTION: "expansion",
    INTENT_STEPPING_BACK: "expansion",
    INTENT_NEUTRAL: "neutral",
}

# Per-intent directional bias.
_INTENT_BIAS: Dict[str, int] = {
    INTENT_PINNING: 0,
    INTENT_ACCUMULATING: +1,
    INTENT_DISTRIBUTING: -1,
    INTENT_HUNTING_STOPS: 0,
    INTENT_FAKING_DIRECTION: 0,
    INTENT_STEPPING_BACK: 0,
    INTENT_NEUTRAL: 0,
}


@dataclass
class MarketMakerMindConfig:
    """Knobs for the MM-mind inference."""
    floor_probability: float = 0.02       # never let any intent go to 0
    confidence_max_patterns: int = 4      # confidence saturates here


class MarketMakerMind:
    """The MM-mind Bayesian aggregator.

    Stateless: ``infer(...)`` returns a fresh ``MMPosterior`` from the
    active pattern matches.
    """

    def __init__(self, cfg: Optional[MarketMakerMindConfig] = None) -> None:
        self.cfg = cfg or MarketMakerMindConfig()

    def infer(self, *, patterns: List[PatternMatch],
               flow_event: Any = None) -> MMPosterior:
        """Run a Bayesian update from flat prior using each pattern's
        likelihood ratios weighted by its confidence."""
        cfg = self.cfg
        # Flat prior.
        posterior: Dict[str, float] = {k: 1.0 / len(ALL_INTENTS)
                                          for k in ALL_INTENTS}

        active_names: List[str] = []
        for m in patterns:
            active_names.append(m.pattern_name)
            ratios = _LIKELIHOODS.get(m.pattern_name)
            if not ratios:
                continue
            # Weight each ratio by the pattern's confidence.
            w = max(0.10, min(1.0, m.confidence))
            for intent, ratio in ratios.items():
                # Blend ratio with 1.0 by pattern confidence.
                effective = 1.0 + (ratio - 1.0) * w
                posterior[intent] = posterior.get(intent, 0.0) * effective

        # Floor + renormalize.
        for k in posterior:
            if posterior[k] < cfg.floor_probability:
                posterior[k] = cfg.floor_probability
        total = sum(posterior.values())
        if total > 0:
            for k in posterior:
                posterior[k] /= total

        dom = max(posterior.items(), key=lambda kv: kv[1])
        confidence = min(1.0, len(patterns) / cfg.confidence_max_patterns)

        bias = _INTENT_BIAS.get(dom[0], 0)
        vol_view = _INTENT_VOL_VIEW.get(dom[0], "neutral")

        # Build operator-readable guidance.
        guidance = _build_guidance(dom[0], dom[1], confidence, active_names)

        notes: List[str] = []
        if confidence < 0.30:
            notes.append("MM intent inference low-confidence; few patterns")
        if dom[1] >= 0.50:
            notes.append(f"MM intent {dom[0]} dominant ({dom[1]:.0%})")

        return MMPosterior(
            intent_distribution=posterior,
            active_patterns=active_names,
            implied_bias=bias,
            implied_volatility_view=vol_view,
            dominant_intent=dom[0],
            dominant_probability=dom[1],
            confidence=confidence,
            operator_guidance=guidance,
            notes=notes,
        )


def _build_guidance(intent: str, prob: float, confidence: float,
                     active: List[str]) -> str:
    """Compose the operator-facing guidance string."""
    base = f"MM appears to be {intent.replace('_', ' ')} ({prob:.0%})"
    if confidence < 0.30:
        base += " — low-confidence read; weight lightly"
    elif confidence < 0.60:
        base += " — moderate confidence"
    else:
        base += " — high confidence"

    if intent == INTENT_PINNING:
        return base + ". Prefer non-directional structures near the strike."
    if intent == INTENT_ACCUMULATING:
        return base + ". Bias long; avoid shorts into absorption."
    if intent == INTENT_DISTRIBUTING:
        return base + ". Bias short; avoid longs into distribution."
    if intent == INTENT_HUNTING_STOPS:
        return base + ". Avoid placing tight stops; wait for the hunt to resolve."
    if intent == INTENT_FAKING_DIRECTION:
        return base + ". Don't chase breakouts; wait for follow-through confirmation."
    if intent == INTENT_STEPPING_BACK:
        return base + ". Reduce size, use limit orders, expect spread widening."
    if intent == INTENT_NEUTRAL:
        return base + ". Standard sizing; no MM tailwind/headwind detected."
    return base + "."
