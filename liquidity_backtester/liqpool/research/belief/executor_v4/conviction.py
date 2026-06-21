"""Conviction score — continuous signal strength (sign + magnitude).

The founder's observation (2026-06-21): "the `direction` number only
shows 0, -1, +1. It should be more flexible with more range, so we can
conclude THIS is a strong signal and THIS is a weak signal."

He's right. ``direction`` is a pure sign — magnitude lives separately in
``confidence`` and in the scenario web's ``directional_consensus``, and
those pieces never get combined into one legible "how strong is this
really?" number.

This module computes a single ``ConvictionScore`` — a continuous signed
value in [-1, +1] where:
  * SIGN  = trade direction (+ bull / - bear)
  * MAGNITUDE = strength, blended from:
      - engine confidence            (0..1)
      - web directional consensus    (|−1..+1|, the multi-scenario vote)
      - MTF alignment score          (0..1, longer-timeframe agreement)
      - (1 − antithesis)             (critic's skepticism, inverted)

It is then classified into a human label
(STRONG_BULL / MODERATE_BULL / WEAK_BULL / NEUTRAL / WEAK_BEAR / ...).

CRITICAL DESIGN CHOICE: this is ADDITIVE. The discrete ``direction``
field that hundreds of downstream comparisons rely on (``direction > 0``,
``== +1``) is untouched. Conviction is a NEW continuous quantity that
feeds sizing and the cockpit — it never replaces the sign.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Strength labels ───────────────────────────────────────────────


CONVICTION_STRONG_BULL = "STRONG_BULL"
CONVICTION_MODERATE_BULL = "MODERATE_BULL"
CONVICTION_WEAK_BULL = "WEAK_BULL"
CONVICTION_NEUTRAL = "NEUTRAL"
CONVICTION_WEAK_BEAR = "WEAK_BEAR"
CONVICTION_MODERATE_BEAR = "MODERATE_BEAR"
CONVICTION_STRONG_BEAR = "STRONG_BEAR"


@dataclass
class ConvictionConfig:
    """Blend weights for the conviction magnitude. Sum ~ 1.0."""
    w_confidence: float = 0.35
    w_web_consensus: float = 0.30
    w_mtf_alignment: float = 0.20
    w_anti_skepticism: float = 0.15
    # Strength band thresholds (on |signed_value|).
    strong_threshold: float = 0.70
    moderate_threshold: float = 0.45
    weak_threshold: float = 0.20

    def __post_init__(self) -> None:
        total = (self.w_confidence + self.w_web_consensus
                 + self.w_mtf_alignment + self.w_anti_skepticism)
        if abs(total - 1.0) > 0.05:
            raise ValueError(
                f"conviction weights must sum to ~1.0; got {total:.3f}")


@dataclass
class ConvictionScore:
    """Continuous signed signal strength."""
    direction: int                # the discrete sign (unchanged contract)
    signed_value: float           # -1..+1 (sign × magnitude)
    magnitude: float              # 0..1 (pure strength)
    label: str                    # STRONG_BULL / ... / STRONG_BEAR
    components: Dict[str, float]  # per-input contribution
    size_multiplier: float        # 0.25..1.0 — direct sizing lever
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "signed_value": round(self.signed_value, 4),
            "magnitude": round(self.magnitude, 4),
            "label": self.label,
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "size_multiplier": round(self.size_multiplier, 3),
            "notes": list(self.notes),
        }


def compute_conviction(*,
                         direction: int,
                         confidence: float,
                         web_directional_consensus: float = 0.0,
                         mtf_alignment_score: float = 0.0,
                         antithesis_score: float = 0.0,
                         cfg: Optional[ConvictionConfig] = None,
                         ) -> ConvictionScore:
    """Blend the strength inputs into one continuous conviction score.

    ``web_directional_consensus`` is signed (−1..+1); we take the
    component that AGREES with the proposed direction (a bull trade only
    gets web credit for bullish consensus, not bearish).
    """
    cfg = cfg or ConvictionConfig()
    notes: List[str] = []

    conf = _clamp01(confidence)
    # Only count web consensus that agrees with our direction.
    if direction > 0:
        web_agree = max(0.0, web_directional_consensus)
    elif direction < 0:
        web_agree = max(0.0, -web_directional_consensus)
    else:
        web_agree = 0.0
    web_agree = _clamp01(web_agree)
    mtf = _clamp01(mtf_alignment_score)
    anti_inv = _clamp01(1.0 - antithesis_score)

    components = {
        "confidence": cfg.w_confidence * conf,
        "web_consensus": cfg.w_web_consensus * web_agree,
        "mtf_alignment": cfg.w_mtf_alignment * mtf,
        "anti_skepticism": cfg.w_anti_skepticism * anti_inv,
    }
    magnitude = _clamp01(sum(components.values()))
    signed = direction * magnitude

    label = _label_for(direction, magnitude, cfg)

    # Sizing lever: maps magnitude → [0.25, 1.0]. A weak signal gets
    # quarter size; a strong one gets full. This is the founder's
    # "strong signal = bigger, weak signal = smaller" made concrete.
    size_multiplier = 0.25 + 0.75 * magnitude

    if magnitude < cfg.weak_threshold:
        notes.append("conviction below weak threshold — sizing heavily reduced")
    if web_agree < 0.05 and abs(web_directional_consensus) > 0.20:
        notes.append("web consensus DISAGREES with proposed direction "
                      "— magnitude penalized")

    return ConvictionScore(
        direction=direction,
        signed_value=round(signed, 4),
        magnitude=round(magnitude, 4),
        label=label,
        components=components,
        size_multiplier=round(size_multiplier, 3),
        notes=notes,
    )


def _label_for(direction: int, magnitude: float,
                cfg: ConvictionConfig) -> str:
    if direction == 0 or magnitude < cfg.weak_threshold:
        return CONVICTION_NEUTRAL
    if direction > 0:
        if magnitude >= cfg.strong_threshold:
            return CONVICTION_STRONG_BULL
        if magnitude >= cfg.moderate_threshold:
            return CONVICTION_MODERATE_BULL
        return CONVICTION_WEAK_BULL
    if magnitude >= cfg.strong_threshold:
        return CONVICTION_STRONG_BEAR
    if magnitude >= cfg.moderate_threshold:
        return CONVICTION_MODERATE_BEAR
    return CONVICTION_WEAK_BEAR


def _clamp01(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return max(0.0, min(1.0, x))
