"""ModificationGate — should we spend a modification?

ChatGPT's exact formula (and the founder agreed):

    modify_score = fill_probability_gain
                 + adverse_move_risk_reduction
                 + thesis_decay_score
                 + portfolio_risk_reduction
                 - modification_budget_cost
                 - noise_penalty

    if modify_score > threshold:
        modify broker order
    else:
        only update ghost price

Threshold rises as budget shrinks:
    mods 0-5:   threshold low      (0.20)
    mods 6-15:  threshold medium   (0.40)
    mods 16-20: threshold high     (0.60)
    mods 21-25: emergency only     (0.85)

Refinements added beyond ChatGPT's sketch:

  * **Liquidity-awareness**: if the spread has widened or friendliness
    dropped, the modification is less likely to fill regardless of
    price, so the gate refuses unless the move is large.

  * **Minimum meaningful change**: the new price must be ≥ 2x spread
    OR ≥ 30% of micro-volatility from the old broker price. Cosmetic
    20-paise tweaks always refused.

  * **Mode-pair multipliers**: KILL mode → always approves (uses
    emergency reserve). HARVEST mode → refuses unless improvement is
    very large.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Config ────────────────────────────────────────────────────────


@dataclass
class ModificationGateConfig:
    """Knobs for the gate."""
    # Threshold schedule
    threshold_low_floor: int = 5           # mods <= this → low threshold
    threshold_medium_floor: int = 15
    threshold_high_floor: int = 20
    threshold_low: float = 0.20
    threshold_medium: float = 0.40
    threshold_high: float = 0.60
    threshold_emergency: float = 0.85

    # Minimum meaningful change (in price terms)
    min_change_spread_multiple: float = 2.0
    min_change_volatility_fraction: float = 0.30
    min_change_absolute_rupees: float = 0.50

    # Score weights (founder + ChatGPT formula)
    w_fill_probability_gain: float = 1.0
    w_adverse_move_risk_reduction: float = 0.8
    w_thesis_decay_score: float = 0.7
    w_portfolio_risk_reduction: float = 0.5
    w_budget_pressure_penalty: float = 1.0
    w_noise_penalty: float = 0.8

    # Mode adjustments
    harvest_mode_threshold_multiplier: float = 1.5    # harder to modify in harvest
    chase_mode_threshold_multiplier: float = 0.7      # easier in chase
    derisk_mode_threshold_multiplier: float = 0.6     # easier in de-risk


@dataclass
class ModificationGateDecision:
    """The gate's verdict on one modification proposal."""
    approve: bool
    score: float
    threshold_used: float
    components: Dict[str, float]
    reasons: List[str]
    suggested_bucket: str             # which budget bucket to charge
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approve": self.approve,
            "score": round(self.score, 4),
            "threshold_used": round(self.threshold_used, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "reasons": list(self.reasons),
            "suggested_bucket": self.suggested_bucket,
            "notes": list(self.notes),
        }


# ── Gate ──────────────────────────────────────────────────────────


from .budget import ModificationBucket


class ModificationGate:
    """Stateless decision functor for modifications."""

    def __init__(self, cfg: Optional[ModificationGateConfig] = None) -> None:
        self.cfg = cfg or ModificationGateConfig()

    def evaluate(self, *,
                   old_broker_price: float,
                   new_ghost_price: float,
                   bars_held: int,
                   total_mods_used: int,
                   budget_pressure: float,    # 0..1
                   exit_mode: str,
                   spread: float,
                   micro_volatility: float,
                   fill_probability_gain: float,    # 0..1
                   thesis_confidence_decay: float,  # 0..1
                   portfolio_risk_relief: float = 0.0,  # 0..1
                   adverse_move_risk_reduction: float = 0.0,
                   ) -> ModificationGateDecision:
        """Decide whether to spend a modification.

        Returns ``approve=True`` if the modification should be sent
        to the broker, with the suggested bucket to charge.
        """
        cfg = self.cfg
        reasons: List[str] = []
        notes: List[str] = []

        if old_broker_price <= 0 or new_ghost_price <= 0:
            return _refuse("non-positive prices",
                            suggested_bucket=ModificationBucket.NORMAL_ADAPTIVE,
                            components={})

        # KILL mode: always approve (emergency).
        if exit_mode == "KILL":
            return ModificationGateDecision(
                approve=True, score=1.0,
                threshold_used=0.0,
                components={"kill_override": 1.0},
                reasons=["KILL mode — bypass gate"],
                suggested_bucket=ModificationBucket.EMERGENCY_RESERVE,
                notes=["bypass: KILL mode"],
            )

        # ── Minimum meaningful change check ─────────────────────────
        change = abs(new_ghost_price - old_broker_price)
        min_change = max(
            spread * cfg.min_change_spread_multiple,
            micro_volatility * cfg.min_change_volatility_fraction,
            cfg.min_change_absolute_rupees,
        )
        if change < min_change:
            return _refuse(
                f"change ₹{change:.2f} < min meaningful ₹{min_change:.2f}",
                suggested_bucket=_pick_bucket(exit_mode, bars_held,
                                                 total_mods_used),
                components={"change": change, "min_change": min_change},
            )

        # ── Compose the score ───────────────────────────────────────
        components: Dict[str, float] = {}
        components["fill_probability_gain"] = (
            cfg.w_fill_probability_gain
            * max(0.0, min(1.0, fill_probability_gain)))
        components["adverse_move_risk_reduction"] = (
            cfg.w_adverse_move_risk_reduction
            * max(0.0, min(1.0, adverse_move_risk_reduction)))
        components["thesis_decay_score"] = (
            cfg.w_thesis_decay_score
            * max(0.0, min(1.0, thesis_confidence_decay)))
        components["portfolio_risk_reduction"] = (
            cfg.w_portfolio_risk_reduction
            * max(0.0, min(1.0, portfolio_risk_relief)))
        components["budget_pressure_penalty"] = (
            -cfg.w_budget_pressure_penalty
            * max(0.0, min(1.0, budget_pressure)))
        # Noise penalty: if change is just slightly above min, penalize a bit.
        noise = max(0.0, 1.0 - (change - min_change) / max(min_change, 1e-6))
        components["noise_penalty"] = -cfg.w_noise_penalty * noise

        score = sum(components.values())

        # ── Threshold (mode-adjusted) ───────────────────────────────
        threshold = self._threshold_for(total_mods_used)
        if exit_mode == "HARVEST":
            threshold *= cfg.harvest_mode_threshold_multiplier
        elif exit_mode == "CHASE_FILL":
            threshold *= cfg.chase_mode_threshold_multiplier
        elif exit_mode == "DE_RISK":
            threshold *= cfg.derisk_mode_threshold_multiplier

        approve = score > threshold
        bucket = _pick_bucket(exit_mode, bars_held, total_mods_used)
        if approve:
            reasons.append(
                f"score {score:.2f} > threshold {threshold:.2f}; "
                f"bucket={bucket}"
            )
        else:
            reasons.append(
                f"score {score:.2f} ≤ threshold {threshold:.2f}; ghost only"
            )

        return ModificationGateDecision(
            approve=approve,
            score=score,
            threshold_used=threshold,
            components=components,
            reasons=reasons,
            suggested_bucket=bucket,
            notes=notes,
        )

    def _threshold_for(self, total_mods_used: int) -> float:
        cfg = self.cfg
        if total_mods_used <= cfg.threshold_low_floor:
            return cfg.threshold_low
        if total_mods_used <= cfg.threshold_medium_floor:
            return cfg.threshold_medium
        if total_mods_used <= cfg.threshold_high_floor:
            return cfg.threshold_high
        return cfg.threshold_emergency


# ── Helpers ───────────────────────────────────────────────────────


def _refuse(reason: str, *, suggested_bucket: str,
              components: Dict[str, float]) -> ModificationGateDecision:
    return ModificationGateDecision(
        approve=False, score=0.0, threshold_used=0.0,
        components=components, reasons=[reason],
        suggested_bucket=suggested_bucket,
        notes=[],
    )


def _pick_bucket(exit_mode: str, bars_held: int,
                  total_mods_used: int) -> str:
    """Pick which budget bucket to charge."""
    if exit_mode == "KILL" or exit_mode == "DE_RISK":
        return ModificationBucket.EMERGENCY_RESERVE
    if bars_held <= 3:
        return ModificationBucket.EARLY_CORRECTION
    if exit_mode == "CHASE_FILL":
        return ModificationBucket.ENDGAME_CHASE
    return ModificationBucket.NORMAL_ADAPTIVE
