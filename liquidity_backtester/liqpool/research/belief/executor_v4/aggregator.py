"""Decision aggregator v1 — the Sprint-2 brain.

Combines the Sprint-2 layered inputs (scenario web, forward projection,
adversarial critic, counterfactual, MTF alignment, economics) into a
single transparent decision rule:

  final_score = 0.30 * base_score          # confidence × engine
              + 0.25 * mtf_alignment       # multi-timeframe consensus
              + 0.20 * projection_factor   # historical-pathway support
              + 0.15 * fees_clearance      # EV margin above min edge
              + 0.10 * portfolio_capacity  # budget room remaining

…with hard overrides:
  * critic recommends REFUSE → refuse
  * tail_mass > tail_alarm   → refuse new entries, tighten open ones
  * chop_mass dominant + directional proposal → refuse or size_down
  * scenario web's dominant_strategy_class disagrees → size_down

The aggregator is transparent (no ML), so every refusal carries a
verbose reason chain that ends up in the ledger.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Outcomes the aggregator can emit.
AGGR_ACCEPT = "ACCEPT"
AGGR_REFUSE = "REFUSE"
AGGR_SIZE_DOWN = "SIZE_DOWN"
AGGR_TIGHTEN_OPEN = "TIGHTEN_OPEN"     # accept-with-portfolio-guard


@dataclass
class AggregatorConfig:
    """Knobs for the decision aggregator."""
    # Component weights — sum to 1.0.
    w_base_score: float = 0.26
    w_mtf_alignment: float = 0.22
    w_projection: float = 0.18
    w_fees_clearance: float = 0.13
    w_portfolio_capacity: float = 0.09
    # Founder 2026-06-22 Tier-2: rehearsal layer is the 6th component.
    # Conditional-kNN off-policy evaluation produces a calibrated
    # P(profit | current state, current entry params); when the
    # ensemble is uncertain it emits 0.50 which behaves as a neutral
    # prior and effectively yields the weight to the other components.
    w_rehearsal: float = 0.12
    # Decision thresholds.
    accept_threshold: float = 0.58
    size_down_threshold: float = 0.50
    # Strategy class agreement weight.
    strategy_disagreement_penalty: float = 0.12
    # Tail/chop alarms (read from web).
    tail_mass_refuse: float = 0.40
    chop_mass_size_down: float = 0.50
    # Projection.
    min_projection_samples: int = 20
    favorable_target_minus_stop: float = 0.10   # P(target hit) - P(stop hit) ≥ this
    # Rehearsal layer escalations — when the rehearsal recommendation is
    # DEFER and confidence is high enough, the aggregator refuses
    # outright (rehearsal saw the analogue distribution telling us this
    # state usually loses).
    rehearsal_defer_min_confidence: float = 0.30
    # ── ManipulationV2 AUTHORITATIVE direction gate (founder 2026-06-22
    # Tier-3): when manipulation_v2 says MM intent is OPPOSITE the
    # proposed direction with confidence ≥ this threshold, REFUSE
    # outright. This is a hard gate — no soft penalty, no neutral
    # prior. It is the "stop bleeding into MM" lever.
    mm_intent_contra_refuse_confidence: float = 0.60
    # When MM intent ALIGNS with proposed direction, multiply position
    # size by up to this much (capped). Default 1.0× = no boost; raise
    # to 1.5–2.0× to ride alignment harder.
    mm_intent_aligned_size_boost_max: float = 1.50
    # When MM intent partially contradicts (confidence between
    # alignment_floor and contra_refuse), size DOWN by this factor.
    mm_intent_partial_contra_size_haircut: float = 0.55
    # MM intent gets its own weight in the composite final score
    # (its directional vote weighted by confidence).
    w_mm_intent: float = 0.00     # Default 0 — pure direction gate; flip on for blended.


@dataclass
class AggregatorDecision:
    """One per-entry decision."""
    decision: str                         # ACCEPT / REFUSE / SIZE_DOWN / TIGHTEN_OPEN
    final_score: float                    # 0..1
    base_score: float
    mtf_alignment_score: float
    projection_factor: float
    fees_clearance_score: float
    portfolio_capacity_score: float
    critic_score: float
    critic_action: str
    web_strategy_class: str
    proposed_strategy_class: str
    recommended_size_multiplier: float    # 0..1
    refuse_reasons: List[str]             # populated when decision != ACCEPT
    component_explanations: List[str]
    # Founder 2026-06-22 Tier-2: rehearsal score + confidence + action.
    rehearsal_score: float = 0.50
    rehearsal_confidence: float = 0.0
    rehearsal_action: str = "PROCEED"
    # Founder 2026-06-22 Tier-3: ManipulationV2 size multiplier (applied
    # by the manager AFTER recommended_size_multiplier — boost when
    # aligned, haircut when partial contra). REFUSE is handled via
    # refuse_reasons, not this field.
    mm_intent_size_multiplier: float = 1.0
    mm_intent_direction: int = 0
    mm_intent_confidence: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "final_score": round(self.final_score, 3),
            "base_score": round(self.base_score, 3),
            "mtf_alignment_score": round(self.mtf_alignment_score, 3),
            "projection_factor": round(self.projection_factor, 3),
            "fees_clearance_score": round(self.fees_clearance_score, 3),
            "portfolio_capacity_score": round(self.portfolio_capacity_score, 3),
            "critic_score": round(self.critic_score, 3),
            "critic_action": self.critic_action,
            "web_strategy_class": self.web_strategy_class,
            "proposed_strategy_class": self.proposed_strategy_class,
            "recommended_size_multiplier": round(self.recommended_size_multiplier, 3),
            "refuse_reasons": list(self.refuse_reasons),
            "component_explanations": list(self.component_explanations),
            "rehearsal_score": round(self.rehearsal_score, 3),
            "rehearsal_confidence": round(self.rehearsal_confidence, 3),
            "rehearsal_action": self.rehearsal_action,
            "mm_intent_size_multiplier": round(
                self.mm_intent_size_multiplier, 3),
            "mm_intent_direction": int(self.mm_intent_direction),
            "mm_intent_confidence": round(
                self.mm_intent_confidence, 3),
            "notes": list(self.notes),
        }


class DecisionAggregator:
    """The brain.

    Stateless; the inputs are everything the manager has already computed
    for this tick.
    """

    def __init__(self, cfg: Optional[AggregatorConfig] = None) -> None:
        self.cfg = cfg or AggregatorConfig()
        # Sanity check the weight sum.
        total = (self.cfg.w_base_score + self.cfg.w_mtf_alignment
                 + self.cfg.w_projection + self.cfg.w_fees_clearance
                 + self.cfg.w_portfolio_capacity
                 + self.cfg.w_rehearsal)
        if abs(total - 1.0) > 0.05:
            raise ValueError(
                f"Aggregator weights must sum to ~1.0; got {total:.3f}"
            )

    def decide(self, *,
                base_confidence: float,
                mtf_alignment: Dict[str, Any],
                projection: Any,
                critique: Any,
                ev_decision: Any,
                web_snapshot: Any,
                proposed_strategy_class: str,
                proposed_direction: int,
                open_positions_count: int,
                max_open_positions: int,
                daily_pnl_rupees: float,
                max_daily_bleed: float,
                rehearsal_decision: Any = None,
                mm_intent: Optional[Dict[str, Any]] = None,
                ) -> AggregatorDecision:
        """Apply the transparent decision rule. Returns AggregatorDecision."""
        cfg = self.cfg
        explanations: List[str] = []
        refuse_reasons: List[str] = []
        notes: List[str] = []

        # ── Component 1: base_score = engine confidence ──────────────
        base_score = max(0.0, min(1.0, base_confidence))
        explanations.append(
            f"base score = engine confidence ({base_score:.2f})"
        )

        # ── Component 2: MTF alignment ──────────────────────────────
        mtf_score = float(mtf_alignment.get("alignment_score", 0.0))
        explanations.append(
            f"MTF alignment score {mtf_score:.2f} "
            f"(L1={mtf_alignment.get('l1_match')}, "
            f"L5={mtf_alignment.get('l5_match')}, "
            f"L15={mtf_alignment.get('l15_match')}, "
            f"L60={mtf_alignment.get('l60_match')})"
        )
        if not bool(mtf_alignment.get("alignment_ok", False)):
            refuse_reasons.append(
                "MTF alignment failed — L1 alone is retail"
            )

        # ── Component 3: projection factor ──────────────────────────
        projection_factor = 0.0
        if projection is not None:
            n = int(getattr(projection, "n_samples", 0))
            if n >= cfg.min_projection_samples:
                p_target = float(getattr(projection, "p_target_hit_first", 0.0))
                p_stop = float(getattr(projection, "p_stop_hit_first", 0.0))
                margin = p_target - p_stop
                # Map margin in [-1, +1] to a [0, 1] factor.
                projection_factor = max(0.0, min(1.0, 0.5 + margin * 0.7))
                explanations.append(
                    f"projection: n={n} matches, "
                    f"P(target_first)={p_target:.2f}, "
                    f"P(stop_first)={p_stop:.2f}, "
                    f"margin {margin:+.2f}"
                )
                if margin < -cfg.favorable_target_minus_stop:
                    refuse_reasons.append(
                        f"projection unfavorable: P(target) - P(stop) "
                        f"= {margin:+.2f} ≤ "
                        f"-{cfg.favorable_target_minus_stop:.2f}"
                    )
            else:
                # Fallback: neutral 0.50 prior with caveat.
                projection_factor = 0.50
                explanations.append(
                    f"projection: only n={n} matches; using neutral prior"
                )
        else:
            projection_factor = 0.50
            explanations.append("projection: unavailable — using neutral prior")

        # ── Component 4: fees clearance ─────────────────────────────
        # Map EV.edge_multiple to a [0,1] score. edge_multiple=1.0 → 0.5
        # (just clears costs). edge_multiple≥2.5 → 1.0.
        fees_score = 0.0
        if ev_decision is not None:
            em = float(getattr(ev_decision, "edge_multiple", 0.0))
            fees_score = max(0.0, min(1.0, (em - 1.0) / 1.5 * 0.5 + 0.5))
            explanations.append(
                f"fees clearance: EV edge multiple {em:.2f}x "
                f"(score {fees_score:.2f})"
            )
            if not bool(getattr(ev_decision, "approve", False)):
                refuse_reasons.append(
                    "EV gate rejected: fees + slippage eat expected profit"
                )

        # ── Component 5: portfolio capacity ─────────────────────────
        used_frac = (open_positions_count / max(1, max_open_positions))
        bleed_used = (-min(0.0, daily_pnl_rupees) / max(1.0, max_daily_bleed))
        portfolio_score = max(0.0, 1.0 - 0.5 * used_frac - 0.5 * bleed_used)
        explanations.append(
            f"portfolio capacity: {open_positions_count}/{max_open_positions} "
            f"slots, daily_pnl=₹{daily_pnl_rupees:.0f} "
            f"(score {portfolio_score:.2f})"
        )
        if bleed_used >= 1.0:
            refuse_reasons.append("daily bleed cap reached")

        # ── Critic refusal short-circuit ────────────────────────────
        critic_score = 0.0
        critic_action = "PROCEED"
        if critique is not None:
            critic_score = float(getattr(critique, "antithesis_score", 0.0))
            critic_action = str(getattr(critique, "recommended_action", "PROCEED"))
            explanations.append(
                f"critic: antithesis_score {critic_score:.2f}, action {critic_action}"
            )
            if critic_action == "REFUSE":
                refuse_reasons.append(
                    f"critic REFUSE (antithesis={critic_score:.2f})"
                )
            else:
                # Append the critic's top reasons as context.
                for r in (getattr(critique, "antithesis_reasons", []) or [])[:3]:
                    notes.append(f"critic: {r}")

        # ── Web overrides ───────────────────────────────────────────
        web_strategy = ""
        if web_snapshot is not None:
            tail_mass = float(getattr(web_snapshot, "tail_mass", 0.0))
            chop_mass = float(getattr(web_snapshot, "chop_mass", 0.0))
            web_strategy = str(getattr(web_snapshot, "dominant_strategy_class", ""))
            consensus = float(getattr(web_snapshot, "directional_consensus", 0.0))
            explanations.append(
                f"web: tail_mass={tail_mass:.2f}, chop_mass={chop_mass:.2f}, "
                f"consensus={consensus:+.2f}, dominant_strategy={web_strategy}"
            )
            if tail_mass >= cfg.tail_mass_refuse:
                refuse_reasons.append(
                    f"web tail_mass {tail_mass:.2f} ≥ "
                    f"refuse threshold {cfg.tail_mass_refuse:.2f}"
                )
            if (proposed_strategy_class in ("long_ce", "long_pe")
                    and chop_mass >= cfg.chop_mass_size_down):
                notes.append(
                    f"web chop_mass {chop_mass:.2f} dominant — directional sizing down"
                )
            # Directional consensus contradicts proposal?
            if (proposed_direction > 0 and consensus < -0.30) \
                    or (proposed_direction < 0 and consensus > 0.30):
                refuse_reasons.append(
                    f"web directional_consensus {consensus:+.2f} contradicts "
                    f"proposed direction {proposed_direction:+d}"
                )

        # ── Component 6: rehearsal (conditional-kNN off-policy eval) ──
        # Founder 2026-06-22 Tier-2: the rehearsal ensemble looks at the
        # K most-similar historical trades to the current state and
        # reports P(profit | this state). When confidence is low (no
        # history yet, or no analogues nearby) the ensemble emits 0.50
        # which is a neutral prior — the aggregator effectively gets
        # nothing from this weight in cold-start sessions but inherits
        # institutional memory once the store accumulates.
        rehearsal_score = 0.50
        rehearsal_confidence = 0.0
        rehearsal_action = "PROCEED"
        if rehearsal_decision is not None:
            rehearsal_score = float(getattr(rehearsal_decision,
                                              "rehearsal_score", 0.50))
            rehearsal_confidence = float(getattr(rehearsal_decision,
                                                   "confidence", 0.0))
            rehearsal_action = str(getattr(rehearsal_decision,
                                              "recommended_action", "PROCEED"))
            n_anal = int(getattr(rehearsal_decision,
                                   "n_analogues_total", 0))
            best_name = str(getattr(rehearsal_decision,
                                       "best_perturbation_name", ""))
            explanations.append(
                f"rehearsal: score {rehearsal_score:.2f} "
                f"(conf {rehearsal_confidence:.2f}, n={n_anal}, "
                f"best='{best_name}', action {rehearsal_action})"
            )
            if (rehearsal_action == "DEFER"
                    and rehearsal_confidence
                    >= cfg.rehearsal_defer_min_confidence):
                refuse_reasons.append(
                    f"rehearsal DEFER (conf {rehearsal_confidence:.2f}): "
                    f"P(profit) too low for analogues of this state")
            # Append the top rehearsal notes as context.
            for r in (getattr(rehearsal_decision, "notes", []) or [])[:2]:
                notes.append(f"rehearsal: {r}")

        # ── ManipulationV2 AUTHORITATIVE direction gate (Tier-3) ────
        # When MMIntent contradicts the proposed direction with
        # confidence above mm_intent_contra_refuse_confidence, we
        # REFUSE outright. This is the founder's "stop bleeding into
        # MM" lever — no soft penalty, hard gate.
        mm_intent_size_multiplier = 1.0
        if mm_intent is not None and isinstance(mm_intent, dict):
            mm_dir = int(mm_intent.get("direction", 0) or 0)
            mm_conf = float(mm_intent.get("confidence", 0.0) or 0.0)
            mm_regime = str(mm_intent.get("regime") or "")
            mm_gamma = str(mm_intent.get("gamma_regime") or "")
            mm_fires = int(mm_intent.get("fire_count", 0) or 0)
            mm_target = mm_intent.get("targeted_strike")
            override = mm_intent.get("operator_override")
            explanations.append(
                f"mm_intent: dir={mm_dir:+d} conf={mm_conf:.2f} "
                f"fires={mm_fires} regime={mm_regime} gamma={mm_gamma}"
                + (" [OVERRIDE]" if override else ""))
            if mm_dir != 0 and proposed_direction != 0:
                if (mm_dir == -proposed_direction
                        and mm_conf >= cfg.mm_intent_contra_refuse_confidence):
                    refuse_reasons.append(
                        f"manipulation_v2: MM intent {mm_dir:+d} "
                        f"contradicts proposed {proposed_direction:+d} "
                        f"with confidence {mm_conf:.2f} ≥ "
                        f"{cfg.mm_intent_contra_refuse_confidence:.2f} "
                        f"— authoritative refuse"
                        + (f" (regime: {mm_regime})" if mm_regime else "")
                    )
                elif (mm_dir == -proposed_direction
                      and mm_conf >= 0.30):
                    mm_intent_size_multiplier *= (
                        cfg.mm_intent_partial_contra_size_haircut)
                    notes.append(
                        f"mm_intent partial contra (conf {mm_conf:.2f}) — "
                        f"sizing × "
                        f"{cfg.mm_intent_partial_contra_size_haircut:.2f}")
                elif mm_dir == proposed_direction and mm_conf >= 0.40:
                    boost = 1.0 + min(
                        cfg.mm_intent_aligned_size_boost_max - 1.0,
                        (mm_conf - 0.40) * 1.5)
                    mm_intent_size_multiplier *= boost
                    notes.append(
                        f"mm_intent aligned (conf {mm_conf:.2f}) — "
                        f"sizing × {boost:.2f}"
                        + (f" → targeting strike {mm_target:.0f}"
                            if mm_target else ""))
            # Optional blended weight — disabled by default
            # (mm_intent_contra_refuse + size haircut is the primary
            # mechanism).
            mm_signed = (mm_dir * mm_conf if (mm_dir == proposed_direction
                                                 and proposed_direction != 0)
                          else (-mm_conf if mm_dir == -proposed_direction
                                  else 0.0))
        else:
            mm_signed = 0.0

        # ── Final composite score ───────────────────────────────────
        final_score = (
            cfg.w_base_score * base_score
            + cfg.w_mtf_alignment * mtf_score
            + cfg.w_projection * projection_factor
            + cfg.w_fees_clearance * fees_score
            + cfg.w_portfolio_capacity * portfolio_score
            + cfg.w_rehearsal * rehearsal_score
            + cfg.w_mm_intent * (mm_signed + 0.5)
        )
        # Strategy class disagreement penalty.
        if web_strategy and web_strategy != "wait" \
                and web_strategy != proposed_strategy_class:
            final_score -= cfg.strategy_disagreement_penalty
            notes.append(
                f"strategy class disagreement: web prefers {web_strategy}, "
                f"proposal {proposed_strategy_class} — penalty applied"
            )
        final_score = max(0.0, min(1.0, final_score))

        # ── Decision ────────────────────────────────────────────────
        recommended_size = 1.0
        decision = AGGR_ACCEPT

        if refuse_reasons:
            decision = AGGR_REFUSE
            recommended_size = 0.0
        elif final_score < cfg.size_down_threshold:
            decision = AGGR_REFUSE
            refuse_reasons.append(
                f"final score {final_score:.2f} < size_down threshold "
                f"{cfg.size_down_threshold:.2f}"
            )
            recommended_size = 0.0
        elif final_score < cfg.accept_threshold:
            decision = AGGR_SIZE_DOWN
            # Linear interpolation between size_down threshold (0.5×) and
            # accept threshold (1.0×).
            span = max(0.01, cfg.accept_threshold - cfg.size_down_threshold)
            frac = (final_score - cfg.size_down_threshold) / span
            recommended_size = 0.5 + 0.5 * frac
        else:
            decision = AGGR_ACCEPT
            recommended_size = 1.0

        # Critic haircut compounds the size.
        if critique is not None:
            critic_haircut = float(getattr(critique, "recommended_size_haircut", 1.0))
            recommended_size *= critic_haircut

        return AggregatorDecision(
            decision=decision,
            final_score=final_score,
            base_score=base_score,
            mtf_alignment_score=mtf_score,
            projection_factor=projection_factor,
            fees_clearance_score=fees_score,
            portfolio_capacity_score=portfolio_score,
            critic_score=critic_score,
            critic_action=critic_action,
            web_strategy_class=web_strategy,
            proposed_strategy_class=proposed_strategy_class,
            recommended_size_multiplier=max(0.0, min(1.0, recommended_size)),
            refuse_reasons=refuse_reasons,
            component_explanations=explanations,
            rehearsal_score=rehearsal_score,
            rehearsal_confidence=rehearsal_confidence,
            rehearsal_action=rehearsal_action,
            mm_intent_size_multiplier=float(mm_intent_size_multiplier),
            mm_intent_direction=int(
                mm_intent.get("direction", 0) if isinstance(mm_intent, dict)
                else 0),
            mm_intent_confidence=float(
                mm_intent.get("confidence", 0.0) if isinstance(mm_intent, dict)
                else 0.0),
            notes=notes,
        )
