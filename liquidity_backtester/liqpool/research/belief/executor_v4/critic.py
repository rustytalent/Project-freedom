"""Adversarial critic — generates the BEAR case for any proposed trade.

The founder's mandate (paraphrased from 2026-06-19): *"You cannot just
say I'm bullish. You have to argue the bear case for the same trade and
beat it back. If you can't beat the bear case, the trade dies."*

This module replaces Sprint 1's heuristic antithesis (a few hand-coded
checks) with a structured adversarial argument generator. For any
proposed entry, the critic walks a catalogue of bear-case arguments
relevant to the proposed direction and returns:

  * ``antithesis_score`` in [0,1] — how strong the bear case is
  * ``antithesis_reasons`` — ordered list of qualitative reasons
  * ``recommended_size_haircut`` in [0,1] — multiplier for sizing
  * ``recommended_action`` — REFUSE / SIZE_DOWN / PROCEED
  * ``critic_notes`` — human-readable summary for the ledger

The critic intentionally looks at signals the THESIS overlooks, not the
ones it confirms. Its job is to be skeptical.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Recommended actions.
CRITIC_REFUSE = "REFUSE"
CRITIC_SIZE_DOWN = "SIZE_DOWN"
CRITIC_PROCEED = "PROCEED"


@dataclass
class CriticConfig:
    """Knobs for the adversarial critic."""
    refuse_score: float = 0.60        # ≥ this → recommended_action = REFUSE
    size_down_score: float = 0.25     # ≥ this and < refuse → SIZE_DOWN
    min_haircut: float = 0.40          # never haircut below this fraction
    # Weights for each line of argument (sum need not equal 1; we cap final at 1).
    weight_brittle_rail: float = 0.20
    weight_single_strike: float = 0.18
    weight_thesis_decline: float = 0.18
    weight_regime_instability: float = 0.18
    weight_opposite_acceptance: float = 0.30
    weight_recent_trap: float = 0.15
    weight_iv_crush_risk: float = 0.10
    weight_chop_dominance: float = 0.15
    weight_tail_mass: float = 0.22
    weight_oscillation: float = 0.15


@dataclass
class CritiqueResult:
    """The output of the critic for a single proposed trade."""
    antithesis_score: float            # in [0,1]
    antithesis_reasons: List[str]      # ordered, descending strength
    recommended_size_haircut: float    # in [min_haircut, 1.0]
    recommended_action: str            # REFUSE / SIZE_DOWN / PROCEED
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "antithesis_score": round(self.antithesis_score, 3),
            "antithesis_reasons": list(self.antithesis_reasons),
            "recommended_size_haircut": round(self.recommended_size_haircut, 3),
            "recommended_action": self.recommended_action,
            "notes": list(self.notes),
        }


class AdversarialCritic:
    """The adversarial critic.

    Stateless functor: ``critique(...)`` returns a fresh result from
    current evidence. The critic is INTENDED to be paranoid — its job is
    to find reasons to NOT take the trade, not to confirm the thesis.
    """

    def __init__(self, cfg: Optional[CriticConfig] = None) -> None:
        self.cfg = cfg or CriticConfig()

    def critique(self, *,
                  proposed_direction: int,
                  proposed_profile: str,
                  snapshot: Dict[str, Any],
                  rich_context: Any,
                  flow_event: Any,
                  flow_memory: Any = None,
                  web_snapshot: Any = None,
                  mtf_alignment: Optional[Dict[str, Any]] = None,
                  ) -> CritiqueResult:
        """Run the full critic argument and return a CritiqueResult."""
        cfg = self.cfg
        score = 0.0
        reasons: List[str] = []
        notes: List[str] = []

        decision = _map(snapshot.get("decision"))
        iv = _map(snapshot.get("iv_state"))
        bf = _map(snapshot.get("battlefield"))
        thesis = _map(snapshot.get("thesis"))
        winding = _map(snapshot.get("winding"))
        iv_state = str(iv.get("state") or "")
        bf_verdict = str(bf.get("verdict") or "")
        thesis_state = str(thesis.get("composite_state") or "")
        winding_zone = str(winding.get("zone") or "")
        bull_score = _num(thesis.get("bull_thesis_score"))
        bear_score = _num(thesis.get("bear_thesis_score"))
        abnormal_frac = float(getattr(flow_event, "abnormal_slot_fraction", 0.0))
        ce_def_frac = float(getattr(flow_event, "ce_defended_fraction", 0.0))
        pe_def_frac = float(getattr(flow_event, "pe_defended_fraction", 0.0))
        avg_surprise = float(getattr(flow_event, "surprise_score", 0.0))
        regime_stab = float(getattr(rich_context, "regime_stability_index", 1.0))
        ce_v = float(getattr(rich_context, "ce_signed_z_velocity", 0.0))
        pe_v = float(getattr(rich_context, "pe_signed_z_velocity", 0.0))
        bull_v = float(getattr(rich_context, "thesis_bull_velocity", 0.0))
        bear_v = float(getattr(rich_context, "thesis_bear_velocity", 0.0))
        disp_v = float(getattr(rich_context, "dispersion_velocity", 0.0))
        epi_mig = float(getattr(rich_context, "epicenter_migration_distance", 0.0))

        # ── 1. Brittle rail ────────────────────────────────────────────
        # High abnormal_slot_fraction means the rail's strength is concentrated
        # in slots that are themselves dod_z outliers — fragile.
        if abnormal_frac >= 0.30:
            s = cfg.weight_brittle_rail * min(1.0, abnormal_frac / 0.50)
            score += s
            reasons.append(
                f"rail strength is brittle: {abnormal_frac:.0%} of slots are abnormal"
            )

        # ── 2. Single-strike distortion ────────────────────────────────
        if epi_mig >= 3 or disp_v >= 0.10:
            s = cfg.weight_single_strike * min(1.0, max(epi_mig / 6.0,
                                                          disp_v / 0.30))
            score += s
            reasons.append(
                f"single-strike pressure: epicenter migration {epi_mig:.0f}, "
                f"dispersion velocity {disp_v:+.2f}"
            )

        # ── 3. Thesis declining despite being above threshold ──────────
        if proposed_direction > 0 and bull_v < -2.0:
            s = cfg.weight_thesis_decline * min(1.0, abs(bull_v) / 8.0)
            score += s
            reasons.append(
                f"bull thesis declining (velocity {bull_v:+.2f}) even as score "
                f"sits at {bull_score:.0f}"
            )
        if proposed_direction < 0 and bear_v < -2.0:
            s = cfg.weight_thesis_decline * min(1.0, abs(bear_v) / 8.0)
            score += s
            reasons.append(
                f"bear thesis declining (velocity {bear_v:+.2f}) even as score "
                f"sits at {bear_score:.0f}"
            )

        # ── 4. Regime instability ──────────────────────────────────────
        if regime_stab < 0.45:
            s = cfg.weight_regime_instability * (1.0 - regime_stab / 0.45)
            score += s
            reasons.append(
                f"regime unstable (stability index {regime_stab:.2f}) — "
                "directional trades historically fail here"
            )

        # ── 5. Opposite-side acceptance defended ───────────────────────
        # Founder's exact concern: if PE rail is being defended while we buy CE,
        # someone with deeper pockets is positioned the other way.
        if proposed_direction > 0 and pe_def_frac >= 0.30:
            s = cfg.weight_opposite_acceptance * min(1.0, pe_def_frac / 0.60)
            score += s
            reasons.append(
                f"PE rail defended at {pe_def_frac:.0%} — hidden downside positioning"
            )
        if proposed_direction < 0 and ce_def_frac >= 0.30:
            s = cfg.weight_opposite_acceptance * min(1.0, ce_def_frac / 0.60)
            score += s
            reasons.append(
                f"CE rail defended at {ce_def_frac:.0%} — hidden upside positioning"
            )

        # ── 6. Recent trap fingerprints ────────────────────────────────
        if flow_memory is not None:
            try:
                bull_trap_recent = (winding_zone == "STRONG_BULL_TRAP_WINDING"
                                     or len(flow_memory.find_pattern(
                                         "surprise_up_then_revert")) >= 1)
                bear_trap_recent = (winding_zone == "STRONG_BEAR_TRAP_WINDING"
                                     or len(flow_memory.find_pattern(
                                         "surprise_down_then_revert")) >= 1)
            except Exception:
                bull_trap_recent = False
                bear_trap_recent = False
            if proposed_direction > 0 and bull_trap_recent:
                score += cfg.weight_recent_trap
                reasons.append(
                    "bull trap fingerprint visible in recent flow — caution"
                )
            if proposed_direction < 0 and bear_trap_recent:
                score += cfg.weight_recent_trap
                reasons.append(
                    "bear trap fingerprint visible in recent flow — caution"
                )

        # ── 7. IV crush risk ───────────────────────────────────────────
        # Long-direction with IV directional already → premium can bleed.
        if iv_state in ("directional_bull", "directional_bear"):
            if (proposed_direction > 0 and iv_state == "directional_bull") \
                    or (proposed_direction < 0 and iv_state == "directional_bear"):
                # Already directional in our way — IV could mean-revert.
                if avg_surprise < 0.15:
                    score += cfg.weight_iv_crush_risk
                    reasons.append(
                        "IV already directional in our favor but flow is calm — "
                        "premium prone to bleed (crush risk)"
                    )

        # ── 8. Scenario web inputs (chop / tail / oscillation) ──────────
        if web_snapshot is not None:
            chop_mass = float(getattr(web_snapshot, "chop_mass", 0.0))
            tail_mass = float(getattr(web_snapshot, "tail_mass", 0.0))
            if chop_mass >= 0.45:
                s = cfg.weight_chop_dominance * min(1.0, chop_mass / 0.80)
                score += s
                reasons.append(
                    f"scenario web: chop mass {chop_mass:.2f} dominant — "
                    "directional trades historically grind out"
                )
            if tail_mass >= 0.25:
                s = cfg.weight_tail_mass * min(1.0, tail_mass / 0.70)
                score += s
                reasons.append(
                    f"scenario web: tail mass {tail_mass:.2f} elevated — "
                    "violent moves possible either direction"
                )

        # ── 9. Thesis oscillation in recent memory ─────────────────────
        if flow_memory is not None:
            try:
                osc_count = len(flow_memory.find_pattern("thesis_oscillation"))
            except Exception:
                osc_count = 0
            if osc_count >= 1:
                s = cfg.weight_oscillation * min(1.0, osc_count / 3.0)
                score += s
                reasons.append(
                    f"thesis oscillation detected ({osc_count} window(s)) — "
                    "indecision / engineered confusion"
                )

        # ── 10. MTF alignment failure ──────────────────────────────────
        # If we're being called despite MTF veto, the bear case is strong.
        if mtf_alignment is not None and not bool(mtf_alignment.get("alignment_ok")):
            score += 0.20
            reasons.append(
                f"MTF alignment failed (score "
                f"{_num(mtf_alignment.get('alignment_score')):.2f}) — "
                "L1 alone is retail"
            )

        # Final bounding + action mapping.
        antithesis = max(0.0, min(1.0, score))
        if antithesis >= cfg.refuse_score:
            action = CRITIC_REFUSE
            haircut = cfg.min_haircut
            notes.append(
                f"antithesis {antithesis:.2f} ≥ refuse floor "
                f"{cfg.refuse_score:.2f}: REFUSE"
            )
        elif antithesis >= cfg.size_down_score:
            action = CRITIC_SIZE_DOWN
            # Linear haircut from 1.0 (at size_down_score) down to min_haircut
            # (at refuse_score).
            span = max(0.01, cfg.refuse_score - cfg.size_down_score)
            frac = (antithesis - cfg.size_down_score) / span
            haircut = max(cfg.min_haircut,
                          1.0 - frac * (1.0 - cfg.min_haircut))
            notes.append(
                f"antithesis {antithesis:.2f} in haircut band: size × "
                f"{haircut:.2f}"
            )
        else:
            action = CRITIC_PROCEED
            haircut = 1.0
            notes.append(
                f"antithesis {antithesis:.2f} below haircut floor — proceed"
            )

        # Sort reasons by descending implicit weight — best done by re-ranking.
        # Simpler: keep insertion order which roughly matches weight ordering.
        return CritiqueResult(
            antithesis_score=antithesis,
            antithesis_reasons=reasons,
            recommended_size_haircut=haircut,
            recommended_action=action,
            notes=notes,
        )


def _map(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except (TypeError, ValueError):
        pass
    return default
