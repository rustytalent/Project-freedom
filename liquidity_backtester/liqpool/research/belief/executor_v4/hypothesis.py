"""Structured position hypothesis — "every order has a why."

Every position the executor opens is born from a complete
:class:`PositionHypothesis` that captures *at the moment of birth*:
  * the triggering decision and full upstream context
  * the multi-timeframe alignment that confirmed it
  * the recent contradicting signals it overcame (or chose to ignore)
  * the explicit validation criteria (what would CONFIRM it's working)
  * the explicit invalidation criteria — hard and soft, by severity
  * the explicit exit rules in numbers
  * the economic facts: fees expected, slippage estimate, EV
  * the operator-readable explanation

This dataclass is the founder's "saturday post-mortem" record. It is
serialized in full to the ledger and can answer "why did you take this
trade?" three months later, line by line.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PositionHypothesis:
    """The complete causal record of why a position exists."""
    # Identity
    position_id: str
    opened_at_ts: Any
    opened_at_bar: int

    # The triggering decision and full snapshot
    trigger_action: str                          # e.g. ENTER_LONG / SCALP_PUT
    trigger_decision: Dict[str, Any]
    trigger_snapshot_summary: Dict[str, Any]     # condensed snapshot — key fields only
    rich_context: Dict[str, Any]                  # the substrate's augmented context

    # The thesis statement (human readable)
    thesis_summary: str
    thesis_state: str                             # BULL_ENTRY / SCALP_CALL / etc
    direction: int                                # +1 / -1
    profile: str                                  # SCALP / INTRADAY

    # Multi-timeframe context
    mtf_alignment: Dict[str, Any]
    mtf_aligned: bool
    contradicting_recent: List[str] = field(default_factory=list)

    # The contract
    contract_side: str = ""                       # CE / PE
    contract_label: str = ""
    contract_level: int = 0
    strike_price: float = 0.0
    entry_premium: float = 0.0
    entry_spot: float = 0.0
    entry_dod_z: float = 0.0
    entry_friendliness: float = 1.0
    entry_spread_state: str = "clean"

    # Sizing rationale
    size_fraction: float = 0.0
    size_lots: int = 0
    rupees_at_risk: float = 0.0
    rupees_at_target: float = 0.0

    # Validation criteria (what confirms the trade is working)
    validation_criteria: List[str] = field(default_factory=list)

    # Invalidation criteria (multi-severity)
    hard_invalidation: List[str] = field(default_factory=list)
    soft_invalidation: List[str] = field(default_factory=list)

    # Exit rules in numbers
    stop_premium: float = 0.0
    target_premium: float = 0.0
    trail_after_r: float = 0.0
    max_hold_bars: int = 0
    premium_stop_pct: float = 0.0

    # Economics
    fee_estimate_rupees: float = 0.0
    slippage_estimate_rupees: float = 0.0
    expected_value_rupees: float = 0.0
    expected_edge_multiple: float = 0.0
    minimum_profitable_premium_delta: float = 0.0

    # Adversarial critique (what almost killed it)
    antithesis_score: float = 0.0
    antithesis_notes: List[str] = field(default_factory=list)

    # Operator-readable explanation
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out = self.__dict__.copy()
        for k in ("contradicting_recent", "validation_criteria",
                  "hard_invalidation", "soft_invalidation", "antithesis_notes"):
            out[k] = list(getattr(self, k))
        out["trigger_decision"] = dict(self.trigger_decision)
        out["trigger_snapshot_summary"] = dict(self.trigger_snapshot_summary)
        out["rich_context"] = dict(self.rich_context)
        out["mtf_alignment"] = dict(self.mtf_alignment)
        return out


def _summarize_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Compact summary of the snapshot dict — keeps the key fields needed
    for post-mortem, drops the large slot_readings."""
    decision = snapshot.get("decision") or {}
    iv = snapshot.get("iv_state") or {}
    bf = snapshot.get("battlefield") or {}
    thesis = snapshot.get("thesis") or {}
    return {
        "ts": snapshot.get("ts"),
        "spot": snapshot.get("spot"),
        "bars_seen": snapshot.get("bars_seen"),
        "decision_action": decision.get("action"),
        "decision_confidence": decision.get("confidence"),
        "iv_state": iv.get("state"),
        "iv_confidence": iv.get("confidence"),
        "clean_mark_fraction": iv.get("clean_mark_fraction"),
        "battlefield_verdict": bf.get("verdict"),
        "battlefield_direction": bf.get("direction"),
        "battlefield_confidence": bf.get("confidence"),
        "thesis_state": thesis.get("composite_state"),
        "bull_score": thesis.get("bull_thesis_score"),
        "bear_score": thesis.get("bear_thesis_score"),
        "winding_zone": (snapshot.get("winding") or {}).get("zone"),
    }


def build_hypothesis_from_snapshot(*,
                                     snapshot: Dict[str, Any],
                                     rich_context_dict: Dict[str, Any],
                                     mtf_alignment: Dict[str, Any],
                                     contradicting_recent: List[str],
                                     direction: int,
                                     profile: str,
                                     contract_side: str,
                                     contract_label: str,
                                     contract_level: int,
                                     strike_price: float,
                                     entry_premium: float,
                                     entry_spot: float,
                                     entry_dod_z: float,
                                     entry_friendliness: float,
                                     entry_spread_state: str,
                                     size_fraction: float,
                                     size_lots: int,
                                     rupees_at_risk: float,
                                     rupees_at_target: float,
                                     stop_premium: float,
                                     target_premium: float,
                                     trail_after_r: float,
                                     max_hold_bars: int,
                                     premium_stop_pct: float,
                                     fee_estimate_rupees: float,
                                     slippage_estimate_rupees: float,
                                     expected_value_rupees: float,
                                     expected_edge_multiple: float,
                                     minimum_profitable_premium_delta: float,
                                     antithesis_score: float,
                                     antithesis_notes: List[str],
                                     bar_index: int,
                                     ) -> PositionHypothesis:
    """Compose a complete hypothesis from the full upstream context."""
    decision = snapshot.get("decision") or {}
    iv = snapshot.get("iv_state") or {}
    bf = snapshot.get("battlefield") or {}
    thesis = snapshot.get("thesis") or {}

    direction_word = "long" if direction > 0 else "short"
    thesis_summary = (
        f"{contract_label} {direction_word} ({profile.lower()}): "
        f"engine action={decision.get('action')} "
        f"thesis_state={thesis.get('composite_state')} "
        f"iv_state={iv.get('state')} "
        f"battlefield={bf.get('verdict')}; "
        f"MTF aligned={mtf_alignment.get('alignment_ok')} "
        f"score={mtf_alignment.get('alignment_score')}"
    )

    # Validation criteria — what would confirm this is the right trade.
    validation: List[str] = [
        f"premium rises ≥0.4R within 8 bars",
        f"thesis composite stays HOLD_{('BULL' if direction > 0 else 'BEAR')} "
        f"for ≥5 bars",
        f"battlefield verdict remains "
        f"{'bullish_agreement' if direction > 0 else 'bearish_agreement'} "
        f"or stronger",
        f"held leg acceptance stays 'defended' or 'normal' (not 'rejected')",
        f"net_intent_z stays {'≥+1.0' if direction > 0 else '≤-1.0'}",
    ]

    # Hard invalidation — immediate exit.
    hard: List[str] = [
        f"premium falls to stop_premium ₹{stop_premium:.2f} (−1R)",
        f"STRONG_{('BULL' if direction > 0 else 'BEAR')}_TRAP on held side",
        f"REGIME_BREAK_{('DOWN' if direction > 0 else 'UP')}",
        "held contract spread state turns 'dangerous'",
        "held contract mark source turns dirty (last_valid/ltp/invalid)",
        "engine warmup flips back to cold mid-trade",
        "daily bleed limit reached (portfolio kill)",
    ]

    # Soft invalidation — after min hold, ONE is enough to exit.
    soft: List[str] = [
        f"thesis composite turns "
        f"{'BEAR_ENTRY/HOLD_BEAR' if direction > 0 else 'BULL_ENTRY/HOLD_BULL'} "
        f"or stays NEUTRAL for ≥3 bars",
        f"{('CE' if direction > 0 else 'PE')} rail mean signed-z drops below 0",
        f"held leg acceptance flips to 'rejected'",
        "stream direction turns opposite to held side",
        f"confidence high-water to current giveback ≥0.30",
        f"profit lock giveback: best R drops by ≥0.45R from peak",
        f"target reached + context fading (votes/rail dropped)",
    ]

    # Explanation — human readable.
    explanation = _build_explanation(
        decision=decision, iv=iv, bf=bf, thesis=thesis,
        direction_word=direction_word, profile=profile,
        contract_label=contract_label, strike_price=strike_price,
        entry_premium=entry_premium, size_lots=size_lots,
        stop_premium=stop_premium, target_premium=target_premium,
        rupees_at_risk=rupees_at_risk,
        fee_estimate=fee_estimate_rupees,
        slippage_estimate=slippage_estimate_rupees,
        expected_value=expected_value_rupees,
        expected_edge_multiple=expected_edge_multiple,
        antithesis_score=antithesis_score,
        antithesis_notes=antithesis_notes,
        mtf_alignment=mtf_alignment,
        contradicting_recent=contradicting_recent,
        hard=hard, soft=soft,
    )

    return PositionHypothesis(
        position_id=uuid.uuid4().hex[:12],
        opened_at_ts=snapshot.get("ts"),
        opened_at_bar=bar_index,
        trigger_action=str(decision.get("action") or ""),
        trigger_decision=dict(decision),
        trigger_snapshot_summary=_summarize_snapshot(snapshot),
        rich_context=dict(rich_context_dict),
        thesis_summary=thesis_summary,
        thesis_state=str(thesis.get("composite_state") or ""),
        direction=direction,
        profile=profile,
        mtf_alignment=dict(mtf_alignment),
        mtf_aligned=bool(mtf_alignment.get("alignment_ok")),
        contradicting_recent=list(contradicting_recent),
        contract_side=contract_side,
        contract_label=contract_label,
        contract_level=contract_level,
        strike_price=strike_price,
        entry_premium=entry_premium,
        entry_spot=entry_spot,
        entry_dod_z=entry_dod_z,
        entry_friendliness=entry_friendliness,
        entry_spread_state=entry_spread_state,
        size_fraction=size_fraction,
        size_lots=size_lots,
        rupees_at_risk=rupees_at_risk,
        rupees_at_target=rupees_at_target,
        validation_criteria=validation,
        hard_invalidation=hard,
        soft_invalidation=soft,
        stop_premium=stop_premium,
        target_premium=target_premium,
        trail_after_r=trail_after_r,
        max_hold_bars=max_hold_bars,
        premium_stop_pct=premium_stop_pct,
        fee_estimate_rupees=fee_estimate_rupees,
        slippage_estimate_rupees=slippage_estimate_rupees,
        expected_value_rupees=expected_value_rupees,
        expected_edge_multiple=expected_edge_multiple,
        minimum_profitable_premium_delta=minimum_profitable_premium_delta,
        antithesis_score=antithesis_score,
        antithesis_notes=list(antithesis_notes),
        explanation=explanation,
    )


def _build_explanation(*,
                       decision: Dict[str, Any], iv: Dict[str, Any],
                       bf: Dict[str, Any], thesis: Dict[str, Any],
                       direction_word: str, profile: str,
                       contract_label: str, strike_price: float,
                       entry_premium: float, size_lots: int,
                       stop_premium: float, target_premium: float,
                       rupees_at_risk: float, fee_estimate: float,
                       slippage_estimate: float, expected_value: float,
                       expected_edge_multiple: float,
                       antithesis_score: float, antithesis_notes: List[str],
                       mtf_alignment: Dict[str, Any],
                       contradicting_recent: List[str],
                       hard: List[str], soft: List[str]) -> str:
    """Compose the operator-readable explanation for the ledger."""
    lines: List[str] = []
    lines.append(
        f"OPENING {direction_word.upper()} {contract_label} "
        f"(strike ₹{strike_price:.0f}) for {size_lots} lot(s) at ₹{entry_premium:.2f}."
    )
    lines.append("")
    lines.append("Why:")
    lines.append(f"  - Engine decision: {decision.get('action')} at "
                  f"confidence {decision.get('confidence'):.2f}")
    lines.append(f"  - Thesis composite: {thesis.get('composite_state')}")
    lines.append(f"  - IV state: {iv.get('state')} (confidence "
                  f"{iv.get('confidence'):.2f})")
    lines.append(f"  - Battlefield: {bf.get('verdict')} "
                  f"(confidence {bf.get('confidence'):.2f})")
    lines.append(
        f"  - Multi-timeframe alignment: "
        f"L1={mtf_alignment.get('l1_match')}, L5={mtf_alignment.get('l5_match')}, "
        f"L15={mtf_alignment.get('l15_match')}, L60={mtf_alignment.get('l60_match')}; "
        f"score={mtf_alignment.get('alignment_score'):.2f}"
    )
    lines.append("")
    lines.append("Economics:")
    lines.append(
        f"  - Fee estimate: ₹{fee_estimate:.0f}, slippage estimate: "
        f"₹{slippage_estimate:.0f}"
    )
    lines.append(
        f"  - Expected value after costs: ₹{expected_value:.0f}, "
        f"edge multiple: {expected_edge_multiple:.2f}x"
    )
    lines.append("")
    lines.append("Risk:")
    lines.append(
        f"  - Stop ₹{stop_premium:.2f} | target ₹{target_premium:.2f}"
    )
    lines.append(f"  - Rupees at risk: ₹{rupees_at_risk:.0f}")
    if contradicting_recent:
        lines.append("")
        lines.append("Recent contradicting signals (we proceeded anyway):")
        for c in contradicting_recent[:5]:
            lines.append(f"  - {c}")
    if antithesis_notes:
        lines.append("")
        lines.append(f"Adversarial critique (antithesis_score={antithesis_score:.2f}):")
        for n in antithesis_notes[:5]:
            lines.append(f"  - {n}")
    lines.append("")
    lines.append("Hard invalidation (immediate exit):")
    for h in hard:
        lines.append(f"  - {h}")
    lines.append("")
    lines.append("Soft invalidation (exit after min hold):")
    for s in soft:
        lines.append(f"  - {s}")
    return "\n".join(lines)
