"""Position journey tracer — "what is each order actually thinking?"

For any open or closed position, returns the complete causal chain:

  1. **Entry context** — the full BeliefSnapshot summary that triggered
     the entry, plus the aggregator's decision dict and the critic's
     antithesis reasons
  2. **Hypothesis** — validation criteria, hard/soft invalidation,
     stop/target, expected EV
  3. **Expected pathway** — the scenario web's top scenarios at entry
     time + the counterfactual's imagined failure narrative
  4. **Sub-pathway expectations** — for each scenario, the expected
     intermediate moves ("price will go up, but FIRST it will go down")
  5. **Bar-by-bar journey** — every BarRecord with the per-bar
     evolution: R, validations that fired, invalidations that
     activated, thesis state, IV state
  6. **Counterfactual kill criteria status** — which hard kills are
     active right now, which haven't fired yet
  7. **Outcome (if closed)** — exit reason, severity, realized R and
     rupees, hypothesis-vs-reality score, post-mortem narrative

The tracer is READ-ONLY: it accesses the manager's internal state and
ledger store via documented attributes; it never modifies anything.

Use case: the operator wants to know "I have 5 open positions; for
each one, what is the engine expecting next? Are any of them
overlapping pathways? Could one of them be the others' counterfactual
killer?"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Data shapes ───────────────────────────────────────────────────


@dataclass
class SubPathway:
    """One expected intermediate move on the way to a scenario's resolution.

    Example: scenario 'shakeout_then_bull' carries a SubPathway
    'expected_dip_first' that warns the operator NOT to be alarmed if
    the position goes negative before going positive.
    """
    name: str
    description: str
    expected_in_bars: int
    expected_magnitude_r: float        # signed R magnitude (e.g. -0.4 = dip 0.4R)
    suppresses_kill: bool = False      # if True, don't kill the position even if this fires

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class PositionJourney:
    """The complete causal chain for one position."""
    position_id: str
    status: str                        # "open" / "closed"
    strategy_name: str
    contract_label: str
    direction: int
    opened_at_bar: int
    bars_open: int
    # Entry-time context
    entry_trigger_decision: Dict[str, Any]
    entry_aggregator_decision: Dict[str, Any]
    entry_critic_antithesis_reasons: List[str]
    entry_snapshot_summary: Dict[str, Any]
    entry_web_snapshot: Optional[Dict[str, Any]]
    entry_mm_posterior: Optional[Dict[str, Any]]
    entry_counterfactual_plan: Optional[Dict[str, Any]]
    # The hypothesis itself
    thesis_summary: str
    validation_criteria: List[str]
    hard_invalidation: List[str]
    soft_invalidation: List[str]
    stop_premium: float
    target_premium: float
    # Expected pathway + sub-pathways
    expected_pathway_narrative: str
    sub_pathways: List[SubPathway]
    # Current state
    current_r: float
    best_r: float
    worst_r: float
    last_premium: float
    high_water_confidence: float
    # Bar-by-bar evolution
    bar_records: List[Dict[str, Any]]
    validations_met_ever: List[str]
    invalidations_active_ever: List[str]
    # Kill criteria status
    active_hard_kill_status: List[Dict[str, Any]]
    # Outcome (if closed)
    closed_outcome: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["sub_pathways"] = [s.to_dict() for s in self.sub_pathways]
        return d


# ── Sub-pathway derivation ────────────────────────────────────────


def _derive_sub_pathways(hypothesis_dict: Dict[str, Any],
                           entry_web: Optional[Dict[str, Any]],
                           counterfactual_plan: Optional[Dict[str, Any]],
                           ) -> List[SubPathway]:
    """Inspect the scenario web at entry to identify expected intermediate
    moves.

    The founder's exact example: scenario 'shakeout_then_bull' implies
    the price will go up BUT FIRST will dip. We surface these
    expectations so the operator knows not to panic on the dip.
    """
    out: List[SubPathway] = []
    top_scenarios = (entry_web.get("top_scenarios") or []) if entry_web else []
    for sc in top_scenarios[:5]:
        name = sc.get("name", "")
        prob = float(sc.get("current_probability", 0.0))
        if prob < 0.10:
            continue
        # Shakeout-then-bull / shakeout-then-bear: explicit dip-first patterns.
        if name == "shakeout_then_bull":
            out.append(SubPathway(
                name="expected_dip_before_bull",
                description=(
                    "Scenario 'shakeout_then_bull' is active. The expected "
                    "path is: spot dips first (shakes out weak longs), THEN "
                    "the bull move resumes. Don't panic on early drawdown."
                ),
                expected_in_bars=8,
                expected_magnitude_r=-0.55,
                suppresses_kill=False,
            ))
        if name == "shakeout_then_bear":
            out.append(SubPathway(
                name="expected_pop_before_bear",
                description=(
                    "Scenario 'shakeout_then_bear' active. Expected path: "
                    "brief pop (shakes out weak shorts), THEN bearish move."
                ),
                expected_in_bars=8,
                expected_magnitude_r=-0.55,
                suppresses_kill=False,
            ))
        if name in ("slow_grind_bull", "slow_grind_bear"):
            out.append(SubPathway(
                name=f"{name}_patient_hold",
                description=(
                    f"'{name}' active. Expect 20-40 bar drift with small "
                    "intermediate retraces. Stops can be wider."
                ),
                expected_in_bars=30,
                expected_magnitude_r=-0.30,
                suppresses_kill=False,
            ))
        if name in ("stop_hunt_up_reverse", "stop_hunt_down_reverse"):
            direction = "up" if "up" in name else "down"
            out.append(SubPathway(
                name=f"stop_hunt_already_priced_{direction}",
                description=(
                    f"'{name}' active — the {direction}-spike was a hunt. "
                    f"Expect quick revert in 5-8 bars."
                ),
                expected_in_bars=8,
                expected_magnitude_r=+0.30,
                suppresses_kill=False,
            ))

    # Counterfactual narrative often encodes the expected failure path —
    # we can show that as the inverse "if-wrong" pathway.
    if counterfactual_plan:
        narrative = counterfactual_plan.get("imagined_failure_path", "")
        if narrative:
            out.append(SubPathway(
                name="counterfactual_failure_path",
                description=f"If wrong, failure unfolds as: {narrative[:200]}",
                expected_in_bars=int(counterfactual_plan.get(
                    "kill_window_bars", 12)),
                expected_magnitude_r=-1.0,
                suppresses_kill=True,
            ))
    return out


def _build_expected_pathway_narrative(hypothesis_dict: Dict[str, Any],
                                         entry_web: Optional[Dict[str, Any]],
                                         sub_pathways: List[SubPathway]
                                         ) -> str:
    """Compose the operator-facing expected-pathway narrative."""
    parts: List[str] = []
    direction = hypothesis_dict.get("direction", 0)
    contract = hypothesis_dict.get("contract_label", "?")
    parts.append(
        f"Position {contract} is {'long' if direction > 0 else 'short'}. "
    )
    if entry_web:
        consensus = float(entry_web.get("directional_consensus", 0.0))
        parts.append(
            f"At entry, the probability web's directional consensus was "
            f"{consensus:+.2f}. "
        )
        top = entry_web.get("top_scenarios") or []
        if top:
            primary = top[0]
            parts.append(
                f"Primary scenario: '{primary.get('name')}' "
                f"(prob {primary.get('probability', 0):.2f}, "
                f"family {primary.get('family')}). "
            )
    if sub_pathways:
        parts.append(
            f"Expected intermediate moves: "
            + "; ".join([f"{s.name} ({s.expected_in_bars}b, "
                         f"{s.expected_magnitude_r:+.2f}R)"
                         for s in sub_pathways[:3]])
            + ". "
        )
    return "".join(parts).strip()


def _check_hard_kill_status(hard_kills: List[str],
                              last_bar_record: Optional[Dict[str, Any]],
                              ) -> List[Dict[str, Any]]:
    """For each hard invalidation phrase, check whether the most recent
    bar record shows it firing."""
    out: List[Dict[str, Any]] = []
    if not last_bar_record:
        return [{"criterion": k, "fired": False,
                  "note": "no bar records yet"} for k in hard_kills]
    invals_active = set(last_bar_record.get("invalidations_active") or [])
    for k in hard_kills:
        out.append({
            "criterion": k,
            "fired": k in invals_active,
            "note": "active in last bar" if k in invals_active else "ok",
        })
    return out


# ── Public API ─────────────────────────────────────────────────────


def build_journey(manager, position_id: str) -> Optional[PositionJourney]:
    """Build the complete causal chain for a single position.

    Looks up the position from the manager's open_states (if open) or
    ledger_store.closed_positions() (if closed). Returns None if not
    found.
    """
    # Search open first.
    open_state = manager._open_states.get(position_id)
    if open_state is not None:
        return _build_open_journey(manager, position_id, open_state)
    # Search closed.
    for ledger in manager.ledger_store.closed_positions():
        if ledger.hypothesis.position_id == position_id:
            return _build_closed_journey(manager, position_id, ledger)
    return None


def build_all_journeys(manager) -> List[PositionJourney]:
    """Build journeys for every open + closed position currently tracked."""
    out: List[PositionJourney] = []
    for pid in list(manager._open_states.keys()):
        j = build_journey(manager, pid)
        if j is not None:
            out.append(j)
    for ledger in manager.ledger_store.closed_positions():
        j = build_journey(manager, ledger.hypothesis.position_id)
        if j is not None:
            out.append(j)
    return out


# ── Internals ──────────────────────────────────────────────────────


def _build_open_journey(manager, position_id: str,
                          open_state) -> PositionJourney:
    h = open_state.hypothesis
    ledger = manager.ledger_store.find_open(position_id)
    cf_plan = manager._counterfactual_plans.get(position_id)
    cf_plan_dict = cf_plan.to_dict() if cf_plan is not None else None
    entry_web = h.rich_context.get("web_snapshot") if isinstance(
        h.rich_context, dict) else None
    sub_pathways = _derive_sub_pathways(h.to_dict(), entry_web, cf_plan_dict)
    expected = _build_expected_pathway_narrative(h.to_dict(), entry_web,
                                                    sub_pathways)
    bar_records: List[Dict[str, Any]] = []
    last_record: Optional[Dict[str, Any]] = None
    if ledger:
        for b in ledger.bar_records:
            br = b.to_dict()
            bar_records.append(br)
            last_record = br
    cur_r = float(last_record.get("current_r", 0.0)) if last_record else 0.0
    validations_met_ever = set()
    invals_ever = set()
    for br in bar_records:
        validations_met_ever.update(br.get("validations_met") or [])
        invals_ever.update(br.get("invalidations_active") or [])
    hard_status = _check_hard_kill_status(list(h.hard_invalidation),
                                             last_record)
    return PositionJourney(
        position_id=position_id,
        status="open",
        strategy_name=h.trigger_action,
        contract_label=h.contract_label,
        direction=h.direction,
        opened_at_bar=h.opened_at_bar,
        bars_open=max(0, (last_record.get("bar_index") if last_record else 0)
                       - h.opened_at_bar),
        entry_trigger_decision=dict(h.trigger_decision),
        entry_aggregator_decision={},   # not stored on hypothesis directly
        entry_critic_antithesis_reasons=list(h.antithesis_notes),
        entry_snapshot_summary=dict(h.trigger_snapshot_summary),
        entry_web_snapshot=entry_web,
        entry_mm_posterior=None,        # Sprint+ could persist this
        entry_counterfactual_plan=cf_plan_dict,
        thesis_summary=h.thesis_summary,
        validation_criteria=list(h.validation_criteria),
        hard_invalidation=list(h.hard_invalidation),
        soft_invalidation=list(h.soft_invalidation),
        stop_premium=h.stop_premium,
        target_premium=h.target_premium,
        expected_pathway_narrative=expected,
        sub_pathways=sub_pathways,
        current_r=cur_r,
        best_r=open_state.best_r,
        worst_r=open_state.worst_r,
        last_premium=open_state.last_premium,
        high_water_confidence=open_state.high_water_confidence,
        bar_records=bar_records,
        validations_met_ever=sorted(validations_met_ever),
        invalidations_active_ever=sorted(invals_ever),
        active_hard_kill_status=hard_status,
        closed_outcome=None,
    )


def _build_closed_journey(manager, position_id: str,
                            ledger) -> PositionJourney:
    h = ledger.hypothesis
    bar_records = [b.to_dict() for b in ledger.bar_records]
    last_record = bar_records[-1] if bar_records else None
    cur_r = (float(last_record.get("current_r", 0.0))
              if last_record else float(ledger.outcome.realized_r
                                          if ledger.outcome else 0.0))
    entry_web = h.rich_context.get("web_snapshot") if isinstance(
        h.rich_context, dict) else None
    sub_pathways = _derive_sub_pathways(h.to_dict(), entry_web, None)
    expected = _build_expected_pathway_narrative(h.to_dict(), entry_web,
                                                    sub_pathways)
    validations_met_ever = set()
    invals_ever = set()
    for br in bar_records:
        validations_met_ever.update(br.get("validations_met") or [])
        invals_ever.update(br.get("invalidations_active") or [])
    hard_status = _check_hard_kill_status(list(h.hard_invalidation),
                                             last_record)
    bars_open = max(0, ((ledger.outcome.exit_bar if ledger.outcome else 0)
                          - h.opened_at_bar))
    return PositionJourney(
        position_id=position_id,
        status="closed",
        strategy_name=h.trigger_action,
        contract_label=h.contract_label,
        direction=h.direction,
        opened_at_bar=h.opened_at_bar,
        bars_open=bars_open,
        entry_trigger_decision=dict(h.trigger_decision),
        entry_aggregator_decision={},
        entry_critic_antithesis_reasons=list(h.antithesis_notes),
        entry_snapshot_summary=dict(h.trigger_snapshot_summary),
        entry_web_snapshot=entry_web,
        entry_mm_posterior=None,
        entry_counterfactual_plan=None,
        thesis_summary=h.thesis_summary,
        validation_criteria=list(h.validation_criteria),
        hard_invalidation=list(h.hard_invalidation),
        soft_invalidation=list(h.soft_invalidation),
        stop_premium=h.stop_premium,
        target_premium=h.target_premium,
        expected_pathway_narrative=expected,
        sub_pathways=sub_pathways,
        current_r=cur_r,
        best_r=max((br.get("best_r_so_far", 0.0)
                       for br in bar_records), default=0.0),
        worst_r=min((br.get("worst_r_so_far", 0.0)
                        for br in bar_records), default=0.0),
        last_premium=(float(last_record.get("held_premium", 0.0))
                       if last_record else 0.0),
        high_water_confidence=max((br.get("high_water_confidence", 0.0)
                                       for br in bar_records), default=0.0),
        bar_records=bar_records,
        validations_met_ever=sorted(validations_met_ever),
        invalidations_active_ever=sorted(invals_ever),
        active_hard_kill_status=hard_status,
        closed_outcome=ledger.outcome.to_dict() if ledger.outcome else None,
    )


# ── Rendering ──────────────────────────────────────────────────────


def render_journey(journey: PositionJourney, *,
                     verbose: bool = False) -> str:
    """Render a journey as a multi-line operator-facing string."""
    lines: List[str] = []
    lines.append(f"━━ JOURNEY: {journey.position_id[:12]} "
                  f"({journey.status}) ━━")
    lines.append(f"  Strategy:  {journey.strategy_name}")
    lines.append(f"  Contract:  {journey.contract_label} "
                  f"({'long' if journey.direction > 0 else 'short'})")
    lines.append(f"  Bars open: {journey.bars_open}")
    lines.append(f"  R now:     {journey.current_r:+.2f}  "
                  f"(best {journey.best_r:+.2f}, worst {journey.worst_r:+.2f})")
    lines.append("")
    lines.append("  THESIS")
    lines.append(f"  > {journey.thesis_summary[:200]}")
    lines.append("")
    lines.append("  EXPECTED PATHWAY")
    lines.append(f"  > {journey.expected_pathway_narrative[:300]}")
    if journey.sub_pathways:
        lines.append("")
        lines.append("  SUB-PATHWAYS (expected intermediate moves):")
        for s in journey.sub_pathways:
            mark = "✓" if s.suppresses_kill else "→"
            lines.append(f"    {mark} {s.name} "
                          f"({s.expected_in_bars}b, {s.expected_magnitude_r:+.2f}R)")
            lines.append(f"        {s.description[:160]}")
    lines.append("")
    lines.append("  KILL CRITERIA STATUS")
    for h in journey.active_hard_kill_status[:6]:
        sym = "⚠" if h["fired"] else "·"
        lines.append(f"    {sym} {h['criterion'][:70]}")
    if journey.closed_outcome:
        lines.append("")
        lines.append("  CLOSED OUTCOME")
        o = journey.closed_outcome
        lines.append(f"  Exit reason: {(o.get('exit_reason') or '')[:120]}")
        lines.append(f"  Realized R:  {float(o.get('realized_r', 0.0)):+.2f}")
        lines.append(f"  Realized ₹:  {float(o.get('realized_rupees', 0.0)):+,.0f}")
        lines.append(f"  Hypothesis-vs-reality score: "
                      f"{float(o.get('hypothesis_vs_reality_score', 0.0)):.2f}")
    if verbose:
        lines.append("")
        lines.append("  VALIDATIONS MET EVER:")
        for v in journey.validations_met_ever[:5]:
            lines.append(f"    ✓ {v[:80]}")
        lines.append("  INVALIDATIONS ACTIVE EVER:")
        for v in journey.invalidations_active_ever[:5]:
            lines.append(f"    ✗ {v[:80]}")
    return "\n".join(lines)
