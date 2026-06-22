"""Probability Web — live multi-scenario tracker.

The founder's mandate (2026-06-19): *"We should not think in ratios.
We should think in probabilities. We should be able to handle many
scenarios at once — shake-out then continuation, step climb, slow
grind, false breakdown then rip — and let the dominant ones drive the
trade."*

This module replaces the v3 executor's single-confidence view with a
live web of competing scenarios. Each ``Scenario`` is born from current
evidence, carries explicit confirming/killer signatures, decays over
time, and is reweighted every tick from fresh observations.

Operationally the ``ScenarioWeb`` holds 30-100 live scenarios across
four families:

  * **directional** — outright bull / bear continuations or reversals
  * **chop**         — sideways with mean-reversion, range-bound regimes
  * **manipulation** — stop hunts, traps, distribution, pin near expiry
  * **fat_tail**     — IV crush, gap, vol expansion, common shock

The aggregator (sprint-2's ``aggregator.py``) reads the web's:

  * ``top_k(k)`` — strongest live scenarios
  * ``directional_consensus()`` — net bullish/bearish probability mass
  * ``tail_mass()`` — total probability over fat_tail scenarios
  * ``dominant_strategy_class()`` — which strategy family the web prefers
  * ``currently_dominant_pathway()`` — the single most likely 10-bar path

…and lets those drive sizing, entry, and exit decisions.

Design choice: TRANSPARENT scoring (no black-box ML). Every scenario's
probability is a closed-form function of its base prior × evidence
multipliers × decay × killer-vote shrinkage. The full reason chain is
serialized to the ledger so the founder can replay any decision.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Scenario family tags.
FAMILY_DIRECTIONAL = "directional"
FAMILY_CHOP = "chop"
FAMILY_MANIPULATION = "manipulation"
FAMILY_FAT_TAIL = "fat_tail"

# Tail class (subdivision of fat_tail).
TAIL_NONE = "none"
TAIL_IV_CRUSH = "iv_crush"
TAIL_GAP = "gap"
TAIL_VOL_EXPANSION = "vol_expansion"
TAIL_COMMON_SHOCK = "common_shock"

# Strategy class hints (consumed by sprint-4 strategy library).
STRAT_LONG_CE = "long_ce"
STRAT_LONG_PE = "long_pe"
STRAT_BULL_VERTICAL = "bull_vertical"
STRAT_BEAR_VERTICAL = "bear_vertical"
STRAT_STRADDLE = "straddle"
STRAT_IRON_CONDOR = "iron_condor"
STRAT_BUTTERFLY = "butterfly"
STRAT_WAIT = "wait"


@dataclass
class ScenarioWebConfig:
    """Knobs for the scenario web."""
    max_scenarios: int = 60
    min_probability_floor: float = 0.02   # below this, retire the scenario
    decay_per_tick: float = 0.985         # multiplicative decay per tick
    boost_per_confirm: float = 1.20       # multiplier per fresh confirming observation
    shrink_per_contradict: float = 0.78   # multiplier per fresh contradiction
    tail_mass_alarm: float = 0.35         # tail_mass() ≥ this → portfolio alarm
    chop_mass_alarm: float = 0.55         # chop family dominant → throttle directional
    consensus_strong: float = 0.55        # |consensus| ≥ this → strong directional bias
    spawn_cooldown_ticks: int = 4         # don't respawn same signature within N ticks
    confirms_cap: int = 12                # max stored confirms per scenario
    contradicts_cap: int = 12             # max stored contradictions per scenario


@dataclass
class Scenario:
    """One live competing scenario in the web."""
    scenario_id: str
    name: str
    family: str                            # directional / chop / manipulation / fat_tail
    trigger_signature: str                 # short label of birth condition
    implied_direction: int                 # +1 / -1 / 0
    implied_horizon_bars: int              # how many bars this pathway is expected to last
    implied_max_drawdown_during_path: float  # in premium R units
    implied_strategy_class: str            # STRAT_* hint for sizing
    tail_class: str = TAIL_NONE
    base_prior: float = 0.10
    current_probability: float = 0.10
    decay_rate: float = 0.985
    confirming_observations: List[str] = field(default_factory=list)
    contradicting_observations: List[str] = field(default_factory=list)
    killer_signatures: List[str] = field(default_factory=list)
    spawned_at_bar: int = 0
    last_updated_bar: int = 0
    last_confirmed_bar: int = 0
    n_confirmed_this_tick: int = 0
    n_contradicted_this_tick: int = 0
    retired: bool = False
    retire_reason: str = ""
    explanation: str = ""
    # Cockpit "pulsing web" visualization (founder ask 2026-06-22):
    # last N probability samples so the UI can render a glow trail
    # showing direction + rate of change.
    recent_probabilities: List[float] = field(default_factory=list)

    def add_confirm(self, evidence: str, bar: int, cap: int) -> None:
        self.confirming_observations.append(evidence)
        if len(self.confirming_observations) > cap:
            self.confirming_observations = self.confirming_observations[-cap:]
        self.n_confirmed_this_tick += 1
        self.last_confirmed_bar = bar

    def add_contradict(self, evidence: str, cap: int) -> None:
        self.contradicting_observations.append(evidence)
        if len(self.contradicting_observations) > cap:
            self.contradicting_observations = self.contradicting_observations[-cap:]
        self.n_contradicted_this_tick += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "family": self.family,
            "trigger_signature": self.trigger_signature,
            "implied_direction": self.implied_direction,
            "implied_horizon_bars": self.implied_horizon_bars,
            "implied_max_drawdown_during_path": round(self.implied_max_drawdown_during_path, 3),
            "implied_strategy_class": self.implied_strategy_class,
            "tail_class": self.tail_class,
            "base_prior": round(self.base_prior, 3),
            "current_probability": round(self.current_probability, 4),
            "decay_rate": round(self.decay_rate, 4),
            "recent_probabilities": [round(p, 4)
                                       for p in self.recent_probabilities[-12:]],
            "confirming_observations": list(self.confirming_observations),
            "contradicting_observations": list(self.contradicting_observations),
            "killer_signatures": list(self.killer_signatures),
            "spawned_at_bar": self.spawned_at_bar,
            "last_updated_bar": self.last_updated_bar,
            "last_confirmed_bar": self.last_confirmed_bar,
            "retired": self.retired,
            "retire_reason": self.retire_reason,
            "explanation": self.explanation,
        }


@dataclass
class WebSnapshot:
    """A per-tick condensed snapshot of the web for downstream consumers."""
    bar_index: int
    n_active: int
    top_scenarios: List[Dict[str, Any]]
    directional_consensus: float           # +1 bullish, -1 bearish, weighted by prob
    tail_mass: float
    chop_mass: float
    manipulation_mass: float
    dominant_strategy_class: str
    currently_dominant_pathway: Optional[Dict[str, Any]]
    family_mass: Dict[str, float]
    # HOT FIX 2026-06-22 LIVE — horizon-weighted consensus (the noise
    # filter the founder discovered live). Default value preserves the
    # legacy API for any constructor that doesn't supply it.
    directional_consensus_horizon_weighted: float = 0.0
    # Founder ask 2026-06-22: ALL active scenarios for the pulsing-web
    # visualization (not just the top 5 list).
    all_scenarios: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bar_index": self.bar_index,
            "n_active": self.n_active,
            "top_scenarios": list(self.top_scenarios),
            "directional_consensus": round(self.directional_consensus, 4),
            "directional_consensus_horizon_weighted": round(
                self.directional_consensus_horizon_weighted, 4),
            "tail_mass": round(self.tail_mass, 4),
            "chop_mass": round(self.chop_mass, 4),
            "manipulation_mass": round(self.manipulation_mass, 4),
            "dominant_strategy_class": self.dominant_strategy_class,
            "currently_dominant_pathway": (dict(self.currently_dominant_pathway)
                                            if self.currently_dominant_pathway else None),
            "family_mass": dict(self.family_mass),
            "all_scenarios": list(self.all_scenarios),
            "notes": list(self.notes),
        }


class ScenarioWeb:
    """The live multi-scenario tracker.

    Call ``observe(snapshot, rich_context, flow_event, mtf_views)`` once per
    tick. Internally:

      1. **DECAY** — every active scenario probability × decay_rate
      2. **SPAWN** — generate new scenarios from current evidence
      3. **UPDATE** — for each active scenario, walk its confirming + killer
         rules against the latest evidence; boost/shrink probability and
         attach reason strings
      4. **RETIRE** — kill scenarios whose killer fired or probability fell
         below the floor
      5. **NORMALIZE** — re-normalize family-wise so total never exceeds 1.0

    Public query API:
      * top_k(k=5)
      * directional_consensus()
      * tail_mass()
      * chop_mass()
      * manipulation_mass()
      * dominant_strategy_class()
      * currently_dominant_pathway()
      * snapshot(bar) — condensed dict for ledger
    """

    def __init__(self, cfg: Optional[ScenarioWebConfig] = None) -> None:
        self.cfg = cfg or ScenarioWebConfig()
        self.scenarios: Dict[str, Scenario] = {}
        self._spawn_history: Dict[str, int] = {}  # signature → last spawn bar
        self._next_id: int = 0
        self._bar_index: int = 0
        # Track recent thesis "side" to detect chop / oscillation.
        self._thesis_side_history: List[str] = []

    def reset(self) -> None:
        self.scenarios.clear()
        self._spawn_history.clear()
        self._next_id = 0
        self._bar_index = 0
        self._thesis_side_history = []

    # ── core per-tick driver ────────────────────────────────────────────

    def observe(self,
                snapshot: Dict[str, Any],
                rich_context: Any,
                flow_event: Any,
                mtf_views: Optional[Dict[str, Any]] = None,
                ) -> WebSnapshot:
        """Drive one tick of the scenario web. Returns a condensed snapshot."""
        cfg = self.cfg
        self._bar_index = int(_num(snapshot.get("bars_seen"), self._bar_index + 1))

        # 1. DECAY all active scenarios.
        for sc in self.scenarios.values():
            sc.n_confirmed_this_tick = 0
            sc.n_contradicted_this_tick = 0
            sc.current_probability *= sc.decay_rate

        # Track thesis side history for chop detection.
        thesis_state_now = str(_map(snapshot.get("thesis")).get(
            "composite_state") or "")
        side = ("bull" if "BULL" in thesis_state_now
                else "bear" if "BEAR" in thesis_state_now else "flat")
        self._thesis_side_history.append(side)
        if len(self._thesis_side_history) > 20:
            self._thesis_side_history = self._thesis_side_history[-20:]

        # 2. SPAWN new scenarios from current evidence.
        self._spawn_from_evidence(snapshot, rich_context, flow_event, mtf_views)

        # 3. UPDATE existing scenarios against latest evidence.
        self._update_from_evidence(snapshot, rich_context, flow_event, mtf_views)

        # 4. RETIRE killed / decayed-out scenarios.
        self._retire_dead()

        # 5. NORMALIZE family masses so total never exceeds 1.0.
        self._normalize()

        return self.snapshot()

    # ── spawn evidence-driven scenarios ─────────────────────────────────

    def _spawn_from_evidence(self,
                              snapshot: Dict[str, Any],
                              rich_context: Any,
                              flow_event: Any,
                              mtf_views: Optional[Dict[str, Any]],
                              ) -> None:
        """Look at the current evidence and birth any scenarios whose
        trigger signature fires *and* hasn't fired within the cooldown."""
        cfg = self.cfg
        bar = self._bar_index
        decision = _map(snapshot.get("decision"))
        thesis = _map(snapshot.get("thesis"))
        iv = _map(snapshot.get("iv_state"))
        bf = _map(snapshot.get("battlefield"))
        winding = _map(snapshot.get("winding"))
        thesis_state = str(thesis.get("composite_state") or "")
        bf_verdict = str(bf.get("verdict") or "")
        iv_state = str(iv.get("state") or "")
        winding_zone = str(winding.get("zone") or "")
        bull_score = _num(thesis.get("bull_thesis_score"))
        bear_score = _num(thesis.get("bear_thesis_score"))
        regime_stab = float(getattr(rich_context, "regime_stability_index", 1.0))
        net_intent = float(getattr(flow_event, "net_intent_z", 0.0))
        net_v = float(getattr(rich_context, "net_intent_velocity", 0.0))
        ce_disp = float(getattr(flow_event, "ce_dispersion", 0.0))
        epi_mig = float(getattr(rich_context, "epicenter_migration_distance", 0.0))

        # ── Directional family ──────────────────────────────────────────
        if "BULL" in thesis_state and bull_score >= 50:
            self._maybe_spawn(
                name="bull_continuation",
                family=FAMILY_DIRECTIONAL,
                trigger_signature=f"bull_thesis+netintent+{int(bull_score)}",
                implied_direction=+1,
                implied_horizon_bars=20,
                implied_max_drawdown_during_path=0.55,
                implied_strategy_class=STRAT_LONG_CE,
                base_prior=0.18 + min(0.15, bull_score / 600.0),
                killers=[
                    "thesis_state contains BEAR",
                    "battlefield verdict is bearish_agreement",
                    "regime_stability < 0.35",
                    "iv_state is dirty_data",
                ],
                explanation=(
                    f"Bull continuation pathway: thesis {thesis_state} with "
                    f"bull_score={bull_score:.0f}; expects 20 bars of upward drift."
                ),
            )

        if "BEAR" in thesis_state and bear_score >= 50:
            self._maybe_spawn(
                name="bear_continuation",
                family=FAMILY_DIRECTIONAL,
                trigger_signature=f"bear_thesis+{int(bear_score)}",
                implied_direction=-1,
                implied_horizon_bars=20,
                implied_max_drawdown_during_path=0.55,
                implied_strategy_class=STRAT_LONG_PE,
                base_prior=0.18 + min(0.15, bear_score / 600.0),
                killers=[
                    "thesis_state contains BULL",
                    "battlefield verdict is bullish_agreement",
                    "regime_stability < 0.35",
                    "iv_state is dirty_data",
                ],
                explanation=(
                    f"Bear continuation pathway: thesis {thesis_state} with "
                    f"bear_score={bear_score:.0f}."
                ),
            )

        # Shake-out then continuation — opposite winding present but underlying
        # thesis still leans bullish.
        if "BULL_TRAP_WINDING" in winding_zone and bull_score >= 30:
            self._maybe_spawn(
                name="shakeout_then_bull",
                family=FAMILY_DIRECTIONAL,
                trigger_signature="bull_trap_then_resume",
                implied_direction=+1,
                implied_horizon_bars=18,
                implied_max_drawdown_during_path=0.85,
                implied_strategy_class=STRAT_LONG_CE,
                base_prior=0.12,
                killers=[
                    "winding_zone STRONG_BEAR",
                    "battlefield verdict is bearish_agreement for >3 bars",
                    "bull_score drops below 15",
                ],
                explanation=(
                    "Shake-out then continuation (bull): bull-trap winding "
                    "while thesis still carries bullish weight — historically "
                    "a fake-out before resumption."
                ),
            )

        if "BEAR_TRAP_WINDING" in winding_zone and bear_score >= 30:
            self._maybe_spawn(
                name="shakeout_then_bear",
                family=FAMILY_DIRECTIONAL,
                trigger_signature="bear_trap_then_resume",
                implied_direction=-1,
                implied_horizon_bars=18,
                implied_max_drawdown_during_path=0.85,
                implied_strategy_class=STRAT_LONG_PE,
                base_prior=0.12,
                killers=[
                    "winding_zone STRONG_BULL",
                    "battlefield verdict is bullish_agreement for >3 bars",
                    "bear_score drops below 15",
                ],
                explanation=(
                    "Shake-out then bear: bear-trap winding while bear thesis "
                    "still has weight."
                ),
            )

        # Slow grind — low surprise, modest direction, stable regime.
        avg_surprise = float(getattr(flow_event, "surprise_score", 0.0))
        if (regime_stab >= 0.65 and abs(net_intent) >= 0.5
                and avg_surprise < 0.3):
            grind_dir = +1 if net_intent > 0 else -1
            grind_name = "slow_grind_bull" if grind_dir > 0 else "slow_grind_bear"
            self._maybe_spawn(
                name=grind_name,
                family=FAMILY_DIRECTIONAL,
                trigger_signature=f"slow_grind_{grind_dir}",
                implied_direction=grind_dir,
                implied_horizon_bars=40,
                implied_max_drawdown_during_path=0.35,
                implied_strategy_class=(STRAT_LONG_CE if grind_dir > 0 else STRAT_LONG_PE),
                base_prior=0.10,
                killers=[
                    "regime_stability < 0.40",
                    "net_intent_z flips sign",
                    "surprise_score > 0.65",
                ],
                explanation=(
                    f"Slow grind {'bull' if grind_dir > 0 else 'bear'}: stable "
                    f"regime + steady net_intent {net_intent:+.2f}σ. Wide stops, "
                    "long horizon."
                ),
            )

        # ── Chop family ─────────────────────────────────────────────────
        thesis_osc_recent = False
        if mtf_views:
            for v in mtf_views.values():
                # tolerate either TimeframeView objects or dicts
                stab = (v.get("regime_stability")
                        if isinstance(v, dict)
                        else getattr(v, "regime_stability", 1.0))
                if stab is not None and float(stab) < 0.40:
                    thesis_osc_recent = True
                    break
        # Detect chop by counting thesis flips in recent history.
        recent_sides = [s for s in self._thesis_side_history[-12:] if s != "flat"]
        recent_flips = sum(1 for a, b in zip(recent_sides, recent_sides[1:])
                            if a != b)
        chop_via_flips = recent_flips >= 3

        if (regime_stab < 0.45 or thesis_osc_recent
                or thesis_state == "NEUTRAL" or chop_via_flips):
            self._maybe_spawn(
                name="chop_range_bound",
                family=FAMILY_CHOP,
                trigger_signature="regime_unstable_or_neutral",
                implied_direction=0,
                implied_horizon_bars=30,
                implied_max_drawdown_during_path=0.50,
                implied_strategy_class=STRAT_IRON_CONDOR,
                base_prior=0.20,
                killers=[
                    "regime_stability > 0.75 for >5 bars",
                    "bull_score>80",
                    "bear_score>80",
                ],
                explanation=(
                    f"Chop / range-bound: regime_stability={regime_stab:.2f}, "
                    "thesis indecisive."
                ),
            )

        # ── Manipulation family ─────────────────────────────────────────
        if epi_mig >= 3 or ce_disp >= 0.55:
            self._maybe_spawn(
                name="single_strike_distortion",
                family=FAMILY_MANIPULATION,
                trigger_signature=f"epicenter_or_disp_high_{int(epi_mig)}",
                implied_direction=0,
                implied_horizon_bars=10,
                implied_max_drawdown_during_path=0.80,
                implied_strategy_class=STRAT_WAIT,
                base_prior=0.15,
                killers=[],   # decays naturally; no single-tick killer reliable here
                explanation=(
                    f"Single-strike distortion suspected: epicenter migration "
                    f"{epi_mig:.0f} strikes, CE dispersion {ce_disp:.2f}. "
                    "Probably MM rebalancing — wait it out."
                ),
            )

        # Surprise-then-revert at the flow level is a stop-hunt fingerprint.
        if float(getattr(flow_event, "surprise_score", 0.0)) >= 0.65:
            spot_pct = float(getattr(flow_event, "spot_pct_change", 0.0))
            if abs(spot_pct) >= 5.0:
                hunt_dir = +1 if spot_pct > 0 else -1
                # The pathway implies REVERSAL: a stop-hunt up usually unwinds DOWN.
                self._maybe_spawn(
                    name=("stop_hunt_up_reverse"
                          if hunt_dir > 0 else "stop_hunt_down_reverse"),
                    family=FAMILY_MANIPULATION,
                    trigger_signature=f"flow_surprise_{hunt_dir}_{int(abs(spot_pct))}bps",
                    implied_direction=-hunt_dir,
                    implied_horizon_bars=12,
                    implied_max_drawdown_during_path=0.75,
                    implied_strategy_class=(STRAT_LONG_PE if hunt_dir > 0
                                            else STRAT_LONG_CE),
                    base_prior=0.14,
                    killers=[
                        "spot moves further in surprise direction for >3 bars",
                        "battlefield verdict aligns with surprise direction",
                    ],
                    explanation=(
                        f"Stop hunt fingerprint: {spot_pct:+.1f}bps surprise "
                        f"({hunt_dir:+d}). Path: revert opposite direction."
                    ),
                )

        # ── Fat-tail family ─────────────────────────────────────────────
        if iv_state in ("dirty_data", "liquidity_distortion", "common_shock"):
            tail_class = (TAIL_COMMON_SHOCK if iv_state == "common_shock"
                          else TAIL_VOL_EXPANSION)
            self._maybe_spawn(
                name="vol_expansion_or_shock",
                family=FAMILY_FAT_TAIL,
                trigger_signature=f"iv_{iv_state}",
                implied_direction=0,
                implied_horizon_bars=8,
                implied_max_drawdown_during_path=1.50,
                implied_strategy_class=STRAT_STRADDLE,
                tail_class=tail_class,
                base_prior=0.18,
                killers=[],    # decays naturally; iv_state already at this value
                explanation=(
                    f"Fat-tail: iv_state={iv_state}. Expect violent moves "
                    "either direction; standard directional sizing unsafe."
                ),
            )
        # IV crush risk when an aggressive directional bull/bear has held
        # for too long without follow-through — premium bleed potential.
        if iv_state in ("directional_bull", "directional_bear") and avg_surprise < 0.10:
            self._maybe_spawn(
                name="iv_crush_risk",
                family=FAMILY_FAT_TAIL,
                trigger_signature=f"iv_crush_{iv_state}",
                implied_direction=0,
                implied_horizon_bars=15,
                implied_max_drawdown_during_path=0.60,
                implied_strategy_class=STRAT_WAIT,
                tail_class=TAIL_IV_CRUSH,
                base_prior=0.05,
                killers=["surprise_score > 0.5 (volatility re-engages)"],
                explanation=(
                    "IV crush risk: directional IV state but flow is calm — "
                    "premium tends to bleed."
                ),
            )

    def _maybe_spawn(self, *, name: str, family: str, trigger_signature: str,
                      implied_direction: int, implied_horizon_bars: int,
                      implied_max_drawdown_during_path: float,
                      implied_strategy_class: str,
                      base_prior: float, killers: List[str],
                      tail_class: str = TAIL_NONE,
                      explanation: str = "") -> None:
        cfg = self.cfg
        if len(self.scenarios) >= cfg.max_scenarios:
            return
        sig_key = f"{name}|{trigger_signature}"
        last_spawn = self._spawn_history.get(sig_key, -999_999)
        if self._bar_index - last_spawn < cfg.spawn_cooldown_ticks:
            # Already alive; just refresh it.
            for sc in self.scenarios.values():
                if sc.name == name and sc.trigger_signature == trigger_signature:
                    sc.last_updated_bar = self._bar_index
                    return
            return
        scenario_id = f"{name}_{self._next_id:04d}"
        self._next_id += 1
        decay = self.cfg.decay_per_tick
        if family == FAMILY_MANIPULATION:
            decay = decay - 0.005   # manipulation pathways die faster
        if family == FAMILY_FAT_TAIL:
            decay = decay - 0.010   # tail risk fades unless freshly confirmed
        prior = max(0.02, min(0.40, base_prior))
        sc = Scenario(
            scenario_id=scenario_id,
            name=name,
            family=family,
            trigger_signature=trigger_signature,
            implied_direction=implied_direction,
            implied_horizon_bars=implied_horizon_bars,
            implied_max_drawdown_during_path=implied_max_drawdown_during_path,
            implied_strategy_class=implied_strategy_class,
            tail_class=tail_class,
            base_prior=prior,
            current_probability=prior,
            decay_rate=decay,
            killer_signatures=list(killers),
            spawned_at_bar=self._bar_index,
            last_updated_bar=self._bar_index,
            last_confirmed_bar=self._bar_index,
            explanation=explanation,
        )
        self.scenarios[scenario_id] = sc
        self._spawn_history[sig_key] = self._bar_index

    # ── update existing scenarios ──────────────────────────────────────

    def _update_from_evidence(self,
                               snapshot: Dict[str, Any],
                               rich_context: Any,
                               flow_event: Any,
                               mtf_views: Optional[Dict[str, Any]],
                               ) -> None:
        cfg = self.cfg
        thesis = _map(snapshot.get("thesis"))
        bf = _map(snapshot.get("battlefield"))
        iv = _map(snapshot.get("iv_state"))
        winding = _map(snapshot.get("winding"))
        thesis_state = str(thesis.get("composite_state") or "")
        bf_verdict = str(bf.get("verdict") or "")
        iv_state = str(iv.get("state") or "")
        winding_zone = str(winding.get("zone") or "")
        bull_score = _num(thesis.get("bull_thesis_score"))
        bear_score = _num(thesis.get("bear_thesis_score"))
        regime_stab = float(getattr(rich_context, "regime_stability_index", 1.0))
        net_intent = float(getattr(flow_event, "net_intent_z", 0.0))
        avg_surprise = float(getattr(flow_event, "surprise_score", 0.0))
        ce_disp = float(getattr(flow_event, "ce_dispersion", 0.0))
        epi_mig = float(getattr(rich_context, "epicenter_migration_distance", 0.0))
        clean_mark = float(getattr(flow_event, "clean_mark_fraction", 1.0))

        bar = self._bar_index

        for sc in self.scenarios.values():
            sc.last_updated_bar = bar
            confirmed: List[str] = []
            contradicted: List[str] = []

            # Universal confirmations / contradictions per family.
            if sc.family == FAMILY_DIRECTIONAL:
                if sc.implied_direction > 0:
                    if "BULL" in thesis_state and "BEAR" not in thesis_state:
                        confirmed.append(f"thesis state {thesis_state}")
                    if bf_verdict == "bullish_agreement":
                        confirmed.append("battlefield bullish_agreement")
                    if net_intent > 0.5:
                        confirmed.append(f"net_intent_z {net_intent:+.2f}σ supports bull")
                    if "BEAR" in thesis_state and "BULL" not in thesis_state:
                        contradicted.append(f"thesis flipped to {thesis_state}")
                    if bf_verdict == "bearish_agreement":
                        contradicted.append("battlefield bearish_agreement")
                    if net_intent < -0.5:
                        contradicted.append(f"net_intent_z {net_intent:+.2f}σ against bull")
                elif sc.implied_direction < 0:
                    if "BEAR" in thesis_state and "BULL" not in thesis_state:
                        confirmed.append(f"thesis state {thesis_state}")
                    if bf_verdict == "bearish_agreement":
                        confirmed.append("battlefield bearish_agreement")
                    if net_intent < -0.5:
                        confirmed.append(f"net_intent_z {net_intent:+.2f}σ supports bear")
                    if "BULL" in thesis_state and "BEAR" not in thesis_state:
                        contradicted.append(f"thesis flipped to {thesis_state}")
                    if bf_verdict == "bullish_agreement":
                        contradicted.append("battlefield bullish_agreement")
                    if net_intent > 0.5:
                        contradicted.append(f"net_intent_z {net_intent:+.2f}σ against bear")
                if regime_stab < 0.35:
                    contradicted.append(f"regime_stability {regime_stab:.2f} low")

            elif sc.family == FAMILY_CHOP:
                if regime_stab < 0.50 or thesis_state == "NEUTRAL":
                    confirmed.append(f"chop env regime={regime_stab:.2f}")
                if max(bull_score, bear_score) > 70:
                    contradicted.append(
                        f"strong directional thesis (bull={bull_score:.0f}, "
                        f"bear={bear_score:.0f})"
                    )
                if regime_stab > 0.75:
                    contradicted.append(f"regime stabilized at {regime_stab:.2f}")

            elif sc.family == FAMILY_MANIPULATION:
                if epi_mig >= 2 or ce_disp >= 0.50:
                    confirmed.append(
                        f"epicenter migration {epi_mig:.0f} | CE disp {ce_disp:.2f}"
                    )
                if avg_surprise >= 0.5:
                    confirmed.append(f"flow surprise {avg_surprise:.2f}")
                if regime_stab > 0.75 and avg_surprise < 0.2:
                    contradicted.append("regime calm + low surprise")

            elif sc.family == FAMILY_FAT_TAIL:
                if iv_state in ("dirty_data", "liquidity_distortion",
                                 "common_shock"):
                    confirmed.append(f"iv_state {iv_state}")
                if clean_mark < 0.85:
                    confirmed.append(f"clean_mark dropped to {clean_mark:.2f}")
                if avg_surprise >= 0.5:
                    confirmed.append(f"surprise {avg_surprise:.2f}")
                if iv_state in ("quiet", "directional_bull", "directional_bear") \
                        and clean_mark > 0.95 and avg_surprise < 0.1:
                    contradicted.append("iv calm + clean marks + low surprise")

            for ev in confirmed:
                sc.add_confirm(ev, bar, cfg.confirms_cap)
            for ev in contradicted:
                sc.add_contradict(ev, cfg.contradicts_cap)

            # Killer signature check — short-circuit retire.
            for k in sc.killer_signatures:
                if self._killer_triggered(k, snapshot, rich_context, flow_event,
                                            mtf_views):
                    sc.retired = True
                    sc.retire_reason = f"killer triggered: {k}"
                    sc.current_probability = 0.0
                    break

            if sc.retired:
                continue

            # Update probability multiplicatively from this tick's evidence.
            mult = (cfg.boost_per_confirm ** sc.n_confirmed_this_tick) \
                * (cfg.shrink_per_contradict ** sc.n_contradicted_this_tick)
            sc.current_probability = max(0.0,
                                          min(0.95, sc.current_probability * mult))
            # Probability-history ring for the cockpit's pulsing-web view.
            sc.recent_probabilities.append(sc.current_probability)
            if len(sc.recent_probabilities) > 12:
                sc.recent_probabilities = sc.recent_probabilities[-12:]

    # ── retire & normalize ──────────────────────────────────────────────

    def _retire_dead(self) -> None:
        cfg = self.cfg
        dead = []
        for sid, sc in self.scenarios.items():
            if sc.retired:
                dead.append(sid)
                continue
            if sc.current_probability < cfg.min_probability_floor:
                sc.retired = True
                sc.retire_reason = sc.retire_reason or "probability decayed below floor"
                dead.append(sid)
        for sid in dead:
            del self.scenarios[sid]

    def _normalize(self) -> None:
        """Cap total probability mass at 1.0. Scenarios are NOT mutually
        exclusive; they are alternative hypotheses. We allow some overlap
        but cap total so dominance comparisons stay meaningful."""
        if not self.scenarios:
            return
        total = sum(sc.current_probability for sc in self.scenarios.values())
        if total <= 1.20:
            return
        scale = 1.20 / total
        for sc in self.scenarios.values():
            sc.current_probability *= scale

    def _killer_triggered(self, killer: str, snapshot: Dict[str, Any],
                           rich_context: Any, flow_event: Any,
                           mtf_views: Optional[Dict[str, Any]]) -> bool:
        """Best-effort killer evaluation. Killers are short English phrases
        — this evaluator just substring-matches against the available
        evidence. Returns True if the killer's described event matches."""
        thesis = _map(snapshot.get("thesis"))
        bf = _map(snapshot.get("battlefield"))
        iv = _map(snapshot.get("iv_state"))
        winding = _map(snapshot.get("winding"))
        thesis_state = str(thesis.get("composite_state") or "")
        bf_verdict = str(bf.get("verdict") or "")
        iv_state = str(iv.get("state") or "")
        winding_zone = str(winding.get("zone") or "")
        bull = _num(thesis.get("bull_thesis_score"))
        bear = _num(thesis.get("bear_thesis_score"))
        regime_stab = float(getattr(rich_context, "regime_stability_index", 1.0))
        net_intent = float(getattr(flow_event, "net_intent_z", 0.0))
        avg_surprise = float(getattr(flow_event, "surprise_score", 0.0))
        clean_mark = float(getattr(flow_event, "clean_mark_fraction", 1.0))
        ce_disp = float(getattr(flow_event, "ce_dispersion", 0.0))
        epi_mig = float(getattr(rich_context, "epicenter_migration_distance", 0.0))

        k = killer.lower()
        if "thesis_state contains bear" in k:
            return "BEAR" in thesis_state
        if "thesis_state contains bull" in k:
            return "BULL" in thesis_state
        if "battlefield verdict is bearish_agreement" in k:
            return bf_verdict == "bearish_agreement"
        if "battlefield verdict is bullish_agreement" in k:
            return bf_verdict == "bullish_agreement"
        if "regime_stability < 0.35" in k:
            return regime_stab < 0.35
        if "regime_stability < 0.40" in k:
            return regime_stab < 0.40
        if "regime_stability > 0.70" in k:
            return regime_stab > 0.70
        if "regime_stability > 0.75" in k:
            return regime_stab > 0.75
        if "iv_state is dirty_data" in k:
            return iv_state == "dirty_data"
        if "iv_state returns to" in k:
            return iv_state in ("directional_bull", "directional_bear", "quiet")
        if "clean_mark_fraction returns to >0.95" in k:
            return clean_mark > 0.95
        if "bull_score drops below 15" in k:
            return bull < 15
        if "bear_score drops below 15" in k:
            return bear < 15
        if "epicenter migration drops to <1" in k:
            return epi_mig < 1
        if "ce_dispersion < 0.30" in k:
            return ce_disp < 0.30
        if "net_intent_z flips sign" in k:
            # Only triggered if we have a clear opposite-sign value (>0.5)
            return False  # path-tracking would need history; skip for now
        if "surprise_score > 0.65" in k or "surprise_score > 0.5" in k:
            return avg_surprise > 0.5
        if "thesis_state becomes strongly bull_*" in k:
            return "BULL" in thesis_state and bull >= 70
        if "thesis_state becomes strongly bear_*" in k:
            return "BEAR" in thesis_state and bear >= 70
        if "bull_score>70" in k:
            return bull > 70
        if "bear_score>70" in k:
            return bear > 70
        if "bull_score>80" in k:
            return bull > 80
        if "bear_score>80" in k:
            return bear > 80
        if "spot moves further in surprise direction" in k:
            return False  # would need spot trajectory; handled by contradictions
        if "spot moves >10bps in the bull/bear direction" in k:
            return False
        if "winding_zone strong_bull" in k:
            return "STRONG_BULL" in winding_zone
        if "winding_zone strong_bear" in k:
            return "STRONG_BEAR" in winding_zone
        return False

    # ── query API ──────────────────────────────────────────────────────

    def active(self) -> List[Scenario]:
        return [sc for sc in self.scenarios.values() if not sc.retired]

    def top_k(self, k: int = 5) -> List[Scenario]:
        return sorted(self.active(),
                      key=lambda s: s.current_probability,
                      reverse=True)[:k]

    def family_mass(self) -> Dict[str, float]:
        out: Dict[str, float] = {
            FAMILY_DIRECTIONAL: 0.0, FAMILY_CHOP: 0.0,
            FAMILY_MANIPULATION: 0.0, FAMILY_FAT_TAIL: 0.0,
        }
        for sc in self.active():
            out[sc.family] = out.get(sc.family, 0.0) + sc.current_probability
        return out

    def directional_consensus(self) -> float:
        """Signed sum of directional scenario probabilities, in [-1, +1].

        Positive = bullish consensus, negative = bearish, ~0 = no consensus.
        """
        score = 0.0
        for sc in self.active():
            if sc.family != FAMILY_DIRECTIONAL or sc.implied_direction == 0:
                continue
            score += sc.implied_direction * sc.current_probability
        return max(-1.0, min(1.0, score))

    def horizon_weighted_consensus(self, *,
                                       min_probability: float = 0.15,
                                       horizon_weight_exp: float = 0.5,
                                       ) -> float:
        """HOT FIX 2026-06-22 LIVE — temporal context for consensus.

        Founder caught it live: short-term scenarios (~1-3 bars) at low
        probability were CONTAMINATING the long-term signal. His
        observation: probs > 0.40 are real 3-10 min reads; probs < 0.10
        are short-term noise.

        This consensus:
          * drops scenarios below ``min_probability`` (default 0.15)
            — kills the short-term noise floor entirely
          * weights survivors by ``horizon_bars ** exp`` so a 30-bar
            scenario gets ~3x the weight of a 3-bar one
          * normalizes by total weight so the result is a true [-1,+1]
        """
        score = 0.0
        total_weight = 0.0
        for sc in self.active():
            if sc.family != FAMILY_DIRECTIONAL or sc.implied_direction == 0:
                continue
            if sc.current_probability < min_probability:
                continue
            h = max(1, int(sc.implied_horizon_bars))
            h_weight = h ** horizon_weight_exp
            weight = sc.current_probability * h_weight
            score += sc.implied_direction * weight
            total_weight += weight
        if total_weight < 1e-6:
            return 0.0
        return max(-1.0, min(1.0, score / total_weight))

    def tail_mass(self) -> float:
        return sum(sc.current_probability for sc in self.active()
                   if sc.family == FAMILY_FAT_TAIL)

    def chop_mass(self) -> float:
        return sum(sc.current_probability for sc in self.active()
                   if sc.family == FAMILY_CHOP)

    def manipulation_mass(self) -> float:
        return sum(sc.current_probability for sc in self.active()
                   if sc.family == FAMILY_MANIPULATION)

    def dominant_strategy_class(self) -> str:
        """Strategy class hint = probability-weighted majority of active
        scenarios' implied_strategy_class."""
        votes: Counter = Counter()
        for sc in self.active():
            votes[sc.implied_strategy_class] += sc.current_probability
        if not votes:
            return STRAT_WAIT
        return votes.most_common(1)[0][0]

    def currently_dominant_pathway(self) -> Optional[Scenario]:
        tops = self.top_k(1)
        return tops[0] if tops else None

    def snapshot(self) -> WebSnapshot:
        active = self.active()
        top = self.top_k(5)
        fam = self.family_mass()
        notes: List[str] = []
        cfg = self.cfg
        tail = self.tail_mass()
        chop = self.chop_mass()
        if tail >= cfg.tail_mass_alarm:
            notes.append(f"tail mass {tail:.2f} ≥ alarm {cfg.tail_mass_alarm:.2f}")
        if chop >= cfg.chop_mass_alarm:
            notes.append(f"chop mass {chop:.2f} ≥ alarm {cfg.chop_mass_alarm:.2f}")
        dominant = self.currently_dominant_pathway()
        return WebSnapshot(
            bar_index=self._bar_index,
            n_active=len(active),
            top_scenarios=[sc.to_dict() for sc in top],
            directional_consensus=self.directional_consensus(),
            directional_consensus_horizon_weighted=self.horizon_weighted_consensus(),
            tail_mass=tail,
            chop_mass=chop,
            manipulation_mass=self.manipulation_mass(),
            dominant_strategy_class=self.dominant_strategy_class(),
            currently_dominant_pathway=dominant.to_dict() if dominant else None,
            family_mass=fam,
            all_scenarios=[sc.to_dict() for sc in active],
            notes=notes,
        )


# ─────────────────────────────────────────────────────────────────
# Local helpers
# ─────────────────────────────────────────────────────────────────


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
