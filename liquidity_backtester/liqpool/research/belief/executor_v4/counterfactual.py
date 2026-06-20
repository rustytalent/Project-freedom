"""Counterfactual kill criteria — "what would tell us early that we picked wrong?"

For every proposed trade, the counterfactual generator imagines the
trade is WRONG and constructs the SPECIFIC early-warning signatures
that would tell us so within the next N bars. These become
position-specific kill criteria attached to the hypothesis (not generic
stops).

Founder's exact instruction (paraphrased): *"Don't give me one stop and
one target. Tell me what would tell you, the engine, that you were
wrong — in this specific context."*

The output is a structured ``CounterfactualPlan``:

  * ``kill_criteria`` — ordered list of position-specific signatures
    that, if observed within the kill window, mean THIS specific trade
    is dying. Each carries a severity and a recommended action.
  * ``imagined_failure_path`` — narrative of how the failure typically
    unfolds in 10 bars
  * ``early_warning_signals`` — the FIRST signals (within 1-3 bars) that
    would trigger acceleration
  * ``kill_window_bars`` — how many bars these criteria apply for
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Kill criterion severities.
SEVERITY_HARD = "hard"     # exit immediately if seen
SEVERITY_SOFT = "soft"     # exit after min hold if confirmed
SEVERITY_INFO = "info"     # log only, don't auto-exit


@dataclass
class KillCriterion:
    """One position-specific killer signal."""
    name: str
    description: str
    severity: str             # hard / soft / info
    monitor_window_bars: int  # check for this many bars after open
    recommended_action: str   # EXIT_NOW / ACCELERATE_EXIT / TIGHTEN_STOP / LOG

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "severity": self.severity,
            "monitor_window_bars": self.monitor_window_bars,
            "recommended_action": self.recommended_action,
        }


@dataclass
class CounterfactualPlan:
    """The position-specific failure plan."""
    proposed_direction: int
    proposed_strategy_class: str
    kill_window_bars: int
    kill_criteria: List[KillCriterion]
    imagined_failure_path: str          # narrative
    early_warning_signals: List[str]     # the 1-3 bar warnings
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposed_direction": self.proposed_direction,
            "proposed_strategy_class": self.proposed_strategy_class,
            "kill_window_bars": self.kill_window_bars,
            "kill_criteria": [c.to_dict() for c in self.kill_criteria],
            "imagined_failure_path": self.imagined_failure_path,
            "early_warning_signals": list(self.early_warning_signals),
            "notes": list(self.notes),
        }


@dataclass
class CounterfactualConfig:
    """Knobs for the counterfactual generator."""
    default_kill_window: int = 12         # bars to watch for failure
    scalp_kill_window: int = 6
    intraday_kill_window: int = 20


class CounterfactualGenerator:
    """Generates the position-specific failure plan.

    Stateless: ``generate(...)`` returns a fresh plan each call. The plan
    is attached to the PositionHypothesis at entry and consulted on every
    subsequent bar by the manager's exit-check pipeline.
    """

    def __init__(self, cfg: Optional[CounterfactualConfig] = None) -> None:
        self.cfg = cfg or CounterfactualConfig()

    def generate(self, *,
                  proposed_direction: int,
                  proposed_profile: str,
                  proposed_strategy_class: str,
                  snapshot: Dict[str, Any],
                  rich_context: Any,
                  flow_event: Any,
                  web_snapshot: Any = None,
                  ) -> CounterfactualPlan:
        """Imagine the trade is wrong; enumerate how the failure unfolds."""
        cfg = self.cfg
        if proposed_profile == "SCALP":
            window = cfg.scalp_kill_window
        elif proposed_profile == "INTRADAY":
            window = cfg.intraday_kill_window
        else:
            window = cfg.default_kill_window

        decision = _map(snapshot.get("decision"))
        thesis = _map(snapshot.get("thesis"))
        iv = _map(snapshot.get("iv_state"))
        bf = _map(snapshot.get("battlefield"))
        thesis_state = str(thesis.get("composite_state") or "")
        bf_verdict = str(bf.get("verdict") or "")
        iv_state = str(iv.get("state") or "")
        epi_mig = float(getattr(rich_context, "epicenter_migration_distance", 0.0))
        regime_stab = float(getattr(rich_context, "regime_stability_index", 1.0))
        ce_def_frac = float(getattr(flow_event, "ce_defended_fraction", 0.0))
        pe_def_frac = float(getattr(flow_event, "pe_defended_fraction", 0.0))

        kill: List[KillCriterion] = []
        warnings: List[str] = []
        notes: List[str] = []

        # ── Direction-specific failure modes ──────────────────────────
        if proposed_direction > 0:
            # Long failure mode #1: thesis flips bearish quickly.
            kill.append(KillCriterion(
                name="thesis_flips_bear",
                description=(
                    f"thesis composite turns BEAR_* within {min(5, window)} bars "
                    "of opening — typical fade signal"
                ),
                severity=SEVERITY_HARD,
                monitor_window_bars=min(5, window),
                recommended_action="EXIT_NOW",
            ))
            warnings.append("thesis_state turning BEAR within 3 bars")

            kill.append(KillCriterion(
                name="ce_rail_collapse",
                description=(
                    "CE rail mean signed-z drops by >1.2σ within "
                    f"{min(6, window)} bars"
                ),
                severity=SEVERITY_HARD,
                monitor_window_bars=min(6, window),
                recommended_action="EXIT_NOW",
            ))
            warnings.append("CE rail signed-z dropping fast")

            kill.append(KillCriterion(
                name="pe_defended_rises",
                description=(
                    "PE defended fraction climbs above 0.50 — hidden short setup"
                ),
                severity=SEVERITY_HARD,
                monitor_window_bars=window,
                recommended_action="EXIT_NOW",
            ))
            warnings.append("PE rail acceptance flipping to defended")

            kill.append(KillCriterion(
                name="held_leg_rejected",
                description=(
                    "Held CE leg acceptance flips to 'rejected' — distribution"
                ),
                severity=SEVERITY_HARD,
                monitor_window_bars=window,
                recommended_action="EXIT_NOW",
            ))

        else:  # proposed_direction < 0
            kill.append(KillCriterion(
                name="thesis_flips_bull",
                description=(
                    f"thesis composite turns BULL_* within {min(5, window)} bars "
                    "of opening"
                ),
                severity=SEVERITY_HARD,
                monitor_window_bars=min(5, window),
                recommended_action="EXIT_NOW",
            ))
            warnings.append("thesis_state turning BULL within 3 bars")

            kill.append(KillCriterion(
                name="pe_rail_collapse",
                description=(
                    "PE rail mean signed-z rises by >1.2σ within "
                    f"{min(6, window)} bars"
                ),
                severity=SEVERITY_HARD,
                monitor_window_bars=min(6, window),
                recommended_action="EXIT_NOW",
            ))
            warnings.append("PE rail signed-z rising fast")

            kill.append(KillCriterion(
                name="ce_defended_rises",
                description=(
                    "CE defended fraction climbs above 0.50 — hidden long setup"
                ),
                severity=SEVERITY_HARD,
                monitor_window_bars=window,
                recommended_action="EXIT_NOW",
            ))
            warnings.append("CE rail acceptance flipping to defended")

            kill.append(KillCriterion(
                name="held_leg_rejected",
                description=(
                    "Held PE leg acceptance flips to 'rejected' — accumulation"
                ),
                severity=SEVERITY_HARD,
                monitor_window_bars=window,
                recommended_action="EXIT_NOW",
            ))

        # ── Context-driven failure modes (apply to both directions) ────

        # If we entered when epicenter was already migrating, ANY further
        # migration confirms manipulation and we should bail.
        if epi_mig >= 2:
            kill.append(KillCriterion(
                name="epicenter_keeps_migrating",
                description=(
                    f"epicenter migration grows above {int(epi_mig)+2} strikes "
                    "from current — MM rebalancing in progress"
                ),
                severity=SEVERITY_HARD,
                monitor_window_bars=window,
                recommended_action="EXIT_NOW",
            ))
            notes.append(
                f"entry context already shows epicenter migrating "
                f"({epi_mig:.0f} strikes); strict watch."
            )

        # If regime was already shaky, sub-2-bar destabilization kills it.
        if regime_stab < 0.65:
            kill.append(KillCriterion(
                name="regime_destabilizes_further",
                description=(
                    f"regime_stability_index drops below "
                    f"{max(0.10, regime_stab - 0.20):.2f}"
                ),
                severity=SEVERITY_HARD,
                monitor_window_bars=window,
                recommended_action="EXIT_NOW",
            ))
            notes.append(
                f"entry regime shaky (stability {regime_stab:.2f}); "
                "watch for further breakdown."
            )

        # If IV state was directional and turns dirty, fat-tail risk activated.
        if iv_state in ("directional_bull", "directional_bear"):
            kill.append(KillCriterion(
                name="iv_state_turns_dirty",
                description=(
                    "iv_state turns dirty_data / liquidity_distortion / common_shock"
                ),
                severity=SEVERITY_HARD,
                monitor_window_bars=window,
                recommended_action="EXIT_NOW",
            ))
            warnings.append("iv_state degrading")

        # Tail mass tracker — if web's tail mass spikes after entry,
        # cut even small positions.
        if web_snapshot is not None:
            tail_now = float(getattr(web_snapshot, "tail_mass", 0.0))
            if tail_now < 0.30:
                kill.append(KillCriterion(
                    name="tail_mass_spikes",
                    description=(
                        f"scenario web tail_mass rises above "
                        f"{tail_now + 0.20:.2f}"
                    ),
                    severity=SEVERITY_SOFT,
                    monitor_window_bars=window,
                    recommended_action="ACCELERATE_EXIT",
                ))

            chop_now = float(getattr(web_snapshot, "chop_mass", 0.0))
            if proposed_strategy_class in ("long_ce", "long_pe") and chop_now < 0.40:
                kill.append(KillCriterion(
                    name="chop_overruns_directional",
                    description=(
                        "scenario web chop_mass exceeds 0.55 — directional thesis "
                        "loses to range-bound regime"
                    ),
                    severity=SEVERITY_SOFT,
                    monitor_window_bars=window,
                    recommended_action="ACCELERATE_EXIT",
                ))

        # Universal early-warning signals.
        warnings.extend([
            "decision.action flips to EXIT",
            "engine warmup flips back to cold",
            "battlefield verdict turns single_distortion / vol_expansion",
        ])

        # Narrative.
        direction_word = "long" if proposed_direction > 0 else "short"
        opposite_word = "bear" if proposed_direction > 0 else "bull"
        narrative = (
            f"If this {direction_word} {proposed_strategy_class} is wrong, the typical "
            f"failure path within {window} bars is: thesis composite first "
            f"weakens (bull/bear velocity drops below 0), then the "
            f"{'CE' if proposed_direction > 0 else 'PE'} rail mean signed-z "
            f"reverses by >1σ, then the {'PE' if proposed_direction > 0 else 'CE'} "
            f"rail acceptance flips to defended (someone else accumulating "
            f"the {opposite_word} side), and finally a sharp premium drop "
            f"toward stop. The 'early warning' is the rail signed-z reversal "
            f"in bars 1-3; the 'kill' is the acceptance flip + thesis turn."
        )

        return CounterfactualPlan(
            proposed_direction=proposed_direction,
            proposed_strategy_class=proposed_strategy_class,
            kill_window_bars=window,
            kill_criteria=kill,
            imagined_failure_path=narrative,
            early_warning_signals=warnings,
            notes=notes,
        )


def _map(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}
