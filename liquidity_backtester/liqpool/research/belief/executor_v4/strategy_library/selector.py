"""Strategy selector — picks the best-fit structure for current state.

Given the executor's Sprint-2/3 outputs, the selector queries every
registered strategy's ``can_enter(ctx)``, ranks the survivors by
``fitness_score``, and returns the winner. The manager then asks that
strategy to ``build(ctx, lots)`` for the actual leg list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .base import BaseStrategy, StrategyContext, StrategyEntryDecision
from .butterfly import ButterflyStrategy
from .iron_condor import IronCondorStrategy
from .jade_lizard import JadeLizardStrategy
from .ratio_spread import BullRatioSpreadStrategy
from .single_leg import SingleLegStrategy
from .straddle import LongStraddleStrategy
from .strangle import LongStrangleStrategy
from .vertical_spread import BearVerticalStrategy, BullVerticalStrategy


@dataclass
class StrategySelectorResult:
    """The selector's per-call output."""
    chosen: Optional[BaseStrategy]
    chosen_name: str
    chosen_fitness: float
    per_strategy_decisions: List[Dict[str, Any]]
    fallback_used: bool = False
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chosen": self.chosen_name,
            "chosen_fitness": round(self.chosen_fitness, 3),
            "per_strategy_decisions": list(self.per_strategy_decisions),
            "fallback_used": self.fallback_used,
            "notes": list(self.notes),
        }


# Strategy fitness bonuses based on context.
_DEFAULT_STRATEGIES: List[BaseStrategy] = [
    SingleLegStrategy(),
    BullVerticalStrategy(),
    BearVerticalStrategy(),
    LongStraddleStrategy(),
    LongStrangleStrategy(),
    IronCondorStrategy(),
    ButterflyStrategy(),
    BullRatioSpreadStrategy(),
    JadeLizardStrategy(),
]


class StrategySelector:
    """Selects the best-fit strategy from a fixed registry."""

    def __init__(self, strategies: Optional[Sequence[BaseStrategy]] = None) -> None:
        self.strategies: List[BaseStrategy] = list(
            strategies if strategies is not None else _DEFAULT_STRATEGIES
        )

    def select(self, ctx: StrategyContext) -> StrategySelectorResult:
        decisions: List[StrategyEntryDecision] = []
        for strat in self.strategies:
            try:
                d = strat.can_enter(ctx)
            except Exception as exc:
                d = StrategyEntryDecision(
                    strategy_name=strat.name, family=strat.family,
                    can_enter=False,
                    reasons=[f"exception in can_enter: {exc}"],
                    fitness_score=0.0,
                )
            decisions.append(d)

        eligible = [(s, d) for s, d in zip(self.strategies, decisions)
                    if d.can_enter]
        notes: List[str] = []
        fallback = False

        if not eligible:
            # Fallback: single_leg if directional thesis, else no entry.
            if ctx.thesis_direction != 0:
                fallback_strat = self.strategies[0]  # SingleLeg
                if not self._lookup_ok(ctx, ctx.thesis_direction):
                    return StrategySelectorResult(
                        chosen=None, chosen_name="",
                        chosen_fitness=0.0,
                        per_strategy_decisions=[d.to_dict() for d in decisions],
                        fallback_used=False,
                        notes=["no eligible strategy and no fallback quote"],
                    )
                fallback = True
                notes.append(
                    "no eligible strategy — falling back to single_leg with low fitness"
                )
                return StrategySelectorResult(
                    chosen=fallback_strat,
                    chosen_name=fallback_strat.name,
                    chosen_fitness=0.30,
                    per_strategy_decisions=[d.to_dict() for d in decisions],
                    fallback_used=True,
                    notes=notes,
                )
            return StrategySelectorResult(
                chosen=None, chosen_name="",
                chosen_fitness=0.0,
                per_strategy_decisions=[d.to_dict() for d in decisions],
                fallback_used=False,
                notes=["no eligible strategy; no directional thesis for fallback"],
            )

        # Rank by fitness — highest wins.
        eligible.sort(key=lambda sd: sd[1].fitness_score, reverse=True)
        winner_strat, winner_decision = eligible[0]
        notes.append(
            f"chose {winner_strat.name} with fitness {winner_decision.fitness_score:.2f}"
        )
        return StrategySelectorResult(
            chosen=winner_strat,
            chosen_name=winner_strat.name,
            chosen_fitness=winner_decision.fitness_score,
            per_strategy_decisions=[d.to_dict() for d in decisions],
            fallback_used=fallback,
            notes=notes,
        )

    @staticmethod
    def _lookup_ok(ctx: StrategyContext, direction: int) -> bool:
        side = "CE" if direction > 0 else "PE"
        return (ctx.strike_lookup is not None
                and (side, 0) in ctx.strike_lookup)
