"""Strategy library — non-directional + structured plays.

Founder's mandate (2026-06-19): *"We don't want to get lost in the ocean
of causality. Sometimes the market is not directional — it's chop, it's
pin, it's vol expansion. We need to be able to play those too."*

This package collects per-strategy modules. Each strategy declares:

  * ``entry_conditions(web_snapshot, mm_posterior, fat_tail_score,
                       crowd_report) -> (bool, reasons)`` — whether the
    current state supports this strategy
  * ``build_legs(spot, strikes, lots) -> StrategyLegs`` — the actual
    contract legs to open
  * ``greek_profile_at_entry(legs) -> dict`` — delta/vega/theta/gamma
  * ``max_loss_rupees(legs, lots) -> float`` — bounded for spread
    structures, +infinity for naked shorts (refused by risk layer)
  * ``max_gain_rupees(legs, lots) -> float`` — bounded for spread
    structures, +infinity for outright longs
  * ``early_warning_signals(legs) -> list[str]`` — context-specific
    invalidation
  * ``adjustment_options(legs, current_state) -> list[str]`` —
    roll/widen/cap actions

A central ``StrategySelector`` (in strategy_selector.py, exported from
this package) takes the executor's current Sprint-2/3 outputs and picks
the highest-fit strategy.
"""
from __future__ import annotations

from .base import (
    StrategyEntryDecision,
    StrategyLeg,
    StrategyLegs,
    StrategyContext,
    StrategyOutcome,
    BaseStrategy,
)
from .single_leg import SingleLegStrategy
from .vertical_spread import (
    BullVerticalStrategy,
    BearVerticalStrategy,
)
from .straddle import LongStraddleStrategy
from .strangle import LongStrangleStrategy
from .iron_condor import IronCondorStrategy
from .butterfly import ButterflyStrategy
from .ratio_spread import BullRatioSpreadStrategy
from .jade_lizard import JadeLizardStrategy
from .selector import StrategySelector, StrategySelectorResult

__all__ = [
    "BaseStrategy",
    "BearVerticalStrategy",
    "BullRatioSpreadStrategy",
    "BullVerticalStrategy",
    "ButterflyStrategy",
    "IronCondorStrategy",
    "JadeLizardStrategy",
    "LongStraddleStrategy",
    "LongStrangleStrategy",
    "SingleLegStrategy",
    "StrategyContext",
    "StrategyEntryDecision",
    "StrategyLeg",
    "StrategyLegs",
    "StrategyOutcome",
    "StrategySelector",
    "StrategySelectorResult",
]
