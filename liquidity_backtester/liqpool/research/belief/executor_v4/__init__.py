"""Premium Belief Engine — Executor v4: the intelligent execution layer.

The next-generation execution layer for the Premium Belief Engine. Replaces
the v3 ``executor.py`` (which was a single-position governor with healed but
still primitive logic) with a layered, probability-driven, fees-aware,
manipulation-aware portfolio manager.

Architecture (5 sprints; this package ships in commitable slices):

  Sprint 1 — Foundation + Survival Math (THIS COMMIT):
    * substrate.py       — exposes module internals + multi-derivative views
    * memory.py          — multi-timeframe memory (1m / 5m / 15m / 60m)
    * flow_memory.py     — rich event-level memory beyond bars
    * economics.py       — fees-FIRST EV gate (the founder's #1 concrete ask)
    * hypothesis.py      — structured trade thesis with validation/invalidation
    * ledger.py          — bar-by-bar audit trail per position
    * manager.py         — minimal multi-position orchestrator using the above

  Sprint 2 — The Probability Web:
    * scenario_web.py, projection.py, critic.py, counterfactual.py,
      aggregator.py (v1)

  Sprint 3 — Market Structure + Fat-Tail Defense:
    * manipulation_patterns.py, market_maker_mind.py,
      fat_tail_amplifier.py, crowd_mirror.py

  Sprint 4 — Strategy Library + Portfolio + Hedging:
    * strategy_library/, risk.py, hedge.py, explainer.py

  Sprint 5 — Polish + Sentinel cockpit panels.

Philosophy (founder-mandated, 2026-06-19):
  * Probability webs, not R/R ratios. The aggregator queries a live web of
    competing scenarios and lets the dominant ones drive sizing + exits.
  * Cut losers fast EVEN when fees are a high fraction of the loss — a
    small loss now is cheaper than a big loss later (fees scale roughly
    with notional, but loss scales with the move).
  * Refuse trades that can't clear (fees + slippage + minimum-edge margin)
    before they fire. Don't take a trade just because the engine said so.
  * Every position is a structured hypothesis with validation/invalidation
    criteria, fully serialized to the ledger. Saturday post-mortems are real.
  * Speed: fast enough to beat retail and slow institutions. We are NOT
    competing with HFT.
"""
from __future__ import annotations

from .aggregator import (
    AggregatorConfig,
    AggregatorDecision,
    DecisionAggregator,
)
from .counterfactual import (
    CounterfactualConfig,
    CounterfactualGenerator,
    CounterfactualPlan,
    KillCriterion,
)
from .critic import (
    AdversarialCritic,
    CriticConfig,
    CritiqueResult,
)
from .economics import (
    ExecutionEconomicsConfig,
    FeeBreakdown,
    NIFTY_LOT_SIZE,
    NIFTY_MAX_LOSS_PER_TRADE,
    fees_for_round_trip,
    minimum_profitable_premium_delta,
    expected_value_after_costs,
    should_accelerate_exit,
)
from .flow_memory import FlowEvent, FlowMemory
from .hypothesis import (
    PositionHypothesis,
    build_hypothesis_from_snapshot,
)
from .ledger import (
    BarRecord,
    LedgerOutcome,
    PositionLedger,
)
from .manager import (
    PortfolioManager,
    PortfolioManagerConfig,
    PortfolioIntent,
)
from .memory import (
    MultiTimeframeMemory,
    TimeframeLevel,
    TimeframeView,
)
from .projection import (
    ForwardProjection,
    ProjectionConfig,
    ProjectionDistribution,
    ProjectionRecord,
)
from .scenario_web import (
    Scenario,
    ScenarioWeb,
    ScenarioWebConfig,
    WebSnapshot,
)
from .substrate import (
    RichContext,
    SubstrateConfig,
    augment_snapshot,
)

__all__ = [
    "AdversarialCritic",
    "AggregatorConfig",
    "AggregatorDecision",
    "BarRecord",
    "CounterfactualConfig",
    "CounterfactualGenerator",
    "CounterfactualPlan",
    "CriticConfig",
    "CritiqueResult",
    "DecisionAggregator",
    "ExecutionEconomicsConfig",
    "FeeBreakdown",
    "FlowEvent",
    "FlowMemory",
    "ForwardProjection",
    "KillCriterion",
    "LedgerOutcome",
    "MultiTimeframeMemory",
    "NIFTY_LOT_SIZE",
    "NIFTY_MAX_LOSS_PER_TRADE",
    "PortfolioIntent",
    "PortfolioManager",
    "PortfolioManagerConfig",
    "PositionHypothesis",
    "PositionLedger",
    "ProjectionConfig",
    "ProjectionDistribution",
    "ProjectionRecord",
    "RichContext",
    "Scenario",
    "ScenarioWeb",
    "ScenarioWebConfig",
    "SubstrateConfig",
    "TimeframeLevel",
    "TimeframeView",
    "WebSnapshot",
    "augment_snapshot",
    "build_hypothesis_from_snapshot",
    "expected_value_after_costs",
    "fees_for_round_trip",
    "minimum_profitable_premium_delta",
    "should_accelerate_exit",
]
