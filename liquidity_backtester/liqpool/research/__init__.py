"""Research primitives for profitability-first alpha discovery.

This package is intentionally separate from the production Arsenal.
It contains two complementary layers:

* the harness/library layer for explicit strategy hypotheses, and
* the miner/atlas layer for discovering and classifying candidate states.

``HypothesisSpec`` remains the harness contract for backwards
compatibility. The rule-miner contract is exported as
``MinerHypothesisSpec`` to avoid mixing the two meanings.
"""

from .harness import (
    BacktestTrade,
    HypothesisHarness,
    HypothesisReport,
    HypothesisSpec,
    simulate_trades,
    trade_stats,
)
from .hypothesis_miner import (
    Condition,
    HypothesisMetrics,
    HypothesisSpec as MinerHypothesisSpec,
    evaluate_hypothesis,
    rank_random_hypotheses,
    sample_random_hypotheses,
)
from .library import HYPOTHESIS_LIBRARY, register_hypothesis
from .manipulation_atlas import (
    ConstituentState,
    ManipulationState,
    classify_index_manipulation,
)
from .state_dataset import (
    StateDatasetConfig,
    build_manipulation_state_frame,
    summarize_state_frame,
)

__all__ = [
    "BacktestTrade",
    "Condition",
    "ConstituentState",
    "HYPOTHESIS_LIBRARY",
    "HypothesisHarness",
    "HypothesisMetrics",
    "HypothesisReport",
    "HypothesisSpec",
    "ManipulationState",
    "MinerHypothesisSpec",
    "StateDatasetConfig",
    "build_manipulation_state_frame",
    "classify_index_manipulation",
    "evaluate_hypothesis",
    "rank_random_hypotheses",
    "register_hypothesis",
    "sample_random_hypotheses",
    "simulate_trades",
    "summarize_state_frame",
    "trade_stats",
]
