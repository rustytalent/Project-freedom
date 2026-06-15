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
from .index_state_builder import (
    DEFAULT_FORWARD_BARS,
    IndexStateBuildConfig,
    build_index_manipulation_dataset,
    parse_weights_text,
    read_weights_csv,
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
    "DEFAULT_FORWARD_BARS",
    "HYPOTHESIS_LIBRARY",
    "HypothesisHarness",
    "HypothesisMetrics",
    "HypothesisReport",
    "HypothesisSpec",
    "IndexStateBuildConfig",
    "ManipulationState",
    "MinerHypothesisSpec",
    "StateDatasetConfig",
    "build_manipulation_state_frame",
    "build_index_manipulation_dataset",
    "classify_index_manipulation",
    "evaluate_hypothesis",
    "parse_weights_text",
    "rank_random_hypotheses",
    "read_weights_csv",
    "register_hypothesis",
    "sample_random_hypotheses",
    "simulate_trades",
    "summarize_state_frame",
    "trade_stats",
]
