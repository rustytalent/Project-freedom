"""Research primitives for profitability-first alpha discovery.

This package contains three complementary layers:

* harness/library — explicit strategy hypotheses and their backtest engine
* miner/atlas — discover and classify candidate market states
* allocator/kill-switch/synthetic-validation — disciplined deployment:
  decide which surviving hypotheses get capital, when to pull a
  decayed one, and whether any survivor invented its edge from noise

``HypothesisSpec`` remains the harness contract for backwards
compatibility. The rule-miner contract is exported as
``MinerHypothesisSpec`` to avoid mixing the two meanings.
"""

from .allocator import (
    AllocationReport,
    HypothesisPosterior,
    allocate_from_metrics,
    allocate_from_reports,
    thompson_allocate,
)
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
from .kill_switch import (
    HypothesisHistory,
    KillVerdict,
    ResurrectionVerdict,
    check_resurrection,
    filter_alive,
    verdict_for_hypothesis,
    verdicts_for_basket,
)
from .library import HYPOTHESIS_LIBRARY, register_hypothesis
from .manipulation_atlas import (
    ConstituentState,
    ManipulationState,
    classify_index_manipulation,
)
from .regime_miner import (
    DEFAULT_MIN_BARS_PER_FOLD,
    DEFAULT_MIN_ROWS_PER_REGIME,
    FoldResult,
    RegimePerformance,
    RegimeSegment,
    WalkForwardMetrics,
    evaluate_across_regimes,
    mine_per_regime,
    rank_walk_forward_hypotheses,
    regime_specialization_score,
    segment_by_regime,
    walk_forward_metrics,
)
from .state_dataset import (
    StateDatasetConfig,
    build_manipulation_state_frame,
    summarize_state_frame,
)
from .synthetic_validation import (
    GENERATORS as SYNTHETIC_GENERATORS,
    RegimeResult,
    SyntheticValidationReport,
    generate_mean_reverting,
    generate_random_walk,
    generate_trending,
    validate_hypothesis_against_noise,
)

__all__ = [
    "AllocationReport",
    "BacktestTrade",
    "Condition",
    "ConstituentState",
    "DEFAULT_FORWARD_BARS",
    "DEFAULT_MIN_BARS_PER_FOLD",
    "DEFAULT_MIN_ROWS_PER_REGIME",
    "FoldResult",
    "HYPOTHESIS_LIBRARY",
    "HypothesisHarness",
    "HypothesisHistory",
    "HypothesisMetrics",
    "HypothesisPosterior",
    "HypothesisReport",
    "HypothesisSpec",
    "IndexStateBuildConfig",
    "KillVerdict",
    "ManipulationState",
    "MinerHypothesisSpec",
    "RegimePerformance",
    "RegimeResult",
    "RegimeSegment",
    "ResurrectionVerdict",
    "StateDatasetConfig",
    "SYNTHETIC_GENERATORS",
    "SyntheticValidationReport",
    "WalkForwardMetrics",
    "allocate_from_metrics",
    "allocate_from_reports",
    "build_manipulation_state_frame",
    "build_index_manipulation_dataset",
    "check_resurrection",
    "classify_index_manipulation",
    "evaluate_across_regimes",
    "evaluate_hypothesis",
    "filter_alive",
    "generate_mean_reverting",
    "generate_random_walk",
    "generate_trending",
    "mine_per_regime",
    "parse_weights_text",
    "rank_random_hypotheses",
    "rank_walk_forward_hypotheses",
    "read_weights_csv",
    "regime_specialization_score",
    "register_hypothesis",
    "sample_random_hypotheses",
    "segment_by_regime",
    "simulate_trades",
    "summarize_state_frame",
    "thompson_allocate",
    "trade_stats",
    "validate_hypothesis_against_noise",
    "verdict_for_hypothesis",
    "verdicts_for_basket",
    "walk_forward_metrics",
]
