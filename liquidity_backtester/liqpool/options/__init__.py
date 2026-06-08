"""Options vertical — per-strike featurizer, conviction cascade, and
executor surfaces. See `docs/options_strategy_methodology.md` and
`docs/options_executor_layered_conviction.md` for the design.

This subpackage is gated to the v1 narrow scope per the methodology:
NIFTY50 + BANKNIFTY weeklies + 2-week, ATM ± 3 strikes, buying-side
exposed to subscribers (selling-side modeled, displayed as research
context).
"""
from .featurizer import (
    OptionsFeaturizerParams,
    build_options_feature_frame,
    OPTIONS_FEATURE_COLUMNS,
)
from .labels import (
    OptionsLabelParams,
    add_options_labels,
    LABEL_OUTPUT_COLUMNS,
)
from .model_config import (
    NUMERIC_FEATURE_COLUMNS,
    CATEGORICAL_FEATURE_COLUMNS,
    MONEYNESS_BUCKET_LEVELS,
    INTERACTION_GROUPS_TEMPLATE,
    MONOTONE_BUY,
    MONOTONE_SELL,
    LIGHTGBM_PARAMS,
    build_constraints,
    derive_tenor,
    expand_categorical_features,
)
from .expected_return_model import (
    BucketKey,
    CalibrationRow,
    OptionsHeadMetrics,
    OptionsExpectedReturnModel,
    OptionsExpectedReturnModelSuite,
    SuiteSummary,
)
from .ccv import (
    CCV,
    apply_causal_adjustments,
    ccv_from_row,
    compute_lcs_scalar,
)
from .executor import (
    PreTradeDecision,
    KillConditions,
    InTradeState,
    InTradeDecision,
    pre_trade_decision,
    in_trade_decision,
    agreeing_layer_count,
    causal_warnings,
)
from .audit import (
    OPTIONS_EXECUTOR_PREDICTION_TYPE,
    OPTIONS_SKIP_PREDICTION_TYPE,
    ExecutorAuditReport,
    SkipCounterfactual,
    TableARow,
    TableB,
    TableBLcsBucketRow,
    audit_to_yesterday_audit_payload,
    compute_executor_audit,
    options_executor_meta,
    options_skip_meta,
)
from .artifact_export import (
    executor_audit_csv,
    options_strikes_csv,
    push_options_executor_audit,
    push_options_strikes,
)
from .macro_scrape import (
    MACRO_TICKERS,
    MacroAdapter,
    MacroOvernightSnapshot,
    StubMacroAdapter,
    YahooMacroAdapter,
    empty_snapshot,
)
from .layer_scores import (
    compute_layer_scores_from_inputs,
    macro_score,
    manipulation_score,
    micro_score,
    options_score,
    pool_score,
    regime_score,
)

__all__ = [
    # Featurizer
    "OptionsFeaturizerParams",
    "build_options_feature_frame",
    "OPTIONS_FEATURE_COLUMNS",
    # Labels
    "OptionsLabelParams",
    "add_options_labels",
    "LABEL_OUTPUT_COLUMNS",
    # Model config
    "NUMERIC_FEATURE_COLUMNS",
    "CATEGORICAL_FEATURE_COLUMNS",
    "MONEYNESS_BUCKET_LEVELS",
    "INTERACTION_GROUPS_TEMPLATE",
    "MONOTONE_BUY",
    "MONOTONE_SELL",
    "LIGHTGBM_PARAMS",
    "build_constraints",
    "derive_tenor",
    "expand_categorical_features",
    # Model
    "BucketKey",
    "CalibrationRow",
    "OptionsHeadMetrics",
    "OptionsExpectedReturnModel",
    "OptionsExpectedReturnModelSuite",
    "SuiteSummary",
    # CCV
    "CCV",
    "apply_causal_adjustments",
    "ccv_from_row",
    "compute_lcs_scalar",
    # Executor
    "PreTradeDecision",
    "KillConditions",
    "InTradeState",
    "InTradeDecision",
    "pre_trade_decision",
    "in_trade_decision",
    "agreeing_layer_count",
    "causal_warnings",
    # Audit
    "OPTIONS_EXECUTOR_PREDICTION_TYPE",
    "OPTIONS_SKIP_PREDICTION_TYPE",
    "ExecutorAuditReport",
    "SkipCounterfactual",
    "TableARow",
    "TableB",
    "TableBLcsBucketRow",
    "audit_to_yesterday_audit_payload",
    "compute_executor_audit",
    "options_executor_meta",
    "options_skip_meta",
    # Artifact export
    "executor_audit_csv",
    "options_strikes_csv",
    "push_options_executor_audit",
    "push_options_strikes",
    # Macro scrape
    "MACRO_TICKERS",
    "MacroAdapter",
    "MacroOvernightSnapshot",
    "StubMacroAdapter",
    "YahooMacroAdapter",
    "empty_snapshot",
    # Layer scores
    "compute_layer_scores_from_inputs",
    "macro_score",
    "manipulation_score",
    "micro_score",
    "options_score",
    "pool_score",
    "regime_score",
]
