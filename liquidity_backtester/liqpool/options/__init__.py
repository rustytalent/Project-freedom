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
]
