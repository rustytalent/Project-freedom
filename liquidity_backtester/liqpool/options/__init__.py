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

__all__ = [
    "OptionsFeaturizerParams",
    "build_options_feature_frame",
    "OPTIONS_FEATURE_COLUMNS",
]
