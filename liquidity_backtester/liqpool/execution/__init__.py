"""Execution-realism models — Stream K.

Three learned models that replace V2's hand-tuned constants with
data-driven estimates. Each is a thin LightGBM wrapper with a
common fit/predict surface, plus a no-op fallback path so V3 stays
deterministic when the models aren't trained yet (or for unit tests
that don't want LightGBM on the hot path).

  * slippage_model — predict realised slippage_bps from state
                     features (regime, distance, ToD, size, side).
  * fill_model — predict P(limit fills before deadline) from
                 (distance_from_best, bars_remaining, vol regime).
  * survival_model — predict E[bars to target] and E[bars to stop]
                     from entry context. Used for sizing and
                     position aging.

All three share a tiny base class (``LGBMRegressorWrapper``) so
``train_execution_models.py`` can iterate over them uniformly.
"""
from .slippage_model import SlippageRealisationModel
from .fill_model import FillProbabilityModel
from .survival_model import TimeToEventModel

__all__ = [
    "SlippageRealisationModel",
    "FillProbabilityModel",
    "TimeToEventModel",
]
