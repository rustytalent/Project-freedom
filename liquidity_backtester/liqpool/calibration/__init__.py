"""Calibration upgrades — Stream N.

Three standalone calibrators that can be composed with the existing
isotonic + bucket-shrinkage pipeline in ``liqpool/ml_model.py``:

  * empirical_bayes — beta-binomial shrinkage that learns the prior
    strength from the data, replacing the fixed Wilson n/(n+20).

  * conformal — distribution-free prediction intervals with a
    coverage guarantee under exchangeability.

  * temperature — scalar temperature / Platt smoother that gently
    reshapes calibrator outputs to better held-out fit.

Each module exposes a fit()/transform()/apply() trio so the existing
pipeline can opt in without restructuring.
"""
from .empirical_bayes import (
    BetaBinomialBucketShrinker,
    fit_beta_binomial_prior,
)
from .conformal import (
    ConformalIntervalCalibrator,
)
from .temperature import (
    TemperatureScaler,
)

__all__ = [
    "BetaBinomialBucketShrinker",
    "fit_beta_binomial_prior",
    "ConformalIntervalCalibrator",
    "TemperatureScaler",
]
