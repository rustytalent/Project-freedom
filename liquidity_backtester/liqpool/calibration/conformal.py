"""Conformal prediction intervals — distribution-free coverage.

Wraps any existing calibrator (isotonic regression, Platt scaling,
empirical-Bayes shrinker) and adds an honest [lower, upper] band
around every output. The contract:

    Under exchangeability of training and test data, the produced
    100*(1 - alpha)% interval covers the true outcome at least
    (1 - alpha) fraction of the time.

For our use case (calibrated probabilities), 'true outcome' is the
binary hit/miss. We use **inductive (split) conformal prediction**:
hold out a calibration set, score residuals on it, take the
appropriate quantile, broadcast to test predictions as a half-width.

Method:
  * On the calibration fold, compute residuals
        s_i = |y_i - p_hat_i|  (absolute prediction error)
  * The conformal quantile q* is the ceil((n+1)(1 - alpha))/n th
    quantile of residuals.
  * For a new prediction p_hat, the interval is
        [p_hat - q*, p_hat + q*] clipped to [0, 1].

Notes:
  * 'Marginal coverage' guarantee — the (1 - alpha) coverage holds
    on AVERAGE over the joint distribution, not conditionally on a
    specific feature value. Conditional coverage requires localised
    conformal (CQR or Mondrian); deferred to v2.
  * Independent of the underlying calibrator's miscalibration —
    a poorly calibrated p_hat still gets the right marginal
    coverage by widening the intervals. That's the *whole point*.
  * The interval width is a useful HONESTY signal: wide intervals
    on important predictions mean the underlying model is unsure,
    and the brief should disclose that.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class ConformalIntervalCalibrator:
    """Inductive split-conformal wrapper.

    Fit on (p_hat_calib, y_calib) where p_hat_calib is the underlying
    calibrator's output on a held-out fold (NOT the training fold —
    that's the whole point of split conformal).

    ``predict_interval(p_hat, alpha)`` returns ``(lower, upper)``
    with marginal (1 - alpha) coverage. ``alpha`` defaults to 0.10
    (a 90% interval).
    """
    half_width_quantile: float = 0.0
    n_calibration: int = 0
    alpha_used_at_fit: float = 0.10

    def fit(
        self,
        p_hat_calib: Sequence[float],
        y_calib: Sequence[float],
        alpha: float = 0.10,
    ) -> "ConformalIntervalCalibrator":
        """Compute the conformal quantile from calibration residuals.

        ``alpha`` is the miscoverage rate at fit time. The same alpha
        is the default at predict time but ``predict_interval`` can
        override it (with a small caveat: a different alpha at
        predict time requires the quantile be re-computed from the
        stored residuals; in v1 we just refit if alpha changes)."""
        p = np.asarray(p_hat_calib, dtype=float)
        y = np.asarray(y_calib, dtype=float)
        mask = np.isfinite(p) & np.isfinite(y)
        p, y = p[mask], y[mask]
        n = len(p)
        if n == 0:
            self.half_width_quantile = 0.5
            self.n_calibration = 0
            self.alpha_used_at_fit = alpha
            return self

        residuals = np.abs(y - p)
        # The split-conformal correction: rank = ceil((n + 1)(1 - alpha)).
        rank = math.ceil((n + 1) * (1.0 - alpha))
        rank = min(max(rank, 1), n)
        sorted_resid = np.sort(residuals)
        # rank is 1-indexed; index = rank - 1.
        self.half_width_quantile = float(sorted_resid[rank - 1])
        self.n_calibration = n
        self.alpha_used_at_fit = alpha
        return self

    def predict_interval(
        self,
        p_hat: float,
        alpha: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Return ``(lower, upper)`` for one point estimate.

        For now, only ``alpha`` values that match the fit alpha are
        supported — passing a different alpha is allowed but logged
        as a no-op (interval width unchanged). v2 will re-quantile
        from stored residuals.
        """
        hw = self.half_width_quantile
        lower = max(0.0, p_hat - hw)
        upper = min(1.0, p_hat + hw)
        return lower, upper

    def predict_intervals(
        self,
        p_hats: Sequence[float],
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Vectorised version. Returns an (n, 2) array."""
        p = np.asarray(p_hats, dtype=float)
        hw = self.half_width_quantile
        lo = np.clip(p - hw, 0.0, 1.0)
        hi = np.clip(p + hw, 0.0, 1.0)
        return np.column_stack([lo, hi])

    def empirical_coverage(
        self,
        p_hat_test: Sequence[float],
        y_test: Sequence[float],
    ) -> float:
        """Diagnostic: what fraction of test points actually fell
        inside their predicted intervals? Should be close to
        ``1 - alpha_used_at_fit`` under exchangeability."""
        intervals = self.predict_intervals(p_hat_test)
        y = np.asarray(y_test, dtype=float)
        inside = (y >= intervals[:, 0]) & (y <= intervals[:, 1])
        return float(inside.mean()) if len(inside) else float("nan")
