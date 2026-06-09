"""Temperature / Platt scaling smoother.

A scalar post-hoc calibrator that gently reshapes probability
outputs without changing their RANK. Used downstream of isotonic
regression to smooth out the staircase artefacts isotonic produces
in sparse-data regions.

Method (single-parameter temperature scaling, Guo et al. 2017):
  * Convert input probabilities back to logits: l_hat = logit(p_hat).
  * Fit a single temperature T > 0 by minimising negative log-loss
    on a held-out set:
        p_T = sigmoid(l_hat / T)
  * T < 1 makes the calibrator more confident; T > 1 makes it
    less confident. The transformation is rank-preserving — no
    prediction crosses another.

This is the simplest possible calibration smoother; the audit
recommended Platt (which is a two-parameter affine logit
transformation). Temperature is one-parameter Platt with intercept
fixed at zero — appropriate when the upstream calibrator already
removed any bias and we only want to fix overall sharpness.

Pros vs full two-parameter Platt:
  * One parameter, can't overfit on a small calibration set.
  * Rank-preserving; doesn't break monotonicity guarantees from
    isotonic.
  * Trivial to implement without scipy.

Cons:
  * Can't fix mean bias (only sharpness). If the upstream calibrator
    is systematically high or low, temperature won't help.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class TemperatureScaler:
    """One-parameter temperature smoother.

    ``temperature == 1.0`` is the identity. Fitting calls a small
    1-D golden-section search over a bounded log-T range so we
    don't pull in scipy.

    Usage::

        ts = TemperatureScaler()
        ts.fit(p_hat_calib, y_calib)   # held-out calibration fold
        p_smoothed = ts.transform(p_hat_test)
    """
    temperature: float = 1.0
    n_calibration: int = 0
    nll_at_fit: float = float("nan")

    def fit(
        self,
        p_hat_calib: Sequence[float],
        y_calib: Sequence[float],
        eps_clip: float = 1e-6,
    ) -> "TemperatureScaler":
        p = np.asarray(p_hat_calib, dtype=float)
        y = np.asarray(y_calib, dtype=float)
        mask = np.isfinite(p) & np.isfinite(y)
        p, y = p[mask], y[mask]
        n = len(p)
        self.n_calibration = n
        if n < 5 or len(np.unique(y)) < 2:
            # Not enough signal to fit; identity transform.
            self.temperature = 1.0
            self.nll_at_fit = float("nan")
            return self

        p = np.clip(p, eps_clip, 1.0 - eps_clip)
        logits = np.log(p / (1.0 - p))

        def nll(t: float) -> float:
            if t <= 0:
                return float("inf")
            z = logits / t
            # Stable log(1 + exp(-z)) and log(1 + exp(z)) computation.
            # NLL = sum( y * log(1+exp(-z)) + (1-y) * log(1+exp(z)) ).
            loss = np.where(
                y > 0.5, _softplus(-z), _softplus(z)
            ).sum()
            return float(loss)

        # Golden-section on log(T) in [log(0.1), log(10)].
        self.temperature = _golden_minimise(
            nll, lo=0.1, hi=10.0, tol=1e-4,
        )
        self.nll_at_fit = nll(self.temperature)
        return self

    def transform(
        self,
        p_hat: Sequence[float],
        eps_clip: float = 1e-6,
    ) -> np.ndarray:
        p = np.asarray(p_hat, dtype=float)
        p_clipped = np.clip(p, eps_clip, 1.0 - eps_clip)
        logits = np.log(p_clipped / (1.0 - p_clipped))
        scaled = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled))

    def transform_one(self, p_hat: float, eps_clip: float = 1e-6) -> float:
        return float(self.transform(np.array([p_hat]), eps_clip)[0])


def _softplus(x: np.ndarray) -> np.ndarray:
    """Stable log(1 + exp(x))."""
    # For x > 0: x + log(1 + exp(-x))
    # For x <= 0: log(1 + exp(x))
    return np.where(
        x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x))
    )


def _golden_minimise(
    f, lo: float, hi: float, tol: float = 1e-4, max_iter: int = 100,
) -> float:
    """Golden-section search for a unimodal scalar function. Avoids
    a scipy dependency; sufficient for the 1-D temperature fit."""
    phi = (math.sqrt(5.0) + 1.0) / 2.0
    a, b = lo, hi
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) / phi
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) / phi
            fd = f(d)
    return (a + b) / 2.0
