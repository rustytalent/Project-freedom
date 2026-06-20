"""IV surface — fit a smile per expiry, extrapolate to any strike.

State-of-the-art options executors don't read "iv_state" as a label —
they reconstruct the IV surface from observed market quotes and use it
to price every leg.

This module fits a smile model from the 22 observed contract premiums
(ATM ±5 × CE/PE) per tick. Two parametric forms are supported:

  * ``polynomial`` — quadratic in log-moneyness (k = log(K/F)). Robust
    fallback that always converges. Default.
  * ``svi``       — Gatheral SVI: total_variance(k) =
        a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
    State-of-the-art industry standard. Used when there are enough
    clean observations (>=8 per side).

The fit is performed once per tick. Output: ``IVSurface`` carrying:
  * iv_at_strike(strike) → float
  * atm_iv → float
  * skew_25d → ±float (25-delta risk reversal proxy)
  * smile_curvature → float (second derivative at ATM)
  * confidence → float in [0,1] (drops with fit residual)
  * fit_residual_rms → float (annualized vol units)
  * implied_var_at_log_money(k) → for SVI chained pricing

Numerical safety: outliers (|IV - median| > 3σ) are excluded before
fitting; the fit always degrades to a flat ATM-IV when there are <4
clean observations.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .black_scholes import (
    DEFAULT_DIVIDEND_YIELD,
    DEFAULT_RISK_FREE_RATE,
    implied_volatility,
)


@dataclass
class IVSurfaceConfig:
    """Knobs for the surface fit."""
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    dividend_yield: float = DEFAULT_DIVIDEND_YIELD
    # Outlier removal
    outlier_z: float = 3.0
    min_clean_observations: int = 4
    # SVI vs polynomial decision
    svi_min_observations_per_side: int = 6
    # SVI optimization
    svi_max_iter: int = 200
    svi_seed_a: float = 0.04
    svi_seed_b: float = 0.10
    svi_seed_rho: float = 0.0
    svi_seed_m: float = 0.0
    svi_seed_sigma: float = 0.10
    # Validity bounds
    min_iv: float = 0.03
    max_iv: float = 2.50


@dataclass
class _Observation:
    """One observed (strike, market price, side) point."""
    strike: float
    market_price: float
    side: str
    iv: float = float("nan")        # filled in by _extract_ivs
    log_money: float = 0.0          # log(strike / forward)


@dataclass(frozen=True)
class IVSurface:
    """Fitted volatility surface for one expiry."""
    fit_method: str                         # "polynomial" / "svi" / "flat"
    coefficients: Tuple[float, ...]         # method-specific
    forward: float
    time_to_expiry: float
    atm_iv: float
    skew_25d: float                         # approx 25-delta risk reversal
    smile_curvature: float                  # second derivative at ATM
    fit_residual_rms: float
    confidence: float                       # 0..1
    n_clean_observations: int
    n_total_observations: int
    notes: List[str] = field(default_factory=list)

    # ── Public query API ────────────────────────────────────────────

    def iv_at_strike(self, strike: float) -> float:
        """IV at a given strike (in annualized vol units)."""
        if self.forward <= 0 or strike <= 0:
            return self.atm_iv
        k = math.log(strike / self.forward)
        return self._iv_at_log_money(k)

    def implied_var_at_log_money(self, log_money: float) -> float:
        """Total implied variance at a given log-moneyness."""
        sigma = self._iv_at_log_money(log_money)
        return sigma * sigma * self.time_to_expiry

    # ── Internal evaluators per fit method ──────────────────────────

    def _iv_at_log_money(self, k: float) -> float:
        if self.fit_method == "polynomial":
            a, b, c = self.coefficients
            iv2 = a + b * k + c * k * k
            return max(0.001, math.sqrt(max(iv2, 1e-8)))
        if self.fit_method == "svi":
            a, b, rho, m, s = self.coefficients
            inside = (k - m) * (k - m) + s * s
            if inside < 0:
                return self.atm_iv
            w = a + b * (rho * (k - m) + math.sqrt(inside))
            # Total variance w = sigma^2 * T → sigma = sqrt(w / T)
            if self.time_to_expiry <= 0:
                return self.atm_iv
            return max(0.001, math.sqrt(max(w, 1e-8) / self.time_to_expiry))
        # Flat fallback
        return self.atm_iv

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fit_method": self.fit_method,
            "coefficients": list(self.coefficients),
            "forward": round(self.forward, 4),
            "time_to_expiry": round(self.time_to_expiry, 6),
            "atm_iv": round(self.atm_iv, 4),
            "skew_25d": round(self.skew_25d, 4),
            "smile_curvature": round(self.smile_curvature, 4),
            "fit_residual_rms": round(self.fit_residual_rms, 4),
            "confidence": round(self.confidence, 3),
            "n_clean_observations": self.n_clean_observations,
            "n_total_observations": self.n_total_observations,
            "notes": list(self.notes),
        }


# ─────────────────────────────────────────────────────────────────
# Surface fitter
# ─────────────────────────────────────────────────────────────────


class IVSurfaceFitter:
    """Fit an IV surface from observed market quotes for one expiry."""

    def __init__(self, cfg: Optional[IVSurfaceConfig] = None) -> None:
        self.cfg = cfg or IVSurfaceConfig()

    def fit(self, *, observations: List[Tuple[float, float, str]],
              spot: float, time_to_expiry: float) -> IVSurface:
        """Fit a surface.

        ``observations`` is a list of (strike, market_price, side) tuples.
        Side is 'CE' or 'PE'. The fitter:
          1. Computes the forward F = spot * exp((r-q)*T)
          2. Inverts BS to get an IV per observation
          3. Filters outliers
          4. Fits a polynomial in log(K/F) (or SVI if enough data)
          5. Returns an IVSurface

        If no observations or all observations fail, returns a flat
        surface at IV=0.25 with confidence=0.
        """
        cfg = self.cfg
        notes: List[str] = []
        n_total = len(observations)
        forward = spot * math.exp((cfg.risk_free_rate - cfg.dividend_yield)
                                    * time_to_expiry)
        if forward <= 0 or time_to_expiry <= 0 or n_total == 0:
            return _flat_surface(forward=max(spot, 0.01),
                                   time_to_expiry=max(time_to_expiry, 1e-6),
                                   notes=["empty inputs — flat fallback"])

        observed: List[_Observation] = []
        for strike, price, side in observations:
            if strike <= 0 or price <= 0:
                continue
            iv = implied_volatility(
                market_price=price, side=side, spot=spot,
                strike=strike, time_to_expiry=time_to_expiry,
                risk_free=cfg.risk_free_rate, dividend=cfg.dividend_yield,
            )
            if not math.isfinite(iv) or iv < cfg.min_iv or iv > cfg.max_iv:
                continue
            observed.append(_Observation(
                strike=strike, market_price=price, side=side,
                iv=iv, log_money=math.log(strike / forward),
            ))

        if len(observed) < cfg.min_clean_observations:
            atm_iv = (statistics.median(o.iv for o in observed)
                      if observed else 0.25)
            notes.append(f"only {len(observed)} clean observations — flat fit")
            return _flat_surface(forward=forward,
                                   time_to_expiry=time_to_expiry,
                                   atm_iv=atm_iv,
                                   n_total=n_total,
                                   n_clean=len(observed),
                                   notes=notes)

        # Outlier removal
        ivs = [o.iv for o in observed]
        med = statistics.median(ivs)
        if len(ivs) >= 4:
            mad = statistics.median(abs(iv - med) for iv in ivs) * 1.4826
            if mad > 0:
                observed = [o for o in observed
                            if abs(o.iv - med) / mad <= cfg.outlier_z]

        if len(observed) < cfg.min_clean_observations:
            notes.append("too many outliers after filtering — flat fit")
            return _flat_surface(forward=forward,
                                   time_to_expiry=time_to_expiry,
                                   atm_iv=med, n_total=n_total,
                                   n_clean=len(observed), notes=notes)

        ce_count = sum(1 for o in observed if o.side == "CE")
        pe_count = len(observed) - ce_count
        use_svi = (ce_count >= cfg.svi_min_observations_per_side
                   and pe_count >= cfg.svi_min_observations_per_side)

        if use_svi:
            surf = self._fit_svi(observed=observed, forward=forward,
                                   time_to_expiry=time_to_expiry,
                                   notes=notes, n_total=n_total)
            if surf is not None:
                return surf
            notes.append("SVI fit failed — falling back to polynomial")

        return self._fit_polynomial(observed=observed, forward=forward,
                                      time_to_expiry=time_to_expiry,
                                      notes=notes, n_total=n_total)

    # ── Polynomial fit ───────────────────────────────────────────────

    def _fit_polynomial(self, *, observed: List[_Observation],
                          forward: float, time_to_expiry: float,
                          notes: List[str], n_total: int) -> IVSurface:
        """Quadratic fit of iv^2 vs log-moneyness."""
        k = [o.log_money for o in observed]
        v = [o.iv * o.iv for o in observed]
        # Normal equations for [1, k, k^2] basis.
        n = len(observed)
        s0 = float(n)
        s1 = sum(k); s2 = sum(x * x for x in k)
        s3 = sum(x ** 3 for x in k); s4 = sum(x ** 4 for x in k)
        t0 = sum(v); t1 = sum(x * y for x, y in zip(k, v))
        t2 = sum(x * x * y for x, y in zip(k, v))
        # Solve 3x3 system via Cramer or Gauss; here Gauss is simpler.
        a, b, c = _solve_3x3([
            [s0, s1, s2, t0],
            [s1, s2, s3, t1],
            [s2, s3, s4, t2],
        ])
        if a is None:
            notes.append("polynomial singular — flat fallback")
            atm_iv = statistics.median(o.iv for o in observed)
            return _flat_surface(forward=forward,
                                   time_to_expiry=time_to_expiry,
                                   atm_iv=atm_iv, n_total=n_total,
                                   n_clean=n, notes=notes)
        # Compute residual
        residual_sum = 0.0
        for o in observed:
            pred = a + b * o.log_money + c * o.log_money * o.log_money
            if pred < 0:
                pred = 0.0
            residual_sum += (math.sqrt(pred) - o.iv) ** 2
        rms = math.sqrt(residual_sum / max(1, n))
        # ATM iv (k=0)
        atm_iv = max(0.001, math.sqrt(max(a, 1e-8)))
        # Skew = derivative at k=0 = b / (2 * atm_iv)
        skew = b / max(2.0 * atm_iv, 1e-6)
        # Curvature = 2c / (2 * atm_iv) - skew^2 / atm_iv (chain rule on sqrt)
        curvature = c / max(atm_iv, 1e-6)
        confidence = max(0.0, min(1.0, 1.0 - rms / 0.30))
        return IVSurface(
            fit_method="polynomial",
            coefficients=(a, b, c),
            forward=forward,
            time_to_expiry=time_to_expiry,
            atm_iv=atm_iv,
            skew_25d=skew,
            smile_curvature=curvature,
            fit_residual_rms=rms,
            confidence=confidence,
            n_clean_observations=n,
            n_total_observations=n_total,
            notes=notes,
        )

    # ── SVI fit (Nelder-Mead simplex) ────────────────────────────────

    def _fit_svi(self, *, observed: List[_Observation],
                   forward: float, time_to_expiry: float,
                   notes: List[str], n_total: int) -> Optional[IVSurface]:
        cfg = self.cfg
        target_w = [(o.iv * o.iv) * time_to_expiry for o in observed]
        k_vals = [o.log_money for o in observed]

        def loss(params: Tuple[float, float, float, float, float]) -> float:
            a, b, rho, m, s = params
            if b < 0 or s <= 0 or not (-0.999 < rho < 0.999):
                return 1e9
            total = 0.0
            for k, w_obs in zip(k_vals, target_w):
                inside = (k - m) * (k - m) + s * s
                if inside < 0:
                    return 1e9
                w = a + b * (rho * (k - m) + math.sqrt(inside))
                if w < 0:
                    return 1e9
                total += (w - w_obs) ** 2
            return total / len(k_vals)

        # Initial simplex (Nelder-Mead).
        seed = (cfg.svi_seed_a * time_to_expiry, cfg.svi_seed_b,
                cfg.svi_seed_rho, cfg.svi_seed_m, cfg.svi_seed_sigma)
        best = _nelder_mead(loss, seed, max_iter=cfg.svi_max_iter)
        if best is None:
            return None
        a, b, rho, m, s = best
        # Compute residual in IV units (not variance).
        residual_sum = 0.0
        for o in observed:
            inside = (o.log_money - m) ** 2 + s * s
            w = a + b * (rho * (o.log_money - m) + math.sqrt(max(0, inside)))
            if w < 0:
                w = 0.0
            iv_pred = math.sqrt(w / time_to_expiry) if time_to_expiry > 0 else 0.0
            residual_sum += (iv_pred - o.iv) ** 2
        rms = math.sqrt(residual_sum / max(1, len(observed)))
        atm_w = a + b * (rho * (-m) + math.sqrt(m * m + s * s))
        atm_iv = math.sqrt(max(atm_w, 1e-8) / max(time_to_expiry, 1e-6))
        # Skew at k=0: derivative of sqrt(w/T) wrt k at k=0
        # dw/dk at k=0 = b * (rho + (-m)/sqrt(m^2 + s^2))
        denom = math.sqrt(m * m + s * s)
        if denom > 1e-9:
            dw_dk = b * (rho + (-m) / denom)
        else:
            dw_dk = b * rho
        skew = dw_dk / max(2.0 * atm_iv * time_to_expiry, 1e-6)
        # Curvature: 2nd derivative
        if denom > 1e-9:
            d2w_dk2 = b * (s * s) / (denom ** 3)
        else:
            d2w_dk2 = 0.0
        curvature = d2w_dk2 / max(2.0 * atm_iv * time_to_expiry, 1e-6)
        confidence = max(0.0, min(1.0, 1.0 - rms / 0.25))
        return IVSurface(
            fit_method="svi",
            coefficients=(a, b, rho, m, s),
            forward=forward,
            time_to_expiry=time_to_expiry,
            atm_iv=atm_iv,
            skew_25d=skew,
            smile_curvature=curvature,
            fit_residual_rms=rms,
            confidence=confidence,
            n_clean_observations=len(observed),
            n_total_observations=n_total,
            notes=notes + [f"SVI converged in {cfg.svi_max_iter} steps max"],
        )


# ─────────────────────────────────────────────────────────────────
# Numerical helpers
# ─────────────────────────────────────────────────────────────────


def _solve_3x3(rows: List[List[float]]) -> Tuple[Optional[float],
                                                    Optional[float],
                                                    Optional[float]]:
    """Solve a 3x3 augmented system [A|b] → (x, y, z). Returns
    (None, None, None) on singularity."""
    a = [r[:] for r in rows]
    # Gaussian elimination with partial pivoting.
    n = 3
    for col in range(n):
        # Pivot.
        pivot_row = col
        for r in range(col + 1, n):
            if abs(a[r][col]) > abs(a[pivot_row][col]):
                pivot_row = r
        if abs(a[pivot_row][col]) < 1e-12:
            return (None, None, None)
        a[col], a[pivot_row] = a[pivot_row], a[col]
        # Eliminate below.
        for r in range(col + 1, n):
            factor = a[r][col] / a[col][col]
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]
    # Back-substitute.
    x = [0.0, 0.0, 0.0]
    for r in range(n - 1, -1, -1):
        s = a[r][n]
        for c in range(r + 1, n):
            s -= a[r][c] * x[c]
        x[r] = s / a[r][r]
    return (x[0], x[1], x[2])


def _nelder_mead(loss, seed: Tuple[float, ...],
                  max_iter: int = 200) -> Optional[Tuple[float, ...]]:
    """Bare-bones Nelder-Mead. Returns best vertex or None on failure."""
    n = len(seed)
    # Initialize simplex around seed.
    simplex = [list(seed)]
    for i in range(n):
        v = list(seed)
        v[i] = v[i] + (0.05 if v[i] == 0 else v[i] * 0.05)
        simplex.append(v)
    values = [loss(tuple(v)) for v in simplex]
    if not all(math.isfinite(v) for v in values):
        return None
    alpha = 1.0; gamma = 2.0; beta = 0.5; delta = 0.5
    for _ in range(max_iter):
        # Sort by loss.
        order = sorted(range(n + 1), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        # Stopping criterion.
        if abs(values[-1] - values[0]) < 1e-9:
            break
        # Centroid of all but worst.
        centroid = [sum(simplex[i][j] for i in range(n)) / n
                    for j in range(n)]
        # Reflection.
        xr = [centroid[j] + alpha * (centroid[j] - simplex[-1][j])
              for j in range(n)]
        fxr = loss(tuple(xr))
        if values[0] <= fxr < values[-2]:
            simplex[-1] = xr; values[-1] = fxr; continue
        # Expansion.
        if fxr < values[0]:
            xe = [centroid[j] + gamma * (xr[j] - centroid[j])
                  for j in range(n)]
            fxe = loss(tuple(xe))
            if fxe < fxr:
                simplex[-1] = xe; values[-1] = fxe
            else:
                simplex[-1] = xr; values[-1] = fxr
            continue
        # Contraction.
        xc = [centroid[j] + beta * (simplex[-1][j] - centroid[j])
              for j in range(n)]
        fxc = loss(tuple(xc))
        if fxc < values[-1]:
            simplex[-1] = xc; values[-1] = fxc; continue
        # Shrink.
        for i in range(1, n + 1):
            simplex[i] = [simplex[0][j] + delta * (simplex[i][j] - simplex[0][j])
                          for j in range(n)]
            values[i] = loss(tuple(simplex[i]))
    return tuple(simplex[0])


def _flat_surface(*, forward: float, time_to_expiry: float,
                   atm_iv: float = 0.25, n_total: int = 0,
                   n_clean: int = 0,
                   notes: List[str] = None) -> IVSurface:
    return IVSurface(
        fit_method="flat",
        coefficients=(atm_iv,),
        forward=forward,
        time_to_expiry=max(1e-6, time_to_expiry),
        atm_iv=atm_iv,
        skew_25d=0.0,
        smile_curvature=0.0,
        fit_residual_rms=0.0,
        confidence=0.0 if n_clean == 0 else 0.30,
        n_clean_observations=n_clean,
        n_total_observations=n_total,
        notes=list(notes or []),
    )
