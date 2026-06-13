"""Black-Scholes Greeks + implied volatility from live premiums.

Kite Connect does NOT provide Greeks or IV. Sentinel computes them
from first principles: back out implied volatility from the live
option premium via bisection on the Black-Scholes price, then compute
delta / gamma / theta / vega analytically at that IV.

This is what powers the scenario engine honestly: "if NIFTY moves
+150, this put loses ~₹X" comes from repricing at the SAME implied
vol, not from a moneyness heuristic.

Limitations (stated, not hidden):
  * European BS on American-style index options — fine for NIFTY/
    BANKNIFTY (European exercise); slightly off for deep-ITM stock
    options (American). Index options are the primary use case.
  * Assumes IV constant under the scenario move. Real IV moves with
    spot (skew slide). v2 can add a sticky-delta adjustment; v1
    keeps the assumption visible in the UI copy.
  * Risk-free rate defaults to 6.5% (RBI repo ballpark); the impact
    on weekly options is negligible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

SQRT_2PI = math.sqrt(2.0 * math.pi)
DEFAULT_RATE = 0.065
IV_LO, IV_HI = 0.01, 4.00
MIN_T_YEARS = 1.0 / (365.0 * 24.0)   # one hour floor; avoids div-by-zero


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def bs_price(spot: float, strike: float, t_years: float, iv: float,
             option_type: str, rate: float = DEFAULT_RATE) -> float:
    """Black-Scholes European price. option_type: "CE" | "PE"."""
    t = max(t_years, MIN_T_YEARS)
    if spot <= 0 or strike <= 0 or iv <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
    d2 = d1 - iv * math.sqrt(t)
    if option_type == "CE":
        return spot * _norm_cdf(d1) - strike * math.exp(-rate * t) * _norm_cdf(d2)
    return strike * math.exp(-rate * t) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def implied_vol(premium: float, spot: float, strike: float, t_years: float,
                option_type: str, rate: float = DEFAULT_RATE,
                tol: float = 1e-4, max_iter: int = 80) -> Optional[float]:
    """Back out IV from a live premium via bisection. Returns None when
    the premium sits outside the no-arbitrage band (stale quote, wide
    spread print, or expiry-second weirdness) — callers must treat a
    None as 'Greeks unavailable for this leg', never crash."""
    t = max(t_years, MIN_T_YEARS)
    if premium <= 0 or spot <= 0 or strike <= 0:
        return None
    # No-arbitrage intrinsic floor.
    if option_type == "CE":
        intrinsic = max(0.0, spot - strike * math.exp(-rate * t))
    else:
        intrinsic = max(0.0, strike * math.exp(-rate * t) - spot)
    if premium < intrinsic - 1e-9:
        return None
    lo, hi = IV_LO, IV_HI
    p_lo = bs_price(spot, strike, t, lo, option_type, rate)
    p_hi = bs_price(spot, strike, t, hi, option_type, rate)
    if not (p_lo <= premium <= p_hi):
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p_mid = bs_price(spot, strike, t, mid, option_type, rate)
        if abs(p_mid - premium) < tol:
            return mid
        if p_mid < premium:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class GreeksView:
    iv: float
    delta: float        # signed: CE in [0,1], PE in [-1,0]
    gamma: float
    theta_per_day: float   # premium decay per calendar day (negative for longs)
    vega_per_pct: float    # premium change per 1 vol-point (1%) move


def greeks(spot: float, strike: float, t_years: float, iv: float,
           option_type: str, rate: float = DEFAULT_RATE) -> GreeksView:
    t = max(t_years, MIN_T_YEARS)
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
    d2 = d1 - iv * math.sqrt(t)
    pdf_d1 = _norm_pdf(d1)
    if option_type == "CE":
        delta = _norm_cdf(d1)
        theta = (-(spot * pdf_d1 * iv) / (2.0 * math.sqrt(t))
                 - rate * strike * math.exp(-rate * t) * _norm_cdf(d2))
    else:
        delta = _norm_cdf(d1) - 1.0
        theta = (-(spot * pdf_d1 * iv) / (2.0 * math.sqrt(t))
                 + rate * strike * math.exp(-rate * t) * _norm_cdf(-d2))
    gamma = pdf_d1 / (spot * iv * math.sqrt(t))
    vega = spot * pdf_d1 * math.sqrt(t)
    return GreeksView(
        iv=iv,
        delta=delta,
        gamma=gamma,
        theta_per_day=theta / 365.0,
        vega_per_pct=vega / 100.0,
    )


def reprice(spot_new: float, strike: float, t_years: float, iv: float,
            option_type: str, rate: float = DEFAULT_RATE) -> float:
    """Scenario repricing: same IV, new spot. The honest first-order
    answer to 'if the index moves to S', what is this leg worth?'"""
    return bs_price(spot_new, strike, t_years, iv, option_type, rate)
