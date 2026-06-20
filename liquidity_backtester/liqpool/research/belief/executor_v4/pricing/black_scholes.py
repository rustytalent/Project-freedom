"""Black-Scholes-Merton pricing and Greeks — numerically robust.

This module is the bedrock of v4's options math. Everything in the
strategy library, risk layer, and hedge proposer that previously used
toy linear-decay greeks now consumes these functions.

Conventions:
  * European options (NIFTY weekly options trade European-style)
  * Continuous compounding for risk-free and dividend yield
  * Time-to-expiry T measured in YEARS (e.g. 7 days = 7/365)
  * Volatility ``sigma`` measured as annualized standard deviation
  * Spot prices and strikes are positive floats; bad inputs return NaN

Numerical robustness:
  * d1/d2 use ``math.log`` with positive-spot/strike guards
  * cumulative normal computed via ``math.erfc`` (more accurate than
    series in deep tails)
  * IV inversion uses Brent's method with safeguarded Newton-Raphson
    seed; handles the case where the option is essentially worthless
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# Square root of 2π — used everywhere in Black-Scholes.
_SQRT_2PI = math.sqrt(2.0 * math.pi)

# Default conventions for NIFTY weekly options on NSE.
DEFAULT_RISK_FREE_RATE = 0.065      # ~6.5% (India 91-day T-bill)
DEFAULT_DIVIDEND_YIELD = 0.012      # NIFTY index dividend yield


# ─────────────────────────────────────────────────────────────────
# Cumulative normal — numerically stable
# ─────────────────────────────────────────────────────────────────


def norm_cdf(x: float) -> float:
    """Standard-normal CDF, accurate in the tails."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def norm_pdf(x: float) -> float:
    """Standard-normal PDF."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


# ─────────────────────────────────────────────────────────────────
# d1, d2
# ─────────────────────────────────────────────────────────────────


def _d1(spot: float, strike: float, time_to_expiry: float, sigma: float,
         risk_free: float, dividend: float) -> float:
    """Black-Scholes d1.

    Guards against ``sigma <= 0``, ``time_to_expiry <= 0``, and
    non-positive spot/strike by returning NaN.
    """
    if (spot <= 0 or strike <= 0 or time_to_expiry <= 0 or sigma <= 0):
        return float("nan")
    return ((math.log(spot / strike)
             + (risk_free - dividend + 0.5 * sigma * sigma) * time_to_expiry)
            / (sigma * math.sqrt(time_to_expiry)))


def _d2(d1: float, sigma: float, time_to_expiry: float) -> float:
    if not math.isfinite(d1) or sigma <= 0 or time_to_expiry <= 0:
        return float("nan")
    return d1 - sigma * math.sqrt(time_to_expiry)


# ─────────────────────────────────────────────────────────────────
# Premium (Black-Scholes-Merton with continuous dividend)
# ─────────────────────────────────────────────────────────────────


def bs_call_price(*, spot: float, strike: float, time_to_expiry: float,
                    sigma: float, risk_free: float = DEFAULT_RISK_FREE_RATE,
                    dividend: float = DEFAULT_DIVIDEND_YIELD) -> float:
    """European call price (Black-Scholes-Merton)."""
    if time_to_expiry <= 0:
        return max(0.0, spot * math.exp(-dividend * 0.0) - strike)
    if sigma <= 0:
        return max(0.0,
                    spot * math.exp(-dividend * time_to_expiry)
                    - strike * math.exp(-risk_free * time_to_expiry))
    d1 = _d1(spot, strike, time_to_expiry, sigma, risk_free, dividend)
    d2 = _d2(d1, sigma, time_to_expiry)
    return (spot * math.exp(-dividend * time_to_expiry) * norm_cdf(d1)
             - strike * math.exp(-risk_free * time_to_expiry) * norm_cdf(d2))


def bs_put_price(*, spot: float, strike: float, time_to_expiry: float,
                   sigma: float, risk_free: float = DEFAULT_RISK_FREE_RATE,
                   dividend: float = DEFAULT_DIVIDEND_YIELD) -> float:
    """European put price (Black-Scholes-Merton)."""
    if time_to_expiry <= 0:
        return max(0.0, strike - spot)
    if sigma <= 0:
        return max(0.0,
                    strike * math.exp(-risk_free * time_to_expiry)
                    - spot * math.exp(-dividend * time_to_expiry))
    d1 = _d1(spot, strike, time_to_expiry, sigma, risk_free, dividend)
    d2 = _d2(d1, sigma, time_to_expiry)
    return (strike * math.exp(-risk_free * time_to_expiry) * norm_cdf(-d2)
             - spot * math.exp(-dividend * time_to_expiry) * norm_cdf(-d1))


# ─────────────────────────────────────────────────────────────────
# Greeks
# ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OptionGreeks:
    """Per-leg Greeks. All annualized (theta in ₹/year, scale to per-day
    by /365 in display code)."""
    price: float
    delta: float
    gamma: float
    vega: float                 # per 1.00 change in sigma
    theta: float                # per year
    rho: float
    vanna: float                # cross-greek: d delta / d sigma
    charm: float                # d delta / d t
    vomma: float                # d vega / d sigma

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return {k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in d.items()}


def call_greeks(*, spot: float, strike: float, time_to_expiry: float,
                  sigma: float, risk_free: float = DEFAULT_RISK_FREE_RATE,
                  dividend: float = DEFAULT_DIVIDEND_YIELD) -> OptionGreeks:
    """Full European call Greeks."""
    if (spot <= 0 or strike <= 0 or time_to_expiry <= 0 or sigma <= 0):
        return OptionGreeks(price=float("nan"), delta=float("nan"),
                              gamma=float("nan"), vega=float("nan"),
                              theta=float("nan"), rho=float("nan"),
                              vanna=float("nan"), charm=float("nan"),
                              vomma=float("nan"))
    d1 = _d1(spot, strike, time_to_expiry, sigma, risk_free, dividend)
    d2 = _d2(d1, sigma, time_to_expiry)
    sqrt_t = math.sqrt(time_to_expiry)
    disc_q = math.exp(-dividend * time_to_expiry)
    disc_r = math.exp(-risk_free * time_to_expiry)
    nd1 = norm_cdf(d1); nd2 = norm_cdf(d2)
    pd1 = norm_pdf(d1)
    price = spot * disc_q * nd1 - strike * disc_r * nd2
    delta = disc_q * nd1
    gamma = disc_q * pd1 / (spot * sigma * sqrt_t)
    vega = spot * disc_q * pd1 * sqrt_t
    theta = (-spot * disc_q * pd1 * sigma / (2.0 * sqrt_t)
             - risk_free * strike * disc_r * nd2
             + dividend * spot * disc_q * nd1)
    rho = strike * time_to_expiry * disc_r * nd2
    vanna = -disc_q * pd1 * d2 / sigma
    charm = (-disc_q * pd1 * (
        (risk_free - dividend) / (sigma * sqrt_t)
        - d2 / (2.0 * time_to_expiry))
        - dividend * disc_q * nd1)
    vomma = vega * (d1 * d2 / sigma)
    return OptionGreeks(
        price=price, delta=delta, gamma=gamma, vega=vega,
        theta=theta, rho=rho, vanna=vanna, charm=charm, vomma=vomma,
    )


def put_greeks(*, spot: float, strike: float, time_to_expiry: float,
                 sigma: float, risk_free: float = DEFAULT_RISK_FREE_RATE,
                 dividend: float = DEFAULT_DIVIDEND_YIELD) -> OptionGreeks:
    """Full European put Greeks."""
    if (spot <= 0 or strike <= 0 or time_to_expiry <= 0 or sigma <= 0):
        return OptionGreeks(price=float("nan"), delta=float("nan"),
                              gamma=float("nan"), vega=float("nan"),
                              theta=float("nan"), rho=float("nan"),
                              vanna=float("nan"), charm=float("nan"),
                              vomma=float("nan"))
    d1 = _d1(spot, strike, time_to_expiry, sigma, risk_free, dividend)
    d2 = _d2(d1, sigma, time_to_expiry)
    sqrt_t = math.sqrt(time_to_expiry)
    disc_q = math.exp(-dividend * time_to_expiry)
    disc_r = math.exp(-risk_free * time_to_expiry)
    n_neg_d1 = norm_cdf(-d1); n_neg_d2 = norm_cdf(-d2)
    pd1 = norm_pdf(d1)
    price = strike * disc_r * n_neg_d2 - spot * disc_q * n_neg_d1
    delta = -disc_q * n_neg_d1
    gamma = disc_q * pd1 / (spot * sigma * sqrt_t)
    vega = spot * disc_q * pd1 * sqrt_t
    theta = (-spot * disc_q * pd1 * sigma / (2.0 * sqrt_t)
             + risk_free * strike * disc_r * n_neg_d2
             - dividend * spot * disc_q * n_neg_d1)
    rho = -strike * time_to_expiry * disc_r * n_neg_d2
    vanna = -disc_q * pd1 * d2 / sigma
    charm = (-disc_q * pd1 * (
        (risk_free - dividend) / (sigma * sqrt_t)
        - d2 / (2.0 * time_to_expiry))
        + dividend * disc_q * n_neg_d1)
    vomma = vega * (d1 * d2 / sigma)
    return OptionGreeks(
        price=price, delta=delta, gamma=gamma, vega=vega,
        theta=theta, rho=rho, vanna=vanna, charm=charm, vomma=vomma,
    )


def greeks_for_side(side: str, *, spot: float, strike: float,
                      time_to_expiry: float, sigma: float,
                      risk_free: float = DEFAULT_RISK_FREE_RATE,
                      dividend: float = DEFAULT_DIVIDEND_YIELD) -> OptionGreeks:
    """Dispatch on side ('CE'/'PE')."""
    if side == "CE":
        return call_greeks(spot=spot, strike=strike,
                              time_to_expiry=time_to_expiry, sigma=sigma,
                              risk_free=risk_free, dividend=dividend)
    if side == "PE":
        return put_greeks(spot=spot, strike=strike,
                             time_to_expiry=time_to_expiry, sigma=sigma,
                             risk_free=risk_free, dividend=dividend)
    raise ValueError(f"Unknown side {side!r}; expected CE or PE")


# ─────────────────────────────────────────────────────────────────
# Implied volatility inversion (Brent's method with Newton seed)
# ─────────────────────────────────────────────────────────────────


def implied_volatility(*, market_price: float, side: str,
                          spot: float, strike: float,
                          time_to_expiry: float,
                          risk_free: float = DEFAULT_RISK_FREE_RATE,
                          dividend: float = DEFAULT_DIVIDEND_YIELD,
                          tolerance: float = 1e-6,
                          max_iter: int = 64,
                          lower: float = 1e-4,
                          upper: float = 5.0,
                          ) -> float:
    """Invert Black-Scholes to recover implied volatility.

    Uses Newton-Raphson seeded near ATM-IV (~0.25 for NIFTY), with
    Brent's bisection fallback if Newton diverges. Returns NaN for
    inputs that can't have a finite IV (price below intrinsic, etc.).
    """
    if (spot <= 0 or strike <= 0 or time_to_expiry <= 0
            or market_price <= 0):
        return float("nan")
    # Lower bound: intrinsic value.
    if side == "CE":
        intrinsic = max(0.0,
                         spot * math.exp(-dividend * time_to_expiry)
                         - strike * math.exp(-risk_free * time_to_expiry))
    else:
        intrinsic = max(0.0,
                         strike * math.exp(-risk_free * time_to_expiry)
                         - spot * math.exp(-dividend * time_to_expiry))
    if market_price < intrinsic - 1e-6:
        return float("nan")     # price below intrinsic — bad data
    # Upper bound: capped by strike or spot.
    max_premium = (spot if side == "CE"
                    else strike * math.exp(-risk_free * time_to_expiry))
    if market_price >= max_premium:
        return float("nan")

    def _price_for(sigma: float) -> float:
        if side == "CE":
            return bs_call_price(spot=spot, strike=strike,
                                    time_to_expiry=time_to_expiry,
                                    sigma=sigma, risk_free=risk_free,
                                    dividend=dividend)
        return bs_put_price(spot=spot, strike=strike,
                               time_to_expiry=time_to_expiry,
                               sigma=sigma, risk_free=risk_free,
                               dividend=dividend)

    # Try Newton-Raphson first.
    sigma = 0.25
    for _ in range(20):
        price = _price_for(sigma)
        if not math.isfinite(price):
            break
        diff = price - market_price
        if abs(diff) < tolerance:
            return max(lower, min(upper, sigma))
        # vega in price terms
        d1 = _d1(spot, strike, time_to_expiry, sigma, risk_free, dividend)
        if not math.isfinite(d1):
            break
        vega = (spot * math.exp(-dividend * time_to_expiry)
                 * norm_pdf(d1) * math.sqrt(time_to_expiry))
        if vega < 1e-10:
            break
        step = diff / vega
        sigma_new = sigma - step
        # Bound it
        if sigma_new <= lower or sigma_new >= upper:
            break
        sigma = sigma_new

    # Brent fallback (bisection — simpler but reliable).
    a, b = lower, upper
    fa = _price_for(a) - market_price
    fb = _price_for(b) - market_price
    if fa * fb > 0:
        return float("nan")     # no root in bracket
    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = _price_for(c) - market_price
        if abs(fc) < tolerance or abs(b - a) < tolerance:
            return max(lower, min(upper, c))
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return max(lower, min(upper, 0.5 * (a + b)))
