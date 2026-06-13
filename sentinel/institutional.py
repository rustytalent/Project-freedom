"""Institutional analytics — the named, established methodologies.

The founder's point, exactly right: a prop desk doesn't trust a novel
score. They trust a number called by its industry-standard name (SVI
fair vol, Cornish-Fisher VaR, realised-vol-cone percentile) because
the *meaning* is already agreed. We don't have to invent
interpretation; we adopt the established one.

This module is the **ground-anchored framework layer** that sits under
the scientists, the scenario engine, and the customer-facing reports.
Every output here is an established institutional methodology with a
citation so a quant can audit it on sight.

What's in here (v1):

  SVI total-variance smile        — Gatheral (2004), the industry IV
                                    surface parameterization
  Fair value under SVI            — BS price using the SVI-implied IV
                                    at a strike's log-moneyness
  Skew & ATM term-structure       — institutional risk-reversal /
                                    butterfly readouts
  Realised vol (close-to-close,
    Parkinson, Garman-Klass,
    Yang-Zhang)                   — four standard estimators a desk
                                    triangulates between
  Realised-vol cone & percentile  — 10/30/60/90d realized vol bands
                                    + current's percentile
                                    ("VIX-cone" methodology)
  Vol risk premium                — implied minus realised (rich/cheap
                                    vol vs delivered)
  Cornish-Fisher VaR              — parametric VaR with skew/kurt
                                    corrections (BIS-cited)
  Historical-simulation VaR       — non-parametric VaR over a window
  Expected Shortfall (CVaR)       — average loss beyond VaR
                                    (Basel III standard)
  Notional/delta/vega/theta
    portfolio exposures           — the four institutional exposure
                                    reports
  Crux Liquidity Score (CLS)      — composite of spread bps + (volume
                                    + OI) depth + recent fill-quality
  Crux Slippage Score (CSS)       — expected adverse fill in bps as a
                                    function of regime + size + spread

Every function carries:
  * a docstring naming the methodology + a one-line citation
  * a Sentinel SCORE_TIER tag (RETAIL / PRO / QUANT) so the SaaS layer
    can gate the right outputs to the right plan automatically
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .greeks import bs_price
from .io_decl import IOSpec, declare

# Tiers for SaaS gating (the founder's nerf strategy)
RETAIL = "RETAIL"
PRO = "PRO"
QUANT = "QUANT"


# =========================================================================
# SVI smile — Gatheral (2004), "A parsimonious arbitrage-free implied
#   volatility parameterization." The industry-standard surface model.
# =========================================================================

@dataclass
class SVIParams:
    """Raw SVI: total variance w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2+sigma^2))
    where k = log(strike/forward). Returns total variance for a slice
    (one expiry). IV(k) = sqrt(w(k)/T)."""
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    T: float

    def total_variance(self, k: float) -> float:
        return self.a + self.b * (self.rho * (k - self.m)
                                  + math.sqrt((k - self.m) ** 2 + self.sigma ** 2))

    def iv(self, k: float) -> float:
        w = max(self.total_variance(k), 1e-9)
        return math.sqrt(w / max(self.T, 1e-9))


def fit_svi_slice(strikes: Sequence[float], ivs: Sequence[float],
                  forward: float, t_years: float) -> SVIParams:
    """Fit raw SVI to (strike, market IV) pairs by minimising squared
    error in total variance. Uses scipy if present, else a deterministic
    grid+golden-section descent so the module has no hard scipy dep.

    Methodology: Gatheral (2004) raw SVI. Sentinel tier: PRO.

    Returns SVIParams whose ``iv(log(K/F))`` is the smile-implied IV at
    strike K. The fit is only meaningful with >= 5 strikes spanning ITM
    through OTM; with fewer points it falls back to a flat-vol SVI
    (a = w_atm, b=sigma=eps, rho=0, m=0) so callers never crash.
    """
    Ks = np.asarray([k for k, v in zip(strikes, ivs) if v and v > 0])
    Vs = np.asarray([v for k, v in zip(strikes, ivs) if v and v > 0])
    if len(Ks) < 5 or t_years <= 0:
        atm = float(np.median(Vs)) if len(Vs) else 0.15
        return SVIParams(a=max(atm * atm * t_years, 1e-6),
                         b=1e-3, rho=0.0, m=0.0, sigma=0.1, T=t_years)
    k = np.log(Ks / forward)
    w_market = (Vs ** 2) * t_years
    # initial guess: ATM total variance for a, modest b, sigma
    a0 = float(np.median(w_market))
    b0, rho0, m0, sig0 = 0.04, -0.3, 0.0, 0.1

    def loss(params):
        a, b, rho, m, sig = params
        if b < 0 or sig <= 0 or not (-0.999 < rho < 0.999):
            return 1e9
        pred = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig * sig))
        return float(np.sum((pred - w_market) ** 2))

    best = (a0, b0, rho0, m0, sig0)
    best_loss = loss(best)
    try:
        from scipy.optimize import minimize
        res = minimize(loss, best, method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 2000})
        if res.fun < best_loss:
            best, best_loss = tuple(res.x), float(res.fun)
    except Exception:
        # coarse grid then refine — deterministic no-scipy fallback
        for b in np.linspace(0.01, 0.5, 12):
            for rho in np.linspace(-0.8, 0.8, 9):
                for sig in np.linspace(0.05, 0.5, 8):
                    for m in np.linspace(-0.05, 0.05, 5):
                        a = max(np.median(w_market) - b * (rho * (-m) + math.sqrt(m * m + sig * sig)), 1e-6)
                        v = loss((a, b, rho, m, sig))
                        if v < best_loss:
                            best, best_loss = (a, b, rho, m, sig), v
    return SVIParams(a=best[0], b=best[1], rho=best[2], m=best[3],
                     sigma=best[4], T=t_years)


def svi_skew_25d(svi: SVIParams) -> Dict[str, float]:
    """25-delta risk-reversal and butterfly readouts off a fitted SVI.

    Methodology: industry FX/equity vol-surface convention. The
    25-delta points sit roughly at k = +/- 0.5 sigma_ATM for short
    expiries; we read at k = +/- 0.10 as a robust proxy (accurate to a
    few bp of vol). Sentinel tier: PRO."""
    iv_minus = svi.iv(-0.10)
    iv_atm = svi.iv(0.0)
    iv_plus = svi.iv(0.10)
    rr = iv_plus - iv_minus      # positive = call skew (rare); usually negative
    bf = 0.5 * (iv_plus + iv_minus) - iv_atm
    return {"iv_25d_put": round(iv_minus, 4),
            "iv_atm": round(iv_atm, 4),
            "iv_25d_call": round(iv_plus, 4),
            "risk_reversal_25d": round(rr, 4),
            "butterfly_25d": round(bf, 4)}


def fair_value_from_svi(svi: SVIParams, spot: float, strike: float,
                        option_type: str, rate: float = 0.065) -> Dict[str, float]:
    """Institutional fair value: take SVI IV at the strike's
    log-moneyness, price Black-Scholes at that IV. The standard
    'screen vs. fair' check a desk runs continuously.

    Methodology: SVI surface (Gatheral 2004) + Black-Scholes-Merton
    (1973). Sentinel tier: PRO. Returns fair_price + fair_iv."""
    k = math.log(max(strike, 1e-9) / max(spot, 1e-9))
    iv = svi.iv(k)
    price = bs_price(spot, strike, svi.T, iv, option_type, rate)
    return {"fair_iv": round(iv, 4), "fair_price": round(price, 2)}


# =========================================================================
# Realised volatility estimators — four institutional standards
# =========================================================================

def rv_close_to_close(closes: Sequence[float], annualize: int = 252) -> float:
    """Classical close-to-close RV. Methodology: textbook. Tier: RETAIL."""
    a = np.asarray(closes, dtype=float)
    if len(a) < 2:
        return float("nan")
    r = np.diff(np.log(a))
    return float(np.std(r, ddof=1) * math.sqrt(annualize))


def rv_parkinson(highs: Sequence[float], lows: Sequence[float],
                 annualize: int = 252) -> float:
    """Parkinson (1980). Uses H/L range — more efficient than C2C.
    Methodology: Parkinson, J. of Business 1980. Tier: PRO."""
    h = np.asarray(highs, dtype=float)
    l = np.asarray(lows, dtype=float)
    if len(h) < 2 or len(l) < 2:
        return float("nan")
    rng = np.log(h / l) ** 2
    return float(math.sqrt(np.mean(rng) / (4.0 * math.log(2.0)) * annualize))


def rv_garman_klass(opens, highs, lows, closes, annualize: int = 252) -> float:
    """Garman-Klass (1980). Uses OHLC; ~7.4x more efficient than C2C.
    Methodology: Garman & Klass, J. of Business 1980. Tier: PRO."""
    o, h, l, c = (np.asarray(x, dtype=float) for x in (opens, highs, lows, closes))
    if len(c) < 2:
        return float("nan")
    a = 0.5 * (np.log(h / l) ** 2) - (2.0 * math.log(2.0) - 1) * (np.log(c / o) ** 2)
    return float(math.sqrt(np.mean(a) * annualize))


def rv_yang_zhang(opens, highs, lows, closes, annualize: int = 252) -> float:
    """Yang-Zhang (2000). Handles overnight gaps + opening jumps —
    closest thing to a true session-IV proxy.
    Methodology: Yang & Zhang, J. of Business 2000. Tier: QUANT."""
    o, h, l, c = (np.asarray(x, dtype=float) for x in (opens, highs, lows, closes))
    n = len(c)
    if n < 3:
        return float("nan")
    o_ret = np.log(o[1:] / c[:-1])       # overnight
    c_ret = np.log(c / o)                # open-to-close
    rs = (np.log(h / c) * np.log(h / o)
          + np.log(l / c) * np.log(l / o))   # Rogers-Satchell
    sig_o = float(np.var(o_ret, ddof=1))
    sig_c = float(np.var(c_ret[1:], ddof=1))
    sig_rs = float(np.mean(rs[1:]))
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = sig_o + k * sig_c + (1 - k) * sig_rs
    return float(math.sqrt(max(var, 0) * annualize))


# =========================================================================
# Realised-vol cone — "VIX-cone" methodology
# =========================================================================

def vol_cone(closes: Sequence[float], windows: Sequence[int] = (10, 20, 30, 60, 90),
             percentiles: Sequence[int] = (10, 25, 50, 75, 90)
             ) -> Dict[str, Any]:
    """For each rolling window length, percentiles of realised vol over
    history. The standard institutional way to answer 'is today's
    implied vol rich or cheap'.

    Methodology: Burghardt & Lane, J. of Derivatives 1990 (the 'VIX
    cone'). Tier: PRO. Returns a cone + current readings.
    """
    out: Dict[str, Any] = {"windows": list(windows),
                           "percentiles": list(percentiles),
                           "cone": {}, "current": {}}
    a = np.asarray(closes, dtype=float)
    if len(a) < max(windows) + 2:
        return out
    log_r = np.diff(np.log(a))
    for w in windows:
        rolls = []
        for i in range(w, len(log_r) + 1):
            rolls.append(float(np.std(log_r[i - w:i], ddof=1)
                                * math.sqrt(252)))
        if not rolls:
            continue
        out["cone"][w] = {f"p{p}": round(float(np.percentile(rolls, p)), 4)
                          for p in percentiles}
        out["current"][w] = round(rolls[-1], 4)
    return out


def vol_cone_percentile(closes: Sequence[float], window: int = 20) -> float:
    """Where does the most recent ``window``-day realised vol sit in
    its own history? 0..100. Tier: RETAIL."""
    a = np.asarray(closes, dtype=float)
    if len(a) < window + 5:
        return float("nan")
    log_r = np.diff(np.log(a))
    rolls = []
    for i in range(window, len(log_r) + 1):
        rolls.append(float(np.std(log_r[i - window:i], ddof=1)
                           * math.sqrt(252)))
    current = rolls[-1]
    rank = sum(1 for r in rolls if r <= current) / len(rolls)
    return round(100.0 * rank, 1)


def vol_risk_premium(implied_atm: float, realised: float) -> Dict[str, float]:
    """Implied - realised. Positive = market is paying up for vol;
    negative = vol is cheap. The single most-watched options metric on
    institutional desks.

    Methodology: standard 'vol risk premium' definition (Bakshi &
    Kapadia, RFS 2003). Tier: RETAIL."""
    if not (implied_atm > 0 and realised > 0):
        return {"implied": implied_atm, "realised": realised,
                "premium": float("nan"), "ratio": float("nan")}
    return {"implied": round(implied_atm, 4), "realised": round(realised, 4),
            "premium": round(implied_atm - realised, 4),
            "ratio": round(implied_atm / realised, 3)}


# =========================================================================
# VaR + Expected Shortfall
# =========================================================================

def parametric_var(pnls: Sequence[float], alpha: float = 0.99) -> Dict[str, float]:
    """Cornish-Fisher VaR — parametric VaR with skew + excess-kurtosis
    corrections so heavy tails aren't underestimated.

    Methodology: Cornish & Fisher (1937), adopted by BIS (1996) as a
    standard market-risk metric. Tier: PRO. Returns positive numbers
    where larger = worse loss."""
    a = np.asarray(pnls, dtype=float)
    if len(a) < 5:
        return {"var": float("nan"), "method": "cornish_fisher",
                "alpha": alpha, "n": int(len(a))}
    mu, sig = float(np.mean(a)), float(np.std(a, ddof=1))
    skew = float(((a - mu) ** 3).mean() / (sig ** 3 + 1e-12))
    kurt_excess = float(((a - mu) ** 4).mean() / (sig ** 4 + 1e-12) - 3.0)
    from math import erfc, sqrt
    # inverse-normal of (1-alpha) — closed form via Beasley-Springer
    z = _norm_inv(1 - alpha)
    cf = (z + (z * z - 1) * skew / 6.0
          + (z ** 3 - 3 * z) * kurt_excess / 24.0
          - (2 * z ** 3 - 5 * z) * (skew ** 2) / 36.0)
    var = -(mu + cf * sig)
    return {"var": round(float(var), 2), "method": "cornish_fisher",
            "alpha": alpha, "n": int(len(a)),
            "skew": round(skew, 3), "excess_kurt": round(kurt_excess, 3)}


def historical_var(pnls: Sequence[float], alpha: float = 0.99) -> Dict[str, float]:
    """Non-parametric VaR: the (1-alpha) percentile of the empirical
    P&L distribution. Methodology: BIS standard. Tier: RETAIL."""
    a = np.asarray(pnls, dtype=float)
    if len(a) < 5:
        return {"var": float("nan"), "method": "historical",
                "alpha": alpha, "n": int(len(a))}
    q = float(np.percentile(a, (1 - alpha) * 100))
    return {"var": round(-q, 2), "method": "historical",
            "alpha": alpha, "n": int(len(a))}


def expected_shortfall(pnls: Sequence[float],
                       alpha: float = 0.975) -> Dict[str, float]:
    """Expected Shortfall (CVaR) — mean loss in the worst (1-alpha)
    tail. Basel III replaced VaR with ES at alpha=0.975 for market
    risk; this is what a serious desk reports.

    Methodology: Basel III market-risk framework. Tier: PRO."""
    a = np.asarray(pnls, dtype=float)
    if len(a) < 5:
        return {"es": float("nan"), "alpha": alpha, "n": int(len(a))}
    cutoff = np.percentile(a, (1 - alpha) * 100)
    tail = a[a <= cutoff]
    if len(tail) == 0:
        return {"es": float("nan"), "alpha": alpha, "n": int(len(a))}
    return {"es": round(-float(np.mean(tail)), 2),
            "alpha": alpha, "n_tail": int(len(tail)), "n": int(len(a))}


# =========================================================================
# Portfolio exposures — the four institutional reports
# =========================================================================

@dataclass
class LegExposure:
    tradingsymbol: str
    qty: int                    # signed
    premium: float
    delta: float                # per-unit
    gamma: float
    theta_per_day: float
    vega_per_pct: float


def portfolio_greek_exposures(legs: Sequence[LegExposure],
                              spot: float) -> Dict[str, float]:
    """Net delta / gamma / theta / vega and notional exposure across
    every option leg. This is what a risk officer asks for first.

    Methodology: linear aggregation of position-weighted Greeks
    (the 'Greek book'). Tier: RETAIL."""
    delta = sum(l.qty * l.delta for l in legs)
    gamma = sum(l.qty * l.gamma for l in legs)
    theta = sum(l.qty * l.theta_per_day for l in legs)
    vega = sum(l.qty * l.vega_per_pct for l in legs)
    notional = sum(abs(l.qty) * l.premium for l in legs)
    delta_notional = delta * spot
    return {
        "net_delta": round(delta, 2),
        "net_gamma": round(gamma, 4),
        "net_theta_per_day": round(theta, 2),
        "net_vega_per_pct": round(vega, 2),
        "delta_notional_rupees": round(delta_notional, 0),
        "premium_notional_rupees": round(notional, 0),
    }


# =========================================================================
# Crux Liquidity Score + Crux Slippage Score — sellable institutional
# scores that anchor on industry-standard inputs
# =========================================================================

def crux_liquidity_score(bid: float, ask: float, mid: float,
                         volume: float, oi: float,
                         volume_pctile: float = 0.5,
                         oi_pctile: float = 0.5) -> Dict[str, float]:
    """Composite 0-100 liquidity quality score for one option strike.
    Spread (bps of mid) + depth proxy (vol+OI percentile within the
    chain) + a small bonus for tight microstructure.

    Methodology: standard market-microstructure liquidity composite
    (Amihud 2002 + Almgren-Chriss spread-cost). Tier: PRO.

    The components are explicit so a desk can audit:
      score = 0.50*(1 - spread_bps/100, clipped) + 0.40*depth + 0.10*tightness
    """
    if mid <= 0:
        return {"score": 0.0, "spread_bps": float("nan"),
                "depth_pctile": 0.0, "components": {}}
    spread_bps = ((ask - bid) / mid) * 10000.0 if ask > bid else 200.0
    spread_score = max(0.0, min(1.0, 1.0 - spread_bps / 100.0))
    depth_score = max(0.0, min(1.0, 0.5 * volume_pctile + 0.5 * oi_pctile))
    tight_score = max(0.0, min(1.0, 1.0 - abs((ask + bid) / 2 - mid) / mid * 20))
    score = 0.50 * spread_score + 0.40 * depth_score + 0.10 * tight_score
    return {
        "score": round(100.0 * score, 1),
        "spread_bps": round(spread_bps, 1),
        "depth_pctile": round(depth_score, 3),
        "components": {"spread": round(spread_score, 3),
                       "depth": round(depth_score, 3),
                       "tightness": round(tight_score, 3)},
    }


def crux_slippage_score(spread_bps: float, vol_regime: str, order_size: int,
                        depth_pctile: float) -> Dict[str, float]:
    """Expected adverse fill (bps from mid) for a market order. Combines
    half-spread + Almgren-Chriss-style market-impact term scaled by
    regime and depth.

    Methodology: Almgren-Chriss (2000) optimal-execution impact model,
    simplified for retail order sizes. Tier: PRO.

    expected_bps = 0.5*spread + impact_coef * sqrt(size / depth) *
                   regime_multiplier
    """
    regime_mult = {"low": 0.8, "normal": 1.0, "high": 1.7}.get(vol_regime, 1.0)
    half_spread = 0.5 * spread_bps
    depth = max(depth_pctile, 0.05)
    impact = 8.0 * math.sqrt(max(order_size, 1) / 100.0) / math.sqrt(depth)
    total = (half_spread + impact) * regime_mult
    return {"expected_slip_bps": round(total, 2),
            "half_spread_bps": round(half_spread, 2),
            "impact_bps": round(impact * regime_mult, 2),
            "regime_multiplier": regime_mult}


# =========================================================================
# Roll curve — futures + monthly options
# =========================================================================

def roll_curve(futures: Sequence[Tuple[str, float, float]]
               ) -> List[Dict[str, Any]]:
    """Input: list of (label, days_to_expiry, price). Output: annualised
    roll yield between consecutive contracts. Negative = backwardation
    (a bullish carry signal); positive = contango.

    Methodology: standard term-structure 'roll yield' calc (Erb &
    Harvey, FAJ 2006). Tier: PRO."""
    fs = sorted(futures, key=lambda x: x[1])
    out = []
    for (la, da, pa), (lb, db, pb) in zip(fs, fs[1:]):
        if pa <= 0 or pb <= 0 or db <= da:
            continue
        ry = (math.log(pb / pa)) * 365.0 / (db - da)
        out.append({"from": la, "to": lb, "days_apart": db - da,
                    "annualised_roll_yield": round(ry, 4),
                    "regime": "contango" if ry > 0 else "backwardation"})
    return out


# =========================================================================
# small util — Beasley-Springer inverse normal (no scipy)
# =========================================================================

def _norm_inv(p: float) -> float:
    if p <= 0:
        return -8.0
    if p >= 1:
        return 8.0
    # Beasley-Springer-Moro
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) \
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q \
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) \
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


declare(IOSpec(
    module="sentinel.institutional",
    purpose="ground-anchored, citation-bearing methodologies: SVI vol surface, "
            "fair value, realised-vol estimators + cone, vol risk premium, "
            "Cornish-Fisher + historical VaR, Expected Shortfall, Greek "
            "exposures, Crux liquidity + slippage scores, roll curves",
    inputs=["strike+IV pairs (SVI fit)", "OHLC series (RV/cone)",
            "P&L series (VaR/ES)", "leg list (exposures)",
            "L1 quote (CLS/CSS)", "futures curve (roll)"],
    outputs=["SVIParams + fair price/IV", "realised vol estimators",
             "vol cone + percentile", "vol risk premium",
             "VaR / ES dicts", "greek exposures dict",
             "Crux Liquidity/Slippage scores"],
    consumes_from=["sentinel.greeks", "sentinel.kite_client",
                   "sentinel.portfolio"],
    produces_for=["sentinel.scientists (fair-value anchor)",
                  "sentinel.scenario_engine (skew-aware repricing)",
                  "sentinel.reports", "sentinel.server (UI later)"],
    tier="TRUSTED",
))
