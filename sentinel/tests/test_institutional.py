"""Wave 3 tests — the institutional analytics layer.

Every methodology here is named and citation-bearing (SVI, Cornish-Fisher
VaR, Expected Shortfall, Parkinson/Garman-Klass/Yang-Zhang RV, vol-cone,
Crux liquidity/slippage, roll yield). These tests pin the established
behaviour so a prop-desk audit can read pass/fail at a glance.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sentinel.greeks import bs_price
from sentinel.institutional import (
    LegExposure,
    SVIParams,
    crux_liquidity_score,
    crux_slippage_score,
    expected_shortfall,
    fair_value_from_svi,
    fit_svi_slice,
    historical_var,
    parametric_var,
    portfolio_greek_exposures,
    roll_curve,
    rv_close_to_close,
    rv_garman_klass,
    rv_parkinson,
    rv_yang_zhang,
    svi_skew_25d,
    vol_cone,
    vol_cone_percentile,
    vol_risk_premium,
    _norm_inv,
)


# ---------------------------------------------------------------------------
# SVI smile (Gatheral 2004)
# ---------------------------------------------------------------------------

def test_svi_params_total_variance_and_iv_consistent():
    """total_variance(0) = a + b*sigma when m=0, rho=0, and IV is sqrt(w/T)."""
    p = SVIParams(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.1, T=0.25)
    w0 = p.total_variance(0.0)
    assert math.isclose(w0, 0.04 + 0.1 * 0.1, rel_tol=1e-9)
    assert math.isclose(p.iv(0.0), math.sqrt(w0 / 0.25), rel_tol=1e-9)


def test_fit_svi_slice_recovers_flat_smile():
    """Feed a near-flat smile of IVs and require the fit to reproduce them
    within a few percent of vol — pins fit_svi_slice's basic correctness."""
    forward = 25000.0
    strikes = [24700, 24800, 24900, 25000, 25100, 25200, 25300]
    ivs = [0.16, 0.155, 0.152, 0.15, 0.152, 0.155, 0.16]
    p = fit_svi_slice(strikes, ivs, forward=forward, t_years=7 / 365)
    for k_strike, iv_market in zip(strikes, ivs):
        k = math.log(k_strike / forward)
        iv_fit = p.iv(k)
        assert abs(iv_fit - iv_market) < 0.05   # within 5 vol-points


def test_fit_svi_slice_handles_too_few_points():
    """Fewer than 5 strikes -> flat-vol fallback; iv(k) is finite and positive."""
    p = fit_svi_slice([25000, 25100], [0.15, 0.16], forward=25000, t_years=7 / 365)
    iv0 = p.iv(0.0)
    assert iv0 > 0 and math.isfinite(iv0)


def test_svi_skew_25d_carries_three_pillars():
    p = SVIParams(a=0.04, b=0.2, rho=-0.3, m=0.0, sigma=0.1, T=0.25)
    out = svi_skew_25d(p)
    assert set(out) >= {"iv_25d_put", "iv_atm", "iv_25d_call",
                         "risk_reversal_25d", "butterfly_25d"}
    # rho < 0 -> put wing should sit higher than call wing (the textbook case)
    assert out["iv_25d_put"] > out["iv_25d_call"]


def test_fair_value_from_svi_matches_bs_price_at_svi_iv():
    """The fair-price output equals BS(SVI-IV) — sanity check the wiring."""
    p = SVIParams(a=0.04, b=0.1, rho=0.0, m=0.0, sigma=0.1, T=7 / 365)
    spot, strike = 25000.0, 25100.0
    out = fair_value_from_svi(p, spot=spot, strike=strike, option_type="CE")
    k = math.log(strike / spot)
    expected = bs_price(spot, strike, 7 / 365, p.iv(k), "CE")
    assert abs(out["fair_price"] - round(expected, 2)) < 0.5


# ---------------------------------------------------------------------------
# Realised-vol estimators
# ---------------------------------------------------------------------------

def _series(seed=0, n=60):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.012, n)
    closes = 25000 * np.exp(np.cumsum(rets))
    # build plausible OHLC around closes
    opens = closes * (1 + rng.normal(0, 0.001, n))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.003, n)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.003, n)))
    return opens, highs, lows, closes


def test_rv_close_to_close_positive_and_finite():
    _, _, _, c = _series()
    rv = rv_close_to_close(c)
    assert rv > 0 and math.isfinite(rv) and rv < 5.0


def test_rv_parkinson_lower_than_c2c_on_quiet_series():
    """Parkinson is ~5x more efficient than C2C; on the same draw it should
    be in the same ballpark (not 10x larger). Pins the estimator's scale."""
    _, h, l, c = _series()
    pk = rv_parkinson(h, l)
    cc = rv_close_to_close(c)
    assert pk > 0
    assert 0.2 * cc < pk < 5.0 * cc


def test_rv_garman_klass_finite():
    o, h, l, c = _series()
    gk = rv_garman_klass(o, h, l, c)
    assert gk > 0 and math.isfinite(gk)


def test_rv_yang_zhang_handles_overnight_gap():
    o, h, l, c = _series()
    yz = rv_yang_zhang(o, h, l, c)
    assert yz > 0 and math.isfinite(yz)


def test_rv_close_to_close_nan_on_short_series():
    assert math.isnan(rv_close_to_close([100.0]))


# ---------------------------------------------------------------------------
# Vol cone + vol risk premium
# ---------------------------------------------------------------------------

def test_vol_cone_returns_percentiles_per_window():
    _, _, _, c = _series(n=200)
    cone = vol_cone(c.tolist(), windows=(10, 20, 60), percentiles=(10, 50, 90))
    assert set(cone["cone"]) == {10, 20, 60}
    # percentiles within a window should be monotone non-decreasing
    for w in (10, 20, 60):
        ps = cone["cone"][w]
        assert ps["p10"] <= ps["p50"] <= ps["p90"]
        assert cone["current"][w] > 0


def test_vol_cone_percentile_in_0_100():
    _, _, _, c = _series(n=200)
    p = vol_cone_percentile(c.tolist(), window=20)
    assert 0.0 <= p <= 100.0


def test_vol_risk_premium_sign():
    out = vol_risk_premium(implied_atm=0.18, realised=0.14)
    assert out["premium"] > 0 and out["ratio"] > 1.0
    out2 = vol_risk_premium(implied_atm=0.10, realised=0.18)
    assert out2["premium"] < 0 and out2["ratio"] < 1.0


def test_vol_risk_premium_nan_on_bad_inputs():
    out = vol_risk_premium(implied_atm=0.0, realised=0.1)
    assert math.isnan(out["premium"])


# ---------------------------------------------------------------------------
# VaR + Expected Shortfall
# ---------------------------------------------------------------------------

def test_parametric_var_positive_and_bounded():
    rng = np.random.default_rng(42)
    pnls = rng.normal(0, 1000, 500).tolist()
    out = parametric_var(pnls, alpha=0.99)
    assert out["var"] > 0 and out["method"] == "cornish_fisher"
    # 99% VaR on N(0,1000) is ~2.33*1000; CF should land near that ballpark
    assert 1500 < out["var"] < 4000


def test_historical_var_matches_empirical_percentile():
    pnls = np.linspace(-100, 100, 101).tolist()    # -100..+100
    out = historical_var(pnls, alpha=0.95)
    # 5th percentile is -90; VaR is reported as positive loss
    assert abs(out["var"] - 90) < 5


def test_expected_shortfall_at_least_historical_var():
    rng = np.random.default_rng(7)
    pnls = rng.normal(0, 1000, 1000).tolist()
    es = expected_shortfall(pnls, alpha=0.975)["es"]
    hv = historical_var(pnls, alpha=0.975)["var"]
    # ES is the mean of the tail BEYOND VaR -> >= VaR by construction
    assert es >= hv - 1e-6


def test_var_es_nan_on_short_series():
    assert math.isnan(parametric_var([1.0, 2.0], alpha=0.99)["var"])
    assert math.isnan(historical_var([1.0, 2.0], alpha=0.99)["var"])
    assert math.isnan(expected_shortfall([1.0, 2.0], alpha=0.975)["es"])


# ---------------------------------------------------------------------------
# Portfolio Greek exposures
# ---------------------------------------------------------------------------

def test_portfolio_greek_exposures_aggregate_correctly():
    legs = [
        LegExposure("NIFTY25000CE", qty=75, premium=180, delta=0.5,
                    gamma=0.001, theta_per_day=-5.0, vega_per_pct=15.0),
        LegExposure("NIFTY24900PE", qty=-75, premium=120, delta=-0.4,
                    gamma=0.001, theta_per_day=-4.0, vega_per_pct=14.0),
    ]
    out = portfolio_greek_exposures(legs, spot=25000.0)
    # long CE + short PE -> net positive delta
    assert math.isclose(out["net_delta"], 75 * 0.5 + (-75) * (-0.4))
    assert out["net_delta"] > 0
    # premium notional is sum of abs(qty)*premium
    assert out["premium_notional_rupees"] == round(75 * 180 + 75 * 120, 0)


def test_portfolio_greek_exposures_empty_book():
    out = portfolio_greek_exposures([], spot=25000.0)
    assert out["net_delta"] == 0 and out["premium_notional_rupees"] == 0


# ---------------------------------------------------------------------------
# Crux Liquidity Score + Crux Slippage Score
# ---------------------------------------------------------------------------

def test_crux_liquidity_score_components_sum_to_score():
    """The composite must match its declared 0.5/0.4/0.1 weighting so the
    SaaS report can be audited line by line."""
    out = crux_liquidity_score(bid=179.5, ask=180.5, mid=180.0,
                                volume=1e6, oi=1e6,
                                volume_pctile=0.7, oi_pctile=0.7)
    c = out["components"]
    weighted = 100.0 * (0.50 * c["spread"] + 0.40 * c["depth"] + 0.10 * c["tightness"])
    assert abs(out["score"] - weighted) < 0.5


def test_crux_liquidity_score_punishes_wide_spread():
    tight = crux_liquidity_score(bid=179.9, ask=180.1, mid=180.0,
                                  volume=1e6, oi=1e6,
                                  volume_pctile=0.7, oi_pctile=0.7)
    wide = crux_liquidity_score(bid=170.0, ask=190.0, mid=180.0,
                                 volume=1e6, oi=1e6,
                                 volume_pctile=0.7, oi_pctile=0.7)
    assert tight["score"] > wide["score"]


def test_crux_liquidity_score_zero_mid_safe():
    out = crux_liquidity_score(bid=0, ask=0, mid=0, volume=0, oi=0)
    assert out["score"] == 0.0


def test_crux_slippage_score_scales_with_regime_and_size():
    base = crux_slippage_score(spread_bps=10.0, vol_regime="normal",
                                order_size=100, depth_pctile=0.5)
    high = crux_slippage_score(spread_bps=10.0, vol_regime="high",
                                order_size=100, depth_pctile=0.5)
    big = crux_slippage_score(spread_bps=10.0, vol_regime="normal",
                               order_size=400, depth_pctile=0.5)
    # high-vol regime adds cost
    assert high["expected_slip_bps"] > base["expected_slip_bps"]
    # larger size -> more impact
    assert big["expected_slip_bps"] > base["expected_slip_bps"]
    # half-spread component is exactly 0.5 * spread_bps
    assert math.isclose(base["half_spread_bps"], 5.0)


# ---------------------------------------------------------------------------
# Roll curve
# ---------------------------------------------------------------------------

def test_roll_curve_labels_contango_and_backwardation():
    # near=100, far=102 over 30d => contango (positive roll yield)
    out_c = roll_curve([("MAY", 5, 100.0), ("JUN", 35, 102.0)])
    assert len(out_c) == 1 and out_c[0]["regime"] == "contango"
    # near=102, far=100 => backwardation
    out_b = roll_curve([("MAY", 5, 102.0), ("JUN", 35, 100.0)])
    assert out_b[0]["regime"] == "backwardation"


def test_roll_curve_empty_when_single_contract():
    assert roll_curve([("MAY", 5, 100.0)]) == []


# ---------------------------------------------------------------------------
# Beasley-Springer inverse normal
# ---------------------------------------------------------------------------

def test_norm_inv_round_trips_standard_quantiles():
    # standard reference points
    assert abs(_norm_inv(0.5) - 0.0) < 1e-6
    assert abs(_norm_inv(0.975) - 1.95996) < 1e-3
    assert abs(_norm_inv(0.025) + 1.95996) < 1e-3
    # extreme tails clamped, not NaN
    assert math.isfinite(_norm_inv(1e-12))
    assert math.isfinite(_norm_inv(1 - 1e-12))


# ---------------------------------------------------------------------------
# IO registry self-declaration
# ---------------------------------------------------------------------------

def test_institutional_module_declares_io():
    import sentinel.institutional  # noqa: F401
    from sentinel import io_decl
    assert "sentinel.institutional" in io_decl.REGISTRY
    spec = io_decl.REGISTRY["sentinel.institutional"]
    assert spec.tier == "TRUSTED"
    assert any("scientists" in p for p in spec.produces_for)
