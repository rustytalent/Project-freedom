"""Tests for executor_v4.pricing — Black-Scholes, IV surface, strategy Greeks."""
from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from liqpool.research.belief.executor_v4.pricing import (
    IVSurfaceConfig,
    IVSurfaceFitter,
    bs_call_price,
    bs_put_price,
    call_greeks,
    compute_strategy_greeks,
    implied_volatility,
    norm_cdf,
    norm_pdf,
    put_greeks,
)


# ── Normal CDF / PDF ───────────────────────────────────────────────


def test_norm_cdf_known_values():
    assert abs(norm_cdf(0) - 0.5) < 1e-10
    assert abs(norm_cdf(1.96) - 0.975) < 1e-4
    assert abs(norm_cdf(-1.96) - 0.025) < 1e-4
    # Deep tails
    assert norm_cdf(5.0) > 0.999999
    assert norm_cdf(-5.0) < 1e-6


def test_norm_pdf_peaked_at_zero():
    assert abs(norm_pdf(0) - 1.0 / math.sqrt(2 * math.pi)) < 1e-10
    assert norm_pdf(0) > norm_pdf(1.0)
    assert norm_pdf(1.0) > norm_pdf(2.0)


# ── Black-Scholes price ────────────────────────────────────────────


def test_atm_call_put_parity():
    spot = 23000.0; strike = 23000.0; T = 7/365
    sigma = 0.18
    c = bs_call_price(spot=spot, strike=strike, time_to_expiry=T, sigma=sigma)
    p = bs_put_price(spot=spot, strike=strike, time_to_expiry=T, sigma=sigma)
    # Put-call parity: C - P = S*e^(-qT) - K*e^(-rT)
    r = 0.065; q = 0.012
    expected = spot * math.exp(-q * T) - strike * math.exp(-r * T)
    assert abs((c - p) - expected) < 0.01


def test_otm_call_value_decreases_with_strike():
    spot = 23000.0; T = 7/365; sigma = 0.18
    c1 = bs_call_price(spot=spot, strike=23100.0,
                         time_to_expiry=T, sigma=sigma)
    c2 = bs_call_price(spot=spot, strike=23200.0,
                         time_to_expiry=T, sigma=sigma)
    c3 = bs_call_price(spot=spot, strike=23500.0,
                         time_to_expiry=T, sigma=sigma)
    assert c1 > c2 > c3
    assert c3 >= 0


def test_zero_time_to_expiry_returns_intrinsic():
    c = bs_call_price(spot=23000.0, strike=22950.0,
                        time_to_expiry=0.0, sigma=0.18)
    assert c == 50.0
    p = bs_put_price(spot=22950.0, strike=23000.0,
                       time_to_expiry=0.0, sigma=0.18)
    assert p == 50.0


def test_zero_vol_returns_discounted_intrinsic_under_carry():
    """With sigma=0, ITM call price = forward - PV(strike).
    Under positive carry this is >= raw intrinsic (50)."""
    c = bs_call_price(spot=23000.0, strike=22950.0,
                        time_to_expiry=7/365, sigma=0.0)
    assert c > 0
    # Discounted-forward intrinsic is somewhat above raw 50 due to carry.
    assert 50.0 <= c <= 100.0


# ── Greeks ─────────────────────────────────────────────────────────


def test_atm_call_delta_around_half():
    g = call_greeks(spot=23000.0, strike=23000.0,
                      time_to_expiry=7/365, sigma=0.18)
    assert 0.40 < g.delta < 0.60


def test_atm_put_delta_negative():
    g = put_greeks(spot=23000.0, strike=23000.0,
                     time_to_expiry=7/365, sigma=0.18)
    assert -0.60 < g.delta < -0.40


def test_otm_call_delta_smaller_than_atm():
    atm = call_greeks(spot=23000.0, strike=23000.0,
                        time_to_expiry=7/365, sigma=0.18)
    otm = call_greeks(spot=23000.0, strike=23200.0,
                        time_to_expiry=7/365, sigma=0.18)
    assert otm.delta < atm.delta
    assert 0 < otm.delta < 0.40


def test_long_call_theta_negative():
    g = call_greeks(spot=23000.0, strike=23000.0,
                      time_to_expiry=7/365, sigma=0.18)
    assert g.theta < 0


def test_call_and_put_gamma_equal():
    """Gamma is the same for calls and puts at the same strike/expiry."""
    c = call_greeks(spot=23000.0, strike=23100.0,
                      time_to_expiry=7/365, sigma=0.18)
    p = put_greeks(spot=23000.0, strike=23100.0,
                     time_to_expiry=7/365, sigma=0.18)
    assert abs(c.gamma - p.gamma) < 1e-10


def test_greeks_serializable():
    g = call_greeks(spot=23000.0, strike=23000.0,
                      time_to_expiry=7/365, sigma=0.18)
    import json
    json.dumps(g.to_dict())


def test_greeks_invalid_inputs_return_nan():
    g = call_greeks(spot=0.0, strike=23000.0,
                      time_to_expiry=7/365, sigma=0.18)
    assert math.isnan(g.delta)


# ── Implied volatility inversion ───────────────────────────────────


def test_iv_roundtrip_call():
    spot = 23000.0; strike = 23100.0; T = 7/365
    true_iv = 0.22
    price = bs_call_price(spot=spot, strike=strike, time_to_expiry=T,
                            sigma=true_iv)
    iv_back = implied_volatility(
        market_price=price, side="CE", spot=spot,
        strike=strike, time_to_expiry=T,
    )
    assert abs(iv_back - true_iv) < 1e-3


def test_iv_roundtrip_put():
    spot = 23000.0; strike = 22900.0; T = 7/365
    true_iv = 0.25
    price = bs_put_price(spot=spot, strike=strike, time_to_expiry=T,
                           sigma=true_iv)
    iv_back = implied_volatility(
        market_price=price, side="PE", spot=spot,
        strike=strike, time_to_expiry=T,
    )
    assert abs(iv_back - true_iv) < 1e-3


def test_iv_below_intrinsic_returns_nan():
    iv = implied_volatility(
        market_price=0.5, side="CE", spot=23000.0,
        strike=22000.0,    # deep ITM: intrinsic ~1000
        time_to_expiry=7/365,
    )
    assert math.isnan(iv)


def test_iv_handles_deep_otm():
    """Deep OTM options have tiny prices but IV inversion should still work."""
    spot = 23000.0; strike = 24000.0; T = 7/365
    true_iv = 0.30
    price = bs_call_price(spot=spot, strike=strike, time_to_expiry=T,
                            sigma=true_iv)
    if price < 0.05:
        pytest.skip("price too small for stable inversion")
    iv_back = implied_volatility(
        market_price=price, side="CE", spot=spot,
        strike=strike, time_to_expiry=T,
    )
    assert abs(iv_back - true_iv) < 0.01


# ── IV surface ─────────────────────────────────────────────────────


def _synthetic_observations(*, spot=23000.0, T=7/365,
                              atm_iv=0.18, skew=-0.5, curvature=2.0):
    """Generate 22 synthetic option prices with a known smile."""
    obs = []
    for level in range(-5, 6):
        strike = 23000.0 + 50 * level
        k = math.log(strike / spot)
        iv = max(0.05, atm_iv + skew * k + curvature * k * k)
        c = bs_call_price(spot=spot, strike=strike, time_to_expiry=T, sigma=iv)
        p = bs_put_price(spot=spot, strike=strike, time_to_expiry=T, sigma=iv)
        if c > 0.5:
            obs.append((strike, c, "CE"))
        if p > 0.5:
            obs.append((strike, p, "PE"))
    return obs


def test_iv_surface_fits_synthetic_smile():
    fitter = IVSurfaceFitter()
    obs = _synthetic_observations(atm_iv=0.18, skew=-0.5, curvature=2.0)
    surf = fitter.fit(observations=obs, spot=23000.0, time_to_expiry=7/365)
    # ATM IV should be close to 0.18.
    assert abs(surf.atm_iv - 0.18) < 0.02
    # Should pick polynomial or svi (we have enough data).
    assert surf.fit_method in ("polynomial", "svi")
    assert surf.n_clean_observations >= 16
    assert surf.confidence > 0.5


def test_iv_surface_handles_empty_observations():
    fitter = IVSurfaceFitter()
    surf = fitter.fit(observations=[], spot=23000.0, time_to_expiry=7/365)
    assert surf.fit_method == "flat"
    assert surf.confidence == 0.0


def test_iv_surface_iv_at_strike_extrapolates():
    fitter = IVSurfaceFitter()
    obs = _synthetic_observations()
    surf = fitter.fit(observations=obs, spot=23000.0, time_to_expiry=7/365)
    # IV at ATM matches atm_iv property.
    atm = surf.iv_at_strike(23000.0)
    assert abs(atm - surf.atm_iv) < 0.005


def test_iv_surface_serializable():
    fitter = IVSurfaceFitter()
    obs = _synthetic_observations()
    surf = fitter.fit(observations=obs, spot=23000.0, time_to_expiry=7/365)
    import json
    json.dumps(surf.to_dict())


# ── Strategy Greeks ─────────────────────────────────────────────────


@dataclass(frozen=True)
class _FakeLeg:
    contract_side: str
    strike_price: float
    direction: int
    lots: int


def test_compute_strategy_greeks_long_ce_at_atm():
    leg = _FakeLeg("CE", 23000.0, +1, 1)
    profile = compute_strategy_greeks(
        legs=[leg], spot=23000.0, time_to_expiry=7/365,
        lot_size=65, fallback_iv=0.18,
    )
    # Long call → positive delta, vega; negative theta.
    assert profile.net_delta > 0
    assert profile.net_vega > 0
    assert profile.net_theta_per_day < 0


def test_compute_strategy_greeks_iron_condor_negative_vega():
    """Iron condor (short OTM strangle, long further OTM) is net short vega."""
    legs = [
        _FakeLeg("CE", 23100.0, -1, 1),   # short OTM CE
        _FakeLeg("CE", 23200.0, +1, 1),   # long further OTM CE
        _FakeLeg("PE", 22900.0, -1, 1),   # short OTM PE
        _FakeLeg("PE", 22800.0, +1, 1),   # long further OTM PE
    ]
    profile = compute_strategy_greeks(
        legs=legs, spot=23000.0, time_to_expiry=7/365,
        lot_size=65, fallback_iv=0.18,
    )
    assert profile.net_vega < 0


def test_compute_strategy_greeks_uses_iv_surface_when_provided():
    fitter = IVSurfaceFitter()
    obs = _synthetic_observations(atm_iv=0.18, skew=-0.5, curvature=2.0)
    surf = fitter.fit(observations=obs, spot=23000.0, time_to_expiry=7/365)
    leg = _FakeLeg("CE", 23000.0, +1, 1)
    profile = compute_strategy_greeks(
        legs=[leg], spot=23000.0, time_to_expiry=7/365,
        lot_size=65, iv_surface=surf, fallback_iv=0.25,
    )
    assert profile.iv_surface_used
    assert profile.per_leg[0].iv_used == pytest.approx(surf.atm_iv,
                                                        abs=0.005)


def test_compute_strategy_greeks_serializable():
    leg = _FakeLeg("CE", 23000.0, +1, 1)
    profile = compute_strategy_greeks(
        legs=[leg], spot=23000.0, time_to_expiry=7/365,
        lot_size=65, fallback_iv=0.18,
    )
    import json
    json.dumps(profile.to_dict())
