"""Scenario engine — the institutional what-if math (backend only; the
UI renders this data later).

The founder's spec, exactly:

  INPUTS  : underlying_move, time_passed_min, iv_move, strike, qty,
            entry_premium, target, stop, option_type
  OUTPUTS : expected_premium, expected_pnl, breakeven, danger_zone,
            theta_bleed, gamma_benefit_risk, best_suggestion,
            ITM/ATM/OTM comparison
  VISUALS : payoff_curve, pnl_heatmap (spot x time), premium_sensitivity,
            risk_waterfall  (NO pie charts)

All repricing uses the Black-Scholes engine in greeks.py: same IV (plus
the scenario's iv_move), advanced clock, new spot. The risk waterfall
decomposes the premium change into delta / gamma / theta / vega
contributions so the operator sees WHAT moves the money, not just that
it moved.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .greeks import bs_price, greeks, implied_vol
from .io_decl import IOSpec, declare

RATE = 0.065


@dataclass
class ScenarioInput:
    spot: float
    strike: float
    option_type: str            # CE | PE
    entry_premium: float
    quantity: int
    t_years: float              # time to expiry at "now"
    underlying_move: float = 0.0    # points, signed
    time_passed_min: float = 0.0
    iv_move: float = 0.0            # vol points, e.g. +2 means +0.02
    target: Optional[float] = None
    stop: Optional[float] = None


def _t_after(t_years: float, minutes: float) -> float:
    return max(t_years - minutes / (365.0 * 24.0 * 60.0), 1.0 / (365 * 24))


def evaluate(inp: ScenarioInput) -> Dict[str, Any]:
    """The full numeric output block."""
    iv0 = implied_vol(inp.entry_premium, inp.spot, inp.strike,
                      inp.t_years, inp.option_type)
    if iv0 is None:
        iv0 = 0.15            # fallback when premium is outside no-arb band
    iv1 = max(0.01, iv0 + inp.iv_move)
    spot1 = inp.spot + inp.underlying_move
    t1 = _t_after(inp.t_years, inp.time_passed_min)

    expected_premium = bs_price(spot1, inp.strike, t1, iv1, inp.option_type, RATE)
    pnl = (expected_premium - inp.entry_premium) * inp.quantity

    g0 = greeks(inp.spot, inp.strike, inp.t_years, iv0, inp.option_type, RATE)
    # decompose the premium change (risk waterfall)
    d_spot = inp.underlying_move
    delta_pnl = g0.delta * d_spot * inp.quantity
    gamma_pnl = 0.5 * g0.gamma * d_spot * d_spot * inp.quantity
    theta_pnl = g0.theta_per_day * (inp.time_passed_min / (24 * 60)) * inp.quantity
    vega_pnl = g0.vega_per_pct * (inp.iv_move * 100) * inp.quantity
    residual = pnl - (delta_pnl + gamma_pnl + theta_pnl + vega_pnl)

    # breakeven spot (where premium returns to entry, holding t1/iv1)
    breakeven_spot = _solve_breakeven(inp, iv1, t1)
    # danger zone: spot range where pnl < -50% of entry cost
    danger = _danger_zone(inp, iv1, t1)

    itm_atm_otm = _compare_strikes(inp, iv0)
    best = _best_suggestion(pnl, theta_pnl, gamma_pnl, inp)

    return {
        "iv_now": round(iv0, 4),
        "iv_scenario": round(iv1, 4),
        "spot_scenario": round(spot1, 1),
        "expected_premium": round(expected_premium, 2),
        "expected_pnl": round(pnl, 0),
        "breakeven_spot": breakeven_spot,
        "danger_zone": danger,
        "theta_bleed_pnl": round(theta_pnl, 0),
        "gamma_benefit_pnl": round(gamma_pnl, 0),
        "delta_pnl": round(delta_pnl, 0),
        "vega_pnl": round(vega_pnl, 0),
        "residual_pnl": round(residual, 0),
        "risk_waterfall": [
            {"factor": "delta", "pnl": round(delta_pnl, 0)},
            {"factor": "gamma", "pnl": round(gamma_pnl, 0)},
            {"factor": "theta", "pnl": round(theta_pnl, 0)},
            {"factor": "vega", "pnl": round(vega_pnl, 0)},
            {"factor": "residual", "pnl": round(residual, 0)},
        ],
        "itm_atm_otm": itm_atm_otm,
        "best_suggestion": best,
    }


def payoff_curve(inp: ScenarioInput, span_pts: float = 300.0,
                 steps: int = 61) -> List[Dict[str, float]]:
    """P&L vs underlying at the scenario's time/IV. The proper payoff
    curve (replaces the pie chart)."""
    iv0 = implied_vol(inp.entry_premium, inp.spot, inp.strike,
                      inp.t_years, inp.option_type) or 0.15
    iv1 = max(0.01, iv0 + inp.iv_move)
    t1 = _t_after(inp.t_years, inp.time_passed_min)
    out = []
    for i in range(steps):
        s = inp.spot - span_pts + (2 * span_pts) * i / (steps - 1)
        prem = bs_price(s, inp.strike, t1, iv1, inp.option_type, RATE)
        out.append({"spot": round(s, 1),
                    "pnl": round((prem - inp.entry_premium) * inp.quantity, 0)})
    return out


def pnl_heatmap(inp: ScenarioInput, span_pts: float = 200.0,
                spot_steps: int = 9, time_steps: int = 6) -> Dict[str, Any]:
    """P&L over (spot move) x (minutes elapsed). The heat map the
    founder asked for: shows how the same move pays very differently
    as theta eats the premium."""
    iv0 = implied_vol(inp.entry_premium, inp.spot, inp.strike,
                      inp.t_years, inp.option_type) or 0.15
    iv1 = max(0.01, iv0 + inp.iv_move)
    spots = [inp.spot - span_pts + 2 * span_pts * i / (spot_steps - 1)
             for i in range(spot_steps)]
    horizon = max(inp.time_passed_min, 60.0)
    times = [horizon * j / (time_steps - 1) for j in range(time_steps)]
    grid = []
    for mins in times:
        t = _t_after(inp.t_years, mins)
        row = []
        for s in spots:
            prem = bs_price(s, inp.strike, t, iv1, inp.option_type, RATE)
            row.append(round((prem - inp.entry_premium) * inp.quantity, 0))
        grid.append(row)
    return {"spots": [round(s, 1) for s in spots],
            "minutes": [round(m, 0) for m in times],
            "pnl_grid": grid}


def premium_sensitivity(inp: ScenarioInput) -> List[Dict[str, Any]]:
    """How much premium moves per unit of each driver, at current state
    — the sensitivity graph (per-Greek bars)."""
    iv0 = implied_vol(inp.entry_premium, inp.spot, inp.strike,
                      inp.t_years, inp.option_type) or 0.15
    g = greeks(inp.spot, inp.strike, inp.t_years, iv0, inp.option_type, RATE)
    return [
        {"driver": "per +50 pts spot", "premium_change": round(g.delta * 50, 2)},
        {"driver": "per +1 day", "premium_change": round(g.theta_per_day, 2)},
        {"driver": "per +1 vol pt", "premium_change": round(g.vega_per_pct, 2)},
        {"driver": "gamma (per 50 pts²)",
         "premium_change": round(0.5 * g.gamma * 50 * 50, 2)},
    ]


# -- internals -------------------------------------------------------------

def _solve_breakeven(inp: ScenarioInput, iv: float, t: float) -> Optional[float]:
    lo, hi = inp.spot - 800, inp.spot + 800
    f = lambda s: bs_price(s, inp.strike, t, iv, inp.option_type, RATE) - inp.entry_premium
    flo, fhi = f(lo), f(hi)
    if flo == 0:
        return round(lo, 1)
    if (flo < 0) == (fhi < 0):
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < 0.05:
            return round(mid, 1)
        if (fm < 0) == (flo < 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return round(0.5 * (lo + hi), 1)


def _danger_zone(inp: ScenarioInput, iv: float, t: float) -> Dict[str, Any]:
    threshold = -0.5 * inp.entry_premium * inp.quantity
    curve = payoff_curve(inp, span_pts=400, steps=81)
    danger = [p["spot"] for p in curve if p["pnl"] <= threshold]
    if not danger:
        return {"present": False}
    return {"present": True, "from_spot": min(danger), "to_spot": max(danger),
            "loss_threshold_pnl": round(threshold, 0)}


def _compare_strikes(inp: ScenarioInput, iv: float) -> List[Dict[str, Any]]:
    """ITM/ATM/OTM comparison: reprice the SAME directional move at three
    strikes and report which captures it best per rupee of premium."""
    step = 50.0
    spot1 = inp.spot + inp.underlying_move
    t1 = _t_after(inp.t_years, inp.time_passed_min)
    rows = []
    for label, k_off in (("ITM", -2), ("ATM", 0), ("OTM", +2)):
        if inp.option_type == "CE":
            strike = round(inp.spot / step) * step + k_off * step
        else:
            strike = round(inp.spot / step) * step - k_off * step
        prem0 = bs_price(inp.spot, strike, inp.t_years, iv, inp.option_type, RATE)
        prem1 = bs_price(spot1, strike, t1, max(0.01, iv + inp.iv_move),
                         inp.option_type, RATE)
        if prem0 <= 0:
            continue
        rows.append({"zone": label, "strike": strike,
                     "premium_now": round(prem0, 2),
                     "premium_after": round(prem1, 2),
                     "return_pct": round((prem1 - prem0) / prem0 * 100, 1)})
    return rows


def _best_suggestion(pnl: float, theta_pnl: float, gamma_pnl: float,
                     inp: ScenarioInput) -> str:
    if pnl <= 0 and theta_pnl < 0 and abs(theta_pnl) > abs(gamma_pnl):
        return ("theta is eating this faster than the move helps — this "
                "scenario favours selling premium, not buying it")
    if gamma_pnl > abs(theta_pnl) * 2:
        return ("gamma strongly positive — a fast move pays far more than "
                "theta costs; OTM convexity is attractive here")
    if pnl > 0:
        return "scenario is net positive; size to the danger-zone distance"
    return "marginal scenario — wait for a cleaner setup or tighten the strike"


declare(IOSpec(
    module="sentinel.scenario_engine",
    purpose="institutional what-if math: payoff curve, P&L heatmap, "
            "premium sensitivity, risk waterfall, ITM/ATM/OTM compare",
    inputs=["sentinel.scenario_engine.ScenarioInput (operator-set)"],
    outputs=["numeric outputs dict + payoff_curve + pnl_heatmap + "
             "premium_sensitivity (consumed by the UI later)"],
    consumes_from=["sentinel.greeks"],
    produces_for=["sentinel.server (UI)"],
    tier="TRUSTED",
))
