"""Customer strategy auditor — paste any multi-leg option position, get a
named institutional verdict.

The founder's customer-facing pillar: retail traders build positions
that the textbook has already named (covered call, iron condor, bull-
put spread). Most don't know which one they built or what it's exposed
to. The auditor:

  1. Computes the multi-leg P&L curve at expiry across a spot grid.
  2. Detects max profit, max loss, breakevens.
  3. Detects WHICH named strategy was constructed.
  4. Aggregates net Greeks (delta/gamma/theta/vega).
  5. Flags structural risks ("uncapped loss", "negative theta in chop",
     "vega-short on a vol-rising day").

Detection rules are the standard option-strategy taxonomy
(Hull, "Options, Futures and Other Derivatives", 11e). Sentinel tier:
RETAIL (the audit itself is free) + PRO (the Greek-book + stress add-on).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .institutional import LegExposure, portfolio_greek_exposures
from .io_decl import IOSpec, declare


@dataclass(frozen=True)
class StrategyLeg:
    option_type: str       # "CE" | "PE"
    strike: float
    qty: int               # signed: + long, - short
    premium: float         # entry premium per unit
    delta: float = 0.0
    gamma: float = 0.0
    theta_per_day: float = 0.0
    vega_per_pct: float = 0.0


@dataclass
class AuditResult:
    detected_strategy: str
    max_profit_rupees: float
    max_loss_rupees: float
    breakeven_points: List[float] = field(default_factory=list)
    payoff_curve: List[Dict[str, float]] = field(default_factory=list)
    net_greeks: Dict[str, float] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Payoff math — per-leg intrinsic at a candidate spot
# ---------------------------------------------------------------------------

def _intrinsic(option_type: str, strike: float, spot: float) -> float:
    if option_type == "CE":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def _leg_pnl_at_expiry(leg: StrategyLeg, spot: float) -> float:
    return leg.qty * (_intrinsic(leg.option_type, leg.strike, spot) - leg.premium)


def _portfolio_pnl_at_expiry(legs: Sequence[StrategyLeg], spot: float) -> float:
    return sum(_leg_pnl_at_expiry(l, spot) for l in legs)


def payoff_curve(legs: Sequence[StrategyLeg],
                 spot_now: float,
                 span_pct: float = 0.08,
                 steps: int = 41) -> List[Dict[str, float]]:
    """Compute payoff (₹) at expiry across a +/- span_pct spot grid."""
    lo = spot_now * (1.0 - span_pct)
    hi = spot_now * (1.0 + span_pct)
    out = []
    for i in range(steps):
        s = lo + (hi - lo) * i / max(steps - 1, 1)
        out.append({"spot": round(s, 2),
                    "pnl": round(_portfolio_pnl_at_expiry(legs, s), 2)})
    return out


def _breakevens(curve: Sequence[Dict[str, float]]) -> List[float]:
    """Linear-interpolated zero crossings of the payoff curve."""
    out = []
    for a, b in zip(curve, curve[1:]):
        if a["pnl"] == 0:
            out.append(a["spot"])
            continue
        if a["pnl"] * b["pnl"] < 0:
            # zero crossing between a and b
            t = a["pnl"] / (a["pnl"] - b["pnl"])
            out.append(round(a["spot"] + t * (b["spot"] - a["spot"]), 2))
    return out


# ---------------------------------------------------------------------------
# Named-strategy detection (Hull 11e taxonomy)
# ---------------------------------------------------------------------------

def detect_strategy(legs: Sequence[StrategyLeg]) -> str:
    """Identify the canonical strategy name. Returns 'CUSTOM' if no match."""
    if not legs:
        return "EMPTY"
    n = len(legs)
    ce = [l for l in legs if l.option_type == "CE"]
    pe = [l for l in legs if l.option_type == "PE"]
    long_ = [l for l in legs if l.qty > 0]
    short_ = [l for l in legs if l.qty < 0]

    # single-leg
    if n == 1:
        l = legs[0]
        if l.qty > 0 and l.option_type == "CE":
            return "LONG_CALL"
        if l.qty > 0 and l.option_type == "PE":
            return "LONG_PUT"
        if l.qty < 0 and l.option_type == "CE":
            return "NAKED_SHORT_CALL"
        if l.qty < 0 and l.option_type == "PE":
            return "NAKED_SHORT_PUT"

    # two-leg combinations
    if n == 2:
        a, b = sorted(legs, key=lambda x: x.strike)
        # vertical spreads (same type, opposite direction, same expiry assumed)
        if a.option_type == b.option_type:
            if a.option_type == "CE":
                if a.qty > 0 and b.qty < 0 and a.qty == -b.qty:
                    return "BULL_CALL_SPREAD"
                if a.qty < 0 and b.qty > 0 and -a.qty == b.qty:
                    return "BEAR_CALL_SPREAD"
            if a.option_type == "PE":
                if a.qty > 0 and b.qty < 0 and a.qty == -b.qty:
                    return "BEAR_PUT_SPREAD"
                if a.qty < 0 and b.qty > 0 and -a.qty == b.qty:
                    return "BULL_PUT_SPREAD"
        # straddle / strangle (CE+PE, both long or both short)
        if {a.option_type, b.option_type} == {"CE", "PE"}:
            if a.qty > 0 and b.qty > 0:
                return "LONG_STRADDLE" if a.strike == b.strike else "LONG_STRANGLE"
            if a.qty < 0 and b.qty < 0:
                return "SHORT_STRADDLE" if a.strike == b.strike else "SHORT_STRANGLE"

    # four-leg combinations
    if n == 4 and len(ce) == 2 and len(pe) == 2:
        # iron condor: long OTM put + short ATM-ish put + short ATM-ish call + long OTM call
        ce_sorted = sorted(ce, key=lambda x: x.strike)
        pe_sorted = sorted(pe, key=lambda x: x.strike)
        # iron condor pattern: PE: long lower, short upper; CE: short lower, long upper
        if (pe_sorted[0].qty > 0 and pe_sorted[1].qty < 0
                and ce_sorted[0].qty < 0 and ce_sorted[1].qty > 0):
            return "IRON_CONDOR"
        # iron butterfly: short straddle + protective wings (same strike on shorts)
        if (pe_sorted[1].strike == ce_sorted[0].strike
                and pe_sorted[0].qty > 0 and pe_sorted[1].qty < 0
                and ce_sorted[0].qty < 0 and ce_sorted[1].qty > 0):
            return "IRON_BUTTERFLY"
        # reverse iron condor
        if (pe_sorted[0].qty < 0 and pe_sorted[1].qty > 0
                and ce_sorted[0].qty > 0 and ce_sorted[1].qty < 0):
            return "REVERSE_IRON_CONDOR"

    # three-leg butterflies (CE: long lower, short 2x middle, long upper)
    if n == 3 and len(ce) == 3:
        ce_sorted = sorted(ce, key=lambda x: x.strike)
        if (ce_sorted[0].qty > 0 and ce_sorted[1].qty < 0
                and ce_sorted[2].qty > 0
                and abs(ce_sorted[1].qty) == ce_sorted[0].qty + ce_sorted[2].qty):
            return "CALL_BUTTERFLY"
    if n == 3 and len(pe) == 3:
        pe_sorted = sorted(pe, key=lambda x: x.strike)
        if (pe_sorted[0].qty > 0 and pe_sorted[1].qty < 0
                and pe_sorted[2].qty > 0
                and abs(pe_sorted[1].qty) == pe_sorted[0].qty + pe_sorted[2].qty):
            return "PUT_BUTTERFLY"

    return "CUSTOM"


# ---------------------------------------------------------------------------
# Risk flagging
# ---------------------------------------------------------------------------

def _flag_uncapped_loss(legs: Sequence[StrategyLeg]) -> Optional[str]:
    """A naked short option has uncapped loss in the direction of the underlying."""
    net_short_call_qty = sum(l.qty for l in legs if l.option_type == "CE" and l.qty < 0)
    long_call_qty_above_short = 0
    short_call_strikes = sorted([l.strike for l in legs if l.option_type == "CE" and l.qty < 0])
    if short_call_strikes:
        highest_short_ce = short_call_strikes[-1]
        long_call_qty_above_short = sum(
            l.qty for l in legs
            if l.option_type == "CE" and l.qty > 0 and l.strike >= highest_short_ce
        )
        if abs(net_short_call_qty) > long_call_qty_above_short:
            return "uncapped UPSIDE loss — net short calls without protection above"

    net_short_put_qty = sum(l.qty for l in legs if l.option_type == "PE" and l.qty < 0)
    short_put_strikes = sorted([l.strike for l in legs if l.option_type == "PE" and l.qty < 0])
    if short_put_strikes:
        lowest_short_pe = short_put_strikes[0]
        long_put_qty_below_short = sum(
            l.qty for l in legs
            if l.option_type == "PE" and l.qty > 0 and l.strike <= lowest_short_pe
        )
        if abs(net_short_put_qty) > long_put_qty_below_short:
            return "uncapped DOWNSIDE loss — net short puts without protection below"
    return None


def _flag_greeks(net: Dict[str, float], regime: Optional[str]) -> List[str]:
    out: List[str] = []
    if regime == "chop" and net["net_theta_per_day"] < 0:
        out.append("negative theta in CHOP regime — paying time decay with no trend")
    if regime == "trending" and net["net_gamma"] < 0:
        out.append("short gamma in TRENDING regime — losses accelerate against you")
    if net["net_vega_per_pct"] < 0 and net["net_delta"] == 0:
        out.append("vega-short delta-neutral — vulnerable to IV spikes")
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def audit_strategy(legs: Sequence[StrategyLeg], spot_now: float,
                   regime: Optional[str] = None) -> AuditResult:
    """The customer-facing audit. Tier: RETAIL.

    Returns a named strategy verdict, the at-expiry payoff curve, max
    profit/loss, breakevens, net Greeks, and structural flags."""
    if not legs:
        return AuditResult(detected_strategy="EMPTY", max_profit_rupees=0,
                           max_loss_rupees=0)
    name = detect_strategy(legs)
    curve = payoff_curve(legs, spot_now)
    pnls = [p["pnl"] for p in curve]
    max_p = max(pnls)
    max_l = min(pnls)
    bes = _breakevens(curve)
    # build LegExposures for the Greek book (1-unit each — qty already baked into payoff)
    exposures = [LegExposure(
        tradingsymbol=f"{l.option_type}{int(l.strike)}",
        qty=l.qty, premium=l.premium,
        delta=l.delta, gamma=l.gamma,
        theta_per_day=l.theta_per_day, vega_per_pct=l.vega_per_pct,
    ) for l in legs]
    net = portfolio_greek_exposures(exposures, spot=spot_now)

    flags: List[str] = []
    u = _flag_uncapped_loss(legs)
    if u:
        flags.append(u)
    flags.extend(_flag_greeks(net, regime))

    notes: List[str] = []
    if name == "CUSTOM":
        notes.append("non-textbook combination — verify intent before holding")
    if name.startswith("NAKED_SHORT"):
        notes.append("naked short option: SEBI margin will be high; consider a "
                     "protective wing to turn this into a defined-risk spread")

    return AuditResult(
        detected_strategy=name,
        max_profit_rupees=round(max_p, 2),
        max_loss_rupees=round(max_l, 2),
        breakeven_points=bes,
        payoff_curve=curve,
        net_greeks=net,
        flags=flags,
        notes=notes,
    )


declare(IOSpec(
    module="sentinel.auditor",
    purpose="customer multi-leg strategy auditor — detect named strategy "
            "(Hull taxonomy), compute payoff curve + max P/L + breakevens, "
            "aggregate Greeks, flag uncapped-loss / regime-mismatched setups",
    inputs=["list[StrategyLeg]", "current spot", "optional regime"],
    outputs=["AuditResult: detected_strategy, payoff_curve, max P/L, "
             "breakevens, net Greeks, flags, notes"],
    consumes_from=["sentinel.institutional"],
    produces_for=["sentinel.server (customer audit endpoint)",
                  "sentinel.reports"],
    tier="TRUSTED",
))
