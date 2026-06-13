"""Multi-leg strategy builder — turn a customer intent into a legged
position.

The customer says "I want premium income, ~delta-neutral, defined risk
for this weekly expiry on NIFTY." Today they have to either guess legs
or pay an advisor. The builder picks the legs.

Five canonical intents (Hull 11e + Natenberg 2nd ed.):

  PREMIUM_SELL_NEUTRAL    short iron condor at +/- 1 sigma wings
  PREMIUM_SELL_INCOME     short strangle at +/- 1 sigma (uncapped — warns)
  DIRECTIONAL_BULL        bull call spread at ATM + 1 sigma OTM cap
  DIRECTIONAL_BEAR        bear put spread at ATM + 1 sigma OTM cap
  VOL_BUY                 long straddle at ATM
  NEUTRAL_INCOME          iron butterfly at ATM with +/- 1 sigma wings

Tier: PRO (the builder is a paid feature; RETAIL gets one
recommendation per session).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .auditor import StrategyLeg, audit_strategy
from .io_decl import IOSpec, declare


INTENTS = (
    "PREMIUM_SELL_NEUTRAL",
    "PREMIUM_SELL_INCOME",
    "DIRECTIONAL_BULL",
    "DIRECTIONAL_BEAR",
    "VOL_BUY",
    "NEUTRAL_INCOME",
)


@dataclass(frozen=True)
class ChainQuote:
    option_type: str
    strike: float
    premium: float
    delta: float = 0.0
    gamma: float = 0.0
    theta_per_day: float = 0.0
    vega_per_pct: float = 0.0


def _pick(chain: Sequence[ChainQuote], option_type: str, target_strike: float
          ) -> Optional[ChainQuote]:
    """Closest available strike on the requested side."""
    cands = [c for c in chain if c.option_type == option_type]
    if not cands:
        return None
    return min(cands, key=lambda c: abs(c.strike - target_strike))


def _one_sigma_strikes(spot: float, iv: float, t_years: float) -> tuple[float, float]:
    """Standard +/- 1 sigma move = spot * exp(+/- iv*sqrt(t)). Rounded to
    nearest 50 (NIFTY tick), 100 for BankNIFTY-scale should be passed by
    caller via the ``strike_step`` arg above."""
    sig = iv * math.sqrt(max(t_years, 1e-6))
    return spot * math.exp(-sig), spot * math.exp(sig)


def _round_to_step(strike: float, step: float) -> float:
    return round(strike / step) * step


def build(intent: str, spot: float, chain: Sequence[ChainQuote],
          iv: float = 0.15, t_years: float = 7 / 365,
          qty: int = 75, strike_step: float = 50.0,
          regime: Optional[str] = None) -> Dict[str, Any]:
    """Construct legs satisfying ``intent`` and run them through the
    auditor so the customer sees the audit alongside the build.

    Returns: ``{intent, legs, audit, missing_strikes (if any)}``. If a
    target strike isn't on the chain, the closest available is used and
    the original target is noted in ``missing_strikes``.
    """
    if intent not in INTENTS:
        raise ValueError(f"unknown intent: {intent}; available: {INTENTS}")
    if qty <= 0:
        raise ValueError("qty must be > 0")
    atm = _round_to_step(spot, strike_step)
    s_lo, s_hi = _one_sigma_strikes(spot, iv, t_years)
    lo = _round_to_step(s_lo, strike_step)
    hi = _round_to_step(s_hi, strike_step)

    legs: List[StrategyLeg] = []
    missing: List[str] = []

    def _add(option_type: str, target_strike: float, signed_qty: int) -> None:
        q = _pick(chain, option_type, target_strike)
        if q is None:
            missing.append(f"{option_type}@{target_strike}")
            return
        if abs(q.strike - target_strike) > 1e-9:
            missing.append(f"{option_type}@{target_strike} -> used {q.strike}")
        legs.append(StrategyLeg(
            option_type=q.option_type, strike=q.strike,
            qty=signed_qty, premium=q.premium,
            delta=q.delta, gamma=q.gamma,
            theta_per_day=q.theta_per_day, vega_per_pct=q.vega_per_pct,
        ))

    if intent == "PREMIUM_SELL_NEUTRAL":
        # short iron condor: long OTM PE wing, short PE inside; short CE inside, long OTM CE wing
        wing_lo = _round_to_step(spot * math.exp(-2 * iv * math.sqrt(t_years)),
                                  strike_step)
        wing_hi = _round_to_step(spot * math.exp(2 * iv * math.sqrt(t_years)),
                                  strike_step)
        _add("PE", wing_lo, +qty)
        _add("PE", lo, -qty)
        _add("CE", hi, -qty)
        _add("CE", wing_hi, +qty)

    elif intent == "PREMIUM_SELL_INCOME":
        # short strangle (uncapped) — warning surfaced by auditor
        _add("PE", lo, -qty)
        _add("CE", hi, -qty)

    elif intent == "DIRECTIONAL_BULL":
        # bull call spread: long ATM, short ~+1 sigma
        _add("CE", atm, +qty)
        _add("CE", hi, -qty)

    elif intent == "DIRECTIONAL_BEAR":
        # bear put spread: long ATM, short ~-1 sigma
        _add("PE", atm, +qty)
        _add("PE", lo, -qty)

    elif intent == "VOL_BUY":
        # long straddle at ATM
        _add("CE", atm, +qty)
        _add("PE", atm, +qty)

    elif intent == "NEUTRAL_INCOME":
        # iron butterfly: short ATM straddle + protective wings at +/- 1 sigma
        _add("PE", lo, +qty)
        _add("PE", atm, -qty)
        _add("CE", atm, -qty)
        _add("CE", hi, +qty)

    audit = audit_strategy(legs, spot_now=spot, regime=regime)
    return {
        "intent": intent,
        "spot": spot,
        "atm_strike": atm,
        "one_sigma_low": lo,
        "one_sigma_high": hi,
        "legs": [
            {
                "option_type": l.option_type, "strike": l.strike,
                "qty": l.qty, "premium": l.premium,
            }
            for l in legs
        ],
        "audit": {
            "detected_strategy": audit.detected_strategy,
            "max_profit_rupees": audit.max_profit_rupees,
            "max_loss_rupees": audit.max_loss_rupees,
            "breakeven_points": audit.breakeven_points,
            "net_greeks": audit.net_greeks,
            "flags": audit.flags,
            "notes": audit.notes,
        },
        "missing_strikes": missing,
    }


declare(IOSpec(
    module="sentinel.strategy_builder",
    purpose="multi-leg builder for customer intents (PREMIUM_SELL_NEUTRAL, "
            "DIRECTIONAL_BULL/BEAR, VOL_BUY, NEUTRAL_INCOME, ...) — picks "
            "legs from the live chain and runs the result through the auditor",
    inputs=["intent string", "spot, iv, t_years, qty",
            "list[ChainQuote] from the live option chain"],
    outputs=["dict with chosen legs, the auditor's verdict, "
             "and any missing/substituted strikes"],
    consumes_from=["sentinel.auditor", "sentinel.kite_client (chain)"],
    produces_for=["sentinel.server (customer builder endpoint)"],
    tier="TRUSTED",
))
