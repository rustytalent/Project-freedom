"""Strategy-level portfolio Greeks — properly computed.

Replaces the toy linear-decay model in strategy_library/base.py. Given:
  * The list of legs (StrategyLeg objects)
  * Current spot
  * Time to expiry (years)
  * An IVSurface (or fallback flat IV)

…produces a full per-position Greek profile:
  * net_delta (in lots, lot-size-scaled)
  * net_vega (₹ per 1.00 IV change × lot_size)
  * net_theta (₹ per day × lot_size)
  * net_gamma
  * net_rho
  * net_vanna, net_charm, net_vomma (cross-Greeks for advanced risk)
  * per-leg Greeks for ledger / cockpit

Conventions:
  * All Greeks are "money-Greeks" scaled by ``lot_size × lots`` so the
    aggregator can sum them across positions.
  * Theta is per CALENDAR day (annual theta / 365).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .black_scholes import (
    DEFAULT_DIVIDEND_YIELD,
    DEFAULT_RISK_FREE_RATE,
    OptionGreeks,
    call_greeks,
    put_greeks,
)
from .iv_surface import IVSurface


@dataclass(frozen=True)
class PerLegGreeks:
    """Greeks for a single leg, scaled by lot_size × lots × direction."""
    contract_side: str
    strike_price: float
    direction: int
    lots: int
    iv_used: float
    price_model: float            # Black-Scholes price for the leg
    delta: float
    gamma: float
    vega: float                   # per 1.00 sigma change (so ₹100/lot if vega=1)
    theta_per_day: float          # ₹/day
    rho: float
    vanna: float
    charm: float
    vomma: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_side": self.contract_side,
            "strike_price": round(self.strike_price, 2),
            "direction": self.direction,
            "lots": self.lots,
            "iv_used": round(self.iv_used, 4),
            "price_model": round(self.price_model, 4),
            "delta": round(self.delta, 4),
            "gamma": round(self.gamma, 6),
            "vega": round(self.vega, 4),
            "theta_per_day": round(self.theta_per_day, 4),
            "rho": round(self.rho, 6),
            "vanna": round(self.vanna, 6),
            "charm": round(self.charm, 6),
            "vomma": round(self.vomma, 6),
        }


@dataclass(frozen=True)
class StrategyGreeksProfile:
    """Aggregated Greek profile for a strategy."""
    net_delta: float              # contracts (lot_size × lots × direction × delta)
    net_gamma: float
    net_vega: float
    net_theta_per_day: float      # ₹/day, signed
    net_rho: float
    net_vanna: float
    net_charm: float
    net_vomma: float
    per_leg: List[PerLegGreeks]
    iv_surface_used: bool
    iv_surface_confidence: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "net_delta": round(self.net_delta, 4),
            "net_gamma": round(self.net_gamma, 6),
            "net_vega": round(self.net_vega, 4),
            "net_theta_per_day": round(self.net_theta_per_day, 4),
            "net_rho": round(self.net_rho, 4),
            "net_vanna": round(self.net_vanna, 6),
            "net_charm": round(self.net_charm, 6),
            "net_vomma": round(self.net_vomma, 6),
            "per_leg": [l.to_dict() for l in self.per_leg],
            "iv_surface_used": self.iv_surface_used,
            "iv_surface_confidence": round(self.iv_surface_confidence, 3),
            "notes": list(self.notes),
        }


def compute_strategy_greeks(*,
                              legs: List[Any],   # StrategyLeg-like objects
                              spot: float,
                              time_to_expiry: float,
                              lot_size: int = 65,
                              iv_surface: Optional[IVSurface] = None,
                              fallback_iv: float = 0.25,
                              risk_free: float = DEFAULT_RISK_FREE_RATE,
                              dividend: float = DEFAULT_DIVIDEND_YIELD,
                              ) -> StrategyGreeksProfile:
    """Compute the aggregated Greek profile for a strategy.

    For each leg:
      1. Look up the IV at the leg's strike from ``iv_surface`` if
         provided, otherwise use ``fallback_iv``
      2. Price the leg with Black-Scholes and extract all Greeks
      3. Scale by ``direction × lots × lot_size``

    Then aggregate across legs.
    """
    notes: List[str] = []
    per_leg: List[PerLegGreeks] = []
    net_delta = net_gamma = net_vega = 0.0
    net_theta = net_rho = net_vanna = net_charm = net_vomma = 0.0

    iv_surface_used = iv_surface is not None and iv_surface.confidence > 0.05
    iv_surface_conf = iv_surface.confidence if iv_surface else 0.0

    for leg in legs:
        side = str(getattr(leg, "contract_side", ""))
        strike = float(getattr(leg, "strike_price", 0.0))
        direction = int(getattr(leg, "direction", 1))
        lots = int(getattr(leg, "lots", 1))

        if iv_surface_used:
            iv = iv_surface.iv_at_strike(strike)
        else:
            iv = fallback_iv

        if side == "CE":
            g = call_greeks(spot=spot, strike=strike,
                              time_to_expiry=max(1e-6, time_to_expiry),
                              sigma=iv, risk_free=risk_free,
                              dividend=dividend)
        else:
            g = put_greeks(spot=spot, strike=strike,
                             time_to_expiry=max(1e-6, time_to_expiry),
                             sigma=iv, risk_free=risk_free,
                             dividend=dividend)

        if not math.isfinite(g.delta):
            notes.append(f"non-finite greeks for {side} K={strike}; skipped")
            continue

        # Scale by direction × lots × lot_size.
        scale = direction * lots * lot_size
        per_leg.append(PerLegGreeks(
            contract_side=side, strike_price=strike,
            direction=direction, lots=lots, iv_used=iv,
            price_model=g.price,
            delta=g.delta * scale,
            gamma=g.gamma * scale,
            vega=g.vega * scale,
            theta_per_day=(g.theta / 365.0) * scale,
            rho=g.rho * scale,
            vanna=g.vanna * scale,
            charm=g.charm * scale,
            vomma=g.vomma * scale,
        ))

        net_delta += g.delta * scale
        net_gamma += g.gamma * scale
        net_vega += g.vega * scale
        net_theta += (g.theta / 365.0) * scale
        net_rho += g.rho * scale
        net_vanna += g.vanna * scale
        net_charm += g.charm * scale
        net_vomma += g.vomma * scale

    if iv_surface_used:
        notes.append(
            f"IV surface ({iv_surface.fit_method}) used, "
            f"confidence {iv_surface.confidence:.2f}, "
            f"atm_iv={iv_surface.atm_iv:.3f}"
        )
    else:
        notes.append(f"flat IV fallback at {fallback_iv:.2f}")

    return StrategyGreeksProfile(
        net_delta=net_delta,
        net_gamma=net_gamma,
        net_vega=net_vega,
        net_theta_per_day=net_theta,
        net_rho=net_rho,
        net_vanna=net_vanna,
        net_charm=net_charm,
        net_vomma=net_vomma,
        per_leg=per_leg,
        iv_surface_used=iv_surface_used,
        iv_surface_confidence=iv_surface_conf,
        notes=notes,
    )
