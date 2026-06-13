"""Scientists — the candidate generators (the founder's "wizards").

The rule the founder set: NOT infinite random trades (99% noise). A
finite set of HIGH-QUALITY, context-aware strategies, each emitting
trades that carry an explicit validation — why this, why now, why it
should beat the market, and what would invalidate it. Quality of the
scientists determines the quality of the whole laboratory.

Each scientist reads a MarketSnapshot (spot + the ATM±5 universe with
live Greeks + context features sourced from the research codebase) and
emits LedgerEvent hypotheses into the shadow ledger at SHADOW tier.
The curator judges them; the orchestration spine graduates the good
ones toward TRUSTED over time.

A scientist that cannot articulate its reasoning does not emit — the
ledger enforces reason_codes at write time, and the base class makes
that the contract, not an afterthought.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .io_decl import IOSpec, declare
from .moneyness import ContractRef
from .shadow_ledger import (
    Context, Hypothesis, Identity, KIND_VIRTUAL, LedgerEvent, new_event,
)


@dataclass
class ContractLive:
    """One universe contract with its live tradable state + Greeks."""
    ref: ContractRef
    premium: float
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    volume: float = 0.0
    oi: float = 0.0
    iv: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    day_high: float = 0.0
    day_low: float = 0.0


@dataclass
class MarketSnapshot:
    """Everything a scientist needs to think. Context fields are filled
    by the research-codebase bridge (wave 3); scientists degrade
    gracefully when a field is None."""
    session: str
    spot: float
    universe: Dict[str, ContractLive]          # moneyness_str -> live
    # context (research bridge fills these; None = unknown)
    trend: Optional[str] = None                # "up" | "down" | "chop"
    vol_regime: Optional[str] = None           # "low" | "normal" | "high"
    minutes_since_open: Optional[float] = None
    pullback_count: Optional[int] = None       # pullbacks in the current leg
    gamma_expansion_early: Optional[bool] = None
    dist_from_vwap_atr: Optional[float] = None
    minutes_to_expiry: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def get(self, moneyness_str: str) -> Optional[ContractLive]:
        return self.universe.get(moneyness_str)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Scientist:
    """A named, validating candidate generator."""
    name = "base"

    def generate(self, snap: MarketSnapshot) -> List[LedgerEvent]:
        raise NotImplementedError

    def _emit(self, snap: MarketSnapshot, c: ContractLive,
              *, expected_move: float, horizon_min: float,
              target_premium: float, stop_premium: float, confidence: float,
              reasons: List[str], invalidations: List[str]) -> LedgerEvent:
        ident = Identity(
            instrument=c.ref.tradingsymbol, option_type=c.ref.option_type,
            strike=c.ref.strike, expiry=None, moneyness_key=c.ref.key.as_str(),
            underlying_price=snap.spot, premium=c.premium, bid=c.bid,
            ask=c.ask, spread=c.spread, volume=c.volume, iv=c.iv, oi=c.oi,
            delta=c.delta, gamma=c.gamma, theta=c.theta, vega=c.vega)
        ctx = Context(
            trend=snap.trend, volatility=snap.vol_regime,
            time_of_day=(None if snap.minutes_since_open is None
                         else f"{int(snap.minutes_since_open)}min_in"))
        hyp = Hypothesis(
            expected_underlying_move=expected_move,
            expected_premium=target_premium,
            expected_horizon_min=horizon_min,
            suggested_entry=c.premium,
            suggested_stop=stop_premium,
            suggested_target=target_premium,
            confidence=confidence, reason_codes=reasons,
            invalidation_conditions=invalidations)
        return new_event(KIND_VIRTUAL, snap.session, ident, ctx, hyp,
                         scientist=self.name)


# ---------------------------------------------------------------------------
# Concrete scientists — each is a documented, logical strategy
# ---------------------------------------------------------------------------

class SecondPullbackScientist(Scientist):
    """The founder's own edge: the best CE entry comes after the SECOND
    pullback in an uptrend, not the first breakout. Buys the ATM call
    when trend is up and at least two pullbacks have printed."""
    name = "second_pullback_ce"

    def generate(self, snap):
        if snap.trend != "up" or (snap.pullback_count or 0) < 2:
            return []
        c = snap.get("NIFTY:CE:+0") or _first_ce(snap)
        if c is None or c.premium <= 0:
            return []
        delta = c.delta or 0.5
        move = 40.0
        target = c.premium + abs(delta) * move
        stop = c.premium - abs(delta) * move * 0.6
        return [self._emit(
            snap, c, expected_move=move, horizon_min=15.0,
            target_premium=round(target, 2), stop_premium=round(stop, 2),
            confidence=0.62,
            reasons=["uptrend", "second_pullback_complete",
                     "atm_delta_efficient"],
            invalidations=["spot_breaks_pullback_low", "trend_flips_to_chop"])]


class GammaScalpScientist(Scientist):
    """OTM calls/puts work ONLY when gamma expansion starts early
    (the founder's observation). Buys +2 OTM in the trend direction
    when gamma_expansion_early is flagged."""
    name = "gamma_scalp_otm"

    def generate(self, snap):
        if not snap.gamma_expansion_early or snap.trend not in ("up", "down"):
            return []
        key = "NIFTY:CE:+2" if snap.trend == "up" else "NIFTY:PE:+2"
        c = snap.get(key)
        if c is None or c.premium <= 0:
            return []
        move = 60.0
        delta = abs(c.delta or 0.3)
        target = c.premium + delta * move + (c.gamma or 0) * move * move * 0.5
        stop = c.premium * 0.6
        return [self._emit(
            snap, c, expected_move=move, horizon_min=10.0,
            target_premium=round(target, 2), stop_premium=round(stop, 2),
            confidence=0.55,
            reasons=["gamma_expansion_early", f"trend_{snap.trend}",
                     "otm_convexity_play"],
            invalidations=["gamma_expansion_stalls", "spot_reverses"])]


class ChopMeanRevScientist(Scientist):
    """ATM works better during chop (the founder's note). When the
    market is choppy and stretched from VWAP, fade with the ATM option
    on the mean-reversion side."""
    name = "chop_atm_meanrev"

    def generate(self, snap):
        if snap.trend != "chop" or snap.dist_from_vwap_atr is None:
            return []
        if abs(snap.dist_from_vwap_atr) < 1.0:
            return []
        # Stretched above VWAP -> expect pullback -> buy PE; below -> CE.
        side = "PE" if snap.dist_from_vwap_atr > 0 else "CE"
        c = snap.get(f"NIFTY:{side}:+0")
        if c is None or c.premium <= 0:
            return []
        move = 30.0
        delta = abs(c.delta or 0.5)
        target = c.premium + delta * move
        stop = c.premium - delta * move * 0.7
        return [self._emit(
            snap, c, expected_move=move, horizon_min=20.0,
            target_premium=round(target, 2), stop_premium=round(stop, 2),
            confidence=0.5,
            reasons=["chop_regime", "stretched_from_vwap",
                     "atm_in_chop_outperforms"],
            invalidations=["chop_resolves_into_trend",
                           "vwap_distance_widens"])]


class ThetaGuardScientist(Scientist):
    """A defensive scientist: near expiry with high theta, it emits a
    REJECTED-style low-confidence note on any OTM buy so the ledger
    captures the counterfactual 'we declined this and were right/wrong'.
    Demonstrates the value of logging what we DON'T do."""
    name = "theta_guard"

    def generate(self, snap):
        if snap.minutes_to_expiry is None or snap.minutes_to_expiry > 120:
            return []
        out: List[LedgerEvent] = []
        for key in ("NIFTY:CE:+3", "NIFTY:PE:+3"):
            c = snap.get(key)
            if c is None or c.premium <= 0 or c.theta is None:
                continue
            if abs(c.theta) < 0.08 * c.premium:
                continue
            move = 50.0
            delta = abs(c.delta or 0.2)
            target = c.premium + delta * move
            out.append(self._emit(
                snap, c, expected_move=move, horizon_min=30.0,
                target_premium=round(target, 2),
                stop_premium=round(c.premium * 0.5, 2),
                confidence=0.30,
                reasons=["near_expiry_high_theta", "low_confidence_guard"],
                invalidations=["fast_directional_move_within_10min"]))
        return out


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------

DEFAULT_SCIENTISTS: List[Scientist] = [
    SecondPullbackScientist(),
    GammaScalpScientist(),
    ChopMeanRevScientist(),
    ThetaGuardScientist(),
]


class ScientistPool:
    """Runs every scientist on a snapshot, returns all candidates. The
    founder's 'finite but high-quality' set; adding a scientist is a
    one-line append, and each must carry its reasoning by construction."""

    def __init__(self, scientists: Optional[List[Scientist]] = None) -> None:
        self.scientists = scientists or list(DEFAULT_SCIENTISTS)

    def run(self, snap: MarketSnapshot) -> List[LedgerEvent]:
        out: List[LedgerEvent] = []
        for sci in self.scientists:
            try:
                out.extend(sci.generate(snap))
            except Exception:
                continue   # a broken scientist must not starve the others
        return out

    def names(self) -> List[str]:
        return [s.name for s in self.scientists]


def _first_ce(snap: MarketSnapshot) -> Optional[ContractLive]:
    for k, v in snap.universe.items():
        if v.ref.option_type == "CE":
            return v
    return None


declare(IOSpec(
    module="sentinel.scientists",
    purpose="candidate generators (wizards): context -> validated trade hypotheses",
    inputs=["sentinel.scientists.MarketSnapshot (spot+universe+context)"],
    outputs=["sentinel.shadow_ledger.LedgerEvent (kind=virtual, SHADOW tier)"],
    consumes_from=["sentinel.portfolio", "sentinel.greeks",
                   "sentinel.moneyness", "liqpool(context bridge, wave3)"],
    produces_for=["sentinel.shadow_ledger", "sentinel.curator"],
    tier="SHADOW",
))
