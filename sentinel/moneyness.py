"""Moneyness-normalized option identity — the expiry-rollover solution.

The problem: on a new weekly's first day, strike 23,100 has no history.
Learning per-strike-level means every expiry the model is born blind.

The fix (the founder's "nil-hitty" insight): learn behaviour around the
ATM, not the strike level. An option's transferable identity is
``(option_type, atm_offset)`` where offset counts strike steps from the
at-the-money strike:

    offset  0  = ATM
    offset +1  = first strike OTM for a CALL (higher strike),
                 first strike OTM for a PUT  (lower strike)
    offset -1  = first strike ITM, etc.

We normalize so that **+offset is always "further OTM"** for both CE and
PE. That makes "+2 OTM call" and "+2 OTM put" directly comparable as
behaviours, which is exactly what the ledger and curator want.

A fresh ATM call on a brand-new weekly inherits everything learned about
ATM-call *behaviour under similar context* — strike levels are
forgotten, behaviour is remembered.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Strike step per underlying (NSE). NIFTY weeklies step by 50, BANKNIFTY
# by 100. Verified against the live chain at runtime; this is the prior.
DEFAULT_STRIKE_STEP = {"NIFTY": 50.0, "BANKNIFTY": 100.0, "FINNIFTY": 50.0}
UNIVERSE_HALF_WIDTH = 5          # ATM +/-5 -> 11 strikes -> 22 contracts


@dataclass(frozen=True)
class MoneynessKey:
    """The transferable, expiry-independent option identity.

    ``offset`` is signed strike steps from ATM, normalized so positive
    = further OTM for BOTH option types. ``label`` is the human form
    ("ATM_CE", "OTM+2_PE", "ITM-1_CE")."""
    underlying: str
    option_type: str          # CE | PE
    offset: int               # +OTM ... 0 ATM ... -ITM

    @property
    def label(self) -> str:
        if self.offset == 0:
            zone = "ATM"
        elif self.offset > 0:
            zone = f"OTM+{self.offset}"
        else:
            zone = f"ITM{self.offset}"
        return f"{zone}_{self.option_type}"

    def as_str(self) -> str:
        return f"{self.underlying}:{self.option_type}:{self.offset:+d}"


def atm_strike(spot: float, step: float) -> float:
    """Nearest strike to spot on the step grid."""
    if step <= 0:
        return spot
    return round(spot / step) * step


def strike_step_for(underlying: str,
                    observed_strikes: Optional[List[float]] = None) -> float:
    """Infer the step from the live chain when possible (robust to NSE
    changing conventions), else fall back to the prior."""
    if observed_strikes and len(observed_strikes) >= 2:
        s = sorted(set(observed_strikes))
        diffs = [round(b - a, 2) for a, b in zip(s, s[1:]) if b > a]
        if diffs:
            # Modal positive diff = the grid step.
            diffs.sort()
            return diffs[len(diffs) // 2]
    return DEFAULT_STRIKE_STEP.get(underlying.upper(), 50.0)


def moneyness_key(underlying: str, option_type: str, strike: float,
                  spot: float, step: float) -> MoneynessKey:
    """Map a concrete (strike, type) to its ATM-relative identity.

    For a CALL: higher strike than ATM = OTM = positive offset.
    For a PUT:  lower  strike than ATM = OTM = positive offset.
    The sign flip on puts is what makes "+2 OTM" mean the same *kind*
    of bet on both sides.
    """
    atm = atm_strike(spot, step)
    raw_steps = round((strike - atm) / step) if step > 0 else 0
    if option_type == "CE":
        offset = raw_steps
    else:                       # PE: OTM is BELOW spot -> flip sign
        offset = -raw_steps
    return MoneynessKey(underlying.upper(), option_type, int(offset))


@dataclass
class ContractRef:
    """A live contract resolved into the universe."""
    tradingsymbol: str
    underlying: str
    option_type: str
    strike: float
    key: MoneynessKey


def build_universe(underlying: str, spot: float,
                   chain: List[Tuple[str, str, float]],
                   half_width: int = UNIVERSE_HALF_WIDTH,
                   ) -> Dict[str, ContractRef]:
    """Select the ATM +/- half_width universe from a chain.

    ``chain`` is a list of (tradingsymbol, option_type, strike) for the
    target underlying + nearest expiry. Returns {moneyness_str ->
    ContractRef} for offsets in [-half_width, +half_width], both types
    — the founder's "5 up, 5 down, both CE and PE" = 22 contracts.

    Missing offsets (illiquid gaps in the chain) are simply absent;
    the caller treats absence as "not tradable right now", never crashes.
    """
    strikes = [c[2] for c in chain]
    step = strike_step_for(underlying, strikes)
    out: Dict[str, ContractRef] = {}
    for sym, otype, strike in chain:
        if otype not in ("CE", "PE"):
            continue
        key = moneyness_key(underlying, otype, strike, spot, step)
        if abs(key.offset) > half_width:
            continue
        ref = ContractRef(sym, underlying.upper(), otype, float(strike), key)
        # If two strikes map to the same key (shouldn't on a clean grid),
        # keep the one closest to its intended slot.
        existing = out.get(key.as_str())
        if existing is None:
            out[key.as_str()] = ref
        else:
            atm = atm_strike(spot, step)
            want = atm + (key.offset if otype == "CE" else -key.offset) * step
            if abs(strike - want) < abs(existing.strike - want):
                out[key.as_str()] = ref
    return out


def universe_symbols(universe: Dict[str, ContractRef]) -> List[str]:
    return [ref.tradingsymbol for ref in universe.values()]
