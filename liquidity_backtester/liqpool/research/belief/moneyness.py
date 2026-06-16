"""Moneyness identity layer (Belief Engine, Phase 3).

The founder's correction: "you just can't treat each of the 22 contracts
as equal." Each strike slot has its own personality, governed by its
Greeks:

  * Deep ITM (|delta| → 1): behaves like the underlying future. Low gamma,
    low vega sensitivity, premium tracks spot nearly 1:1. The founder's
    rule of thumb — "from the second ITM they move like futures."
  * Near ITM: directional but still carries some optionality.
  * ATM (|delta| ≈ 0.5): maximum gamma and vega — the most reactive slot,
    the most information-rich, but also where the residual swings hardest.
  * OTM (|delta| < 0.4): cheap, convex, noisier; premium responds
    non-linearly and is the most distorted by spread.
  * Deep OTM (|delta| → 0): lottery tickets; mostly noise and spread.

Why this matters: you cannot call a contract "abnormal" until you know
what NORMAL is for that exact slot. A ₹0.70 residual on an ATM option is
ordinary; on a 4-OTM it is enormous. And — the founder's sharp use case —
when a 2-ITM option STOPS behaving like a future and starts behaving like
an ATM/OTM (its effective delta drops toward 0.5), that itself is a tell
that something is being engineered in that strike.

This module:
  * classifies a (spot, strike, option_type) into a signed moneyness level
    (negative = ITM depth, 0 = ATM, positive = OTM depth);
  * supplies the EXPECTED |delta| baseline per level (a documented
    weekly-NIFTY-ish profile, extrapolated past ±5);
  * labels the behavioral regime of each slot;
  * detects identity anomalies — a realized effective delta that matches a
    DIFFERENT slot's expectation than the one the strike actually occupies.

The expected-delta numbers are deliberately approximate. The engine uses
them as a *reference shape*, not a pricing model — the live residual layer
calibrates the real effective delta per slot from the tape. Where a true
Black-Scholes / SVI delta is available it can be passed in to override.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


# Expected |delta| by signed moneyness level for an ATM-ish weekly option.
# Negative level = ITM depth, 0 = ATM, positive = OTM depth.
_ABS_DELTA_BY_LEVEL: Dict[int, float] = {
    -5: 0.93, -4: 0.88, -3: 0.82, -2: 0.73, -1: 0.62,
    0: 0.50,
    1: 0.38, 2: 0.27, 3: 0.18, 4: 0.12, 5: 0.08,
}

# Behavioral-regime thresholds on |delta|.
_FUTURE_LIKE_DELTA = 0.85       # deep ITM ≈ future
_DIRECTIONAL_ITM_DELTA = 0.60   # near ITM — directional, some optionality
_GAMMA_ATM_LOW = 0.40           # ATM band [0.40, 0.60): max gamma/vega
_CONVEX_OTM_LOW = 0.18          # OTM band [0.18, 0.40): convex, noisier
                                # below 0.18 → lottery / pure noise


def expected_abs_delta(level: int) -> float:
    """Expected |delta| for a signed moneyness level.

    Levels within ±5 use the calibrated table; beyond that we extrapolate
    monotonically toward the asymptotes (0.99 deep ITM, 0.02 deep OTM)."""
    if level in _ABS_DELTA_BY_LEVEL:
        return _ABS_DELTA_BY_LEVEL[level]
    if level < -5:
        # Each extra strike of ITM depth adds a shrinking step toward 0.99.
        steps = -level - 5
        return min(0.99, _ABS_DELTA_BY_LEVEL[-5] + 0.012 * steps)
    # level > 5 → deeper OTM, decay toward 0.02.
    steps = level - 5
    return max(0.02, _ABS_DELTA_BY_LEVEL[5] - 0.012 * steps)


def moneyness_behavior(abs_delta: float) -> str:
    """Map |delta| to a behavioral-regime label."""
    if abs_delta >= _FUTURE_LIKE_DELTA:
        return "future_like"
    if abs_delta >= _DIRECTIONAL_ITM_DELTA:
        return "directional_itm"
    if abs_delta >= _GAMMA_ATM_LOW:
        return "gamma_atm"
    if abs_delta >= _CONVEX_OTM_LOW:
        return "convex_otm"
    return "lottery_otm"


def _level_label(option_type: str, level: int) -> str:
    ot = option_type.upper()
    if level == 0:
        return f"{ot}_ATM"
    if level < 0:
        return f"{ot}_ITM{abs(level)}"
    return f"{ot}_OTM{level}"


@dataclass(frozen=True)
class MoneynessSlot:
    """The identity of one contract on the battlefield."""
    option_type: str          # "CE" or "PE"
    strike: float
    level: int                # signed: <0 ITM, 0 ATM, >0 OTM
    label: str                # e.g. "CE_ATM", "PE_ITM2"
    expected_abs_delta: float
    expected_signed_delta: float   # +abs for CE, -abs for PE
    behavior: str
    is_future_like: bool

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def classify_moneyness(spot: float, strike: float, option_type: str,
                       strike_step: float = 50.0) -> MoneynessSlot:
    """Classify one contract into a moneyness slot.

    For a CALL, strikes above spot are OTM (positive level), below are ITM
    (negative). For a PUT the direction is inverted: strikes above spot are
    ITM, below are OTM.
    """
    ot = option_type.upper()
    if ot not in ("CE", "PE"):
        raise ValueError(f"option_type must be CE or PE, got {option_type!r}")
    if strike_step <= 0:
        raise ValueError("strike_step must be positive")
    atm_strike = round(spot / strike_step) * strike_step
    strikes_above = int(round((strike - atm_strike) / strike_step))
    # CE: above ATM = OTM (positive). PE: above ATM = ITM (negative).
    level = strikes_above if ot == "CE" else -strikes_above
    abs_delta = expected_abs_delta(level)
    signed = abs_delta if ot == "CE" else -abs_delta
    behavior = moneyness_behavior(abs_delta)
    return MoneynessSlot(
        option_type=ot,
        strike=float(strike),
        level=int(level),
        label=_level_label(ot, level),
        expected_abs_delta=abs_delta,
        expected_signed_delta=signed,
        behavior=behavior,
        is_future_like=abs_delta >= _FUTURE_LIKE_DELTA,
    )


def build_chain_slots(spot: float,
                      contracts: Sequence[Tuple[float, str]],
                      strike_step: float = 50.0) -> List[MoneynessSlot]:
    """Classify a whole battlefield. ``contracts`` is a sequence of
    (strike, option_type) pairs — typically the 22 contracts of ATM ±5 ×
    CE/PE. Returns slots sorted by option_type then level."""
    slots = [classify_moneyness(spot, k, ot, strike_step) for k, ot in contracts]
    return sorted(slots, key=lambda s: (s.option_type, s.level))


def detect_identity_anomaly(slot: MoneynessSlot,
                            realized_abs_delta: float,
                            *,
                            tolerance: float = 0.12) -> Dict[str, Any]:
    """The founder's "ITM not behaving like ITM" detector.

    Compares the realized effective |delta| (estimated live from the tape)
    against the slot's expected |delta|. If the realized value matches a
    DIFFERENT level's expectation by more than ``tolerance``, the slot is
    flagged as impersonating that other regime — a sign that something is
    being engineered in this strike.

    Returns a dict with:
      * ``is_anomaly`` — bool
      * ``expected_abs_delta`` / ``realized_abs_delta``
      * ``delta_gap`` — realized − expected (signed magnitude)
      * ``behaving_like`` — the behavioral label the realized delta implies
      * ``note`` — human-readable summary
    """
    expected = slot.expected_abs_delta
    realized = float(realized_abs_delta)
    gap = realized - expected
    behaving_like = moneyness_behavior(realized)
    is_anomaly = abs(gap) > tolerance and behaving_like != slot.behavior
    if not is_anomaly:
        note = (f"{slot.label} behaving normally "
                f"(|Δ|≈{realized:.2f} vs expected {expected:.2f})")
    else:
        direction = "stronger/more future-like" if gap > 0 else "weaker/more OTM-like"
        note = (f"{slot.label} ABNORMAL: |Δ|≈{realized:.2f} ({behaving_like}) "
                f"vs expected {expected:.2f} ({slot.behavior}) — {direction}")
    return {
        "label": slot.label,
        "is_anomaly": bool(is_anomaly),
        "expected_abs_delta": expected,
        "realized_abs_delta": realized,
        "delta_gap": float(gap),
        "expected_behavior": slot.behavior,
        "behaving_like": behaving_like,
        "note": note,
    }
