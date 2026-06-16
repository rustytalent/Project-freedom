"""IV / skew pressure classifier (Belief Engine, Phase 5).

The founder's correction: "IV shock is not model failure — it should be
a STATE." The engine needs to distinguish:

  * **Common IV shock**: BOTH legs strengthen (or both weaken). Volatility
    expansion / straddle bid / event risk priced in. Direction unclear —
    do NOT interpret as bullish or bearish.

  * **Directional skew shock**: one rail strengthens while the other
    weakens — the canonical bullish or bearish positioning.

  * **Liquidity distortion**: spread state is dangerous / friendliness
    has collapsed — the premium signal is not trustworthy regardless of
    direction. Engineered to subsume any other reading.

  * **Dirty data**: too many slots have invalid marks (Phase 2 flagged
    them). Refuse to act.

This is the LAST layer before the Phase-6 thesis memory. It takes the
battlefield rail summaries (Phase 5 ``battlefield.py``) plus a battlefield
mark-quality + spread-friendliness aggregate, and emits one IV state with
a human-readable note. The state machine intentionally OVERRIDES the
battlefield's directional reading when data quality is poor — a clean
"bullish_agreement" verdict is worthless if half the rail has wide spreads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from .battlefield import (
    BattlefieldSnapshot,
    FIELD_BEARISH_AGREEMENT,
    FIELD_BULLISH_AGREEMENT,
    FIELD_QUIET,
    FIELD_SINGLE_DISTORTION,
    FIELD_VOL_CONTRACTION,
    FIELD_VOL_EXPANSION,
    SlotReading,
)


# IV state labels.
IV_DIRECTIONAL_BULL = "directional_bull"   # one-sided positioning, bullish
IV_DIRECTIONAL_BEAR = "directional_bear"   # one-sided positioning, bearish
IV_COMMON_SHOCK = "common_shock"           # both rails strong same way → vol expansion
IV_VOL_CONTRACTION = "vol_contraction"     # both rails quiet → calm
IV_LIQUIDITY_DISTORTION = "liquidity_distortion"  # spread state dangerous
IV_DIRTY_DATA = "dirty_data"               # too many invalid marks
IV_NEUTRAL = "neutral"                      # nothing actionable


@dataclass
class IVStateConfig:
    """Knobs for the IV/skew state classifier."""
    # Below this fraction of slots reporting clean (microprice/mid) marks,
    # we declare dirty data and refuse to read direction.
    min_clean_mark_fraction: float = 0.65
    # Average friendliness below this → liquidity distortion dominates.
    min_avg_friendliness: float = 0.40
    # Fraction of rail slots whose acceptance state is "defended" or
    # "rejected" — used to flag a one-rail directional skew shock even if
    # the battlefield itself didn't call agreement.
    skew_acceptance_fraction: float = 0.35
    eps: float = 1e-9


@dataclass(frozen=True)
class IVState:
    """The IV/skew state machine output for one bar."""
    state: str
    direction: int                  # +1 / -1 / 0
    confidence: float               # [0,1]
    avg_friendliness: float
    clean_mark_fraction: float
    ce_defended_fraction: float
    pe_defended_fraction: float
    ce_rejected_fraction: float
    pe_rejected_fraction: float
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _fraction_with_acceptance(readings: Iterable[SlotReading], side: str,
                              acceptance: str) -> float:
    side_slots = [r for r in readings if r.slot.option_type == side]
    if not side_slots:
        return 0.0
    n = sum(1 for r in side_slots if r.acceptance == acceptance)
    return float(n) / len(side_slots)


def classify_iv_state(snapshot: BattlefieldSnapshot,
                      readings: Sequence[SlotReading],
                      *,
                      clean_mark_fraction: Optional[float] = None,
                      cfg: Optional[IVStateConfig] = None) -> IVState:
    """Classify the IV/skew state for the current bar.

    ``snapshot`` is the Phase-5 battlefield reading. ``readings`` is the
    per-slot input feeding it — they carry the friendliness and acceptance
    state we need for the data-quality layer. ``clean_mark_fraction`` is
    the fraction of slots whose Phase-2 mark came from microprice/mid (not
    last_valid/ltp/invalid); the caller computes it.

    Returns one of the seven IV states with direction + confidence."""
    cfg = cfg or IVStateConfig()

    n = len(readings)
    if n == 0:
        return IVState(
            state=IV_DIRTY_DATA, direction=0, confidence=0.0,
            avg_friendliness=0.0, clean_mark_fraction=0.0,
            ce_defended_fraction=0.0, pe_defended_fraction=0.0,
            ce_rejected_fraction=0.0, pe_rejected_fraction=0.0,
            note="no slot readings provided",
        )

    avg_friend = float(np.mean([r.friendliness for r in readings]))
    cmf = float(clean_mark_fraction) if clean_mark_fraction is not None else 1.0
    ce_def = _fraction_with_acceptance(readings, "CE", "defended")
    pe_def = _fraction_with_acceptance(readings, "PE", "defended")
    ce_rej = _fraction_with_acceptance(readings, "CE", "rejected")
    pe_rej = _fraction_with_acceptance(readings, "PE", "rejected")

    # 1. Dirty data dominates.
    if cmf < cfg.min_clean_mark_fraction:
        return IVState(
            state=IV_DIRTY_DATA, direction=0, confidence=1.0 - cmf,
            avg_friendliness=avg_friend, clean_mark_fraction=cmf,
            ce_defended_fraction=ce_def, pe_defended_fraction=pe_def,
            ce_rejected_fraction=ce_rej, pe_rejected_fraction=pe_rej,
            note=(f"only {cmf:.0%} of slots have clean marks "
                  f"(< {cfg.min_clean_mark_fraction:.0%}) — refuse to read direction"),
        )

    # 2. Liquidity distortion next — wide spreads kill the signal even
    #    when the battlefield itself looks directional.
    if avg_friend < cfg.min_avg_friendliness:
        return IVState(
            state=IV_LIQUIDITY_DISTORTION, direction=0,
            confidence=1.0 - (avg_friend / max(cfg.min_avg_friendliness, cfg.eps)),
            avg_friendliness=avg_friend, clean_mark_fraction=cmf,
            ce_defended_fraction=ce_def, pe_defended_fraction=pe_def,
            ce_rejected_fraction=ce_rej, pe_rejected_fraction=pe_rej,
            note=(f"avg spread friendliness {avg_friend:.2f} below "
                  f"{cfg.min_avg_friendliness:.2f} — signals not trustworthy"),
        )

    # 3. Common IV shock — both rails moving the SAME way.
    if snapshot.verdict == FIELD_VOL_EXPANSION:
        return IVState(
            state=IV_COMMON_SHOCK, direction=0,
            confidence=snapshot.confidence,
            avg_friendliness=avg_friend, clean_mark_fraction=cmf,
            ce_defended_fraction=ce_def, pe_defended_fraction=pe_def,
            ce_rejected_fraction=ce_rej, pe_rejected_fraction=pe_rej,
            note=("both rails active in same direction — common IV shock / "
                  "straddle bid; do NOT read directionally"),
        )

    # 4. Battlefield-declared directional agreement.
    if snapshot.verdict == FIELD_BULLISH_AGREEMENT:
        return IVState(
            state=IV_DIRECTIONAL_BULL, direction=+1,
            confidence=snapshot.confidence,
            avg_friendliness=avg_friend, clean_mark_fraction=cmf,
            ce_defended_fraction=ce_def, pe_defended_fraction=pe_def,
            ce_rejected_fraction=ce_rej, pe_rejected_fraction=pe_rej,
            note=snapshot.note,
        )
    if snapshot.verdict == FIELD_BEARISH_AGREEMENT:
        return IVState(
            state=IV_DIRECTIONAL_BEAR, direction=-1,
            confidence=snapshot.confidence,
            avg_friendliness=avg_friend, clean_mark_fraction=cmf,
            ce_defended_fraction=ce_def, pe_defended_fraction=pe_def,
            ce_rejected_fraction=ce_rej, pe_rejected_fraction=pe_rej,
            note=snapshot.note,
        )

    # 5. Acceptance-driven skew shock — even when the battlefield aggregation
    #    didn't call agreement, a broad acceptance pattern across a rail
    #    (calls being defended in many strikes, puts being rejected in many)
    #    is itself the directional skew read. Checked BEFORE vol_contraction
    #    because acceptance state can fire on bars whose rail-mean dod_z is
    #    quiet — the founder's whole point is that broad acceptance is a
    #    directional signal even when the per-bar |z| isn't.
    if ce_def >= cfg.skew_acceptance_fraction and pe_rej >= cfg.skew_acceptance_fraction:
        return IVState(
            state=IV_DIRECTIONAL_BULL, direction=+1,
            confidence=min(1.0, 0.5 * (ce_def + pe_rej)),
            avg_friendliness=avg_friend, clean_mark_fraction=cmf,
            ce_defended_fraction=ce_def, pe_defended_fraction=pe_def,
            ce_rejected_fraction=ce_rej, pe_rejected_fraction=pe_rej,
            note=(f"acceptance skew: {ce_def:.0%} CE defended, "
                  f"{pe_rej:.0%} PE rejected — bullish belief skew"),
        )
    if pe_def >= cfg.skew_acceptance_fraction and ce_rej >= cfg.skew_acceptance_fraction:
        return IVState(
            state=IV_DIRECTIONAL_BEAR, direction=-1,
            confidence=min(1.0, 0.5 * (pe_def + ce_rej)),
            avg_friendliness=avg_friend, clean_mark_fraction=cmf,
            ce_defended_fraction=ce_def, pe_defended_fraction=pe_def,
            ce_rejected_fraction=ce_rej, pe_rejected_fraction=pe_rej,
            note=(f"acceptance skew: {pe_def:.0%} PE defended, "
                  f"{ce_rej:.0%} CE rejected — bearish belief skew"),
        )

    # 6. Vol contraction — both rails quiet, no directional acceptance skew,
    #    spreads clean.
    if snapshot.verdict == FIELD_QUIET and avg_friend > 0.7:
        return IVState(
            state=IV_VOL_CONTRACTION, direction=0,
            confidence=min(1.0, avg_friend),
            avg_friendliness=avg_friend, clean_mark_fraction=cmf,
            ce_defended_fraction=ce_def, pe_defended_fraction=pe_def,
            ce_rejected_fraction=ce_rej, pe_rejected_fraction=pe_rej,
            note="both rails quiet on clean spread — calm market",
        )

    # 7. Default neutral.
    return IVState(
        state=IV_NEUTRAL, direction=0, confidence=0.0,
        avg_friendliness=avg_friend, clean_mark_fraction=cmf,
        ce_defended_fraction=ce_def, pe_defended_fraction=pe_def,
        ce_rejected_fraction=ce_rej, pe_rejected_fraction=pe_rej,
        note="nothing actionable on this bar",
    )
