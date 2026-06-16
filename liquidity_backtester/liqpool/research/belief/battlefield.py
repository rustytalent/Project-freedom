"""Multi-strike battlefield (Belief Engine, Phase 5).

The founder's principle: "having multiple-strike confirmation is a very
good thing. Because if we missed one activity — say one strike did not
move expected — the strikes above and below telling the same story is
itself the story."

This module stacks the per-slot reads (Phase 4) into a battlefield view
of all 22 contracts (ATM ±5 × CE/PE). The output answers:

  * Is the CE rail broadly strong / weak / mixed?  (call-side belief)
  * Is the PE rail broadly strong / weak / mixed?  (put-side belief)
  * Is one strike doing something the others aren't? (single-strike distortion)
  * Where does the abnormality concentrate — ATM, ITM, OTM?
  * Are the two rails AGREEING on direction (bullish/bearish), or
    showing volatility expansion / dirty data?

Each slot contributes its own deviation-of-deviation (Phase 4 ``dod_z``)
which is already in slot-specific robust-σ units, so they aggregate
cleanly without per-slot rescaling. We DO weight by behavior:

  * future-like deep-ITM carries more directional information per σ
    (delta is close to ±1, so its residual is more meaningful as belief)
  * lottery deep-OTM carries less (cheap, convex, noisier)
  * the weighting is gentle (0.4..1.0) — a strong signal on a lottery
    strike is still worth seeing, just discounted.

The battlefield refuses to call agreement on a single-strike spike. A
``dispersion_score`` measures how concentrated abnormality is in one
slot vs spread across the rail; high dispersion → single-strike
distortion → trust the read less.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .moneyness import MoneynessSlot


# Behavior-band weights. Future-like deep-ITM carries the most directional
# information per σ of residual; lottery OTM carries the least.
_BEHAVIOR_WEIGHT: Dict[str, float] = {
    "future_like": 1.00,
    "directional_itm": 0.95,
    "gamma_atm": 0.90,
    "convex_otm": 0.65,
    "lottery_otm": 0.40,
}

# Rail-state labels.
RAIL_STRONG_BULL = "strong_bull"   # CE rail broadly strong / PE rail broadly weak
RAIL_STRONG_BEAR = "strong_bear"   # PE rail broadly strong / CE rail broadly weak
RAIL_MIXED = "mixed"               # rail abnormal but not directionally clean
RAIL_QUIET = "quiet"               # nothing notable on this rail

# Battlefield-verdict labels (the cross-rail read).
FIELD_BULLISH_AGREEMENT = "bullish_agreement"   # CE strong AND PE weak
FIELD_BEARISH_AGREEMENT = "bearish_agreement"   # PE strong AND CE weak
FIELD_VOL_EXPANSION = "vol_expansion"            # BOTH rails strong (common shock)
FIELD_VOL_CONTRACTION = "vol_contraction"        # BOTH rails weak
FIELD_SINGLE_DISTORTION = "single_distortion"    # high dispersion — one strike only
FIELD_QUIET = "quiet"                            # no signal


@dataclass
class BattlefieldConfig:
    """Knobs for the battlefield aggregation."""
    # Minimum |dod_z| on the contrarian rail to count a slot as "abnormal"
    # for the dispersion calculation. Slots below this are noise.
    slot_abnormal_z: float = 1.0
    # Rail-level threshold for the weighted-mean |dod_z| to be considered
    # broadly strong / weak. Below this the rail is "quiet".
    rail_active_threshold: float = 0.75
    # Dispersion ceiling. Above this the rail is judged as a single-strike
    # distortion rather than broad agreement.
    max_dispersion_for_agreement: float = 0.65
    # Cross-rail threshold for declaring a directional agreement — the
    # absolute difference of the two signed means must exceed this.
    agreement_gap_z: float = 1.0
    eps: float = 1e-9


@dataclass(frozen=True)
class SlotReading:
    """One slot's current read fed into the battlefield."""
    slot: MoneynessSlot
    dod_z: float
    is_abnormal: bool = False
    friendliness: float = 1.0
    acceptance: str = "normal"


@dataclass(frozen=True)
class RailSummary:
    """One rail's (all CE slots OR all PE slots) summary."""
    side: str                       # "CE" or "PE"
    n_slots: int
    weighted_mean_signed_z: float   # signed weighted mean of dod_z
    weighted_mean_abs_z: float      # weighted mean of |dod_z|
    fraction_abnormal: float        # fraction of slots beyond slot_abnormal_z
    dispersion_score: float         # 0..1, higher = abnormality concentrated in few slots
    n_abnormal_strong: int          # slots with dod_z > +slot_abnormal_z (strong)
    n_abnormal_weak: int            # slots with dod_z < -slot_abnormal_z (weak)
    epicenter_label: str            # the label of the slot with max |dod_z|
    epicenter_level: int            # signed level of that slot
    state: str                      # RAIL_STRONG_BULL / RAIL_STRONG_BEAR / RAIL_MIXED / RAIL_QUIET

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class BattlefieldSnapshot:
    """One bar's reading of the whole battlefield."""
    ce_rail: RailSummary
    pe_rail: RailSummary
    verdict: str                    # one of FIELD_* labels
    direction: int                  # +1 bullish, -1 bearish, 0 none
    confidence: float               # [0,1] — agreement strength × (1 - dispersion)
    note: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ce_rail": self.ce_rail.to_dict(),
            "pe_rail": self.pe_rail.to_dict(),
            "verdict": self.verdict,
            "direction": int(self.direction),
            "confidence": float(self.confidence),
            "note": self.note,
        }


def _gini_like_dispersion(weights: np.ndarray) -> float:
    """Dispersion ∈ [0,1]: 0 = perfectly even, 1 = all mass on one slot.

    Defined as ``1 − (Σwᵢ)² / (n·Σwᵢ²)`` complement-ish — actually we use
    the standard Herfindahl/concentration measure normalised to [0,1]:
    ``H = Σ(pᵢ²)`` where ``pᵢ = wᵢ/Σw``; then ``dispersion = (n·H − 1)/(n − 1)``.

    Even allocation → H = 1/n → dispersion = 0.
    One-slot allocation → H = 1 → dispersion = 1.
    """
    w = np.asarray(weights, dtype=float)
    w = np.where(w < 0, 0.0, w)
    s = float(w.sum())
    n = int(w.size)
    if n <= 1 or s <= 1e-12:
        return 0.0
    p = w / s
    h = float((p * p).sum())
    return max(0.0, min(1.0, (n * h - 1.0) / (n - 1)))


def _rail_summary(side: str, readings: Sequence[SlotReading],
                  cfg: BattlefieldConfig) -> RailSummary:
    own = [r for r in readings if r.slot.option_type == side]
    n = len(own)
    if n == 0:
        return RailSummary(
            side=side, n_slots=0, weighted_mean_signed_z=0.0,
            weighted_mean_abs_z=0.0, fraction_abnormal=0.0,
            dispersion_score=0.0, n_abnormal_strong=0, n_abnormal_weak=0,
            epicenter_label="", epicenter_level=0, state=RAIL_QUIET,
        )
    weights = np.array([_BEHAVIOR_WEIGHT.get(r.slot.behavior, 0.5) for r in own])
    z = np.array([r.dod_z for r in own])
    abs_z = np.abs(z)
    # Weighted means.
    w_total = max(weights.sum(), cfg.eps)
    weighted_signed = float((weights * z).sum() / w_total)
    weighted_abs = float((weights * abs_z).sum() / w_total)
    # Abnormal counts.
    is_abn = abs_z > cfg.slot_abnormal_z
    strong = int(((z > cfg.slot_abnormal_z) & is_abn).sum())
    weak = int(((z < -cfg.slot_abnormal_z) & is_abn).sum())
    fraction_abn = float(is_abn.sum()) / n
    # Dispersion measured over the |dod_z| contributions to the rail.
    contributions = weights * abs_z
    dispersion = _gini_like_dispersion(contributions)
    # Epicenter — slot of maximum |dod_z|.
    epi_idx = int(np.argmax(abs_z))
    epicenter_label = own[epi_idx].slot.label
    epicenter_level = own[epi_idx].slot.level
    # State.
    if weighted_abs < cfg.rail_active_threshold:
        state = RAIL_QUIET
    elif dispersion > cfg.max_dispersion_for_agreement:
        state = RAIL_MIXED
    elif weighted_signed > cfg.rail_active_threshold:
        state = RAIL_STRONG_BULL if side == "CE" else RAIL_STRONG_BEAR
    elif weighted_signed < -cfg.rail_active_threshold:
        # Reverse — CE rail weak collectively reads as "bearish belief on CE
        # side" (calls being sold harder than delta would say). For PE rail
        # weak → bullish (puts being sold harder than delta would say).
        state = RAIL_STRONG_BEAR if side == "CE" else RAIL_STRONG_BULL
    else:
        state = RAIL_MIXED
    return RailSummary(
        side=side, n_slots=n,
        weighted_mean_signed_z=weighted_signed,
        weighted_mean_abs_z=weighted_abs,
        fraction_abnormal=fraction_abn,
        dispersion_score=dispersion,
        n_abnormal_strong=strong,
        n_abnormal_weak=weak,
        epicenter_label=epicenter_label,
        epicenter_level=epicenter_level,
        state=state,
    )


def battlefield_snapshot(readings: Sequence[SlotReading],
                         cfg: Optional[BattlefieldConfig] = None,
                         ) -> BattlefieldSnapshot:
    """Aggregate per-slot reads into one battlefield verdict.

    The verdict is the cross-rail read:
      * CE strong + PE weak → bullish_agreement
      * PE strong + CE weak → bearish_agreement
      * BOTH rails strong → vol_expansion (common shock / straddle bid)
      * BOTH rails quiet  → vol_contraction
      * dispersion on either rail too high → single_distortion
      * otherwise quiet
    """
    cfg = cfg or BattlefieldConfig()
    ce = _rail_summary("CE", readings, cfg)
    pe = _rail_summary("PE", readings, cfg)

    # Direction signed for the cross-rail comparison. A bullish field means
    # CE side biased strong (positive signed mean), PE side biased weak
    # (negative signed mean — put residuals decaying faster than fair).
    ce_signed = ce.weighted_mean_signed_z
    pe_signed = pe.weighted_mean_signed_z
    cross_gap = ce_signed - pe_signed   # >0 favours bullish, <0 favours bearish

    # Single-strike distortion takes priority — even if the rail averages
    # look directional, if one slot is carrying the rail, do not trust it.
    # Distortion fires when EITHER:
    #   * the rail is active (weighted_mean_abs_z above threshold) AND
    #     dispersion is too high to call broad agreement, OR
    #   * the rail's max |dod_z| is above the slot-abnormal threshold AND
    #     dispersion is high (a single strike screams while the rest sits
    #     quiet — the quiet rail mean is itself the distortion fingerprint).
    def _is_distorted(rail: RailSummary) -> bool:
        if rail.dispersion_score <= cfg.max_dispersion_for_agreement:
            return False
        if rail.weighted_mean_abs_z > cfg.rail_active_threshold:
            return True
        # Recover the max |z| from the rail's epicenter info if needed.
        max_abs = max((abs(r.dod_z) for r in readings
                       if r.slot.option_type == rail.side), default=0.0)
        return max_abs > cfg.slot_abnormal_z
    ce_distorted = _is_distorted(ce)
    pe_distorted = _is_distorted(pe)
    if ce_distorted or pe_distorted:
        verdict = FIELD_SINGLE_DISTORTION
        direction = 0
        # Confidence reflects how clearly the distortion is one-strike.
        confidence = max(ce.dispersion_score, pe.dispersion_score)
        epi = (ce.epicenter_label if ce.dispersion_score >= pe.dispersion_score
               else pe.epicenter_label)
        note = (f"single-strike distortion at {epi}; "
                f"do not trust rail aggregation")
        return BattlefieldSnapshot(ce, pe, verdict, direction, confidence, note)

    if ce.state == RAIL_QUIET and pe.state == RAIL_QUIET:
        return BattlefieldSnapshot(ce, pe, FIELD_QUIET, 0, 0.0,
                                   "both rails quiet — no actionable read")

    if (ce.weighted_mean_abs_z > cfg.rail_active_threshold
            and pe.weighted_mean_abs_z > cfg.rail_active_threshold
            and ce_signed * pe_signed > 0):
        # Both rails moving the SAME signed direction → common-mode IV shock.
        verdict = FIELD_VOL_EXPANSION
        direction = 0
        confidence = min(1.0, 0.5 * (abs(ce_signed) + abs(pe_signed)) / 3.0)
        note = (f"both rails active in same direction "
                f"(CE {ce_signed:+.2f}, PE {pe_signed:+.2f}) — "
                f"common IV shock / straddle bid")
        return BattlefieldSnapshot(ce, pe, verdict, direction, confidence, note)

    if abs(cross_gap) > cfg.agreement_gap_z:
        direction = +1 if cross_gap > 0 else -1
        verdict = (FIELD_BULLISH_AGREEMENT if direction > 0
                   else FIELD_BEARISH_AGREEMENT)
        # Confidence scales with the cross-gap, dampened by the worse
        # rail's dispersion.
        max_disp = max(ce.dispersion_score, pe.dispersion_score)
        confidence = min(1.0, abs(cross_gap) / 4.0) * (1.0 - max_disp)
        note = (f"CE {ce_signed:+.2f} vs PE {pe_signed:+.2f} — "
                f"rails agree {('bullish' if direction>0 else 'bearish')}; "
                f"CE epicenter={ce.epicenter_label}, PE epicenter={pe.epicenter_label}")
        return BattlefieldSnapshot(ce, pe, verdict, direction, confidence, note)

    return BattlefieldSnapshot(ce, pe, FIELD_QUIET, 0, 0.0,
                               "rails not aligned strongly enough")


def make_readings(slots_and_z: Sequence[Any]) -> List[SlotReading]:
    """Convenience builder: accepts a sequence of (slot, dod_z) tuples or
    full SlotReading objects, returns a list of SlotReading."""
    out: List[SlotReading] = []
    for item in slots_and_z:
        if isinstance(item, SlotReading):
            out.append(item)
            continue
        slot, z = item[0], float(item[1])
        out.append(SlotReading(slot=slot, dod_z=z,
                                is_abnormal=abs(z) > 1.0))
    return out
