"""Causal Context Vector + LCS scalar.

Implements §3.2 (CCV), §3.3 (apply_causal_adjustments), and §3.4
(LCS scalar) of ``docs/options_executor_layered_conviction.md``.

The CCV is the rich object the executor and the LightGBM model
consume. The LCS scalar is a digest of the CCV for human display in
the brief.

Design discipline (encoded as tests):

  * Local collapse only. ``apply_causal_adjustments`` damps
    micro and pool when manipulation has a downstream effect on
    them. Macro and regime are NEVER touched — they are upstream
    in the layer DAG (§3.1).
  * Macro gate. When ``|macro| < 0.10`` the LCS scalar is exactly
    0.0; no other layer can manufacture conviction when the
    possibility space is empty.
  * Direction follows macro. When the gate is open, sign(LCS) =
    sign(macro). Other layers shape MAGNITUDE only.
  * Magnitude is the geometric mean of agreement factors. A near-
    zero agreement factor damps the whole cascade more than a high
    factor lifts it — the user's "string under tension" intuition.

The module is pure: no LightGBM, no I/O, no randomness. Every
function is deterministic and small enough to inspect at a glance.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Tunables — keep small; v1 v2 will tune these from outcomes.
# ---------------------------------------------------------------------------

MACRO_GATE_THRESHOLD: float = 0.10
MICRO_FORCING_MANIP_THRESHOLD: float = 0.40
POOL_DISTORTION_MANIP_THRESHOLD: float = 0.50
MICRO_FORCING_DAMPING: float = 0.6
POOL_DISTORTION_DAMPING: float = 0.6


# ---------------------------------------------------------------------------
# Causal Context Vector
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CCV:
    """The full causal state at one prediction or in-trade bar.

    Every layer score is in ``[-1, +1]`` with the sign convention that
    positive = bullish on the underlying, negative = bearish. The
    flags and adjusted scores are derived; the alignments are useful
    LightGBM input features that encode pairwise consistency.
    """
    # Raw per-layer scores
    macro_score: float
    regime_score: float
    pool_score: float
    options_score: float
    micro_score: float
    manipulation_score: float

    # Causal flags derived from inter-layer state
    micro_forced_flag: bool
    pool_distortion_flag: bool

    # Adjusted scores (post local-collapse)
    micro_organic_score: float
    pool_holding_strength: float

    # Cross-layer interaction features
    manip_micro_alignment: float
    manip_pool_alignment: float
    regime_pool_alignment: float
    macro_regime_alignment: float

    # Summary scalar for human display
    lcs: float

    def to_dict(self) -> Dict[str, float]:
        """Plain dict for JSON serialization / LightGBM input rows."""
        d = asdict(self)
        d["micro_forced_flag"] = int(self.micro_forced_flag)
        d["pool_distortion_flag"] = int(self.pool_distortion_flag)
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sign(x: float) -> float:
    """Numeric sign without numpy. Returns +1 / -1 / 0 — note that the
    cascade gate treats exactly-0 as "no direction"; downstream
    functions that branch on sign already guard against this."""
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _alignment(a: float, b: float) -> float:
    """Pairwise alignment feature in ``[-1, +1]``.

    The product ``sign(a) * sign(b) * min(|a|, |b|)`` is the
    smaller-magnitude-signed-by-agreement statistic used in §3.3.
    It is 0 whenever either layer is silent and gracefully shrinks
    when one layer is weak.
    """
    return _sign(a) * _sign(b) * min(abs(a), abs(b))


# ---------------------------------------------------------------------------
# Local-collapse adjustment (§3.3 of the executor doc)
# ---------------------------------------------------------------------------

def apply_causal_adjustments(raw: Dict[str, float]) -> CCV:
    """Compute the full CCV from raw layer scores.

    ``raw`` must carry keys: macro, regime, pool, options, micro,
    manipulation. Any missing key is treated as 0.0 (silent layer).

    The function applies the §3.3 adjustments LOCALLY: manipulation
    can damp micro and pool because the DAG places those layers
    downstream of manipulation. Macro and regime are untouched.
    """
    macro = float(raw.get("macro", 0.0))
    regime = float(raw.get("regime", 0.0))
    pool = float(raw.get("pool", 0.0))
    options = float(raw.get("options", 0.0))
    micro = float(raw.get("micro", 0.0))
    manip = float(raw.get("manipulation", 0.0))

    # Manipulation → microstructure (forced if aligned)
    micro_forced = False
    if (abs(manip) >= MICRO_FORCING_MANIP_THRESHOLD
            and _sign(manip) == _sign(micro)
            and _sign(micro) != 0):
        micro_forced = True
        explained_share = min(1.0, abs(manip))
        micro_organic = micro * (1.0 - MICRO_FORCING_DAMPING * explained_share)
    else:
        micro_organic = micro

    # Manipulation → pool (distortion if opposed)
    pool_distortion = False
    if (abs(manip) >= POOL_DISTORTION_MANIP_THRESHOLD
            and _sign(manip) != _sign(pool)
            and _sign(pool) != 0):
        pool_distortion = True
        pool_holding = pool * POOL_DISTORTION_DAMPING
    else:
        pool_holding = pool

    lcs = compute_lcs_scalar(macro, regime, pool_holding, options,
                             micro_organic, manip)

    return CCV(
        macro_score=macro,
        regime_score=regime,
        pool_score=pool,
        options_score=options,
        micro_score=micro,
        manipulation_score=manip,
        micro_forced_flag=micro_forced,
        pool_distortion_flag=pool_distortion,
        micro_organic_score=micro_organic,
        pool_holding_strength=pool_holding,
        manip_micro_alignment=_alignment(manip, micro),
        manip_pool_alignment=_alignment(manip, pool),
        regime_pool_alignment=_alignment(regime, pool),
        macro_regime_alignment=_alignment(macro, regime),
        lcs=lcs,
    )


# ---------------------------------------------------------------------------
# LCS scalar (§3.4 of the executor doc)
# ---------------------------------------------------------------------------

def compute_lcs_scalar(macro: float, regime: float, pool_holding: float,
                       options: float, micro_organic: float,
                       manipulation: float) -> float:
    """Compute the LCS scalar from the ADJUSTED layer state.

    Properties:
      * Returns 0.0 when ``|macro| < MACRO_GATE_THRESHOLD``.
      * Otherwise, ``sign(LCS) == sign(macro)``.
      * Magnitude = ``|macro| * geom_mean(agreement_factors)``
        clipped to ``[0, 1]``.
    """
    if abs(macro) < MACRO_GATE_THRESHOLD:
        return 0.0
    direction = 1.0 if macro > 0 else -1.0
    other_scores = [regime, pool_holding, options, micro_organic,
                    manipulation]
    factors = [1.0 + 0.5 * (direction * s) for s in other_scores]
    # Geometric mean.
    log_sum = sum(math.log(max(f, 1e-12)) for f in factors)
    geom_mean = math.exp(log_sum / len(factors))
    magnitude = _clamp(abs(macro) * geom_mean, 0.0, 1.0)
    return direction * magnitude


# ---------------------------------------------------------------------------
# Convenience: build CCV from a row dict (e.g. an outcome-log row)
# ---------------------------------------------------------------------------

def ccv_from_row(row: Dict[str, float],
                 layer_aliases: Optional[Dict[str, str]] = None) -> CCV:
    """Build a CCV from a wide row that may use long column names.

    ``layer_aliases`` lets the caller map e.g. ``"macro_score"`` →
    ``"macro"`` when the row's column names are verbose. Default
    aliasing matches the OptionsExpectedReturnModel's input feature
    catalogue.
    """
    default_aliases = {
        "macro_score": "macro",
        "regime_score": "regime",
        "pool_score": "pool",
        "options_score": "options",
        "micro_score": "micro",
        "manipulation_score": "manipulation",
    }
    aliases = layer_aliases or default_aliases
    raw: Dict[str, float] = {}
    for src, dst in aliases.items():
        if src in row:
            raw[dst] = float(row[src])
        elif dst in row:
            raw[dst] = float(row[dst])
    return apply_causal_adjustments(raw)
