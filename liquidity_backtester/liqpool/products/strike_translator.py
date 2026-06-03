"""Level-to-strike translator for NSE index options.

The proximity model produces predictions of the form "P(level L tested
within horizon H)" for raw price levels. Index option traders think in
strikes — quantised price levels with a fixed step (e.g. Nifty steps
every 50). This module maps the engine's output to that vocabulary.

v1 scope (per ``docs/COORDINATION.md`` NEXT UP #5):

  * Pure level-to-strike mapping. Given an index name and a level,
    return the nearest strike(s) with the original probability.
  * NO Greeks. NO IV. NO PCR. NO OI. Those are higher-tier additions
    once paying customers exist and have asked for them.
  * Five indexes covered: Nifty 50, Bank Nifty, Fin Nifty, Nifty
    Midcap Select, Sensex. Configurable per-index strike step and
    lot size constants (current as of 2026; document the source so
    future drift is auditable).

Public API:

  * ``INDEX_CONFIGS`` — per-index dict of (strike_step, lot_size,
    instrument_root). Single source of truth for the strike grid.
  * ``nearest_strikes(level, index_name, k=2)`` — return the k strikes
    surrounding the level.
  * ``translate_proximity(predictions, index_name)`` — turn a list of
    (level, p_test) into a list of (strike, side, p_test) for the
    options_suitability section of the brief.

Hard constraint: the output here is regime-language, not trade-language.
We say "the 24500 strike has P(tested) = 0.82" — never "buy the 24500 CE".
The brief renderer's tipster-vocabulary guardrail enforces this at
render time as a second line of defence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class IndexConfig:
    """Per-index strike-grid configuration.

    Source of truth for strike_step and lot_size: NSE F&O segment as of
    2026-06. If these change, update here AND bump a note in the
    coordination doc — every brief published with stale constants is
    technically wrong for the trader translating it back to lots.
    """
    name: str
    instrument_root: str            # how the trader writes the option
                                    # symbol (e.g. NIFTY, BANKNIFTY)
    strike_step: float              # smallest gap between adjacent strikes
    lot_size: int                   # contracts per lot
    weekly_expiry_weekday: int      # 0=Mon, 1=Tue, ..., 3=Thu


INDEX_CONFIGS: Dict[str, IndexConfig] = {
    "NIFTY50": IndexConfig(
        name="NIFTY50", instrument_root="NIFTY",
        strike_step=50.0, lot_size=75,
        weekly_expiry_weekday=3,
    ),
    "BANKNIFTY": IndexConfig(
        name="BANKNIFTY", instrument_root="BANKNIFTY",
        strike_step=100.0, lot_size=30,
        weekly_expiry_weekday=3,
    ),
    "FINNIFTY": IndexConfig(
        name="FINNIFTY", instrument_root="FINNIFTY",
        strike_step=50.0, lot_size=65,
        weekly_expiry_weekday=3,
    ),
    "NIFTYMIDCAPSELECT": IndexConfig(
        name="NIFTYMIDCAPSELECT", instrument_root="MIDCPNIFTY",
        strike_step=25.0, lot_size=120,
        weekly_expiry_weekday=3,
    ),
    "SENSEX": IndexConfig(
        name="SENSEX", instrument_root="SENSEX",
        strike_step=100.0, lot_size=20,
        weekly_expiry_weekday=4,        # BSE Sensex options expire Fri
    ),
}


# ---------------------------------------------------------------------------
# Strike-grid math
# ---------------------------------------------------------------------------

def nearest_strike(level: float, index_name: str) -> float:
    """Snap a continuous level to the nearest strike on the index grid."""
    cfg = INDEX_CONFIGS.get(index_name)
    if cfg is None:
        raise KeyError(f"unknown index {index_name!r}; "
                        f"known: {sorted(INDEX_CONFIGS)}")
    step = cfg.strike_step
    return round(level / step) * step


def nearest_strikes(level: float, index_name: str, k: int = 2) -> List[float]:
    """Return the ``k`` strikes surrounding (and including) the nearest.

    For k=2: the strike at-or-below + the strike at-or-above.
    For k=4: those two plus the next pair out (one wider step each side).
    Used when the brief wants to flag a probability range, not a point.
    """
    cfg = INDEX_CONFIGS.get(index_name)
    if cfg is None:
        raise KeyError(f"unknown index {index_name!r}")
    step = cfg.strike_step
    base = round(level / step) * step
    # Build symmetric outward steps.
    half = max(1, k // 2)
    strikes = sorted(
        {base + i * step for i in range(-half, half + 1)}
    )
    return [s for s in strikes if s > 0][:k] if k < len(strikes) else strikes


# ---------------------------------------------------------------------------
# Proximity → options_suitability translator
# ---------------------------------------------------------------------------

@dataclass
class StrikeLevelInPlay:
    """One row of the options_suitability.strike_levels_in_play list."""
    strike: float
    p_test_today: float
    p_test_within_60min: float
    side_from_open: str             # above | below
    key_level_type: str             # demand_pool | supply_pool | poc | etc.
    underlying_level: float         # the original (unrounded) level


def translate_proximity(predictions: List[Dict[str, Any]],
                         index_name: str,
                         min_p_test_today: float = 0.15,
                         ) -> List[StrikeLevelInPlay]:
    """Map proximity-model output to strike-grid language.

    ``predictions`` is a list of dicts each carrying at minimum:
        level: float
        p_test_today: float
        p_test_within_60min: float
        side_from_open: "above" | "below"
        key_level_type: str (e.g. "demand_pool", "supply_pool", "poc")

    Returns one StrikeLevelInPlay per input prediction whose
    p_test_today >= ``min_p_test_today``. Multiple raw levels can snap
    to the same strike; we DON'T deduplicate (the brief renderer can
    show "12450 strike — 82% from demand-pool; 48% from POC"; both
    rows preserve attribution).
    """
    cfg = INDEX_CONFIGS.get(index_name)
    if cfg is None:
        raise KeyError(f"unknown index {index_name!r}")
    out: List[StrikeLevelInPlay] = []
    for p in predictions:
        p_today = float(p.get("p_test_today", 0.0))
        if p_today < min_p_test_today:
            continue
        level = float(p["level"])
        strike = nearest_strike(level, index_name)
        out.append(StrikeLevelInPlay(
            strike=strike,
            p_test_today=p_today,
            p_test_within_60min=float(p.get("p_test_within_60min", 0.0)),
            side_from_open=str(p.get("side_from_open", "above")),
            key_level_type=str(p.get("key_level_type", "unknown")),
            underlying_level=level,
        ))
    out.sort(key=lambda x: -x.p_test_today)
    return out


# ---------------------------------------------------------------------------
# Regime helpers — used by the brief's options_suitability section
# ---------------------------------------------------------------------------

def theta_danger_score(path_efficiency_30: float,
                        direction_changes_30: float,
                        vol_regime_zscore_20d: float) -> float:
    """Compute the per-index theta-danger score.

    Range 0..1. High = chop predicted; premium decay is the dominant
    risk for option BUYERS regardless of directional thesis. Drawn from
    the path/context features the bundle already computes.

    Heuristic (tunable in v2):
      * Low path efficiency means choppy → theta dangerous to buyers.
      * High direction-change count amplifies that.
      * Elevated vol_regime makes IV crush worse, so we tilt the
        score slightly higher in elevated vol.
    """
    eff = max(0.0, min(1.0, float(path_efficiency_30)))
    chop_contribution = 1.0 - eff
    change_contribution = min(1.0,
                                max(0.0, float(direction_changes_30) / 25.0))
    vol_contribution = max(0.0, min(0.5,
                                      float(vol_regime_zscore_20d) * 0.15))
    score = 0.55 * chop_contribution + 0.30 * change_contribution + 0.15 * vol_contribution
    return float(max(0.0, min(1.0, score)))


def premium_regime_for_buyers_and_sellers(theta_score: float,
                                            ) -> Dict[str, str]:
    """Map theta-danger score to a buyer/seller regime label pair.

    Returns the structure that fits into options_suitability:
        {"regime_for_premium_buyers": ..., "regime_for_premium_sellers": ...}

    Labels: 'unfavourable' / 'moderate' / 'favourable'. This is the
    closest the brief comes to a directional suggestion, and it's
    still framed as REGIME, not action.
    """
    if theta_score >= 0.65:
        buyers = "unfavourable"
        sellers = "favourable"
    elif theta_score >= 0.40:
        buyers = "moderate"
        sellers = "moderate"
    else:
        buyers = "favourable"
        sellers = "unfavourable"
    return {
        "regime_for_premium_buyers": buyers,
        "regime_for_premium_sellers": sellers,
    }


def is_known_index(index_name: str) -> bool:
    return index_name in INDEX_CONFIGS
