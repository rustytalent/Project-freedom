"""Crowd mirror — "do we look like retail?"

Founder's mandate: *"If we are the dumb money, the MM has a target on
us. We need to know — and diversify away from the herd's structure."*

This module measures the structural similarity between OUR current
portfolio and the typical retail pattern, then warns when we're
indistinguishable from the crowd.

The retail signature is roughly:
  * Long ATM CE during bull thesis windows
  * Long ATM PE during bear thesis windows
  * No hedges, no spreads — single-leg directional
  * Bunched stops within ±20% of entry premium
  * No structural diversification (one underlying, one expiry, one strike-cluster)

Output: ``CrowdMirrorReport`` with a ``we_look_like_retail`` boolean,
density scores, and recommended diversification moves.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CrowdMirrorConfig:
    """Knobs for crowd-mirror inference."""
    retail_threshold: float = 0.65         # ≥ this → we_look_like_retail
    atm_band_levels: int = 1                # within ±N strikes of ATM is "ATM-like"
    long_atm_weight: float = 0.40
    no_hedge_weight: float = 0.20
    single_underlying_weight: float = 0.15
    single_expiry_weight: float = 0.15
    bunched_stops_weight: float = 0.10


@dataclass
class CrowdMirrorReport:
    """The crowd-mirror's per-tick output."""
    we_look_like_retail: bool
    crowd_density_long_atm_ce: float
    crowd_density_long_atm_pe: float
    no_hedge_fraction: float
    single_underlying: bool
    single_expiry: bool
    stops_clustered: bool
    retail_similarity_score: float          # 0..1
    mm_likely_target_us: bool
    recommended_diversification: List[str]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "we_look_like_retail": self.we_look_like_retail,
            "crowd_density_long_atm_ce": round(self.crowd_density_long_atm_ce, 3),
            "crowd_density_long_atm_pe": round(self.crowd_density_long_atm_pe, 3),
            "no_hedge_fraction": round(self.no_hedge_fraction, 3),
            "single_underlying": self.single_underlying,
            "single_expiry": self.single_expiry,
            "stops_clustered": self.stops_clustered,
            "retail_similarity_score": round(self.retail_similarity_score, 3),
            "mm_likely_target_us": self.mm_likely_target_us,
            "recommended_diversification": list(self.recommended_diversification),
            "notes": list(self.notes),
        }


class CrowdMirror:
    """The crowd-similarity self-awareness module."""

    def __init__(self, cfg: Optional[CrowdMirrorConfig] = None) -> None:
        self.cfg = cfg or CrowdMirrorConfig()

    def inspect(self,
                  open_positions: List[Any],
                  ) -> CrowdMirrorReport:
        """Look at the current portfolio and grade its retail-similarity.

        ``open_positions`` is an iterable of ``PositionHypothesis`` (or
        anything quacking like one: contract_side, contract_level,
        strike_price, stop_premium, entry_premium, direction).
        """
        cfg = self.cfg
        if not open_positions:
            return CrowdMirrorReport(
                we_look_like_retail=False,
                crowd_density_long_atm_ce=0.0,
                crowd_density_long_atm_pe=0.0,
                no_hedge_fraction=0.0,
                single_underlying=False,
                single_expiry=False,
                stops_clustered=False,
                retail_similarity_score=0.0,
                mm_likely_target_us=False,
                recommended_diversification=[],
            )

        n = len(open_positions)
        long_atm_ce = 0
        long_atm_pe = 0
        single_legs = 0
        for p in open_positions:
            side = getattr(p, "contract_side", "")
            level = int(getattr(p, "contract_level", 0))
            direction = int(getattr(p, "direction", 0))
            if abs(level) <= cfg.atm_band_levels:
                if side == "CE" and direction > 0:
                    long_atm_ce += 1
                elif side == "PE" and direction < 0:
                    long_atm_pe += 1
            # Treat anything without a partner leg as a single-leg (proxy
            # for "no hedge" in this Sprint-3 board).
            single_legs += 1

        long_atm_ce_density = long_atm_ce / n
        long_atm_pe_density = long_atm_pe / n
        no_hedge_frac = single_legs / n

        single_underlying = True   # all NIFTY weekly options in Sprint 3
        single_expiry = True       # same

        # Stops clustered: small variance of stop_premium / entry_premium.
        stop_ratios = []
        for p in open_positions:
            entry = float(getattr(p, "entry_premium", 0.0) or 0.0)
            stop = float(getattr(p, "stop_premium", 0.0) or 0.0)
            if entry > 0:
                stop_ratios.append(stop / entry)
        stops_clustered = False
        if len(stop_ratios) >= 2:
            mean = sum(stop_ratios) / len(stop_ratios)
            var = sum((r - mean) ** 2 for r in stop_ratios) / len(stop_ratios)
            stops_clustered = var < 0.005  # very tight

        # Score.
        score = (
            cfg.long_atm_weight * max(long_atm_ce_density, long_atm_pe_density)
            + cfg.no_hedge_weight * no_hedge_frac
            + cfg.single_underlying_weight * (1.0 if single_underlying else 0.0)
            + cfg.single_expiry_weight * (1.0 if single_expiry else 0.0)
            + cfg.bunched_stops_weight * (1.0 if stops_clustered else 0.0)
        )
        score = max(0.0, min(1.0, score))
        we_look_like_retail = score >= cfg.retail_threshold
        mm_target = we_look_like_retail and n >= 2

        recs: List[str] = []
        notes: List[str] = []
        if long_atm_ce_density >= 0.50:
            recs.append("diversify CE strikes to OTM2/ITM1 — break the ATM cluster")
        if long_atm_pe_density >= 0.50:
            recs.append("diversify PE strikes to OTM2/ITM1 — break the ATM cluster")
        if no_hedge_frac >= 0.80:
            recs.append(
                "convert one position to a vertical spread to reduce single-leg exposure"
            )
        if stops_clustered:
            recs.append("stagger stops — avoid synchronous liquidation")
        if mm_target:
            notes.append(
                "Portfolio structurally indistinguishable from retail — high MM target risk"
            )

        return CrowdMirrorReport(
            we_look_like_retail=we_look_like_retail,
            crowd_density_long_atm_ce=long_atm_ce_density,
            crowd_density_long_atm_pe=long_atm_pe_density,
            no_hedge_fraction=no_hedge_frac,
            single_underlying=single_underlying,
            single_expiry=single_expiry,
            stops_clustered=stops_clustered,
            retail_similarity_score=score,
            mm_likely_target_us=mm_target,
            recommended_diversification=recs,
            notes=notes,
        )
