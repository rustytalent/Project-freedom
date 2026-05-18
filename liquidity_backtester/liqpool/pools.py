"""Liquidity pool synthesis: cluster level candidates from multiple timeframes into pools, score them.

A `Pool` is a price *zone* (low..high) anchored to a formation time, with contributors from one
or more detectors and timeframes. Confluence is the headline idea: a 5m EQH that lines up with
a PWH and a 1H FVG is far more meaningful than any one of those alone."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict
import math
import pandas as pd

from .config import Config, DetectionParams, FactorWeights
from .features import LevelCandidate, detect_all, previous_period_levels
from .indicators import atr


@dataclass
class Pool:
    side: str                          # "high" (sell-side / supply) or "low" (buy-side / demand)
    price_low: float
    price_high: float
    formed_at: pd.Timestamp            # latest contributor formation time (price-wise)
    available_at: pd.Timestamp         # when the LAST contributor became known to a trader.
                                       # Tester walks forward strictly after this — no look-ahead.
    contributors: List[LevelCandidate] = field(default_factory=list)
    score: float = 0.0
    tfs: List[str] = field(default_factory=list)
    asset: str = ""                    # symbol this pool belongs to (Track 4 multi-asset)

    @property
    def mid(self) -> float:
        return (self.price_low + self.price_high) / 2

    @property
    def width(self) -> float:
        return self.price_high - self.price_low

    def summary(self) -> Dict:
        return {
            "side": self.side,
            "price_low": self.price_low,
            "price_high": self.price_high,
            "mid": self.mid,
            "formed_at": self.formed_at,
            "available_at": self.available_at,
            "score": self.score,
            "tfs": self.tfs,
            "n_contributors": len(self.contributors),
            "sources": sorted({c.source for c in self.contributors}),
        }


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _zone_of(c: LevelCandidate) -> tuple[float, float]:
    return c.price - c.half_width, c.price + c.half_width


def _overlap(a_low: float, a_high: float, b_low: float, b_high: float, tol: float) -> bool:
    return not (a_high + tol < b_low or b_high + tol < a_low)


def cluster_candidates(cands: List[LevelCandidate], merge_tol: float,
                       max_pool_width: float | None = None) -> List[Pool]:
    """Centroid-based price clustering on each side.

    A candidate joins an existing cluster if its price is within `merge_tol` of the cluster's
    running centroid. This avoids the runaway "transitive expansion" of zone-overlap merging,
    where one wide zone (typical of high-TF FVGs/OBs) swallows everything nearby.

    The final pool zone is derived from the candidates' price range plus a small cushion, and
    capped at `max_pool_width` if set.
    """
    pools: List[Pool] = []
    by_side: Dict[str, List[LevelCandidate]] = {"high": [], "low": []}
    for c in cands:
        if c.side in by_side:
            by_side[c.side].append(c)

    cap = max_pool_width if max_pool_width is not None else (merge_tol * 4.0)

    for side, items in by_side.items():
        items.sort(key=lambda x: x.price)
        i = 0
        while i < len(items):
            group = [items[i]]
            centroid = items[i].price
            j = i + 1
            while j < len(items) and abs(items[j].price - centroid) <= merge_tol:
                group.append(items[j])
                centroid = sum(c.price for c in group) / len(group)
                j += 1

            prices = [c.price for c in group]
            pmin, pmax = min(prices), max(prices)
            # Cushion: take the largest contributor half-width but clamp so we don't blow up.
            cushion = min(max((c.half_width for c in group), default=merge_tol * 0.5),
                          merge_tol * 1.5)
            pool_low = pmin - cushion
            pool_high = pmax + cushion
            # Cap total width: shrink symmetrically around centroid.
            if pool_high - pool_low > cap:
                pool_low = centroid - cap / 2
                pool_high = centroid + cap / 2

            pools.append(Pool(
                side=side,
                price_low=float(pool_low),
                price_high=float(pool_high),
                formed_at=max(c.ts for c in group),
                # Honest "available to trader" time = when the LATEST contributor became known.
                available_at=max(c.known_at for c in group),
                contributors=list(group),
                tfs=sorted({c.tf for c in group}),
            ))
            i = j
    return pools


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_FACTOR_OF_SOURCE = {
    "SWING_H": None, "SWING_L": None,                   # swings are baseline, weight handled implicitly
    "EQH": "eqhl", "EQL": "eqhl",
    "PDH": "prev_day", "PDL": "prev_day",
    "PWH": "prev_week", "PWL": "prev_week",
    "PMH": "prev_month", "PML": "prev_month",
    "FVG_bull": "fvg", "FVG_bear": "fvg",
    "OB_bull": "order_block", "OB_bear": "order_block",
    "REJ_high": "in_candle_imbalance", "REJ_low": "in_candle_imbalance",
    "HVN": "volume_node",
    "ORB_H": "orb_extreme", "ORB_L": "orb_extreme",
}


def _factor_of(source: str) -> str | None:
    # Source string is e.g. "EQH@15min" — strip the suffix.
    head = source.split("@", 1)[0]
    return _FACTOR_OF_SOURCE.get(head)


def score_pool(pool: Pool, weights: FactorWeights) -> float:
    if not pool.contributors:
        return 0.0
    w = weights.as_dict()
    score = 0.0
    factor_seen: Dict[str, float] = {}
    for c in pool.contributors:
        f = _factor_of(c.source)
        contrib = c.strength
        if f is None:
            # baseline swing contributes a small fixed amount
            contrib *= 0.4
        else:
            contrib *= w.get(f, 1.0)
        # Diminishing returns: each repeated factor adds sqrt-scaled amount.
        prior = factor_seen.get(f or "swing", 0.0)
        marginal = contrib / math.sqrt(1 + prior)
        score += marginal
        factor_seen[f or "swing"] = prior + 1
    # Multi-TF confluence multiplier: more *distinct* TFs aligning → exponential-ish boost.
    distinct_tfs = len(set(pool.tfs))
    if distinct_tfs > 1:
        score *= (1.0 + (distinct_tfs - 1) * (w["multi_tf_overlap"] - 1.0))
    return score


# ---------------------------------------------------------------------------
# Build pools across multi-TF data
# ---------------------------------------------------------------------------

def build_pools(tf_data: Dict[str, pd.DataFrame], cfg: Config) -> List[Pool]:
    """tf_data: {'base': df_5m, '15min': df_15m, ...}.
    Pools are clustered using base-TF ATR as the merge tolerance."""
    base = tf_data["base"]
    base_atr = atr(base, cfg.detect.atr_period).bfill()
    median_atr = float(base_atr.median())
    merge_tol = float(median_atr * cfg.detect.merge_atr)
    max_pool_width = float(median_atr * 1.5)  # hard cap so pools stay readable on the 5m chart

    cands: List[LevelCandidate] = []

    # Per-TF detectors (everything except previous-period levels, which we want on the base TF).
    for tf, df in tf_data.items():
        if df is None or df.empty:
            continue
        # ORB only on base.
        cands.extend(detect_all(df, cfg.detect, tf=tf, include_orb=(tf == "base")))

    # Previous D/W/M from base TF (these naturally span the lookback).
    cands.extend(previous_period_levels(base))

    # Give every candidate a half-width if it has none.
    for c in cands:
        if c.half_width <= 0:
            c.half_width = float(base_atr.iloc[-1]) * cfg.detect.pool_halfwidth_atr

    pools = cluster_candidates(cands, merge_tol=merge_tol, max_pool_width=max_pool_width)
    for p in pools:
        p.score = score_pool(p, cfg.weights)

    # Filter by minimum score and sort: highest score first, most recent breaks ties.
    pools = [p for p in pools if p.score >= cfg.min_pool_score]
    pools.sort(key=lambda p: (p.score, p.formed_at), reverse=True)
    return pools


def project_to_base(pools: List[Pool], base_index: pd.DatetimeIndex) -> List[Pool]:
    """Drop pools whose `available_at` falls outside the base TF window; clamp `formed_at` so
    plotting on the 5m chart always has a valid anchor. We drop (not clamp) pools whose
    `available_at` is after the base data ends, because there's no forward data to test them on
    and forcing them in would be misleading."""
    if len(base_index) == 0:
        return pools
    bmin, bmax = base_index[0], base_index[-1]
    out = []
    for p in pools:
        if p.available_at > bmax:
            continue
        if p.formed_at < bmin:
            p.formed_at = bmin
        if p.available_at < bmin:
            p.available_at = bmin
        out.append(p)
    return out
