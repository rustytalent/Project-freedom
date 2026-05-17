"""Walk-forward pool tester (forward-bias-free, multi-tier outcomes).

For each pool we start scanning the base TF *strictly after* `pool.available_at` — the moment
the latest contributor would have been visible to a real trader — and classify into ONE of:

  untouched             price never entered the zone within the horizon
  touched_no_signal     entered but neither a meaningful reaction nor a close-break occurred
  respected_weak        reaction in [weak, strong) ATR within respect_within_bars; no break
  respected_strong      reaction >= strong_reaction_atr; no break
  broken_weak           close beyond zone by [weak, strong) ATR (mild breach, often wick-close)
  broken_strong         close beyond zone by >= strong_break_atr (decisive close-through)
  horizon_insufficient  available_at too close to data end — excluded from rates

A strong break wins over any reaction (it overrides — the pool ultimately failed). A weak break
is only fatal if there was no strong reaction (the pool survived the wick-through).

Compared to the previous single-threshold tester, this gives us TEXTURE: the share of pools
that broke decisively vs barely, and the share that respected meaningfully vs marginally."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import numpy as np
import pandas as pd

from .config import Config
from .indicators import atr
from .pools import Pool


_RESPECT_OUTCOMES = ("respected_strong", "respected_weak")
_BREAK_OUTCOMES = ("broken_strong", "broken_weak")
_DECISIVE = ("respected_strong", "broken_strong")


@dataclass
class PoolResult:
    pool_idx: int
    side: str
    formed_at: pd.Timestamp
    price_low: float
    price_high: float
    score: float
    outcome: str
    touched_at: Optional[pd.Timestamp] = None
    broken_at: Optional[pd.Timestamp] = None
    bars_to_touch: Optional[int] = None
    bars_to_break: Optional[int] = None
    max_excursion_through: float = 0.0      # in ATR units; max close-through past the zone
    reaction_atr: float = 0.0               # max reverse move post-touch in ATR units
    n_touches: int = 0
    tfs: List[str] = field(default_factory=list)

    @property
    def is_respect(self) -> bool:
        return self.outcome in _RESPECT_OUTCOMES

    @property
    def is_break(self) -> bool:
        return self.outcome in _BREAK_OUTCOMES

    @property
    def is_decisive(self) -> bool:
        return self.outcome in _DECISIVE

    @property
    def is_tested(self) -> bool:
        """A pool is 'tested' if we got either a respect or break outcome (any tier)."""
        return self.is_respect or self.is_break

    def as_dict(self):
        d = asdict(self)
        d["formed_at"] = str(self.formed_at)
        d["touched_at"] = str(self.touched_at) if self.touched_at is not None else None
        d["broken_at"] = str(self.broken_at) if self.broken_at is not None else None
        return d


def _touch(bar_low: float, bar_high: float, pool_low: float, pool_high: float) -> bool:
    return not (bar_high < pool_low or bar_low > pool_high)


def test_pools(df_base: pd.DataFrame, pools: List[Pool], cfg: Config) -> List[PoolResult]:
    if df_base.empty or not pools:
        return []
    a = atr(df_base, cfg.detect.atr_period).bfill().values
    idx = df_base.index
    h, l, c = df_base["high"].values, df_base["low"].values, df_base["close"].values
    results: List[PoolResult] = []
    min_horizon = max(cfg.respect_within_bars + 1, 10)

    for k, p in enumerate(pools):
        start = int(np.searchsorted(idx.values, np.datetime64(p.available_at), side="right"))
        end = min(start + cfg.test_horizon_bars, len(df_base))
        forward_bars = max(0, end - start)

        if forward_bars < min_horizon:
            results.append(PoolResult(k, p.side, p.formed_at, p.price_low, p.price_high, p.score,
                                       outcome="horizon_insufficient", tfs=list(p.tfs)))
            continue

        did_touch = False
        touched_at, bars_to_touch = None, None
        broken_at, bars_to_break = None, None
        max_close_break = 0.0
        max_reaction = 0.0
        n_touches = 0
        outside_seen = False

        for j in range(start, end):
            in_zone = _touch(l[j], h[j], p.price_low, p.price_high)
            if not in_zone:
                outside_seen = True
            # A touch only counts after price has cleanly left the zone at least once.
            if in_zone and outside_seen:
                if not did_touch:
                    did_touch = True
                    touched_at = idx[j]
                    bars_to_touch = j - start
                n_touches += 1

            # Close-through magnitude in ATR units (post-touch only).
            if did_touch:
                if p.side == "high":
                    cb = (c[j] - p.price_high) / max(a[j], 1e-9)
                else:
                    cb = (p.price_low - c[j]) / max(a[j], 1e-9)
                if cb > max_close_break:
                    max_close_break = cb
                    if cb >= cfg.weak_break_atr and broken_at is None:
                        broken_at = idx[j]
                        bars_to_break = j - start

            # Reverse move within respect_within_bars after first touch.
            if did_touch:
                bars_since_touch = j - (start + bars_to_touch)
                if bars_since_touch <= cfg.respect_within_bars:
                    if p.side == "high":
                        rev = max(0.0, (p.price_low - l[j]) / max(a[j], 1e-9))
                    else:
                        rev = max(0.0, (h[j] - p.price_high) / max(a[j], 1e-9))
                    if rev > max_reaction:
                        max_reaction = rev

            # Stop walking once a STRONG break has happened — that's a final state.
            if max_close_break >= cfg.strong_break_atr:
                break

        # Classify with priority: strong-break overrides; otherwise strong-respect; then weak tier.
        if not did_touch:
            outcome = "untouched"
        elif max_close_break >= cfg.strong_break_atr:
            outcome = "broken_strong"
        elif max_reaction >= cfg.strong_reaction_atr and max_close_break < cfg.weak_break_atr:
            outcome = "respected_strong"
        elif max_close_break >= cfg.weak_break_atr and max_reaction < cfg.strong_reaction_atr:
            outcome = "broken_weak"
        elif max_reaction >= cfg.weak_reaction_atr:
            outcome = "respected_weak"
        else:
            outcome = "touched_no_signal"

        results.append(PoolResult(
            pool_idx=k, side=p.side, formed_at=p.formed_at,
            price_low=p.price_low, price_high=p.price_high, score=p.score,
            outcome=outcome, touched_at=touched_at, broken_at=broken_at,
            bars_to_touch=bars_to_touch, bars_to_break=bars_to_break,
            max_excursion_through=float(max_close_break),
            reaction_atr=float(max_reaction),
            n_touches=int(n_touches),
            tfs=list(p.tfs),
        ))
    return results


def summarise(results: List[PoolResult]) -> dict:
    """Returns headline metrics + per-outcome counts.

    `respect_rate` is the BROAD rate: (all respects) / (all respects + all breaks).
    `respect_rate_strict` is the DECISIVE rate: respected_strong / (respected_strong + broken_strong).
    Strict ignores weak/ambiguous outcomes and gives the cleanest signal of "pools that mattered".
    """
    if not results:
        return {"n": 0, "tested_n": 0, "respect_rate": 0.0, "break_rate": 0.0,
                "respect_rate_strict": 0.0, "decisive_n": 0,
                "untouched_rate": 0.0, "touched_no_signal_rate": 0.0,
                "horizon_insufficient": 0, "median_bars_to_touch": None,
                "avg_score": 0.0, "outcomes": {}}

    insufficient = [r for r in results if r.outcome == "horizon_insufficient"]
    eligible = [r for r in results if r.outcome != "horizon_insufficient"]

    counts: dict = {}
    for r in eligible:
        counts[r.outcome] = counts.get(r.outcome, 0) + 1

    respected = [r for r in eligible if r.is_respect]
    broken = [r for r in eligible if r.is_break]
    untouched = [r for r in eligible if r.outcome == "untouched"]
    no_signal = [r for r in eligible if r.outcome == "touched_no_signal"]

    decisive_resp = counts.get("respected_strong", 0)
    decisive_break = counts.get("broken_strong", 0)
    decisive_total = decisive_resp + decisive_break

    tested = len(respected) + len(broken)
    bars_to_touch = [r.bars_to_touch for r in (respected + broken) if r.bars_to_touch is not None]

    return {
        "n": len(results),
        "eligible_n": len(eligible),
        "tested_n": tested,
        "horizon_insufficient": len(insufficient),
        "respect_rate": (len(respected) / tested) if tested else 0.0,
        "break_rate": (len(broken) / tested) if tested else 0.0,
        "respect_rate_strict": (decisive_resp / decisive_total) if decisive_total else 0.0,
        "decisive_n": decisive_total,
        "untouched_rate": (len(untouched) / len(eligible)) if eligible else 0.0,
        "touched_no_signal_rate": (len(no_signal) / len(eligible)) if eligible else 0.0,
        "median_bars_to_touch": float(np.median(bars_to_touch)) if bars_to_touch else None,
        "avg_score": float(np.mean([r.score for r in results])),
        "outcomes": counts,
    }
