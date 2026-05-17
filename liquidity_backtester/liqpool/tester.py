"""Walk-forward pool tester (forward-bias-free, multi-tier outcomes with confirmation logic).

For each pool we start scanning the base TF *strictly after* `pool.available_at` — the moment
the latest contributor would have been visible to a real trader — and classify into ONE of:

  untouched             price never entered the zone within the horizon
  touched_no_signal     entered but neither a meaningful reaction nor a break occurred
  respected_weak        reaction in [weak_reaction_atr, strong_reaction_atr) within
                        respect_within_bars; no decisive break
  respected_strong      reaction >= strong_reaction_atr; no break >= weak_break_atr
  broken_weak           close beyond by [weak_break_atr, strong_break_atr) AND no strong reaction
                        OR a single-bar close past >= strong_break_atr that DIDN'T confirm
                        (i.e. next bar didn't also close past) — classical "wick break" / fakeout
  broken_strong         strong_break_confirm_bars CONSECUTIVE bars close past by
                        >= strong_break_atr AND price didn't reclaim back inside within
                        reclaim_within_bars — definitive failure
  swept_and_reclaimed   price closed past by >= strong_break_atr (even just once) then closed
                        back INSIDE the zone within reclaim_within_bars — the canonical SMC
                        stop-hunt + hold pattern. Treated as a (decisive) respect.
  horizon_insufficient  pool too close to data end to test fairly

Priorities at classification time:
  1. untouched (overrides all)
  2. swept_and_reclaimed (close-past followed by reclaim)
  3. broken_strong (confirmed consecutive strong-close-past, no reclaim)
  4. respected_strong (strong reaction, no weak-break-or-greater anywhere)
  5. broken_weak (some close past, no strong reaction)
  6. respected_weak (some reaction)
  7. touched_no_signal (touched but nothing meaningful)
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence
import numpy as np
import pandas as pd

from .config import Config
from .indicators import atr
from .pools import Pool


_RESPECT_OUTCOMES = ("respected_strong", "respected_weak", "swept_and_reclaimed")
_BREAK_OUTCOMES = ("broken_strong", "broken_weak")
_DECISIVE = ("respected_strong", "broken_strong", "swept_and_reclaimed")


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
    max_excursion_through: float = 0.0      # max close-through past the zone in ATR units
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
        return self.is_respect or self.is_break

    def as_dict(self):
        d = asdict(self)
        d["formed_at"] = str(self.formed_at)
        d["touched_at"] = str(self.touched_at) if self.touched_at is not None else None
        d["broken_at"] = str(self.broken_at) if self.broken_at is not None else None
        return d


def _touch(bar_low: float, bar_high: float, pool_low: float, pool_high: float) -> bool:
    return not (bar_high < pool_low or bar_low > pool_high)


def _has_consecutive_run(indices: Sequence[int], n: int) -> bool:
    """True if `indices` contains a run of n consecutive integers."""
    if n <= 1:
        return len(indices) >= 1
    if len(indices) < n:
        return False
    s = sorted(set(indices))
    streak = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1] + 1:
            streak += 1
            if streak >= n:
                return True
        else:
            streak = 1
    return False


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
        strong_break_bars: List[int] = []   # bar indices where close past by >= strong_break_atr
        first_strong_break_at: Optional[int] = None
        reclaim_at: Optional[int] = None
        confirmed_strong_break = False

        for j in range(start, end):
            in_zone = _touch(l[j], h[j], p.price_low, p.price_high)
            if not in_zone:
                outside_seen = True
            if in_zone and outside_seen:
                if not did_touch:
                    did_touch = True
                    touched_at = idx[j]
                    bars_to_touch = j - start
                n_touches += 1

            if did_touch:
                # Close-through magnitude (post-touch only).
                if p.side == "high":
                    cb = (c[j] - p.price_high) / max(a[j], 1e-9)
                else:
                    cb = (p.price_low - c[j]) / max(a[j], 1e-9)
                if cb > max_close_break:
                    max_close_break = cb
                    if cb >= cfg.weak_break_atr and broken_at is None:
                        broken_at = idx[j]
                        bars_to_break = j - start

                # Track strong-break bars (for confirmation logic).
                if cb >= cfg.strong_break_atr:
                    strong_break_bars.append(j)
                    if first_strong_break_at is None:
                        first_strong_break_at = j

                # Reclaim detection: after the first strong-break bar, did price close BACK
                # INSIDE the zone within reclaim_within_bars?
                if first_strong_break_at is not None and reclaim_at is None:
                    delta = j - first_strong_break_at
                    if 0 < delta <= cfg.reclaim_within_bars:
                        if p.price_low <= c[j] <= p.price_high:
                            reclaim_at = j

                # Reaction tracking: max reverse move within respect_within_bars after touch.
                bars_since_touch = j - (start + bars_to_touch)
                if bars_since_touch <= cfg.respect_within_bars:
                    if p.side == "high":
                        rev = max(0.0, (p.price_low - l[j]) / max(a[j], 1e-9))
                    else:
                        rev = max(0.0, (h[j] - p.price_high) / max(a[j], 1e-9))
                    if rev > max_reaction:
                        max_reaction = rev

            # Cheap early exit: confirmed strong break with no chance of reclaim left.
            if (first_strong_break_at is not None
                    and reclaim_at is None
                    and (j - first_strong_break_at) > cfg.reclaim_within_bars
                    and _has_consecutive_run(strong_break_bars, cfg.strong_break_confirm_bars)):
                confirmed_strong_break = True
                break

        # Final consolidation if we didn't early-exit.
        if not confirmed_strong_break:
            confirmed_strong_break = _has_consecutive_run(strong_break_bars,
                                                          cfg.strong_break_confirm_bars)

        # Classify (priority order matters).
        if not did_touch:
            outcome = "untouched"
        elif first_strong_break_at is not None and reclaim_at is not None:
            outcome = "swept_and_reclaimed"
        elif confirmed_strong_break:
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

    `respect_rate` (broad)        = (all respects incl swept_and_reclaimed) / (respects + breaks)
    `respect_rate_strict`         = (respected_strong + swept_and_reclaimed) /
                                    (respected_strong + swept_and_reclaimed + broken_strong)
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

    decisive_resp = counts.get("respected_strong", 0) + counts.get("swept_and_reclaimed", 0)
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
