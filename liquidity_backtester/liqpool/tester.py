"""Walk-forward pool tester.

For each pool we look at the future bars on the base TF and classify the outcome:
  - untouched: price never entered the zone within the horizon
  - respected: price touched the zone and then reversed >= reaction_atr*ATR within respect_within_bars,
               *without* closing through the zone
  - broken: a bar closed beyond the zone by break_close_buffer_atr*ATR
We also record max excursion through the zone and time-to-touch / time-to-break."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List
import numpy as np
import pandas as pd

from .config import Config
from .indicators import atr
from .pools import Pool


@dataclass
class PoolResult:
    pool_idx: int
    side: str
    formed_at: pd.Timestamp
    price_low: float
    price_high: float
    score: float
    outcome: str                # "untouched" | "respected" | "broken"
    touched_at: pd.Timestamp | None = None
    broken_at: pd.Timestamp | None = None
    bars_to_touch: int | None = None
    bars_to_break: int | None = None
    max_excursion_through: float = 0.0   # in ATR units; how far price closed beyond the pool
    reaction_atr: float = 0.0            # how far price reversed after touch
    n_touches: int = 0
    tfs: List[str] = field(default_factory=list)

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
    pos_of_ts = {ts: i for i, ts in enumerate(idx)}
    results: List[PoolResult] = []

    for k, p in enumerate(pools):
        # Start scanning at the first bar strictly after formation.
        start = pos_of_ts.get(p.formed_at)
        if start is None:
            # Pool formed_at not in base index: find the next bar by searchsorted.
            start = int(np.searchsorted(idx.values, np.datetime64(p.formed_at)))
        start = max(start + 1, 0)
        end = min(start + cfg.test_horizon_bars, len(df_base))

        if start >= end:
            results.append(PoolResult(k, p.side, p.formed_at, p.price_low, p.price_high, p.score,
                                       outcome="untouched", tfs=list(p.tfs)))
            continue

        touched_at = None
        bars_to_touch = None
        broken_at = None
        bars_to_break = None
        max_through = 0.0
        reaction = 0.0
        n_touches = 0
        outcome = "untouched"
        # Many higher-TF features (FVGs, OBs) sit on the bar that produced them, so the very next
        # bar is often still inside the zone. A real "touch" only counts after price has cleanly
        # left the zone at least once.
        outside_seen = False

        for j in range(start, end):
            in_zone = _touch(l[j], h[j], p.price_low, p.price_high)
            if not in_zone:
                outside_seen = True
            if in_zone and outside_seen:
                if touched_at is None:
                    touched_at = idx[j]
                    bars_to_touch = j - start
                n_touches += 1

            # Break check: close beyond pool by buffer.
            buf = cfg.break_close_buffer_atr * a[j]
            if p.side == "high":
                # Sell-side pool above: broken if close > pool_high + buf.
                through = c[j] - (p.price_high + buf)
            else:
                # Buy-side pool below: broken if close < pool_low - buf.
                through = (p.price_low - buf) - c[j]
            if through > 0:
                max_through = max(max_through, through / max(a[j], 1e-9))
                if broken_at is None:
                    broken_at = idx[j]
                    bars_to_break = j - start
                    outcome = "broken"
                    break  # first close-through ends the test

            # Respect check (only after first touch, before any break).
            if touched_at is not None and outcome != "broken":
                bars_since_touch = j - pos_of_ts.get(touched_at, j)
                if bars_since_touch <= cfg.respect_within_bars:
                    if p.side == "high":
                        rev = max(0.0, p.price_low - l[j])  # how far below the pool we travelled
                    else:
                        rev = max(0.0, h[j] - p.price_high)
                    reaction = max(reaction, rev / max(a[j], 1e-9))
                    if reaction >= cfg.respect_reaction_atr:
                        outcome = "respected"
                        # don't break — we still want to see if a later break happens, but result is locked
                        # Actually for cleanliness, lock the first reached terminal state.
                        break

        results.append(PoolResult(
            pool_idx=k, side=p.side, formed_at=p.formed_at,
            price_low=p.price_low, price_high=p.price_high, score=p.score,
            outcome=outcome,
            touched_at=touched_at, broken_at=broken_at,
            bars_to_touch=bars_to_touch, bars_to_break=bars_to_break,
            max_excursion_through=float(max_through),
            reaction_atr=float(reaction),
            n_touches=int(n_touches),
            tfs=list(p.tfs),
        ))
    return results


def summarise(results: List[PoolResult]) -> dict:
    if not results:
        return {"n": 0, "respect_rate": 0.0, "break_rate": 0.0, "untouched_rate": 0.0,
                "tested_n": 0, "median_bars_to_touch": None, "avg_score": 0.0}
    touched = [r for r in results if r.outcome != "untouched"]
    respected = [r for r in results if r.outcome == "respected"]
    broken = [r for r in results if r.outcome == "broken"]
    untouched = [r for r in results if r.outcome == "untouched"]
    bars_to_touch = [r.bars_to_touch for r in touched if r.bars_to_touch is not None]
    return {
        "n": len(results),
        "tested_n": len(touched),
        "respect_rate": (len(respected) / len(touched)) if touched else 0.0,
        "break_rate": (len(broken) / len(touched)) if touched else 0.0,
        "untouched_rate": len(untouched) / len(results),
        "median_bars_to_touch": float(np.median(bars_to_touch)) if bars_to_touch else None,
        "avg_score": float(np.mean([r.score for r in results])),
    }
