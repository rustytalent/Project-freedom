"""Spread & liquidity friendliness engine (Belief Engine, Phase 4).

The founder's correction: "spread cannot be only a filter — it needs its
own engine." A wide or widening spread is not just an execution cost; it
is a *market-condition signal*. In consolidation the spread can eat you
alive, and a premium move that looks abnormal may be pure spread noise.

This module produces, per contract, a continuous **Spread Friendliness
Score** ∈ [0,1] and a state label, from:

  * ``spread_pct``   — (ask − bid) / mark, scale-free across strikes
  * ``spread_z``     — current spread vs the contract's OWN rolling normal
                       spread (robust median + IQR). A 2σ-wide spread for
                       a normally-tight strike is a real distortion even if
                       its absolute pct looks small.
  * ``depth_score``  — saturating function of bid_qty + ask_qty
  * (freshness handled upstream in the mark-price layer)

States:
  * ``clean``       — tight, stable, deep → trade-friendly
  * ``widening``    — spread expanding above its norm → caution
  * ``dangerous``   — wide AND thin → do not execute; premium signals here
                      are suspect
  * ``improving``   — spread contracting back toward norm after a widening

The friendliness score is what downstream layers multiply into a signal's
confidence: a correct direction with an ugly spread is still a bad trade.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


_IQR_TO_SIGMA = 1.349

CLEAN = "clean"
WIDENING = "widening"
DANGEROUS = "dangerous"
IMPROVING = "improving"


@dataclass
class SpreadConfig:
    """Knobs for spread friendliness."""
    norm_window: int = 60          # rolling window for the contract's normal spread
    depth_full_qty: float = 1000.0  # total qty at which depth_score saturates
    tight_pct: float = 0.01        # ≤1% spread is tight
    wide_pct: float = 0.05         # ≥5% spread is wide
    widening_z: float = 1.5        # spread_z beyond this = widening
    dangerous_z: float = 2.5       # spread_z beyond this (and wide) = dangerous
    spread_weight: float = 0.6     # friendliness = w·tightness + (1-w)·depth, minus z penalty
    eps: float = 1e-9

    def __post_init__(self) -> None:
        if self.wide_pct <= self.tight_pct:
            raise ValueError("wide_pct must be > tight_pct")
        if not (0.0 <= self.spread_weight <= 1.0):
            raise ValueError("spread_weight must be in [0,1]")


def spread_friendliness_frame(df: pd.DataFrame,
                              cfg: Optional[SpreadConfig] = None) -> pd.DataFrame:
    """Compute spread friendliness for one contract's quote series.

    ``df`` needs columns ``bid``, ``ask``, ``mark`` and optionally
    ``bid_qty`` / ``ask_qty`` (default 0). Returns a copy with:
      * ``spread_pct`` / ``spread_norm`` / ``spread_z``
      * ``depth_score``
      * ``friendliness`` ∈ [0,1]
      * ``spread_state`` — clean / widening / dangerous / improving
    """
    cfg = cfg or SpreadConfig()
    required = {"bid", "ask", "mark"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"spread_friendliness_frame missing columns: {sorted(missing)}")
    out = df.copy().reset_index(drop=True)
    bid = pd.to_numeric(out["bid"], errors="coerce")
    ask = pd.to_numeric(out["ask"], errors="coerce")
    mark = pd.to_numeric(out["mark"], errors="coerce")
    bid_qty = pd.to_numeric(out["bid_qty"], errors="coerce") if "bid_qty" in out.columns else pd.Series(0.0, index=out.index)
    ask_qty = pd.to_numeric(out["ask_qty"], errors="coerce") if "ask_qty" in out.columns else pd.Series(0.0, index=out.index)

    spread = (ask - bid).clip(lower=0.0)
    spread_pct = (spread / mark.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    w = cfg.norm_window
    min_p = max(8, w // 2)
    norm = spread_pct.rolling(w, min_periods=min_p).median()
    q75 = spread_pct.rolling(w, min_periods=min_p).quantile(0.75)
    q25 = spread_pct.rolling(w, min_periods=min_p).quantile(0.25)
    sigma = ((q75 - q25) / _IQR_TO_SIGMA).clip(lower=cfg.eps)
    spread_z = ((spread_pct - norm) / sigma).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    depth = (bid_qty + ask_qty).fillna(0.0)
    depth_score = (depth / (cfg.depth_full_qty + cfg.eps)).clip(upper=1.0)

    # Tightness: 1 at/under tight_pct, 0 at/over wide_pct, linear between.
    tightness = (1.0 - (spread_pct - cfg.tight_pct) / (cfg.wide_pct - cfg.tight_pct))
    tightness = tightness.clip(lower=0.0, upper=1.0)

    z_penalty = (spread_z.clip(lower=0.0) / (cfg.dangerous_z + cfg.eps)).clip(upper=1.0)
    friendliness = (cfg.spread_weight * tightness
                    + (1.0 - cfg.spread_weight) * depth_score) * (1.0 - 0.5 * z_penalty)
    friendliness = friendliness.clip(lower=0.0, upper=1.0)

    # State machine. A spread that is still tighter than ``tight_pct`` cannot
    # be "widening" in any way that matters — a 0.30%→0.31% wiggle is a huge
    # robust-z on a tiny-variance baseline but is not a real distortion. Gate
    # the elevated states on the absolute pct so the z only matters once the
    # spread is meaningfully off its tight floor.
    above_tight = spread_pct > cfg.tight_pct
    is_wide = spread_pct >= cfg.wide_pct
    widening = (spread_z > cfg.widening_z) & above_tight
    dangerous = (spread_z > cfg.dangerous_z) & is_wide
    improving = (spread_z < -cfg.widening_z) & above_tight
    state = np.full(len(out), CLEAN, dtype=object)
    state = np.where(widening.values, WIDENING, state)
    state = np.where(improving.values, IMPROVING, state)
    state = np.where(dangerous.values, DANGEROUS, state)

    out["spread_pct"] = spread_pct.values
    out["spread_norm"] = norm.fillna(spread_pct).values
    out["spread_z"] = spread_z.values
    out["depth_score"] = depth_score.values
    out["friendliness"] = friendliness.values
    out["spread_state"] = state
    return out


def is_execution_friendly(friendliness: float, spread_state: str,
                          *, min_friendliness: float = 0.45) -> bool:
    """Gate helper: True when the contract is friendly enough to execute.
    Dangerous spread is never friendly regardless of score."""
    if spread_state == DANGEROUS:
        return False
    return friendliness >= min_friendliness
