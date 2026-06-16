"""Fair response model (Belief Engine, Phase 4).

"What should this contract's premium do, given the underlying move?"

The expected premium change for a move ΔS in the underlying is the
contract's effective delta times ΔS:

    fair_change = effective_delta · ΔS

The effective delta is NOT assumed — it is estimated live from the tape
(rolling Cov(Δmark, Δspot) / Var(Δspot)) and SHRUNK toward the moneyness
prior (``slot.expected_signed_delta``) when the sample is thin. This is
the founder's principle made rigorous: the moneyness identity gives the
*reference* delta, the live tape corrects it, and the blend is honest
about how much evidence it actually has.

Two reasons the shrinkage matters:
  * Early in a session there aren't enough ticks to trust a raw
    regression; the prior carries it.
  * A degenerate window (spot barely moved → Var≈0) produces a wild raw
    beta; the prior + clipping tame it.

The output ``residual = Δmark − fair_change`` is the per-bar surprise:
the part of the premium move NOT explained by direction. Phase-4
``residual.py`` then turns that into the slot-specific
deviation-of-deviation.

Everything runs on the clean MARK series (Phase 2), never raw LTP, and is
strictly causal (rolling windows look backward only).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .moneyness import MoneynessSlot


@dataclass
class FairResponseConfig:
    """Knobs for the effective-delta estimate."""
    delta_window: int = 30           # rolling window for the empirical delta
    prior_strength: float = 10.0     # pseudo-observations of the moneyness prior
    max_abs_delta: float = 1.25      # clip — a single option's |Δ| shouldn't exceed this
    eps: float = 1e-9

    def __post_init__(self) -> None:
        if self.delta_window < 3:
            raise ValueError("delta_window must be >= 3")
        if self.prior_strength < 0:
            raise ValueError("prior_strength must be >= 0")


def estimate_effective_delta(mark: pd.Series, spot: pd.Series,
                             slot: MoneynessSlot,
                             cfg: Optional[FairResponseConfig] = None,
                             ) -> pd.Series:
    """Rolling effective delta for one contract, shrunk toward the
    moneyness prior and clipped to the option's plausible sign/range.

    Returns a Series aligned to ``mark``."""
    cfg = cfg or FairResponseConfig()
    mark = pd.to_numeric(mark, errors="coerce")
    spot = pd.to_numeric(spot, errors="coerce")
    d_mark = mark.diff()
    d_spot = spot.diff()
    w = cfg.delta_window
    min_p = max(3, w // 2)
    cov = d_mark.rolling(w, min_periods=min_p).cov(d_spot)
    var = d_spot.rolling(w, min_periods=min_p).var()
    beta = cov / (var + cfg.eps)
    n = d_spot.rolling(w, min_periods=1).count()

    prior = float(slot.expected_signed_delta)
    # Sample-size shrinkage: weight the empirical beta by n/(n+prior_strength).
    w_emp = n / (n + cfg.prior_strength + cfg.eps)
    eff = w_emp * beta + (1.0 - w_emp) * prior
    eff = eff.fillna(prior)
    # Clip by option sign — a call's effective delta can't be negative,
    # a put's can't be positive.
    if slot.option_type == "CE":
        eff = eff.clip(lower=0.0, upper=cfg.max_abs_delta)
    else:
        eff = eff.clip(lower=-cfg.max_abs_delta, upper=0.0)
    return eff


def fair_response_frame(df: pd.DataFrame,
                        slot: MoneynessSlot,
                        cfg: Optional[FairResponseConfig] = None,
                        ) -> pd.DataFrame:
    """Compute the fair response + residual for one contract's series.

    ``df`` must have columns ``spot`` and ``mark`` (the clean mark price
    from Phase 2). Returns a copy with added columns:
      * ``eff_delta``   — live effective delta (shrunk toward prior)
      * ``fair_change`` — expected premium change = eff_delta · Δspot
      * ``actual_change`` — Δmark
      * ``residual``    — actual_change − fair_change (the per-bar surprise)
    """
    cfg = cfg or FairResponseConfig()
    required = {"spot", "mark"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"fair_response_frame missing columns: {sorted(missing)}")
    out = df.copy().reset_index(drop=True)
    spot = pd.to_numeric(out["spot"], errors="coerce")
    mark = pd.to_numeric(out["mark"], errors="coerce")
    eff = estimate_effective_delta(mark, spot, slot, cfg)
    d_spot = spot.diff()
    d_mark = mark.diff()
    fair_change = eff * d_spot
    out["eff_delta"] = eff.values
    out["fair_change"] = fair_change.values
    out["actual_change"] = d_mark.values
    out["residual"] = (d_mark - fair_change).values
    return out
