"""MeanReversionAlpha — z-score-based intraday mean reversion.

Thesis: short-term price moves that extend significantly beyond a rolling
mean tend to revert. Compute a z-score of the current close vs. its
N-bar mean and std. When |z| exceeds a threshold and recent volatility
isn't pathological, fade the extension:

  z > +threshold  ->  SHORT  (overextended up; revert down)
  z < -threshold  ->  LONG   (overextended down; revert up)

This alpha is structurally INDEPENDENT of the liquidity-pool engine — it
doesn't use pools at all. Useful in the arsenal because if it has edge
in pockets where pool_reach doesn't, the two diversify; if it doesn't,
that's also informative (it tells us micro-momentum dominates over
mean-reversion at this timeframe).

Barriers: stop_atr=0.5, target_atr=1.5 (closer than pool_reach because
mean-revert moves tend to be tighter), horizon_bars=12 (1 hour — these
are short-term setups).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..base import Alpha, AlphaSignal


class MeanReversionAlpha(Alpha):
    """Z-score extension reversal."""

    @property
    def name(self) -> str:
        return "mean_reversion"

    @property
    def description(self) -> str:
        return (
            "Z-score-based intraday mean reversion. Compute close vs. 20-bar "
            "rolling mean/std; when |z| > 2.0, fade the extension. Stop 0.5 "
            "ATR, target 1.5 ATR, horizon 12 bars (~1h)."
        )

    Z_THRESHOLD: float = 2.0          # |z| above this triggers a signal
    ZSCORE_WINDOW: int = 20           # rolling window for mean/std
    MIN_BAR_GAP: int = 6              # don't re-fire within this many bars
    STOP_ATR: float = 0.5
    TARGET_ATR: float = 1.5
    HORIZON_BARS: int = 12

    def candidates(self, *, symbol: str, df_base: pd.DataFrame,
                   atr_series: pd.Series,
                   extra: Optional[Dict[str, Any]] = None,
                   ) -> List[AlphaSignal]:
        if df_base is None or len(df_base) < self.ZSCORE_WINDOW + 2:
            return []
        close = df_base["close"].astype(float).to_numpy()
        # Rolling stats over [i-W, i-1] (excluding the decision bar itself
        # for honest causality). Use pandas for the rolling and shift by 1.
        s = pd.Series(close)
        roll_mean = s.rolling(self.ZSCORE_WINDOW, min_periods=self.ZSCORE_WINDOW).mean().shift(1).to_numpy()
        roll_std = s.rolling(self.ZSCORE_WINDOW, min_periods=self.ZSCORE_WINDOW).std(ddof=1).shift(1).to_numpy()

        signals: List[AlphaSignal] = []
        last_fire_idx = -10**9
        for i in range(self.ZSCORE_WINDOW + 1, len(df_base)):
            mu = roll_mean[i]
            sigma = roll_std[i]
            if not (np.isfinite(mu) and np.isfinite(sigma) and sigma > 0):
                continue
            z = (close[i] - mu) / sigma
            if abs(z) < self.Z_THRESHOLD:
                continue
            if i - last_fire_idx < self.MIN_BAR_GAP:
                continue
            side = "short" if z > 0 else "long"
            signals.append(AlphaSignal(
                alpha_name=self.name, symbol=symbol,
                decision_at=df_base.index[i],
                decision_idx=i,
                side=side,
                entry_reference=float(close[i]),
                stop_atr=self.STOP_ATR,
                target_atr=self.TARGET_ATR,
                horizon_bars=self.HORIZON_BARS,
                # Confidence scales with z-magnitude, capped at 1.0.
                confidence=float(min(1.0, abs(z) / 4.0)),
                state={
                    "zscore": float(z),
                    "rolling_mean": float(mu),
                    "rolling_std": float(sigma),
                },
            ))
            last_fire_idx = i
        return signals

    def regime_tags(self, signal, df_base, atr_series):
        base = super().regime_tags(signal, df_base, atr_series)
        z = float(signal.state.get("zscore", 0.0))
        if abs(z) >= 3.0:
            base["zscore_bucket"] = "extreme_3plus"
        elif abs(z) >= 2.5:
            base["zscore_bucket"] = "high_2.5_to_3"
        else:
            base["zscore_bucket"] = "moderate_2_to_2.5"
        return base
