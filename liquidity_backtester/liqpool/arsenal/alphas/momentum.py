"""MomentumAlpha — multi-bar trend-following continuation.

Thesis: when a stock has shown consistent directional movement in the
last K bars AND the current volatility regime isn't degenerate, the
move tends to continue in the short term.

Trigger conditions for LONG:
  * 12-bar return > +TREND_THRESHOLD ATR (price has moved up significantly)
  * ADX-like trend strength (recent range expansion)
  * Current ATR not in the extreme tail (vol_ratio < REGIME_CEILING)

Mirror for SHORT.

Barriers: stop_atr=0.75 (wider than mean-reversion because we're going
with the trend and want room), target_atr=2.0, horizon_bars=24 (~2h).

This is the trend-continuation complement to MeanReversionAlpha. In
practice they should compete: when one fires, the other often shouldn't.
The arsenal evaluator's combination view will show that.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..base import Alpha, AlphaSignal


class MomentumAlpha(Alpha):
    """Multi-bar momentum continuation."""

    @property
    def name(self) -> str:
        return "momentum"

    @property
    def description(self) -> str:
        return (
            "Multi-bar momentum continuation. Trigger when the 12-bar log "
            "return exceeds TREND_THRESHOLD * ATR and the volatility regime "
            "is not pathological. Stop 0.75 ATR, target 2.0 ATR, horizon 24 "
            "bars (~2h)."
        )

    MOMENTUM_WINDOW: int = 12         # bars to measure momentum over
    TREND_THRESHOLD: float = 1.5      # log-return in ATR units to trigger
    REGIME_CEILING: float = 2.5       # skip if current ATR / median ATR > this
    MIN_BAR_GAP: int = 12             # don't re-fire within this many bars
    STOP_ATR: float = 0.75
    TARGET_ATR: float = 2.0
    HORIZON_BARS: int = 24

    def candidates(self, *, symbol: str, df_base: pd.DataFrame,
                   atr_series: pd.Series,
                   extra: Optional[Dict[str, Any]] = None,
                   ) -> List[AlphaSignal]:
        if df_base is None or len(df_base) < self.MOMENTUM_WINDOW + 30:
            return []
        close = df_base["close"].astype(float).to_numpy()
        atr_arr = atr_series.to_numpy(dtype=float)

        # log return over MOMENTUM_WINDOW bars, ATR-normalized.
        # ret_atr[i] = log(close[i] / close[i - W]) / atr[i]   (atr in price units)
        # Because atr is in price units and log returns are dimensionless, we
        # convert: use absolute close change / atr instead for direct compare.
        n = len(close)
        ret_atr = np.zeros(n, dtype=float)
        for i in range(self.MOMENTUM_WINDOW, n):
            prev = close[i - self.MOMENTUM_WINDOW]
            if prev <= 0 or atr_arr[i] <= 0:
                continue
            ret_atr[i] = (close[i] - prev) / atr_arr[i]

        # Volatility regime: ATR(now) vs median of last 60 bars.
        atr_series_pd = pd.Series(atr_arr)
        median_60 = atr_series_pd.rolling(60, min_periods=20).median().to_numpy()
        regime = np.where(median_60 > 0, atr_arr / median_60, 1.0)

        signals: List[AlphaSignal] = []
        last_fire_idx = -10**9
        for i in range(self.MOMENTUM_WINDOW + 60, n):
            if regime[i] > self.REGIME_CEILING or not np.isfinite(regime[i]):
                continue            # extreme volatility regime, skip
            r = ret_atr[i]
            if not np.isfinite(r):
                continue
            if abs(r) < self.TREND_THRESHOLD:
                continue
            if i - last_fire_idx < self.MIN_BAR_GAP:
                continue
            side = "long" if r > 0 else "short"
            signals.append(AlphaSignal(
                alpha_name=self.name, symbol=symbol,
                decision_at=df_base.index[i],
                decision_idx=i,
                side=side,
                entry_reference=float(close[i]),
                stop_atr=self.STOP_ATR,
                target_atr=self.TARGET_ATR,
                horizon_bars=self.HORIZON_BARS,
                confidence=float(min(1.0, abs(r) / 3.0)),   # |r|=3 -> 1.0
                state={
                    "ret_atr_12b": float(r),
                    "vol_regime": float(regime[i]),
                },
            ))
            last_fire_idx = i
        return signals

    def regime_tags(self, signal, df_base, atr_series):
        base = super().regime_tags(signal, df_base, atr_series)
        regime = float(signal.state.get("vol_regime", 1.0))
        if regime < 0.75:
            base["vol_regime_bucket"] = "low_vol"
        elif regime < 1.25:
            base["vol_regime_bucket"] = "normal_vol"
        else:
            base["vol_regime_bucket"] = "high_vol"
        return base
