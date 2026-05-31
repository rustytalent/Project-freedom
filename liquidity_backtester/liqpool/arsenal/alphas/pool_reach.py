"""LiquidityPoolReachAlpha — the existing strategy as an alpha.

Wraps the OOS-pool execution path that lives in
:mod:`liqpool.execution_backtest`. Emits one signal per touched OOS pool
in the asset's bundle, with:

  * side = long if pool side == "low" (demand below price), short if
    pool side == "high" (supply above price).
  * decision_at = the bar BEFORE the pool's first touch (so entry on the
    next bar's open mirrors the V1 touch_confirmed mode).
  * stop_atr = 0.5, target_atr = 2.0, horizon_bars = 40 — the V1 defaults.
  * confidence = pool's Q score if present, else 0.5.

The thesis: when price touches a multi-timeframe confluence zone, there
is a non-trivial reaction. This is the existing system reformulated as
an arsenal member so it can be compared to other alphas head-to-head.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ..base import Alpha, AlphaSignal


class LiquidityPoolReachAlpha(Alpha):
    """Touched-pool entry, V1 default barriers, alpha-framework wrapper."""

    @property
    def name(self) -> str:
        return "pool_reach"

    @property
    def description(self) -> str:
        return (
            "Trade the touch of a multi-timeframe liquidity confluence zone. "
            "Long demand zones from above, short supply zones from below. "
            "Stop 0.5 ATR; target 2.0 ATR; horizon 40 bars (~3.3h)."
        )

    DEFAULT_STOP_ATR: float = 0.5
    DEFAULT_TARGET_ATR: float = 2.0
    DEFAULT_HORIZON_BARS: int = 40

    def candidates(self, *, symbol: str, df_base: pd.DataFrame,
                   atr_series: pd.Series,
                   extra: Optional[Dict[str, Any]] = None,
                   ) -> List[AlphaSignal]:
        extra = extra or {}
        ad = extra.get("asset_data")
        if ad is None:
            return []
        pools = ad.walkforward.oos_pools
        results = ad.walkforward.oos_results

        signals: List[AlphaSignal] = []
        idx_values = df_base.index.values
        for pool, result in zip(pools, results):
            if result.touched_at is None:
                continue
            touch_idx = int(df_base.index.searchsorted(
                pd.Timestamp(result.touched_at), side="left"
            ))
            if touch_idx <= 0 or touch_idx >= len(df_base):
                continue
            # Decision happens at touch close; entry next bar open (evaluator
            # handles that step). So decision_idx = touch bar.
            decision_idx = touch_idx
            side = "long" if pool.side == "low" else "short"
            entry_ref = float(pool.price_high if pool.side == "low"
                              else pool.price_low)
            signals.append(AlphaSignal(
                alpha_name=self.name, symbol=symbol,
                decision_at=df_base.index[decision_idx],
                decision_idx=decision_idx,
                side=side,
                entry_reference=entry_ref,
                stop_atr=self.DEFAULT_STOP_ATR,
                target_atr=self.DEFAULT_TARGET_ATR,
                horizon_bars=self.DEFAULT_HORIZON_BARS,
                confidence=float(getattr(result, "pool_quality", 0.5) or 0.5),
                state={
                    "pool_idx": int(result.pool_idx),
                    "pool_score": float(pool.score),
                    "factor": _headline_for_pool(pool),
                    "tf_count": int(len(set(pool.tfs))),
                },
            ))
        return signals

    def regime_tags(self, signal, df_base, atr_series):
        base = super().regime_tags(signal, df_base, atr_series)
        # Add alpha-specific regime dimensions.
        base["factor"] = str(signal.state.get("factor", "OTHER"))
        base["tf_bucket"] = _tf_bucket(int(signal.state.get("tf_count", 0)))
        return base


def _tf_bucket(n: int) -> str:
    if n <= 1:
        return "1tf"
    if n == 2:
        return "2tf"
    if n == 3:
        return "3tf"
    return "4plus_tf"


def _headline_for_pool(pool) -> str:
    """Lightweight rewrite of liqpool.execution_backtest._headline_factor
    to avoid an import cycle (execution_backtest imports from this module
    via the registry path)."""
    priority = ("OB", "FVG", "EQHL", "REJ", "ORB", "HVN", "SWING",
                "PD", "PW", "PM")
    families = set()
    for c in pool.contributors:
        src = str(getattr(c, "source", "")).split("@", 1)[0]
        if src.startswith("OB_"):
            families.add("OB")
        elif src.startswith("FVG_"):
            families.add("FVG")
        elif src in ("EQH", "EQL"):
            families.add("EQHL")
        elif src.startswith("REJ_"):
            families.add("REJ")
        elif src.startswith("ORB_"):
            families.add("ORB")
        elif src == "HVN":
            families.add("HVN")
        elif src.startswith("SWING_"):
            families.add("SWING")
        elif src.startswith("PD"):
            families.add("PD")
        elif src.startswith("PW"):
            families.add("PW")
        elif src.startswith("PM"):
            families.add("PM")
    for item in priority:
        if item in families:
            return item
    return "OTHER"
