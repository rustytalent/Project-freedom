"""Time-to-event survival model — Stream K.M.3.

Predicts expected bars to target and expected bars to stop given
the entry context. Both heads share the same feature set; the
caller picks which to ask for.

For v1 we implement this as two parallel GBM regressors on the
*observed* time-to-event from V2 trades. Strict Cox-style
survival models (with censoring) come in v2 — they're more
principled but require censoring labels (time-exit trades are
right-censored at horizon_bars, not "took 150 bars to target").
The v1 regression approach is OK because the V2 trade output's
``bars_held`` captures the observed barrier hit; ``outcome``
encodes which barrier hit so the trainer can filter to the right
subset.

The output plugs into V3 in two places:

  1. Sizing: an entry expected to take 80 bars to resolve gets
     a smaller position than one expected to resolve in 15 — slow
     trades tie up margin and accumulate time-decay costs.
  2. Position aging: V3 can compare actual bars_held vs
     E[bars_to_target] and exit early when a trade is materially
     past its expected resolution window without progress.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ._base import LGBMConfig, LGBMRegressorWrapper


FEATURE_COLUMNS: List[str] = [
    "distance_atr",
    "vol_frac",
    "side_long",
    "session_open",
    "session_close",
    "tf_count",       # number of timeframes the contributing pool spans
    "score",          # the pool's composite score from the detector
]

FALLBACK_BARS_TO_TARGET: float = 30.0
FALLBACK_BARS_TO_STOP: float = 30.0
BARS_FLOOR: float = 1.0
BARS_CEIL: float = 400.0    # MIS horizon cap


def _featurize(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["distance_atr"] = df.get("distance_atr", df.get(
        "dist_atr_at_last_bar", np.nan,
    ))
    out["vol_frac"] = df.get("vol_frac", np.nan)
    side = df.get("side", pd.Series("long", index=df.index)).astype(str)
    out["side_long"] = (side == "long").astype(float)
    phase = df.get("session_phase",
                   pd.Series("mid", index=df.index)).astype(str)
    out["session_open"] = (phase == "open").astype(float)
    out["session_close"] = (phase == "close").astype(float)
    out["tf_count"] = df.get("tf_count", np.nan)
    out["score"] = df.get("score", np.nan)
    return out


class TimeToEventModel:
    """Two-head time-to-event predictor.

    Fit takes a trades frame containing per-row ``outcome`` and
    ``bars_held`` (the V2 output columns). The trainer splits the
    rows by outcome and fits one regressor per head.
    """

    def __init__(self, config: Optional[LGBMConfig] = None) -> None:
        self._target_head = LGBMRegressorWrapper(
            feature_names=FEATURE_COLUMNS,
            fallback=FALLBACK_BARS_TO_TARGET,
            config=config,
        )
        self._stop_head = LGBMRegressorWrapper(
            feature_names=FEATURE_COLUMNS,
            fallback=FALLBACK_BARS_TO_STOP,
            config=config,
        )

    def fit(self, trades: pd.DataFrame) -> "TimeToEventModel":
        if trades.empty or "outcome" not in trades.columns:
            return self
        outcome = trades["outcome"].astype(str)
        target_rows = trades[outcome.str.contains("target", case=False)]
        stop_rows = trades[outcome.str.contains("stop", case=False)]
        if not target_rows.empty:
            X = _featurize(target_rows)
            y = np.clip(
                target_rows["bars_held"].astype(float).to_numpy(),
                BARS_FLOOR, BARS_CEIL,
            )
            self._target_head.fit(X, y)
        if not stop_rows.empty:
            X = _featurize(stop_rows)
            y = np.clip(
                stop_rows["bars_held"].astype(float).to_numpy(),
                BARS_FLOOR, BARS_CEIL,
            )
            self._stop_head.fit(X, y)
        return self

    def predict_bars_to_target(self, state: Dict[str, Any]) -> float:
        df = pd.DataFrame([state])
        X = _featurize(df)
        raw = float(self._target_head.predict(X)[0])
        if not math.isfinite(raw):
            raw = FALLBACK_BARS_TO_TARGET
        return max(BARS_FLOOR, min(raw, BARS_CEIL))

    def predict_bars_to_stop(self, state: Dict[str, Any]) -> float:
        df = pd.DataFrame([state])
        X = _featurize(df)
        raw = float(self._stop_head.predict(X)[0])
        if not math.isfinite(raw):
            raw = FALLBACK_BARS_TO_STOP
        return max(BARS_FLOOR, min(raw, BARS_CEIL))

    # -- persistence -------------------------------------------------

    def save(self, root: Path) -> None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self._target_head.save(root / "target_head.txt")
        self._stop_head.save(root / "stop_head.txt")

    @classmethod
    def load(cls, root: Path) -> "TimeToEventModel":
        obj = cls()
        root = Path(root)
        obj._target_head = LGBMRegressorWrapper.load(root / "target_head.txt")
        obj._stop_head = LGBMRegressorWrapper.load(root / "stop_head.txt")
        return obj

    @property
    def is_fitted(self) -> bool:
        return self._target_head.is_fitted and self._stop_head.is_fitted
