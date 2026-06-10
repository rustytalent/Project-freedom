"""Fill probability model — Stream K.M.2.

Predicts P(limit order fills before deadline) from:

  * limit_distance_from_best — how far the resting limit sits from
    the best opposing quote, in bps of price
  * bars_remaining — how many 5-minute bars until the order
    expires / time-stops out
  * vol_frac — ATR / price at the entry bar (regime proxy)
  * session_open / session_close — boolean phase indicators

Training data comes from V2's fill-policy outcomes (the V2 simulator
emits an ``entry_reason`` that captures whether a limit filled
generously, neutrally, or never). For v1 we synthesise the
positive/negative training signal directly from V2's outcome
labels; once live data lands we replace the synthetic targets with
real fills.

The trained probability plugs back into V3 in place of V2's
"assume target fills 100% of the time" approximation. V3 then
discounts target reward by ``p_fill * reward + (1 - p_fill) * (
    time-exit fallback)`` so the backtest stops over-counting fills.
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
    "distance_bps",
    "bars_remaining",
    "vol_frac",
    "session_open",
    "session_close",
]

FALLBACK_PROB: float = 0.65    # the V2 baseline neutral fill-policy assumption
PROB_FLOOR: float = 0.05
PROB_CEIL: float = 0.99


def _featurize(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["distance_bps"] = df.get("distance_bps", np.nan)
    out["bars_remaining"] = df.get("bars_remaining", np.nan)
    out["vol_frac"] = df.get("vol_frac", np.nan)
    phase = df.get("session_phase",
                   pd.Series("mid", index=df.index)).astype(str)
    out["session_open"] = (phase == "open").astype(float)
    out["session_close"] = (phase == "close").astype(float)
    return out


class FillProbabilityModel:
    """Public surface for V3: ``fit(events_df)`` and
    ``predict_prob(state_dict) -> float``.

    The training event frame has one row per "limit order would have
    been placed here" event with a binary ``filled`` target. The
    trainer fits a regression on the binary target with output
    clamped to (PROB_FLOOR, PROB_CEIL) so the V3 path always sees
    a valid probability.
    """

    def __init__(self, config: Optional[LGBMConfig] = None) -> None:
        self._wrapper = LGBMRegressorWrapper(
            feature_names=FEATURE_COLUMNS,
            fallback=FALLBACK_PROB,
            config=config,
            # Regression on a 0/1 target is fine for our purposes —
            # we're modelling a probability, not classifying. The
            # output is clamped at predict time. Using "regression"
            # rather than "binary" keeps the loss tolerant to
            # slightly out-of-range labels that some upstream
            # mappings produce (e.g. partial fills as 0.5).
            objective="regression",
        )

    def fit(
        self,
        events_df: pd.DataFrame,
        target_col: str = "filled",
    ) -> "FillProbabilityModel":
        if events_df.empty:
            return self
        X = _featurize(events_df)
        y = events_df[target_col].astype(float).clip(0.0, 1.0).to_numpy()
        self._wrapper.fit(X, y)
        return self

    def predict_prob(self, state: Dict[str, Any]) -> float:
        df = pd.DataFrame([state])
        X = _featurize(df)
        raw = float(self._wrapper.predict(X)[0])
        if not math.isfinite(raw):
            raw = FALLBACK_PROB
        return max(PROB_FLOOR, min(raw, PROB_CEIL))

    def save(self, path: Path) -> None:
        self._wrapper.save(path)

    @classmethod
    def load(cls, path: Path) -> "FillProbabilityModel":
        obj = cls()
        obj._wrapper = LGBMRegressorWrapper.load(path)
        return obj

    @property
    def is_fitted(self) -> bool:
        return self._wrapper.is_fitted

    @property
    def trained_n(self) -> int:
        return self._wrapper.trained_n

    @property
    def oos_rmse(self) -> Optional[float]:
        return self._wrapper.oos_metric
