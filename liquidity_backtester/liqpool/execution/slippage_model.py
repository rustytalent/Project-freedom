"""Slippage realisation model — Stream K.M.1.

Predicts realised slippage in basis points from the entry context.
Training data comes from paired (intended_price, realised_price)
observations — either:

  * V2 simulator output (entry_slippage_bps / exit_slippage_bps
    columns of ExecutionTradeV2.to_dict()). Bootstrap for v1 since
    no live fill data exists yet.
  * Once live trades start landing: real broker-fill prices vs the
    intended price. Strictly preferred over V2 self-training.

The trained model plugs back into V3 in place of the V2 hand-tuned
``compute_slippage_bps`` formula, so V3's backtests reflect the
slippage distribution the engine actually realises rather than
V2's rule-based approximation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ._base import LGBMConfig, LGBMRegressorWrapper


# Feature columns the trainer extracts from a trade frame and the
# inference path expects. Adding a column = update both ends.
FEATURE_COLUMNS: List[str] = [
    "vol_frac",            # ATR / price at the entry bar
    "distance_atr",        # ATR-normalised distance to the structural level
    "size_notional",       # entry_notional_inr
    "side_long",           # 1 if long else 0
    "session_open",        # 1 if session_phase == "open"
    "session_close",       # 1 if session_phase == "close"
    "mode_market",         # 1 if entry was a market-style fill
    "mode_limit",          # 1 if entry was a limit fill
]

# When the model isn't trained, V3 falls back to the V2 default
# (2 bps). Predict-time also clamps any pathological prediction
# into [0.5, 50] bps so a stale model can't silently zero or
# explode slippage.
FALLBACK_BPS: float = 2.0
PREDICT_FLOOR_BPS: float = 0.5
PREDICT_CEIL_BPS: float = 50.0


def _first_present(df: pd.DataFrame, names: Sequence[str],
                   default: Any = np.nan) -> pd.Series:
    """Return the first column in `names` that exists in df, else a
    constant-default series. Used to support both the V2 trade-frame
    column names (entry_notional_inr, dist_atr_at_last_bar, etc.) and
    the V3 inference shape (size_notional, distance_atr).
    """
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(default, index=df.index)


def _featurize(df: pd.DataFrame) -> pd.DataFrame:
    """Pull the inference features out of a trade-row dataframe.

    Accepts BOTH the V2 trade-frame schema and the V3 inference
    schema by trying each canonical name in order. Anything truly
    absent stays NaN (the LightGBM booster handles those natively
    at predict time).
    """
    out = pd.DataFrame(index=df.index)
    out["vol_frac"] = _first_present(df, ["vol_frac"])
    if out["vol_frac"].isna().all():
        atr = _first_present(df, ["atr_at_entry"])
        ent = _first_present(df, ["entry"])
        out["vol_frac"] = atr.astype(float) / ent.replace(0, np.nan).astype(float)
    out["distance_atr"] = _first_present(
        df, ["distance_atr", "dist_atr_at_last_bar"]
    ).astype(float)
    out["size_notional"] = _first_present(
        df, ["size_notional", "entry_notional_inr", "notional_inr"]
    ).astype(float)
    if "side_long" in df.columns:
        out["side_long"] = df["side_long"].astype(float)
    else:
        side = _first_present(df, ["side"], "long").astype(str)
        out["side_long"] = (side == "long").astype(float)
    if "session_open" in df.columns:
        out["session_open"] = df["session_open"].astype(float)
        out["session_close"] = _first_present(df, ["session_close"], 0.0).astype(float)
    else:
        phase = _first_present(df, ["session_phase"], "mid").astype(str)
        out["session_open"] = (phase == "open").astype(float)
        out["session_close"] = (phase == "close").astype(float)
    if "mode_market" in df.columns:
        out["mode_market"] = df["mode_market"].astype(float)
        out["mode_limit"] = _first_present(df, ["mode_limit"], 0.0).astype(float)
    else:
        mode = _first_present(df, ["mode", "entry_reason"], "").astype(str)
        mode_lc = mode.str.lower()
        out["mode_market"] = mode_lc.str.contains("market").astype(float)
        out["mode_limit"] = mode_lc.str.contains("limit").astype(float)
    return out


class SlippageRealisationModel:
    """Public surface for V3: ``fit(trades_df)`` and
    ``predict_bps(state_dict) -> float``.

    Trained from a long-format trade frame that carries one row per
    fill (entry leg, exit leg). The target column defaults to
    ``slippage_bps``; the V2 trade frame stores them as
    ``entry_slippage_bps`` / ``exit_slippage_bps`` so the caller
    splits and concatenates before passing in.
    """

    def __init__(self, config: Optional[LGBMConfig] = None) -> None:
        self._wrapper = LGBMRegressorWrapper(
            feature_names=FEATURE_COLUMNS,
            fallback=FALLBACK_BPS,
            config=config,
            objective="regression",
        )

    @staticmethod
    def trade_frame_to_long(trades: pd.DataFrame) -> pd.DataFrame:
        """Convert an ``ExecutionTradeV2.to_dict()`` frame into a
        long-format (one row per leg) training frame.

        Each input row produces two output rows: entry and exit.
        The ``slippage_bps`` column is the per-leg realised slippage
        the V2 simulator emitted; ``leg`` records which side.
        """
        if trades.empty:
            return trades
        rows: List[Dict[str, Any]] = []
        for _, r in trades.iterrows():
            base = {
                "vol_frac": _safe_float(r.get("atr_at_entry"))
                            / max(_safe_float(r.get("entry"), 1.0), 1e-9),
                "distance_atr": _safe_float(r.get("distance_atr",
                                                  r.get("dist_atr_at_last_bar", 0.0))),
                "size_notional": _safe_float(r.get("entry_notional_inr",
                                                   r.get("notional_inr", 0.0))),
                "side": str(r.get("side", "long")),
                "mode": str(r.get("mode", "")),
            }
            rows.append({
                **base, "leg": "entry",
                "session_phase": _phase_from_ts(r.get("entry_at")),
                "slippage_bps": _safe_float(r.get("entry_slippage_bps")),
            })
            rows.append({
                **base, "leg": "exit",
                "session_phase": _phase_from_ts(r.get("exit_at")),
                "slippage_bps": _safe_float(r.get("exit_slippage_bps")),
            })
        return pd.DataFrame(rows)

    def fit(
        self,
        trades_long: pd.DataFrame,
        target_col: str = "slippage_bps",
    ) -> "SlippageRealisationModel":
        if trades_long.empty:
            return self
        X = _featurize(trades_long)
        y = trades_long[target_col].astype(float).to_numpy()
        # Cap absurd target values to keep the loss well-behaved.
        y = np.clip(y, PREDICT_FLOOR_BPS, PREDICT_CEIL_BPS * 2)
        self._wrapper.fit(X, y)
        return self

    def predict_bps(self, state: Dict[str, Any]) -> float:
        """V3 hot path: pass the entry-state dict, get a clamped bps
        estimate back. Falls through to the configured fallback when
        the model isn't trained or the input is non-finite."""
        df = pd.DataFrame([state])
        X = _featurize(df)
        raw = float(self._wrapper.predict(X)[0])
        # NaN / inf guard before the clamp — max(0.5, nan) returns nan.
        if not math.isfinite(raw):
            raw = FALLBACK_BPS
        # Clamp to the public predict-band so a stale or thinly-
        # trained model can't silently break V3.
        return max(PREDICT_FLOOR_BPS, min(raw, PREDICT_CEIL_BPS))

    def save(self, path: Path) -> None:
        self._wrapper.save(path)

    @classmethod
    def load(cls, path: Path) -> "SlippageRealisationModel":
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


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _phase_from_ts(ts: Any) -> str:
    """Reconstruct the V2 session phase label from an ISO timestamp
    string or pandas timestamp. Defaults to "mid" on parse failures."""
    try:
        t = pd.Timestamp(ts).time()
    except Exception:
        return "mid"
    if t.hour < 10:
        return "open"
    if t.hour >= 14 and (t.hour > 14 or t.minute >= 30):
        return "close"
    return "mid"
