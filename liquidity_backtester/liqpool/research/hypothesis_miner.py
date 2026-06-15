"""Small, honest hypothesis miner.

This is the first gear of the alpha factory. It does not claim to find
tradable edges by itself. It samples transparent rule hypotheses from a
numeric feature frame and ranks them on a provided forward-return target.
The ranked candidates must still go through Arsenal, walk-forward,
cost stress, nulls, and paper/live validation before they can matter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import json
import math
import random

import numpy as np
import pandas as pd


OPS = (">", "<")


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    threshold: float

    def evaluate(self, frame: pd.DataFrame) -> pd.Series:
        if self.feature not in frame.columns:
            return pd.Series(False, index=frame.index)
        x = pd.to_numeric(frame[self.feature], errors="coerce")
        if self.op == ">":
            return x > self.threshold
        if self.op == "<":
            return x < self.threshold
        raise ValueError(f"unsupported op {self.op!r}")

    def to_dict(self) -> Dict:
        return {
            "feature": self.feature,
            "op": self.op,
            "threshold": float(self.threshold),
        }


@dataclass(frozen=True)
class HypothesisSpec:
    name: str
    side: str
    conditions: List[Condition]
    hold_bars: int = 12
    stop_atr: float = 1.0
    target_atr: float = 2.0
    metadata: Dict = field(default_factory=dict)

    def mask(self, frame: pd.DataFrame) -> pd.Series:
        if not self.conditions:
            return pd.Series(False, index=frame.index)
        out = pd.Series(True, index=frame.index)
        for cond in self.conditions:
            out &= cond.evaluate(frame)
        return out.fillna(False)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "side": self.side,
            "conditions": [c.to_dict() for c in self.conditions],
            "hold_bars": int(self.hold_bars),
            "stop_atr": float(self.stop_atr),
            "target_atr": float(self.target_atr),
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass(frozen=True)
class HypothesisMetrics:
    name: str
    trades: int
    mean_r: float
    sharpe: float
    win_rate: float
    max_drawdown_r: float
    t_stat: float
    score: float
    spec_json: str

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "trades": int(self.trades),
            "mean_R": float(self.mean_r),
            "sharpe": float(self.sharpe),
            "win_rate": float(self.win_rate),
            "max_drawdown_R": float(self.max_drawdown_r),
            "t_stat": float(self.t_stat),
            "score": float(self.score),
            "spec_json": self.spec_json,
        }


def sample_random_hypotheses(
    frame: pd.DataFrame,
    *,
    n: int,
    seed: int = 7,
    feature_columns: Optional[Sequence[str]] = None,
    max_conditions: int = 3,
    quantiles: Sequence[float] = (0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85),
) -> List[HypothesisSpec]:
    """Sample transparent AND-rule hypotheses from numeric columns."""
    rng = random.Random(seed)
    features = _usable_features(frame, feature_columns)
    if not features:
        return []
    thresholds = _feature_thresholds(frame, features, quantiles)
    out: List[HypothesisSpec] = []
    for i in range(n):
        n_conds = rng.randint(1, max(1, max_conditions))
        picked = rng.sample(features, k=min(n_conds, len(features)))
        conds: List[Condition] = []
        for feature in picked:
            pool = thresholds.get(feature) or [0.0]
            conds.append(Condition(
                feature=feature,
                op=rng.choice(OPS),
                threshold=float(rng.choice(pool)),
            ))
        side = rng.choice(["long", "short"])
        out.append(HypothesisSpec(
            name=f"mined_{seed}_{i:05d}",
            side=side,
            conditions=conds,
            hold_bars=rng.choice([3, 6, 12, 24, 36, 60]),
            stop_atr=rng.choice([0.5, 0.75, 1.0, 1.5, 2.0]),
            target_atr=rng.choice([0.75, 1.0, 1.5, 2.0, 3.0]),
            metadata={"generator": "random_rule_v1", "seed": seed},
        ))
    return out


def evaluate_hypothesis(
    frame: pd.DataFrame,
    spec: HypothesisSpec,
    *,
    forward_return_col: str,
    cost_r_col: Optional[str] = None,
    min_trades: int = 30,
) -> HypothesisMetrics:
    """Score one hypothesis on an already prepared forward-return column.

    ``forward_return_col`` should be expressed in R or another stable
    per-trade return unit. For shorts, the sign is inverted. If a
    ``cost_r_col`` exists, it is subtracted per selected row.
    """
    mask = spec.mask(frame)
    if forward_return_col not in frame.columns:
        raise KeyError(f"missing forward return column {forward_return_col!r}")
    r = pd.to_numeric(frame.loc[mask, forward_return_col], errors="coerce")
    if spec.side == "short":
        r = -r
    if cost_r_col and cost_r_col in frame.columns:
        r = r - pd.to_numeric(frame.loc[mask, cost_r_col], errors="coerce").fillna(0.0)
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(r))
    if n == 0:
        return _empty_metrics(spec)
    mean = float(r.mean())
    std = float(r.std(ddof=1)) if n > 1 else 0.0
    sharpe = mean / std if std > 1e-12 else 0.0
    t_stat = mean / (std / math.sqrt(n)) if std > 1e-12 and n > 1 else 0.0
    win_rate = float((r > 0).mean())
    dd = _max_drawdown(r.to_numpy(dtype=float))
    sparsity_penalty = 0.0 if n >= min_trades else (min_trades - n) / max(min_trades, 1)
    score = mean * min(1.0, math.sqrt(n / max(min_trades, 1))) + 0.05 * sharpe - 0.10 * sparsity_penalty
    return HypothesisMetrics(
        name=spec.name,
        trades=n,
        mean_r=mean,
        sharpe=float(sharpe),
        win_rate=win_rate,
        max_drawdown_r=dd,
        t_stat=float(t_stat),
        score=float(score),
        spec_json=spec.to_json(),
    )


def rank_random_hypotheses(
    frame: pd.DataFrame,
    *,
    forward_return_col: str,
    cost_r_col: Optional[str] = None,
    n: int = 1000,
    seed: int = 7,
    feature_columns: Optional[Sequence[str]] = None,
    min_trades: int = 30,
) -> pd.DataFrame:
    """Sample, score, and return a ranking dataframe."""
    specs = sample_random_hypotheses(
        frame, n=n, seed=seed, feature_columns=feature_columns
    )
    rows = [
        evaluate_hypothesis(
            frame,
            spec,
            forward_return_col=forward_return_col,
            cost_r_col=cost_r_col,
            min_trades=min_trades,
        ).to_dict()
        for spec in specs
    ]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["score", "mean_R", "trades"], ascending=[False, False, False]
    ).reset_index(drop=True)


def _usable_features(frame: pd.DataFrame, feature_columns: Optional[Sequence[str]]) -> List[str]:
    if feature_columns:
        candidates = [c for c in feature_columns if c in frame.columns]
    else:
        candidates = list(frame.columns)
    out = []
    for c in candidates:
        s = pd.to_numeric(frame[c], errors="coerce")
        if s.notna().sum() >= 20 and s.nunique(dropna=True) >= 5:
            out.append(c)
    return out


def _feature_thresholds(
    frame: pd.DataFrame,
    features: Iterable[str],
    quantiles: Sequence[float],
) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for feature in features:
        s = pd.to_numeric(frame[feature], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if s.empty:
            continue
        vals = [float(s.quantile(q)) for q in quantiles]
        out[feature] = sorted(set(v for v in vals if math.isfinite(v)))
    return out


def _max_drawdown(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    equity = np.cumsum(values)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min())


def _empty_metrics(spec: HypothesisSpec) -> HypothesisMetrics:
    return HypothesisMetrics(
        name=spec.name,
        trades=0,
        mean_r=0.0,
        sharpe=0.0,
        win_rate=0.0,
        max_drawdown_r=0.0,
        t_stat=0.0,
        score=-1.0,
        spec_json=spec.to_json(),
    )

