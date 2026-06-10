"""M.5 — Detector trust router.

Per-(factor, regime) posterior trust in each detector family,
learned from the pools the detectors actually fired on and how
those pools resolved.

The detector audit's biggest finding: every detector threshold is a
magic number with no validation, and detectors are equal-voters in
pool scoring. This model replaces equal-voting with empirical trust:

    trust(factor, regime) = shrunk P(pool respected | factor, regime)

Shrinkage reuses the Stream N beta-binomial empirical Bayes — small
(factor, regime) buckets shrink toward the global respect rate, big
buckets keep their observed rate. No LightGBM needed at v1: the
bucket space is tiny (≈ 8 factors × 4 regimes) and a parametric
posterior beats a tree model on 32 cells.

Data shape (one row per resolved pool):
    factor   — the pool's dominant factor family (Pool factor string)
    regime   — coarse regime label at formation ("trend"/"range"/...)
    respected — binary outcome from the walkforward tester

Usage at scoring time:
    router.trust("SWEEP", "trend") -> 0.61
The pool builder can multiply contributor strength by trust, turning
the equal-voter median into a trust-weighted aggregate.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from ..calibration.empirical_bayes import (
    BetaBinomialPrior,
    fit_beta_binomial_prior,
)

FALLBACK_TRUST: float = 0.5


class DetectorTrustRouter:
    """Empirical-Bayes trust per (factor, regime) bucket."""

    def __init__(self) -> None:
        self._prior: Optional[BetaBinomialPrior] = None
        self._counts: Dict[Tuple[str, str], Tuple[float, float]] = {}

    def fit(self, outcomes: pd.DataFrame) -> "DetectorTrustRouter":
        """``outcomes`` columns: factor, regime, respected (0/1)."""
        if outcomes.empty:
            return self
        need = {"factor", "regime", "respected"}
        if not need.issubset(outcomes.columns):
            raise ValueError(f"outcomes needs columns {need}")
        counts: Dict[Tuple[str, str], Tuple[float, float]] = {}
        for (factor, regime), g in outcomes.groupby(["factor", "regime"]):
            hits = float(g["respected"].astype(float).sum())
            n = float(len(g))
            counts[(str(factor), str(regime))] = (hits, n)
        self._counts = counts
        self._prior = fit_beta_binomial_prior(list(counts.values()))
        return self

    def trust(self, factor: str, regime: str) -> float:
        """Posterior-mean respect rate for the bucket. Unknown
        buckets get the global prior mean; unfit router returns the
        max-entropy fallback."""
        if self._prior is None:
            return FALLBACK_TRUST
        key = (str(factor), str(regime))
        hits, n = self._counts.get(key, (0.0, 0.0))
        return float(self._prior.shrink(hits, n))

    def trust_table(self) -> pd.DataFrame:
        """Full bucket table for the audit / sensitivity report."""
        if self._prior is None:
            return pd.DataFrame()
        rows = []
        for (factor, regime), (hits, n) in sorted(self._counts.items()):
            rows.append({
                "factor": factor, "regime": regime,
                "n": int(n), "raw_rate": hits / n if n else float("nan"),
                "trust": self._prior.shrink(hits, n),
            })
        return pd.DataFrame(rows)

    @property
    def is_fitted(self) -> bool:
        return self._prior is not None

    # -- persistence (plain JSON; no boosters involved) --------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "counts": {f"{k[0]}||{k[1]}": v for k, v in self._counts.items()},
            "prior": None if self._prior is None else {
                "alpha": self._prior.alpha, "beta": self._prior.beta,
                "global_rate": self._prior.global_rate,
                "prior_strength": self._prior.prior_strength,
                "n_buckets_used": self._prior.n_buckets_used,
            },
        }
        path.write_text(json.dumps(blob))

    @classmethod
    def load(cls, path: Path) -> "DetectorTrustRouter":
        blob = json.loads(Path(path).read_text())
        obj = cls()
        obj._counts = {
            tuple(k.split("||", 1)): tuple(v)
            for k, v in blob["counts"].items()
        }
        p = blob.get("prior")
        if p is not None:
            obj._prior = BetaBinomialPrior(**p)
        return obj
