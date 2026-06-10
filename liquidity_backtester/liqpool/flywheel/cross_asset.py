"""M.9 — Cross-asset signal transferability.

The unified multi-asset models implicitly assume signal transfers
uniformly across the ~25-stock universe. In practice transfer is
regime- and pair-dependent: a banking-sector structural read often
transfers to its sector peers but not to IT. This module builds an
EMPIRICAL transfer matrix from per-asset OOS outcomes and shrinks it
with the Stream N empirical-Bayes machinery:

    transfer(a, b) = shrunk P(outcome agrees | A's signal fired and
                              B had a comparable setup same session)

Data shape (one row per same-session co-occurrence):
    asset_a, asset_b — the two symbols
    agreed           — 1 if the two pools resolved the same way
                       (both respected or both broke), else 0

The matrix feeds two consumers:
  * The unified trainer can downweight cross-asset sample sharing
    for low-transfer pairs.
  * The brief's sector-regime block can qualify claims ("banking
    read transfers weakly to PSU banks this regime").

v1 keeps it symmetric: transfer(a,b) == transfer(b,a). Direction-
ality (A leads B) is a v2 question that needs lead-lag data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from ..calibration.empirical_bayes import (
    BetaBinomialPrior,
    fit_beta_binomial_prior,
)

FALLBACK_TRANSFER: float = 0.5


def _key(a: str, b: str) -> Tuple[str, str]:
    """Symmetric pair key."""
    return (a, b) if a <= b else (b, a)


class CrossAssetTransferMatrix:
    """Empirical-Bayes shrunk pairwise agreement rates."""

    def __init__(self) -> None:
        self._prior: Optional[BetaBinomialPrior] = None
        self._counts: Dict[Tuple[str, str], Tuple[float, float]] = {}

    def fit(self, co_occurrences: pd.DataFrame) -> "CrossAssetTransferMatrix":
        """``co_occurrences`` columns: asset_a, asset_b, agreed."""
        if co_occurrences.empty:
            return self
        need = {"asset_a", "asset_b", "agreed"}
        if not need.issubset(co_occurrences.columns):
            raise ValueError(f"co_occurrences needs columns {need}")
        counts: Dict[Tuple[str, str], Tuple[float, float]] = {}
        for _, r in co_occurrences.iterrows():
            k = _key(str(r["asset_a"]), str(r["asset_b"]))
            hits, n = counts.get(k, (0.0, 0.0))
            counts[k] = (hits + float(r["agreed"]), n + 1.0)
        self._counts = counts
        self._prior = fit_beta_binomial_prior(list(counts.values()))
        return self

    def transfer(self, asset_a: str, asset_b: str) -> float:
        """Shrunk agreement rate for the pair. Same-asset is 1.0 by
        definition; unknown pairs get the prior mean; unfit matrix
        returns the max-entropy fallback."""
        if asset_a == asset_b:
            return 1.0
        if self._prior is None:
            return FALLBACK_TRANSFER
        hits, n = self._counts.get(_key(asset_a, asset_b), (0.0, 0.0))
        return float(self._prior.shrink(hits, n))

    def matrix(self) -> pd.DataFrame:
        """Long-format table of all observed pairs for the audit."""
        if self._prior is None:
            return pd.DataFrame()
        rows = []
        for (a, b), (hits, n) in sorted(self._counts.items()):
            rows.append({
                "asset_a": a, "asset_b": b, "n": int(n),
                "raw_agreement": hits / n if n else float("nan"),
                "transfer": self._prior.shrink(hits, n),
            })
        return pd.DataFrame(rows)

    @property
    def is_fitted(self) -> bool:
        return self._prior is not None

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "counts": {f"{k[0]}||{k[1]}": v for k, v in self._counts.items()},
            "prior": None if self._prior is None else {
                "alpha": self._prior.alpha, "beta": self._prior.beta,
                "global_rate": self._prior.global_rate,
                "prior_strength": self._prior.prior_strength,
                "n_buckets_used": self._prior.n_buckets_used,
            },
        }))

    @classmethod
    def load(cls, path: Path) -> "CrossAssetTransferMatrix":
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
