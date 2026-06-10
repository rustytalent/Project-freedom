"""M.7 — Reaction archetype clustering + classification.

Today every post-touch reaction is summarised as one scalar (respect
/ break / reclaim). In reality the shapes differ: V-bounce, slow
grind through, fake-break-and-reclaim, drift-along-the-level. This
module:

  1. CLUSTERS normalised post-touch price paths into K archetypes
     (k-means on the shape vectors), and
  2. CLASSIFIES new touches into an archetype from the entry
     context, before the reaction happens.

The archetype id becomes a new label dimension for downstream
models and a customer-facing vocabulary for the brief ("this looks
like an archetype-3 setup: slow grind, ~4-bar mean exit").

Path normalisation: each reaction is the (close - touch_price) /
ATR series over the first ``path_bars`` bars after touch, side-
flipped so "respect" is always positive — long and short pools
share archetypes.

v1 scope: clustering is sklearn KMeans (deterministic via seed);
the classifier is the shared LightGBM wrapper. Naming the archetypes
("V-bounce" etc.) is a human step — the operator reviews
``archetype_profiles()`` and assigns names in config; the model
only emits stable integer ids.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..execution._base import LGBMConfig, LGBMRegressorWrapper

CONTEXT_FEATURES: List[str] = [
    "distance_atr", "vol_frac", "side_long",
    "session_open", "session_close", "tf_count", "score",
]

DEFAULT_N_ARCHETYPES = 5
DEFAULT_PATH_BARS = 6


def normalise_paths(paths: np.ndarray, sides: np.ndarray) -> np.ndarray:
    """``paths``: (n, path_bars) of (close - touch)/ATR. ``sides``:
    +1 for pools below price (long-respect = bounce up), -1 above.
    Flip so positive always means 'respected'."""
    return paths * sides.reshape(-1, 1)


class ReactionArchetypeModel:
    """Cluster + classify post-touch reaction shapes."""

    def __init__(
        self,
        n_archetypes: int = DEFAULT_N_ARCHETYPES,
        path_bars: int = DEFAULT_PATH_BARS,
        seed: int = 42,
        config: Optional[LGBMConfig] = None,
    ) -> None:
        self.n_archetypes = int(n_archetypes)
        self.path_bars = int(path_bars)
        self.seed = int(seed)
        self._kmeans = None
        self._centroids: Optional[np.ndarray] = None
        # One regressor per archetype: P(archetype k | entry context).
        # (Multiclass via one-vs-rest keeps the shared wrapper usable.)
        self._heads: List[LGBMRegressorWrapper] = [
            LGBMRegressorWrapper(
                feature_names=CONTEXT_FEATURES,
                fallback=1.0 / max(1, n_archetypes),
                config=config,
            )
            for _ in range(self.n_archetypes)
        ]

    # -- clustering ---------------------------------------------------

    def fit_clusters(self, paths: np.ndarray,
                     sides: np.ndarray) -> np.ndarray:
        """Cluster the historical reaction shapes. Returns the per-row
        archetype assignment (also stored for fit_classifier)."""
        from sklearn.cluster import KMeans

        shapes = normalise_paths(np.asarray(paths, dtype=float),
                                 np.asarray(sides, dtype=float))
        mask = np.isfinite(shapes).all(axis=1)
        if mask.sum() < self.n_archetypes * 5:
            # Too little data to support K clusters; record nothing.
            return np.full(len(shapes), -1)
        km = KMeans(n_clusters=self.n_archetypes, random_state=self.seed,
                    n_init=10)
        labels_valid = km.fit_predict(shapes[mask])
        self._kmeans = km
        self._centroids = km.cluster_centers_
        labels = np.full(len(shapes), -1)
        labels[mask] = labels_valid
        return labels

    def assign(self, path: np.ndarray, side: float) -> int:
        """Archetype id for one realised reaction path; -1 if the
        cluster model isn't fit or the path is malformed."""
        if self._kmeans is None:
            return -1
        shape = normalise_paths(
            np.asarray(path, dtype=float).reshape(1, -1),
            np.asarray([side], dtype=float),
        )
        if not np.isfinite(shape).all():
            return -1
        return int(self._kmeans.predict(shape)[0])

    def archetype_profiles(self) -> pd.DataFrame:
        """Centroid table for the operator's naming pass. Each row is
        one archetype's mean normalised path."""
        if self._centroids is None:
            return pd.DataFrame()
        cols = [f"bar_{i+1}_atr" for i in range(self._centroids.shape[1])]
        df = pd.DataFrame(self._centroids, columns=cols)
        df.insert(0, "archetype_id", range(len(df)))
        df["terminal_atr"] = self._centroids[:, -1]
        df["max_excursion_atr"] = self._centroids.max(axis=1)
        df["min_excursion_atr"] = self._centroids.min(axis=1)
        return df

    # -- classification ------------------------------------------------

    def fit_classifier(self, context: pd.DataFrame,
                       archetype_labels: np.ndarray) -> "ReactionArchetypeModel":
        """One-vs-rest heads on the entry context. Rows labelled -1
        (unassigned) are dropped."""
        labels = np.asarray(archetype_labels)
        mask = labels >= 0
        if mask.sum() == 0:
            return self
        ctx = context.loc[mask].reset_index(drop=True)
        labels = labels[mask]
        for k, head in enumerate(self._heads):
            y = (labels == k).astype(float)
            head.fit(ctx[CONTEXT_FEATURES], y)
        return self

    def predict_archetype_probs(self, state: Dict[str, Any]) -> np.ndarray:
        """Normalised archetype distribution from entry context.
        Unfit heads emit their uniform fallback, so the unfit model
        returns the uniform distribution."""
        df = pd.DataFrame([state])
        raw = np.array([
            max(0.0, float(h.predict(df[CONTEXT_FEATURES]
                                     if set(CONTEXT_FEATURES) <= set(df.columns)
                                     else df)[0]))
            for h in self._heads
        ])
        if not np.isfinite(raw).all() or raw.sum() <= 0:
            return np.full(self.n_archetypes, 1.0 / self.n_archetypes)
        return raw / raw.sum()

    def predict_archetype(self, state: Dict[str, Any]) -> int:
        return int(np.argmax(self.predict_archetype_probs(state)))

    @property
    def is_fitted(self) -> bool:
        return self._kmeans is not None
