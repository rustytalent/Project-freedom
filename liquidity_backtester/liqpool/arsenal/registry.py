"""Alpha registry — a small named lookup for declared alphas.

We deliberately keep this lightweight: it's a dict with a few validation
guards. Anything more (capabilities, versioning, etc.) can be added later
without breaking callers.

Why a registry rather than just passing a list of Alpha instances around:
the runner CLI accepts ``--alphas pool_reach,mean_reversion`` and the
registry resolves names to instances. It also catches duplicate
registration early (typos in the alphas/ folder) instead of producing
two trade frames with the same name later.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

from .base import Alpha


class AlphaRegistry:
    """Process-wide singleton-style registry.

    Usage:
        registry = AlphaRegistry()
        registry.register(MyAlpha())
        for alpha in registry.all():
            ...
        only = registry.subset(["pool_reach", "momentum"])
    """

    def __init__(self) -> None:
        self._by_name: Dict[str, Alpha] = {}

    def register(self, alpha: Alpha) -> None:
        """Register one alpha. Duplicate names raise to catch typos."""
        if not isinstance(alpha, Alpha):
            raise TypeError(f"expected Alpha subclass, got {type(alpha).__name__}")
        name = alpha.name
        if not isinstance(name, str) or not name:
            raise ValueError(f"alpha.name must be a non-empty string, got {name!r}")
        if name in self._by_name:
            raise ValueError(f"alpha {name!r} already registered")
        self._by_name[name] = alpha

    def get(self, name: str) -> Alpha:
        if name not in self._by_name:
            raise KeyError(f"alpha {name!r} not registered "
                            f"(known: {sorted(self._by_name)})")
        return self._by_name[name]

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __len__(self) -> int:
        return len(self._by_name)

    def names(self) -> List[str]:
        return sorted(self._by_name.keys())

    def all(self) -> List[Alpha]:
        """All registered alphas, in name-sorted order for deterministic runs."""
        return [self._by_name[n] for n in sorted(self._by_name)]

    def subset(self, names: Iterable[str]) -> List[Alpha]:
        """Return alphas matching the given names, in the order supplied."""
        names = list(names)
        unknown = [n for n in names if n not in self._by_name]
        if unknown:
            raise KeyError(f"unknown alpha names: {unknown} "
                            f"(known: {sorted(self._by_name)})")
        return [self._by_name[n] for n in names]


def default_registry() -> AlphaRegistry:
    """Production-ready alphas suitable for the daily Arsenal sweep.

    Includes baselines with sufficient trade volume to be statistically
    interpretable (>=200 OOS trades per asset in typical bundles) plus
    the proximity-journey alpha + its single-input baseline for
    attribution.

    Sparse / experimental alphas (typically <200 trades) live in
    :func:`research_registry` so they don't contaminate routine sweeps.

    Importing alphas at module top would create a circular dependency
    (registry depends on alphas/ which depends on Alpha). We construct
    lazily in the function body.
    """
    from .alphas.pool_reach import LiquidityPoolReachAlpha
    from .alphas.mean_reversion import MeanReversionAlpha
    from .alphas.momentum import MomentumAlpha
    from .alphas.model_filtered import ProximityFilteredPoolAlpha

    reg = AlphaRegistry()
    reg.register(LiquidityPoolReachAlpha())
    reg.register(MeanReversionAlpha())
    reg.register(MomentumAlpha())
    reg.register(ProximityFilteredPoolAlpha(
        name="proximity_journey_baseline",
        min_p_touch=0.60,
        max_dist_atr=5.0,
        use_soft_score=False,
        use_direction_score=False,
        use_sector_rotation=False,
        use_vol_regime=False,
        use_multi_horizon=False,
        require_opening_breakout=False,
    ))
    reg.register(ProximityFilteredPoolAlpha(
        name="proximity_journey",
        min_p_touch=0.50,
        min_score=0.58,
        max_dist_atr=10.0,
        use_soft_score=True,
        use_multi_horizon=True,
    ))
    return reg


def research_registry() -> AlphaRegistry:
    """``default_registry()`` plus the sparse / experimental alphas.

    The additional members typically produce fewer than 200 OOS trades
    per Arsenal run, which means their per-alpha mean_R and p-value
    estimates are statistically uninformative. They are kept here for
    explicit research sweeps where the sparse signal is acceptable —
    e.g. when running on a much larger basket, or when sweeping the
    alphas' own configuration knobs.

    DO NOT use this registry as the default for routine retrains; the
    extra alphas inflate the multiple-comparison problem without
    contributing reliable signal at the current basket size.
    """
    from .alphas.model_filtered import (
        DistanceBandJourneyAlpha,
        OpeningRangeToPoolAlpha,
        ProximityDirectionSoftAlpha,
        SectorRotationJourneyAlpha,
    )

    reg = default_registry()
    reg.register(DistanceBandJourneyAlpha())
    reg.register(ProximityDirectionSoftAlpha())
    reg.register(OpeningRangeToPoolAlpha())
    reg.register(SectorRotationJourneyAlpha())
    return reg
