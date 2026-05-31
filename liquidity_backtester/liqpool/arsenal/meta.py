"""Meta alphas — combine multiple base alphas into one.

The arsenal evaluator's pairwise_combinations view already shows
"when alpha A AND alpha B both fire on the same bar, what's the
combined outcome?" — but only as a diagnostic. Meta alphas turn that
diagnostic into a FIRST-CLASS alpha that the framework treats like any
other: it has a name, it emits AlphaSignals, it's null-tested.

Three combination operators:

  MetaANDAlpha(components):
    Emit a signal only when EVERY component would emit at the same
    (symbol, decision_idx). This is the strictest filter — agreement
    of multiple independent signals. Geometry (stop, target, horizon)
    is the per-component MINIMUM by default (tightest stop, smallest
    target, shortest horizon) for conservative trading; configurable.

  MetaORAlpha(components):
    Emit a signal when ANY component would emit. Geometry follows the
    FIRST component that fired (or a configurable selector). Useful
    for diversification: "fire on anyone's good idea."

  MetaWeightedAlpha(components, weights):
    Emit only when the weighted confidence sum across firing components
    exceeds a threshold. The most flexible combinator; defaults to
    equal-weight average. Useful for soft voting.

This is the self-growth mechanism the user asked about: once N base
alphas exist, the arsenal can mechanically construct C(N, 2) AND-pairs
and C(N, 3) AND-triples and so on — each evaluated independently. This
turns "alpha discovery" into "combination search," which is both more
tractable and more statistically defensible than searching over raw
feature space.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .base import Alpha, AlphaSignal


def _key(sig: AlphaSignal) -> tuple:
    """Key signals by (symbol, decision_idx). Two alphas "agree" at the
    same bar if they emit signals with the same key."""
    return (sig.symbol, sig.decision_idx)


def _merge_geometry_min(components: Sequence[AlphaSignal]) -> Dict[str, float]:
    """Conservative geometry merge: tightest stop, smallest target, shortest
    horizon. Use the FIRST component's side (callers ensure side agreement
    is checked separately if desired)."""
    stops = [c.stop_atr for c in components]
    targets = [c.target_atr for c in components]
    horizons = [c.horizon_bars for c in components]
    return {
        "stop_atr": float(min(stops)),
        "target_atr": float(min(targets)),
        "horizon_bars": int(min(horizons)),
    }


def _merge_geometry_max_horizon_min_risk(components: Sequence[AlphaSignal]) -> Dict[str, float]:
    """Alternative merge: tightest stop (lowest risk), largest target
    (highest reward), shortest horizon (intraday compatible)."""
    return {
        "stop_atr": float(min(c.stop_atr for c in components)),
        "target_atr": float(max(c.target_atr for c in components)),
        "horizon_bars": int(min(c.horizon_bars for c in components)),
    }


# ---------------------------------------------------------------------------
# AND combinator
# ---------------------------------------------------------------------------

class MetaANDAlpha(Alpha):
    """Emit a signal only when EVERY component alpha emits at the same
    (symbol, decision_idx) with the SAME side.

    The combined geometry defaults to the conservative merge
    (min stop / min target / min horizon). Pass ``geometry_merge=
    _merge_geometry_max_horizon_min_risk`` for a different policy.
    """

    def __init__(self, components: Sequence[Alpha],
                 name: Optional[str] = None,
                 require_same_side: bool = True,
                 geometry_merge=_merge_geometry_min) -> None:
        if len(components) < 2:
            raise ValueError("MetaANDAlpha needs at least 2 component alphas")
        self.components = list(components)
        self._name = name or f"AND_{'_X_'.join(a.name for a in components)}"
        self.require_same_side = bool(require_same_side)
        self._geometry_merge = geometry_merge

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return (f"AND combination of: {[a.name for a in self.components]}. "
                f"Emits only when all components agree at the same "
                f"(symbol, decision_idx)"
                f"{' with same side' if self.require_same_side else ''}.")

    def candidates(self, *, symbol: str, df_base: pd.DataFrame,
                   atr_series: pd.Series,
                   extra: Optional[Dict[str, Any]] = None,
                   ) -> List[AlphaSignal]:
        per_component: List[Dict[tuple, AlphaSignal]] = []
        for c in self.components:
            sigs = c.candidates(symbol=symbol, df_base=df_base,
                                atr_series=atr_series, extra=extra)
            per_component.append({_key(s): s for s in sigs})
        if not per_component or not per_component[0]:
            return []

        common: set = set(per_component[0].keys())
        for d in per_component[1:]:
            common &= set(d.keys())
        if not common:
            return []

        out: List[AlphaSignal] = []
        for key in sorted(common):                      # deterministic order
            comp_sigs = [d[key] for d in per_component]
            if self.require_same_side:
                sides = {s.side for s in comp_sigs}
                if len(sides) > 1:
                    continue
            geom = self._geometry_merge(comp_sigs)
            avg_conf = sum(s.confidence for s in comp_sigs) / len(comp_sigs)
            # Use the first component's entry_reference as canonical.
            anchor = comp_sigs[0]
            out.append(AlphaSignal(
                alpha_name=self._name, symbol=anchor.symbol,
                decision_at=anchor.decision_at,
                decision_idx=anchor.decision_idx,
                side=anchor.side,
                entry_reference=anchor.entry_reference,
                stop_atr=float(geom["stop_atr"]),
                target_atr=float(geom["target_atr"]),
                horizon_bars=int(geom["horizon_bars"]),
                confidence=float(min(1.0, max(0.0, avg_conf))),
                state={
                    "components": [s.alpha_name for s in comp_sigs],
                    "component_confidences": [s.confidence for s in comp_sigs],
                },
            ))
        return out


# ---------------------------------------------------------------------------
# OR combinator
# ---------------------------------------------------------------------------

class MetaORAlpha(Alpha):
    """Emit a signal when ANY component would emit. When multiple components
    fire at the same (symbol, decision_idx), the highest-confidence one is
    chosen as the canonical signal."""

    def __init__(self, components: Sequence[Alpha],
                 name: Optional[str] = None) -> None:
        if len(components) < 2:
            raise ValueError("MetaORAlpha needs at least 2 component alphas")
        self.components = list(components)
        self._name = name or f"OR_{'_or_'.join(a.name for a in components)}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return (f"OR combination of: {[a.name for a in self.components]}. "
                f"Emits when any component does; per-bar dedup to the "
                f"highest-confidence component.")

    def candidates(self, *, symbol: str, df_base: pd.DataFrame,
                   atr_series: pd.Series,
                   extra: Optional[Dict[str, Any]] = None,
                   ) -> List[AlphaSignal]:
        best_by_key: Dict[tuple, AlphaSignal] = {}
        for c in self.components:
            sigs = c.candidates(symbol=symbol, df_base=df_base,
                                atr_series=atr_series, extra=extra)
            for s in sigs:
                k = _key(s)
                existing = best_by_key.get(k)
                if existing is None or s.confidence > existing.confidence:
                    best_by_key[k] = s
        # Re-emit with the meta-alpha's name so the evaluator attributes
        # correctly. Preserve the component's geometry verbatim.
        out: List[AlphaSignal] = []
        for k in sorted(best_by_key.keys()):
            src = best_by_key[k]
            out.append(AlphaSignal(
                alpha_name=self._name, symbol=src.symbol,
                decision_at=src.decision_at, decision_idx=src.decision_idx,
                side=src.side, entry_reference=src.entry_reference,
                stop_atr=src.stop_atr, target_atr=src.target_atr,
                horizon_bars=src.horizon_bars,
                confidence=src.confidence,
                state={
                    "source_alpha": src.alpha_name,
                    "source_confidence": src.confidence,
                },
            ))
        return out


# ---------------------------------------------------------------------------
# Weighted combinator (soft voting)
# ---------------------------------------------------------------------------

class MetaWeightedAlpha(Alpha):
    """Emit only when the weighted sum of firing components' confidences
    exceeds ``threshold``. Components that do not fire at a given bar
    contribute 0.

    Default ``weights`` is equal across components; ``threshold`` defaults
    to ``0.5 * sum(weights)`` which means "majority confidence."
    """

    def __init__(self, components: Sequence[Alpha],
                 weights: Optional[Sequence[float]] = None,
                 threshold: Optional[float] = None,
                 require_same_side: bool = True,
                 name: Optional[str] = None,
                 geometry_merge=_merge_geometry_min) -> None:
        if len(components) < 2:
            raise ValueError("MetaWeightedAlpha needs at least 2 components")
        if weights is not None and len(weights) != len(components):
            raise ValueError("weights length must match components")
        self.components = list(components)
        self.weights = [float(w) for w in (weights or [1.0] * len(components))]
        self.threshold = (float(threshold) if threshold is not None
                          else 0.5 * sum(self.weights))
        self.require_same_side = bool(require_same_side)
        self._name = name or f"WEIGHTED_{'_'.join(a.name for a in components)}"
        self._geometry_merge = geometry_merge

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return (f"Weighted combination of {[a.name for a in self.components]} "
                f"with weights {self.weights}; threshold {self.threshold:.2f}.")

    def candidates(self, *, symbol: str, df_base: pd.DataFrame,
                   atr_series: pd.Series,
                   extra: Optional[Dict[str, Any]] = None,
                   ) -> List[AlphaSignal]:
        per_component: List[Dict[tuple, AlphaSignal]] = []
        for c in self.components:
            sigs = c.candidates(symbol=symbol, df_base=df_base,
                                atr_series=atr_series, extra=extra)
            per_component.append({_key(s): s for s in sigs})

        # Union of all decision keys.
        all_keys: set = set()
        for d in per_component:
            all_keys |= set(d.keys())

        out: List[AlphaSignal] = []
        for key in sorted(all_keys):
            firing = []
            firing_weights = []
            for c_idx, d in enumerate(per_component):
                if key in d:
                    firing.append(d[key])
                    firing_weights.append(self.weights[c_idx])
            if not firing:
                continue
            if self.require_same_side and len({s.side for s in firing}) > 1:
                continue
            weighted_conf = sum(s.confidence * w
                                 for s, w in zip(firing, firing_weights))
            if weighted_conf < self.threshold:
                continue
            geom = self._geometry_merge(firing)
            anchor = firing[0]
            out.append(AlphaSignal(
                alpha_name=self._name, symbol=anchor.symbol,
                decision_at=anchor.decision_at,
                decision_idx=anchor.decision_idx,
                side=anchor.side,
                entry_reference=anchor.entry_reference,
                stop_atr=float(geom["stop_atr"]),
                target_atr=float(geom["target_atr"]),
                horizon_bars=int(geom["horizon_bars"]),
                confidence=float(min(1.0, weighted_conf / max(sum(self.weights), 1e-9))),
                state={
                    "weighted_confidence": float(weighted_conf),
                    "firing_components": [s.alpha_name for s in firing],
                    "n_firing": len(firing),
                },
            ))
        return out
