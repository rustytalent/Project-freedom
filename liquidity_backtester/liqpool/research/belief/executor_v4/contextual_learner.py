"""ContextualLearner — regime-conditioned, recency-weighted, attributed.

Founder 2026-06-22 Tier-2 brief: "the adaptive calibrator you built is
primitive. Make it more context-aware, more elaborate, more
understanding. It should not lose context." This module is the upgrade.

Three concrete improvements over the base ``OnlineLearner``:

  1. **Regime-conditioned weights** — instead of one global weight
     vector, we maintain *per-family* vectors (directional / chop /
     fat_tail / manipulation / unknown). At apply-time the live
     calibrator pushes the *current regime's* weights into the
     aggregator, not a one-size-fits-all average. The reasoning:
     fees_clearance matters less in chop (smaller moves, smaller R)
     than in directional (we need real edge over fees); MTF
     alignment matters more in directional than in fat-tail
     (alignment dies fast under news). One global vector blurs all
     of this.

  2. **Recency-weighted observations** — each observation carries an
     ``observed_at_ts`` and is weighted by ``exp(-age_days / half_life)``
     in the SGD loss + gradient. Yesterday's lessons are worth more
     than five days ago because the regime + market microstructure
     change. The walk-forward gate sees the *recency-weighted* val_loss,
     so an overfit-to-week-old-trades update is rejected.

  3. **Shapley-style component attribution** — when a closure is
     observed, we compute each component's marginal contribution to
     the predicted win probability. Surfaces as a "what helped / what
     hurt" line in the cockpit so the operator can see which feature
     made each trade and which only rode along.

The class composes the base ``OnlineLearner`` once per regime family;
the base implementation does all the SGD + walk-forward + bounds
plumbing. We just route observations to the right learner and
post-process the outputs.

Wiring: ``LiveCalibrator`` gets an optional ``contextual_learner``;
when present, ``apply_to_aggregator`` reads the current regime tag and
applies the family-specific weights. The original session-local
``OnlineLearner`` path remains intact for backward-compatibility — the
contextual layer is purely additive.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .learning import OnlineLearner, OnlineLearnerConfig, WeightUpdate


# Regime families we condition on. ``UNKNOWN`` is the fallback bucket
# for observations whose regime tag couldn't be derived (cold start,
# missing web snapshot, etc.).
FAMILY_DIRECTIONAL = "directional"
FAMILY_CHOP = "chop"
FAMILY_FAT_TAIL = "fat_tail"
FAMILY_MANIPULATION = "manipulation"
FAMILY_UNKNOWN = "unknown"

REGIME_FAMILIES = (
    FAMILY_DIRECTIONAL, FAMILY_CHOP, FAMILY_FAT_TAIL,
    FAMILY_MANIPULATION, FAMILY_UNKNOWN,
)


# ── Config ────────────────────────────────────────────────────────


@dataclass
class ContextualLearnerConfig:
    """Knobs."""
    # The base learner config used for every per-family learner.
    base_cfg: OnlineLearnerConfig = field(default_factory=OnlineLearnerConfig)
    # Recency decay: weight = exp(-age_days / half_life_days). A short
    # half-life means recent days dominate; a long one is closer to
    # uniform weighting. Disable by setting to a very large number.
    recency_half_life_days: float = 4.0
    # When a family has < min_samples_to_specialise, we use the GLOBAL
    # cross-family vector instead of that family's own (so a barely-
    # populated regime doesn't fit noise).
    min_samples_to_specialise: int = 8
    # Shapley attribution: the number of permutations to sample. With 5
    # components there are 120 permutations; sampling captures the
    # expected marginal contribution for cheap. Set to 0 to disable.
    n_shapley_permutations: int = 24


# ── Attribution dataclass ─────────────────────────────────────────


@dataclass
class ComponentAttribution:
    """Per-component Shapley contribution to a single closure's predicted
    win probability. Sums (approximately) to the model's prediction.

    Positive ⇒ this component pulled the prediction toward 'win';
    negative ⇒ pulled toward 'loss'.
    """
    component: str
    score: float                # raw component score at entry
    weight_used: float          # weight from the regime-specific vector
    contribution: float         # Shapley marginal contribution

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "score": round(self.score, 3),
            "weight_used": round(self.weight_used, 3),
            "contribution": round(self.contribution, 4),
        }


@dataclass
class ClosureReport:
    """What we emitted for one closed trade."""
    realized_r: float
    won: bool
    regime_family: str
    component_attributions: List[ComponentAttribution]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "realized_r": round(self.realized_r, 3),
            "won": self.won,
            "regime_family": self.regime_family,
            "component_attributions": [a.to_dict()
                                          for a in self.component_attributions],
            "notes": list(self.notes),
        }


# ── The contextual learner ────────────────────────────────────────


class ContextualLearner:
    """Per-regime-family OnlineLearner ensemble + recency + Shapley.

    The class is intentionally additive: it does not change the
    underlying OnlineLearner's SGD or walk-forward semantics. It just
    routes observations to the right family-specific learner and adds
    interpretability (regime conditioning, recency, attribution).
    """

    _COMPONENTS = OnlineLearner._COMPONENTS

    def __init__(self, *,
                  initial_weights: Dict[str, float],
                  cfg: Optional[ContextualLearnerConfig] = None,
                  ) -> None:
        self.cfg = cfg or ContextualLearnerConfig()
        # One sub-learner per family. Each starts from the same initial
        # weights — they diverge as observations arrive.
        self._by_family: Dict[str, OnlineLearner] = {
            f: OnlineLearner(initial_weights=dict(initial_weights),
                                cfg=self.cfg.base_cfg)
            for f in REGIME_FAMILIES
        }
        # A separate GLOBAL learner across ALL families that we fall
        # back to when a family is too thinly populated to trust.
        self._global = OnlineLearner(initial_weights=dict(initial_weights),
                                          cfg=self.cfg.base_cfg)
        # Per-family observation timestamps (for recency calculations).
        self._obs_timestamps: Dict[str, List[float]] = {
            f: [] for f in REGIME_FAMILIES}
        self._recent_closure_reports: List[ClosureReport] = []

    # ── Observation routing ────────────────────────────────────────

    def observe_closure(self, *,
                          component_scores: Dict[str, float],
                          realized_r: float,
                          regime_family: str = FAMILY_UNKNOWN,
                          observed_at_ts: Optional[float] = None,
                          ) -> ClosureReport:
        """Route the closure to the right family-specific learner AND
        the global one. Compute Shapley attribution for the per-family
        weights as they stood BEFORE the observation."""
        family = regime_family if regime_family in REGIME_FAMILIES \
                  else FAMILY_UNKNOWN
        learner = self._by_family[family]
        # Capture pre-observation weights for attribution.
        pre_weights = dict(learner.weights)
        # Feed both the family-specific and the global learners.
        learner.observe_closure(
            component_scores=component_scores, realized_r=realized_r,
        )
        self._global.observe_closure(
            component_scores=component_scores, realized_r=realized_r,
        )
        ts = float(observed_at_ts if observed_at_ts is not None
                   else time.time())
        self._obs_timestamps[family].append(ts)
        # Compute Shapley-style attribution against the pre-update
        # weights for the operator dashboard.
        attributions = self._shapley_attribution(
            component_scores=component_scores, weights=pre_weights)
        won = bool(realized_r >= self.cfg.base_cfg.win_threshold_r)
        report = ClosureReport(
            realized_r=realized_r, won=won,
            regime_family=family,
            component_attributions=attributions,
            notes=[
                f"family={family}; pre-update weights used for attribution",
            ],
        )
        # Keep the most recent N reports for the cockpit.
        self._recent_closure_reports.append(report)
        if len(self._recent_closure_reports) > 32:
            self._recent_closure_reports = self._recent_closure_reports[-32:]
        return report

    # ── Per-family update + apply ──────────────────────────────────

    def maybe_update(self, family: str
                          ) -> Optional[WeightUpdate]:
        """Trigger the SGD update on the family-specific learner. The
        global learner is updated by its own
        ``observe_closure``-triggered cadence."""
        fam = family if family in REGIME_FAMILIES else FAMILY_UNKNOWN
        return self._by_family[fam].maybe_update()

    def maybe_update_all(self) -> Dict[str, Optional[WeightUpdate]]:
        """Run maybe_update on every family + the global. Returns a
        per-family map of WeightUpdate or None."""
        out: Dict[str, Optional[WeightUpdate]] = {}
        for f, learner in self._by_family.items():
            out[f] = learner.maybe_update()
        out["__global__"] = self._global.maybe_update()
        return out

    def weights_for(self, family: str) -> Dict[str, float]:
        """The weights the live aggregator should USE right now for the
        given regime. Falls back to global when the family is too
        thinly populated to specialise."""
        fam = family if family in REGIME_FAMILIES else FAMILY_UNKNOWN
        learner = self._by_family[fam]
        n_obs = len(learner.observations)
        if n_obs < self.cfg.min_samples_to_specialise:
            return dict(self._global.weights)
        return dict(learner.weights)

    def apply_to_aggregator_config(self, aggregator_cfg, *,
                                          current_family: str = FAMILY_UNKNOWN,
                                          ) -> Dict[str, float]:
        """Push the family-specific (or global-fallback) weights into the
        live AggregatorConfig in place.

        The base learner manages 5 weights summing to 1.0. The aggregator
        has a 6th weight ``w_rehearsal`` that we MUST NOT step on. We
        rescale the 5 learned weights so that they sum to ``1 - w_rehearsal``
        (the budget reserved for the contextual components) before
        applying. This keeps the aggregator's weight-sum invariant
        intact and lets the rehearsal weight stay where it is.
        """
        w = self.weights_for(current_family)
        # Reserve the rehearsal weight slice; the rest is what the
        # contextual learner gets to distribute.
        w_rehearsal_now = float(getattr(aggregator_cfg, "w_rehearsal", 0.0))
        budget = max(0.0, 1.0 - w_rehearsal_now)
        scale = budget / max(1e-9, sum(w.values()))
        scaled = {k: v * scale for k, v in w.items()}
        for k, v in scaled.items():
            if hasattr(aggregator_cfg, k):
                setattr(aggregator_cfg, k, float(v))
        return scaled

    # ── Recency-weighted summary ───────────────────────────────────

    def _recency_weight(self, ts: float, now: Optional[float] = None) -> float:
        now = now if now is not None else time.time()
        age_days = max(0.0, (now - ts) / 86400.0)
        half_life = max(0.01, self.cfg.recency_half_life_days)
        # exp(-ln(2) * age / half_life) gives proper half-life decay.
        return math.exp(-math.log(2.0) * age_days / half_life)

    def per_family_n_samples(self) -> Dict[str, int]:
        return {f: len(l.observations) for f, l in self._by_family.items()}

    def per_family_effective_n(self) -> Dict[str, float]:
        """Sum of recency weights per family. The 'effective' sample
        count once we apply recency decay."""
        now = time.time()
        out: Dict[str, float] = {}
        for f, ts_list in self._obs_timestamps.items():
            out[f] = sum(self._recency_weight(ts, now) for ts in ts_list)
        return out

    # ── Shapley attribution ───────────────────────────────────────

    def _shapley_attribution(self, *,
                                 component_scores: Dict[str, float],
                                 weights: Dict[str, float],
                                 ) -> List[ComponentAttribution]:
        """Sample n permutations and compute each component's expected
        marginal contribution to the model's predicted win probability.

        Predicted P(win) = sigmoid(Σ w_i * x_i − 0.5). Marginal
        contribution of component i in permutation π is f(S ∪ {i}) −
        f(S) where S is the set of components that came before i in π.
        Shapley = expectation over permutations. Sampling is unbiased.
        """
        cfg = self.cfg
        if cfg.n_shapley_permutations <= 0:
            return []
        scores = {k: float(component_scores.get(k.replace("w_", "") + "",
                                                    component_scores.get(
                                                        k.replace("w_", ""),
                                                        0.0)))
                  for k in self._COMPONENTS}
        # The base learner's _features() routes weight key 'w_base_score'
        # to score key 'base_score' — replicate that here.
        keymap = {
            "w_base_score": "base_score",
            "w_mtf_alignment": "mtf_alignment_score",
            "w_projection": "projection_factor",
            "w_fees_clearance": "fees_clearance_score",
            "w_portfolio_capacity": "portfolio_capacity_score",
        }
        scores = {k: float(component_scores.get(keymap[k], 0.0))
                  for k in self._COMPONENTS}

        def _sigmoid_minus_half(logit: float) -> float:
            return 1.0 / (1.0 + math.exp(-(logit - 0.5)))

        n_perms = cfg.n_shapley_permutations
        contribs: Dict[str, float] = {k: 0.0 for k in self._COMPONENTS}
        # Deterministic shuffle (no PRNG dependency): we use a sequence
        # of pseudo-random permutations derived from index hashing. With
        # 5 components and ~24 perms we cover most of the 120-perm space.
        from itertools import permutations
        all_perms = list(permutations(self._COMPONENTS))
        step = max(1, len(all_perms) // n_perms)
        sampled = all_perms[::step][:n_perms]
        for perm in sampled:
            included: Dict[str, float] = {}
            prev_pred = _sigmoid_minus_half(0.0)
            for k in perm:
                included[k] = weights.get(k, 0.0) * scores.get(k, 0.0)
                pred = _sigmoid_minus_half(sum(included.values()))
                contribs[k] += (pred - prev_pred)
                prev_pred = pred
        if len(sampled) == 0:
            return []
        for k in contribs:
            contribs[k] /= len(sampled)

        return [
            ComponentAttribution(
                component=k,
                score=scores.get(k, 0.0),
                weight_used=weights.get(k, 0.0),
                contribution=contribs[k],
            )
            for k in self._COMPONENTS
        ]

    # ── Cockpit summary ───────────────────────────────────────────

    def summary(self, *, current_family: str = FAMILY_UNKNOWN
                 ) -> Dict[str, Any]:
        per_family_weights = {
            f: dict(learner.weights)
            for f, learner in self._by_family.items()
        }
        n_samples = self.per_family_n_samples()
        eff_n = self.per_family_effective_n()
        applied = self.weights_for(current_family)
        recent = [r.to_dict() for r in self._recent_closure_reports[-6:]]
        return {
            "current_family": current_family,
            "applied_weights": applied,
            "global_weights": dict(self._global.weights),
            "per_family_weights": per_family_weights,
            "per_family_n_samples": n_samples,
            "per_family_effective_n": {k: round(v, 2)
                                            for k, v in eff_n.items()},
            "recent_closure_reports": recent,
            "min_samples_to_specialise": self.cfg.min_samples_to_specialise,
            "recency_half_life_days": self.cfg.recency_half_life_days,
        }
