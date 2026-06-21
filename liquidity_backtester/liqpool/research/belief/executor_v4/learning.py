"""Online learning — adaptive weights from realized outcomes.

State-of-the-art trading systems learn from their own ledger. This
module reads closed positions from the LedgerStore, scores how much
each aggregator component (base/MTF/projection/fees/portfolio) actually
contributed to win-vs-loss, and proposes weight updates that the
manager applies on the fly.

Method: regularized stochastic gradient over a logistic model that
predicts P(trade wins | components). The loss is binary cross-entropy
weighted by realized R magnitude (so big wins/losses dominate updates).

This is intentionally CONSERVATIVE:
  * Weights move slowly (default learning rate 0.005)
  * Hard bounds [0.05, 0.50] per component to avoid degenerate weights
  * Sum-to-1 constraint preserved by re-normalization
  * Updates only fire when >=N_MIN_SAMPLES closures exist
  * The current AggregatorConfig values are the prior; learning gently
    nudges them
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OnlineLearnerConfig:
    """Knobs for the online learner."""
    learning_rate: float = 0.005
    min_samples_per_update: int = 12
    update_every_n_closures: int = 6
    weight_min: float = 0.05
    weight_max: float = 0.50
    regularization: float = 0.001
    # Win definition: realized_r >= this is a "win"
    win_threshold_r: float = 0.20
    # Walk-forward calibration
    walk_forward_enabled: bool = True
    walk_forward_train_fraction: float = 0.70
    walk_forward_overfit_threshold: float = 1.30  # val_loss / train_loss
    walk_forward_min_val_samples: int = 4


@dataclass
class WeightUpdate:
    """One round of suggested weight changes."""
    samples_used: int
    pre_update_weights: Dict[str, float]
    post_update_weights: Dict[str, float]
    weight_deltas: Dict[str, float]
    model_loss: float
    train_loss: float = 0.0
    val_loss: float = 0.0
    walk_forward_accepted: bool = True
    walk_forward_overfit_ratio: float = 1.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "samples_used": self.samples_used,
            "pre_update_weights": dict(self.pre_update_weights),
            "post_update_weights": dict(self.post_update_weights),
            "weight_deltas": dict(self.weight_deltas),
            "model_loss": round(self.model_loss, 4),
            "train_loss": round(self.train_loss, 4),
            "val_loss": round(self.val_loss, 4),
            "walk_forward_accepted": self.walk_forward_accepted,
            "walk_forward_overfit_ratio": round(
                self.walk_forward_overfit_ratio, 3),
            "notes": list(self.notes),
        }


class OnlineLearner:
    """Adaptive weight updater.

    Holds the current weights; ``observe_closure(...)`` ingests one
    closed position's record; ``maybe_update()`` performs an update
    when enough samples have arrived.
    """

    _COMPONENTS = (
        "w_base_score", "w_mtf_alignment", "w_projection",
        "w_fees_clearance", "w_portfolio_capacity",
    )

    def __init__(self, *,
                  initial_weights: Dict[str, float],
                  cfg: Optional[OnlineLearnerConfig] = None) -> None:
        self.cfg = cfg or OnlineLearnerConfig()
        self.weights = {k: float(initial_weights.get(k, 0.0))
                        for k in self._COMPONENTS}
        self._renormalize()
        self.observations: List[Dict[str, float]] = []
        self.closures_since_update = 0
        self.update_history: List[WeightUpdate] = []

    def observe_closure(self, *,
                          component_scores: Dict[str, float],
                          realized_r: float,
                          win_threshold_r: Optional[float] = None) -> None:
        """Record one closed-position datapoint.

        ``component_scores`` is the per-component score that the
        aggregator computed at entry (base_score, mtf_alignment_score,
        projection_factor, fees_clearance_score, portfolio_capacity_score).
        ``realized_r`` is the actual outcome.
        """
        threshold = (win_threshold_r if win_threshold_r is not None
                     else self.cfg.win_threshold_r)
        record = {
            "base_score": float(component_scores.get("base_score", 0.0)),
            "mtf_alignment_score": float(component_scores.get(
                "mtf_alignment_score", 0.0)),
            "projection_factor": float(component_scores.get(
                "projection_factor", 0.0)),
            "fees_clearance_score": float(component_scores.get(
                "fees_clearance_score", 0.0)),
            "portfolio_capacity_score": float(component_scores.get(
                "portfolio_capacity_score", 0.0)),
            "won": 1.0 if realized_r >= threshold else 0.0,
            "weight": min(3.0, abs(realized_r) + 0.5),
        }
        self.observations.append(record)
        self.closures_since_update += 1

    def maybe_update(self) -> Optional[WeightUpdate]:
        """Run an SGD step when conditions met. Returns the update or None.

        With walk_forward_enabled (default), the observations are split
        into TRAIN (oldest 70%) and VAL (newest 30%). We compute weights
        from TRAIN only, then score on VAL. If val_loss > train_loss *
        overfit_threshold, we REJECT the update — the proposed weights
        overfit and won't generalize. The current weights are kept.
        """
        cfg = self.cfg
        if self.closures_since_update < cfg.update_every_n_closures:
            return None
        if len(self.observations) < cfg.min_samples_per_update:
            return None

        pre = dict(self.weights)
        all_obs = list(self.observations)
        n_total = len(all_obs)

        if cfg.walk_forward_enabled:
            split = max(1, int(n_total * cfg.walk_forward_train_fraction))
            train_obs = all_obs[:split]
            val_obs = all_obs[split:]
            if len(val_obs) < cfg.walk_forward_min_val_samples:
                # Not enough holdout — fall back to single-pass update.
                train_obs = all_obs
                val_obs = []
        else:
            train_obs = all_obs
            val_obs = []

        # ── Train pass: compute gradients + new weights ─────────────
        proposed_weights = self._gradient_step(train_obs, pre)
        train_loss = self._compute_loss(train_obs, proposed_weights)

        # ── Validation pass (if any) ────────────────────────────────
        val_loss = 0.0
        wf_accepted = True
        overfit_ratio = 1.0
        if val_obs:
            val_loss = self._compute_loss(val_obs, proposed_weights)
            ratio_denom = max(0.01, train_loss)
            overfit_ratio = val_loss / ratio_denom
            wf_accepted = (overfit_ratio
                            <= cfg.walk_forward_overfit_threshold)

        # ── Apply or reject ────────────────────────────────────────
        deltas: Dict[str, float] = {}
        notes: List[str] = []
        if wf_accepted:
            for k in self._COMPONENTS:
                deltas[k] = proposed_weights[k] - self.weights[k]
                self.weights[k] = proposed_weights[k]
            self._renormalize()
            notes.append(
                f"online_learner: applied; n_train={len(train_obs)}, "
                f"n_val={len(val_obs)}, train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}"
            )
        else:
            # Rejected: keep current weights, but record the attempt.
            for k in self._COMPONENTS:
                deltas[k] = 0.0
            notes.append(
                f"online_learner: REJECTED (overfit ratio "
                f"{overfit_ratio:.2f} > {cfg.walk_forward_overfit_threshold:.2f}); "
                f"weights unchanged"
            )

        post = dict(self.weights)
        loss_for_summary = (val_loss if val_obs and wf_accepted else train_loss)
        update = WeightUpdate(
            samples_used=n_total,
            pre_update_weights=pre,
            post_update_weights=post,
            weight_deltas=deltas,
            model_loss=loss_for_summary,
            train_loss=train_loss,
            val_loss=val_loss,
            walk_forward_accepted=wf_accepted,
            walk_forward_overfit_ratio=overfit_ratio,
            notes=notes,
        )
        self.update_history.append(update)
        self.closures_since_update = 0
        return update

    def _gradient_step(self, observations: List[Dict[str, float]],
                         prior: Dict[str, float]) -> Dict[str, float]:
        """One SGD step over the given observations. Returns proposed weights
        WITHOUT mutating self.weights."""
        cfg = self.cfg
        n = len(observations)
        grads = {k: 0.0 for k in self._COMPONENTS}
        for obs in observations:
            features = self._features(obs)
            logit = sum(self.weights[k] * features[k] for k in self._COMPONENTS)
            p = 1.0 / (1.0 + math.exp(-(logit - 0.5)))
            err = obs["won"] - p
            sample_w = obs["weight"]
            for k in self._COMPONENTS:
                grads[k] += sample_w * err * features[k]
        # Regularization toward prior.
        for k in self._COMPONENTS:
            grads[k] -= cfg.regularization * (self.weights[k] - prior[k]) * n
        proposed = {}
        for k in self._COMPONENTS:
            new_w = self.weights[k] + (cfg.learning_rate / max(1, n)) * grads[k]
            proposed[k] = max(cfg.weight_min, min(cfg.weight_max, new_w))
        return proposed

    def _compute_loss(self, observations: List[Dict[str, float]],
                        weights: Dict[str, float]) -> float:
        """Average sample-weighted cross-entropy under given weights."""
        if not observations:
            return 0.0
        total = 0.0
        for obs in observations:
            features = self._features(obs)
            logit = sum(weights[k] * features[k] for k in self._COMPONENTS)
            p = 1.0 / (1.0 + math.exp(-(logit - 0.5)))
            y = obs["won"]
            sample_w = obs["weight"]
            total += sample_w * (-(y * math.log(max(1e-9, p))
                                     + (1.0 - y) * math.log(max(1e-9, 1.0 - p))))
        return total / max(1, len(observations))

    @staticmethod
    def _features(obs: Dict[str, float]) -> Dict[str, float]:
        return {
            "w_base_score": obs["base_score"],
            "w_mtf_alignment": obs["mtf_alignment_score"],
            "w_projection": obs["projection_factor"],
            "w_fees_clearance": obs["fees_clearance_score"],
            "w_portfolio_capacity": obs["portfolio_capacity_score"],
        }

    def apply_to_aggregator_config(self, aggregator_cfg) -> None:
        """Push the current learned weights into a live AggregatorConfig.

        Done in-place. Tests use this to verify integration.
        """
        for k, v in self.weights.items():
            if hasattr(aggregator_cfg, k):
                setattr(aggregator_cfg, k, float(v))

    def summary(self) -> Dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "n_observations": len(self.observations),
            "n_updates": len(self.update_history),
            "last_update": (self.update_history[-1].to_dict()
                             if self.update_history else None),
        }

    # ── Internal ────────────────────────────────────────────────────

    def _renormalize(self) -> None:
        total = sum(self.weights.values())
        if total <= 0:
            return
        for k in self.weights:
            self.weights[k] /= total
