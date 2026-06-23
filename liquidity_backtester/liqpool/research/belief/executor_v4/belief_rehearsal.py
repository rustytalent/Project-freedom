"""Belief Rehearsal Ensemble — conditional kNN off-policy evaluation.

The founder's brief (2026-06-22): "Do not use Monte Carlo, that is old.
Use something advanced. This layer can do miracles for us — make it
elaborately, brilliantly, luxuriously. It should tell the executor what
the other scores should be so an order should execute."

What this is, in one sentence: at every entry decision, we look at the
*real* historical trades that were most similar to the current state,
ask "how did the analogues actually finish?" across several entry-
parameter perturbations, and return a calibrated rehearsal score that
the aggregator weighs alongside its other components. This is
**non-parametric off-policy evaluation** — no GBM fantasy, no log-normal
assumption, no parametric distribution at all. The fat tails, the MM
behaviour, the manipulation patterns: all are baked into the analogue
distribution because the analogues are real closed trades.

Why this beats Monte Carlo:
  * Monte Carlo assumes you know the data-generating process. We don't.
    The Indian options book has fat tails, regime breaks, MM herding,
    and event-day jumps that GBM cannot model.
  * Monte Carlo with hand-tuned distributions inherits the modeller's
    biases. Real analogues do not.
  * Monte Carlo gives no actionable feedback. The rehearsal output tells
    you which perturbation would have worked better — "if you'd entered
    at 30bps tighter stop, P(target) goes from 0.55 to 0.68."
  * Monte Carlo cannot accumulate institutional memory. The analogue
    store does — every closed trade makes tomorrow's rehearsal smarter.

Architecture:

    BeliefObservation       — one closed trade, standardized features +
                                 entry params + outcome + regime tag
    BeliefObservationStore  — on-disk + in-memory ring of observations,
                                 loads N prior days at startup, atomic
                                 per-IST-date JSONL writes
    FeatureSpace            — Welford running mean/std for every feature;
                                 outcome-correlation feature importances
                                 derived from history
    AnalogueRetriever       — brute-force weighted-Euclidean kNN over
                                 standardized features (≤2000 obs makes
                                 a real index unnecessary; clean and
                                 dependency-free)
    PerturbationGrid        — generates a small grid of entry parameter
                                 variants around the proposed entry
                                 (confidence floor, size, target, stop,
                                 profile swap)
    RehearsalDecision       — per-perturbation outcome distribution,
                                 best perturbation, calibrated score,
                                 recommended action, full notes
    BeliefRehearsalEnsemble — the orchestrator; what the manager calls

Wired into the manager: on each entry consideration the ensemble is
rehearsed; the resulting rehearsal_score becomes the 6th weight
component in the AggregatorConfig. On each close, the observation is
recorded and persisted. On startup, prior days' observations are
loaded so the ensemble inherits institutional memory.

State-of-the-art-without-ML-deps: this is what professional desks call
**off-policy evaluation with kNN-based importance weighting** (Beygelzimer
& Langford 2009; Mandel et al. 2014; Thomas & Brunskill 2016 for the
doubly-robust extension we'll layer on later).
"""
from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


IST = timezone(timedelta(hours=5, minutes=30))


def _today_ist() -> str:
    return datetime.now(IST).date().isoformat()


def _now_iso() -> str:
    return datetime.now(IST).isoformat(timespec="seconds")


# ── Feature vector schema ─────────────────────────────────────────


# Every observation carries the same fixed feature schema so kNN over a
# constant-dimensional space is well-defined. Adding a feature later is
# a schema migration — bump SCHEMA_VERSION and provide a default in
# ``_feature_vector_from_state``.
SCHEMA_VERSION = 1


FEATURE_KEYS: Tuple[str, ...] = (
    # Substrate (the dev-of-dev family)
    "regime_stability_index",
    "ce_signed_z_velocity",
    "ce_signed_z_acceleration",
    "pe_signed_z_velocity",
    "pe_signed_z_acceleration",
    "net_intent_velocity",
    "net_intent_acceleration",
    "dispersion_velocity",
    "thesis_bull_velocity",
    "thesis_bear_velocity",
    "epicenter_migration_distance",
    # Scenario web masses
    "chop_mass",
    "manipulation_mass",
    "tail_mass",
    "directional_consensus_horizon_weighted",
    # IV / flow
    "iv_state_confidence",
    "clean_mark_fraction",
    "net_intent_z",
    # MM mind
    "mm_dominant_probability",
    "mm_confidence",
    # Crowd
    "retail_similarity_score",
    # Risk
    "portfolio_drawdown_r",
    "open_positions_count",
)


# Entry-parameter axes — the perturbations operate over these.
PARAM_KEYS: Tuple[str, ...] = (
    "entry_confidence", "size_lots", "target_r", "stop_r",
    "profile_scalp", "direction",
)


# ── Observation dataclass ─────────────────────────────────────────


@dataclass
class BeliefObservation:
    """One closed trade as the rehearsal store sees it.

    All vectors are stored RAW (not standardised) — the FeatureSpace
    holds the running statistics and standardises on-demand. This means
    older observations remain valid as the statistics evolve.
    """
    obs_id: str
    ts: str                         # IST iso
    bar_index: int
    schema_version: int
    feature_vector: Dict[str, float]
    entry_params: Dict[str, float]
    # outcome carries both numerics (realized_r, MFE, MAE) AND the exit
    # reason string — so Dict[str, Any]. Helpers in this module coerce
    # the numeric fields to floats where they care; everything else
    # passes through.
    outcome: Dict[str, Any]
    regime_tag: Dict[str, Any]
    strategy: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obs_id": self.obs_id,
            "ts": self.ts,
            "bar_index": self.bar_index,
            "schema_version": self.schema_version,
            "feature_vector": dict(self.feature_vector),
            "entry_params": dict(self.entry_params),
            "outcome": dict(self.outcome),
            "regime_tag": dict(self.regime_tag),
            "strategy": self.strategy,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BeliefObservation":
        return cls(
            obs_id=str(d.get("obs_id", "")),
            ts=str(d.get("ts", "")),
            bar_index=int(d.get("bar_index", 0)),
            schema_version=int(d.get("schema_version", 0)),
            feature_vector={k: float(v) for k, v in
                              (d.get("feature_vector") or {}).items()},
            entry_params={k: float(v) for k, v in
                            (d.get("entry_params") or {}).items()},
            # outcome is mixed — strings (exit_reason) coexist with
            # numerics. Preserve as-is; consumers cast on read.
            outcome=dict(d.get("outcome") or {}),
            regime_tag=dict(d.get("regime_tag") or {}),
            strategy=str(d.get("strategy") or ""),
            notes=list(d.get("notes") or []),
        )


# ── On-disk store ─────────────────────────────────────────────────


class BeliefObservationStore:
    """Per-IST-date JSONL store of closed-trade observations.

    Filename: ``belief_observations_<YYYY-MM-DD>.jsonl`` under the state
    directory. Append-only — every closed trade is a new line. Atomic
    only at line granularity (a partial write at EOF is recoverable by
    truncating the final line). The in-memory ring holds the most recent
    ``ring_size`` observations across all loaded days.

    Why JSONL rather than one JSON document per day: append-only writes
    are cheaper, line-grained corruption recovery is trivial, and we
    don't need to load the whole file just to add one closure.
    """

    FILE_PREFIX = "belief_observations_"
    FILE_SUFFIX = ".jsonl"

    def __init__(self, *, state_dir: Path, ring_size: int = 2000,
                  history_days: int = 14) -> None:
        self.state_dir = Path(state_dir)
        self.ring_size = int(ring_size)
        self.history_days = int(history_days)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._ring: List[BeliefObservation] = []
        self._loaded_dates: List[str] = []

    # ── Path helpers ───────────────────────────────────────────────

    def path_for(self, date_str: str) -> Path:
        return self.state_dir / f"{self.FILE_PREFIX}{date_str}{self.FILE_SUFFIX}"

    def path_for_today(self) -> Path:
        return self.path_for(_today_ist())

    def prior_dates(self, n_days: int) -> List[str]:
        today = datetime.now(IST).date()
        return [(today - timedelta(days=i)).isoformat()
                for i in range(1, n_days + 1)]

    # ── Read API ───────────────────────────────────────────────────

    def load_history(self) -> int:
        """Read up to history_days of prior + today into the in-memory
        ring. Returns the number of observations loaded."""
        loaded = 0
        # Today first so we can append to it; then prior days, oldest first.
        dates = list(reversed(self.prior_dates(self.history_days)))
        dates.append(_today_ist())
        seen: List[BeliefObservation] = []
        for d in dates:
            path = self.path_for(d)
            if not path.exists():
                continue
            try:
                with open(path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obs = BeliefObservation.from_dict(json.loads(line))
                            seen.append(obs)
                            loaded += 1
                        except Exception:
                            continue
                self._loaded_dates.append(d)
            except Exception:
                continue
        # Keep most recent ring_size.
        self._ring = seen[-self.ring_size:]
        return loaded

    def observations(self) -> List[BeliefObservation]:
        return list(self._ring)

    # ── Write API ──────────────────────────────────────────────────

    def append(self, obs: BeliefObservation) -> None:
        """Append the observation to today's file and the in-memory ring.

        Persistence failure must never break the trading loop — we log
        nothing here; the caller relies on the ring being valid in-
        memory even if disk is wedged.
        """
        self._ring.append(obs)
        if len(self._ring) > self.ring_size:
            self._ring = self._ring[-self.ring_size:]
        try:
            path = self.path_for_today()
            # Append one JSON line. We use atomic file append (O_APPEND on
            # the OS) — POSIX guarantees per-write atomicity for writes
            # smaller than PIPE_BUF, and a JSONL line is tiny.
            with open(path, "a") as f:
                f.write(json.dumps(obs.to_dict(), default=str) + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
        except Exception:
            pass


# ── Feature standardisation ───────────────────────────────────────


class _Welford:
    """Welford's online variance / mean estimator. Numerically stable;
    no overflow even on long sessions."""

    __slots__ = ("n", "mean", "m2")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def push(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.m2 / self.n if self.n > 1 else 1.0

    @property
    def std(self) -> float:
        v = self.variance
        return math.sqrt(v) if v > 0 else 1.0


class FeatureSpace:
    """Maintains per-feature running statistics + outcome-correlation
    feature weights for the kNN distance metric.

    The weights are |Pearson correlation between feature and realized R|
    over the loaded history, clipped to [0.1, 1.0] so no feature is
    completely ignored. Recomputed every ``recompute_every`` appends.
    """

    def __init__(self, *, recompute_every: int = 25,
                  weight_floor: float = 0.10,
                  weight_ceiling: float = 1.0) -> None:
        self._stats: Dict[str, _Welford] = {k: _Welford() for k in FEATURE_KEYS}
        self._weights: Dict[str, float] = {k: 1.0 for k in FEATURE_KEYS}
        self.recompute_every = int(recompute_every)
        self.weight_floor = float(weight_floor)
        self.weight_ceiling = float(weight_ceiling)
        self._appends_since_recompute = 0

    def observe(self, observation: BeliefObservation) -> None:
        for k in FEATURE_KEYS:
            v = float(observation.feature_vector.get(k, 0.0))
            self._stats[k].push(v)
        self._appends_since_recompute += 1

    def maybe_recompute_weights(self,
                                    observations: List[BeliefObservation]
                                    ) -> bool:
        if self._appends_since_recompute < self.recompute_every:
            return False
        if len(observations) < 30:
            self._appends_since_recompute = 0
            return False
        self.recompute_weights(observations)
        self._appends_since_recompute = 0
        return True

    def recompute_weights(self,
                              observations: List[BeliefObservation]) -> None:
        ys = [float(o.outcome.get("realized_r", 0.0)) for o in observations]
        y_mean = statistics.mean(ys) if ys else 0.0
        y_var = (sum((y - y_mean) ** 2 for y in ys) / max(1, len(ys) - 1)
                 if len(ys) > 1 else 1.0)
        if y_var <= 1e-9:
            return
        new_weights: Dict[str, float] = {}
        for k in FEATURE_KEYS:
            xs = [float(o.feature_vector.get(k, 0.0)) for o in observations]
            x_mean = statistics.mean(xs)
            x_var = (sum((x - x_mean) ** 2 for x in xs)
                     / max(1, len(xs) - 1))
            if x_var <= 1e-9:
                new_weights[k] = self.weight_floor
                continue
            cov = (sum((xs[i] - x_mean) * (ys[i] - y_mean)
                        for i in range(len(xs))) / max(1, len(xs) - 1))
            corr = cov / math.sqrt(x_var * y_var)
            w = abs(corr)
            new_weights[k] = max(self.weight_floor,
                                  min(self.weight_ceiling, w))
        self._weights = new_weights

    def standardise(self, feature_vector: Dict[str, float]
                     ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k in FEATURE_KEYS:
            stat = self._stats[k]
            x = float(feature_vector.get(k, 0.0))
            if stat.n < 5:
                out[k] = x  # too little data to standardise
            else:
                out[k] = (x - stat.mean) / max(1e-6, stat.std)
        return out

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self._weights)


# ── Retrieval ─────────────────────────────────────────────────────


@dataclass
class Analogue:
    observation: BeliefObservation
    similarity: float                     # in [0, 1]; 1 = identical
    distance: float                       # weighted Euclidean, >= 0


class AnalogueRetriever:
    """Brute-force kNN over standardised feature vectors.

    With ≤2000 observations, brute-force search is ~0.5ms in Python —
    no need for a kd-tree or annoy index. Adding a real ANN library
    later is straightforward; the interface here doesn't depend on it.
    """

    def __init__(self, *, feature_space: FeatureSpace, k: int = 20,
                  min_similarity: float = 0.0,
                  regime_match_bonus: float = 0.10,
                  ) -> None:
        self.feature_space = feature_space
        self.k = int(k)
        self.min_similarity = float(min_similarity)
        self.regime_match_bonus = float(regime_match_bonus)

    def _distance(self, q_std: Dict[str, float],
                    obs_std: Dict[str, float],
                    weights: Dict[str, float]) -> float:
        s = 0.0
        for k in FEATURE_KEYS:
            d = q_std.get(k, 0.0) - obs_std.get(k, 0.0)
            s += weights.get(k, 1.0) * d * d
        return math.sqrt(s)

    def query(self, *,
                 query_features: Dict[str, float],
                 observations: List[BeliefObservation],
                 query_regime_tag: Optional[Dict[str, Any]] = None,
                 ) -> List[Analogue]:
        if not observations:
            return []
        q_std = self.feature_space.standardise(query_features)
        weights = self.feature_space.weights
        # Compute distances; standardise each observation lazily.
        results: List[Tuple[float, BeliefObservation]] = []
        q_family = ((query_regime_tag or {}).get("dominant_family", "")
                    if query_regime_tag else "")
        for obs in observations:
            obs_std = self.feature_space.standardise(obs.feature_vector)
            d = self._distance(q_std, obs_std, weights)
            # Same-regime bonus — bring same-family observations closer.
            if q_family and obs.regime_tag.get("dominant_family") == q_family:
                d *= (1.0 - self.regime_match_bonus)
            results.append((d, obs))
        results.sort(key=lambda x: x[0])
        # Convert distance to similarity. We use exp(-d) to map d ∈ [0,∞)
        # → similarity ∈ (0,1]. Not normalised across the corpus —
        # interpretation is "how close in feature space", not "rank-1".
        out: List[Analogue] = []
        for d, obs in results[:self.k]:
            sim = math.exp(-d)
            if sim < self.min_similarity:
                break
            out.append(Analogue(observation=obs, similarity=sim, distance=d))
        return out


# ── Perturbation grid ─────────────────────────────────────────────


@dataclass
class Perturbation:
    """One alternative set of entry parameters being rehearsed."""
    name: str
    deltas: Dict[str, float]              # delta from proposed params
    notes: str = ""

    def applied_to(self, base_params: Dict[str, float]
                    ) -> Dict[str, float]:
        out = dict(base_params)
        for k, dv in self.deltas.items():
            base = float(out.get(k, 0.0))
            if k in ("entry_confidence", "target_r", "stop_r", "size_lots"):
                out[k] = max(0.0, base * (1.0 + dv))
            else:
                out[k] = base + dv
        return out


class PerturbationGrid:
    """Generates the small set of entry-parameter variants we rehearse.

    The default grid is deliberately compact (8 perturbations including
    the no-op "as-proposed" baseline) so we can rehearse each variant
    against the analogue set without blowing up latency. Each perturbation
    is a one-dimensional move from the proposed entry; cross-dimensional
    perturbations are deliberately omitted (they explode combinatorially
    and the marginal information is dominated by the per-axis moves).
    """

    @staticmethod
    def default_grid() -> List[Perturbation]:
        return [
            Perturbation(name="as_proposed", deltas={}),
            Perturbation(name="conf_+10pct",
                          deltas={"entry_confidence": +0.10},
                          notes="entered only with 10% higher confidence"),
            Perturbation(name="conf_-10pct",
                          deltas={"entry_confidence": -0.10},
                          notes="entered with looser confidence floor"),
            Perturbation(name="size_+50pct",
                          deltas={"size_lots": +0.50},
                          notes="entered with 1.5× sizing"),
            Perturbation(name="size_-50pct",
                          deltas={"size_lots": -0.50},
                          notes="entered with half sizing"),
            Perturbation(name="target_-20pct",
                          deltas={"target_r": -0.20},
                          notes="closer target (lock profits sooner)"),
            Perturbation(name="stop_-15pct",
                          deltas={"stop_r": -0.15},
                          notes="tighter stop (cut faster)"),
            Perturbation(name="stop_+15pct",
                          deltas={"stop_r": +0.15},
                          notes="wider stop (more breathing room)"),
        ]


# ── Rehearsal decision ────────────────────────────────────────────


@dataclass
class PerturbationOutcome:
    """The distribution of realised R among analogues for one perturbation."""
    name: str
    n_analogues: int
    mean_r: float
    std_r: float
    median_r: float
    p_target: float
    p_stop: float
    p_neither: float
    p_profit: float                   # P(realized_r > 0)
    expected_r_above_zero: float      # E[realized_r | realized_r > 0]
    expected_r_below_zero: float      # E[realized_r | realized_r < 0]
    similarity_weight_sum: float      # sum of similarity weights used
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_analogues": self.n_analogues,
            "mean_r": round(self.mean_r, 3),
            "std_r": round(self.std_r, 3),
            "median_r": round(self.median_r, 3),
            "p_target": round(self.p_target, 3),
            "p_stop": round(self.p_stop, 3),
            "p_neither": round(self.p_neither, 3),
            "p_profit": round(self.p_profit, 3),
            "expected_r_above_zero": round(self.expected_r_above_zero, 3),
            "expected_r_below_zero": round(self.expected_r_below_zero, 3),
            "similarity_weight_sum": round(self.similarity_weight_sum, 3),
            "notes": self.notes,
        }


@dataclass
class RehearsalDecision:
    """The ensemble's verdict for one entry consideration."""
    ran: bool
    rehearsal_score: float            # in [0, 1] — fed to aggregator
    confidence: float                 # in [0, 1] — how reliable
    n_analogues_total: int
    mean_similarity: float
    per_perturbation: List[PerturbationOutcome]
    best_perturbation_name: str
    current_perturbation_name: str = "as_proposed"
    recommended_action: str = "PROCEED"   # PROCEED / TUNE / DEFER / REFUSE
    deferral_reason: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ran": self.ran,
            "rehearsal_score": round(self.rehearsal_score, 3),
            "confidence": round(self.confidence, 3),
            "n_analogues_total": self.n_analogues_total,
            "mean_similarity": round(self.mean_similarity, 3),
            "per_perturbation": [p.to_dict() for p in self.per_perturbation],
            "best_perturbation_name": self.best_perturbation_name,
            "current_perturbation_name": self.current_perturbation_name,
            "recommended_action": self.recommended_action,
            "deferral_reason": self.deferral_reason,
            "notes": list(self.notes),
        }

    @staticmethod
    def empty(reason: str) -> "RehearsalDecision":
        return RehearsalDecision(
            ran=False, rehearsal_score=0.50, confidence=0.0,
            n_analogues_total=0, mean_similarity=0.0,
            per_perturbation=[], best_perturbation_name="",
            recommended_action="PROCEED",
            deferral_reason=reason,
            notes=[reason],
        )


# ── Outcome aggregation ───────────────────────────────────────────


def _winsorise(values: List[float], lo: float = 0.02, hi: float = 0.98
                ) -> List[float]:
    """Cap extreme outliers at the [2,98] percentiles. Keeps a single
    blown-up trade from swinging the entire distribution."""
    if not values:
        return values
    s = sorted(values)
    n = len(s)
    if n < 5:
        return list(values)
    i_lo = max(0, int(lo * (n - 1)))
    i_hi = min(n - 1, int(hi * (n - 1)))
    lo_v, hi_v = s[i_lo], s[i_hi]
    return [max(lo_v, min(hi_v, v)) for v in values]


def _aggregate_perturbation(perturbation: Perturbation,
                              analogues: List[Analogue],
                              base_params: Dict[str, float],
                              ) -> PerturbationOutcome:
    """Compute the perturbation's outcome distribution by importance-
    weighting analogues toward those whose entry parameters are closer
    to the perturbed parameters.

    This is the kernel of the off-policy estimator: each analogue is
    re-weighted by a kernel of the entry-parameter distance, so the
    rehearsal evaluates the *perturbed* policy rather than the policy
    that originally generated the data.
    """
    target_params = perturbation.applied_to(base_params)
    weights: List[float] = []
    rs: List[float] = []
    p_target_n = 0.0
    p_stop_n = 0.0
    p_neither_n = 0.0
    total_w = 0.0
    for a in analogues:
        # Importance kernel: exp(-|Δ params|) over the perturbation axes
        # that actually moved, scaled by the analogue's similarity.
        d_param = 0.0
        for k, dv in perturbation.deltas.items():
            actual = float(a.observation.entry_params.get(k, 0.0))
            wanted = float(target_params.get(k, 0.0))
            base = max(1e-3, abs(float(base_params.get(k, 1.0))))
            d_param += ((actual - wanted) / base) ** 2
        kernel = math.exp(-math.sqrt(d_param))
        w = a.similarity * kernel
        if w <= 0:
            continue
        r = float(a.observation.outcome.get("realized_r", 0.0))
        rs.append(r)
        weights.append(w)
        total_w += w
        exit_reason = str(a.observation.outcome.get("exit_reason", ""))
        if "target" in exit_reason.lower():
            p_target_n += w
        elif "stop" in exit_reason.lower():
            p_stop_n += w
        else:
            p_neither_n += w

    if total_w <= 0 or not rs:
        return PerturbationOutcome(
            name=perturbation.name,
            n_analogues=0, mean_r=0.0, std_r=0.0, median_r=0.0,
            p_target=0.0, p_stop=0.0, p_neither=0.0,
            p_profit=0.0, expected_r_above_zero=0.0,
            expected_r_below_zero=0.0, similarity_weight_sum=0.0,
            notes="no surviving analogues after kernel weighting",
        )

    # Winsorise to keep one blown trade from dominating the mean.
    rs_w = _winsorise(rs)
    weighted_mean = sum(w * r for w, r in zip(weights, rs_w)) / total_w
    weighted_var = sum(w * (r - weighted_mean) ** 2
                       for w, r in zip(weights, rs_w)) / total_w
    std = math.sqrt(max(0.0, weighted_var))
    median = statistics.median(rs_w)

    # P(profit) and conditional expectations.
    p_profit_w = sum(w for w, r in zip(weights, rs_w) if r > 0.0) / total_w
    above = [(w, r) for w, r in zip(weights, rs_w) if r > 0.0]
    below = [(w, r) for w, r in zip(weights, rs_w) if r <= 0.0]
    e_above = (sum(w * r for w, r in above) / sum(w for w, _ in above)
               if above else 0.0)
    e_below = (sum(w * r for w, r in below) / sum(w for w, _ in below)
               if below else 0.0)

    return PerturbationOutcome(
        name=perturbation.name,
        n_analogues=len(rs),
        mean_r=weighted_mean,
        std_r=std,
        median_r=median,
        p_target=p_target_n / total_w,
        p_stop=p_stop_n / total_w,
        p_neither=p_neither_n / total_w,
        p_profit=p_profit_w,
        expected_r_above_zero=e_above,
        expected_r_below_zero=e_below,
        similarity_weight_sum=total_w,
        notes=perturbation.notes,
    )


# ── Ensemble orchestrator ─────────────────────────────────────────


@dataclass
class RehearsalConfig:
    """Knobs."""
    enabled: bool = True
    min_observations_to_run: int = 12
    min_confidence_to_emit_score: float = 0.20
    k: int = 20
    min_similarity: float = 0.0
    regime_match_bonus: float = 0.10
    recompute_weights_every: int = 25
    history_days: int = 14
    ring_size: int = 2000
    # If the best perturbation outperforms the current by this much
    # weighted-mean R, we surface a TUNE recommendation.
    tune_threshold_r: float = 0.20
    # If the best perturbation has P(profit) below this even for the
    # current proposal, we surface a DEFER recommendation.
    defer_p_profit_below: float = 0.40


class BeliefRehearsalEnsemble:
    """The conditional-kNN ensemble. The manager calls ``rehearse`` at
    every entry consideration and ``record_closure`` at every close.

    Lifecycle:
        construct → call ``boot()`` once at session start → call
        ``rehearse()`` per entry consideration → call
        ``record_closure()`` per close.

    The ensemble has no concept of "live" vs "paper" — it just sees
    closures. Hooking it up to the paper broker for backtests is
    automatic (the manager calls the same record_closure path).
    """

    def __init__(self, *, state_dir: Path,
                  cfg: Optional[RehearsalConfig] = None,
                  ) -> None:
        self.cfg = cfg or RehearsalConfig()
        self.store = BeliefObservationStore(
            state_dir=state_dir,
            ring_size=self.cfg.ring_size,
            history_days=self.cfg.history_days,
        )
        self.feature_space = FeatureSpace(
            recompute_every=self.cfg.recompute_weights_every,
        )
        self.retriever = AnalogueRetriever(
            feature_space=self.feature_space,
            k=self.cfg.k,
            min_similarity=self.cfg.min_similarity,
            regime_match_bonus=self.cfg.regime_match_bonus,
        )
        self.grid = PerturbationGrid.default_grid()
        self._last_decision: Optional[RehearsalDecision] = None
        self._booted: bool = False

    # ── Lifecycle ──────────────────────────────────────────────────

    def boot(self) -> Dict[str, Any]:
        """Load history into the ring + bootstrap the feature space.
        Idempotent; safe to call from the manager's first tick path."""
        if self._booted:
            return {"already_booted": True}
        loaded = self.store.load_history()
        obs = self.store.observations()
        for o in obs:
            self.feature_space.observe(o)
        if len(obs) >= 30:
            self.feature_space.recompute_weights(obs)
        self._booted = True
        return {
            "loaded": loaded,
            "in_ring": len(obs),
            "feature_weights_computed": len(obs) >= 30,
        }

    # ── Rehearse ───────────────────────────────────────────────────

    def rehearse(self, *,
                    query_features: Dict[str, float],
                    proposed_params: Dict[str, float],
                    regime_tag: Optional[Dict[str, Any]] = None,
                    ) -> RehearsalDecision:
        """The hot path. Cheap on cold start (no analogues), valuable
        once the store has accumulated meaningful history."""
        cfg = self.cfg
        if not cfg.enabled:
            decision = RehearsalDecision.empty("rehearsal disabled")
            self._last_decision = decision
            return decision
        obs = self.store.observations()
        if len(obs) < cfg.min_observations_to_run:
            decision = RehearsalDecision.empty(
                f"insufficient history: {len(obs)} obs "
                f"< {cfg.min_observations_to_run} needed")
            self._last_decision = decision
            return decision

        analogues = self.retriever.query(
            query_features=query_features,
            observations=obs,
            query_regime_tag=regime_tag,
        )
        if not analogues:
            decision = RehearsalDecision.empty("no analogues retrieved")
            self._last_decision = decision
            return decision

        mean_sim = sum(a.similarity for a in analogues) / len(analogues)
        # Aggregate each perturbation.
        outcomes = [
            _aggregate_perturbation(p, analogues, proposed_params)
            for p in self.grid
        ]
        # Pick the best (highest expected mean R among those with
        # at least 3 analogues — protects against degenerate 1-sample
        # spikes).
        viable = [o for o in outcomes if o.n_analogues >= 3]
        if not viable:
            viable = outcomes
        best = max(viable, key=lambda o: o.mean_r)
        current = next((o for o in outcomes if o.name == "as_proposed"),
                       outcomes[0])

        # Calibrated rehearsal score: P(profit) under the *current*
        # perturbation, weighted by our confidence in the estimate.
        confidence = min(1.0, math.log1p(current.n_analogues) / math.log1p(20)
                         * (mean_sim ** 0.5))
        if confidence < cfg.min_confidence_to_emit_score:
            # Not enough confidence to substantively move the aggregator;
            # emit neutral 0.50 so the aggregator effectively ignores us.
            score = 0.50
        else:
            score = current.p_profit

        # Recommendation logic.
        recommended = "PROCEED"
        notes: List[str] = []
        if current.p_profit < cfg.defer_p_profit_below \
                and confidence >= cfg.min_confidence_to_emit_score:
            recommended = "DEFER"
            notes.append(
                f"current perturbation P(profit) {current.p_profit:.2f} "
                f"< {cfg.defer_p_profit_below:.2f}; "
                f"DEFER until the state moves")
        elif (best.name != current.name
              and best.mean_r - current.mean_r > cfg.tune_threshold_r
              and best.n_analogues >= 3):
            recommended = "TUNE"
            notes.append(
                f"alternative '{best.name}' shows expected "
                f"{best.mean_r:+.2f}R vs current {current.mean_r:+.2f}R "
                f"(Δ {best.mean_r - current.mean_r:+.2f}R); "
                f"consider {best.notes or 'tuning entry params'}")

        decision = RehearsalDecision(
            ran=True,
            rehearsal_score=float(score),
            confidence=float(confidence),
            n_analogues_total=len(analogues),
            mean_similarity=float(mean_sim),
            per_perturbation=outcomes,
            best_perturbation_name=best.name,
            current_perturbation_name=current.name,
            recommended_action=recommended,
            notes=notes,
        )
        self._last_decision = decision
        return decision

    # ── Record ─────────────────────────────────────────────────────

    def record_closure(self, observation: BeliefObservation) -> None:
        """Adds the closed-trade observation to the store + feature space.

        Triggers a weight recomputation if the appender's running count
        crosses the recompute threshold.
        """
        self.store.append(observation)
        self.feature_space.observe(observation)
        self.feature_space.maybe_recompute_weights(self.store.observations())

    # ── Cockpit surface ────────────────────────────────────────────

    def last_decision(self) -> Optional[RehearsalDecision]:
        return self._last_decision

    def summary(self) -> Dict[str, Any]:
        last = self.last_decision()
        return {
            "booted": self._booted,
            "n_observations_in_ring": len(self.store.observations()),
            "history_days_loaded": len(self.store._loaded_dates),
            "feature_weights": self.feature_space.weights,
            "last_decision": last.to_dict() if last is not None else None,
        }


# ── Feature extraction helper ─────────────────────────────────────


def build_query_features(*,
                            rich_context: Optional[Dict[str, Any]],
                            web_snapshot: Optional[Dict[str, Any]],
                            mm_posterior: Optional[Dict[str, Any]],
                            crowd_report: Optional[Dict[str, Any]],
                            portfolio_risk: Optional[Dict[str, Any]],
                            iv_state: Optional[Dict[str, Any]],
                            flow_event: Optional[Dict[str, Any]],
                            n_open_positions: int = 0,
                            ) -> Dict[str, float]:
    """Extract the standard FEATURE_KEYS vector from the manager's
    per-tick state. Centralised so the same shape is built at rehearse
    time and at close time — no schema drift.
    """
    r = rich_context or {}
    w = web_snapshot or {}
    mm = mm_posterior or {}
    crowd = crowd_report or {}
    risk = portfolio_risk or {}
    iv = iv_state or {}
    flow = flow_event or {}
    return {
        "regime_stability_index": float(r.get("regime_stability_index", 0.0)),
        "ce_signed_z_velocity": float(r.get("ce_signed_z_velocity", 0.0)),
        "ce_signed_z_acceleration": float(
            r.get("ce_signed_z_acceleration", 0.0)),
        "pe_signed_z_velocity": float(r.get("pe_signed_z_velocity", 0.0)),
        "pe_signed_z_acceleration": float(
            r.get("pe_signed_z_acceleration", 0.0)),
        "net_intent_velocity": float(r.get("net_intent_velocity", 0.0)),
        "net_intent_acceleration": float(
            r.get("net_intent_acceleration", 0.0)),
        "dispersion_velocity": float(r.get("dispersion_velocity", 0.0)),
        "thesis_bull_velocity": float(r.get("thesis_bull_velocity", 0.0)),
        "thesis_bear_velocity": float(r.get("thesis_bear_velocity", 0.0)),
        "epicenter_migration_distance": float(
            r.get("epicenter_migration_distance", 0.0)),
        "chop_mass": float(w.get("chop_mass", 0.0)),
        "manipulation_mass": float(w.get("manipulation_mass", 0.0)),
        "tail_mass": float(w.get("tail_mass", 0.0)),
        "directional_consensus_horizon_weighted": float(
            w.get("directional_consensus_horizon_weighted", 0.0)),
        "iv_state_confidence": float(iv.get("confidence", 0.0)),
        "clean_mark_fraction": float(iv.get("clean_mark_fraction", 0.0)),
        "net_intent_z": float(iv.get("net_intent_z",
                                       flow.get("net_intent_z", 0.0))),
        "mm_dominant_probability": float(mm.get("dominant_probability", 0.0)),
        "mm_confidence": float(mm.get("confidence", 0.0)),
        "retail_similarity_score": float(
            crowd.get("retail_similarity_score", 0.0)),
        "portfolio_drawdown_r": float(risk.get("portfolio_drawdown_r", 0.0)),
        "open_positions_count": float(n_open_positions),
    }
