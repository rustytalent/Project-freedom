"""Thompson-sampling capital allocator across surviving hypotheses (idea #26).

Once two or more hypotheses survive the harness/walk-forward/null gauntlet
the next question is: *given limited capital, how much should each one get
this week?* Equal split is naive (a 0.8-Sharpe hypothesis is twice as
trustworthy as a 0.4-Sharpe one but gets the same weight). Greedy "all to
the highest Sharpe" ignores that the Sharpe estimate itself is noisy and
the highest in-sample is rarely the highest live.

Thompson sampling is the principled middle ground from the multi-armed-
bandit literature (Thompson 1933; Russo & Van Roy 2014). For each arm
(hypothesis) we maintain a posterior over its true Sharpe. To allocate, we
*sample* one Sharpe from each posterior and weight by the sampled Sharpes.
Repeat many times; the average weights are the allocation. Hypotheses
with high mean and tight posteriors get most of the weight most of the
time (exploit); hypotheses with high uncertainty occasionally sample very
high and get explored.

What this module is NOT:
  * It does not decide whether a hypothesis is good. That is what the
    harness + null tests + ``kill_switch`` are for.
  * It does not size bet-by-bet. That is the per-trade sizer's job. This
    module says "70% of the bankroll goes to H1 this week, 25% to H2,
    5% to H3."
  * It does not handle correlated hypotheses. Two strategies that fire
    on the same setup get weighted independently here; a future
    extension would shrink their joint weight.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# A hypothesis with fewer than this many trades has a noisy Sharpe estimate
# and the allocator should weight the prior more heavily.
MIN_TRADES_PRIOR_DOMINANT: int = 30

# Below this floor we treat a hypothesis as having no edge — the posterior
# sample is replaced by zero when forming weights.
SHARPE_FLOOR: float = 0.0

# Default per-arm cap. Without this a single hypothesis with the best
# in-sample Sharpe and a tight posterior would get 100% — disastrous when
# the in-sample estimate is wrong.
DEFAULT_MAX_WEIGHT: float = 0.60

# Default lower clamp on the allocation. A hypothesis the allocator wants
# to send to zero stays at zero — the floor is for hypotheses still
# allowed to trade, to keep some exploration alive when they are not
# the current best.
DEFAULT_FLOOR_WEIGHT: float = 0.0


@dataclass(frozen=True)
class HypothesisPosterior:
    """Sample-size-aware Sharpe posterior for one hypothesis.

    Built from per-hypothesis trade statistics. The posterior on true
    Sharpe is approximated as ``N(mean_sharpe, posterior_std**2)`` with
    ``posterior_std = stderr_sharpe * (1 + prior_weight)`` where
    ``prior_weight`` shrinks when ``n_trades`` is large. This keeps the
    inflation honest: small samples have wide posteriors regardless of
    their point estimate.
    """
    name: str
    n_trades: int
    mean_sharpe: float
    stderr_sharpe: float

    @classmethod
    def from_report(cls, report: Any) -> "HypothesisPosterior":
        """Build from a ``HypothesisReport`` (research/harness.py)."""
        overall = getattr(report, "overall", None)
        if overall is None:
            raise ValueError(f"report {report!r} has no .overall")
        return cls(
            name=str(getattr(report, "name", "hypothesis")),
            n_trades=int(getattr(overall, "n_trades", 0) or 0),
            mean_sharpe=float(getattr(overall, "sharpe", 0.0) or 0.0),
            stderr_sharpe=_sharpe_stderr_from_n(
                float(getattr(overall, "sharpe", 0.0) or 0.0),
                int(getattr(overall, "n_trades", 0) or 0),
            ),
        )

    @classmethod
    def from_metrics(cls, metrics: Any) -> "HypothesisPosterior":
        """Build from a miner ``HypothesisMetrics``-shaped object."""
        n = int(getattr(metrics, "trades", 0) or 0)
        sharpe = float(getattr(metrics, "sharpe", 0.0) or 0.0)
        return cls(
            name=str(getattr(metrics, "name", "hypothesis")),
            n_trades=n,
            mean_sharpe=sharpe,
            stderr_sharpe=_sharpe_stderr_from_n(sharpe, n),
        )

    @classmethod
    def from_stats(cls, *, name: str, n_trades: int, mean_sharpe: float,
                   stderr_sharpe: Optional[float] = None) -> "HypothesisPosterior":
        """Direct construction for callers that already have a Sharpe + n.

        Pass ``stderr_sharpe`` explicitly when the caller has a better
        estimate than the Lo (2002) closed form (e.g. bootstrap), else
        leave ``None`` and the function falls back to the closed form."""
        se = stderr_sharpe
        if se is None or not np.isfinite(se):
            se = _sharpe_stderr_from_n(mean_sharpe, n_trades)
        return cls(name=str(name), n_trades=int(n_trades),
                   mean_sharpe=float(mean_sharpe), stderr_sharpe=float(se))


def _sharpe_stderr_from_n(sharpe: float, n_trades: int) -> float:
    """Closed-form standard error of the Sharpe ratio (Lo 2002).

    ``SE(SR) ≈ sqrt((1 + 0.5 SR²) / n)``. Conservative because the Lo
    correction assumes iid normal returns; real trade returns are
    fatter-tailed so the true SE is bigger — which is the *safe*
    direction for the allocator (errs toward more exploration).
    """
    if n_trades <= 1:
        return 5.0   # huge posterior std — basically prior-dominated
    sr = float(sharpe)
    return float(math.sqrt(max(1e-9, (1.0 + 0.5 * sr * sr) / float(n_trades))))


@dataclass
class AllocationReport:
    """Result of one allocator run.

    ``weights`` is the recommended capital share per hypothesis. Sums to
    ≤ 1.0; the remainder (1 - sum) is the implied "cash" reserve — when
    no hypothesis sampled above the floor often enough to justify full
    deployment, the allocator deliberately leaves capital uninvested.
    """
    weights: Dict[str, float] = field(default_factory=dict)
    win_share: Dict[str, float] = field(default_factory=dict)
    posteriors: Dict[str, HypothesisPosterior] = field(default_factory=dict)
    n_draws: int = 0
    cash_share: float = 1.0
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "win_share": dict(self.win_share),
            "posteriors": {k: v.__dict__.copy() for k, v in self.posteriors.items()},
            "n_draws": int(self.n_draws),
            "cash_share": float(self.cash_share),
            "note": self.note,
        }


def thompson_allocate(posteriors: Sequence[HypothesisPosterior],
                      *,
                      n_draws: int = 4000,
                      seed: int = 19,
                      max_weight: float = DEFAULT_MAX_WEIGHT,
                      floor_weight: float = DEFAULT_FLOOR_WEIGHT,
                      sharpe_floor: float = SHARPE_FLOOR,
                      min_trades_to_allocate: int = MIN_TRADES_PRIOR_DOMINANT,
                      ) -> AllocationReport:
    """Allocate capital across hypotheses by Thompson-sampling each
    posterior and averaging the per-draw winner shares.

    Per draw:
      1. Sample one Sharpe per arm from N(mean, stderr²).
      2. Clip negatives to zero (``sharpe_floor=0`` means "no edge → no
         weight"; the option exists if a caller wants symmetry instead).
      3. Normalise to weights summing to 1; cap any single weight at
         ``max_weight`` and re-normalise; floor positive weights at
         ``floor_weight`` for exploration headroom.
    The reported ``weights`` are the mean of those per-draw weights;
    ``win_share`` is the fraction of draws each arm had the largest
    sampled Sharpe.

    Hypotheses with ``n_trades < min_trades_to_allocate`` participate in
    the draws but their posterior is widened to reflect the small
    sample. They do not get a special boost — exploration emerges from
    the wide posterior, not from a fixed exploration bonus.
    """
    if not posteriors:
        return AllocationReport(note="no hypotheses provided")
    rng = np.random.default_rng(seed)
    names = [p.name for p in posteriors]
    means = np.array([p.mean_sharpe for p in posteriors], dtype=float)
    stds = np.array([p.stderr_sharpe for p in posteriors], dtype=float)
    # Inflate posterior std for sample-poor hypotheses. The shrinkage
    # factor is 1 when n=0, 0 when n >= min_trades_to_allocate.
    shrink = np.array(
        [max(0.0, (min_trades_to_allocate - p.n_trades) / max(1, min_trades_to_allocate))
         for p in posteriors],
        dtype=float,
    )
    inflated_stds = stds * (1.0 + shrink)
    sum_weights = np.zeros(len(posteriors), dtype=float)
    win_counts = np.zeros(len(posteriors), dtype=int)
    for _ in range(int(n_draws)):
        draws = rng.normal(loc=means, scale=inflated_stds)
        if draws.max() > sharpe_floor:
            win_counts[int(np.argmax(draws))] += 1
        positive = np.clip(draws - sharpe_floor, 0.0, None)
        if not positive.sum():
            continue   # this draw contributes zero — uninvested
        raw = positive / positive.sum()
        # Cap any single weight; redistribute overflow to others
        # proportional to their pre-cap weights so the cap doesn't
        # silently inflate weak arms.
        capped = _cap_redistribute(raw, max_weight)
        # Floor: lift small positive weights up to floor_weight; pull
        # excess from the biggest arm. Floor stays optional.
        capped = _floor_redistribute(capped, floor_weight, max_weight)
        sum_weights += capped
    mean_weights = sum_weights / float(n_draws)
    cash_share = max(0.0, 1.0 - float(mean_weights.sum()))
    win_share = win_counts.astype(float) / float(n_draws)
    weights_dict = {n: float(w) for n, w in zip(names, mean_weights)}
    note = "ok"
    if cash_share > 0.5:
        note = (f"cash_share={cash_share:.0%} — over half of draws had no "
                f"positive sampled Sharpe; review hypotheses or lower "
                f"sharpe_floor before deploying")
    return AllocationReport(
        weights=weights_dict,
        win_share={n: float(s) for n, s in zip(names, win_share)},
        posteriors={p.name: p for p in posteriors},
        n_draws=int(n_draws),
        cash_share=cash_share,
        note=note,
    )


def _cap_redistribute(weights: np.ndarray, max_weight: float) -> np.ndarray:
    """If any weight exceeds ``max_weight``, cap it and redistribute the
    overflow to the other positive-weight arms in proportion."""
    if max_weight >= 1.0 or weights.size == 0:
        return weights.copy()
    out = weights.copy()
    for _ in range(weights.size):  # at most n iterations: each cap removes one degree of freedom
        over = out > max_weight
        if not over.any():
            break
        overflow = float(out[over].sum() - over.sum() * max_weight)
        out[over] = max_weight
        non_capped = out[~over]
        if non_capped.sum() <= 0:
            # No room to redistribute → leave the overflow uninvested.
            break
        out[~over] = non_capped + overflow * (non_capped / non_capped.sum())
    return out


def _floor_redistribute(weights: np.ndarray, floor_weight: float,
                         max_weight: float) -> np.ndarray:
    """Lift positive weights below ``floor_weight`` up to it; pay for
    that lift by trimming the largest weights down. Zero weights stay
    zero (the floor is exploration only for arms already allocated)."""
    if floor_weight <= 0.0 or weights.size == 0:
        return weights.copy()
    out = weights.copy()
    positive = out > 0.0
    below = positive & (out < floor_weight)
    if not below.any():
        return out
    needed = float((floor_weight - out[below]).sum())
    out[below] = floor_weight
    # Take the deficit out of the largest remaining weight, but never
    # below floor_weight itself.
    while needed > 1e-12:
        donor_idx = int(np.argmax(out))
        donor_room = float(out[donor_idx] - floor_weight)
        if donor_room <= 1e-12:
            break          # nobody has spare capacity above the floor
        take = min(donor_room, needed)
        out[donor_idx] -= take
        needed -= take
    return out


def allocate_from_reports(reports: Sequence[Any], **kwargs) -> AllocationReport:
    """Convenience wrapper: build posteriors from a list of
    ``HypothesisReport`` objects and run the allocator."""
    posteriors = [HypothesisPosterior.from_report(r) for r in reports]
    return thompson_allocate(posteriors, **kwargs)


def allocate_from_metrics(metrics: Sequence[Any], **kwargs) -> AllocationReport:
    """Convenience wrapper: build posteriors from a list of miner
    ``HypothesisMetrics`` objects and run the allocator."""
    posteriors = [HypothesisPosterior.from_metrics(m) for m in metrics]
    return thompson_allocate(posteriors, **kwargs)
