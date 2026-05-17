"""Stratified out-of-sample probability model for pools.

Differentiated per-pool P(respect) without ML. We bucket OOS pools by (n_TFs_bucket, headline
factor) and store each bucket's respect rate (with Wilson 90% CI). At predict time the pool is
mapped to its bucket; if the bucket has too few samples we fall back to a coarser bucket
(`n_TFs` only) and finally to the overall OOS rate. This solves the "every pool shows the same
probability" problem from the aggregate-only reporting.

Note on broad vs strict rates: we report both. The broad rate counts all respects (weak+strong)
vs all breaks; the strict rate uses only the decisive tier (respected_strong vs broken_strong).
For per-pool predictions we expose both so the user can see how clean the signal is.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Iterable
from collections import defaultdict

from .pools import Pool
from .tester import PoolResult
from .correlation import pool_factors
from .stats import wilson_score_interval


def _tf_bucket(n: int) -> str:
    if n <= 1: return "1"
    if n == 2: return "2"
    if n == 3: return "3"
    return "4+"


# Priority order for picking a single "headline" factor for bucketing. We bucket by ONE factor
# (not all factors) to keep the bucket-space small enough that each bucket gets samples; the
# priority order picks the most informative factor present in the pool.
_FACTOR_PRIORITY = ["EQHL", "OB", "FVG", "PM", "PW", "REJ", "PD", "ORB", "HVN", "SWING"]


def _headline_factor(pool: Pool) -> str:
    pf = pool_factors(pool)
    for f in _FACTOR_PRIORITY:
        if f in pf:
            return f
    return "OTHER"


@dataclass
class BucketStat:
    n: int                  # eligible pools in bucket
    tested: int             # pools with a respect-or-break outcome
    respected: int          # respected_strong + respected_weak
    decisive_total: int     # respected_strong + broken_strong (denominator for strict rate)
    decisive_respected: int # respected_strong
    rate_broad: float
    rate_strict: float
    ci_low: float
    ci_high: float


@dataclass
class StratifiedRespectModel:
    """OOS-calibrated lookup model for P(respect | pool features).

    Fit on (pool, result) pairs gathered from walk-forward OOS slices, where each pool was
    evaluated under a config the optimiser tuned WITHOUT seeing that pool's outcome. Predict
    falls back through bucket hierarchy when a finer bucket lacks samples.
    """
    overall: BucketStat
    by_tf: Dict[str, BucketStat] = field(default_factory=dict)
    by_tf_factor: Dict[Tuple[str, str], BucketStat] = field(default_factory=dict)
    min_sample: int = 8

    @staticmethod
    def _bucket_stats(items: Iterable[Tuple[Pool, PoolResult]], ci: float = 0.90) -> BucketStat:
        items = list(items)
        n = len(items)
        tested = [r for _, r in items if r.is_tested]
        respected = [r for r in tested if r.is_respect]
        decisive = [r for r in tested if r.is_decisive]
        decisive_resp = [r for r in decisive if r.is_respect]
        broad = (len(respected) / len(tested)) if tested else 0.0
        strict = (len(decisive_resp) / len(decisive)) if decisive else 0.0
        lo, hi = wilson_score_interval(len(respected), len(tested), ci=ci)
        return BucketStat(
            n=n, tested=len(tested), respected=len(respected),
            decisive_total=len(decisive), decisive_respected=len(decisive_resp),
            rate_broad=broad, rate_strict=strict,
            ci_low=lo, ci_high=hi,
        )

    @classmethod
    def fit(cls, pools: List[Pool], results: List[PoolResult],
            min_sample: int = 8) -> "StratifiedRespectModel":
        if len(pools) != len(results):
            raise ValueError(f"pools/results length mismatch: {len(pools)} vs {len(results)}")
        paired = list(zip(pools, results))
        overall = cls._bucket_stats(paired)

        by_tf: Dict[str, list] = defaultdict(list)
        by_tf_factor: Dict[Tuple[str, str], list] = defaultdict(list)
        for p, r in paired:
            tfb = _tf_bucket(len(set(p.tfs)))
            f = _headline_factor(p)
            by_tf[tfb].append((p, r))
            by_tf_factor[(tfb, f)].append((p, r))

        return cls(
            overall=overall,
            by_tf={k: cls._bucket_stats(v) for k, v in by_tf.items()},
            by_tf_factor={k: cls._bucket_stats(v) for k, v in by_tf_factor.items()},
            min_sample=min_sample,
        )

    def predict(self, pool: Pool) -> Tuple[BucketStat, str]:
        """Return (BucketStat, source_label) for `pool`, falling back through coarser buckets."""
        tfb = _tf_bucket(len(set(pool.tfs)))
        f = _headline_factor(pool)
        fine = self.by_tf_factor.get((tfb, f))
        if fine is not None and fine.tested >= self.min_sample:
            return fine, f"TFs={tfb} & {f}"
        coarse = self.by_tf.get(tfb)
        if coarse is not None and coarse.tested >= self.min_sample:
            return coarse, f"TFs={tfb}"
        return self.overall, "overall"


def print_model(model: StratifiedRespectModel, file=None) -> None:
    print("\n=== Stratified OOS Probability Model ===", file=file)
    print(f"  overall: n={model.overall.n} tested={model.overall.tested}  "
          f"broad={model.overall.rate_broad:.1%}  strict={model.overall.rate_strict:.1%}  "
          f"CI={model.overall.ci_low:.1%}-{model.overall.ci_high:.1%}", file=file)
    print(f"\n[by TF count]", file=file)
    print(f"  {'TFs':<5} {'n':>5} {'tested':>7} {'broad':>8} {'strict':>8} {'90% CI':>16}", file=file)
    for k in sorted(model.by_tf, key=lambda x: (x.endswith("+"), x)):
        s = model.by_tf[k]
        ci = f"{s.ci_low*100:.0f}-{s.ci_high*100:.0f}%"
        print(f"  {k:<5} {s.n:>5} {s.tested:>7} {s.rate_broad:>7.1%} "
              f"{s.rate_strict:>7.1%} {ci:>16}", file=file)
    print(f"\n[by TF count × headline factor]  (only buckets with tested >= {model.min_sample})",
          file=file)
    print(f"  {'TFs':<5} {'factor':<8} {'n':>5} {'tested':>7} {'broad':>8} {'strict':>8} {'90% CI':>16}",
          file=file)
    sig_rows = sorted(model.by_tf_factor.items(),
                      key=lambda kv: -kv[1].rate_broad)
    for (tfb, f), s in sig_rows:
        if s.tested < model.min_sample:
            continue
        ci = f"{s.ci_low*100:.0f}-{s.ci_high*100:.0f}%"
        print(f"  {tfb:<5} {f:<8} {s.n:>5} {s.tested:>7} {s.rate_broad:>7.1%} "
              f"{s.rate_strict:>7.1%} {ci:>16}", file=file)
