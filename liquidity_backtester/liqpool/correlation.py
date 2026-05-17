"""Factor / TF correlation analysis.

After a backtest run, we want to know not just "which weight was high", but **which factors
and which timeframes actually drive respected pools, and which combinations work together**.

This module:
  - For each detector family (side-collapsed: EQHL, FVG, OB, REJ, PD, PW, PM, HVN, ORB, SWING)
    computes the conditional respect rate of pools containing that family.
  - Does the same for every TF and for every TF-count bucket (1 TF, 2 TFs, ...).
  - Enumerates co-occurring factor pairs and triplets, again with conditional stats.
  - Reports the "lift" over the baseline respect rate so it's clear which combos are *actually*
    additive vs just popular.
  - Provides `filter_by_required` so you can re-test the pool set restricted to the top combo.

This is what lets the optimizer's weight choices be cross-checked: high weight on a factor that
shows low conditional lift is a red flag (the optimizer found a noisy local optimum).
"""
from __future__ import annotations
from itertools import combinations
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Iterable

from .pools import Pool
from .tester import PoolResult


# Collapse "EQH"/"EQL" → "EQHL" etc., so the analysis is about WHAT TYPE of feature contributed,
# not which side of the chart it was on. A pool only contains features of its own side.
_FAMILY = {
    "SWING_H": "SWING", "SWING_L": "SWING",
    "EQH": "EQHL", "EQL": "EQHL",
    "PDH": "PD", "PDL": "PD",
    "PWH": "PW", "PWL": "PW",
    "PMH": "PM", "PML": "PM",
    "FVG_bull": "FVG", "FVG_bear": "FVG",
    "OB_bull": "OB", "OB_bear": "OB",
    "REJ_high": "REJ", "REJ_low": "REJ",
    "HVN": "HVN",
    "ORB_H": "ORB", "ORB_L": "ORB",
}


def pool_factors(pool: Pool) -> Set[str]:
    return {_FAMILY.get(c.source.split("@", 1)[0], c.source.split("@", 1)[0])
            for c in pool.contributors}


def _stats(pairs: List[Tuple[Pool, PoolResult]]) -> Dict:
    """Returns common metrics: counts, respect, break, touch and untested rates."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "tested": 0, "respect": 0.0, "break": 0.0, "untested_rate": 0.0}
    eligible = [(p, r) for p, r in pairs if r.outcome != "horizon_insufficient"]
    touched = [(p, r) for p, r in eligible if r.outcome in ("respected", "broken")]
    if not touched:
        return {"n": n, "tested": 0, "respect": 0.0, "break": 0.0,
                "untested_rate": 1.0 - len(touched) / max(len(eligible), 1)}
    respected = sum(1 for _, r in touched if r.outcome == "respected")
    broken = sum(1 for _, r in touched if r.outcome == "broken")
    return {
        "n": n,
        "tested": len(touched),
        "respect": respected / len(touched),
        "break": broken / len(touched),
        "untested_rate": 1.0 - len(touched) / max(len(eligible), 1),
    }


def _lift(rate: float, baseline: float) -> float:
    if baseline <= 0:
        return 0.0
    return rate / baseline - 1.0


def analyze(pools: List[Pool], results: List[PoolResult],
            min_sample: int = 10, top_k: int = 12) -> Dict:
    if not pools or not results:
        return {}
    paired = list(zip(pools, results))
    base = _stats(paired)
    baseline = base["respect"]

    # Per-factor family
    families = sorted({f for p in pools for f in pool_factors(p)})
    singles = []
    for f in families:
        match = [(p, r) for p, r in paired if f in pool_factors(p)]
        s = _stats(match)
        if s["tested"] >= min_sample:
            s.update({"factor": f, "lift": _lift(s["respect"], baseline)})
            singles.append(s)

    # Per-pair (only pairs that actually co-occur somewhere)
    seen_pairs: Set[Tuple[str, str]] = set()
    for p, _ in paired:
        pf = pool_factors(p)
        for a, b in combinations(sorted(pf), 2):
            seen_pairs.add((a, b))
    pairs = []
    for a, b in seen_pairs:
        match = [(p, r) for p, r in paired if {a, b}.issubset(pool_factors(p))]
        s = _stats(match)
        if s["tested"] >= min_sample:
            s.update({"factors": (a, b), "lift": _lift(s["respect"], baseline)})
            pairs.append(s)

    # Per-triplet
    seen_trip: Set[Tuple[str, str, str]] = set()
    for p, _ in paired:
        pf = pool_factors(p)
        if len(pf) >= 3:
            for a, b, c in combinations(sorted(pf), 3):
                seen_trip.add((a, b, c))
    triplets = []
    for a, b, c in seen_trip:
        match = [(p, r) for p, r in paired if {a, b, c}.issubset(pool_factors(p))]
        s = _stats(match)
        if s["tested"] >= min_sample:
            s.update({"factors": (a, b, c), "lift": _lift(s["respect"], baseline)})
            triplets.append(s)

    # Per-TF
    all_tfs = sorted({t for p in pools for t in p.tfs})
    tfs = []
    for tf in all_tfs:
        match = [(p, r) for p, r in paired if tf in p.tfs]
        s = _stats(match)
        if s["tested"] >= min_sample:
            s.update({"tf": tf, "lift": _lift(s["respect"], baseline)})
            tfs.append(s)

    # TF-count: respect rate vs how many distinct TFs confirmed
    groups = defaultdict(list)
    for p, r in paired:
        groups[len(set(p.tfs))].append((p, r))
    tf_count_stats = []
    for k in sorted(groups):
        s = _stats(groups[k])
        s.update({"n_tfs": k, "lift": _lift(s["respect"], baseline)})
        tf_count_stats.append(s)

    return {
        "baseline_respect": baseline,
        "n_pools": base["n"],
        "n_tested": base["tested"],
        "singles": sorted(singles, key=lambda x: -x["respect"])[:top_k],
        "pairs": sorted(pairs, key=lambda x: -x["respect"])[:top_k],
        "triplets": sorted(triplets, key=lambda x: -x["respect"])[:top_k],
        "tfs": sorted(tfs, key=lambda x: -x["respect"]),
        "tf_count_stats": tf_count_stats,
    }


def filter_by_required(pools: List[Pool], results: List[PoolResult],
                       required_factors: Iterable[str] = (),
                       required_tfs: Iterable[str] = (),
                       min_distinct_tfs: int = 1
                       ) -> Tuple[List[Pool], List[PoolResult]]:
    """Return the subset of (pool, result) where ALL `required_factors` are present AND each of
    `required_tfs` is among the contributing TFs AND the pool spans `>= min_distinct_tfs` TFs."""
    req_f = set(required_factors)
    req_tf = set(required_tfs)
    out_p, out_r = [], []
    for p, r in zip(pools, results):
        if not req_f.issubset(pool_factors(p)):
            continue
        if not req_tf.issubset(set(p.tfs)):
            continue
        if len(set(p.tfs)) < min_distinct_tfs:
            continue
        # Reindex pool_idx so downstream code that uses it as a key remains consistent.
        out_p.append(p)
        out_r.append(r)
    return out_p, out_r


def print_report(report: Dict, file=None) -> None:
    if not report:
        print("(no pools / no results)", file=file)
        return
    base = report.get("baseline_respect", 0.0)
    print(f"\n=== correlation report  (baseline respect = {base:.1%}, "
          f"pools={report['n_pools']}, tested={report['n_tested']}) ===", file=file)

    print("\n[per factor family]   factor → respect / lift vs baseline", file=file)
    print(f"  {'factor':<8} {'n':>5} {'tested':>7} {'respect':>9} {'lift':>9}", file=file)
    for s in report["singles"]:
        print(f"  {s['factor']:<8} {s['n']:>5} {s['tested']:>7} "
              f"{s['respect']:>8.1%} {s['lift']:>+8.1%}", file=file)

    print("\n[per timeframe]", file=file)
    print(f"  {'tf':<8} {'n':>5} {'tested':>7} {'respect':>9} {'lift':>9}", file=file)
    for s in report["tfs"]:
        print(f"  {s['tf']:<8} {s['n']:>5} {s['tested']:>7} "
              f"{s['respect']:>8.1%} {s['lift']:>+8.1%}", file=file)

    print("\n[multi-TF confluence: respect rate vs # of distinct TFs in a pool]", file=file)
    print(f"  {'n_tfs':>5} {'n':>5} {'tested':>7} {'respect':>9} {'lift':>9}", file=file)
    for s in report["tf_count_stats"]:
        print(f"  {s['n_tfs']:>5} {s['n']:>5} {s['tested']:>7} "
              f"{s['respect']:>8.1%} {s['lift']:>+8.1%}", file=file)

    print("\n[top factor pairs by conditional respect rate]", file=file)
    print(f"  {'pair':<20} {'n':>5} {'tested':>7} {'respect':>9} {'lift':>9}", file=file)
    for s in report["pairs"]:
        a, b = s["factors"]
        print(f"  {a+'+'+b:<20} {s['n']:>5} {s['tested']:>7} "
              f"{s['respect']:>8.1%} {s['lift']:>+8.1%}", file=file)

    print("\n[top factor triplets by conditional respect rate]", file=file)
    print(f"  {'triplet':<28} {'n':>5} {'tested':>7} {'respect':>9} {'lift':>9}", file=file)
    for s in report["triplets"]:
        a, b, c = s["factors"]
        print(f"  {a+'+'+b+'+'+c:<28} {s['n']:>5} {s['tested']:>7} "
              f"{s['respect']:>8.1%} {s['lift']:>+8.1%}", file=file)
