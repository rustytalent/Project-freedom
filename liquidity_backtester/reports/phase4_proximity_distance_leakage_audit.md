# Phase 4 Proximity Distance Leakage Audit

**Status:** PASS

This Workstream 2 audit checks whether persisted proximity `distance_atr` rows can be
recomputed from the decision-time 5m close, trailing ATR(14), and the referenced pool
zone. This is the critical causality probe for Track A because `distance_atr` is the
dominant proximity feature.

## Scope

- Feature store: `output_feature_store/core25_fresh_may25`
- Split: `oos`
- Horizons: `78, 156, 312`
- Tolerance: `1e-09`
- Files checked: 375
- Rows checked: 7,673,506 / 7,673,506
- Sampling: Audited every row in the selected proximity shards.

## Result

- Distance mismatches: 0
- Side mismatches: 0
- Rows already inside pool zone: 0
- Invalid bar/pool indexes: 0
- Max absolute distance error: `0.000e+00`

## By Horizon

| Horizon | Files | Rows Checked | Distance Mismatches | Side Mismatches | Inside-Zone Rows | Max Abs Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 78 | 125 | 2,572,441 | 0 | 0 | 0 | 0.000e+00 |
| 156 | 125 | 2,560,144 | 0 | 0 | 0 | 0.000e+00 |
| 312 | 125 | 2,540,921 | 0 | 0 | 0 | 0.000e+00 |

## Pool Index Semantics

Proximity shards store `pool_idx` against the in-memory `train_pools + oos_pools` list,
not the local pool index inside `pools/symbol=<SYMBOL>/oos.parquet`. The audit recreates
that combined table before recomputing distances. This is now documented because using
the local OOS pool index creates false mismatch alarms.

## Interpretation

The persisted proximity `distance_atr` feature is reproducible from decision-time
inputs for the audited artifacts. This does **not** prove the full proximity model
is leakage-free; it specifically clears the highest-priority distance feature probe.

## Issue Rows

No issue rows were found; `reports/phase4_proximity_distance_leakage_issues.csv` contains headers only.

## Next Step

Continue Workstream 2 with broader artifact-backed probes: MTF closed-bar replay and
pool availability replay. Track A should still wait for those checks plus the distance
bucket AUC gate.
