# Phase 4 Pool Availability Replay Audit

**Status:** PASS

This Workstream 2 audit checks whether saved liquidity pools are point-in-time
available before labels/trades consume them. It combines a full contributor
timestamp pass with a sampled detector replay truncated at `available_at`.

## Scope

- Model report: `output_models/core25_latest/multi_asset_report.pkl`
- Pool rows audited: 237,364
- Contributor rows audited: 3,696,717
- Replay samples per symbol: `1`
- Replay checked/misses/skipped: 25 / 0 / 0
- Issues: ERROR=0, WARN=0

## Availability Metadata By Pool Set

| Pool Set | Pools | Contributors | Missing Available | Known Before Source Close | Available Before Known | Available After Final Bar | Touch Before Available | Break Before Available |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| final | 51,622 | 862,641 | 0 | 0 | 0 | 0 | 0 | 0 |
| oos | 33,494 | 662,631 | 0 | 0 | 0 | 0 | 0 | 0 |
| train | 152,248 | 2,171,445 | 0 | 0 | 0 | 0 | 0 | 0 |

## Detector Replay By Symbol

| Symbol | Checked | Misses | Skipped |
| --- | ---: | ---: | ---: |
| AXISBANK | 1 | 0 | 0 |
| BAJAJ-AUTO | 1 | 0 | 0 |
| BRITANNIA | 1 | 0 | 0 |
| CIPLA | 1 | 0 | 0 |
| DABUR | 1 | 0 | 0 |
| DIVISLAB | 1 | 0 | 0 |
| DRREDDY | 1 | 0 | 0 |
| EICHERMOT | 1 | 0 | 0 |
| HCLTECH | 1 | 0 | 0 |
| HDFCBANK | 1 | 0 | 0 |
| HINDUNILVR | 1 | 0 | 0 |
| ICICIBANK | 1 | 0 | 0 |
| INFY | 1 | 0 | 0 |
| ITC | 1 | 0 | 0 |
| KOTAKBANK | 1 | 0 | 0 |
| LUPIN | 1 | 0 | 0 |
| M&M | 1 | 0 | 0 |
| MARUTI | 1 | 0 | 0 |
| NESTLEIND | 1 | 0 | 0 |
| SBIN | 1 | 0 | 0 |
| SUNPHARMA | 1 | 0 | 0 |
| TCS | 1 | 0 | 0 |
| TECHM | 1 | 0 | 0 |
| TVSMOTOR | 1 | 0 | 0 |
| WIPRO | 1 | 0 | 0 |

## Interpretation

No hard pool-availability leakage was found in the audited model report.
The sampled replay reproduced every checked pool from data truncated at
`available_at`.

Full replay of every Core25 pool remains intentionally disabled by default because
each replay re-runs the detector stack over truncated history. Increase
`--samples-per-symbol` for slower local or cloud runs.

## Issue Rows

Detailed rows were written to `reports/phase4_pool_availability_replay_issues.csv`.

## Next Step

With Q scale, fast probes, distance causality, MTF closed-bar semantics, and pool
availability replay covered, the next Phase 4 gate is the Track A proximity
distance-bucket AUC audit.
