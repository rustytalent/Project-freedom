# Phase 4 Branch A Q Percentile Policy Research

## Verdict

- Q percentile ranking improves several cohorts but does not create a positive existing-policy edge yet.
- This is a research report only. It does not change live gates.
- Existing policy outcomes are still from the v1 execution simulator, so this is not final live evidence.

## Inputs

- Q audit CSV: `output_core25_phase3d_train_fresh_may25/phase3_oos_prediction_audit.csv`
- Policy outcome labels: `output_core25_phase3d_train_fresh_may25/execution_policy_outcomes.parquet`
- Q column: `blended_q`
- Q observed range: 23.4% to 55.7%

## Best Existing Policy Cohort By Q Percentile

| mode | best_cohort | base_mean_R | best_mean_R | delta_R | base_PF | best_PF | trades |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blind_limit | top_1pct | -0.802 | -0.583 | 0.219 | 0.346 | 0.459 | 190 |
| displacement_confirmed | top_1pct | -0.515 | -0.408 | 0.107 | 0.396 | 0.462 | 55 |
| reclaim_confirmed | top_1pct | -1.046 | -0.604 | 0.442 | 0.325 | 0.545 | 60 |
| touch_confirmed | top_1pct | -0.553 | -0.448 | 0.106 | 0.350 | 0.413 | 157 |

## Full Cohort Table

| mode | cohort | pools | trades | trade% | win | mean_R | median_R | PF | actual | mean_Q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blind_limit | all | 33494 | 23116 | 69.0% | 26.7% | -0.802 | -1.568 | 0.346 | 43.5% | 43.5% |
| blind_limit | top_50pct | 16747 | 11031 | 65.9% | 27.0% | -0.790 | -1.542 | 0.340 | 45.3% | 44.8% |
| blind_limit | top_25pct | 8374 | 5340 | 63.8% | 28.3% | -0.773 | -1.519 | 0.338 | 47.2% | 45.6% |
| blind_limit | top_10pct | 3350 | 2058 | 61.4% | 29.3% | -0.754 | -1.531 | 0.357 | 48.9% | 45.9% |
| blind_limit | top_5pct | 1675 | 1016 | 60.7% | 27.6% | -0.811 | -1.542 | 0.320 | 49.8% | 46.2% |
| blind_limit | top_1pct | 335 | 190 | 56.7% | 34.2% | -0.583 | -1.485 | 0.459 | 55.4% | 47.1% |
| displacement_confirmed | all | 33494 | 7591 | 22.7% | 39.2% | -0.515 | -1.290 | 0.396 | 43.5% | 43.5% |
| displacement_confirmed | top_50pct | 16747 | 3412 | 20.4% | 39.3% | -0.512 | -1.269 | 0.383 | 45.3% | 44.8% |
| displacement_confirmed | top_25pct | 8374 | 1506 | 18.0% | 40.2% | -0.496 | -1.253 | 0.389 | 47.2% | 45.6% |
| displacement_confirmed | top_10pct | 3350 | 599 | 17.9% | 40.9% | -0.489 | -1.257 | 0.403 | 48.9% | 45.9% |
| displacement_confirmed | top_5pct | 1675 | 301 | 18.0% | 42.5% | -0.416 | -1.077 | 0.454 | 49.8% | 46.2% |
| displacement_confirmed | top_1pct | 335 | 55 | 16.4% | 41.8% | -0.408 | -1.077 | 0.462 | 55.4% | 47.1% |
| reclaim_confirmed | all | 33494 | 7519 | 22.4% | 21.3% | -1.046 | -1.793 | 0.325 | 43.5% | 43.5% |
| reclaim_confirmed | top_50pct | 16747 | 3707 | 22.1% | 21.1% | -1.046 | -1.778 | 0.323 | 45.3% | 44.8% |
| reclaim_confirmed | top_25pct | 8374 | 1805 | 21.6% | 20.1% | -1.114 | -1.793 | 0.299 | 47.2% | 45.6% |
| reclaim_confirmed | top_10pct | 3350 | 689 | 20.6% | 21.6% | -1.034 | -1.788 | 0.341 | 48.9% | 45.9% |
| reclaim_confirmed | top_5pct | 1675 | 353 | 21.1% | 23.2% | -0.939 | -1.766 | 0.386 | 49.8% | 46.2% |
| reclaim_confirmed | top_1pct | 335 | 60 | 17.9% | 28.3% | -0.604 | -1.672 | 0.545 | 55.4% | 47.1% |
| touch_confirmed | all | 33494 | 19267 | 57.5% | 37.6% | -0.553 | -1.286 | 0.350 | 43.5% | 43.5% |
| touch_confirmed | top_50pct | 16747 | 9046 | 54.0% | 38.1% | -0.541 | -1.263 | 0.346 | 45.3% | 44.8% |
| touch_confirmed | top_25pct | 8374 | 4294 | 51.3% | 39.4% | -0.525 | -1.227 | 0.344 | 47.2% | 45.6% |
| touch_confirmed | top_10pct | 3350 | 1683 | 50.2% | 40.0% | -0.512 | -1.193 | 0.358 | 48.9% | 45.9% |
| touch_confirmed | top_5pct | 1675 | 833 | 49.7% | 37.8% | -0.546 | -1.230 | 0.334 | 49.8% | 46.2% |
| touch_confirmed | top_1pct | 335 | 157 | 46.9% | 41.4% | -0.448 | -0.605 | 0.413 | 55.4% | 47.1% |

## Interpretation

- Branch A was correct: absolute Q scale is compressed and rank signal exists.
- However, filtering existing blind/touch/reclaim/displacement policies by top-Q percentile does not by itself make those existing policies profitable after costs.
- The strongest improvement appears in the rare top 1% cohorts, but sample sizes are small and still need leakage checks, 1-minute execution v2, and DSR before trust.
- Next engineering path should continue with leakage probes and execution simulator v2 before any live gate rewrite.
