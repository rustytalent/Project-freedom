# Phase 4 Multi-Pocket Mini-Sweeps

## Verdict

- Decision: `ECONOMIC_SURVIVORS_NEED_CPCV`
- 3 deduplicated economic survivor cells found, but none clear DSR.
- Cells tested: `1944`
- DSR trials used: `2000`
- Minimum trades per cell: `100`
- This is Option B: reuse existing v2 sweep trades; no new 1-minute simulation was run.
- Because the source sweep admitted candidates around `T>=0.75` and `D>=0.55`, this run cannot test looser entry gates such as `T=0.70` from scratch.
- Survivors here still need CPCV and synthetic random/ATR nulls before paper trading.

## Pocket Family Verdicts

| family | verdict | cells | econ | strict | dedupe | best_n | best_R | best_PF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| normal_vol_5_8_long | FRAGILE_NEEDS_CPCV | 432 | 83 | 0 | 1 | 175 | 1.112 | 3.978 |
| base_5_8_long | FRAGILE_NEEDS_CPCV | 432 | 80 | 0 | 1 | 228 | 0.686 | 2.308 |
| eqhl_5_8_long | FRAGILE_NEEDS_CPCV | 432 | 27 | 0 | 1 | 111 | 0.958 | 3.305 |
| midday_5_8_long | DEAD | 216 | 0 | 0 | 0 | 10 | 3.553 | 23.428 |
| morning_5_8_long | DEAD | 432 | 0 | 0 | 0 | 8 | 2.622 | 18.725 |

## Deduplicated Economic Survivors

| family | T | D | tf | sl | h | n | win | mean_R | PF | cost1.5x | 1st/2nd | temp | pos_sec | ci_low | DSR | sel_ov | dedupe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| normal_vol_5_8_long | 0.75 | 0.65 | 1.0 | 1.5 | 60 | 175 | 68.6% | 1.112 | 3.978 | 0.889 | 1.097/1.138 | 0.96 | 4 | 0.836 | 0.0% | 0.638 | True |
| eqhl_5_8_long | 0.75 | 0.55 | 1.0 | 1.5 | 60 | 111 | 65.8% | 0.958 | 3.305 | 0.702 | 0.985/0.916 | 0.93 | 3 | 0.614 | 0.0% | 0.329 | True |
| base_5_8_long | 0.75 | 0.70 | 1.0 | 1.5 | 60 | 228 | 56.1% | 0.686 | 2.308 | 0.464 | 0.663/0.723 | 0.91 | 3 | 0.439 | 0.0% | 0.638 | True |

## Top Economic Survivors Before Deduplication

| family | T | D | tf | sl | h | n | win | mean_R | PF | cost1.5x | 1st/2nd | temp | pos_sec | ci_low | DSR | sel_ov | dedupe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| normal_vol_5_8_long | 0.75 | 0.65 | 1.0 | 1.5 | 60 | 175 | 68.6% | 1.112 | 3.978 | 0.889 | 1.097/1.138 | 0.96 | 4 | 0.836 | 0.0% | 0.638 | True |
| normal_vol_5_8_long | 0.75 | 0.60 | 1.0 | 1.5 | 60 | 176 | 68.2% | 1.105 | 3.970 | 0.881 | 1.086/1.138 | 0.95 | 4 | 0.830 | 0.0% | 0.994 | False |
| normal_vol_5_8_long | 0.75 | 0.65 | 1.0 | 1.5 | 36 | 175 | 69.7% | 1.054 | 4.218 | 0.834 | 1.015/1.121 | 0.90 | 4 | 0.790 | 0.0% | 1.000 | False |
| normal_vol_5_8_long | 0.75 | 0.60 | 1.0 | 1.5 | 36 | 176 | 69.3% | 1.047 | 4.207 | 0.826 | 1.005/1.121 | 0.89 | 4 | 0.785 | 0.0% | 0.994 | False |
| normal_vol_5_8_long | 0.75 | 0.65 | 1.0 | 1.0 | 60 | 175 | 49.1% | 1.036 | 2.360 | 0.718 | 1.036/1.036 | 1.00 | 4 | 0.616 | 0.0% | 1.000 | False |
| normal_vol_5_8_long | 0.75 | 0.60 | 1.0 | 1.0 | 60 | 176 | 48.9% | 1.018 | 2.323 | 0.698 | 1.008/1.036 | 0.97 | 4 | 0.599 | 0.0% | 0.994 | False |
| normal_vol_5_8_long | 0.75 | 0.70 | 0.8 | 1.5 | 60 | 157 | 68.8% | 0.991 | 3.721 | 0.777 | 0.917/1.126 | 0.79 | 3 | 0.733 | 0.0% | 0.897 | False |
| normal_vol_5_8_long | 0.75 | 0.55 | 1.0 | 1.5 | 60 | 199 | 65.8% | 0.984 | 3.455 | 0.754 | 1.070/0.862 | 0.79 | 4 | 0.729 | 0.0% | 0.879 | False |
| eqhl_5_8_long | 0.75 | 0.55 | 1.0 | 1.5 | 60 | 111 | 65.8% | 0.958 | 3.305 | 0.702 | 0.985/0.916 | 0.93 | 3 | 0.614 | 0.0% | 0.329 | True |
| normal_vol_5_8_long | 0.75 | 0.65 | 1.0 | 1.0 | 36 | 175 | 49.7% | 0.951 | 2.273 | 0.635 | 0.933/0.981 | 0.95 | 5 | 0.545 | 0.0% | 1.000 | False |
| eqhl_5_8_long | 0.75 | 0.55 | 1.0 | 1.0 | 60 | 111 | 49.5% | 0.943 | 2.214 | 0.580 | 0.875/1.052 | 0.81 | 3 | 0.429 | 0.0% | 1.000 | False |
| normal_vol_5_8_long | 0.75 | 0.60 | 1.0 | 1.0 | 36 | 176 | 49.4% | 0.933 | 2.237 | 0.615 | 0.906/0.981 | 0.92 | 4 | 0.528 | 0.0% | 0.994 | False |
| normal_vol_5_8_long | 0.75 | 0.65 | 0.8 | 1.5 | 60 | 175 | 68.6% | 0.933 | 3.500 | 0.712 | 0.953/0.898 | 0.94 | 4 | 0.691 | 0.0% | 1.000 | False |
| normal_vol_5_8_long | 0.75 | 0.55 | 1.0 | 1.5 | 36 | 199 | 66.8% | 0.928 | 3.573 | 0.700 | 0.983/0.848 | 0.85 | 4 | 0.683 | 0.0% | 0.879 | False |
| normal_vol_5_8_long | 0.75 | 0.60 | 0.8 | 1.5 | 60 | 176 | 68.2% | 0.927 | 3.492 | 0.704 | 0.943/0.898 | 0.95 | 4 | 0.686 | 0.0% | 0.994 | False |
| eqhl_5_8_long | 0.75 | 0.55 | 1.0 | 1.5 | 36 | 111 | 65.8% | 0.900 | 3.416 | 0.646 | 0.910/0.885 | 0.97 | 3 | 0.574 | 0.0% | 1.000 | False |
| normal_vol_5_8_long | 0.75 | 0.65 | 0.8 | 1.5 | 36 | 175 | 69.7% | 0.886 | 3.705 | 0.666 | 0.871/0.912 | 0.95 | 4 | 0.655 | 0.0% | 1.000 | False |
| normal_vol_5_8_long | 0.75 | 0.70 | 1.0 | 2.0 | 60 | 157 | 69.4% | 0.884 | 4.119 | 0.719 | 0.798/1.038 | 0.73 | 3 | 0.657 | 0.0% | 0.897 | False |
| normal_vol_5_8_long | 0.75 | 0.60 | 0.8 | 1.5 | 36 | 176 | 69.3% | 0.880 | 3.696 | 0.659 | 0.861/0.912 | 0.94 | 4 | 0.650 | 0.0% | 0.994 | False |
| normal_vol_5_8_long | 0.75 | 0.65 | 1.0 | 1.5 | 24 | 175 | 68.0% | 0.869 | 3.586 | 0.650 | 0.778/1.027 | 0.71 | 4 | 0.621 | 0.0% | 1.000 | False |
| normal_vol_5_8_long | 0.75 | 0.60 | 1.0 | 1.5 | 24 | 176 | 67.6% | 0.863 | 3.577 | 0.642 | 0.770/1.027 | 0.70 | 4 | 0.617 | 0.0% | 0.994 | False |
| eqhl_5_8_long | 0.75 | 0.55 | 1.0 | 1.0 | 36 | 111 | 49.5% | 0.846 | 2.105 | 0.484 | 0.773/0.960 | 0.78 | 3 | 0.351 | 0.0% | 1.000 | False |
| normal_vol_5_8_long | 0.75 | 0.65 | 1.0 | 2.0 | 60 | 175 | 69.1% | 0.836 | 3.820 | 0.665 | 0.843/0.824 | 0.98 | 4 | 0.622 | 0.0% | 1.000 | False |
| normal_vol_5_8_long | 0.75 | 0.60 | 1.0 | 2.0 | 60 | 176 | 68.8% | 0.830 | 3.812 | 0.658 | 0.834/0.824 | 0.99 | 4 | 0.617 | 0.0% | 0.994 | False |
| eqhl_5_8_long | 0.75 | 0.55 | 0.8 | 1.5 | 60 | 111 | 67.6% | 0.818 | 2.973 | 0.564 | 0.880/0.721 | 0.81 | 3 | 0.515 | 0.0% | 1.000 | False |

## Top Cells By Mean R

| family | T | D | tf | sl | h | n | win | mean_R | PF | cost1.5x | 1st/2nd | temp | pos_sec | ci_low | DSR | sel_ov | dedupe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| midday_5_8_long | 0.75 | 0.60 | 1.0 | 1.0 | 60 | 10 | 90.0% | 3.553 | 23.428 | 3.259 | 3.185/3.594 | 0.89 | 0 | 2.055 | 0.2% | 0.061 | False |
| midday_5_8_long | 0.75 | 0.65 | 1.0 | 1.0 | 60 | 10 | 90.0% | 3.553 | 23.428 | 3.259 | 3.185/3.594 | 0.89 | 0 | 2.055 | 0.2% | 0.061 | False |
| midday_5_8_long | 0.75 | 0.70 | 1.0 | 1.0 | 60 | 9 | 88.9% | 3.388 | 20.248 | 3.108 | 3.185/3.413 | 0.93 | 0 | 1.753 | 0.1% | 0.053 | False |
| midday_5_8_long | 0.75 | 0.60 | 1.0 | 1.0 | 36 | 10 | 90.0% | 3.073 | 20.397 | 2.779 | 3.185/3.060 | 0.96 | 0 | 1.526 | 0.0% | 0.061 | False |
| midday_5_8_long | 0.75 | 0.65 | 1.0 | 1.0 | 36 | 10 | 90.0% | 3.073 | 20.397 | 2.779 | 3.185/3.060 | 0.96 | 0 | 1.526 | 0.0% | 0.061 | False |
| midday_5_8_long | 0.75 | 0.70 | 1.0 | 1.0 | 36 | 9 | 88.9% | 2.854 | 17.217 | 2.575 | 3.185/2.813 | 0.87 | 0 | 1.193 | 0.0% | 0.053 | False |
| midday_5_8_long | 0.75 | 0.60 | 0.8 | 1.0 | 60 | 10 | 90.0% | 2.788 | 18.599 | 2.494 | 2.368/2.835 | 0.83 | 0 | 1.599 | 0.4% | 0.061 | False |
| midday_5_8_long | 0.75 | 0.65 | 0.8 | 1.0 | 60 | 10 | 90.0% | 2.788 | 18.599 | 2.494 | 2.368/2.835 | 0.83 | 0 | 1.599 | 0.4% | 0.061 | False |
| midday_5_8_long | 0.75 | 0.60 | 1.0 | 1.0 | 24 | 10 | 100.0% | 2.722 | inf | 2.435 | 3.185/2.671 | 0.81 | 0 | 1.341 | 0.0% | 0.061 | False |
| midday_5_8_long | 0.75 | 0.65 | 1.0 | 1.0 | 24 | 10 | 100.0% | 2.722 | inf | 2.435 | 3.185/2.671 | 0.81 | 0 | 1.341 | 0.0% | 0.061 | False |
| midday_5_8_long | 0.75 | 0.70 | 0.8 | 1.0 | 60 | 9 | 88.9% | 2.669 | 16.161 | 2.389 | 2.368/2.706 | 0.87 | 0 | 1.365 | 0.2% | 0.053 | False |
| morning_5_8_long | 0.85 | 0.55 | 1.0 | 1.5 | 60 | 8 | 87.5% | 2.622 | 18.725 | 2.426 | 0.858/2.874 | 0.23 | 0 | 1.315 | 0.6% | 0.040 | False |
| morning_5_8_long | 0.85 | 0.60 | 1.0 | 1.5 | 60 | 8 | 87.5% | 2.622 | 18.725 | 2.426 | 0.858/2.874 | 0.23 | 0 | 1.315 | 0.6% | 0.040 | False |
| morning_5_8_long | 0.85 | 0.65 | 1.0 | 1.5 | 60 | 8 | 87.5% | 2.622 | 18.725 | 2.426 | 0.858/2.874 | 0.23 | 0 | 1.315 | 0.6% | 0.040 | False |
| morning_5_8_long | 0.85 | 0.70 | 1.0 | 1.5 | 60 | 8 | 87.5% | 2.622 | 18.725 | 2.426 | 0.858/2.874 | 0.23 | 0 | 1.315 | 0.6% | 0.040 | False |
| midday_5_8_long | 0.75 | 0.70 | 1.0 | 1.0 | 24 | 9 | 100.0% | 2.580 | inf | 2.308 | 3.185/2.505 | 0.74 | 0 | 1.068 | 0.0% | 0.053 | False |
| midday_5_8_long | 0.75 | 0.60 | 0.8 | 1.0 | 36 | 10 | 90.0% | 2.527 | 16.954 | 2.234 | 2.368/2.545 | 0.93 | 0 | 1.250 | 0.0% | 0.061 | False |
| midday_5_8_long | 0.75 | 0.65 | 0.8 | 1.0 | 36 | 10 | 90.0% | 2.527 | 16.954 | 2.234 | 2.368/2.545 | 0.93 | 0 | 1.250 | 0.0% | 0.061 | False |
| morning_5_8_long | 0.85 | 0.55 | 1.0 | 1.5 | 36 | 8 | 87.5% | 2.527 | 18.082 | 2.334 | 0.098/2.874 | 0.00 | 0 | 1.139 | 0.1% | 0.040 | False |
| morning_5_8_long | 0.85 | 0.60 | 1.0 | 1.5 | 36 | 8 | 87.5% | 2.527 | 18.082 | 2.334 | 0.098/2.874 | 0.00 | 0 | 1.139 | 0.1% | 0.040 | False |

## Cross-Pocket Overlap

| cell_a | cell_b | overlap | independent |
| --- | --- | --- | --- |
| normal_vol_5_8_long|T0.75|D0.70|tf0.8|sl1.5|h60 | eqhl_5_8_long|T0.75|D0.55|tf1.0|sl1.5|h60 | 0.307 | partial |
| normal_vol_5_8_long|T0.75|D0.70|tf0.8|sl1.5|h60 | eqhl_5_8_long|T0.75|D0.55|tf1.0|sl1.0|h60 | 0.307 | partial |
| normal_vol_5_8_long|T0.75|D0.70|tf0.8|sl1.5|h60 | eqhl_5_8_long|T0.75|D0.55|tf1.0|sl1.5|h36 | 0.307 | partial |
| normal_vol_5_8_long|T0.75|D0.70|tf0.8|sl1.5|h60 | eqhl_5_8_long|T0.75|D0.55|tf1.0|sl1.0|h36 | 0.307 | partial |
| normal_vol_5_8_long|T0.75|D0.70|tf0.8|sl1.5|h60 | eqhl_5_8_long|T0.75|D0.55|tf0.8|sl1.5|h60 | 0.307 | partial |
| normal_vol_5_8_long|T0.75|D0.70|tf0.8|sl1.5|h60 | eqhl_5_8_long|T0.75|D0.55|tf0.8|sl1.5|h36 | 0.307 | partial |
| normal_vol_5_8_long|T0.75|D0.70|tf0.8|sl1.5|h60 | eqhl_5_8_long|T0.75|D0.55|tf1.0|sl1.5|h24 | 0.307 | partial |
| normal_vol_5_8_long|T0.75|D0.70|tf0.8|sl1.5|h60 | eqhl_5_8_long|T0.75|D0.55|tf1.0|sl2.0|h60 | 0.307 | partial |
| eqhl_5_8_long|T0.75|D0.55|tf1.0|sl1.5|h60 | normal_vol_5_8_long|T0.75|D0.70|tf1.0|sl2.0|h60 | 0.307 | partial |
| eqhl_5_8_long|T0.75|D0.55|tf1.0|sl1.5|h60 | normal_vol_5_8_long|T0.75|D0.70|tf0.8|sl2.0|h60 | 0.307 | partial |
| eqhl_5_8_long|T0.75|D0.55|tf1.0|sl1.0|h60 | normal_vol_5_8_long|T0.75|D0.70|tf1.0|sl2.0|h60 | 0.307 | partial |
| eqhl_5_8_long|T0.75|D0.55|tf1.0|sl1.0|h60 | normal_vol_5_8_long|T0.75|D0.70|tf0.8|sl2.0|h60 | 0.307 | partial |

## Survivor Criteria

- `n_trades >= min_trades`
- `mean_R > +0.10`
- `PF > 1.40`
- `temporal_stability_score > 0.70`
- at least `3` sectors with positive mean R
- mean R remains positive under `1.5x` costs
- approximate 95% lower confidence bound of mean R is above zero
- strict survivor additionally requires `DSR >= 95%` and Bonferroni-adjusted DSR pass.

## Unavailable Or Skipped Dimensions

| dimension |
| --- |
| none |

## Interpretation

- If only one deduplicated survivor remains, the likely result is a refined single Track A pocket, not a multi-pocket portfolio.
- If multiple deduplicated survivors remain with low overlap, they become candidates for CPCV and synthetic nulls.
- DSR is expected to be hard to clear at this stage because this project has already tested many hypotheses. Treat economic survivors as candidates, not validated edges.

## Outputs

- Cells CSV: `reports/phase4_multi_pocket_mini_sweep_cells.csv`
- Survivors CSV: `reports/phase4_multi_pocket_mini_sweep_survivors.csv`
