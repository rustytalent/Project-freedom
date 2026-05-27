# Phase 4 Track A Constrained Validation

## Verdict

- Decision: `CONSTRAINED_PASS`
- At least one constrained pocket survives validation filters. Best: long_ex_banking_it / T>=0.75|D>=0.65|tf=1|sl=2|h=60 / all with validation R +0.349.
- This is still research validation, not live-trade approval.
- The test reuses the completed v2 1-minute Track A sweep; no simulator rerun was needed.

## Setup

- Trade source: `output_phase4_track_a_pretouch_sweep/pretouch_sweep_trades.parquet`
- Chronological validation split midpoint: `2025-10-14 08:00:00`
- DSR trials: `550`
- Validated hypothesis: Track A pre-touch, long-only; BANKING/IT exclusion and no-afternoon are sensitivity checks.

## Predeclared Anchor Pocket

Anchor cell: `T>=0.75`, `D>=0.60`, `3-8ATR`, target fraction `1.0`, stop `2.0ATR`, hold `60` bars, Q=`all`.

| scenario | cell | Q | n | mean_R | PF | val_n | val_R | val_PF | cost1.5x | sectors | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_ex_banking_it | T>=0.75|D>=0.60|tf=1|sl=2|h=60 | all | 440 | 0.360 | 1.890 | 217 | 0.340 | 1.772 | 0.185 | 3 | True |
| long_auto_pharma_fmcg | T>=0.75|D>=0.60|tf=1|sl=2|h=60 | all | 440 | 0.360 | 1.890 | 217 | 0.340 | 1.772 | 0.185 | 3 | True |
| long_all_sectors | T>=0.75|D>=0.60|tf=1|sl=2|h=60 | all | 541 | 0.307 | 1.752 | 276 | 0.253 | 1.581 | 0.130 | 5 | True |
| long_no_afternoon_ex_banking_it | T>=0.75|D>=0.60|tf=1|sl=2|h=60 | all | 188 | 0.762 | 3.465 | 130 | 0.613 | 2.595 | 0.614 | 2 | False |
| long_no_afternoon | T>=0.75|D>=0.60|tf=1|sl=2|h=60 | all | 231 | 0.640 | 2.791 | 154 | 0.507 | 2.200 | 0.496 | 2 | False |
| short_all_sectors_control | T>=0.75|D>=0.60|tf=1|sl=2|h=60 | all | 59 | -0.560 | 0.328 | 53 | -0.619 | 0.271 | -0.786 | 0 | False |
| banking_it_long_control | T>=0.75|D>=0.60|tf=1|sl=2|h=60 | all | 101 | 0.074 | 1.176 | 59 | -0.068 | 0.835 | -0.111 | 2 | False |

## Constrained Passes

| scenario | cell | Q | n | mean_R | PF | val_n | val_R | val_PF | cost1.5x | sectors | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_ex_banking_it | T>=0.75|D>=0.65|tf=1|sl=2|h=60 | all | 434 | 0.365 | 1.908 | 211 | 0.349 | 1.802 | 0.190 | 3 | True |
| long_auto_pharma_fmcg | T>=0.75|D>=0.65|tf=1|sl=2|h=60 | all | 434 | 0.365 | 1.908 | 211 | 0.349 | 1.802 | 0.190 | 3 | True |
| long_ex_banking_it | T>=0.75|D>=0.60|tf=1|sl=2|h=60 | all | 440 | 0.360 | 1.890 | 217 | 0.340 | 1.772 | 0.185 | 3 | True |
| long_auto_pharma_fmcg | T>=0.75|D>=0.60|tf=1|sl=2|h=60 | all | 440 | 0.360 | 1.890 | 217 | 0.340 | 1.772 | 0.185 | 3 | True |
| long_ex_banking_it | T>=0.75|D>=0.65|tf=1|sl=2|h=36 | all | 434 | 0.315 | 1.804 | 211 | 0.324 | 1.802 | 0.141 | 3 | True |
| long_auto_pharma_fmcg | T>=0.75|D>=0.65|tf=1|sl=2|h=36 | all | 434 | 0.315 | 1.804 | 211 | 0.324 | 1.802 | 0.141 | 3 | True |
| long_ex_banking_it | T>=0.75|D>=0.60|tf=1|sl=2|h=36 | all | 440 | 0.311 | 1.787 | 217 | 0.315 | 1.770 | 0.137 | 3 | True |
| long_auto_pharma_fmcg | T>=0.75|D>=0.60|tf=1|sl=2|h=36 | all | 440 | 0.311 | 1.787 | 217 | 0.315 | 1.770 | 0.137 | 3 | True |
| long_ex_banking_it | T>=0.75|D>=0.65|tf=1|sl=1.5|h=60 | all | 434 | 0.366 | 1.612 | 211 | 0.301 | 1.467 | 0.139 | 3 | True |
| long_auto_pharma_fmcg | T>=0.75|D>=0.65|tf=1|sl=1.5|h=60 | all | 434 | 0.366 | 1.612 | 211 | 0.301 | 1.467 | 0.139 | 3 | True |
| long_ex_banking_it | T>=0.75|D>=0.65|tf=1|sl=2|h=24 | all | 434 | 0.251 | 1.644 | 211 | 0.289 | 1.719 | 0.078 | 3 | True |
| long_auto_pharma_fmcg | T>=0.75|D>=0.65|tf=1|sl=2|h=24 | all | 434 | 0.251 | 1.644 | 211 | 0.289 | 1.719 | 0.078 | 3 | True |
| long_ex_banking_it | T>=0.75|D>=0.60|tf=1|sl=1.5|h=60 | all | 440 | 0.359 | 1.596 | 217 | 0.288 | 1.442 | 0.132 | 3 | True |
| long_auto_pharma_fmcg | T>=0.75|D>=0.60|tf=1|sl=1.5|h=60 | all | 440 | 0.359 | 1.596 | 217 | 0.288 | 1.442 | 0.132 | 3 | True |
| long_all_sectors | T>=0.75|D>=0.65|tf=1|sl=2|h=60 | all | 526 | 0.321 | 1.791 | 263 | 0.278 | 1.649 | 0.146 | 5 | True |
| long_ex_banking_it | T>=0.75|D>=0.60|tf=1|sl=2|h=24 | all | 440 | 0.246 | 1.627 | 217 | 0.278 | 1.682 | 0.073 | 3 | True |
| long_auto_pharma_fmcg | T>=0.75|D>=0.60|tf=1|sl=2|h=24 | all | 440 | 0.246 | 1.627 | 217 | 0.278 | 1.682 | 0.073 | 3 | True |
| long_ex_banking_it | T>=0.75|D>=0.65|tf=1|sl=1.5|h=36 | all | 434 | 0.306 | 1.518 | 211 | 0.276 | 1.445 | 0.080 | 3 | True |
| long_auto_pharma_fmcg | T>=0.75|D>=0.65|tf=1|sl=1.5|h=36 | all | 434 | 0.306 | 1.518 | 211 | 0.276 | 1.445 | 0.080 | 3 | True |
| long_ex_banking_it | T>=0.75|D>=0.65|tf=0.8|sl=2|h=60 | all | 434 | 0.295 | 1.755 | 211 | 0.269 | 1.641 | 0.120 | 3 | True |

## Holdout Selection Check

For each scenario, the cell is selected by first-half OOS mean R only, then scored on the second half.

| scenario | cell | Q | n | mean_R | PF | val_n | val_R | val_PF | cost1.5x | sectors | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_auto_pharma_fmcg | T>=0.75|D>=0.60|tf=1|sl=1.5|h=60 | all | 440 | 0.359 | 1.596 | 217 | 0.288 | 1.442 | 0.132 | 3 | True |
| long_ex_banking_it | T>=0.75|D>=0.60|tf=1|sl=1.5|h=60 | all | 440 | 0.359 | 1.596 | 217 | 0.288 | 1.442 | 0.132 | 3 | True |
| long_all_sectors | T>=0.75|D>=0.60|tf=1|sl=1|h=60 | top_50pct | 271 | 0.227 | 1.240 | 151 | -0.032 | 0.970 | -0.103 | 3 | False |

## Top Validation Rows

These rows are sorted by validation R after requiring at least 100 validation trades and 2 sector buckets.

| scenario | cell | Q | n | mean_R | PF | val_n | val_R | val_PF | cost1.5x | sectors | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_no_afternoon_ex_banking_it | T>=0.75|D>=0.60|tf=1|sl=1.5|h=60 | all | 188 | 0.939 | 3.259 | 130 | 0.702 | 2.317 | 0.746 | 2 | False |
| long_no_afternoon_ex_banking_it | T>=0.75|D>=0.65|tf=1|sl=1.5|h=60 | all | 186 | 0.924 | 3.199 | 128 | 0.677 | 2.249 | 0.731 | 2 | False |
| long_no_afternoon_ex_banking_it | T>=0.75|D>=0.60|tf=1|sl=1.5|h=36 | all | 188 | 0.800 | 3.007 | 130 | 0.662 | 2.338 | 0.610 | 2 | False |
| long_no_afternoon_ex_banking_it | T>=0.75|D>=0.65|tf=1|sl=1.5|h=36 | all | 186 | 0.784 | 2.944 | 128 | 0.635 | 2.265 | 0.594 | 2 | False |
| long_no_afternoon_ex_banking_it | T>=0.75|D>=0.60|tf=1|sl=2|h=60 | all | 188 | 0.762 | 3.465 | 130 | 0.613 | 2.595 | 0.614 | 2 | False |
| long_no_afternoon | T>=0.75|D>=0.60|tf=1|sl=1.5|h=60 | all | 231 | 0.800 | 2.716 | 154 | 0.598 | 2.065 | 0.612 | 2 | False |
| long_no_afternoon_ex_banking_it | T>=0.75|D>=0.65|tf=1|sl=2|h=60 | all | 186 | 0.751 | 3.403 | 128 | 0.594 | 2.522 | 0.603 | 2 | False |
| long_no_afternoon | T>=0.75|D>=0.65|tf=1|sl=1.5|h=60 | all | 228 | 0.797 | 2.713 | 151 | 0.589 | 2.049 | 0.611 | 2 | False |
| long_no_afternoon_ex_banking_it | T>=0.75|D>=0.55|tf=1|sl=2|h=60 | all | 216 | 0.734 | 3.323 | 147 | 0.580 | 2.453 | 0.575 | 2 | False |
| long_no_afternoon_ex_banking_it | T>=0.75|D>=0.60|tf=1|sl=2|h=36 | all | 188 | 0.646 | 3.266 | 130 | 0.572 | 2.717 | 0.501 | 2 | False |
| long_no_afternoon_ex_banking_it | T>=0.75|D>=0.60|tf=1|sl=1.5|h=24 | all | 188 | 0.582 | 2.413 | 130 | 0.571 | 2.155 | 0.393 | 2 | False |
| long_no_afternoon_ex_banking_it | T>=0.75|D>=0.55|tf=1|sl=1.5|h=60 | all | 216 | 0.842 | 2.894 | 147 | 0.569 | 1.977 | 0.633 | 2 | False |
| long_no_afternoon_ex_banking_it | T>=0.75|D>=0.55|tf=1|sl=2|h=36 | all | 216 | 0.624 | 3.210 | 147 | 0.558 | 2.667 | 0.467 | 2 | False |
| long_no_afternoon_ex_banking_it | T>=0.75|D>=0.65|tf=1|sl=2|h=36 | all | 186 | 0.633 | 3.199 | 128 | 0.553 | 2.634 | 0.489 | 2 | False |
| long_no_afternoon_ex_banking_it | T>=0.75|D>=0.65|tf=1|sl=1.5|h=24 | all | 186 | 0.568 | 2.363 | 128 | 0.550 | 2.095 | 0.379 | 2 | False |
| long_no_afternoon | T>=0.75|D>=0.60|tf=1|sl=1.5|h=36 | all | 231 | 0.700 | 2.649 | 154 | 0.549 | 2.058 | 0.515 | 2 | False |
| long_no_afternoon | T>=0.75|D>=0.65|tf=1|sl=1.5|h=36 | all | 228 | 0.695 | 2.645 | 151 | 0.539 | 2.040 | 0.512 | 2 | False |
| long_no_afternoon_ex_banking_it | T>=0.75|D>=0.55|tf=1|sl=1.5|h=36 | all | 216 | 0.698 | 2.645 | 147 | 0.536 | 1.993 | 0.493 | 2 | False |
| long_no_afternoon_ex_banking_it | T>=0.75|D>=0.60|tf=0.8|sl=1.5|h=60 | all | 188 | 0.785 | 2.986 | 130 | 0.515 | 1.999 | 0.593 | 2 | False |
| long_no_afternoon_ex_banking_it | T>=0.75|D>=0.60|tf=1|sl=2|h=24 | all | 188 | 0.496 | 2.758 | 130 | 0.509 | 2.546 | 0.352 | 2 | False |

## Controls

| scenario | cell | Q | n | mean_R | PF | val_n | val_R | val_PF | cost1.5x | sectors | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| banking_it_long_control | T>=0.75|D>=0.65|tf=1|sl=2|h=60 | all | 92 | 0.113 | 1.268 | 52 | -0.010 | 0.975 | -0.060 | 2 | False |
| banking_it_long_control | T>=0.75|D>=0.65|tf=1|sl=2|h=24 | all | 92 | 0.179 | 1.498 | 52 | -0.014 | 0.963 | 0.006 | 2 | False |
| banking_it_long_control | T>=0.75|D>=0.65|tf=1|sl=2|h=36 | all | 92 | 0.131 | 1.348 | 52 | -0.030 | 0.918 | -0.041 | 2 | False |
| banking_it_long_control | T>=0.75|D>=0.65|tf=0.8|sl=2|h=24 | all | 92 | 0.251 | 1.824 | 52 | -0.045 | 0.880 | 0.080 | 2 | False |
| banking_it_long_control | T>=0.75|D>=0.65|tf=0.8|sl=2|h=60 | all | 92 | 0.167 | 1.453 | 52 | -0.058 | 0.855 | -0.006 | 2 | False |
| banking_it_long_control | T>=0.75|D>=0.65|tf=0.6|sl=2|h=24 | all | 92 | 0.145 | 1.480 | 52 | -0.060 | 0.837 | -0.026 | 2 | False |
| banking_it_long_control | T>=0.75|D>=0.65|tf=0.8|sl=2|h=12 | all | 92 | 0.189 | 1.620 | 52 | -0.061 | 0.831 | 0.017 | 2 | False |
| banking_it_long_control | T>=0.75|D>=0.65|tf=0.6|sl=2|h=60 | all | 92 | 0.067 | 1.186 | 52 | -0.062 | 0.837 | -0.105 | 2 | False |
| banking_it_long_control | T>=0.75|D>=0.60|tf=1|sl=2|h=60 | all | 101 | 0.074 | 1.176 | 59 | -0.068 | 0.835 | -0.111 | 2 | False |
| banking_it_long_control | T>=0.75|D>=0.65|tf=0.8|sl=2|h=36 | all | 92 | 0.189 | 1.588 | 52 | -0.070 | 0.811 | 0.018 | 2 | False |
| banking_it_long_control | T>=0.75|D>=0.60|tf=1|sl=2|h=24 | all | 101 | 0.134 | 1.369 | 59 | -0.072 | 0.816 | -0.050 | 2 | False |
| banking_it_long_control | T>=0.75|D>=0.65|tf=1|sl=2|h=12 | all | 92 | 0.128 | 1.411 | 52 | -0.075 | 0.792 | -0.044 | 2 | False |

## Interpretation

- The short side remains a negative control, not a strategy candidate.
- The broad long-only pocket is the key signal; BANKING/IT exclusion is a risk-control variant, not proof of a new model.
- The best constrained rows use Q=`all`; Q percentile filtering is not required for this pocket and should stay diagnostic for now.
- No-afternoon variants can improve mean R but reduce breadth, so they should be treated as a timing overlay after validation.
- If constrained passes remain positive, the next step is CPCV/null baselines before any paper-trading workflow.
- Triple-barrier labels should be built on these executable Track A outcomes only after this constrained pocket survives robustness checks.
