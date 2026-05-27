# Phase 4 Track A CPCV / Component Nulls

## Verdict

- Decision: `CPCV_COMPONENT_MARGINAL`
- The fixed pocket is still positive across many paths, but robustness is not decisive.
- Scope note: this is a fixed-pocket CPCV-style path audit using existing v2 Track A trades.
- It includes shuffled-direction and sector/short controls, but not synthetic random-pool or ATR-offset nulls yet.

## Fixed Pocket

- T >= `0.75`
- Direction-to-pool >= `0.65`
- Distance `3.0-8.0 ATR`
- Geometry target fraction `1.0`, stop `2.0 ATR`, hold `60` bars
- Primary pocket: long-only, sectors `AUTO, FMCG, PHARMA`

## Path Summary

| scenario | paths | median_n | mean_R | median_R | worst_R | positive | PF>1 | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_all_sectors | 45 | 105.0 | 0.322 | 0.312 | -0.301 | 84.4% | 84.4% | 84.4% |
| long_ex_banking_it | 45 | 87.0 | 0.367 | 0.339 | -0.251 | 80.0% | 80.0% | 80.0% |
| banking_it_long_control | 45 | 18.0 | 0.117 | 0.115 | -0.720 | 62.2% | 62.2% | 0.0% |
| short_all_sectors_control | 45 | 9.0 | -0.547 | -0.579 | -1.432 | 11.1% | 11.1% | 0.0% |

## Component Nulls

| kind | label | n | win | mean_R | PF | DSR |
| --- | --- | --- | --- | --- | --- | --- |
| actual | actual_long_ex_banking_it | 434 | 57.4% | 0.365 | 1.908 | 0.0% |
| sector_control | actual_long_all_sectors | 526 | 56.7% | 0.321 | 1.791 | 0.0% |
| negative_control | actual_banking_it_long_control | 92 | 53.3% | 0.113 | 1.268 | 0.0% |
| negative_control | actual_short_all_sectors_control | 44 | 29.5% | -0.546 | 0.385 | 0.0% |
| shuffled_direction_null | 500 shuffles; p(null>=actual)=0.000 | 434 median | 51.8% | 0.222 median | 1.473 median | 0.0% |

## Weakest Pocket Paths

| path | blocks | n | mean_R | PF | pass |
| --- | --- | --- | --- | --- | --- |
| 37 | 5+7 | 88 | -0.251 | 0.622 | False |
| 5 | 0+5 | 88 | -0.200 | 0.700 | False |
| 7 | 0+7 | 88 | -0.198 | 0.722 | False |
| 36 | 5+6 | 87 | -0.089 | 0.840 | False |
| 40 | 6+7 | 87 | -0.087 | 0.856 | False |
| 13 | 1+5 | 87 | -0.069 | 0.870 | False |
| 15 | 1+7 | 87 | -0.066 | 0.884 | False |
| 6 | 0+6 | 87 | -0.035 | 0.942 | False |

## Interpretation

- If the pocket passes here, the next missing test is the expensive synthetic-null suite: random pools and ATR-offset pools.
- If all-sectors is close to ex-BANKING/IT, the sector exclusion should become a soft risk weight rather than a hard rule.
- If shuffled direction is close to actual, the direction model is not contributing enough and Track A should simplify toward proximity-only.
- This report still does not authorize live trading; it only decides whether to invest compute in full nulls/final-stage labels.

## Outputs

- Path CSV: `reports/phase4_track_a_cpcv_paths.csv`
- Null CSV: `reports/phase4_track_a_component_nulls.csv`