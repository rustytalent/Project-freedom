# Q Percentile Filtering Does Not Rescue Track B Under V2 Simulator

## Verdict

- Decision: `TRACK_B_DEAD`
- Best neutral top-1% Q cohort is -0.62R, worse than -0.50R.
- Official decision uses the neutral fill policy; generous/conservative are sensitivity checks.
- This report does not change live gates and does not prove any tradeable edge.

## Inputs

- `output_core25_phase4_v2_neutral`
- `output_core25_phase4_v2_generous`
- `output_core25_phase4_v2_conservative`

## Required Matrix: Neutral Fill Policy

Each cell is `mean R (trade count)`.

| mode | all | top_25pct | top_10pct | top_5pct | top_1pct |
| --- | --- | --- | --- | --- | --- |
| blind_limit | -1.198R (20696) | -1.161R (5399) | -1.115R (2147) | -1.038R (1044) | -1.255R (202) |
| displacement_confirmed | -0.677R (7730) | -0.658R (1863) | -0.634R (691) | -0.470R (318) | -0.618R (85) |
| reclaim_confirmed | -1.420R (7683) | -1.400R (2064) | -1.421R (852) | -1.299R (413) | -1.175R (71) |
| touch_confirmed | -0.709R (19107) | -0.700R (4853) | -0.679R (1879) | -0.595R (919) | -0.714R (189) |

## Sensitivity: Generous Fill Policy

| mode | all | top_25pct | top_10pct | top_5pct | top_1pct |
| --- | --- | --- | --- | --- | --- |
| blind_limit | -1.135R (24145) | -1.096R (6291) | -1.047R (2490) | -0.962R (1213) | -1.184R (231) |
| displacement_confirmed | -0.677R (7730) | -0.654R (1871) | -0.631R (689) | -0.479R (317) | -0.653R (86) |
| reclaim_confirmed | -1.420R (7683) | -1.402R (2073) | -1.424R (852) | -1.296R (413) | -1.211R (71) |
| touch_confirmed | -0.709R (19107) | -0.698R (4857) | -0.675R (1876) | -0.591R (916) | -0.703R (189) |

## Sensitivity: Conservative Fill Policy

| mode | all | top_25pct | top_10pct | top_5pct | top_1pct |
| --- | --- | --- | --- | --- | --- |
| blind_limit | -1.193R (19482) | -1.166R (5063) | -1.131R (2014) | -1.056R (982) | -1.256R (186) |
| displacement_confirmed | -0.677R (7730) | -0.656R (1853) | -0.629R (683) | -0.478R (315) | -0.635R (84) |
| reclaim_confirmed | -1.420R (7683) | -1.399R (2059) | -1.424R (849) | -1.303R (411) | -1.201R (70) |
| touch_confirmed | -0.709R (19107) | -0.699R (4834) | -0.678R (1858) | -0.593R (915) | -0.706R (187) |

## Full Cohort Table

| fill | mode | cohort | trades | win | mean_R | median_R | PF | maxDD_R | actual | mean_Q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conservative | blind_limit | all | 19482 | 19.0% | -1.193 | -1.614 | 0.089 | -23238.9 | 41.7% | 42.9% |
| conservative | blind_limit | top_25pct | 5063 | 19.2% | -1.166 | -1.580 | 0.093 | -5901.7 | 43.4% | 44.9% |
| conservative | blind_limit | top_10pct | 2014 | 19.9% | -1.131 | -1.550 | 0.098 | -2278.8 | 44.3% | 45.4% |
| conservative | blind_limit | top_5pct | 982 | 21.7% | -1.056 | -1.468 | 0.114 | -1034.8 | 47.3% | 45.9% |
| conservative | blind_limit | top_1pct | 186 | 17.2% | -1.256 | -1.632 | 0.074 | -231.9 | 43.9% | 48.3% |
| conservative | displacement_confirmed | all | 7730 | 37.3% | -0.677 | -1.318 | 0.232 | -5234.0 | 59.2% | 42.8% |
| conservative | displacement_confirmed | top_25pct | 1853 | 39.1% | -0.656 | -1.287 | 0.236 | -1215.9 | 62.6% | 44.9% |
| conservative | displacement_confirmed | top_10pct | 683 | 39.4% | -0.629 | -1.269 | 0.255 | -429.9 | 64.1% | 45.5% |
| conservative | displacement_confirmed | top_5pct | 315 | 46.0% | -0.478 | -0.702 | 0.354 | -151.3 | 69.1% | 46.1% |
| conservative | displacement_confirmed | top_1pct | 84 | 39.3% | -0.635 | -1.277 | 0.282 | -54.1 | 62.5% | 48.5% |
| conservative | reclaim_confirmed | all | 7683 | 14.7% | -1.420 | -1.833 | 0.073 | -10907.7 | 79.4% | 43.0% |
| conservative | reclaim_confirmed | top_25pct | 2059 | 15.9% | -1.399 | -1.820 | 0.084 | -2878.2 | 80.0% | 44.8% |
| conservative | reclaim_confirmed | top_10pct | 849 | 14.8% | -1.424 | -1.819 | 0.082 | -1206.7 | 78.4% | 45.3% |
| conservative | reclaim_confirmed | top_5pct | 411 | 17.5% | -1.303 | -1.775 | 0.115 | -535.3 | 81.7% | 45.8% |
| conservative | reclaim_confirmed | top_1pct | 70 | 21.4% | -1.201 | -1.818 | 0.121 | -82.1 | 83.1% | 48.3% |
| conservative | touch_confirmed | all | 19107 | 35.7% | -0.709 | -1.326 | 0.200 | -13542.2 | 48.5% | 42.9% |
| conservative | touch_confirmed | top_25pct | 4834 | 35.6% | -0.699 | -1.300 | 0.194 | -3379.7 | 51.6% | 44.8% |
| conservative | touch_confirmed | top_10pct | 1858 | 36.1% | -0.678 | -1.285 | 0.199 | -1262.3 | 52.9% | 45.4% |
| conservative | touch_confirmed | top_5pct | 915 | 39.3% | -0.593 | -1.206 | 0.234 | -544.7 | 56.1% | 45.9% |
| conservative | touch_confirmed | top_1pct | 187 | 34.8% | -0.706 | -1.333 | 0.214 | -134.1 | 51.7% | 48.2% |
| generous | blind_limit | all | 24145 | 22.4% | -1.135 | -1.632 | 0.109 | -27411.4 | 42.9% | 42.9% |
| generous | blind_limit | top_25pct | 6291 | 23.1% | -1.096 | -1.596 | 0.118 | -6895.2 | 45.2% | 44.8% |
| generous | blind_limit | top_10pct | 2490 | 24.3% | -1.047 | -1.558 | 0.130 | -2608.3 | 45.9% | 45.4% |
| generous | blind_limit | top_5pct | 1213 | 26.6% | -0.962 | -1.479 | 0.155 | -1164.8 | 49.2% | 45.8% |
| generous | blind_limit | top_1pct | 231 | 22.5% | -1.184 | -1.659 | 0.093 | -273.8 | 47.2% | 48.2% |
| generous | displacement_confirmed | all | 7730 | 37.3% | -0.677 | -1.318 | 0.232 | -5234.0 | 59.2% | 42.8% |
| generous | displacement_confirmed | top_25pct | 1871 | 39.1% | -0.654 | -1.287 | 0.237 | -1223.7 | 62.6% | 44.8% |
| generous | displacement_confirmed | top_10pct | 689 | 39.5% | -0.631 | -1.274 | 0.254 | -435.0 | 63.5% | 45.5% |
| generous | displacement_confirmed | top_5pct | 317 | 46.1% | -0.479 | -0.702 | 0.352 | -152.7 | 67.9% | 46.1% |
| generous | displacement_confirmed | top_1pct | 86 | 38.4% | -0.653 | -1.277 | 0.272 | -56.9 | 62.2% | 48.4% |
| generous | reclaim_confirmed | all | 7683 | 14.7% | -1.420 | -1.833 | 0.073 | -10907.7 | 79.4% | 43.0% |
| generous | reclaim_confirmed | top_25pct | 2073 | 15.8% | -1.402 | -1.819 | 0.083 | -2904.5 | 80.0% | 44.8% |
| generous | reclaim_confirmed | top_10pct | 852 | 14.9% | -1.424 | -1.817 | 0.082 | -1211.2 | 77.9% | 45.3% |
| generous | reclaim_confirmed | top_5pct | 413 | 17.7% | -1.296 | -1.766 | 0.116 | -535.3 | 81.5% | 45.8% |
| generous | reclaim_confirmed | top_1pct | 71 | 21.1% | -1.211 | -1.819 | 0.119 | -84.0 | 83.3% | 48.4% |
| generous | touch_confirmed | all | 19107 | 35.7% | -0.709 | -1.326 | 0.200 | -13542.2 | 48.5% | 42.9% |
| generous | touch_confirmed | top_25pct | 4857 | 35.5% | -0.698 | -1.300 | 0.195 | -3392.9 | 51.7% | 44.8% |
| generous | touch_confirmed | top_10pct | 1876 | 36.2% | -0.675 | -1.285 | 0.201 | -1269.0 | 52.9% | 45.4% |
| generous | touch_confirmed | top_5pct | 916 | 39.4% | -0.591 | -1.201 | 0.234 | -543.9 | 56.3% | 45.9% |
| generous | touch_confirmed | top_1pct | 189 | 34.9% | -0.703 | -1.333 | 0.215 | -135.1 | 51.7% | 48.2% |
| neutral | blind_limit | all | 20696 | 19.7% | -1.198 | -1.641 | 0.095 | -24797.2 | 42.0% | 42.9% |
| neutral | blind_limit | top_25pct | 5399 | 20.3% | -1.161 | -1.604 | 0.102 | -6268.5 | 44.0% | 44.9% |
| neutral | blind_limit | top_10pct | 2147 | 21.4% | -1.115 | -1.568 | 0.112 | -2394.1 | 44.9% | 45.4% |
| neutral | blind_limit | top_5pct | 1044 | 23.4% | -1.038 | -1.501 | 0.133 | -1081.5 | 48.0% | 45.9% |
| neutral | blind_limit | top_1pct | 202 | 18.8% | -1.255 | -1.662 | 0.077 | -251.8 | 46.0% | 48.4% |
| neutral | displacement_confirmed | all | 7730 | 37.3% | -0.677 | -1.318 | 0.232 | -5234.0 | 59.2% | 42.8% |
| neutral | displacement_confirmed | top_25pct | 1863 | 39.0% | -0.658 | -1.287 | 0.234 | -1227.0 | 62.5% | 44.9% |
| neutral | displacement_confirmed | top_10pct | 691 | 39.4% | -0.634 | -1.274 | 0.253 | -438.1 | 64.1% | 45.5% |
| neutral | displacement_confirmed | top_5pct | 318 | 46.5% | -0.470 | -0.380 | 0.359 | -150.1 | 69.1% | 46.1% |
| neutral | displacement_confirmed | top_1pct | 85 | 40.0% | -0.618 | -1.271 | 0.294 | -53.3 | 63.0% | 48.5% |
| neutral | reclaim_confirmed | all | 7683 | 14.7% | -1.420 | -1.833 | 0.073 | -10907.7 | 79.4% | 43.0% |
| neutral | reclaim_confirmed | top_25pct | 2064 | 15.8% | -1.400 | -1.820 | 0.083 | -2888.6 | 80.0% | 44.8% |
| neutral | reclaim_confirmed | top_10pct | 852 | 14.9% | -1.421 | -1.819 | 0.082 | -1208.8 | 78.4% | 45.3% |
| neutral | reclaim_confirmed | top_5pct | 413 | 17.7% | -1.299 | -1.771 | 0.116 | -536.4 | 81.7% | 45.8% |
| neutral | reclaim_confirmed | top_1pct | 71 | 22.5% | -1.175 | -1.817 | 0.128 | -81.5 | 83.3% | 48.4% |
| neutral | touch_confirmed | all | 19107 | 35.7% | -0.709 | -1.326 | 0.200 | -13542.2 | 48.5% | 42.9% |
| neutral | touch_confirmed | top_25pct | 4853 | 35.6% | -0.700 | -1.301 | 0.194 | -3396.5 | 51.6% | 44.8% |
| neutral | touch_confirmed | top_10pct | 1879 | 36.1% | -0.679 | -1.285 | 0.199 | -1277.5 | 53.0% | 45.4% |
| neutral | touch_confirmed | top_5pct | 919 | 39.3% | -0.595 | -1.206 | 0.233 | -549.5 | 56.1% | 45.9% |
| neutral | touch_confirmed | top_1pct | 189 | 34.4% | -0.714 | -1.333 | 0.211 | -137.0 | 51.7% | 48.2% |

## Interpretation

- Q ranking still improves some cohorts, but not enough to overcome v2 execution friction.
- The old post-touch policies remain negative even after filtering to the highest-Q pools.
- Track B should stay deprioritized unless a future execution policy changes the payoff geometry.
- The next primary research path remains Track A pre-touch directional sweep under v2.
