# Phase 4 Multi-Pocket Slice Analysis

## Verdict

- Decision: `MULTI_POCKET_CANDIDATES_FOUND`
- 6 candidate slices passed the anti-overfit filters.
- Slice hypotheses tested: `125`
- This is discovery only. It does not authorize live trading or paper trading.
- Distance `1-3 ATR` is only partially covered because the existing Track A sweep mostly spans `2-10 ATR`.
- Slice rows can include multiple geometries for the same market event; candidates are hypotheses for mini-sweeps.

## Sources

- Trades: `output_phase4_track_a_pretouch_sweep/pretouch_sweep_trades.parquet`
- Candidate enrichment: `output_phase4_track_a_pretouch_sweep/pretouch_candidates.parquet`

## Candidate Pockets

| family | slice | n | win | mean_R | PF | cost1.5x | 1st/2nd | symbols | sectors | top | reject |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| time_x_distance | midday / 5-8 | 576 | 65.6% | 0.722 | 2.766 | 0.511 | 1.617/0.662 | 8 | 4 | LUPIN:144 | candidate |
| time_x_distance | morning / 5-8 | 3348 | 58.2% | 0.668 | 2.600 | 0.471 | 0.680/0.651 | 15 | 5 | MARUTI:792 | candidate |
| time_x_direction | midday / UP | 1368 | 64.5% | 0.480 | 2.069 | 0.215 | -0.531/0.708 | 10 | 4 | LUPIN:288 | candidate |
| factor_x_distance | EQHL / 5-8 | 4428 | 59.8% | 0.478 | 1.954 | 0.200 | 0.551/0.371 | 18 | 5 | MARUTI:756 | candidate |
| volatility_x_distance | normal_vol / 5-8 | 7632 | 57.0% | 0.437 | 1.850 | 0.190 | 0.572/0.261 | 21 | 5 | MARUTI:1584 | candidate |
| time_bucket | midday | 1584 | 58.6% | 0.295 | 1.569 | 0.031 | -0.675/0.511 | 12 | 4 | LUPIN:288 | candidate |

## Diagnostic-Only Pockets

| family | slice | n | win | mean_R | PF | cost1.5x | 1st/2nd | symbols | sectors | top | reject |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| symbol | M&M | 612 | 85.0% | 0.916 | 6.669 | 0.659 | 1.114/0.740 | 1 | 1 | M&M:612 | one_symbol; one_sector |
| factor | ORB | 36 | 100.0% | 0.801 | inf | 0.418 | n/a/0.801 | 1 | 1 | MARUTI:36 | n<100; PF<=1.30; one_symbol; one_sector |
| factor_x_distance | ORB / 3-5 | 36 | 100.0% | 0.801 | inf | 0.418 | n/a/0.801 | 1 | 1 | MARUTI:36 | n<100; PF<=1.30; one_symbol; one_sector |
| symbol | ICICIBANK | 180 | 85.0% | 0.754 | 8.237 | 0.425 | 0.462/1.192 | 1 | 1 | ICICIBANK:180 | one_symbol; one_sector |
| symbol | MARUTI | 3600 | 62.5% | 0.305 | 1.600 | -0.006 | 0.439/0.112 | 1 | 1 | MARUTI:3600 | cost1.5x<=0; one_symbol; one_sector |
| symbol | DIVISLAB | 756 | 65.1% | 0.287 | 1.746 | 0.013 | -0.057/0.600 | 1 | 1 | DIVISLAB:756 | one_symbol; one_sector |
| day_of_week | Friday | 4716 | 53.7% | 0.236 | 1.397 | -0.056 | 1.044/-0.405 | 17 | 5 | TVSMOTOR:1224 | second_half<=0; cost1.5x<=0; first_half_only |
| sector_x_direction | AUTO / UP | 8712 | 55.8% | 0.221 | 1.418 | -0.064 | 0.271/0.142 | 5 | 1 | MARUTI:3492 | cost1.5x<=0; one_sector |
| time_x_direction | morning / UP | 8424 | 51.2% | 0.219 | 1.406 | 0.014 | 0.723/-0.090 | 19 | 5 | MARUTI:1476 | second_half<=0; first_half_only |
| sector | AUTO | 8856 | 55.3% | 0.201 | 1.372 | -0.085 | 0.278/0.082 | 5 | 1 | MARUTI:3600 | cost1.5x<=0; one_sector |
| distance_bucket | 5-8 | 12204 | 51.1% | 0.185 | 1.310 | -0.080 | 0.384/-0.012 | 23 | 5 | MARUTI:1908 | second_half<=0; cost1.5x<=0; first_half_only |
| time_bucket | morning | 8748 | 49.9% | 0.177 | 1.314 | -0.029 | 0.723/-0.139 | 20 | 5 | MARUTI:1476 | win<=50%; second_half<=0; cost1.5x<=0; first_half_only |
| symbol | EICHERMOT | 504 | 50.0% | 0.152 | 1.268 | -0.197 | 0.149/0.158 | 1 | 1 | EICHERMOT:504 | PF<=1.30; win<=50%; cost1.5x<=0; one_symbol; one_sector |
| factor_x_distance | PW / 3-5 | 36 | 50.0% | 0.127 | 1.253 | -0.053 | n/a/0.127 | 1 | 1 | TCS:36 | n<100; PF<=1.30; win<=50%; cost1.5x<=0; one_symbol; one_sector |
| vol_regime | normal_vol | 15804 | 50.7% | 0.114 | 1.186 | -0.142 | 0.398/-0.131 | 23 | 5 | MARUTI:2844 | PF<=1.30; second_half<=0; cost1.5x<=0; first_half_only |
| factor_x_distance | FVG / 5-8 | 5292 | 48.5% | 0.084 | 1.130 | -0.167 | 0.436/-0.187 | 19 | 5 | TECHM:972 | mean_R<=0.10; PF<=1.30; win<=50%; second_half<=0; cost1.5x<=0; first_half_only |
| factor_x_distance | PD / 5-8 | 36 | 50.0% | 0.076 | 1.192 | -0.403 | n/a/0.076 | 1 | 1 | ICICIBANK:36 | n<100; mean_R<=0.10; PF<=1.30; win<=50%; cost1.5x<=0; one_symbol; one_sector |
| symbol | TVSMOTOR | 3024 | 45.4% | 0.031 | 1.051 | -0.245 | 0.145/-0.589 | 1 | 1 | TVSMOTOR:3024 | mean_R<=0.10; PF<=1.30; win<=50%; second_half<=0; cost1.5x<=0; one_symbol; one_sector; first_half_only |
| volatility_x_distance | high_vol / 5-8 | 612 | 41.2% | 0.028 | 1.043 | -0.157 | 0.908/-0.243 | 7 | 4 | LUPIN:252 | mean_R<=0.10; PF<=1.30; win<=50%; second_half<=0; cost1.5x<=0; first_half_only |
| factor_x_distance | REJ / 5-8 | 720 | 49.6% | 0.020 | 1.037 | -0.296 | 0.295/-0.163 | 11 | 4 | TVSMOTOR:216 | mean_R<=0.10; PF<=1.30; win<=50%; second_half<=0; cost1.5x<=0; first_half_only |

## Rejected High-R Slices

| family | slice | n | win | mean_R | PF | cost1.5x | 1st/2nd | symbols | sectors | top | reject |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q_percentile_tier | top25-10 | 3888 | 52.7% | 0.214 | 1.348 | -0.046 | 0.368/0.071 | 14 | 4 | LUPIN:864 | cost1.5x<=0 |
| day_of_week | Thursday | 5832 | 55.8% | 0.189 | 1.330 | -0.057 | 0.361/0.074 | 21 | 5 | MARUTI:1872 | cost1.5x<=0 |
| direction_conf_tier | 80+ | 14688 | 51.7% | 0.179 | 1.311 | -0.047 | 0.282/0.066 | 20 | 5 | DRREDDY:3744 | cost1.5x<=0 |
| time_x_distance | midday / 3-5 | 1008 | 54.6% | 0.051 | 1.088 | -0.243 | -1.003/0.402 | 10 | 4 | SUNPHARMA:252 | mean_R<=0.10; PF<=1.30; cost1.5x<=0 |
| confluence_bucket | 2 | 504 | 37.5% | -0.073 | 0.907 | -0.393 | -0.597/0.870 | 7 | 4 | TVSMOTOR:216 | mean_R<=0.10; PF<=1.30; win<=50%; cost1.5x<=0 |
| factor_x_distance | OB / 5-8 | 1548 | 41.7% | -0.092 | 0.868 | -0.334 | -0.260/0.084 | 14 | 5 | DRREDDY:324 | mean_R<=0.10; PF<=1.30; win<=50%; cost1.5x<=0 |
| factor | OB | 4248 | 43.8% | -0.145 | 0.787 | -0.395 | -0.283/-0.032 | 18 | 5 | DRREDDY:828 | mean_R<=0.10; PF<=1.30; win<=50%; second_half<=0; cost1.5x<=0 |
| factor_x_distance | OB / 3-5 | 2700 | 45.0% | -0.175 | 0.738 | -0.430 | -0.300/-0.087 | 17 | 5 | DRREDDY:504 | mean_R<=0.10; PF<=1.30; win<=50%; second_half<=0; cost1.5x<=0 |
| p_touch_tier | 80-85 | 9468 | 41.5% | -0.214 | 0.714 | -0.493 | -0.025/-0.358 | 23 | 5 | MARUTI:1512 | mean_R<=0.10; PF<=1.30; win<=50%; second_half<=0; cost1.5x<=0 |
| tf_bucket | 3 | 5724 | 41.3% | -0.230 | 0.686 | -0.509 | -0.042/-0.377 | 19 | 5 | MARUTI:792 | mean_R<=0.10; PF<=1.30; win<=50%; second_half<=0; cost1.5x<=0 |
| tf_count_bucket | 3 | 5724 | 41.3% | -0.230 | 0.686 | -0.509 | -0.042/-0.377 | 19 | 5 | MARUTI:792 | mean_R<=0.10; PF<=1.30; win<=50%; second_half<=0; cost1.5x<=0 |
| q_percentile_tier | top50-25 | 6732 | 40.3% | -0.231 | 0.683 | -0.487 | -0.135/-0.273 | 20 | 5 | DRREDDY:1692 | mean_R<=0.10; PF<=1.30; win<=50%; second_half<=0; cost1.5x<=0 |
| p_touch_tier | 85-90 | 3780 | 38.8% | -0.237 | 0.669 | -0.516 | -0.201/-0.260 | 16 | 5 | DRREDDY:972 | mean_R<=0.10; PF<=1.30; win<=50%; second_half<=0; cost1.5x<=0 |
| confluence_bucket | 1 | 216 | 40.3% | -0.263 | 0.609 | -0.665 | -0.036/-1.398 | 4 | 3 | SBIN:108 | mean_R<=0.10; PF<=1.30; win<=50%; second_half<=0; cost1.5x<=0 |
| volatility_x_distance | low_vol / 5-8 | 3960 | 41.1% | -0.275 | 0.635 | -0.588 | -0.164/-0.351 | 20 | 5 | DRREDDY:864 | mean_R<=0.10; PF<=1.30; win<=50%; second_half<=0; cost1.5x<=0 |

## Diversification View

| candidate_a | candidate_b | corr | diversifying |
| --- | --- | --- | --- |
| time_x_distance:midday / 5-8 | time_x_distance:morning / 5-8 | -0.088 | yes |
| time_x_direction:midday / UP | time_x_distance:morning / 5-8 | -0.074 | yes |
| time_bucket:midday | time_x_distance:morning / 5-8 | -0.054 | yes |
| time_x_direction:midday / UP | volatility_x_distance:normal_vol / 5-8 | 0.064 | yes |
| time_bucket:midday | volatility_x_distance:normal_vol / 5-8 | 0.072 | yes |
| time_x_distance:midday / 5-8 | volatility_x_distance:normal_vol / 5-8 | 0.078 | yes |
| factor_x_distance:EQHL / 5-8 | time_bucket:midday | 0.202 | yes |
| factor_x_distance:EQHL / 5-8 | time_x_direction:midday / UP | 0.202 | yes |
| factor_x_distance:EQHL / 5-8 | time_x_distance:midday / 5-8 | 0.228 | yes |
| factor_x_distance:EQHL / 5-8 | time_x_distance:morning / 5-8 | 0.372 | yes |

## Recommended Mini-Sweeps

| candidate | filter | why | mini_sweep |
| --- | --- | --- | --- |
| time_x_distance:midday / 5-8 | time_bucket == 'midday' & distance_bucket == '5-8' | n=576, R=+0.722, PF=2.77 | T [0.70,0.75,0.80], D [0.60,0.65,0.70], target [0.6,0.8,1.0], stop [1.5,2.0,2.5], hold [24,36,60] |
| time_x_distance:morning / 5-8 | time_bucket == 'morning' & distance_bucket == '5-8' | n=3348, R=+0.668, PF=2.60 | T [0.70,0.75,0.80], D [0.60,0.65,0.70], target [0.6,0.8,1.0], stop [1.5,2.0,2.5], hold [24,36,60] |
| time_x_direction:midday / UP | time_bucket == 'midday' & direction == 'UP' | n=1368, R=+0.480, PF=2.07 | T [0.70,0.75,0.80], D [0.60,0.65,0.70], target [0.6,0.8,1.0], stop [1.5,2.0,2.5], hold [24,36,60] |
| factor_x_distance:EQHL / 5-8 | factor_enriched == 'EQHL' & distance_bucket == '5-8' | n=4428, R=+0.478, PF=1.95 | T [0.70,0.75,0.80], D [0.60,0.65,0.70], target [0.6,0.8,1.0], stop [1.5,2.0,2.5], hold [24,36,60] |
| volatility_x_distance:normal_vol / 5-8 | vol_regime == 'normal_vol' & distance_bucket == '5-8' | n=7632, R=+0.437, PF=1.85 | T [0.70,0.75,0.80], D [0.60,0.65,0.70], target [0.6,0.8,1.0], stop [1.5,2.0,2.5], hold [24,36,60] |

## Unavailable Or Skipped Dimensions

| dimension |
| --- |
| none |

## Interpretation

- Treat candidate pockets as pre-registered hypotheses for the next mini-sweep, not as results.
- A useful multi-pocket portfolio needs candidates that survive validation and are not highly correlated.
- If candidates mostly repeat the existing long AUTO/FMCG/PHARMA theme, the project still has one broad pocket, not many independent alphas.
- Full synthetic nulls remain required before final-stage trade modeling or paper trading.

## Outputs

- All slices: `reports/phase4_multi_pocket_slices.csv`
- Candidate pockets: `reports/phase4_multi_pocket_candidates.csv`
- Correlation matrix: `reports/phase4_multi_pocket_correlations.csv`
