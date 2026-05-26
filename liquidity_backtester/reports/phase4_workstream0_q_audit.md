# Phase 4 Workstream 0 Q Scale Audit

## Verdict

- **Decision:** Branch A - Q is compressed and has edge.
- **Current Q gate reachable?** No.
- **Actual blended Q range:** 23.4% to 55.7%.
- **Blended Q standard deviation:** 1.84%.
- **Base strict-respect rate:** 43.5%.
- **Exact top 5% lift:** +5.91pp.
- **Exact top 1% lift:** +8.82pp.

The 70% absolute Q gate is broken for this model scale. Replace production research gates with percentile-based Q gates, then re-run execution backtests before enabling any live trade confidence.

Plain English: the quality model is not expressing confidence on a 0-100% live-gate scale. Its useful ranking signal lives inside a very narrow band around the base rate. A `Q >= 70%` live gate is therefore mathematically unreachable for the current conservative model outputs.

## Inputs

- Audit CSV: `output_core25_phase3d_train_fresh_may25/phase3_oos_prediction_audit.csv`
- Live plan JSON: `output_core25_phase3d_train_fresh_may25/live_plan.json`
- No retraining was performed.
- Only decisive OOS rows (`actual` not null) are used for realized strict-respect lift.

## Q Distribution

| q_col | n | min | p50 | p90 | p95 | p99 | max | mean | std | range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global_q | 33494 | 0.2344 | 0.4348 | 0.4552 | 0.4589 | 0.4598 | 0.5486 | 0.4361 | 0.0185 | 0.3142 |
| sector_q | 33494 | 0.1500 | 0.4395 | 0.4848 | 0.5060 | 0.5090 | 0.5970 | 0.4393 | 0.0333 | 0.4470 |
| blended_q | 33494 | 0.2344 | 0.4347 | 0.4543 | 0.4587 | 0.4599 | 0.5573 | 0.4355 | 0.0184 | 0.3230 |

## Gate Reachability

| check | value | result |
| --- | --- | --- |
| current absolute gate | 70.0% | UNREACHABLE |
| max observed blended_q | 55.7% | below current gate |
| setups passing current gate | 0.0000% | none |
| top 25% threshold | 45.4% | candidate watch threshold |
| top 10% threshold | 45.4% | candidate strict threshold |
| top 5% threshold | 45.9% | candidate high-confidence research threshold |
| top 1% threshold | 46.0% | rare tail threshold |

## Q Decile Actual Strict-Respect Rates

| decile | n | mean_q | min_q | max_q | actual | lift | rel_lift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2230 | 0.3946 | 0.2727 | 0.4180 | 38.7% | -4.80pp | -11.0% |
| 1 | 2207 | 0.4249 | 0.4180 | 0.4270 | 42.1% | -1.36pp | -3.1% |
| 2 | 2270 | 0.4283 | 0.4270 | 0.4296 | 42.6% | -0.90pp | -2.1% |
| 3 | 2126 | 0.4315 | 0.4296 | 0.4328 | 41.0% | -2.48pp | -5.7% |
| 4 | 2242 | 0.4341 | 0.4329 | 0.4346 | 45.1% | +1.69pp | +3.9% |
| 5 | 2110 | 0.4349 | 0.4346 | 0.4360 | 42.4% | -1.03pp | -2.4% |
| 6 | 2182 | 0.4375 | 0.4360 | 0.4448 | 45.5% | +2.01pp | +4.6% |
| 7 | 2283 | 0.4505 | 0.4449 | 0.4538 | 41.9% | -1.53pp | -3.5% |
| 8 | 2425 | 0.4540 | 0.4539 | 0.4543 | 47.0% | +3.52pp | +8.1% |
| 9 | 1874 | 0.4591 | 0.4544 | 0.5486 | 48.8% | +5.32pp | +12.2% |

## Exact Top-Percentile Lift

| slice | n | min_q | mean_q | max_q | actual | lift | rel_lift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| top 50pct | 10975 | 0.4346 | 0.4470 | 0.5486 | 45.1% | +1.64pp | +3.8% |
| top 25pct | 5488 | 0.4535 | 0.4557 | 0.5486 | 46.6% | +3.16pp | +7.3% |
| top 10pct | 2195 | 0.4543 | 0.4584 | 0.5486 | 48.7% | +5.25pp | +12.1% |
| top 5pct | 1098 | 0.4581 | 0.4613 | 0.5486 | 49.4% | +5.91pp | +13.6% |
| top 1pct | 220 | 0.4598 | 0.4694 | 0.5486 | 52.3% | +8.82pp | +20.3% |
| top 0 1pct | 22 | 0.5069 | 0.5263 | 0.5486 | 54.5% | +11.09pp | +25.5% |

## Per-Sector Compression

| sector | min | max | mean | std | range |
| --- | --- | --- | --- | --- | --- |
| BANKING | 0.2727 | 0.5449 | 0.4352 | 0.0193 | 0.2722 |
| AUTO | 0.2920 | 0.5573 | 0.4290 | 0.0181 | 0.2653 |
| FMCG | 0.3272 | 0.5486 | 0.4391 | 0.0181 | 0.2214 |
| PHARMA | 0.2344 | 0.5142 | 0.4382 | 0.0171 | 0.2798 |
| IT | 0.3272 | 0.5486 | 0.4393 | 0.0166 | 0.2214 |

## Per-Asset Compression

| asset | min | max | mean | std | range |
| --- | --- | --- | --- | --- | --- |
| SBIN | 0.2727 | 0.5449 | 0.4296 | 0.0228 | 0.2722 |
| LUPIN | 0.2344 | 0.5142 | 0.4349 | 0.0215 | 0.2798 |
| ITC | 0.3291 | 0.5486 | 0.4356 | 0.0211 | 0.2195 |
| INFY | 0.3503 | 0.5486 | 0.4393 | 0.0204 | 0.1983 |
| EICHERMOT | 0.2920 | 0.5483 | 0.4264 | 0.0201 | 0.2563 |
| NESTLEIND | 0.3272 | 0.4801 | 0.4380 | 0.0200 | 0.1529 |
| TVSMOTOR | 0.3386 | 0.4888 | 0.4269 | 0.0199 | 0.1501 |
| DABUR | 0.3586 | 0.5449 | 0.4406 | 0.0189 | 0.1863 |
| AXISBANK | 0.3457 | 0.5315 | 0.4364 | 0.0181 | 0.1858 |
| HDFCBANK | 0.3451 | 0.5069 | 0.4361 | 0.0179 | 0.1618 |
| M&M | 0.3483 | 0.5573 | 0.4326 | 0.0177 | 0.2090 |
| MARUTI | 0.3325 | 0.5184 | 0.4281 | 0.0174 | 0.1858 |
| ICICIBANK | 0.3325 | 0.4606 | 0.4422 | 0.0171 | 0.1281 |
| WIPRO | 0.3272 | 0.4629 | 0.4378 | 0.0170 | 0.1358 |
| SUNPHARMA | 0.3580 | 0.4606 | 0.4425 | 0.0165 | 0.1026 |
| DIVISLAB | 0.3291 | 0.4606 | 0.4337 | 0.0162 | 0.1315 |
| KOTAKBANK | 0.3129 | 0.4606 | 0.4367 | 0.0154 | 0.1477 |
| TCS | 0.3592 | 0.4659 | 0.4402 | 0.0154 | 0.1067 |
| HCLTECH | 0.3586 | 0.5069 | 0.4426 | 0.0150 | 0.1483 |
| TECHM | 0.3457 | 0.4989 | 0.4378 | 0.0146 | 0.1532 |
| CIPLA | 0.3580 | 0.4615 | 0.4392 | 0.0146 | 0.1035 |
| HINDUNILVR | 0.3718 | 0.4598 | 0.4406 | 0.0140 | 0.0879 |
| BRITANNIA | 0.3565 | 0.4606 | 0.4412 | 0.0138 | 0.1041 |
| BAJAJ-AUTO | 0.3539 | 0.4954 | 0.4324 | 0.0138 | 0.1415 |
| DRREDDY | 0.3544 | 0.4801 | 0.4429 | 0.0134 | 0.1257 |

## Percentile-Gate Replacement Spec

This report does not implement gate changes. If the user approves Branch A follow-up, the next code change should replace absolute Q thresholds with percentile-aware thresholds learned from OOS calibration artifacts.

- Persist Q distribution metadata with the model bundle: `q_col`, OOS quantiles, base strict rate, top-slice lifts, and source artifact hash if available.
- Convert each live candidate's `blended_q` into `q_percentile` using the saved OOS distribution.
- Replace the current hard `Q >= 70%` gate with a research gate such as `q_percentile >= 0.90` or `q_percentile >= 0.95`, then backtest before promoting it.
- Keep `P_touch`, `P_reaction`, direction alignment, distance, bucket sample size, sector regime, and net expectancy gates. Percentile Q is not allowed to create a trade by itself.
- Re-run execution backtests on top-percentile-Q cohorts and report PF, mean R, max drawdown, and DSR before changing live behavior.

## Next Step

Stop here for review. Do not start Track A, Track B, simulator v2, or leakage CI until this audit is accepted. The immediate likely follow-up is a percentile-gate research backtest, not live trading.
