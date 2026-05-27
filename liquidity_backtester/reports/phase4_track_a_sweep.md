# Phase 4 Track A Pre-Touch Directional Sweep

## Verdict

- Decision: `MARGINAL`
- Positive cells exist, but best deployable evidence fails robustness filters: mean_R=+0.222.
- Cumulative DSR trials used: `550`
- Intraday session exit enforced at `15:10` IST.

## Pre-Flight Direction Stability

| distance | n | base | mean_p | AUC |
| --- | --- | --- | --- | --- |
| 2-5 ATR | 131078 | 41.3% | 41.0% | 0.708 |
| 3-8 ATR | 128751 | 43.1% | 40.7% | 0.701 |
| 5-10 ATR | 96233 | 43.9% | 39.9% | 0.686 |

## Inputs

- Model report: `output_models/core25_phase4_v2_neutral/multi_asset_report.pkl`
- Feature store: `output_feature_store/core25_phase4_v2_neutral`
- Raw 1m dir: `/Users/abc/Library/CloudStorage/GoogleDrive-garvitkatyal312@gmail.com/My Drive/kite_indian_market_data/raw_1m`
- Candidate rows after floor gates: `744`
- Joined proximity rows: `2,577,956`
- Dropped rows without direction labels: `0`

## Top Cells By Mean R

| gate | geom | Q | n | win | mean_R | PF | DSR | long/short | sectors | stable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T>=0.75|D>=0.60|3-8ATR | tf=1, sl=2, h=60 | all | 600 | 52.8% | 0.222 | 1.493 | 0.0% | 541/59 | 5 | False |
| T>=0.75|D>=0.60|3-8ATR | tf=1, sl=1.5, h=60 | all | 600 | 47.2% | 0.203 | 1.314 | 0.0% | 541/59 | 5 | False |
| T>=0.75|D>=0.60|3-8ATR | tf=1, sl=2, h=36 | all | 600 | 53.0% | 0.188 | 1.433 | 0.0% | 541/59 | 5 | False |
| T>=0.75|D>=0.60|3-8ATR | tf=0.8, sl=2, h=60 | all | 600 | 55.7% | 0.177 | 1.410 | 0.0% | 541/59 | 5 | False |
| T>=0.75|D>=0.60|3-8ATR | tf=1, sl=1.5, h=36 | all | 600 | 48.0% | 0.165 | 1.262 | 0.0% | 541/59 | 5 | False |
| T>=0.75|D>=0.60|3-8ATR | tf=1, sl=2, h=24 | all | 600 | 52.8% | 0.154 | 1.363 | 0.0% | 541/59 | 5 | False |
| T>=0.75|D>=0.60|3-8ATR | tf=0.8, sl=2, h=36 | all | 600 | 55.3% | 0.149 | 1.357 | 0.0% | 541/59 | 5 | False |
| T>=0.75|D>=0.60|3-8ATR | tf=0.8, sl=1.5, h=60 | all | 600 | 50.0% | 0.141 | 1.226 | 0.0% | 541/59 | 5 | False |
| T>=0.75|D>=0.60|3-8ATR | tf=0.8, sl=2, h=24 | all | 600 | 55.3% | 0.130 | 1.319 | 0.0% | 541/59 | 5 | False |
| T>=0.75|D>=0.60|3-8ATR | tf=1, sl=1.5, h=24 | all | 600 | 47.7% | 0.110 | 1.177 | 0.0% | 541/59 | 5 | False |
| T>=0.75|D>=0.60|3-8ATR | tf=0.8, sl=1.5, h=36 | all | 600 | 50.3% | 0.108 | 1.177 | 0.0% | 541/59 | 5 | False |
| T>=0.75|D>=0.60|3-8ATR | tf=0.6, sl=2, h=60 | all | 600 | 59.0% | 0.094 | 1.228 | 0.0% | 541/59 | 5 | False |
| T>=0.75|D>=0.55|3-8ATR | tf=1, sl=2, h=60 | all | 744 | 47.8% | 0.086 | 1.165 | 0.0% | 670/74 | 5 | False |
| T>=0.75|D>=0.60|3-8ATR | tf=0.6, sl=2, h=36 | all | 600 | 57.8% | 0.085 | 1.214 | 0.0% | 541/59 | 5 | False |
| T>=0.75|D>=0.60|3-8ATR | tf=0.8, sl=1.5, h=24 | all | 600 | 50.2% | 0.074 | 1.123 | 0.0% | 541/59 | 5 | False |
| T>=0.75|D>=0.60|3-8ATR | tf=0.6, sl=2, h=24 | all | 600 | 58.7% | 0.074 | 1.191 | 0.0% | 541/59 | 5 | False |
| T>=0.75|D>=0.55|3-8ATR | tf=1, sl=2, h=36 | all | 744 | 48.1% | 0.056 | 1.111 | 0.0% | 670/74 | 5 | False |
| T>=0.75|D>=0.55|3-8ATR | tf=0.8, sl=2, h=60 | all | 744 | 50.8% | 0.053 | 1.106 | 0.0% | 670/74 | 5 | False |
| T>=0.75|D>=0.55|3-8ATR | tf=1, sl=1.5, h=60 | all | 744 | 42.6% | 0.039 | 1.055 | 0.0% | 670/74 | 5 | False |
| T>=0.75|D>=0.55|3-8ATR | tf=0.8, sl=2, h=36 | all | 744 | 50.5% | 0.030 | 1.062 | 0.0% | 670/74 | 5 | False |

## Top Cells By DSR

| gate | geom | Q | n | win | mean_R | PF | DSR | long/short | sectors | stable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T>=0.75|D>=0.55|3-8ATR | tf=0.6, sl=1, h=12 | all | 744 | 38.8% | -0.395 | 0.549 | 0.0% | 670/74 | 5 | False |
| T>=0.75|D>=0.60|3-8ATR | tf=0.6, sl=1, h=12 | all | 600 | 42.7% | -0.266 | 0.675 | 0.0% | 541/59 | 5 | False |
| T>=0.75|D>=0.55|3-8ATR | tf=1, sl=1, h=36 | all | 744 | 31.5% | -0.221 | 0.789 | 0.0% | 670/74 | 5 | False |
| T>=0.75|D>=0.60|3-8ATR | tf=1, sl=1, h=24 | all | 600 | 34.8% | -0.094 | 0.902 | 0.0% | 541/59 | 5 | False |
| T>=0.75|D>=0.55|3-8ATR | tf=1, sl=1, h=24 | all | 744 | 31.3% | -0.285 | 0.722 | 0.0% | 670/74 | 5 | False |
| T>=0.75|D>=0.60|3-8ATR | tf=1, sl=1, h=12 | all | 600 | 39.3% | -0.215 | 0.747 | 0.0% | 541/59 | 5 | False |
| T>=0.75|D>=0.55|3-8ATR | tf=1, sl=1, h=12 | all | 744 | 35.5% | -0.361 | 0.598 | 0.0% | 670/74 | 5 | False |
| T>=0.75|D>=0.60|3-8ATR | tf=0.8, sl=2, h=60 | all | 600 | 55.7% | 0.177 | 1.410 | 0.0% | 541/59 | 5 | False |
| T>=0.75|D>=0.55|3-8ATR | tf=0.8, sl=2, h=60 | all | 744 | 50.8% | 0.053 | 1.106 | 0.0% | 670/74 | 5 | False |
| T>=0.75|D>=0.60|3-8ATR | tf=0.8, sl=2, h=36 | all | 600 | 55.3% | 0.149 | 1.357 | 0.0% | 541/59 | 5 | False |

## Interpretation

- Q is applied post-hoc within each cell, not as a separate sweep dimension.
- Cells must clear minimum total trades, long/short trades, and sector breadth before they count as passing.
- `mean_R_cost_1_25x` and `mean_R_cost_1_50x` are included in the CSV for slippage/cost stress checks.
- This report is research-only and does not change live/predict gates.
