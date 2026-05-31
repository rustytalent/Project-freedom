# Post-MIS Baseline Re-Measurement

Date: 2026-06-01

## Verdict

R-MEASURE is complete.

- Track B / post-touch execution is still dead under corrected MIS arithmetic.
- Track A / pre-touch journey-to-liquidity remains the only credible equity path.
- Track A must not move to paper/live yet because V2 still needs MIS enforcement and synthetic null validation.

## Inputs

- Model bundle: `output_models/core25_latest/multi_asset_report.pkl`
- Track A trade parquet: `output_phase4_track_a_pretouch_sweep/pretouch_sweep_trades.parquet`
- Scripts:
  - `analysis/cost_sensitivity.py`
  - `analysis/pocket_sensitivity.py`
  - `analysis/pretouch_pocket_sensitivity.py`

## Cost Sensitivity

The post-MIS cost sensitivity run shows that sizing helps reduce cost drag, but does not create a deployable Track B edge.

| mode | net_R qty=1 | net_R 50k | net_R 100k | net_R 200k | gross_R at 200k |
| --- | ---: | ---: | ---: | ---: | ---: |
| blind_limit | -0.91 | -0.90 | -0.79 | -0.66 | -0.26 |
| displacement_confirmed | -0.47 | -0.47 | -0.38 | -0.29 | +0.00 |
| reclaim_confirmed | -1.02 | -1.02 | -0.85 | -0.66 | -0.06 |
| touch_confirmed | -0.53 | -0.52 | -0.44 | -0.35 | -0.06 |

Interpretation:

- `displacement_confirmed` is the least bad legacy mode.
- Its cost-free ceiling is only about breakeven.
- Even at 200k notional, it remains negative after costs.
- Position sizing is helpful but not sufficient.

## Track B Pocket Sensitivity

The post-MIS pocket replay found no positive Track B pocket.

Script verdict:

```text
Minimum sample threshold: n >= 50
Statistically positive cells (CI lower bound > 0): 0
Borderline (positive mean but CI crosses zero): 0
  (no positive cells at any sizing or pocket - even with MIS, the strategy has no positive-net-R subset on this OOS data)
```

Representative best-looking cells are still negative:

| pocket | mode | sizing | net_R | 95% CI |
| --- | --- | --- | ---: | --- |
| kitchen_sink | displacement_confirmed | 200k | -0.29 | [-0.33, -0.25] |
| auto_only | displacement_confirmed | 200k | -0.24 | [-0.31, -0.16] |
| favorable_sectors | displacement_confirmed | 200k | -0.27 | [-0.32, -0.22] |
| winning_combo | displacement_confirmed | 200k | -0.28 | [-0.36, -0.19] |
| winning_combo | touch_confirmed | 200k | -0.32 | [-0.37, -0.27] |

Interpretation:

- Track B is not merely weak in aggregate.
- Track B fails across the documented pockets, including the old winning combo.
- Reaction/post-touch models may still be useful as exit/context features, but not as primary entry engines.

## Track A Pre-Touch Pocket Sensitivity

Track A pre-touch sensitivity remains positive in multiple pockets.

| pocket | trades | win | gross_R | net_R | 95% CI | PF |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| kitchen_sink | 26784 | 43.6% | +0.42 | -0.12 | [-0.14, -0.11] | 1.39 |
| morning | 8748 | 49.9% | +0.59 | +0.18 | [+0.15, +0.21] | 2.67 |
| midday | 1584 | 58.6% | +0.82 | +0.30 | [+0.22, +0.37] | 1.54 |
| afternoon | 16452 | 38.8% | +0.29 | -0.32 | [-0.35, -0.30] | 0.77 |
| morning_midday | 10332 | 51.2% | +0.62 | +0.19 | [+0.17, +0.22] | 2.54 |
| auto_only | 8856 | 55.3% | +0.77 | +0.20 | [+0.17, +0.23] | 1.83 |
| favorable_sectors | 20088 | 46.6% | +0.50 | -0.04 | [-0.06, -0.02] | 1.51 |
| unfavorable_sectors | 6696 | 34.6% | +0.18 | -0.38 | [-0.41, -0.35] | 0.53 |
| long_only | 24120 | 45.1% | +0.46 | -0.07 | [-0.09, -0.05] | 1.47 |
| short_only | 2664 | 30.4% | +0.07 | -0.59 | [-0.64, -0.54] | 0.29 |
| winning_combo | 7776 | 56.8% | +0.79 | +0.35 | [+0.32, +0.38] | 2.86 |
| losing_combo | 4140 | 34.8% | +0.21 | -0.45 | [-0.49, -0.41] | 0.41 |

Script verdict:

```text
Minimum sample threshold: n >= 50
Statistically positive pockets (CI lower bound > 0): 5
  + morning                n=8748 net_R=+0.18 CI=[+0.15,+0.21] PF=2.67
  + midday                 n=1584 net_R=+0.30 CI=[+0.22,+0.37] PF=1.54
  + morning_midday         n=10332 net_R=+0.19 CI=[+0.17,+0.22] PF=2.54
  + auto_only              n=8856 net_R=+0.20 CI=[+0.17,+0.23] PF=1.83
  + winning_combo          n=7776 net_R=+0.35 CI=[+0.32,+0.38] PF=2.86
Borderline (positive mean, CI crosses zero): 0
```

Interpretation:

- The alpha is in the journey to liquidity, not the post-touch reaction.
- Afternoon is materially harmful and should stay excluded from any intraday Track A candidate.
- The strongest current research pocket is still `winning_combo`.
- The Track A result is promising but not live-approved.

## Decision

Track B is deprioritized as an entry strategy.

Track A advances to validation, but not immediately. Before synthetic nulls or CPCV, V2 must receive the same MIS enforcement as V1 because Track A evidence depends on V2/pre-touch execution artifacts.

Next task:

1. R-MIS-V2: enforce no-late-entry and same-day square-off in `execution_simulator_v2.py`.
2. Re-run or re-check Track A pre-touch artifacts under MIS-honest V2.
3. Then run synthetic nulls against the surviving Track A pockets.

