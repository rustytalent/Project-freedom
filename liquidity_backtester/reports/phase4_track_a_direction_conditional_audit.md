# Phase 4 Track A Direction-Conditional Audit

**Decision:** VIABLE

Direction signal is preserved inside high-P_touch, 2-8 ATR setup rows.

This audit tests whether direction still helps after conditioning on the setup rows
that Track A would care about: high P_touch and tradeable distance. Metrics are
pool-side relative: above-pool rows want UP, below-pool rows want DOWN.

## Scope

- Model report: `output_models/core25_latest/multi_asset_report.pkl`
- Feature store: `output_feature_store/core25_fresh_may25`
- Split: `oos`
- Proximity horizon: `78`
- Proximity shards read: `125`
- Joined setup rows: `2,572,441`
- Rows dropped with no direction label: `0`
- Metrics CSV: `reports/phase4_track_a_direction_conditional_audit.csv`

## Baseline Direction Model

- Full OOS direction rows: `22,000`
- Full OOS p_up AUC: `0.621`
- Full OOS accuracy at 0.50: `58.4%`

## Primary Gate

- Segment: `p_touch>=0.85 & distance 2-8 ATR`
- Rows: `173`
- AUC: `0.724`
- Accuracy at 0.50: `70.5%`
- Top-quartile confidence accuracy: `100.0%`
- Long/short rows: `148` / `25`
- Long/short AUC: `0.723` / `0.673`
- Sample caution: this strict high-P_touch segment is small; use broader sensitivity bands in the sweep.

Decision rules:

- `AUC >= 0.60`: Direction conditionality passes.
- `0.55 <= AUC < 0.60`: Marginal; continue cautiously.
- `AUC < 0.55`: Track A direction edge is not preserved.

## Conditional Metrics

| Segment | N | Base Correct Dir | Mean P_dir | AUC | Acc@0.50 | Top-Q Acc | Long Rows | Short Rows | Long AUC | Short AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all proximity setup rows | 2,572,441 | 47.7% | 47.1% | 0.634 | 59.4% | 70.7% | 666,259 | 1,906,182 | 0.650 | 0.628 |
| distance 2-8 ATR | 180,790 | 41.9% | 40.3% | 0.715 | 67.1% | 81.8% | 83,501 | 97,289 | 0.712 | 0.710 |
| distance 3-8 ATR | 124,632 | 42.6% | 40.0% | 0.711 | 66.4% | 81.5% | 56,860 | 67,772 | 0.707 | 0.706 |
| p_touch>=0.80 & distance 2-8 ATR | 287 | 79.4% | 70.9% | 0.758 | 75.3% | 98.7% | 245 | 42 | 0.773 | 0.653 |
| p_touch>=0.85 & distance 2-8 ATR | 173 | 80.3% | 71.2% | 0.724 | 70.5% | 100.0% | 148 | 25 | 0.723 | 0.673 |
| p_touch>=0.90 & distance 2-8 ATR | 24 | 70.8% | 70.5% | 0.782 | 62.5% | 100.0% | 20 | 4 | 0.787 | 1.000 |
| p_touch>=0.85 & distance 3-8 ATR | 137 | 84.7% | 72.1% | 0.658 | 68.6% | 100.0% | 122 | 15 | 0.682 | 0.444 |
| p_touch>=0.85 & distance 5-8 ATR | 58 | 84.5% | 71.6% | 0.609 | 67.2% | 100.0% | 51 | 7 | 0.594 | 0.500 |
| p_touch<0.50 all distances | 2,555,708 | 47.6% | 47.1% | 0.633 | 59.4% | 70.6% | 655,166 | 1,900,542 | 0.648 | 0.628 |

## Interpretation

Direction remains useful inside high-P_touch tradeable-distance setup rows.
Track A can proceed, but the next implementation should be Execution Simulator
v2 before any large pre-touch sweep, because profitability still depends on fills,
slippage, stops, targets, and costs.
