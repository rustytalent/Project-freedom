# Phase 4 Track A Proximity Distance-Bucket Audit

**Decision:** VIABLE

Primary 3-8 ATR proximity AUC is above 0.70.

This is the Track A viability gate. It tests whether P_touch remains predictive
at tradeable pre-touch distances, rather than only looking strong because very
nearby pools are easy to classify.

## Scope

- Model report: `output_models/core25_latest/multi_asset_report.pkl`
- Feature store: `output_feature_store/core25_fresh_may25`
- Split: `oos`
- Shards read by horizon: `{78: 125, 156: 125, 312: 125}`
- Metrics CSV: `reports/phase4_track_a_proximity_bucket_audit.csv`

## Primary Gate

- Primary horizon: `78` bars
- Tradeable bucket: `3-8 ATR`
- Rows: `124,632`
- Base touch rate: `12.4%`
- AUC: `0.835`
- Top-decile actual touched: `46.9%`
- Lift over bucket base: `+34.55pp`

Decision rules from the Phase 4 brief:

- `AUC > 0.70`: Track A is viable, run the full pre-touch sweep.
- `AUC < 0.55`: Track A is not viable; proximity is mostly trivial/too weak here.
- `0.55 <= AUC <= 0.70`: Track A is marginal; continue cautiously.

## Bucket Metrics

| Horizon | Bucket | N | Base Touch | Mean P_touch | AUC | Top-Decile Actual | Lift |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 78 | 0-1 ATR | 35,165 | 11.1% | 15.5% | 0.791 | 37.0% | +25.94pp |
| 78 | 1-3 ATR | 105,149 | 12.1% | 15.6% | 0.796 | 39.9% | +27.75pp |
| 78 | 3-5 ATR | 70,262 | 12.8% | 13.9% | 0.813 | 45.1% | +32.26pp |
| 78 | 5-8 ATR | 54,370 | 11.7% | 11.1% | 0.862 | 49.8% | +38.05pp |
| 78 | 8-12 ATR | 50,894 | 4.7% | 3.5% | 0.881 | 27.3% | +22.58pp |
| 78 | 12+ ATR | 2,256,601 | 0.0% | 0.0% | 0.997 | 0.3% | +0.31pp |
| 78 | 3-8 ATR | 124,632 | 12.4% | 12.7% | 0.835 | 46.9% | +34.55pp |
| 156 | 0-1 ATR | 34,985 | 11.5% | 15.5% | 0.788 | 37.5% | +26.04pp |
| 156 | 1-3 ATR | 104,670 | 12.8% | 16.2% | 0.797 | 42.0% | +29.16pp |
| 156 | 3-5 ATR | 70,667 | 14.3% | 15.4% | 0.820 | 49.1% | +34.77pp |
| 156 | 5-8 ATR | 54,576 | 13.7% | 13.5% | 0.864 | 56.3% | +42.56pp |
| 156 | 8-12 ATR | 51,160 | 5.7% | 4.3% | 0.876 | 31.7% | +25.94pp |
| 156 | 12+ ATR | 2,244,086 | 0.0% | 0.0% | 0.994 | 0.4% | +0.39pp |
| 156 | 3-8 ATR | 125,243 | 14.0% | 14.6% | 0.840 | 52.0% | +38.00pp |
| 312 | 0-1 ATR | 34,610 | 11.5% | 15.4% | 0.790 | 37.9% | +26.39pp |
| 312 | 1-3 ATR | 103,459 | 12.8% | 15.9% | 0.799 | 42.1% | +29.31pp |
| 312 | 3-5 ATR | 69,681 | 14.4% | 15.2% | 0.821 | 49.7% | +35.31pp |
| 312 | 5-8 ATR | 54,288 | 13.7% | 13.4% | 0.862 | 55.1% | +41.39pp |
| 312 | 8-12 ATR | 50,330 | 5.8% | 4.3% | 0.871 | 31.7% | +25.95pp |
| 312 | 12+ ATR | 2,228,553 | 0.0% | 0.0% | 0.995 | 0.4% | +0.39pp |
| 312 | 3-8 ATR | 123,969 | 14.1% | 14.4% | 0.839 | 52.0% | +37.90pp |

## Interpretation

The 3-8 ATR bucket clears the strict viability threshold on the primary horizon.
Track A can proceed to the direction-conditional audit and then the reduced
pre-touch parameter sweep under the v2 execution simulator.
