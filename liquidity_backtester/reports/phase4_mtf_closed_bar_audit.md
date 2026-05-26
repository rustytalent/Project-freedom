# Phase 4 MTF Closed-Bar Audit

**Status:** WARN

This Workstream 2 artifact audit checks multi-timeframe bar shards for closed-bar
semantics before Track A/Track B strategy work continues.

## Scope

- Feature store: `output_feature_store/core25_fresh_may25`
- Higher timeframes: `15m, 60m, 180m, 1D, 1W`
- Decision split for join-risk scan: `oos` direction rows
- Audit rows: 125
- Issues: ERROR=0, WARN=50, INFO=125

## Key Findings

- Hard errors: `0`.
- Exact higher-TF aggregations: `15m, 180m, 60m`.
- Naive open-timestamp join risk: `110,000` decision rows.
- Safe rule for any future MTF join: use only bars where `bar_open + timeframe <= decision_ts`.
- `1D`: 50 stored bars missing from expected.
- `1W`: 50 aggregate value mismatches.

## Summary

| TF | Symbols | Stored Rows | Expected Rows | Stored Not In Expected | Expected Not In Stored | Aggregate Mismatches | Naive Join Risk Rows | Naive Join Risk % | Max OHLC Error | Max Volume Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 15m | 25 | 464,200 | 464,200 | 0 | 0 | 0 | 22,000 | 100.0% | 0 | 0 |
| 180m | 25 | 55,775 | 55,775 | 0 | 0 | 0 | 22,000 | 100.0% | 0 | 0 |
| 1D | 25 | 18,675 | 18,625 | 50 | 0 | 0 | 22,000 | 100.0% | 0 | 0 |
| 1W | 25 | 3,950 | 3,950 | 0 | 0 | 50 | 22,000 | 100.0% | 145 | 3.86e+06 |
| 60m | 25 | 130,050 | 130,050 | 0 | 0 | 0 | 22,000 | 100.0% | 0 | 0 |

## Anchoring Rules Used

- `15m`: pandas `15min`, left-labeled closed-left bars.
- `60m`: pandas `60min`, left-labeled bars anchored to the first 5m timestamp.
- `180m`: pandas `180min`, left-labeled bars anchored to the first 5m timestamp.
- `1D`: IST-midnight calendar day, represented as naive UTC-style `18:30` timestamps.
- `1W`: IST Friday-week anchor, represented as naive UTC-style `18:30` timestamps.

## Interpretation

The audit found no hard closed-bar errors. Intraday higher-timeframe bars that
match exactly are listed in Key Findings; daily/weekly warning rows should be
reviewed as data-generation consistency issues, not automatically as lookahead.
The `naive_join_leak_risk_rows` column is not a current-code failure; it shows how many
direction decision timestamps would leak if future code used a plain backward join
on the higher-TF bar open timestamp instead of requiring `bar_open + timeframe <= decision_ts`.
WARN rows mean some stored higher-timeframe bars differ from exact 5m aggregation or coverage. These should be reviewed, but they are not automatically lookahead errors.

## Issue Rows

Detailed rows were written to `reports/phase4_mtf_closed_bar_issues.csv`.

## Next Step

Continue Workstream 2 with pool availability replay. Track A should still wait for
the remaining leakage checks plus the proximity distance-bucket AUC viability gate.
