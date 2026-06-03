# Outcome Logging — Schema & Storage Spec

This is the data flywheel. Every prediction the Daily Brief makes goes
into a log on creation. Every prediction's resolution is written when
the outcome becomes known. The join of these two tables is the moat.

## Why this is the most important non-customer-facing artifact

The pool-detection algorithm is reverse-engineerable. The model
architecture is reverse-engineerable. The calibration stack is too.
None of those are the moat.

The moat is the **outcome log**: a continuously-growing dataset of
"what we predicted, with what confidence, in what regime — and what
actually happened." It compounds weekly. It can't be reverse-
engineered because it requires having made the predictions in real
time. It is simultaneously:

* **Marketing** — honest, audited track record. Customers can verify.
* **R&D signal** — tells us which features actually predict and which
  are noise. Drives the next retrain.
* **Retention mechanism** — the "Yesterday Audit" section of the brief
  is rendered from this log. No log → no audit → no trust.
* **SEBI safe-harbour evidence** — proves the service is research
  (probabilistic, calibrated, with honest miss disclosure) and not
  tipster advice.

If this log doesn't exist before customer #1's first brief, the most
valuable early data is lost forever. **Build the log first.**

## Two tables

### Table 1: `predictions`

Append-only. One row per prediction made by the Daily Brief generator.
Written at brief-generation time.

```json
{
  "prediction_id": "PRED_2026_06_03_NIFTY50_p_touch_24500_h60",
  "brief_id": "BRIEF_2026_06_03",
  "generated_at_utc": "2026-06-03T03:00:00Z",
  "trading_date_ist": "2026-06-03",
  "model_bundle_version": "core25_head_alpha@2b6782d",
  "feature_version": "42_features_v3",

  "prediction_type": "proximity",
  "instrument_kind": "index",
  "symbol": "NIFTY50",

  "target_description": {
    "kind": "level_touch",
    "level": 24500.0,
    "level_type": "demand_pool",
    "side_from_open": "below",
    "horizon_bars": 60,
    "horizon_label": "60min"
  },

  "predicted_value": 0.78,
  "predicted_value_kind": "calibrated_probability",
  "confidence_bucket": "high",
  "raw_score_diagnostic": 0.82,

  "regime_tags_at_prediction": {
    "vol_regime": "normal",
    "session_character_predicted": "trend_day",
    "sector_state": "banking_trending_up",
    "weekly_expiry_day": false,
    "trap_pattern_active": null
  },

  "logged_for_audit": true
}
```

Fields explained:

* `prediction_id` — globally unique. The format encodes the brief, the
  symbol, the prediction type, and the horizon so it's human-readable
  too. Joins to the resolutions table.
* `prediction_type` — one of:
  * `proximity` — "will level L be touched within horizon H"
  * `direction` — "will price end higher/lower over horizon H"
  * `avoidance` — "stand aside today" recommendation
  * `regime` — meta predictions (e.g. "today is a trend day, 60%")
  * `options_strike` — "will strike K be tested" (the level-to-strike
    translator's output, redundant with proximity but logged separately
    so options track record is auditable on its own)
* `target_description` — schema varies by `prediction_type`. Always
  specifies what would resolve the prediction. The renderer uses this
  to write the human prose.
* `predicted_value` + `predicted_value_kind` — the predicted_value is
  always calibrated. Kind is one of `calibrated_probability`,
  `direction_sign` (-1 / 0 / +1), `boolean`, `continuous_atr_units`.
* `raw_score_diagnostic` — the uncalibrated model output, kept for
  audit/debug. Never user-facing.
* `regime_tags_at_prediction` — frozen at prediction time. The
  calibration calculator slices by these to compute per-regime
  reliability.

### Table 2: `resolutions`

Append-only. One row per prediction once outcome becomes known.
Written by an end-of-day (or end-of-horizon for longer windows)
resolution job.

```json
{
  "prediction_id": "PRED_2026_06_03_NIFTY50_p_touch_24500_h60",
  "resolved_at_utc": "2026-06-03T04:00:00Z",
  "resolution_method": "level_touched",

  "outcome_boolean": true,
  "outcome_continuous": null,

  "resolution_details": {
    "touched_at_bar_offset": 12,
    "touched_at_utc": "2026-06-03T03:58:00Z",
    "max_favorable_excursion_atr": 1.8,
    "max_adverse_excursion_atr": 0.4,
    "exit_reason": "horizon_expired_within_zone"
  },

  "had_data_gap": false,
  "resolution_quality": "clean"
}
```

Fields explained:

* `resolution_method` — one of:
  * `level_touched` — proximity prediction, the level was hit.
  * `level_not_touched_horizon_expired` — proximity, level not hit by
    end of horizon.
  * `direction_matched` / `direction_wrong` / `direction_neutral` —
    for direction predictions.
  * `avoidance_validated` — for avoidance predictions, the instrument
    actually had a bad day (any reasonable hypothetical trade lost
    money). The "what we told you to skip would have hurt".
  * `avoidance_unnecessary` — instrument had a good day, our avoid was
    wrong.
  * `data_gap` — the underlying data did not arrive in time to resolve.
* `outcome_boolean` — true / false for binary predictions.
* `outcome_continuous` — used for direction (signed-ATR) or regime
  (predicted_value compared to realised).
* `resolution_details` — schema varies per resolution method. Always
  includes timing and excursion data so we can compute time-to-touch
  distributions, drawdown along the way, etc.
* `had_data_gap` / `resolution_quality` — flags for excluding bad
  resolutions from calibration stats.

## Calibration derivation (the daily report)

The Yesterday Audit section of the brief is computed by joining
`predictions` to `resolutions` for the previous trading day:

```sql
-- pseudo-SQL for illustration; actual implementation uses pandas
SELECT
  p.confidence_bucket,
  p.prediction_type,
  COUNT(*) as n,
  AVG(CASE WHEN r.outcome_boolean THEN 1.0 ELSE 0.0 END) as hit_rate,
  AVG(p.predicted_value) as mean_predicted_p,
  AVG(p.predicted_value) - AVG(CASE WHEN r.outcome_boolean THEN 1.0 ELSE 0.0 END)
    as calibration_error
FROM predictions p
JOIN resolutions r USING (prediction_id)
WHERE p.trading_date_ist = '2026-06-02'
  AND r.had_data_gap = false
GROUP BY p.confidence_bucket, p.prediction_type;
```

The `calibration_error` column feeds the `drifting_today` field of
tomorrow's brief. If any (bucket, type) pair has |calibration_error|
> 0.08 on the last 30 trading days, that head is flagged DRIFTING and
the brief warns the reader.

## Storage

v1: Parquet files partitioned by trading_date_ist.

```
data/outcome_log/
  predictions/
    trading_date_ist=2026-06-03/
      predictions.parquet
  resolutions/
    trading_date_ist=2026-06-03/
      resolutions.parquet
```

* Append-only, never edit-in-place. If a resolution corrects a
  previous resolution (rare, e.g. data correction), write a new row
  with `correction_of_resolution_id` field; never overwrite.
* Partition by `trading_date_ist` for cheap pruning when computing the
  trailing-30-day calibration window.
* Schema-evolution rule: any new field must be additive. Old readers
  must not break on new fields. Field removal requires a schema
  version bump.

v2 (when paying customers > 10): Postgres for fast joins + a parquet
mirror for analytics. Not v1.

## Backfilling from existing OOS bundles

Before customer #1 ever sees a brief, we should have a populated log
covering at minimum the last 90 trading days. This gives the brief's
"Yesterday Audit" + calibration sections real data on day one.

The backfill procedure:

1. For each historical trading date in the last 90 days:
   * Load the bundle that was current as of that date
     (`model_bundle_version` field stamped accordingly).
   * Run the brief generator in "historical mode" — generates the same
     JSON it would have on that morning.
   * Write the predictions to the log with stamped historical
     `generated_at_utc`.
2. For each prediction, look up the actual resolution from existing
   OOS arsenal trade frames + raw bar data.
3. Write resolutions to the log.

This is a one-time job. Idempotent (re-runs overwrite the historical
partitions; not the live-collected ones).

## Outcome-log retention vs SEBI

* **Retain forever.** Storage is cheap, and the audit value compounds.
* The log is the EVIDENCE that the product is research-with-calibration,
  not tipster-with-luck. If SEBI ever asks "what did you actually say,
  to whom, and what happened", the log answers it cleanly.
* **Never log a customer-side action.** This log tracks our
  predictions, not what customers did with them. Customers' trade logs
  are theirs and they can opt-in to sharing for the Strategy Diagnosis
  product separately.

## Avoidance prediction logging — the special case

Avoidance predictions are negative-space — "don't trade". Resolving
them is asymmetric:

* If the customer avoided AND the day was bad → avoidance correct
  (validated).
* If the customer avoided AND the day was fine → avoidance wrong
  (over-conservative).
* If the customer traded anyway → not our concern; we logged the
  recommendation.

For our log purposes, we resolve the avoidance against the
hypothetical: "if a unit-position trade had been taken at session
open on this instrument with the model's standard barriers, would it
have lost?". This is computable from bar data. The resolution method
is `avoidance_validated` if the hypothetical lost more than 0.5 ATR,
`avoidance_unnecessary` otherwise.

This is one of the most important metrics to track because the
avoidance call is the SEBI-safest, customer-most-valuable section of
the brief. We need real numbers on whether it earns its keep.

## Implementation order

1. Implement `liqpool/products/outcome_log.py` — append-only writer
   with schema validation.
2. Backfill 90 days from existing bundles + OOS frames.
3. Wire the writer into the brief generator (predictions get logged
   at brief-creation time).
4. Implement `liqpool/products/resolver.py` — end-of-day job that
   reads predictions and writes resolutions.
5. Implement `liqpool/products/calibration_audit.py` — reads the
   joined log and produces the Yesterday Audit JSON block.

## Acceptance for this spec

1. User reviews and confirms the two-table model.
2. The five prediction_types cover everything the brief will produce.
3. Storage location + partition scheme is acceptable on the existing
   Google Drive warehouse (or wherever the user prefers).
4. The backfill plan is clear enough that Codex could execute it
   without further questions.
