# Daily Research Brief — Schema & Renderer Spec

The Daily Brief is the FIRST commercial product. One artifact, generated
once per trading day, in two synchronised forms:

* **Machine layer** — a single JSON document (this schema). Consumed by
  quant / engineer customers who want raw probabilities and tags to feed
  into their own systems.
* **Human layer** — a 5-to-7-minute read in PDF + plain email. Rendered
  deterministically from the JSON. No content lives outside the JSON;
  the renderer is purely a presentation layer.

Both come from the same generator. The JSON IS the contract.

## Hard constraints (do not violate)

1. **No tipster outputs.** Every section describes context, probabilities,
   regimes, and audit. Never "buy X at Y stop Z". This is the SEBI
   safe-harbour AND the moat.
2. **The human layer must read in 5–7 minutes.** Section length budgets
   are enforced by the renderer. If new data won't fit, it goes into the
   machine layer only.
3. **Every probability is calibrated.** No numbers shown that haven't
   been through the calibration stack (isotonic + bucket + distance).
   Uncalibrated raw scores are renamed `_raw_score` and never user-facing.
4. **Every claim is dated.** The brief is for one specific trading day.
   No "general" advice. No "this week" sections — those belong in a
   separate weekly product.
5. **Every prediction is logged.** The same generator writes to the
   outcome log (see `outcome_logging_schema.md`). A brief that doesn't
   log is a brief that can't be audited tomorrow.

## Top-level JSON shape

```json
{
  "schema_version": "1.0",
  "brief_metadata": { ... },
  "index_regime": { ... },
  "sector_regime": { ... },
  "top_watchlist": [ ... ],
  "options_suitability": { ... },
  "avoid_list": [ ... ],
  "key_zones": [ ... ],
  "confidence_notes": { ... },
  "yesterday_audit": { ... }
}
```

## brief_metadata

```json
{
  "brief_id": "BRIEF_2026_06_03",
  "trading_date_ist": "2026-06-03",
  "generated_at_utc": "2026-06-03T03:00:00Z",
  "model_bundle_version": "core25_head_alpha@2b6782d",
  "feature_version": "42_features_v3",
  "session_status": "pre_open",
  "reading_time_minutes": 6,
  "indexes_covered": ["NIFTY50", "BANKNIFTY", "FINNIFTY",
                       "NIFTYMIDCAPSELECT", "SENSEX"]
}
```

* `brief_id` — globally unique. Joins to the outcome log.
* `model_bundle_version` — bundle name + git sha. Allows post-hoc
  attribution of which model version made which call.
* `session_status` — `pre_open` (delivered before 09:00 IST) or
  `intraday_update` (a mid-session refresh, optional v2). v1 is
  pre_open only.

## index_regime

One block per index in `indexes_covered`. Length budget in the human
layer: ~30 seconds per index = ~2.5 minutes for all 5.

```json
{
  "NIFTY50": {
    "previous_close": 24521.30,
    "expected_gap_atr": 0.42,
    "expected_gap_classification": "small_gap_up",
    "vol_regime": "normal",
    "vol_regime_zscore_20d": 0.34,
    "today_session_character_prediction": "trend_day_60pct",
    "trap_risk_score": 0.18,
    "trap_pattern_active": null
  }
}
```

* `expected_gap_atr` — from overnight futures basis where available;
  null on days where futures data is missing.
* `vol_regime` — one of `low` / `normal` / `elevated` / `extreme`.
  Maps to `vol_regime_zscore_20d` quartile buckets.
* `today_session_character_prediction` — one of `trend_day`,
  `range_day`, `chop_day`, `inside_day`, `mixed` — plus a probability.
* `trap_risk_score` — 0..1, derived from `is_gap_up_trap_fade` and
  `is_gap_down_reversal` features pre-market.
* `trap_pattern_active` — null pre-open; populated mid-session if the
  brief re-renders intraday.

## sector_regime

One row per major NSE sector. Length budget: 1 paragraph total in human
layer.

```json
{
  "as_of_ist": "2026-06-03T09:15:00+05:30",
  "trending_up": ["BANKING", "IT"],
  "trending_down": ["METALS"],
  "chopping": ["PHARMA", "AUTO", "ENERGY"],
  "neutral": ["FMCG", "REALTY"],
  "leadership_change_vs_yesterday": ["BANKING", "METALS"]
}
```

The sector-rotation alpha already provides this signal. The
`leadership_change_vs_yesterday` field is the actionable bit — the
trader's eye should go to sectors that just flipped.

## top_watchlist

5–10 instruments where the model has actionable conviction. Length
budget: 1 line per entry in human layer.

```json
[
  {
    "symbol": "HDFCBANK",
    "sector": "BANKING",
    "side": "long",
    "reason_tag": "proximity_high_p_touch_journey",
    "key_level": 1742.50,
    "key_level_type": "demand_pool",
    "p_touch_today": 0.74,
    "p_touch_within_60min": 0.41,
    "model_confidence_bucket": "high",
    "regime_tags": ["banking_trending_up", "low_vol_regime"],
    "avoidance_note": null
  }
]
```

* `reason_tag` — one of a controlled vocabulary
  (`proximity_high_p_touch_journey`, `direction_alignment_strong`,
  `confluence_pool_volume_confirmed`, etc.). Renderer maps to prose.
* `key_level` + `key_level_type` — the structural level we are
  flagging. Never an entry recommendation, just "this level is in play
  today, with these odds".
* `model_confidence_bucket` — `low` / `moderate` / `high` / `very_high`,
  mapped from the model's calibrated probability against the per-
  prediction calibration table.
* `avoidance_note` — populated only if the avoidance alpha says
  "watchlisted but stand aside today for X reason".

## options_suitability

Per index. Length budget: ~30 seconds per index in human layer.

```json
{
  "NIFTY50": {
    "directional_bias": "mild_up",
    "expected_range_today_atr": 1.6,
    "expected_range_today_points": 392,
    "movement_quality_prediction": "clean_60pct_choppy_40pct",
    "theta_danger_score": 0.42,
    "strike_levels_in_play": [
      {
        "strike": 24500,
        "p_test_today": 0.78,
        "p_test_within_60min": 0.34,
        "side_from_open": "below",
        "key_level_type": "demand_pool"
      }
    ],
    "regime_for_premium_buyers": "moderate",
    "regime_for_premium_sellers": "favourable",
    "session_recommendation": "context_only_no_directional_call"
  }
}
```

* `directional_bias` — `strong_down` / `mild_down` / `neutral` /
  `mild_up` / `strong_up`. From the direction model + sector context.
* `expected_range_today_atr` — from vol regime model. Points version
  is for the trader's convenience.
* `movement_quality_prediction` — one of `clean_trend`,
  `choppy_in_range`, `clean_then_chop`, `mixed`. From the
  `path_efficiency_30` + `direction_changes_30` features.
* `theta_danger_score` — 0..1, high on chop-predicted days. Renderer
  uses this to write the "premium decay risk" sentence.
* `strike_levels_in_play` — the level→strike translator output. v1:
  pure level-to-strike mapping. v2: add IV / Greeks.
* `regime_for_premium_buyers` / `_sellers` — `unfavourable` /
  `moderate` / `favourable`. This is the closest the brief comes to
  a directional suggestion, and it's still framed as regime, not action.
* `session_recommendation` — always
  `context_only_no_directional_call` in v1. Future versions may add
  `range_play_setup_potential` or `breakout_watch` (still regime, not
  action).

## avoid_list

Instruments + regimes where the model says "stand aside today".
Length budget: short — usually <5 entries.

```json
[
  {
    "symbol": "RELIANCE",
    "reason": "vol_regime_elevated_AND_path_efficiency_collapsed",
    "regime_tags": ["elevated_vol", "chop_predicted"],
    "model_confidence": "high"
  },
  {
    "symbol": "ALL_INDEX_OPTIONS",
    "reason": "expiry_day_chop_pattern_active",
    "regime_tags": ["weekly_expiry_day", "is_gap_up_trap_fade"],
    "model_confidence": "moderate"
  }
]
```

The avoidance list is one of the most valuable sections to the
customer because it prevents losses. It is also THE most important
section for the SEBI safe-harbour: a service that helps people NOT
trade is unambiguously not advisory.

## key_zones

Active proximity-model levels across all watched instruments. Machine-
layer only; not rendered in the human PDF (would blow the length
budget). Customers who want the raw zone map consume this directly.

```json
[
  {
    "symbol": "BANKNIFTY",
    "level": 52000,
    "level_type": "supply_pool",
    "side_from_open": "above",
    "p_test_today": 0.66,
    "p_test_within_60min": 0.22,
    "horizons_evaluated": [12, 36, 60],
    "p_per_horizon": {"12": 0.22, "36": 0.51, "60": 0.66},
    "confluence_with_poc": true,
    "confluence_with_vah_val": false,
    "model_confidence_bucket": "moderate"
  }
]
```

## confidence_notes

Meta-section: which model heads are CALIBRATED today vs DRIFTING.
Length budget: 1 paragraph.

```json
{
  "calibrated_today": ["proximity_12bar", "proximity_36bar",
                        "direction_60bar"],
  "drifting_today": ["reaction_post_touch"],
  "drift_reason": {
    "reaction_post_touch": "calibration_error > 0.08 on last_30_days"
  },
  "overall_brief_confidence": "moderate_to_high",
  "operator_note": null
}
```

* `operator_note` — null by default. Human override field for days
  when something genuinely unusual is happening that the model can't
  see (RBI policy day, geopolitical event, etc.). Use sparingly — the
  whole point is to NOT be an opinion shop.

## yesterday_audit

Closes the trust loop. Length budget: 1 paragraph in human layer.

```json
{
  "yesterday_brief_id": "BRIEF_2026_06_02",
  "predictions_made": 23,
  "predictions_resolved": 21,
  "predictions_pending": 0,
  "predictions_data_gap": 2,
  "hit_rate_by_confidence_bucket": {
    "high": {"n": 8, "hit_rate": 0.875},
    "moderate": {"n": 11, "hit_rate": 0.636},
    "low": {"n": 2, "hit_rate": 0.500}
  },
  "avoid_list_validated": {
    "instruments_avoided": 3,
    "avoid_calls_correct": 2,
    "avoid_calls_wrong": 1,
    "estimated_capital_protected_atr": 4.2
  },
  "directional_bias_results": {
    "NIFTY50": "matched", "BANKNIFTY": "matched",
    "FINNIFTY": "partial_match", "NIFTYMIDCAPSELECT": "matched",
    "SENSEX": "wrong"
  },
  "notable_misses": [
    {
      "symbol": "HDFCBANK",
      "predicted": "p_touch_high_long_journey",
      "actual": "level_not_tested_by_eod",
      "diagnosis": "vol_regime_collapsed_unexpectedly"
    }
  ]
}
```

The `notable_misses` field is the trust mechanism. Most services hide
losses. Exposing them — with diagnoses — is the moat.

## Human renderer rules

Section ordering in the human PDF/email:

1. **One-paragraph TLDR** — generated last, summarises the brief.
2. **Index regime block** — table form, one row per index.
3. **Options suitability per index** — short prose per index.
4. **Sector regime** — one paragraph.
5. **Top watchlist** — table, one row per instrument.
6. **Avoid list** — bullet list.
7. **Confidence notes** — one paragraph.
8. **Yesterday audit** — one paragraph + the notable_misses if any.

Sections 1–8 are mandatory. `key_zones` is JSON-layer-only.

Tone:
* Never "buy", "sell", "go long", "go short" in human prose.
* Use "the model assigns a probability of X to event Y" and "if you
  are positioned in Z, the regime suggests W".
* Use "context" not "signal". Use "regime" not "setup".
* If the model is uncertain, say so explicitly. "The direction model
  is calibrated today but the reaction model is drifting; treat
  post-touch sections with extra caution."

## What v1 does NOT include

* No IV / Greeks (deferred until level-to-strike layer is paying).
* No fundamentals / corporate-event flags (no data source wired yet).
* No social-sentiment / news features.
* No multi-day swing brief (separate product, separate spec).
* No intraday re-render (pre_open only).
* No interactive web dashboard (PDF + email is v1).

## Acceptance for the schema commit (this doc)

1. User reviews schema and confirms it represents the product they
   want to ship.
2. The five top-level fields a customer would see (top_watchlist,
   options_suitability, avoid_list, confidence_notes, yesterday_audit)
   each map to a section of the human PDF.
3. The JSON document, alone, is sufficient to reproduce the human
   PDF deterministically — no information lives outside it.

## Next steps after this spec is approved

1. Implement `liqpool/products/daily_brief.py` — the generator. Reads
   from a fitted bundle, produces the JSON.
2. Implement `liqpool/products/brief_renderer.py` — JSON → PDF/email.
3. Wire outcome logging at JSON-write time (see
   `outcome_logging_schema.md`).
4. Smoke test on a historical date: generate brief for 2026-05-25,
   compare to actual outcomes.
