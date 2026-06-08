# Codex Prompt — Options Vertical Training Run (Stream D)

> Paste this as a single message to Codex. The prompt is self-
> contained — Codex hasn't seen the design conversation that
> produced these modules. Treat it as a briefing for a smart
> colleague who just walked into the room.

---

## §0 Mission

The options vertical (Stream D) is engineering-complete on
`claude/liquidity-pool-backtester-1uskb`. The featurizer, label
generator, OptionsExpectedReturnModel suite, the layered-conviction
executor, the Yesterday-Audit aggregator, the artifact-export
hooks, and the overnight macro scrape are all shipped with 118
new tests + lint clean. **What's left is the actual VPS training
run that produces Gate-1 and Gate-2 numbers.** That is your job in
this session.

The methodology + executor design lives in:
- `docs/options_strategy_methodology.md` (the full strategy)
- `docs/options_executor_layered_conviction.md` (the cascade
  executor amendment, four revisions in a day, latest is the
  truth)
- `docs/MASTER_PLAN.md` §3 Stream D + §5 deviations D11-D15

You don't need to read them cover-to-cover. The TL;DR:

- The model is a LightGBM Huber regressor per (side × tenor × ToD)
  bucket. Target = realized_premium_atr_units_60min after slippage
  and STT.
- Structural priors enforce a cascade DAG via LightGBM's
  `interaction_constraints` + `monotone_constraints`.
- The executor uses a Causal Context Vector → LCS scalar → rule-
  based pre-trade (ENTER/WAIT/SKIP at |LCS| ≥ 0.30 bootstrap) and
  in-trade (HOLD/EXIT_KILL/EXIT_INVALIDATION/EXIT_TARGET/TRAIL_STOP).
- There is NO drawdown floor in production. Layer collapse is the
  only soft-exit. See deviation D13 for the reasoning.
- Gate 1 = "does the model rank predictions?" — Spearman > 0.10
  AND top-decile realized R > 0 AND IV-decile calibration error
  ≤ 0.15 on at least ONE bucket.
- Gate 2 = "does conviction-hold beat naive percent-stop?" —
  conviction-hold mean realized R minus naive-baseline mean
  realized R ≥ +0.20 on a 60-day OOS replay.

If Gate 1 fails, the options vertical pauses; we re-evaluate
features (D.1) before any commercial step. **Do not retry-train
with different hyperparameters in that case.** The methodology
doc commits us to "if Gate 1 fails, fix the features, not the
loss function." That's the discipline.

## §1 Where things live

Branch: `claude/liquidity-pool-backtester-1uskb` at commit
`3cf4fbc` (or later). Latest commit list (`git log --oneline -10`)
should show D.1 → D.7 → D.5a in that order at the top.

Module locations:

```
liquidity_backtester/liqpool/options/
├── __init__.py                  # re-exports the full public API
├── featurizer.py                # D.1 build_options_feature_frame
├── labels.py                    # D.2 add_options_labels
├── model_config.py              # D.3 constraints + hyperparams
├── expected_return_model.py     # D.3 OptionsExpectedReturnModel +
│                                #     OptionsExpectedReturnModelSuite
├── ccv.py                       # D.5 CCV + apply_causal_adjustments
│                                #     + compute_lcs_scalar
├── executor.py                  # D.5 pre_trade_decision +
│                                #     in_trade_decision + kill conds
├── audit.py                     # D.6 compute_executor_audit
├── artifact_export.py           # D.7 push_options_strikes +
│                                #     push_options_executor_audit
├── macro_scrape.py              # D.5a YahooMacroAdapter
└── layer_scores.py              # D.5a compute_layer_scores_from_inputs
```

Tests live under `liquidity_backtester/tests/test_options_*.py`.
**Run them first** before doing anything else:

```bash
cd liquidity_backtester
PYTHONPATH=. python -m pytest tests/test_options_*.py -q
```

You should see ≥ 118 tests passing. If anything fails, STOP and
report the failure — do not start training on a broken tree.

## §2 Pre-flight checks (do all of these)

### §2.1 Yahoo macro endpoint reachability

The macro scrape needs to fetch from Yahoo Finance's free public
endpoint. The development environment was blocked (403 on default
egress IP). On a real VPS this usually works. Test it:

```bash
PYTHONPATH=. python -c "
from liqpool.options import YahooMacroAdapter
snap = YahooMacroAdapter().fetch()
print(snap)
print('sources:', snap.sources)
"
```

**Three possible outcomes:**

- All sources tagged `ok` → macro layer feeds full cascade context.
  Continue normally.
- Some sources tagged `ok`, others tagged with an error class →
  partial coverage. The macro_score function skips NaN components.
  Continue.
- All sources tagged with error class names (`URLError`,
  `HTTPError`, etc.) → Yahoo is firewalled from the VPS. The macro
  layer will return 0 for every row → cascade gate closes every
  bar → executor will SKIP every day. **Report this back so we can
  decide:** swap to a paid feed adapter (Polygon / Twelve Data /
  AlphaVantage at ~$30/month) OR run with macro-blind training
  (the model still fits on the rest of the cascade).

### §2.2 Dependencies + lint

```bash
PYTHONPATH=. python -c "import lightgbm; print(lightgbm.__version__)"
# expect 4.x

PYTHONPATH=. python -m pytest -q   # FULL suite (not just options)
# expect 597+ passed, 0 failed

PYTHONPATH=. python scripts/compliance_lint.py
# expect enforced_failures=0
```

### §2.3 Bundle availability

Identify the latest fitted bundle on the VPS (typically under
`output_models/core25_head_alpha_*`). You need:

- The intraday underlying OHLCV bars per index (NIFTY50, BANKNIFTY)
  for the training window
- The daily Greeks parquet (delta, gamma, theta, vega, IV,
  premium_close) per (strike, date)
- The daily macro frame (India VIX close + USDINR close)
- The intraday option premium bars per (strike, bar) — this is
  what the label generator consumes

If any of those are not in the warehouse for the training window,
STOP and report which.

## §3 The one wiring step that isn't yet automated

The featurizer (D.1) does NOT yet emit the six CCV layer-score
columns (macro_score, regime_score, pool_score, options_score,
micro_score, manipulation_score). The layer-score functions exist
(D.5a) but nothing in the engine calls them and joins the result
onto the feature frame. **You need to add that step.**

Add a small helper to the training script. Roughly:

```python
from liqpool.options import (
    build_options_feature_frame, add_options_labels,
    compute_layer_scores_from_inputs,
    YahooMacroAdapter, empty_snapshot,
)

# 1. Build the base feature frame from D.1
features = build_options_feature_frame(
    underlying_bars=bars_df,
    greeks_daily=greeks_df,
    macro_daily=macro_df,
    strikes=strike_list,
    underlying="NIFTY50",
)

# 2. Add the D.2 labels
labelled = add_options_labels(
    features=features,
    option_bars=option_bars_df,
)

# 3. Per-row, compute the 6 layer scores and join onto the frame
try:
    macro_snap = YahooMacroAdapter().fetch()
except Exception:
    macro_snap = empty_snapshot()

# Vectorised approach: build a list of dicts in one pass
layer_score_rows = []
for _, row in labelled.iterrows():
    scores = compute_layer_scores_from_inputs(
        macro_snapshot=macro_snap,
        # Underlying intraday
        underlying_30m_return=row.get("underlying_30m_return"),
        underlying_5m_return=row.get("underlying_5m_return"),
        # IV / Greeks already on the frame
        iv_percentile_60d=row.get("iv_percentile_60d"),
        theta_per_day_pct=row.get("theta_per_day_pct_yday"),
        dte_trading_days=row.get("dte_trading_days"),
        vega_per_volpoint_pct=row.get("vega_per_volpoint_pct_yday"),
        # Path / regime — fill what the existing bundle exposes
        path_efficiency_30=row.get("path_efficiency_30"),
        direction_changes_30=row.get("direction_changes_30"),
        vol_regime_zscore_20d=row.get("vol_regime_zscore_20d"),
        p_up=row.get("direction_p_up"),  # if available
        # Pool — from the existing proximity model output
        proximity_p_60min=row.get("p_test_within_60min"),
        pool_side_from_spot=row.get("side_from_open"),
        # Manipulation — boolean flags from Stream C detectors
        # (fill from your existing detector output frame; default
        # False if not surfaced per row yet)
    )
    layer_score_rows.append({
        "macro_score": scores["macro"],
        "regime_score": scores["regime"],
        "pool_score": scores["pool"],
        "options_score": scores["options"],
        "micro_score": scores["micro"],
        "manipulation_score": scores["manipulation"],
    })

import pandas as pd
labelled = pd.concat(
    [labelled.reset_index(drop=True),
     pd.DataFrame(layer_score_rows)], axis=1)
```

If you DON'T have any of the L1 features the layer-score functions
want (path_efficiency, p_up, etc.), just pass `None` for those —
the score functions degrade gracefully. The model still trains; it
just has a thinner cascade input.

**Note:** the model trains cleanly even if you skip §3 entirely.
LightGBM treats missing layer-score columns as not-present and the
interaction_constraints / monotone_constraints simply don't bind
on them. Skipping §3 means the model trains on raw L1 features
(VIX, USDINR, IV, theta, etc.) without the cascade summary
columns. Mention which you did in the report.

## §4 The training run

```python
from liqpool.options import OptionsExpectedReturnModelSuite

suite = OptionsExpectedReturnModelSuite(
    min_trades=200,          # methodology doc gate
    val_frac=0.25,           # chronological holdout
    embargo_bars=13,         # 5 * 12 = 60 min label horizon + 1 bar
    label_clip=(-3.0, +5.0), # winsorize TRAIN target only
    seed=41,
)

summary = suite.fit(labelled)

# Print per-head metrics
for label, metrics in summary.per_head.items():
    print(f"\n[{label}]  status={metrics.status}  reason={metrics.reason}")
    if metrics.status != "fit":
        continue
    print(f"  n_train={metrics.n_train}  n_oos={metrics.n_oos}")
    print(f"  mean_predicted_oos={metrics.mean_predicted_oos:+.3f}")
    print(f"  mean_realized_oos ={metrics.mean_realized_oos:+.3f}")
    print(f"  spearman          ={metrics.spearman_pred_vs_real:+.3f}")
    print(f"  top_decile_R_oos  ={metrics.top_decile_mean_realized:+.3f}")
    print(f"  top_decile_hit    ={metrics.top_decile_hit_rate:.1%}")
    print(f"  decile reliability:")
    for r in metrics.calibration_table:
        print(f"    d{r.decile}: n={r.n} predicted={r.mean_predicted:+.2f} "
              f"realized={r.mean_realized:+.2f} err={r.calibration_error:+.2f}")
    print(f"  IV-decile sub-slice calibration:")
    for r in metrics.sub_slice_calibration_by_iv_decile:
        print(f"    iv-d{r.decile}: n={r.n} err={r.calibration_error:+.2f}")
    print(f"  Top features by gain:")
    for name, gain in metrics.feature_importance_top_20[:8]:
        print(f"    {name:35s} gain={gain:>10.1f}")

print(f"\nGate 1 pass: {summary.gate1_pass}")
print(f"Winning buckets: {summary.gate1_winning_buckets}")
```

Persist `summary` (pickle, JSON, whatever fits the existing bundle
plumbing) — we want to inspect it later.

## §5 Gate 1 verdict — what to do with the numbers

Gate 1 = at least one bucket has:
- `spearman_pred_vs_real > 0.10`, AND
- `top_decile_mean_realized > 0`, AND
- `max(|iv_decile_sub_slice_err|) ≤ 0.15`

The `summary.gate1_pass` and `summary.gate1_winning_buckets`
booleans already compute this. Three outcomes:

- **Gate 1 passes (one or more winning buckets).** Move on to §6
  Gate 2. Report the winning buckets + their numbers.
- **Gate 1 fails but Spearman is close (any bucket > 0.05).**
  Report the numbers. Do NOT retune hyperparameters. The methodology
  doc commits us to fixing features (D.1), not the loss function.
- **Gate 1 fails everywhere with Spearman ≤ 0.05.** The model has
  no rank skill. Report this clearly. We will re-evaluate the
  feature set before any further work.

## §6 Gate 2 — executor value-add (only if Gate 1 passed)

Gate 2 needs trade outcomes from a replay. The audit module
expects the outcome log in the existing PredictionRecord +
ResolutionRecord shape (from `liqpool.products.outcome_log`).

If you don't have a replay harness for the options executor yet,
the simplest path is:

```python
# Pseudo-sketch — adapt to your existing replay infrastructure
from liqpool.options import (
    OptionsExpectedReturnModelSuite,
    ccv_from_row, pre_trade_decision, in_trade_decision,
    KillConditions, InTradeState,
    options_executor_meta, options_skip_meta,
    OPTIONS_EXECUTOR_PREDICTION_TYPE, OPTIONS_SKIP_PREDICTION_TYPE,
    compute_executor_audit,
)
from liqpool.products.outcome_log import (
    OutcomeLogWriter, PredictionRecord, ResolutionRecord)

writer = OutcomeLogWriter(root="/tmp/options_replay_log")

for bar in oos_bars_iter:                # walk the OOS window
    # Build a CCV from the row's layer scores
    ccv = ccv_from_row(bar)
    pre = pre_trade_decision(ccv)
    if pre.action == "SKIP":
        # Compute counter-factual: what the trade WOULD have done
        cf_r = simulate_60min_trade(bar)  # your existing replay primitive
        meta = options_skip_meta(
            bucket_key=bar["bucket_key"],
            lcs_at_decision=ccv.lcs,
            counterfactual_r_atr=cf_r,
            skip_reason=pre.reason,
        )
        writer.write_prediction(PredictionRecord(
            prediction_id=bar["row_id"],
            ...,
            prediction_type=OPTIONS_SKIP_PREDICTION_TYPE,
            regime_tags_at_prediction=meta,
            ...,
        ))
        continue
    if pre.action != "ENTER":
        continue
    # Simulate the in-trade state machine
    trace = simulate_trade_with_executor(bar, ccv, predicted_R)
    meta = options_executor_meta(
        bucket_key=bar["bucket_key"],
        lcs_at_entry=ccv.lcs,
        lcs_at_exit=trace.lcs_at_exit,
        agreeing_layers_at_entry=trace.agreeing_at_entry,
        agreeing_layers_at_exit=trace.agreeing_at_exit,
        min_lcs_during_trade=trace.min_lcs,
        min_realized_r_during_trade=trace.min_r,
        realized_r_at_exit_atr=trace.realized_r,
        naive_exit_r_atr=trace.naive_baseline_r,  # SIMPLE PERCENT-STOP
        was_conviction_hold=trace.held_through_drawdown,
        exit_action=trace.exit_action,
        predicted_target_r_atr=predicted_R,
    )
    writer.write_prediction(PredictionRecord(
        ...,
        prediction_type=OPTIONS_EXECUTOR_PREDICTION_TYPE,
        regime_tags_at_prediction=meta,
        ...,
    ))
    writer.write_resolution(ResolutionRecord(
        ...,
        resolution_method="executor_exit",
        outcome_continuous=trace.realized_r,
        ...,
    ))

writer.commit()
```

Then aggregate:

```python
joined = writer.read_joined(trading_date_ist=...)
report = compute_executor_audit(joined, trading_date_ist=...)

print("Gate 2 verdict:", report.gate2_pass)
print("Naive baseline mean R:", report.gate2_naive_baseline_mean_r)
print("Conviction minus naive:", report.gate2_conviction_minus_naive)
print()
print("Table A (non-conviction trades):")
for r in report.table_a:
    print(f"  {r.bucket}: n={r.n} mean_R={r.mean_realized_r_atr:+.3f} "
          f"win_rate={r.win_rate:.1%}")
print()
print("Table B (conviction-holds):")
print(f"  n_total={report.table_b.n_total} "
      f"mean_R={report.table_b.mean_realized_r_atr:+.3f} "
      f"pct_recovered={report.table_b.pct_recovered_to_profit:.1%}")
for r in report.table_b.by_lcs_at_hold:
    print(f"  lcs={r.lcs_at_hold_bucket}: n={r.n} mean_R={r.mean_realized_r_atr:+.3f} "
          f"pct_recovered={r.pct_recovered_to_profit:.1%}")
print()
print("SKIP counter-factual:")
print(f"  n={report.skip_counterfactual.n} "
      f"mean_R_if_taken={report.skip_counterfactual.mean_counterfactual_r_atr:+.3f}")
```

Persist the report. Run the artifact pusher to upload it to the
website:

```python
from liqpool.options import push_options_executor_audit

# Env vars must be set:
#   WEBSITE_BASE_URL = https://<the deployed website domain>
#   ENGINE_INGEST_TOKEN = <shared secret>
push_options_executor_audit(report, tier="free_signup")
```

## §7 What to report back

A single concise message (Markdown OK):

1. **Pre-flight summary**
   - `pnpm-equivalent`: pytest count, lint count
   - Yahoo reachability: which tickers fetched OK
   - Bundle window: start-end IST dates, n bars total
2. **Featurizer wiring** — did you add the layer-score join (§3),
   or train on raw L1 features only?
3. **Training run results** — the per-head metrics dump from §4
4. **Gate 1 verdict** — pass/fail per bucket + the winning bucket
   numbers if any
5. **Gate 2 verdict** (if Gate 1 passed) — Table A summary, Table B
   summary by LCS-at-hold bucket, SKIP counter-factual mean, Gate-2
   delta vs naive baseline
6. **What's in the bundle now** — file path of the persisted
   summary + audit report
7. **Any failures or anomalies** — be specific, do not hedge

Keep it under ~500 lines. The numbers carry the message.

## §8 Things NOT to do

- Do NOT retune hyperparameters if Gate 1 fails. The methodology
  doc commits us to fixing features instead.
- Do NOT add a drawdown floor to the executor "to be safe." See
  deviation D13. Either we trust the LCS or we don't ship.
- Do NOT widen the win-rate threshold or change the calibration-
  error tolerance to make Gate 1 pass. If it fails, it fails.
- Do NOT push artifacts to the live website until Gate 1 passes.
  Customer-facing surfaces should only see numbers we'd publish
  unedited.

## §9 If something is broken

Branch is `claude/liquidity-pool-backtester-1uskb`. Latest commit
`3cf4fbc` (or whatever's at HEAD). Tests, lint, methodology docs
are all current at that commit. If something is genuinely broken
beyond a small fix, stop the training, push a clear failure
report with the stack trace, and don't try to patch over it.

Good luck. The discipline is honesty.
