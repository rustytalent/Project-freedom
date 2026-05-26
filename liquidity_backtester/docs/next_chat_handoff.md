# Next Chat Handoff

## Repo And Branch

- Repo path on Mac: `/Users/abc/Projects/Project-freedom`
- Work folder for running Python: `/Users/abc/Projects/Project-freedom/liquidity_backtester`
- GitHub repo: `https://github.com/rustytalent/Project-freedom`
- Actual GitHub default branch: `claude/liquidity-pool-backtester-1uskb`
- Current local branch: `claude/liquidity-pool-backtester-1uskb`
- Do not assume `main` is the default branch. The GitHub repo default is `claude/liquidity-pool-backtester-1uskb`, and that is the branch to update for handoffs/code changes.
- Push is working from terminal.

## Important Local Status

The active Phase 4 branch/default branch is
`claude/liquidity-pool-backtester-1uskb`. Recent Phase 4 work has been pushed
to GitHub through:

- `987183b Add Phase 4 Q scale audit`
- `fcab4cd Add Q percentile policy research`
- `1d27a89 Add fast leakage probe foundation`

This handoff is now being updated for the next Workstream 2 artifact-backed
leakage probe: the real Core25 OOS proximity `distance_atr` audit.

Note: GitHub rejected creation of active `.github/workflows/leakage_tests.yml`
because the current token does not have `workflow` scope. The workflow template
exists at `docs/workflows/leakage_tests.yml`; a user/token with workflow scope
can copy it into `.github/workflows/leakage_tests.yml`.

Do not accidentally commit these local/untracked runtime files unless explicitly requested:

- `liquidity_backtester/.env`
- `liquidity_backtester/.venv/`
- `liquidity_backtester/generate_token.py`
- `liquidity_backtester/run_morning.sh`
- `liquidity_backtester/output_*`
- `liquidity_backtester/.DS_Store`

## Phase 4 Current Status

### Workstream 0: Q Scale Audit — COMPLETE

- Report: `reports/phase4_workstream0_q_audit.md`
- Verdict: Branch A, Q is compressed and has real ranking edge.
- Current absolute `Q >= 70%` gate is unreachable on the compressed scale.
- Blended Q range was roughly 23.4% to 55.7%, with about 1.84 percentage
  points of standard deviation.
- Top 5% blended Q produced about +5.91pp strict-respect lift.
- Top 1% blended Q produced about +8.82pp strict-respect lift.
- Important: this proves the quality gate scale was broken, not that the old
  execution policies are profitable.

### Q Percentile Policy Research — COMPLETE

- Report: `reports/phase4_q_percentile_policy_backtest.md`
- Percentile Q improves existing policy cohorts but does not make them
  profitable:
  - `blind_limit`: roughly -0.802R to -0.583R
  - `displacement_confirmed`: roughly -0.515R to -0.408R
  - `reclaim_confirmed`: roughly -1.046R to -0.604R
  - `touch_confirmed`: roughly -0.553R to -0.448R
- Do not rewrite production live gates from this alone. It is research evidence
  that percentile ranking matters, but v1 execution still has negative edge.

### Workstream 2: Fast Leakage Probe Foundation — COMPLETE

- Added `liqpool/leakage_probes.py`.
- Added `tests/leakage/test_fast_leakage_probes.py`.
- Added workflow template at `docs/workflows/leakage_tests.yml`.
- Fast tests cover label-shuffle association, feature timestamp masks, and
  snapshot-level `distance_atr` causality.

### Workstream 2: Artifact-Backed Proximity Distance Audit — COMPLETE

- Script: `analysis/run_phase4_proximity_distance_leakage_audit.py`
- Report: `reports/phase4_proximity_distance_leakage_audit.md`
- Issue sample CSV: `reports/phase4_proximity_distance_leakage_issues.csv`
- Scope: Core25 feature store `output_feature_store/core25_fresh_may25`, OOS
  proximity shards, horizons 78/156/312.
- Rows checked: 7,673,506 across 375 parquet files.
- Result: PASS.
- Distance mismatches: 0.
- Side mismatches: 0.
- Invalid bar/pool indexes: 0.
- Max absolute distance error: 0.
- Important discovered/indexing detail: proximity shards store `pool_idx`
  against the in-memory `train_pools + oos_pools` list, not the local OOS
  pool index. Any future artifact audit must recreate that combined pool table.

### Workstream 2: Artifact-Backed MTF Closed-Bar Audit — COMPLETE

- Script: `analysis/run_phase4_mtf_closed_bar_audit.py`
- Report: `reports/phase4_mtf_closed_bar_audit.md`
- Issue CSV: `reports/phase4_mtf_closed_bar_issues.csv`
- Scope: Core25 feature store `output_feature_store/core25_fresh_may25`,
  higher timeframes `15m,60m,180m,1D,1W`, OOS direction decision rows.
- Status: WARN, with `0 ERROR`, `50 WARN`, `125 INFO`.
- Intraday higher-TF aggregations reproduced exactly from 5m:
  `15m`, `60m`, `180m`.
- Daily/weekly data-generation warnings:
  - `1D`: 50 stored rows not present in the 5m-derived expected index
    (2 per symbol).
  - `1W`: 50 aggregate value mismatches (2 per symbol).
- The audit also showed `110,000` decision-time join-risk rows: a naive
  backward join on higher-TF bar-open timestamps would select unclosed bars
  for every OOS direction decision. Future MTF joins must enforce:
  `bar_open + timeframe <= decision_ts`.
- This is not a current-code hard failure, but it is a guardrail for Track A/B
  and should be reviewed before adding any new MTF point-in-time features.

### Workstream 2: Pool Availability Replay Audit — COMPLETE

- Script: `analysis/run_phase4_pool_availability_replay_audit.py`
- Report: `reports/phase4_pool_availability_replay_audit.md`
- Issue CSV: `reports/phase4_pool_availability_replay_issues.csv`
- Model report source: `output_models/core25_latest/multi_asset_report.pkl`.
- Full metadata coverage:
  - Pool rows audited: `237,364`
  - Contributor rows audited: `3,696,717`
  - Pool sets: `final`, `train`, `oos`
- Sampled detector replay:
  - Samples per symbol: `1`
  - Checked/misses/skipped: `25 / 0 / 0`
- Result: PASS, `0 ERROR`, `0 WARN`.
- Meaning: no saved pool was consumed before its contributors were known, no
  contributor was known before its source bar close, no touch/break label started
  before pool availability, and every sampled pool was reproduced from detector
  replay truncated at `available_at`.
- Full replay of every pool remains expensive on Mac because each replay reruns
  the detector stack over truncated history. Increase `--samples-per-symbol` for
  slower local/cloud validation.

### Track A Gate: Proximity Distance-Bucket AUC — COMPLETE

- Script: `analysis/run_phase4_track_a_proximity_bucket_audit.py`
- Report: `reports/phase4_track_a_proximity_bucket_audit.md`
- Metrics CSV: `reports/phase4_track_a_proximity_bucket_audit.csv`
- Source: saved model report `output_models/core25_latest/multi_asset_report.pkl`
  and feature store `output_feature_store/core25_fresh_may25`.
- Decision: `VIABLE`.
- Primary h=78 tradeable bucket `3-8 ATR`:
  - Rows: `124,632`
  - Base touch rate: `12.4%`
  - AUC: `0.835`
  - Top-decile actual touched: `46.9%`
  - Lift over bucket base: `+34.55pp`
- Longer horizons also remain strong:
  - h=156 `3-8 ATR` AUC: `0.840`
  - h=312 `3-8 ATR` AUC: `0.839`
- Meaning: Track A is not just a trivial "pool is already close" effect. P_touch
  remains predictive in the tradeable pre-touch zone. This does not prove
  profitability; it clears the gate to test direction conditionality and then
  execution under a realistic simulator.

### Track A Gate: Direction-Conditional Audit — COMPLETE

- Script: `analysis/run_phase4_track_a_direction_conditional_audit.py`
- Report: `reports/phase4_track_a_direction_conditional_audit.md`
- Metrics CSV: `reports/phase4_track_a_direction_conditional_audit.csv`
- Source: saved model report `output_models/core25_latest/multi_asset_report.pkl`
  and feature store `output_feature_store/core25_fresh_may25`.
- Decision: `VIABLE`.
- Full OOS direction baseline:
  - Direction rows: `22,000`
  - p_up AUC: `0.621`
  - Accuracy at 0.50: `58.4%`
- Broad Track A subset:
  - Segment `distance 2-8 ATR`
  - Rows: `180,790`
  - Pool-relative direction AUC: `0.715`
  - Accuracy at 0.50: `67.1%`
  - Top-quartile confidence accuracy: `81.8%`
- Primary strict high-proximity subset:
  - Segment `P_touch>=0.85 & distance 2-8 ATR`
  - Rows: `173`
  - Pool-relative direction AUC: `0.724`
  - Accuracy at 0.50: `70.5%`
  - Top-quartile confidence accuracy: `100.0%`
  - Long/short rows: `148 / 25`
  - Long/short AUC: `0.723 / 0.673`
- Caution: the strict high-P_touch subset is small and long-heavy. The future
  sweep should include broader thresholds such as `P_touch>=0.80` and distance
  bands like `2-8`/`3-8`, then use DSR/multiple-testing correction.
- Meaning: Direction survives Track A conditioning. The pre-touch thesis has now
  passed the proximity and direction gates, but profitability is still unproven.

### Next Recommended Step

Continue Workstream 2 before Track A/B strategy work:

1. Build Execution Simulator v2 before any large parameter sweep:
   - 1m intra-bar path resolution.
   - adverse-selection-aware fills.
   - state-dependent slippage.
   - itemized Zerodha intraday equity cost stack.
   - pre-touch directional simulator hook.
2. Then run the Track A pre-touch sweep only under realistic execution/costs.
3. Keep daily/weekly MTF warnings in mind before adding any new MTF joined
   features.

## Recent Uploaded Commits / Completed Work

- `b9bec66 Add post-touch reaction model alerts`
  - Added Phase 3B reaction confirmation alerts in predict/live output.
  - Writes `reaction_alerts.csv` when recent touched pools can be scored.
  - Not pushed at the time this handoff was edited unless `git status` says otherwise.

- `9167860 Add execution mode backtest reports`
  - Added Phase 2B execution-mode historical PnL reports.
  - Writes `execution_backtest_summary.csv` and `execution_backtest_trades.csv`.

- `7c1df5b Add direction-aware execution reporting`
  - Added Phase 2C UP/DOWN execution reporting and direction-coded R fields.
  - Added `displacement_confirmed` execution mode.
  - Pushed to `claude/liquidity-pool-backtester-1uskb`.

- `cd9e43f Add reaction-aware execution cost gates`
  - Added Phase 2A reaction-aware live gates and Zerodha equity friction.

- `af76f8a Add Core25 feature-store training path`
  - Added Core25 local universe and feature-store-first training path.

- `b198e2e Update handoff with scaling roadmap`
  - Recorded the Core25/local-vs-cloud scaling plan.

- `fc35a0f Add parallel asset checkpoints`
  - Added `--asset-workers`, `--checkpoint-dir`, and `--resume`.
  - Per-symbol parquet training can run in parallel with deterministic final merge order.
  - Completed symbols are saved as pickle checkpoints, so a killed run can resume without repeating all asset work.
  - Worker processes cap BLAS/OpenMP thread counts to avoid oversubscribing the M4 laptop.

Phase 2A reaction-aware gate work:

- Reframed live decision from static pool "quality" into a liquidity lifecycle:
  - `P_touch`: probability price reaches the pool/liquidity magnet.
  - `P_respect`: pre-touch calibrated quality prior.
  - `P_reaction`: post-touch reaction prior, shrunk from historical reaction buckets.
  - `P_trade`: `P_touch * P_reaction`, used as an alert/actionability measure.
- Added an explicit Zerodha equity cost model in `liqpool/costs.py`:
  - Intraday brokerage: 0.03% or ₹20 per executed order, whichever is lower.
  - Delivery brokerage: ₹0.
  - Intraday STT: 0.025% sell side.
  - Delivery STT: 0.1% buy and sell.
  - Exchange transaction charge, SEBI fee, stamp duty, GST, delivery DP charge, and slippage assumptions are all parameterised.
  - Official source used: `https://zerodha.com/charges/`, checked 2026-05-25.
- `examples/multi_asset_run.py` now has Phase 2A execution/friction flags:
  - `--execution-mode blind_limit|touch_confirmed|reclaim_confirmed`
  - `--cost-product intraday|delivery`
  - `--slippage-bps`
  - `--cost-quantity`
  - `--gate-min-post-touch-strict`
- Live gate now blocks trades when:
  - post-touch bucket sample is too small,
  - post-touch strict reaction rate is below threshold,
  - REJ factor is not validated,
  - cost-adjusted net expectancy is <= 0.
- `post_touch_events.parquet` rows now include Phase 2 reaction labels/features:
  - `reaction_label`, `reaction_family`
  - `is_hard_reject`, `is_sweep_reclaim`, `is_absorption`, `is_fail_continue`, `is_no_signal`
  - `reclaim_success_label`, `break_continuation_label`, `mae_minus_mfe_atr`
- Important conceptual change:
  - High `P_touch` is now treated as "price may come here", not as "take a trade".
  - The default live execution mode is `reclaim_confirmed`, so the plan should arm alerts and wait for touch/reclaim/displacement confirmation instead of blind limit entries.

Phase 2B execution-backtest work:

- Added `liqpool/execution_backtest.py`.
- It replays historical pool opportunities and compares:
  - `blind_limit`
  - `touch_confirmed`
  - `reclaim_confirmed`
- Each simulated trade includes:
  - entry/exit timestamps, entry, stop, target, exit reason
  - gross PnL, Zerodha/slippage cost, net PnL, net R
  - reaction label, MAE/MFE, factor, TF count, sector, symbol
- `examples/multi_asset_run.py` now writes:
  - `execution_backtest_summary.csv`
  - `execution_backtest_trades.csv`
- Console prints execution-mode profitability metrics:
  - trades, win rate, net expectancy, net R, profit factor, max drawdown.
- New CLI flags:
  - `--skip-execution-backtest`
  - `--execution-backtest-split oos|train`
- This is still a first-pass simulator. It uses conservative same-bar handling: if stop and target are both hit in the same candle, stop wins. Next likely improvement is to use 1-minute post-touch replay for more precise intrabar ordering.

Phase 2C direction-aware execution reporting / logic upgrade:

- Keep profitability and direction separate:
  - `net_r`: profit/loss R. Positive means profitable; negative means loss.
  - `direction`: `UP` for long/buy setups, `DOWN` for short/sell setups.
  - `direction_sign`: `+1` for UP, `-1` for DOWN.
  - `directional_net_r`: sign shows direction (`+` up, `-` down), magnitude is `abs(net_r)`.
- Live expectancy also carries:
  - `net_expectancy_r`: profit/loss expected R after costs.
  - `directional_net_expectancy_r`: direction-coded expected R; positive means UP/long setup, negative means DOWN/short setup. Do not use this as profitability.
- Added `execution_backtest_by_direction.csv`.
- Execution summary now reports UP/DOWN trade counts and separate UP/DOWN net expectancy R.
- Added new stricter execution mode:
  - `displacement_confirmed`
  - waits for a sweep/close-through, then requires a close back beyond the pool boundary in the trade direction before entering.
- CLI `--execution-mode` now accepts `displacement_confirmed`.
- `live_gate_decisions.csv` and `live_plan.json` now include `direction`, `direction_sign`, and `directional_net_expectancy_r` for tradeable/live candidates.
- Bug fix:
  - historical execution replay now uses each asset's optimized `final_cfg`, not only the top-level CLI config. This keeps horizons, reclaim windows, ATR settings, and tester thresholds consistent with the fitted symbol setup.
- Interpretation rule:
  - Do not read a negative `directional_net_r` as a losing trade by itself. Use `net_r`/`net_pnl` for profitability and `direction`/`directional_net_r` for long-vs-short mapping.

Phase 3A post-touch reaction model work, started after Phase 2C:

- Added standalone post-touch reaction model suite:
  - Module: `liqpool/reaction_model.py`
  - Targets:
    - `strict_reaction` from `strict_respect_label`
    - `reclaim_success` from `reclaim_success_label`
    - `break_continuation` from `break_continuation_label`
- Feature-store reaction events now include `pt_*` confirmation-window features from the first few bars after touch:
  - touch-bar range/body/volume ratio
  - close-through magnitude
  - max close-through and reaction over first 3/6 bars
  - close-back-inside/reclaim features
  - wick rejection ratio
- The reaction model trains only in feature-store train mode, after `reaction_events` shards are written.
- New artifacts:
  - `reaction_model_report.csv`
  - `reaction_model_calibration.csv`
  - `reaction_feature_importance.csv`
- Important interpretation:
  - This is a **post-touch confirmation model**, not a pre-touch signal.
  - Current live pre-touch candidates still use the conservative bucket-shrunk `P_reaction` prior.
  - Next step is to add a post-touch/paper-trading workflow that scores an event after touch candles print.

Phase 3C Consistency Layer v1 work, started after Opus review:

- Opus review pushed the roadmap toward leakage/execution foundations before more alpha layers.
- Added leakage audit module:
  - Module: `liqpool/leakage_audit.py`
  - Checks bar index integrity, required OHLCV columns, contributor `known_at`,
    higher-timeframe closed-bar timing, pool `available_at`, label/touch timing,
    sampled detector replay, and whether effective embargo covers the maximum
    active label horizon.
  - `--embargo-bars` is now only the requested minimum. Default strict behavior
    computes `effective_embargo_bars = max(requested_embargo_bars, max_active_label_horizon)`.
  - `max_active_label_horizon` currently comes from the quality horizon and
    proximity horizons, typically maxing at `312` bars.
  - `--allow-short-embargo` exists only for debug/legacy comparison.
  - Leakage `ERROR` rows are run-blocking by default after artifacts are written.
    Use `--allow-leakage-errors` only for debugging.
  - Sampled replay audit is controlled by `--replay-audit-samples` and
    `--replay-audit-severity warn|error`; default replay misses are warnings so
    the detector can be audited before making replay strict.
  - Writes `leakage_audit.json` and `leakage_issues.csv`.
- Added execution-policy label module:
  - Module: `liqpool/policy_labels.py`
  - Emits one row per pool-policy pair for `blind_limit`, `touch_confirmed`,
    `reclaim_confirmed`, and `displacement_confirmed`.
  - Keeps no-trade rows so gated/live strategy evaluation cannot hide selection bias.
  - Adds first-class supervision fields:
    - `policy_target_return_r`
    - `policy_target_win`
    - `policy_target_trade_generated`
    - `policy_target_censored`
  - In train mode, policy labels are also written into the feature store under:
    `policy_labels/mode=<mode>/split=<split>/part.parquet`.
  - Writes `execution_policy_outcomes.parquet` and `policy_label_summary.csv`.
- Console Phase 3C block now reports:
  - leakage status/error/warning counts
  - requested vs effective embargo and max active horizon
  - replay audit checked/miss/skipped counts
  - best/worst policy mean R
- `research_summary.json` and `live_plan.json` include:
  - `consistency_status`
  - `leakage_audit`
  - `policy_label_summary`
  - requested/effective embargo metadata
- Important interpretation:
  - Reaction taxonomies remain diagnostics.
  - Supervision should move toward policy-specific realized net return/R labels.
  - `P_touch * P_reaction` should eventually be replaced by a final-stage trade model
    that learns dependence between touch, direction, respect, reaction, regime, and costs.

Phase 3D policy outcome model work:

- Added research-only executable policy outcome modeling:
  - Module: `liqpool/policy_model.py`
  - Class: `PolicyOutcomeModelSuite`
  - Trains one small LightGBM + isotonic classifier per execution mode.
  - Target is `policy_target_win` on generated trades only.
  - Uses pool structure/timing metadata and excludes realized outcomes, MAE/MFE,
    `bars_to_touch`, entry/exit prices, and barrier fields from model features.
- `examples/multi_asset_run.py` now has:
  - `--skip-policy-model`
  - `--policy-model-min-trades`
- In train mode, after policy labels are generated:
  - builds train/oos policy-label tables,
  - trains policy outcome models per mode,
  - evaluates on OOS generated trades,
  - refreshes the saved model bundle so future predict/shadow-live work can access
    `report.policy_model`.
- New Phase 3D artifacts:
  - `policy_model_report.csv`
  - `policy_model_calibration.csv`
  - `policy_model_feature_importance.csv`
  - `policy_model_oos_predictions.csv`
- Important interpretation:
  - This is **not** used by live gates yet.
  - The immediate purpose is model selection: does the top-ranked slice of a real
    execution policy improve OOS net R after costs compared with the full policy?

- `c3b53a4 Support per-symbol parquet folders`
  - `ParquetProvider` can read both all-symbol parquet files and folder-style per-symbol files such as `resampled/5m/TCS_5m.parquet`.
  - This matches the user's Google Drive warehouse layout.

- `39b7461 Add parquet universe training pipeline`
  - Added the large India universe config, parquet provider, train/predict modes, model artifact save/load, and structured live outputs.

- `280516d Add causal momentum quality features`
  - Added causal return/momentum/range/trend-into-pool features for the quality model.

- `3ac1333 Record real-data baseline results`
  - Captured the real yfinance baseline readout before parquet migration.

- `008129d Add sector-aware MoE quality model`
  - Added `SectorMoERespectModel`
  - Global LightGBM model + sector experts
  - Initial hard sector routing + 70/30 sector/global blend
  - Multi-asset training uses sector MoE

- `02040e1 Add sector regime execution filter`
  - Added richer sector regime scoring from 5d/20d/60d sector returns
  - Added `ALIGNED`, `AGAINST`, `NEUTRAL` sector execution logic
  - Weak setups against a strong sector regime are blocked unless Q is high enough
  - Wired into `examples/live_run.py` and `examples/multi_asset_run.py`

- `7b79238 Add purged validation and dynamic quality gates`
  - Added purged/embargoed quality-model validation with logged fold stats
  - Added configurable `regularization_preset`, including `conservative_finml`
  - Added train-vs-validation quality model metrics so fit gap is visible
  - Replaced fixed 70/30 sector blend with dynamic sector shrinkage weights
  - Added Phase 3A OOS prediction audit artifacts for global/sector/blended Q
  - Added quality and proximity distance-binned metrics
  - Added post-touch reaction quality diagnostics
  - Added strict live gate with `TRADEABLE`, `WATCH_ONLY`, and `REJECTED_WITH_REASON`
  - Updated JSON artifacts with validation, distance, shrinkage, post-touch, and gate data

## Commands To Run

Legacy yfinance smoke from repo root:

```bash
cd /Users/abc/Projects/Project-freedom/liquidity_backtester
PYTHONPATH=. .venv/bin/python examples/multi_asset_run.py
```

Useful comparison run:

```bash
cd /Users/abc/Projects/Project-freedom/liquidity_backtester
PYTHONPATH=. .venv/bin/python examples/multi_asset_run.py --regularization-preset conservative_finml --out output_conservative
```

Primary parquet run command used by the user on Mac:

```bash
cd /Users/abc/Projects/Project-freedom/liquidity_backtester

export DATA_ROOT="/Users/abc/Library/CloudStorage/GoogleDrive-garvitkatyal312@gmail.com/My Drive/kite_indian_market_data"
export RESAMPLED_DIR="$DATA_ROOT/resampled"
export SYMBOLS='HDFCBANK,ICICIBANK,SBIN,AXISBANK,KOTAKBANK,INDUSINDBK,BANKBARODA,PNB,FEDERALBNK,AUBANK,IDFCFIRSTB,TCS,INFY,HCLTECH,WIPRO,TECHM,LTIM,PERSISTENT,COFORGE,MPHASIS,HINDUNILVR,ITC,NESTLEIND,BRITANNIA,DABUR,GODREJCP,MARICO,COLPAL,TATACONSUM,MARUTI,TATAMOTORS,M&M,BAJAJ-AUTO,HEROMOTOCO,EICHERMOT,TVSMOTOR,ASHOKLEY,RELIANCE,ONGC,BPCL,IOC,HINDPETRO,GAIL,BAJFINANCE,BAJAJFINSV,CHOLAFIN,SHRIRAMFIN,SBICARD,SUNPHARMA,DRREDDY,CIPLA,DIVISLAB,LUPIN,TORNTPHARM,TATASTEEL,JSWSTEEL,HINDALCO,VEDL'

PYTHONPATH=. .venv/bin/python examples/multi_asset_run.py \
  --data-source parquet \
  --data-dir "$RESAMPLED_DIR" \
  --mode train \
  --model-dir output_models/latest \
  --symbols "$SYMBOLS" \
  --asset-workers 3 \
  --checkpoint-dir output_checkpoints/parquet_large \
  --resume \
  --regularization-preset conservative_finml \
  --out output_parquet_train
```

If committing/pushing:

```bash
cd /Users/abc/Projects/Project-freedom
git status --short --branch
git add <files>
git commit -m "<message>"
git push origin claude/liquidity-pool-backtester-1uskb
```

## Verification Already Done

- `python3 -m compileall liqpool examples`
- `git diff --check`
- Synthetic smoke test for purged/embargoed `PoolRespectModel.fit`
- Synthetic smoke test for dynamic `SectorMoERespectModel` shrinkage
- Synthetic smoke test for OOS audit distance buckets and post-touch metrics
- Phase 3A synthetic reaction-model smoke:
  - `ReactionModelSuite` trained strict/reclaim/break targets and predicted event rows.
- Phase 3A real parquet smoke:
  - `TCS,INFY`, `--folds 2 --iters 1 --final-iters 1`
  - temporary feature store/model/output under `/private/tmp`
  - produced `reaction_model_report.csv`, `reaction_model_calibration.csv`, and `reaction_feature_importance.csv`
  - observed OOS reaction AUCs in the smoke around:
    - strict reaction: `0.759`
    - reclaim success: `0.716`
    - break continuation: `0.765`
  - Treat these only as a smoke sanity check, not a production benchmark.
- Phase 3B code path added after Phase 3A:
  - prediction/live output can score recent touched pools with the saved `ReactionModelSuite`
  - writes `reaction_alerts.csv` when there are eligible recent touch events
  - embeds `reaction_confirmation` inside `live_plan.json`
  - prints a concise `POST-TOUCH REACTION CONFIRMATIONS` console section
  - these alerts are confirmation intelligence only and do **not** automatically make a setup tradeable.
- Phase 3C Consistency Layer v1:
  - `python3 -m compileall liqpool examples`
  - `git diff --check`
  - Real TCS/INFY parquet train smoke with temporary `/private/tmp/lb_phase3c_*` paths:

```bash
PYTHONPATH=. .venv/bin/python examples/multi_asset_run.py \
  --symbols TCS,INFY \
  --data-source parquet \
  --data-dir "/Users/abc/Library/CloudStorage/GoogleDrive-garvitkatyal312@gmail.com/My Drive/kite_indian_market_data/resampled" \
  --mode train \
  --model-dir /private/tmp/lb_phase3c_model_smoke \
  --feature-store-dir /private/tmp/lb_phase3c_feature_store_smoke \
  --checkpoint-dir /private/tmp/lb_phase3c_checkpoints_smoke \
  --folds 2 --iters 1 --final-iters 1 \
  --asset-workers 1 \
  --regularization-preset conservative_finml \
  --skip-execution-backtest \
  --replay-audit-samples 2 \
  --out /private/tmp/lb_phase3c_output_smoke
```

  - Produced:
    - `leakage_audit.json`
    - `leakage_issues.csv`
    - `policy_label_summary.csv`
    - `execution_policy_outcomes.parquet`
    - feature-store policy labels at
      `/private/tmp/lb_phase3c_feature_store_smoke/policy_labels/mode=<mode>/split=oos/part.parquet`
  - Smoke leakage audit:
    - status `WARN`
    - `0` errors
    - `54` warnings, all non-blocking label-window warnings near the data end
    - requested embargo `78`, effective embargo `312`, max active horizon `312`, covers `true`
    - sampled replay audit checked `4`, misses `0`, skipped `0`
  - Policy label artifact sanity:
    - `execution_policy_outcomes.parquet` shape was `(11764, 46)`
    - new columns present:
      `split`, `policy_target_return_r`, `policy_target_win`,
      `policy_target_trade_generated`, `policy_target_censored`
  - Predict-mode smoke from `/private/tmp/lb_phase3c_model_smoke` also passed:
    - no retraining / no model save
    - parquet data source only
    - leakage audit still `0` errors, effective embargo `312`, replay misses `0`
  - Smoke policy labels on TCS/INFY OOS were negative for all execution policies:
    - `displacement_confirmed`: mean R `-0.43`, trade rate `15.8%`
    - `touch_confirmed`: mean R `-0.55`, trade rate `46.1%`
    - `blind_limit`: mean R `-0.84`, trade rate `59.5%`
    - `reclaim_confirmed`: mean R `-1.06`, trade rate `19.4%`
  - Interpretation: this validates the artifact path and reinforces the Opus point that
    execution-policy outcomes, not respect taxonomy alone, must become the main training target.
- Phase 3D policy outcome model smoke:
  - Real TCS/INFY parquet train smoke with temporary `/private/tmp/lb_phase3d_*` paths.
  - Same strict effective embargo behavior: requested `78`, effective `312`.
  - Produced:
    - `policy_model_report.csv`
    - `policy_model_calibration.csv`
    - `policy_model_feature_importance.csv`
    - `policy_model_oos_predictions.csv`
  - Artifact sanity:
    - `policy_model_report.csv` shape `(4, 20)`
    - `policy_model_calibration.csv` shape `(16, 7)`
    - `policy_model_feature_importance.csv` shape `(81, 3)`
    - `policy_model_oos_predictions.csv` shape `(800, 10)`
    - saved model bundle contains `report.policy_model`
  - Smoke policy outcome model results:
    - `blind_limit`: OOS AUC `0.559`, base win `25.1%`, top10 R `-0.68`, mean R `-0.84`
    - `displacement_confirmed`: OOS AUC `0.537`, base win `41.2%`, top10 R `+0.01`, mean R `-0.43`
    - `reclaim_confirmed`: OOS AUC `0.523`, base win `19.9%`, top10 R `-1.00`, mean R `-1.06`
    - `touch_confirmed`: OOS AUC `0.561`, base win `37.0%`, top10 R `-0.26`, mean R `-0.55`
  - Interpretation: weak but useful research signal. On this tiny IT smoke, the model
    separated the least-bad policy slices, but only `displacement_confirmed` top decile
    reached near breakeven. Do not use this live until Core25/full OOS confirms it.

Real-data smoke note:

- A tiny yfinance integration run was attempted with:

```bash
PYTHONPATH=. .venv/bin/python -u examples/multi_asset_run.py \
  --symbols HDFCBANK.NS,ICICIBANK.NS \
  --period 30d --folds 2 --iters 2 --final-iters 2 \
  --out output_smoke --regularization-preset conservative_finml \
  --gate-t-today 0.10
```

- It failed because yfinance/Yahoo returned no usable data in that environment.
- The code correctly refused to train on synthetic fallback data.
- Do not treat this as a model failure; rerun when market data fetch works.

Full real-data baseline note, completed 2026-05-21:

- Full default and `conservative_finml` baselines ran successfully against live yfinance data.
- Output folders:
  - `liquidity_backtester/output_default_baseline/`
  - `liquidity_backtester/output_conservative_baseline/`
- `TATAMOTORS.NS` returned no yfinance data in both runs and was safely skipped; both baselines used the same 9-asset basket.
- Shared OOS outcome results:
  - Total OOS tested: 785
  - Pooled OOS broad: 43.2%, Wilson CI 40.3%-46.1%
  - Pooled OOS strict: 40.8%
  - Mean per-asset overfit gap: +15.7%
  - Post-touch touched n: 786, strict respect 40.8%, broken_strong 56.7%
- Default quality model:
  - Validation Brier/log-loss/AUC: 0.2476 / 0.6883 / 0.512
  - Phase 3A blended Brier/log-loss/AUC: 0.2387 / 0.6703 / 0.569
  - Sector weights: AUTO 0.0%, BANKING 6.5%, FMCG 19.3%, IT 0.0%
- Conservative quality model:
  - Validation Brier/log-loss/AUC: 0.2455 / 0.6839 / 0.535
  - Phase 3A blended Brier/log-loss/AUC: 0.2385 / 0.6698 / 0.575
  - Sector weights: AUTO 0.0%, BANKING 10.5%, FMCG 9.3%, IT 0.0%
- Current read: `conservative_finml` is the better baseline on quality validation and Phase 3A blended audit, but the lift is modest. Pooled OOS, post-touch reaction quality, direction AUC, and proximity behavior are unchanged because the same OOS events were evaluated.
- Live gate result:
  - No `TRADEABLE` setups in either run.
  - Default: 6 `WATCH_ONLY`, 79 `REJECTED_WITH_REASON`.
  - Conservative: 5 `WATCH_ONLY`, 80 `REJECTED_WITH_REASON`.
  - Main reason is low Q versus strict `Q >= 70%`; the best Q values are only around 52%. Do not relax the gate just because there are no trades. If changing it, first inspect whether new features can lift post-touch quality rather than simply lowering Q.
- Small code fix after the run:
  - `examples/multi_asset_run.py` verdict text now references `GATE_Q`/the strict live gate instead of the old 55% watch threshold, so the summary no longer contradicts the printed `Q>=70%` gate.

Post-touch feature step, completed after the baseline commit:

- Added causal momentum/context features to `liqpool/featurize.py` for the quality model:
  - `ret_1`, `ret_6`, `ret_24`, `ret_78`
  - `mom_12_atr`, `zscore_close_50`, `range_6_atr`
  - `trend_into_pool_6`, `trend_into_pool_24`
- Intent: help Q separate pools likely to respect after touch from pools likely to break, without changing labels or relaxing live gates.
- Verification completed:
  - `python3 -m compileall liqpool examples`
  - venv synthetic featurizer smoke
  - venv synthetic `PoolRespectModel.fit` smoke with the expanded 52-column feature matrix
- Tiny yfinance smoke with HDFCBANK/ICICIBANK was retried, including escalated network access, but yfinance returned no usable data for both symbols. Treat that as data-feed availability, not a model failure.

Parquet warehouse migration step, started 2026-05-22:

- Added `config/universe_india_large.yaml` with the expanded NSE universe and context-only index symbols.
- Added `ParquetProvider` and `YFinanceProvider` abstractions in `liqpool/data.py`.
  - `--data-source parquet --data-dir <resampled_dir>` reads local `all_5m/all_15m/all_60m/all_180m/all_1D/all_1W` parquet files.
  - Parquet mode never calls yfinance and never creates synthetic fallback data.
  - Missing symbols are skipped cleanly with reasons.
- Added `examples/resample_kite_parquet.py` to resample raw 1-minute Kite/Zerodha parquet into the expected all-symbol timeframe files.
- Wired `examples/multi_asset_run.py`:
  - `--data-source parquet`
  - `--data-dir`
  - `--mode train`
  - `--mode predict`
  - `--model-dir`
- Train mode now saves a pickled `multi_asset_report.pkl` plus metadata under `--model-dir`.
- Predict mode loads the saved model bundle, reads latest parquet candles, rebuilds current pools with saved per-symbol configs, and does zero walk-forward/model training.
- Added structured live outputs:
  - `live_plan.json`
  - `live_gate_decisions.csv`
  - `tradeable_setups.csv`
  - `watchlist.csv`
  - `rejected_setups.csv`
  - `validation_report.csv`
  - `feature_importance.csv`
- Expanded sector mapping and made `sector_of()` normalize both Kite symbols like `TCS` and yfinance symbols like `TCS.NS`.
- Verification completed:
  - `python3 -m compileall liqpool examples`
  - `git diff --check`
  - synthetic local parquet provider smoke from `/private/tmp/lb_parquet_smoke`
  - predict-mode missing-model check fails clearly before training

Parallel parquet training step, completed 2026-05-22:

- Added `--asset-workers`, `--checkpoint-dir`, and `--resume` to `examples/multi_asset_run.py`.
- Per-symbol walk-forward/final-fit work can now run through `ProcessPoolExecutor`.
- Each completed symbol writes a pickle checkpoint under `--checkpoint-dir`; `--resume` loads completed symbols and only runs pending ones.
- Cross-asset merge remains deterministic in the caller's original symbol order before unified model training.
- Worker processes set BLAS/OpenMP thread env vars to 1 to avoid oversubscribing an M4/16GB laptop.
- Verified with a real local parquet smoke:
  - `--symbols TCS,INFY --folds 2 --iters 1 --final-iters 1 --asset-workers 2`
  - Both per-symbol jobs completed in parallel, checkpointed, then unified training/reporting completed.

Large parquet run status, observed 2026-05-23:

- User's local data root:
  - `/Users/abc/Library/CloudStorage/GoogleDrive-garvitkatyal312@gmail.com/My Drive/kite_indian_market_data`
- Data layout confirmed:
  - `raw_1m/` has per-symbol `SYMBOL_1m.parquet` files.
  - `resampled/` has subfolders `5m`, `15m`, `60m`, `180m`, `1D`, `1W`.
- The 58-symbol parquet run reached the unified stage twice and was killed by macOS:
  - Last printed stage: `[[unified]]  training cross-asset direction + proximity models`
  - Terminal result: `zsh: killed`
- This was assessed as RAM pressure, not missing-symbol failure:
  - `LTIM` and `TATAMOTORS` were skipped cleanly before unified training due missing/mismatched 5m parquet paths.
  - Missing symbols reduce the run size and would normally produce a Python traceback if unhandled.
  - `zsh: killed` means the OS terminated the process externally, consistent with memory pressure.
- Disk was not the problem:
  - User reported about 134 GiB free.
- Per-symbol checkpoint work should not be lost:
  - The run used `--checkpoint-dir output_checkpoints/parquet_large --resume`.
  - Rerunning with `--resume` should reuse completed asset checkpoints and go directly toward unified work.
- The current bottleneck is the unified timing/proximity stage:
  - Code builds `all_train_snaps` and `all_oos_snaps` as Python lists across all assets/folds.
  - `ProximityModel.fit()` expands `snapshots x active_pools` into a large list of dict rows, then converts it to a pandas DataFrame.
  - This is much larger than the pool-quality dataset.

Core25 data freshness, checked 2026-05-25:

- User updated Google Drive parquet data.
- Core25 raw and resampled parquet files are current through the May 25, 2026 close:
  - `raw_1m`: latest `2026-05-25 15:29 IST`
  - `5m`: latest `2026-05-25 15:25 IST`
  - `15m`, `60m`, `180m`: latest `2026-05-25 15:15 IST`
  - `1D`: latest `2026-05-25`
  - `1W`: latest `2026-05-22` because weekly bars are week-labeled
- Core25 had 25/25 symbols present with no laggards.
- After daily data updates, run predict mode from parquet before judging any live plan.

## Current Architecture Read

The current code does **not** appear to merge all 5m/15m/60m/180m/1D/1W candles into one giant candle matrix.

Current flow:

```text
per symbol:
  load base 5m dataframe
  load separate higher timeframe dataframes
  detect levels on each timeframe
  project/merge levels into liquidity pools on base timeframe
  train/evaluate pool quality from pool-level rows
```

The memory-heavy flow is different:

```text
per asset, per fold:
  generate snapshots over base 5m time
  each snapshot stores market state plus all active pool touch labels
  append snapshots into global Python lists

then:
  expand snapshots x active pools into proximity rows
  build one large pandas DataFrame
  train/evaluate direction and proximity models
```

Important interpretation:

- Pool-quality training is one row per pool/event and is comparatively small.
- Proximity training is many rows per pool because the same active pool can be observed from many timestamps before touch/break.
- Reducing proximity snapshots does not reduce pool-quality training. It limits the redundant timing/reachability table.
- Proximity predicts `P_touch`, not `P_respect` or profitability. It is alert intelligence, not trade permission.

## Cloud Vs Local Decision

Current recommendation:

```text
MacBook M4 / 16GB:
  development
  parquet validation
  resampling
  small smoke runs
  prediction-only mode
  sampled/memory-safe training

Cloud CPU / high RAM:
  full 58-symbol training
  full proximity table benchmark
  full post-touch event database
  execution backtests
  feature-pruning sweeps
  later sequence-model experiments
```

For the current LightGBM/tabular/parquet pipeline, prioritize CPU RAM over GPU:

- Minimum useful cloud target: 8 vCPU, 64GB RAM, 50GB SSD.
- Better target: 16 vCPU, 128GB RAM, 100GB SSD.
- Storage does not need to be 300GB right now. The user's current parquet files are small; 50-100GB is enough for current runs plus artifacts.
- GPU is not needed yet. It only becomes relevant later for sequence/transformer-style models.

Cheaper cloud options discussed:

- Colab Pro/Pro+:
  - Easiest because data is already in Google Drive.
  - Good for experiments but runtime availability/disconnects can vary.
- Hetzner 64GB/128GB CPU server:
  - Better for stable serious long-running research if the user is comfortable uploading/mounting data and using Linux terminal.
- RunPod:
  - Useful if it offers enough system RAM, but GPU is not the main need at this stage.

Even after architecture improvements, full benchmark runs should eventually move to cloud. The architecture work makes local development and future phases healthier; it does not remove the need for cloud when the user wants full unsampled universe runs.

## Proposed Phase 1.5 Architecture Work

The next high-impact engineering step is a feature-store/chunked-training architecture, not merely "use Polars".

Implementation status as of 2026-05-24:

- Core25 local universe has been added for MacBook development:
  - Config: `config/universe_core25.yaml`
  - CLI: `--universe core25`
  - Symbols: 5 each from BANKING, IT, FMCG, AUTO, PHARMA
- Feature-store foundation has been added:
  - Module: `liqpool/feature_store.py`
  - CLI: `--feature-store-dir`
  - If `--universe core25 --mode train` is used and no feature-store dir is supplied, it defaults to `output_feature_store/core25`.
- The new Mac-safe timing path writes per-symbol parquet shards and trains direction/proximity from feature tables instead of a full-universe in-memory snapshot list:
  - `bars/timeframe=<tf>/symbol=<symbol>/year=<year>/part.parquet`
  - `pools/symbol=<symbol>/<split>.parquet`
  - `quality/symbol=<symbol>/<split>.parquet`
  - `direction/symbol=<symbol>/split=<split>/fold=<n>.parquet`
  - `proximity/horizon=<h>/symbol=<symbol>/split=<split>/fold=<n>.parquet`
  - `reaction_events/symbol=<symbol>/<split>.parquet`
- `DirectionModel.fit_frame()` and `ProximityModel.fit_frame()` now fit from persisted feature rows.
- `evaluate_timing_frames()` evaluates OOS direction/proximity from feature-store rows.
- Note: the quality model still fits from the existing full in-memory pool-level DataFrame in this pass because that table is not the RAM bottleneck. Full quality rows are persisted to the feature store for audit/future table-native training, and no quality rows are sampled away.
- Smart proximity reduction is implemented in shard generation:
  - keep all touch-positive rows
  - keep all rows inside 0-3 ATR
  - keep 50% of 3-5 ATR negatives
  - keep 20% of 5-10 ATR negatives
  - keep 5% of 10+ ATR negatives
- `examples/multi_asset_run.py` now exports:
  - `research_summary.json`
  - `post_touch_events.parquet`
  - `post_touch_report.csv`
  - existing live/validation/feature-importance files

Core25 local run template:

```bash
cd /Users/abc/Projects/Project-freedom/liquidity_backtester

export DATA_ROOT="/Users/abc/Library/CloudStorage/GoogleDrive-garvitkatyal312@gmail.com/My Drive/kite_indian_market_data"
export RESAMPLED_DIR="$DATA_ROOT/resampled"

PYTHONPATH=. .venv/bin/python examples/multi_asset_run.py \
  --universe core25 \
  --data-source parquet \
  --data-dir "$RESAMPLED_DIR" \
  --mode train \
  --model-dir output_models/core25_latest \
  --asset-workers 2 \
  --checkpoint-dir output_checkpoints/core25 \
  --resume \
  --regularization-preset conservative_finml \
  --out output_core25_train
```

Core25 prediction-only template after daily parquet update:

```bash
cd /Users/abc/Projects/Project-freedom/liquidity_backtester

export DATA_ROOT="/Users/abc/Library/CloudStorage/GoogleDrive-garvitkatyal312@gmail.com/My Drive/kite_indian_market_data"
export RESAMPLED_DIR="$DATA_ROOT/resampled"

PYTHONPATH=. .venv/bin/python examples/multi_asset_run.py \
  --universe core25 \
  --data-source parquet \
  --data-dir "$RESAMPLED_DIR" \
  --mode predict \
  --model-dir output_models/core25_latest \
  --execution-mode reclaim_confirmed \
  --out output_core25_predict_latest
```

Phase 2C stricter confirmation test:

```bash
PYTHONPATH=. .venv/bin/python examples/multi_asset_run.py \
  --universe core25 \
  --data-source parquet \
  --data-dir "$RESAMPLED_DIR" \
  --mode predict \
  --model-dir output_models/core25_latest \
  --execution-mode displacement_confirmed \
  --out output_core25_predict_displacement
```

Core idea:

```text
raw 1m parquet
  -> resampled partitioned bars
  -> pool event table
  -> quality feature table
  -> proximity feature shards
  -> post-touch reaction event table
  -> execution trial table
  -> model training / validation
  -> prediction-only live plan
```

Recommended structural changes:

1. Partition bars and derived features by symbol/timeframe/date or year.
   - Example:
     - `features/bars/timeframe=5m/symbol=TCS/year=2024/*.parquet`
     - `features/pools/symbol=TCS/*.parquet`
     - `features/proximity/horizon=78/symbol=TCS/*.parquet`
     - `features/reaction_events/symbol=TCS/*.parquet`
     - `features/execution_trials/mode=reclaim_confirmed/symbol=TCS/*.parquet`

2. Stop keeping full-universe snapshot objects in RAM.
   - Generate proximity rows per symbol/fold/horizon.
   - Write shard parquet immediately.
   - Train by scanning shards, filtering, and collecting only the selected training matrix.

3. Use Polars/DuckDB where it helps.
   - Polars lazy scans and partition pruning can reduce memory.
   - DuckDB can query parquet shards without loading everything.
   - Do not simply replace "giant pandas DataFrame" with "giant Polars DataFrame"; the real fix is shard-first/chunked design.

4. Make proximity reduction explicit and intelligent.
   - Keep all touch-positive rows.
   - Keep all near-pool rows, e.g. 0-3 ATR.
   - Downsample far-away repeated rows.
   - Preserve sector/symbol balance.
   - Use class weights or sample weights if negatives are downsampled.

5. Save intermediate artifacts after each stage.
   - If unified proximity dies, quality model artifacts and feature shards should still be preserved.
   - Avoid repeating expensive per-asset detection/training work.

Expected practical difference between full and smart-reduced proximity:

- Full proximity run is the benchmark.
- Smart reduced proximity should preserve most practical watchlist value because far-away repeated rows are highly redundant.
- Approximate expectation discussed with the user:
  - Full proximity reference quality: 100%.
  - Smart reduced proximity: roughly 95-99% practical touch-timing quality if near/touched cases are preserved.
  - Heavy random reduction can materially hurt calibration and rare-case timing; avoid that.

## Remaining Roadmap / Phases

The user explicitly noted that the current work is only Phase 1 and wants future phases forced into the roadmap. Use this order.

### Phase 1 - Data And Research Engine Foundation

Mostly implemented:

- ParquetProvider and yfinance separation.
- Large India universe config.
- No synthetic fallback in parquet mode.
- Train/predict mode split.
- Model artifact saving/loading.
- Structured live outputs.
- Per-symbol parallelism and checkpoint resume.

Remaining Phase 1 work:

- Fix/inspect missing `LTIM` and `TATAMOTORS` parquet naming/path issue.
- Add memory-safe unified timing controls or the feature-store design above.
- Add a clearer failure path when unified timing is too large for RAM.
- Optionally save partial artifacts after quality/sector training before proximity training.

### Phase 2 - Validation, Calibration, And Gate Discipline

Partially implemented:

- Purged/embargoed validation exists for quality model.
- Dynamic sector shrinkage exists.
- OOS prediction audit exists.
- Strict live gate exists.

Remaining:

- Make purged/embargoed validation mandatory everywhere relevant, not just quality.
- Ensure no random time-series KFold remains in model-selection paths.
- Save richer validation reports for each model family:
  - train rows
  - purged rows
  - embargo rows
  - validation rows
  - train/val AUC, Brier, logloss
  - fit gap
  - calibration bins
- Reject or warn on model configs with excessive train-validation gap.
- Add empirical Bayes / Wilson interval bucket reliability tables everywhere live gates use bucket stats.

### Phase 3 - Post-Touch Reaction Intelligence

Partially implemented. Current code can export `post_touch_events.parquet` from
feature-store reaction shards and includes reaction labels/diagnostics. Phase 3A
now also trains a standalone post-touch reaction model suite from those events
when feature-store train mode is used. Phase 3B makes that model operational in
live/predict output by scoring recent touched pools after a small confirmation
window has printed. After the Opus review, Phase 3C shifted the immediate priority
toward leakage probes and execution-policy labels before adding more model
complexity.

Implemented:

- Export `post_touch_events.parquet`.
- Label events:
  - `HARD_REJECT`
  - `SWEEP_RECLAIM`
  - `ABSORPTION`
  - `FAIL_CONTINUE`
  - `LIQUIDITY_VACUUM`
  - `NO_SIGNAL`
- Use historical post-touch bucket strict rates as a shrunk `P_reaction` prior in live gates.
- Add first-few-bars-after-touch confirmation features with `pt_*` prefixes.
- Train/evaluate model targets:
  - strict reaction
  - reclaim success
  - break continuation
- Save model report, calibration, and feature-importance artifacts.
- Score recent touched pools in train/predict output:
  - CLI flags:
    - `--reaction-feature-bars` default `6`
    - `--reaction-alert-lookback-bars` default `78`
    - `--reaction-confirm-threshold` default `0.60`
    - `--reaction-break-risk-threshold` default `0.60`
  - Outputs:
    - `reaction_alerts.csv`
    - `live_plan.json.reaction_confirmation`
    - `multi_asset_summary.json.reaction_alerts`
  - Actions are informational:
    - `CONFIRM_UP`
    - `CONFIRM_DOWN`
    - `RECLAIM_WATCH_UP`
    - `RECLAIM_WATCH_DOWN`
    - `AVOID_BREAK_CONTINUATION`
    - `WATCH_REACTION`
- First-pass leakage audit:
  - Module: `liqpool/leakage_audit.py`
  - Artifacts: `leakage_audit.json`, `leakage_issues.csv`
  - Checks pool/contributor availability timestamps, label-window sanity,
    OHLCV/index integrity, and active-horizon vs embargo coverage.
- First-pass execution-policy labels:
  - Module: `liqpool/policy_labels.py`
  - Artifacts: `execution_policy_outcomes.parquet`, `policy_label_summary.csv`
  - Keeps no-trade rows and labels realized net R under each explicit execution policy.

Remaining:

- Add sector regime and direction-state-at-touch features into the event table.
- Add 1-minute post-touch replay so confirmation features use finer intrabar ordering.
- Add bar-by-bar detector replay vs batch-generation diff checks.
- Add MTF closed-bar/as-of join audit with explicit higher-timeframe completion timestamps.
- Add final-stage trade model only after leakage/execution artifacts are trusted.
- Turn reaction alerts into a true stateful alert workflow:
  - pre-touch watchlist arms a pool
  - after touch, a follow-up command scores that pool using only candles printed since touch
  - execution entry is allowed only when confirmation, regime, direction, and expectancy align.
- Live plan should be able to arm an alert pre-touch and then require reaction-model
  confirmation after touch before entry.

### Phase 4 - Execution And PnL Engine

Partially implemented through Phase 2A/2B/2C.

Implemented:

- Add execution modes:
  - `blind_limit`
  - `touch_confirmed`
  - `reclaim_confirmed`
- `displacement_confirmed`
- Prefer `reclaim_confirmed` by default for live decisions.
- Simulate Zerodha/friction-adjusted entries with stop, target, and time exit.
- Report profitability metrics:
  - net expectancy per trade
  - win rate
  - avg win/loss
  - profit factor
  - max drawdown
- Output trade-level fills and execution summary CSVs.
- Separate profitability R from direction-coded R.

Remaining:

- Use 1-minute post-touch replay for better intrabar order instead of conservative same-bar handling.
- Add partial TP simulation.
- Use target sizing from historical MFE distribution, not only fixed ATR target.
- Add richer exposure-time, trades/month, per-symbol PnL, and per-sector PnL reports.
- Add confirmation-filter comparisons after the standalone reaction model exists.
- Never call a setup profitable from respect rate alone.

### Phase 5 - Live Decision System

Partially implemented through predict mode and live reports.

Remaining:

- Make predict mode the daily workflow:
  - loads saved models
  - reads latest parquet
  - performs zero training
  - outputs concise plan
- Keep outputs separated:
  - `live_plan.json`
  - `watchlist.csv`
  - `tradeable_setups.csv`
  - `rejected_setups.csv`
- Console should remain concise:
  - verdict
  - tradeable setups
  - watch-only liquidity magnets
  - rejected with reason
  - model health warning
- Explicitly separate:
  - `P_touch`
  - `P_respect`
  - `P_trade`
- High `P_touch` plus weak `P_respect` must remain watch-only.

### Phase 6 - Scaling, Feature Store, And Cloud Benchmarking

This is now a major priority because the Mac was killed twice at unified proximity.

Required:

- Implement Phase 1.5 feature-store/chunking design.
- Add optional memory-safe proximity training mode for Mac development.
- Add cloud full-run instructions.
- Compare:
  - full unsampled proximity on cloud
  - smart reduced proximity on Mac/cloud
  - quality/reaction/execution metrics unchanged or improved
- Add model/data versioning so cloud artifacts can be brought back to Mac for predict mode.

### Phase 7 - Advanced Modeling

Do not jump directly to 7B/30B/120B models.

Current view:

- Bigger parameter count is not the bottleneck.
- The bottleneck is label quality, leakage-safe validation, reaction/execution modeling, and memory architecture.
- For future deep learning, start with small market-specific sequence models:
  - TCN
  - temporal transformer
  - TFT/PatchTST-style candle encoder
  - cross-asset/sector context encoder
- Practical first target is roughly 1M-50M parameters, not 7B.
- GPU becomes useful only in this phase.

## Key User Preferences / Context

- User wants the system to become a disciplined quant research/live-decision engine, not a loose signal generator.
- User does not want to weaken the quality model just to fit the Mac.
- Explain clearly when a reduction affects only the proximity snapshot table and not pool-quality training.
- User is comfortable running terminal commands but wants exact step-by-step commands.
- User may paste long terminal output; answer directly and practically.
- User is planning to move full training to cloud eventually, but wants local Mac development to continue.
- User gave permission to commit this handoff update.

## Current Phase 3 Status

The previous handoff said Phase 3A-E remained. Most of that is now implemented:

- OOS prediction audit: implemented
- Dynamic sector shrinkage replacing fixed 70/30 behavior: implemented
- Live output includes global/sector/blended component fields where available: implemented through `predict_components`
- Safety checks/global fallback: implemented through zero-weight sector shrinkage and fallback reasons
- Conservative quality model regularization preset: implemented
- Distance-binned quality/proximity diagnostics: implemented
- Post-touch reaction diagnostics: implemented
- Strict conditional-edge live gate: implemented
- Phase 3A reaction model suite: implemented
- Phase 3B post-touch reaction alerts: implemented
- Phase 3C Consistency Layer v1: implemented and smoke-tested
- Phase 3D policy outcome model diagnostics: implemented and smoke-tested

## Next Goals

1. Run prediction-only mode from the freshly updated May 25 parquet.
   - Use `--mode predict --universe core25 --data-source parquet`.
   - Start with `--execution-mode reclaim_confirmed`.
   - Optionally compare `--execution-mode displacement_confirmed`.
   - Predict mode must do zero training and should consume `output_models/core25_latest`.

2. Treat `conservative_finml` as the current preferred training preset unless a future run reverses the quality audit.
   - Re-run train only when the historical data window changes materially, after major feature changes, or when model artifacts are missing/stale.
   - Compare pooled OOS broad/strict, mean per-asset overfit gap, quality Brier/log-loss/AUC, post-touch strict respect, and execution net expectancy.
   - Do not judge success from proximity AUC alone; proximity is the reachability engine and distance dominates by design.

3. Tune the live gate only after seeing real OOS bucket counts and execution results.
   - Defaults are intentionally strict: Q >= 70%, T_today >= 50%, `DIR_ALIGN`, distance 0.5-12 ATR, bucket n >= 30.
   - If it rejects everything again, inspect `gate_decisions` and `distance_bucket_metrics` before relaxing thresholds.
   - Trade count can drop; that is acceptable if post-touch quality improves.
   - High `P_touch` remains watch-only unless `P_reaction`, bucket reliability, direction, sector, and net expectancy all pass.

4. Inspect sector shrinkage behavior.
   - Check `sector_shrinkage_report` in `multi_asset_summary.json`.
   - Good behavior: small/noisy sector experts get near-zero weights.
   - Only trust sector experts that beat global fallback on validation loss with reasonable AUC.

5. Use post-touch reaction quality as the main alpha diagnostic.
   - Focus on `post_touch_reaction_metrics`.
   - Look for factor/TF/sector/Q buckets with high strict respect and low broken_strong rate.
   - This should drive future feature or gate changes more than aggregate touch probability.

6. Harden Phase 3C toward strict replay/point-in-time enforcement.
   - Run Core25 train/predict after Phase 3C and inspect `leakage_audit.json`.
   - Any leakage `ERROR` now blocks the run by default after artifacts are written.
   - `embargo_bars < max_active_horizon` should no longer appear unless
     `--allow-short-embargo` is explicitly passed.
   - Inspect `policy_label_summary.csv` and feature-store `policy_labels/*` to see whether
     gates improve explicit policy returns or only reduce sample size.
   - Next hardening step: make replay audit stricter after fixing any detector-causality misses,
     then add point-in-time feature-store join tests.

7. Validate Phase 3D policy outcome models on Core25 before using them anywhere live.
   - Inspect `policy_model_report.csv`.
   - Good sign: top-decile OOS `policy_target_return_r` improves materially over
     full-policy mean R and stays positive after costs.
   - Bad sign: AUC near 0.50 and top-decile R still negative. In that case the model is
     only a diagnostic and should not be promoted.
   - Do not let `policy_model_p_win` enter the live gate until Core25/full-universe OOS
     and shadow-live validation agree.

8. Continue Phase 3 proper: make the reaction model operational after touch.
   - The standalone reaction model suite now exists and trains from `post_touch_events.parquet`.
   - Phase 3B now adds first-pass reaction alerts for recently touched pools.
   - Next harden this into a stateful "pool armed -> pool touched -> score event -> entry allowed/blocked" workflow.
   - Add sector regime at touch, direction state at touch, and eventually 1-minute replay features.
   - Live plan should eventually require touch + confirmation + reaction model agreement.

9. If metrics still show high overfit, the next high-impact options are:
   - Raise `min_sector_train_n` / `min_sector_class_n`
   - Increase `min_data_in_leaf` for `conservative_finml`
   - Reduce feature_fraction/bagging_fraction further
   - Require stronger OOS bucket support before arming trades
   - Add better post-touch labels/features, not more reachability features

## Explicit Non-Goals For Now

- Do not implement fractional differencing yet.
- Do not broadly neutralize features yet.
- Do not remove `distance_atr` from the proximity model.
- Do not treat high proximity AUC as automatic leakage.
- Do not commit runtime/secrets files unless explicitly requested.

## Phase 4 Complete Revised Context

### Do Not Skip

- Workstream 0 blocks everything.
- No live trading confidence until execution PF > 1.0 OOS.
- No Track A/B implementation before Q compression and leakage checks.
- No production gate rewrite until the Q audit report is reviewed.

### Phase 4 Workstream 0 Status

Phase 4 Workstream 0 has now been implemented as analysis-only scripts and a
markdown report:

- `analysis/q_distribution_audit.py`
- `analysis/q_decile_actual_rate.py`
- `analysis/q_top_percentile_lift.py`
- `analysis/q_gate_reachability.py`
- `analysis/run_phase4_workstream0_q_audit.py`
- `reports/phase4_workstream0_q_audit.md`

Current audit verdict:

- Branch A: Q is compressed and has edge.
- Current `Q >= 70%` absolute live gate is unreachable.
- Blended Q range: about 23.4% to 55.7%.
- Blended Q standard deviation: about 1.84 percentage points.
- Base strict-respect rate: about 43.5%.
- Exact top 5% lift: about +5.91pp.
- Exact top 1% lift: about +8.82pp.

Interpretation:

- The quality model is not expressing confidence on a 0-100% gate scale.
- The 70% Q gate is mathematically broken for the current conservative model outputs.
- Q still carries useful rank signal in the tail.
- Do not make this a live-trading change yet. The next step is a
  percentile-gate research backtest and then execution validation.

Branch A follow-up status:

- Added `analysis/q_rescale.py`.
- Added `analysis/run_phase4_q_percentile_research.py`.
- Generated `reports/phase4_q_percentile_policy_backtest.md`.
- Generated `reports/phase4_q_percentile_policy_backtest.csv`.
- Finding: Q percentile ranking improves several existing-policy cohorts, but does
  not make any current v1 execution policy profitable after costs.
- Best top-1% cohort deltas:
  - `blind_limit`: mean R improves from about -0.802 to -0.583.
  - `displacement_confirmed`: mean R improves from about -0.515 to -0.408.
  - `reclaim_confirmed`: mean R improves from about -1.046 to -0.604.
  - `touch_confirmed`: mean R improves from about -0.553 to -0.448.
- Interpretation: the Q scale bug is real and Q ranking is useful, but percentile
  Q alone does not rescue the current execution policies. Continue with leakage
  probes and execution simulator v2 before any live gate rewrite.

Workstream 2 leakage probe status:

- Added fast leakage probe utilities in `liqpool/leakage_probes.py`.
- Added CI-runnable tests in `tests/leakage/test_fast_leakage_probes.py`.
- Added GitHub workflow template in `docs/workflows/leakage_tests.yml`.
- Active `.github/workflows/` could not be pushed from this environment because
  the current GitHub token lacks `workflow` scope. A user/token with workflow
  scope can copy the template into `.github/workflows/leakage_tests.yml`.
- Added `reports/phase4_workstream2_leakage_probes.md`.
- Current v1 probes cover label-shuffle association destruction, generic
  `as_of_ts <= decision_ts` future-mask checks, snapshot `distance_atr`
  recomputation, and a future-mutation regression for snapshot pool distances.
- This is not full Workstream 2 completion yet. Remaining slow probes:
  retrain models on shuffled labels, full feature metadata future-mask audit,
  full MTF replay, full pool availability replay, and real OOS proximity
  distance-at-decision audit.
- Do not start Track A sweep before the real distance/proximity leakage audit.

### Earlier Opus Review — Verbatim

# Review

## Top-line assessment

The document reads like someone who has been burned by their own backtest and is doing the right thing about it. The decomposition into P_touch, P_respect/quality, P_reaction, and P_trade is conceptually defensible, and the explicit refusal to equate "respect rate" with profitability already puts you ahead of most retail/semi-pro frameworks. The main weaknesses, in order of how badly they will hurt you: (1) the labeling abstraction is one level too high — it should be tied to a concrete execution policy, not to event-type taxonomies; (2) several plausible leakage paths in pool availability, MTF alignment, and outcome-window embargo that the document acknowledges in principle but does not show concrete mechanics for; (3) the execution simulation is mentioned but not specified, and the strategy class (liquidity sweeps) is exactly the class where naive slippage assumptions blow up. Until execution and leakage are clean, every model-quality result is uninformative.

## Leakage — where I'd look first

The single most common source of leakage in this class of system is **pool availability time**. A swing high that requires N bars of right-side confirmation is *not* knowable at the swing bar — it is knowable at the swing bar plus N. Equal-highs/lows clusters and order-block-like structures often need re-confirmation as new pivots form. If your pool detection function is run over a historical window and produces a pool with `formation_ts`, the model must consume that pool with availability `formation_ts + confirmation_lag`, where the lag is detector-specific. I would force this through the type system: every pool object carries `available_from_ts`, and any feature/label evaluation that touches the pool before that timestamp is an error, not a warning. Add a fuzz test that re-runs detection on truncated history at each `available_from_ts` and asserts that the pool would have been produced. If it wouldn't, the lag is wrong.

The second is **multi-timeframe join semantics**. At a 5-minute decision bar, the 60m/180m/daily/weekly features must reflect the *last fully closed* higher-TF bar, not the currently forming one. `merge_asof` with `direction='backward'` and a tolerance equal to the higher-TF period, joined on completion timestamp (open + period), is the only safe pattern. If you ever resample with `.resample('60min').last()` and then forward-fill, you have probably leaked.

Third, **outcome-window embargo**. Walk-forward with purging is necessary but not sufficient — the embargo must be at least as long as the maximum outcome horizon used in any label in the training fold. If the post-touch reaction window is, say, 24 bars, and the proximity-touch horizon is 200 bars, the embargo for the joint model is 200, not 24. A frequent bug is configuring one embargo size for "the system" rather than per-label.

Fourth, **volatility normalization**. "Width normalized by volatility" must use trailing-only realized vol with a strict cutoff at the decision timestamp. `rolling().std()` in pandas is fine if you're sure the window is right-closed and shifted; many implementations subtly center the window or include the current bar.

Fifth, **pool revision / replay determinism**. If you regenerate pools from a longer history (e.g., for a new backtest range), do you get the same pools at the same `available_from_ts` as you did before? If not, your detector is non-causal. The cleanest test: produce pools incrementally bar-by-bar from t=0 and compare against batch-produced pools — they must be byte-identical.

Sixth, **label contamination across the basket**. Cross-sectional features (sector aggregates, sector regime, index regime) computed from constituents that include the symbol being predicted produce a mild form of self-leakage. Either exclude the focal symbol from its own sector aggregate, or be explicit that you accept this and bound its impact.

Seventh, and this is subtle: the **direction model and the quality model likely share features and training periods**. If they do, treating their outputs as "two independent confirmations" in the trade gate is wrong — their errors are correlated. Either model the joint, or measure the correlation of their errors and apply a credibility haircut.

I would build a leakage probe suite that runs on every model release: (a) shuffle 10% of labels and confirm metrics degrade proportionally; (b) replace each feature in turn with white noise and confirm the contribution rankings are sane; (c) shift labels backward by one bar and confirm a noticeable performance gain (proof that strict causality matters); (d) bar-by-bar replay vs batch generation diff. Any of these failing silently is a leakage smell.

## Labeling — the abstraction is one level too high

Categorical pool labels ("respected strongly", "swept and reclaimed", "weak respect", "broken strongly", "no signal") are a *human description of what happened*, not a *machine-supervisable function of the trade outcome you actually care about*. The boundaries between these classes are inevitably arbitrary thresholds you'll tune, and they couple labeling decisions to model performance in ways that look like alpha when they're really label hacking.

The cleaner framing is the triple-barrier method, applied **conditional on an explicit entry policy**. Define a small set of policies (blind-limit, touch-confirmed, reclaim-confirmed, displacement-confirmed). For each policy, each pool either generates a trade or doesn't. For trades, the label is the realized vol-normalized return at the first of: take-profit, stop-loss, or time barrier. Now your models predict the *return distribution under policy P*, which is the thing that determines PnL. A "sweep and reclaim" event is then automatically not a failure for the reclaim-confirmed policy if the reclaim trade was a winner. You stop arguing about taxonomy.

A few specifics: use vol-normalized targets/stops (e.g., k × ATR) and report both the discrete win/loss and the continuous MFE/MAE distribution. For pools that don't get touched in horizon, treat them as **censored** in a survival-analysis sense rather than as a failed event — the Cox or AFT framing for time-to-touch is more honest than a binary "touched within H bars" label and stops you from picking H arbitrarily. The MFE/MAE post-touch should also be reported with time-to-MFE; a 1R move that took 3 bars and a 1R move that took 30 are not the same trade even if your stop didn't hit.

One implementation point: if you keep categorical labels for diagnostics, derive them *from* the continuous outcomes rather than computing them independently from the price action. That guarantees consistency.

## Validation

Walk-forward with purge and embargo is the right family. A few additions:

Use **combinatorial purged cross-validation (CPCV)** rather than single-path walk-forward for model selection. You get many backtest paths from the same data, you can compute path-level Sharpe distributions, and you can apply Bailey/López de Prado's **Probability of Backtest Overfitting** and **Deflated Sharpe Ratio** to penalize for the number of configurations you've tried. Without DSR-style deflation, any system that searches over thresholds, features, or sector definitions will produce an inflated headline Sharpe — typically by a factor of 2–3× for moderate search.

The embargo should equal the max outcome horizon active in the fold, not a global constant. Report `train_end_ts`, `embargo_end_ts`, `validation_start_ts` explicitly for every fold and reject configs where the embargo was shorter than any active horizon.

Report not only AUC and Brier but **per-decile lift** at the operating threshold you'd actually use live, with Wilson intervals. AUC can be high while the top decile (where you actually trade) is uncalibrated noise.

For the cross-sectional dimension (multiple symbols), do not pool symbols into one validation set without checking that the in-sample/out-of-sample split is **strictly temporal across all symbols simultaneously**. If symbol A's validation period overlaps symbol B's training period and there is index/sector co-movement, you have a leakage path that doesn't show up in per-symbol diagnostics.

Finally, run a **null-model baseline**: random pools, mid-bar levels, levels chosen by simple ATR offsets. Your system needs to beat these by an OOS margin that survives DSR, or the "edge" is in the detector's selection of where to look rather than in any predictive signal.

## Calibration

Brier and reliability diagrams are necessary but not sufficient. Specific recommendations:

Fit calibration with **isotonic regression** on a held-out calibration fold (separate from train and validation), not on the validation set you report. If data is too thin for isotonic, use Platt scaling. Re-fit per regime (high-vol vs low-vol, trending vs ranging index) if you observe reliability differences across those slices — single-curve calibration averages over regimes and will look fine in aggregate while being miscalibrated where you trade.

For low-sample buckets, apply **empirical Bayes shrinkage** to the bucket's empirical base rate (beta-binomial conjugate is fine for Bernoulli outcomes). The shrinkage prior should be the parent population (e.g., the global model's overall rate for that label), and the shrinkage weight should fall out of the bucket's prior and posterior variances, not be hand-tuned. Wilson intervals are good for displaying uncertainty; they should not be the basis for the live gate. The right gate is "posterior expected utility under the calibrated, shrunk probability."

**Calibration drift** is the silent killer. Track rolling calibration on a 60–90 day window OOS and alert if Brier deteriorates or the reliability curve tilts. Live gates should be soft-disabled when drift exceeds a threshold.

For the live decision specifically, consider **conformal prediction** for prediction intervals on the post-touch return. It gives distribution-free coverage guarantees, which is a stronger property than calibrated point probabilities when the loss function is asymmetric.

## Execution realism — the section I'd push hardest on

Liquidity-sweep strategies are exactly the strategies where naive execution assumptions produce phantom alpha. Specifically:

**Adverse selection on limit orders.** A blind limit resting at a known liquidity level fills you in two cases: the level holds (you get filled by traffic that doesn't continue) and the level fails (you get filled because price is going through). Conditional on fill, the distribution of outcomes is heavily skewed toward "level failed" because that's the regime that generates volume at the level. Any simulation that assumes "if low ≤ limit price ≤ high during bar X, you got filled at limit price" overstates fill quality. At minimum, model fill probability as a function of how far price went *through* the limit; ideally, model fills only when price trades past the limit by some buffer, and accept that you may not have been at the front of the queue.

**Slippage correlated with the signal.** When pools get swept, spreads widen and volatility spikes. Constant slippage in basis points understates true cost during the exact bars your strategy trades. Model slippage as `f(short-term realized vol, distance from level, time of day)` and calibrate it to actual broker fills if you have any (paper trading on a real venue is the cheapest way to gather this).

**Intra-bar path ambiguity.** On a 5m bar with stop and target both inside the bar's H/L range, which hit first is undefined without finer data. Default assumption "stop hit first" is the right conservative choice; better is to use 1m bars (which you already have) to resolve within the 5m bar.

**Indian market specifics** (relevant since you're in Punjab): for intraday equity, round-trip cost on a discount broker is approximately STT 0.025% (sell side only), exchange transaction charge ~0.00325% NSE / 0.00375% BSE, GST 18% on (brokerage + transaction charges), SEBI charges, stamp duty 0.003% (buy). Round-trip on liquid large-caps is typically 3–6 bps at retail size; smaller caps and F&O are materially more. STT on options is on premium and brutal. Hard-code these and never let a backtest report "edge" that's smaller than the cost band.

**Capacity.** Liquidity-event strategies have hard capacity limits because the *event itself* is being created by other size-takers. Test at multiple sizes; the strategy that works at 1 lot may not work at 50 because your own order moves the level.

**Bar timing.** Decision on bar close, action on next bar open, fill within that bar — that's the only sequence that doesn't leak. Anything else needs justification.

If I were reviewing this for a fund, the first deliverable I'd require before any further model work is an execution simulator that ingests 1m (or tick if you can get it) data and produces fills under a stated policy, with cost layers itemized. Then re-run the existing models through it. If expectancy collapses, no model improvement matters until execution does.

## Architecture

The P_touch × P_reaction × P_direction decomposition is useful for organizing the research but **don't multiply the probabilities in the live gate as if they were independent**. They almost certainly aren't. Direction state at decision time and touch probability share macro features; quality and reaction share microstructure features. The clean fix is to train a final-stage *trade* model that consumes the three calibrated probabilities (and possibly their raw features) and predicts the actual trade outcome under your execution policy. That model can learn the interaction structure. The decomposition is then a feature-engineering convenience, not a probabilistic factorization.

On **sector experts with dynamic shrinkage**: the framework is fine, but the comparison criterion matters. Compare on **strictly proper scoring rules** (log-loss or Brier) on a held-out fold, not on accuracy or AUC, and apply DSR-style multiple-testing correction since you're running one comparison per sector. James-Stein-style shrinkage toward the global model is the right shape; the shrinkage weight should fall out of the variance of the sector expert's OOS performance, not be tuned.

On **the feature store**: the principle "don't build the giant object in memory" is correct. Concrete additions: (a) every feature has a `feature_version` and `as_of_ts`, and the store supports point-in-time joins, not just merge_asof on timestamps; (b) feature computation is deterministic given (raw data hash, code version, params hash) — make this a CI check; (c) the proximity-table reduction should preserve all positive (touch) events and all near-touch negatives, and use stratified subsampling of far-away negatives with importance weights stored alongside, so calibration on full population can be recovered.

On **watch-only setups in evaluation**: if you gate them out before measuring PnL, you have a selection-biased evaluation. Run two evaluations in parallel — the gated one (your live strategy) and the ungated one (every setup that crossed a minimum threshold). The gap between them tells you whether the gates are helping or whether you've just shrunk the sample to a lucky slice.

## Prioritized roadmap

1. **Leakage audit and probe suite as CI.** Pool availability lag, MTF join semantics, embargo per-horizon, replay determinism, label-shuffle probe, future-mask probe. Until this is green, ignore all metrics.
2. **Reframe labels as triple-barrier outcomes under explicit policies.** Drop categorical pool labels as supervision targets; keep them as diagnostics derived from continuous outcomes.
3. **Build a realistic execution simulator** (1m intra-bar resolution, adverse-selection-aware fills, vol-correlated slippage, Indian cost stack) and re-baseline existing results. Many "edges" will not survive this.
4. **CPCV + DSR harness** to replace single-path walk-forward for model selection. Track number of configurations tried and deflate accordingly.
5. **Calibration with isotonic + empirical-Bayes shrinkage, rolling drift monitoring.** Live gates only on shrunk, drift-checked probabilities.
6. **Final-stage trade model** that consumes the three component probabilities plus raw features, replacing multiplicative gating.
7. **Post-touch reaction model only after the above.** This is where you said alpha lives, and I think you're right — but adding signal on top of a leaky base will just amplify the leakage.
8. **Shadow-live mode** that ingests data bar-by-bar with strict point-in-time semantics and diffs decisions against the research engine. Any divergence is a bug to fix before paper trading.
9. **Paper trade for at least one full quarter** with shadow-live decisions vs actual broker fills. Compare predicted to actual slippage and fill rate; recalibrate execution sim.
10. **Capacity tests** at 1×, 5×, 10× intended live size; abandon configs whose edge dies under realistic size.

## Red flags

The mention that synthetic fallback data needs to be banned suggests it has been used. Audit historical results for any contamination; any run that used synthetic fills should be retired entirely.

The observation that P_touch is much stronger than P_reaction is consistent with both genuine signal (distance is informative) and with leakage through MTF features. Run the leakage probe on the touch model specifically; if shuffling the 60m/daily features doesn't degrade it, you have a problem.

"Local parquet, no API during training" is good *if and only if* the parquet was built causally. If it was built from broker data with retroactive split/dividend adjustments applied, your historical levels are not the levels traders actually saw, and pool detection on adjusted data produces pools that didn't exist live. Confirm the adjustment policy and either (a) detect pools on unadjusted data and adjust outcomes, or (b) accept that the system has a small structural bias and bound it.

5-minute base TF on Indian equities gives ~75 bars/day and many liquidity events will be statistically thin per symbol per year. Consider whether the effective sample size after purging and per-bucket conditioning supports the model complexity you have. If buckets are routinely under a few hundred samples, you are mostly fitting noise.

Finally: the document is intellectually honest, which is a real asset. The dangerous failure mode for systems with this much self-awareness is that the discipline gets applied to the model layer but not the execution and leakage layers, because those feel less interesting. Resist that. The model is the easy part.

### Phase 4 Complete Revised Brief — Verbatim

# Codex Brief: Phase 4 COMPLETE REVISED — The Hidden Q Compression Bug, Pre-Touch Directional Strategy, and the AUC-to-PnL Gap

## Preamble: Read This First

This brief supersedes the original Phase 4 brief and the user's current Phase 4A/4B/4C plan. Both prior plans are technically sound but contain a critical blind spot that emerged from user-led analysis during a live trading session debrief on 2026-05-26.

**The blind spot:** The system has been diagnosed as having "weak pre-touch quality" (AUC 0.527, max Q ~46%) and "strong post-touch reaction" (AUC 0.776). This diagnosis led the entire architectural trajectory toward post-touch confirmation as the primary signal source. However, the user identified that:

1. The quality model output may be **severely compressed by `conservative_finml` regularization**, meaning "max Q = 46%" might actually be the 95th percentile of the model's real distribution — not a weak ceiling, but a compressed strong signal.
2. The strongest model in the stack (Proximity AUC 0.978) is being used only as a "reachability filter" when it could power a **pre-touch directional strategy** that captures the journey to the pool (e.g., ₹100→₹110 = 10% move), not just the post-touch reaction (e.g., ₹110→₹108 = 2% move).
3. The live gate threshold (Q ≥ 70%) may be **mathematically unreachable** if Q's true output range is 38-48%. This would explain the system printing "no tradeable today" every single day — not because it's correctly identifying low-edge days, but because the gate cannot fire.

This brief reframes Phase 4 around three parallel investigations:

- **Track 0 (URGENT, blocks everything else):** Audit whether Q is compressed or genuinely weak. If compressed, every threshold in the system is potentially miscalibrated.
- **Track A (NEW):** Pre-touch directional strategy using Proximity × Direction.
- **Track B (CURRENT PLAN):** Post-touch reaction execution (preserving the user's Phase 4A/4B/4C work).

All three tracks share the same execution simulator upgrades, leakage probes, and calibration framework. The execution mechanics are shared infrastructure; the strategy logic is parallel.

A live trading session on 2026-05-26 confirmed backtest predictions exactly. The user lost approximately ₹2,000 across 4 trades, with realized per-trade R closely matching OOS backtest expectations of −0.52R to −1.05R. **The model is behaving as advertised; the issue is that current execution policies have no demonstrated positive edge AND the gates evaluating those policies may be fundamentally broken.**

This brief is intentionally exhaustive. Vague briefs produce vague code. Every workstream has explicit acceptance criteria, expected artifacts, and decision gates. Do not skip sections. Do not parallelize workstreams beyond the explicit PR sequencing. Do not propose live trading enhancements until Tracks 0, A, and B all reach Definition of Done.

---

## 1. Project Context (For Codex Memory)

**Repository:** `/Users/abc/Projects/Project-freedom`
**Working directory:** `liquidity_backtester/`
**Branch:** `claude/liquidity-pool-backtester-1uskb`
**Data root:** `/Users/abc/Library/CloudStorage/GoogleDrive-garvitkatyal312@gmail.com/My Drive/kite_indian_market_data`
**Universe for this phase:** Core25 only (5 stocks × 5 sectors: BANKING, IT, FMCG, AUTO, PHARMA)
**Model artifacts:** `output_models/core25_latest/`
**Existing handoff doc:** Must be read in full before starting any workstream.

**Critical files for Workstream 0 audit:**
- `output_core25_phase3d_train_fresh_may25/phase3_oos_prediction_audit.csv`
- `output_core25_phase3d_train_fresh_may25/multi_asset_summary.json`
- `output_core25_phase3d_train_fresh_may25/research_summary.json`
- `output_models/core25_latest/multi_asset_report.pkl`

**Current model performance (handoff snapshot):**
- Quality model: Brier 0.246, AUC 0.527, max observed Q ≈ 0.46
- Direction model: AUC 0.621
- Proximity model: AUC 0.978
- Reaction model suite: AUC 0.748-0.776
- Execution backtest: ALL modes negative R (-0.52 to -1.05), PF 0.37-0.42
- Live gate: requires Q ≥ 0.70, T_today ≥ 0.50, DIR_ALIGN, distance 0.5-12 ATR, bucket n ≥ 30
- Live verdict: WATCH (no tradeable setups) virtually every day

**Live trading session 2026-05-26 results:**
- HCLTECH long: -₹50 approximately
- WIPRO long: -₹400 approximately (filled at gap-up, never recovered)
- HDFCBANK long: +₹445 then -₹650 (gave back profits, stopped out)
- KOTAKBANK long: -₹1,450 (stopped out)
- HINDUNILVR short: +₹160 (only winner, exited cleanly)
- Net day: approximately -₹1,940 on ~₹2.5L deployed
- **This matches backtest projections of -0.55R per trade × 4 trades**

The user has committed to paper trading only until execution PF > 1.0 OOS is achieved.

---

## 2. The Central Hypothesis (THIS IS THE BIGGEST INSIGHT)

The original Phase 4 brief asked: "Why does a 0.78-AUC reaction signal not translate into profit?"

The revised central question is: **"Have we been evaluating the model on the wrong scale this entire time?"**

Evidence supporting this hypothesis:

1. **Suspiciously narrow Q range across diverse assets:** Every one of 25 assets across 5 sectors prints `top_Q` between 45-46%. That is a 1-percentage-point spread across HDFCBANK (banking, large-cap), TVSMOTOR (auto, mid-cap), DIVISLAB (pharma, mid-cap), ITC (FMCG, large-cap), etc. If Q operated on a true 0-100% scale, you would expect 15-30% variance across such different assets and regimes.

2. **Conservative regularization preset:** The user runs `conservative_finml` regularization, which is specifically designed to shrink predictions toward the base rate to prevent overfitting. This is correct ML practice but creates a critical interaction: **the live gate threshold (Q ≥ 70%) was set on the un-shrunk scale and is potentially mathematically unreachable on the shrunk scale.**

3. **Daily "no tradeable" verdict:** The system has printed "NO TRADEABLE SETUP IN BASKET TODAY" virtually every day. This is either:
   - (a) The system correctly identifying low-edge days (current interpretation), OR
   - (b) The gate is algorithmically broken and cannot fire (untested hypothesis)

4. **AUC 0.527 with compressed output IS edge:** AUC measures ranking, not magnitude. If Q only varies from 0.38 to 0.50 (12-point range), an AUC of 0.527 over that compressed range means the model IS extracting signal — it just lacks headroom to express high confidence on an absolute scale.

5. **Sector experts shrunk to zero:** 3 of 5 sector experts (BANKING, FMCG, IT, PHARMA) have shrinkage weight of 0% because they "did not beat global enough" or "weight shrunk below usable floor." Only AUTO survives at 21.2% weight. This shrinkage logic also evaluates on absolute scales — if Q is compressed, the comparison logic may be unfair.

If this hypothesis confirms, the implications cascade through the entire system:

- The quality model isn't broken — it's been mis-evaluated for the entire project lifetime
- Execution backtests may have been testing wrong cohorts (no setups passing the broken gate)
- Sector expert shrinkage decisions need re-evaluation on percentile scales
- The pivot to post-touch reactions (Phase 3A/3B/3D) may have been premature
- Pre-touch trading may be entirely viable once Q is rescaled

This is why **Workstream 0 must complete before any other workstream begins.**

---

## 3. Workstream 0 — The Q Scale Audit (BLOCKING ALL OTHER WORK)

### 3.1 Purpose

Determine whether the quality model's output is compressed by regularization (and therefore mis-evaluated by absolute-scale gates) or genuinely weak (in which case the current diagnosis stands).

### 3.2 This is a half-day investigation, not a workstream

No code changes to production. No new features. No retraining. Pure analysis on existing artifacts.

### 3.3 Required scripts to write

**Script 1: `analysis/q_distribution_audit.py`**

```python
"""
Analyzes the actual output distribution of the quality model from OOS predictions.
Determines if Q is operating on a compressed range due to regularization.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_q_distribution(audit_csv_path: Path) -> dict:
    """
    Load the phase3_oos_prediction_audit.csv and compute:
    - Full distribution statistics for q_global, q_sector_expert, q_blended
    - Percentile breakdown (50, 75, 90, 95, 99, 99.9)
    - Min, max, mean, std
    - Asset-level Q range variance
    - Sector-level Q range variance
    """
    df = pd.read_csv(audit_csv_path)

    results = {}
    for q_col in ['q_global', 'q_sector_expert', 'q_blended']:
        if q_col not in df.columns:
            continue

        q = df[q_col].dropna()
        results[q_col] = {
            'n': len(q),
            'min': q.min(),
            'max': q.max(),
            'mean': q.mean(),
            'std': q.std(),
            'percentiles': {
                p: q.quantile(p/100) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
            },
            'range': q.max() - q.min(),
            'compression_ratio': (q.max() - q.min()) / 1.0,  # vs theoretical 0-1 range
        }

    # Per-asset Q distribution variance
    asset_q_stats = df.groupby('asset')['q_blended'].agg(['min', 'max', 'mean', 'std'])
    asset_q_stats['range'] = asset_q_stats['max'] - asset_q_stats['min']

    # Per-sector Q distribution variance
    sector_q_stats = df.groupby('sector')['q_blended'].agg(['min', 'max', 'mean', 'std'])

    return {
        'overall': results,
        'per_asset': asset_q_stats.to_dict(),
        'per_sector': sector_q_stats.to_dict(),
    }

if __name__ == '__main__':
    audit_path = Path('output_core25_phase3d_train_fresh_may25/phase3_oos_prediction_audit.csv')
    results = analyze_q_distribution(audit_path)

    # Output to reports/q_distribution_audit.md
    # Critical decision points to highlight:
    # 1. Is max(q_blended) < 0.60? If yes, gate at 0.70 is unreachable
    # 2. Is std(q_blended) < 0.05? If yes, Q is severely compressed
    # 3. Does per-asset Q range vary by more than 5%? If no, Q is too uniform
```

**Script 2: `analysis/q_decile_actual_rate.py`**

```python
"""
For each Q decile (compressed or not), compute the ACTUAL respect rate.
If the model is extracting real signal from a compressed range, top deciles
will have meaningfully higher actual rates than bottom deciles.
"""

def analyze_q_decile_performance(audit_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(audit_csv_path)

    # Use blended Q, fall back to global if blended unavailable
    q_col = 'q_blended' if 'q_blended' in df.columns else 'q_global'

    # Decile the predictions
    df['q_decile'] = pd.qcut(df[q_col], 10, labels=False, duplicates='drop')

    # For each decile, compute predicted vs actual
    decile_stats = df.groupby('q_decile').agg(
        n=('q_decile', 'size'),
        mean_q_predicted=(q_col, 'mean'),
        actual_strict_respect=('y_strict', 'mean'),
        actual_broad_respect=('y_broad', 'mean'),
    ).reset_index()

    # Calculate lift over base rate
    base_rate = df['y_strict'].mean()
    decile_stats['lift_over_base'] = decile_stats['actual_strict_respect'] - base_rate
    decile_stats['lift_pct'] = (decile_stats['actual_strict_respect'] / base_rate - 1) * 100

    return decile_stats

# Decision criteria:
# If top decile actual_strict_respect >= base_rate + 0.05 (5pp lift), Q has real edge
# If top decile actual_strict_respect <= base_rate + 0.01 (1pp), Q is noise
# Anywhere in between is "weak but real signal"
```

**Script 3: `analysis/q_top_percentile_lift.py`**

```python
"""
Computes lift at the top 5%, top 1%, top 0.1% of Q predictions.
This is the critical test: if compressed Q is hiding real edge, it shows here.
"""

def analyze_top_percentile_lift(audit_csv_path: Path) -> dict:
    df = pd.read_csv(audit_csv_path)
    q_col = 'q_blended' if 'q_blended' in df.columns else 'q_global'

    df['q_percentile'] = df[q_col].rank(pct=True)
    base_rate = df['y_strict'].mean()

    results = {'base_rate': base_rate}
    for cutoff_pct in [0.99, 0.95, 0.90, 0.75, 0.50]:
        top = df[df['q_percentile'] >= cutoff_pct]
        if len(top) > 0:
            results[f'top_{int((1-cutoff_pct)*100)}pct'] = {
                'n': len(top),
                'mean_q': top[q_col].mean(),
                'actual_respect': top['y_strict'].mean(),
                'lift_over_base': top['y_strict'].mean() - base_rate,
                'lift_pct_relative': (top['y_strict'].mean() / base_rate - 1) * 100,
            }

    return results

# Decision criteria for "Q is compressed but has edge":
# top 5% should have actual_respect >= base_rate + 0.05
# top 1% should have actual_respect >= base_rate + 0.08
```

**Script 4: `analysis/q_gate_reachability.py`**

```python
"""
Tests whether the current live gate thresholds (Q >= 0.70, etc.) are mathematically
reachable given the actual Q distribution.
"""

def analyze_gate_reachability(audit_csv_path: Path, live_plan_json: Path) -> dict:
    df = pd.read_csv(audit_csv_path)
    with open(live_plan_json) as f:
        live_plan = json.load(f)

    q_col = 'q_blended' if 'q_blended' in df.columns else 'q_global'
    current_q_gate = live_plan.get('gate_config', {}).get('min_q', 0.70)

    pct_above_gate = (df[q_col] >= current_q_gate).mean() * 100
    max_q_ever = df[q_col].max()

    # Find the percentile that the current gate corresponds to
    if max_q_ever < current_q_gate:
        gate_status = 'UNREACHABLE'
        gate_percentile = None
    else:
        gate_percentile = (df[q_col] < current_q_gate).mean() * 100

    # Suggest percentile-based equivalents
    suggested_thresholds = {
        'top_5pct_threshold': df[q_col].quantile(0.95),
        'top_10pct_threshold': df[q_col].quantile(0.90),
        'top_25pct_threshold': df[q_col].quantile(0.75),
    }

    return {
        'current_absolute_gate': current_q_gate,
        'max_q_observed': max_q_ever,
        'pct_setups_passing_gate': pct_above_gate,
        'gate_status': gate_status,
        'gate_percentile_equivalent': gate_percentile,
        'suggested_percentile_thresholds': suggested_thresholds,
    }
```

### 3.4 Required deliverable: `reports/phase4_workstream0_q_audit.md`

This markdown report must answer, explicitly and in plain English, the following questions:

1. What is the actual range of Q produced by the quality model? (min, max, std)
2. Is the current Q gate (0.70) reachable? What percentile does it correspond to?
3. Does the top decile / top 5% / top 1% of Q predictions have meaningful lift over base rate?
4. Per asset and per sector, how compressed is Q?
5. Should the gate logic be replaced with percentile-based thresholds?

### 3.5 Decision tree based on findings

**Branch A — Q is compressed (max Q < 0.60, top 5% has lift > 5pp):**

This confirms the user's hypothesis. The quality model has been mis-evaluated. Actions:

1. Document the finding prominently
2. Create `analysis/q_rescale.py` that converts compressed Q to percentile-based Q
3. Replace ALL Q thresholds throughout the codebase with percentile-based gates
4. Re-run live plan with corrected gates
5. Re-run execution backtest filtered to top-percentile Q setups only
6. **Re-evaluate the entire Phase 3 architecture** because the assumption "pre-touch quality is weak" may be wrong

**Branch B — Q is genuinely weak (max Q approaches 0.70, top 5% has lift < 2pp):**

The current diagnosis stands. Quality model has limited edge. Continue with original Phase 4 plan (Tracks A and B as written below).

**Branch C — Q is compressed but lift is also weak (max Q < 0.60, top 5% has lift < 2pp):**

The model output is compressed AND the underlying signal is genuinely weak. This is the worst case — fix the gates (because they're broken either way) but acknowledge that even percentile-corrected gates won't reveal hidden edge.

### 3.6 Acceptance criteria for Workstream 0

Single markdown report `reports/phase4_workstream0_q_audit.md` containing:

- All four analysis script outputs as embedded tables
- Explicit answer to "is Q compressed?" (yes/no/partially)
- Explicit answer to "is the current gate reachable?" (yes/no)
- If Branch A: complete percentile-based gate replacement spec
- If Branch B or C: confirmation that current absolute-scale evaluation is correct
- Decision: proceed to Track A, Track B, or both

**This report must be reviewed by the user before any other workstream begins.** If Branch A is confirmed, the entire Phase 4 plan below may need adjustment based on findings.

### 3.7 Time budget

Workstream 0 must complete within 2 working days. It is analysis-only on existing artifacts. No model training, no new data, no architecture changes.

---

## 4. Workstream 1 — Execution Simulator v2 (Shared Foundation)

### 4.1 Purpose

The current execution simulator uses 5-minute bars and treats stop-target ties pessimistically (stop wins). It does not model adverse selection, state-dependent slippage, or the realistic Indian cost stack. Every diagnosis below this layer is suspect until the simulator is realistic.

This workstream must complete before any strategy backtest (Track A or Track B) can be trusted.

### 4.2 Required changes

#### 4.2.1 1-minute intra-bar resolution

The data warehouse contains `raw_1m/` parquet files for all Core25 symbols. The 5-minute execution backtest currently cannot resolve which barrier (stop or target) hit first within a 5-min bar. Use 1-minute data to resolve this exactly.

Implementation:

```python
def resolve_intrabar_path(
    entry_bar_5m_timestamp: pd.Timestamp,
    entry_price: float,
    stop_price: float,
    target_price: float,
    symbol: str,
    direction: str,  # 'long' or 'short'
    intrabar_data_1m: pd.DataFrame,
) -> dict:
    """
    Returns:
    - resolution: 'stop_hit', 'target_hit', 'no_hit'
    - resolution_timestamp: when within the 5m bar
    - resolution_price: actual 1m price at resolution
    - intrabar_path: list of 1m close prices
    """
    bar_1m_slice = intrabar_data_1m[
        (intrabar_data_1m['timestamp'] >= entry_bar_5m_timestamp) &
        (intrabar_data_1m['timestamp'] < entry_bar_5m_timestamp + pd.Timedelta(minutes=5))
    ]

    for idx, row in bar_1m_slice.iterrows():
        if direction == 'long':
            if row['low'] <= stop_price:
                return {'resolution': 'stop_hit', 'timestamp': row['timestamp'], 'price': stop_price}
            if row['high'] >= target_price:
                return {'resolution': 'target_hit', 'timestamp': row['timestamp'], 'price': target_price}
        elif direction == 'short':
            if row['high'] >= stop_price:
                return {'resolution': 'stop_hit', 'timestamp': row['timestamp'], 'price': stop_price}
            if row['low'] <= target_price:
                return {'resolution': 'target_hit', 'timestamp': row['timestamp'], 'price': target_price}

    return {'resolution': 'no_hit', 'timestamp': None, 'price': None}
```

For trades that span multiple 5-min bars, iterate through all 1-min bars in the holding period.

#### 4.2.2 Adverse-selection-aware fills

Currently the simulator fills limit orders the moment price touches the limit level. This is too generous. Real market microstructure requires price to trade THROUGH a level for a marketable limit to fill against weak liquidity.

```python
FILL_POLICY_OPTIONS = {
    'generous': {
        'description': 'Current behavior — fills at limit when touched',
        'fill_buffer_atr': 0.0,
    },
    'neutral': {
        'description': 'Requires price to trade through limit by buffer',
        'fill_buffer_atr': 0.1,
    },
    'conservative': {
        'description': 'Mid-point between limit and current price + buffer',
        'fill_buffer_atr': 0.15,
        'use_midpoint_fill': True,
    },
}
```

Add CLI flag `--fill-policy`. Re-baseline all four existing execution modes (`blind_limit`, `touch_confirmed`, `reclaim_confirmed`, `displacement_confirmed`) under each policy.

#### 4.2.3 State-dependent slippage model

Current slippage is a flat 1 bps per side. Real slippage depends on volatility, distance to level (closer = thinner book), and time of day (open/close are wider).

```python
def compute_slippage_bps(
    realized_vol_5m: float,  # ATR or similar
    distance_to_level_atr: float,  # how close are we to the pool
    session_phase: str,  # 'open' (9:15-10:00), 'mid' (10:00-14:30), 'close' (14:30-15:30)
    base_bps: float = 2.0,
    k1_vol: float = 10.0,
    k2_distance: float = 1.0,
    k3_session: dict = None,
) -> float:
    if k3_session is None:
        k3_session = {'open': 2.0, 'mid': 1.0, 'close': 1.5}

    slip = base_bps
    slip += k1_vol * realized_vol_5m
    if distance_to_level_atr > 0:
        slip += k2_distance / distance_to_level_atr
    slip *= k3_session.get(session_phase, 1.0)

    return slip
```

Calibration: start with the parameters above. After running full Core25 backtest, compare simulated slippage to actual fills from the user's 2026-05-26 paper trades (HCLTECH, WIPRO, HDFCBANK, KOTAKBANK, HINDUNILVR). Tune parameters to match.

#### 4.2.4 Itemized Indian cost stack

Replace any flat-fee approximation with the exact Zerodha intraday equity cost stack:

```python
def compute_zerodha_intraday_costs(
    buy_price: float,
    sell_price: float,
    quantity: int,
) -> dict:
    """
    Returns itemized round-trip costs in rupees and as bps of turnover.
    Reference: https://zerodha.com/charges/ (verified 2026-05-25)
    """
    buy_turnover = buy_price * quantity
    sell_turnover = sell_price * quantity
    total_turnover = buy_turnover + sell_turnover

    # Brokerage: 0.03% or Rs 20 per executed order, whichever is lower
    brokerage_buy = min(0.0003 * buy_turnover, 20.0)
    brokerage_sell = min(0.0003 * sell_turnover, 20.0)
    brokerage_total = brokerage_buy + brokerage_sell

    # STT: 0.025% on sell side for intraday equity
    stt = 0.00025 * sell_turnover

    # Exchange transaction charges: NSE 0.00297%, both sides
    exchange = 0.0000297 * total_turnover

    # SEBI charges: Rs 10 per crore
    sebi = (10.0 / 10000000) * total_turnover

    # Stamp duty: 0.003% on buy side
    stamp = 0.00003 * buy_turnover

    # GST: 18% on (brokerage + exchange + SEBI)
    gst = 0.18 * (brokerage_total + exchange + sebi)

    total_cost = brokerage_total + stt + exchange + sebi + stamp + gst

    return {
        'brokerage_buy': brokerage_buy,
        'brokerage_sell': brokerage_sell,
        'brokerage_total': brokerage_total,
        'stt': stt,
        'exchange': exchange,
        'sebi': sebi,
        'stamp': stamp,
        'gst': gst,
        'total_cost_inr': total_cost,
        'total_cost_bps': (total_cost / total_turnover) * 10000,
        'breakeven_pct_move': (total_cost / buy_turnover) * 100,
    }
```

Output these fields in every trade row of `execution_backtest_v2_trades.csv`.

#### 4.2.5 Pre-touch trade simulation hooks (NEW capability)

The current simulator assumes entry happens at or after touch. Track A (Pre-Touch Directional) requires entry BEFORE touch, with the pool as a target. Add:

```python
class PreTouchDirectionalSimulator:
    """
    Simulates trades that enter at current price, target the pool boundary,
    and exit on stop or time barrier.
    """
    def simulate(
        self,
        pool_zone: tuple,  # (low, high) of pool
        pool_side: str,  # 'above' (long target) or 'below' (short target)
        entry_timestamp: pd.Timestamp,
        entry_price: float,
        bars_5m: pd.DataFrame,
        bars_1m: pd.DataFrame,
        stop_atr_mult: float = 1.5,
        target_fraction: float = 0.8,  # exit at 80% of distance to pool
        max_hold_bars: int = 78,
        atr_at_entry: float,
    ) -> dict:
        # Compute stop and target
        if pool_side == 'above':
            direction = 'long'
            target = entry_price + target_fraction * (pool_zone[0] - entry_price)
            stop = entry_price - stop_atr_mult * atr_at_entry
        else:
            direction = 'short'
            target = entry_price - target_fraction * (entry_price - pool_zone[1])
            stop = entry_price + stop_atr_mult * atr_at_entry

        # Iterate through 1m bars to find resolution
        # ... (uses resolve_intrabar_path above)
        # Apply slippage and costs
        # Return trade result
```

### 4.3 Required outputs

- `liquidity_backtester/liqpool/execution_simulator_v2.py` (new module)
- `examples/multi_asset_run.py` CLI flags: `--fill-policy`, `--use-1m-resolution`, `--slippage-model state_dependent|flat`
- `output_<run>/execution_backtest_v2_summary.csv`
- `output_<run>/execution_backtest_v2_trades.csv` (with all itemized cost fields)
- `output_<run>/execution_backtest_v2_vs_v1_delta.csv`

### 4.4 Acceptance criteria

1. All four existing modes re-run under v2 simulator with all three fill policies (12 backtest runs total)
2. Delta table v1 vs v2 included in PR description
3. New `PRE_TOUCH_DIRECTIONAL` simulator stub passes synthetic test (trades fire, resolve, compute costs correctly)
4. Itemized cost output validated against manual Zerodha calculation for one sample trade (cross-check against user's 2026-05-26 actual broker statement once available)
5. 1-minute intrabar resolution validated against synthetic edge cases (target hit before stop, stop hit before target, neither hit, both bars span gap)
6. No regression in existing Core25 backtest pipeline (run completes end-to-end)

### 4.5 Time budget

5 working days. This is the foundation for everything else and must be solid.

---

## 5. Workstream 2 — Leakage Probes as CI (MUST FAIL BUILD)

### 5.1 Purpose

Before any strategy testing, leakage must be ironclad. The proximity model's 0.978 AUC is suspiciously high and could harbor subtle lookahead bias. The current Phase 3C sampled replay audit checked only 4 samples — not nearly enough.

If leakage exists anywhere, all downstream backtests are meaningless. This workstream is non-negotiable.

### 5.2 Required probes

#### 5.2.1 Label-shuffle probe

```python
def test_label_shuffle_destroys_proximity_auc():
    """
    Shuffle y in training, retrain proximity model, assert OOS AUC ≈ 0.50.
    If OOS AUC stays high with shuffled labels, the model is learning from
    leaked future information.
    """
    np.random.seed(42)
    y_shuffled = np.random.permutation(y_train)
    model = train_proximity(X_train, y_shuffled)
    auc_oos = evaluate(model, X_oos, y_oos)
    assert 0.45 <= auc_oos <= 0.55, f"LEAKAGE: shuffled-label AUC = {auc_oos:.3f}"

def test_label_shuffle_destroys_direction_auc():
    # Same for direction model

def test_label_shuffle_destroys_quality_auc():
    # Same for quality model

def test_label_shuffle_destroys_reaction_auc():
    # Same for reaction model suite (strict, reclaim, break)
```

Run as part of CI. Must pass for all four models.

#### 5.2.2 Future-mask probe

For every feature column, assert that `as_of_ts <= decision_ts`. Mask any violators, retrain, assert byte-equality of predictions.

```python
def test_no_future_features_in_proximity():
    feature_metadata = load_feature_metadata()  # must exist with as_of_ts per feature

    for feature_name, meta in feature_metadata.items():
        if meta['as_of_ts'] > meta['decision_ts']:
            pytest.fail(f"LEAKAGE: feature {feature_name} uses future data")

    # Also do byte-equality test: mask any unsafe feature, retrain, compare predictions
    masked_features = mask_unsafe_features(features)
    masked_model = train_proximity(masked_features, y)
    assert_predictions_byte_equal(model, masked_model)
```

This requires every feature to have associated metadata with `as_of_ts` and `decision_ts`. If this metadata doesn't exist, building it is part of this workstream.

#### 5.2.3 Full MTF replay probe (not sampled)

Current Phase 3C samples 4 decision points. Replace with full audit:

```python
def test_mtf_replay_byte_equality_full():
    """
    For EVERY decision_ts in OOS:
    1. Truncate raw data to ts <= decision_ts
    2. Recompute 15m, 60m, 180m, 1D features from truncated data
    3. Assert byte-equality with stored values
    """
    decisions = load_oos_decisions()
    mismatches = []

    for decision in decisions:
        truncated = truncate_history(raw_data, decision.ts)
        recomputed = compute_mtf_features(truncated)
        stored = load_stored_features(decision)

        if not np.array_equal(recomputed, stored):
            mismatches.append(decision)

    assert len(mismatches) == 0, f"LEAKAGE: {len(mismatches)} MTF mismatches"
```

This will be expensive (potentially hours for full Core25). Optimize by parallelizing per symbol. Run nightly in CI, not on every PR.

#### 5.2.4 Pool availability probe (full, not sampled)

For every pool's `available_from_ts`, rerun pool detection on truncated history and assert the pool is produced identically.

```python
def test_pool_availability_full():
    pools = load_all_pools()
    misses = []

    for pool in pools:
        truncated = truncate_history(raw_data, pool.available_from_ts)
        redetected_pools = detect_pools(truncated)

        if pool not in redetected_pools:
            misses.append(pool)

    assert len(misses) == 0, f"LEAKAGE: {len(misses)} pools not reproducible"
```

#### 5.2.5 Distance leakage probe (SPECIFIC to Track A)

The proximity model uses `distance_atr` as its top feature (8.3M importance score, vs next at 627K). If `distance_atr` at decision_ts is computed using the close of decision_ts bar (a common bug), it contains forward information for any decision made mid-bar.

```python
def test_distance_atr_uses_only_past_data():
    """
    For each decision, recompute distance_atr using only bars strictly before decision_ts.
    Assert equality with stored distance_atr.
    """
    decisions = load_oos_decisions()
    for decision in decisions:
        past_bars_only = raw_data[raw_data['ts'] < decision.ts]
        recomputed_distance = compute_distance_atr(decision.pool, past_bars_only.iloc[-1])
        stored_distance = decision.distance_atr

        assert abs(recomputed_distance - stored_distance) < 1e-9, \
            f"LEAKAGE in distance_atr at {decision.ts}: stored={stored_distance}, recomputed={recomputed_distance}"
```

This is the single most important probe for validating Track A's thesis. If `distance_atr` is leaky, the entire proximity AUC of 0.978 may be artifact.

### 5.3 CI integration

Add `.github/workflows/leakage_tests.yml`:

```yaml
name: Leakage Probes
on: [push, pull_request]

jobs:
  leakage-fast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pytest tests/leakage/test_label_shuffle.py
      - run: pytest tests/leakage/test_future_mask.py
      - run: pytest tests/leakage/test_distance_leakage.py

  leakage-slow:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'  # nightly only
    steps:
      - uses: actions/checkout@v4
      - run: pytest tests/leakage/test_mtf_replay_full.py
      - run: pytest tests/leakage/test_pool_availability_full.py
```

### 5.4 Acceptance criteria

1. All five probes implemented as pytest tests
2. All probes pass on current production model
3. CI integration live (fast probes on every PR, slow probes nightly)
4. If any probe fails: halt all Phase 4 work, document leakage source in `reports/leakage_findings.md`, patch before continuing
5. Specific written verification of `distance_atr` causality (most important for Track A)

### 5.5 Time budget

4 working days. Parallel to Workstream 1.

---

## 6. Workstream 3 — Track A: Pre-Touch Directional Strategy (THE NEW THESIS)

### 6.1 Purpose

Test the user's central insight: the proximity model (AUC 0.978) combined with the direction model (AUC 0.621) can predict the JOURNEY of price toward a pool. If P_touch = 95% that price will move from ₹100 to ₹110, AND direction says UP, then a pre-touch directional trade entered NOW at ₹100, targeting ₹110, captures the full 10% move.

The current architecture ignores this trade entirely and waits to trade the ₹110→₹108 reaction (2% move). The post-touch move is harder (reactions are noisy), smaller (less reward), and more cost-sensitive. The pre-touch journey is bigger, more predictable, and uses the strongest model in the stack.

### 6.2 CRITICAL gating analysis BEFORE running 240-cell sweep

The proximity model's 0.978 AUC could be:
- (a) Genuinely strong directional prediction across all distances (Track A thesis is real)
- (b) Trivially good at close-range "is the pool 0.1 ATR away?" prediction, with no edge at trade-relevant distances (Track A thesis fails)

Run this analysis FIRST, before any sweep:

```python
def analyze_proximity_auc_by_distance_bucket(audit_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(audit_csv)

    distance_buckets = [
        (0, 1, '0-1 ATR (trivial)'),
        (1, 3, '1-3 ATR (scalping)'),
        (3, 5, '3-5 ATR (intraday)'),
        (5, 8, '5-8 ATR (sweet spot)'),
        (8, 12, '8-12 ATR (swing)'),
        (12, 100, '12+ ATR (untradeable)'),
    ]

    results = []
    for low, high, label in distance_buckets:
        bucket = df[(df['distance_atr'] >= low) & (df['distance_atr'] < high)]
        if len(bucket) < 100:
            continue

        # Compute proximity AUC for this bucket
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(bucket['touched'], bucket['p_touch'])
        base_rate = bucket['touched'].mean()

        results.append({
            'bucket': label,
            'n': len(bucket),
            'auc': auc,
            'base_rate_touched': base_rate,
            'mean_p_touch': bucket['p_touch'].mean(),
            'top_decile_actual_touched': bucket.nlargest(len(bucket)//10, 'p_touch')['touched'].mean(),
        })

    return pd.DataFrame(results)
```

**Decision gate based on output:**

- If 3-8 ATR bucket has AUC > 0.70 → Track A is viable, run full sweep
- If 3-8 ATR bucket has AUC < 0.55 → Track A is NOT viable, proximity is just trivial close-distance prediction, abandon Track A
- If 3-8 ATR AUC is 0.55-0.70 → marginal, run sweep with low expectations

**Do not proceed to 6.3 unless this gate passes.**

### 6.3 Strategy specification

```python
class PreTouchDirectionalStrategy:
    """
    Enters at current price when proximity says pool will be touched
    and direction model agrees with the pool direction.

    Captures the journey to the pool, not the reaction at the pool.
    """

    def evaluate_setup(
        self,
        pool: dict,
        spot: float,
        models: dict,
        config: dict,
    ) -> dict:
        # Compute features
        distance_atr = abs(pool['mid_price'] - spot) / pool['atr']
        pool_side = 'above' if pool['mid_price'] > spot else 'below'

        # Get model predictions
        p_touch = models['proximity'].predict(features_for_proximity)
        p_direction_up = models['direction'].predict(features_for_direction)

        # Determine if direction aligns with pool side
        if pool_side == 'above':
            direction_aligned = p_direction_up >= config['min_p_direction']
            trade_direction = 'long'
        else:
            direction_aligned = (1 - p_direction_up) >= config['min_p_direction']
            trade_direction = 'short'

        # Apply gates
        passes = all([
            p_touch >= config['min_p_touch'],          # default 0.85
            direction_aligned,
            config['min_distance_atr'] <= distance_atr <= config['max_distance_atr'],  # 2-8 ATR
            self.check_direction_conditional_auc_above_threshold(),
        ])

        if not passes:
            return {'tradeable': False, 'reason': '...'}

        # Compute entry/stop/target
        if trade_direction == 'long':
            entry = spot
            target = spot + config['target_fraction'] * (pool['low'] - spot)
            stop = spot - config['stop_atr_mult'] * pool['atr']
        else:
            entry = spot
            target = spot - config['target_fraction'] * (spot - pool['high'])
            stop = spot + config['stop_atr_mult'] * pool['atr']

        # Risk/reward check
        reward = abs(target - entry)
        risk = abs(entry - stop)
        rr = reward / risk

        if rr < config['min_rr_ratio']:
            return {'tradeable': False, 'reason': f'R:R {rr:.2f} below minimum'}

        return {
            'tradeable': True,
            'direction': trade_direction,
            'entry': entry,
            'target': target,
            'stop': stop,
            'p_touch': p_touch,
            'p_direction': p_direction_up if trade_direction == 'long' else (1 - p_direction_up),
            'distance_atr': distance_atr,
            'expected_r': rr,
        }
```

### 6.4 Backtest sweep

Using the v2 execution simulator, sweep this parameter grid:

```python
PARAMETER_GRID = {
    'min_p_touch': [0.70, 0.80, 0.85, 0.90, 0.95],          # 5 values
    'min_p_direction': [0.55, 0.60, 0.65, 0.70],            # 4 values
    'distance_range': [(2, 5), (3, 6), (3, 8), (5, 10)],    # 4 ranges
    'target_fraction': [0.5, 0.8, 1.0],                     # 3 values
    'stop_atr_mult': [1.0, 1.5, 2.0, 3.0],                  # 4 values
    'max_hold_bars': [30, 78, 156],                          # 3 values
}
# Total: 5 * 4 * 4 * 3 * 4 * 3 = 2,880 cells
```

That's too many for clean analysis. Reduce to a tractable grid:

```python
REDUCED_GRID = {
    'min_p_touch': [0.80, 0.85, 0.90],                       # 3
    'min_p_direction': [0.55, 0.60, 0.65],                   # 3
    'distance_range': [(2, 5), (3, 8), (5, 10)],            # 3
    'target_fraction': [0.6, 0.8, 1.0],                      # 3
    'stop_atr_mult': [1.0, 1.5, 2.0],                        # 3
    'max_hold_bars': [78, 156],                              # 2
}
# Total: 3^5 * 2 = 486 cells
```

For each cell, run full Core25 OOS backtest. Output per cell:
- Number of trades
- Win rate
- Mean R
- Median R
- Max drawdown
- Profit factor
- Sharpe (annualized assuming 250 trading days)
- Per-sector breakdown

### 6.5 Deflated Sharpe Ratio (CRITICAL)

With 486 cells, the best cell by raw PF is overfit by construction. Apply DSR deflation (Bailey/López de Prado 2014):

```python
def deflated_sharpe_ratio(
    sharpe_observed: float,
    n_trials: int,
    n_obs: int,
    skewness: float,
    kurtosis: float,
) -> float:
    """
    Returns probability that the observed Sharpe is genuinely > 0
    after multiple-testing deflation.

    A DSR > 0.95 means 95% confidence the strategy has real edge
    after accounting for selection bias from running n_trials.
    """
    from scipy.stats import norm

    # Expected max Sharpe under null hypothesis (no skill)
    emc = 0.5772  # Euler-Mascheroni constant
    expected_max_sharpe = (
        (1 - emc) * norm.ppf(1 - 1/n_trials) +
        emc * norm.ppf(1 - 1/(n_trials * np.e))
    )

    # Variance adjustment
    var_sharpe = (
        (1 - skewness * sharpe_observed + ((kurtosis - 1) / 4) * sharpe_observed**2)
        / (n_obs - 1)
    )

    deflated = norm.cdf(
        (sharpe_observed - expected_max_sharpe) / np.sqrt(var_sharpe)
    )

    return deflated
```

Only consider cells with DSR > 0.95 as "real edge."

### 6.6 Comparison vs post-touch baseline

Side-by-side report:

| Strategy | Best PF | Mean R | Trades | DSR | Max DD | Avg Hold |
|---|---|---|---|---|---|---|
| Track A: Pre-touch directional (best cell) | ? | ? | ? | ? | ? | ? |
| Track B: Post-touch reclaim_confirmed (current) | 0.37 | -1.05 | 7591 | 0.0 | ₹31,729 | ? |
| Combined: Pre-touch entry + post-touch hold | ? | ? | ? | ? | ? | ? |

### 6.7 Acceptance criteria

1. Distance-bucket AUC gate analysis completed first
2. If gate passes (3-8 ATR AUC > 0.55): full 486-cell sweep executed
3. DSR computed for top 10 cells
4. Comparison table vs Track B baseline
5. `reports/phase4_track_a_pretouch.md` with explicit conclusion: viable, marginal, or not viable
6. If viable, the top cell becomes a new execution mode `pretouch_directional` in `--execution-mode` CLI flag

### 6.8 Time budget

7 working days (depends on whether gate passes — if it fails, this becomes a 1-day "not viable" writeup).

---

## 7. Workstream 4 — Track B: Post-Touch Execution (THE CURRENT PLAN)

### 7.1 Purpose

The user's existing Phase 4A/4B/4C plan is preserved here as Track B. After Workstreams 0, 1, 2 complete, run this in parallel with Workstream 3.

### 7.2 Phase 4A: 1-minute intra-bar replay (now part of Workstream 1)

This was the user's original Phase 4A. It is now folded into Workstream 1 (Execution Simulator v2) so that both Track A and Track B benefit. No separate work needed here.

### 7.3 Phase 4B: Stateful post-touch workflow

Build the post-touch state machine:

```
POOL_ARMED
  → TOUCHED
    → CONFIRMATION_WINDOW (collect first N 1m/5m candles)
      → REACTION_SCORED (run reaction model on confirmation features)
        → ENTRY_ALLOWED (if all gates pass) or ENTRY_BLOCKED
          → MANAGED_EXIT (trail / target / stop / time)
```

Implementation:

```python
class PostTouchStateMachine:
    """
    Tracks the lifecycle of a pool from arming through touch through reaction
    through entry through exit.
    """

    states = ['ARMED', 'TOUCHED', 'IN_CONFIRMATION', 'REACTION_SCORED', 'ENTERED', 'EXITED', 'CANCELLED']

    def arm_pool(self, pool: dict, ts: pd.Timestamp):
        self.pools[pool.id] = {
            'state': 'ARMED',
            'pool': pool,
            'armed_at': ts,
        }

    def on_new_bar(self, bar: dict, ts: pd.Timestamp):
        for pool_id, state in self.pools.items():
            if state['state'] == 'ARMED':
                if self._is_touch(state['pool'], bar):
                    state['state'] = 'TOUCHED'
                    state['touched_at'] = ts
                    state['touch_bar_ohlcv'] = bar

            elif state['state'] == 'TOUCHED':
                bars_since_touch = (ts - state['touched_at']).total_seconds() / 300  # 5-min bars
                if bars_since_touch >= self.config['confirmation_window']:
                    state['state'] = 'IN_CONFIRMATION'
                    state['confirmation_bars'] = self._collect_confirmation_bars(state)

            elif state['state'] == 'IN_CONFIRMATION':
                reaction_features = self._compute_reaction_features(state)
                reaction_pred = self.reaction_model.predict(reaction_features)

                if self._reaction_passes_gate(reaction_pred):
                    state['state'] = 'ENTERED'
                    state['entry_details'] = self._compute_entry(state, reaction_pred)
                else:
                    state['state'] = 'CANCELLED'
                    state['cancel_reason'] = 'reaction_failed_gate'
```

CLI flags:
- `--use-stateful-post-touch` (enables this workflow)
- `--confirmation-window-bars` (default 3)
- `--reaction-confirm-threshold` (default 0.60)

### 7.4 Phase 4C: Reaction model calibration

The reaction model is producing 98% probabilities for some alerts (the `CONFIRM_UP` / `CONFIRM_DOWN` examples in the live output). This is calibration saturation — likely because of:
- Overfit on small confirmation buckets
- Isotonic calibration on the same fold used for evaluation
- No regime-conditional calibration

Required fixes:

#### 7.4.1 Held-out 3rd fold for isotonic

Current setup uses 2 folds (train, val) and fits isotonic on val. This understates calibration error. Switch to 3 folds (train, calib, val):

```python
def train_with_proper_calibration(X, y, n_folds=3):
    train_fold, calib_fold, val_fold = split_temporal(X, y, n_folds)

    base_model = lgb.train(params, train_fold)
    isotonic = IsotonicRegression().fit(
        base_model.predict(calib_fold.X),
        calib_fold.y
    )

    # Report metrics on truly held-out val fold
    val_predictions_raw = base_model.predict(val_fold.X)
    val_predictions_calibrated = isotonic.transform(val_predictions_raw)
    val_brier = brier_score_loss(val_fold.y, val_predictions_calibrated)
    val_auc = roc_auc_score(val_fold.y, val_predictions_calibrated)

    return base_model, isotonic, {'brier': val_brier, 'auc': val_auc}
```

#### 7.4.2 Per-regime calibration

Split validation into volatility terciles and fit separate isotonic per regime:

```python
def fit_per_regime_calibration(predictions, labels, vol_features):
    vol_terciles = pd.qcut(vol_features, 3, labels=['low_vol', 'mid_vol', 'high_vol'])

    calibrators = {}
    for regime in ['low_vol', 'mid_vol', 'high_vol']:
        mask = vol_terciles == regime
        if mask.sum() < 100:
            continue
        calibrators[regime] = IsotonicRegression().fit(
            predictions[mask], labels[mask]
        )

    return calibrators
```

#### 7.4.3 Bucket-n minimums and shrinkage

For each reaction bucket (TF count × factor type × sector × distance), require minimum n=50 events before reporting calibrated probability. Below threshold, use shrunk prior toward base rate:

```python
def shrunk_probability(bucket_prob: float, bucket_n: int, base_rate: float, prior_n: int = 100) -> float:
    """
    Empirical Bayes shrinkage: small buckets pull toward base rate.
    """
    return (bucket_prob * bucket_n + base_rate * prior_n) / (bucket_n + prior_n)
```

#### 7.4.4 Output format

Replace single `p_reaction` field with full diagnostic set:

```python
{
    'raw_p_reaction': 0.98,
    'calibrated_p_reaction': 0.74,  # after isotonic
    'shrunk_p_reaction': 0.65,      # after empirical Bayes
    'bucket_n': 23,                  # sample size for this bucket
    'calibration_error': 0.08,       # observed - predicted in bucket
    'regime': 'high_vol',            # which calibrator was used
    'confidence_warning': 'small bucket, shrunk heavily',
}
```

### 7.5 Acceptance criteria

1. State machine implemented and tested with synthetic events
2. Stateful workflow integrated into `--mode predict` output
3. Reaction model recalibrated with 3-fold + per-regime + shrinkage
4. No reaction alert prints >90% probability after recalibration (sanity check)
5. Reaction model OOS Brier after recalibration is lower than current
6. `reports/phase4_track_b_post_touch.md` with results

### 7.6 Time budget

10 working days. This is the bulk of the user's current plan and needs to be done right.

---

## 8. Workstream 5 — Direction Model Conditional Audit

### 8.1 Purpose

Direction AUC 0.621 globally. But the trades that Track A would actually take are a SUBSET of all decisions: high P_touch + specific distance range. Is the conditional AUC on that subset still 0.621, or does it drop?

This is a Track A viability gate.

### 8.2 Required script

```python
def audit_direction_conditional_on_proximity(audit_csv: Path):
    df = pd.read_csv(audit_csv)

    full_auc = roc_auc_score(df['y_direction'], df['p_direction'])

    # Subset: high P_touch + 2-8 ATR distance
    high_p_subset = df[
        (df['p_touch'] >= 0.85) &
        (df['distance_atr'].between(2, 8))
    ]
    high_p_auc = roc_auc_score(high_p_subset['y_direction'], high_p_subset['p_direction'])

    # Subset: low P_touch (sanity check — should be similar to full)
    low_p_subset = df[df['p_touch'] < 0.50]
    low_p_auc = roc_auc_score(low_p_subset['y_direction'], low_p_subset['p_direction'])

    return {
        'full_sample_auc': full_auc,
        'full_sample_n': len(df),
        'high_proximity_subset_auc': high_p_auc,
        'high_proximity_subset_n': len(high_p_subset),
        'low_proximity_subset_auc': low_p_auc,
        'low_proximity_subset_n': len(low_p_subset),
    }
```

### 8.3 Decision criteria

- If `high_proximity_subset_auc >= 0.60`: Track A direction signal is preserved, viable
- If `high_proximity_subset_auc` in 0.55-0.60: Marginal, proceed with caution
- If `high_proximity_subset_auc < 0.55`: Track A loses direction edge in trade conditions, abandon Track A

### 8.4 Acceptance criteria

Single markdown report `reports/phase4_workstream5_direction_conditional.md` with the three AUC numbers and the explicit go/no-go decision for Track A.

### 8.5 Time budget

1 working day.

---

## 9. Workstream 6 — Geometry Sweep (Stops/Targets)

### 9.1 Purpose

Post-touch average MAE is 4.15 ATR but current stops are nowhere near that. Sweep stop/target combinations to find what actually works.

### 9.2 Sweep grid

Apply to both `touch_confirmed` and `displacement_confirmed`:

```python
SWEEP_GRID = {
    'sl_atr': [0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    'tp_atr': [1.0, 1.5, 2.0, 3.0, 5.0, 'next_opposing_pool'],
    'time_barrier': [30, 78, 156, 312],
    'trailing': ['none', 'breakeven_at_1R', 'atr_trail_2x', 'atr_trail_3x'],
}
# Total: 6 * 6 * 4 * 4 = 576 cells per mode = 1152 total
```

For each cell, report:
- Number of trades (skip cells with n<200)
- Win rate
- Mean R
- Median R
- Max drawdown
- Profit factor
- Sharpe
- DSR (multiple-testing deflated)

### 9.3 Acceptance criteria

`reports/phase4_geometry_sweep.md` with top 10 cells by DSR-adjusted PF. Recommended cell with justification.

### 9.4 Time budget

5 working days (computationally expensive).

---

## 10. Workstream 7 — Calibration & Validation Upgrades

### 10.1 Purpose

Strengthen the validation framework so future workstreams don't rediscover the same flaws.

### 10.2 Required upgrades

#### 10.2.1 CPCV harness

Replace single-path walk-forward with Combinatorial Purged Cross-Validation:

```python
def cpcv_validate(
    X, y,
    n_paths: int = 10,
    purge_bars: int = 312,
    embargo_bars: int = 312,
) -> dict:
    """
    Generates n_paths different train/test combinations with proper purging.
    Returns Sharpe distribution and DSR.
    """
    # ... implementation
```

#### 10.2.2 Null baselines

Build two null models that the system must beat:

```python
def null_model_random_pools(spot_history, pool_density):
    """Scatter random levels at the same spatial density as detected pools."""

def null_model_atr_offset(spot, atr, k_range=[1, 2, 3, 5]):
    """Use spot ± k×ATR as 'pools'."""
```

System must beat both nulls on OOS mean R post-DSR. If it doesn't, the "edge" is detector selection bias.

### 10.3 Acceptance criteria

`reports/phase4_validation_upgrades.md` showing:
- CPCV Sharpe distribution for all current models
- Comparison vs random pool null
- Comparison vs ATR offset null

### 10.4 Time budget

5 working days.

---

## 11. Workstream 8 — Live Trading Lessons Integration

### 11.1 Purpose

The 2026-05-26 live session surfaced four issues. Encode them as system features.

### 11.2 Required features

#### 11.2.1 BSE/NSE routing

```python
DEFAULT_EXCHANGE_PRIORITY = ['NSE', 'BSE']

def select_exchange(symbol: str, available_exchanges: list) -> str:
    for preferred in DEFAULT_EXCHANGE_PRIORITY:
        if preferred in available_exchanges:
            return preferred
    raise ValueError(f"No usable exchange for {symbol}")
```

Live plan output must specify exchange explicitly. Warn if symbol BSE-only.

#### 11.2.2 Time-of-day windows

```python
TRADING_WINDOWS = {
    'morning_active': (time(9, 15), time(11, 30)),
    'lunch_lull': (time(11, 30), time(13, 30)),
    'afternoon_active': (time(13, 30), time(14, 45)),
    'closing': (time(14, 45), time(15, 30)),
}

def is_entry_allowed(now: time, mode: str = 'MIS') -> tuple[bool, str]:
    if mode == 'MIS' and now >= time(14, 45):
        return False, 'Insufficient time before MIS auto-squareoff'
    if TRADING_WINDOWS['lunch_lull'][0] <= now < TRADING_WINDOWS['lunch_lull'][1]:
        return False, 'Lunch lull: low edge, false breakouts'
    return True, 'OK'
```

CLI flag: `--allowed-trading-window`.

#### 11.2.3 Post-news euphoria detector

```python
def detect_news_euphoria(
    gift_nifty_overnight_return: float,
    vix_change: float,
    consecutive_positive_days: int,
) -> dict:
    euphoria_score = 0

    if gift_nifty_overnight_return > 0.01:
        euphoria_score += 1
    if vix_change < -0.10:
        euphoria_score += 1
    if consecutive_positive_days >= 3:
        euphoria_score += 1

    if euphoria_score >= 2:
        return {
            'is_euphoric': True,
            'recommended_action': 'reduce_position_size_50pct_or_block',
            'reason': f'Euphoria score {euphoria_score}/3',
        }
    return {'is_euphoric': False}
```

New features for the model: `gift_nifty_overnight_return`, `vix_overnight_change`, `consecutive_positive_days`.

#### 11.2.4 Daily drawdown circuit breaker

```python
class DailyDrawdownCircuitBreaker:
    def __init__(self, max_loss_pct: float = 0.005):
        self.max_loss_pct = max_loss_pct
        self.daily_pnl = 0.0
        self.capital = None

    def update(self, trade_pnl: float):
        self.daily_pnl += trade_pnl
        if self.daily_pnl / self.capital < -self.max_loss_pct:
            return 'STOP_TRADING_TODAY'
        return 'CONTINUE'
```

CLI flag: `--max-daily-loss-pct` (default 0.005).

### 11.3 Acceptance criteria

1. All four features implemented
2. Live plan output respects all four
3. Integration tests pass
4. Documentation in `docs/live_trading_safeguards.md`

### 11.4 Time budget

3 working days.

---

## 12. Workstream 9 — Combined Strategy (CONDITIONAL)

### 12.1 Purpose

If both Track A and Track B show edge, build a hybrid:

- Track A enters before touch (captures journey)
- Track B decides at touch (hold, scale, exit, or reverse)

### 12.2 Only proceed if

- Workstream 3 shows Track A DSR > 0.95 on best cell
- Workstream 4 shows at least one Track B mode with PF > 1.0

### 12.3 Acceptance criteria

If proceeded, the combined strategy must beat both standalone tracks on OOS mean R.

### 12.4 Time budget

5 working days (conditional, possibly skipped).

---

## 13. Workstream 10 — Options Data Foundation (PARALLEL, LOW PRIORITY)

### 13.1 Purpose

Lay options data foundation for Phase 5+. Do NOT build options strategies in this phase.

### 13.2 Required work

1. Kite Connect setup for options chain pulls
2. Historical OHLC + OI for NIFTY/BANKNIFTY weekly + monthly (1 year)
3. Back-calculate IV using `py_vollib`:
   ```python
   from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
   iv = implied_volatility(price=opt_close, S=spot, K=strike, t=tte_years, r=0.065, q=0.0, flag='c')
   ```
4. Compute Greeks
5. Store IV percentile / rank per strike per day

### 13.3 Justification

The proximity model's strength (AUC 0.978) is naturally suited to options: "will price reach strike X by expiry?" is exactly the proximity question. If Track A works on equities, it likely works better on options due to leverage and defined risk.

### 13.4 Acceptance criteria

1. Historical IV/Greeks for NIFTY, BANKNIFTY (1 year)
2. Validation against external source (Sensibull or Opstra) for 5 sample dates
3. No strategies built yet — just clean data layer

### 13.5 Time budget

7 working days, parallel to other workstreams.

---

## 14. Workstream 11 — Timeframe Audit (RESEARCH SPIKE)

### 14.1 Purpose

The 78-bar horizon on 5-min timeframe = 1 trading day. For afternoon entries, labels span overnight gaps which are news-driven and uncapturable from intraday features.

### 14.2 Required tests

1. Train identical models at 5m, 15m, 30m, 1H base timeframes
2. Compare AUC, Brier, execution PF
3. Test intraday-only horizon (h=30, 2.5 hours)
4. Test time-of-day-filtered training (afternoon entries use shorter horizons)

### 14.3 Acceptance criteria

`reports/phase4_timeframe_audit.md` with recommendation.

### 14.4 Time budget

5 working days, low priority.

---

## 15. PR Sequencing (NON-NEGOTIABLE)

Codex will be tempted to do all eleven workstreams in parallel and produce a 5000-line monster diff. **DO NOT.** Land in this exact order:

| PR | Workstream | Why this order |
|---|---|---|
| #1 | Workstream 0 (Q Scale Audit) | Blocks everything else if Q is broken |
| #2 | Workstream 2 (Leakage Probes) | Foundation — if leakage exists, everything else is moot |
| #3 | Workstream 1 (Execution Simulator v2) | Shared infrastructure for Tracks A and B |
| #4 | Workstream 5 (Direction Conditional Audit) | Track A viability gate, fast |
| #5 | Track A distance-bucket AUC gate (part of WS3) | Track A viability gate, fast |
| #6 | Workstream 3 full sweep (only if PR #4 and #5 pass) | Track A test |
| #7 | Workstream 4 (Track B — current plan's 4A/4B/4C) | Track B test |
| #8 | Workstream 8 (Live Trading Lessons) | Safety features |
| #9 | Workstream 6 (Geometry Sweep) | Improves Track B specifically |
| #10 | Workstream 7 (Calibration / Validation) | Strengthens future work |
| #11 | Workstream 9 (Combined Strategy) | Only if both tracks succeed |
| #12 | Workstream 10 (Options Data) | Parallel, foundation for Phase 5 |
| #13 | Workstream 11 (Timeframe Audit) | Research spike, lowest priority |

Each PR must:
- Pass existing tests
- Pass new leakage CI from PR #2
- Include explicit acceptance criteria section
- Include before/after metrics where applicable
- Reference this brief by workstream number
- Contain its own markdown report in `reports/`

---

## 16. Out of Scope This Phase

Explicitly DO NOT:

- Add new features beyond what's specified in workstreams above
- Train new models beyond what's specified
- Expand universe beyond Core25
- Retrain on more data (use existing OOS audit data for analysis)
- Trade real money until DoD met
- Build sequence/transformer models
- Remove `distance_atr` from proximity model
- Propose Phase 5 changes
- Refactor unrelated code
- Add cloud infrastructure (Mac is fine for Core25)
- Build options strategies (only data foundation in WS10)

---

## 17. Definition of Done

Phase 4 is complete when ALL of the following hold:

1. **Workstream 0 complete** with clear answer on Q compression hypothesis
2. **Workstream 2 complete** with all leakage probes passing in CI
3. **Workstream 1 complete** with v2 simulator operational
4. **Track A definitive answer**: viable (with cells > DSR 0.95) or not viable (with evidence)
5. **Track B definitive answer**: at least one execution mode with PF > 1.0 OOS, or documented evidence that post-touch alone cannot be made profitable
6. **Calibration sane**: no reaction alerts >90% after recalibration
7. **Live trading lessons integrated**: all four safeguards (BSE/NSE, time windows, euphoria, circuit breaker) operational
8. **Final report**: `reports/phase4_conclusion.md` summarizing:
   - Was Q compressed? (yes/no)
   - Is Track A viable? (yes/no, with numbers)
   - Is Track B viable? (yes/no, with numbers)
   - What's the recommended live strategy?
   - What's the go/no-go decision for paper trading?

---

## 18. Critical Reframes from Original Plan

If you (Codex) are reading the previous Phase 4 brief and this one and noticing differences, the key reframes are:

| Original brief said | This brief says | Why |
|---|---|---|
| "Quality model is weak (AUC 0.527)" | "Quality model may be COMPRESSED, not weak. Audit first." | User insight on regularization shrinkage |
| "Pivot to post-touch" | "Test pre-touch AND post-touch in parallel" | Proximity AUC 0.978 is too strong to ignore |
| "Phase 4A is 1m intrabar replay" | "1m replay is Workstream 1 (shared infra)" | Both tracks need it |
| "Phase 4B is stateful post-touch" | "Workstream 4 is Track B post-touch" | Reframed as one of two tracks |
| "486-cell sweep for geometry" | "486-cell sweep for Track A, separate 1152-cell for Track B geometry" | Different strategies need different sweeps |
| "DSR is optional" | "DSR is mandatory" | Without DSR, best cell is overfit by construction |

---

## 19. Honest Communication Standard for Reports

When something doesn't work, say so plainly. The user has been live-trading the model against advice and lost ₹2,000 on 2026-05-26 — exactly what the OOS backtest predicted. The biggest enemy is false confidence.

If Workstream 0 finds Q is genuinely weak (not compressed), write "Q is not compressed, current diagnosis stands" in the report title. Do not soften it.

If Track A's distance-bucket AUC gate fails, write "Track A not viable" in the title. Do not write "promising preliminary signal."

If Track B reaches DoD with one mode at PF 1.02 after all the work, write "marginal edge, requires paper validation" — not "Track B successful."

Honesty in writeups is worth more than apparent progress. The user can handle bad news. The user cannot handle losing money on a system they were told was working.

---

## 20. User Context

The user is a discretionary trader running this system semi-systematically. The user has demonstrated:

- Deep model understanding (caught the Q compression hypothesis)
- Strong intuition (questioned the 78-bar horizon, the post-touch pivot, the percentile interpretation)
- Willingness to lose small to learn (paper-trading commitment after live loss)
- Patience for proper engineering (specifically asked for "long, very elaborated, highly defining" brief)

The user wants this system to become a disciplined quant research engine. Not a loose signal generator. Not a gambling tool.

The user will:
- Run terminal commands carefully (provide exact commands)
- Read long reports thoroughly (markdown reports are read, not skimmed)
- Question assumptions you make (be explicit about your assumptions)
- Notice if you skip workstreams (don't skip)

Treat this user as a peer engineer who happens to also be the domain expert. Explain decisions, don't just make them.

---

## 21. Existing Run Commands (Reference)

Core25 train (re-run only if data window changes):
```bash
PYTHONPATH=. .venv/bin/python examples/multi_asset_run.py \
  --universe core25 \
  --data-source parquet \
  --data-dir "$RESAMPLED_DIR" \
  --mode train \
  --model-dir output_models/core25_latest \
  --asset-workers 2 \
  --checkpoint-dir output_checkpoints/core25 \
  --resume \
  --regularization-preset conservative_finml \
  --out output_core25_train
```

Core25 predict (daily workflow):
```bash
PYTHONPATH=. .venv/bin/python examples/multi_asset_run.py \
  --universe core25 \
  --data-source parquet \
  --data-dir "$RESAMPLED_DIR" \
  --mode predict \
  --model-dir output_models/core25_latest \
  --execution-mode reclaim_confirmed \
  --out output_core25_predict_latest
```

---

## 22. Final Note

This brief is long because the work is consequential. The user has identified what may be the most important bug in the project's history (Q compression). The strategic question (pre-touch vs post-touch) is the most important architectural question since the project began.

Get Workstream 0 right. Get Workstream 2 right. Then run Tracks A and B in parallel under a sound simulator.

If Q is compressed AND Track A works, the system suddenly has clear edge it couldn't see before. That is the prize. Worth doing properly.

If neither works, the answer is options (Phase 5). But we have to know first.

Start with Workstream 0. Single markdown report. Then check in before any code changes.
