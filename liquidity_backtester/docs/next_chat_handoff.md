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

Local branch is ahead of GitHub with the Phase 3B post-touch reaction alert
commit:

- `b9bec66 Add post-touch reaction model alerts`

This handoff is being updated for Phase 3C Consistency Layer v1. Check
`git log --oneline -5` and `git status --short --branch` at the start of the
next session to confirm whether the Phase 3C commit has already been created
or pushed.

Do not accidentally commit these local/untracked runtime files unless explicitly requested:

- `liquidity_backtester/.env`
- `liquidity_backtester/.venv/`
- `liquidity_backtester/generate_token.py`
- `liquidity_backtester/run_morning.sh`
- `liquidity_backtester/output_*`
- `liquidity_backtester/.DS_Store`

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

7. Continue Phase 3 proper: make the reaction model operational after touch.
   - The standalone reaction model suite now exists and trains from `post_touch_events.parquet`.
   - Phase 3B now adds first-pass reaction alerts for recently touched pools.
   - Next harden this into a stateful "pool armed -> pool touched -> score event -> entry allowed/blocked" workflow.
   - Add sector regime at touch, direction state at touch, and eventually 1-minute replay features.
   - Live plan should eventually require touch + confirmation + reaction model agreement.

8. If metrics still show high overfit, the next high-impact options are:
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
