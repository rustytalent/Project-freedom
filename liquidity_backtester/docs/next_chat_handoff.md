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

The default branch is synced with GitHub after the Phase 3 stability/gating commit.

Do not accidentally commit these local/untracked runtime files unless explicitly requested:

- `liquidity_backtester/.env`
- `liquidity_backtester/.venv/`
- `liquidity_backtester/generate_token.py`
- `liquidity_backtester/run_morning.sh`

## Recent Uploaded Commits

- `fc35a0f Add parallel asset checkpoints`
  - Added `--asset-workers`, `--checkpoint-dir`, and `--resume`.
  - Per-symbol parquet training can run in parallel with deterministic final merge order.
  - Completed symbols are saved as pickle checkpoints, so a killed run can resume without repeating all asset work.
  - Worker processes cap BLAS/OpenMP thread counts to avoid oversubscribing the M4 laptop.

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

Not fully implemented. Current code has post-touch diagnostics, but not a standalone event database/model.

Required:

- Generate `post_touch_events.parquet`.
- Label events:
  - `HARD_REJECT`
  - `SWEEP_RECLAIM`
  - `ABSORPTION`
  - `FAIL_CONTINUE`
  - `LIQUIDITY_VACUUM`
  - `NO_SIGNAL`
- Add features from the first N candles after touch:
  - reclaim candle presence
  - displacement size
  - volume spike
  - close back inside/outside pool
  - wick rejection ratio
  - time-to-reclaim
  - MAE/MFE after touch
  - sector regime at touch
  - direction state at touch
- Train a reaction model separately.
- Live plan should be able to require confirmation after touch instead of blind limit entry.

### Phase 4 - Execution And PnL Engine

Not implemented yet.

Required:

- Add execution modes:
  - `blind_limit`
  - `touch_confirmed`
  - `reclaim_confirmed`
- Prefer `reclaim_confirmed` by default for live decisions.
- Simulate:
  - slippage
  - brokerage/fees placeholder
  - stop by pool width + ATR
  - target by historical MFE distribution
  - partial TP
  - time stop
  - direction filter
  - sector filter
  - confirmation filter
- Report profitability metrics:
  - net expectancy per trade
  - win rate
  - avg win/loss
  - profit factor
  - max drawdown
  - trades/month or trades/year
  - exposure time
  - per-symbol PnL
  - per-sector PnL
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

## Next Goals

1. Treat `conservative_finml` as the current preferred baseline unless a future run reverses the quality audit.
   - First rerun `conservative_finml` after the new causal momentum features and compare against `output_conservative_baseline`.
   - Then rerun from the real local parquet warehouse using `--data-source parquet`.
   - On the MacBook M4/16GB, start with `--asset-workers 3`; use `--resume` with a checkpoint directory for restartability.
   - Re-run default and conservative baselines when the data window changes materially or when `TATAMOTORS.NS` data becomes available.
   - Compare pooled OOS broad/strict, mean per-asset overfit gap, quality Brier/log-loss/AUC, and post-touch strict respect.
   - Do not judge success from proximity AUC alone; proximity is the reachability engine and distance dominates by design.

2. Tune the new live gate only after seeing real OOS bucket counts.
   - Defaults are intentionally strict: Q >= 70%, T_today >= 50%, `DIR_ALIGN`, distance 0.5-12 ATR, bucket n >= 30.
   - The 2026-05-21 baselines reject everything mainly because Q is far below 70%, not because every bucket is under-supported.
   - If it rejects everything again, inspect `gate_decisions` and `distance_bucket_metrics` before relaxing thresholds.
   - Trade count can drop; that is acceptable if post-touch quality improves.

3. Inspect sector shrinkage behavior.
   - Check `sector_shrinkage_report` in `multi_asset_summary.json`.
   - Good behavior: small/noisy sector experts get near-zero weights.
   - Only trust sector experts that beat global fallback on validation loss with reasonable AUC.

4. Use post-touch reaction quality as the main alpha diagnostic.
   - Focus on `post_touch_reaction_metrics`.
   - Look for factor/TF/sector/Q buckets with high strict respect and low broken_strong rate.
   - This should drive future feature or gate changes more than aggregate touch probability.

5. If metrics still show high overfit, the next high-impact options are:
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
