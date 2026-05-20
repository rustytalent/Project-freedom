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

From repo root:

```bash
cd /Users/abc/Projects/Project-freedom/liquidity_backtester
PYTHONPATH=. .venv/bin/python examples/multi_asset_run.py
```

Useful comparison run:

```bash
cd /Users/abc/Projects/Project-freedom/liquidity_backtester
PYTHONPATH=. .venv/bin/python examples/multi_asset_run.py --regularization-preset conservative_finml --out output_conservative
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
