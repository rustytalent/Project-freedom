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

1. Run full real-data baselines when yfinance/data access works.
   - Run default preset and `conservative_finml`.
   - Compare pooled OOS broad/strict, mean per-asset overfit gap, quality Brier/log-loss/AUC, and post-touch strict respect.
   - Do not judge success from proximity AUC alone; proximity is the reachability engine and distance dominates by design.

2. Tune the new live gate only after seeing real OOS bucket counts.
   - Defaults are intentionally strict: Q >= 70%, T_today >= 50%, `DIR_ALIGN`, distance 0.5-12 ATR, bucket n >= 30.
   - If it rejects everything, inspect `gate_decisions` and `distance_bucket_metrics` before relaxing thresholds.
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
