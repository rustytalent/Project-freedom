# Next Chat Handoff

## Repo And Branch

- Repo path on Mac: `/Users/abc/Projects/Project-freedom`
- Work folder for running Python: `/Users/abc/Projects/Project-freedom/liquidity_backtester`
- GitHub repo: `https://github.com/rustytalent/Project-freedom`
- Active branch: `claude/liquidity-pool-backtester-1uskb`
- Push is working from terminal now.

## Important Local Status

The branch is synced with GitHub after Phase 2.

Do not accidentally commit these local/untracked runtime files unless explicitly requested:

- `liquidity_backtester/.env`
- `liquidity_backtester/.venv/`
- `liquidity_backtester/generate_token.py`
- `liquidity_backtester/run_morning.sh`

## Recent Uploaded Commits

- `008129d Add sector-aware MoE quality model`
  - Added `SectorMoERespectModel`
  - Global LightGBM model + sector experts
  - Hard sector routing + 70/30 sector/global blend
  - Multi-asset training now uses sector MoE

- `02040e1 Add sector regime execution filter`
  - Added richer sector regime scoring from 5d/20d/60d sector returns
  - Added `ALIGNED`, `AGAINST`, `NEUTRAL` sector execution logic
  - Weak setups against a strong sector regime are blocked unless Q is high enough
  - Wired into `examples/live_run.py` and `examples/multi_asset_run.py`

## Commands To Run

From repo root:

```bash
cd /Users/abc/Projects/Project-freedom/liquidity_backtester
PYTHONPATH=. .venv/bin/python examples/multi_asset_run.py
```

If committing/pushing:

```bash
cd /Users/abc/Projects/Project-freedom
git status --short --branch
git add <files>
git commit -m "<message>"
git push origin claude/liquidity-pool-backtester-1uskb
```

## Phase 3 / True MoE Remaining Work

Phase 1 and Phase 2 are structural but still mostly rule-based:

- Phase 1: sector expert blend is fixed at 70% sector / 30% global when expert exists.
- Phase 2: sector regime gate is heuristic using 5d/20d/60d returns.

Phase 3 should make the gate learned and measurable.

### Phase 3A: OOS Prediction Audit

Goal: measure whether Phase 1/2 actually improved results.

Tasks:

- Add an OOS comparison table:
  - global Q
  - sector expert Q
  - blended Q
  - actual outcome
  - sector
  - asset
- Report metrics by model:
  - Brier score
  - log-loss
  - AUC
  - top-decile hit rate
  - per-sector calibration
- This should be done before changing the gate again.

### Phase 3B: Learned Gate Dataset

Goal: create training rows for a gate model.

Each row should include:

- global prediction
- sector prediction
- absolute disagreement between global and sector expert
- sector regime score / conviction
- asset reliability
- sector OOS respect rate
- pool features already used by Q model
- label: which prediction was closer / whether blended prediction improved calibration

### Phase 3C: Learned Dynamic Gate

Goal: replace fixed 70/30 with dynamic weights.

Output:

```text
final_q = gate_weight * sector_q + (1 - gate_weight) * global_q
```

Gate model can start simple:

- logistic regression or shallow LightGBM regressor/classifier
- target should optimize calibration / Brier improvement, not only hit rate
- clip gate weights, for example `0.20 <= gate_weight <= 0.85`, to avoid unstable all-in routing

### Phase 3D: Live Explanation Upgrade

Goal: make live output explain why the model trusted sector/global.

Add to trade cards:

- global Q
- sector expert Q
- gate weight
- final blended Q
- sector regime alignment
- reason for fallback if no expert trained

### Phase 3E: Safety Checks

Before trusting Phase 3:

- Keep global fallback always available.
- If sector expert train sample is too small, skip expert.
- If gate model cannot train cleanly, fall back to Phase 1 fixed blend.
- Compare against Phase 1/2 baseline before declaring improvement.

