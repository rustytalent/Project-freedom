# System Overview

> The single doc that explains what this is, how it works, and how the
> parts fit together. Read this if you are new to the project or
> returning after time away. Update only when the underlying SHAPE
> changes; for tactical state see `COORDINATION.md`; for architecture
> facts see `COMPANY_MAP.md`.

---

## 1. The 30-second mental model

We have built **one engine that answers one question at multiple time
horizons**:

> Will price reach level L within time T?

Everything else is presentation, packaging, and commerce.

The engine is fed by:
* 5-minute OHLCV bars for ~25 NSE equities (and soon 5 indexes)
* 42 features computed per bar (price structure + volume profile +
  session context + calendar events + path shape)

The engine produces:
* **Quality (Q)** — P(a pool will be respected) — *currently weak*
* **Direction** — P(price goes up over horizon) — *useful as filter*
* **Proximity** — P(level L is touched within H bars) — *the strong signal*
* **Reaction** — P(post-touch confirmation works) — *calibrated*
* **Policy Return (R1)** — predicted realised R per candidate trade — *can rank, can't yet find positive expectation*

The engine outputs go into three products:
1. **Daily Research Brief** — JSON + plain-text email; the customer-facing
   artifact (this is the FIRST product to ship)
2. **Strategy Diagnosis** — one-time deliverable; sells the subscription
3. **Private Alpha** — self-trading; furthest from ready

---

## 2. The two-axis framework

Almost every confusion in this project's history has come from collapsing
the two axes into one. Hold them separately.

### Axis 1 — the STACK (bottom to top, "necessity chain")

```
L0  Data
    ↓
L1  Features
    ↓
L2  Models
    ↓
L3  Signals / Alpha
    ↓
L4  Execution & Validation
    ↓
L5  Audit
    ↓
L6  Products
    ↓
L7  Customers
```

Each layer was born because the layer below couldn't answer the
question alone. Data needed features to be readable. Features needed
models to be predictive. Models needed signals to be tradeable. Signals
needed validation to be trusted. Validation needed audit to be
verifiable. **Infrastructure** is the spine that holds L0–L5 upright —
not a separate layer but the orthogonal connective tissue.

### Axis 2 — the HORIZON (the (L, T) parameter)

The same engine, pointed at different (level, time) pairs, produces
three commercial products:

| Horizon | (L, T) | Customer | Status |
|---|---|---|---|
| Swing (days–weeks) | structural pools, T = days | Positional traders | Engine ready; presentation not built |
| **Intraday MIS** | same-session pools, T = 60min | **Day-traders, current focus** | **Engine + labels + brief built** |
| Index Options | strike grid, T = expiry | **Index/options traders, our pilot customers** | Engine ready; level-to-strike translator built |

---

## 3. Each layer in detail

### L0 — Data

**What it is**: 5-minute OHLCV bars for ~25 NSE equities. Stored on
Google Drive (the user's warehouse), loaded via `liqpool/data.py`.
Normalised tz-naive UTC; IST = UTC + 5h30m everywhere downstream.

**Status**: built, working, sufficient for v1. Index OHLCV (Nifty,
Bank Nifty, etc.) is NOT yet wired in — that's a known follow-up.

**Cost to compute**: trivial (just load from disk).

**Commercial value**: zero standalone. Free public data, no moat. But
this is the substrate everything else stands on.

### L1 — Features

**What it is**: `liqpool/timing.py::StateFeaturizer`. Per-bar feature
computation. Currently 42 features in five groups:

| Group | Count | Examples | Why |
|---|---|---|---|
| Price-structure | 17 | `ret_6`, `mom_12_atr`, `zscore_close_50`, `range_6_atr` | The original snapshot features |
| MTF session-context | 5 | `htf_today_range_atr`, `htf_session_volume_ratio`, `htf_overnight_gap_atr` | "Where are we within today's session?" |
| AVWAP + FRVP | 9 | `avwap_today_dist_atr`, `poc_today_dist_atr`, `in_value_area_today` | Volume-weighted location |
| Expiry calendar | 4 | `days_to_monthly_expiry`, `is_weekly_expiry_day` | F&O cycle effects |
| Path / shape | 7 | `path_efficiency_30`, `is_gap_up_trap_fade`, `vol_regime_zscore_20d` | How we got here, not just where we are |

**All causal**: at bar j only bars i ≤ j contribute. Pinned by truncation
tests in `test_*_features.py`.

**Status**: built, validated. `vol_regime_zscore_20d` and
`days_to_monthly_expiry` are the #1 and #5 direction-model features in
the latest retrain.

**Cost to compute**: ~O(n) per asset at featurizer init time, where n is
bar count.

**Commercial value**: indirectly via the models. The feature engineering
itself is a real asset (replicating this would take a competitor
months), but the features are not the product.

### L2 — Models

The five trained heads:

#### Quality (Q) — `SectorMoERespectModel` in `liqpool/ml_model.py`
- **What it predicts**: P(pool is respected | features at touch)
- **Target**: binary respect label (≥0.5 ATR reaction post-touch)
- **Calibration**: isotonic + per-(TF, factor) bucket shrinkage with cap 0.30
- **Status**: WEAK (val AUC 0.541, barely above noise). The Q-DECOMPRESS
  commit revealed the truth that bucket shrinkage was hiding. Q is now
  context-only, NOT a hard live gate.
- **Sector experts**: 5 trained, 4 typically get 0% MoE weight (the
  global model dominates). Honest: the sector layer adds little.

#### Direction — `DirectionModel` in `liqpool/timing.py`
- **What it predicts**: P(price goes up | state at decision)
- **Target**: binary direction over 60-bar horizon
- **Status**: useful filter. OOS AUC 0.567. Top-quartile-confidence
  accuracy 58.8% — when the model is most certain, it's right ~59% of
  the time.

#### Proximity — `ProximityModel` (one per horizon) in `liqpool/timing.py`
- **What it predicts**: P(pool touched within H bars)
- **Horizons trained**: 12, 36, 60 (under MIS labels; was 78/156/312 pre-MIS)
- **Status**: THE STRONG MODEL. Overall AUC 0.92–0.95.
- **Honest caveat**: at 0-1 ATR distance (trader-relevant), AUC is
  0.73–0.76. The headline 0.95 is inflated by easy 10+ ATR cases where
  the answer is trivially "no". Still strong; just don't oversell.

#### Reaction — `ReactionModelSuite` in `liqpool/reaction_model.py`
- **What it predicts**: Three sub-targets — strict reaction post-touch,
  reclaim success, break continuation.
- **Status**: strong. strict_reaction OOS AUC 0.78, top-10 hit 88.7%.

#### Policy Return (R1) — `PolicyReturnModel` in `liqpool/policy_model.py`
- **What it predicts**: realised R per (mode, candidate trade) pair
- **Modes**: displacement_confirmed / touch_confirmed / blind_limit /
  reclaim_confirmed
- **Status**: spearman > 0.10 in all 4 modes (0.16–0.32). Can RANK
  trades. Top decile is "less catastrophic" but still negative R.

### L3 — Signals / Alpha

The arsenal — `liqpool/arsenal/`. Multiple alpha classes combine L2
outputs into trade candidates. Each alpha is an independent signal
generator with `candidates()` and `regime_tags()` methods. The arsenal
evaluator runs them all through the SAME cost + MIS + barrier rules so
no alpha can hide behind generous assumptions.

| Alpha | What it does | Status |
|---|---|---|
| `pool_reach` | Touched pool, V1 default barriers | Negative R; cost-dominated |
| `mean_reversion` | Z-score reversion (pool-free) | Negative R |
| `momentum` | Multi-bar continuation | Negative R |
| `quality_filtered_pool` | pool_reach gated by Q | Sparse |
| `direction_confirmed_pool` | pool_reach gated by direction | Sparse |
| `proximity_journey_baseline` | Single-input pre-touch (Opus original) | Sparse |
| `proximity_journey` | 7-input pre-touch ensemble (Codex upgrade) | Sparse but negative |
| `distance_5_8_journey` | Distance-band variant | Too sparse |
| `proximity_direction_soft` | Soft direction + proximity score | Sparse |
| `opening_range_to_pool` | ORB-filtered pre-touch | Too sparse |
| `sector_rotation_journey` | Sector-rotation gated | Sparse |
| `policy_return_*` | R1 regressor as gate | Research-only |

**Critical**: the soft-ensemble alphas (proximity_journey and family)
blend 7 inputs each. If they win or lose, we don't know which input is
responsible. The `proximity_journey_baseline` exists for attribution —
delta to `proximity_journey` is the only honest measure of whether the
extra inputs help.

### L4 — Execution & Validation

`liqpool/execution_backtest.py` (V1) and `liqpool/execution_simulator_v2.py`
(V2 with state-dependent slippage). Also `liqpool/arsenal/evaluator.py`
for the alpha framework's shared MIS + cost + barrier pipeline.

The cost model is real Zerodha intraday rupees: brokerage (capped
₹20/order) + STT (sell leg only) + exchange + SEBI + stamp duty + GST +
slippage (default 1 bps/side). See `liqpool/costs.py`.

**The min_target_to_cost_ratio filter** (added recently): rejects any
signal whose best-case target reward in INR is < 3× the round-trip
cost. Encodes the user's HDFC sizing instinct: small notional + tiny
target = uneconomic regardless of edge.

**Default notional**: ₹1,00,000 (bumped from ₹50k because ₹50k on a
₹770 stock gives qty 64, below practitioner sizing floor).

**Validation methods available**:
- Purged + embargoed walk-forward (default)
- CPCV (Combinatorial Purged Cross-Validation) — Lopez de Prado
- Time-shuffle null, sign-flip null, pool-level null, synthetic null

### L5 — Audit

**Phase 3A**: OOS prediction audit per model (global / sector / blended)
with calibration error per sector + distance bucket.

**Phase 3C**: Consistency / leakage replay. Re-runs labels on a sample
of OOS pools and compares — catches data leakage and label-window
truncation issues.

**Null tests**: Are there four of them:
- **Time-shuffle null** — shuffles bar timestamps within asset, retrains,
  checks alpha P&L collapses. Catches calendar leakage.
- **Sign-flip null** — flips direction labels at random, retrains, checks
  P&L collapses. Catches feature-target accidental correlation.
- **Pool-level null** — randomly relabels pool side. Catches pool-detection
  bias.
- **Synthetic null** — GBM regression against constant target. Catches
  optimisation bias.

**DSR** (Deflated Sharpe Ratio) — adjusts Sharpe for multiple-testing.

**Calibration audit** — Brier, log-loss, AUC, calibration error per
distance bucket, per sector, per (TF × factor) pocket.

**Outcome log** (just shipped) — predictions + resolutions table joined
nightly to produce the Yesterday Audit section of the daily brief.
This is the flywheel.

### L6 — Products

Three products planned, in sales order:

#### Daily Research Brief — `liqpool/products/daily_brief.py`
The customer-facing artifact. JSON contract in
`docs/daily_brief_schema.md`. Eight sections (brief_metadata, index_regime,
sector_regime, top_watchlist, options_suitability, avoid_list, key_zones,
confidence_notes, yesterday_audit). Renders to plain-text email body via
`brief_renderer.py`. Tipster-vocabulary guardrail enforced at render time.

#### Strategy Diagnosis ("X-Ray")
Not yet built as a discrete product, but every piece exists: per-asset
OOS gap analysis, per-sector calibration errors, R1 mode breakdowns,
cost-wall decomposition. One-time deliverable; converts customers to
the subscription.

#### Private Alpha
Self-trading. Currently negative R across all modes. Months out, maybe
never in current form. NOT the product that pays first.

### L7 — Customers

Three hand-picked pilot users (mostly options traders). Free first
week. PDF + email for v1. Web dashboard only if it costs < ₹300/mo
(static free-tier hosting).

---

## 4. The training pipeline — what happens when you run `multi_asset_run.py`

A training pass touches every layer top-to-bottom. Here's what
each step does and what it produces.

### Step 1: data load (`liqpool/data.py`)

For each symbol in the basket:
1. Read parquet from Google Drive warehouse.
2. Normalise to tz-naive UTC.
3. Validate OHLCV invariants (high ≥ max(o, c), low ≤ min(o, c)).
4. Return as a pandas DataFrame keyed by timestamp.

Per-asset cost: O(bars). Multi-asset cost: linear in basket size.

### Step 2: detection (`liqpool/pools.py` + `liqpool/detectors/*`)

For each symbol:
1. Compute swings (high/low fractals).
2. Apply factor detectors: EQH/EQL, FVG, Order Blocks, Rejection
   candles, ORB extremes, Volume Profile HVN, Previous Day/Week/Month
   highs and lows, Swing pivots.
3. Merge overlapping detections on the same side into pools.
4. Cross-TF reinforce: a pool detected on 15min that aligns with a
   60min pool gets a multi-TF count bonus.

Output: a list of `Pool` objects per asset, each with `price_low`,
`price_high`, `score`, `tfs`, `contributors`, `available_at`,
`formed_at`.

### Step 3: testing (`liqpool/tester.py`)

For each pool:
1. Find the first bar where price touches the zone (`touched_at`).
2. Walk forward from touch under MIS-aware horizon cap (`intraday_session_only`).
3. Classify outcome: `respected_strong`, `respected_weak`,
   `broken_strong`, `broken_weak`, `swept_and_reclaimed`,
   `touched_no_signal`, `untouched`.

This is where the **MIS-honest** labels happen. The label window is
capped at same-IST-day EOD (15:15 IST) so a pool can't be credited
with cross-session price action a MIS trader couldn't capture.

### Step 4: walk-forward (`liqpool/walkforward.py`)

For each symbol:
1. Split pools chronologically into train and OOS windows (with purge
   + embargo gap — Lopez de Prado discipline).
2. Refit per-asset baseline + per-asset quality calibration on train.
3. Carry the OOS pools forward unchanged.

This is what makes the OOS numbers honest — no model has seen the
OOS bars at training time.

### Step 5: unified ML (`liqpool/multi_asset.py`)

Cross-asset training:
1. Build the multi-asset feature frame from all train pools.
2. Fit the global Q model on the pooled data.
3. Fit per-sector experts.
4. Dynamic shrinkage: each expert's MoE weight is scaled by how much
   it beats the global model on held-out data. Experts that don't beat
   the global get 0% weight.
5. Bucket calibration: for each (TF count, factor) bucket with ≥30
   OOS samples, pull predictions toward the bucket's empirical OOS
   rate (capped at 0.30 to avoid Q compression).

### Step 6: direction + proximity (`liqpool/timing.py`)

For each symbol's OOS window:
1. Generate snapshots every `sample_every` bars (default 12 = hourly).
2. Each snapshot carries the state features + the future bars'
   max-high / min-low arrays + the per-pool bars-to-touch.
3. Fit `DirectionModel` on (state → direction_label) over the basket.
4. Fit one `ProximityModel` per horizon (12, 36, 60) on
   (state + pool features → touched_within_horizon).
5. Optional Stage C: per-distance-bucket isotonic recalibration
   composed on top of the global isotonic.

### Step 7: reaction (`liqpool/reaction_model.py`)

For pools that were touched:
1. Build features at touch.
2. Fit three sub-models: strict_reaction (post-touch ≥0.5 ATR reaction),
   reclaim_success (sweep-and-reclaim pattern), break_continuation
   (decisive break followed by continuation).

### Step 8: policy labels + R1 (`liqpool/policy_labels.py` + `policy_model.py`)

For each pool:
1. Simulate each execution mode (displacement_confirmed, touch_confirmed,
   blind_limit, reclaim_confirmed).
2. Record (trade_generated, net_R) per (pool, mode).
3. Fit `PolicyReturnModel` per mode: Huber regression on winsorised
   `policy_target_return_r`. Reports spearman + top10/top25 R.

### Step 9: execution backtests

V1 (`liqpool/execution_backtest.py`) and V2
(`liqpool/execution_simulator_v2.py`) replay the OOS pools under each
mode with realistic Zerodha costs. Output: net R per mode, profit
factor, max drawdown.

### Step 10: arsenal evaluation (`analysis/run_arsenal.py`)

Wraps everything in the alpha framework:
1. Register all alphas in the registry.
2. Walk each (alpha × asset) and emit signal candidates.
3. Apply MIS + cost + barrier resolution via the shared evaluator.
4. Compute alpha correlation matrix, per-alpha turnover, cost-wall
   summary at multiple cost multipliers.
5. Run null tests (time-shuffle + sign-flip) on each alpha.
6. Hold-out: select winners on first 80%, report on last 20%.

### Step 11: artifact emission

Writes ~20 CSV / parquet / JSON files into the output directory:
* `multi_asset_summary.json` — top-level metrics
* `research_summary.json` — verdict + per-asset dashboard
* `live_plan.json` — tradeable / watch_only / rejected_with_reason
* `phase3_oos_prediction_audit.csv` — per-prediction audit
* `phase3_sector_calibration.csv` — per-sector calibration error
* `execution_backtest_*.csv` — V1 + V2 trade fills + summaries
* `policy_return_model_report.csv` — R1 metrics
* `reaction_alerts.csv` — post-touch confirmation alerts
* (after `BRIEF-V1` and `PRODUCT-CORE` land in the pipeline) `daily_brief.json` + `daily_brief.txt`

---

## 5. How to read a training output

The summary you get back from a training pass has these big sections:

### MULTI-ASSET SUMMARY
The headline: pooled OOS respect rate, per-asset OOS gap, basket-level overfit.

**Interpretation**:
- Pooled OOS broad respect 50.4% means about half of OOS pools react.
  Above 50% = there's structure, not pure noise.
- Per-asset overfit gap > +10% on individual stocks (e.g. HINDUNILVR
  +16.5%) means the train fit isn't generalising for those names —
  candidates for stricter validation or exclusion.

### unified ML feature importance
Lists the top 15 features by LightGBM gain.

**Interpretation**:
- Look for the NEW features (the 25 added this week) in the top 10.
- `vol_regime_zscore_20d` at #1 in direction means "where is today's vol
  relative to the last 20 days" was the missing signal.

### Phase 3A OOS prediction audit
Per-model (global / sector_expert / blended) Brier + log-loss + AUC +
top10_hit + calibration error.

**Interpretation**:
- AUC 0.541 = barely above random. AUC 0.55+ = usable. AUC 0.65+ =
  strong. AUC 0.80+ = exceptional (and probably leaky if it appears in
  a new model — be suspicious).
- top10_hit > base_rate by 5%+ = model can rank meaningfully.
- |calibration_error| < 0.05 = well-calibrated; > 0.08 = drifting.

### unified proximity models
Per-horizon AUC + decile lift + per-distance-bucket metrics.

**Interpretation**:
- Overall AUC can be inflated by easy cases. ALWAYS check the 0-1 ATR
  bucket — that's the trader-relevant case. AUC 0.70+ at 0-1 ATR is real.
- decile_lift = top-decile-base / bottom-decile-base. inf or 100x+ is
  normal when base rate is < 1%.

### POST-TOUCH REACTION QUALITY
Per-bucket breakdown of touched-pool outcomes.

**Interpretation**:
- "broken_strong" share > "respected_strong" share = bias toward
  breaks. Currently 48.6% vs 16.7% — the post-touch reaction story is
  more about breakdowns than respects.

### EXECUTION BACKTEST
Per-mode net R with Zerodha costs.

**Interpretation**:
- Negative net R = the strategy loses money at the configured cost level.
- Compare V1 to V2 — V2 uses state-dependent slippage; the V2 numbers
  are more realistic.
- Profit Factor (PF) < 1.0 = losing strategy; 1.2+ = barely positive;
  1.5+ = real edge; 2.0+ = exceptional.

### POLICY RETURN MODEL (R1)
Per-mode mean realised R + top10/top25 R + spearman.

**Interpretation**:
- spearman > 0.10 = regressor can rank. spearman > 0.30 = strong ranker.
- top10_R > 0 = positive expectancy when filtered to top decile.
  Currently negative for all modes — we can rank but not yet find a
  positive subset.

### TODAY'S CROSS-ASSET PLAN
Per-asset dashboard + verdict.

**Interpretation**:
- VERDICT = WATCH means no setup; AVOID means actively stand aside;
  TRADEABLE means at least one pool passed every gate.
- Best gated net EV today > 0 = at least one candidate has positive
  expectancy after costs.

### SECTOR INTELLIGENCE
Per-sector regime + money rotation + correlation matrix.

**Interpretation**:
- conv (conviction) > 0.5 = strong regime signal.
- 5d vs 60d divergence = money rotation in or out.

---

## 6. What the audits catch (and what they don't)

**The audits are not equally valuable.** Some are catching real bugs;
some are producing 400+ warnings that nobody reads.

| Audit | What it catches | Current value |
|---|---|---|
| Phase 3A OOS prediction audit | Calibration drift per sector / distance | High — feeds confidence_notes |
| Phase 3C leakage replay | Label-window inconsistency, time leakage | Medium — currently 480 WARN rows nobody reads |
| Time-shuffle null | Calendar / time-based leakage in features | High — would catch a serious class of bugs |
| Sign-flip null | Accidental feature-target correlation | High — same |
| Pool-level null | Pool-detection bias confused with alpha | High — Track A POOL_LEVEL_FAIL was caught by this |
| Synthetic null | Optimisation bias from feature selection | Low-medium — overlap with sign-flip |
| Moment null | Time-series moment artifacts | Spec'd, not run yet |
| DSR | Multiple-testing-adjusted Sharpe | High — but currently underused in reports |

---

## 7. Glossary

| Term | Meaning |
|---|---|
| **R** | Risk units. 1R = 1 stop-distance of move. Standard quant unit. |
| **ATR** | Average True Range. Volatility measure. We use 14-period and 60-period. |
| **MIS** | Margin Intraday Square-off. Indian intraday-only product. Position must close by 15:15 IST. |
| **Pool** | A liquidity-aligned price zone (e.g. EQH, FVG, OB). The model's primary structural unit. |
| **Pocket** | A specific (TF count, factor type, direction) combination. We bucket pools into pockets for calibration. |
| **Touch** | Price entering the pool's [price_low, price_high] zone. |
| **Respect** | Post-touch ≥0.5 ATR reaction in the pool's expected direction. |
| **Break** | Post-touch close beyond the pool's far side by ≥0.5 ATR. |
| **Reclaim** | Initial break followed by close back inside within 12 bars. |
| **Q (quality)** | P(respect | touched). |
| **T_today** | P(touch within today's session). |
| **EV** | Expected Value, in INR or R. Net of costs. |
| **Brier** | Mean squared error of probability predictions. Lower = better calibrated. |
| **AUC** | Area under ROC curve. 0.5 = random; 1.0 = perfect. |
| **Spearman** | Rank correlation. -1 to +1. 0 = no rank relationship. |
| **Decile lift** | top-decile-hit-rate / bottom-decile-hit-rate. |

---

## 8. How the layers connect commercially

```
L0 Data  →  L1 Features  →  L2 Models  →  L6 Daily Brief  →  L7 Customer pays
                                ↓
                         L3 Alpha
                                ↓
                         L4 Execution
                                ↓
                         L5 Audit  →  L6 Yesterday Audit section
                                ↓
                         L5 Audit  →  L6 Strategy Diagnosis (sales wedge)
```

Three commercial paths emerge from the same engine:

1. **Subscription** (research/context) — L0 → L2 → L6 Daily Brief.
   Lowest bar; sells first. Customer pays for context, makes own decisions.
2. **Diagnosis** (infrastructure-as-service) — L4 + L5 → L6 X-Ray.
   Sales wedge that converts subscriptions. One-time deliverable.
3. **Private alpha** — L0 → L3 → L4. Self-trading. Furthest from ready;
   may never clear costs in current form.

The brief is the entry. The diagnosis is the conversion. The alpha is
the dream. Don't confuse the order.

---

## 9. What this repo IS and IS NOT

**This is**:
* A tier-3 research-grade quant infrastructure (CPCV, triple-barrier,
  null tests, MIS-aware labels, calibration stack).
* A 42-feature engine producing 5 model outputs across 25 NSE stocks
  with 90+ days of OOS validation per pass.
* The first commercial product (Daily Brief) as code, with the
  data flywheel (outcome log) wired in.
* A SEBI-safe-harbour designed product (regime-and-probability
  language only, tipster guardrail enforced at render time).

**This is NOT**:
* An auto-trading bot. Private alpha is still negative-R after costs.
* A signal service. There are no "buy X at Y" outputs anywhere in the
  customer-facing path — that's enforced by code, not just policy.
* A tier-4+ shop. We lack portfolio-level risk, execution algos, alt
  data, and microstructure. Those are future work.

For where each layer stands commercially see `COMPANY_MAP.md`.
For the live state board (in-flight work + decisions log + queue) see
`COORDINATION.md`.
