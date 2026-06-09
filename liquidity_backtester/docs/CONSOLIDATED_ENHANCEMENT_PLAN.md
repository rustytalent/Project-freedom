# CONSOLIDATED ENHANCEMENT PLAN — v1

> **Owner**: project owner
> **Authored**: 2026-06-09
> **Status**: living document; supersedes nothing
> **Cross-references**:
>   - `docs/MASTER_PLAN.md` — work board / streams A-H
>   - `docs/RESEARCH_ROADMAP.md` — strategic backlog T0-T6
>   - `docs/SYSTEM_OVERVIEW.md` — what the pipeline does
>   - `docs/strategy_diagnosis_spec.md` — diagnosis product spec

---

## §0 — Purpose

This document consolidates two recent inputs into one prioritised plan:

1. The **six-subsystem repo audit** (June 2026) — boundary bugs in
   validation/calibration, gaps in execution realism, statistics that
   overstate edge, integrity gaps in the outcome log, and methodology
   weaknesses across detectors, models, options, and arsenal.
2. The **synthetic-data flywheel generalisation** — extending T4.5 of
   the research roadmap into a full inventory of paired-data factories
   already living inside the engine, plus the new models each one
   unlocks.

It is **not** a replacement for `MASTER_PLAN.md` (the master work
board) or `RESEARCH_ROADMAP.md` (the strategic backlog). It is the
tactical synthesis: *here is what to do, in what order, with what
acceptance gates, to close the audit and unlock the flywheel at the
same time.*

Streams A-H are reserved by the master plan. This plan opens streams
I-P. Each stream is independent enough to start in parallel; cross-
dependencies are explicit in §3.

---

## §1 — Unified picture

The audit and the flywheel idea reinforce each other in a way that
isn't obvious at first.

The audit's recurring failure pattern is **small boundary errors in
exactly the places that protect calibration**: off-by-one in walk-
forward, embargo edge using `<=` instead of `<`, ATR features
backfilled across session-start, unresolved predictions silently
dropped from the public calibration table. None of these show up in
headline backtest numbers. They show up as quiet drift in live
calibration — which is the product.

The flywheel idea fixes the *opposite* problem: the engine throws away
the data it generates. Every SKIP, every avoided pool, every macro-
gate close, every slippage realisation, every drift event has a
training row paired with it. Today, almost none of these rows are
collected.

The unified play is:

- **First close the boundaries.** No flywheel that trains on poisoned
  data is worth building.
- **Then build the shadow-data collector** that captures every decision
  the engine declines.
- **Then train the new models** on the shadow data, on the simulator's
  own output, on the outcome log itself.
- **Then upgrade the calibrators** (empirical Bayes, conformal,
  temperature scaling) on the cleaned-up substrate.

That sequence — integrity → substrate → models → calibrators — is the
backbone of streams I through P below.

---

## §2 — Streams

Each stream entry has:
- **Goal** — what it ships
- **Files** — concrete paths added/edited
- **Acceptance** — how we know it's done
- **Effort** — rough commit count + wall-clock
- **Depends on** — preconditions
- **Unlocks** — downstream work it enables

---

### Stream I — Calibration integrity (the audit boundary fixes)

**Goal**: close every boundary bug the audit flagged that touches the
calibration contract. These are small diffs, large consequences.

**Files**:
- `liqpool/walkforward.py:122-132` — fix off-by-one in `_safe_end_ts`.
- `liqpool/validation.py:228-230` — embargo edge `<=` → `<`.
- `liqpool/data.py:371-375` — replace `dropna(how="any")` after
  resample with high/low-from-close fill, then drop only on
  close+volume.
- `liqpool/timing.py:131-132` — remove `bfill()` on early-session ATR;
  emit NaN + a `is_warmup` flag for downstream.
- `liqpool/ml_model.py:77-80` — make `label_end_time` raise on
  all-None candidates instead of silently returning `available_at`.
- `liqpool/ml_model.py:134` — per-sample base period for mixed-
  frequency embargo.
- `liqpool/triple_barrier.py:88-91` — log intraday-truncated outcomes
  with a sentinel so downstream calibrators can stratify on it.
- `liqpool/products/daily_brief.py` — add hard publish-time data-cutoff
  assertion (`base_df.index[-1]` must be before 08:30 IST cutoff or
  raise).
- `liqpool/products/daily_brief.py:115` — rename `side_from_open` →
  `side_from_close` (it's computed from prior close, not open). Keep a
  deprecation alias for one cycle so the outcome-log schema doesn't
  break.
- `liqpool/featurize.py:145` — fix 12-bar momentum (was 11 bars).
- `liqpool/sample_weights.py:72-80` — replace mean-normalisation with
  min-floor-preserving rescale.
- `liqpool/leakage_audit.py:314-319` — document bar-timestamp convention
  (index = open) and add an assertion.

**Acceptance**:
- All existing tests still pass.
- New tests added for each fix that would have caught the bug.
- Calibration audit numbers on the existing bundle change by < ±1%
  (proves none of the fixes accidentally invalidated training).
- One unified `tests/test_boundary_invariants.py` rolling all of these
  into a single audit suite.

**Effort**: ~6-8 commits, one session each.

**Depends on**: nothing. Pure cleanup.

**Unlocks**: everything downstream that trains on bundle outputs.

---

### Stream J — Statistical honesty

**Goal**: stop overstating edge in arsenal + brief + outcome log.

**Files**:
- `liqpool/arsenal/null_tests.py:169` — default `n_trials=1000`
  (currently 100); document that `>= 1000` is required.
- `liqpool/arsenal/evaluator.py` — add Sharpe + Bailey-style deflated
  Sharpe to `per_alpha_summary`.
- `liqpool/arsenal/registry.py` — wrap `research_registry()` with
  Bonferroni (or, better, White's Reality Check / Hansen's SPA) when
  more than one alpha is null-tested in a session.
- `liqpool/products/outcome_log.py:320-348` — report `n_resolved`,
  `n_unresolved`, and a `coverage_pct` per bucket. Calibration error
  is computed on resolved only; unresolved are disclosed.
- `liqpool/products/outcome_log.py` — SHA-256 sidecar hash chain per
  partition. New module `liqpool/products/outcome_hash_chain.py`.
- `liqpool/products/brief_renderer.py` — surface
  `coverage_pct < 100%` and the hash-chain proof URL in the Yesterday
  Audit section.

**Acceptance**:
- Arsenal evaluator emits Bonferroni-adjusted significance + deflated
  Sharpe in every alpha summary row.
- Re-run the existing arsenal report; previously "significant" alphas
  with marginal p-values are correctly downgraded.
- Outcome log writes `*.sha256` sidecars; a `verify_hash_chain` CLI
  re-derives them and matches.
- Yesterday Audit displays coverage and never hides unresolved
  predictions.

**Effort**: ~5 commits.

**Depends on**: Stream I (don't add honesty layers on top of broken
boundaries).

**Unlocks**: the calibration dataset (T4.3) becomes externally
verifiable rather than trust-us. Diagnosis product (T3.1) gets the
deflated Sharpe column it needs.

---

### Stream K — Execution realism v3

**Goal**: V2 simulator is decent. V3 swaps the magic constants for
learned components, trained on the simulator's own paired output. This
is the slippage idea, applied across the execution surface.

**Components**:
- **Slippage realisation model** (the seed): trains on every V2 trade's
  `(state features, intended price, realised price)` tuple. Predicts
  `slippage_bps` from `(session_phase, vol_regime, distance_atr,
  size, side, mode)`. Replaces the constant 1 bps default + the rule-
  based V2 state-dependent slippage. The trained model is plugged
  back into V3 so the simulator uses it.
- **Fill probability model**: trains on `(limit distance from best,
  bars-to-deadline, regime) → did this hypothetical limit fill?`.
  Replaces the V2 assumption that targets fill 100% of the time.
- **Time-to-touch / time-to-stop survival model**: trains on triple-
  barrier outcomes. Outputs `E[bars to target]` and `E[bars to stop]`
  conditional on the entry context. Used for sizing (smaller size on
  slow trades) and position aging.
- **Market-impact model (parametric, not trained)**: add a sqrt-impact
  term to V3 for `notional > 200k`, plus include STT/stamp on the
  correct legs for shorts (audit flagged the asymmetry).
- **Near-zero price guard**: reject sizing when `entry_price_ref < 0.01`
  before notional ÷ price explodes.
- **Same-bar tie-breaker is configurable** in the V3 config.

**Files**:
- `liqpool/execution_simulator_v3.py` — wraps V2, plugs in the learned
  models. V2 stays untouched as the ground-truth simulator.
- `liqpool/execution/slippage_model.py` — trainer + inference.
- `liqpool/execution/fill_model.py` — trainer + inference.
- `liqpool/execution/survival_model.py` — trainer + inference (Cox or
  GBM survival; LightGBM has `objective='survival'` support).
- `scripts/train_execution_models.py` — runs all three trainers.
- `tests/test_execution_v3_realism.py` — V3 must produce *less
  optimistic* P&L than V2 on the same bundle; otherwise the trained
  models are wrong.

**Acceptance**:
- V3 backtests on the existing bundle show mean_R between V2 (too
  optimistic) and a stress-test baseline (1.5x V2 cost). Specifically:
  V3 mean_R ≈ V2 mean_R − 30 to 80 bps.
- Slippage model OOS RMSE on held-out trades < 5 bps.
- Survival model concordance index > 0.60.
- All audit fixes #8–#12 from the execution audit incorporated.

**Effort**: ~10-12 commits. Two-week project.

**Depends on**: Stream I (V2 outputs we train on must be correct).

**Unlocks**: T4.1 (slippage dataset) becomes shippable. Sizing layer
gets honest expectations. Brief publishes "expected hold time" per
trade as a feature.

---

### Stream L — Shadow-data collector (the substrate)

**Goal**: every decision the engine declines today is data thrown away.
This stream installs a single collector that captures every gate's
counterfactual row to one append-only parquet.

The points to hook:
- Every SKIP from the options executor (T1.1).
- Every avoidance flag (`AvoidEntry`) — what would the avoided
  basket have done in the next session?
- Every pool below `min_target_to_cost_ratio` — record the would-be
  trade's economics.
- Every drift flag — record the precursor trajectory and the recovery
  time once it clears.
- Every macro-gate-closed day — record the would-be trades the gate
  blocked.
- Every Q < threshold pool that was touched — hard negative for Q.
- Every alpha that didn't fire when another did — co-firing pattern.

**Files**:
- `liqpool/products/shadow_log.py` — new module. Single class
  `ShadowLogger` with `record(event_kind, payload)` interface.
  Append-only parquet partitioned by `(trading_date_ist, event_kind)`.
  Schema versioned.
- Integration points: ~12 call sites across `options/executor.py`,
  `products/daily_brief.py`, `drift.py`, `arsenal/registry.py`.
- `analysis/replay_shadow_counterfactuals.py` — for each shadow row,
  resolve the counterfactual outcome (what would the trade have done?)
  and append a resolution.
- `tests/test_shadow_log.py` — schema invariants, append-only proof,
  resolution correctness.

**Acceptance**:
- One run of the existing pipeline emits ≥ 5 distinct event kinds.
- Replay script resolves ≥ 80% of recorded counterfactuals on the
  next session of data.
- Counterfactual outcomes can be joined with the outcome log on
  `trading_date_ist`.

**Effort**: ~4 commits, one week. ~200-300 LOC + tests + docs.

**Depends on**: Stream I (we want clean labels in the shadow log).

**Unlocks**: every Stream M model below. This is the single highest-
leverage piece of infrastructure in the plan.

---

### Stream M — Synthetic-data model suite

**Goal**: ten new models, each one trained on data the engine already
generates (or that Stream L will start generating). This is the bulk
of the flywheel.

| # | Model | Trains on | Replaces / adds | Effort |
|---|---|---|---|---|
| M.1 | Slippage realisation | V2 trade outputs | V3 simulator default | covered in Stream K |
| M.2 | Fill probability | Limit-order sim runs | V3 target-fill assumption | covered in Stream K |
| M.3 | Time-to-touch survival | Triple-barrier outcomes | New sizing input | covered in Stream K |
| M.4 | **SKIP regret estimator** | Shadow log (SKIP events) | New executor input | 4 commits |
| M.5 | **Detector trust router** | Detector firings + outcomes | Per-detector trust weight | 3 commits |
| M.6 | **Drift-imminent model** | Calibration history + drift events | Earlier drift warning | 3 commits |
| M.7 | **Reaction archetype clusterer + classifier** | Touch reactions (5-bar paths) | New label dimension | 5 commits |
| M.8 | **Bucket-aging model** | Q-bucket calibration over time | Replaces fixed Wilson `n/(n+20)` | 3 commits |
| M.9 | **Cross-asset transferability model** | Multi-asset prediction pairs | Routes signal cross-asset | 4 commits |
| M.10 | **Brief confidence meta-calibrator** | Outcome log per-bucket trajectories | Recalibrates tomorrow's confidence | 3 commits |

For each model:
- Trainer script in `scripts/train_<model>.py`.
- Inference module in `liqpool/<area>/<model>.py`.
- Tests in `tests/test_<model>.py` with causality + truncation pins.
- A short methodology doc in `docs/models/<model>.md`.

**Acceptance per model**: spelled out in the model's doc; common
template = OOS metric beats baseline + calibration error within
±0.05 + causality test passes.

**Effort**: ~25 commits total across the ten models. Spread over 6-8
weeks.

**Depends on**: Streams I, K (M.1-M.3), L (M.4, M.6, M.8 explicitly).

**Unlocks**: directly improves the brief (M.7, M.10), the executor
(M.4), the dashboard's drift warnings (M.6), the calibration story
(M.8). M.5 cleans up the detector audit's biggest finding.

---

### Stream N — Calibration upgrades

**Goal**: the calibration audit verdict was "adequate, not primitive."
This stream pushes it to "good." Three components in priority order.

**Components**:
- **Empirical Bayes (beta-binomial) bucket shrinkage** — replaces the
  fixed Wilson `n/(n+20)` shrinkage in `ml_model.py:361-398`. Removes
  the arbitrary "20", fits the prior strength from the data itself.
  Fixes Q-compression at the root rather than capping it with
  `shrinkage_max=0.30`.
- **Conformal prediction intervals** — adds honest `[lower, upper]`
  bands around every published probability. The brief becomes "P(touch)
  = 62% [48–74]" instead of a bare point estimate. This is the single
  most upgradable item from a customer-facing angle.
- **Temperature scaling / Platt smoothing** — optional post-isotonic
  layer. Improves OOS calibration in regions the isotonic regression
  was sparse in.

**Files**:
- `liqpool/calibration/empirical_bayes.py`
- `liqpool/calibration/conformal.py`
- `liqpool/calibration/temperature.py`
- `liqpool/ml_model.py` — opt-in flags for each layer.
- `liqpool/products/brief_renderer.py` — render `[lower, upper]` bands.
- `tests/test_calibration_upgrades.py`

**Acceptance**:
- Empirical Bayes shrinkage produces a wider Q range than fixed-
  shrinkage on the same bundle (Q-decompress validation).
- Conformal coverage on OOS test set within ±2% of nominal
  (90% conformal interval covers 88-92% of held-out outcomes).
- Temperature scaling reduces log-loss on the held-out distance
  buckets by ≥ 2%.

**Effort**: ~6 commits. Two weeks.

**Depends on**: Stream I (clean training labels).

**Unlocks**: the brief publishes intervals not points (huge customer-
trust win). T4.3 calibration dataset has its first sellable
differentiator. Diagnosis product can use conformal bands as a
reliability metric.

---

### Stream O — Brain v0 (alpha mixer)

**Goal**: T2.1 (alpha mixer that conditions on regime) is the only
piece of the Brain that's safe to start before there's a validated
positive-edge alpha. This stream builds the minimal version on top of
the existing arsenal, so the moment an alpha proves positive (Stream
D Gate 4), the mixer is ready to combine it with the others.

**Components**:
- Per-(regime, alpha) historical net-R lookup table, built from the
  arsenal evaluator's existing per-trade output.
- A regime classifier using the existing path/context features
  (`vol_regime_zscore_20d`, `path_efficiency_30`, `is_gap_up_trap_fade`).
- An inference-time mixer that: classifies current regime → looks up
  alpha weights for that regime → blends.
- A **null-test for the mixer itself** (most important): on a shuffle
  of regime labels, the mixer must not produce a positive expectation.

**Files**:
- `liqpool/brain/regime_classifier.py`
- `liqpool/brain/alpha_mixer.py`
- `liqpool/brain/mixer_null_test.py`
- `analysis/run_brain_v0_backtest.py`
- `docs/brain_v0_spec.md`

**Acceptance**:
- Mixer beats the best individual alpha by ≥ +0.05 ATR on OOS test
  set, with `p < 0.01` under shuffle null.
- If no individual alpha is positive yet, the mixer must explicitly
  return `abstain` rather than silently combining negatives. (This
  enforces the masterplan's named failure mode: never combine
  negative-edge alphas into a more elaborate negative-edge system.)

**Effort**: ~6 commits. Two weeks.

**Depends on**: Stream I; at least one positive-edge alpha (most
likely from Stream D Gate 4 — the options executor).

**Unlocks**: T2.2 (strategy synthesis layer) — the real Brain — once
the mixer has proven the regime-conditioning premise.

---

### Stream P — New data inputs (T5.1 extensions)

**Goal**: the more independent data sources, the more uncorrelated
alpha sources possible. T5.1 lists 7. This stream adds the cheap
incremental ones and wires them.

**New inputs (free, in priority of leverage)**:
- **NSE FII/DII daily flows** — `liqpool/ingest/nse_flows.py`.
- **Block/bulk deal data** — `liqpool/ingest/nse_deals.py`.
- **F&O bhavcopy OI build / unwind** — extends existing T5.1 #3.
- **Promoter pledge change filings** — `liqpool/ingest/sebi_pledge.py`.
- **Insider trade filings** — `liqpool/ingest/sebi_insider.py`.
- **USD/INR currency** — macro overlay feature group.
- **Sector ETF flows** — cleaner than the basket synthesis.
- **Pre-open auction data** — first-bar prediction feature.
- **Earnings surprise table** — event-window feature.

**Files**:
- `liqpool/ingest/<source>.py` per source.
- `liqpool/features.py` — feature extractors that consume the new
  warehouses.
- `tests/test_new_data_ingest.py` — schema invariants.

**Acceptance**:
- Each source ingests cleanly into a versioned parquet partition.
- At least three new features land in the next training run.
- Feature-importance ranking shows ≥ one of the new features in the
  top 15 of any model.

**Effort**: ~9-12 commits across the sources. One per source.

**Depends on**: nothing engineering-wise; requires user to confirm
data-source ToS for each.

**Unlocks**: the diversity case from T5.1 — independent alpha sources
because the inputs themselves are independent.

---

## §3 — Sequencing

```
┌──────────────────────────────────────────────────────┐
│ Stream I — Calibration integrity (close audit gaps)  │   ← do first
└──────────────┬───────────────────────────────────────┘
               │
   ┌───────────┼────────────────┬──────────────────┐
   ▼           ▼                ▼                  ▼
┌───────┐  ┌───────┐       ┌────────┐         ┌────────┐
│ J     │  │ L     │       │ N      │         │ P      │
│ Stats │  │ Shadow│       │ Calib. │         │ Data   │
│ honest│  │collect│       │ upgrade│         │ inputs │
└───┬───┘  └───┬───┘       └────────┘         └────────┘
    │          │
    │          └─────┬──────────────┐
    ▼                ▼              ▼
┌────────┐      ┌────────┐     ┌────────┐
│ K      │      │ M.4    │     │ M.5-10 │
│ Exec   │      │ Regret │     │ Models │
│ v3     │      └────────┘     └────────┘
└────────┘
                                  │
                                  ▼
                            ┌──────────┐
                            │ O        │
                            │ Brain v0 │   ← only after one
                            │ (mixer)  │      positive-edge alpha
                            └──────────┘
```

**Recommended execution order if doing serial**:
1. Stream I (one to two weeks)
2. Stream L (one week, in parallel with I if possible)
3. Stream J + N (two weeks each, parallel)
4. Stream K (two weeks)
5. Stream M.4 → M.5 → M.6 → M.7 → M.8 → M.9 → M.10 (~6-8 weeks)
6. Stream P (rolling, one source per week, no blocking)
7. Stream O (only when Stream D produces its first Gate-4 positive)

**Total wall-clock**: 12-16 weeks of focused engineering, assuming one
to two parallel work-streams.

---

## §4 — What I can ship cleanly

Honest accounting of what an autonomous coding agent can land vs what
requires the project owner.

| Stream | What I can ship as code+tests+docs | What needs the owner |
|---|---|---|
| **I** | 100% — every fix is a known boundary, the tests are writable | Just review + merge |
| **J** | 90% — Bonferroni, deflated Sharpe, hash chain, coverage disclosure all clean | Decide which multiple-testing correction (Bonferroni vs SPA vs Reality Check); pick once |
| **K** | 60% — V3 framework, slippage/fill/survival trainers, V3 simulator wrapper | Train on real VPS bundle data; validate against live fills |
| **L** | 100% — pure infrastructure, integrations at known sites, tests | Just review + merge |
| **M.1-M.3** | 60% — same as Stream K — framework yes, training run no | VPS training |
| **M.4** | 80% — regret model framework + trainer | VPS training on shadow log once L is populated |
| **M.5** | 80% — detector trust router framework | VPS training |
| **M.6** | 70% — drift-imminent model | Owner labels which historical events count as drift events |
| **M.7** | 70% — reaction archetype clustering | Owner reviews the cluster catalog and names the archetypes |
| **M.8** | 90% — bucket-aging model | VPS training |
| **M.9** | 70% — cross-asset transferability | Owner reviews the asset-pair similarity matrix |
| **M.10** | 90% — brief confidence meta-calibrator | VPS training |
| **N** | 80% — all three calibrators clean to land as code | VPS retrain |
| **O** | 40% — alpha-mixer skeleton + null test framework | Waits on a positive-edge alpha; owner decides when to flip the switch |
| **P** | 70% — ingestion parsers + feature extractors per source | Owner provides creds for paid sources; reviews each source's ToS |

Aggregated: **~75-80% of the plan I can land as clean code-and-tests
in the repo**. The remaining 20-25% is owner-side: training on real
data, business decisions, source credentials.

That 75% is enough to land Streams I, J, L, the *frameworks* for K and
M, all of N, and most of P inside this branch. The training runs and
the Brain v0 flip-switch are reserved for the owner.

---

## §5 — End-state capabilities

After the full plan ships, the engine has:

**Methodology axis**:
- Purged + embargoed walk-forward with corrected boundaries.
- Triple-barrier labels with intraday-truncation stratification.
- Leakage probes (existing) + boundary-invariant CI tests (new).
- Multi-stage calibration: isotonic → empirical-Bayes-shrunk buckets
  → temperature smoothing → conformal intervals.
- Statistical honesty: Bonferroni or SPA on alpha registry,
  deflated Sharpe everywhere, ≥ 1000 permutations on null tests.
- Drift detection: from absolute threshold (current) to learned
  drift-imminent model with hysteresis.

**Execution axis**:
- V2 (rule-based state-dependent) preserved as ground truth.
- V3 (learned slippage / fill / survival) for production sizing.
- Indian cost stack: STT/stamp on correct legs both sides, GST,
  exchange, SEBI, stamp, brokerage. Optional circuit-halt scenario.
- Near-zero price guard. Configurable same-bar tie-breaker.

**Data axis**:
- 7 existing + 9 new data inputs (FII/DII flows, deal data, pledge,
  insider, USD/INR, sector ETFs, pre-open, earnings, F&O OI).
- Shadow log collecting every declined decision for future training.
- Outcome log with hash-chain integrity, coverage disclosure, and
  learned meta-calibration of confidence.

**Model axis**:
- Existing five heads (Q, Direction, Proximity, Reaction, R1)
  recalibrated against the new substrate.
- Ten new models: slippage, fill, survival, regret, detector trust,
  drift-imminent, reaction archetype, bucket aging, cross-asset
  transferability, brief confidence meta-calibrator.
- Brain v0 (alpha mixer) waiting for the first Gate-4 positive.

**Audit axis**:
- Hash-chained outcome log (verifiable from outside).
- Public calibration with coverage disclosure.
- Conformal intervals on every published probability.
- Drift events have precursor trajectory + recovery time.

**Where this places the engine in the landscape**:
- **Above 95% of Indian retail quants** — they don't run purged WF,
  realistic costs, or any calibration audit at all.
- **Above most small Indian prop desks on methodology axis** — they
  rarely have all of: purged WF + leakage probes + hash-chain audit
  + deflated Sharpe + conformal intervals. They optimise for capital
  deployment; you optimise for honest disclosure.
- **Comparable to institutional research infrastructure stacks** —
  not Renaissance / Two Sigma / DE Shaw scale. They have decades of
  talent, capital, and exotic data this engine doesn't.
- **The audit-in-public contract is unique in Indian retail/prop.**
  Nobody else publishes per-bucket calibration error with hash-chain
  proofs and coverage disclosure. That's not a tech advantage — it's
  category creation.

**What this is not**:
- Not HFT. Sub-minute latency is not in scope.
- Not capacity-deployable for ₹100Cr+ books without microstructure
  upgrades.
- Not a guarantee of trading-edge profitability. The product moat is
  the *honest research artifact*, not a promise that the alpha wins.

---

## §6 — Productization snapshot

Sellable products after the plan ships, in increasing order of
maturity / customer commitment.

**Tier 1 — low-friction, high-volume**:
1. **Newsletter** (free, distribution) — already in Stream H.
2. **Daily research brief — equity intraday** — exists.
3. **Daily research brief — index** — Stream D in flight.
4. **Daily research brief — options** — Stream D Gate 3.
5. **Swing brief** — Stream F + this plan's M.10 confidence layer.

**Tier 2 — paid datasets** (each one a sellable feed):
6. **Slippage dataset** (T4.1) — direct output of Stream K's data
   generation. Sellable by Stream K Gate 1.
7. **Calibration / outcome dataset** (T4.3) — direct output of
   Stream J's hash-chained log.
8. **Pool / liquidity-zone dataset** (T4.2) — with 5-day delay so it
   doesn't erode the brief edge.
9. **Detector event dataset** (T4.4) — same delay pattern.
10. **Regime atlas dataset** (new) — labeled regime windows with
    transition probabilities. Direct output of Stream O's regime
    classifier.
11. **Reaction archetype atlas** (new) — direct output of M.7.

**Tier 3 — high-margin services**:
12. **Strategy Diagnosis service** (T3.1) — components exist; gets
    teeth from Streams I, J, N. The SEBI-safest scalable product.
13. **Brief-as-API** (new) — B2B endpoint for desks that want to run
    the publish pipeline on their own universe.
14. **Calibrated probabilities API** (new) — "give me a strike, I
    return P(touch) with conformal bands." Direct output of
    Streams N and M.10.

**Tier 4 — aspirational, post-revenue**:
15. **Hedging overlay** (T3.3) — needs live Greeks + portfolio view.
16. **Failure mode taxonomy dataset** — once Strategy Diagnosis runs
    on enough customer strategies, the labeled catalog of "how
    strategies fail" is irreplaceable.

**Realistic ship-able portfolio in the first year**:
- 1 free distribution channel (newsletter)
- 3-4 daily briefs (equity, index, options, swing)
- 1 paid diagnosis service
- 2-3 paid datasets (slippage, calibration, regime atlas)
- 1 API (calibrated probabilities)

That's **6-8 distinct products** with a coherent moat story: every one
of them is a window into the same audited engine.

---

## §7 — Production-readiness assessment

After full ship, the engine sits at:

**Reliability**:
- Atomic parquet writes ✓
- Hash-chained outcome log ✓
- Drift monitoring with learned precursor ✓
- Leakage CI tests ✓
- Boundary-invariant CI tests ✓ (new)
- Reconciliation tests ✓
→ SaaS-grade reliability for a small operation.

**Observability**:
- Outcome log + arsenal evaluator + Phase 3A audit
- Shadow log (new) covering declined decisions
- Drift dashboard with learned precursors
→ Better than most institutional research stacks. Genuinely world-class.

**Scale**:
- Today: 1 founder, 0-3 pilot customers.
- After plan: 50-200 customers without re-architecting.
- 200-2000 customers: needs ops layer (queue, worker pool, multi-
  tenant outcome log). Out of scope.
- HFT scale: out of scope by design.

**Latency**:
- Brief generation: offline batch, sub-five-minute target.
- API endpoints (calibrated probabilities): single-digit ms.
- Live desk intraday updates: would need 30-60s path; doable but not
  in this plan.

**Compliance posture**:
- Non-advisory research artifact ✓
- Disclosures in every brief ✓
- Hash-chained audit verifiable externally ✓
- Strategy Diagnosis is research, not advice ✓
- No tip channels, no signal lists, no follow-up retargeting ✓
→ SEBI-safe under current Indian regulations.

---

## §8 — Living document maintenance

This document is updated when:
- A stream's acceptance gates change.
- A stream completes (mark it ✓, link the commit).
- A new finding (audit or flywheel idea) doesn't fit any existing
  stream → open a new stream letter.
- The dependency graph in §3 changes.

It is **not** updated for:
- Individual commit progress (that's in `MASTER_PLAN.md` and
  `COORDINATION.md`).
- Strategic backlog items that don't have an active stream
  (those go in `RESEARCH_ROADMAP.md`).

---

*End of plan.*
