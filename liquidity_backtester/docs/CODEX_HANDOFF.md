# Codex Handoff — State of the Project after Opus Iteration

**Branch:** `claude/liquidity-pool-backtester-1uskb`
**Latest commit:** `ac0a9cc Arsenal grows itself: model-wrapped alphas + meta combinators`
**Date:** 2026-05-31
**Test suite:** 219 passing, 0 failing, enforced compliance lint clean.

---

## 0. READ THIS FIRST — One-screen TL;DR

**Role split:** Opus/Claude is the **brain + debugger + designer**. Codex is the
**main implementer + runtime worker** (long sweeps, training runs, big
refactors). This document is Opus's full state dump for Codex to act on.

**Where the project is structurally:**
1. The execution simulator had TWO silent bugs (R-explosion on gap, no MIS
   enforcement). **Both are now fixed in `liqpool/execution_backtest.py`** — but
   V2 simulator still has no MIS, and pre-Opus measurements were taken on the
   buggy V1, so every R number from before commit `ecb08c9` needs re-baselining.
2. The strategy thesis from the prior chat — **journey-to-liquidity**, Track A
   pre-touch directional — is still the most credible alpha hypothesis. The
   Track A constrained pocket (long AUTO/FMCG/PHARMA, 3-8 ATR, P_touch≥0.75,
   Direction≥0.65) was last measured at **+0.365R / PF 1.91** under V2
   simulator, **CPCV_COMPONENT_MARGINAL**. That measurement is still valid;
   V2 was MIS-honest already.
3. **New infrastructure shipped by Opus:** SaaS feed, R1 force regressor on net R,
   triple-barrier labels + 48-cell grid sweep, alpha arsenal framework with 3
   base + 3 model-wrapped alphas + 3 meta-combinators + null tests. This makes
   the arsenal **mechanically self-growing** along two axes: threshold sweeps
   and combination algebra. See §5.
4. **Codex's "next best engineering step"** from `reports/phase4_status_for_opus.md`
   was the **synthetic-null suite** (random pools, ATR-offset pools, sector-
   neutral, full-V2 shuffled-direction). **Opus did NOT do this.** Opus jumped
   ahead to a final-stage model (R3 force model). The synthetic-null suite is
   still outstanding and is the most defensible next deliverable. See §6 and §10.

**Next concrete moves for Codex** are in §10, priority-ordered, actionable.

---

## 1. Roles and how we work together

| | Opus (Claude) | Codex |
|---|---|---|
| **Primary mode** | Diagnose, plan, design, write code that proves a concept, write tests, write docs. | Execute long-running training runs, multi-day refactors across many files, large sweeps, data pipelines. |
| **What I do well** | Spot bugs by reading code; design clean abstractions; small focused changes with comprehensive tests; honest critique of plans; statistical reasoning (CI95 / nulls / DSR). | Long iteration loops; consistent style across many files; reliable on multi-step refactors that don't need re-design. |
| **What I avoid** | Multi-hour training runs (waste my context); large rote refactors; building things on top of unvalidated foundations. | Open-ended design decisions; deciding what to prove; choosing between competing strategic directions. |
| **Hand-off contract** | I push commits + write this doc + leave a clear "next task" list. | Codex picks up from §10, completes one task end-to-end, opens PR, hands back. |

**One important contract:** if Codex hits a structural decision (which alpha to
implement next, whether to extend or replace a model, etc.), **stop and ask
Opus.** Don't make architectural decisions silently — that's where the project
went off-rails before (e.g. the gap-explosion bug went unnoticed because no one
audited the simulator at the same time as adding new modes on top of it).

---

## 2. Project state — current truth, not aspirational

### 2.1 What we know is solid

- **Pool detection is causal.** Pre-Opus replay audit
  (`reports/phase4_pool_availability_replay_audit.md`) passed: 237,364 pool rows,
  zero "used before known" violations.
- **Proximity distance feature has no leakage.** 7.67M rows audited, zero
  mismatches (`reports/phase4_proximity_distance_leakage_audit.md`).
- **MIS enforcement is now correct in V1.** Opus's commit `ecb08c9` added
  `EOD_SQUAREOFF_IST_MIN=15:15`, `NO_NEW_ENTRY_AFTER_IST_MIN=14:30`, and
  same-day bar iteration. See §4.2 for the bug, §5.7 for verification.
- **The R-explosion bug is fixed.** V1 confirmation modes used to silently
  produce net_R in the millions when the next bar's open gapped past the
  protective stop. Fix in `2e89e77`. See §4.1.
- **Trained models work mechanically.** Quality (`unified_ml`), direction,
  proximity, reaction, policy outcome (Phase 3D), policy return (R1) — all
  saved in the bundle and queryable. They DO discriminate (AUCs documented
  per model). What's questionable is whether their discrimination translates
  to tradeable edge — see §2.2.

### 2.2 What we know is broken or marginal

- **High model AUCs do NOT translate to net positive R in the kitchen sink.**
  Track B (post-touch reaction / touch_confirmed / displacement_confirmed) is
  net-negative on every pocket Opus tested (`pocket_sensitivity.py` verdict).
  The cause is path-dependence: models predict endpoints, trades depend on
  paths.
- **Track A constrained pocket is positive but marginal.** +0.365R, PF 1.91
  on 434 trades — survives the first CPCV/component-null layer but the
  shuffled-direction null still makes +0.222R (Codex's measurement from
  `reports/phase4_track_a_cpcv_nulls.md`). So proximity + geometry + sector +
  time-of-day are doing most of the work; direction adds lift but isn't the
  edge.
- **Proximity AUC = 0.97 is suspiciously high.** The handoff and Opus both
  flag this. Sampled replay passed but full replay hasn't been done. If there
  IS residual leakage, the Track A pocket's edge is even smaller than the
  marginal CPCV verdict suggests.
- **Q model AUC ~0.53.** Barely above 0.5. The "Q≥70% live gate" is
  mathematically unreachable (Q range is 23-56% per
  `reports/phase4_workstream0_q_audit.md`). Q model is essentially noise as a
  ranker. Codex documented this; Opus did NOT recalibrate it.
- **Reaction confirmations display ~98% by construction.** Top-10 sort on
  reaction probability + isotonic clipping at 0.98 made it look like the
  model was saturated. Opus's audit (`analysis/audit_reaction_distribution.py`)
  proved this was a display artifact, not model saturation. Fixed in the
  predict path display, but the model itself is fine.

### 2.3 What's brand new and not yet tested on real data

- **R1 PolicyReturnModel** (Huber regression on net R per generated trade).
  Code passes its tests; only run on synthetic data and the saved bundle.
- **R3 triple-barrier labels + force-model sweep** (48 configs). Code passes
  tests; needs cloud run to produce real R3 verdict.
- **Alpha arsenal framework + 3 base alphas + 3 model-wrapped alphas + 3
  meta-combinators.** Code passes 80 tests; never run end-to-end on the real
  bundle.
- **Pocket sensitivity verdicts.** Opus's `pocket_sensitivity.py` (Track B)
  and `pretouch_pocket_sensitivity.py` (Track A) are both no-retrain
  analyses that the user ran on their Mac. Track B confirmed dead on every
  pocket; Track A wasn't run to completion (laptop died mid-output).

---

## 3. Opus's answers to Codex's "Questions for Opus"

From `reports/phase4_status_for_opus.md` §"Questions for Opus":

**Q1: Given shuffled-direction median is still +0.222R while actual is +0.365R,
should direction remain a hard gate or become a soft feature in a final-stage
model?**

**Opus:** Soft feature in a final-stage model. The fact that shuffled direction
still produces +0.222R means most of the edge is in (proximity × distance × sector
× time-of-day), not direction. A hard direction gate throws away setups where
proximity is excellent even when direction is mediocre. The R1 PolicyReturnModel
(`liqpool/policy_model.py:PolicyReturnModel`) is the scaffolding for this: it
trains a regressor on `policy_target_return_r` with direction-related features
as inputs but not gates. Once R1 is trained on real data, the
DirectionConfirmedPoolAlpha thresholds become a sweep parameter, not a hard
rule.

**Q2: Is excluding BANKING/IT too post-hoc, given all-sector long still performs
at +0.321R?**

**Opus:** Yes, exclusion is too post-hoc. The +0.044R lift from excluding
BANKING/IT (+0.365 vs +0.321) is well within the noise band of the CPCV worst
path (-0.251R). Treating it as a hard production rule fails the falsifiability
test — if BANKING/IT regimes shift, the rule will silently underperform. **Make
sector a feature in the final-stage model, not an exclusion list.** The arsenal
framework already adds `sector` as a regime-stratification dimension; the
PolicyReturnAlpha already gets sector as a model input. The exclusion of
BANKING/IT should be re-examined after the next CPCV pass on a final-stage
model.

**Q3: Should the next null suite simulate random/ATR pools through the full V2
pre-touch simulator, or can we derive a valid faster approximation from
existing candidate rows?**

**Opus:** Full V2 simulator. The approximation will leak the cost model's
asymmetry (which Opus already proved bites Track A by ~0.30R at Rs50k notional
in `analysis/cost_sensitivity.py`). The arsenal's `null_tests.py` already
implements time-shuffle and sign-flip nulls but does NOT yet implement the four
synthetic nulls Codex's plan requires (random pools at matched spatial density,
ATR-offset pools at k=1,2,3,5, sector-neutral random pools, full-V2 shuffled
direction). **Adding the four synthetic nulls is task R-NULLS in §10.**

**Q4: Should triple-barrier labels define "journey completed before adverse
move" rather than "pool respected after touch"?**

**Opus:** Yes, and Opus already shipped this as R3. The label in
`liqpool/triple_barrier.py:triple_barrier_label` is realized R from entry to
first barrier hit (target, stop, or time/EOD). It's parametrized by
`(stop_atr, target_atr, horizon_bars)` so the "journey vs respect" definition
becomes a configuration, not a hard-coded label. The 48-cell grid sweep in
`analysis/force_model_sweep.py` tests this: 4 stops × 4 targets × 3 horizons.
Codex's "journey completed" can be approximated by `target_atr=4.0,
stop_atr=1.0, horizon_bars=24` (long horizon, tight stop) — that combination
is in the grid. Once Codex runs the sweep on real data, the verdict on
"journey vs respect" is the per-config table.

**Q5: Should Opus prioritize final-stage model before or after synthetic
random-pool/ATR nulls, given this first CPCV layer is marginal but positive?**

**Opus:** **Codex's preferred ordering was nulls first; Opus did final-stage
first.** This is an honest disagreement. The reasoning:
- Codex's argument (nulls first): if the alpha doesn't survive synthetic nulls,
  the final-stage model is wasted work.
- Opus's argument (final-stage first): the final-stage model is small and
  produces statistical artifacts (top-decile R per config with CI95) that the
  null tests can then validate. Without a final-stage model the nulls only
  validate the existing hard-rule pocket, which we already know is marginal.

In hindsight, both should run. The arsenal framework supports running synthetic
nulls on ANY alpha including the wrapped policy-return model. **Codex should
implement the synthetic-null suite next (§10 task R-NULLS) and run it against
both (a) the existing Track A pocket and (b) the R1+R3 trained models.**

---

## 4. Bugs Opus found and fixed

### 4.1 R-explosion: V1 simulator produced net_R in the MILLIONS on gap-past-stop

**Symptom:** User reported `net_R = +40,673,861.30` on `displacement_confirmed`
mode in a clean train run, with up-leg and down-leg signs flipped.

**Root cause:** In `liqpool/execution_backtest.py:simulate_pool_trade`, the
confirmation modes (touch / reclaim / displacement) reassign entry to the NEXT
bar's open. When that bar gapped past the protective stop (gap-down for long,
gap-up for short), the natural risk per share went NEGATIVE:

```python
risk_per_share = max(float(entry_price - stop), 1e-9)  # long
```

`max(negative, 1e-9)` returns `1e-9`. A small rupee loss divided by near-zero
risk produced absurd net_R. blind_limit mode was immune because it never
reassigns entry.

**Fix (commit `2e89e77`):** Two-part defense:
1. If `natural_risk_per_share <= 0`, return None — the trade would have been
   stopped out at the open. It's a slippage event, not a real trade.
2. Floor `risk_per_share` at `0.10 × ATR` per share (minimum-risk guard).

V2 simulator already had equivalent logic
(`test_v2_skips_confirmed_entry_when_next_open_invalidates_geometry`). The fix
brings V1 in line.

**Tests:** `tests/test_execution_backtest_risk.py` — 4 cases pinning the
invariant.

### 4.2 No MIS enforcement: V1 simulator iterated bars across overnight gaps

**Symptom:** Every R number computed by V1 silently treated overnight bars as
intraday continuation. A trade entered Friday 14:00 IST could "exit" Monday
11:00 IST without any model of the weekend.

**Root cause:** `_exit_trade()` iterated bar indices with no day/session
boundary check. `cfg.respect_within_bars=40` was 40 bar *indices*, not 40
same-day bars. `cfg.test_horizon_bars=200` was ~2.7 calendar days.
`liqpool/regime.py:nse_session()` existed and was used as a feature, but never
as a gate.

**Fix (commit `ecb08c9`):**
- New constants: `SESSION_OPEN_IST_MIN=09:15`, `SESSION_CLOSE_IST_MIN=15:30`,
  `EOD_SQUAREOFF_IST_MIN=15:15`, `NO_NEW_ENTRY_AFTER_IST_MIN=14:30`.
- New helpers: `_ist_minute_of_day`, `_ist_date`, `_last_intraday_bar_idx`.
- Convention: tz-naive timestamps treated as UTC (matches
  `liqpool.data._normalise_parquet_ohlcv` which strips tz at lines 37-38,
  74-75). Same convention as `nse_session`.
- `simulate_pool_trade` now refuses entries outside session or after 14:30 IST,
  and caps the time-stop at the same-day EOD bar.

**Effect on R numbers (measured at user retrain after fix):**
- gross_R lift across modes: blind_limit +0.09, displacement +0.05,
  reclaim +0.07, touch +0.07.
- Trade count dropped ~15-17% per mode (late-session entries refused).
- `displacement_confirmed` reached `gross_R = +0.00` (essentially break-even
  on edge before costs).

**Tests:** `tests/test_execution_backtest_risk.py` (3 MIS tests added).

**V2 still has no MIS gate.** It has `session_phase()` for slippage tagging
only. **Task R-MIS-V2 in §10 is to apply the same fix to V2.**

### 4.3 Reaction "every confirmation 98%" was a display artifact, not saturation

**Symptom:** User saw every confirmation in predict output show
`strict=98% reclaim=98% break=2%`. Plausible interpretation: model is
saturated and useless.

**Root cause:** In `examples/multi_asset_run.py:~1167`, the reaction
confirmations section did `reaction_alerts[:10]` AFTER sorting descending by
reaction probability. By construction those 10 are the highest-confidence
subset. The reaction model is properly calibrated (per-target isotonic on
disjoint holdout in `liqpool/reaction_model.py:241-259`, clipped [0.02, 0.98]).

**Validation:** Opus built `analysis/audit_reaction_distribution.py` to
histogram the full OOS distribution. User ran it on Mac:

| target | median | p75 | %≥0.85 | OOS AUC |
|---|---|---|---|---|
| strict_reaction | 0.44 | 0.67 | 10.2% | 0.770 |
| reclaim_success | 0.39 | 0.45 | 6.3% | 0.742 |
| break_continuation | 0.46 | 0.62 | 13.4% | 0.762 |

Verdict: `DISPLAY ARTIFACT`. Model is genuinely calibrated and discriminating.

**Fix (commit `a484e8d`):** Display block in `examples/multi_asset_run.py`
now shows the distribution summary above the top-10 list and labels the
truncation explicitly ("highest-confidence subset"). No model change.

**Tests:** `tests/test_audit_reaction_distribution.py`.

### 4.4 StateFeaturizer shape-mismatch on small bar indices

**Symptom:** When the DirectionConfirmedPoolAlpha tried to query the direction
model at touch bars near the start of the session window, `features_at(j, ...)`
raised `ValueError: shapes (j+1,) (j,) could not be broadcast`.

**Root cause:** `liqpool/timing.py:108`:
```python
oa = (c[i0:j + 1] - c[max(0, i0 - 1):j])[:j - i0 + 1]
```
When `i0 = max(0, j-11) = 0` (early bars), `c[i0:j+1]` is `j+1` elements but
`c[max(0, i0-1):j]` is `j` elements. Subtracting fails.

The bug didn't manifest in production because StateFeaturizer was only called
via timing.py snapshot building with `j` well above 12.

**Fix (commit `ac0a9cc`):** Replaced with `np.diff`:
```python
slice_start = max(0, j - 11)
moves = np.diff(c[slice_start:j + 1])
mom_12 = float(np.sum(moves) / a) if len(moves) else 0.0
```

### 4.5 Length-mismatch in reaction-distribution audit

**Symptom:** Audit script crashed with
`Found input variables with inconsistent numbers of samples: [21064, 22241]`
when user ran it on Mac.

**Root cause:** `_summarise_row` computed AUC by pairing
`y_labelled` (21,064 labelled rows) with `p_all` (22,241 predictions including
unlabelled).

**Fix (commit `3d61b8c`):** `_predict_target` returns 3-tuple
`(p_all, p_labelled, y_labelled)` with the labelled pair guaranteed
equal-length; `_summarise_row` uses each for its proper purpose.

---

## 5. Features and infrastructure Opus shipped

### 5.1 SaaS feed engine + service (commits `ec2b835`, `ded0147`, `bc50e4a`)

- `liqpool/scoring.py`: opaque-feature scoring (`feature_intensity_score` 0-100
  rank, `feature_state` 3-class non-directional label, `level_zone` geometry).
  Per-customer watermarking via deterministic jitter keyed on
  `(customer_id, day)`.
- `liqpool/ingest.py`: reads raw predict CSV/JSON, scores per row.
- `liqpool/serving.py`: handler that builds the customer envelope including
  feed-level `compliance_notice`.
- `service/app.py`: FastAPI server with API-key auth + rate limiting.
- `scripts/export_saas_feed.py`: offline CSV/JSON feed generator (per-customer).
- `scripts/synthetic_smoke.py`: end-to-end smoke test that doesn't need real
  model data.
- `docs/data_product_spec.md`, `docs/legal/compliance.md`,
  `docs/sebi_compliance_notes.md`: customer-facing + internal legal docs.

The schema was neutralized after a compliance red-team (`bc50e4a`): no `G`,
no `D`, no `+/-/=` directional glyphs; opaque integer + non-directional state
label; per-observation `level_usage_note`; full disclaimer in feed-level
`compliance_notice` (not repeated per record).

**Note for Codex:** the SaaS export path is **frozen** during all alpha-side
work. Any change to public field names breaks customer contracts. If the
arsenal verdict changes the underlying signal, **the encoding stays the same**;
only the numeric scores change.

### 5.2 Compliance linter (commit `bc50e4a`)

`scripts/compliance_lint.py`: scans for hard-banned words (buy/sell/hold/
signal/recommendation/etc.) in customer-facing files. Internal code is allowed
to use these terms when they're inevitable (e.g. `side_label = "buy"` for a
trained model's required input). The linter has both **hard** (non-customer-
facing okay) and **enforced** (blocks CI) tiers. `enforced_failures=0` is the
gate.

### 5.3 End-to-end RUNBOOK (commit `a91e3b0` + updates)

`RUNBOOK.md`: setup → data → train → predict → SaaS feed → distribute, in one
document. Mac-safe (python3, explicit `.venv/bin/python` paths so no `source`
needed). Documents the real Google Drive data path. Documents `--asset-workers`
default.

### 5.4 Pillar 1: reaction-distribution audit + display fix + per-distance calibration (commit `a484e8d`)

- `analysis/audit_reaction_distribution.py`: histograms reaction model output
  across full OOS to detect saturation. Used to refute the "98% saturation"
  hypothesis.
- `examples/multi_asset_run.py` display fix: confirmations section now prints
  the distribution summary, labels the top-10 list explicitly.
- `liqpool/distance_calibration.py`: `DistanceCalibrator` class that fits one
  isotonic regressor per distance bucket (0-1 / 1-3 / 3-5 / 5-10 / 10+ ATR).
  Sparse buckets fall back to identity so the layer can never make calibration
  worse. Wired into `ProximityModel` (`liqpool/timing.py`) and into
  `PoolRespectModel` / `SectorMoERespectModel` (`liqpool/ml_model.py`) as a
  ready hook — but **NOT YET wired into the joint Q × proximity eval site** in
  `walkforward.py`. That last wiring is task `R-DISTCAL-WIRE` in §10.

### 5.5 R1: PolicyReturnModel (commit `b0643df`)

`liqpool/policy_model.py:PolicyReturnModel`,
`PolicyReturnModelSuite`: Huber regression on realized net R per generated
trade, winsorized at [-2.5, +5.0]. Per execution mode. Trained alongside the
existing binary `PolicyOutcomeModel` (untouched). New attrs on the report:
`policy_return_model`, `policy_return_model_report`,
`policy_return_model_calibration`, `policy_return_model_feature_importance`.
Backward-compat default in `_load_predict_report`.

CLI flag: `--r-policy-threshold` (default None = report only). Plumbed but the
predict-path gate is intentionally not wired yet — R1 first has to prove edge
on the kitchen sink before changing live behaviour. **Task R-R1-VERIFY in §10
is to produce the verdict from real data.**

User's first verification run on Mac: `displacement_confirmed` reached
`top10_R = -0.34R`, Spearman +0.091. After bug 4.1 fix and MIS enforcement,
re-ran: `top10_R = -0.34R` again, Spearman +0.091 (consistent). Verdict: R1
ranking is honest but kitchen-sink R is dead. The interesting test is the
pocket-filtered R1 verdict — not yet run.

### 5.6 Cost-sensitivity analysis (commit `de72842`)

`analysis/cost_sensitivity.py`: no-retrain test. Loads bundle, re-runs the
execution backtest at qty=1 + a list of rupee notionals (default
50k/100k/200k). Reports per mode: `gross_R` (cost-free ceiling), `net_R`,
`cost_R` (drag), `PF`, `avg_cost_inr`. The decisive number is `gross_R`: if
≤0, no amount of sizing can produce winning trades.

User's run on Mac (post-MIS fix): every mode's `gross_R` is between -0.06 and
-0.35. **The strategy has no positive-gross-R region in the kitchen sink under
honest MIS arithmetic.** This was the verdict that triggered the pocket
analysis next.

### 5.7 Pocket-sensitivity analyses (commits `868a912`, `e153012`)

- `analysis/pocket_sensitivity.py` (Track B post-touch): filters OOS pools by
  session × sector × side × factor, re-runs `simulate_execution_modes` with
  MIS active, reports per-pocket CI95. User's run: **no pocket clears net_R > 0
  with CI lower bound > 0**. Every pocket dead even at ₹200k notional.
  Conclusion: Track B post-touch is structurally dead.
- `analysis/pretouch_pocket_sensitivity.py` (Track A pre-touch): consumes the
  existing Track A sweep parquet (`output_phase4_track_a_pretouch_sweep/
  pretouch_sweep_trades.parquet`) and reports CI95 per pocket. Not yet run on
  real data — user's laptop died mid-output last time. **Task R-TRACKA-VERIFY
  in §10.**

### 5.8 R3: Triple-barrier labels + force-model sweep (commit `5e7ece4`)

- `liqpool/triple_barrier.py:triple_barrier_label`: pure function that returns
  realized R from entry to first barrier hit. MIS-aware via
  `_last_intraday_bar_idx`. Exit reasons: target / stop / time_exit /
  eod_squareoff / no_room.
- `analysis/force_model_sweep.py`: two-part analysis.
  - **Part 1** static grid sweep: per-config mean R / win / PF / CI95 across
    every touched OOS pool × every (stop, target, horizon) config in
    {0.3,0.5,0.75,1.0} × {1.0,1.5,2.0,3.0} × {12,24,78}.
  - **Part 2** force model: LightGBM Huber regressor on
    (features + stop_atr + target_atr + horizon_bars) → realized R.
    Chronological 70/30 train/val split. For each config, slice val
    predictions, take top decile, report realized R with CI95.
  - Verdict cross-comparison: "ALPHA EXISTS but requires selective trading"
    vs "Both fixed and learned positive" vs "Neither finds positive R."
- Feature modes: basic (geometry + state) and `--use-extended-features` adds
  volume_ratio_5b, volume_zscore_20b, gap_atr, range_compression,
  time_quartile, vol_ratio_regime.

**Not yet run on real data.** Task R-R3-VERIFY in §10.

### 5.9 Alpha arsenal framework (commits `fcc9cf9`, `ac0a9cc`)

```
liqpool/arsenal/
├── base.py              Alpha (ABC), AlphaSignal (validated dataclass)
├── registry.py          AlphaRegistry, default_registry()
├── evaluator.py         ArsenalEvaluator (shared MIS+cost+barrier pipeline)
├── null_tests.py        time_shuffle_null, sign_flip_null, NullResult
├── meta.py              MetaANDAlpha, MetaORAlpha, MetaWeightedAlpha
└── alphas/
    ├── pool_reach.py        LiquidityPoolReachAlpha (existing strategy wrapped)
    ├── mean_reversion.py    MeanReversionAlpha (z-score fade, pool-independent)
    ├── momentum.py          MomentumAlpha (multi-bar trend, pool-independent)
    └── model_filtered.py    QualityFilteredPoolAlpha, DirectionConfirmedPoolAlpha,
                             PolicyReturnAlpha (existing trained models as alphas)
analysis/run_arsenal.py     CLI runner with per-alpha + regime + pairwise + null verdict
```

The contract every alpha satisfies:
```python
class MyAlpha(Alpha):
    name = "..."
    def candidates(self, *, symbol, df_base, atr_series, extra=None)
        -> List[AlphaSignal]: ...
    def regime_tags(self, signal, df_base, atr_series) -> Dict[str, str]: ...
```

The evaluator applies, for every signal from every alpha:
1. MIS check (entry past 14:30 IST → drop).
2. Enter on next bar's open (no look-ahead).
3. Compute realized R via `triple_barrier_label` (MIS-aware).
4. Apply Zerodha intraday costs + flat bps slippage.
5. Record regime tags from `alpha.regime_tags()`.

Aggregations: `per_alpha_summary` (CI95), `per_regime_summary` (any tag column),
`pairwise_combinations` (signals firing at same (symbol, decision_idx)).

Null tests: time-shuffle (randomize decision_idx within each asset's range,
re-execute, compare null mean R to actual; one-sided p-value) and sign-flip
(flip a fraction of signals' sides; test whether direction is the edge).

Meta combinators:
- `MetaANDAlpha([alpha_a, alpha_b])`: emits only when BOTH fire at same
  (symbol, decision_idx) with same side. Conservative geometry merge.
- `MetaORAlpha([alpha_a, alpha_b])`: emits union, dedup to highest-confidence.
- `MetaWeightedAlpha([alpha_a, alpha_b], weights, threshold)`: soft voting.

**Self-growth mechanism for the arsenal:**
1. **Layer 1 (threshold sweep)**: register the same wrapped alpha N times with
   different thresholds (e.g. `QualityFilteredPoolAlpha(min_q=q)` for
   `q in (0.45, 0.50, 0.55, 0.60, 0.65)`). The evaluator treats each as
   separate, reports CI95 per threshold.
2. **Layer 2 (combination algebra)**: with N base alphas, mechanically
   construct C(N,2) AND-pairs and C(N,3) AND-triples. Each is itself an Alpha;
   the framework handles it.
3. **Layer 3 (auto-discovery of new alpha LOGIC)**: not yet built. Dangerous
   without strict multiple-comparisons controls. Defer until 1+2 exhausted.

---

## 6. Codex's roadmap — what's validated and what's invalidated

### 6.1 Validated by Opus's work

| Codex plan item | Opus validation |
|---|---|
| "Use Q percentiles, not Q≥70%" | Confirmed. Q range is 23-56%, gate at 70% is unreachable. Used pocket sensitivity instead. |
| "V2 simulator is stricter and more honest than V1" | Confirmed. But Opus discovered V1 was ALSO double-broken (R-explosion + no MIS). Both V1 bugs are now fixed; V2 still has no MIS. |
| "Post-touch execution modes are negative" | Confirmed under honest MIS arithmetic. No pocket clears net_R > 0. |
| "Track A pre-touch is the active path" | Validated. Built pretouch_pocket_sensitivity to verify with CI95. |
| "Triple-barrier as next labelling step" | Built (R3 in `liqpool/triple_barrier.py`). |
| "Direction as soft feature, not hard gate" | Built into R1 / wrapped alphas with thresholds as sweep params. |

### 6.2 Invalidated or revised by Opus

| Codex plan item | Opus revision |
|---|---|
| "Q ≥ 70% live gate" | Mathematically impossible; replaced with pocket-filtered + force-model-filtered selection. |
| "Reaction model is saturated (every confirmation 98%)" | False — it's a display artifact. Model is properly calibrated. Display fixed; model untouched. |
| "Track B is dead under V2 simulator" | True but the V1 numbers Codex showed were ALSO corrupted by the R-explosion + no-MIS bugs. New baseline measurements needed across all execution analyses run before commit `ecb08c9`. |
| "Run synthetic-null suite FIRST, then final-stage model" | Opus reversed the order. Final-stage (R1, R3, arsenal) built first. **Synthetic-null suite is now the most defensible outstanding task** — see §10 R-NULLS. |

### 6.3 Still valid and unaddressed (Opus did not work on these)

| Codex plan item | Why Opus skipped | Priority |
|---|---|---|
| Full CPCV on Track A under post-MIS-fix measurement | User couldn't retrain on Mac, deferred to cloud. | **HIGH** (Cloud now ready.) |
| 4-test synthetic-null suite (random / ATR-offset / sector-neutral / full-V2-shuffled-direction) | Opus jumped to R1/R3/arsenal instead. | **HIGH** |
| Q model recalibration | Codex documented; Opus added distance-calibration scaffolding but didn't wire into Q joint-eval site. | MEDIUM |
| Proximity AUC 0.97 full replay audit (deep leakage check) | Sampled replay passes; full replay never done. | MEDIUM-HIGH |
| MTF closed-bar daily/weekly WARN cleanup | Pre-Opus; never touched. | LOW (warnings, not errors) |
| Options-data foundation (mentioned as pivot if Track A fails nulls) | Track A hasn't been re-verified post-MIS yet. | DEFER until null verdict |
| Final-stage journey-trade model with continuous labels | Opus's R1 + R3 partially address this; needs real-data verification. | HIGH (verify R3 then iterate) |

---

## 7. Architecture map — every file, every connection

```
DATA WAREHOUSE (Google Drive: kite_indian_market_data/resampled/)
    │
    ▼
liqpool/data.py:ParquetProvider
    - load_base(): 5-min OHLCV per symbol; tz-stripped to naive UTC
    │
    ▼
liqpool/pools.py:build_pools(tf, best)
    - Multi-TF detection: OB / FVG / EQHL / REJ / ORB / SWING / PD / PW / PM
    - pool.available_at = max(contributor.known_at)
    │
    ├─ liqpool/featurize.py:Featurizer.transform_batch(pools)
    │     - Pool-level features: score, tf_count, width_atr, factor_*, age_bars, etc.
    │     - All causal at pool.available_at
    │
    ├─ liqpool/timing.py
    │     - StateFeaturizer: per-bar state features (ret_*, vol_ratio, etc.)
    │     - DirectionModel: P(up) over horizon H, LightGBM + isotonic
    │     - ProximityModel(h): P(touch within h bars), LightGBM + isotonic
    │     - Now wired with optional DistanceCalibrator (Opus added; not yet
    │       invoked by walkforward)
    │
    ├─ liqpool/ml_model.py
    │     - PoolRespectModel: per-pool P(respect), LightGBM + isotonic
    │     - BucketCalib (existing): per-(TF, factor) recalibration
    │     - DistanceCalibrator hook (Opus added; not yet wired into joint eval)
    │     - SectorMoERespectModel: global + sector experts with dynamic shrinkage
    │
    ├─ liqpool/reaction_model.py
    │     - ReactionBinaryModel × 3 targets (strict / reclaim / break)
    │     - Per-target isotonic on disjoint holdout (correct calibration)
    │     - Output clipped [0.02, 0.98]
    │
    └─ liqpool/policy_model.py
          - PolicyOutcomeModel (Phase 3D): binary classifier per execution mode
          - PolicyReturnModel (R1, Opus added): Huber regression per mode
          - PolicyReturnModelSuite: one regressor per mode + aggregate

EXECUTION
    │
    ├─ liqpool/execution_backtest.py (V1)
    │     - simulate_pool_trade: now MIS-aware (Opus fix ecb08c9)
    │     - 4 modes: blind_limit / touch_confirmed / reclaim_confirmed /
    │       displacement_confirmed
    │     - _exit_trade: bar-by-bar, capped at same-day EOD (15:15 IST)
    │     - R-explosion bug fixed (Opus fix 2e89e77)
    │
    ├─ liqpool/execution_simulator_v2.py
    │     - simulate_pool_trade_v2: 1m intra-bar resolution
    │     - State-dependent slippage (vol × distance × session phase)
    │     - PreTouchDirectionalSimulator: Track A pre-touch entries
    │     - !! V2 STILL HAS NO MIS GATE !! (task R-MIS-V2)
    │
    ├─ liqpool/triple_barrier.py (Opus added, R3)
    │     - triple_barrier_label: pure label function
    │     - Used by force_model_sweep and arsenal evaluator
    │
    └─ liqpool/costs.py
          - ZerodhaEquityCostConfig (brokerage cap Rs20, STT, exchange, etc.)
          - estimate_round_trip_charges: turnover-proportional

ALPHA ARSENAL (Opus added, fcc9cf9 + ac0a9cc)
    │
    ├─ liqpool/arsenal/base.py
    │     - AlphaSignal (frozen dataclass, validated)
    │     - Alpha (ABC): name, description, candidates(), regime_tags()
    │
    ├─ liqpool/arsenal/registry.py
    │     - AlphaRegistry: named lookup
    │     - default_registry(): 3 base alphas pre-registered
    │
    ├─ liqpool/arsenal/evaluator.py
    │     - ArsenalEvaluator: shared MIS+cost+barrier pipeline
    │     - per_alpha_summary, per_regime_summary, pairwise_combinations
    │
    ├─ liqpool/arsenal/null_tests.py
    │     - time_shuffle_null, sign_flip_null
    │     - !! synthetic null suite (random/ATR-offset/sector-neutral) NOT
    │        YET IMPLEMENTED !! (task R-NULLS)
    │
    ├─ liqpool/arsenal/meta.py
    │     - MetaANDAlpha, MetaORAlpha, MetaWeightedAlpha
    │
    └─ liqpool/arsenal/alphas/
          - pool_reach.py: LiquidityPoolReachAlpha
          - mean_reversion.py: MeanReversionAlpha
          - momentum.py: MomentumAlpha
          - model_filtered.py: QualityFilteredPoolAlpha,
                                DirectionConfirmedPoolAlpha, PolicyReturnAlpha

ANALYSIS SCRIPTS (Opus added unless noted)
    │
    ├─ analysis/audit_reaction_distribution.py (Pillar 1)
    ├─ analysis/cost_sensitivity.py (no-retrain cost analysis)
    ├─ analysis/pocket_sensitivity.py (Track B per-pocket CI95)
    ├─ analysis/pretouch_pocket_sensitivity.py (Track A verdict layer)
    ├─ analysis/force_model_sweep.py (R3, triple-barrier × 48 configs)
    └─ analysis/run_arsenal.py (arsenal CLI runner)
    [Pre-Opus: analysis/run_phase4_*.py — Q audit, multi-pocket slices, CPCV
     nulls, MTF audit, replay audit, pretouch sweep, focused diagnostics]

SAAS LAYER (Opus added, ec2b835/ded0147/bc50e4a)
    │
    ├─ liqpool/scoring.py (opaque feature_intensity_score + feature_state)
    ├─ liqpool/ingest.py (raw predict → scored)
    ├─ liqpool/serving.py (request handler)
    ├─ service/app.py (FastAPI server)
    ├─ scripts/export_saas_feed.py (offline CSV/JSON generator)
    ├─ scripts/synthetic_smoke.py (no-data smoke test)
    └─ scripts/compliance_lint.py (banned-word linter)

DOCUMENTATION
    │
    ├─ RUNBOOK.md (end-to-end, Mac-safe)
    ├─ docs/next_chat_handoff.md (3293 lines, pre-Opus + Opus annotations)
    ├─ docs/data_product_spec.md (customer-facing SaaS spec)
    ├─ docs/sebi_compliance_notes.md (internal legal)
    ├─ docs/legal/compliance.md (ToS basis)
    ├─ docs/compliance_audit_report.md (lint output snapshot)
    ├─ docs/CODEX_HANDOFF.md (this doc)
    └─ reports/phase4_*.md (15+ research reports, pre-Opus)
```

---

## 8. Test inventory (219 tests, 0 failing)

Pre-Opus test files (kept):
- `tests/test_execution_simulator_v2.py` (6 tests)
- `tests/leakage/test_fast_leakage_probes.py`
- `tests/test_validation.py`

Opus-added test files (40+):
| File | Tests | Coverage |
|---|---|---|
| `tests/test_arsenal_alphas.py` | 10 | pool_reach, mean_reversion, momentum |
| `tests/test_arsenal_base.py` | 10 | AlphaSignal validation, regime tags |
| `tests/test_arsenal_evaluator.py` | 10 | MIS execution, aggregations |
| `tests/test_arsenal_meta.py` | 12 | AND/OR/Weighted combinators |
| `tests/test_arsenal_model_filtered.py` | 16 | wrapped alphas with stub models |
| `tests/test_arsenal_null_tests.py` | 12 | time-shuffle, sign-flip helpers |
| `tests/test_arsenal_registry.py` | 10 | registry validation |
| `tests/test_audit_reaction_distribution.py` | 5 | length-mismatch regression |
| `tests/test_compliance_language.py` | varies | enforced banned-word gate |
| `tests/test_cost_sensitivity.py` | 5 | gross/net/cost R arithmetic, PF, CI |
| `tests/test_distance_calibration.py` | 8 | per-bucket isotonic, monotone, fallback |
| `tests/test_execution_backtest_risk.py` | 7 | R-explosion + MIS regression |
| `tests/test_force_model_sweep.py` | 15 | features, aggregation, training, verdict |
| `tests/test_pocket_sensitivity.py` | 14 | filter logic, summary, verdict |
| `tests/test_policy_return_model.py` | 10 | R1 fit/predict, winsor, splits, backcompat |
| `tests/test_pretouch_pocket_sensitivity.py` | 14 | Track A pocket filters + verdict |
| `tests/test_scoring.py` | 11 | opaque scoring + leak guard |
| `tests/test_service_api.py` | 3 | FastAPI endpoints |
| `tests/test_serving_ingest.py` | 4 | ingestion roundtrip |
| `tests/test_triple_barrier.py` | 11 | barrier arithmetic, gap, EOD, sides |

---

## 9. Next concrete tasks for Codex (priority-ordered, actionable)

Each task has: **Inputs**, **Outputs**, **Acceptance criteria**, **Estimated
effort**. Codex should pick from the top, complete end-to-end, push, then come
back for the next.

### TASK R-MEASURE: Re-baseline measurements on post-MIS-fix data **[HIGHEST]**

**Why:** Every R number measured before commit `ecb08c9` is corrupted by the
R-explosion + no-MIS bugs. We need a clean baseline before any further work.

**Inputs:**
- Saved bundle: `output_models/core25_latest/multi_asset_report.pkl`
- Track A sweep parquet (existing or re-generated)

**What to do:**
1. Run `analysis/cost_sensitivity.py` at qty=1 / Rs50k / Rs100k / Rs200k.
2. Run `analysis/pocket_sensitivity.py` (Track B) on the same bundle.
3. Run `analysis/pretouch_pocket_sensitivity.py` on the existing Track A
   sweep parquet (or regenerate via
   `analysis/run_phase4_track_a_pretouch_sweep.py` first).
4. Write `reports/post_mis_baseline.md` summarizing all three verdicts.

**Acceptance:**
- Verdict block from each script copied verbatim into report.
- One-paragraph human conclusion: "Track B status under MIS = X; Track A
  status under MIS = Y; next implied step = Z."

**Estimated effort:** 2-4 hours (mostly waiting for sweep regen if needed).

### TASK R-NULLS: Full synthetic-null suite (4 tests)

**Why:** Codex's original "next best engineering step" from
`reports/phase4_status_for_opus.md`. Opus skipped this to build R1/R3/arsenal;
now overdue.

**Inputs:**
- `liqpool/arsenal/null_tests.py` (has time-shuffle and sign-flip; needs
  three more null types).
- Saved bundle.
- Existing Track A sweep parquet.

**What to do:**
1. Add `liqpool/arsenal/null_tests.py:random_pool_null` — generate random
   pools at matched spatial density (same count per asset, random
   formed_at/touched_at), re-execute, compute mean R, repeat 100 trials,
   compute one-sided p-value.
2. Add `liqpool/arsenal/null_tests.py:atr_offset_pool_null` — shift each
   real pool's price band by k × ATR for k ∈ {1, 2, 3, 5}, re-execute,
   compute mean R, p-value per k.
3. Add `liqpool/arsenal/null_tests.py:sector_neutral_pool_null` — randomize
   pools' sector tags while preserving distribution.
4. Add `liqpool/arsenal/null_tests.py:full_v2_shuffled_direction_null` —
   same as sign-flip but specifically routed through V2 simulator (which
   has slippage realism the V1-based sign-flip lacks).
5. Add corresponding tests in `tests/test_arsenal_null_tests.py`.
6. Wire all four into `analysis/run_arsenal.py --null-suite full`.
7. Write `reports/synthetic_null_verdict.md` running all four against:
   (a) LiquidityPoolReachAlpha baseline, (b) the Track A constrained pocket
   alpha, (c) the meta-AND of (Quality × Direction × PolicyReturn).

**Acceptance:**
- Each null returns a `NullResult` with finite p-value.
- For each tested alpha: p-value table across all 4 nulls + 2 existing.
- Human verdict: "Track A under all 6 nulls = X."
- All new tests pass; full suite still passes.

**Estimated effort:** 1-2 days.

### TASK R-R3-VERIFY: Run force-model sweep and report verdict

**Why:** R3 is built but never run on real data. The verdict ("ALPHA EXISTS but
requires selective trading" vs "neither finds positive R") directly answers
whether path-conditional learning extracts edge.

**Inputs:**
- Saved bundle: `output_models/core25_latest/multi_asset_report.pkl`

**What to do:**
1. Run `analysis/force_model_sweep.py --use-extended-features false
   --out output_audit/force_model_basic`.
2. Run `analysis/force_model_sweep.py --use-extended-features true
   --out output_audit/force_model_extended`.
3. Compare verdicts. Specifically:
   - Best gross_R in static grid sweep (Part 1) — does any config cross
     zero?
   - Best top-decile predicted R in force model (Part 2) — does any cell
     have ci95_lo > 0?
   - Lift of learned over fixed.
   - Does adding extended features change the verdict (basic vs extended)?
4. Write `reports/force_model_verdict.md` with both verdicts side-by-side.

**Acceptance:**
- Each run produces `static_grid_sweep.csv`, `force_model_per_config.csv`,
  `force_model_val_predictions.parquet`.
- Report has explicit answer to "did Opus's R3 hypothesis survive?"

**Estimated effort:** 30 min (cloud has 16 cores, sweep is fast).

### TASK R-ARSENAL-VERIFY: Run arsenal end-to-end on real bundle

**Why:** Arsenal is built but never run on real data. Until it runs, we don't
know if any of the model-wrapped alphas adds edge over the base alphas.

**Inputs:**
- Saved bundle.

**What to do:**
1. Run `analysis/run_arsenal.py --null-trials 1000` (cloud can handle it).
2. Read the per-alpha summary. Identify any alphas with `ci95_lo > 0` AND
   both nulls passing (p < 0.05).
3. Read the pairwise combinations. Identify any (alpha_a, alpha_b) pair
   with `combined_mean_R` higher than either individual.
4. Write `reports/arsenal_verdict.md`.

**Acceptance:**
- All artifacts written.
- Explicit listing of: positive alphas, positive combinations, alphas that
  pass nulls.

**Estimated effort:** 1 hour.

### TASK R-AUTOTUNE: Threshold sweep for wrapped alphas (Layer 1 self-growth)

**Why:** Demonstrates the arsenal's self-growth Layer 1. Easy to verify.

**Inputs:**
- Working arsenal (R-ARSENAL-VERIFY complete).

**What to do:**
1. Create `analysis/arsenal_threshold_sweep.py`.
2. For QualityFilteredPoolAlpha, register at min_q ∈ {0.40, 0.45, 0.50,
   0.55, 0.60, 0.65, 0.70}.
3. For DirectionConfirmedPoolAlpha, register at min_direction ∈ {0.51,
   0.55, 0.60, 0.65, 0.70, 0.75}.
4. For PolicyReturnAlpha, register at min_predicted_r ∈ {-0.2, -0.1, 0.0,
   +0.1, +0.2, +0.3}.
5. Run arsenal evaluator with this expanded registry (≈19 alphas).
6. Report best threshold per wrapped alpha.

**Acceptance:**
- `reports/arsenal_threshold_sweep.md` with the best-threshold table.
- New unit tests in `tests/test_arsenal_threshold_sweep.py`.

**Estimated effort:** 4-6 hours.

### TASK R-MIS-V2: Apply MIS enforcement to V2 simulator

**Why:** V2 already uses 1m intra-bar resolution but has no session boundary
enforcement. All V2 measurements have the same overnight-gap bug V1 used to
have.

**Inputs:**
- `liqpool/execution_simulator_v2.py`.

**What to do:**
1. Import the MIS constants and helpers from
   `liqpool.execution_backtest`.
2. In `simulate_pool_trade_v2` and `_exit_5m_fallback`, add the same
   late-entry refusal + same-day exit cap as V1's
   `simulate_pool_trade`.
3. Mirror Opus's V1 tests for V2:
   `tests/test_execution_simulator_v2_mis.py`.
4. Re-run any analysis that uses V2 (e.g. `analysis/run_phase4_track_a_
   pretouch_sweep.py`) and compare numbers to the V2 baseline in
   `reports/phase4_track_a_sweep.md`.

**Acceptance:**
- All new tests pass.
- V2 trades that would have spanned overnight gaps return None or are
  capped at EOD bar.
- A note in `reports/post_mis_baseline.md` (or new file) explaining the
  delta to old V2 numbers.

**Estimated effort:** 4-6 hours.

### TASK R-WRAPPED-PROX: Add proximity + reaction as wrapped alphas

**Why:** 3 of the 5 trained models are now wrapped (Q, direction, R1). The
remaining two (proximity, reaction) need richer input pipelines.

**Inputs:**
- Saved bundle (`report.unified_proximity`, `report.reaction_model`).

**What to do:**
1. `liqpool/arsenal/alphas/proximity.py:ProximityFilteredPoolAlpha`. Use
   the StateFeaturizer to build per-touch features, query
   `report.unified_proximity[h]` at the chosen horizon, gate by
   `min_p_touch` threshold.
2. `liqpool/arsenal/alphas/reaction.py:ReactionConfirmedPoolAlpha`. After
   touch + `confirm_window_bars`, build post-touch event features
   (`liqpool.feature_store.post_touch_event_row` is the template), query
   `report.reaction_model.predict_event_frame()`, gate by
   `min_strict_reaction`.
3. Register both in `default_registry()`.
4. Add tests with stub models in `tests/test_arsenal_model_filtered.py`.

**Acceptance:**
- Both alphas fail-graceful when bundle attrs missing.
- Tests pass.

**Estimated effort:** 1-2 days.

### TASK R-DISTCAL-WIRE: Wire per-distance calibration into joint eval

**Why:** Opus shipped `DistanceCalibrator` as a class with hooks on
ProximityModel + PoolRespectModel but never wired it into the
`walkforward.py` joint-Q×proximity evaluation site, so the distance-binned
cal_err table in the predict summary doesn't reflect the calibration.

**Inputs:**
- `liqpool/walkforward.py:evaluate_timing_frames`.

**What to do:**
1. After existing proximity model fit, call
   `proximity_model.fit_distance_calib(oos_proximity_frame)`.
2. Same for Q model via `model.fit_distance_calib(p_predicted,
   distance_atr, y_true)` at the joint eval site.
3. Persist calibrators in the bundle (they're picklable).
4. Re-run train + predict, compare per-distance cal_err to old values.
5. Update tests.

**Acceptance:**
- Per-distance cal_err shrinks for buckets with n ≥ 300.
- AUC unchanged.

**Estimated effort:** 1 day.

### TASK R-PROX-LEAK: Deep replay audit of proximity AUC 0.97

**Why:** Sampled replay passes, but full replay never done. Proximity AUC of
0.97 is the linchpin of Track A's claimed edge. If there's residual leakage,
Track A's marginal CPCV verdict is even worse than it looks.

**Inputs:**
- `analysis/run_phase4_pool_availability_replay_audit.py` (existing template
  for replay audits).

**What to do:**
1. Adapt the replay-audit pattern to the proximity model. For each OOS
   sample, truncate `df_base` at the snapshot's decision time, recompute
   the proximity prediction, compare to the saved prediction.
2. Tolerance: 1e-6 difference.
3. Report mismatches.
4. Write `reports/phase4_proximity_full_replay_audit.md`.

**Acceptance:**
- Either: zero mismatches → proximity is causal, AUC 0.97 is real.
- Or: mismatches > 0 → leakage source identified, fix proposed.

**Estimated effort:** 2-3 days.

---

## 10. Risk register

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| Track A pocket fails the synthetic-null suite | HIGH | MEDIUM-HIGH | Codex completes R-NULLS first; if all 4 nulls beaten, pocket is robust. If 1+ null is competitive, pocket is partial-noise; downgrade to research-only. |
| V2 simulator MIS bug invalidates Track A baseline | MEDIUM | HIGH | Codex completes R-MIS-V2 then re-runs Track A sweep. |
| Proximity AUC 0.97 is partly leakage | HIGH | UNKNOWN | R-PROX-LEAK. If real leakage found, all proximity-dependent verdicts must be re-run. |
| Force model overfits the 48 configs × pool features | MEDIUM | MEDIUM | R-R3-VERIFY shows whether top-decile generalises. If not, increase val_n threshold; add walk-forward CV inside the regressor. |
| Arsenal nulls find every alpha is noise | MEDIUM | MEDIUM | Honest answer; pivot to options data foundation (Codex's original Plan B). |
| SaaS feed customer commitment before alpha is validated | HIGH | LOW (Opus pushed back) | Continue: research-grade only. No pricing or commercial commit until at least one alpha clears nulls. |
| Cloud bill for long sweeps | LOW | LOW | 16-core box; force_model_sweep is ~10 min. Only R-NULLS at high trial count (1000+) is non-trivial. |

---

## 11. How to run everything — cheat sheet

```bash
# Setup (one-time)
cd liquidity_backtester
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r service/requirements.txt
export DATA_ROOT="/path/to/kite_indian_market_data"
export RESAMPLED_DIR="$DATA_ROOT/resampled"

# Train (heavy; ~30-60 min on cloud)
PYTHONPATH=. .venv/bin/python examples/multi_asset_run.py \
  --universe core25 --data-source parquet --data-dir "$RESAMPLED_DIR" \
  --mode train --asset-workers 8 \
  --model-dir output_models/core25_latest \
  --out output_core25_train

# Predict (fast)
PYTHONPATH=. .venv/bin/python examples/multi_asset_run.py \
  --universe core25 --data-source parquet --data-dir "$RESAMPLED_DIR" \
  --mode predict --asset-workers 8 \
  --model-dir output_models/core25_latest \
  --out output_core25_predict_latest

# Post-MIS-fix baselines (all no-retrain, < 10 min each)
PYTHONPATH=. .venv/bin/python analysis/cost_sensitivity.py \
  --model-dir output_models/core25_latest \
  --notionals 50000,100000,200000 --out output_audit/cost
PYTHONPATH=. .venv/bin/python analysis/pocket_sensitivity.py \
  --model-dir output_models/core25_latest --out output_audit/pockets
PYTHONPATH=. .venv/bin/python analysis/pretouch_pocket_sensitivity.py \
  --trades output_phase4_track_a_pretouch_sweep/pretouch_sweep_trades.parquet \
  --out output_audit/pretouch_pockets

# R3 force model (cloud-fast)
PYTHONPATH=. .venv/bin/python analysis/force_model_sweep.py \
  --use-extended-features false --out output_audit/force_basic
PYTHONPATH=. .venv/bin/python analysis/force_model_sweep.py \
  --use-extended-features true --out output_audit/force_extended

# Arsenal
PYTHONPATH=. .venv/bin/python analysis/run_arsenal.py \
  --model-dir output_models/core25_latest \
  --null-trials 1000 --out output_audit/arsenal

# Tests + lint (before any commit)
PYTHONPATH=. .venv/bin/python -m pytest -q
python scripts/compliance_lint.py
```

---

## 12. Open questions Opus needs Codex's input on

1. **The user mentioned they trade Rs50k notional but the cost-sensitivity
   verdict suggested Rs100k+ where the Rs20 brokerage cap binds.** Should the
   arsenal evaluator default `notional_inr` to 100k instead of 50k? Or expose
   per-asset notional scaling (Rs50k per stock = different qty per asset)?

2. **Per-asset notional vs basket notional** — the user wants 50k per position
   but the simulator currently uses 50k as a global parameter. Codex: please
   verify which the arsenal evaluator should default to and update if needed.

3. **V2 simulator path for Track A.** Codex's `run_phase4_track_a_pretouch_
   sweep.py` already enforces 15:10 IST exit but uses V2 internals. After
   Codex completes R-MIS-V2, should the pretouch sweep be re-run? The user
   may want to re-baseline Track A numbers.

4. **Synthetic-null infrastructure choice.** Codex's roadmap describes 4 null
   types. Opus's null_tests.py uses signal-level permutations (preserve count,
   shuffle attributes). For "random pools at matched spatial density," should
   the null be implemented at the **pool level** (generate fake pools, run
   full pipeline) or **signal level** (preserve real pools, randomize signal
   tags)? Codex's preference: pool-level is more honest but more expensive.

5. **R1 / R3 / arsenal overlap.** All three produce "predicted R per
   candidate" with different feature scopes. After R-R3-VERIFY and
   R-ARSENAL-VERIFY, do we keep all three? Or should one be the canonical
   force model? Opus prefers: keep R1 as the per-mode regressor (intrinsic to
   policy_model.py), R3 as the cross-config sweep tool, arsenal as the
   composition engine. They serve different purposes.

---

## Appendix A: Opus's commit log on this branch (chronological)

```
ec2b835  Add opaque analytics-feed engine over raw predict outputs
ded0147  Add SaaS feed exporter + synthetic smoke test
bc50e4a  Neutralize public feed schema per compliance red-team
a91e3b0  Add end-to-end runbook (setup -> train -> predict -> SaaS feed)
6f85179  Increase default asset workers from 1 to 4 (user-authored, accepted)
a8aadeb  Make runbook macOS-safe and default asset-workers to 4
8240063  Document real data source (Google Drive warehouse) in runbook
a484e8d  Pillar 1: reaction-distribution audit, display fix, per-distance calibration
3d61b8c  Fix audit_reaction_distribution length-mismatch crash
b0643df  R1: PolicyReturnModel — Huber regression on realized net R
2e89e77  Fix V1 execution net_R explosion when next-bar opens past stop
de72842  Add no-retrain cost/position-size sensitivity analysis
ecb08c9  Enforce intraday MIS in v1 execution simulator (no-late-entry + EOD square-off)
868a912  Add pocket-sensitivity analysis: does the alpha live in documented subsets?
e153012  Track A pre-touch pocket sensitivity (CI-aware verdict on existing sweep)
5e7ece4  R3: triple-barrier labels + force-model sweep (48-cell grid, CI95 verdict)
fcc9cf9  Alpha arsenal: framework + 3 seed alphas + null tests + CLI
ac0a9cc  Arsenal grows itself: model-wrapped alphas + meta combinators
```

## Appendix B: pre-Opus Phase 4 commit chain (Codex authored)

Reference only; Opus did not author these but built on them.

```
2746f3e  Add Phase 4 multi-pocket mini-sweeps
5544019  Add Phase 4 multi-pocket slice analysis
c6e2f1a  Add Track A CPCV component null audit
6c9355a  Wire Track A pre-touch research pocket into predict
136071d  Add Track A constrained validation
f762205  Add Track A focused diagnostics
45c5251  Add Track A pre-touch sweep results
4971ceb  Add Track A pre-touch sweep runner
83827e7  Add Phase 4 Q percentile V2 audit
58a21da  Add Phase 4 v2 terminal runbook
d79092a  Fix execution v2 invalid entry geometry
6710b93  Add execution simulator v2 foundation
e650eea  Add Track A direction conditional audit
759a7c7  Add Track A proximity bucket audit
f81a008  Add pool availability replay audit
438df51  Add MTF closed-bar leakage audit
8600b1c  Add proximity distance leakage audit
1d27a89  Add fast leakage probe foundation
fcab4cd  Add Q percentile policy research
987183b  Add Phase 4 Q scale audit
```

---

---

## Codex Update — 2026-06-01 after post-MIS baseline

This section supersedes the old "Next move: R-MEASURE" line above.

### Completed after Opus handoff

1. **R-MEASURE completed and reported**
   - Report: `reports/post_mis_baseline.md`
   - Track B/post-touch remains dead under corrected MIS/cost arithmetic.
   - Cost scaling helps but does not rescue Track B. Best post-touch mode at
     Rs200k notional is still negative (`displacement_confirmed` about -0.29R).
   - Track A/pre-touch remains the only credible equity path.

2. **V2 MIS enforcement completed**
   - File: `liqpool/execution_simulator_v2.py`
   - V2 now refuses out-of-session/late MIS entries and caps same-day exits.
   - Tests: `tests/test_execution_simulator_v2.py`

3. **Fast Track A synthetic-null preflight added**
   - Script: `analysis/run_phase4_track_a_synthetic_nulls.py`
   - Test: `tests/test_track_a_synthetic_nulls.py`
   - Report: `reports/phase4_track_a_synthetic_nulls.md`

### Latest synthetic-null preflight read

Primary pocket: `winning_combo` = morning/midday + AUTO/FMCG/PHARMA + UP.

Result: `SYNTHETIC_PREFLIGHT_CORE_PASS_DIRECTION_WEAK`

- Matched-random rows: PASS, p~=0.001
- Time-bucket shuffle: PASS, p~=0.001
- Sector shuffle: PASS, p~=0.001
- Direction-label shuffle: FAIL, p~=0.512

Interpretation: the core journey pocket is not explained by random matched
rows, time labels, or sector labels. But the hard UP direction gate does not
add distinguishable edge over shuffled direction labels. Direction should be a
soft feature in the final-stage model, not a hard production gate, unless a
later full-V2 replay proves otherwise.

### Next exact order

1. Pull latest code on VPS.
2. Run the heavier pool-level null replay:
   - direction-hard winning combo
   - same core pocket with direction softened/removed
3. Do not paper trade or live trade before full synthetic nulls pass.

### VPS command to confirm preflight

```bash
cd /root/Project-freedom/liquidity_backtester
git pull origin claude/liquidity-pool-backtester-1uskb
mkdir -p logs output_audit/track_a_synthetic_nulls

PYTHONPATH=. .venv/bin/python -u analysis/run_phase4_track_a_synthetic_nulls.py \
  --trades output_phase4_track_a_pretouch_sweep/pretouch_sweep_trades.parquet \
  --trials 2000 \
  --workers 12 \
  --out-dir output_audit/track_a_synthetic_nulls \
  --report-out reports/phase4_track_a_synthetic_nulls.md \
  2>&1 | tee logs/track_a_synthetic_nulls.log
```

**End of Codex handoff. Next move: confirm synthetic-null preflight on VPS, then build full pool-level nulls.**

### Codex Update — pool-level null runner added

After the VPS confirmed `SYNTHETIC_PREFLIGHT_CORE_PASS_DIRECTION_WEAK`, Codex
added:

- `analysis/run_phase4_track_a_pool_level_nulls.py`
- `tests/test_track_a_pool_level_nulls.py`

This runner mutates pool locations and replays them through the Track A v2
execution path. It tests:

- `random_pool_matched`
- `atr_offset_1`, `atr_offset_2`, `atr_offset_3`, `atr_offset_5`
- `sector_neutral_random`
- `shuffled_direction_replay`

It tests both `direction_hard` and `direction_soft` pockets. Use 5m fallback
first on VPS for speed; rerun with `--use-1m-resolution --raw-1m-dir ...` only
if raw 1m data is copied to the VPS.

```bash
cd /root/Project-freedom/liquidity_backtester
git pull origin claude/liquidity-pool-backtester-1uskb
mkdir -p logs output_audit/track_a_pool_level_nulls

PYTHONPATH=. .venv/bin/python -u analysis/run_phase4_track_a_pool_level_nulls.py \
  --model-report output_models/core25_latest/multi_asset_report.pkl \
  --candidates output_phase4_track_a_pretouch_sweep/pretouch_candidates.parquet \
  --trials 100 \
  --workers 12 \
  --out-dir output_audit/track_a_pool_level_nulls \
  --report-out reports/phase4_track_a_pool_level_nulls.md \
  2>&1 | tee logs/track_a_pool_level_nulls.log
```

**End of Codex handoff. Next move: run pool-level null replay on VPS and interpret the result.**
