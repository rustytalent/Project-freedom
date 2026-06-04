# Coordination

Live state board between Opus, Codex, and the user. Read on every
session start. Update on every commit.

For long-lived architecture see `COMPANY_MAP.md`. Don't put architecture
here; don't put tactical state there.

---

## IN FLIGHT

Work an agent is actively holding. Don't touch another agent's entry —
add your own or hand off via a NEXT UP item.

| Owner | Branch | Description | Started | Status |
|-------|--------|-------------|---------|--------|
| (none currently) | | | | |

---

## RECENTLY DECIDED

Append-only log of decisions. One line each.
Format: `YYYY-MM-DD [agent] commit_sha — decision`

```
2026-06-03 [opus]  bc3eb84 — EXPIRY-CTX: 4 calendar-event features added
2026-06-03 [opus]  c709635 — AVWAP-FRVP: 9 volume-weighted features added
2026-06-03 [opus]  eb21b15 — MTF-CTX: 5 today-relative session features added
2026-06-03 [opus]  7202cc1 — PROX-JOURNEY: ProximityFilteredPoolAlpha shipped
2026-06-03 [opus]  68ae5cf — Q-DECOMPRESS: bucket shrinkage capped at 0.30
2026-06-03 [codex] e6cd496 — Arsenal upgraded: 5 journey alphas in default_registry, daily return streams, correlation matrix
2026-06-03 [codex] 9cd710a — Null-test report-passing bug fixed
2026-06-03 [codex] 309a87a — Holdout discipline + Bonferroni + per-alpha turnover + proximity_journey_baseline added
2026-06-03 [opus]  03835f0 — COST-REALISM: min_target_to_cost_ratio filter + default notional bumped to ₹1L + gross_R surfaced
2026-06-03 [opus]  809fb20 — PATH-CTX: 7 path/shape features (trap detector, path efficiency, vol regime)
```

Strategic decisions (recorded for posterity, no commit attached):

```
2026-06-03 [opus + user] — Company has THREE revenue lines: research (sells first),
                            infra/diagnosis (sells second), private alpha (last).
                            Stop chasing private alpha as if it's the only product.
2026-06-03 [opus + user] — First commercial product: Daily Research Brief + post-market
                            Audit, for index/options-flavoured intraday traders. 3 hand-
                            picked customers at high price, not many at low price.
2026-06-03 [opus + user] — One engine, three horizons (swing / intraday / options).
                            Not three separate systems.
2026-06-03 [opus + user] — Strategy-diagnosis is the SALES MOTION for the subscription,
                            not a parallel product line.
2026-06-03 [opus + user] — Outcome logging must be built BEFORE customer #1's first
                            brief, or the most valuable early data is lost forever.
2026-06-03 [opus + user] — Stop adding alpha modules. Next code work is the daily
                            artifact pipeline, not feature #43.
2026-06-03 [opus]  fdbaf08 — BRIEF-V1: Daily Brief generator + plain-text renderer + 18 tests
2026-06-03 [opus]  3ca8a21 — PRODUCT-CORE: outcome log + strike translator + brief integration + backfill + 42 tests
2026-06-03 [opus]  e2c6ccf — SYSTEM-DOC: single comprehensive overview of architecture + pipeline (docs/SYSTEM_OVERVIEW.md)
2026-06-03 [opus]  5689a94 — CLEANUP-1: phase3c noise reduction + stale handoff docs archived + sparse alphas split into research_registry()
2026-06-03 [opus]  6877bce — CLEANUP-2: SectorMoE per-sector experts default OFF (Config.train_sector_experts=False); saves ~40% Q training time with no measurable AUC impact
2026-06-03 [opus]  a6aab91 — PROX-DIST-FIX: pass real distance + state + Q to predict_one (was 1.0 placeholder causing 98% calibration error)
2026-06-04 [opus]  (this)  — ROADMAP-T0: docs/RESEARCH_ROADMAP.md + time-decay sample weighting (Q model only, opt-in via Config.sample_decay_halflife_days) + analysis/pack_artifacts.py (3-file consolidation for chat-upload)
2026-06-04 [codex] 6856af1 — PRODUCT: wire `multi_asset_run.py` to emit
                            `daily_brief.json` / `daily_brief.txt`; fix
                            outcome-log backfill resolution partitioning.
2026-06-04 [user+codex VPS proof] — Task A backfill complete on
                            `core25_head_alpha_710362b`: 90 prediction
                            partitions, 90 resolution partitions, 1260
                            predictions, 1260 resolutions.
2026-06-04 [user+codex VPS proof] — Task B artifact integration complete:
                            `output_predict_brief_710362b_fast/`
                            contains Daily Brief JSON/text and outcome-log
                            prediction parquet. Copy/calibration hardening
                            remains before customer delivery.
2026-06-04 [codex] bf5cffb — EXEC-GEOM: configurable V2 geometry,
                            `break_confirmed`, `sweep_reclaim`, reclaim
                            tests, and consolidated
                            `analysis/run_geometry_mode_sweep.py`.
2026-06-03 [user retrain core25_head_alpha @ 710362b — findings recorded by opus]:
  * vol_regime_zscore_20d is #1 direction feature (gain 4919) — PATH-CTX validated
  * days_to_monthly_expiry is #5 direction feature (gain 2658) — EXPIRY-CTX validated
  * vol_regime_zscore_20d is #6 proximity h=12 feature (gain 11520) — generalizes
  * Q model OOS val AUC = 0.541 (barely above noise) — DEMOTE Q to context-only,
    do not use as a hard live gate; Q-DECOMPRESS revealed honest weakness
  * 4 of 5 sector experts got 0% MoE weight (only FMCG 5.4%) — sector MoE largely
    collapses to global model; sector layer adds little
  * Proximity overall AUC 0.92-0.95 but at 0-1 ATR (trader-relevant distance)
    the honest AUC is 0.73-0.76 — strong, but the headline number is inflated
    by easy far-away cases
  * R1 policy-return regressor: spearman > 0.10 in ALL FOUR modes
    (blind_limit 0.32, displacement 0.17, touch 0.16, reclaim 0.16) — the
    regressor CAN rank trades; top-decile is 27-33% less negative than mean,
    BUT top decile is still negative across all modes
  * Direction model OOS AUC 0.567, top-quartile-confidence accuracy 58.8% —
    useful filter, not standalone signal
  * Reaction model OOS AUC 0.78 (strict_reaction) — strong, calibrated
  * Today's basket verdict: NO TRADEABLE SETUP, max T_today 12% on HCLTECH,
    max T_2d 27% — exactly the kind of AVOID day the daily brief surfaces
  * Cost-realism filter NOT yet observable here (only fires in Arsenal evaluator,
    not bundle-level execution_backtest). Codex NEXT UP #1 still pending.
```

---

## NEXT UP

Ordered priority queue. Each item has owner + acceptance criteria so an
agent can pick it up cold and know when it's done.

### 1. Codex — verify cost-realism + path-context retrain results (HIGHEST PRIORITY)

Pull HEAD (`809fb20`). Retrain `core25_head_alpha`. Run Arsenal with
`--workers 15` and the discipline already wired (holdout + Bonferroni).
Report back on the three cost-decomposition questions:

- After `min_target_to_cost_ratio=3.0` filter, is `pool_reach` cost_R
  down from ~1.1R to <0.3R?
- Is `pool_reach` gross_R still around -0.16R (confirms our diagnosis)
  or did filtering change the population enough to shift it?
- Do any of the 7 PATH-CTX features (`first_15min_range_atr`,
  `path_efficiency_30`, `is_gap_up_trap_fade`, `vol_regime_zscore_20d`,
  etc.) land in top-10 importance for direction or proximity?

Acceptance: the COST WALL SUMMARY table renders with the three new
columns (`mean_gross_R`, `mean_cost_R`, `mean_R`). Feature importance
dump lists which of the 25 new features (across all 5 commits this week)
landed top-10/top-20 per model. Results pasted back into chat or summary
appended here under RECENTLY DECIDED.

### 2. Opus — Daily Brief generator + renderer ✓ COMPLETE (commit fdbaf08)

`liqpool/products/daily_brief.py` + `brief_renderer.py` + 18 tests
in commit `fdbaf08`. JSON contract matches `docs/daily_brief_schema.md`.
Renderer enforces no-tipster-vocabulary guardrail at render time.

### 3. Opus — Outcome log writer + integration + backfill ✓ COMPLETE (commit 3ca8a21)

`liqpool/products/outcome_log.py` + `strike_translator.py` + brief
integration + `analysis/backfill_outcome_log.py` + 42 new tests in
commit `3ca8a21`. The brief generator now logs three classes of
predictions (proximity / avoidance / options_strike) to the
append-only Parquet flywheel, populates options_suitability via the
5-index strike translator when index data is provided, and renders
yesterday_audit from the joined log when history exists.

### 5. Opus — 5-index strike translator ✓ COMPLETE (commit 3ca8a21)

INDEX_CONFIGS for Nifty 50 / Bank Nifty / Fin Nifty / Nifty Midcap
Select / Sensex pinned with strike_step + lot_size + weekly expiry
weekday. translate_proximity + theta_danger_score +
premium_regime_for_buyers_and_sellers shipped. No Greeks / IV / PCR
in v1 — higher-tier additions when paying customers ask.

---

## NEW QUEUE — post-PRODUCT-CORE

The original items 1-5 are complete or owned. The following sequence
unblocks the first customer-shippable brief on the real bundle.

### A. Codex — run outcome-log backfill on the real bundle (BLOCKER)

```
PYTHONPATH=. python analysis/backfill_outcome_log.py \
    --bundle output_models/core25_head_alpha_710362b/multi_asset_report.pkl \
    --output-root data/outcome_log \
    --days 90
```

Acceptance: parquet files appear under `data/outcome_log/predictions/`
and `data/outcome_log/resolutions/`, one partition per IST trading
date. Print row counts per table per partition. Verify
`OutcomeLogWriter().read_joined("YYYY-MM-DD")` returns a non-empty
frame for at least 60 of the 90 partitions (some weekends and
holidays will be empty; the rest must populate).

If the backfill crashes on a real bundle attribute the synthetic
stubs didn't exercise, report the traceback and STOP — don't paper
over it. Opus will fix the generator.

### B. Codex — wire generate_brief() into multi_asset_run.py

Find the artifact-emission section of `examples/multi_asset_run.py`
(near where `live_plan.json` is written) and add a brief-generation
call that writes:
  * `daily_brief.json` (the dict from `BriefDocument.to_dict()`)
  * `daily_brief.txt` (the output of `render_email(brief)`)
into the existing output directory.

The brief generator's optional kwargs (`outcome_log_writer`,
`index_data`) start as None / empty in v1. Wiring index_data
requires index OHLCV which is not in the bundle yet — that's a
separate later task.

Acceptance: after a multi_asset_run training pass, both files exist
in the output directory. The `daily_brief.txt` is human-readable and
non-empty.

### C. Codex — verify cost-realism Arsenal result (STILL OWED)

Original NEXT UP #1, never closed. Pull HEAD (now at `3ca8a21`),
retrain on a small basket (5 assets is fine for verification),
run Arsenal with `--workers 15`. Report the COST WALL SUMMARY
table — specifically whether the new `mean_gross_R` and `mean_cost_R`
columns confirm or refute the gross/cost decomposition that motivated
the COST-REALISM commit (Opus's prediction: gross_R near -0.16,
cost_R near 1.1 at multiplier 1.0).

### D. Opus — wire StateFeaturizer-derived index_data into the brief

When Codex has index OHLCV in a bundle (separate work item, currently
unblocked but unassigned), populate the `index_data` dict from the
StateFeaturizer's already-computed `vol_regime_zscore_20d`,
`path_efficiency_30`, `direction_changes_30` per index. This unstubs
the index_regime and options_suitability sections for production
briefs.

---

## NOT NEXT UP (explicitly deferred)

### 4. Codex — wire avoidance-alpha as first-class output

The "do not trade today" call is as valuable as the trade calls.
Trigger conditions (proposed; user to confirm): `vol_regime_zscore_20d
> 1.5` AND `path_efficiency_30 < 0.20` over the last 30 bars AND/OR
`is_gap_up_trap_fade == 1` AND `direction_changes_30 > 15`. Emit a
single boolean per asset per day in the OOS frame so the brief can
read it.

Acceptance: a new column `avoidance_recommended` in the trade frame
(or a sibling table per asset per day), with a test pinning the
trigger logic.

(Items 1, 4 above remain owed by Codex. Items 2, 3, 5 are complete —
see new queue A/B/C/D below.)

---

## OPEN QUESTIONS FOR USER

Answer these inline (replace the question with the answer + initials).
They block NEXT UP items.

1. **Which index first for the options-language layer?**
   **ANSWERED (user, 2026-06-03):** Five major indexes —
   Nifty 50, Bank Nifty, Fin Nifty, plus two more (likely Nifty Midcap
   Select + Sensex; user to confirm exact two). Engine is the same per
   index, only the strike-grid translator repeats.

2. **First three customers — when do we want to be brief-ready?**
   **ANSWERED (user, 2026-06-03):** A few weeks. Drives parallel build
   of outcome-logging alongside the brief, not sequenced after.

3. **Pricing for the first 3 — confirm ₹20–25k/month?**
   **ANSWERED (user, 2026-06-03):** FREE for the first week to the
   small pool of hand-picked pilot users. Charge from week 2 onward.
   Strategy-diagnosis wedge: pricing TBD — likely free during the same
   pilot week, paid afterwards. Number per-customer: not locked, will
   set after pilot week-1 outcomes are observed.

4. **Delivery format for the daily brief?**
   **ANSWERED (user, 2026-06-03):** PDF + plain email YES for v1.
   Web dashboard ONLY if total monthly hosting + build cost is < ₹300.
   In practice this means GitHub Pages / Vercel free tier / Cloudflare
   Pages — anything that's ₹0/month for static content. No paid SaaS
   tooling for the dashboard until paid customers exist.

---

## NOT NEXT UP (explicitly deferred)

These are things one of us might be tempted to do. Don't, until they
graduate to NEXT UP.

- Mobile app
- Public community / Discord / Telegram
- Dhan/Truedata data acquisition (until OHLCV-derived features prove insufficient)
- Greeks computation pipeline (until level-to-strike layer is paying)
- API/dashboard product (until 3 paying customers exist)
- More alpha modules (until daily brief is shipping)
- More state features (the 42 we have is enough until retrain proves they aren't)
- VIX/Nifty cross-section integration (deferred until current OHLCV-derived signal proves orthogonal lift)

---

## How to update this doc

When you finish a piece of work:
1. Append a one-line entry to RECENTLY DECIDED with date, agent,
   commit sha, and what shipped.
2. Remove your entry from IN FLIGHT if it's done; update status if
   it's blocked or paused.
3. If your work created new follow-ups, add them to NEXT UP with
   owner + acceptance criteria. Don't leave loose ends in chat that
   the next session won't see.

When you START a piece of work:
1. Add yourself to IN FLIGHT with a one-line description.
2. Check if it overlaps with anyone else's IN FLIGHT entry. If yes,
   coordinate before editing the same files.
