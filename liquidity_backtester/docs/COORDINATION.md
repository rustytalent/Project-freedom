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

### 2. Opus — Daily Brief generator + renderer ✓ SCHEMA APPROVED, IMPLEMENTATION IN PROGRESS

**Status:** schema at `docs/daily_brief_schema.md` validated against
retrain output 710362b — every section maps to data already in the
bundle. User approved the schema implicitly by greenlighting "do 2
then 1" after the schema review. Implementation now active.

**Scope for v1 implementation:**
  * `liqpool/products/daily_brief.py` — reads a MultiAssetReport
    bundle, produces the JSON.
  * `liqpool/products/brief_renderer.py` — JSON → plain-text email
    body. PDF rendering deferred to v2 (after first paying customer).
  * Tests for both modules.
  * Sections fully implemented for v1: brief_metadata, sector_regime,
    top_watchlist, avoid_list, key_zones, confidence_notes.
  * Sections stubbed for v1 (rendered as "pending"): index_regime
    (needs index data not in current bundle), options_suitability
    (blocked on level-to-strike translator), yesterday_audit (blocked
    on outcome log).

**Acceptance:** generate a brief from the existing 710362b bundle for
trading_date_ist=2026-06-03 with a non-empty avoid_list and verdict
matching the bundle's verdict ("NO TRADEABLE SETUP"). Both JSON and
text renderers round-trip cleanly. Tests pin section presence and
the "no tipster outputs" constraint (no "buy/sell/long/short" in
human prose).

### 3. Opus — Outcome log writer ✓ SCHEMA APPROVED, NOT YET STARTED

**Status:** schema at `docs/outcome_logging_schema.md` approved per
"do 2 then 1". Implementation queued AFTER the Daily Brief generator
because the brief generator's emit-prediction calls feed the log,
not the other way around.

**Sub-tasks:**
  (a) Opus — implement `liqpool/products/outcome_log.py` writer.
  (b) Opus or Codex — backfill 90 days from existing bundles.
  (c) Codex — implement `resolver.py` end-of-day job.

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

### 5. Opus or Codex — options-language presentation layer for 5 indexes

Unblocked. Build a translator that maps proximity output to strike-grid
language for Nifty 50, Bank Nifty, Fin Nifty, plus two more (user to
confirm exact two — likely Nifty Midcap Select + Sensex). v1 is pure
level-to-strike mapping: "P(Nifty touches 24,500 today) = 0.82" →
"24,500 CE / 24,500 PE: 82% chance of being tested before today's EOD".
No Greeks engine, no IV, no PCR — those are higher tiers later. Single
translator module reused across all 5 indexes via a config dict of
(index_name, lot_size, strike_step). Acceptance: the daily-brief schema
(NEXT UP #2) carries an options-language sub-section per index, fed
from this translator.

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
