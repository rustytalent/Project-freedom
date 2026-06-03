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

### 2. Opus — design the Daily Brief artifact schema

Define the data contract for the daily brief BEFORE either agent writes
generation code. Sections, per the design discussion:

1. Index regime (today's vol bucket, gap classification, trap risk)
2. Sector regime (which sectors trending vs chopping)
3. Top watchlist (5–10 instruments with proximity scores)
4. Options suitability — for Nifty/BankNifty: directional bias, expected
   range bucket, theta-danger flag, trend-vs-chop call
5. Avoid list (instruments + regimes where model says "stand aside")
6. Key zones (the proximity-model levels with P(touch) + horizon)
7. Confidence notes (which model heads are calibrated today, which are
   drifting)
8. Yesterday audit (what we predicted yesterday → what happened)

Acceptance: `docs/data_product_spec.md` updated (or new file
`docs/daily_brief_schema.md`) with JSON schema + human-prose template.
No code yet — the schema must be reviewed by user before generation
code is written.

### 3. Opus — design outcome-logging schema

The data flywheel starts here. Define what gets logged per prediction:

- Prediction ID, timestamp, instrument, level, horizon, P(touch),
  model version, regime tags
- Resolution: did the level get touched within horizon? exit price?
  drawdown along the way? whether the avoid-list call would have
  protected capital?

Acceptance: schema doc committed under `docs/`. No code yet — pin the
contract first. Must support backfilling from existing OOS bundles so
we have history from day one, not just from customer #1.

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

### 5. (deferred — see OPEN QUESTIONS) — options-language presentation layer

Pending user decision on which index to start with (Nifty? BankNifty?
both?). When chosen: build a small translator that maps proximity
output ("P(Nifty touches 24,500 today) = 0.82") to strike-grid language
("the 24,500 strike has 82% chance of being tested before today's EOD").
No Greeks engine yet. v1 is pure level-to-strike mapping.

---

## OPEN QUESTIONS FOR USER

Answer these inline (replace the question with the answer + initials).
They block NEXT UP items.

1. **Which index first for the options-language layer?** Nifty 50, Bank
   Nifty, both, or Fin Nifty? (Affects strike-grid math and the customer
   ergonomics of v1.)

2. **First three customers — when do we want to be brief-ready?** A
   week? A month? This affects whether outcome-logging is built in
   parallel with the brief or sequenced after it.

3. **Pricing for the first 3 — confirm ₹20–25k/month per customer or
   different number?** Affects whether we offer the strategy-diagnosis
   wedge as free or as a paid one-time deliverable (₹5k? ₹10k?).

4. **Do you want the daily brief delivered as PDF, plain email, web
   dashboard, or all of the above for v1?** Affects what the generation
   pipeline has to render to.

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
