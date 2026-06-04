# Research Roadmap

> Every idea that came out of the strategy conversations, captured in
> detail before context compaction loses it. Items are ranked, sequenced,
> and (where possible) scoped to a concrete implementation step. This is
> NOT a backlog of nice-to-haves — it is the canonical research+product
> plan we reason against going forward.
>
> Update rules: items move from "queued" -> "in flight" -> "shipped" via
> COORDINATION.md. NEVER delete an idea from this file; if we decide not
> to do something, move it to the bottom under "explicitly declined" with
> the reason. The point is that future cycles can re-evaluate.

Cross-references:
* `docs/SYSTEM_OVERVIEW.md` — what we have today.
* `docs/COMPANY_MAP.md` — long-lived architecture.
* `docs/COORDINATION.md` — live tactical state.

---

## TIER 0 — fold into the current cycle (this week)

Items that are cheap, founded, and lift the product we are about to
ship. Doing these BEFORE customer #1 reads brief #1 is correct.

### T0.1 — Time-decay sample weighting

**Idea**: treat 3-year-old training data as less informative than last
month's. Today we weight every sample equally; regimes drift; stale
samples can drag the model toward patterns that no longer work.

**Implementation scope (committing in this cycle)**:
* `liqpool/sample_weights.py` — `time_decay_weights(times, half_life_days)`.
* `Config.sample_decay_halflife_days: Optional[float] = None` (default
  off, opt-in).
* Thread weight array through `PoolRespectModel.fit` -> LightGBM
  `sample_weight=`.
* Apply to Q (global) only in v1. Direction/proximity/reaction are
  separate add-ons in T1.

**Founded references**: Lopez de Prado AFML ch. 4 (sample weights from
return attribution + time-decay).

**Expected lift**: small (~1-2% AUC). Most valuable when regime is
shifting — exactly the case we have now (Q AUC 0.541, post-touch
breaks dominating).

### T0.2 — Artifact compression

**Idea**: the training pipeline emits ~20 files per run; the user can't
upload that many to chat. Compress to 3-4 consolidated files without
breaking downstream consumers.

**Implementation scope**:
* `analysis/pack_artifacts.py` — reads an output directory, emits:
  * `consolidated_summary.json` — multi_asset_summary + research_summary
    + live_plan + brief metadata.
  * `consolidated_models.json` — per-model OOS + calibration + feature
    importance.
  * `consolidated_backtest.json` — execution v1/v2 + arsenal + policy_R + cost wall.
* Original files stay untouched (downstream code unchanged); the
  consolidated bundles are for upload to chat.

### T0.3 — Detector upgrade (frontloaded per user override)

**Idea**: current detectors (EQH/EQL, FVG, OB, REJ, ORB, HVN, PD/PW/PM,
SWING) are foundational but not differentiated. To make the brief
*novel* — your word — we need detector signals competitors don't have.

**Concrete candidates to add (research cycle, 2-3 commits)**:

* **Liquidity sweep** — high pierced by wick of size >= K * ATR AND
  closed back below within N bars. This is the "stop hunt" pattern
  the user described in the manipulation-aware section of the
  brain-dump. Distinct from break-and-hold; currently undetected.
* **Multi-bar imbalance / inefficiency** — generalisation of single-bar
  FVG to 3-bar imbalances where bar[i].high < bar[i+2].low (or vice
  versa). Captures larger displacement events.
* **Premium / discount midpoint** — 50% retracement level of the last
  impulse leg (most recent swing-to-swing move). Standard
  institutional reference point.
* **Volume-weighted swing levels** — swing high/low weighted by volume
  traded within K bars of that swing. A swing at 1M-share volume is
  structurally stronger than one at 100k-share volume.
* **Cumulative delta divergence proxy** — without tick data, proxy via
  ratio of up-volume to down-volume over a window vs price direction.
  Detects accumulation/distribution invisible to pure price detectors.
* **Asian / overnight range markers** — for index work, the
  pre-Asian / Asian session high+low marked as same-day reference
  levels.
* **Stop-run + reclaim pattern** — explicit detector for the canonical
  SMC sweep-then-reclaim sequence (the "swept_and_reclaimed" outcome
  is already labelled; what's missing is a DETECTOR that flags the
  setup ex ante, before the reclaim is confirmed).

**Acceptance per detector**:
* Causal — no look-ahead.
* Added to `Pool.contributors` so existing scoring + factor_count work.
* Test fixture pinning detection on a known pattern + non-detection on
  a non-pattern bar.
* OOS feature importance check — does the new factor show up in any
  model's top 15 after retrain?

**Scope discipline**: 2 detectors per commit, not 7 in one shot. Each
gets its own importance/lift verification.

---

## TIER 1 — fund with revenue (post first 3 paying customers)

Real research projects that move the product but take days-to-weeks.
Each is its own commit-train cycle.

### T1.1 — Manipulation-aware / path-dependent direction model

**The single deepest idea in the brain-dump.** Today's direction model
predicts P(up over horizon H) from a snapshot at decision time. But a
human pro doesn't trade on a snapshot — they trade on the *path* and
read manipulation in real time. A liquidity sweep that takes out their
stop and reverses is observable in the bar stream but invisible to a
snapshot-only model.

**Architecture proposal**:
* New feature group `path_along_route`: as we predict whether price
  reaches level L, also compute the proximity-model output at every
  intermediate pool *along the path*. So the model sees not just
  "direction is up" but "the journey goes via these 3 liquidity pools
  with these P(touch) values".
* New feature `sweep_recently_completed`: 1 if a liquidity sweep
  (T0.3) fired in the last K bars, 0 otherwise. Sweeps frequently
  precede reversal.
* Regime-conditional stop sizing: when `sweep_recently_completed` AND
  proximity to a known pool is high, *widen* the implied stop to
  survive the swap. This is the user's "I'd stay in the trade because
  I know it's a liquidity grab" intuition, implemented.

**Expected impact**: direction AUC could go from 0.567 to 0.62+. If
this works, it's the single thing that turns private-alpha viable.

**Prerequisite**: detector upgrade (T0.3) for the liquidity-sweep
detector. Can't build manipulation-aware features without a
manipulation detector.

### T1.2 — Sub-alpha library (sub-models per alpha)

**The "momentum with vs without volume" insight.** Each current alpha
is monolithic. Reality is: momentum + high volume = continuation
likely; momentum + low volume = trap likely (and shortable).

**Architecture proposal**:
* Each top-level alpha gets a *sub-classifier* — a small model that
  outputs a sub-label per signal: `momentum.with_volume`,
  `momentum.fake_breakout`, `mean_reversion.in_range`,
  `mean_reversion.fading_trend`, etc.
* The arsenal evaluator routes by sub-label; sub-label P&L is reported
  separately.
* The fake-breakout sub-label can be inverted into a SHORT signal (the
  user's "if we know it's fake, we can short it" point).

**Synthetic-data flywheel**: every fake-breakout we detect becomes a
labelled training example for *both* the sub-classifier AND the
direction model (it learns "this is what manipulation looks like").
We're generating our own training data from our own signal stream.

**Scope**: start with `momentum.with_volume` vs `momentum.fake_breakout`
as the v1 split. If lift is real, extend.

### T1.3 — Q model: target enrichment + sub-models

Q is binary (respect / break). User's right that this is crude.

**Improvements queued**:
* Multi-class target: `respected_strong`, `respected_weak`,
  `swept_and_reclaimed`, `broken_weak`, `broken_strong`. We already
  emit these as outcome labels; switch Q from binary to multinomial.
* Per-factor sub-models: a separate Q head for each headline factor
  family (EQHL, FVG, OB, REJ). Factors behave differently; one model
  blurs them.
* Q regression on continuous reaction strength (in ATR) instead of /
  alongside the binary respect classifier.

**Prerequisite**: Q AUC is currently 0.541. Improvements only matter if
some Q-related signal is recoverable; T1.3 is a structured search for
that signal, not a build-from-faith.

### T1.4 — Continuous (vs snapshot) feature pipeline

The user's observation that snapshot features lose the information
between snapshots is correct. A snapshot every 12 bars throws away
~11 bars of evolution.

**Proposal**:
* Run featurization every bar (not every 12) on the OOS window.
* Add `bar_seq_X_atr` features for the last N bars of price evolution
  encoded as a fixed-length vector (mini-sequence input).
* Train models on the denser snapshot set; cost increase ~12x but
  manageable on the VPS.

**Risk**: compute cost. Mitigate by sampling at 6-bar instead of 1-bar
intervals in v1 (still 2x denser than today).

---

## TIER 2 — the Brain / Strategy Executor organ

The user's deepest architectural ask: a real strategic brain with
sub-organs that cross-talk, sees alphas as ingredients to mix surgically
(aqua regia, not bulk-blend), and adapts strategy to context.

This is the destination architecture. It is months out. **Do not start
it before there is one validated edge that the brain has to combine.**
Combining negative-edge alphas into a more elaborate negative-edge
system is a known failure mode.

### T2.1 — Alpha mixer that conditions on regime context

The "right ratio under right conditions" idea, made concrete:
* Per (regime, alpha) lookup of historical net R.
* At inference: classify current regime, pull the alpha weights for
  THAT regime, blend.
* Regime classification uses the path/context features we already have
  (vol_regime_zscore_20d, path_efficiency_30, is_gap_up_trap_fade, etc.).

### T2.2 — Strategy synthesis layer

Combines L2 model outputs + L3 alpha selections into multi-leg
strategies (pre-touch position + post-touch reaction sizing + hedge
overlay) where each leg is contextual. This is what the user means by
"strategy comes from alpha, alpha comes from inputs". v2+ territory.

### T2.3 — Cross-organ communication protocol

Each sub-organ (data, features, models, alphas, executor, audit)
emits structured messages the others consume. The brief's
yesterday_audit already does this in primitive form (audit informs
tomorrow's confidence_notes). Generalising is the work.

---

## TIER 3 — product surfaces (real, but post-revenue)

### T3.1 — Strategy Diagnosis as the SCALABLE LEAD product

User insight worth promoting up the priority order:

> The data we are giving to the people should be limited to minimum
> highest paying people. But this strategy diagnosis can be given to
> everyone, small to small retail retailer. It is improving their
> profitability. It is also improving the market quality. It is also
> not giving them specific targets. So that is a very good thing.

This is a more scalable + safer product than the daily brief:
* SEBI-safer (you analyse THEIR strategy; you don't tell them what to
  do).
* Doesn't risk our edge by exposing actionable levels to a wide
  audience.
* Higher unit economics: one-time deliverable, can charge upfront.
* Markets globally — quants outside India will pay too.

**Status**: the components exist (arsenal evaluator + per-mode R + cost
decomposition + null tests). What's missing is the *product wrapper*:
a customer ingests their strategy as a function or rule-set, we run it
through our pipeline, we deliver the diagnosis PDF.

**Scope for first version**: build after the daily brief proves out
with the first 3 customers. Use lessons learned about what they
actually wanted.

### T3.2 — Brief tiering (T1 / T2 / T3 / T4)

User's framing:
* T1 — one-liner / headline. Lowest price, widest audience.
* T2 — full brief as v1 schema. Mid tier.
* T3 — brief + customer-strategy diagnosis layered in. High tier.
* T4 — bespoke research / custom analytics. Highest tier.

Per instrument class:
* Brief for equity (intraday MIS) — different content from
* Brief for index (intraday + options) — different from
* Brief for swing (multi-day positional) — different from
* Brief for options (Greeks-aware) — eventually.

### T3.3 — Hedging / portfolio / risk-management product

Once intraday + swing + index briefs exist, the natural cross-sell is a
hedging overlay: "you're long this; here's the right put to hedge it".
Requires Greeks. Requires real options data. Requires portfolio-level
view we don't have. Real work, real money, real timeline.

### T3.4 — Swing trading product (engine ready)

User: "if swing is ready, ship it". The engine IS ready — the same
proximity model trained at horizon = days works. Build the swing brief
schema (different from intraday: no MIS cap, longer T, level
persistence over multi-day windows). The daily brief generator already
has the structural shape; this is a swing-specific extension.

### T3.5 — Index trading product (engine ready, data pending)

User confirmed they will provide index OHLCV. Once that lands:
* Run the existing pipeline on the index symbols.
* Wire `index_data` into the brief generator (it's already a kwarg).
* options_suitability section unstubs automatically via the strike
  translator.

---

## TIER 4 — sellable synthetic-data products (Moats)

User explicitly called out that the data WE generate as a byproduct of
our pipeline could be sold standalone. These are moats — nobody else
has them.

### T4.1 — Slippage dataset

**What it is**: for every historical trade in our backtest, the delta
between the IDEAL entry/exit price (signal bar close) and the REALISED
fill price after slippage modelling. Time series, per asset, per
session, per side, per session-character regime.

**Why it's a moat**: not for sale anywhere else. Indian retail quants
desperately need realistic slippage to make their backtests honest.

**How to package**: nightly snapshot of `execution_backtest_*_trades.csv`
re-shaped into a `slippage_by_(symbol, side, regime, hour)` parquet.

**Customer**: retail quants, algo desks, small props. Subscription
~₹2-5k/mo, very long-tail.

**Scope to ship**: extraction script from existing trade outputs.
One-week build. Post-revenue.

### T4.2 — Pool / liquidity-zone dataset

**What it is**: every detected pool with score, factor, TF count,
respect outcome (when known), per asset, per session.

**Why it's a moat**: the detection algorithm is reverse-engineerable,
but the curated, audited time-series of detected pools with outcomes is
a real dataset.

**Customer**: institutional research, prop desks who want raw
structural data feeds.

**Risk**: this is the closer-to-edge dataset. Selling it could erode
our own usage. Tier accordingly: sell with a 5-day delay, or sell only
to non-competing audiences (quant academia, retail traders not on our
own brokers).

### T4.3 — Outcome-log / calibration dataset

**What it is**: the prediction+resolution log itself (T0/T1 already
written). Compounds into a track record nobody else can replicate.

**Why it's a moat**: a 3-year audited "predicted P → actual P" trail
for proximity, direction, reaction is genuinely irreplaceable. Both
proof-of-skill and a saleable dataset (anonymised, aggregated).

**Customer**: research consumers, due diligence for any future capital
raise.

### T4.4 — Detector / event-marker dataset

If T0.3 ships the new detector library (liquidity sweeps, multi-bar
imbalances, premium/discount, volume-weighted swings), the *detected
events themselves* — time-series of "sweep occurred on HDFCBANK at
this time" — is a sellable feed. Same risk class as T4.2; same
mitigation.

### T4.5 — Synthetic-data flywheel (internal first)

User insight: every signal we generate (including ones our gates
REJECT) is a training data point for our own future models.

**Concrete instances**:
* Rejected `momentum.fake_breakout` candidate -> labelled training row
  for the direction model: "here is what manipulation looks like".
* Rejected avoidance day where the basket actually had a winner ->
  training row for the avoidance model: "here is a false positive".
* Touched-but-rejected pool (Q < threshold) -> Q model gets a hard
  negative example.

We feed the flywheel back into ourselves first; selling it externally
is a downstream T4 item.

---

## TIER 5 — data sourcing improvements (independent of model work)

### T5.1 — More diverse data inputs

User: "if we have a diversity of features, then we can also have
different alphas. Because the source was inherently different, these
alphas are also not interdependent."

**Concrete next data sources, ranked by leverage / cost**:
1. **India VIX** (free via Kite, already specced). Vol regime.
2. **Nifty / Bank Nifty / Fin Nifty / Midcap Select / Sensex OHLCV**
   (free, user is providing). Index context per stock + the engine for
   the index product.
3. **F&O bhavcopy daily** (free from NSE). OI build / unwind.
4. **Dhan API option chain with Greeks** (free with account, validate
   claim). IV / Greeks for the options brief.
5. **RBI policy calendar** (scrape, free). Event-window features.
6. **Sectoral indices** (free via Kite). Cross-sectional rotation
   stronger signal than the stock-baskets we synthesize today.
7. **Implied vol surface** (paid, deferred). Premium pricing for tier-3
   options brief.

### T5.2 — Data quality / latency

When live: tick-data capture from Kite WebSocket for the forward-only
microstructure flywheel. Day-1 of customer #1's brief.

---

## TIER 6 — housekeeping & cleanup (continuing the discipline)

### T6.1 — Causality enforcement audit

User specifically called this a backbone. Treat it as such.

**Action**: comprehensive `tests/test_causality_invariants.py` that
walks every feature, every model fit, every label generator, and
asserts truncation-stability + no-look-ahead. We have piecemeal
versions today (test_path_context_features etc.). Consolidate into a
single auditable invariant suite.

### T6.2 — Continued cleanup cycles

Following the same discipline as CLEANUP-1 and CLEANUP-2:
* Periodically re-audit what's earning compute vs. wasted.
* Move stale code to `experiments/` or `archive/`, never delete
  prematurely.
* Mark every kept-but-deprecated thing with a comment explaining
  the disposition.

---

## Cross-cutting principles

These rules govern HOW we work, regardless of which tier.

1. **No new alpha modules until existing ones clear costs.** Tested
   in CLEANUP-1's `default_registry` shrink to 5.

2. **Ship before optimising.** The detector upgrade (T0.3) is the one
   case where the user explicitly overrode this — and only because
   detectors directly affect the brief's quality. Generally: working
   v1 > perfect v2.

3. **The outcome log is the moat.** Every prediction we make logs.
   Every resolution we observe logs. Three years from now this is the
   single hardest-to-copy asset we own. Never break the log writer.

4. **Tipster-language guardrail is non-negotiable.** Render-time check
   raises on any forbidden phrase. SEBI-safe by code, not by policy.

5. **Detect, document, defer.** New idea -> write it here -> sequence
   it -> revisit when its tier fires. Don't start it mid-cycle.

6. **Causal-by-construction.** Every feature, every label, every audit
   passes a truncation test. Always.

7. **Three revenue lines, sequenced.** Research subscription first;
   strategy diagnosis as scalable wedge; private alpha last. The
   diagnosis line is now elevated (per T3.1) to potentially-equal
   priority with the subscription.

---

## Explicitly declined (with reasons)

> Items we considered and decided NOT to do. Kept here so future
> cycles don't re-litigate.

* **Full Greeks engine for v1 options brief** — wait until paying
  customer asks. Level-to-strike mapping suffices for v1.
* **Mobile app** — distribution problem, not a product problem.
* **Public Discord / Telegram community** — SEBI risk, support burden.
* **Copy-trading** — outside the SEBI-safe product corridor.
* **Tipster-style "buy X at Y stop Z" outputs** — never. Hard architectural
  constraint enforced at render time.
* **Post-touch cash-equity MIS self-trading at current cost structure**
  — empirically dead per the 2026-06-04 geometry-mode sweep (commit
  1e9a9de). All 18 cells across {6 geometries} × {3 modes} negative;
  best (2.5/2.5 / touch_confirmed) at mean_R = -0.324R. The model
  ranks correctly (R1 spearman > 0.10, reaction AUC 0.78), but the
  cost arithmetic on small-ATR cash-equity moves consumes any gross
  edge that exists. Reclaim mode: gross_R only -0.052R but cost_R
  +0.906R — 17× cost-to-gross ratio.
  Implication: do not retry post-touch geometry tweaks unless one of
  three structural changes has been made: (a) cost regime drops
  dramatically (e.g. broker waives brokerage), (b) trade is moved to
  options where ATR move → multi-R premium move, (c) trade is moved
  to a substantially larger-ATR instrument class.
  The proximity model itself is unaffected — predictions remain
  calibrated and valuable as RESEARCH product output to customers
  whose execution cost structure differs from ours.

---

## Maintenance

Every cycle, the COORDINATION.md RECENTLY DECIDED section records what
shipped. When an item from this roadmap ships, mark it here as DONE
with the commit sha. When an item is decided against, move it to
"Explicitly declined" with the reason. The point of this document is
that no idea is lost — even ones we choose not to do, we choose them
explicitly with context.
