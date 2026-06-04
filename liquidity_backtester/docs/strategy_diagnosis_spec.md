# Strategy Diagnosis Product — Specification

> The customer-facing infrastructure-as-service product. Given a
> customer's strategy (described as a rule-set, code, or a frozen
> trade log), produce an audited diagnostic report covering: regime-
> stratified performance, leakage check, cost decomposition,
> calibration audit, robustness null tests, and prescriptive
> next-steps.
>
> Elevated to potential LEAD scalable product per the 2026-06-04
> post-touch finding: customers paying us to X-ray their strategies
> is SEBI-safest, lowest-friction, most-scalable channel and works
> regardless of whether OUR private trading clears costs.
>
> Cross-references:
>   * docs/SYSTEM_OVERVIEW.md  — what our pipeline does.
>   * docs/RESEARCH_ROADMAP.md  — T3.1, T6.1.
>   * docs/COORDINATION.md  — current state.

## 1. Product positioning

### What it is

"Send us your strategy. We run it through our pipeline. We send back
an audited report covering 8 dimensions you cannot easily measure on
your own."

### What it is NOT

- It is not a backtest service. Backtest services give you a P&L
  curve. We give you a P&L curve PLUS the diagnostic context that
  explains WHY the curve has the shape it has.
- It is not an advisory service. We do not tell the customer whether
  to trade their strategy. We tell them what its true characteristics
  are and let them decide.
- It is not a one-shot consulting engagement. It is a structured
  deliverable with consistent format the customer can compare
  against future iterations of their own strategy.

### Why it exists

Three reasons stack:

1. **Indian retail and small-prop quants have no honest audit
   infrastructure.** They run their strategies in their own
   backtesters which are often leaky, ignore costs, or assume
   unrealistic fills. They KNOW this. They want third-party audit
   they can trust.

2. **The infrastructure we built for our own research is the exact
   infrastructure they need.** Purged-embargoed walk-forward,
   triple-barrier labels, calibration audits, null tests, regime
   stratification, MIS-honest execution simulation, cost realism.
   They cannot easily build these.

3. **The 2026-06-04 post-touch result PROVES this is a viable
   business model.** Our own pipeline correctly identified that
   our private alpha didn't work, with full diagnostic context.
   That is exactly the service we sell — clarity that costs less
   than the savings from not deploying a doomed strategy live.

## 2. The 8 dimensions of the diagnostic

Every diagnosis report covers these 8 sections. Customer ingests
their strategy (rule-set, code, or trade log), we produce a fixed-
shape report.

### Dim 1 — Survivorship & Coverage

What the strategy traded over what window, what it didn't trade,
which instruments and regimes are under-represented in the trade log,
and what that biases the headline metrics by.

Concrete output:
- N trades total, by month, by instrument, by side.
- "Untraded windows" — periods where the strategy had no signal
  fire (could be conservative gating or could be hidden lookahead).
- Sample-size warning: any sub-cohort with N < 30 flagged as
  statistically uninterpretable.

### Dim 2 — Cost Realism

Re-run the strategy under realistic Zerodha intraday costs (or
customer's broker if specified). Decompose realised R into gross_R
and cost_R per trade. Compute the cost-to-gross-ratio. Stress at
1.0x / 1.5x / 2.0x cost multipliers.

Concrete output:
- mean_gross_R, mean_cost_R, mean_R per mode, per regime.
- Cost-wall verdict: "this strategy clears the 1.5x cost stress in
  [X] of [Y] sub-cohorts."
- The cost dominance ratio that triggered our own post-touch death.

### Dim 3 — Leakage Detection

Run the strategy's signal generation against our leakage-replay
harness. Catches: bar-aligned timing leaks, label-window truncation,
forward-looking feature usage, embargo violations.

Concrete output:
- Per-signal leak class with severity (WARN / ERROR).
- Replay reconstruction: "we re-derived your signal on bar j using
  only bars i <= j and got [match / mismatch]."
- The single largest source of false confidence in retail
  backtesting; flagging it is itself a service.

### Dim 4 — Regime Stratification

Stratify performance by vol regime, session character (trend /
range / chop), sector trend, F&O expiry cycle, gap classification,
liquidity-sweep recency.

Concrete output:
- Per-regime mean_R, hit rate, sample size, p-value.
- "Your strategy works in [X regimes] and breaks in [Y regimes]."
- The diagnostic that lets a customer add their OWN regime filter
  (without us giving them one — which would be advisory).

### Dim 5 — Calibration Audit

If the strategy emits confidence scores, isotonic calibration plus
per-bucket calibration error. If purely binary (trade / don't trade),
report the implied confidence calibration via hit-rate distribution.

Concrete output:
- Calibration error per confidence bucket.
- Reliability diagram.
- "Top-decile-confidence trades are X% accurate vs Y% base rate."

### Dim 6 — Null Robustness

Time-shuffle null, sign-flip null, pool-level null, synthetic null
applied to the strategy. Reports whether reported edge survives.

Concrete output:
- Per-null-test: p-value of "is real performance better than null?"
- Bonferroni-adjusted thresholds (configurable per total-test count).
- "Your strategy beats [N of 4] null tests at p < 0.05."
- A strategy that beats all four nulls is genuinely robust. One
  that beats 0-1 is curve-fit.

### Dim 7 — Cost-to-Edge Sensitivity

Stress test: at what realistic cost multiplier does the strategy
cross zero? At what notional? At what slippage assumption?

Concrete output:
- Break-even cost multiplier per mode.
- Notional sweet-spot: the position sizing where cost/R ratio is
  minimised.
- The arithmetic that lets a customer make a sizing decision they
  can defend.

### Dim 8 — Prescriptive Next-Steps

Three concrete improvements the customer could try, ranked by
expected impact, derived from the diagnostic findings.

Concrete output (examples):
- "Your strategy loses 67% of its gross edge to cost on
  intraday-MIS HDFCBANK. Consider routing to options on the same
  underlying, or to a longer-horizon swing version."
- "Your direction filter has positive AUC in trending regimes but
  negative AUC in chop. Adding a path_efficiency gate would have
  cut your chop-day loss rate from 65% to ~40% in OOS replay."
- "Your sample size on FMCG names is N=23 over 18 months — too
  sparse to draw conclusions either way. Either trade them at the
  same rate as your majority sectors or remove them from the
  strategy specification."

This is the section that turns the diagnosis from a report into a
relationship. Customers come back because the prescriptions work.

## 3. Customer ingestion paths

### Tier A — Rule-set diagnosis (cheapest, fastest)

Customer fills a structured form describing their strategy in
plain rules:
- entry condition (e.g. "RSI < 30 on 5-min")
- exit condition
- stop / target / horizon
- instrument basket
- direction

We translate to an executable signal generator. We run it through
the pipeline. We deliver the report.

**Pricing target**: ₹5,000–10,000 one-time. Turnaround: 48 hours.
This is the volume product.

### Tier B — Code diagnosis (more thorough)

Customer sends their strategy code (Python, AmiBroker, MQL, or
similar). We:
1. Reproduce their backtest output as-is in our environment.
2. Run all 8 dimensions.
3. Identify any divergence between their reported numbers and our
   replication (the leak detection in disguise).
4. Deliver report + the reproduction notebook.

**Pricing target**: ₹15,000–35,000 one-time. Turnaround: 5–7 days.

### Tier C — Live trade-log audit (for working systems)

Customer sends their actual live trade log (CSV of fills). We
augment the 8 dimensions with:
- Implementation shortfall: how much of the strategy's intended
  edge is being lost in execution.
- Slippage breakdown by instrument, time-of-day, market state.
- Drift detection: is the live performance diverging from backtest,
  and if so, when did it start?

**Pricing target**: ₹50,000–1,50,000 one-time + ₹15,000-30,000/month
ongoing for quarterly re-audits.

### Tier D — Strategy R&D engagement (institutional)

For prop desks and funds. Customer sends multi-strategy book +
returns + we go deeper:
- Multi-strategy correlation under regime.
- Portfolio-level cost decomposition.
- Strategy-decay analysis: which strategies are in decay, which
  are in regime mismatch, which are dead.

**Pricing target**: ₹2,00,000–10,00,000 per engagement. Project-
based, 4–8 weeks. Limited slots per quarter.

## 4. The deliverable artifact

Same shape across tiers; depth varies.

### Format

PDF report + interactive HTML dashboard + raw JSON for customer's
own analysis. The PDF is what they read; the JSON is what they
keep.

### Length

- Tier A: 8 pages PDF (one per dimension), ~5 min to read.
- Tier B: 15-25 pages PDF.
- Tier C: 25-40 pages PDF + recurring monthly delta reports.
- Tier D: 50+ pages + presentation deck delivered live.

### Structure per dimension

Each of the 8 sections follows the same structure:
1. **One-paragraph plain-prose summary.**
2. **Headline numbers in a table.**
3. **One chart that visualizes the key finding.**
4. **Interpretation: what this means for the strategy.**
5. **Caveats: where the diagnostic might be wrong (e.g. small N,
   regime under-representation).**

## 5. Honesty constraints (the moat)

The same architectural constraints as the daily brief apply:

- **No tipster vocabulary in the report.** "Your strategy works in
  trending regimes" is OK. "Add this filter" — only allowed in the
  Prescriptive Next-Steps section, framed as "consider X" not "do X".
- **The diagnostic shows weaknesses honestly.** A strategy that
  fails 3 of 4 null tests gets that fact reported in plain language.
  Customers reward this honesty; competitors hide it.
- **We diagnose against THEIR strategy, not OUR judgment.** We don't
  tell them their strategy is bad; we tell them what its measured
  characteristics are. They decide.

## 6. SEBI positioning

Strategy Diagnosis is the safest possible product corridor in
Indian financial regulation:
- We do not provide trading signals to the customer.
- We do not manage their capital.
- We do not give buy/sell/target/stop recommendations.
- We provide RESEARCH about the customer's OWN strategy.

This is the same legal category as a backtest software vendor
or a quantitative research consultancy. No RA / IA registration
required for this product corridor.

## 7. Distribution & sales motion

### How customers find us

1. **Daily Brief audit section.** Every customer reading the brief
   sees us audit OUR OWN predictions honestly. The natural follow-up
   conversation: "you audit yourselves — can you audit me?"
2. **Reddit / X organic.** Indian quant Reddit + X is small and
   tightly connected. Sharing redacted diagnosis snippets ("here is
   what we found in an anonymous strategy") drives inbound.
3. **Conference talks / writeups.** One published case study per
   quarter ("we audited this anonymous strategy and found a 67%
   cost leakage") drives 5-10 high-quality inbounds per piece.

### Why customers come back

- Tier A → Tier B conversion: "the rule-set diagnosis was good;
  send us your code".
- Tier B → Tier C conversion: "the code diagnosis matched our
  internal numbers; once we go live, audit our actual fills".
- Tier C → ongoing retainer: "you caught our strategy drift in
  month 3; we want this monthly forever".
- Tier D customers refer other institutions.

### What does NOT scale

- Selling the diagnosis as a SaaS dashboard. The depth that makes
  the diagnosis valuable comes from human interpretation; we don't
  productize that to self-serve.
- One-shot template reports for every retail trader. Tier A is the
  furthest down-market we go; below ₹5,000 the unit economics
  break.

## 8. Revenue model

Realistic 24-month projection (assuming brief subscription business
in parallel):

| Tier | Customers / quarter | Avg revenue | Quarterly revenue |
|---|---|---|---|
| A — Rule-set | 20-40 | ₹7,500 | ₹1.5-3 L |
| B — Code | 8-15 | ₹25,000 | ₹2-4 L |
| C — Live audit | 3-5 ongoing + 2 new | ₹2,00,000 (annualised) | ₹6-10 L |
| D — Institutional | 1-2 / quarter | ₹5,00,000 | ₹5-10 L |

Total Q4 2026 (24 months out): ₹15-27 L per quarter, ~₹60L-1Cr ARR
from Strategy Diagnosis alone — at the same time the brief
subscription is at ₹5-10L MRR.

The two product lines reinforce each other: brief subscribers see
the audit section, become diagnosis customers; diagnosis customers
see the brief's quality, become subscribers.

## 9. Build sequence (post-customer-#1)

1. **Tier A first** (1 week of build). Rule-set form + automated
   pipeline ingest + report generator. Use the bundle's existing
   evaluator + audit + cost + null infrastructure. Reuse the daily
   brief's renderer module shape.
2. **Tier B next** (2-3 weeks). Adds code-ingestion + replay-vs-
   customer-report comparison. Most of the pipeline is reused.
3. **Tier C** (4-6 weeks). Adds the live-trade-log ingestion +
   slippage decomposition + drift detection. Requires building a
   per-customer continuous-monitoring layer.
4. **Tier D** (when first institutional customer signs). Bespoke
   build per engagement; no pre-built code.

## 10. The single decision this product hinges on

Strategy diagnosis only works if customers TRUST the diagnostic. They
trust it because:
- Our own daily brief audits OUR predictions honestly (visible proof
  of method).
- The outcome log is public to subscribers (track record).
- The diagnostic flags weaknesses in our own work (we don't hide
  failures — see the 2026-06-04 post-touch death documented in
  RESEARCH_ROADMAP).

This is why the brief must ship first. The brief's audit section IS
the marketing for the diagnosis product. Without 60+ days of audited
brief history, the diagnosis pitch is "trust us"; with it, the pitch
is "look at our work."

The brief and the diagnosis are not two products. They are one
product strategy with two delivery surfaces.
