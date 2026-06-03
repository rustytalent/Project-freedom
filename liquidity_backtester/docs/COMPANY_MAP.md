# Company Map

This is the single source of truth for what this project is. It is the map
every agent reasons against. Update only when an architectural decision
genuinely changes; everything tactical lives in `COORDINATION.md`.

## What this is

A market-intelligence operating system for NSE Indian equities and (soon)
index options. **One engine answering one question at multiple horizons.**

The engine: a calibrated probability that price will reach a level L within
a time horizon T, conditioned on rich market context.

The question changes only its (L, T) parameters:
- **Swing** — large T (days–weeks), L = structural pools.
- **Intraday MIS** — small T (minutes–hours), L = same-session pools.
- **Index options** — T = expiry, L = the strike grid.

This framing is load-bearing. We do not build three separate strategy
systems. We build one engine and three presentation layers.

## Two axes

The project is organised along two orthogonal axes. Almost all confusion
in past sessions came from collapsing them into one.

### Axis 1 — The Stack (necessity chain, bottom to top)

| Layer | What it is | Status |
|-------|------------|--------|
| L0  Data | Raw OHLCV bars, MIS-honest timestamps, multi-asset NSE warehouse | Built |
| L1  Features | The state featurizer — 42 features across structure, MTF, AVWAP/FRVP, expiry, path/context | Built |
| L2  Models | Q (SectorMoE), direction, **proximity (3 horizons)**, reaction, policy R | Built; proximity is the strongest (AUC ~0.93) |
| L3  Signals / Alpha | The arsenal — combines L2 outputs into trade candidates. Multiple alpha classes, soft-scored ensemble for journey alpha. | Built; gross edge near break-even, costs dominate |
| L4  Execution & Validation | Backtest + forward test + MIS-aware simulator + cost model + min-economic-position filter | Built |
| L5  Audit | Null tests (time-shuffle, sign-flip, pool-level, moment), calibration audits, cost-stress decomposition | Built; under-surfaced in reports |

**Infrastructure** is the spine that holds L0–L5 upright. It is born of
necessity (training pipeline, parallel asset workers, bundle persistence,
report generation, sweep runners). It is the second most monetizable layer
because other quants face the same necessity.

### Axis 2 — Horizon / Instrument (the same engine, different (L, T))

| Horizon | Question | Customer | Status |
|---------|----------|----------|--------|
| Swing (days–weeks) | "Will price reach L by end-of-week?" | Positional + option-buy-and-hold | Engine ready; presentation layer not built |
| Intraday MIS (minutes–hours) | "Will price reach L same-session?" | Day-traders + intraday algo | Engine + MIS labels + ProximityFilteredPoolAlpha built |
| Index options (T = expiry, L = strike) | "Will Nifty/BankNifty test strike K before expiry?" | Our actual customer pool today | Engine ready; presentation layer not built — **this is the next product** |

## Commercial layer — three revenue lines, sequenced

1. **Research / context subscription** (sells FIRST — lowest bar, lowest SEBI risk).
   Daily research brief + post-market audit, hand-priced, hand-picked
   customers. Product is judgment-augmentation, not trade calls.

2. **Infrastructure as a service** (sells SECOND — highest margin).
   Strategy diagnosis ("X-ray your strategy") sold as a one-time deliverable
   AND as the sales motion that converts customers into the research
   subscription. Eventually: backtest-as-a-service + custom-alpha research.

3. **Private alpha** (sells LAST — highest bar, slowest path).
   Self-trading. Requires gross edge to clear realistic costs. Not the
   thing that pays rent first. Will improve as the data flywheel improves.

These are not three teams or three timelines. They are one funnel.
Diagnosis → subscription → flywheel data → better engine → maybe later
the private alpha clears costs.

## The data flywheel (the moat)

The pool-detection algorithm is reverse-engineerable. The model
architecture is reverse-engineerable. The thing that is NOT reverse-
engineerable is the **outcome log** of every prediction we ever made
crossed with every actual market resolution. That dataset compounds
weekly, becomes the marketing (honest track record), the R&D signal
(what features actually predict), and the retention mechanism (audit
layer customers can verify).

Implication: **outcome logging must exist BEFORE customer #1 reads
brief #1**, or the most valuable early data is lost forever.

## The two layers of every product output

Every research artifact ships in two synchronised forms:

- **Human layer** — clean prose summary, 5–7 minute read, no jargon.
  Decision context, not trade instructions.
- **Machine layer** — JSON / CSV with probabilities, scores, feature
  tags, regime tags. For the quant / engineer customer who wants the
  raw shape to combine with their own system.

Both come from the same generation pipeline. Both can be priced
separately or bundled.

## The product hierarchy (what is exposed at what tier)

- **Private Core** (never exposed) — raw model code, feature weights,
  pocket-selection logic, internal execution rules.
- **Paid Research Layer** — regime, probabilities, rankings, risk tags,
  zones, confidence buckets, post-trade audit.
- **Premium Analytics Layer** (higher tier) — custom backtests, strategy
  diagnostics, personalised watchlists, API access.
- **Public Marketing Layer** — philosophy, anonymised sample reports,
  educational explainers. Never direct trade calls.

## What we DO NOT do

- No "buy X at Y stop Z" outputs. Ever. Both legal (SEBI) and product.
- No mobile app, no public community, no copy-trading, no signal room.
- No new alpha modules until the daily brief deliverable is live and
  paying customers exist.
- No simultaneous building of all three horizon products. Pick one
  (options first, given the customer pool), ship, then expand.

## Update protocol for this document

Edit this file only when an architectural fact changes. Tactical state
(in-flight work, queue, decisions of the week) goes in
`COORDINATION.md`. If you find yourself wanting to put a date or a
sprint into this file, you are editing the wrong document.
