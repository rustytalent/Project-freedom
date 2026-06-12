# Equity Strategy & Methodology — Active Commercial Vertical

> Status: active commercial vertical as of June 2026.
>
> This document mirrors the options-vertical structure, but uses the data
> we actually own today: liquid NSE equities, index spot context, the
> trained proximity/direction/reaction/R1 stack, Daily Brief artifacts,
> and the outcome log.
>
> Options remain a future vertical until historical option intraday OHLCV
> bars exist. Greeks-only data is not enough to train the options
> executor honestly.

## 0. Why This Exists

The options vertical was designed because index options improve the
trade-economics of a working proximity signal. That logic is still
correct. The blocker is data, not theory: we currently do not have
historical intraday option OHLCV bars across old expiries, so the
options Gate-1 model cannot label realized premium paths.

Equity is therefore the active vertical. The goal is not to resurrect
cash-equity MIS self-trading as a private-alpha machine. Multiple
realistic execution tests already showed that small same-session equity
moves do not clear Indian intraday costs. The goal is to package the
equity engine as a research product:

- intraday context for active traders,
- swing and multi-day context for lower-friction customers,
- machine-readable zones and probabilities for infrastructure clients,
- outcome logs that turn every brief into a compounding calibration
  dataset.

This is the same vertical discipline as options: prediction layer,
decision layer, audit layer, delivery layer, and commercial gates.

## 1. Hard Claims This Vertical Commits To

These claims are product claims, not tipster claims.

1. **The equity engine ranks market state better than naive baselines.**
   Proximity, direction, reaction, and R1 must be reported as ranking
   tools with OOS metrics. We do not hide weak heads.

2. **The brief is valuable even when no setup qualifies.** Avoidance
   days, low-touch days, sector regime shifts, and post-touch
   confirmation context are valid paid research outputs.

3. **Cash-equity same-session execution is research-only until it clears
   costs.** No user-facing copy can imply that current MIS equity
   signals are deployable as trade calls.

4. **Swing/delivery is the primary equity monetization path.** The same
   proximity engine should be extended to 5d/10d/20d horizons where
   larger moves and lower delivery friction can make the economics more
   realistic.

5. **Every published prediction is logged.** The outcome log is the
   moat: it records what was said, when it was said, what actually
   happened, and whether the prediction was live or retrospective.

## 2. Instrument Universe

V1 uses the 25-stock core equity universe:

- BANKING: HDFCBANK, ICICIBANK, SBIN, AXISBANK, KOTAKBANK
- IT: TCS, INFY, HCLTECH, WIPRO, TECHM
- FMCG: HINDUNILVR, ITC, NESTLEIND, BRITANNIA, DABUR
- AUTO: MARUTI, M&M, BAJAJ-AUTO, EICHERMOT, TVSMOTOR
- PHARMA: SUNPHARMA, DRREDDY, CIPLA, DIVISLAB, LUPIN

Index spot context should be used for market regime and customer
language, but the v1 equity artifact remains stock-first.

## 3. Data We Use

### L0 — owned today

- 1-minute equity OHLCV for the core universe.
- Resampled 5-minute, 15-minute, 60-minute, 180-minute, daily, weekly
  equity bars.
- Index spot OHLCV for broad regime context where available.
- Sector mapping and sector rotation metrics.
- Daily Brief artifacts and outcome-log partitions.

### L1 — features

The equity vertical uses the full state featurizer:

- base returns, volatility, ATR, z-scores, pull metrics,
- structural pool features,
- multi-timeframe session context,
- AVWAP/FRVP features,
- expiry-cycle context,
- path/shape context,
- sweep, stop-run-reclaim, imbalance, premium/discount, volume-weighted
  swing, and cumulative-delta-proxy detectors when present in the
  trained bundle.

## 4. Model Layer

The equity vertical exposes four model families.

### Q / quality

Purpose: context and ranking only.

If Q validation AUC is weak, it must be demoted. Q should not be a hard
gate unless a current bundle proves calibration and lift.

### Direction

Purpose: directional context, not a standalone signal.

The direction head helps explain whether a level test is aligned or
fighting the current move. It can support watchlist prioritization and
avoidance notes.

### Proximity

Purpose: core engine.

The proximity heads answer:

```
Will price test level L within horizon T?
```

Intraday horizons remain 12/36/60 bars. Swing horizons should be added
as separate heads at 5d/10d/20d. Intraday and swing calibration must not
be pooled.

### Reaction / R1

Purpose:

- reaction: post-touch confirmation context,
- R1: ranking candidate execution policies and diagnosing whether the
  best decile is still negative after costs.

R1 is valuable even when it says "do not trade": it explains that the
engine can rank opportunities but the execution economics still fail.

## 5. Decision Layer

The equity vertical has three product modes.

### Equity Intraday Research

Question:

```
Which stocks have meaningful same-session probability of testing an
important structural level, and which contexts should be avoided?
```

Output:

- watchlist,
- avoid list,
- key zones,
- sector regime,
- model-health notes,
- post-touch confirmation alerts,
- no-trade/stand-aside days.

No trade instructions.

### Equity Swing Research

Question:

```
Which stocks have meaningful multi-day probability of testing a
structural level, with delivery-style cost assumptions?
```

Output:

- 5d/10d/20d proximity ranks,
- expected hold-period bucket,
- delivery breakeven move,
- sector and index regime context,
- weekly audit.

This is the primary equity path for paid customers because the economics
are less hostile than MIS.

### Equity Infrastructure Feed

Question:

```
What non-invertible, machine-readable market-state features can a
customer plug into their own research stack?
```

Output:

- opaque zone bands,
- feature intensity,
- regime tags,
- confidence buckets,
- customer watermark,
- manifest and hashes.

Exact private model internals are never exposed.

## 6. Executor Layer

Equity execution is split by use case.

### Intraday MIS

Status: research-only.

It can be replayed for diagnosis, but cannot be sold as a signal until:

- at least one policy has positive OOS top-decile net R,
- it clears 1.5x cost stress,
- it beats same-slice nulls,
- it survives live paper outcome logging.

### Delivery / Swing

Status: next build target.

Required execution assumptions:

- no intraday auto-square-off,
- realistic delivery charges,
- overnight gap handling,
- max holding window,
- larger rupee floors,
- per-symbol liquidity and turnover caps.

## 7. Audit Layer

Every equity output must report:

- OOS model metrics,
- calibration by confidence bucket,
- cost stress for execution claims,
- null-test status for any alpha claim,
- retrospective/live status for outcome-log rows,
- exact number of predictions made and resolved.

The product can say "model confidence is low today". That honesty is a
feature, not a defect.

## 8. Shipping Layer

The equity vertical ships in two synchronized forms.

### Human layer

Daily Research Brief:

- 5-7 minute read,
- no tipster vocabulary,
- no exact public/demo levels,
- exact paid/licensed levels only,
- avoidance and uncertainty clearly surfaced.

### Machine layer

Customer delivery pack:

- daily_brief.json,
- daily_brief.txt,
- consolidated_summary.json,
- consolidated_models.json,
- consolidated_backtest.json,
- outcome-log summary,
- manifest with hashes,
- optional zip.

The public/demo feed must stay abstracted and watermarked.

## 9. Acceptance Gates

### Gate 1 — current equity brief is production-ready

Pass when:

- predict run produces daily_brief.json and daily_brief.txt,
- artifact packer emits a customer-delivery pack,
- no exact levels leak to public/demo/sample tiers,
- outcome-log predictions are written,
- tipster vocabulary guardrail passes.

### Gate 2 — swing equity model proves useful

Pass when:

- 5d/10d/20d swing proximity heads train,
- each horizon reports n_train, n_oos, AUC, calibration,
- at least one horizon has OOS AUC > 0.55 and usable top-bucket lift,
- swing brief renders with at least one non-trivial historical watchlist
  entry.

### Gate 3 — customer delivery loop closes

Pass when:

- website artifact ingest receives the equity brief pack,
- Google auth + paid entitlement can read the paid artifact,
- demo/free preview receives abstracted levels only,
- email/webhook delivery path is verified.

### Gate 4 — live calibration

Pass after 20-30 trading sessions of live logged predictions:

- calibration error remains within declared tolerance,
- no schema or artifact delivery failures,
- no methodology leaks in customer-facing artifacts.

## 10. Pricing Implication

Because equity is the active vertical, pricing should be framed as:

- **Equity Research**: core daily brief + outcome audit.
- **Equity + Index Context**: equity brief with index regime and weekly
  swing context.
- **Professional / Infrastructure**: exact paid levels, machine layer,
  custom watchlists, delivery packs, and strategy diagnosis.

Options can be listed as "coming after historical options data
coverage", not sold as a current model-backed product.

## 11. What Not To Do

- Do not claim cash-equity MIS is deployable while current cost-stress
  metrics remain negative.
- Do not re-run options Gate-1 until historical option OHLCV bars exist.
- Do not expose feature weights, model internals, or reverse-engineering
  details in customer artifacts.
- Do not add trade-call language to the brief.
- Do not treat a no-trade day as a failed product day.

## 12. Immediate Implementation Queue

1. Make the website product language equity-first.
2. Run the latest equity predict job from the latest trained bundle.
3. Pack a customer-delivery artifact.
4. Push/import the artifact into the website.
5. Backfill/continue the outcome log.
6. Build swing-horizon equity heads and swing brief after the daily brief
   delivery loop is stable.
