# The Profitability Doctrine

**The pivot, written down so it can't drift.**
**Date: 2026-06-15**
**Author: Opus 4.7 / Claude Code session, in partnership with the founder**

> *"We are no longer building a SaaS product. We are building a quant
> system whose only customer is the founder. Profitability is the only
> metric. Distribution, branding, multi-tenant, billing — all
> deprioritised. The cockpit is OUR cockpit. The research engine is
> OUR research engine. Nothing else matters."*

This document exists so a year from now you can read it and check
whether we kept the discipline.

---

## 1. The brutal truth about where we are

**We have ZERO proven edges.** Let me say this clearly so it isn't lost
in the celebration of 1,290 passing tests:

- The four Sentinel "scientists" (SecondPullback, GammaScalp, ChopMeanRev,
  ThetaGuard) are heuristic generators. They have **never been
  backtested**. We do not know their hit rate, expectancy, or
  drawdown.
- The six live models (reaction, proximity, liquidity, quality,
  post-reaction, manipulation) are heuristic transforms of recent
  ticks. They have **never been validated as predictive**.
- The liqpool research engine's three head suite (direction, proximity,
  quality) has training infrastructure but the **trained model
  bundles in this repo are demo fixtures**. No real OOS Sharpe is on
  file.
- The Crux meta-signal composes signals that themselves are unproven.
  Composing noise gets you composed noise.

**What we DO have** is *infrastructure that can find edges, run them,
risk-manage them, and audit them.* That's a lot. It's also not the
same as having edges.

We have been building a beautiful kitchen and assuming the kitchen
implies food.

## 2. What it actually takes to make money trading options in India

Strip the romance away. Profitability is a finite chain. Break any
link and you lose money no matter how good the other links are.

```
EDGE → COSTS → SIZING → RISK → DISCIPLINE → COMPOUND
```

| Link | What it means | What we have | What we need |
|---|---|---|---|
| **Edge** | A signal with statistical advantage after costs | Nothing proven | A library of validated hypotheses with OOS Sharpe ≥ 1.0 |
| **Costs** | STT + GST + brokerage + slippage + IV mispricing | Math is in `india_tax.py` | Realistic slippage model per liquidity bucket |
| **Sizing** | How much to bet on each edge | Nothing | Kelly-fractional sizer per signal + per portfolio |
| **Risk** | Daily / weekly / monthly loss caps | Profit lock + intention contract | Volatility-targeted position sizing, drawdown gates |
| **Discipline** | Following the system when emotional | Behavioral engine (tilt, biases, intention) | This is actually our strongest link |
| **Compound** | Letting wins ride into more wins | Nothing | Bankroll growth function, weekly reinvestment rule |

We have built link 5 (discipline) magnificently. Links 1, 3, 6 don't
exist yet. Without those, the rest doesn't print.

## 3. The 10 hypotheses worth testing (in priority order)

A real quant operation tests many hypotheses, keeps the few that
work, kills the rest. Here's our hypothesis backlog, ranked by my
honest expected-value estimate (intuition, not math — that's exactly
what the testing harness is for):

| # | Hypothesis | Why I believe it might work | What kills it |
|---|---|---|---|
| 1 | **Expiry-Thursday theta scalping**: short ATM straddle on Thursday 9:30 IST, close at 14:30 | Theta is non-linear in last day; intraday vol on expiry day is usually below historical realized | Gap moves, news, BankNIFTY-shock days |
| 2 | **Opening-range failure fade**: on +1σ-gap days, fade the open after first 30 min if range fails to extend | Mean-reversion documented in Indian retail flow; option premiums priced for continuation | Trending news-driven days |
| 3 | **BankNIFTY-vs-NIFTY relative strength rotation**: when ratio crosses 2-day MA, trade pair | Two correlated indices with asymmetric liquidity — convergence trade with options | When both indices break regime together |
| 4 | **VIX-percentile premium selling**: short premium when India VIX > 80th percentile of trailing 60 days | Vol risk premium documented (Bakshi-Kapadia 2003) | Vol-of-vol spikes; black swans |
| 5 | **Time-of-day expectancy filter**: certain 30-min windows have positive expectancy others don't | Microstructure: 9:15-9:45 + 14:45-15:15 carry information; lunch fade is noise | Calendar effects (RBI, expiry, results) override |
| 6 | **Heavyweight divergence trade**: when our constituent board flags MANIPULATED, fade the index | Already coded; just need backtest | If 1-2 stocks dominate, momentum trumps mean-reversion |
| 7 | **Post-event vol crush**: short ATM straddle right after a scheduled event | IV pumps before, deflates after; well-known | Underlying actually moves |
| 8 | **Gap-fill probability per gap size**: trade gap-fills when historical gap-fill probability > 60% | Empirical pattern; conditional probability | When the gap is news-driven and continues |
| 9 | **Liquidity-sweep-and-reclaim entry**: enter after the LiquidityModel fires + spot reclaims | Already coded; need backtest | Microstructure noise mimics the pattern |
| 10 | **Bank-of-Maharashtra pattern**: small/mid-cap NIFTY weights mean-reverting after sector ETF moves | Inefficiency in retail flow | Capacity is small; alpha may not scale |

**Decision rule**: test each in order; keep ones with OOS Sharpe > 1.0
post-cost on 18+ months of data; kill the rest. **Expected outcome:
2-3 of 10 will pass.** That's roughly the base rate at every prop
desk I've ever read about. If 0 pass, we have a deeper problem (data
quality, label design, regime change) and we stop and figure that out
before trying more.

## 4. The build sequence — concrete next 4 weeks

### Week 1 — Build the hypothesis testing harness
Code (in `liquidity_backtester/liqpool/research/`):
- `HypothesisSpec` dataclass: entry rule, exit rule, position type, symbol, date range
- `HypothesisHarness.run(spec)`: walks the warehouse, simulates with `execution_simulator_v2`, applies real costs from `sentinel.india_tax`, returns `HypothesisReport`
- `HypothesisReport`: Sharpe, Sortino, max_dd, hit_rate, expectancy_R, avg_hold_min, total_trades, by-regime breakdown, by-month breakdown
- `HypothesisLibrary`: a registry of the 10 hypotheses above, each as a callable

### Week 2 — Run all 10 hypotheses
- Run on every NIFTY weekly expiry in the warehouse
- Score, rank, save reports
- Pick top 3 candidates

### Week 3 — Paper-trade the top 3 candidates
- Use our existing `sentinel.paper.PaperAccount`
- Let them run live for 1 week
- Compare live results to backtest expectations
- Confidence interval check: live trades within ±2σ of backtest distribution?

### Week 4 — Sizing + risk + go live with #1
- Build `liqpool/research/kelly.py`: position size = (edge × win_prob - loss_prob) / loss_size, capped at 0.25 of full Kelly
- Tie into `sentinel.orchestration.Orchestrator`: graduate the winning hypothesis source to TRUSTED, then to EXECUTION (one signal at a time)
- Start with **1% of bankroll per trade**. Tiny. Boring. Survivable.
- Daily P&L attribution journal: which hypothesis made/lost what

## 5. What to STOP doing (deprecation list)

These were good engineering. They aren't profit-critical. They go in
the freezer:

| Stop building | Why | Status |
|---|---|---|
| Customer console (`/console`) | We aren't selling to anyone | Keep code, remove from priority |
| BYOK multi-tenant | One user (you) | Keep code, irrelevant |
| Stripe webhook | No customers | Keep code, irrelevant |
| Notification system | One user, you'll see the cockpit | Keep code, no urgency |
| Mobile responsive | You trade on desktop | Done, no further work |
| Auditor/Builder/Stress as customer surfaces | They're useful to YOU as analysis tools — keep but don't polish | Reposition as personal tools |
| Help text expansion | You wrote it; you know what it does | Frozen |
| Rate limiting | Solo use | Frozen |
| Health endpoints | Useful, but not profit-critical | Frozen at current state |

Approximate code we stop touching: ~30% of the codebase. That's
not a sign of waste — it's a sign of focus.

## 6. What to KEEP working on

Profit-critical surfaces. Everything else is decoration:

| Keep building | Why |
|---|---|
| `liquidity_backtester` research engine | This is where edges are found |
| The full StateFeaturizer wire-up to live mode | Slim feature builder limits prediction quality |
| Sentinel's `trust spine` + orchestrator | This is the safety wall that lets us auto-execute |
| `paper.PaperAccount` | Validation step before real money |
| The shadow ledger | Decision audit / debugging |
| Behavioral engine | Prevents YOU from killing the system in tilt |
| Crux composer (rewired) | One verdict aggregating proven signals |
| Curator + calibration | Decay detection on live signals |
| `india_tax.py` cost math | Edge AFTER costs is the only real edge |

## 7. The hardest acknowledgement

You asked, sincerely, *"why are we still failing so bad at execution
that it's not becoming profitable?"*

The honest answer at this stage of the pivot:
**We haven't yet asked the market a single statistically rigorous
question.** We've built tools. We haven't sent the tools after a
specific quarry.

Profitability is downstream of:
1. Asking the right question (hypothesis design)
2. Letting the data answer (backtest)
3. Believing the answer even when it says no (discipline)
4. Executing the survivors (the trust spine)
5. Killing the decayed (curator)

We've never done #1 or #2 honestly. Until we do, expecting #4 to
print is hoping.

This isn't a failure of execution. It's a sequencing problem. We
built link 5 (discipline) before link 1 (edge). Discipline without
edge just means you lose money more slowly. The pivot is: stop
adding links 5-7. Build link 1 first. **The harness is link 1.**

## 8. The promise to you

Mom (since you called me that): I will tell you the truth even when
it's hard. I will warn you when your enthusiasm is running past the
evidence. I will push you to kill features you love when they don't
print. I will commit code even when it scares me.

In exchange: when the harness says "this hypothesis is noise," you
kill it. When it says "this works at 1.2 Sharpe," you size at 1%
not 10%. When you want to skip the paper-trade step, you don't.

**That discipline IS the moat.** Anyone can copy the code. Nobody
can copy *patient empirical testing*. That's our edge before any
specific hypothesis is.

---

## 9. The first concrete action — today, this session

Build the `HypothesisHarness`. Not in a week. Today. Even a v0 that
takes a hypothesis function + bar series + cost function and returns
{Sharpe, max_dd, expectancy} is enough to start.

Then over the next session pick hypothesis #1 from the list above
(Thursday theta scalp), code it, run it. We'll have an honest
answer to "does any edge exist for us" within the week.

Everything else waits.
