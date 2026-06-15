# Creative Profitability Playbook

**Read order**: After `PROFITABILITY_DOCTRINE.md` (the why) and
`CODEBASE_TRUTH_MAP.md` (the what). This is the *how to be creative*.

**For**: future Claude sessions, Codex, the founder, or any
collaborator picking this up cold. Everything here is meant to be
**actionable** — every idea has a one-paragraph implementation sketch
so it can be coded by whoever has the context window next.

**Date**: 2026-06-15

---

## The single insight that reframes everything

> *Hypotheses guessed from domain intuition are finite. Hypotheses
> SEARCHED from data are infinite.*

The standard quant playbook (which is what `liqpool/research/library.py`
implements today) is:

```
domain expert → hypothesis → backtest → keep / kill
```

That pipeline has a hard ceiling: the number of hypotheses a human can
imagine. For NIFTY options that's maybe ~50 ideas. Most retail
literature has already explored them.

The data-driven alternative is:

```
data → hypothesis GENERATOR → backtest at scale → filter survivors
       → meta-search for what made them survive → generate next batch
```

This is **hypothesis SEARCH** (a.k.a. systematic alpha mining,
symbolic regression, genetic programming, AutoML for finance). The
search space is combinatorial, not enumerated. You don't run out.

Both approaches need our `HypothesisHarness` as the validator. The
difference is upstream: are we hand-writing recipes, or generating
them?

**This playbook is biased toward search, because intuition runs out
fast.**

---

## The 30 creative angles — each with an implementation sketch

These are organised by class of edge, then by implementation
difficulty. The marker on each entry:

- 🟢 = a few hours of code from where we are now
- 🟡 = one focused day
- 🔴 = a week+ but the payoff potential is highest
- 💎 = uniquely possible because of OUR codebase (the trust spine, the
       behavioral engine, the cross-codebase ledger)

### A. DATA-DRIVEN HYPOTHESIS MINING (the path beyond intuition)

#### 1. 🟢 Random hypothesis generator over a feature × rule grammar

Define a small grammar of primitives:
- features: `ret_1`, `ret_6`, `ret_24`, `atr_proxy`, `zscore_close_50`,
  `session_minutes_in`, `streak`, `dow`, plus any of our 30+ liqpool
  features
- comparison ops: `>`, `<`, `>= ATR`, `<= 0`, `crosses_above`,
  `crosses_below`, `within_N_bars_of`
- entry composition: `AND` of 1-3 conditions
- exit composition: time stop `N bars`, profit target `K ATR`, hard
  stop `−K ATR`, opposite signal
- side: long / short / both

Sample 10,000 random hypotheses. Run through `HypothesisHarness`. Sort
by Sharpe. Look at the top 100. Three of them WILL look like patterns
you wouldn't have guessed. That's the gold.

```python
# liqpool/research/hypothesis_miner.py  (sketch)
def sample_random_hypothesis(seed: int) -> HypothesisSpec:
    rng = random.Random(seed)
    n_conds = rng.choice([1, 2, 3])
    conditions = [rng.choice(PRIMITIVES) for _ in range(n_conds)]
    return HypothesisSpec(
        name=f"mined_{seed}",
        fn=compile_conditions(conditions),
        params={"hold_bars": rng.choice([6, 12, 25, 78])},
    )
```

#### 2. 🟡 Genetic programming over the same grammar

Same primitives, but instead of random sampling, run mutation + crossover
on the top survivors. Sklearn has DEAP; PyGAD also works. After 50
generations of 100 individuals, the fittest survivors are the candidates
you couldn't have guessed.

**Warning**: this overfits massively if you don't walk-forward CV the
fitness. Our `walkforward._fold_windows` is exactly the discipline. Use
the OOS Sharpe of each fold, average across folds, fitness = that
average.

#### 3. 🟡 Bayesian optimization over hypothesis hyperparameters

For each existing hypothesis (H1, H2, ..., or any mined one), the
parameters (`enter_minute_of_day`, `hold_minutes`, `or_minutes`,
`fail_threshold_pct`) form a search space. Use scikit-optimize or
Optuna to find the joint maximum.

**Catch**: this overfits the params to in-sample. Mitigation: walk-
forward optimization — optimize on fold 1, test on fold 2; optimize on
fold 1+2, test on fold 3; etc. Report only the *out-of-sample* P&L
trajectory.

#### 4. 🔴💎 Conditional-on-regime hypothesis search

Our `liqpool.regime` already classifies bars into ADX / vol_ratio /
session buckets. Run the hypothesis miner separately within each
regime. Some hypotheses will have alpha only in "trending + low vol"
days. The miner discovers these conditional edges directly.

Then live: the cockpit's existing regime classifier determines which
mined hypothesis is allowed to fire today. **This is how prop desks
actually do it.** Statistical learning per regime, not blanket.

#### 5. 🔴 Symbolic regression for option-premium dynamics

We have option premium history per contract (`premium_tracker.py`).
Use `pysr` (Cranmer's symbolic regression library) to discover closed-
form relationships between premium velocity and spot motion. The
formulas it spits out may be *new* statistical relationships nobody's
published. Even noise survives some of the time and that's worth
seeing.

---

### B. THE COST MODEL IS PROBABLY WRONG — CALIBRATE IT

#### 6. 🟢 Reverse-engineer real costs from Zerodha trade confirmations

Every Kite order returns a `charges` block: brokerage, STT, exchange,
SEBI, GST, stamp. After ~20 round-trips we can fit a linear regression
of `actual_total_charges ~ buy_premium + sell_premium + qty + side`
and compare coefficients to our `CostsModel`.

Concretely, build a `CostCalibrator` module:

```python
# liqpool/research/cost_calibration.py  (sketch)
def fit_from_confirmations(trades: list[dict]) -> CostsModel:
    """Each trade dict has buy_premium, sell_premium, qty, side, and
    the actual charges dict from Kite. Returns a CostsModel whose
    coefficients are FIT to reality, not assumed."""
```

#### 7. 🟢 Slippage profile per liquidity bucket

Currently we assume 2 ticks of slippage. Reality: it depends on
contract liquidity. After a few weeks of paper trades on real chains,
group fills by (volume bucket × OI bucket × spread_pct) and fit
slippage_ticks per bucket.

#### 8. 🟢 Stress-test the cost model with adversarial parameters

Run every surviving hypothesis under +1 tick, +2 tick, +3 tick
slippage. The ones that still survive at +3 ticks are robust to cost-
model uncertainty. The ones that die at +1 are paper-only.

#### 9. 🟡 Build a cost-aware position sizer

`size = (expected_edge × prob_win) / (cost_per_trade + risk_per_unit)`.

Some hypotheses with thin edges only work at large size; others only
at small size. The sizer makes the trade-off explicit per hypothesis.

---

### C. MICROSTRUCTURE & ORDER FLOW (when we get tick data)

#### 10. 🟡💎 Trade imbalance signals

KiteTicker `MODE_FULL` gives us buy/sell quantities per tick. A
running buy-vs-sell imbalance is one of the most-cited
microstructure features in literature (Kyle 1985, Lehmann 1992).

```python
# liqpool/live_models.py — add OrderFlowImbalance scientist
def tick(self, recent_ticks):
    buys = sum(t["buy_qty"] for t in recent_ticks[-60:])
    sells = sum(t["sell_qty"] for t in recent_ticks[-60:])
    if buys + sells < 100: return None
    imbalance = (buys - sells) / (buys + sells)
    if abs(imbalance) < 0.4: return None
    # fire signal
```

#### 11. 🟡 Bid-ask spread mean-reversion

Spread widens during volatility. After it widens by N sigmas, it
mean-reverts. Trading the spread itself (selling the wider leg,
buying the tighter) is a thing some shops do.

#### 12. 🔴 Predicting next-tick direction from order book depth

Train a tiny classifier (XGBoost is fine) on (book imbalance,
ask-pressure, bid-pressure, recent trades) → direction of next-tick
move. If AUC > 0.55 we have an entry timing signal that wraps any
strategy.

---

### D. VOL SURFACE EDGES (we have institutional infrastructure already)

#### 13. 🟢💎 SVI fair-value vs market premium scan

`sentinel/institutional.py` already fits SVI. Every morning, fit it
to today's chain. For each strike, compare market premium to SVI fair
price. Strikes >5% above fair are SELL candidates; <5% below are BUY.

This is the most directly actionable edge in our codebase that's
unexploited. Code is mostly written.

#### 14. 🟡 Calendar spread mispricing

When near-month and far-month IVs diverge by more than 3σ of their
trailing distribution, the spread reverts. Classic statarb on the
term structure.

#### 15. 🔴 Skew rotation trade

Risk reversal (`OTM call IV − OTM put IV`) rotates between bullish
and bearish regimes. Our `svi_skew_25d()` already computes it.
Trading the rotation: when RR crosses its 20-day moving average,
flip the directional bias.

#### 16. 🟡 Vol risk premium harvesting

Sell ATM straddles when 30-day IV / 30-day realized vol > 1.1 (vol is
cheap to sell). Our `vol_risk_premium()` computes this. Backtest with
weekly rebalancing.

---

### E. CROSS-ASSET ARBITRAGE

#### 17. 🟢 SGX NIFTY vs spot NIFTY mean reversion

SGX NIFTY leads (8 hours ahead). When SGX is up >0.5% after-hours,
spot opens at premium. The discount typically closes in first 30 min.
Fade openings that gap > predicted.

#### 18. 🟡💎 NIFTY-BANKNIFTY pair statistical arbitrage

Compute rolling 60-min beta. When ratio deviates >2σ from its mean,
trade the convergence. Options-leg implementation: long NIFTY
straddle + short BANKNIFTY straddle (delta-neutral pair).

#### 19. 🟡 USD/INR vs IT stock basket

Strong inverse correlation (rupee weakness → IT export earnings up).
When USD/INR moves 0.5% but IT basket hasn't, the lag is tradeable.

#### 20. 🔴 Cross-listed ADR arbitrage

INFY / WIT / HDB on NYSE close at 1:30 IST. Their close vs NSE close
should track. Gap > 1% is an arbitrage trade (with currency hedge).

---

### F. BEHAVIORAL EDGES (retail flow patterns) 💎

#### 21. 🟢 Round-number strike attraction

Retail buys 25000, 25100, 25200 (round 100s). 25050 is underpriced
in implied terms because nobody buys it. Calendar-spread the
mispricing.

#### 22. 🟢 Friday-afternoon FOMO

Retail loads OTM CEs Fri afternoon hoping for Monday gap-up. On
average those CEs decay over the weekend. Sell them.

#### 23. 🟡 Monday-morning panic put

Retail panics on Sunday-night news, buys OTM PEs Monday morning.
Those PEs decay if Monday is normal. Sell-side calendar.

#### 24. 🟡 Lunch fade

Indian retail eats 12:30-13:30; volume drops; mean reversion in spot.
Statistical tested in academic literature.

#### 25. 🔴 Telegram tipster sentiment

Scrape (legally) the top 50 NIFTY Telegram channels. Count mentions
of "buy CE" vs "buy PE" today. Fade the consensus. (Retail is the
predictable counterparty.)

---

### G. ENSEMBLE & PORTFOLIO-LEVEL EDGES

#### 26. 🟡 Multi-armed bandit allocation across hypotheses

Once 3+ hypotheses survive, use Thompson sampling to allocate capital
each day proportional to each hypothesis's recent Sharpe. Concretely:

```python
# liqpool/research/allocator.py  (sketch)
def thompson_allocate(reports: list[HypothesisReport],
                       capital: float) -> dict[str, float]:
    """Sample from each hypothesis's posterior Sharpe distribution,
    allocate capital proportional to sampled Sharpes. Explore +
    exploit naturally."""
```

#### 27. 🟡 Walk-forward ensemble

Train a meta-classifier: "should I trust hypothesis H1 today?" using
features: recent Sharpe, current regime, days since last winner.
Trade only when the meta says yes.

#### 28. 🔴💎 Decay-aware kill-switch per hypothesis

Track per-hypothesis rolling 30-day Sharpe. When it drops below 0.5,
stop trading that hypothesis automatically. The `curator.py` already
has the bones for this. Wire it.

---

### H. THE WEIRD ONES (low probability, high payoff)

#### 29. 🔴 Synthetic-data validation

Generate fake bar series with KNOWN properties (mean-reverting,
trending, random walk, regime-switching). Run hypotheses on them. If
a hypothesis "works" on random-walk synthetic data, it's overfit.
This is one of the strongest tools against fooling yourself.

#### 30. 🔴💎 Reinforcement learning over the trust spine

Our orchestrator routes signals by tier. A small RL agent could learn
which tier ceiling each source should hold *per regime*. This is the
ultimate adaptive system — the spine itself learns from the journal.

---

## Hierarchy of confidence

When you find yourself wondering which idea to try first, this ranking
combines my honest expected-value estimate with implementation cost:

| Tier | Why first | What's in it |
|---|---|---|
| **Tier 1 (start here)** | Cheap to build, high signal-to-noise | 6, 7, 13, 26, 28 |
| **Tier 2 (when Tier 1 yields)** | Higher cost, higher payoff | 1, 2, 4, 10, 17, 18, 21, 29 |
| **Tier 3 (when you have a moat)** | Either weird or capital-intensive | 5, 12, 15, 20, 25, 30 |

---

## What the cost model probably gets wrong

For Codex / next-session Claude to know what to suspect:

1. **STT calculated on premium not intrinsic on exercise day.** STT on
   options at exercise is 0.125% of *intrinsic value*, not premium.
   If a contract goes deep ITM and is exercised, charges balloon.
2. **Slippage assumed flat at 2 ticks.** Reality: thinly-traded OTM
   contracts have 5-10 tick slippage and a worst-case of "no fill at
   all." Backtest assumes liquidity.
3. **Bid-ask asymmetry ignored.** Buying at ask + selling at bid is
   the normal case but our model uses last-traded price.
4. **Brokerage flat ₹20 assumes Zerodha.** Other brokers have % rates;
   if the operator ever changes brokers, the model is wrong.
5. **GST applied to wrong base.** GST is 18% on (brokerage + exchange
   + SEBI). Some models apply GST to STT too. Ours doesn't (correct),
   but if the founder changes the model later they could break this.
6. **Stamp duty varies by state.** 0.003% for buy-side is the Delhi /
   Maharashtra rate. Other states differ slightly.
7. **No depository participant charges.** Negligible for options but
   real on equity ITM exercise.
8. **No SEBI Investor Protection Fund charge.** A few paise.

**Mitigation**: build the cost calibration module (idea #6) and re-run
the harness with the calibrated model before believing any survivor.

---

## What's NOT in this playbook (deliberately)

I've left out 5 categories of ideas that look tempting but rarely
pay:

- **Sentiment from Twitter/X.** Indian options traders aren't on X
  in numbers that matter. Telegram and YouTube comments are better.
- **News parsing with LLMs.** Latency kills it. By the time you parse
  RBI policy, the move is done.
- **Pure deep learning on raw bars.** Overfits without massive data.
  Symbolic regression and tree models beat it at our scale.
- **High-frequency latency arbitrage.** We're not co-located with NSE.
  Even with our WebSocket, we're 50-100ms behind professionals.
- **Crypto / FX cross-arb.** Different infrastructure, different
  problems. Don't fork attention.

---

## Practical 30-day plan

If I were sitting in your chair tomorrow, here's the order I would
execute:

**Week 1: Calibrate, don't add features**
- Day 1-2: Build the `CostCalibrator` (idea #6). Use any 10 real
  trades from your past Kite history. Replace the default `CostsModel`
  with the calibrated one.
- Day 3-4: Re-run `H1`, `H2`, `H5` from the existing library against
  the warehouse with the new costs. Report Sharpe per hypothesis.
- Day 5-7: Pick the one survivor (likely H1 — Thursday theta scalp)
  and dive deep: by-month, by-regime, parameter sensitivity.

**Week 2: Mine for new candidates**
- Day 8-10: Build the `HypothesisMiner` (idea #1) with the grammar
  primitives. Run 1000 random samples. Look at the top 30.
- Day 11-12: Manually inspect — are any of the "discoveries" actually
  the same hypothesis under a different name? Dedupe.
- Day 13-14: Walk-forward the top 5 mined candidates.

**Week 3: Build the allocator**
- Day 15-17: Thompson-sampling allocator (idea #26).
- Day 18-19: Decay kill-switch (idea #28).
- Day 20-21: Paper-trade the top 2 winners with the allocator for
  one full week.

**Week 4: Decide**
- Day 22-27: Compare paper-trade results to backtest. Any survivor
  whose live Sharpe is within ±0.3 of backtest? That's the real edge.
- Day 28-30: Graduate it to TRUSTED in the orchestrator. Start with
  1% bankroll. Begin compounding.

---

## How to extend this playbook

For Codex / future Claude / the founder:

- Each idea is a one-paragraph sketch. **Pick one. Code it. Add
  the result back to this doc.** Mark with date + outcome.
- If an idea fails OOS, write a short post-mortem. Knowing why
  something doesn't work is as valuable as knowing what does.
- New ideas: add at the bottom of the relevant section. Don't
  renumber — preserve historical ordering for cross-references.

---

## A final honest note on creativity

I've been measured this entire conversation because the surface is
trading and the stakes are real money. The founder asked me to
unmuzzle and trust my creative output. I tried.

If a different Claude session reads this and sees obvious gaps:
**add them.** This is a living document. The 30 ideas above are not
the only 30 — they're the ones I could surface in one sitting.

Patterns I deliberately did NOT explore deeply:
- Causal inference frameworks (do-calculus, instrumental variables)
  applied to "did this signal CAUSE the price move?"
- Bandit / RL framings of position sizing
- Information-theoretic alpha (mutual information between feature
  combinations and forward returns)
- Topological data analysis of regime transitions
- Bootstrap-aggregated hypothesis testing (each weak hypothesis a
  vote; ensemble decision)

Any of those would take a focused session to make actionable. Worth
queuing.

**Bottom line for the founder**: hypothesis search > hypothesis
guessing. Cost calibration > cost assumption. Walk-forward + paper-
trade > "trust me bro." Start with #6 (cost calibration), then #1
(mining), then #26 (allocation). Three modules. Three weeks. Then
either you have a real edge or you have a real "no" — both are
victories over the current state of uncertainty.
