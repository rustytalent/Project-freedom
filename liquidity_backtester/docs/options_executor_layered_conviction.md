# Options Executor — Layered-Conviction Amendment

> Supersedes §5 of `options_strategy_methodology.md`. Same goal
> (an executor layer that does not exist in the retail Indian options
> space), but reframes the executor from a rule-based gate into a
> **context-layered conviction engine** aligned with the user's actual
> trading style.
>
> Author: written by Opus under user direction after the 2026-06-09
> trading-philosophy briefing. The user's lived trading style is the
> design input; the discipline layer is added so the style survives
> being wrong.

## §0. Why this amendment exists

The original §5 design was **predictor + rule-based gate**: the
model emits `predicted_net_R`, a small rule list says ENTER / WAIT /
SKIP, then a state machine manages the trade with fixed stops and
targets.

The user pointed out — correctly — that this is not how they actually
make money. Their lived edge is:

1. **Reading context BEFORE the trade.** Yesterday: global indexes,
   overnight Bloomberg/RBI flow, US close, gap direction — synthesised
   into a hypothesis about the day's regime.
2. **Forcing the entry when the context aligns**, not waiting for a
   clean intra-bar setup.
3. **Holding through initial drawdown** when the context layers still
   agree (the drawdown is read as manipulation, not invalidation).
4. **Exiting only when the layers themselves start disagreeing**, not
   when a fixed-percent stop fires.
5. **Sizing by conviction**, where conviction is the count and
   strength of agreeing context layers.

That is not a rule-based gate. It is a **layered-conviction
discipline** with discretion encoded into the layer-agreement
calculation.

This amendment redesigns §5 around that.

## §1. The honest tension this design has to resolve

There's a real tension and it would be dishonest to pretend otherwise.

The user's style worked yesterday because:
- HDFC fell within 5 minutes after a -₹1,300 unbooked drawdown.
- INFOSYS resolved in 87 seconds.
- The macro hypothesis was right.

The same style will fail on a different day when:
- The market keeps moving against the conviction for 90 minutes.
- The "manipulation" read is wrong — the trend is real.
- The macro hypothesis was right in direction but wrong in timing.

Survivorship bias is real. Conviction-hold-through-drawdown produces
spectacular wins on the days it works and large losses on the days it
doesn't. A retail trader can absorb the variance with discipline; a
publisher of research has to surface BOTH outcomes honestly in the
audit log.

So this amendment:
1. **Encodes the layered-conviction model faithfully** so the system
   can reproduce the user's style on subscribers' behalf.
2. **Adds explicit hard guardrails** that override conviction when
   the drawdown crosses thresholds that statistically cannot be
   recovered. These are not stop losses in the conventional sense —
   they are disaster floors that exist because the model has to
   survive the days when the user's read is wrong.
3. **Audits the conviction-hold outcomes separately from the clean
   signal outcomes** so subscribers can see, with their own eyes,
   how often "manipulation" reads are actually right vs how often
   they're wishful thinking. The truth is in the data, not in the
   philosophy.

## §2. The six context layers

Each layer produces a **bounded score in [-1, +1]** at the prediction
time. Sign indicates direction (positive = bullish on the underlying;
negative = bearish). Magnitude indicates strength. The layered-
conviction score (§3) combines them.

### Layer 1 — Macro overnight context (the "morning briefing" layer)

**What it captures**: where global markets ended overnight, USDINR
direction, US futures direction at IST 08:30, major overnight news
(RBI flow, FOMC, oil shocks, geopolitical events).

**Inputs**:
- SPX close → DJI close → NDX close (signed change)
- SGX Nifty futures level at IST 08:00 vs prior NIFTY close
- USDINR overnight change
- VIX-US overnight change
- A small structured news-flow feed (RBI calendar, scheduled events)

**Output**: `macro_score ∈ [-1, +1]`. Concretely:
- +0.8 if SPX +0.5%, SGX Nifty +0.3%, USDINR flat, no policy events
- -0.8 if SPX -0.5%, SGX Nifty -0.5%, USDINR +0.3%, hawkish RBI flow
- ±0.3 if mixed / conflicting signals
- 0.0 only when data is unavailable

**Sourcing**: this is the layer most cleanly automated — we already
have warehouse data for SGX-Nifty proxy, USDINR, and INR risk-free
rate. SPX/DJI/NDX needs one new data feed (cheap, multiple providers).
RBI calendar is a static scrape that doesn't change often.

**Why this matters**: yesterday's edge started here. The model needs
to be able to form this view the same way the user does.

### Layer 2 — Index regime layer

**What it captures**: the underlying index's structural state at
the bar before the trade — trend direction, regime stability,
sector rotation signal.

**Inputs** (all from the existing engine):
- Direction model output at h=60 (probability + magnitude)
- Path efficiency (run vs chop predictor)
- Direction changes per 30-bar window
- Vol regime z-score (20d rolling)
- Sector rotation (which sectors leading vs lagging)

**Output**: `index_regime_score ∈ [-1, +1]`. Sign tracks the direction
the underlying is leaning. Magnitude scales with path efficiency
(stronger when the recent move has been run-like, weaker when chop).

**Why this matters**: this is the layer the existing engine produces
naturally. It is the "the structural read" component.

### Layer 3 — Structural-level layer

**What it captures**: whether the underlying is approaching a
high-conviction pool, and which side of price the pool sits on.

**Inputs** (all from the existing engine):
- Proximity probability at h=12 / h=36 / h=60 to the nearest pool
- Pool side (above current price = supply; below = demand)
- Pool composition (how many detector sources contribute)
- Distance from current spot in ATR units

**Output**: `pool_score ∈ [-1, +1]`. Sign is bearish if the nearest
high-strength pool is below (supply-being-tested-from-above is
discretionary; demand-being-tested-from-above is a bounce target).
Magnitude scales with pool strength × proximity probability.

**Why this matters**: this is what the existing brief surfaces as
"watchlist" today. It's the structural foundation for the strike
choice.

### Layer 4 — Options-specific context layer

**What it captures**: the option-side conditions independent of the
underlying — IV regime, theta dominance, gamma exposure.

**Inputs**:
- IV percentile (60d rolling) of the strike being considered
- IV day-over-day change
- Theta per day as % of premium (current option)
- Vega per vol point as % of premium
- Time of day (theta non-linear in this)
- DTE in trading days

**Output**: `options_score ∈ [-1, +1]`. For a BUY thesis:
- Positive when IV is rich enough that vega works for you, theta is
  manageable, DTE > 2.
- Negative when IV percentile > 0.85 (vol mean-reversion risk) or
  theta_per_day > 0.10 (decay will eat the trade).

For a SELL thesis: signs flip.

**Why this matters**: this is the options-specific filter. Without
it the system would recommend the same strike at IV percentile 95
that it recommends at IV percentile 30, and those are very different
trades.

### Layer 5 — Microstructure layer (the user's own discretionary inputs)

**What it captures**: what the user himself uses to time entries —
Anchored VWAP with standard-deviation bands, EMA stack alignment,
candle structure at the level.

**Inputs**:
- AVWAP from session open (and from prior key bar — yesterday's
  close, prior swing) with ±1σ, ±2σ, ±3σ bands
- EMA stack: 25 / 75 / 125 / 250 alignment (all sloping the same way
  = strong trend; tangled = chop)
- Wick/body ratio of the last 3 bars (rejection pattern)
- Cumulative-delta-proxy direction (Stream C-3, already shipped)

**Output**: `microstructure_score ∈ [-1, +1]`. Sign tracks the
direction the bar structure is leaning. Magnitude scales with EMA
stack alignment × candle conviction.

**Why this matters**: this is the user's discretionary edge encoded.
The user said it explicitly: "anchored VWAP with three lines of
standard deviation along with a 4-line context provider of EMA
25/75/125/250." The system needs to compute this the same way the
user reads it.

### Layer 6 — Manipulation layer

**What it captures**: whether the current move looks like a liquidity
sweep / stop-run / sweep-and-reclaim — i.e. the kind of move where
"the drawdown against you is manipulation, not real."

**Inputs** (all from Stream C, already shipped):
- Recent sweep detector outputs (SWEEP_H / SWEEP_L)
- Stop-run-and-reclaim detector outputs (SR_H / SR_L)
- Cumulative-delta divergences
- Volume-weighted swings

**Output**: `manipulation_score ∈ [-1, +1]`. Positive when recent
microstructure shows institutional sweep-reclaim patterns FAVOURING
the current thesis (i.e. someone harvested liquidity against the
direction we're entering, suggesting institutional positioning is
ON our side). Negative when the patterns are AGAINST us.

**Why this matters**: this is the layer that distinguishes
"drawdown-is-manipulation" from "drawdown-is-trend." When this layer
stays strongly positive during an in-trade drawdown, conviction is
maintained. When it drops, conviction breaks.

## §3. The Layered Conviction Score (LCS)

The LCS is a single scalar in [-1, +1] computed at every prediction
time and every 5-min in-trade bar:

```
LCS = w1 * macro_score
    + w2 * index_regime_score
    + w3 * pool_score
    + w4 * options_score
    + w5 * microstructure_score
    + w6 * manipulation_score
```

with weights summing to 1.0. Starting weights (will tune from data):

```
w1 = 0.15   macro
w2 = 0.20   index regime
w3 = 0.20   pool
w4 = 0.15   options
w5 = 0.20   microstructure
w6 = 0.10   manipulation
```

The weights are NOT learned at v1 — they are the user's stated
priorities expressed numerically. v2 learns them from outcomes once
≥200 OOS trades exist per bucket.

### LCS thresholds

```
LCS ≥ +0.50   →  high-conviction BUY thesis
LCS ≥ +0.30   →  moderate-conviction BUY thesis
+0.10 to +0.30 → low-conviction; system WAITs
-0.10 to +0.10 → no signal; system SKIPs
-0.30 to -0.10 → low-conviction SELL
LCS ≤ -0.30   →  moderate-conviction SELL
LCS ≤ -0.50   →  high-conviction SELL
```

### "Force entry" rule (the user's actual discipline)

When LCS ≥ +0.30 (or ≤ -0.30), **the executor recommends ENTER
immediately** at the next bar, regardless of intra-bar timing
considerations.

This is the user's "force trade" principle encoded: do not wait for a
perfect intra-bar setup if the layered context already agrees.

Rationale: in the user's experience (and in the data we have),
waiting for a clean bar pattern after 4+ context layers agree
typically means missing the move. The conviction model says: trust
the layer-agreement, take the entry, accept that the first few minutes
may be against you.

## §4. The conviction-driven hold-vs-exit logic (the moat)

This is the part nobody else builds. Standard executors exit on
percent drawdown. This executor exits on **layer-agreement collapse**.

### Pre-trade snapshot

At entry, the executor records:
- LCS at entry
- Score per layer at entry
- Which layers were "agreeing" (same sign as the thesis)
- The minimum LCS observed during the entry bar

### Every 5-min bar in-trade, recompute

```
LCS_now             = weighted sum, current
agreeing_layers_now = count of layers with same sign as thesis
LCS_drop            = LCS_at_entry - LCS_now
drawdown_R          = realized R in ATR units (negative = against us)
```

### The hold-vs-exit decision tree

The exit decision is **driven by LCS, not by P&L**:

```
if any kill_condition triggered:
    EXIT_KILL
elif drawdown_R <= disaster_floor_R:           # see §6
    EXIT_DISASTER
elif LCS_drop >= 0.40 and agreeing_layers_now <= 2:
    # Layers are collapsing. This is "real invalidation",
    # not manipulation. The user's own discipline says exit.
    EXIT_INVALIDATION
elif drawdown_R <= -1.0 and agreeing_layers_now <= 3:
    # Moderate drawdown AND moderate layer-disagreement.
    # We hold one more bar, then re-decide.
    HOLD_WATCHED
elif LCS_now >= 0.60 and realized_R >= 1.0:
    # Strong conviction AND profitable. Trail.
    TRAIL_STOP
elif realized_R >= predicted_target_R:
    EXIT_TARGET
else:
    HOLD
```

### The key insight

A drawdown of -1.5 ATR with LCS still at +0.55 and 5/6 layers
agreeing is **NOT an exit signal under this logic**. The layered
read of the market still says the thesis is correct; the drawdown is
manipulation noise.

A drawdown of -0.5 ATR with LCS collapsed from +0.55 to +0.05 and
agreeing layers dropped from 5 to 2 **IS an exit signal**. The
drawdown is small, but the underlying read has fallen apart, which
the user's own discipline says is the real invalidation.

This is the user's lived trading style encoded as a decision tree.
It will lose money on days the layers stay aligned but the market
keeps moving anyway (the "everyone is wrong; manipulation just kept
going" scenario). Those losses are bounded by §6.

### Why this is novel

Every other retail options executor I know of exits on percent
drawdown. That works fine for systems whose edge is statistical
because their conviction never changes during the trade.

This executor exits on conviction collapse. Conviction is recomputed
from the same six layers every 5 minutes. When the layers agree, we
hold. When they disagree, we exit. The percent drawdown is a
secondary check, not the primary trigger.

## §5. Dynamic rupee floor (data-derived per bucket)

The original methodology hardcoded `rupee_reward_floor_inr=600`. Per
the user's correction, this should be **derived from the per-bucket
statistical sweet spot** in the OOS evaluation window.

### The formula

For each (side × tenor × ToD) bucket, after training the model and
holding out the OOS slice:

```
rupee_floor_per_bucket = max(
    300,                                      # hard minimum to clear cost
    p20_of_top_decile_realized_R_inr          # 20th percentile of the
                                              # rupee outcomes from
                                              # trades that scored in
                                              # the top decile of
                                              # predicted_R
)
```

Where `p20_of_top_decile_realized_R_inr` is computed as:

```
top_decile_oos_trades = oos_trades where predicted_R >= 90th_percentile
realized_R_inr = predicted_R_atr × per_lot_R_inr × default_qty
floor = numpy.percentile(realized_R_inr, 20)
```

### What this gives us

A statistically-honest "expected minimum profit per top-tier trade"
per bucket. The 20th percentile means: even on the worst 1-in-5 days
in the OOS window, this trade would clear at least this many rupees.

If `floor < 300` for a bucket, that bucket fails the cost-arithmetic
test for retail-scale trading and we don't expose it to subscribers
even if the model has rank skill.

### Monthly refresh

The floor recomputes monthly on a rolling 90-day window of realized
outcomes. As live data accumulates, the floor refines toward the
true distribution rather than the (smaller) OOS sample distribution.

### What this REPLACES

The previous flat `rupee_reward_floor_inr=600` is gone. Sizing math
in §6 below uses `rupee_floor_per_bucket` instead.

## §6. Sizing — the reward = risk^p formulation

The user gave the formula. Let me formalise it:

```
√risk = reward^p

→ risk = reward^(2p)
```

with `p` derived from **conviction × quantity-vs-price ratio**:

```
p = base_p + conviction_bonus + size_bonus

base_p              = 1.0           # baseline; risk = reward^2
conviction_bonus    = 0.5 * (LCS - 0.30)    # only above the entry threshold
size_bonus          = 0.3 * log(notional / typical_notional_for_strike)
```

So:
- High LCS (+0.60) + moderate size → p ≈ 1.15 + 0.1 = 1.25
  → risk = reward^2.5 (acceptable downside given high conviction)
- Low LCS (+0.30) + small size → p ≈ 1.0
  → risk = reward^2 (conservative)
- High LCS + LARGE size → p ≈ 1.3
  → risk = reward^2.6 (system aware large size carries more cost
                       asymmetry; demands higher conviction)

### What "risk" and "reward" mean here

- **Reward** = `predicted_net_R_atr × per_lot_R_inr × qty` (rupee
  upside if the prediction is right)
- **Risk** = the absolute rupee cap the executor will allow on this
  trade (the disaster floor; see below)

### The disaster floor

Given the sizing formula:

```
max_rupee_risk = reward^(2p)
disaster_floor_R = -max_rupee_risk / (per_lot_R_inr × qty)
```

This is the rupee number the executor will NOT let the trade drop
past, regardless of conviction. If the trade hits this, EXIT_DISASTER
fires (§4 decision tree) even if LCS is still +0.80.

### Why this disciplines the conviction-hold

Conviction-driven holds can in principle absorb arbitrary drawdown.
The disaster floor caps absolute rupee loss per trade so that even a
day where the user's read is wrong AND the layers stay misleadingly
aligned (the "everyone is wrong" scenario) is bounded.

This is the explicit safety the §1 honesty discussion demanded.

## §7. What this design does NOT do

Same scope discipline as the original methodology doc, plus:

1. **Does not learn the layer weights at v1.** The 0.15/0.20/0.20/
   0.15/0.20/0.10 weights are stated priorities. v2 fits them from
   outcomes.
2. **Does not predict "manipulation" — it detects it after the fact.**
   The manipulation layer (Layer 6) reads recent sweep/reclaim
   patterns. It cannot predict whether the NEXT 30 minutes will be
   a sweep. It can only say "the last 30 minutes looked like one,
   which means institutional flow may be on our side."
3. **Does not auto-execute trades.** The brief surfaces the
   executor's view and rationale; the subscriber decides whether to
   take it. The system is a context engine, not an order router at
   v1 (broker integration is Stream D.8, deferred post-revenue).
4. **Does not cover events.** FOMC days, RBI policy days, earnings,
   expiry-day morning — the macro layer flags these and the executor
   downgrades all signals to WAIT regardless of other layers.
5. **Does not predict the next "manipulation hold" outcome
   individually.** It produces the conviction score; the audit log
   tracks how often conviction-holds win vs lose.

## §8. The audit gain — proving this design works

The Yesterday Audit section (Stream G + D.6) gains TWO sub-tables
for options conviction-holds:

### Table A — outcomes for "took the trade and exited at target/stop"

Standard executor performance. Mean realized R, calibration error,
hit rate per bucket. Same as the original §5 design audited.

### Table B — outcomes for "took the trade, hit drawdown, HELD via conviction, then exited"

This is the new table that proves whether the conviction-hold
discipline actually works on real data.

```
                  n   mean_realized_R   pct_recovered_to_profit  pct_lost_more
all conviction-holds     ...      ...           ...                    ...
buy / weekly / morning   ...      ...           ...                    ...
sell / weekly / morning  ...      ...           ...                    ...
LCS at hold > 0.50       ...      ...           ...                    ...
LCS at hold 0.30-0.50    ...      ...           ...                    ...
```

A successful design has:
- `pct_recovered_to_profit > 60%` for LCS > 0.50 at hold
- `mean_realized_R > 0` for high-LCS holds
- BOTH visibly less for lower-LCS holds (proves the LCS threshold
  is meaningful)

If Table B shows that conviction-holds are actually break-even or
losing on average, **the conviction-hold logic gets retired** and
the executor reverts to the simpler percent-stop discipline of the
original §5. The data decides.

## §9. Updated acceptance gates

Gate 2 (the executor-value gate) is updated to test this new design:

**Gate 2 — does layered-conviction add value beyond a simple stop?**

Pass criteria:
1. The naive baseline (`always enter at LCS ≥ 0.30; exit at -30%
   stop or +50% target`) has its mean realized R measured.
2. The conviction-hold variant of the same trades has its mean
   realized R measured.
3. Conviction-hold variant beats naive by ≥ +0.20 R per trade,
   measured on the OOS window.
4. Disaster_floor never gets hit on > 5% of trades. If it gets hit
   more often, sizing is too aggressive; reduce conviction_bonus
   coefficient.

If Gate 2 fails: retire the conviction-hold logic (use the simpler
§5 executor with rule-based stops), and ship the model + simpler
executor as v1.

## §10. Why this matters for the brand

The Yesterday Audit table on conviction-holds is the audit nobody
in retail Indian options has. Every other service in the space
either (a) doesn't track hold-vs-exit decisions at all, or (b) only
reports the winners.

Publishing the conviction-hold success rate alongside the simple
executor's success rate is the **calibrated honesty that
distinguishes this product**:

- "Of the 48 conviction-hold trades in the last 90 days, 31 (65%)
  recovered to profit, 17 (35%) closed at the disaster floor. Average
  outcome: +0.42 R."

This is what makes the product trustable. Not the philosophy. The
audit.

## §11. Open questions to settle before D.5 implementation

These need user input before I write code:

1. **Macro layer data source.** Do we use a paid data feed for
   SPX/DJI/NDX (e.g. AlphaVantage, Polygon, Twelve Data; ~$30/month)
   or scrape Yahoo Finance for free? The free option works but has
   no SLA.
2. **Microstructure layer's AVWAP anchor.** The user computes AVWAP
   from session open. Do we also compute from prior swing
   high/low / prior day's close as alternative anchors? Multiple
   anchors give more layers but more complexity.
3. **Conviction weights starting values.** I used 0.15/0.20/0.20/
   0.15/0.20/0.10 above. Are those right or should the user nudge
   them?
4. **Force-entry threshold.** I used LCS ≥ +0.30 for ENTER. Is that
   right or should it be +0.40 (more conservative) or +0.20 (more
   aggressive)?
5. **Disaster floor coefficient.** Currently `reward^2p` gives a
   typical disaster floor of -1.5 to -2.5 R per trade. Comfortable
   with that range or want tighter?

I default to the values above and proceed. Override any of them
before D.5 lands and I'll re-derive.

End of amendment.
