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

## §1. The design's central commitment

This design has one central commitment: **either we trust the
Layered Conviction Score, or we don't ship.**

Earlier drafts tried to hedge — keep the conviction model AND add a
fixed-percent drawdown stop "just in case." After user pushback on
2026-06-09, that hedge is removed. It was incoherent: a system that
overrides its own conviction layer on a fixed-percent threshold is
admitting it doesn't trust the layer. Shipping a model we don't
trust is the actual risk; the override is just retail-stop reflex
disguised as discipline.

The argument for trusting the LCS in production is **not** that
conviction is always right. It will sometimes be catastrophically
wrong. The argument is that the place where we discover whether to
trust the LCS is **the OOS replay at Gate 2 and the live audit at
Gate 4** — not a runtime hedge that contaminates every trade.

So the philosophy this design enforces is:
1. **In production**: the LCS is the only exit signal. If layers
   agree, we hold. Always.
2. **At Gate 2** (before ship): the conviction-hold variant must
   beat the naive percent-stop baseline by ≥ +0.20 R per trade on a
   60-day OOS replay. If it doesn't, we ship the simpler design
   instead. **This is where "do we trust it?" gets answered.**
3. **At Gate 4** (first 30 live days): the conviction-hold outcomes
   must hold OOS calibration. If drawdown distribution shifts
   meaningfully, the LCS weights or thresholds get retuned.
4. **In the audit** (every brief): conviction-hold outcomes are
   published separately from naive outcomes. The subscriber sees
   what the discipline cost on the bad days AND what it earned on
   the good ones. The discipline either proves itself or it doesn't,
   in public.

This means: the system WILL lose money on days when layers agree
but the market keeps moving anyway. Those losses are real and the
audit shows them. The bet is that, on average, across 200+ trades,
the days the layers are right pay for the days they're wrong by
more than the naive percent-stop baseline pays. Gate 2 answers that
bet empirically before any subscriber sees it.

Survivorship bias is the live risk. The user's three winning trades
yesterday are not evidence the discipline works — they are evidence
the discipline worked on three specific trades. The 60-day OOS
replay is what tells us whether the philosophy generalises. The
30-day live window is what tells us whether the OOS result
generalises further. The audit is what tells subscribers the answer
in their own time.

If the data says yes: this is the genuinely novel executor described
in §4. If the data says no: this becomes the simpler executor of the
original methodology §5, no apology needed.

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

## §3. The Layered Conviction Score (LCS) and the Causal Context Vector (CCV)

> **Redesigned 2026-06-09 (third revision, same day)** after user
> pushback. The previous geometric-mean cascade was right about
> hierarchy but wrong about (a) collapsing all the causal state into
> a single scalar and (b) applying uniform global damping when one
> layer collapsed. Real layer interactions are LOCAL: when
> manipulation fires, it forces microstructure and distorts pool —
> not macro, not regime. The downstream LightGBM model must encode
> this graph structure, not just consume layer scores as flat
> features.

### §3.1 The causal layer graph

```
            ┌─────────────────────────────────────────┐
            │                                         │
   macro ──►│   regime ──► pool ──► options           │
            │                ▲                        │
            │                │                        │
            │   micro ◄──────┴──── manipulation       │
            │     ▲                      │            │
            │     └──────────────────────┘            │
            │   (manipulation forces micro;           │
            │    micro reflects forced-or-organic)    │
            └─────────────────────────────────────────┘
```

- **Forward edges** (top to bottom): macro defines possibility space;
  regime selects the realised regime; pool defines the target; options
  defines the vehicle.
- **Manipulation has backward edges** into pool and micro. Aggressive
  stacked orders can distort pool boundaries (apparent demand pool
  broken by stacked-buy pressure, not by organic demand failure). The
  same orders force the microstructure (EMA stack flipping because
  the price was pushed there, not because trend reversed).
- **Micro is dual-purpose**: a micro signal can be either organic
  (trend-induced) or forced (manipulation-induced). The two have
  different information content and must be distinguished, not summed.

### §3.2 The Causal Context Vector (CCV) — what the executor and the
        model actually consume

The single-scalar LCS throws away the causal state. The system
internally carries a richer object — the **Causal Context Vector**
(CCV) — which preserves the layer-by-layer information:

```python
@dataclass
class CCV:
    # Raw per-layer scores, in [-1, +1]
    macro_score: float
    regime_score: float
    pool_score: float
    options_score: float
    micro_score: float
    manipulation_score: float

    # Causal flags derived from inter-layer state (see §3.3)
    micro_forced_flag: bool        # micro signal likely caused by manip
    pool_distortion_flag: bool     # pool boundary distorted by aggressive flow
    micro_organic_score: float     # micro_score after removing forced component
    pool_holding_strength: float   # pool_score after distortion adjustment

    # Cross-layer interaction features (for the LightGBM input)
    manip_micro_alignment: float   # signed agreement between manip and micro
    manip_pool_alignment: float
    regime_pool_alignment: float
    macro_regime_alignment: float

    # Summary scalar for human consumption only (the brief shows this)
    lcs: float                     # cascade output per §3.4
```

The brief and the customer see the LCS scalar. The executor and the
LightGBM force-entry classifier consume the full CCV. The scalar is
a digest for humans; the vector is the actual signal.

### §3.3 Local collapse — the causal adjustment layer

When a layer fires strongly and the graph says it has a downstream
effect, we apply a LOCAL adjustment. Not a global damping.

```python
def apply_causal_adjustments(raw: dict[str, float]) -> CCV:
    # Start with raw scores
    micro = raw["micro"]
    pool = raw["pool"]
    manip = raw["manipulation"]

    micro_forced_flag = False
    pool_distortion_flag = False

    # ── Manipulation → microstructure ──────────────────────────────
    # If manipulation is firing in the same direction as microstructure,
    # the micro signal is partly EXPLAINED by manipulation. It is not
    # additional independent evidence — it is the same evidence read
    # through a different layer.
    if abs(manip) >= 0.40 and sign(manip) == sign(micro):
        micro_forced_flag = True
        # Remove the manipulation-attributable component from micro.
        # The remaining "organic" signal is what the model should weight.
        explained_share = min(1.0, abs(manip))
        micro_organic = micro * (1.0 - 0.6 * explained_share)
    else:
        micro_organic = micro

    # ── Manipulation → pool distortion ─────────────────────────────
    # Strong manipulation with sign opposite to pool means aggressive
    # flow is pushing AGAINST the pool. The pool's holding power is
    # weaker than its raw score suggests.
    if abs(manip) >= 0.50 and sign(manip) != sign(pool):
        pool_distortion_flag = True
        pool_holding = pool * 0.6
    else:
        pool_holding = pool

    # ── Cross-layer alignment features (for the model) ─────────────
    return CCV(
        macro_score=raw["macro"],
        regime_score=raw["regime"],
        pool_score=pool,
        options_score=raw["options"],
        micro_score=micro,
        manipulation_score=manip,
        micro_forced_flag=micro_forced_flag,
        pool_distortion_flag=pool_distortion_flag,
        micro_organic_score=micro_organic,
        pool_holding_strength=pool_holding,
        manip_micro_alignment=sign(manip) * sign(micro) * min(abs(manip), abs(micro)),
        manip_pool_alignment=sign(manip) * sign(pool) * min(abs(manip), abs(pool)),
        regime_pool_alignment=sign(raw["regime"]) * sign(pool) * min(abs(raw["regime"]), abs(pool)),
        macro_regime_alignment=sign(raw["macro"]) * sign(raw["regime"]) * min(abs(raw["macro"]), abs(raw["regime"])),
        lcs=_compute_lcs_scalar(raw, micro_organic, pool_holding),
    )
```

Key points:
- The adjustment is **LOCAL**: it affects micro and pool when
  manipulation fires. It does NOT affect macro, regime, or options.
  Macro is upstream; it cannot be forced by downstream manipulation.
- The flags persist: `micro_forced_flag` is a feature the model
  sees. The model learns that "micro is bullish + micro_forced_flag
  is True" is a DIFFERENT regime than "micro is bullish +
  micro_forced_flag is False."
- Both raw and adjusted versions of micro/pool are kept. The model
  decides which to rely on, per bucket.

### §3.4 The LCS scalar — for human display only

The LCS is now a digest of the adjusted state:

```python
def _compute_lcs_scalar(raw, micro_organic, pool_holding):
    macro = raw["macro"]
    if abs(macro) < 0.10:
        return 0.0                       # macro gate closed
    direction = +1.0 if macro > 0 else -1.0

    # Use the ADJUSTED scores for micro and pool. Manipulation enters
    # as its own factor (it is real causal evidence about institutional
    # positioning).
    adjusted = [
        raw["regime"],
        pool_holding,
        raw["options"],
        micro_organic,
        raw["manipulation"],
    ]
    factors = [1.0 + 0.5 * (direction * s) for s in adjusted]
    n = len(factors)
    product = 1.0
    for f in factors:
        product *= f
    geom_mean = product ** (1.0 / n)
    return direction * min(1.0, abs(macro) * geom_mean)
```

The scalar is what gets rendered in the brief ("Today's LCS for
NIFTY ATM-PE buying: +0.42"). It is convenient for human attention
but is NOT what the executor or the model use for decisions.

### §3.5 The LightGBM modifications — structural priors via
        `interaction_constraints` and `monotone_constraints`

The OptionsExpectedReturnModel (§4 of the parent methodology doc) is
a LightGBM Huber regressor. We use two LightGBM features to encode
the causal graph as structural priors:

**Interaction constraints** — restrict which features can appear
together in the same tree-split path. This forces the model to
respect the causal DAG:

```python
# Group definitions encode the layer DAG of §3.1.
# A feature can belong to multiple groups (overlap = allowed
# interaction across groups). A split path can only combine features
# that share at least one group.
INTERACTION_GROUPS = [
    # Top-of-cascade
    ["macro_score", "regime_score", "macro_regime_alignment"],
    # Mid-cascade
    ["regime_score", "pool_score", "options_score",
     "regime_pool_alignment", "pool_holding_strength"],
    # Manipulation cluster (the local-collapse zone)
    ["pool_score", "micro_score", "manipulation_score",
     "manip_micro_alignment", "manip_pool_alignment",
     "micro_forced_flag", "pool_distortion_flag",
     "micro_organic_score", "pool_holding_strength"],
    # Options-specific cluster
    ["options_score", "iv_percentile", "theta_per_day_pct",
     "dte_trading_days"],
]
```

Effect: the model cannot fit a tree that splits on
`{macro_score, options_score}` directly (those don't share a group),
but it CAN split on `{macro_score, regime_score, pool_score,
options_score}` because the chain connects through shared-group
features. This enforces top-down information flow.

**Monotone constraints** — pin known directional relationships so
the model cannot learn implausible reversals:

```python
MONOTONE = {
    # Higher macro (more bullish) → higher expected R for BUY side
    "macro_score": +1,
    # Higher micro_organic → higher expected R (after manipulation
    # adjustment; organic micro is the trusted signal)
    "micro_organic_score": +1,
    # Higher options_score (better vol regime) → higher expected R
    "options_score": +1,
    # Higher theta_per_day_pct → lower expected R for BUY side
    # (theta is a drag for premium buyers)
    "theta_per_day_pct": -1,
    # Higher IV percentile → lower expected R for BUY side
    # (mean reversion risk)
    "iv_percentile": -1,
    # The flags and alignments are not monotone-constrained; let the
    # model find the right shape.
}
```

Effect: the model is mathematically prevented from learning, e.g.,
"more bullish macro → lower expected buy-side R." That kind of
pattern can appear in noisy training data and would silently destroy
the model. Monotone constraints kill it at the loss-function level.

**Why this is a real LightGBM modification and not just "throw
features at it"**: the standard ML practice is to add features and
hope the model learns the right structure. With ~200 OOS trades per
bucket the model does NOT have enough data to learn structure from
scratch; it overfits the noise. Causal interaction constraints +
monotone constraints encode the prior knowledge that we DO have
(the layer DAG) and lets the model spend its limited data budget
on the questions we don't already know the answers to (e.g.
"how strongly does theta drag interact with vol regime?").

### §3.6 What about the force-entry classifier (separate model)

The force-entry classifier (introduced in §3.5 of the earlier draft,
now reframed) is a SECOND LightGBM model — a binary classifier on
`p_force_entry_profitable`. It uses the same CCV as input plus
predicted_R from the regression model:

```python
FORCE_ENTRY_INTERACTION_GROUPS = [
    INTERACTION_GROUPS[0],
    INTERACTION_GROUPS[1],
    INTERACTION_GROUPS[2],     # manipulation cluster - most important here
    INTERACTION_GROUPS[3],
    ["predicted_R", "predicted_R_lower_ci", "predicted_R_upper_ci"],
]

FORCE_ENTRY_MONOTONE = {
    "predicted_R": +1,           # higher predicted R → more likely profitable
    "micro_forced_flag": -1,     # forced micro is weaker evidence; reduce P
    "pool_distortion_flag": -1,  # distorted pool is weaker holding; reduce P
}
```

So the force-entry classifier is encouraged by the model structure
itself to be SKEPTICAL of forced microstructure and distorted pools
— exactly the user's "originality collapsed mean nearby thing also
get distorted" intuition, encoded as monotone constraints.

### §3.7 What the brief actually displays

The customer sees the LCS scalar but ALSO sees the causal flags
when they fire. The renderer surface for an options strike entry:

```
NIFTY50 24500 PE — Buy-side analysis:
  LCS = +0.42 (moderate bearish conviction on NIFTY)
  Per-layer state:
    macro: +0.55 (US closed -0.4%, SGX Nifty -0.3%)
    regime: +0.40 (index trending down, mild)
    pool: +0.60 (24500 is strong demand pool from prior session)
    options: +0.30 (IV percentile 0.55, theta modest)
    micro: +0.25 (EMA stack rolling down, modest)
    manipulation: +0.45 ⚠ MICRO FORCED
  Causal note: aggressive sell-side flow detected. The bearish
    microstructure read is partly explained by recent stop-run
    activity; the organic micro contribution is +0.10. Read with
    caution.
  Predicted net R (60min, buy-side): +0.31 ATR. Bootstrap rule says
    ENTER (LCS ≥ 0.30). Learned classifier (when fitted) will weigh
    the MICRO_FORCED flag and may downgrade.
```

This is the rich object. The customer sees not just "the model says
ENTER" but the chain of why — including the warning that the
microstructure read is forced. They can disagree.

### §3.8 Engineering commitments

- `lib/options/ccv.py` — CCV dataclass + `apply_causal_adjustments`
  function (deterministic, no learned components).
- `lib/options/lcs.py` — `_compute_lcs_scalar` (a pure function of
  the adjusted scores).
- `lib/options/featurizer.py` — when building the LightGBM input
  frame, emits the full CCV per row + the cross-layer interaction
  features.
- `lib/options/model_config.py` — `INTERACTION_GROUPS` and
  `MONOTONE` constraints; consumed by both the
  OptionsExpectedReturnModel fit and the force-entry classifier fit.
- Pin tests on the causal adjustments (manipulation aligned with
  micro → forced flag fires + micro_organic damped; manipulation
  opposite to pool → distortion flag fires + pool_holding damped;
  no flags fire when manipulation is weak or absent).
- Brief renderer surfaces the per-layer table + causal warnings
  when flags fire.

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

The exit decision is **driven by LCS only. P&L is never a standalone
exit trigger.** Drawdown alone, no matter how deep, does not exit
the trade. The layered conviction model is either trusted or it
isn't; we trust it.

```
if any kill_condition triggered:                        # §4.1
    EXIT_KILL
elif LCS_drop >= 0.40 and agreeing_layers_now <= 2:    # real invalidation
    EXIT_INVALIDATION
elif realized_R >= predicted_target_R:
    EXIT_TARGET
elif LCS_now >= 0.60 and realized_R >= 1.0:            # strong + profitable
    TRAIL_STOP
else:
    HOLD
```

That's the entire tree. There is no `drawdown_R <= disaster_floor`
clause. If the layers say HOLD, the executor holds, regardless of
how deep the drawdown gets.

### §4.1 Kill conditions — what CAN override the layers

Only events the context layers literally cannot react to inside a
5-minute bar:

1. **Exchange-level halt** on the underlying or the option chain
   (NSE F&O circuit breaker).
2. **Broker connection loss** lasting > 60 seconds. (Audit-only at
   v1 since we don't auto-execute; the brief just notes "if you are
   in this trade and lost connection, your discipline takes over".)
3. **VIX intra-bar spike > +3 points** (this is a magnitude that
   means the volatility regime has changed faster than any layer
   can update; the trade's pricing assumptions are no longer valid).
4. **Underlying spot intra-bar move > 4 ATR** in either direction
   (flash event; layers cannot reprice in one bar).

None of these are "the trade went against you." Every one of them
is "the world changed faster than the model can read."

### The key insight

A drawdown of -1.5 ATR with LCS still at +0.55 and 5/6 layers
agreeing is **NOT an exit signal**. The layered read of the market
still says the thesis is correct; the drawdown is manipulation.

A drawdown of -3.0 ATR with LCS still at +0.55 and 5/6 layers
agreeing is **STILL NOT an exit signal**. Same reason.

A drawdown of -0.5 ATR with LCS collapsed from +0.55 to +0.05 and
agreeing layers dropped from 5 to 2 **IS an exit signal**. The
drawdown is small, but the underlying read has fallen apart, which
is the real invalidation.

A flash event that moves the underlying 4 ATR in 5 minutes triggers
EXIT_KILL because the layers cannot react that fast. Not because
the P&L is bad.

### Why this is novel

Every other retail options executor I know of exits on percent
drawdown. They have to, because their underlying edge is purely
statistical and their conviction never changes during the trade —
the only sensible discipline is "cut at -X%".

This executor doesn't have a fixed-percent stop at all. Conviction
is recomputed from the six layers every 5 minutes. When the layers
agree, we hold — for as long as they agree. When they disagree, we
exit. There is no "but just in case the model is wrong" override.
The model's correctness is the question Gate 2 answers, not a
parameter to hedge against in production.

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
- **Risk** = the rupee figure used for sizing the position so that
  conviction-weighted upside justifies the notional. This is a
  **sizing input**, not a runtime stop.

### What "risk" is NOT, anymore

In an earlier revision of this document the `risk` figure also acted
as a runtime stop — a "disaster floor" that fired EXIT_DISASTER even
when LCS was high. That clause has been **removed** (2026-06-09
revision after user pushback).

The user's argument was correct: a runtime drawdown floor that
overrides the conviction model is incoherent with the design's
premise. Either we trust the LCS, or we don't. If we layer a
"but-just-in-case" P&L stop on top, we are admitting we don't trust
our own model — and shipping a model we don't trust is the actual
risk. The mid-position is the worst position.

The runtime executor therefore has no fixed-percent stop. The only
exits are: layer-collapse (EXIT_INVALIDATION), target (EXIT_TARGET),
trail (TRAIL_STOP after profit), kill-condition (EXIT_KILL: exchange
halt, broker outage, VIX spike > +3 points, intra-bar spot move > 4
ATR — events the layers literally cannot react to).

### Where the safety actually lives

Safety is moved up-stack to where it belongs:

1. **At sizing**: `√risk = reward^p` is the sizing rule. The harder
   the conviction (higher LCS), the bigger the position. This caps
   exposure at the moment of decision rather than after.
2. **At Gate 2**: the conviction-hold logic must prove itself on
   real OOS data — if it doesn't beat naive percent-stop discipline
   by ≥ +0.20 R per trade, the entire conviction-hold engine gets
   retired and we ship the simpler version. **The data decides
   whether to trust the philosophy.**
3. **At Gate 4**: 30 days of live conviction-holds must hold the OOS
   calibration. If they don't, the head retrains.
4. **At the audit table B (§8 of this doc)**: every conviction-hold
   outcome is published. Subscribers see the win rate and the worst
   loss alongside each other. The product's honesty is in the audit,
   not in the rules.

This means: in production, if the user (or any subscriber following
the executor) takes a -₹4,000 conviction-hold trade because layers
agreed and the trade ran away anyway, that's a real loss the system
recorded. The audit publishes it. The subscriber knows. The next
day's calibration table shows the impact. If those losses accumulate
faster than the wins, Gate 4 fires and the conviction-hold parameters
get retuned or the discipline gets retired.

That is honest risk management. A hidden override that fires the
emergency-exit on the customer's behalf is not.

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

## §11. Open questions — answered 2026-06-09

1. **Macro layer data source — RESOLVED → free.** Yahoo Finance
   public endpoints for SPX/DJI/NDX/SGX-Nifty + the existing
   warehouse for USDINR. No paid feed at v1. Caveat documented:
   no SLA, no overnight backfill on the rare days Yahoo's endpoint
   is degraded; on those days the macro layer reports `score=0`
   (which closes the cascade gate per §3.1 and the executor SKIPs
   the day entirely until macro is recovered). When revenue allows,
   swap in a paid feed (Polygon / Twelve Data / AlphaVantage) by
   replacing one adapter class; no other code changes.

2. **Microstructure AVWAP anchor — RESOLVED → session-open only.**
   No prior-swing or prior-day-close anchors at v1. Keeps the
   microstructure layer auditable in one line: "anchored from the
   day's 09:15 IST bar with ±1σ, ±2σ, ±3σ bands." If we add anchors
   later they go behind a feature flag so the audit can compare
   one-anchor vs multi-anchor calibration on real data.

3. **Conviction weights — OBSOLETE.** The original §3 had per-layer
   weights `w1..w6` summing to 1.0 inside a linear sum. The cascade
   redesign (§3.1) does not use linear weights. Macro is a gate
   (no weight); the other five layers contribute multiplicative
   factors in `[0.5, 1.5]` aggregated by geometric mean. Each
   non-macro layer contributes equally per the geometric mean. v2
   may add per-layer powers (`factor_i ^ w_i`) if data justifies
   layer asymmetry; v1 stays equal.

4. **Force-entry threshold — RESOLVED → bootstrap rule, then
   learned.** Per §3.5: at first deploy ENTER fires at `|LCS| ≥
   0.30` (bootstrap so the system has defined behaviour from day 1).
   After ≥200 trades per bucket accumulate, a small logistic
   regression per (side × tenor × ToD) bucket replaces the
   threshold. Bootstrap and learned classifier run in parallel for
   90 days; whichever beats the other on realized R per trade
   becomes the production policy. Audit publishes both decisions
   side-by-side throughout.

5. **Disaster floor — OBSOLETE.** Removed entirely per D13. The
   only runtime exits are EXIT_KILL (events the 5-min layer
   recomputation cannot detect), EXIT_INVALIDATION (LCS collapse),
   EXIT_TARGET, TRAIL_STOP. Safety lives at sizing, at Gate 2, at
   Gate 4, and in audit table B. No rupee-cap runtime override.

All five answered. D.5 implementation has no remaining structural
ambiguity. Next implementation step is D.1 (per-strike featurizer).

End of amendment.
