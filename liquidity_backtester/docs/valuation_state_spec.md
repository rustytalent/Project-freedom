# Valuation State — Stream Q specification

> The founder's "bubble" model, formalised. Origin: founder's own
> pre-market gap trading observations, 2026-06. Captured verbatim
> intent first, then the engineering translation.

## 1. The founder's thesis (preserved)

The Indian market behaves like a compressible bubble: under macro
pressure (crude spikes, global stress) value gets wiped; with no
technical pressure it re-expands. We — retailers — never get insider
information early. But insiders PRICE IT IN, and their pricing leaves
a footprint in the tape before the news layer reaches us. If we build
a **valuation reference** from everything we have (the substrate:
years of OHLCV, macro series, synthetic replay), anchored at an
arbitrary X (absolute level is unknowable and irrelevant — only
*changes* and *divergences* matter), then:

- **Gap vs valuation divergence is a manipulation signal.** Valuation
  weak + gap UP → the gap is unsupported, expect it to come back down.
  Valuation weakly down + gap DOWN that "over-justifies" the weakness
  → expect reversal up. "The first scandal of every market day is the
  biggest scandal" — the opening gap is where informed positioning
  shows itself.
- **Intraday valuation trajectory** (each day's drift from its own
  opening valuation, at hour granularity) may reveal where the day's
  peak/low forms relative to fair value.
- Layered correctly, this can **front-run the news layer** for
  retail: we never see the news early, but we can see the *pricing
  footprint* of those who do.

## 2. Engineering translation

What the thesis calls "valuation" is a **latent fair-value state**
estimated from market-internal + macro inputs, deliberately anchored
at an arbitrary origin (call it X = 100 at series start). Three
nested reference frames:

| Frame | Window | What it answers |
|---|---|---|
| Structural | yearly | which year-regime are we in (cheap/dear vs history) |
| Cyclical | monthly | is this month stretched vs its year |
| Tactical | daily/hourly | today's drift vs today's own open |

### 2.1 The state vector (inputs)

The escape from circularity is the core design constraint: a
"valuation" built only from price is just smoothed price, and
divergence from it is just momentum/mean-reversion in disguise. The
state MUST blend orthogonal inputs:

- price/return structure (we have: OHLCV 5m, 3y)
- **breadth** — advance/decline across the 25-name basket (computable
  from existing per-asset data TODAY; index-wide breadth needs the
  index data the founder committed to provide, T3.5)
- volume character — basket volume z-score, delivery % (bhavcopy, free)
- **macro pressure channel** — crude, USD/INR, India VIX, US close
  (extends the existing options macro_scrape; all free)
- FII/DII flows (Stream P ingest, free)
- futures basis + OI build/unwind when index data lands (the single
  best "insider footprint" proxy available to retail)

### 2.2 The model (two stages)

**Stage 1 — ValuationState estimator.** v1 is deliberately humble: a
weighted composite z-score of the orthogonal channels, EWMA-smoothed,
per frame. (A Kalman latent-state upgrade is v2; do not start there —
an interpretable composite that the founder can sanity-check against
his own trading instinct beats a black-box latent on day one.)
Output per day t:

    V_struct(t), V_cycl(t), V_tact(t, h)   — anchored, unitless
    dV(t) = V(t) - V(t-1)                  — the valuation drift
    div(t) = sign/magnitude divergence between dV and realised return

**Stage 2 — GapBehaviorModel.** The tradeable head. LGBMX binary heads
on (valuation state, gap context) at 09:15:

    P(gap fades >= 50% by 11:00 | V-state, gap size, gap direction)
    P(close > open | V-state, gap context)
    P(day high before 11:00 | V-state, gap context)   [peak-early head]

Labels computable from existing 5m data the moment index OHLCV lands.
The existing `is_gap_up_trap_fade` / `is_gap_down_reversal` features
(timing.py) are the primitive ancestors of this — Stream Q generalises
them by conditioning on the valuation state instead of raw gap size
alone.

### 2.3 Where it plugs into the organism

- **CCV macro layer upgrade**: the options executor's macro_score is
  currently a thin DoD% composite from Yahoo. `V_tact` + `dV_cycl` is
  a strictly richer macro layer — same gate semantics, better food.
  This directly upgrades Stream D's executor without touching its
  state machine.
- **Brief**: a new "valuation context" block — disclosed as the same
  calibrated-probability artifact as everything else (P(gap fade)
  with conformal bands), never as a directional command.
- **Brain (T2)**: V-state is exactly the kind of slow-context channel
  the path-finder will condition on later.

## 3. Honest assessment

**Strong**: gap-conditional-on-state is testable, labelable, and the
data is nearly free. Overnight-gap mean reversion is a documented
effect in Indian indices; conditioning on an orthogonal valuation
state is a genuine refinement over unconditional gap-fade stats. The
"insider footprint" framing maps to real, measurable proxies (futures
basis, OI shifts, late-day drift) — not mysticism.

**Risks**:
1. **Circularity** — if the state collapses to smoothed price, the
   whole edifice is momentum in a costume. Mitigation: the breadth /
   flows / macro channels must carry >= 50% of the composite weight,
   and an ablation (price-only state vs full state) is a mandatory
   acceptance gate.
2. **Story overfit on gaps** — every gap has a narrative in hindsight.
   Mitigation: the heads ship through the same harness as everything
   else (purged WF, Bonferroni/DSR in the arsenal, calibration audit,
   outcome-log resolution).
3. **Regime dependence** — gap behaviour in 2024's melt-up differs
   from 2026. The bucket-aging organ (M.8) and drift monitors apply
   unchanged.

## 4. Acceptance gates

- **Gate Q1 (anti-circularity)**: full-state GapBehaviorModel beats
  the price-only-state ablation by >= 0.03 AUC OOS, else the state
  isn't adding information and Q pauses.
- **Gate Q2 (calibration)**: P(gap fade) calibration error <= 0.10
  per confidence bucket over a 60-day OOS window.
- **Gate Q3 (economic)**: a gap-fade alpha through the arsenal
  (same costs, same nulls, deflated Sharpe) shows DSR > 0.90 on the
  index instrument before anything reaches the brief.

## 5. Data the founder must provide / approve

1. Index OHLCV (Nifty, BankNifty at minimum) — committed under T3.5.
2. F&O bhavcopy ingestion approval (free, NSE) — for basis/OI channel.
3. FII/DII daily flows (free) — Stream P item, now promoted.
4. Confirmation that crude + USDINR via free sources is acceptable
   for v1 (same fragility class as the existing Yahoo macro scrape).

## 6. Build order (once data lands)

1. `liqpool/valuation/state.py` — composite ValuationState, 3 frames,
   anchored at 100. Pure, causal, truncation-tested.
2. `liqpool/valuation/gap_model.py` — LGBMX heads + labels.
3. Arsenal alpha `gap_fade_valuation` for Gate Q3.
4. CCV macro-layer adapter (feature-flagged; old scrape stays the
   fallback).
5. Brief block + outcome-log prediction type `gap_behavior` so every
   published gap probability resolves next session like everything
   else.
