# Sentinel Architecture — the organism, sequenced

> Not "we have organs." **"We know the order, and why the order
> exists."** This document is the systematization the founder demanded:
> every component, what it consumes, what it emits, which tier of trust
> its output carries, and the build sequence that turns the organic mess
> into a predictable machine.

---

## 0. The one-paragraph model

Sentinel is a **live experiment factory** wrapped around a Zerodha
account. The substrate is not OHLCV — it is the **Shadow Ledger**: a
rich, contextual, journey-complete record of every real, virtual,
rejected, and counterfactual decision. Organs (research models,
scientists, calibration, curator) feed on the ledger and feed back
into it. The **Orchestration spine** decides what data flows where and
which output is allowed to touch real money. The product *is* the
ledger and what the organism learns from it.

---

## 1. The trust tiers (the spine's core contract)

Every output in the system carries exactly one tier. This is the
single most important rule — it is what prevents a research signal
from ever silently moving real money.

| Tier | Meaning | Can place orders? | Shown to operator? | Logged to ledger? |
|---|---|---|---|---|
| `EXECUTION` | trusted enough to act | **yes** (trails/locks only) | yes | yes |
| `TRUSTED` | shown as a recommendation | no | yes | yes |
| `LOGGED` | recorded for research | no | optional | yes |
| `SHADOW` | research-only, never surfaced | no | no | yes |

The only `EXECUTION`-tier actors in v1 are the **trailing stop** and
the **profit lock** — and they only ever *exit* positions the operator
opened. No model output is `EXECUTION` tier until it has earned it
through the curator (months of validated calls), and even then it
exits, never enters.

---

## 2. The data-flow contract (where everything goes)

```
                        KITE (live)
                            │ quotes, positions, funds, chain
                            ▼
                    ┌───────────────┐
                    │  MARKET FEED  │  (poller, rate-limited)
                    └───────┬───────┘
                            │ ticks for 22-contract universe (ATM±5 CE/PE)
            ┌───────────────┼────────────────────────────┐
            ▼               ▼                            ▼
   ┌────────────────┐ ┌──────────────┐        ┌────────────────────┐
   │  GREEKS ENGINE │ │ CONTEXT LAYER│        │  PORTFOLIO STATE   │
   │ IV, Δ Γ Θ ν    │ │ (research    │        │ positions+Greeks   │
   │                │ │  codebase    │        │ pairs, exposure    │
   └───────┬────────┘ │  organs)     │        └─────────┬──────────┘
           │          └──────┬───────┘                  │
           │                 │                          │
           ▼                 ▼                          ▼
   ┌───────────────────────────────────────────────────────────┐
   │                    SCIENTISTS (candidate generators)        │
   │  finite, HIGH-QUALITY, context-aware virtual traders. each  │
   │  trade carries a VALIDATION: why this, why now, why it      │
   │  beats the market, what invalidates it.                     │
   └───────────────────────────┬───────────────────────────────┘
                                │ hypotheses (virtual + counterfactual)
                                ▼
   ┌───────────────────────────────────────────────────────────┐
   │                      SHADOW LEDGER                          │
   │  identity · greeks · context · hypothesis · journey ·       │
   │  judgment    (the substrate / the product)                  │
   └───────────────────────────┬───────────────────────────────┘
                                │ closed, judged experiments
                                ▼
   ┌───────────────────────────────────────────────────────────┐
   │                   CURATOR + CALIBRATION                     │
   │  reads the ledger → tunes KNOBS (entry threshold, target    │
   │  distance, stop distance, strike selection, hold time,      │
   │  confidence map, trail aggressiveness). re-runs history to  │
   │  test if the tune would have helped. NOT blind retraining.  │
   └───────────────────────────┬───────────────────────────────┘
                                │ validated knob deltas
                                ▼
                         ORCHESTRATION SPINE
                   (routes by trust tier; only EXECUTION
                    tier reaches the order path)
```

---

## 3. The organs (component register)

Each row: what it eats, what it emits, its trust tier, build status.

| Organ | Eats | Emits | Tier | Status |
|---|---|---|---|---|
| Market Feed | Kite | ticks | — | ✅ v1 (polling) |
| Greeks Engine | premiums | IV, Δ Γ Θ ν | — | ✅ `greeks.py` |
| Moneyness Identity | spot+chain | ATM-relative keys | — | ✅ this pass |
| Portfolio State | positions+quotes | enriched legs, pairs | TRUSTED | ✅ `portfolio.py` |
| Profit Lock (ratchet) | live P&L | locked/unlocked/fire | EXECUTION | ✅ this pass |
| Trailing Stop | premium ticks | exits | EXECUTION | ✅ `trails.py` |
| Scenario Engine | leg + scenario | payoff/heatmap/waterfall | TRUSTED | 🔜 wave 2 |
| **Shadow Ledger** | every decision | the substrate | LOGGED/SHADOW | ✅ this pass (schema) |
| Scientists | context+greeks | validated hypotheses | SHADOW→TRUSTED | 🔜 wave 2 |
| Context Layer | research codebase | regime/liquidity/levels | LOGGED | 🔜 bridge wave 3 |
| Equity Context | top-10 NIFTY stocks | weightage divergence | LOGGED | 🔜 wave 3 |
| Curator | ledger | improvement verdicts | — | 🔜 wave 2 |
| Calibration Engine | curator | knob deltas | TRUSTED | 🔜 wave 2 |
| Slippage Models | fills+ledger | Crux liquidity scores | TRUSTED | 🔜 wave 3 |
| VaR / Stress | portfolio+context | crisis simulation | TRUSTED | 🔜 wave 4 |
| Orchestration | all of above | tier-routed flow | — | ✅ this pass (spine) |

---

## 4. The Shadow Ledger schema (the institutional record)

Every event — real, virtual, rejected, alternative, model-suggestion,
counterfactual — carries five blocks. This is the founder's exact
specification, formalized:

**IDENTITY** — event_id, ts, session, underlying_price, instrument,
option_type, strike, expiry, premium, bid, ask, spread, volume, IV,
OI, delta, gamma, theta, vega, **moneyness_key** (the ATM-relative
identity that survives expiry rollover).

**CONTEXT** — trend, liquidity, volatility, opening_range, time_of_day,
support_resistance, model_signal, reaction_model, option_chain.
(Sourced from the research codebase organs once bridged.)

**HYPOTHESIS** — expected_underlying_move, expected_premium,
expected_horizon, suggested_entry, suggested_stop, suggested_target,
confidence, reason_codes, invalidation_conditions. *Every trade must
carry a validation: why this works, why it beats the market.*

**JOURNEY** — outcomes at T+1/3/5/10/15/30/60 min, MFE, MAE,
time_to_target, time_to_stop, premium_decay, slippage_estimate.

**JUDGMENT** — was_entry_good, was_exit_good, was_stop_too_tight,
was_target_too_far, was_confidence_calibrated, did_selection_outperform,
itm_vs_otm, atm_vs_otm, ce_pe_as_expected.

---

## 5. Moneyness-normalized memory (the Thursday solution)

The problem: on a new weekly's first day ("nil-hitty"), strike 23,100
has no history. The fix the founder named: **learn behaviour around
the ATM, not the strike level.** The ledger's `moneyness_key` is
`(option_type, atm_offset)` where offset ∈ {−5..+5}: ATM=0, +1 = first
OTM call / first OTM put, etc. A fresh ATM call inherits everything the
system learned about ATM-call *behaviour* under similar context. Strike
levels are forgotten; behaviour is remembered.

---

## 6. The Scientists (why finite-but-good beats infinite-but-dumb)

Infinite random virtual trades = 99% noise. The founder's rule:
**fewer trades, higher quality, every one carrying intent and
validation.** A scientist is a candidate generator that:

1. reads the full context (price, chart, regime, chain, equity-weight
   divergence),
2. forms a hypothesis with a stated reason it should beat the market,
3. states what would invalidate it,
4. emits at SHADOW tier into the ledger.

Only after the curator shows a scientist's calls are calibrated does
its output graduate toward TRUSTED. This is the quality gate that
keeps the laboratory honest.

---

## 7. The Calibration Engine (knobs, not blind retraining)

The ledger reveals truths like *"target usually 12.2 pts too far,"
"stop usually 8.2 pts too tight," "OTM works only when gamma expansion
starts early," "ATM better in chop," "confidence 0.7 actually wins
53%."* The calibration engine adjusts **knobs** — entry threshold,
target distance, stop distance, strike selection, hold time, confidence
map, partial-booking rule, trail aggressiveness — and then **re-runs
the ledger's stored journeys** to test whether the tune would have
helped *before* it goes live. This is safer than blind weight retrain,
which can overfit one weird day and ruin weeks.

**Training-in-live policy (decided):** weights are NOT updated intraday.
The ledger accumulates live; calibration proposes knob deltas; the
operator (or an overnight job) accepts them after the re-run test
passes. Live data *informs* immediately via knobs; it *retrains*
weights only on a deliberate cadence. This is the founder's own
instinct, adopted.

---

## 8. Build sequence (the "why this order" the founder asked for)

1. **Bedrock (this pass):** Shadow Ledger schema + Moneyness Identity +
   Orchestration spine + Profit-Lock ratchet. *Why first: every later
   organ reads/writes the ledger and routes through the spine. Build
   the nervous system and the spine before the muscles.*
2. **Wave 2:** Scientists (candidate generators) + Curator + Calibration
   + Scenario Engine visuals. *Why: now there is a ledger to fill and a
   spine to route through; the laboratory can run.*
3. **Wave 3:** Research-codebase context bridge + Equity context layer
   (top-10 weightage divergence) + Slippage models. *Why: enrich the
   context block once the loop is closed and proven.*
4. **Wave 4:** VaR / real-crisis stress simulator + customer-facing
   strategy auditor + nerf-tier SaaS gating. *Why: monetization and
   advanced research come after the engine demonstrably improves the
   founder's own trading for 30+ sessions.*

---

## 9. Safety invariants (never violated, any tier, any wave)

1. Only EXECUTION-tier actors place orders; in v1 that is trails +
   profit lock, and they only EXIT.
2. No scientist / model output is EXECUTION tier until curator-validated.
3. Dry-run default; `SENTINEL_CONFIRM_REAL=1` is the only real-order path.
4. Max-loss-allowed-today is a hard floor checked before any entry the
   operator makes is allowed to be trailed up.
5. The ledger is append-only; judgments are added, never overwritten.
6. Weights never update intraday; knobs may, after the re-run test.
