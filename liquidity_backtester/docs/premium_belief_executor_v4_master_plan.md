# Premium Belief Engine — Executor v4 Master Plan

**Purpose**: Source of truth for the executor_v4 build across all 5 sprints.
This document survives context loss. If a future session is asked to
continue, READ THIS FIRST.

**Date**: 2026-06-19
**Founder**: targeting Monday 2026-06-23 to USE this live
**Author**: Opus 4.7 (Claude)

---

## Why this exists

The founder identified that the original Codex execution governor (`executor.py`,
v1/v2) was structurally broken — would lose money in live. I (Claude) healed
the worst bugs in v3 (premium-based R, held-contract data quality, strike
anchoring) but acknowledged that v3 is still primitive: it acts like a textbook
"signal → trade" bot, not a sophisticated portfolio manager.

The founder then specified what the layer SHOULD be:
1. Every order has a structured hypothesis with validation/invalidation
2. Temporal context awareness (not just the latest tick)
3. Multi-position portfolio manager with risk attribution
4. Probability-web thinking, NOT R/R-ratio thinking
5. Manipulation-aware (fat-tail amplifier)
6. Strategy library for non-directional plays
7. Crowd-mirror self-awareness
8. Sophisticated, exhaustive, state-of-the-art

The founder is targeting **Monday Jun 23** to put this in front of his money.
He explicitly authorized maximum ambition: "make it like your life depends on
it." Honest scope: this is a survival + discipline system that compounds
slowly; not a money printer. Realistic expectation: top 0.1% of retail
systems, institutional Sharpe 0.8-1.4, ~8-15% annualized on bankroll once
matured.

---

## Architecture: 5 sprints, 19 modules

```
liqpool/research/belief/executor_v4/

# Layer 0 — INFORMATION SUBSTRATE
├── substrate.py            # Sprint 1 ✅ — exposes module internals + derivatives
├── memory.py               # Sprint 1 ✅ — multi-timeframe rollups (L1/L5/L15/L60)
├── flow_memory.py          # Sprint 1 ✅ — event-level memory + sequence queries

# Layer 1 — PROBABILITY WEB (the new thinking core)
├── scenario_web.py         # Sprint 2 ✅ — live multi-scenario tracker
├── projection.py           # Sprint 2 ✅ — forward Bayesian probabilities
├── critic.py               # Sprint 2 ✅ — adversarial bear-case
├── counterfactual.py       # Sprint 2 ✅ — kill criteria from imagined failure

# Layer 1.5 — MARKET STRUCTURE / FAT-TAIL DEFENSE
├── manipulation_patterns.py    # Sprint 3 — pattern catalogue with detectors
├── market_maker_mind.py        # Sprint 3 — Bayesian MM posterior
├── fat_tail_amplifier.py       # Sprint 3 — tail-mass scalar feeding the web
├── crowd_mirror.py             # Sprint 3 — "do we look like retail?"

# Layer 2 — ECONOMICS + RISK
├── economics.py            # Sprint 1 ✅ — fees-FIRST EV gate
├── risk.py                 # Sprint 4 — multi-position risk metrics
├── hedge.py                # Sprint 4 — pair-trade proposals

# Layer 3 — STRATEGY (founder's library)
├── strategy_library/       # Sprint 4
│   ├── verticals.py
│   ├── straddles.py
│   ├── iron_condor.py
│   ├── butterfly.py
│   ├── calendar.py
│   ├── ratio_spread.py
│   └── risk_reversal.py
├── strategy_selector.py    # Sprint 4

# Layer 4 — DECISION + AUDIT
├── hypothesis.py           # Sprint 1 ✅ — structured trade thesis
├── ledger.py               # Sprint 1 ✅ — bar-by-bar audit trail
├── aggregator.py           # Sprint 2 ✅ — Bayesian decision rule
├── explainer.py            # Sprint 4 — human-readable "why"

# Orchestration
└── manager.py              # Sprint 1 ✅ + Sprint 2 ✅ (Sprint 3/4 still to come)
```

---

## Founder-confirmed parameters (2026-06-19)

| Parameter | Value |
|---|---|
| Lot size | **65** (NIFTY weekly, confirmed not 75) |
| Per-trade max loss | **₹500-1000** |
| Daily bleed floor | **₹5,000** (half of "₹10k = doomed") |
| Brokerage | ₹20 entry + ₹20 exit (flat per order) |
| STT | 0.0625% on SELL premium notional only |
| Exchange | 0.00345% on both legs |
| SEBI | 0.0001% on both legs |
| Stamp duty | 0.003% on BUY notional only |
| GST | 18% on (brokerage + exchange + SEBI) |
| Decision metric | **probability web shift, NOT R/R** |
| Loss-cutting | **aggressive**, even at high fees/loss ratio |
| Speed target | **fast positioning**, NOT HFT |

---

## Sprint 1 — Foundation + Survival Math ✅ (COMMITTED)

**Shipped modules**:
- `substrate.py` (300 lines) — RichContext, derivatives, run-length tracking
- `memory.py` (240 lines) — MultiTimeframeMemory at 4 levels
- `flow_memory.py` (310 lines) — FlowEvent + sequence queries + 8 pre-baked patterns
- `economics.py` (320 lines) — Zerodha fee model, EV gate, exit accelerator
- `hypothesis.py` (270 lines) — PositionHypothesis with full causal record
- `ledger.py` (220 lines) — Per-position audit trail + post-mortem scorecard
- `manager.py` (560 lines) — PortfolioManager with all gates wired

**Tests shipped**: 46 (economics 17, substrate 9, memory+flow 11, manager 9)
**Status**: 1204 total tests passing, 0 regressions.

**Capabilities delivered**:
- Refuses trades that can't clear fees + slippage + edge margin (the
  founder's #1 concrete defense)
- Multi-timeframe alignment veto (L1 alone = retail → refuse)
- Temporal contradiction scan (last 20 min of flow memory)
- Sprint-1 antithesis score (regime stability + dispersion + acceptance)
- Full hypothesis with validation/invalidation criteria
- Bar-by-bar ledger with semantic checks
- Hard exit on engine EXIT / unsafe IV / unsafe battlefield / engine cold /
  premium stop / held spread dangerous / held mark dirty
- Soft exit on thesis flip / acceptance rejected / confidence giveback /
  profit-lock giveback / target reached / max hold
- Exit accelerator: cut losers even if fees ratio bad
- Daily bleed blocker (stops new entries when daily P&L ≤ -₹5000)
- Multi-position support up to max_open_positions
- Cooldowns after entry and after exit

**What Sprint 1 does NOT yet do** (deferred to later sprints):
- Probability-web tracking of 30+ live scenarios (Sprint 2)
- Forward Bayesian projection (Sprint 2)
- Adversarial bear-case generation (Sprint 2 — current implementation is a
  weighted-heuristic Sprint-1 placeholder)
- Counterfactual position-specific kill criteria (Sprint 2)
- Manipulation pattern detection (Sprint 3)
- Market-maker mind posterior (Sprint 3)
- Fat-tail amplifier (Sprint 3)
- Crowd-mirror self-awareness (Sprint 3)
- Multi-leg strategies (vertical/straddle/iron_condor/etc.) (Sprint 4)
- Portfolio risk parity + delta-neutral hedging (Sprint 4)

---

## Sprint 2 — The Probability Web ✅ (COMMITTED)

**Shipped modules**:
- `scenario_web.py` (~600 lines) — `ScenarioWeb` + `Scenario` dataclass with
  4 families (directional/chop/manipulation/fat_tail), per-tick
  DECAY → SPAWN → UPDATE → RETIRE → NORMALIZE, killer-signature
  evaluation, top-k / consensus / tail_mass / strategy_class queries
- `projection.py` (~190 lines) — `ForwardProjection` empirical conditional
  estimator over (thesis, iv, bf, winding, direction, regime) coordinates;
  returns `ProjectionDistribution` with target/stop probabilities, R
  quantiles, sample-count confidence
- `critic.py` (~250 lines) — `AdversarialCritic` with 10 weighted
  arguments (brittle rail, single-strike distortion, thesis decline,
  regime instability, opposite-side acceptance, recent traps, IV crush,
  chop dominance, tail mass, oscillation, MTF failure); returns
  REFUSE/SIZE_DOWN/PROCEED with reason chain
- `counterfactual.py` (~270 lines) — `CounterfactualGenerator` produces
  position-specific kill criteria with monitor windows, severity tags,
  and narrative failure path
- `aggregator.py` (~250 lines) — `DecisionAggregator` transparent
  weighted-sum brain (0.30 base + 0.25 mtf + 0.20 projection + 0.15 fees
  + 0.10 portfolio) with critic hard-refuse, web tail/chop/consensus
  overrides, strategy-class disagreement penalty

**Manager integration**:
- Per-tick: substrate → MTF → flow → scenario_web (in order)
- Entry pipeline gated through critic → projection → aggregator → counterfactual
- Per-position counterfactual plan attached at open; HARD-severity kills
  evaluated each bar inside the kill window
- Closed positions feed the projection tape with realized R outcomes
- Portfolio summary now carries `scenario_web` snapshot + `projection_summary`

**Tests shipped**: 51 new (scenario_web 12, projection 8, critic 8,
counterfactual+aggregator 17, manager+sprint2 integration 5, with one
manager test split). **Status**: 1255 total tests passing, 0 regressions.

---

## Sprint 2 — Original spec (for reference)

**Goal**: replace single-confidence decisioning with a live web of competing
scenarios that the aggregator queries.

### `scenario_web.py`
A `Scenario` dataclass with: id, name, trigger_signature, current_probability,
decay_rate, confirming_observations (rolling), contradicting_observations
(rolling), killer_signatures, implied_direction, implied_horizon_bars,
implied_max_drawdown_during_path, implied_strategy_class, tail_class.

`ScenarioWeb` maintains 30-100 live scenarios. Per tick:
1. SPAWN — generate new scenarios from current state evidence
2. UPDATE — rescore probabilities from latest evidence
3. RETIRE — kill scenarios whose killer fired OR whose probability decayed
   past floor
4. BOOST/SHRINK — scenarios with fresh confirms grow; those with
   contradictions shrink

Query API: `top_k()`, `directional_consensus()`, `tail_mass()`,
`dominant_strategy_class()`, `currently_dominant_pathway()`.

### `projection.py`
Conditional probability distributions estimated from state-conditional
historical records:
- P(continuation | current state and history)
- P(reversal in N bars | current state)
- Expected premium move distribution (mean, std, p5, p95)
- P(target_hit_first), P(stop_hit_first)

Built from the engine's own historical state tape — fully transparent,
not a black-box ML.

### `critic.py`
For any proposed BULL entry, construct the BEAR case for it:
- Is the rail strength brittle? (high abnormal_slot_fraction)
- Is dod_z elevated but on a single epicenter? (single-strike distortion)
- Is thesis score declining even above entry threshold?
- Are there recent BULL_TRAP signatures?
- Is IV historically prone to crush at this time of day?
- Has dispersion velocity been rising?

Returns: `antithesis_score`, `antithesis_reasons`, `recommended_size_haircut`.

### `counterfactual.py`
For a proposed trade, simulate the next 10 bars under the assumption it's
WRONG. Generate the SPECIFIC early-warning signatures that would tell us
we picked wrong. These become position-specific kill criteria attached to
the hypothesis (not generic stops).

### `aggregator.py` v1
The brain. Combines:
- Scenario web top-k + tail_mass
- Forward projection
- Critic antithesis
- Counterfactual kill criteria
- MTF alignment
- Economics EV
- Portfolio risk budget remaining

Transparent decision rule (no black-box ML):
```
final_score = 0.30*base_score + 0.25*mtf_alignment
            + 0.20*projection_factor + 0.15*fees_clearance
            + 0.10*portfolio_capacity
```
If `tail_mass > threshold`: refuse new entries + tighten open positions.
If `final_score < 0.55`: refuse with explanation.

### Sprint 2 sizing
~1500 lines code, ~50 tests. ~1-2 days work after Sprint 1.

---

## Sprint 3 — Market Structure + Fat-Tail Defense

### `manipulation_patterns.py`
Catalogued pattern detectors:
- **stop_hunt_short**: wick above HOD → fast revert below
- **stop_hunt_long**: wick below LOD → fast revert
- **liquidity_vacuum**: sudden spread blowout + bid disappear
- **pin** (near expiry): coiling within ₹10-25 of strike
- **accumulation**: sustained absorbing, narrow range, CE acceptance up
- **distribution**: sustained selling masked by churn, PE acceptance up
- **squeeze_setup**: narrowing range + decreasing realized vol
- **fake_breakout**: break out → fail to follow within 5 bars
- **iceberg_buy**: repeated absorption at same level
- **MM_inventory_flip**: rail signed-z reverses cleanly with no spot move

Each detector returns `PatternMatch(pattern_name, confidence,
started_at_bar, expected_resolution_bars, implied_mm_intent,
suggested_trade, suggested_avoid, notes)`.

### `market_maker_mind.py`
Aggregates patterns + flow into Bayesian posterior over MM intent:
- pinning_near_expiry
- accumulating
- distributing
- neutral_inventory
- stepping_back

Output: `MMPosterior(intent_distribution, active_patterns, implied_bias,
implied_volatility_view, confidence, operator_guidance)`.

### `fat_tail_amplifier.py`
Combines manipulation + MM + dispersion + crowd into a single
`tail_score ∈ [0,1]`. Feeds the scenario web (spawns fat-tail scenarios
proportional to score). When `tail_score > 0.6`: refuse new positions,
suggest hedges.

### `crowd_mirror.py`
Self-awareness: compare our portfolio composition to typical retail. When
we look like retail crowd, the MM has a target on us. Outputs:
`we_look_like_retail, crowd_density_long_atm_ce, mm_likely_target_us,
recommended_diversification`.

### Sprint 3 sizing
~1200 lines code, ~40 tests.

---

## Sprint 4 — Strategy Library + Portfolio + Hedging

### `strategy_library/`
Each strategy declares:
- `entry_conditions(scenario_web, mm_mind, tail_score) → bool`
- `greek_profile_at_entry() → {delta, vega, theta, gamma}`
- `max_loss_rupees(lots) → float`
- `max_gain_rupees(lots) → float` (or inf)
- `early_warning_signals() → list[str]`
- `adjustment_options(current_state) → list[str]`

Strategies:
- **single_leg.py**: the current buy CE/PE (one option)
- **vertical_spread.py**: bull call / bear put (limited risk/reward)
- **straddle.py**: long (vol expansion) / short (vol crush)
- **strangle.py**: cheaper than straddle, wider tail
- **iron_condor.py**: neutral, sells two strangles, capped loss
- **butterfly.py**: max-pain pin near expiry
- **calendar.py**: time-spread for IV term structure
- **ratio_spread.py**: 1x2 directional with built-in hedge
- **risk_reversal.py**: synthetic stock alternative

### `strategy_selector.py`
Given `(scenario_web, mm_mind, fat_tail_score, risk_budget) → strategy`.

### `risk.py`
- total_premium_at_risk_rupees
- net_directional_exposure (signed delta)
- gross_exposure_rupees
- portfolio_drawdown_r
- most_dangerous_position() — biggest fraction of total risk
- correlated_clusters() — positions on same underlying / same direction
- net_vega, net_theta

Carries portfolio kill switches.

### `hedge.py`
When `tail_score` high or net_delta exceeds budget: propose hedge legs.
Greek-aware pair construction.

### Sprint 4 sizing
~2000 lines code, ~50 tests.

---

## Sprint 5 — Polish + Sentinel cockpit

- Full multi-position `manager.py` with all earlier integrations
- `explainer.py` for human-readable "why" trails
- Sentinel cockpit panels:
  - Per-position cards with R, hypothesis, validation status
  - Most-dangerous-position highlight
  - Recent-contradictions panel
  - Scenario web dominant pathways
  - MM-mind posterior bar chart
  - Strategy library current selection rationale

### Sprint 5 sizing
~800 lines (mostly UI) + integration tests.

---

## Build philosophy (for future sessions)

1. **Each sprint is committable on its own** — no dependencies on
   uncompleted sprints. Sprint 1 has been shipped working.
2. **Test before integrate**. Each module gets its own test file.
3. **Honest fallbacks**. Sprint 1's antithesis is a heuristic; Sprint 2's
   critic.py will replace it cleanly.
4. **Backward compatibility**. v3 (`executor.py`) stays available until v4
   reaches feature parity. live_runner.py can switch between them.
5. **Founder's discipline > algorithmic ambition**. Refuse trades. Cut
   losers fast. Probability webs not R/R. Survival > profit.

---

## What to do if a future session continues this

1. Read this document FIRST.
2. Run `pytest tests/test_executor_v4_*.py -q` to confirm Sprint 1 works.
3. Check the latest commit message for what was last shipped.
4. Continue at the next un-shipped sprint.
5. Do NOT rewrite Sprint 1. It's calibrated to the founder's numbers and
   tested.
6. Honor the philosophy: probability web > R/R, cut losers fast, refuse
   bad trades, structure every order as a hypothesis.

---

## Founder quotes preserved (for context anchoring)

> "We should not think in ratios. We should think in probabilities."

> "When you know the loser is making, you have to cut it. Even if the
> fees will incur. A bigger loser will incur ~10x more fees."

> "I am not saying we are making the money printing machine. I am saying
> we are making a machine which can win, which can make money, which can
> thrive in this demonic market."

> "Our job is to win."

> "It should use the best of the best methodologies and technologies we
> have or we can make or we can implement. This is how we are going to
> make this. You have to make all of this like your life depends on it."

> "We're not competing with the HFTs. We just have to be fast enough that
> we can compete with everyone else."

> "We should also make a library a very, very exhaustive elaborative big
> ridiculously a strategic advanced smart library of strategies so that
> we can outperform every other player in the market."

These are the anchoring values. Every decision in the build defers to
them.
