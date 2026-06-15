# Profitability Doctrine

Date: 2026-06-15

This document supersedes any prior commercial-launch priority until the founder explicitly reverses it. The project is currently a private profitability laboratory. Website, pricing, onboarding, and public delivery are secondary. The only primary question is:

> Does this machine help us preserve capital, understand engineered market behavior, and graduate real edges into carefully sized live risk?

## North Star

The system is not a signal bot. It is a market cognition and alpha-governance machine:

```
data -> state understanding -> hypothesis generation -> brutal validation
     -> shadow/paper trading -> curated graduation -> tiny live allocation
     -> continuous learning
```

The machine should become better at three jobs:

1. Detect when visible price action is engineered, sponsored, trapped, absorbed, or decaying.
2. Generate many explicit hypotheses from that state, each with an invalidation path.
3. Decide which hypotheses deserve zero capital, paper capital, tiny live capital, or retirement.

## Core Beliefs

### Markets Are Not Treated As Random

The working assumption is that Indian index and options movement often carries intent: FII pressure, DII cushioning, heavyweight index masking, premium decay, strike pinning, liquidity sweeps, and trap construction. These are not random-walk phenomena to a human trader, and they should not be flattened into random-walk assumptions by the machine.

### Random Data Is Only A Lie Detector

Synthetic random data is not proof that an alpha works. It is a negative control. If a hypothesis appears profitable on random garbage, then it is likely exploiting leakage, a simulator bug, or overfit. Real validation must come from chronological market data, paper trading, and live drift checks.

### Manipulation Is A State, Not A Candle

The system must learn manipulation as a full state vector:

- index location and gap context
- top-weight constituent positioning
- anchored VWAP relationships
- prior-session value area
- breadth and participation
- premium richness/cheapness by moneyness
- IV/OI behavior
- sector sponsorship
- proximity to liquidity pools
- session phase
- model probabilities
- cost and fill viability

A manipulation state can produce a trade alpha or an avoidance alpha. Avoidance alpha is real profit if it prevents bad trades.

### Capital Survival Beats Cleverness

No model, scientist, or mined rule touches serious capital directly. The path is:

```
REJECTED -> SHADOW -> PAPER -> TINY_LIVE -> TRUSTED -> SIZED
```

Any hypothesis can move down the ladder quickly. Moving up requires evidence.

## Non-Negotiable Gates

Every candidate that wants capital must pass:

1. Chronological walk-forward validation.
2. Explicit cost and slippage stress.
3. Null tests or negative controls appropriate to the signal.
4. Drawdown and loss-streak checks.
5. Paper/live drift comparison.
6. Decay kill-switch.
7. Position sizing bounded by bankroll survival.

Anything weaker is research-only.

## The Profitability Stack

### L0 Data

Own the raw evidence: equity bars, index bars, option quotes, option bars when available, fills, order book snapshots, constituent weights, macro calendars, and historical outcomes.

### L1 State Understanding

Convert data into market state: manipulation atlas, top-weight pressure, premium fair value, AVWAP defense, breadth, session regime, path shape, liquidity pool map, and cost/fill environment.

### L2 Hypothesis Generation

Scientists and miners generate hypotheses. Human ideas and machine-mined ideas both become structured `HypothesisSpec` objects, not ad hoc scripts.

### L3 Validation Harness

Every hypothesis is replayed through the same cost, fill, timing, and walk-forward rules. A hypothesis can win only after paying reality.

### L4 Shadow Ledger

Sentinel records what fired, what was skipped, what would have happened, what invalidated, and what the system learned.

### L5 Curator And Flywheel

The curator turns outcomes into findings. The flywheel turns findings into calibrated trust, decay, regret, and allocation.

### L6 Capital Governor

The governor decides size, not the alpha. It enforces daily loss, profit lock, max trades, max exposure, decay, and kill-switches.

## Priority Order

1. Real cost and slippage calibration from broker fills.
2. Manipulation Atlas for live state classification.
3. HypothesisMiner for systematic rule search.
4. Walk-forward + null + stress reporting for every candidate.
5. Sentinel scientists wired to the atlas.
6. Paper ledger and curator-driven graduation.
7. Tiny live capital with strict governor.

## What To Stop For Now

- Public launch polish.
- Pricing work.
- Payment gateway work.
- Resurrecting stale pocket names for emotional comfort.
- Adding features that no hypothesis engine consumes.
- Trusting options backtests without historical option OHLC/quote coverage.
- Letting any model output act without a ledger trail.

## Definition Of Progress

Progress is not a pretty dashboard. Progress is:

- one more false edge killed,
- one more cost assumption replaced with observed reality,
- one more manipulation state classified,
- one more paper hypothesis judged,
- one more bad trade avoided,
- one more tiny live edge proven without damaging the bankroll.

