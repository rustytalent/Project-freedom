# Premium Belief Cockpit Guide

This guide explains what the Sentinel Premium Belief tab is showing, what the live
engine is doing, and what it is not doing yet. Read this before using the cockpit
as a live trading aid.

## One Sentence

Premium Belief is a shadow-only options battlefield reader. It polls live Kite
quotes, converts the option book into clean marks, studies how CE and PE rails
are accepting or rejecting the underlying move, keeps a thesis memory, and emits
a decision such as `HOLD`, `WAIT`, `NO_TRADE`, `ENTER_LONG`, `ENTER_SHORT`,
`SCALP_CALL`, `SCALP_PUT`, or `EXIT`.

It does not place orders.

## Data Reality

The current live runner is not true exchange tick-by-tick streaming.

Current data path:

1. Kite spot quote for the configured underlying, normally `NSE:NIFTY 50`.
2. Kite option-chain instruments for the nearest expiry.
3. Batched Kite full quotes for ATM plus or minus `levels` strikes, both CE and PE.
4. Per-contract mark price from quote depth:
   - first choice: microprice from best bid, ask, and visible depth,
   - second choice: mid price,
   - fallback: fresh LTP only when book data is not clean,
   - refusal: stale, locked, crossed, missing, or dirty quote conditions.

The browser tab refreshes once per second. That refresh is only the human display
loop. It must not be confused with execution timing. If live execution is ever
added, the execution loop should run inside the backend runner or a separate
order adapter, not inside the browser.

## Why The Browser Refresh Is One Second

The live runner's normal Kite polling config is around one second with a
`min_quote_gap_seconds` guard. A faster browser refresh can make the UI feel busy,
but it cannot create more market truth than the runner writes to the JSONL stream.

Use this mental model:

- Runner polling speed decides how often the engine receives market information.
- Sentinel tailing speed decides how fast the cockpit sees the latest engine row.
- Browser refresh speed decides how fast your screen redraws.
- Order execution speed, if added later, must be a separate audited backend loop.

For monitoring, one second is useful. For execution, the browser is the wrong
place to depend on.

## The 8 Phase Engine Map

The cockpit now shows each phase with a short description and metric pills. This
is not decoration. Each phase is a guardrail in the live decision chain.

### Phase 1: Logging + Live Bus

What it does:

The standalone belief runner writes a `ModelSignal` JSON row. Sentinel tails that
stream and displays the row. This phase confirms that the engine row has reached
the cockpit.

What to watch:

- `bars`: how much history the live engine has accumulated.
- `contracts`: how many option contracts were selected.
- `warm`: whether the engine has passed warmup.

If this phase is missing, the cockpit is not seeing the runner.

### Phase 2: Mark Price + Data Quality

What it does:

This is the engine's truth filter. It refuses to treat a bad LTP print as truth.
It builds a tradable mark from microprice or mid when possible. If the book is
dirty, the engine should block directional interpretation.

What to watch:

- `clean`: fraction of selected option slots using clean microprice or mid marks.
- `friendly`: spread friendliness.
- `IV conf`: confidence of the IV/skew classifier.

Interpretation:

- High clean mark fraction is necessary.
- Low spread friendliness means live execution is dangerous even if direction
  looks attractive.
- Dirty data is a system condition, not a market setup.

### Phase 3: Moneyness Identity

What it does:

Every contract is assigned a behavior identity:

- future-like ITM,
- directional ITM,
- gamma ATM,
- convex OTM,
- lottery OTM.

The decision layer uses this identity when it recommends `CE_ATM`, `PE_ATM`, or
another strike class.

What to watch:

- selected strike label,
- selected side,
- selected level.

Interpretation:

ATM is usually the maximum gamma read. ITM is often cleaner but less convex. Deep
OTM can be informative but is noisier and easier to distort.

### Phase 4: Residual + Spread Read

What it does:

This phase asks: did option premium move the way it should have moved given the
underlying? It compares observed option mark change against a fair response and
computes deviation-of-deviation.

What to watch:

- `CE z`: call rail signed deviation.
- `PE z`: put rail signed deviation.
- `disp`: dispersion, meaning whether one strike is carrying the signal alone.

Interpretation:

Broad CE strength with broad PE weakness can support bullish belief. Broad PE
strength with broad CE weakness can support bearish belief. High dispersion means
one strike is screaming while the rest is not, so the read may be a distortion.

### Phase 5: Battlefield + IV State

What it does:

This phase combines the CE rail and PE rail into one battlefield verdict. It also
separates directional skew from common IV shock and dirty liquidity.

Important states:

- `directional_bull`: calls and puts agree in a bullish way.
- `directional_bear`: calls and puts agree in a bearish way.
- `common_shock`: both rails are expanding. Direction is unclear.
- `vol_contraction`: quiet book.
- `single_distortion`: one slot or a narrow pocket is distorting the rail.
- `liquidity_distortion` or `dirty_data`: do not trust the book.

What to watch:

- battlefield verdict,
- battlefield confidence,
- IV state,
- IV confidence,
- direction.

Interpretation:

This is the main "are options accepting or rejecting the underlying move" panel.

### Phase 6: Thesis Memory

What it does:

This phase prevents the engine from flipping every tick. It keeps persistent
scores:

- bull thesis,
- bear thesis,
- vol expansion,
- liquidity danger,
- no-trade danger.

It uses hysteresis. Entry requires a stronger score than exit. That means a thesis
can survive small pullbacks instead of dying instantly.

What to watch:

- bull thesis,
- bear thesis,
- no-trade danger,
- composite state,
- held direction.

Interpretation:

This is the engine's memory. A single tick can influence it, but a single tick
should not dominate it.

### Phase 7: Winding + State Machines

What it does:

This phase tracks stories, not just snapshots:

- bull continuation,
- bear continuation,
- liquidity sweep.

The winding detector looks for paused excursions around upper or lower bands,
including trap and continuation zones.

What to watch:

- winding zone,
- winding confidence,
- bull machine state,
- bear machine state,
- sweep machine state.

Interpretation:

This is the part that starts to look like a human trader's narrative: impulse,
pullback, survival, winding, continuation, damage, invalidation.

### Phase 8: Decision Layer

What it does:

This layer applies precedence:

1. no-trade rules,
2. exit rules,
3. scalp winding rules,
4. entry rules,
5. hold rules,
6. wait.

What to watch:

- action,
- trade allowed,
- confidence,
- strike,
- no-trade reason,
- exit rule,
- invalidation rule.

Interpretation:

This is the final shadow answer. It is useful as a cockpit decision aid, but it is
not yet an automated order system.

## Cockpit Panels

### Premium Belief Engine

This is the top decision panel. It shows the latest action, spot, confidence,
warmup state, bars seen, direction, selected strike, battlefield verdict, and IV
state.

The data path line tells you that the current engine is quote-polling and
shadow-only.

### Battlefield Rails

This panel is intentionally placed where your eye naturally goes first on the
right side. It shows CE and PE rail summaries:

- signed z,
- rail state,
- absolute z,
- abnormal fraction,
- dispersion,
- epicenter.

Use it to answer: is the move accepted broadly, rejected broadly, or distorted by
one strike?

### Live Trace

This chart shows normalized spot range, decision confidence, and battlefield
confidence. Spot is normalized so it can be viewed on the same chart as
probabilities.

### Thesis Memory Trace

This chart shows bull, bear, no-trade, and vol-expansion scores. This is the best
place to see whether the system is building conviction or just flickering.

### Mark Quality + IV

This chart shows clean mark fraction, spread friendliness, and IV confidence.
If this chart is unhealthy, do not over-trust directional conclusions.

### Rail Pressure

This chart shows CE signed z, PE signed z, and both rail dispersion measures.
It is the compact battlefield history.

### Selected Contracts

This is a supporting table, not the main decision object. It tells you which
nearest-expiry contracts the engine is reading.

### Raw Snapshot

This is for debugging. It exposes the exact JSON snapshot that Sentinel received.

## Phase-4 Slot Heatmap Contract

The cockpit now receives a compact `slot_readings` array from
`BeliefSnapshot.to_dict()`. Each row is one CE or PE moneyness slot and is the
atomic evidence behind the aggregate battlefield rails.

Each slot row includes:

- `strike`,
- `option_type`,
- `moneyness_label`,
- `level`,
- `behavior`,
- `expected_abs_delta` and `expected_signed_delta`,
- `mark_source`,
- `mark_quality` and `mark_quality_label`,
- `mark_price`, `mark_spread`, and `mark_spread_pct`,
- `ltp_confirms` and `mark_flags`,
- `friendliness`,
- `spread_state`,
- `acceptance`,
- `dod_z`,
- `is_abnormal`.

Sentinel also exposes the same array at the top level of `/api/premium_belief`
as `slot_readings`, so the UI does not need to dig into `raw_snapshot`.

### Reading The Heatmap

Green cells mean premium is defending above the fair-response path. Red cells
mean premium is rejecting below the fair-response path. Amber or dim cells mean
the read is contaminated by mark quality or spread friendliness and should be
treated as weaker evidence. An outlined cell means the slot crossed the
abnormality threshold.

This is the first cockpit surface that lets the operator see whether the rail
verdict is broad agreement across the chain or just one noisy contract.

## Practical Operating Rules

1. Do not trust the decision when warmup is false.
2. Do not trust direction when clean marks are weak.
3. Treat common IV shock as no directional read.
4. Treat single-strike distortion as a warning, not an entry.
5. Treat thesis memory as the main continuity layer.
6. Treat battlefield rails as the main acceptance or rejection layer.
7. Treat the browser as monitoring only.
8. Do not add live orders until the shadow stream has been audited against real
   outcomes.

## Run Reminder

Market-closed demo:

```bash
cd /root/Project-freedom/liquidity_backtester

PYTHONPATH=. .venv/bin/python -u scripts/run_belief_live.py \
  --demo \
  --terminal \
  --underlying NIFTY \
  --levels 5 \
  --poll-seconds 0.5 \
  --warmup-bars 20 \
  --max-ticks 300 \
  --out-jsonl /root/.sentinel/liqpool_live_signals.jsonl
```

Sentinel browser:

```bash
cd /root/Project-freedom
source .venv_sentinel/bin/activate

SENTINEL_DEMO=1 \
SENTINEL_TOKEN="your-token" \
SENTINEL_JOURNAL_DIR="/root/.sentinel" \
uvicorn sentinel.server:app --host 127.0.0.1 --port 8800
```

Then open:

```text
http://127.0.0.1:8800/console
```

If running through an SSH tunnel from the Mac:

```bash
ssh -L 8800:127.0.0.1:8800 crux-live
```

Then open the same local URL on the Mac.
