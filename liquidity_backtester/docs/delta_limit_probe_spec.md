# Delta-Implied Limit Probe — Stream R specification

> The founder's options-entry method, formalised into a self-labelling
> data flywheel. Origin: founder's live options trading, 2026-06.

## 1. The founder's method (preserved, math corrected)

When the founder forms a directional view on the underlying, he does
NOT enter at market. He converts the view into a Greeks-implied target
premium and places a **limit order there**. Worked example (corrected):

- Underlying view: price rises 30 points.
- Option: a put with premium 20, delta ~= -0.5.
- Greeks implication: put premium falls ~= |delta| * move = 0.5 * 30
  = 15 → target premium ~= 5.
- Action: place a limit BUY for that put at 5.

The insight — and it is a real one:

> **If the limit fills, the market itself confirmed the underlying
> prediction. If it never fills, the prediction was wrong — and the
> closest the premium came tells you BY HOW MUCH it was wrong.**

Every such order is a **falsifiable experiment the market scores for
free**. The fill / no-fill / how-close signal is a calibration label
for the underlying price-move magnitude prediction. Collect enough of
them and you can reverse-engineer a well-calibrated "how far will it
move" model from nothing but limit-order outcomes.

## 2. Why this is genuinely valuable

1. **Free, clean, abundant labels.** No simulator assumptions — the
   exchange resolves each probe. Fill = move >= implied; gap-to-fill =
   magnitude error. This is the cleanest label source in the whole
   engine.
2. **Magnitude calibration, not just direction.** Our current heads
   say P(touch) / P(up). They are weak on HOW FAR. The probe outcome
   is a direct magnitude signal: the premium only reaches the limit if
   the underlying moved the predicted distance.
3. **It is the missing inverse of the options ER model.** The existing
   `OptionsExpectedReturnModel` predicts premium return from features.
   The probe goes the other way: observed premium behaviour →
   calibrated underlying-move estimate. The two can cross-check.
4. **Execution-native.** It fuses prediction and execution into one
   act — the order IS the hypothesis test.

## 3. Engineering translation

### 3.1 The probe object

    DeltaLimitProbe:
        underlying, strike, option_type, expiry
        entry_premium, delta_at_entry, gamma_at_entry
        predicted_underlying_move        # signed, ATR or points
        implied_target_premium           # entry + delta*move (+ 0.5*gamma*move^2)
        limit_price                      # = implied_target_premium
        placed_at, deadline_bars

Note the gamma term: for larger predicted moves the linear
delta-only target is biased; include the 0.5*gamma*move^2 convexity
correction so the implied target is honest for big predictions. The
founder's instinct (delta*move) is the first-order version; gamma is
the second-order fix that matters most exactly when the move is large
enough to be worth trading.

### 3.2 Outcome labels (resolver)

For each probe at its deadline:

    FILLED              -> move_magnitude >= predicted (direction right,
                           size >= predicted)
    NOT_FILLED          -> closest_premium gives the magnitude shortfall:
                           implied_realised_move = (entry - min_premium_seen)
                                                   / |delta|   (gamma-adjusted)
    magnitude_error     = predicted_move - implied_realised_move

This is exactly the Stream L shadow-log pattern (a decision + a
counterfactual the next bars resolve), specialised to options. It
SHOULD be implemented as a new shadow event kind so it rides the
existing collector + replay machinery:

    EVENT_KIND_DELTA_LIMIT_PROBE = "delta_limit_probe"

and a resolver in `replay_shadow_counterfactuals.py` that reads the
option's premium path to the deadline.

### 3.3 The model it trains

`MoveMagnitudeCalibrator` (a flywheel organ, M.11):

    input:  underlying-move PREDICTION (from direction/proximity heads)
            + context (vol regime, dte, distance, time-of-day)
    target: implied_realised_move from probe outcomes
    output: calibrated expected move magnitude + conformal band

This closes a real gap: it turns the engine's directional heads into
*magnitude-calibrated* heads using labels the market gives for free.
Feeds straight back into:
- the options ER model (better magnitude → better premium-return est)
- the brief (P(touch) gains a companion "expected move: X ATR [lo-hi]")
- position sizing in V3 (magnitude conviction → size)

### 3.4 Where it plugs into the organism

- **Shadow log**: new event kind, no new infrastructure.
- **Replay resolver**: new entry in the RESOLVERS dict (the pattern
  Stream L was built to extend).
- **Flywheel hub**: M.11 joins as an eighth organ; same fit/serve/
  starve-safe contract.
- **Options executor**: the probe IS the entry mechanism — this is
  how the executor places orders in live mode, not a separate system.

## 4. Honest assessment

**Strong**: the self-labelling property is rare and real. Most ML in
trading struggles for clean labels; this manufactures them as a
byproduct of trading. The magnitude-calibration angle attacks a
genuine weakness (our heads are direction-strong, magnitude-weak).

**Risks**:
1. **Fill ≠ pure prediction confirmation.** A limit can fill from an
   IV spike or a vol-crush unrelated to the directional move. The
   resolver MUST decompose: was the premium move delta-driven (price)
   or vega-driven (IV)? Without this the labels are contaminated. This
   is the single hardest part and the make-or-break of Stream R.
   Mitigation: record IV at placement and at fill; attribute the
   premium change to delta vs vega vs theta before labelling.
2. **Liquidity.** Indian option strikes away from ATM have wide
   spreads; a limit at the implied target may sit behind a wall and
   never fill even when the prediction was right. The fill model
   (K.2) must condition the label — "didn't fill" is ambiguous
   between "wrong prediction" and "right prediction, no liquidity".
3. **Survivorship in the label set.** Only placed probes generate
   labels; predictions never converted to probes are missing. Log
   the *hypothetical* probe for every directional view, not just the
   ones actually traded, so the label set isn't biased toward
   high-conviction calls.

## 5. Acceptance gates

- **Gate R1 (attribution)**: the resolver correctly decomposes >= 80%
  of premium moves into delta/vega/theta on a labelled validation set,
  else the labels are too contaminated to train on.
- **Gate R2 (calibration)**: MoveMagnitudeCalibrator's predicted move
  vs implied-realised move has calibration error <= 0.15 ATR across
  deciles on OOS probes.
- **Gate R3 (lift)**: magnitude-calibrated sizing in V3 beats
  flat-conviction sizing by >= 0.10 R on a 60-day OOS replay.

## 6. Dependencies

- Per-strike option premium + Greeks time series (the options
  warehouse — partially have it; intraday Greeks are the gap, same
  one the options ER model already faces).
- Live order book / fill data once live (until then, simulate fills
  against the historical premium path with the K.2 fill model).
- IV time series per strike for the delta/vega attribution (Gate R1).

## 7. Build order

1. Add `EVENT_KIND_DELTA_LIMIT_PROBE` to shadow_log.py.
2. `liqpool/options/delta_probe.py` — the probe object + implied
   target math (with gamma correction).
3. Premium-path resolver in replay_shadow_counterfactuals.py with
   delta/vega/theta attribution (Gate R1 lives here).
4. `liqpool/flywheel/move_magnitude.py` — M.11 organ.
5. Wire as the options executor's live entry mechanism.
6. Brief: attach "expected move [lo-hi]" to options strike entries.
