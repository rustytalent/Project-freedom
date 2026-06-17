# Premium Belief Execution Governor

The Execution Governor is the shadow execution layer above the Premium Belief
Engine. Premium Belief reads the live option battlefield. The governor decides
whether that read is tradable, blocked, already in-position, or invalidated.

This layer is intentionally broker-agnostic. It never places orders. It emits a
serializable execution intent that a future broker adapter can consume only
after the intent stream has been audited.

## Why This Layer Exists

The raw belief action is not enough for execution. A live trader also needs:

- whether the data is clean enough to trust
- whether CE/PE rails, IV state, thesis, and stream direction agree
- whether a setup is blocked by dirty marks, abnormal slots, bad spreads, or
  no-trade danger
- what contract side and profile the system wants
- how large the shadow unit should be
- where the stop, target, trail, and max-hold boundaries are
- when an open position should be held or invalidated

The governor turns those requirements into one row per belief tick.

## Inputs

The governor consumes the same dictionary payload that the Premium Belief runner
writes into `extras.belief_snapshot`.

Required high-value fields:

- `decision.action`, `decision.direction`, `decision.confidence`
- `decision.trade_allowed`, `decision.spread_friendliness`
- `decision.strike.side`, `decision.strike.label`, `decision.strike.level`
- `iv_state.state`, `iv_state.direction`, `iv_state.clean_mark_fraction`
- `battlefield.verdict`, `battlefield.direction`, CE/PE rail signed z values
- `thesis.composite_state`, `thesis.no_trade_score`
- `slot_readings[].mark_source`, `friendliness`, `is_abnormal`,
  `mark_quality.confidence`
- optional `streaming_divergence.direction`

## Output Contract

Every emitted intent includes:

- `executor_name`
- `executor_version`
- `order_mode`
- `shadow_only`
- `live_orders_enabled`
- `intent`
- `action`
- `allowed`
- `direction`
- `confidence`
- `size_fraction`
- `contract_side`
- `contract_label`
- `contract_level`
- `profile`
- `stop_r`
- `target_r`
- `trail_after_r`
- `max_hold_bars`
- `invalidation`
- `reason_codes`
- `reject_reasons`
- `state`
- `telemetry`

`order_mode` is always `SHADOW_ONLY` in this implementation. If a future broker
adapter is added, it should require an explicit separate configuration and must
not infer live permission from this row.

## Intent Meanings

- `OPEN_LONG`: shadow-open an intraday long directional read.
- `OPEN_SHORT`: shadow-open an intraday short directional read.
- `OPEN_SCALP_CALL`: shadow-open a call scalp.
- `OPEN_SCALP_PUT`: shadow-open a put scalp.
- `HOLD_POSITION`: keep the current shadow position.
- `EXIT_POSITION`: close the shadow position because the thesis flipped, data
  quality failed, max hold expired, or an adverse spot move crossed the guard.
- `FLAT_WAIT`: no position and no acceptable entry action.
- `BLOCKED`: the belief action was not safe enough to execute.

## Entry Guards

The governor blocks entries when any of these are true:

- warmup is incomplete
- the belief decision explicitly disallows entry
- IV state is dirty, distorted, or common-shock
- battlefield says single distortion or volatility expansion
- clean mark fraction is below threshold
- average spread friendliness is below threshold
- abnormal slot fraction is too high
- dirty mark-source fraction is too high
- no-trade score is too high
- confidence is below the profile threshold
- directional vote count is too low
- CE/PE rail alignment is too weak

Directional votes come from decision direction, IV direction, battlefield
direction, thesis state, and optional streaming divergence direction.

## Lifecycle

The governor is stateful. Once it opens a shadow position, later rows are judged
against that position.

Exit triggers include:

- Premium Belief emits `EXIT`
- IV or battlefield becomes unsafe
- no-trade danger rises
- opposite directional votes form
- profile max-hold expires
- spot moves adversely beyond the configured guard

The state is deliberately simple: one shadow position at a time. This is correct
for the current cockpit because the UI is focused on one Premium Belief stream.

## Sizing

`size_fraction` is a normalized unit fraction, not quantity and not capital. It
combines confidence, spread friendliness, clean mark fraction, rail strength,
and directional vote count. The output is tiered:

- high conviction: `1.00`
- strong conviction: `0.70`
- acceptable conviction: `0.45`
- weak but allowed: `0.25`

Position quantity, rupee risk, brokerage constraints, and exchange rules belong
in a later broker/risk adapter.

## Live Command Example

Demo mode:

```bash
cd /root/Project-freedom/liquidity_backtester

PYTHONPATH=. .venv/bin/python -u scripts/run_belief_live.py \
  --demo \
  --underlying NIFTY \
  --levels 5 \
  --poll-seconds 0.25 \
  --warmup-bars 20 \
  --out-jsonl /root/.sentinel/liqpool_live_signals.jsonl \
  --executor-out-jsonl /root/.sentinel/belief_executor_intents.jsonl
```

Live Kite shadow mode:

```bash
cd /root/Project-freedom/liquidity_backtester

export KITE_API_KEY="..."
export KITE_ACCESS_TOKEN="..."

PYTHONPATH=. .venv/bin/python -u scripts/run_belief_live.py \
  --underlying NIFTY \
  --levels 5 \
  --poll-seconds 0.5 \
  --min-quote-gap-seconds 0.5 \
  --warmup-bars 80 \
  --out-jsonl /root/.sentinel/liqpool_live_signals.jsonl \
  --executor-out-jsonl /root/.sentinel/belief_executor_intents.jsonl
```

## Sentinel Display

Sentinel reads the executor intent from:

```text
latest.extras.executor
```

The cockpit can show intent, allowed/blocked, size, profile, stop/target,
reasons, order mode, and cost-wall status alongside the eight-phase belief map.

## Safety Contract

This layer is not a live trading bot yet. It is the missing middle layer between
market interpretation and order placement.

Before live orders, this intent stream must be audited against:

- stale feed detection
- missed tick behavior
- repeated-entry cooldowns
- broker reject simulation
- slippage and fill model
- max daily loss
- max position age
- disconnect handling
- duplicate order suppression
- manual kill switch

