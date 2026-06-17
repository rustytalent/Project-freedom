# Premium Belief Live Runbook

This runbook turns the Premium Belief Engine into a standalone shadow
software process. It can run by itself from Kite credentials and can also
feed Sentinel through the shared `ModelSignal` JSONL stream.

## What Claude Shipped

The engine is split into phases:

1. Phase 2: mark price and data quality in `liqpool/research/belief/mark_price.py`
2. Phase 3: moneyness identity in `liqpool/research/belief/moneyness.py`
3. Phase 4: fair response, residual, deviation-of-deviation, and spread friendliness
4. Phase 5: multi-strike battlefield and IV state classification
5. Phase 6: thesis memory with hysteresis
6. Phase 7: winding zones plus bull, bear, and liquidity-sweep state machines
7. Phase 8: decision layer and engine orchestrator
8. Phase 9 and 10: reserved for paper execution and tiny live execution

The live runner in `scripts/run_belief_live.py` wires phases 2 to 8 to Kite.
It is shadow-only. It never places orders. It also has `--demo` mode for
market-closed terminal rehearsals.

## Inputs

Required environment variables:

```bash
export KITE_API_KEY="..."
export KITE_ACCESS_TOKEN="..."
```

Required live market access:

- Kite `NSE:NIFTY 50` spot LTP
- Kite `NFO` instruments dump
- Kite full quotes for nearest-expiry NIFTY ATM +/- 5 strikes, both CE and PE

Normal ticks use one batched Kite `quote` call containing spot plus the option
contracts. Contract-refresh ticks may take an extra guarded quote call.

## Output

The runner writes append-only JSONL rows. Each row is a shared
`ModelSignal` with the full belief snapshot in `extras.belief_snapshot`.

Default path:

```bash
/var/lib/sentinel/liqpool_live_signals.jsonl
```

Sentinel already tails `<SENTINEL_JOURNAL_DIR>/liqpool_live_signals.jsonl`
through `sentinel.liqpool_bridge.LiveSignalsTail`. If your Sentinel service
uses `SENTINEL_JOURNAL_DIR=/root/.sentinel`, pass
`--out-jsonl /root/.sentinel/liqpool_live_signals.jsonl`.

## Run Standalone

From `liquidity_backtester`:

```bash
mkdir -p logs /var/lib/sentinel

nohup env PYTHONPATH=. \
  KITE_API_KEY="$KITE_API_KEY" \
  KITE_ACCESS_TOKEN="$KITE_ACCESS_TOKEN" \
  .venv/bin/python -u scripts/run_belief_live.py \
  --underlying NIFTY \
  --spot-key "NSE:NIFTY 50" \
  --exchange NFO \
  --strike-step 50 \
  --levels 5 \
  --poll-seconds 1 \
  --min-quote-gap-seconds 1.05 \
  --warmup-bars 80 \
  --terminal \
  --out-jsonl /var/lib/sentinel/liqpool_live_signals.jsonl \
  > logs/belief_live.log 2>&1 &

echo $! > logs/belief_live.pid
cat logs/belief_live.pid
```

## Watch It

```bash
cd /root/Project-freedom/liquidity_backtester
PID="$(cat logs/belief_live.pid)"

watch -n 5 '
date
ps -p '"$PID"' -o pid,stat,pcpu,pmem,rss,etime,args 2>/dev/null || echo "belief runner stopped"
echo
tail -40 logs/belief_live.log 2>/dev/null
echo
echo "LATEST SIGNALS"
tail -5 /var/lib/sentinel/liqpool_live_signals.jsonl 2>/dev/null
'
```

## Market-Closed Terminal Test

Use this when the market is closed. It uses synthetic option quotes, so it
tests the software path, not real profitability.

```bash
mkdir -p logs /tmp/sentinel_demo

PYTHONPATH=. .venv/bin/python -u scripts/run_belief_live.py \
  --demo \
  --terminal \
  --underlying NIFTY \
  --levels 5 \
  --poll-seconds 0.5 \
  --warmup-bars 20 \
  --max-ticks 120 \
  --out-jsonl /tmp/sentinel_demo/liqpool_live_signals.jsonl
```

If you want to watch the file separately:

```bash
tail -f /tmp/sentinel_demo/liqpool_live_signals.jsonl
```

## Test Without Infinite Loop

```bash
PYTHONPATH=. .venv/bin/python -u scripts/run_belief_live.py \
  --demo \
  --max-ticks 3 \
  --out-jsonl /tmp/belief_live_test.jsonl
```

## Sentinel Integration

Sentinel does not need to import the belief engine. It only tails the
JSONL file:

```python
from pathlib import Path
from sentinel.liqpool_bridge import LiveSignalsTail

tail = LiveSignalsTail(
    Path("/var/lib/sentinel/liqpool_live_signals.jsonl"),
    publisher=sentinel.publisher,
)
sentinel.add_tick_hook(tail.poll)
```

This keeps the engine independent and lets the same stream work for:

- standalone terminal operation
- Sentinel cockpit display
- shadow ledger replay
- later paper execution

## Safety

The current runner is shadow-only. A future execution adapter must not be
added until the JSONL stream has been shadow-tested, audited for dirty quote
behavior, and reviewed against real trade outcomes.
