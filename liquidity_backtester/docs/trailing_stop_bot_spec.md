# Trailing-Stop Bot — Spec & Runbook

> The first piece of live trading infrastructure that runs alongside
> the founder's manual options trading, not as a replacement for it.
> Captures the operator's hard-won realisation: "I let very good
> profit turn into very medium profit because greediness kicks in
> after the position is already up."

## 1. The job, in one sentence

When you tap "trail this," the bot watches the live premium, tracks
the running peak, and exits at market the instant the premium falls
from peak by your specified rupee cushion — so a winning trade can
never round-trip while you're looking at something else.

## 2. Hard scope

**Does:**
- Watches a position you opened and tracks its peak premium.
- Fires a market exit through Zerodha when premium gives back ≥ cushion.
- Auto-flattens before 15:14 IST so MIS auto-square-off doesn't bite
  at worse prices.
- Persists state to a journal — crash + restart resumes active trails.
- Lets you cancel mid-flight or update the cushion mid-flight.

**Doesn't:**
- Decide entries (you do).
- Pick the cushion (you do, per position, in rupees).
- Analyse the market (you already do).
- Trade overnight.
- Touch anything you didn't explicitly arm.

## 3. Why cushion is in RUPEES, not %

A ₹300 give-back is one tick of noise on a fat ATM call running ₹250
premium. The same ₹300 is the whole prize on a deep OTM put running
₹15. Percentage cushions punish thin trades and over-protect fat ones.
Rupees scale naturally with the position size YOU chose. The CLI asks
you for it every time so you commit before you tap arm.

## 4. State machine

```
arm(symbol, side, qty, cushion, premium)
    │
    ▼
  ARMED ──────────────► peak update on every tick where premium beats peak
    │
    ├── premium - peak ≥ cushion  (LONG)
    │   peak - premium ≥ cushion  (SHORT, mirrored)
    │   OR clock ≥ 15:14 IST
    ▼
  EXIT_TRIGGERED  (exit_reason = "TRIGGER" | "SQUAREOFF")
    │
    ├── broker accepts ──► EXITED
    └── broker rejects ──► ERROR (preserves exit_error)

cancel(trail_id) on ARMED ──► CANCELLED
```

## 5. Safety rails baked in

1. **Dry-run is the default.** `--confirm-real-orders` is the only
   flag that turns on real placement, and it requires
   `KITE_API_KEY` + `KITE_ACCESS_TOKEN` to be set.
2. **Bad quotes don't fire.** A `None` or zero from the broker quote
   is treated as "skip this tick." Peak is never updated downward by
   a bad reading.
3. **No double fire.** The exit handler is idempotent on `state`;
   subsequent ticks on an EXIT_TRIGGERED trail are no-ops.
4. **Journal append-only.** Every state change is written. A crash
   mid-trade replays cleanly from disk.
5. **The bot only touches positions YOU arm.** Nothing automatic.

## 6. Files

- `liqpool/trading/trailing_stop_bot.py` — the bot module + journal.
- `scripts/trailing_stop_cli.py` — the operator CLI.
- `tests/test_trailing_stop_bot.py` — 18 tests, all transitions pinned.

## 7. Runbook

### Setup (once)

```bash
export KITE_API_KEY=...
export KITE_ACCESS_TOKEN=...          # generated via examples/zerodha_login.py
```

### Daily flow (every trading session)

1. **Start the bot in a tmux session at the start of the day:**

       tmux new -s trail
       python -m scripts.trailing_stop_cli serve --confirm-real-orders

   It runs all day, polling every 2 seconds, picking up arm/cancel/
   update commands from other shells via the journal file.

2. **You open positions normally** in the Zerodha app.

3. **When one runs in profit and greediness kicks in:**

       python -m scripts.trailing_stop_cli arm \
           --symbol NIFTY26JUN24500CE --side long --qty 75 \
           --cushion 30 --premium 195

   The bot starts watching from this premium. ₹30 cushion means:
   if premium peaks at 250, an exit fires at 220. If it peaks at 195
   (never moves up), an exit fires at 165.

4. **Check what's active:**

       python -m scripts.trailing_stop_cli list

5. **Cancel if you want to ride further:**

       python -m scripts.trailing_stop_cli cancel --trail-id TRAIL_X_...

6. **Tighten as the trade keeps running** (lock in more):

       python -m scripts.trailing_stop_cli update --trail-id TRAIL_X_... \
           --cushion 50

7. **At 15:14 IST**, the bot auto-flattens every still-armed trail at
   market. You don't need to do anything — but you can also flatten
   manually before then.

## 8. What it explicitly doesn't do at v1

- **No WebSocket streaming.** v1 polls every 2s — slightly higher
  latency, hugely simpler to reason about. Move to streaming in v2
  if the latency proves expensive on tight trails.
- **No multi-leg awareness.** Each trail is one symbol. If you trade
  a spread, you arm two trails. v2 could couple them.
- **No partial exits.** Cushion fires → entire quantity exits at
  market. Scaling out is a v2 feature once we know what scaling
  rule the operator actually wants (50%/25%/25% etc).
- **No "BSL" / "trail tightener as it runs."** The cushion stays
  fixed unless you `update`. Auto-tightening is tempting but tends
  to choke out winners; v2 can add it as an opt-in mode.

## 9. Failure modes & how the bot handles them

| Failure | Behaviour |
|---|---|
| Broker quote returns None | Skip tick; preserve peak; no fire |
| Broker rejects exit order | `state=ERROR`, `exit_error=<msg>`, trail removed from active set so it doesn't retry every 2s |
| Bot crash mid-session | Journal replay on next start recovers all ARMED + EXIT_TRIGGERED trails |
| Token expires mid-session | First failed quote/order surfaces error; trail goes to ERROR; operator restarts |
| Wall clock passes 15:14 IST | Every ARMED trail auto-fires with `exit_reason=SQUAREOFF` |

## 10. The flywheel angle (later)

Every armed trail is a labelled data point about the operator's
intuition: "I thought this position was at peak, and I gave it ₹X to
prove me wrong." When fired, we know peak premium, trigger premium,
time elapsed, and (eventually) the underlying move attribution. That
joins the shadow log naturally as a new event kind in a future pass:

    EVENT_KIND_MANUAL_TRAIL_FIRED

trained on, this becomes a "when does the operator's eye get the
peak right" signal — a small organ but a high-signal one, because
every label is the operator's own verified-by-money judgment.

Not built today. Logged here so it isn't forgotten.
