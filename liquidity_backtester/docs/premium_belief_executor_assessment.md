# Premium Belief Engine — Executor Assessment (founder-requested healing)

**Audit date**: 2026-06-19
**Auditor**: Opus 4.7 (Claude)
**Subject**: `liqpool/research/belief/executor.py` (Codex commits c0849c4 + b0a617e)
**Founder's report**: "very very poor, will lose money"

---

## Honest separation: what works vs what is structurally wrong

### What Codex got right (keep these)

1. **Shadow-only by construction.** `order_mode = "SHADOW_ONLY"`, `shadow_only = True`,
   `live_orders_enabled = False` baked in. A future broker adapter must opt-in
   separately. This is correct discipline.
2. **Position lifecycle** (entry → hold → cooldown → exit) with state tracked
   across calls.
3. **Guard categories are right**: warmup, dirty marks, abnormal slots, dirty
   sources, spread friendliness, no-trade score, unsafe IV state, unsafe
   battlefield verdict.
4. **Exit reasons partition into hard (immediate) vs soft (after min_hold)**.
   Correct two-tier exit pattern.
5. **High-water confidence + best-R / worst-R tracking** so a giveback can be
   computed.
6. **Cooldown after entry AND after exit** — prevents same-bar churn.
7. **Telemetry block is thorough** — directional votes, rail alignment, the
   slot-quality summary all surface for an auditor.

### What is structurally wrong (these are what would lose money)

#### Critical-1: R-multiple is computed on **spot**, not on the **option premium** held

```python
def _position_r(self, pos: ExecutorPosition, spot: float) -> float:
    direction = 1 if pos.side == "LONG" else -1
    spot_move = direction * ((spot - pos.entry_spot) / max(1e-9, pos.entry_spot))
    return spot_move / max(1e-9, self.cfg.adverse_spot_stop_pct)
```

The position is in an option contract (`CE_ATM`, `PE_ATM`, ...). The P&L,
the stop, the target, the profit-lock — ALL must be computed against the
**held contract's premium**, not the spot.

Why this matters in live:
- A flat spot for 30 bars can theta-bleed the premium 10-25%. Codex's logic
  sees `current_r ≈ 0`, holds → real position is bleeding while "R" stays
  near zero.
- An IV crush can drop the premium 20% on a 0.05% spot move. Codex sees
  `current_r` barely move; the option premium has just collapsed.
- A spot move that triggers the engine's `EXIT` (correct) might show
  `current_r > 0` on spot because the option leg moved differently from
  the underlying delta-expectation.

**Live impact**: hard stops fire late or never; profit locks fire on the
wrong metric; best_r / worst_r tracking is meaningless. This alone is a
"will lose money" defect.

#### Critical-2: `adverse_spot_stop_pct = 0.0025` (0.25%) is the wrong reference

NIFTY moves 0.25% several times per session, often within a single 15-min
candle. With this divisor:
- 0.5% adverse spot move = `2R` (= hard stop) — fires on noise.
- 1R = 0.25% spot adverse — too tight.

Combined with Critical-1 the R numbers are simultaneously meaningless
and miscalibrated.

#### Critical-3: No held-contract data quality

The executor receives `slot_readings` but does NOT pull the held
contract's own:
- `friendliness` (spread state on the held contract right now)
- `acceptance` (defended / rejected / normal on the held leg)
- `dod_z` (deviation-of-deviation on the held leg)
- `mark_source` / `mark_quality` (is the held quote even tradeable?)

The founder's *whole point*: **the spread can eat you alive**, and **the
held contract's acceptance flipping is the first sign to exit**. Neither
is checked. The executor exits on rail-level / battlefield-level signals
but is blind to its own contract.

### What is soft-wrong (will leak edge, not lose money outright)

1. **`min_directional_votes = 2`** with the decision's own direction counted
   as one of five votes → only 1 *additional* confirming source is required.
   For the founder's "multi-strike confirmation" insistence this should be
   `≥3` non-decision votes.
2. **`min_directional_rail_abs = 0.30`** — 0.30 robust-σ is barely above
   the chosen abnormality threshold (`anomaly_threshold = 1.5`); a position
   can open while neither rail is decisively aligned.
3. **`size_fraction` is dimensionless** — there is no rupees-per-R anchor.
   The operator cannot translate `size=0.7` to a lot count.
4. **No "engine went cold" exit** — if `is_warm` flips back to False mid-
   session, the executor still holds.
5. **No theta-aware exit** — see Critical-1; same root cause.
6. **No first-weakness exit on sweep scalps** — the decision says "exit at
   first weakness on call rail" but the executor ignores per-slot reads.

---

## The healing plan (this PR)

Surgical, not a rewrite. Keep Codex's lifecycle + guards + intent shape;
replace the broken P&L core with a correct one and add the missing reads.

### Changes

1. **New `HeldContractRead` payload** passed to `evaluate(snapshot, stream,
   contract_quote)` — carries the LIVE bid/ask/mark/spread_state/acceptance
   of the contract the position holds (or the contract the entry will use).
2. **R is now on premium**:
   ```
   R = (premium_now - entry_premium) × side / (entry_premium × premium_stop_pct)
   ```
   `premium_stop_pct` defaults: 35% for scalps, 50% for intraday (options
   are leveraged — a 50% drop on a ₹100 ATM CE is normal at 1R).
3. **Strike is now anchored**: position carries the actual strike number
   plus the label, so a broker adapter can route.
4. **Rupees-per-R anchor**: new config `risk_rupees_per_lot` + `lot_size`
   so `size_fraction` converts to `lots` deterministically.
5. **Held-contract guards**: if held contract's `spread_state == "dangerous"`
   → hard exit. If held contract's `acceptance` flips against the held
   side → soft exit.
6. **Engine-cold guard**: if `is_warm` flips False while holding → hard exit.
7. **First-weakness sweep exit**: on `profile == SCALP`, if the held leg's
   `dod_z` magnitude drops below the entry magnitude by a factor → soft exit.
8. **Stricter quorum**: `min_directional_votes` default raised, decision's
   own direction NOT counted, rail floor raised.

### Backwards compatibility

The existing tests Codex shipped pin **wrong behavior** (the spot-based R).
I will rewrite them to test the correct (premium-based) behavior, retaining
the test names where the *intent* still applies. Tests that pin the broken
behavior get marked deprecated and replaced.

### What this commit does NOT change

- The decision layer (Phase 8 of the engine) stays as-is — it's the right
  shape.
- The sentinel cockpit display Codex built stays — the display is fine,
  the executor was the issue.
- Shadow-only mode stays. No live orders.

### What still won't be perfect after this

This is honest: even with the heal, the executor cannot:
- model implied-vol-crush around expiry (would need an IV term-structure
  reader);
- handle gap risk (need overnight / news-day filters);
- simulate slippage (would need a fill model trained on Kite live fills).
These are explicitly out-of-scope for the healing pass — they would
need real Kite fill data to calibrate honestly.
