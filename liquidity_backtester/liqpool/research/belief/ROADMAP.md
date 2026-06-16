# Premium Belief Engine — Roadmap

**Origin:** founder directive, 2026-06-16. A live option-battlefield
interpreter, not a candle/wick indicator.

> Normal traders watch where the underlying went. This engine watches
> whether the option battlefield *accepted* that move.

## The core question

Given the current underlying move, the time of day, each strike's
moneyness identity, the spread/liquidity state, the IV/skew pressure, and
the recent premium memory — are the 22 contracts (ATM ±5 × CE/PE)
behaving **normally**, or revealing **abnormal belief**?

## Founding corrections (must never be violated)

1. **Mark price, not wick.** The unit of analysis is a *premium response
   event* from the live quote stream, not an OHLC candle. Candles are
   summaries used only as multi-timeframe context.
2. **LTP is not truth.** Use microprice → mid → (LTP only as a fresh,
   in-book confirmation). A stale/locked/crossed quote must be flagged
   dirty and must not drive signals.
3. **Moneyness identity.** Each of the 22 slots has its own personality
   (Greeks). You cannot call a contract abnormal until you know what
   normal is *for that slot*.
4. **Deviation of deviation is slot-specific, time-specific,
   spread-specific.** A ₹0.70 residual means different things at ATM vs
   4-OTM, at 9:20 vs 2:55 expiry, at ₹0.05 spread vs ₹0.80 spread.
5. **Memory is non-negotiable.** Thesis scores with hysteresis so the
   engine does not flip like a scared retail trader.
6. **IV shock is a STATE, not a failure.** Classify common (both legs up)
   vs directional skew shock vs liquidity distortion vs dirty data.
7. **Multi-timeframe assimilation has ROLES, not equal votes.** 15m =
   context, 5m = structure, 1m = execution, tick = truth. The assimilation
   layer is where good data can become a bad interpretation — build it
   carefully.

## Build order (shadow mode first — no automation until late)

| Phase | Layer | Module | Status |
|------|-------|--------|--------|
| 1 | Logging engine | (quote frames) | substrate |
| 2 | **Mark-price + data quality** | `mark_price.py` | ✅ shipped |
| 3 | **Moneyness identity** | `moneyness.py` | ✅ shipped |
| 4 | Fair response model | `fair_response.py` | next |
| 4 | Residual / deviation-of-deviation (slot-specific) | `residual.py` | next |
| 4 | Spread & liquidity friendliness | `spread.py` | next |
| 5 | Multi-strike battlefield heatmap (σ maps) | `battlefield.py` | |
| 5 | IV / skew pressure state | `iv_state.py` | |
| 6 | Thesis memory (bull/bear/vol/liquidity/no-trade + hysteresis) | `thesis_memory.py` | |
| 7 | Winding-zone detector (4 types) | `winding.py` | |
| 7 | State machines (bull continuation / bear continuation / liquidity sweep) | `state_machines.py` | |
| 8 | Decision layer (avoid-trade + hold/exit + best strike) | `decision.py` | |
| 8 | Engine orchestrator (streaming) | `engine.py` | |
| 9 | Paper execution | | |
| 10 | Tiny live execution | | |

## Design notes captured so far

### Mark price (Phase 2 — done)
- `microprice = (ask·bid_qty + bid·ask_qty)/(bid_qty+ask_qty)` (Stoikov).
- Quality = `spread_weight·spread_score + (1-spread_weight)·depth_score`.
- Decision: microprice when quality ≥ floor AND spread ≤ good AND depth ok;
  else mid; else (crossed/locked/non-positive) → last good → fresh LTP →
  invalid.
- `MarkPriceTracker` remembers last good mark per contract for fallback.

### Moneyness identity (Phase 3 — done)
- Signed level: <0 ITM, 0 ATM, >0 OTM. Puts inverted.
- Expected |Δ| baseline table for ±5, extrapolated beyond.
- Behavior bands: future_like (≥0.85) / directional_itm (≥0.60) /
  gamma_atm (≥0.40) / convex_otm (≥0.18) / lottery_otm (<0.18).
- `detect_identity_anomaly` — the "ITM not behaving like ITM" tell:
  realized effective |Δ| matching a *different* slot's regime → flag.

### Winding zone (Phase 7 — design pinned, not built)
Inside an unfinished candle/segment, `X` = reference, `Y+`/`Y-` =
excursions. When price reaches an excursion and pauses, that region is a
*winding zone* — a temporary auction where premium decides
continuation vs reversal. Four types:
- **Bullish winding-up:** spot near upper range, CE strong, PE weak,
  spread clean, multi-strike CE agreement, pullbacks don't damage CE →
  likely continuation up.
- **Bearish winding-down:** mirror → likely continuation down.
- **Bull-trap winding:** spot near upper range but CE stops expanding, PE
  refuses to fall, spread worsens, CE residual fades → up move may fail,
  PE scalp.
- **Bear-trap winding:** mirror → CE scalp.
This is the entry/exit subsystem for scalping.

### Thesis memory hysteresis (Phase 6 — design pinned)
- entry confidence threshold = 70, exit = 45, no-trade danger = 65.
- Scores decay/update per bar; pullback that survives nicks the score a
  little; pullback that breaks structure cuts it hard.

### Honest scope
- Strong: bad-trade avoidance, fake-pullback holding, exit improvement,
  dirty-data avoidance.
- Possible after shadow testing: standalone scalper.
- Impossible: every-situation-proof (news/IV shocks).
