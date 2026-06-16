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
| 4 | **Fair response model** | `fair_response.py` | ✅ shipped |
| 4 | **Residual / deviation-of-deviation (slot-specific)** | `residual.py` | ✅ shipped |
| 4 | **Spread & liquidity friendliness** | `spread.py` | ✅ shipped |
| 5 | **Multi-strike battlefield heatmap (σ maps)** | `battlefield.py` | ✅ shipped |
| 5 | **IV / skew pressure state** | `iv_state.py` | ✅ shipped |
| 6 | **Thesis memory (bull/bear/vol/liquidity/no-trade + hysteresis)** | `thesis_memory.py` | ✅ shipped |
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

### Fair response + residual (Phase 4 — done)
- `estimate_effective_delta`: rolling Cov(Δmark,Δspot)/Var(Δspot), shrunk
  toward `slot.expected_signed_delta` by `n/(n+prior_strength)`, clipped to
  the option's sign. Converges to the true delta, prior-carried when thin.
- `fair_change = eff_delta · Δspot`; `residual = Δmark − fair_change`.
- `deviation_of_deviation`: robust z of residual vs its own rolling
  median ± IQR band (slot-specific), capped at ±8.
- `acceptance` state: "defended" (residual persistently abnormally
  positive — premium held above fair) / "rejected" (persistently abnormally
  negative) / "normal". The ₹43/₹44 read at the residual level.

### Spread friendliness (Phase 4 — done)
- `spread_pct`, `spread_z` (vs the contract's own rolling normal),
  `depth_score`, `friendliness` ∈ [0,1], and a state machine
  (clean/widening/dangerous/improving). Elevated states gated on absolute
  pct so a sub-tight-threshold wiggle can't false-trigger on robust-z.
- `is_execution_friendly` gate: dangerous spread is never friendly.

### Battlefield + IV state (Phase 5 — done)
- `RailSummary` per side: weighted (by behavior band) signed mean dod_z,
  weighted abs mean, fraction abnormal, **dispersion_score** (Herfindahl-
  based: 0 = even spread across slots, 1 = one-slot concentration),
  epicenter label + signed level, state (strong_bull/strong_bear/mixed/
  quiet). Behavior weights: future_like 1.0, directional_itm 0.95,
  gamma_atm 0.90, convex_otm 0.65, lottery_otm 0.40.
- `battlefield_snapshot` cross-rail verdict: bullish_agreement /
  bearish_agreement / vol_expansion / vol_contraction / single_distortion /
  quiet. Distortion takes priority — fires either when an active rail
  is too disperse OR when the rail-mean is quiet but a single slot's
  |dod_z| exceeds the abnormal threshold (the quiet rail mean IS the
  distortion fingerprint).
- `classify_iv_state` 7-state machine: dirty_data → liquidity_distortion
  → common_shock → directional_bull/bear (battlefield-declared) →
  acceptance-skew directional_bull/bear (broad CE-defended + PE-rejected
  patterns even when rail-mean is quiet — the founder's whole point) →
  vol_contraction → neutral. Data quality dominates direction: a clean
  bullish agreement on a dirty book is worthless.

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

### Thesis memory hysteresis (Phase 6 — done)
- Five scores in [0,100]: bull_thesis, bear_thesis, vol_expansion,
  liquidity_danger, no_trade.
- Hysteresis thresholds (founder's numbers): entry = 70, exit = 45,
  no_trade_danger = 65. The 25-pt gap between entry and exit is what
  stops the engine from flipping like a scared retail trader.
- Update dynamics: per-bar decay 0.94 (≈16-bar half-life); confirming
  signals add positive deltas (battlefield agreement, with extra bonus
  for acceptance-skew IV); damaging signals subtract — counter-direction
  read = small nick, regime_break on held side = large cut, STRONG_TRAP
  on held side = cut, basic trap = half cut.
- Composite states: BULL_ENTRY / BEAR_ENTRY / HOLD_BULL / HOLD_BEAR /
  EXIT_BULL / EXIT_BEAR / NO_TRADE_DANGER / NEUTRAL.
- Entry requires the chosen direction to STRICTLY dominate the other by
  ≥10 — avoids entering during vol expansion when both sides are elevated.
- NO_TRADE_DANGER overrides direction: any composite of liquidity_danger
  and vol_expansion/2 above the danger threshold blocks new entries
  regardless of how high bull_thesis is.

### Honest scope
- Strong: bad-trade avoidance, fake-pullback holding, exit improvement,
  dirty-data avoidance.
- Possible after shadow testing: standalone scalper.
- Impossible: every-situation-proof (news/IV shocks).
