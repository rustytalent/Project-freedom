# Swing Vertical — Spec (Stream F, design)

The original engine is **intraday-MIS** native: same-session reactions,
strict end-of-day caps, no overnight risk. The swing vertical is the
**multi-day proximity + delivery** product line.

## Why this matters (cost arithmetic)

The intraday product fights brokerage + STT + slippage on every trade.
Per-trade cost is ~0.05% on liquid equities (round-trip), which on a
0.4% intraday move leaves ~0.35% gross before any alpha.

Swing on delivery:
- ZERO brokerage on most Indian brokers (Zerodha CNC = 0).
- Zero MIS penalty (no auto-square-off).
- Larger moves per trade (3-7 day holds → 2-5% typical range).
- DP / STT remains but is a fraction of the larger move.

Per-trade cost as % of move is ~0.05% on a 3% move = 1.7% cost ratio,
vs ~12% cost ratio intraday. **The same edge is 7x more profitable
on the swing product.** This is why this stream is the most likely
to land profitable without further alpha changes.

This is the *design* commit. Training and pipeline wiring land
post-Stream-B.

## Hard constraints

1. **Different label horizon.** The intraday respect/break/reaction
   labels are capped at the same-IST-session EOD (Config flag
   `intraday_session_only=True`). Swing labels must use
   `intraday_session_only=False` AND a longer `test_horizon_bars`
   (default for swing: 20 trading days × 78 bars-per-day = 1560 bars
   on the 5m base TF).
2. **Per-pool multi-day persistence.** An intraday pool is touched
   once and resolved by EOD. A swing pool can be touched on day 1,
   bounce, retraced on day 3, retested on day 4. The pool builder
   already supports `available_at` (when the pool became known) and
   `broken_at` (when it was definitively broken); swing inherits
   that — no new pool-builder code.
3. **No tipster outputs.** Same rule as the intraday brief. The
   swing brief speaks in regime / probability / context.
4. **Swing proximity heads are SEPARATE from intraday.** A separate
   `unified_proximity_swing` dict keyed by `h5d`, `h10d`, `h20d` (in
   trading-day units, NOT bar units). The intraday `unified_proximity`
   stays unchanged.
5. **Sample-weight decay shorter for swing.** Regime drift on
   multi-day moves is faster (a 3-day move reads regime A; the next
   3-day move reads regime B). Default
   `sample_decay_halflife_days = 90` for swing vs `180` for intraday.

## Layer additions

### L2 — new model heads

```python
# Per-symbol per-horizon binary classifier: P(touch within N trading days)
class SwingProximityModel(PoolProximityModel):
    """Sibling of intraday PoolProximityModel.

    Differences:
      * label horizons in trading-day units (5d, 10d, 20d)
      * intraday_session_only=False
      * sample_decay_halflife_days=90 (faster decay)
      * separate calibration table (NOT pooled with intraday)
    """

class SwingDirectionModel(UnifiedDirectionModel):
    """Sibling. Multi-day direction conviction.

    Inputs: same featurizer as intraday but pooled to daily resolution
    (last bar of the trading day).
    Horizon: 5d, 10d, 20d directional sign + magnitude.
    """
```

### L4 — execution simulator changes

The intraday execution simulator caps every label/trade at EOD. The
swing simulator must:
- Allow positions to persist across days (no auto-square-off).
- Compute realized R using delivery-cost model (0 brokerage, STT only
  on sell, DP on sell of held shares).
- Use a wider stop (default 1.5x ATR-D, where ATR-D is daily ATR).
- Use a wider target (default 3.0x ATR-D, asymmetric R:R = 2:1).

```python
class SwingExecutionConfig:
    cost_model: str = "delivery"         # vs "intraday_mis"
    stop_atr_d: float = 1.5
    target_atr_d: float = 3.0
    max_hold_trading_days: int = 20
    overnight_gap_handling: str = "open_fill"  # gap risk realistic
    rupee_floor_inr: float = 1500.0      # higher floor than intraday
                                          # (delivery cost amortization
                                          # only makes sense above ~₹1500
                                          # gross R; tune by symbol class)
```

### L6 — new brief section

The swing brief is a SEPARATE artifact from the intraday brief — same
generator, different schema, different cadence (weekly Monday morning
+ ad-hoc when a swing pool becomes available).

```json
{
  "schema_version": "swing_1.0",
  "brief_metadata": { ...intraday-like... },
  "swing_regime": {
    "trending_up_5d_horizon": [...symbols...],
    "trending_down_20d_horizon": [...],
    "leadership_change_vs_last_week": [...]
  },
  "swing_watchlist": [
    {
      "symbol": "HDFCBANK",
      "side": "long",
      "key_level": 1742.5,
      "key_level_type": "demand_pool",
      "p_test_5d": 0.62,
      "p_test_10d": 0.78,
      "p_test_20d": 0.85,
      "expected_hold_trading_days": 7,
      "model_confidence_bucket": "high",
      "delivery_cost_breakeven_move_pct": 0.32
    },
    ...
  ],
  "swing_avoid_list": [...],
  "swing_yesterday_audit": { ...with retrospective flag (Stream G)... }
}
```

Renderer rule: same tipster guardrail, plus an explicit
"hold-period budget" line per entry so the reader knows the call
is multi-day.

## Multi-day persistence in the outcome log

The existing outcome log partitions predictions by `trading_date_ist`.
Swing predictions add a `prediction_horizon_trading_days` field
(5, 10, 20) so the resolver knows how long to wait before declaring
a prediction unresolved. The resolver becomes:

- For horizon=5: declare resolved when 5 trading days pass OR when
  the level is touched, whichever first.
- For horizon=20: declare resolved when 20 trading days pass OR
  touched.

No schema bump needed — the `prediction_horizon_trading_days` field
is additive and old partitions read back NA (treated as "intraday"
= same-day).

## Two-product brief routing

The brief generator gets a new top-level mode:

```python
generate_brief(report, ..., product_mode: str = "intraday")
# product_mode in {"intraday", "swing", "both"}
```

When `"both"`, produce TWO BriefDocuments (intraday for today, swing
for the trailing week + ad-hoc swing setups). The renderer composes
them into a single email body with section headers per product.

Default stays `"intraday"` so existing callers are unaffected.

## Pricing implication

The swing product justifies a **higher subscription tier** because:
- Multi-day holds = customer needs less screen time.
- Cost arithmetic genuinely favours the customer.
- Lower call frequency = less noise to evaluate.

Suggested initial tier structure (decision for §11 once first pilots
land):
- Intraday-only: base tier
- Swing-only: base + 30%
- Both: base + 60% (vs base + 100% if priced linearly — discount
  encourages bundling, since cost to us of producing both is
  marginal once the bundle is loaded)

## Acceptance (post-B, training session)

1. `SwingProximityModel` trains on the existing bundle's daily
   resampled frames (`base_df_daily` from the warehouse reader).
2. Reported metrics per horizon:
   - `n_train`, `n_oos`
   - `oos_auc_h5d`, `oos_auc_h10d`, `oos_auc_h20d` (target: each
     > 0.55, mirror of intraday acceptance bar)
   - Calibration table per horizon
3. Swing brief renders end-to-end with at least one swing watchlist
   entry on the historical bundle (smoke test).
4. Outcome log resolver handles `prediction_horizon_trading_days=5`
   and `=20` correctly.
5. Full test suite green; lint `enforced_failures=0`.

## Out of scope (deferred)

- Options on swing pools (the swing trade is delivery; options would
  add a layer on top — a Stream D × Stream F crossover).
- Sector rotation as a swing-specific brief input.
- Pairs / spread structures across symbols.
- Tax-optimized exit timing.

## Verification before training lands

1. New tests in `tests/test_swing_proximity_model.py`:
   - Determinism, chronological split, no-lookahead on daily features.
   - Label-horizon distinct from intraday (no contamination across
     model instances).
2. New tests in `tests/test_swing_brief.py`:
   - Schema shape matches `swing_1.0`.
   - Renderer surfaces hold-period budget.
   - Tipster-vocabulary guardrail applies.
3. Manual swing brief generation against the saved bundle — verify
   at least one symbol gets a non-trivial multi-day proximity.
