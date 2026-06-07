# Options Strategy & Methodology — The Full Plan

> Supersedes `options_expected_return_model_spec.md` (June 2026) which
> covered only the prediction head. This document covers the full
> stack: why options now, the calibrated prediction layer, the
> execution layer (the moat), the audit layer, and how it ships.
>
> Status: design committed (this commit). Implementation streams
> opened in §10. Acceptance gates in §11.
>
> Author: written by Opus under user direction. Trading consequence
> is the user's.

## §0. Why this exists

The equity-MIS post-touch product line was empirically closed under
two execution regimes (D2, D5 in the master plan): -0.324R mean
across 18 geometry cells under ATR-symmetric sizing, and -0.400R net
under user's actual rupee-floor discipline. The proximity head was
fine — val AUC 0.92-0.95 overall, 0.73-0.76 at 0-1 ATR distance. The
model was correctly ranking opportunities; the trade economics on
small intraday equity moves consumed the gross edge.

Options change the trade economics structurally. Same proximity
signal, same rank, different instrument: a 1% NIFTY move that costs
12% in friction on equity-MIS produces a 30-50% premium move on ATM
weeklies with friction near 1-2%. The cost ratio inverts. This is
not a "second try at the same thing" — it is the same engine on the
right instrument.

## §1. Hard claims this plan commits to

These are the things we are willing to be wrong about publicly.
Empirical failure on any of them sends the relevant stream back to
the drawing board, not into production.

1. **Per-tenor predicted net return is positive on the top decile, on
   the right entries.** The Options Expected-Return Model must
   produce a top-decile realized net premium ATR > 0 on out-of-sample
   data, on at least one (side × tenor × time-of-day) bucket, for
   the engine to ship publicly.
2. **The executor adds value beyond the predictor.** A naive
   "always trade the top-decile prediction" baseline must be
   beaten by the executor's ENTER/WAIT/SKIP logic on out-of-sample
   data, measured in net realized R per trade. If the executor
   doesn't beat the naive policy, the executor is over-engineering.
3. **Calibration holds under regime shifts.** The reported
   calibration error must stay within ±0.10 across vol regimes
   (VIX deciles) on out-of-sample data. Vol regime is the most
   likely source of model drift; we test for it explicitly.
4. **Theta accounting is correct.** Theta decay is a feature input,
   not a separately-subtracted adjustment. The label includes the
   bar's actual time, so the model learns the conditional impact of
   theta from the data directly.
5. **No look-ahead.** Every feature timestamp ≤ prediction `as_of`.
   The daily-Greeks-to-intraday-equity join uses
   `align_daily_to_intraday` which lags daily features by exactly
   one trading day. Truncating any dataset to the bar where a
   feature becomes known must produce the same value as running on
   the full dataset.

## §2. The instrument universe (intentionally narrow at v1)

We trade only the two most-liquid index option chains, and we cap
strike scope tightly:

- **Index**: NIFTY50, BANKNIFTY (and FINNIFTY once it stabilises
  post-2026 lot-size change)
- **Tenors**: weekly (DTE 0-7), 2-week (DTE 8-14), monthly only
  as research context (no v1 trades)
- **Strikes**: ATM ± 3 strike steps. Anything further OTM is
  research-only at v1 (low calibration confidence; lottery payoff
  profile that the regressor can't price reliably).
- **Sides**: buying and selling both modeled; only buying side
  exposed to retail subscribers at v1 (selling demands margin,
  risk controls, and broker-side defined-loss structure we don't
  ship at v1).

Why narrow: the engine learns more per bucket when we don't dilute
the sample with thin-volume strikes. We can widen later, off the
back of evidence.

## §3. Data we use (none of which is hypothetical — all in warehouse)

Every input below is already populated in the Kite warehouse. Audited
and reachable via `liqpool/warehouse.py:WarehouseReader`.

### L0 — already in warehouse
- 5-min OHLCV + OI for the active option contract per (underlying ×
  strike × tenor)
- Bhavcopy EOD for every strike, since 2024-01-01 (3y)
- Daily Greeks parquet (delta, gamma, theta, vega per strike × day)
- India VIX intraday (3y) + daily (5y)
- Index spot OHLCV 3y on 1-min
- Macro / risk-free rate 7% flat for v1 (sufficient; we are not
  pricing options here, we are predicting realized P&L)

### L1 — derived per-bar features for option pricing prediction

**From the underlying** (reused from the equity stack):
- Proximity probability at h=12/36/60 for the underlying touching
  the strike
- Direction probability + magnitude over the next h bars
- Path efficiency (run vs chop predictor)
- Direction changes per 30-bar window
- Vol regime z-score (20-day rolling)
- Expected range today in ATR units

**From the option chain / Greeks (lagged by 1 trading day via
`align_daily_to_intraday`)**:
- IV percentile of the strike, 60d rolling
- IV rank, 60d
- IV day-over-day change in basis points
- Theta per day as percent of the previous close's premium
- Vega per vol point as percent of premium
- Gamma per spot point as percent of premium
- Delta of the strike
- Strike's PCR-OI (put-call ratio open interest)

**Instrument-specific structural features**:
- Distance from spot to strike, in ATR units of the underlying
- Distance from strike to the brief's flagged proximity level, in
  ATR units
- DTE in trading days (NOT calendar days — Indian equity F&O ignores
  weekends)
- Weekly expiry flag (Thursday for NIFTY, Wednesday for BANKNIFTY)
- Time of day in minutes since open (theta non-linear in this)
- Time to EOD in minutes (theta acceleration window)

**Macro**:
- India VIX close yesterday
- India VIX day-over-day change in basis points
- USDINR day-over-day change in basis points

### Label

Per (strike × entry-bar × side × hold horizon):

```
realized_premium_pct_60min = (premium[entry + 12 bars] - premium[entry]) / premium[entry]
                              after slippage subtraction
realized_premium_atr_units = realized_premium_pct_60min /
                              rolling_60d_atr_of_premium_pct[strike]
```

The denominator removes the strike-vs-strike scale problem (a 5%
move on a deep-OTM premium is statistically different from a 5%
move on an ATM premium).

Winsorized at [-3.0, +5.0] for fitting. Reporting uses unclipped
values so that the displayed metrics describe live behaviour.

Slippage assumption baked into the label, not the model:
- 2 ticks each side (₹0.10 for index options)
- STT for the selling side per the current SEBI grid
- No brokerage assumed (options buying is brokerage-free on most
  Indian retail brokers)

Why label normalisation matters: if we fit on raw premium percent,
the model overweights deep-OTM lottery winners that don't represent
realistic outcomes. Normalising by per-strike ATR forces the model
to predict the *typical* move for that strike, not the rare outlier.

## §4. The model layer — `OptionsExpectedReturnModel`

### Architecture

One LightGBM Huber regressor per **(side × tenor × time-of-day
bucket)** triple. Buckets:

- side ∈ {buy, sell}
- tenor ∈ {weekly, 2week}
- time_of_day ∈ {open_window (09:15-10:30), mid_session (10:30-13:00),
  pre_close (13:00-14:45), close_window (14:45-15:15)}

That's 16 heads. Empirically we expect to train only the ~8 with
≥200 OOS trades. The remaining 8 fall back to context-only display
in the brief.

### LightGBM settings (preset, not tunable at v1)

```python
objective="huber"
alpha=0.9                     # Huber transition: 90th-percentile
                              # absolute residual is the cutover
                              # between L2 and L1 loss
num_leaves=31
min_data_in_leaf=80
bagging_fraction=0.8
bagging_freq=3
feature_fraction=0.85
lambda_l2=12.0
learning_rate=0.04
num_boost_round=400 with early stopping on validation R²
```

These mirror the regularisation preset of the existing
`PolicyReturnModel` (a sibling regressor that already exists in
`policy_model.py`). The point is to *not* hyper-tune at v1; we want
to know whether the signal is there, not whether we can squeeze the
last 5% out of a marginal head.

### Validation: chronological + embargo, NEVER random

Random k-fold leaks: a January 2025 trade in train and a January 2025
trade in validation share regime context. Chronological:

- Train on 2024-01-01 → 2025-12-31
- Embargo 5 trading days (no fits use bars within this gap on either
  side of the split)
- Validate on 2026-01-06 → 2026-06-30

Embargo prevents label-spill: a 60-min-horizon label written at
14:30 IST overlaps with the next bar's feature inputs.

### Calibration

No isotonic (output is realized R, not a probability). Calibration
is the decile reliability table: for each decile of `predicted_R`,
report `mean(predicted_R)` vs `mean(realized_R)`. A calibrated head
has the points lying on or near the diagonal.

Per the §1 commit: calibration error must stay within ±0.10 ATR
units across VIX-decile sub-slices of the OOS window. This is the
explicit regime-shift test.

### What we report per head

For each (side × tenor × ToD) head:

```
n_train, n_oos
mean_predicted_R_oos, mean_realized_R_oos
spearman(predicted_R, realized_R)_oos
top_decile_mean_realized_R_oos
top_decile_realized_hit_rate          # fraction with realized > 0
calibration_table                      # 10 rows: predicted vs realized
feature_importance_top_20
sub_slice_calibration_error_by_vix_decile
sub_slice_calibration_error_by_dte_bucket
```

Top-decile realized R is the *commercial* metric. Spearman is the
*scientific* metric. We commit to both.

## §5. The executor layer — the actually novel piece

> This is the part that does not exist in the retail Indian options
> space and is the moat. The prediction layer alone is a research
> asset; the executor layer is the product.

### What the executor receives

For each in-play strike that the Daily Brief flags:

```
{
  underlying: "NIFTY50",
  strike: 24500,
  side: "buy",
  tenor: "weekly",
  predicted_net_return_atr: -0.45,        # from §4 model
  predicted_net_return_atr_lower_ci: -0.78,
  predicted_net_return_atr_upper_ci: -0.12,
  spot_proximity_p_60min: 0.34,
  spot_proximity_p_today: 0.78,
  iv_percentile: 0.62,
  theta_per_day_pct: 0.07,
  vix_now: 14.2,
  vix_regime: "normal",
  as_of_ist: "2026-06-09T09:45:00+05:30",
  bars_to_eod: 65,
}
```

### What the executor outputs (pre-trade)

A structured decision with rationale:

```
{
  action: "ENTER" | "WAIT" | "SKIP",
  reason: "string explaining why",
  recommended_qty: int (lots),
  rupee_reward_floor_inr: 600,
  recommended_stop_premium_pct: -0.30,
  recommended_target_premium_pct: +0.50,
  hard_exit_at_ist: "2026-06-09T13:15:00+05:30",
  kill_conditions: [
    "underlying makes 3 opposite-direction closes of 0.4 ATR",
    "VIX moves > +2 points intra-trade",
    "premium loses > 30% from entry"
  ],
  confidence_bucket: "moderate" | "high" | "very_high",
}
```

### Pre-trade decision rules (rule-based, not learned, at v1)

The executor is **a rule-based layer over a calibrated model**. We
do not learn the executor at v1 — the rules are explicit, auditable,
and disagreement with the customer's own judgement is interpretable.

#### Hard SKIP conditions
1. Predicted net R upper CI < 0 (model says even the optimistic case
   loses)
2. DTE < 2 trading days AND side = "buy" (theta dominates;
   non-buyable for any retail-scale account)
3. IV percentile > 0.90 (vol is expensive; mean-reversion risk
   makes premium-buying a coin-flip)
4. Strike is > 5% OTM on a weekly (lottery; expected R can be
   positive on small sample but not robust)
5. Brief's confidence_notes flags the proximity head as drifting
   (don't trust the input that the executor depends on)

#### Hard WAIT conditions
1. Spot proximity p_60min < 0.20 (the underlying isn't moving toward
   the strike fast enough; check again in 30 minutes)
2. Underlying is < 0.30 ATR from a confirmed regime flip in the last
   30 bars (let the new regime stabilise)
3. Time-of-day is "close_window" AND side = "buy" weekly (theta
   acceleration overwhelms; wait for fresh setup tomorrow)

#### ENTER otherwise, with sizing:
```
recommended_qty = max(
  1 lot,
  ceil(rupee_reward_floor_inr / per_lot_expected_R_inr)
)
per_lot_expected_R_inr = (predicted_net_return_atr * lot_size
                          * average_premium_inr_per_unit
                          * atr_of_premium_pct[strike])
```

with a hard cap at `max_notional_inr` (default ₹2,00,000) so a stop
loss can't exceed customer's risk budget per trade.

### In-trade state machine

Every 5 minutes after entry, the executor recomputes:

```
state = {
  realized_R_so_far: float,         # in ATR units
  bars_since_entry: int,
  underlying_move_since_entry: float,  # in ATR units
  premium_change_pct: float,
  theta_consumed_pct: float,         # estimated from time elapsed
  vix_change: float,
  spot_proximity_p_residual_60min: float,
  any_kill_condition_triggered: bool,
}
```

And outputs one of:

```
HOLD              # default; nothing changed
TRAIL             # if realized_R > 1.0 ATR; raise stop to entry +0.3 ATR
SCALE_OUT_HALF    # if realized_R > 0.8 * predicted_target_R
                  # take half off, ride the rest to the original target
EXIT_TARGET       # premium reached target_pct
EXIT_STOP         # premium reached stop_pct
EXIT_KILL         # any kill_condition triggered
EXIT_HARD         # current time >= hard_exit_at_ist
```

### Post-trade calibration update

Every realized trade closes the loop:

```
outcome_log_record = {
  predicted_net_R_atr: ...,
  realized_net_R_atr: ...,
  decision_path: ["ENTER", "HOLD", "HOLD", "TRAIL", "EXIT_TARGET"],
  kill_conditions_evaluated: [...],
  bars_held: ...,
  underlying_path: [...],          # sampled at 5-min intervals
}
```

This row goes into the outcome log (`liqpool/products/outcome_log.py`)
with `prediction_type="options_executor"` so the audit dashboard
shows realized-vs-predicted distribution per executor decision class
(naive ENTER vs WAIT-then-ENTER vs SKIP-saves).

### Why the executor is rule-based at v1, not learned

Three reasons:
1. **Interpretability for the customer.** When the executor says
   SKIP, the reason must be a sentence the trader can disagree with.
   A learned policy is a black box; a rule list is auditable.
2. **Sample size.** A learned policy needs many trades to fit. At v1
   we expect ~150 trades per (side × tenor × ToD) bucket in the OOS
   window. That's enough to fit a regressor on continuous R; it is
   *not* enough to fit a discrete-action policy that beats a
   well-tuned rule.
3. **Failure mode.** A rule-based executor fails gracefully (you
   know which rule blocked the trade). A learned policy fails
   opaquely (you don't know why it skipped).

v2 (post-revenue, post-real-OOS-history) replaces the rules with a
learned policy that uses the rules as feature inputs. That's after
acceptance §11 passes on real money.

## §6. The brief layer — how this lands in the customer's inbox

The Daily Brief's existing `options_suitability` section gains four
fields per strike entry:

```json
{
  "strike": 24500,
  "p_test_today": 0.78,
  "p_test_within_60min": 0.34,
  "side_from_open": "below",
  "key_level_type": "demand_pool",
  "underlying_level": 24512,
  "predicted_net_return_buy_atr": -0.45,
  "predicted_net_return_sell_atr": 0.30,
  "executor_decision_buy": "WAIT",
  "executor_rationale_buy": "spot_proximity_p_60min<0.20, retest within 30min",
  "executor_decision_sell": "SKIP",
  "executor_rationale_sell": "iv_percentile=0.62, premium not rich enough"
}
```

Renderer surfaces this in the human view as one paragraph per
strike:

```
NIFTY50 24500 PE (in-play demand pool):
  P(test today) = 78%, P(test within 60min) = 34%
  Buy-side net expected return (60min): -0.45 ATR. Executor: WAIT —
    spot proximity has not yet crossed the entry threshold.
  Sell-side net expected return (60min): +0.30 ATR. Executor: SKIP —
    implied vol is not rich enough to justify premium selling here.
```

Tipster guardrail still applies: phrases like "buy-side net return"
and "sell-side net return" do not match the forbidden vocabulary;
"WAIT" and "SKIP" are descriptions of model state, not commands.
The renderer's grep check passes.

The customer reads the brief, sees the executor's view, and decides
for themselves whether to act. We have published context. They have
made the decision. SEBI safe-harbour holds.

## §7. The audit layer — closing the trust loop

Every executor decision is logged with the same flywheel as the
existing brief predictions:

- `outcome_log/predictions/trading_date_ist=YYYY-MM-DD/predictions.parquet`
  carries one row per strike-side-executor-decision per brief.
- `outcome_log/resolutions/trading_date_ist=YYYY-MM-DD/resolutions.parquet`
  carries the realized outcome once known (EOD or strike-touched
  intraday).
- `calibration_by_bucket` computes per (side × tenor × ToD ×
  executor_decision) hit rate + calibration error.

The Yesterday Audit section of tomorrow's brief surfaces:

```
Options executor calibration (last 30 days):
  buy / weekly / open_window / ENTER:    n=42, mean_predicted_R=+0.31,
                                          mean_realized_R=+0.28,
                                          calibration_error=-0.03
  buy / weekly / open_window / WAIT-then-ENTER: n=23, +0.18, +0.42, +0.24
  buy / weekly / open_window / SKIP:     n=87, NaN, +0.04 (the
                                          counter-factual: WHAT the
                                          trade WOULD have done)
```

The SKIP-counter-factual row is what makes the executor
demonstrably-valuable: we show the customer that the trades the
executor refused would have averaged +0.04 R if taken (basically
zero — confirming the skip discipline).

If SKIP counter-factual ever averages > +0.20 R consistently, the
executor rules need re-tuning. That's the audit trigger.

## §8. What this does NOT do at v1 (deferred to v2)

Explicit scope discipline. Each deferred item is named so we don't
quietly drift into it:

- **Multi-leg structures** (spreads, straddles, condors, butterflies).
  A whole separate research lane.
- **Selling-side product for retail customers.** Modeled, but
  surfaced only as research context (`predicted_net_return_sell_atr`
  and `executor_decision_sell` are read-only at v1). Selling
  requires margin-aware risk infrastructure we don't ship at v1.
- **Position sizing beyond the rupee-floor.** No Kelly, no
  volatility-adjusted notional, no portfolio-level allocation. v1
  is per-trade only.
- **Cross-strike dynamics.** The model treats each strike
  independently. We do not (yet) model "if you bought 24500 PE,
  what does that imply for 24700 PE pricing today".
- **Greek hedging.** No delta-hedging, no gamma-scalping. Pure
  directional / vol-regime plays at v1.
- **Same-day exits beyond 60min.** Hard horizon is 60 minutes at
  v1. Sells go to EOD or kill condition; buys exit at target /
  stop / kill / hard_exit_at_ist (typically 2 hours before EOD).
- **Earnings / event-window trades.** Skipped entirely at v1. The
  vol distortion is regime-change risk the model wasn't trained on.

## §9. Cost arithmetic — concrete

The equity failure was empirically -0.4R net. Here is the same
arithmetic for ATM weekly NIFTY options for context:

Typical setup we'd target (ATM weekly NIFTY, mid-session, expected
30-50% move):

```
Premium entry:      ₹120 (per unit)
Lot size:           75
Premium notional:   ₹9,000 per lot
Slippage 2 ticks:   ₹15 per side = ₹30 round-trip per lot
STT (buying):       0 (selling pays)
Brokerage:          0 (most retail Indian brokers, options buy)
Round-trip cost:    ₹30 per lot
Cost as % move:     ₹30 / ₹120 = 25% of premium → BUT
                    typical move target = 30-50% of premium
                    Cost-to-target ratio = 25/50 = 0.5 vs 17x on equity
```

The ratio is 34× friendlier for options than for equity-MIS
post-touch. That single arithmetic shift is the structural reason
this stream can work where the equity stream couldn't.

This is not "leveraged equity wins for free." It's: the same
prediction edge, applied to an instrument where the cost ratio
allows the edge to express. A wrong call still loses. A right call
finally pays for itself.

## §10. Implementation streams (what unblocks ship)

Mapped against the master plan's stream framework. Each is opened
as a §3 entry with explicit acceptance and ETA.

### Stream D.1 — Feature joining (data → ML-ready frame)
**Layers**: L0 + L1
**Owner**: Opus design + Codex implementation
**Status**: warehouse reader exists; per-strike feature builder doesn't
**Scope**:
- New `liqpool/options/featurizer.py`: walks a (underlying × strike ×
  tenor × bar) tuple and returns the feature row defined in §3.
- Uses `align_daily_to_intraday` for the daily-Greeks join.
- No-lookahead test: truncating the dataset to the prediction bar
  produces identical features to running on the full frame.
- Causality test: every feature timestamp ≤ as_of.
**Acceptance**: 8 tests passing, lint clean, 100k-row featurizer
runs in <60s on a sample bundle.
**ETA**: 1 commit.

### Stream D.2 — Label generation (realized_premium_atr_units)
**Layers**: L1 + L2
**Owner**: Codex
**Status**: not built
**Scope**:
- `liqpool/options/labels.py`: produces the `realized_premium_atr_units`
  label at the 60-minute horizon, per the §3 formula.
- Includes the slippage assumption (2 ticks each side; STT for sells)
  baked into the label.
- Tests pin: zero-volume bars produce NaN labels (skipped), not zero;
  winsorization happens at fit time, not at label generation; the
  per-strike ATR denominator is computed from a 60d rolling window
  that ends strictly before the label window opens.
**Acceptance**: 6 tests, lint clean.
**ETA**: 1 commit.

### Stream D.3 — Train `OptionsExpectedReturnModel` suite
**Layers**: L2
**Owner**: Codex
**Depends on**: D.1 + D.2
**Status**: not built
**Scope**:
- One head per (side × tenor × ToD), gated at `min_trades=200`.
- Reports per-head metrics per §4.
- Bundle persistence so the brief generator can load it via
  `getattr` (backward-compat with bundles that don't have it).
- Chronological split + embargo per §4.
**Acceptance**:
1. At least one (side × tenor × ToD) head has Spearman > 0.10 AND
   top-decile realized_R > 0 on OOS data. If yes, D.4 proceeds.
   If no, we re-evaluate D.1 features before ANY commercial step.
2. Calibration error across VIX-decile sub-slices stays within
   ±0.10 ATR for the qualifying heads.
**ETA**: 2 commits (1 fitting; 1 evaluation + report).

### Stream D.4 — Brief integration (predicted_net_return surfaces)
**Layers**: L6
**Owner**: Opus
**Depends on**: D.3 acceptance
**Status**: schema designed in `options_expected_return_model_spec.md`;
implementation not built
**Scope**:
- `liqpool/products/daily_brief.py` reads the bundle's
  `options_expected_return_model` and populates
  `predicted_net_return_*_atr` on each strike entry.
- Renderer surfaces the per-strike paragraph in §6.
- Tipster guardrail verified on the new copy.
**Acceptance**: existing brief tests still pass + 4 new tests pin
the new fields' presence and prose.
**ETA**: 1 commit.

### Stream D.5 — The executor layer (the moat)
**Layers**: L4 + L6
**Owner**: Opus design + Codex implementation
**Depends on**: D.4 (executor wraps the predicted_net_return)
**Status**: designed (§5); not built
**Scope**:
- `liqpool/options/executor.py`: implements the §5 rule list as a
  pure function. Inputs are §5's pre-trade dict; outputs are §5's
  decision dict.
- `liqpool/options/state_machine.py`: in-trade state machine per §5.
- Brief integration: `executor_decision_buy/sell` and `executor_rationale_buy/sell`
  populated on each strike entry.
- Renderer: per §6 paragraph format.
- Tests pin every rule + every kill condition + the rupee-floor
  sizing math.
**Acceptance**:
1. 20 tests passing across pre-trade rules, in-trade state machine,
   sizing math, kill-condition evaluation.
2. End-to-end synthetic-bundle test: brief contains executor
   decisions, renderer produces tipster-clean prose.
3. Counter-factual audit fixture: on a replay window of 60 trading
   days, the SKIP counter-factual mean realized R must be ≤ +0.20 R.
   If higher, the executor is too conservative; re-tune rules
   before shipping.
**ETA**: 3 commits.

### Stream D.6 — Outcome log + Yesterday Audit for options
**Layers**: L5 + L6
**Owner**: Opus
**Depends on**: D.5
**Status**: schema additive; not built
**Scope**:
- `outcome_log` gains `prediction_type="options_executor"` rows.
- Yesterday Audit section gains the §7 executor-calibration table
  including the SKIP counter-factual row.
- Retrospective-replay flag (Stream G) carries through to options
  audit identically.
**Acceptance**: 6 tests; full outcome-log round-trip on a synthetic
options frame.
**ETA**: 1 commit.

### Stream D.7 — Live-publish hook
**Layers**: L4
**Owner**: Codex
**Depends on**: D.5
**Status**: artifact pusher exists; options-artifact specifically not
**Scope**:
- New artifact kind `options_executor_call_pdf` for the per-strike
  executor card.
- New kind `options_strikes_csv` (already in the website's allow-list).
- Publish hook on the live brief runner pushes both whenever a brief
  is generated with options content.
**Acceptance**: smoke test from VPS produces both artifact kinds;
portal shows them in the Options section.
**ETA**: 1 commit.

### Stream D.8 — Live broker hardening for options (post-pilot)
**Layers**: L4
**Owner**: Codex
**Depends on**: D.5 + first paying customer requesting auto-execute
**Status**: deferred until first customer demand; explicit non-goal
at MVP
**Scope**: broker integration for options-specific quirks (lot-size
multiples, freeze-quantity caps, option-symbol resolution from
strike + expiry).
**ETA**: 3 commits, post-revenue.

## §11. Acceptance gates — concrete, non-negotiable

The streams above ship in order. Each gate is a hard pass/fail.

### Gate 1 — D.1 + D.2 + D.3 complete
**Question**: Does the predicted-net-return model have edge?

**Pass criterion**: at least one (side × tenor × ToD) head shows:
- Spearman(predicted_R, realized_R) > 0.10 on OOS
- Top-decile realized R > 0 on OOS, n ≥ 200
- Calibration error ≤ 0.15 ATR on the top-decile bucket
- Sub-slice calibration error stays ≤ 0.15 ATR across VIX deciles

**Fail action**: stream stops here. We do not build the executor on
top of an uncalibrated head. We re-evaluate features (D.1) and
possibly the label horizon (D.2). If repeated fits fail, the
options vertical itself is closed-pending-better-data, same as
equity-MIS was.

### Gate 2 — D.4 + D.5 complete
**Question**: Does the executor add value beyond the naive baseline?

**Pass criterion**:
- Executor's "trades it took" (ENTER paths) on a 60-day OOS replay
  produce mean realized R > 0
- Executor's "trades it skipped" (SKIP paths) have counter-factual
  mean realized R ≤ +0.20 (proving SKIP wasn't pessimistic)
- Net realized R per executor-decided trade beats the naive
  "always take top-decile predicted R" baseline by ≥ +0.15 ATR

**Fail action**: re-tune executor rules. The model passes Gate 1
but the rules are wrong. We hold ship for ≤ 2 weeks of rule tuning;
beyond that, ship Gate-1 results with executor disabled (research-
only options block in brief).

### Gate 3 — D.6 + D.7 complete + smoke test passes
**Question**: Does the whole pipeline ship cleanly to the customer?

**Pass criterion**:
- Engine generates brief with options section + executor decisions
- Auto-publishes both PDF and CSV to website
- Portal renders options block with the executor paragraph
- Yesterday Audit on day N+1 shows day N options calibration
- Tipster guardrail passes on all rendered copy

**Fail action**: standard build-fix-redeploy; no methodology change.

### Gate 4 — first 30 trading days of live calibration
**Question**: Did the model hold up in live conditions?

**Pass criterion** (audited 30 days after Gate 3):
- Live calibration error within ±0.15 ATR vs OOS calibration error
  per head
- No drift flag persisting > 5 trading days
- Net realized R of executor's ENTER paths positive on a sample of
  ≥ 50 live trades

**Fail action**: identify the drifting head, retrain on a window
that includes the live data, re-run from Gate 1 for that head only.
Other heads continue in production.

## §12. Risk to the project, named explicitly

Six things that can plausibly kill this stream. Naming them so we
don't pretend they won't.

1. **Top-decile realized R might be zero.** The model can rank
   correctly while none of the deciles are absolutely profitable
   after slippage. This is the equity outcome reproduced. If it
   happens, see Gate 1's fail action.
2. **The 60-min horizon might be wrong.** Theta accelerates inside
   60 min during the close window; intra-day vol can crush in 15
   min on an event. We may need 30-min, 15-min, and 60-min heads
   per (side × tenor × ToD). v1 ships 60-min only; v2 widens
   horizons if Gate 4 reveals horizon-specific drift.
3. **Daily Greeks lag may be too stale.** A morning trade uses
   yesterday's Greeks. On a fast-moving day this can be substantially
   wrong. Mitigation: include intra-day IV as a feature (computable
   from the option's own price + Black-Scholes inversion), not
   just lagged daily IV.
4. **Calibration may break in regime shift.** A May 2025-trained
   model meeting a 2026-March VIX spike will display drift. The
   ±0.10 calibration tolerance across VIX deciles is the explicit
   guard. If it fails, we retrain monthly rather than quarterly.
5. **Executor rules may be over-fit to backtest.** Rule choices
   like "skip if VIX > 90th percentile" come from us reading the
   data, which is a form of look-ahead. Mitigation: rules are
   monotone-direction (no thresholds that flip sign across
   conditions); the SKIP counter-factual gate (Gate 2) catches
   this.
6. **Broker / live data limitations.** Reading live IV per strike
   at the cadence the executor wants (5-min) may exceed broker
   rate limits. Mitigation: snapshot IV at signal time only;
   in-trade re-checks use last-traded-premium changes, not
   re-inverted IV.

## §13. What "tier 3" means here

Tier 3 in quant-research vocabulary is: every output is calibrated
against held-out data, every drift is detected and disclosed, every
counter-factual is auditable, every rule is documented and testable.
This document and the streams it opens are tier 3.

Tier 3 is not "perfect"; it is *honest*. We will be wrong sometimes.
The system reports when it's wrong, faster than the customer can
detect it on their own. That is the difference between a research
service and a tipster service.

## §14. Glossary

- **ATM** — at-the-money. Strike closest to the current spot.
- **OTM** — out-of-the-money. Strike further from spot in the
  direction premium is not exercised in.
- **ATR** — Average True Range. Volatility scale used throughout.
- **DTE** — days to expiry.
- **IV** — implied volatility.
- **PCR-OI** — put-call ratio, by open interest.
- **ToD** — time of day, bucketed into open / mid / pre-close / close
  windows.
- **R** — return in ATR units. "+1R" means the move was 1 ATR in
  our favour.
- **Top decile** — the top 10% of predicted-R candidates in the
  OOS evaluation window.
- **Counter-factual** — what the trade would have earned if the
  executor had not skipped it.

End of document.
