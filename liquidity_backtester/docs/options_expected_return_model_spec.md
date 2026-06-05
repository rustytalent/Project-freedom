# Options Expected-Return Model — Spec (Stream D, design)

The intraday equity stack outputs calibrated probabilities (proximity,
direction, Q). For the **options vertical slice**, those probabilities
alone are insufficient — a customer reading the brief needs to know
not just "P(NIFTY tests 24500 today) = 0.78" but "given this regime,
buying the 24500 PE here has an expected NET return of -0.45 ATR
premium-units over the next 60 minutes (theta dominates the move),
while selling the 24500 PE has expected return +0.30 (vol-of-vol
favorable)". The Options Expected-Return Model produces that number.

This is the *design* commit. Training and pipeline wiring land
post-Stream-B (after Codex's V2-notional patches reach the remote).

## Hard constraints

1. **Inputs are causal.** Every feature timestamp must be at-or-before
   the prediction `as_of` time. The daily-Greeks-to-intraday-equity
   join uses `align_daily_to_intraday` (already shipped in
   `liqpool/warehouse.py`) which lags daily features by one day —
   never the same trading day.
2. **Calibration is reported in premium ATR units, not raw rupees.**
   Options have wildly varying notional (NIFTY weekly 50-delta vs
   FINNIFTY 5-delta), and raw-rupee MAE hides the comparison. The
   model's loss is computed on `(realized_premium_pct - predicted) /
   premium_ATR_at_entry`. Reporting is in those normalized units.
3. **Theta is a feature, not a separate adjustment.** The model
   learns the conditional impact of theta directly. We do NOT
   post-hoc subtract theta from the predicted return — that would
   double-count signals the model already encoded.
4. **One head per (side × tenor × time-of-day-bucket).** Buying ATM
   weekly at 09:30 IST is a different problem from selling 2-week
   OTM at 14:00. Stacking them under one head erases the strongest
   conditional signal (theta dominance changes drastically with
   time-of-day and tenor).
5. **Only used when ≥ 200 trades exist for the (side × tenor × ToD)
   bucket.** Below that, the brief falls back to context-only options
   suitability (the existing `strike_translator` + theta-danger
   score path).

## Top-level signature

```python
class OptionsExpectedReturnModel:
    def __init__(self, mode: str):
        """mode = 'buy_ATM_weekly_morning' |
                 'buy_OTM_weekly_morning'  |
                 'sell_ATM_weekly_afternoon' | ...
        """
    def fit(self, train_labels: OptionsLabelFrame,
            oos_labels: OptionsLabelFrame) -> OptionsERMetrics: ...
    def predict_frame(self, X: pd.DataFrame) -> pd.Series: ...
    def predict_one(self, *,
                    spot: float, strike: float, dte_days: int,
                    side: str, regime_features: dict,
                    daily_greeks: dict,
                    as_of: pd.Timestamp) -> float:
        """Returns predicted NET return (after assumed slippage of
        2 ticks each side, broker brokerage 0 for options buying,
        STT for options selling included) in premium-ATR units."""
```

## Feature set

Inputs grouped by L-layer:

### L1 (intraday equity / index features)
- `p_touch_h12`, `p_touch_h36`, `p_touch_h60` from `unified_proximity`
- `direction_p_long` from `unified_direction`
- `path_efficiency_30`, `direction_changes_30`
- `vol_regime` (categorical: low / normal / high)
- `vol_regime_zscore_20d`
- `expected_range_today_atr`
- `time_of_day_minutes_since_open` (numeric; used for time-decay
  emphasis even within a single ToD bucket since the bucket boundaries
  are coarse)

### L1 (option-distance / structure features)
- `dist_strike_to_spot_atr` — strike's distance from current spot in
  ATR units (NOT raw points; an ATM 24500 PE is closer in ATR when
  vol is high than when vol is low)
- `dist_strike_to_proximity_target_atr` — strike's distance from the
  daily brief's flagged proximity level
- `is_ATM_strike` (boolean, |dist_strike_to_spot| < 0.25 ATR)
- `is_strike_in_play` (boolean, was this strike in the brief's
  strike_levels_in_play list)

### L0 (daily Greeks + IV features, joined via align_daily_to_intraday)
- `iv_percentile_60d` — 60-day rolling percentile of the strike's IV
- `iv_rank_60d`
- `theta_per_day_pct` — yesterday's theta as % of premium
- `vega_per_volpoint_pct`
- `gamma_per_spot_pct`
- `delta` — yesterday's strike delta
- `iv_dod_change_bps` — IV day-over-day change (basis points)
- `days_to_expiry`

### L0 (macro / underlying)
- `india_vix_close_yday` — daily VIX
- `india_vix_dod_change`
- `nifty_oi_pcr_yday` — put-call ratio (open interest)
- `usdinr_dod_change_bps`

All daily features are **lagged by one trading day** via the
`align_daily_to_intraday` join — never same-day, since real-time at
09:15 IST tomorrow we only know today's EOD values.

## Label

Per (strike × entry-time × side):
- `realized_premium_pct_60min` — premium change over the next 60min
  IST minutes, sampled at 5-minute intervals, fixed entry at 09:30 IST
  bucket open (10:00 / 13:00 / 14:00 etc per bucket).
- Normalized to `realized_premium_atr_units = realized_premium_pct /
  rolling_60d_atr_of_premium_pct[strike]`.

Slippage assumption baked into the label, not the model:
- 2 ticks each side (0.10 INR for index options, 0.05 for stock).
- STT for selling-side trades (per current SEBI grid).
- No brokerage assumed (options BUYING is brokerage-free on most
  Indian brokers).

## Model

LightGBM regressor (Huber loss, `alpha=0.9`):
- Target: `realized_premium_atr_units`, winsorized at `[-3.0, +5.0]`.
- Calibration: decile reliability table (mean predicted vs mean
  realized per decile of predicted). No isotonic (output is R, not
  a probability).
- Validation: chronological split with embargo (same pattern as
  `PolicyReturnModel` from the R1 stream that exists in
  `policy_model.py`).

## Brief integration

The brief's `options_suitability` section currently surfaces strikes
in play + theta-danger / premium-regime context (via
`strike_translator`). Stream D adds three fields per strike entry:

```json
{
  "strike": 24500.0,
  "p_test_today": 0.78,
  "predicted_net_return_buy_atr": -0.45,
  "predicted_net_return_sell_atr": 0.30,
  "expected_return_confidence_bucket": "moderate"
}
```

`expected_return_confidence_bucket` follows the same mapping as
proximity (`>=0.80 very_high`, `0.65 high`, `0.50 moderate`, else
`low`) but applied to `1 - sigmoid(-predicted * 5)` so a strongly
negative-R prediction gets `very_high` confidence too (we are highly
confident this strike is a bad buy).

## Render

The brief renderer's options block gains one line per strike:

```
NIFTY50 24500 PE (in-play demand pool):
  P(test today) = 78%, P(within 60min) = 34%
  Buying premium net expected return (60min): -0.45 ATR (theta drags)
  Selling premium net expected return (60min): +0.30 ATR (favorable)
```

The tipster-guardrail still applies: no "buy" / "sell" verbs. We
say "buying-side net return" and "selling-side net return".

## Acceptance (post-B, training session)

1. Each (side × tenor × ToD) bucket with `>=200` trades trains a head.
2. Reported metrics per head:
   - `n_train`, `n_oos`
   - `mean_predicted_atr`, `mean_realized_atr`
   - `spearman(predicted, realized)` (target: > 0.10 for the head to
     be considered "has edge")
   - `top_decile_realized_atr` (target: positive for the buying-side
     model on at least one (tenor × ToD) bucket — proves there exists
     a positive-R buyable subset, even if narrow)
3. CSV artifacts: `options_er_model_report.csv`,
   `options_er_model_calibration.csv`,
   `options_er_model_feature_importance.csv`.
4. Brief renders end-to-end with the new fields.

## Out of scope (deferred to a follow-on stream)

- Multi-leg structures (spreads, straddles, condors).
- Same-day exits beyond 60-minute horizons.
- Greeks-aware position sizing.
- The R-policy stack on options (a separate options policy model).
- Cross-strike dynamics (e.g. the 24500 PE pricing implies something
  about the 24700 PE).

## Verification before this lands

1. `align_daily_to_intraday` round-trip test confirms the lag is
   exactly one trading day (no same-day leakage).
2. New tests in `tests/test_options_expected_return_model.py`:
   - Causality: no feature timestamp later than `as_of`.
   - Determinism: same seed produces identical predictions.
   - Chronological split: validation segment strictly later than train.
3. Compliance lint `enforced_failures=0`.
4. Manual brief generation against the saved bundle + a real index
   options snapshot — verify the new fields render.
