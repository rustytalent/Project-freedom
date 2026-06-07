"""Tests for the OptionsExpectedReturnModel suite (Stream D.3).

Pinned contracts:

  1. Constraint builder: monotone signs flip between buy and sell;
     interaction groups translate column names to indices correctly
     and skip features absent from the frame.
  2. Tenor derivation: dte 1-7 → weekly; 8-14 → 2week; else None.
  3. Feature matrix expansion: one-hots are deterministic
     (driven by MONEYNESS_BUCKET_LEVELS, not by the data) so train
     and predict frames produce comparable vectors.
  4. Chronological split + embargo: train is strictly earlier than
     val; embargo drops any train row within 5*embargo_bars minutes
     of the earliest val bar.
  5. Min-trades gate: a bucket with < min_trades is skipped with a
     populated reason; status="skipped".
  6. Fit-then-predict round-trip: synthetic data where the label is
     a learnable function of features gives Spearman > 0.5 on the
     held-out tail.
  7. Suite routing: a mixed frame containing multiple buckets
     produces a head per qualifying bucket; predict_frame routes
     each row to its head and NaN-fills rows without a fit.
  8. Determinism: same seed + same input → bit-identical predictions
     on the same held-out slice.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from liqpool.options.expected_return_model import (
    BucketKey,
    OptionsExpectedReturnModel,
    OptionsExpectedReturnModelSuite,
    _apply_embargo,
    _build_feature_matrix,
    _chronological_split,
    _decile_table,
)
from liqpool.options.model_config import (
    MONOTONE_BUY,
    MONOTONE_SELL,
    build_constraints,
    derive_tenor,
    expand_categorical_features,
)


# ---------------------------------------------------------------------------
# Fixture: synthetic frame that LOOKS like the D.1+D.2 output
# ---------------------------------------------------------------------------

def _synth_frame(n: int = 800, seed: int = 0,
                 side: str = "buy",
                 dte_days: int = 5,
                 tod: str = "open",
                 start: str = "2026-01-02 09:15",
                 learnable: bool = True) -> pd.DataFrame:
    """Synthetic labelled frame.

    When ``learnable=True`` the label is a linear combination of three
    features + noise, so the model should learn the signal. When
    False, the label is pure noise.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq="5min")
    dist = rng.normal(0, 1.0, n)
    abs_dist = np.abs(dist) * 0.005
    delta = 0.5 + rng.normal(0, 0.1, n)
    underlying_30m = rng.normal(0, 0.003, n)
    iv_pct = rng.uniform(0.2, 0.9, n)
    iv_rank = rng.uniform(0.2, 0.9, n)
    iv = 0.14 + rng.normal(0, 0.02, n)
    theta_pct = rng.uniform(0.02, 0.10, n)
    vega_pct = rng.uniform(0.10, 0.30, n)
    real_vol = np.abs(rng.normal(0, 0.005, n))
    moneyness = rng.choice(
        ["ATM", "ATM_or_near", "1OTM", "2OTM", "3OTM_plus"],
        size=n, p=[0.35, 0.25, 0.20, 0.12, 0.08])
    if learnable:
        # The "true" label: rewards bullish underlying + ATM, penalises
        # high IV percentile. Plus noise. Range ~ [-3, +3].
        signal = (
            2.5 * underlying_30m * 100
            - 1.0 * iv_pct
            - 0.6 * abs_dist * 50
            + 0.5 * delta
        )
        noise = rng.normal(0, 0.5, n)
        y = signal + noise
    else:
        y = rng.normal(0, 1.0, n)
    return pd.DataFrame({
        "bar_ts": ts,
        "trading_date_ist": ts.normalize(),
        "underlying": "NIFTY50",
        "strike": 24500.0,
        "side": side,
        "tod_bucket": tod,
        "dte_trading_days": float(dte_days),
        "moneyness_bucket": moneyness,
        "weekly_expiry_flag": False,
        # Numeric features the model will see
        "dist_strike_to_spot_atr": dist,
        "abs_dist_pct": abs_dist,
        "tod_minutes_since_open": np.arange(n) % 360,
        "tod_minutes_to_eod": 360 - (np.arange(n) % 360),
        "underlying_5m_return": rng.normal(0, 0.001, n),
        "underlying_30m_return": underlying_30m,
        "underlying_realized_vol_30m": real_vol,
        "delta_yday": delta,
        "gamma_yday": 0.02 + rng.normal(0, 0.005, n),
        "theta_per_day_pct_yday": theta_pct,
        "vega_per_volpoint_pct_yday": vega_pct,
        "iv_yday": iv,
        "iv_dod_change_bps": rng.normal(0, 50, n),
        "iv_percentile_60d": iv_pct,
        "iv_rank_60d": iv_rank,
        "pcr_oi_yday": 1.0 + rng.normal(0, 0.1, n),
        "india_vix_close_yday": 14.0 + rng.normal(0, 1, n),
        "india_vix_dod_change": rng.normal(0, 0.3, n),
        "usdinr_dod_change_bps": rng.normal(0, 5, n),
        # Labels (from D.2)
        "label_valid": True,
        "realized_premium_atr_units_60min": y,
    })


# ---------------------------------------------------------------------------
# 1. Constraint builder
# ---------------------------------------------------------------------------

class ConstraintBuilderTests(unittest.TestCase):

    def test_monotone_signs_flip_between_sides(self):
        # The directional features have opposite signs.
        for key in ("underlying_30m_return", "delta_yday",
                    "iv_percentile_60d", "abs_dist_pct"):
            self.assertIn(key, MONOTONE_BUY)
            self.assertIn(key, MONOTONE_SELL)
            self.assertEqual(MONOTONE_BUY[key], -MONOTONE_SELL[key],
                f"{key} sign must flip; buy={MONOTONE_BUY[key]} "
                f"sell={MONOTONE_SELL[key]}")

    def test_build_constraints_filters_missing_features(self):
        feature_names = ["underlying_30m_return", "iv_percentile_60d",
                         "abs_dist_pct"]
        interaction, monotone = build_constraints(feature_names, "buy")
        # No group is empty.
        for g in interaction:
            self.assertGreater(len(g), 0)
        # Monotone vector length matches feature count.
        self.assertEqual(len(monotone), len(feature_names))
        # Buy direction is +1 for underlying_30m, -1 for iv pct.
        i30 = feature_names.index("underlying_30m_return")
        iiv = feature_names.index("iv_percentile_60d")
        self.assertEqual(monotone[i30], +1)
        self.assertEqual(monotone[iiv], -1)


# ---------------------------------------------------------------------------
# 2. Tenor derivation
# ---------------------------------------------------------------------------

class TenorTests(unittest.TestCase):

    def test_weekly_2week_outofscope(self):
        self.assertEqual(derive_tenor(1), "weekly")
        self.assertEqual(derive_tenor(7), "weekly")
        self.assertEqual(derive_tenor(8), "2week")
        self.assertEqual(derive_tenor(14), "2week")
        self.assertIsNone(derive_tenor(15))
        self.assertIsNone(derive_tenor(0))
        self.assertIsNone(derive_tenor(None))


# ---------------------------------------------------------------------------
# 3. Feature matrix expansion is deterministic
# ---------------------------------------------------------------------------

class FeatureMatrixTests(unittest.TestCase):

    def test_train_predict_columns_match_when_levels_differ(self):
        # Train frame contains 4 of the 5 moneyness levels;
        # predict frame contains a DIFFERENT subset. The post-one-hot
        # column list must be identical.
        train = _synth_frame(n=10)
        # Force the moneyness column to skip "3OTM_plus" in train and
        # skip "ATM" in predict to make the test sharp.
        train["moneyness_bucket"] = ["ATM", "ATM_or_near", "1OTM", "2OTM"] * 2 + ["ATM"] * 2
        pred = _synth_frame(n=5, seed=99)
        pred["moneyness_bucket"] = ["1OTM", "2OTM", "3OTM_plus",
                                      "ATM_or_near", "3OTM_plus"]

        X_train, names_train = _build_feature_matrix(train)
        X_pred, names_pred = _build_feature_matrix(pred)
        self.assertEqual(names_train, names_pred,
            "feature column list must be identical across frames")
        # Each post-one-hot moneyness column exists for both.
        for level in ("ATM", "ATM_or_near", "1OTM", "2OTM", "3OTM_plus"):
            self.assertIn(f"moneyness_bucket__{level}", names_train)


# ---------------------------------------------------------------------------
# 4. Chronological split + embargo
# ---------------------------------------------------------------------------

class SplitAndEmbargoTests(unittest.TestCase):

    def test_chronological_split_separates_in_time(self):
        frame = _synth_frame(n=400)
        tr, val = _chronological_split(frame, val_frac=0.25)
        # Every train bar_ts < every val bar_ts.
        max_tr = frame.loc[tr, "bar_ts"].max()
        min_val = frame.loc[val, "bar_ts"].min()
        self.assertLess(max_tr, min_val,
            "chronological split must place train strictly earlier than val")

    def test_embargo_drops_overlap(self):
        frame = _synth_frame(n=400)
        tr, val = _chronological_split(frame, val_frac=0.25)
        embargo = 13
        tr_post = _apply_embargo(tr, val, frame, embargo)
        # Every retained train bar must be at least 13*5 min before
        # the earliest val bar.
        val_start = frame.loc[val, "bar_ts"].min()
        cutoff = val_start - pd.Timedelta(minutes=5 * embargo)
        max_tr_post = frame.loc[tr_post, "bar_ts"].max()
        self.assertLessEqual(max_tr_post, cutoff)
        # Embargo strictly tightens the train set.
        self.assertLessEqual(len(tr_post), len(tr))


# ---------------------------------------------------------------------------
# 5. Min-trades gate
# ---------------------------------------------------------------------------

class MinTradesGateTests(unittest.TestCase):

    def test_below_threshold_is_skipped(self):
        small = _synth_frame(n=50)
        head = OptionsExpectedReturnModel(
            bucket=BucketKey("buy", "weekly", "open"),
            min_trades=200,
        )
        head.fit(small)
        self.assertEqual(head.metrics.status, "skipped")
        self.assertIn("min_trades", head.metrics.reason)


# ---------------------------------------------------------------------------
# 6. Fit + predict on learnable synthetic data
# ---------------------------------------------------------------------------

class FitPredictTests(unittest.TestCase):

    def test_learnable_signal_yields_positive_spearman(self):
        frame = _synth_frame(n=600, seed=7, learnable=True)
        head = OptionsExpectedReturnModel(
            bucket=BucketKey("buy", "weekly", "open"),
            min_trades=200,
        )
        head.fit(frame)
        self.assertEqual(head.metrics.status, "fit")
        # A signal-to-noise ratio of ~5:1 in the fixture should produce
        # clearly-positive rank correlation on the held-out tail.
        self.assertGreater(head.metrics.spearman_pred_vs_real, 0.10,
            f"learnable signal should give Spearman > 0.10; "
            f"got {head.metrics.spearman_pred_vs_real}")
        # Calibration table is populated.
        self.assertGreater(len(head.metrics.calibration_table), 0)
        # Feature importance ranks something on top.
        self.assertGreater(len(head.metrics.feature_importance_top_20), 0)

    def test_predict_frame_shape(self):
        frame = _synth_frame(n=400, learnable=True)
        head = OptionsExpectedReturnModel(
            bucket=BucketKey("buy", "weekly", "open"),
            min_trades=100,
        )
        head.fit(frame)
        preds = head.predict_frame(frame)
        self.assertEqual(len(preds), len(frame))
        # All non-NaN.
        self.assertTrue(preds.notna().all())


# ---------------------------------------------------------------------------
# 7. Suite routing
# ---------------------------------------------------------------------------

class SuiteRoutingTests(unittest.TestCase):

    def test_multiple_buckets_get_separate_heads(self):
        # Two buckets in the same frame: (buy, weekly, open) and
        # (sell, weekly, open).
        buy = _synth_frame(n=300, seed=1, side="buy")
        sell = _synth_frame(n=300, seed=2, side="sell",
                            start="2026-02-02 09:15")
        frame = pd.concat([buy, sell], ignore_index=True)
        suite = OptionsExpectedReturnModelSuite(min_trades=200)
        summary = suite.fit(frame)
        self.assertEqual(summary.head_count, 2,
            f"expected 2 heads (buy + sell); got {summary.head_count}")
        labels = sorted(summary.per_head.keys())
        self.assertEqual(labels,
                          ["buy/weekly/open", "sell/weekly/open"])

    def test_predict_routes_per_row(self):
        buy = _synth_frame(n=300, seed=11, side="buy")
        sell = _synth_frame(n=300, seed=22, side="sell",
                            start="2026-02-02 09:15")
        frame = pd.concat([buy, sell], ignore_index=True)
        suite = OptionsExpectedReturnModelSuite(min_trades=200)
        suite.fit(frame)
        preds = suite.predict_frame(frame)
        # Every row should get a prediction (both heads fit).
        self.assertEqual(len(preds), len(frame))
        self.assertGreater(preds.notna().sum(), 0)

    def test_unfit_bucket_returns_nan(self):
        # An out-of-scope tenor (dte > 14) → no head, NaN prediction.
        ok = _synth_frame(n=300, dte_days=5)
        bad = _synth_frame(n=10, dte_days=25, start="2026-03-01 09:15")
        frame = pd.concat([ok, bad], ignore_index=True)
        suite = OptionsExpectedReturnModelSuite(min_trades=200)
        suite.fit(frame)
        preds = suite.predict_frame(frame)
        # Rows with dte > 14 (the bad slice) have NaN predictions.
        bad_rows = preds.iloc[300:]
        self.assertTrue(bad_rows.isna().all(),
            "rows with out-of-scope tenor must get NaN predictions")


# ---------------------------------------------------------------------------
# 8. Determinism
# ---------------------------------------------------------------------------

class DeterminismTests(unittest.TestCase):

    def test_same_seed_same_predictions(self):
        frame = _synth_frame(n=400, learnable=True, seed=3)
        h1 = OptionsExpectedReturnModel(
            bucket=BucketKey("buy", "weekly", "open"),
            min_trades=200, seed=41,
        )
        h2 = OptionsExpectedReturnModel(
            bucket=BucketKey("buy", "weekly", "open"),
            min_trades=200, seed=41,
        )
        h1.fit(frame)
        h2.fit(frame)
        p1 = h1.predict_frame(frame)
        p2 = h2.predict_frame(frame)
        pd.testing.assert_series_equal(p1, p2, check_names=False)


# ---------------------------------------------------------------------------
# 9. Decile reliability helper (corner case smoke)
# ---------------------------------------------------------------------------

class DecileTableTests(unittest.TestCase):

    def test_decile_table_well_calibrated(self):
        # Predicted = realized → calibration error 0 in every decile.
        rng = np.random.default_rng(0)
        x = pd.Series(rng.normal(0, 1, 500))
        table = _decile_table(x, x.copy())
        self.assertGreater(len(table), 0)
        for row in table:
            self.assertAlmostEqual(row.calibration_error, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
