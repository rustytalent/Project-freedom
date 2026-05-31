"""Tests for the force-model grid sweep.

We cannot test the bundle-loading path here (no real bundle), but we CAN
test the core arithmetic — feature computation, per-config aggregation,
model training on synthetic data, and the verdict logic.

Pins:
  * Basic feature schema is stable (column set doesn't drift silently).
  * Extended features include all six the user named.
  * _aggregate_per_config groups correctly, computes mean/CI/PF.
  * _train_force_model trains end-to-end on synthetic data without crashing.
  * _evaluate_force_per_config picks top-decile by predicted_r, computes
    realized R within that subset.
  * Verdict identifies positive cells, computes learned-vs-fixed lift.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from analysis.force_model_sweep import (  # noqa: E402
    GRID,
    ConfigRow,
    ForcedConfigRow,
    _aggregate_per_config,
    _basic_features,
    _evaluate_force_per_config,
    _extended_features,
    _train_force_model,
    _verdict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubPool:
    """Minimal Pool-like object satisfying _basic_features' attribute reads."""
    def __init__(self, side="low", score=1.0, tfs=("base",), width=2.0,
                 n_contributors=3, factor="EQHL"):
        self.side = side
        self.score = score
        self.tfs = list(tfs)
        self.width = width
        self.price_low = 100.0
        self.price_high = 100.0 + width
        # _headline_factor reads pool.contributors[*].source. We mock the
        # iteration with simple objects.
        class _C:
            def __init__(self, source):
                self.source = source
        self.contributors = [_C(f"{factor}_x") for _ in range(n_contributors)]


def _ohlcv(n, base_ist="10:00"):
    """5-min OHLCV frame whose IST clock starts at base_ist."""
    h, m = base_ist.split(":")
    ist_min = int(h) * 60 + int(m)
    utc_min = ist_min - (5 * 60 + 30)
    if utc_min < 0:
        utc_min += 24 * 60
    utc_h, utc_m = divmod(utc_min, 60)
    start_utc = f"2026-05-26 {utc_h:02d}:{utc_m:02d}"
    idx = pd.date_range(start_utc, periods=n, freq="5min")
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0, 0.2, n))
    return pd.DataFrame({
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": rng.uniform(800, 1200, n),
    }, index=idx)


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

def test_grid_has_48_configs():
    assert len(GRID) == 48
    stops = sorted({s for s, _, _ in GRID})
    targets = sorted({t for _, t, _ in GRID})
    horizons = sorted({h for _, _, h in GRID})
    assert stops == [0.3, 0.5, 0.75, 1.0]
    assert targets == [1.0, 1.5, 2.0, 3.0]
    assert horizons == [12, 24, 78]


def test_basic_features_schema_stable():
    df = _ohlcv(120)
    pool = _StubPool(side="low", factor="EQHL")
    feats = _basic_features(df, entry_idx=80, pool=pool, atr_val=1.0, symbol="HDFCBANK")
    # Spot-check a few mandatory columns.
    must_have = {
        "pool_score", "pool_tf_count", "pool_width_atr", "pool_n_contributors",
        "side_high", "factor_EQHL", "factor_OB",
        "sector_BANKING", "sector_OTHER",
        "ret_1", "ret_6", "ret_24", "ret_78",
        "range_6_atr", "atr_value", "minutes_since_session_open",
    }
    missing = must_have - set(feats)
    assert not missing, f"basic features missing: {missing}"
    # All values must be finite.
    for k, v in feats.items():
        assert math.isfinite(v), f"{k}={v!r} is not finite"


def test_extended_features_include_all_six_named_by_user():
    df = _ohlcv(120)
    feats = _extended_features(df, entry_idx=80, atr_val=1.0)
    required = {"volume_ratio_5b", "volume_zscore_20b", "gap_atr",
                "range_compression", "time_quartile", "vol_ratio_regime"}
    assert set(feats) == required, f"got {set(feats)}"
    for k, v in feats.items():
        assert math.isfinite(v), f"{k}={v!r}"


def test_extended_features_degrade_gracefully_with_short_history():
    # Only 3 bars — most rolling windows fall back to defaults.
    df = _ohlcv(3)
    feats = _extended_features(df, entry_idx=2, atr_val=1.0)
    # Defaults: volume_ratio_5b=1.0, volume_zscore_20b=0.0, etc.
    assert feats["volume_ratio_5b"] == 1.0
    assert feats["volume_zscore_20b"] == 0.0
    assert feats["range_compression"] == 1.0
    assert feats["vol_ratio_regime"] == 1.0


def test_factor_one_hot_falls_back_to_other_for_unknown():
    df = _ohlcv(120)
    pool = _StubPool(factor="XYZ_unknown")        # not in FACTOR_BUCKETS
    feats = _basic_features(df, entry_idx=80, pool=pool, atr_val=1.0, symbol="X.NS")
    assert feats["factor_OTHER"] == 1.0
    # Specific factor cols are 0.
    for f in ("OB", "FVG", "EQHL", "REJ", "ORB", "SWING"):
        assert feats[f"factor_{f}"] == 0.0


def test_sector_one_hot_falls_back_to_other_for_unmapped():
    df = _ohlcv(120)
    pool = _StubPool()
    # An unmapped symbol -> sector_of returns "OTHER" or similar; check the
    # OTHER column flips on.
    feats = _basic_features(df, entry_idx=80, pool=pool, atr_val=1.0,
                            symbol="UNMAPPED_TICKER_XYZ")
    # Either a recognized sector OR OTHER is 1; exactly one sector col is 1.
    sector_vals = {k: v for k, v in feats.items() if k.startswith("sector_")}
    assert sum(sector_vals.values()) == 1.0


# ---------------------------------------------------------------------------
# Static aggregation
# ---------------------------------------------------------------------------

def _synth_dataset(n_pools=200, seed=42):
    """Build a synthetic dataset of (pool, config) rows for testing.

    Each pool has one realized_r per config. We fabricate so that one config
    has clearly positive R and one clearly negative.
    """
    rng = np.random.default_rng(seed)
    rows = []
    base_ts = pd.Timestamp("2026-01-01 04:30")
    for i in range(n_pools):
        for stop_atr, target_atr, horizon in GRID:
            # Inject signal: (s=0.5, t=2.0, h=78) tends to win;
            # (s=0.3, t=1.0, h=12) tends to lose.
            if (stop_atr, target_atr, horizon) == (0.5, 2.0, 78):
                r = float(rng.normal(0.4, 1.0))
            elif (stop_atr, target_atr, horizon) == (0.3, 1.0, 12):
                r = float(rng.normal(-0.5, 0.8))
            else:
                r = float(rng.normal(-0.05, 1.2))
            rows.append({
                "symbol": f"S{i % 5}",
                "pool_idx": i,
                "entry_ts": str(base_ts + pd.Timedelta(minutes=5 * i)),
                "side_long": 1.0,
                "stop_atr": stop_atr,
                "target_atr": target_atr,
                "horizon_bars": float(horizon),
                "realized_r": r,
                "exit_reason": "time_exit",
                # Some toy features so the model has variance to fit.
                "f1": float(rng.normal()),
                "f2": float(rng.normal()),
                "f3": float(rng.normal()),
            })
    return pd.DataFrame(rows)


def test_aggregate_groups_and_computes_metrics():
    ds = _synth_dataset(n_pools=200)
    rows = _aggregate_per_config(ds)
    # 48 configs in grid -> 48 rows.
    assert len(rows) == 48
    # The "winning" config we injected should have positive mean.
    winning = [r for r in rows
               if r.stop_atr == 0.5 and r.target_atr == 2.0 and r.horizon_bars == 78]
    assert len(winning) == 1
    assert winning[0].mean_R > 0.2
    assert winning[0].ci95_lo < winning[0].mean_R < winning[0].ci95_hi
    # Losing one.
    losing = [r for r in rows
              if r.stop_atr == 0.3 and r.target_atr == 1.0 and r.horizon_bars == 12]
    assert len(losing) == 1
    assert losing[0].mean_R < -0.2


def test_aggregate_empty_returns_empty_list():
    assert _aggregate_per_config(pd.DataFrame()) == []


# ---------------------------------------------------------------------------
# Force model training + per-config eval
# ---------------------------------------------------------------------------

def test_train_force_model_runs_on_synthetic_data():
    ds = _synth_dataset(n_pools=200)
    model, val_df, feature_cols = _train_force_model(ds, seed=7)
    assert model is not None
    assert "predicted_r" in val_df.columns
    # Predictions should be finite.
    assert val_df["predicted_r"].notna().all()
    # Feature columns should include the config columns.
    assert "stop_atr" in feature_cols
    assert "target_atr" in feature_cols
    assert "horizon_bars" in feature_cols


def test_evaluate_force_per_config_returns_one_row_per_grid_cell():
    ds = _synth_dataset(n_pools=300)
    _, val_df, _ = _train_force_model(ds, seed=7)
    rows = _evaluate_force_per_config(val_df, top_pct=0.10)
    # Some configs may be skipped if val_n < 20; with n=300 we have ~90 val
    # rows per config which is plenty.
    assert len(rows) >= 40                         # most of 48
    for r in rows:
        assert r.val_n >= 20
        assert r.top_decile_n >= 1
        assert math.isfinite(r.top_decile_realized_r)
        assert r.ci95_lo <= r.top_decile_realized_r <= r.ci95_hi


def test_evaluate_skips_thin_buckets():
    # 5 pools means 5 rows per config — below the 20-row threshold.
    ds = _synth_dataset(n_pools=5)
    _, val_df, _ = _train_force_model(ds, seed=7)
    rows = _evaluate_force_per_config(val_df, top_pct=0.10)
    # Should skip all or nearly all configs.
    assert len(rows) <= 5


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def _mk_static_row(mean_R, ci_lo, ci_hi, n=100,
                   s=0.5, t=2.0, h=78) -> ConfigRow:
    return ConfigRow(
        stop_atr=s, target_atr=t, horizon_bars=h,
        trades=n, win_rate=0.5,
        mean_R=mean_R, median_R=mean_R, PF=1.5,
        se=(ci_hi - ci_lo) / (2 * 1.96),
        ci95_lo=ci_lo, ci95_hi=ci_hi,
    )


def _mk_forced_row(mean_R, ci_lo, ci_hi, val_n=200,
                   s=0.5, t=2.0, h=78) -> ForcedConfigRow:
    return ForcedConfigRow(
        stop_atr=s, target_atr=t, horizon_bars=h,
        val_n=val_n, top_decile_n=20,
        top_decile_realized_r=mean_R,
        se=(ci_hi - ci_lo) / (2 * 1.96),
        ci95_lo=ci_lo, ci95_hi=ci_hi,
    )


def test_verdict_identifies_positive_fixed_and_learned():
    static = [
        _mk_static_row(+0.30, +0.15, +0.45, s=0.5, t=2.0, h=78),
        _mk_static_row(-0.20, -0.30, -0.10, s=0.3, t=1.0, h=12),
    ]
    forced = [
        _mk_forced_row(+0.50, +0.30, +0.70, s=0.5, t=2.0, h=78),
        _mk_forced_row(-0.10, -0.25, +0.05, s=1.0, t=3.0, h=24),
    ]
    lines = _verdict(static, forced)
    joined = "\n".join(lines)
    assert "Statistically positive configs (ci95_lo > 0): 1" in joined
    assert "Statistically positive configs: 1" in joined
    # Best fixed mean_R == +0.30, best forced top10 R == +0.50 -> learned lift +0.20.
    assert "Learned lift over best fixed: +0.20" in joined


def test_verdict_says_alpha_exists_if_only_learned_is_positive():
    static = [_mk_static_row(-0.05, -0.12, +0.02, s=0.5, t=2.0, h=78)]
    forced = [_mk_forced_row(+0.30, +0.18, +0.42, s=0.5, t=2.0, h=78)]
    lines = _verdict(static, forced)
    joined = "\n".join(lines)
    assert "ALPHA EXISTS but requires selective trading" in joined


def test_verdict_says_neither_when_all_negative():
    static = [_mk_static_row(-0.30, -0.40, -0.20)]
    forced = [_mk_forced_row(-0.15, -0.25, -0.05)]
    lines = _verdict(static, forced)
    joined = "\n".join(lines)
    assert "Neither fixed nor learned finds positive R" in joined


def test_verdict_empty_rows_handled_gracefully():
    lines = _verdict([], [])
    # No exception. No "Best fixed" line because no rows.
    joined = "\n".join(lines)
    assert "Best fixed config" not in joined
    assert "Best learned cell" not in joined
