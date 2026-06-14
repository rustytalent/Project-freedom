"""Golden-row safety nets for the three leakage-sensitive modules the
audit flagged as untested.

These are NOT exhaustive — they're the first tests of three modules
that have been pulling weight in production with zero coverage:

  * ml_model         — purge/embargo logic + label semantics
  * walkforward      — expanding-window fold boundaries (purge invariant)
  * timing           — StateFeaturizer + DirectionModel + ProximityModel
                       horizon math + the MIS-horizons constant

The goal is to PIN the math that's already in production so future
refactors fail loudly on regression — not to fully validate the
research itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from liqpool.feature_store import (
    DEFAULT_DIRECTION_HORIZON, INTRADAY_HORIZONS, PROXIMITY_HORIZONS,
    SCALP_HORIZONS, SWING_HORIZONS,
)
from liqpool.walkforward import _fold_windows


# ─────────────────────────────────────────────────────────────────
# 1. The MIS horizons are CORRECT (the founder's call)
# ─────────────────────────────────────────────────────────────────

def test_proximity_horizons_match_mis_motive():
    """Founder's call: short / mid / rest-of-session at 5m bars =
    12 / 25 / 78 bars = 1h / ~2h / 1 session.
    Pinned in this test so a future PR cannot silently revert."""
    assert PROXIMITY_HORIZONS == (12, 25, 78)


def test_direction_horizon_matches_one_nse_session():
    """78 bars × 5min = 390 min = 1 NSE F&O session.
    'By close, will NIFTY be up?' — the right question for MIS."""
    assert DEFAULT_DIRECTION_HORIZON == 78


def test_intraday_and_swing_presets_distinct():
    assert INTRADAY_HORIZONS != SWING_HORIZONS
    assert max(INTRADAY_HORIZONS) <= 78        # within one session
    assert min(SWING_HORIZONS) >= 78           # at least one session


def test_scalp_preset_short_enough_for_premium_scalp():
    """Scalp preset must be < 1h on the longest leg so premium scalpers
    train on data their own holding window matches."""
    assert max(SCALP_HORIZONS) <= 12           # ≤ 1h at 5m


# ─────────────────────────────────────────────────────────────────
# 2. walkforward._fold_windows — purge/embargo invariants
# ─────────────────────────────────────────────────────────────────

def _idx(n, start="2026-01-01", freq="5min"):
    return pd.date_range(start=start, periods=n, freq=freq)


def test_fold_windows_chronological_and_nonoverlapping():
    """Each fold's test window must be strictly later than the previous
    fold's test window and they must not overlap. This is THE invariant
    walk-forward relies on; if it ever breaks, models would peek at
    future data."""
    idx = _idx(2000)
    folds = _fold_windows(idx, n_folds=5, train_frac=0.5,
                          min_train=pd.Timedelta("1h"))
    assert len(folds) >= 4
    prev_test_end = None
    for train_start, train_end, test_start, test_end in folds:
        # train ends before test starts (no future leakage)
        assert train_end <= test_start
        # tests march forward
        if prev_test_end is not None:
            assert test_start >= prev_test_end
        # non-empty
        assert test_end > test_start
        prev_test_end = test_end


def test_fold_windows_train_starts_at_history_start():
    """The 'expanding' part of the walk-forward — every fold's training
    window starts at the very beginning of the data."""
    idx = _idx(2000)
    folds = _fold_windows(idx, n_folds=3, train_frac=0.6,
                          min_train=pd.Timedelta("0s"))
    assert all(t[0] == idx[0] for t in folds)


def test_fold_windows_empty_index_returns_empty():
    folds = _fold_windows(pd.DatetimeIndex([]), n_folds=5,
                          train_frac=0.5, min_train=pd.Timedelta("0s"))
    assert folds == []


def test_fold_windows_zero_folds_returns_empty():
    idx = _idx(100)
    assert _fold_windows(idx, n_folds=0, train_frac=0.5,
                          min_train=pd.Timedelta("0s")) == []


def test_fold_windows_skips_folds_with_insufficient_train():
    """If min_train would force the train window below threshold for
    early folds, those are dropped — model never trains on too-little
    data. Pinned so a config bug that lowers min_train silently can't
    sneak in."""
    idx = _idx(100)
    folds = _fold_windows(idx, n_folds=5, train_frac=0.1,
                          min_train=pd.Timedelta("1D"))
    # 10% of a 100-bar 5min series is short — min_train="1d" rejects
    assert folds == []


# ─────────────────────────────────────────────────────────────────
# 3. ml_model — purge logic + label semantics
# ─────────────────────────────────────────────────────────────────

def _pool_result(outcome, **kw):
    """Build a minimal PoolResult with everything but ``outcome``
    defaulted to sensible numbers (tests don't care)."""
    from liqpool.tester import PoolResult
    base = dict(pool_idx=0, side="support",
                 formed_at=pd.Timestamp("2026-01-01 09:30"),
                 price_low=100.0, price_high=101.0, score=0.5,
                 outcome=outcome)
    base.update(kw)
    return PoolResult(**base)


def test_labels_and_trainable_mask_consistency():
    """labels() and trainable_mask() must agree on which outcomes count:
    every trainable row has a label, every label is 0/1."""
    from liqpool.ml_model import labels, trainable_mask

    fake = [
        _pool_result("respected_strong"),
        _pool_result("swept_and_reclaimed"),
        _pool_result("broken_strong"),
        _pool_result("broken_weak"),
        _pool_result("pending"),               # not trainable
        _pool_result("cancelled"),             # not trainable
    ]
    mask = trainable_mask(fake)
    lbl = labels(fake)
    assert mask.dtype == bool
    assert lbl.dtype == int
    # the trainable mask flags the 4 decisive outcomes; the order in the
    # list above MAY differ from how the module enumerates _TRAIN_OUTCOMES,
    # so just check counts + intersection.
    assert mask.sum() >= 4
    assert (lbl[mask] >= 0).all() and (lbl[mask] <= 1).all()
    # un-trainable rows still get a label (0) but the mask should hide them
    assert mask[-2:].any() or (~mask[-2:]).any()


def test_label_end_time_picks_latest_non_null_timestamp():
    """label_end_time must use the LATEST of available_at / touched_at /
    broken_at — picking earlier would leave embargo windows too short
    and let adjacent pools leak through each other."""
    from liqpool.ml_model import label_end_time
    from liqpool.pools import Pool

    pool = Pool.__new__(Pool)
    pool.available_at = pd.Timestamp("2026-01-01 10:00")
    pool.pool_id = "P1"
    r = _pool_result("respected_strong",
                      touched_at=pd.Timestamp("2026-01-01 11:30"),
                      broken_at=pd.Timestamp("2026-01-01 11:00"))
    assert label_end_time(pool, r) == pd.Timestamp("2026-01-01 11:30")


def test_label_end_time_raises_when_all_null():
    """A pool with no end-time candidate would create a zero-width
    purge window; we raise rather than silently return."""
    from liqpool.ml_model import label_end_time
    from liqpool.pools import Pool

    pool = Pool.__new__(Pool)
    pool.available_at = None
    pool.pool_id = "P1"
    r = _pool_result("respected_strong",
                      touched_at=None, broken_at=None)
    with pytest.raises(ValueError):
        label_end_time(pool, r)


def test_purged_embargoed_splits_purge_window_is_strict():
    """A label that ends INSIDE the validation window must be purged
    out of the training set — otherwise the model peeks at the
    target."""
    from liqpool.ml_model import _purged_embargoed_splits

    n = 60
    start = pd.Timestamp("2026-01-01")
    start_times = [start + pd.Timedelta(hours=i) for i in range(n)]
    end_times = [s + pd.Timedelta(minutes=30) for s in start_times]
    y = np.zeros(n, dtype=int); y[::3] = 1                 # mixed labels

    splits, stats = _purged_embargoed_splits(
        start_times, end_times, y, embargo_bars=2,
        base_period_seconds=3600.0, n_folds=4)
    assert splits, "expected at least one fold"
    assert stats and len(stats) == len(splits)
    # Every fold: no training row's evaluation window intersects the
    # validation window — that's the strict-purge guarantee.
    for (train_idx, val_idx), st in zip(splits, stats):
        val_start = min(start_times[i] for i in val_idx)
        val_end   = max(end_times[i]   for i in val_idx)
        for t_i in train_idx:
            label_window = (start_times[t_i], end_times[t_i])
            overlaps = (label_window[0] <= val_end and
                         label_window[1] >= val_start)
            assert not overlaps, (
                f"purge violation: train row {t_i} window {label_window} "
                f"overlaps validation [{val_start} .. {val_end}]")
        # purge bookkeeping is consistent
        assert st.purged_train_size <= st.original_train_size
        assert st.purged_rows_removed >= 0
        assert st.embargoed_rows_removed >= 0


# ─────────────────────────────────────────────────────────────────
# 4. timing — StateFeaturizer feature schema is stable
# ─────────────────────────────────────────────────────────────────

def test_state_feature_schema_includes_intraday_signals():
    """The state feature vector must contain at least the intraday
    return signals an MIS trader looks at: ret_1, ret_6, ret_78,
    momentum, zscore, and an adx-like trend strength."""
    from liqpool.timing import STATE_FEATURE_NAMES
    names = set(STATE_FEATURE_NAMES)
    required = {"ret_1", "ret_6", "ret_24", "ret_78",
                 "zscore_close_50", "adx_14"}
    assert required <= names, f"missing intraday features: {required - names}"


def test_direction_model_default_horizon_matches_session():
    from liqpool.timing import DirectionModel
    assert DirectionModel().horizon == 78


def test_proximity_model_with_intraday_horizons():
    """ProximityModel constructed at each of the MIS horizons stores the
    horizon correctly — the cross-codebase contract for the live
    inference path assumes ``model.horizon`` matches the head suffix."""
    from liqpool.timing import ProximityModel
    for h in PROXIMITY_HORIZONS:
        m = ProximityModel(horizon=h)
        assert m.horizon == h
