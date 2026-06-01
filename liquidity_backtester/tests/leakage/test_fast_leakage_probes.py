from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from liqpool.features import LevelCandidate
from liqpool.leakage_probes import (
    assert_no_future_features,
    assert_snapshot_distance_causal,
    label_shuffle_auc_probe,
)
from liqpool.pools import Pool
from liqpool.tester import PoolResult
from liqpool.timing import Snapshot, StateFeaturizer, generate_snapshots


def _bars(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01 09:15", periods=n, freq="5min")
    base = np.linspace(100.0, 102.0, n)
    return pd.DataFrame({
        "open": base - 0.05,
        "high": base + 0.50,
        "low": base - 0.50,
        "close": base,
        "volume": np.full(n, 1000.0),
    }, index=idx)


def _pool(df: pd.DataFrame) -> Pool:
    known_at = df.index[50]
    contributor = LevelCandidate(
        side="high",
        price=130.0,
        ts=df.index[45],
        known_at=known_at,
        source="TEST",
        half_width=0.25,
        tf="base",
    )
    return Pool(
        side="high",
        price_low=129.75,
        price_high=130.25,
        formed_at=df.index[45],
        available_at=known_at,
        contributors=[contributor],
        score=1.0,
        tfs=["base"],
        asset="TEST",
    )


class FastLeakageProbeTests(unittest.TestCase):
    def test_label_shuffle_probe_destroys_auc_association(self) -> None:
        rng = np.random.default_rng(7)
        y = rng.binomial(1, 0.5, 2000)
        score = y * 0.70 + rng.normal(0, 0.20, len(y))
        result = label_shuffle_auc_probe(y, score, seed=11, repeats=96)

        self.assertGreater(result.observed_auc, 0.90)
        self.assertTrue(result.passed)
        self.assertGreaterEqual(result.shuffled_mean_auc, 0.45)
        self.assertLessEqual(result.shuffled_mean_auc, 0.55)

    def test_future_mask_probe_rejects_feature_after_decision(self) -> None:
        safe = [{
            "feature": "ret_1",
            "as_of_ts": "2026-01-01 09:20:00",
            "decision_ts": "2026-01-01 09:20:00",
        }]
        assert_no_future_features(safe)

        unsafe = [{
            "feature": "ret_1",
            "as_of_ts": "2026-01-01 09:25:00",
            "decision_ts": "2026-01-01 09:20:00",
        }]
        with self.assertRaises(AssertionError):
            assert_no_future_features(unsafe)

    def test_snapshot_distance_probe_matches_decision_close(self) -> None:
        ts = pd.Timestamp("2026-01-01 10:00:00")
        pool = Pool(
            side="high",
            price_low=119.5,
            price_high=120.5,
            formed_at=ts,
            available_at=ts,
            contributors=[],
            score=1.0,
            tfs=["base"],
        )
        snap = Snapshot(
            bar_idx=10,
            ts=ts,
            close=100.0,
            atr_val=2.0,
            state={},
            pool_touches=[(0, None, 10.0, "above")],
            n_future_bars=10,
        )
        assert_snapshot_distance_causal(snap, [pool])

        bad = Snapshot(
            bar_idx=10,
            ts=ts,
            close=100.0,
            atr_val=2.0,
            state={},
            pool_touches=[(0, None, 1.0, "above")],
            n_future_bars=10,
        )
        with self.assertRaises(AssertionError):
            assert_snapshot_distance_causal(bad, [pool])

    def test_generate_snapshots_distance_is_unchanged_by_future_mutation(self) -> None:
        df = _bars(105)
        pool = _pool(df)
        result = PoolResult(
            pool_idx=0,
            side=pool.side,
            formed_at=pool.formed_at,
            price_low=pool.price_low,
            price_high=pool.price_high,
            score=pool.score,
            outcome="untouched",
        )

        snap = generate_snapshots(
            df, [pool], [result], StateFeaturizer(df),
            window_start=df.index[80],
            window_end=df.index[80],
            sample_every=1,
            max_horizon=10,
            intraday_session_only=False,    # leakage probe — not an MIS test
        )[0]
        assert_snapshot_distance_causal(snap, [pool])
        distance_before = snap.pool_touches[0][2]

        mutated = df.copy()
        mutated.iloc[81:, mutated.columns.get_loc("close")] += 1000.0
        mutated.iloc[81:, mutated.columns.get_loc("high")] += 1000.0
        mutated.iloc[81:, mutated.columns.get_loc("low")] += 1000.0
        mutated_snap = generate_snapshots(
            mutated, [pool], [result], StateFeaturizer(mutated),
            window_start=mutated.index[80],
            window_end=mutated.index[80],
            sample_every=1,
            max_horizon=10,
            intraday_session_only=False,    # leakage probe — not an MIS test
        )[0]
        assert_snapshot_distance_causal(mutated_snap, [pool])
        self.assertAlmostEqual(distance_before, mutated_snap.pool_touches[0][2], places=12)


if __name__ == "__main__":
    unittest.main()
