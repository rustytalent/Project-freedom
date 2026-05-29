from __future__ import annotations

import math
import unittest

import numpy as np

from liqpool.validation import (
    assign_time_groups,
    bootstrap_mean_ci,
    cpcv_test_group_sets,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    n_cpcv_backtest_paths,
    probabilistic_sharpe_ratio,
    purged_train_mask,
    sharpe_ratio,
)
import pandas as pd


class SharpeTests(unittest.TestCase):
    def test_sharpe_ratio_basic(self) -> None:
        self.assertEqual(sharpe_ratio([1.0]), 0.0)          # too few
        self.assertEqual(sharpe_ratio([2.0, 2.0, 2.0]), 0.0)  # zero variance
        x = [1.0, -1.0, 1.0, -1.0]
        self.assertAlmostEqual(sharpe_ratio(x), 0.0, places=9)

    def test_psr_monotonic_in_n(self) -> None:
        # Same observed Sharpe, more observations -> more confident edge is real.
        p_small = probabilistic_sharpe_ratio(0.1, n_obs=30)
        p_large = probabilistic_sharpe_ratio(0.1, n_obs=3000)
        self.assertLess(p_small, p_large)
        self.assertGreater(p_large, 0.9)
        # A zero observed Sharpe vs a zero benchmark is a coin flip.
        self.assertAlmostEqual(probabilistic_sharpe_ratio(0.0, n_obs=500), 0.5, places=6)

    def test_expected_max_sharpe_grows_with_trials(self) -> None:
        self.assertEqual(expected_max_sharpe(1, 0.04), 0.0)   # single trial -> no deflation
        self.assertEqual(expected_max_sharpe(50, 0.0), 0.0)   # no spread -> no deflation
        e10 = expected_max_sharpe(10, 0.04)
        e1000 = expected_max_sharpe(1000, 0.04)
        self.assertGreater(e10, 0.0)
        self.assertGreater(e1000, e10)

    def test_dsr_discounts_for_many_trials(self) -> None:
        rng = np.random.default_rng(3)
        # A genuinely positive return stream.
        r = rng.normal(0.06, 1.0, 800)
        d1 = deflated_sharpe_ratio(r, n_trials=1)
        d_many = deflated_sharpe_ratio(r, n_trials=5000)
        self.assertGreater(d1["dsr"], d_many["dsr"])         # more trials -> harder to pass
        self.assertGreaterEqual(d_many["deflation_benchmark_sharpe"],
                                d1["deflation_benchmark_sharpe"])
        # n_trials=1 deflation benchmark must be exactly 0 (reduces to PSR vs 0).
        self.assertEqual(d1["deflation_benchmark_sharpe"], 0.0)

    def test_dsr_kills_a_cherry_picked_zero_edge(self) -> None:
        rng = np.random.default_rng(11)
        # Essentially zero edge but a tiny positive sample Sharpe; under thousands of
        # trials the DSR should be unconvinced (well below 0.95).
        r = rng.normal(0.01, 1.0, 300)
        d = deflated_sharpe_ratio(r, n_trials=2000)
        self.assertLess(d["dsr"], 0.95)


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_mean_ci(self) -> None:
        rng = np.random.default_rng(0)
        r = rng.normal(0.3, 1.0, 2000)
        res = bootstrap_mean_ci(r, ci=0.95, n_boot=2000, seed=1)
        self.assertEqual(res.n, 2000)
        self.assertLess(res.ci_low, res.mean)
        self.assertLess(res.mean, res.ci_high)
        self.assertGreater(res.prob_positive, 0.99)

    def test_bootstrap_empty(self) -> None:
        res = bootstrap_mean_ci([], n_boot=10)
        self.assertEqual(res.n, 0)
        self.assertEqual(res.prob_positive, 0.0)


class CPCVTests(unittest.TestCase):
    def test_test_group_sets_and_paths(self) -> None:
        sets = cpcv_test_group_sets(6, 2)
        self.assertEqual(len(sets), math.comb(6, 2))         # 15 splits
        self.assertTrue(all(len(s) == 2 for s in sets))
        self.assertEqual(n_cpcv_backtest_paths(6, 2), 5)     # C(6,2)*2/6 = 5 paths
        with self.assertRaises(ValueError):
            cpcv_test_group_sets(4, 4)

    def test_assign_time_groups_contiguous(self) -> None:
        ts = pd.date_range("2026-01-01", periods=20, freq="D")
        groups = assign_time_groups(ts, n_groups=4)
        self.assertEqual(groups.shape[0], 20)
        self.assertEqual(set(groups.tolist()), {0, 1, 2, 3})
        # time-ordered: group ids are non-decreasing along time
        self.assertTrue(np.all(np.diff(groups) >= 0))

    def test_purged_train_mask_excludes_overlap_and_embargo(self) -> None:
        # 12 daily events in 4 groups of 3. Test group = {1}. Event whose label window
        # reaches into group 1 must be purged; the event just after must be embargoed.
        starts = pd.date_range("2026-01-01", periods=12, freq="D")
        ends = starts + pd.Timedelta(days=1)
        groups = assign_time_groups(starts, n_groups=4)
        embargo = np.timedelta64(1, "D")
        mask = purged_train_mask(groups, [1], starts, ends, embargo_td=embargo)
        # No test-group event is in train.
        self.assertFalse(mask[groups == 1].any())
        # At least one non-test event was removed by purge/embargo.
        self.assertLess(int(mask.sum()), int((groups != 1).sum()))


if __name__ == "__main__":
    unittest.main()
