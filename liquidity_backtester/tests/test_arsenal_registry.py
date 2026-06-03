"""Tests for liqpool.arsenal.registry — named lookup of alphas."""
from __future__ import annotations

import unittest

import pandas as pd

from liqpool.arsenal.base import Alpha, AlphaSignal
from liqpool.arsenal.registry import (
    AlphaRegistry,
    default_registry,
    research_registry,
)


class _NamedAlpha(Alpha):
    """Minimal Alpha subclass used as a test fixture."""
    def __init__(self, name: str):
        self._n = name
    @property
    def name(self) -> str:
        return self._n
    def candidates(self, *, symbol, df_base, atr_series, extra=None):
        return []


class AlphaRegistryTests(unittest.TestCase):
    def test_register_and_get(self):
        reg = AlphaRegistry()
        reg.register(_NamedAlpha("a"))
        reg.register(_NamedAlpha("b"))
        self.assertEqual(reg.names(), ["a", "b"])
        self.assertEqual(reg.get("a").name, "a")
        self.assertIn("a", reg)
        self.assertEqual(len(reg), 2)

    def test_register_duplicate_raises(self):
        reg = AlphaRegistry()
        reg.register(_NamedAlpha("a"))
        with self.assertRaises(ValueError):
            reg.register(_NamedAlpha("a"))

    def test_register_non_alpha_raises(self):
        reg = AlphaRegistry()
        with self.assertRaises(TypeError):
            reg.register("not an alpha")          # type: ignore[arg-type]

    def test_register_empty_name_raises(self):
        class _Empty(_NamedAlpha):
            @property
            def name(self):
                return ""
        reg = AlphaRegistry()
        with self.assertRaises(ValueError):
            reg.register(_Empty(""))

    def test_get_unknown_raises(self):
        reg = AlphaRegistry()
        with self.assertRaises(KeyError):
            reg.get("nope")

    def test_subset_preserves_order(self):
        reg = AlphaRegistry()
        for n in ("a", "b", "c"):
            reg.register(_NamedAlpha(n))
        sub = reg.subset(["c", "a"])
        self.assertEqual([a.name for a in sub], ["c", "a"])

    def test_subset_unknown_raises(self):
        reg = AlphaRegistry()
        reg.register(_NamedAlpha("a"))
        with self.assertRaises(KeyError):
            reg.subset(["a", "missing"])

    def test_all_returns_sorted(self):
        reg = AlphaRegistry()
        for n in ("c", "a", "b"):
            reg.register(_NamedAlpha(n))
        self.assertEqual([a.name for a in reg.all()], ["a", "b", "c"])


class DefaultRegistryTests(unittest.TestCase):
    def test_default_registry_has_baseline_alphas(self):
        reg = default_registry()
        self.assertTrue({
            "pool_reach",
            "mean_reversion",
            "momentum",
            "proximity_journey_baseline",
            "proximity_journey",
        }.issubset(set(reg.names())))

    def test_default_registry_excludes_sparse_alphas(self):
        # The 4 sparse alphas moved to research_registry to keep
        # routine sweeps statistically interpretable.
        reg = default_registry()
        for sparse in ("distance_5_8_journey",
                        "proximity_direction_soft",
                        "opening_range_to_pool",
                        "sector_rotation_journey"):
            self.assertNotIn(sparse, reg.names(),
                f"sparse alpha {sparse!r} should be in research_registry, "
                f"not default_registry — it produces <200 OOS trades "
                f"per typical Arsenal run")


class ResearchRegistryTests(unittest.TestCase):
    def test_research_registry_is_superset_of_default(self):
        d = set(default_registry().names())
        r = set(research_registry().names())
        self.assertTrue(d.issubset(r),
            f"research_registry must be a superset of default_registry. "
            f"Missing: {d - r}")

    def test_research_registry_includes_sparse_alphas(self):
        reg = research_registry()
        for sparse in ("distance_5_8_journey",
                        "proximity_direction_soft",
                        "opening_range_to_pool",
                        "sector_rotation_journey"):
            self.assertIn(sparse, reg.names(),
                f"sparse alpha {sparse!r} should be available via "
                f"research_registry for explicit research sweeps")

    def test_research_registry_total_count(self):
        # default (5) + sparse (4) = 9 alphas.
        self.assertEqual(len(research_registry()), 9)


if __name__ == "__main__":
    unittest.main()
