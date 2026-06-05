"""Tests for the Daily Research Brief generator + plain-text renderer.

Pinned contracts:

  * generate_brief produces a BriefDocument with all 9 top-level
    sections, even when the input report is a sparse stub.
  * Missing report attributes produce ``_status: "pending"`` stub
    blocks, not crashes.
  * Sector regime classification matches the documented rules
    (trending_up / trending_down / chopping / neutral buckets).
  * Top watchlist drops candidates below the proximity threshold and
    flags sector-misalignment in avoidance_note.
  * The avoid_list always emits ALL_BASKET when the watchlist is
    empty — this is the SEBI-safest, customer-most-valuable section.
  * The renderer never emits tipster vocabulary
    (buy/sell/long/short/entry/stop/target).
  * JSON serialisation round-trips (to_dict produces a JSON-compatible
    plain dict).
"""
from __future__ import annotations

import json
import unittest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from liqpool.products.brief_renderer import (
    TIPSTER_VOCABULARY,
    _assert_no_tipster_language,
    render_email,
)
from liqpool.products.daily_brief import (
    BriefDocument,
    SectorRegime,
    SCHEMA_VERSION,
    _classify_sector_regimes,
    _confidence_bucket,
    generate_brief,
)


# ---------------------------------------------------------------------------
# Stub bundle — minimal MultiAssetReport-shaped object the generator reads.
# ---------------------------------------------------------------------------

@dataclass
class _StubContributor:
    source: str = "EQH"
    ts: Any = field(default_factory=lambda: pd.Timestamp("2026-05-01"))


@dataclass
class _StubPool:
    side: str
    price_low: float
    price_high: float
    tfs: List[str]
    score: float = 1.0
    contributors: List[_StubContributor] = field(
        default_factory=lambda: [_StubContributor("EQH")]
    )
    available_at: Any = field(default_factory=lambda: pd.Timestamp("2026-05-01"))
    formed_at: Any = field(default_factory=lambda: pd.Timestamp("2026-05-01"))
    width: float = 1.0

    @property
    def mid(self) -> float:
        return (self.price_low + self.price_high) / 2.0


@dataclass
class _StubResult:
    pool_idx: int = 0
    touched_at: Any = None
    broken_at: Any = None


@dataclass
class _StubWalkforward:
    oos_pools: List[_StubPool] = field(default_factory=list)
    oos_results: List[_StubResult] = field(default_factory=list)


@dataclass
class _StubAssetData:
    base_df: pd.DataFrame
    walkforward: _StubWalkforward


def _bars(n: int = 80, base_price: float = 100.0, drift: float = 0.0,
          start: str = "2026-05-22 03:45") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="5min")
    rng = np.random.default_rng(0)
    close = base_price + np.cumsum(rng.normal(drift, 0.05, n))
    return pd.DataFrame({
        "open": close, "high": close + 0.2, "low": close - 0.2,
        "close": close, "volume": np.full(n, 1000.0),
    }, index=idx)


class _StubProximityModel:
    def __init__(self, returned_p: float = 0.40) -> None:
        self.returned_p = float(returned_p)
        self.val_auc = 0.85
        # Capture the kwargs every predict_one call sees, so tests can
        # assert the generator passes REAL inputs (not placeholders).
        self.last_call_kwargs: Dict[str, Any] = {}

    def predict_one(self, pool, dist_atr, side, state, quality_pred,
                    atr_val=1.0):
        self.last_call_kwargs = {
            "dist_atr": float(dist_atr),
            "side": str(side),
            "state_keys": sorted(state.keys()) if state else [],
            "quality_pred": float(quality_pred),
            "atr_val": float(atr_val),
        }
        return self.returned_p


class _DistanceSensitiveStubModel:
    """Returns P that scales inversely with distance. Used to pin that
    the generator passes real per-pool distances (not a placeholder)."""
    def __init__(self) -> None:
        self.val_auc = 0.85

    def predict_one(self, pool, dist_atr, side, state, quality_pred,
                    atr_val=1.0):
        # Far away -> low P; close -> high P. Hard linear shape so the
        # test can pin the relationship.
        return max(0.0, 1.0 - float(dist_atr) / 50.0)


class _StubDirectionModel:
    val_auc = 0.60


@dataclass
class _StubReport:
    assets: Dict[str, _StubAssetData]
    unified_direction: Optional[Any] = None
    unified_proximity: Dict[int, Any] = field(default_factory=dict)
    unified_oos_audit: Optional[Any] = None


def _two_asset_report(p_touch: float = 0.40) -> _StubReport:
    df_a = _bars(base_price=100.0)
    df_b = _bars(base_price=500.0)
    # Pool below price -> long candidate.
    pool_a = _StubPool("low", price_low=90.0, price_high=91.0,
                       tfs=["base", "15min"])
    # Pool above price -> short candidate.
    pool_b = _StubPool("high", price_low=520.0, price_high=521.0,
                       tfs=["base"])
    ad_a = _StubAssetData(
        base_df=df_a,
        walkforward=_StubWalkforward(
            oos_pools=[pool_a],
            oos_results=[_StubResult(pool_idx=0)],
        ),
    )
    ad_b = _StubAssetData(
        base_df=df_b,
        walkforward=_StubWalkforward(
            oos_pools=[pool_b],
            oos_results=[_StubResult(pool_idx=0)],
        ),
    )
    return _StubReport(
        assets={"HDFCBANK": ad_a, "TCS": ad_b},
        unified_direction=_StubDirectionModel(),
        unified_proximity={12: _StubProximityModel(p_touch)},
    )


# ---------------------------------------------------------------------------
# Sector regime classifier tests
# ---------------------------------------------------------------------------

class ClassifySectorRegimesTests(unittest.TestCase):

    def test_trending_up_when_both_returns_positive(self) -> None:
        out = _classify_sector_regimes({
            "BANKING": {"ret_5d": 0.011, "ret_20d": 0.030},
        })
        self.assertEqual(out["trending_up"], ["BANKING"])
        self.assertEqual(out["trending_down"], [])

    def test_trending_down_when_both_returns_negative(self) -> None:
        out = _classify_sector_regimes({
            "FMCG": {"ret_5d": -0.011, "ret_20d": -0.052},
        })
        self.assertEqual(out["trending_down"], ["FMCG"])

    def test_chopping_when_opposite_signs_small_magnitudes(self) -> None:
        out = _classify_sector_regimes({
            "AUTO": {"ret_5d": 0.005, "ret_20d": -0.003},
        })
        self.assertEqual(out["chopping"], ["AUTO"])

    def test_neutral_when_opposite_signs_large_magnitudes(self) -> None:
        # Big opposite-sign moves get bucketed as neutral, not chopping.
        out = _classify_sector_regimes({
            "PHARMA": {"ret_5d": 0.04, "ret_20d": -0.05},
        })
        self.assertEqual(out["neutral"], ["PHARMA"])


# ---------------------------------------------------------------------------
# Confidence bucket helper tests
# ---------------------------------------------------------------------------

class ConfidenceBucketTests(unittest.TestCase):

    def test_buckets(self) -> None:
        self.assertEqual(_confidence_bucket(0.20), "low")
        self.assertEqual(_confidence_bucket(0.50), "moderate")
        self.assertEqual(_confidence_bucket(0.65), "high")
        self.assertEqual(_confidence_bucket(0.85), "very_high")


# ---------------------------------------------------------------------------
# Top-level generator tests
# ---------------------------------------------------------------------------

class GenerateBriefTests(unittest.TestCase):

    def test_all_nine_top_level_sections_present(self) -> None:
        report = _two_asset_report()
        brief = generate_brief(report, trading_date_ist="2026-06-03")
        d = brief.to_dict()
        for k in ("schema_version", "brief_metadata", "index_regime",
                   "sector_regime", "top_watchlist",
                   "options_suitability", "avoid_list", "key_zones",
                   "confidence_notes", "yesterday_audit"):
            self.assertIn(k, d, f"section {k} missing from brief")

    def test_schema_version_pinned(self) -> None:
        report = _two_asset_report()
        brief = generate_brief(report, trading_date_ist="2026-06-03")
        self.assertEqual(brief.schema_version, SCHEMA_VERSION)
        self.assertEqual(brief.schema_version, "1.0")

    def test_empty_report_does_not_crash(self) -> None:
        # Sparse stub with no assets at all.
        report = _StubReport(assets={})
        brief = generate_brief(report, trading_date_ist="2026-06-03")
        self.assertEqual(brief.top_watchlist, [])
        # avoid_list must contain ALL_BASKET when watchlist is empty.
        self.assertTrue(
            any(a.symbol == "ALL_BASKET" for a in brief.avoid_list),
            "Empty bundle should yield ALL_BASKET avoid recommendation"
        )

    def test_stub_sections_are_pending(self) -> None:
        report = _two_asset_report()
        brief = generate_brief(report, trading_date_ist="2026-06-03")
        for stub_name in ("index_regime", "options_suitability",
                           "yesterday_audit"):
            stub = getattr(brief, stub_name)
            self.assertEqual(stub.get("_status"), "pending",
                f"{stub_name} should have _status='pending' in v1")

    def test_brief_id_format(self) -> None:
        report = _two_asset_report()
        brief = generate_brief(report, trading_date_ist="2026-06-03")
        self.assertEqual(brief.brief_metadata.brief_id, "BRIEF_2026_06_03")

    def test_watchlist_populated_when_p_touch_above_threshold(self) -> None:
        # Stub returns p=0.40 from the proximity model — above 0.20.
        report = _two_asset_report(p_touch=0.40)
        brief = generate_brief(report, trading_date_ist="2026-06-03")
        self.assertGreater(len(brief.top_watchlist), 0)

    def test_watchlist_empty_when_p_touch_below_threshold(self) -> None:
        # Stub returns p=0.10 — below 0.20 threshold.
        report = _two_asset_report(p_touch=0.10)
        brief = generate_brief(report, trading_date_ist="2026-06-03")
        self.assertEqual(len(brief.top_watchlist), 0)
        # ALL_BASKET avoid must fire.
        self.assertTrue(any(a.symbol == "ALL_BASKET"
                             for a in brief.avoid_list))

    def test_json_roundtrip(self) -> None:
        report = _two_asset_report()
        brief = generate_brief(report, trading_date_ist="2026-06-03")
        as_dict = brief.to_dict()
        serialised = json.dumps(as_dict, default=str)
        deserialised = json.loads(serialised)
        self.assertEqual(deserialised["schema_version"], "1.0")
        self.assertEqual(deserialised["brief_metadata"]["brief_id"],
                          "BRIEF_2026_06_03")


# ---------------------------------------------------------------------------
# Renderer tests
# ---------------------------------------------------------------------------

class RenderEmailTests(unittest.TestCase):

    def test_renders_non_empty_text(self) -> None:
        report = _two_asset_report()
        brief = generate_brief(report, trading_date_ist="2026-06-03")
        text = render_email(brief)
        self.assertGreater(len(text), 200,
            "rendered brief should not be a stub")
        self.assertIn("Daily Research Brief", text)
        self.assertIn("2026-06-03", text)

    def test_no_tipster_vocabulary_in_rendered_brief(self) -> None:
        report = _two_asset_report()
        brief = generate_brief(report, trading_date_ist="2026-06-03")
        text = render_email(brief)
        lowered = text.lower()
        for phrase in TIPSTER_VOCABULARY:
            self.assertNotIn(phrase, lowered,
                f"tipster phrase {phrase!r} leaked into rendered brief")

    def test_empty_bundle_renders_avoid_basket(self) -> None:
        report = _StubReport(assets={})
        brief = generate_brief(report, trading_date_ist="2026-06-03")
        text = render_email(brief)
        self.assertIn("ALL_BASKET", text)
        self.assertIn("no_setup_today_low_proximity_across_basket", text)

    def test_guardrail_raises_on_tipster_phrase(self) -> None:
        with self.assertRaises(RuntimeError):
            _assert_no_tipster_language(
                "the model says buy hdfcbank at 770 with stop loss at 760")

    def test_guardrail_passes_on_research_language(self) -> None:
        # Pure context language must NOT trigger.
        _assert_no_tipster_language(
            "The proximity model assigns a 70% probability to the level "
            "being tested today. Context only; consult your own thesis."
        )


# ---------------------------------------------------------------------------
# Integration: outcome log + strike translator wired into brief generator
# ---------------------------------------------------------------------------

class BriefWithIndexDataTests(unittest.TestCase):
    """When the caller passes index_data + indexes_covered, the
    options_suitability section is populated by the strike translator.
    """

    def _index_data(self) -> Dict[str, Any]:
        return {
            "NIFTY50": {
                "previous_close": 24521.30,
                "vol_regime": "normal",
                "vol_regime_zscore_20d": 0.34,
                "directional_bias": "mild_up",
                "expected_range_today_atr": 1.6,
                "path_efficiency_30": 0.40,    # chop-ish
                "direction_changes_30": 12.0,
                "proximity_predictions": [
                    {"level": 24500, "p_test_today": 0.78,
                     "p_test_within_60min": 0.34,
                     "side_from_open": "below",
                     "key_level_type": "demand_pool"},
                    {"level": 24735, "p_test_today": 0.22,
                     "p_test_within_60min": 0.05,
                     "side_from_open": "above",
                     "key_level_type": "supply_pool"},
                ],
            },
        }

    def test_options_suitability_populated_when_index_data_passed(self) -> None:
        report = _two_asset_report()
        brief = generate_brief(
            report,
            trading_date_ist="2026-06-03",
            indexes_covered=["NIFTY50"],
            index_data=self._index_data(),
        )
        suit = brief.options_suitability
        self.assertIn("NIFTY50", suit)
        nifty = suit["NIFTY50"]
        self.assertEqual(nifty["directional_bias"], "mild_up")
        self.assertIn("strike_levels_in_play", nifty)
        self.assertEqual(len(nifty["strike_levels_in_play"]), 2)
        self.assertEqual(nifty["strike_levels_in_play"][0]["strike"], 24500.0)
        self.assertIn("regime_for_premium_buyers", nifty)

    def test_options_suitability_per_index_stub_for_missing_data(self) -> None:
        # NIFTY50 has data, BANKNIFTY does not -> per-index stub for BANKNIFTY.
        report = _two_asset_report()
        brief = generate_brief(
            report,
            trading_date_ist="2026-06-03",
            indexes_covered=["NIFTY50", "BANKNIFTY"],
            index_data=self._index_data(),
        )
        self.assertIn("NIFTY50", brief.options_suitability)
        self.assertEqual(
            brief.options_suitability["BANKNIFTY"]["_status"], "pending")


class BriefWithOutcomeLogTests(unittest.TestCase):
    """When the caller passes an OutcomeLogWriter, the generator logs
    one PredictionRecord per call (watchlist + avoid + strike).
    """

    def test_writer_logs_proximity_predictions_from_watchlist(self) -> None:
        from liqpool.products.outcome_log import OutcomeLogWriter
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            writer = OutcomeLogWriter(root=tmp)
            report = _two_asset_report(p_touch=0.40)
            brief = generate_brief(
                report,
                trading_date_ist="2026-06-03",
                outcome_log_writer=writer,
            )
            preds = writer.read_predictions("2026-06-03")
            # 2 watchlist entries + at least 1 avoid (ALL_BASKET fires
            # when watchlist exists but no setup AND when watchlist empty;
            # here watchlist non-empty so ALL_BASKET should NOT fire).
            prox_count = (preds["prediction_type"] == "proximity").sum()
            self.assertGreaterEqual(int(prox_count), 2,
                "should log one proximity prediction per watchlist entry")

    def test_writer_logs_avoid_basket_when_no_setup(self) -> None:
        from liqpool.products.outcome_log import OutcomeLogWriter
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            writer = OutcomeLogWriter(root=tmp)
            report = _two_asset_report(p_touch=0.05)    # below threshold
            brief = generate_brief(
                report,
                trading_date_ist="2026-06-03",
                outcome_log_writer=writer,
            )
            preds = writer.read_predictions("2026-06-03")
            avoid_rows = preds[preds["prediction_type"] == "avoidance"]
            self.assertGreater(len(avoid_rows), 0)
            self.assertIn("ALL_BASKET", set(avoid_rows["symbol"]))

    def test_writer_logs_options_strike_predictions(self) -> None:
        from liqpool.products.outcome_log import OutcomeLogWriter
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            writer = OutcomeLogWriter(root=tmp)
            report = _two_asset_report()
            brief = generate_brief(
                report,
                trading_date_ist="2026-06-03",
                indexes_covered=["NIFTY50"],
                index_data=BriefWithIndexDataTests()._index_data(),
                outcome_log_writer=writer,
            )
            preds = writer.read_predictions("2026-06-03")
            strike_rows = preds[preds["prediction_type"] == "options_strike"]
            self.assertGreater(len(strike_rows), 0)
            self.assertIn("NIFTY50", set(strike_rows["symbol"]))

    def test_predict_one_receives_real_distance_not_placeholder(self) -> None:
        """REGRESSION TEST.

        Before this test existed, the daily brief generator passed
        dist_atr=1.0 as a hardcoded placeholder to every predict_one
        call, regardless of the actual price-to-pool distance. That
        caused the proximity model to return ~98% probability for
        every pool on a real bundle (Codex's first 90-day backfill
        produced calibration_error 0.98 on the very_high bucket).

        This test pins that the generator passes a DIFFERENT dist_atr
        for two pools at different distances. If dist_atr ever goes
        back to a constant for all pools, this test fails.
        """
        report = _two_asset_report(p_touch=0.40)
        # Replace the unified_proximity with a stub that captures the
        # dist_atr it was called with.
        captured: List[float] = []

        class _DistanceCaptureStub:
            val_auc = 0.85
            def predict_one(self, pool, dist_atr, side, state,
                            quality_pred, atr_val=1.0):
                captured.append(float(dist_atr))
                return 0.40

        report.unified_proximity = {12: _DistanceCaptureStub()}
        _ = generate_brief(report, trading_date_ist="2026-06-03")
        # We have 2 active pools at different prices (90-91 below close
        # ~100, 520-521 above close ~500). Each gets one predict_one
        # call. The two distances MUST differ.
        self.assertGreaterEqual(len(captured), 2,
            f"generator should make >=2 predict_one calls; got {len(captured)}")
        self.assertNotEqual(captured[0], captured[1],
            f"dist_atr must differ across pools at different distances; "
            f"got {captured} — this indicates the dist_atr placeholder "
            f"bug regressed")
        # Both distances must be > 0 (placeholder was 1.0; real values
        # for these fixtures are roughly 10-20 ATR or more).
        for d in captured:
            self.assertGreater(d, 0.0)

    def test_yesterday_audit_renders_pending_when_no_history(self) -> None:
        from liqpool.products.outcome_log import OutcomeLogWriter
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            writer = OutcomeLogWriter(root=tmp)
            report = _two_asset_report()
            brief = generate_brief(
                report,
                trading_date_ist="2026-06-03",
                outcome_log_writer=writer,
            )
            # No history written for 2026-06-02 -> pending.
            self.assertEqual(brief.yesterday_audit["_status"], "pending")


# ---------------------------------------------------------------------------
# Stream G — retrospective-audit flag end-to-end
# ---------------------------------------------------------------------------

class RetrospectiveAuditFlagTests(unittest.TestCase):
    """End-to-end pin for Stream G:

      * ``retrospective=True`` propagates from ``generate_brief`` through
        ``_log_brief_predictions`` onto every PredictionRecord written.
      * Default ``retrospective=False`` keeps the live path untouched.
      * The Yesterday Audit payload exposes ``retrospective_share`` and
        ``is_retrospective_calibration`` when the previous-day predictions
        were retrospective.
      * The renderer surfaces a customer-readable disclosure line when
        ``is_retrospective_calibration`` is True.
    """

    def _writer_with_retrospective_yesterday(self, tmp: str
                                              ) -> "OutcomeLogWriter":
        """Write a yesterday-shaped batch of retrospective predictions
        + resolutions so today's brief can read them via ``read_joined``.
        """
        from liqpool.products.outcome_log import (
            OutcomeLogWriter, ResolutionRecord)
        writer = OutcomeLogWriter(root=tmp)
        # Generate yesterday's brief in retrospective mode.
        report = _two_asset_report(p_touch=0.40)
        generate_brief(
            report,
            trading_date_ist="2026-06-02",
            outcome_log_writer=writer,
            retrospective=True,
        )
        # Resolve each prediction (outcome doesn't matter for this test;
        # we just need resolutions so the calibration table is non-empty).
        preds = writer.read_predictions("2026-06-02")
        for _, row in preds.iterrows():
            writer.write_resolution(ResolutionRecord(
                prediction_id=row["prediction_id"],
                resolved_at_utc="2026-06-02T10:30:00Z",
                resolution_method="level_touched",
                outcome_boolean=True, outcome_continuous=None,
                resolution_details={"touched_at_bar_offset": 12},
                had_data_gap=False, resolution_quality="clean",
                trading_date_ist="2026-06-02",
            ))
        writer.commit()
        return writer

    def test_retrospective_flag_propagates_to_prediction_records(self) -> None:
        from liqpool.products.outcome_log import OutcomeLogWriter
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            writer = OutcomeLogWriter(root=tmp)
            report = _two_asset_report(p_touch=0.40)
            generate_brief(
                report,
                trading_date_ist="2026-06-03",
                outcome_log_writer=writer,
                retrospective=True,
            )
            preds = writer.read_predictions("2026-06-03")
            self.assertGreater(len(preds), 0)
            self.assertTrue(bool(preds["is_retrospective"].all()),
                "every logged prediction must carry is_retrospective=True "
                "when generate_brief is called with retrospective=True")

    def test_live_default_keeps_is_retrospective_false(self) -> None:
        from liqpool.products.outcome_log import OutcomeLogWriter
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            writer = OutcomeLogWriter(root=tmp)
            report = _two_asset_report(p_touch=0.40)
            generate_brief(
                report,
                trading_date_ist="2026-06-03",
                outcome_log_writer=writer,
            )
            preds = writer.read_predictions("2026-06-03")
            self.assertGreater(len(preds), 0)
            self.assertFalse(bool(preds["is_retrospective"].any()),
                "default live brief generation must not tag predictions "
                "as retrospective")

    def test_yesterday_audit_exposes_retrospective_share(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            writer = self._writer_with_retrospective_yesterday(tmp)
            # Today's brief reads yesterday's joined log.
            report = _two_asset_report(p_touch=0.40)
            brief = generate_brief(
                report,
                trading_date_ist="2026-06-03",
                outcome_log_writer=writer,
            )
            audit = brief.yesterday_audit
            # Populated, not pending.
            self.assertNotEqual(audit.get("_status"), "pending",
                f"audit should be populated when yesterday's log exists; "
                f"got {audit}")
            self.assertAlmostEqual(audit["retrospective_share"], 1.0,
                                    places=4)
            self.assertTrue(audit["is_retrospective_calibration"])
            self.assertGreater(audit["predictions_made"], 0)

    def test_renderer_surfaces_retrospective_disclosure(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            writer = self._writer_with_retrospective_yesterday(tmp)
            report = _two_asset_report(p_touch=0.40)
            brief = generate_brief(
                report,
                trading_date_ist="2026-06-03",
                outcome_log_writer=writer,
            )
            text = render_email(brief)
            self.assertIn("YESTERDAY AUDIT", text)
            self.assertIn("retrospective replay", text,
                "renderer must disclose retrospective calibration to "
                "the customer when >50% of yesterday's predictions were "
                "backfilled")

    def test_renderer_omits_disclosure_for_live_audit(self) -> None:
        """If yesterday's predictions were live (is_retrospective=False),
        the renderer must NOT print the retrospective disclosure line."""
        from liqpool.products.outcome_log import (
            OutcomeLogWriter, ResolutionRecord)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            writer = OutcomeLogWriter(root=tmp)
            report = _two_asset_report(p_touch=0.40)
            # Live yesterday — retrospective stays False.
            generate_brief(
                report,
                trading_date_ist="2026-06-02",
                outcome_log_writer=writer,
            )
            preds = writer.read_predictions("2026-06-02")
            for _, row in preds.iterrows():
                writer.write_resolution(ResolutionRecord(
                    prediction_id=row["prediction_id"],
                    resolved_at_utc="2026-06-02T10:30:00Z",
                    resolution_method="level_touched",
                    outcome_boolean=True, outcome_continuous=None,
                    resolution_details={"touched_at_bar_offset": 12},
                    had_data_gap=False, resolution_quality="clean",
                    trading_date_ist="2026-06-02",
                ))
            writer.commit()
            brief = generate_brief(
                report,
                trading_date_ist="2026-06-03",
                outcome_log_writer=writer,
            )
            self.assertFalse(
                brief.yesterday_audit["is_retrospective_calibration"])
            text = render_email(brief)
            self.assertNotIn("retrospective replay", text)


if __name__ == "__main__":
    unittest.main()
