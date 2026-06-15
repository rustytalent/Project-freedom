import json
import unittest

import numpy as np
import pandas as pd

from liqpool.research import (
    ConstituentState,
    build_manipulation_state_frame,
    classify_index_manipulation,
    rank_random_hypotheses,
    summarize_state_frame,
)


class ManipulationAtlasTests(unittest.TestCase):
    def test_index_masking_when_green_move_is_concentrated_and_breadth_weak(self):
        rows = [
            ConstituentState("HDFCBANK", 0.18, return_pct=1.4, today_avwap_dist_atr=0.8),
            ConstituentState("ICICIBANK", 0.12, return_pct=1.1, today_avwap_dist_atr=0.6),
            ConstituentState("RELIANCE", 0.10, return_pct=1.0, today_avwap_dist_atr=0.5),
            ConstituentState("INFY", 0.08, return_pct=-0.4, today_avwap_dist_atr=-0.3),
            ConstituentState("TCS", 0.07, return_pct=-0.3, today_avwap_dist_atr=-0.2),
        ]
        state = classify_index_manipulation(
            rows, index_return_pct=0.55, breadth_positive_frac=0.35
        )
        self.assertEqual(state.label, "index_masking")
        self.assertEqual(state.action_bias, "avoid_chasing_calls")
        self.assertIn("top3_heavyweight_concentration", state.reason_codes)

    def test_broad_sponsorship_when_breadth_and_avwap_agree(self):
        rows = [
            {"symbol": "A", "weight": 0.2, "return_pct": 0.8, "today_avwap_dist_atr": 0.6, "prev_session_avwap_dist_atr": 0.3},
            {"symbol": "B", "weight": 0.2, "return_pct": 0.6, "today_avwap_dist_atr": 0.4, "prev_session_avwap_dist_atr": 0.2},
            {"symbol": "C", "weight": 0.2, "return_pct": 0.5, "today_avwap_dist_atr": 0.5, "prev_session_avwap_dist_atr": 0.1},
            {"symbol": "D", "weight": 0.2, "return_pct": 0.4, "today_avwap_dist_atr": 0.3, "prev_session_avwap_dist_atr": 0.1},
            {"symbol": "E", "weight": 0.2, "return_pct": -0.1, "today_avwap_dist_atr": 0.1, "prev_session_avwap_dist_atr": 0.1},
        ]
        state = classify_index_manipulation(
            rows, index_return_pct=0.45, breadth_positive_frac=0.8
        )
        self.assertEqual(state.label, "broad_sponsorship")
        self.assertEqual(state.action_bias, "directional_setups_allowed")


class HypothesisMinerTests(unittest.TestCase):
    def test_random_miner_finds_simple_embedded_rule(self):
        rng = np.random.default_rng(11)
        n = 400
        frame = pd.DataFrame({
            "x": rng.normal(size=n),
            "noise": rng.normal(size=n),
        })
        frame["forward_R"] = np.where(frame["x"] > 0.7, 0.8, -0.05)
        ranked = rank_random_hypotheses(
            frame,
            forward_return_col="forward_R",
            n=300,
            seed=3,
            feature_columns=["x", "noise"],
            min_trades=20,
        )
        self.assertFalse(ranked.empty)
        self.assertGreater(ranked.iloc[0]["mean_R"], 0.1)
        spec = json.loads(ranked.iloc[0]["spec_json"])
        self.assertIn(spec["side"], {"long", "short"})
        self.assertTrue(spec["conditions"])


class StateDatasetTests(unittest.TestCase):
    def test_build_manipulation_state_frame_groups_constituents(self):
        frame = pd.DataFrame(
            [
                {"ts": "2026-06-15 09:20", "symbol": "A", "weight": 0.2, "return_pct": 1.0, "today_avwap_dist_atr": 0.7, "prev_session_avwap_dist_atr": 0.4, "index_return_pct": 0.5, "breadth_positive_frac": 0.8},
                {"ts": "2026-06-15 09:20", "symbol": "B", "weight": 0.2, "return_pct": 0.8, "today_avwap_dist_atr": 0.6, "prev_session_avwap_dist_atr": 0.3, "index_return_pct": 0.5, "breadth_positive_frac": 0.8},
                {"ts": "2026-06-15 09:20", "symbol": "C", "weight": 0.2, "return_pct": 0.6, "today_avwap_dist_atr": 0.5, "prev_session_avwap_dist_atr": 0.2, "index_return_pct": 0.5, "breadth_positive_frac": 0.8},
                {"ts": "2026-06-15 09:20", "symbol": "D", "weight": 0.2, "return_pct": 0.4, "today_avwap_dist_atr": 0.4, "prev_session_avwap_dist_atr": 0.1, "index_return_pct": 0.5, "breadth_positive_frac": 0.8},
                {"ts": "2026-06-15 09:20", "symbol": "E", "weight": 0.2, "return_pct": -0.1, "today_avwap_dist_atr": 0.1, "prev_session_avwap_dist_atr": 0.1, "index_return_pct": 0.5, "breadth_positive_frac": 0.8},
            ]
        )
        states = build_manipulation_state_frame(frame)
        self.assertEqual(len(states), 1)
        self.assertEqual(states.iloc[0]["state_label"], "broad_sponsorship")
        self.assertEqual(states.iloc[0]["n_constituents"], 5)

        summary = summarize_state_frame(states)
        self.assertEqual(summary["rows"], 1)
        self.assertEqual(summary["labels"]["broad_sponsorship"], 1)


if __name__ == "__main__":
    unittest.main()
