import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from liqpool.research import (
    ConstituentState,
    IndexStateBuildConfig,
    build_index_manipulation_dataset,
    build_manipulation_state_frame,
    classify_index_manipulation,
    rank_random_hypotheses,
    summarize_state_frame,
)
from liqpool.warehouse import WarehouseReader


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


class IndexStateBuilderTests(unittest.TestCase):
    def test_build_index_manipulation_dataset_from_synthetic_warehouse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            symbols = ("A", "B", "C", "D", "E")
            for i, symbol in enumerate(symbols):
                _write_resampled_symbol(root, symbol, base=100 + i * 10)
            _write_spot_index(root, base=1000)

            reader = WarehouseReader(root=str(root))
            constituents, states, summary = build_index_manipulation_dataset(
                reader,
                IndexStateBuildConfig(
                    symbols=symbols,
                    index="NIFTY50",
                    tf="5m",
                    historical_window_bars=3,
                    min_history_bars=1,
                    forward_bars=(1, 2),
                    min_constituents=5,
                ),
            )

            self.assertFalse(constituents.empty)
            self.assertFalse(states.empty)
            self.assertIn("today_avwap_dist_atr", constituents.columns)
            self.assertIn("prev_session_avwap_dist_atr", constituents.columns)
            self.assertIn("state_label", states.columns)
            self.assertIn("forward_index_return_1b_pct", states.columns)
            self.assertEqual(summary["symbols_missing"], [])
            self.assertEqual(summary["rows"], len(states))


def _synthetic_bars(symbol: str, base: float) -> pd.DataFrame:
    ts = list(pd.date_range("2026-06-10 09:15", periods=8, freq="5min", tz="Asia/Kolkata"))
    ts += list(pd.date_range("2026-06-11 09:15", periods=8, freq="5min", tz="Asia/Kolkata"))
    drift = np.linspace(0, 3.0, len(ts))
    close = base + drift
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": 1000,
            "symbol": symbol,
        }
    )


def _write_resampled_symbol(root: Path, symbol: str, base: float) -> None:
    path = root / "resampled" / "5m"
    path.mkdir(parents=True, exist_ok=True)
    _synthetic_bars(symbol, base).to_parquet(path / f"{symbol}_5m.parquet")


def _write_spot_index(root: Path, base: float) -> None:
    path = root / "spot" / "NIFTY50"
    path.mkdir(parents=True, exist_ok=True)
    frame = _synthetic_bars("NIFTY50", base)
    frame["open_interest"] = 0
    frame.to_parquet(path / "5min.parquet")


if __name__ == "__main__":
    unittest.main()
