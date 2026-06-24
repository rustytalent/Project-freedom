from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from auction_state_graph import AuctionGraphConfig, run_auction_state_graph
from auction_state_graph.core.candle_builder import build_candles
from auction_state_graph.core.state_classifier import annotate_candles
from auction_state_graph.ingest.replay_adapter import demo_ticks, load_replay_ticks


def test_ohlc_replay_expands_to_path_ticks(tmp_path: Path) -> None:
    replay = tmp_path / "ohlc.csv"
    pd.DataFrame(
        [
            {"ts": "2026-06-18 09:15:00", "open": 100, "high": 105, "low": 98, "close": 104, "volume": 1000},
            {"ts": "2026-06-18 09:16:00", "open": 104, "high": 106, "low": 101, "close": 102, "volume": 1200},
        ]
    ).to_csv(replay, index=False)

    ticks = load_replay_ticks(replay, symbol="TEST")
    assert len(ticks) > 2
    assert ticks[0].source_kind == "ohlc_path_replay"
    assert min(t.price for t in ticks) == 98
    assert max(t.price for t in ticks) == 106


def test_builder_keeps_bounded_history() -> None:
    config = AuctionGraphConfig(symbol="DEMO", timeframe="1m", max_candles=12, bin_count=8)
    candles = build_candles(demo_ticks(periods=1800), config)
    assert len(candles) <= 12
    assert all(len(c.price_bins) == 8 for c in candles)


def test_features_labels_and_story_are_present() -> None:
    config = AuctionGraphConfig(symbol="DEMO", timeframe="1m", max_candles=60, bin_count=10)
    candles = annotate_candles(build_candles(demo_ticks(periods=900), config))
    assert candles
    last = candles[-1]
    assert last.label
    assert last.story is not None
    assert last.next_states
    assert last.path_efficiency >= 0
    story_text = " ".join(last.story.bullets + last.story.warnings).lower()
    assert "place order" not in story_text
    assert "buy " not in story_text
    assert "sell " not in story_text


def test_runner_writes_dashboard_and_summary(tmp_path: Path) -> None:
    config = AuctionGraphConfig(symbol="DEMO", timeframe="1m", max_candles=40, bin_count=8)
    summary = run_auction_state_graph(config=config, out_dir=tmp_path, demo=True)
    html = Path(summary["artifacts"]["dashboard_html"])
    summary_json = Path(summary["artifacts"]["summary_json"])
    assert html.exists()
    assert "No model bundle, no Sentinel, no broker, no orders" in html.read_text()
    payload = json.loads(summary_json.read_text())
    assert payload["bounded_history"] is True
    assert payload["no_execution"] is True
