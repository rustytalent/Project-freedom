"""Standalone runner for Auction State Graph replay."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import AuctionGraphConfig
from .core.candle_builder import build_candles
from .core.state_classifier import annotate_candles
from .dashboard.render import render_dashboard
from .ingest.replay_adapter import demo_ticks, load_replay_ticks


def run_auction_state_graph(
    *,
    config: AuctionGraphConfig,
    out_dir: str | Path,
    replay_path: str | Path | None = None,
    demo: bool = False,
) -> dict:
    config.validate()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if demo:
        ticks = demo_ticks(symbol=config.symbol)
    elif replay_path is not None:
        ticks = load_replay_ticks(replay_path, symbol=config.symbol, max_rows=config.max_replay_rows)
    else:
        raise ValueError("pass demo=True or replay_path")

    candles = annotate_candles(build_candles(ticks, config))
    rows = [c.to_dict() for c in candles]
    summary = _summary(candles, source="demo" if demo else str(replay_path))

    candles_csv = out_dir / "auction_candles.csv"
    summary_json = out_dir / "auction_summary.json"
    raw_json = out_dir / "auction_candles.json"
    dashboard_html = out_dir / "auction_state_graph.html"

    pd.DataFrame([_flat_for_csv(c) for c in candles]).to_csv(candles_csv, index=False)
    raw_json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    render_dashboard(candles, dashboard_html, config.dashboard_title)

    summary["artifacts"] = {
        "dashboard_html": str(dashboard_html),
        "summary_json": str(summary_json),
        "candles_csv": str(candles_csv),
        "candles_json": str(raw_json),
    }
    return summary


def _flat_for_csv(c) -> dict:
    return {
        "symbol": c.symbol,
        "timeframe": c.timeframe,
        "start_ts": c.start_ts,
        "open": c.open,
        "high": c.high,
        "low": c.low,
        "close": c.close,
        "range": c.range,
        "body": c.body,
        "body_ratio": c.body_ratio,
        "close_location": c.close_location,
        "path_efficiency": c.path_efficiency,
        "churn_ratio": c.churn_ratio,
        "acceptance_proxy": c.acceptance_proxy,
        "urgency_plus": c.urgency_plus,
        "urgency_minus": c.urgency_minus,
        "urgency_net": c.urgency_net,
        "premium_bias": c.premium_bias,
        "premium_compression": c.premium_compression,
        "label": c.label,
        "label_reasons": " | ".join(c.label_reasons),
    }


def _summary(candles, source: str) -> dict:
    labels: dict[str, int] = {}
    for c in candles:
        labels[c.label] = labels.get(c.label, 0) + 1
    last = candles[-1] if candles else None
    return {
        "source": source,
        "candles": len(candles),
        "bounded_history": True,
        "labels": labels,
        "last_candle": last.to_dict() if last else None,
        "no_execution": True,
        "no_broker_connection": True,
    }
