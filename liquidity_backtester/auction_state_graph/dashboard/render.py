"""Plotly dashboard for standalone auction-state replay."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from ..core.models import CandleFeatures


def render_dashboard(candles: list[CandleFeatures], out_html: str | Path, title: str) -> Path:
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    if not candles:
        out_html.write_text(_page(title, "<p>No candles were produced.</p>", {}), encoding="utf-8")
        return out_html

    df = pd.DataFrame([_flat_candle(c) for c in candles])
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.34, 0.16, 0.16, 0.16, 0.18],
        subplot_titles=(
            "Candles with Auction State",
            "Path Efficiency and Churn",
            "Urgency U+/U- and Net",
            "CE/PE Premium Response",
            "Latest Candle Price-Level Map",
        ),
    )
    _add_candles(fig, df)
    _add_path_panel(fig, df)
    _add_urgency_panel(fig, df)
    _add_premium_panel(fig, df)
    _add_latest_heatmap(fig, candles[-1])
    fig.update_layout(
        template="plotly_dark",
        height=1280,
        margin=dict(l=60, r=30, t=70, b=40),
        showlegend=True,
        hovermode="x unified",
        title=title,
    )
    fig.update_xaxes(rangeslider_visible=False)
    chart_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False, config={"responsive": True})
    out_html.write_text(_page(title, chart_html, _summary(candles)), encoding="utf-8")
    return out_html


def _flat_candle(c: CandleFeatures) -> dict:
    return {
        "start_ts": c.start_ts,
        "open": c.open,
        "high": c.high,
        "low": c.low,
        "close": c.close,
        "label": c.label,
        "range": c.range,
        "body_ratio": c.body_ratio,
        "path_efficiency": c.path_efficiency,
        "churn_ratio": min(c.churn_ratio, 8.0),
        "urgency_plus": c.urgency_plus,
        "urgency_minus": c.urgency_minus,
        "urgency_net": c.urgency_net,
        "ce_efficiency": c.ce_efficiency,
        "pe_efficiency": c.pe_efficiency,
        "premium_bias": c.premium_bias,
        "premium_compression": c.premium_compression,
        "acceptance_proxy": c.acceptance_proxy,
        "story": c.story.title if c.story else c.label,
    }


def _add_candles(fig: go.Figure, df: pd.DataFrame) -> None:
    fig.add_trace(
        go.Candlestick(
            x=df["start_ts"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
        ),
        row=1,
        col=1,
    )
    label_colors = {
        "Clean Bullish Impulse": "#79d28d",
        "Clean Bearish Impulse": "#d86c61",
        "Upside Sweep Rejection": "#e0a64a",
        "Downside Sweep Reclaim": "#6fb4ff",
        "Two-Sided Fight": "#b38cff",
        "Premium Compression": "#c0c0c0",
        "No-Trade Churn": "#777777",
    }
    for label, rows in df.groupby("label"):
        fig.add_trace(
            go.Scatter(
                x=rows["start_ts"],
                y=rows["close"],
                mode="markers",
                marker=dict(size=8, color=label_colors.get(label, "#f2c46d")),
                name=label,
                text=rows["story"],
                hovertemplate="%{text}<br>close=%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )


def _add_path_panel(fig: go.Figure, df: pd.DataFrame) -> None:
    fig.add_trace(go.Scatter(x=df["start_ts"], y=df["path_efficiency"], name="path efficiency", line=dict(color="#79d28d")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["start_ts"], y=df["acceptance_proxy"], name="acceptance proxy", line=dict(color="#6fb4ff")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["start_ts"], y=df["churn_ratio"] / 8.0, name="churn ratio scaled", line=dict(color="#e0a64a")), row=2, col=1)


def _add_urgency_panel(fig: go.Figure, df: pd.DataFrame) -> None:
    fig.add_trace(go.Scatter(x=df["start_ts"], y=df["urgency_plus"], name="U+", line=dict(color="#79d28d")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["start_ts"], y=df["urgency_minus"], name="U-", line=dict(color="#d86c61")), row=3, col=1)
    fig.add_trace(go.Bar(x=df["start_ts"], y=df["urgency_net"], name="U net", marker_color="#f2c46d", opacity=0.45), row=3, col=1)


def _add_premium_panel(fig: go.Figure, df: pd.DataFrame) -> None:
    fig.add_trace(go.Scatter(x=df["start_ts"], y=df["ce_efficiency"], name="CE efficiency", line=dict(color="#79d28d")), row=4, col=1)
    fig.add_trace(go.Scatter(x=df["start_ts"], y=df["pe_efficiency"], name="PE efficiency", line=dict(color="#d86c61")), row=4, col=1)
    fig.add_trace(go.Scatter(x=df["start_ts"], y=df["premium_bias"], name="premium bias", line=dict(color="#f2c46d")), row=4, col=1)
    fig.add_trace(go.Scatter(x=df["start_ts"], y=df["premium_compression"], name="premium compression", line=dict(color="#b38cff")), row=4, col=1)


def _add_latest_heatmap(fig: go.Figure, candle: CandleFeatures) -> None:
    bins = list(reversed(candle.price_bins))
    y = [f"{row.low:.2f}-{row.high:.2f}" for row in bins]
    z = [[row.net_tick_delta] for row in bins]
    hover = [[
        f"ticks={row.tick_count}<br>time={row.time_spent_seconds:.1f}s<br>"
        f"volume~{row.approx_volume:.0f}<br>urgency={row.urgency_net:+.2f}<br>"
        f"premium_bias={row.premium_bias if row.premium_bias is not None else 'n/a'}"
    ] for row in bins]
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=["net tick delta"],
            y=y,
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            colorscale=[[0.0, "#6e302d"], [0.5, "#111820"], [1.0, "#2f6b45"]],
            name="price bin map",
        ),
        row=5,
        col=1,
    )


def _summary(candles: list[CandleFeatures]) -> dict:
    last = candles[-1]
    return {
        "symbol": last.symbol,
        "timeframe": last.timeframe,
        "candles_rendered": len(candles),
        "last_label": last.label,
        "last_story": last.story.to_dict() if last.story else None,
        "next_states": [row.to_dict() for row in last.next_states],
    }


def _page(title: str, chart_html: str, summary: dict) -> str:
    safe_summary = json.dumps(summary, indent=2, default=str)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ margin: 0; background: #07090c; color: #e6e2d7; font: 14px Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }}
    header {{ padding: 22px 28px; border-bottom: 1px solid #1f2733; background: #0b0f15; }}
    h1 {{ margin: 0; font-size: 22px; letter-spacing: 0; }}
    .sub {{ color: #9aa2af; margin-top: 6px; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 18px; }}
    .panel {{ border: 1px solid #202a36; border-radius: 8px; background: #0d1219; padding: 14px; margin-bottom: 18px; }}
    pre {{ white-space: pre-wrap; color: #c9d1d9; }}
    .guard {{ color: #f2c46d; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="sub">Standalone replay. No model bundle, no Sentinel, no broker, no orders.</div>
  </header>
  <main>
    <section class="panel">
      <strong class="guard">Current candle story and next-state possibilities</strong>
      <pre>{safe_summary}</pre>
    </section>
    <section class="panel">{chart_html}</section>
  </main>
</body>
</html>
"""
