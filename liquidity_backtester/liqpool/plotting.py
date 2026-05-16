"""Plotly visualisation: 5m candles + pool zones + outcome markers."""
from __future__ import annotations
from typing import List, Iterable
import pandas as pd

from .pools import Pool
from .tester import PoolResult


def _color_for(p: Pool, outcome: str | None) -> tuple[str, str]:
    if outcome == "respected":
        line = "rgba(46,160,67,0.9)"; fill = "rgba(46,160,67,0.18)"
    elif outcome == "broken":
        line = "rgba(220,80,60,0.9)"; fill = "rgba(220,80,60,0.15)"
    elif outcome == "untouched":
        line = "rgba(140,140,160,0.8)"; fill = "rgba(140,140,160,0.10)"
    else:
        if p.side == "high":
            line = "rgba(230,140,40,0.9)"; fill = "rgba(230,140,40,0.15)"
        else:
            line = "rgba(70,130,220,0.9)"; fill = "rgba(70,130,220,0.15)"
    return line, fill


def plot_chart(df_base: pd.DataFrame, pools: List[Pool],
               results: Iterable[PoolResult] | None = None,
               title: str = "Liquidity Pools", out_path: str | None = None,
               top_n: int = 25):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    res_by_idx = {}
    if results is not None:
        res_by_idx = {r.pool_idx: r for r in results}

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.8, 0.2], vertical_spacing=0.02)

    fig.add_trace(go.Candlestick(
        x=df_base.index,
        open=df_base["open"], high=df_base["high"],
        low=df_base["low"], close=df_base["close"],
        name="5m", showlegend=False,
    ), row=1, col=1)

    if "volume" in df_base.columns:
        fig.add_trace(go.Bar(x=df_base.index, y=df_base["volume"],
                             marker_color="rgba(120,120,140,0.5)", showlegend=False),
                      row=2, col=1)

    x_end = df_base.index[-1]
    pools_sorted = sorted(pools, key=lambda p: p.score, reverse=True)[:top_n]
    for i, p in enumerate(pools_sorted):
        outcome = res_by_idx.get(pools.index(p)).outcome if pools.index(p) in res_by_idx else None
        line, fill = _color_for(p, outcome)
        x0 = p.formed_at
        x1 = x_end
        fig.add_shape(type="rect", xref="x", yref="y",
                      x0=x0, x1=x1, y0=p.price_low, y1=p.price_high,
                      line=dict(color=line, width=1), fillcolor=fill, layer="below",
                      row=1, col=1)
        sources = ",".join(sorted({c.source.split('@')[0] for c in p.contributors}))
        label = f"{p.side.upper()} | {p.mid:.2f} | s={p.score:.2f} | tf={'+'.join(p.tfs)} | {sources}"
        fig.add_annotation(x=x1, y=p.price_high if p.side == "high" else p.price_low,
                           text=label, showarrow=False, xanchor="right",
                           font=dict(size=9, color=line), bgcolor="rgba(0,0,0,0.25)",
                           row=1, col=1)

        r = res_by_idx.get(pools.index(p))
        if r and r.touched_at is not None:
            fig.add_trace(go.Scatter(x=[r.touched_at], y=[(p.price_low + p.price_high) / 2],
                                     mode="markers", marker=dict(color=line, size=8, symbol="x"),
                                     showlegend=False, hovertext=f"touched ({r.outcome})"),
                          row=1, col=1)
        if r and r.broken_at is not None:
            fig.add_trace(go.Scatter(x=[r.broken_at],
                                     y=[p.price_high if p.side == "high" else p.price_low],
                                     mode="markers", marker=dict(color="rgba(220,80,60,1)",
                                                                 size=10, symbol="triangle-down"
                                                                 if p.side == "high" else "triangle-up"),
                                     showlegend=False, hovertext="broken"),
                          row=1, col=1)

    fig.update_layout(title=title, template="plotly_dark", height=900, xaxis_rangeslider_visible=False,
                      margin=dict(l=40, r=40, t=60, b=40))
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(rangeslider_visible=False, row=2, col=1)

    if out_path:
        fig.write_html(out_path, include_plotlyjs="cdn")
    return fig
