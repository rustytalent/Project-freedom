"""Build research-ready market-state datasets.

The Manipulation Atlas is deliberately small and deterministic. This
module turns bar/constituent rows into a chronological state frame so
the atlas can be joined to outcomes, mined for rules, or fed into
Sentinel.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd

from .manipulation_atlas import ConstituentState, classify_index_manipulation


@dataclass(frozen=True)
class StateDatasetConfig:
    """Column mapping for a top-constituent manipulation-state frame."""

    group_cols: tuple[str, ...] = ("ts",)
    symbol_col: str = "symbol"
    weight_col: str = "weight"
    return_col: str = "return_pct"
    today_avwap_col: str = "today_avwap_dist_atr"
    prev_avwap_col: str = "prev_session_avwap_dist_atr"
    historical_position_col: str = "historical_position_pct"
    gap_col: str = "gap_pct"
    index_return_col: str | None = "index_return_pct"
    breadth_col: str | None = "breadth_positive_frac"
    min_constituents: int = 5


def _row_to_constituent(row: Mapping[str, object], cfg: StateDatasetConfig) -> ConstituentState:
    today_avwap_dist_atr = float(row.get(cfg.today_avwap_col, 0.0) or 0.0)
    prev_session_avwap_dist_atr = float(row.get(cfg.prev_avwap_col, 0.0) or 0.0)
    return ConstituentState(
        symbol=str(row.get(cfg.symbol_col, "")),
        weight=float(row.get(cfg.weight_col, 0.0) or 0.0),
        return_pct=float(row.get(cfg.return_col, 0.0) or 0.0),
        today_avwap_dist_atr=today_avwap_dist_atr,
        prev_session_avwap_dist_atr=prev_session_avwap_dist_atr,
        historical_position_pct=float(row.get(cfg.historical_position_col, 0.5) or 0.5),
        gap_pct=float(row.get(cfg.gap_col, 0.0) or 0.0),
        above_today_avwap=today_avwap_dist_atr > 0,
        above_prev_avwap=prev_session_avwap_dist_atr > 0,
    )


def build_manipulation_state_frame(
    frame: pd.DataFrame,
    cfg: StateDatasetConfig | None = None,
) -> pd.DataFrame:
    """Classify each timestamp/session group into a manipulation state.

    Input rows should be one constituent per group. For example, if
    ``group_cols=("ts",)`` and there are ten symbols per timestamp, the
    output contains one state row per timestamp.
    """

    cfg = cfg or StateDatasetConfig()
    if frame.empty:
        return pd.DataFrame()

    missing = [
        col
        for col in (
            *cfg.group_cols,
            cfg.symbol_col,
            cfg.weight_col,
            cfg.return_col,
            cfg.today_avwap_col,
        )
        if col not in frame.columns
    ]
    if missing:
        raise ValueError(f"missing required state columns: {missing}")

    out: list[dict[str, object]] = []
    group_key: str | list[str]
    group_key = list(cfg.group_cols) if len(cfg.group_cols) > 1 else cfg.group_cols[0]

    for key, group in frame.groupby(group_key, sort=True, dropna=False):
        if len(group) < cfg.min_constituents:
            continue

        rows = group.to_dict("records")
        constituents = [_row_to_constituent(row, cfg) for row in rows]

        index_return = 0.0
        if cfg.index_return_col and cfg.index_return_col in group.columns:
            index_return = float(pd.to_numeric(group[cfg.index_return_col], errors="coerce").dropna().mean() or 0.0)

        breadth = None
        if cfg.breadth_col and cfg.breadth_col in group.columns:
            breadth_values = pd.to_numeric(group[cfg.breadth_col], errors="coerce").dropna()
            if len(breadth_values):
                breadth = float(breadth_values.mean())

        state = classify_index_manipulation(
            constituents,
            index_return_pct=index_return,
            breadth_positive_frac=breadth,
        )

        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(cfg.group_cols, key)}
        row.update(
            {
                "state_label": state.label,
                "state_confidence": state.confidence,
                "action_bias": state.action_bias,
                "reason_codes": json.dumps(state.reason_codes, sort_keys=True),
                "state_metrics": json.dumps(state.metrics, sort_keys=True),
                "n_constituents": int(len(group)),
                "index_return_pct": index_return,
                "breadth_positive_frac": breadth,
            }
        )
        out.append(row)

    return pd.DataFrame(out)


def summarize_state_frame(frame: pd.DataFrame) -> dict[str, object]:
    """Return a compact JSON-friendly summary for reports."""

    if frame.empty:
        return {
            "rows": 0,
            "labels": {},
            "action_bias": {},
            "avg_confidence": None,
        }
    return {
        "rows": int(len(frame)),
        "labels": frame["state_label"].value_counts(dropna=False).to_dict(),
        "action_bias": frame["action_bias"].value_counts(dropna=False).to_dict(),
        "avg_confidence": float(frame["state_confidence"].mean()),
    }


__all__ = [
    "StateDatasetConfig",
    "build_manipulation_state_frame",
    "summarize_state_frame",
]
