"""Regression test for the Stage-A reaction-distribution audit script.

The original bug: ``_summary_row`` was computing AUC by pairing the labelled
subset's ``y`` with the FULL predictions array, which mismatched lengths
whenever any OOS rows had NaN labels (most runs).

These tests pin the contract: ``_predict_target`` must return predictions on
the full population alongside ``(p_labelled, y_labelled)`` pairs of equal
length, and ``_summary_row`` must compute AUC only on that pair (never
crashing on length mismatch).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.audit_reaction_distribution import (
    _predict_target, _summary_row,
)


class _StubReactionModel:
    """Minimal stand-in for ReactionBinaryModel — only the audit's API surface."""

    def __init__(self, label_col: str = "strict_respect_label"):
        self.label_col = label_col

    def predict_frame(self, events: pd.DataFrame) -> np.ndarray:
        # Deterministic predictions that vary per row but stay in [0.02, 0.98]
        # like the real model's isotonic-clipped output.
        return np.clip(np.arange(1, len(events) + 1) / (len(events) + 1), 0.02, 0.98)


def _events_with_some_nan_labels(label_col: str) -> pd.DataFrame:
    return pd.DataFrame({
        label_col: [1, 0, 1, np.nan, np.nan, 0, 1, np.nan, 0, 1],
        "score": np.arange(10, dtype=float),
    })


def test_predict_target_lengths_match():
    label_col = "strict_respect_label"
    events = _events_with_some_nan_labels(label_col)
    model = _StubReactionModel(label_col=label_col)
    p_all, p_lab, y_lab = _predict_target(model, events)
    assert len(p_all) == len(events)
    assert p_lab is not None and y_lab is not None
    assert len(p_lab) == len(y_lab) == int(events[label_col].notna().sum())


def test_predict_target_no_labels_returns_none_pair():
    events = pd.DataFrame({"unrelated": [1.0, 2.0, 3.0]})
    model = _StubReactionModel(label_col="strict_respect_label")
    p_all, p_lab, y_lab = _predict_target(model, events)
    assert len(p_all) == 3
    assert p_lab is None and y_lab is None


def test_predict_target_all_nan_labels_returns_none_pair():
    label_col = "strict_respect_label"
    events = pd.DataFrame({label_col: [np.nan, np.nan, np.nan]})
    model = _StubReactionModel(label_col=label_col)
    p_all, p_lab, y_lab = _predict_target(model, events)
    assert len(p_all) == 3
    assert p_lab is None and y_lab is None


def test_summary_row_computes_auc_without_length_mismatch():
    """The original crash: roc_auc_score(y, p) where y is shorter than p."""
    label_col = "strict_respect_label"
    events = _events_with_some_nan_labels(label_col)
    model = _StubReactionModel(label_col=label_col)
    p_all, p_lab, y_lab = _predict_target(model, events)
    row = _summary_row("strict_reaction", model, p_all, p_lab, y_lab)
    assert row["n_predictions"] == len(events)
    assert row["n_labelled"] == int(events[label_col].notna().sum())
    # Both classes present, so AUC is computed.
    assert row["oos_auc"] is not None
    assert 0.0 <= row["oos_auc"] <= 1.0


def test_summary_row_skips_auc_on_single_class():
    label_col = "strict_respect_label"
    events = pd.DataFrame({label_col: [1, 1, 1, np.nan, 1]})
    model = _StubReactionModel(label_col=label_col)
    p_all, p_lab, y_lab = _predict_target(model, events)
    row = _summary_row("strict_reaction", model, p_all, p_lab, y_lab)
    assert row["oos_auc"] is None
    assert row["n_predictions"] == 5
    assert row["n_labelled"] == 4
