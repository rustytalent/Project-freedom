"""Tests for the premium-vs-spot divergence detector.

These pin the founder's live-observed edge (2026-06-16) and the one
refinement that makes it robust — IV common-mode cancellation.

The crown-jewel test is ``test_iv_common_mode_does_not_trap``: when a
general IV spike lifts BOTH call and put premiums together (vega, not
direction), the detector must NOT fire a trap. That is the difference
between reading order flow and being fooled by a vol expansion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from liqpool.research.premium_divergence import (
    BEAR_TRAP,
    BULL_TRAP,
    CONFIRMED_UP,
    NEUTRAL,
    DivergenceSignal,
    PremiumDivergenceConfig,
    compute_divergence_frame,
    divergence_exit_reason,
    divergence_signals,
    summarize_divergence,
)


# ─────────────────────────────────────────────────────────────────
# Synthetic builders
# ─────────────────────────────────────────────────────────────────

_WIN_A, _WIN_B = 200, 235     # injected event window [200, 235)


def _base_frame(n: int = 400, seed: int = 1) -> pd.DataFrame:
    """Honest baseline: call/put track spot by delta ±0.5 + noise + theta."""
    rng = np.random.default_rng(seed)
    spot = 23000 + np.cumsum(rng.normal(0, 5, n))
    d_spot = np.diff(spot, prepend=spot[0])
    call = 100 + np.cumsum(0.5 * d_spot + rng.normal(0, 1.0, n)) - np.arange(n) * 0.02
    put = 100 + np.cumsum(-0.5 * d_spot + rng.normal(0, 1.0, n)) - np.arange(n) * 0.02
    return pd.DataFrame({
        "ts": pd.date_range("2026-06-16 09:15", periods=n, freq="1min"),
        "spot": spot,
        "call_premium": np.clip(call, 1, None),
        "put_premium": np.clip(put, 1, None),
    })


def _inject_bull_trap(df: pd.DataFrame) -> pd.DataFrame:
    """Spot ramps UP over the window, but PUTS are held up (accumulation):
    puts rise instead of decaying — a bull trap."""
    df = df.copy()
    a, b = _WIN_A, _WIN_B
    L = b - a
    ramp = np.linspace(0, 120, L)
    df.loc[a:b - 1, "spot"] = df.loc[a:b - 1, "spot"].values + ramp
    df.loc[a:b - 1, "put_premium"] = df.loc[a:b - 1, "put_premium"].values + np.linspace(0, 60, L)
    df.loc[a:b - 1, "call_premium"] = df.loc[a:b - 1, "call_premium"].values + 0.5 * ramp
    df["call_premium"] = np.clip(df["call_premium"], 1, None)
    df["put_premium"] = np.clip(df["put_premium"], 1, None)
    return df


def _inject_bear_trap(df: pd.DataFrame) -> pd.DataFrame:
    """Spot ramps DOWN over the window, but CALLS are held up — a bear trap."""
    df = df.copy()
    a, b = _WIN_A, _WIN_B
    L = b - a
    ramp = np.linspace(0, 120, L)
    df.loc[a:b - 1, "spot"] = df.loc[a:b - 1, "spot"].values - ramp
    df.loc[a:b - 1, "call_premium"] = df.loc[a:b - 1, "call_premium"].values + np.linspace(0, 60, L)
    df.loc[a:b - 1, "put_premium"] = df.loc[a:b - 1, "put_premium"].values + 0.5 * ramp
    df["call_premium"] = np.clip(df["call_premium"], 1, None)
    df["put_premium"] = np.clip(df["put_premium"], 1, None)
    return df


def _inject_iv_spike(df: pd.DataFrame) -> pd.DataFrame:
    """Spot ramps UP, premiums track by delta, but a COMMON IV bump lifts
    BOTH call and put by the same amount — pure vega, no directional intent.
    The detector must NOT call this a trap."""
    df = df.copy()
    a, b = _WIN_A, _WIN_B
    L = b - a
    ramp = np.linspace(0, 120, L)
    iv_bump = np.linspace(0, 40, L)
    df.loc[a:b - 1, "spot"] = df.loc[a:b - 1, "spot"].values + ramp
    df.loc[a:b - 1, "call_premium"] = df.loc[a:b - 1, "call_premium"].values + 0.5 * ramp + iv_bump
    df.loc[a:b - 1, "put_premium"] = df.loc[a:b - 1, "put_premium"].values - 0.5 * ramp + iv_bump
    df["call_premium"] = np.clip(df["call_premium"], 1, None)
    df["put_premium"] = np.clip(df["put_premium"], 1, None)
    return df


def _in_window(sigs, lo=195, hi=250):
    return [s for s in sigs if lo <= s.index <= hi]


# ─────────────────────────────────────────────────────────────────
# Trap detection
# ─────────────────────────────────────────────────────────────────

def test_bull_trap_detected_when_puts_hold_up_in_uptrend():
    df = _inject_bull_trap(_base_frame(seed=1))
    sigs = divergence_signals(df)
    in_win = [s for s in _in_window(sigs) if s.verdict == BULL_TRAP]
    assert in_win, "should detect the planted bull trap"
    s = in_win[0]
    assert s.leg == "put"
    assert s.direction == -1            # expect spot down
    assert s.net_intent < 0             # bearish positioning (z-score)
    assert s.entry_zone_low <= s.entry_premium <= s.entry_zone_high


def test_bear_trap_detected_when_calls_hold_up_in_downtrend():
    df = _inject_bear_trap(_base_frame(seed=2))
    sigs = divergence_signals(df)
    in_win = [s for s in _in_window(sigs) if s.verdict == BEAR_TRAP]
    assert in_win, "should detect the planted bear trap"
    s = in_win[0]
    assert s.leg == "call"
    assert s.direction == +1            # expect spot up
    assert s.net_intent > 0             # bullish positioning


def test_iv_common_mode_does_not_trap():
    """CROWN JEWEL: a pure IV spike (both legs lifted equally) must not be
    mistaken for directional positioning. No trap inside the window."""
    df = _inject_iv_spike(_base_frame(seed=1))
    sigs = divergence_signals(df)
    traps_in_window = _in_window(sigs)
    assert len(traps_in_window) == 0, (
        f"IV common-mode must cancel — got {len(traps_in_window)} false traps"
    )


def test_clean_random_has_low_signal_rate():
    """On honest data with no injected event, the signal rate must be low —
    the detector is selective, not a noise generator."""
    df = _base_frame(seed=3)
    summ = summarize_divergence(df)
    assert summ["signal_rate"] < 0.10


# ─────────────────────────────────────────────────────────────────
# Frame / schema
# ─────────────────────────────────────────────────────────────────

def test_compute_frame_adds_expected_columns():
    df = _base_frame(seed=4)
    enriched = compute_divergence_frame(df)
    for col in ("call_resid", "put_resid", "call_pressure", "put_pressure",
                "net_intent", "net_intent_z", "spot_move_norm", "verdict", "signal"):
        assert col in enriched.columns
    # Signal is consistent with verdict.
    assert (enriched.loc[enriched["verdict"] == BULL_TRAP, "signal"] == -1).all()
    assert (enriched.loc[enriched["verdict"] == BEAR_TRAP, "signal"] == +1).all()
    assert (enriched.loc[enriched["verdict"] == NEUTRAL, "signal"] == 0).all()


def test_compute_frame_raises_on_missing_columns():
    df = pd.DataFrame({"spot": [1, 2, 3], "call_premium": [1, 2, 3]})
    with pytest.raises(ValueError, match="put_premium"):
        compute_divergence_frame(df)


def test_min_premium_filter_suppresses_deep_otm_noise():
    """If premiums are below min_premium, no trap fires regardless of the
    residual — deep-OTM tick noise must not generate signals."""
    df = _inject_bull_trap(_base_frame(seed=1))
    df["put_premium"] = 0.1     # below default min_premium=0.5
    df["call_premium"] = 0.1
    sigs = divergence_signals(df)
    assert sigs == []


def test_activity_guard_suppresses_pinned_leg():
    """A leg pinned at its floor (deep-OTM ₹1 lottery ticket) has near-zero
    change-vol — its 'holding up' is pinning, not accumulation. Even with a
    spot move and apparent intent, no trap should fire while a leg is dead."""
    df = _inject_bear_trap(_base_frame(seed=2))
    # Pin the call leg flat at 1.0 across the whole session — it's "dead".
    df["call_premium"] = 1.0
    sigs = divergence_signals(df)
    # A bear trap reads the call leg; a pinned call must not produce signals.
    assert all(s.leg != "call" for s in sigs), (
        "pinned/dead call leg must not generate bear-trap signals"
    )


def test_cooldown_collapses_a_cluster_to_one_signal():
    """A single trap event spans many consecutive bars. With a cooldown,
    one event yields far fewer (one-per-cluster) signals."""
    df = _inject_bull_trap(_base_frame(seed=1))
    raw = divergence_signals(df, cooldown_bars=0)
    cooled = divergence_signals(df, cooldown_bars=10)
    assert len(cooled) < len(raw), "cooldown should reduce repeat signals"
    # No two same-verdict signals are within the cooldown window.
    bull = [s.index for s in cooled if s.verdict == BULL_TRAP]
    for a, b in zip(bull, bull[1:]):
        assert b - a > 10


def test_delta_override_columns_are_used():
    """When call_delta/put_delta columns are supplied, they override the
    model-free estimate (and the frame still computes)."""
    df = _base_frame(seed=5)
    df["call_delta"] = 0.5
    df["put_delta"] = -0.5
    enriched = compute_divergence_frame(df)
    # The supplied deltas are clipped into bounds and used verbatim.
    assert np.allclose(enriched["call_delta_eff"], 0.5)
    assert np.allclose(enriched["put_delta_eff"], -0.5)


# ─────────────────────────────────────────────────────────────────
# Causality
# ─────────────────────────────────────────────────────────────────

def test_signal_is_causal_no_lookahead():
    """The verdict at bar T must depend only on data up to T. Truncating
    the frame after T must leave T's verdict unchanged."""
    df = _inject_bull_trap(_base_frame(seed=1))
    full = compute_divergence_frame(df)
    # Pick a bar with a definite verdict inside the window.
    trap_idx = full.index[(full["verdict"] == BULL_TRAP) & (full.index >= _WIN_A)]
    assert len(trap_idx) > 0
    k = int(trap_idx[0])
    truncated = compute_divergence_frame(df.iloc[:k + 1])
    assert truncated["verdict"].iloc[k] == full["verdict"].iloc[k]
    assert truncated["signal"].iloc[k] == full["signal"].iloc[k]


# ─────────────────────────────────────────────────────────────────
# Exit logic
# ─────────────────────────────────────────────────────────────────

def test_exit_reason_fires_when_intent_flips_against_put():
    """Holding a put (dir=-1); if the last bar's net_intent_z is bullish,
    the put-accumulation thesis is gone → exit."""
    df = _base_frame(seed=6)
    enriched = compute_divergence_frame(df)
    # Force a strongly bullish last-bar intent.
    enriched.loc[enriched.index[-1], "net_intent_z"] = 5.0
    enriched.loc[enriched.index[-1], "spot_move_norm"] = 0.0
    reason = divergence_exit_reason(enriched, position_dir=-1)
    assert reason is not None
    assert "bullish" in reason


def test_exit_reason_none_when_thesis_intact():
    df = _base_frame(seed=7)
    enriched = compute_divergence_frame(df)
    # Put thesis intact: intent still bearish, spot hasn't moved down yet.
    enriched.loc[enriched.index[-1], "net_intent_z"] = -3.0
    enriched.loc[enriched.index[-1], "spot_move_norm"] = 0.0
    assert divergence_exit_reason(enriched, position_dir=-1) is None


def test_exit_reason_take_profit_after_reversal_realized():
    df = _base_frame(seed=8)
    enriched = compute_divergence_frame(df)
    # Put position, spot has now moved down past the threshold → realized.
    enriched.loc[enriched.index[-1], "net_intent_z"] = -3.0
    enriched.loc[enriched.index[-1], "spot_move_norm"] = -2.0
    reason = divergence_exit_reason(enriched, position_dir=-1)
    assert reason is not None
    assert "reversal realized" in reason


# ─────────────────────────────────────────────────────────────────
# Determinism + summary
# ─────────────────────────────────────────────────────────────────

def test_determinism_same_frame_same_signals():
    df = _inject_bull_trap(_base_frame(seed=1))
    s1 = divergence_signals(df)
    s2 = divergence_signals(df)
    assert [s.to_dict() for s in s1] == [s.to_dict() for s in s2]


def test_summarize_schema_and_counts_add_up():
    df = _inject_bull_trap(_base_frame(seed=1))
    summ = summarize_divergence(df)
    expected_keys = {"n_bars", "n_signals", "signal_rate", "bull_traps",
                     "bear_traps", "confirmed_up", "confirmed_down", "neutral"}
    assert expected_keys.issubset(summ.keys())
    # Verdict counts partition the bars.
    total = (summ["bull_traps"] + summ["bear_traps"] + summ["confirmed_up"]
             + summ["confirmed_down"] + summ["neutral"])
    assert total == summ["n_bars"]
    # Signals are exactly the trap bars.
    assert summ["n_signals"] == summ["bull_traps"] + summ["bear_traps"]


def test_signal_to_dict_is_json_serializable():
    df = _inject_bull_trap(_base_frame(seed=1))
    sigs = divergence_signals(df)
    assert sigs
    import json
    json.dumps([s.to_dict() for s in sigs], default=str)
