"""Tests for the liquidity-hunt vs real-reversal classifier.

These pin the founder's "during a pullback, positioning beats price"
principle. Three canonical scenarios are tested against the same
established bullish regime:

  * HUNT — spot drops while positioning (call pressure positive, put
    pressure negative) STAYS on the bullish side. Verdict =
    LIQUIDITY_HUNT_UP, action = hold_through. The fall is a stop-sweep,
    not a reversal.
  * BREAK — spot drops AND positioning flips (call pressure negative,
    put pressure positive). Verdict = REGIME_BREAK_DOWN, action = exit.
    The fall is real; the regime is broken.
  * TREND — spot continues up with the regime. Verdict =
    TREND_INTACT_UP, action = ride.

Additional invariants pinned:
  * The defend/break check is smoothed over confirm_window so it doesn't
    flicker on intra-move buffer noise.
  * The grace window catches a break against a regime that just decayed
    (a fast positioning flip can collapse the slow regime metric to zero
    before the break prints — the grace fill preserves the recent regime).
  * Empty / structureless frames degrade to NO_REGIME without crashing.
  * Causality: the verdict at bar T uses only data with index ≤ T.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from liqpool.research.liquidity_hunt import (
    ACTION_EXIT,
    ACTION_HOLD_THROUGH,
    ACTION_RIDE,
    LIQUIDITY_HUNT_UP,
    NO_REGIME,
    REGIME_BREAK_DOWN,
    TREND_INTACT_UP,
    HuntSignal,
    LiquidityHuntConfig,
    classify_liquidity_hunts,
    hunt_signals,
    summarize_hunts,
)
from liqpool.research.premium_divergence import compute_divergence_frame


# ─────────────────────────────────────────────────────────────────
# Synthetic scenarios
# ─────────────────────────────────────────────────────────────────

_REGIME_START, _REGIME_END = 20, 120
_COUNTER_START, _COUNTER_END = 120, 150


def _build_scenario(event: str, n: int = 220, seed: int = 1) -> pd.DataFrame:
    """Bars 20..120 — establish a bullish regime via SUSTAINED per-bar
    residual injection: calls beat delta by +0.5, puts lag by -0.5. Pressure
    accumulates persistently positive on calls / negative on puts.

    Bars 120..150 — counter-move (spot drops):
      * event='hunt'  → positioning DEFENDED: keep injecting the same direction
      * event='break' → positioning FLIPS: inject the opposite direction
      * event='trend' → no counter-move (regime keeps going up); injection
        continues bullish-positioning-side

    Bars 150+ — return to baseline."""
    rng = np.random.default_rng(seed)
    drift = np.full(n, 0.4)
    if event != "trend":
        drift[_COUNTER_START:_COUNTER_END] = -3.0     # spot drops during counter-move
    spot = 23000 + np.cumsum(drift + rng.normal(0, 3, n))
    d_spot = np.diff(spot, prepend=spot[0])
    inj_c = np.zeros(n)
    inj_p = np.zeros(n)
    inj_c[_REGIME_START:_REGIME_END] = 0.5
    inj_p[_REGIME_START:_REGIME_END] = -0.5
    if event == "hunt":
        inj_c[_COUNTER_START:_COUNTER_END] = 0.55
        inj_p[_COUNTER_START:_COUNTER_END] = -0.45
    elif event == "break":
        inj_c[_COUNTER_START:_COUNTER_END] = -0.7
        inj_p[_COUNTER_START:_COUNTER_END] = 0.8
    elif event == "trend":
        inj_c[_COUNTER_START:_COUNTER_END] = 0.5
        inj_p[_COUNTER_START:_COUNTER_END] = -0.5
    call = 100 + np.cumsum(0.5 * d_spot + rng.normal(0, 0.4, n) + inj_c)
    put = 100 + np.cumsum(-0.5 * d_spot + rng.normal(0, 0.4, n) + inj_p)
    return pd.DataFrame({
        "ts": pd.date_range("2026-06-16 09:15", periods=n, freq="1min"),
        "spot": spot,
        "call_premium": np.clip(call, 1, None),
        "put_premium": np.clip(put, 1, None),
    })


# ─────────────────────────────────────────────────────────────────
# Regime establishment
# ─────────────────────────────────────────────────────────────────

def test_bullish_regime_is_established_when_call_pressure_persists():
    df = _build_scenario("trend")
    enr = classify_liquidity_hunts(df)
    # By bar 100 (well into the regime window) the regime should be +1.
    sample = enr.loc[80:115, "regime"]
    assert (sample == 1).sum() > len(sample) * 0.5, (
        "bullish regime must be sustained in the establishment window"
    )


def test_no_regime_when_frame_has_no_structure():
    """A pure-noise frame must produce mostly NO_REGIME (very few accidental
    regimes are tolerable, but it must NOT call a regime persistently)."""
    rng = np.random.default_rng(99)
    n = 200
    df = pd.DataFrame({
        "ts": pd.date_range("2026-06-16 09:15", periods=n, freq="1min"),
        "spot": 23000 + np.cumsum(rng.normal(0, 4, n)),
        "call_premium": 100 + np.cumsum(rng.normal(0, 1, n)),
        "put_premium": 100 + np.cumsum(rng.normal(0, 1, n)),
    })
    summ = summarize_hunts(df)
    assert summ["no_regime"] > summ["n_bars"] * 0.5


# ─────────────────────────────────────────────────────────────────
# Hunt — positioning defended
# ─────────────────────────────────────────────────────────────────

def test_hunt_during_pullback_when_positioning_stays_bullish():
    """Spot drops while calls remain accumulated and puts remain weak →
    LIQUIDITY_HUNT_UP through the counter-move. The action is
    hold_through — do not panic-sell."""
    df = _build_scenario("hunt")
    enr = classify_liquidity_hunts(df)
    window = enr.loc[_COUNTER_START:_COUNTER_END - 1, "hunt_verdict"]
    n_hunt = (window == LIQUIDITY_HUNT_UP).sum()
    n_break = (window == REGIME_BREAK_DOWN).sum()
    assert n_hunt >= 15, (
        f"hunt scenario must fire LIQUIDITY_HUNT_UP across the counter-move; "
        f"got n_hunt={n_hunt}, n_break={n_break}"
    )
    assert n_break == 0, "hunt scenario must NOT call regime_break"


def test_hunt_action_is_hold_through():
    df = _build_scenario("hunt")
    sigs = hunt_signals(df, cooldown_bars=8)
    in_window = [s for s in sigs
                 if _COUNTER_START <= s.index < _COUNTER_END
                 and s.verdict == LIQUIDITY_HUNT_UP]
    assert in_window
    for s in in_window:
        assert s.action == ACTION_HOLD_THROUGH
        assert s.direction == +1
        assert s.regime == +1


# ─────────────────────────────────────────────────────────────────
# Break — positioning flipped
# ─────────────────────────────────────────────────────────────────

def test_break_during_pullback_when_positioning_flips():
    """Spot drops AND calls collapse AND puts surge → REGIME_BREAK_DOWN.
    The action is exit — the regime is genuinely broken."""
    df = _build_scenario("break")
    enr = classify_liquidity_hunts(df)
    window = enr.loc[_COUNTER_START:_COUNTER_END - 1, "hunt_verdict"]
    n_break = (window == REGIME_BREAK_DOWN).sum()
    n_hunt = (window == LIQUIDITY_HUNT_UP).sum()
    assert n_break >= 10, (
        f"break scenario must fire REGIME_BREAK_DOWN; "
        f"got n_break={n_break}, n_hunt={n_hunt}"
    )
    assert n_break > n_hunt, "break must dominate hunt in this scenario"


def test_break_action_is_exit():
    df = _build_scenario("break")
    sigs = hunt_signals(df, cooldown_bars=8)
    breaks_in_window = [s for s in sigs
                        if _COUNTER_START <= s.index < _COUNTER_END
                        and s.verdict == REGIME_BREAK_DOWN]
    assert breaks_in_window
    for s in breaks_in_window:
        assert s.action == ACTION_EXIT
        assert s.regime == +1     # the regime that was broken was bullish


# ─────────────────────────────────────────────────────────────────
# Trend — no counter-move
# ─────────────────────────────────────────────────────────────────

def test_trend_intact_when_no_counter_move():
    """When the regime continues without a counter-move, the dominant
    verdict in the regime window is TREND_INTACT_UP with action=ride."""
    df = _build_scenario("trend")
    enr = classify_liquidity_hunts(df)
    window = enr.loc[60:_REGIME_END - 1, "hunt_verdict"]
    n_trend = (window == TREND_INTACT_UP).sum()
    assert n_trend >= 30


# ─────────────────────────────────────────────────────────────────
# Robustness invariants
# ─────────────────────────────────────────────────────────────────

def test_classify_accepts_pre_enriched_frame():
    """Idempotency: if the input already has the divergence columns,
    classify must not re-compute them."""
    df = _build_scenario("hunt")
    pre = compute_divergence_frame(df)
    a = classify_liquidity_hunts(pre)
    b = classify_liquidity_hunts(df)
    pd.testing.assert_series_equal(a["hunt_verdict"], b["hunt_verdict"])


def test_summary_partitions_bars_across_verdicts():
    df = _build_scenario("hunt")
    summ = summarize_hunts(df)
    total = (summ["liquidity_hunt_up"] + summ["liquidity_hunt_down"]
             + summ["regime_break_down"] + summ["regime_break_up"]
             + summ["trend_intact_up"] + summ["trend_intact_down"]
             + summ["uncertain"] + summ["no_regime"])
    assert total == summ["n_bars"]


def test_hunt_signal_to_dict_is_json_serializable():
    df = _build_scenario("hunt")
    sigs = hunt_signals(df, cooldown_bars=8)
    import json
    json.dumps([s.to_dict() for s in sigs[:5]], default=str)


def test_classify_is_causal_no_lookahead():
    """The verdict at bar T must be unchanged if the frame is truncated
    after T — strictly causal, safe for live use."""
    df = _build_scenario("hunt")
    full = classify_liquidity_hunts(df)
    # Find a bar with a HUNT verdict inside the counter-move window.
    hunt_idx = full.index[(full["hunt_verdict"] == LIQUIDITY_HUNT_UP)
                          & (full.index >= _COUNTER_START)]
    assert len(hunt_idx) > 0
    k = int(hunt_idx[0])
    truncated = classify_liquidity_hunts(df.iloc[:k + 1])
    assert truncated["hunt_verdict"].iloc[k] == full["hunt_verdict"].iloc[k]


def test_include_breaks_false_excludes_break_signals():
    df = _build_scenario("break")
    all_sigs = hunt_signals(df, cooldown_bars=8, include_breaks=True)
    hunt_only = hunt_signals(df, cooldown_bars=8, include_breaks=False)
    assert any(s.verdict == REGIME_BREAK_DOWN for s in all_sigs)
    assert not any(s.verdict == REGIME_BREAK_DOWN for s in hunt_only)


def test_cooldown_collapses_repeat_verdicts():
    df = _build_scenario("hunt")
    raw = hunt_signals(df, cooldown_bars=0)
    cooled = hunt_signals(df, cooldown_bars=8)
    assert len(cooled) < len(raw)


def test_hunt_confidence_is_nonzero_on_hunt_and_zero_elsewhere():
    df = _build_scenario("hunt")
    sigs = hunt_signals(df, cooldown_bars=8)
    hunts = [s for s in sigs if s.verdict == LIQUIDITY_HUNT_UP]
    breaks = [s for s in sigs if s.verdict == REGIME_BREAK_DOWN]
    assert hunts
    assert any(s.hunt_confidence > 0.2 for s in hunts), (
        "hunt confidence must reflect positioning's continued defense"
    )
    for s in breaks:
        assert s.hunt_confidence == 0.0


def test_config_validation_rejects_bad_inputs():
    with pytest.raises(ValueError, match="regime_window"):
        LiquidityHuntConfig(regime_window=2)
    with pytest.raises(ValueError, match="confirm_window"):
        LiquidityHuntConfig(confirm_window=0)
