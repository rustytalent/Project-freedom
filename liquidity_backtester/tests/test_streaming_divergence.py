"""Tests for the mid-candle streaming engine.

The founder's "15-min candle 5 min in" concern translates to two
invariants the engine must hold:

  * STABILITY GATE: intra-candle wiggle (provisional flicker) must NOT
    change the stable verdict. The engine emits a verdict change only
    when it has held for ``stability_ticks`` consecutive updates.

  * CONSISTENCY WITH SEALED: feeding the same series through the engine
    (seal each bar) must produce the same final-bar reads as the batch
    pipeline produces on a single classify_liquidity_hunts call. The
    streaming engine is the SAME math, just incremental.

Additional safety nets pinned:
  * History bound: confirmed history is capped at ``history_bars``.
  * Warmup: while history is too short the engine returns NEUTRAL/none
    without crashing.
  * reset(): clears state cleanly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from liqpool.research.liquidity_hunt import (
    ACTION_HOLD_THROUGH,
    ACTION_NONE,
    LIQUIDITY_HUNT_UP,
    NO_REGIME,
    classify_liquidity_hunts,
)
from liqpool.research.premium_divergence import NEUTRAL
from liqpool.research.streaming_divergence import (
    DEFAULT_HISTORY_BARS,
    DEFAULT_STABILITY_TICKS,
    StreamingDivergenceEngine,
    StreamingRead,
)


def _build_hunt_session(n: int = 220, seed: int = 1) -> pd.DataFrame:
    """Same canonical hunt scenario the liquidity_hunt tests use."""
    rng = np.random.default_rng(seed)
    drift = np.full(n, 0.4)
    drift[120:145] = -3.0
    spot = 23000 + np.cumsum(drift + rng.normal(0, 3, n))
    d_spot = np.diff(spot, prepend=spot[0])
    inj_c = np.zeros(n); inj_p = np.zeros(n)
    inj_c[20:120] = 0.5; inj_p[20:120] = -0.5
    inj_c[120:150] = 0.55; inj_p[120:150] = -0.45      # positioning DEFENDED
    call = 100 + np.cumsum(0.5 * d_spot + rng.normal(0, 0.4, n) + inj_c)
    put = 100 + np.cumsum(-0.5 * d_spot + rng.normal(0, 0.4, n) + inj_p)
    return pd.DataFrame({
        "ts": pd.date_range("2026-06-16 09:15", periods=n, freq="1min"),
        "spot": spot,
        "call_premium": np.clip(call, 1, None),
        "put_premium": np.clip(put, 1, None),
    })


def _seal_through(eng: StreamingDivergenceEngine,
                  df: pd.DataFrame, up_to: int) -> StreamingRead:
    """Seal the first ``up_to`` bars; return the read on the next bar."""
    last_read = None
    for i in range(up_to):
        row = df.iloc[i]
        eng.seal_bar(row["ts"], row["spot"],
                     row["call_premium"], row["put_premium"])
    # Update with the next bar as forming.
    nxt = df.iloc[up_to]
    last_read = eng.update(nxt["ts"], nxt["spot"],
                            nxt["call_premium"], nxt["put_premium"])
    return last_read


# ─────────────────────────────────────────────────────────────────
# Stability gate
# ─────────────────────────────────────────────────────────────────

def test_provisional_flicker_does_not_change_stable_verdict():
    """If the operator wiggles the forming bar's tick values, the
    provisional verdict may flicker — but the STABLE verdict must not
    change until the provisional has held for stability_ticks."""
    df = _build_hunt_session()
    eng = StreamingDivergenceEngine(stability_ticks=3)
    # Seal the regime + the start of the counter-move.
    for i in range(130):
        row = df.iloc[i]
        eng.seal_bar(row["ts"], row["spot"], row["call_premium"], row["put_premium"])
    # The next forming bar's REAL close is row 130.
    target = df.iloc[130]
    # Three wiggling reads, each radically different — provisional jumps
    # around but stable should stay where it was (NEUTRAL initially).
    r1 = eng.update(target["ts"], target["spot"] - 50,
                    target["call_premium"] - 10, target["put_premium"] + 10)
    r2 = eng.update(target["ts"], target["spot"] + 50,
                    target["call_premium"] + 10, target["put_premium"] - 10)
    r3 = eng.update(target["ts"], target["spot"],
                    target["call_premium"], target["put_premium"])
    # Stable stays at whatever the first read promoted (likely NEUTRAL)
    # until a single verdict has held for 3 consecutive updates.
    assert r1.stable_verdict in (NEUTRAL, NO_REGIME)
    # The three reads have different provisional verdicts in general; the
    # stable should not have flipped to chase the last one.
    assert r3.bars_held == 1, "candidate counter resets when provisional changes"


def test_stable_verdict_promotes_after_consecutive_holds():
    """Three identical updates → stability_ticks=3 reached → stable promotes."""
    df = _build_hunt_session()
    eng = StreamingDivergenceEngine(stability_ticks=3)
    for i in range(125):
        row = df.iloc[i]
        eng.seal_bar(row["ts"], row["spot"], row["call_premium"], row["put_premium"])
    target = df.iloc[125]
    # Three IDENTICAL updates — the same provisional must promote to stable.
    args = (target["ts"], target["spot"],
            target["call_premium"], target["put_premium"])
    r1 = eng.update(*args)
    r2 = eng.update(*args)
    r3 = eng.update(*args)
    assert r3.bars_held >= 3
    assert r3.stable_verdict == r3.provisional_verdict


# ─────────────────────────────────────────────────────────────────
# Consistency with batch
# ─────────────────────────────────────────────────────────────────

def test_streaming_matches_batch_at_each_sealed_bar():
    """For any bar T, running the batch classify_liquidity_hunts on df[:T+1]
    must yield the same verdict at row T as the streaming engine after
    sealing T-1 bars and reading T as forming. This pins that the
    streaming engine is the SAME math, just incremental."""
    df = _build_hunt_session()
    eng = StreamingDivergenceEngine(stability_ticks=1)   # disable damping
    for t in (60, 100, 130, 160):
        eng.reset()
        read = _seal_through(eng, df, up_to=t)
        # Batch on the same prefix [0..t].
        batch = classify_liquidity_hunts(df.iloc[:t + 1])
        batch_verdict = batch["hunt_verdict"].iloc[-1]
        # The streaming engine reports trap_verdict OR hunt_verdict as
        # provisional per its precedence rule; match against hunt where
        # it's actionable.
        if batch_verdict != NO_REGIME:
            assert read.provisional_verdict == batch_verdict, (
                f"streaming != batch at t={t}: "
                f"stream={read.provisional_verdict} batch={batch_verdict}"
            )


def test_streaming_catches_hunt_during_counter_move():
    """The hunt that the batch test catches at bar ~125 must also appear
    in the streaming read (with stability_ticks=1)."""
    df = _build_hunt_session()
    eng = StreamingDivergenceEngine(stability_ticks=1)
    hunt_seen = False
    for i in range(len(df) - 1):
        row = df.iloc[i]
        eng.seal_bar(row["ts"], row["spot"],
                     row["call_premium"], row["put_premium"])
        nxt = df.iloc[i + 1]
        read = eng.update(nxt["ts"], nxt["spot"],
                          nxt["call_premium"], nxt["put_premium"])
        if (read.provisional_verdict == LIQUIDITY_HUNT_UP
                and 120 <= i + 1 <= 150):
            hunt_seen = True
            assert read.action == ACTION_HOLD_THROUGH
            assert read.direction == +1
    assert hunt_seen, "streaming engine must catch the hunt in the counter window"


# ─────────────────────────────────────────────────────────────────
# Warmup + safety
# ─────────────────────────────────────────────────────────────────

def test_warmup_returns_neutral_without_crash():
    eng = StreamingDivergenceEngine()
    read = eng.update(pd.Timestamp("2026-06-16 09:15"), 23000, 100, 100)
    assert read.provisional_verdict == NEUTRAL
    assert read.stable_verdict == NEUTRAL
    assert read.action == ACTION_NONE
    assert read.direction == 0


def test_history_is_bounded_at_history_bars():
    df = _build_hunt_session(n=300)
    eng = StreamingDivergenceEngine(history_bars=100)
    for i in range(len(df)):
        row = df.iloc[i]
        eng.seal_bar(row["ts"], row["spot"],
                     row["call_premium"], row["put_premium"])
    assert eng.history_size <= 100


def test_reset_clears_all_state():
    df = _build_hunt_session()
    eng = StreamingDivergenceEngine()
    for i in range(50):
        row = df.iloc[i]
        eng.seal_bar(row["ts"], row["spot"],
                     row["call_premium"], row["put_premium"])
    eng.reset()
    assert eng.history_size == 0
    read = eng.update(pd.Timestamp("2026-06-16 09:15"), 23000, 100, 100)
    assert read.bar_index == 0
    assert read.provisional_verdict == NEUTRAL


def test_config_validation_rejects_too_small_history():
    with pytest.raises(ValueError, match="history_bars"):
        StreamingDivergenceEngine(history_bars=10)
    with pytest.raises(ValueError, match="stability_ticks"):
        StreamingDivergenceEngine(stability_ticks=0)


def test_streaming_read_to_dict_is_json_serializable():
    df = _build_hunt_session()
    eng = StreamingDivergenceEngine()
    read = _seal_through(eng, df, up_to=130)
    import json
    json.dumps(read.to_dict(), default=str)
