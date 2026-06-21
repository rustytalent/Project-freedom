"""Tests for the advanced (opt-in) Phase-6 thesis builder features.

Founder audit (2026-06-22): the original thesis-memory used a fixed
decay rate regardless of tape speed, and surfaced no uncertainty on
its scores (downstream couldn't distinguish 72 ± 8 from 72 ± 1).

These tests cover the additive upgrades:
  * adaptive decay (tape-speed-aware)
  * uncertainty bands on every score (wider when inputs are flip-flopping)

Default behavior tests live in the original test_belief_phase6.py and
are unaffected by these changes (regression suite already green).
"""
from __future__ import annotations

import pytest

from liqpool.research.belief.iv_state import (
    IV_COMMON_SHOCK,
    IV_DIRECTIONAL_BEAR,
    IV_DIRECTIONAL_BULL,
    IV_NEUTRAL,
    IVState,
)
from liqpool.research.belief.thesis_memory import (
    ThesisMemory,
    ThesisMemoryConfig,
)


def _iv(state: str, confidence: float = 0.7,
         direction: int = 0) -> IVState:
    return IVState(
        state=state, direction=direction, confidence=confidence,
        avg_friendliness=0.92, clean_mark_fraction=0.98,
        ce_defended_fraction=0.0, pe_defended_fraction=0.0,
        ce_rejected_fraction=0.0, pe_rejected_fraction=0.0,
        note="test",
    )


# ── Default behavior: snapshot shape preserved ───────────────────


def test_default_snapshot_has_zero_ci_and_legacy_decay():
    mem = ThesisMemory()
    snap = mem.update(iv_state=_iv(IV_DIRECTIONAL_BULL,
                                       confidence=0.85, direction=1))
    # Legacy fields present with same semantics.
    assert hasattr(snap, "bull_thesis_score")
    assert hasattr(snap, "composite_state")
    # New CI fields default to 0 (legacy "no uncertainty surfaced" world).
    assert snap.bull_thesis_ci == 0.0
    assert snap.bear_thesis_ci == 0.0
    assert snap.vol_expansion_ci == 0.0
    # Effective decay must equal the static cfg decay.
    assert snap.effective_decay == pytest.approx(0.94)


def test_advanced_default_off_in_config():
    cfg = ThesisMemoryConfig()
    assert cfg.enable_adaptive_decay is False
    assert cfg.enable_uncertainty is False


# ── Adaptive decay ───────────────────────────────────────────────


def test_adaptive_decay_faster_in_fast_tape():
    cfg = ThesisMemoryConfig(enable_adaptive_decay=True,
                               decay_min=0.80, decay_max=0.99,
                               adaptive_decay_sensitivity=0.10)
    mem = ThesisMemory(cfg=cfg)
    # Build up a positive bull score with no tape speed signal.
    for _ in range(8):
        mem.update(iv_state=_iv(IV_DIRECTIONAL_BULL,
                                  confidence=0.85, direction=1))
    base_score = mem._state.bull
    # Fast tape: decay should drop closer to decay_min.
    snap_fast = mem.update(iv_state=None, tape_speed=2.0)
    assert snap_fast.effective_decay < 0.94
    # Slow tape: decay should rise toward decay_max.
    mem2 = ThesisMemory(cfg=cfg)
    for _ in range(8):
        mem2.update(iv_state=_iv(IV_DIRECTIONAL_BULL,
                                   confidence=0.85, direction=1))
    snap_slow = mem2.update(iv_state=None, tape_speed=0.3)
    assert snap_slow.effective_decay > 0.94


def test_adaptive_decay_clipped_to_bounds():
    cfg = ThesisMemoryConfig(enable_adaptive_decay=True,
                               decay_min=0.90, decay_max=0.98,
                               adaptive_decay_sensitivity=1.0)  # extreme
    mem = ThesisMemory(cfg=cfg)
    # Catastrophically high tape speed must be clipped to decay_min.
    snap = mem.update(iv_state=None, tape_speed=20.0)
    assert snap.effective_decay >= 0.90
    # Catastrophically low → clipped to decay_max.
    snap2 = mem.update(iv_state=None, tape_speed=-5.0)
    assert snap2.effective_decay <= 0.98


def test_adaptive_decay_default_when_no_tape_speed():
    cfg = ThesisMemoryConfig(enable_adaptive_decay=True,
                               adaptive_decay_sensitivity=0.10)
    mem = ThesisMemory(cfg=cfg)
    snap = mem.update(iv_state=None)   # tape_speed omitted
    # tape_speed=None → adjust=0 → effective_decay == decay_per_bar
    assert snap.effective_decay == pytest.approx(0.94)


def test_adaptive_decay_rejects_bad_config():
    with pytest.raises(ValueError):
        ThesisMemoryConfig(enable_adaptive_decay=True,
                            decay_min=0.99, decay_max=0.94)


# ── Uncertainty bands ────────────────────────────────────────────


def test_uncertainty_bands_widen_under_flip_flop():
    cfg = ThesisMemoryConfig(enable_uncertainty=True,
                               uncertainty_window=8,
                               uncertainty_max_band=20.0)
    mem = ThesisMemory(cfg=cfg)
    # Flip-flopping IV state: BULL ↔ BEAR alternating.
    for i in range(10):
        iv = _iv(IV_DIRECTIONAL_BULL if i % 2 == 0
                  else IV_DIRECTIONAL_BEAR,
                  confidence=0.85,
                  direction=(1 if i % 2 == 0 else -1))
        flippy = mem.update(iv_state=iv)
    # Now run a consistent series in a fresh memory.
    mem2 = ThesisMemory(cfg=cfg)
    for _ in range(10):
        consistent = mem2.update(iv_state=_iv(IV_DIRECTIONAL_BULL,
                                                  confidence=0.85,
                                                  direction=1))
    assert flippy.bull_thesis_ci > consistent.bull_thesis_ci


def test_uncertainty_bands_bounded_by_max():
    cfg = ThesisMemoryConfig(enable_uncertainty=True,
                               uncertainty_max_band=10.0)
    mem = ThesisMemory(cfg=cfg)
    for i in range(20):
        iv = _iv([IV_DIRECTIONAL_BULL, IV_DIRECTIONAL_BEAR,
                    IV_NEUTRAL, IV_COMMON_SHOCK][i % 4],
                   confidence=0.85)
        snap = mem.update(iv_state=iv)
    assert snap.bull_thesis_ci <= 10.0
    assert snap.bull_thesis_ci >= 0.0


def test_uncertainty_early_session_wide_band():
    cfg = ThesisMemoryConfig(enable_uncertainty=True,
                               uncertainty_max_band=20.0)
    mem = ThesisMemory(cfg=cfg)
    # Fewer than 3 observations → wide default band.
    snap = mem.update(iv_state=_iv(IV_DIRECTIONAL_BULL,
                                       confidence=0.85, direction=1))
    assert snap.bull_thesis_ci > 0.0


def test_uncertainty_off_means_zero_ci_always():
    cfg = ThesisMemoryConfig(enable_uncertainty=False)
    mem = ThesisMemory(cfg=cfg)
    # Hit it with chaotic input — CI must still be 0 (legacy contract).
    for i in range(20):
        iv = _iv([IV_DIRECTIONAL_BULL, IV_DIRECTIONAL_BEAR,
                    IV_NEUTRAL][i % 3], confidence=0.85)
        snap = mem.update(iv_state=iv)
    assert snap.bull_thesis_ci == 0.0


def test_uncertainty_rejects_bad_config():
    with pytest.raises(ValueError):
        ThesisMemoryConfig(enable_uncertainty=True,
                            uncertainty_window=1)


# ── Serialization sanity ─────────────────────────────────────────


def test_snapshot_with_advanced_fields_serializes():
    import json
    cfg = ThesisMemoryConfig(enable_adaptive_decay=True,
                               enable_uncertainty=True)
    mem = ThesisMemory(cfg=cfg)
    for i in range(6):
        snap = mem.update(iv_state=_iv(IV_DIRECTIONAL_BULL,
                                            confidence=0.85, direction=1),
                              tape_speed=1.2)
    blob = json.dumps(snap.to_dict())
    rehydrated = json.loads(blob)
    assert "bull_thesis_ci" in rehydrated
    assert "effective_decay" in rehydrated
