"""Exit modes — Harvest, Shade, Chase-fill, De-risk, Kill.

Each open position is always in exactly one mode. The mode determines
how aggressively the engine moves the ghost price and how willingly it
spends modification budget.

Mode semantics (founder + ChatGPT alignment):

  HARVEST   — Thesis working, premium behaving well. Hold target,
              don't waste mods. Maybe raise ghost.
  SHADE     — Thesis OK but target weakening. Lower ghost target.
              Modify broker only if new target is meaningfully more
              fillable.
  CHASE_FILL — Price is near exit but not filling. Move limit slightly
               toward market. Spend mods carefully.
  DE_RISK   — Portfolio risk too high OR thesis decaying. Prefer fill
              over perfect price. Aggressive ghost reduction.
  KILL      — Thesis broken. Cancel passive order, exit aggressively
              at market. No ego.

Mode transitions use hysteresis to avoid whiplash: a 2-tick dwell
minimum at each mode, plus mode-pair scores for the transition gate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Mode constants ────────────────────────────────────────────────


EXIT_MODE_HARVEST = "HARVEST"
EXIT_MODE_SHADE = "SHADE"
EXIT_MODE_CHASE_FILL = "CHASE_FILL"
EXIT_MODE_DE_RISK = "DE_RISK"
EXIT_MODE_KILL = "KILL"

ExitMode = str   # Lightweight; just one of the above.


# ── Decision shape ────────────────────────────────────────────────


@dataclass
class ExitModeDecision:
    """The result of a mode decision for one position."""
    mode: ExitMode
    previous_mode: ExitMode
    transitioned: bool
    confidence: float                # 0..1, how strong the case for this mode is
    reasons: List[str]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "previous_mode": self.previous_mode,
            "transitioned": self.transitioned,
            "confidence": round(self.confidence, 3),
            "reasons": list(self.reasons),
            "notes": list(self.notes),
        }


# ── Hysteresis state ──────────────────────────────────────────────


@dataclass
class _ModeHysteresis:
    """Per-position dwell counters to prevent whiplash."""
    current_mode: ExitMode = EXIT_MODE_HARVEST
    bars_in_current_mode: int = 0
    last_proposed_mode: Optional[ExitMode] = None
    bars_proposing: int = 0


# ── Decider ───────────────────────────────────────────────────────


def decide_exit_mode(*,
                       hysteresis: _ModeHysteresis,
                       thesis_intact: bool,
                       thesis_confidence_decay: float,
                       current_r: float,
                       revisit_probability: float,
                       high_trajectory: str,
                       bars_held: int,
                       max_hold_bars: int,
                       fat_tail_action: str = "NORMAL",
                       portfolio_under_pressure: bool = False,
                       price_near_exit: bool = False,
                       min_dwell_bars: int = 2,
                       ) -> ExitModeDecision:
    """Pick the next exit mode based on the current evidence.

    The decision rule is deterministic and transparent — each mode has
    a set of trigger conditions, and the strongest-match wins, subject
    to hysteresis.
    """
    reasons: List[str] = []
    notes: List[str] = []

    # ── KILL: thesis broken / hard invalidation ─────────────────────
    if not thesis_intact:
        return _maybe_transition(hysteresis, EXIT_MODE_KILL,
                                    confidence=1.0,
                                    reasons=["thesis intact = False"],
                                    notes=notes, min_dwell_bars=0)
    if thesis_confidence_decay >= 0.40:
        return _maybe_transition(hysteresis, EXIT_MODE_KILL,
                                    confidence=0.85,
                                    reasons=[f"confidence decay "
                                              f"{thesis_confidence_decay:.2f} > 0.40"],
                                    notes=notes, min_dwell_bars=0)
    if current_r <= -0.85:
        return _maybe_transition(hysteresis, EXIT_MODE_KILL,
                                    confidence=0.90,
                                    reasons=[f"R {current_r:.2f} near stop"],
                                    notes=notes, min_dwell_bars=0)
    if bars_held >= max_hold_bars:
        return _maybe_transition(hysteresis, EXIT_MODE_KILL,
                                    confidence=0.80,
                                    reasons=[f"max hold {max_hold_bars} reached"],
                                    notes=notes, min_dwell_bars=0)
    if fat_tail_action == "REFUSE":
        return _maybe_transition(hysteresis, EXIT_MODE_KILL,
                                    confidence=0.75,
                                    reasons=["fat-tail REFUSE — flat now"],
                                    notes=notes, min_dwell_bars=0)

    # ── DE_RISK: portfolio pressure or strong thesis decay ──────────
    if portfolio_under_pressure or thesis_confidence_decay >= 0.25:
        return _maybe_transition(hysteresis, EXIT_MODE_DE_RISK,
                                    confidence=0.70,
                                    reasons=[
                                        "portfolio under pressure"
                                        if portfolio_under_pressure
                                        else f"thesis decay "
                                              f"{thesis_confidence_decay:.2f}"
                                    ], notes=notes,
                                    min_dwell_bars=min_dwell_bars)

    # ── CHASE_FILL: price near exit but not filling ─────────────────
    if price_near_exit and revisit_probability >= 0.40:
        return _maybe_transition(hysteresis, EXIT_MODE_CHASE_FILL,
                                    confidence=0.65,
                                    reasons=[
                                        f"price near exit; "
                                        f"revisit prob {revisit_probability:.2f}"
                                    ], notes=notes,
                                    min_dwell_bars=min_dwell_bars)

    # ── HARVEST: highs rising or working in our favor ───────────────
    if (high_trajectory == "rising" and revisit_probability >= 0.50
            and current_r >= 0.30):
        return _maybe_transition(hysteresis, EXIT_MODE_HARVEST,
                                    confidence=0.75,
                                    reasons=[
                                        f"highs rising; revisit prob "
                                        f"{revisit_probability:.2f}; R "
                                        f"{current_r:.2f}"
                                    ], notes=notes,
                                    min_dwell_bars=min_dwell_bars)

    # ── SHADE: highs flat/falling but thesis OK ─────────────────────
    if high_trajectory in ("flat", "falling") or revisit_probability < 0.40:
        return _maybe_transition(hysteresis, EXIT_MODE_SHADE,
                                    confidence=0.55,
                                    reasons=[
                                        f"highs {high_trajectory}; "
                                        f"revisit prob {revisit_probability:.2f}"
                                    ], notes=notes,
                                    min_dwell_bars=min_dwell_bars)

    # ── Default: HARVEST ────────────────────────────────────────────
    return _maybe_transition(hysteresis, EXIT_MODE_HARVEST,
                                confidence=0.50,
                                reasons=["no special condition; default harvest"],
                                notes=notes, min_dwell_bars=min_dwell_bars)


# ── Internals ──────────────────────────────────────────────────────


def _maybe_transition(hysteresis: _ModeHysteresis,
                        proposed_mode: ExitMode, *,
                        confidence: float, reasons: List[str],
                        notes: List[str], min_dwell_bars: int
                        ) -> ExitModeDecision:
    """Apply hysteresis to the proposal."""
    current = hysteresis.current_mode

    # Always advance dwell counter for the CURRENT mode — regardless of
    # what's being proposed. That way "I've been in HARVEST for 5 ticks"
    # is true whether or not other modes are being proposed each tick.
    hysteresis.bars_in_current_mode += 1

    if proposed_mode == current:
        # Same mode confirmed — reset proposal-of-something-else counter.
        hysteresis.last_proposed_mode = None
        hysteresis.bars_proposing = 0
        return ExitModeDecision(
            mode=current, previous_mode=current, transitioned=False,
            confidence=confidence, reasons=reasons, notes=notes,
        )

    # KILL mode bypasses hysteresis — safety first.
    if proposed_mode == EXIT_MODE_KILL:
        previous = current
        hysteresis.current_mode = EXIT_MODE_KILL
        hysteresis.bars_in_current_mode = 1
        hysteresis.last_proposed_mode = None
        hysteresis.bars_proposing = 0
        notes.append(f"KILL transition from {previous} (no hysteresis)")
        return ExitModeDecision(
            mode=EXIT_MODE_KILL, previous_mode=previous,
            transitioned=True, confidence=confidence,
            reasons=reasons, notes=notes,
        )

    # Other transitions require minimum dwell in current mode + repeat proposal.
    if hysteresis.bars_in_current_mode < min_dwell_bars:
        notes.append(f"dwell {hysteresis.bars_in_current_mode}/{min_dwell_bars} "
                      f"in {current}; refusing transition to {proposed_mode}")
        return ExitModeDecision(
            mode=current, previous_mode=current, transitioned=False,
            confidence=confidence, reasons=reasons, notes=notes,
        )

    # Need 2 ticks of same proposal to transition.
    if hysteresis.last_proposed_mode == proposed_mode:
        hysteresis.bars_proposing += 1
    else:
        hysteresis.last_proposed_mode = proposed_mode
        hysteresis.bars_proposing = 1

    if hysteresis.bars_proposing < 2:
        notes.append(f"proposing {proposed_mode} only "
                      f"{hysteresis.bars_proposing}/2 — holding {current}")
        return ExitModeDecision(
            mode=current, previous_mode=current, transitioned=False,
            confidence=confidence, reasons=reasons, notes=notes,
        )

    # Transition!
    previous = current
    hysteresis.current_mode = proposed_mode
    hysteresis.bars_in_current_mode = 1
    hysteresis.last_proposed_mode = None
    hysteresis.bars_proposing = 0
    return ExitModeDecision(
        mode=proposed_mode, previous_mode=previous,
        transitioned=True, confidence=confidence,
        reasons=reasons, notes=notes,
    )
