"""Fat-tail amplifier — single scalar driving fat-tail defense.

Combines manipulation patterns + MM-mind + dispersion + crowd into a
single ``tail_score ∈ [0, 1]``. The aggregator uses this to:

  * refuse new positions when tail_score > 0.6
  * propose hedges when tail_score is mid (0.3-0.6) but a position is open
  * scale up size when tail_score is very low (<0.15) and other gates pass

The score is a transparent weighted sum of:

  1. **manipulation density**  — count of active manipulation patterns
  2. **stepping-back evidence** — MM stepping back / liquidity vacuum
  3. **dispersion velocity**    — single-strike pressure rising
  4. **regime instability**     — substrate's regime stability inverse
  5. **iv state risk**          — dirty/distortion/common-shock states
  6. **clean mark fraction**    — flow's mark quality
  7. **crowd density**          — from crowd_mirror, if available
  8. **mm intent volatility view** — expansion ↑, neutral ↓, crush ↓

Each component is normalized to [0, 1] and combined with weights that
sum to roughly 1.0. The final clip keeps the result in [0, 1].
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .manipulation_patterns import (
    PATTERN_LIQUIDITY_VACUUM, PATTERN_STOP_HUNT_LONG, PATTERN_STOP_HUNT_SHORT,
    PATTERN_FAKE_BREAKOUT, PATTERN_SQUEEZE_SETUP, PatternMatch,
)
from .market_maker_mind import (
    INTENT_STEPPING_BACK, INTENT_FAKING_DIRECTION, INTENT_HUNTING_STOPS,
    MMPosterior,
)


@dataclass
class FatTailAmplifierConfig:
    """Knobs for the amplifier."""
    # Component weights — sum to ~1.0.
    w_manipulation: float = 0.18
    w_stepping_back: float = 0.15
    w_dispersion: float = 0.12
    w_regime_instability: float = 0.12
    w_iv_state: float = 0.16
    w_clean_marks: float = 0.10
    w_crowd: float = 0.07
    w_mm_vol_view: float = 0.10
    # Action thresholds.
    refuse_score: float = 0.60
    hedge_score: float = 0.35
    scale_up_score: float = 0.15


@dataclass
class FatTailScore:
    """The amplifier's per-tick output."""
    tail_score: float                       # 0..1
    components: Dict[str, float]            # per-component contributions
    recommended_action: str                 # SCALE_UP / NORMAL / HEDGE / REFUSE
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tail_score": round(self.tail_score, 4),
            "components": {k: round(v, 4)
                            for k, v in self.components.items()},
            "recommended_action": self.recommended_action,
            "notes": list(self.notes),
        }


# Actions.
TAIL_ACTION_SCALE_UP = "SCALE_UP"
TAIL_ACTION_NORMAL = "NORMAL"
TAIL_ACTION_HEDGE = "HEDGE"
TAIL_ACTION_REFUSE = "REFUSE"


class FatTailAmplifier:
    """The single-scalar tail-risk synthesizer."""

    def __init__(self, cfg: Optional[FatTailAmplifierConfig] = None) -> None:
        self.cfg = cfg or FatTailAmplifierConfig()

    def amplify(self, *,
                  patterns: List[PatternMatch],
                  mm_posterior: Optional[MMPosterior],
                  flow_event: Any,
                  rich_context: Any,
                  iv_state: str,
                  crowd_density: float = 0.0,
                  ) -> FatTailScore:
        """Compute the tail score and recommended action."""
        cfg = self.cfg
        components: Dict[str, float] = {}
        notes: List[str] = []

        # 1. Manipulation density.
        risky_patterns = [m for m in patterns
                            if m.pattern_name in {PATTERN_LIQUIDITY_VACUUM,
                                                    PATTERN_STOP_HUNT_LONG,
                                                    PATTERN_STOP_HUNT_SHORT,
                                                    PATTERN_FAKE_BREAKOUT,
                                                    PATTERN_SQUEEZE_SETUP}]
        manip = min(1.0, len(risky_patterns) / 3.0)
        components["manipulation"] = manip * cfg.w_manipulation
        if manip > 0.3:
            notes.append(f"{len(risky_patterns)} risky manipulation patterns active")

        # 2. Stepping-back evidence.
        stepping = 0.0
        if mm_posterior is not None:
            stepping_p = mm_posterior.intent_distribution.get(
                INTENT_STEPPING_BACK, 0.0)
            stepping = min(1.0, stepping_p * 2.0)  # boost
            if stepping_p >= 0.30:
                notes.append(
                    f"MM stepping_back probability {stepping_p:.0%}"
                )
        components["stepping_back"] = stepping * cfg.w_stepping_back

        # 3. Dispersion velocity (substrate).
        disp_v = float(getattr(rich_context, "dispersion_velocity", 0.0))
        disp = min(1.0, abs(disp_v) / 0.30)
        components["dispersion"] = disp * cfg.w_dispersion

        # 4. Regime instability.
        regime_stab = float(getattr(rich_context, "regime_stability_index", 1.0))
        regime_inst = max(0.0, 1.0 - regime_stab)
        components["regime_instability"] = regime_inst * cfg.w_regime_instability

        # 5. IV state risk.
        iv_risk = 0.0
        if iv_state == "common_shock":
            iv_risk = 1.0
        elif iv_state == "dirty_data":
            iv_risk = 0.80
        elif iv_state == "liquidity_distortion":
            iv_risk = 0.70
        components["iv_state"] = iv_risk * cfg.w_iv_state
        if iv_risk > 0.50:
            notes.append(f"iv_state {iv_state} carries fat-tail risk")

        # 6. Clean marks.
        clean = float(getattr(flow_event, "clean_mark_fraction", 1.0))
        clean_risk = max(0.0, 1.0 - clean)   # 1 - clean = dirty fraction
        components["clean_marks"] = min(1.0, clean_risk * 3.0) * cfg.w_clean_marks

        # 7. Crowd density (we look like retail → MMs hunt).
        crowd_risk = max(0.0, min(1.0, crowd_density))
        components["crowd"] = crowd_risk * cfg.w_crowd

        # 8. MM volatility view.
        mm_vol_risk = 0.0
        if mm_posterior is not None:
            if mm_posterior.implied_volatility_view == "expansion":
                mm_vol_risk = 0.7
            elif mm_posterior.implied_volatility_view == "crush":
                mm_vol_risk = 0.4   # crush bleeds premiums; still a tail
            else:
                mm_vol_risk = 0.0
        components["mm_vol_view"] = mm_vol_risk * cfg.w_mm_vol_view

        score = sum(components.values())
        score = max(0.0, min(1.0, score))

        if score >= cfg.refuse_score:
            action = TAIL_ACTION_REFUSE
            notes.append(
                f"tail_score {score:.2f} ≥ refuse {cfg.refuse_score:.2f}"
            )
        elif score >= cfg.hedge_score:
            action = TAIL_ACTION_HEDGE
            notes.append(
                f"tail_score {score:.2f} in hedge band — propose protection"
            )
        elif score < cfg.scale_up_score:
            action = TAIL_ACTION_SCALE_UP
            notes.append(
                f"tail_score {score:.2f} very low — scale up allowed"
            )
        else:
            action = TAIL_ACTION_NORMAL

        return FatTailScore(
            tail_score=score,
            components=components,
            recommended_action=action,
            notes=notes,
        )
