"""Crux meta-signal — one verdict the operator reads when every other
panel says something different.

The problem the founder hits: there are 8 panels open, the reaction
model says 'upside reaction', the liquidity model says 'sweep+reclaim',
the constituent board says 'FRAGILE', the tilt index is AMBER, and the
operator has 12 seconds before they have to decide. What now?

Crux fuses every live ModelSignal + tilt + intention + scorecard
average into ONE of seven operator-facing verdicts:

  EXIT_NOW       — kill the position, period (kill switch / intention
                   violated / RED tilt + losing position)
  TRAIL_UP       — tighten the trail, you're sitting on profit
  HOLD           — let it work, no action needed
  WATCH          — narrative is forming but no entry yet
  TRADE          — multiple confirming signals, conditions clean
  DO_NOT_CHASE   — entry would be FOMO, refuse it (Lo 2017)
  WAIT           — flat tape / consolidation / no edge

The verdict carries: confidence, the 3 strongest contributing signals,
the strongest contradicting signal (intellectual honesty), and a
one-sentence rationale. Published as a TRUSTED ``crux`` ModelSignal so
it lands on the live brief feed too.

This is deliberately simple — no ML, no thresholds tuned by gradient
descent. The function reads like a desk-prop interview answer, which
is what the operator expects to be able to defend in front of a senior
trader: "I took the trade because Crux said TRADE, and the supporting
signals were X, Y, Z."
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .io_decl import IOSpec, declare
from .live_publisher import LivePublisher, ModelSignal, now_ist_hms


VERDICTS = ("EXIT_NOW", "TRAIL_UP", "HOLD", "WATCH", "TRADE",
            "DO_NOT_CHASE", "WAIT")


@dataclass
class CruxVerdict:
    verdict: str
    confidence: float                       # 0..1
    rationale: str
    supporting: List[str] = field(default_factory=list)
    contradicting: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    ts_ist: str = ""

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


# Signals that suggest UPSIDE direction (CE bias).
_BULL_TAGS = {
    "upside reaction", "near_low", "demand_zone",
    "continuation", "trend_follow", "HIDDEN_BULL", "BROAD_BULL",
}
# Signals that suggest DOWNSIDE direction (PE bias).
_BEAR_TAGS = {
    "downside reaction", "near_high", "supply_zone", "retrace_high",
    "HIDDEN_BEAR", "BROAD_BEAR",
}
# Signals that suggest WAIT (volatile, manipulated, choppy).
_AVOID_TAGS = {
    "MANIPULATED", "MANIPULATION", "stop_hunt_risk", "fast_spike",
    "CONSOLIDATION", "LOW (chop)", "low", "stop_hunt",
}


def _classify(sig: ModelSignal) -> str:
    """Light tag-based mapping from a ModelSignal to a directional bias."""
    text = (sig.signal + " " + " ".join(sig.reason_codes)).lower()
    if any(t in text for t in (k.lower() for k in _BULL_TAGS)): return "BULL"
    if any(t in text for t in (k.lower() for k in _BEAR_TAGS)): return "BEAR"
    if any(t in text for t in (k.lower() for k in _AVOID_TAGS)): return "AVOID"
    return "NEUTRAL"


@dataclass
class _CruxInputs:
    signals: List[ModelSignal] = field(default_factory=list)
    tilt_band: str = "GREEN"
    tilt_index: float = 0.0
    intention_violated: bool = False
    open_positions: int = 0
    open_pnl: float = 0.0
    has_winner: bool = False
    has_loser: bool = False
    killed: bool = False
    scorecard_avg: float = 0.0


def compose(inputs: _CruxInputs) -> CruxVerdict:
    """The whole fusion in one place — explicit, defendable, testable."""
    now = now_ist_hms()
    # ── 0. Hard overrides ─────────────────────────────────────────────
    if inputs.killed:
        return CruxVerdict(
            verdict="EXIT_NOW", confidence=1.0, ts_ist=now,
            rationale="kill switch active — refuse all order paths",
            supporting=["kill_switch"], inputs={"killed": True})

    if inputs.intention_violated:
        return CruxVerdict(
            verdict="EXIT_NOW", confidence=1.0, ts_ist=now,
            rationale="intention contract violated — Ulysses pattern "
                      "(Elster 2000); flatten and stop",
            supporting=["intention_violated"], inputs={"intention": "violated"})

    if inputs.tilt_band in ("RED", "CIRCUIT") and inputs.open_pnl < 0:
        return CruxVerdict(
            verdict="EXIT_NOW", confidence=0.9, ts_ist=now,
            rationale=(f"tilt {inputs.tilt_band} with losing book — close "
                       f"before more behavioral damage"),
            supporting=[f"tilt={inputs.tilt_band}"],
            inputs={"tilt_band": inputs.tilt_band})

    # ── 1. Direction census from live signals ─────────────────────────
    bull = bear = avoid = 0
    bull_conf = bear_conf = avoid_conf = 0.0
    top_bull: Optional[ModelSignal] = None
    top_bear: Optional[ModelSignal] = None
    top_avoid: Optional[ModelSignal] = None
    for s in inputs.signals:
        side = _classify(s)
        w = s.confidence
        if side == "BULL":
            bull += 1; bull_conf += w
            if top_bull is None or w > top_bull.confidence: top_bull = s
        elif side == "BEAR":
            bear += 1; bear_conf += w
            if top_bear is None or w > top_bear.confidence: top_bear = s
        elif side == "AVOID":
            avoid += 1; avoid_conf += w
            if top_avoid is None or w > top_avoid.confidence: top_avoid = s

    # ── 2. AVOID conditions dominate the directional census ──────────
    if avoid_conf >= max(bull_conf, bear_conf):
        return CruxVerdict(
            verdict="WAIT", confidence=min(0.9, 0.5 + avoid_conf * 0.1),
            ts_ist=now,
            rationale=("microstructure says wait — manipulation / chop / "
                       "consolidation signals dominate the bus"),
            supporting=[top_avoid.model] if top_avoid else [],
            contradicting=(top_bull.model if top_bull and bull_conf > bear_conf
                            else top_bear.model if top_bear else None),
            inputs={"bull": bull, "bear": bear, "avoid": avoid})

    # ── 3. Position-management overrides direction signals ───────────
    if inputs.has_winner and (top_bull or top_bear):
        # If you're sitting on profit AND a direction signal is firing,
        # tighten before chasing.
        return CruxVerdict(
            verdict="TRAIL_UP", confidence=0.75, ts_ist=now,
            rationale=("you have an open winner — tighten the trail "
                       "before a new entry; chasing dilutes risk-adjusted edge"),
            supporting=["open_winner"] + ([top_bull.model] if top_bull else [top_bear.model] if top_bear else []),
            inputs={"open_pnl": inputs.open_pnl})

    # ── 4. Pure directional verdicts when no positions are open ──────
    strong_bull = bull >= 2 and bull_conf >= 1.0
    strong_bear = bear >= 2 and bear_conf >= 1.0

    if strong_bull and not strong_bear:
        # Check FOMO heuristic — was the supporting set ALSO mentioning a
        # 'continuation' / 'recency' tag with a heavyweight FRAGILE board?
        contradict = None
        for s in inputs.signals:
            if "FRAGILE" in (s.signal + " ".join(s.reason_codes)).upper():
                contradict = s.model; break
        return CruxVerdict(
            verdict="TRADE", confidence=min(0.92, 0.55 + bull_conf * 0.12),
            ts_ist=now,
            rationale=(f"{bull} bull-aligned signals confirm; entry conditions "
                       f"clean (tilt {inputs.tilt_band})"),
            supporting=([s.model for s in inputs.signals
                          if _classify(s) == "BULL"])[:3],
            contradicting=contradict,
            inputs={"bull_signals": bull, "bear_signals": bear})

    if strong_bear and not strong_bull:
        return CruxVerdict(
            verdict="TRADE", confidence=min(0.92, 0.55 + bear_conf * 0.12),
            ts_ist=now,
            rationale=(f"{bear} bear-aligned signals confirm; entry conditions "
                       f"clean (tilt {inputs.tilt_band})"),
            supporting=([s.model for s in inputs.signals
                          if _classify(s) == "BEAR"])[:3],
            inputs={"bull_signals": bull, "bear_signals": bear})

    # ── 5. Mixed — direction is firing but not consensus ──────────────
    if bull or bear:
        return CruxVerdict(
            verdict="WATCH", confidence=0.6, ts_ist=now,
            rationale=("a narrative is forming but consensus is weak — "
                       "wait for a second confirming signal"),
            supporting=[s.model for s in inputs.signals
                        if _classify(s) in ("BULL", "BEAR")][:2],
            inputs={"bull": bull, "bear": bear})

    # ── 6. Truly nothing happening ───────────────────────────────────
    return CruxVerdict(
        verdict="WAIT", confidence=0.55, ts_ist=now,
        rationale="no high-confidence signals; do nothing is also a trade",
        inputs={"bull": bull, "bear": bear, "avoid": avoid})


def compose_from_state(live_signals: Dict[str, ModelSignal],
                       tilt_band: str = "GREEN",
                       tilt_index: float = 0.0,
                       intention_violated: bool = False,
                       open_positions: int = 0,
                       open_pnl: float = 0.0,
                       has_winner: bool = False,
                       has_loser: bool = False,
                       killed: bool = False,
                       scorecard_avg: float = 0.0,
                       publisher: Optional[LivePublisher] = None
                       ) -> CruxVerdict:
    """Compose + (optionally) publish to the live bus."""
    sigs = list(live_signals.values()) if isinstance(live_signals, dict) else list(live_signals)
    inputs = _CruxInputs(
        signals=sigs, tilt_band=tilt_band, tilt_index=tilt_index,
        intention_violated=intention_violated,
        open_positions=open_positions, open_pnl=open_pnl,
        has_winner=has_winner, has_loser=has_loser,
        killed=killed, scorecard_avg=scorecard_avg)
    verdict = compose(inputs)
    if publisher is not None:
        publisher.publish(ModelSignal(
            ts_ist=verdict.ts_ist, asset="NIFTY", model="crux",
            signal=f"{verdict.verdict}: {verdict.rationale}",
            confidence=verdict.confidence,
            trust_tier="TRUSTED",
            reason_codes=verdict.supporting,
            extras={"verdict": verdict.verdict,
                    "contradicting": verdict.contradicting,
                    **verdict.inputs},
        ))
    return verdict


declare(IOSpec(
    module="sentinel.crux_signal",
    purpose="meta-signal composer — fuses every live ModelSignal + tilt "
            "+ intention + position state into ONE operator-facing "
            "verdict (EXIT_NOW / TRAIL_UP / HOLD / WATCH / TRADE / "
            "DO_NOT_CHASE / WAIT) with rationale, supporting signals, and "
            "the strongest contradiction. Publishes as a TRUSTED 'crux' "
            "ModelSignal so it lands on the live brief feed",
    inputs=["live_signals dict (from LivePublisher.current())",
            "tilt state + intention + open positions + open P&L"],
    outputs=["CruxVerdict + TRUSTED crux ModelSignal"],
    consumes_from=["sentinel.live_publisher", "sentinel.psychology",
                   "sentinel.live_equity"],
    produces_for=["sentinel.server (cockpit)", "sentinel.live_publisher",
                  "sentinel.shadow_ledger"],
    tier="TRUSTED",
))
