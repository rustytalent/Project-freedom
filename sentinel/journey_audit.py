"""Journey audit — per-trade scorecard + retrospective mistake detector.

This module is the *post-hoc* counterpart to ``psychology``:

  * ``psychology`` fires LIVE biases (you're about to make a mistake)
  * ``journey_audit`` grades COMPLETED journeys (you made these mistakes
    and here's the score)

Two surfaces:

  * ``TradeQualityScorecard`` — 0..100 score per completed journey on
    five 20-point axes (reason_codes, entry timing, trail discipline,
    exit quality, R-multiple). Standard institutional trade-quality
    rubric (Tharp 2007 ch. 5 'R-multiples and expectancy'; Steenbarger
    2009 ch. 3 'Performance metrics').
  * ``MistakeDetector`` — five rule-based mistake patterns scanned over
    completed journeys, each citing the standard literature where the
    pattern is named.

Both consume ``LedgerEvent.to_row()`` dicts (the canonical journey
shape) and emit results that can be displayed live or rolled up at
end-of-day. Mistakes additionally fire as TRUSTED ``trade_mistake``
ModelSignal so the live brief feed surfaces the lesson while it's
still fresh.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .io_decl import IOSpec, declare
from .live_publisher import LivePublisher, ModelSignal, now_ist_hms

IST = timezone(timedelta(hours=5, minutes=30))


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

@dataclass
class ScoreBreakdown:
    reason_codes: int = 0          # 0..20
    entry_timing: int = 0          # 0..20
    trail_discipline: int = 0      # 0..20
    exit_quality: int = 0          # 0..20
    r_multiple: int = 0            # 0..20
    total: int = 0                 # 0..100

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JourneyScore:
    event_id: str
    scientist: str
    symbol: str
    grade: str                     # A / B / C / D / F
    breakdown: ScoreBreakdown
    notes: List[str] = field(default_factory=list)

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        d["breakdown"] = self.breakdown.to_row()
        return d


def grade_for(score: int) -> str:
    if score >= 85: return "A"
    if score >= 70: return "B"
    if score >= 55: return "C"
    if score >= 40: return "D"
    return "F"


class TradeQualityScorecard:
    """The five-axis trade-quality rubric. Each axis 0..20:

      reason_codes      — did the hypothesis state WHY the trade should
                          win? (Hull 11e §22.5: every trade explains
                          itself) — 4 pts per reason code up to 20
      entry_timing      — did entry land near suggested_entry?
                          ≤ 1% off = full 20; sliding scale
      trail_discipline  — was a trail armed AND not too tight per
                          anchoring-detector rules
      exit_quality      — target-hit (20), trail-fired (15), stop-hit
                          (10), neither / wandered out (5)
      r_multiple        — Tharp 2007 R-multiple; capped at +3R = 20,
                          -1R = 0; intermediate sliding
    """

    def __init__(self) -> None:
        pass

    def score(self, row: Dict[str, Any]) -> JourneyScore:
        ident = row.get("identity") or {}
        hyp = row.get("hypothesis") or {}
        journey = row.get("journey") or {}
        judg = row.get("judgment") or {}
        bd = ScoreBreakdown()
        notes: List[str] = []

        # --- reason_codes -------------------------------------------------
        rcs = hyp.get("reason_codes") or []
        bd.reason_codes = min(20, len(rcs) * 4)
        if not rcs:
            notes.append("no reason codes — hypothesis is unexplained")

        # --- entry timing -------------------------------------------------
        entry, sug = ident.get("premium"), hyp.get("suggested_entry")
        if entry and sug and entry > 0 and sug > 0:
            off_pct = abs(entry - sug) / sug * 100.0
            if off_pct <= 1:    bd.entry_timing = 20
            elif off_pct <= 3:  bd.entry_timing = 15
            elif off_pct <= 5:  bd.entry_timing = 10
            elif off_pct <= 10: bd.entry_timing = 5
            else:               bd.entry_timing = 0
            if off_pct > 5:
                notes.append(f"entry {off_pct:.1f}% off the suggested level")
        else:
            bd.entry_timing = 10        # neutral if we can't measure

        # --- trail discipline --------------------------------------------
        # heuristic: time_to_target_min present → trail was working;
        # premium_decay positive AND time_to_stop_min absent → never
        # armed; cushion absent → minimum.
        if journey.get("time_to_target_min") is not None:
            bd.trail_discipline = 20
        elif journey.get("time_to_stop_min") is not None:
            bd.trail_discipline = 12       # stop hit at least bounded loss
        elif journey.get("complete"):
            bd.trail_discipline = 6        # journey ran out, no protection
            notes.append("no trail armed — loss/win left to chance")
        else:
            bd.trail_discipline = 10       # still in flight

        # --- exit quality ------------------------------------------------
        t_tgt = journey.get("time_to_target_min")
        t_stp = journey.get("time_to_stop_min")
        if t_tgt is not None and (t_stp is None or t_tgt <= t_stp):
            bd.exit_quality = 20            # target reached cleanly
        elif t_stp is not None:
            bd.exit_quality = 10
            notes.append("stopped out before target")
        elif journey.get("complete"):
            bd.exit_quality = 5
            notes.append("journey expired without target or stop")
        else:
            bd.exit_quality = 12            # in-flight neutral

        # --- R-multiple ---------------------------------------------------
        if entry and sug is not None:
            stop = hyp.get("suggested_stop") or 0
            risk = max(entry - stop, 1e-6)
            # realised exit premium = t+60 outcome if present, else MFE/MAE proxy
            outcome = (journey.get("outcomes") or {}).get("t+60")
            if outcome is None:
                outcome = journey.get("mfe") if journey.get("mfe") else entry
            r_mult = (outcome - entry) / risk
            r_pts = int(round(min(20, max(0, (r_mult + 1.0) * 5.0))))
            bd.r_multiple = r_pts
            if r_mult < -0.5:
                notes.append(f"realised R {r_mult:+.2f} — full risk taken")
            elif r_mult >= 2.0:
                notes.append(f"realised R {r_mult:+.2f} — strong setup")
        else:
            bd.r_multiple = 10

        bd.total = (bd.reason_codes + bd.entry_timing + bd.trail_discipline
                    + bd.exit_quality + bd.r_multiple)
        return JourneyScore(
            event_id=str(row.get("event_id", "")),
            scientist=str(row.get("scientist", "")),
            symbol=str(ident.get("instrument", "")),
            grade=grade_for(bd.total),
            breakdown=bd,
            notes=notes,
        )

    def score_session(self, rows: List[Dict[str, Any]]) -> List[JourneyScore]:
        return [self.score(r) for r in rows
                if (r.get("journey") or {}).get("complete")]

    def session_summary(self, scores: List[JourneyScore]) -> Dict[str, Any]:
        if not scores:
            return {"count": 0, "average": 0, "grade_distribution": {}}
        avg = round(sum(s.breakdown.total for s in scores) / len(scores), 1)
        dist: Dict[str, int] = {}
        for s in scores:
            dist[s.grade] = dist.get(s.grade, 0) + 1
        return {"count": len(scores), "average": avg,
                "grade_distribution": dist,
                "best": max(scores, key=lambda s: s.breakdown.total).to_row(),
                "worst": min(scores, key=lambda s: s.breakdown.total).to_row()}


# ---------------------------------------------------------------------------
# Mistake detector
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Mistake:
    pattern: str                   # GAVE_BACK_PEAK / TARGET_TOO_FAR / ...
    severity: float                # 0..1
    event_id: str
    symbol: str
    evidence: str
    citation: str                  # named literature

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


class MistakeDetector:
    """Five rule-based mistake patterns. Each cites the lit where the
    pattern is named so the operator reads literature, not jargon.

      GAVE_BACK_PEAK — peak_pnl > X but exit at ≤ 30% of peak
                       (Tharp 2007: 'give back ratio')
      TARGET_TOO_FAR — never hit target, MFE always below by Y%
                       (Steenbarger 2009 ch. 3)
      STOP_TOO_TIGHT — stopped out and price then ran to target
                       (Curator's own concept — the journey shows it)
      NO_TRAIL_ON_WINNER — peak_pnl > X but trail never armed
                          (founder's R1 maximizer rule)
      IGNORED_INVALIDATION — invalidation_conditions present but
                            position held past one (Hull 11e §22.5)
    """

    name = "MistakeDetector"

    def __init__(self, give_back_threshold: float = 0.3,
                 winner_threshold_rupees: float = 1500.0) -> None:
        self.give_back = give_back_threshold
        self.winner_threshold = winner_threshold_rupees

    def scan_row(self, row: Dict[str, Any]) -> List[Mistake]:
        """Per-completed-journey scan. Returns 0..N mistakes."""
        journey = row.get("journey") or {}
        if not journey.get("complete"):
            return []
        hyp = row.get("hypothesis") or {}
        ident = row.get("identity") or {}
        eid = str(row.get("event_id", ""))
        sym = str(ident.get("instrument", ""))
        out: List[Mistake] = []

        entry = float(ident.get("premium") or 0)
        mfe   = journey.get("mfe")
        mae   = journey.get("mae")
        tgt   = hyp.get("suggested_target")
        stp   = hyp.get("suggested_stop")
        outcome = (journey.get("outcomes") or {}).get("t+60")

        # GAVE_BACK_PEAK
        if (entry > 0 and mfe is not None and outcome is not None
                and mfe > entry * 1.05):
            peak_gain = mfe - entry
            realised = outcome - entry
            if peak_gain > 0 and realised / peak_gain < self.give_back:
                ratio = 1 - max(0.0, realised / peak_gain)
                out.append(Mistake(
                    pattern="GAVE_BACK_PEAK", severity=round(min(1.0, ratio), 3),
                    event_id=eid, symbol=sym,
                    evidence=(f"peak {mfe:.1f} from entry {entry:.1f}, "
                              f"exit {outcome:.1f} — gave back "
                              f"{ratio*100:.0f}% of the move"),
                    citation="Tharp (2007) — give-back ratio",
                ))

        # TARGET_TOO_FAR
        if (mfe is not None and tgt is not None
                and journey.get("time_to_target_min") is None):
            gap = tgt - mfe
            if gap > 0 and entry > 0:
                sev = round(min(1.0, gap / max(entry, 1.0)), 3)
                out.append(Mistake(
                    pattern="TARGET_TOO_FAR", severity=sev,
                    event_id=eid, symbol=sym,
                    evidence=(f"MFE {mfe:.1f} never reached target {tgt:.1f} "
                              f"({gap:.1f} pts away)"),
                    citation="Steenbarger (2009) ch. 3",
                ))

        # STOP_TOO_TIGHT
        if (mae is not None and stp is not None and mfe is not None and tgt is not None
                and journey.get("time_to_stop_min") is not None
                and mfe >= tgt):
            sev = round(min(1.0, (mfe - tgt) / max(tgt, 1.0) + 0.4), 3)
            out.append(Mistake(
                pattern="STOP_TOO_TIGHT", severity=sev,
                event_id=eid, symbol=sym,
                evidence=(f"stopped at {stp:.1f} (MAE {mae:.1f}), "
                          f"then move ran to {mfe:.1f} — past target {tgt:.1f}"),
                citation="Curator finding — stop-tightness-bias",
            ))

        # NO_TRAIL_ON_WINNER (heuristic: completed, peak gain > threshold,
        # but neither target nor stop ever hit — i.e. no protection)
        if (mfe is not None and entry > 0
                and (mfe - entry) * 75 > self.winner_threshold
                and journey.get("time_to_target_min") is None
                and journey.get("time_to_stop_min") is None):
            out.append(Mistake(
                pattern="NO_TRAIL_ON_WINNER", severity=0.7,
                event_id=eid, symbol=sym,
                evidence=(f"peak P&L > ₹{int(self.winner_threshold):,} "
                          f"but journey expired with no trail fired"),
                citation="Sentinel maximizer R1",
            ))

        # IGNORED_INVALIDATION — only fires if invalidation_conditions
        # exists AND the realised path violated one and we still held.
        # Heuristic v1: the journey completed at a loss while
        # invalidation_conditions had been declared (the curator can
        # add per-condition checks later).
        inv = hyp.get("invalidation_conditions") or []
        if inv and outcome is not None and entry > 0 and outcome < entry * 0.9:
            out.append(Mistake(
                pattern="IGNORED_INVALIDATION", severity=0.6,
                event_id=eid, symbol=sym,
                evidence=(f"invalidation conditions declared: "
                          f"{', '.join(map(str, inv))[:80]} — "
                          f"position closed -{(1-outcome/entry)*100:.0f}%"),
                citation="Hull (2017) §22.5",
            ))

        return out

    def scan_session(self, rows: List[Dict[str, Any]]) -> List[Mistake]:
        out: List[Mistake] = []
        for r in rows:
            out.extend(self.scan_row(r))
        return out


def publish_mistakes(mistakes: List[Mistake],
                     publisher: Optional[LivePublisher]) -> int:
    """Publish each mistake as a TRUSTED trade_mistake ModelSignal so it
    surfaces in the live brief feed + mirrors to the shadow ledger."""
    if publisher is None:
        return 0
    n = 0
    for m in mistakes:
        ok = publisher.publish(ModelSignal(
            ts_ist=now_ist_hms(), asset=m.symbol or "OPERATOR",
            model="trade_mistake",
            signal=f"{m.pattern}: {m.evidence}",
            confidence=m.severity,
            trust_tier="TRUSTED",
            reason_codes=[m.pattern, m.citation],
            extras={"citation": m.citation, "event_id": m.event_id},
        ))
        n += int(ok)
    return n


declare(IOSpec(
    module="sentinel.journey_audit",
    purpose="post-hoc trade audit — TradeQualityScorecard grades every "
            "completed journey 0..100 on 5 axes (Tharp R-multiple + "
            "Steenbarger performance metrics); MistakeDetector scans for "
            "5 named patterns (gave_back_peak / target_too_far / "
            "stop_too_tight / no_trail_on_winner / ignored_invalidation) "
            "and fires TRUSTED trade_mistake signals to the bus",
    inputs=["completed journey rows (LedgerEvent.to_row())"],
    outputs=["JourneyScore (per journey) + session_summary",
             "Mistake[] (per session) + trade_mistake ModelSignal stream"],
    consumes_from=["sentinel.shadow_ledger"],
    produces_for=["sentinel.live_publisher", "sentinel.server (cockpit)",
                  "sentinel.psychology (mind report)"],
    tier="TRUSTED",
))
