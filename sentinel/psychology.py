"""Behavioral engine — the founder's 'mechanize anti-greed psychology'
made testable.

Almost every Indian options platform shows Greeks and a P&L curve.
None of them tell the operator they are about to make a *revenge
trade*, that their last three winners made them *enlarge their next
size* (the hot-hand fallacy), or that their stop-loss is glued to a
round-number anchor of their entry price.

That is the gap. Behavioral finance has fifty years of literature on
exactly the failure modes a discretionary options trader walks into;
this module reads our shadow ledger and live state and fires a signal
the moment any of the canonical biases is firing.

Every detector carries its academic citation in plain text, so the
operator sees not just "REVENGE" but "Steenbarger (2009): trade
following a loss within 5 min is statistically worse." Same
ground-anchored framework rule the rest of the codebase obeys.

Detectors in this v1:

  RevengeTradeDetector       — Steenbarger 2009 "Enhancing Trader
                               Performance"; loss-aversion over-
                               compensation (Kahneman 2011 §26)
  FomoEntryDetector          — Lo 2017 "Adaptive Markets"; recency
                               + herd bias (Shiller 2000)
  HotHandDetector            — Gilovich, Vallone & Tversky 1985;
                               size escalation after winning streak
  DispositionEffectDetector  — Shefrin & Statman 1985; holding
                               losers too long, taking winners too
                               fast (loss aversion + mental
                               accounting)
  AnchoringDetector          — Tversky & Kahneman 1974; stops
                               clustered on round numbers near entry
  RiskBudgetDetector         — Tharp 2007 "Trade Your Way to
                               Financial Freedom"; R-multiple risk
                               budgeting; today's realized loss vs
                               session VaR budget

Above the detectors:

  TiltIndex                  — composite 0-100 emotional-state
                               gauge with EWMA decay; the operator's
                               "should I be trading right now?" dial
  IntentionContract          — Ulysses-contract pattern (Elster 2000):
                               pre-market the operator commits to a
                               max loss / max positions / no-trade-
                               after time; live state checks it
  MindReport                 — end-of-session reflection: tilt avg
                               & peak, biases-by-type, intention
                               kept / violated, with citations.

Wired into the spine: every bias event is also a TRUSTED ModelSignal
published to the LivePublisher so it shows in the live brief feed and
mirrors to the canonical ShadowLedger.
"""
from __future__ import annotations

import json
import math
import threading
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from .io_decl import IOSpec, declare
from .live_publisher import LivePublisher, ModelSignal, now_ist_hms

IST = timezone(timedelta(hours=5, minutes=30))


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BiasEvent:
    """One detected bias firing. Frozen so the bus can hand the same
    instance to many readers."""
    bias: str               # REVENGE / FOMO / HOT_HAND / DISPOSITION / ANCHORING / RISK_BUDGET
    severity: float         # 0..1
    detected_at_ist: str    # HH:MM:SS
    citation: str
    evidence: str
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TiltState:
    """Composite emotional-state read-out."""
    index: float = 0.0                   # 0..100 (higher = more tilted)
    band: str = "GREEN"                  # GREEN < 30 < AMBER < 60 < RED < 85 < CIRCUIT
    biases_active: List[str] = field(default_factory=list)
    recent: List[BiasEvent] = field(default_factory=list)
    last_update_ist: str = ""

    def to_row(self) -> Dict[str, Any]:
        return {
            "index": round(self.index, 1),
            "band": self.band,
            "biases_active": list(self.biases_active),
            "recent": [b.to_row() for b in self.recent],
            "last_update_ist": self.last_update_ist,
        }


@dataclass
class IntentionContract:
    """The operator's pre-market commitment (Ulysses contract pattern,
    Elster 2000 "Ulysses Unbound"). Set once at the open; the live
    state checks every cycle.

    Once set, breaching the max day loss or hitting the no-trade-after
    time flips ``violated`` to True. Server-side, the spine should
    refuse non-EXECUTION upgrades while violated — making the
    behavioral commitment teeth, not just a sticky note.
    """
    set_at_ist: str = ""
    max_day_loss_rupees: float = 0.0
    target_day_profit_rupees: float = 0.0
    max_positions: int = 0               # 0 = unlimited
    no_trade_after_ist: str = "15:15"
    notes: str = ""
    violated: bool = False
    violation_reason: str = ""

    @property
    def is_set(self) -> bool:
        return bool(self.set_at_ist)

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)

    def check(self, current_pnl: float, open_positions: int,
              now_ist_hms: str) -> Optional[str]:
        """Returns the FIRST violation reason found, or None. Idempotent —
        callers can re-check every cycle."""
        if not self.is_set:
            return None
        if self.max_day_loss_rupees > 0 and current_pnl <= -abs(self.max_day_loss_rupees):
            return (f"day loss ₹{abs(current_pnl):,.0f} exceeded committed "
                    f"max ₹{self.max_day_loss_rupees:,.0f}")
        if self.max_positions > 0 and open_positions > self.max_positions:
            return (f"{open_positions} open positions exceeds committed "
                    f"max {self.max_positions}")
        if self.no_trade_after_ist and now_ist_hms >= self.no_trade_after_ist:
            if open_positions > 0:
                return (f"clock past committed no-trade-after "
                        f"{self.no_trade_after_ist} with {open_positions} "
                        f"positions still open")
        return None


@dataclass
class MindReport:
    """End-of-session reflection. Persisted to the journal so the
    operator can read the same report next morning."""
    session: str
    tilt_avg: float = 0.0
    tilt_peak: float = 0.0
    biases_by_type: Dict[str, int] = field(default_factory=dict)
    most_common_bias: Optional[str] = None
    revenge_trades: int = 0
    fomo_entries: int = 0
    disciplined_trails: int = 0          # times the operator armed a trail
    intention_was_set: bool = False
    intention_violated: bool = False
    intention_violation_reason: str = ""
    notable_quotes: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Detectors — each grounded in a citation, each pure (no I/O)
# ---------------------------------------------------------------------------

# Convention for the action records the detectors consume.
# Sentinel feeds them from the shadow ledger + the trail engine + the
# kite_client's orders_log; each row is normalized to this shape so the
# detectors are easy to unit-test.
@dataclass(frozen=True)
class TradeAction:
    ts_ist: str                         # HH:MM:SS
    action: str                         # ENTRY / EXIT / TRAIL_ARMED / TRAIL_FIRED / SUGGESTION
    symbol: str = ""
    qty: int = 0
    pnl: float = 0.0                    # only meaningful on EXIT
    premium: float = 0.0
    spot: float = 0.0
    note: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)


def _hms_minus(a: str, b: str) -> float:
    """Minutes from b -> a, assuming both are HH:MM:SS in the same day."""
    fmt = "%H:%M:%S"
    try:
        da = datetime.strptime(a, fmt); db = datetime.strptime(b, fmt)
        return (da - db).total_seconds() / 60.0
    except Exception:
        return 0.0


class RevengeTradeDetector:
    """Steenbarger 2009 (ch. "Coaching Through Drawdowns"): a fresh
    entry within ``window_min`` of a closed loss is statistically worse
    than the trader's baseline — loss-aversion overcompensation
    (Kahneman & Tversky 1979).

    Severity scales with: how recently the loss closed, how big it was,
    and whether the new size is >= the losing size (escalation)."""

    name = "REVENGE"
    citation = "Steenbarger (2009); Kahneman & Tversky (1979)"

    def __init__(self, window_min: float = 5.0,
                 min_loss_rupees: float = 500.0) -> None:
        self.window_min = window_min
        self.min_loss = min_loss_rupees

    def scan(self, recent: List[TradeAction], now_ist: str) -> Optional[BiasEvent]:
        # find the most recent ENTRY action
        entries = [a for a in recent if a.action == "ENTRY"]
        exits   = [a for a in recent if a.action == "EXIT" and a.pnl < -self.min_loss]
        if not entries or not exits:
            return None
        last_entry = entries[-1]
        # the closest losing exit BEFORE this entry
        prior = [e for e in exits if e.ts_ist < last_entry.ts_ist]
        if not prior:
            return None
        loss = prior[-1]
        dt = _hms_minus(last_entry.ts_ist, loss.ts_ist)
        if not (0 <= dt <= self.window_min):
            return None
        # severity components
        size_escalation = (last_entry.qty / max(1, loss.qty)) if loss.qty else 1.0
        loss_severity = min(1.0, abs(loss.pnl) / 5000.0)
        time_severity = max(0.0, 1.0 - dt / self.window_min)
        sev = round(min(1.0, 0.4 * loss_severity + 0.4 * time_severity
                        + 0.2 * min(1.0, size_escalation / 2.0)), 3)
        return BiasEvent(
            bias=self.name, severity=sev, detected_at_ist=now_ist,
            citation=self.citation,
            evidence=(f"entry on {last_entry.symbol} {dt:.1f} min after "
                      f"closing {loss.symbol} for ₹{loss.pnl:,.0f}"),
            extras={"loss_rupees": loss.pnl, "delay_min": round(dt, 2),
                    "size_escalation": round(size_escalation, 2)},
        )


class FomoEntryDetector:
    """Lo (2017) "Adaptive Markets" + Shiller (2000) "Irrational
    Exuberance": entries placed AFTER a fast multi-percent expansion in
    the spot are statistically worse — recency + herd bias.

    Detects: ENTRY action where the spot moved more than ``move_pct``
    in the ``lookback_min`` window immediately before the entry, with
    the entry direction lining up with the recent move (chasing it).
    """

    name = "FOMO"
    citation = "Lo (2017); Shiller (2000)"

    def __init__(self, lookback_min: float = 10.0,
                 move_pct: float = 0.4) -> None:
        self.lookback_min = lookback_min
        self.move_pct = move_pct

    def scan(self, recent: List[TradeAction],
             spot_hist: List[tuple], now_ist: str) -> Optional[BiasEvent]:
        """spot_hist is [(ts_ist, spot), ...] in chronological order."""
        entries = [a for a in recent if a.action == "ENTRY"]
        if not entries or len(spot_hist) < 3:
            return None
        last = entries[-1]
        ts_then = next((s for s in spot_hist if _hms_minus(last.ts_ist, s[0])
                        <= self.lookback_min and _hms_minus(last.ts_ist, s[0]) > 0), None)
        spot_at_entry = next((s for s in reversed(spot_hist) if s[0] <= last.ts_ist), None)
        if ts_then is None or spot_at_entry is None or ts_then[1] <= 0:
            return None
        move = (spot_at_entry[1] - ts_then[1]) / ts_then[1] * 100.0
        if abs(move) < self.move_pct:
            return None
        # CE entries chase upside, PE entries chase downside; without
        # option_type info we use the note as a fallback heuristic.
        ce = "CE" in last.symbol.upper()
        pe = "PE" in last.symbol.upper()
        chasing = (ce and move > 0) or (pe and move < 0) or (not ce and not pe)
        if not chasing:
            return None
        sev = round(min(1.0, abs(move) / (self.move_pct * 3.0)), 3)
        return BiasEvent(
            bias=self.name, severity=sev, detected_at_ist=now_ist,
            citation=self.citation,
            evidence=(f"entered {last.symbol} after spot moved "
                      f"{move:+.2f}% in last {self.lookback_min:.0f} min"),
            extras={"prior_move_pct": round(move, 2)},
        )


class HotHandDetector:
    """Gilovich, Vallone & Tversky (1985): there is no 'hot hand' in
    statistically independent outcomes — but humans size up after
    winning streaks. Detects ``streak_min`` consecutive wins followed
    by a position whose size is materially larger than the streak's
    average."""

    name = "HOT_HAND"
    citation = "Gilovich, Vallone & Tversky (1985)"

    def __init__(self, streak_min: int = 3,
                 size_multiplier: float = 1.5) -> None:
        self.streak_min = streak_min
        self.size_multiplier = size_multiplier

    def scan(self, recent: List[TradeAction], now_ist: str) -> Optional[BiasEvent]:
        exits = [a for a in recent if a.action == "EXIT"]
        if len(exits) < self.streak_min:
            return None
        last_streak = exits[-self.streak_min:]
        if not all(e.pnl > 0 for e in last_streak):
            return None
        entries = [a for a in recent if a.action == "ENTRY"]
        if not entries:
            return None
        last = entries[-1]
        if last.ts_ist <= last_streak[-1].ts_ist:
            return None     # entry must be AFTER the streak completed
        avg_size = sum(e.qty for e in last_streak) / len(last_streak)
        if avg_size <= 0 or last.qty < avg_size * self.size_multiplier:
            return None
        sev = round(min(1.0, (last.qty / avg_size - 1.0) / 2.0 + 0.5), 3)
        return BiasEvent(
            bias=self.name, severity=sev, detected_at_ist=now_ist,
            citation=self.citation,
            evidence=(f"sized up to {last.qty} after {self.streak_min} "
                      f"consecutive wins (avg streak size {avg_size:.0f})"),
            extras={"streak_avg_qty": round(avg_size, 1),
                    "new_qty": last.qty},
        )


class DispositionEffectDetector:
    """Shefrin & Statman (1985) "The Disposition to Sell Winners Too
    Early and Ride Losers Too Long" — measures the asymmetry in
    holding-time / trail-arming between winning and losing positions
    across the session.
    """

    name = "DISPOSITION"
    citation = "Shefrin & Statman (1985)"

    def __init__(self, asymmetry_threshold: float = 1.6) -> None:
        self.asymmetry_threshold = asymmetry_threshold

    def scan(self, completed: List[TradeAction],
             now_ist: str) -> Optional[BiasEvent]:
        wins, losses = [], []
        for a in completed:
            if a.action != "EXIT":
                continue
            hold = a.extras.get("hold_minutes")
            if hold is None:
                continue
            (wins if a.pnl > 0 else losses).append(float(hold))
        if len(wins) < 2 or len(losses) < 2:
            return None
        mean_win = sum(wins) / len(wins)
        mean_loss = sum(losses) / len(losses)
        if mean_win <= 0:
            return None
        ratio = mean_loss / mean_win
        if ratio < self.asymmetry_threshold:
            return None
        sev = round(min(1.0, (ratio - 1) / 3.0), 3)
        return BiasEvent(
            bias=self.name, severity=sev, detected_at_ist=now_ist,
            citation=self.citation,
            evidence=(f"avg loss held {mean_loss:.1f} min vs avg winner "
                      f"{mean_win:.1f} min (ratio {ratio:.2f})"),
            extras={"hold_ratio": round(ratio, 2),
                    "mean_loss_min": round(mean_loss, 1),
                    "mean_win_min": round(mean_win, 1)},
        )


class AnchoringDetector:
    """Tversky & Kahneman (1974) "Judgment under Uncertainty:
    Heuristics and Biases": stops set at round numbers near the entry
    price are anchored, not analytical. Detects trail-arming actions
    where the cushion is suspiciously round.
    """

    name = "ANCHORING"
    citation = "Tversky & Kahneman (1974)"

    def __init__(self, round_units: tuple = (50.0, 100.0)) -> None:
        self.round_units = round_units

    def scan(self, recent: List[TradeAction],
             now_ist: str) -> Optional[BiasEvent]:
        arms = [a for a in recent if a.action == "TRAIL_ARMED"]
        if not arms:
            return None
        last = arms[-1]
        cushion = float(last.extras.get("cushion_rupees", 0.0))
        if cushion <= 0:
            return None
        round_hit = next((u for u in self.round_units
                          if abs(cushion - u) < 0.01), None)
        if round_hit is None:
            return None
        # Only flag if the cushion appears to ignore the trade's actual
        # premium scale (e.g. ₹100 cushion on a ₹15 premium = anchored).
        premium = float(last.extras.get("premium", 0)) or 1.0
        if cushion / premium < 0.3:
            return None
        sev = round(min(1.0, cushion / max(premium, 1.0) * 0.3), 3)
        return BiasEvent(
            bias=self.name, severity=sev, detected_at_ist=now_ist,
            citation=self.citation,
            evidence=(f"trail cushion ₹{cushion:.0f} on a ₹{premium:.0f} "
                      f"premium — round-number anchor, not ATR-derived"),
            extras={"cushion_rupees": cushion, "premium": premium},
        )


class RiskBudgetDetector:
    """Tharp (2007) "Trade Your Way to Financial Freedom": every
    session has a finite R-budget. When realized loss + worst-case
    open exposure exceeds the budget, additional risk-taking is
    statistically unjustified. Fires when day P&L is below a hard
    -₹X threshold AND open exposure widens the drawdown band further.
    """

    name = "RISK_BUDGET"
    citation = "Tharp (2007)"

    def __init__(self, hard_loss_rupees: float = 3000.0) -> None:
        self.hard_loss = hard_loss_rupees

    def scan(self, day_pnl: float, open_positions: int,
             now_ist: str) -> Optional[BiasEvent]:
        if day_pnl >= -self.hard_loss:
            return None
        sev = round(min(1.0, abs(day_pnl) / (self.hard_loss * 2.0)), 3)
        return BiasEvent(
            bias=self.name, severity=sev, detected_at_ist=now_ist,
            citation=self.citation,
            evidence=(f"day P&L ₹{day_pnl:,.0f} below hard budget "
                      f"₹-{self.hard_loss:,.0f} "
                      f"with {open_positions} position(s) still open"),
            extras={"day_pnl": day_pnl, "hard_loss": self.hard_loss,
                    "open_positions": open_positions},
        )


# ---------------------------------------------------------------------------
# TiltIndex — composite gauge
# ---------------------------------------------------------------------------

class TiltIndex:
    """0-100 EWMA composite of detected bias severities. Bands:

      0..29   GREEN    — operator's frontal lobe is in charge
      30..59  AMBER    — system 1 starting to take over (Kahneman 2011)
      60..84  RED      — multiple biases firing concurrently; spine
                          should refuse new EXECUTION upgrades
      85..100 CIRCUIT  — emotional circuit-breaker; kill-switch level

    The constants are chosen so that one severity-0.7 event raises tilt
    by ~20 points; a quiet stretch decays it back at ~5 points / minute.
    """

    def __init__(self, decay_per_min: float = 5.0,
                 spike_gain: float = 35.0) -> None:
        self.decay_per_min = decay_per_min
        self.spike_gain = spike_gain
        self._value = 0.0
        self._last_ts: Optional[datetime] = None
        self._biases_active: Deque[str] = deque(maxlen=8)

    def _now(self) -> datetime:
        return datetime.now(IST)

    def _decay(self) -> None:
        now = self._now()
        if self._last_ts is None:
            self._last_ts = now; return
        dt_min = (now - self._last_ts).total_seconds() / 60.0
        self._value = max(0.0, self._value - dt_min * self.decay_per_min)
        self._last_ts = now

    def observe(self, ev: BiasEvent) -> float:
        """Returns the new tilt index after absorbing the event."""
        self._decay()
        # quadratic-ish weighting so high-severity events count more
        spike = self.spike_gain * (0.4 + ev.severity * ev.severity * 0.6)
        self._value = min(100.0, self._value + spike)
        if ev.bias not in self._biases_active:
            self._biases_active.append(ev.bias)
        return self._value

    def peek(self) -> float:
        self._decay()
        return self._value

    @staticmethod
    def band_for(value: float) -> str:
        if value < 30: return "GREEN"
        if value < 60: return "AMBER"
        if value < 85: return "RED"
        return "CIRCUIT"

    def biases_active(self) -> List[str]:
        return list(self._biases_active)


# ---------------------------------------------------------------------------
# PsychologyEngine — the orchestrator
# ---------------------------------------------------------------------------

class PsychologyEngine:
    """Drives the detectors on a sliding action window, updates the
    TiltIndex, publishes a TRUSTED `psychology` ModelSignal whenever a
    new bias fires, and accumulates the daily MindReport.

    Thread-safe; reads + writes happen under a single lock so the live
    cockpit can poll concurrently with the tick loop.
    """

    BAND_LIMIT_FOR_EXECUTION = "RED"   # at RED or worse, refuse new EXEC

    def __init__(self, publisher: Optional[LivePublisher] = None,
                 session: str = "",
                 journal_path: Optional[Path] = None) -> None:
        self.publisher = publisher
        self.session = session
        self.journal_path = Path(journal_path) if journal_path else None
        self.detectors_action = [
            RevengeTradeDetector(),
            FomoEntryDetector(),
            HotHandDetector(),
            DispositionEffectDetector(),
            AnchoringDetector(),
        ]
        self.budget_detector = RiskBudgetDetector()
        self.tilt = TiltIndex()
        self.intention = IntentionContract()
        self.actions: Deque[TradeAction] = deque(maxlen=200)
        self.spot_hist: Deque[tuple] = deque(maxlen=120)   # (ts_ist, spot)
        self.recent_biases: Deque[BiasEvent] = deque(maxlen=12)
        self._lock = threading.Lock()
        # accumulators for the MindReport
        self._tilt_samples: List[float] = []
        self._bias_counts: Counter = Counter()
        self._published_keys: set = set()  # dedupe by (bias, evidence)

    # -- recording ---------------------------------------------------------

    def record(self, action: TradeAction) -> None:
        with self._lock:
            self.actions.append(action)

    def record_spot(self, ts_ist: str, spot: float) -> None:
        with self._lock:
            self.spot_hist.append((ts_ist, spot))

    # -- intention contract -----------------------------------------------

    def set_intention(self, max_day_loss_rupees: float,
                      target_day_profit_rupees: float = 0.0,
                      max_positions: int = 0,
                      no_trade_after_ist: str = "15:15",
                      notes: str = "") -> IntentionContract:
        with self._lock:
            self.intention = IntentionContract(
                set_at_ist=now_ist_hms(),
                max_day_loss_rupees=float(max_day_loss_rupees),
                target_day_profit_rupees=float(target_day_profit_rupees),
                max_positions=int(max_positions),
                no_trade_after_ist=no_trade_after_ist,
                notes=notes,
            )
            return self.intention

    # -- detection cycle --------------------------------------------------

    def cycle(self, day_pnl: float = 0.0,
              open_positions: int = 0) -> List[BiasEvent]:
        """Run every detector once; publish any new bias events; return
        the events fired this cycle (may be empty)."""
        now = now_ist_hms()
        events: List[BiasEvent] = []
        with self._lock:
            recent = list(self.actions)
            spot_hist = list(self.spot_hist)

            for det in self.detectors_action:
                if isinstance(det, FomoEntryDetector):
                    ev = det.scan(recent, spot_hist, now)
                else:
                    ev = det.scan(recent, now)
                if ev is None:
                    continue
                key = (ev.bias, ev.evidence)
                if key in self._published_keys:
                    continue
                self._published_keys.add(key)
                events.append(ev)

            ev = self.budget_detector.scan(day_pnl, open_positions, now)
            if ev is not None:
                key = (ev.bias, ev.evidence)
                if key not in self._published_keys:
                    self._published_keys.add(key)
                    events.append(ev)

            # intention violation also counts as a bias event
            v = self.intention.check(day_pnl, open_positions, now)
            if v and not self.intention.violated:
                self.intention.violated = True
                self.intention.violation_reason = v
                events.append(BiasEvent(
                    bias="INTENTION_VIOLATED", severity=1.0,
                    detected_at_ist=now,
                    citation="Elster (2000) 'Ulysses Unbound'",
                    evidence=f"intention contract breached: {v}",
                    extras={"reason": v},
                ))

            for ev in events:
                self.tilt.observe(ev)
                self.recent_biases.appendleft(ev)
                self._bias_counts[ev.bias] += 1

            current_tilt = self.tilt.peek()
            self._tilt_samples.append(current_tilt)

        # publish outside the lock to avoid re-entrance into shadow ledger
        if self.publisher is not None:
            for ev in events:
                self.publisher.publish(ModelSignal(
                    ts_ist=ev.detected_at_ist, asset="OPERATOR",
                    model="psychology",
                    signal=f"{ev.bias}: {ev.evidence}",
                    confidence=ev.severity,
                    trust_tier="TRUSTED",
                    reason_codes=[ev.bias, ev.citation],
                    extras={"citation": ev.citation, **ev.extras},
                ))
        return events

    # -- read sides -------------------------------------------------------

    def state(self) -> TiltState:
        with self._lock:
            value = self.tilt.peek()
            return TiltState(
                index=value, band=TiltIndex.band_for(value),
                biases_active=self.tilt.biases_active(),
                recent=list(self.recent_biases),
                last_update_ist=now_ist_hms(),
            )

    def should_block_execution(self) -> bool:
        """The spine asks this before promoting any non-hard-wired
        signal to EXECUTION. RED and CIRCUIT bands block; AMBER warns."""
        value = self.tilt.peek()
        band = TiltIndex.band_for(value)
        return band in ("RED", "CIRCUIT")

    # -- end-of-session ---------------------------------------------------

    def mind_report(self) -> MindReport:
        with self._lock:
            tilt_avg = (sum(self._tilt_samples) / len(self._tilt_samples)
                        if self._tilt_samples else 0.0)
            tilt_peak = max(self._tilt_samples) if self._tilt_samples else 0.0
            most = self._bias_counts.most_common(1)
            citations = sorted({
                d.citation for d in self.detectors_action
                if self._bias_counts.get(d.name, 0) > 0
            } | ({self.budget_detector.citation}
                  if self._bias_counts.get(self.budget_detector.name, 0) > 0
                  else set()))
            report = MindReport(
                session=self.session,
                tilt_avg=round(tilt_avg, 1),
                tilt_peak=round(tilt_peak, 1),
                biases_by_type=dict(self._bias_counts),
                most_common_bias=most[0][0] if most else None,
                revenge_trades=self._bias_counts.get("REVENGE", 0),
                fomo_entries=self._bias_counts.get("FOMO", 0),
                disciplined_trails=sum(1 for a in self.actions
                                       if a.action == "TRAIL_ARMED"),
                intention_was_set=self.intention.is_set,
                intention_violated=self.intention.violated,
                intention_violation_reason=self.intention.violation_reason,
                citations=citations,
                notable_quotes=_NOTABLE_QUOTES,
            )
        if self.journal_path is not None:
            try:
                self.journal_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.journal_path, "a") as f:
                    f.write(json.dumps(report.to_row()) + "\n")
            except Exception:
                pass
        return report


_NOTABLE_QUOTES = [
    "“The investor's chief problem — and even his worst enemy — is likely to be himself.” — Benjamin Graham",
    "“Trade in size that lets you sleep at night.” — Paul Tudor Jones",
    "“Risk comes from not knowing what you are doing.” — Warren Buffett",
    "“The market can stay irrational longer than you can stay solvent.” — attr. J.M. Keynes",
]


declare(IOSpec(
    module="sentinel.psychology",
    purpose="behavioral engine — six citation-bearing bias detectors "
            "(REVENGE/FOMO/HOT_HAND/DISPOSITION/ANCHORING/RISK_BUDGET) + "
            "TiltIndex composite gauge + IntentionContract (Ulysses "
            "pattern) + MindReport. Fires TRUSTED psychology signals to "
            "the live bus; spine can refuse EXECUTION while in RED tilt",
    inputs=["TradeAction stream (ENTRY/EXIT/TRAIL_ARMED with ts, qty, "
            "pnl, premium, spot)",
            "spot history snapshots",
            "day P&L + open positions count"],
    outputs=["BiasEvent stream",
             "TiltState (live)",
             "MindReport (end of session)",
             "TRUSTED psychology ModelSignal to LivePublisher"],
    consumes_from=["sentinel.shadow_ledger", "sentinel.trails",
                   "sentinel.kite_client (orders log)"],
    produces_for=["sentinel.live_publisher", "sentinel.server (cockpit)",
                  "sentinel.shadow_ledger (mind report)"],
    tier="TRUSTED",
))
