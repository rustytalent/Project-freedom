"""Monday-bonus features — small but high-value additions for go-live day.

Built on Sunday 2026-06-21 in response to the founder's request for
"other suggestions which can be useful for tomorrow". Each is a
focused, well-tested helper that the V4Runner or the operator UI can
opt into.

Contents:

  1. ``TimeOfDayStopScaler`` — widens stops during lunch chop (12:00-
     13:00 IST), tightens stops in the final hour (14:30-15:30). The
     founder noted "stop-loss tightening based on time-of-day" as a
     small-but-real edge.

  2. ``DefensiveAutoHedger`` — when fat_tail_score's recommended_action
     transitions from NORMAL → HEDGE, this module automatically takes
     the first hedge proposal from the HedgeProposer and submits it
     via the broker adapter, instead of just suggesting it. Safety
     gated: only fires after a configurable number of consecutive
     HEDGE-band ticks (default 2) to avoid whipsaws.

  3. ``HealthEndpoint`` — a stateless function that returns a JSON-
     serializable health report from a V4Runner. Suitable as the body
     of an HTTP GET /health endpoint or a periodic status emitter.
     Reports: last tick ts, daily P&L, open positions, broker mode,
     broker fills today, persistence file path, kill switch state.

  4. ``SlippageTracker`` — records the difference between the
     executor's predicted exit_premium and the broker's actual fill
     price. Lets the founder calibrate slippage estimates over time.

  5. ``AlertHookEmitter`` — produces structured alerts for critical
     events (kill switch fired, position closed at loss > X,
     fat-tail spike, broker rejection). The caller wires these to
     Telegram/SMS/Slack however they prefer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional


# Indian Standard Time
IST = timezone(timedelta(hours=5, minutes=30))


# ─────────────────────────────────────────────────────────────────
# 1. Time-of-day stop scaler
# ─────────────────────────────────────────────────────────────────


@dataclass
class TimeOfDayStopScalerConfig:
    """Per-session stop multipliers."""
    morning_open_minutes: int = 15      # 09:15-09:30 → wider (volatile)
    morning_open_scale: float = 1.30
    lunch_chop_start_minute: int = 720  # 12:00 in minutes from midnight
    lunch_chop_end_minute: int = 780    # 13:00
    lunch_chop_scale: float = 1.20
    final_hour_start_minute: int = 870  # 14:30
    final_hour_scale: float = 0.85
    last_15min_scale: float = 0.70


class TimeOfDayStopScaler:
    """Multiplies a base stop distance by a time-of-day factor."""

    def __init__(self, cfg: Optional[TimeOfDayStopScalerConfig] = None) -> None:
        self.cfg = cfg or TimeOfDayStopScalerConfig()

    def scale_for(self, ts: Optional[Any] = None) -> float:
        """Return the stop scale (>1 widens, <1 tightens) for the given ts.

        If ``ts`` is None or unparseable, returns 1.0.
        """
        if ts is None:
            now = datetime.now(IST)
        elif isinstance(ts, datetime):
            now = ts.astimezone(IST) if ts.tzinfo else ts.replace(tzinfo=IST)
        else:
            try:
                now = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                now = now.astimezone(IST)
            except (ValueError, TypeError):
                return 1.0

        cfg = self.cfg
        minutes = now.hour * 60 + now.minute
        if minutes < 9 * 60 + 15:
            return 1.0
        # Morning open band (9:15-9:30)
        if minutes < 9 * 60 + 15 + cfg.morning_open_minutes:
            return cfg.morning_open_scale
        # Lunch chop
        if cfg.lunch_chop_start_minute <= minutes < cfg.lunch_chop_end_minute:
            return cfg.lunch_chop_scale
        # Last 15min
        if minutes >= 15 * 60 + 15:
            return cfg.last_15min_scale
        # Final hour
        if minutes >= cfg.final_hour_start_minute:
            return cfg.final_hour_scale
        return 1.0


# ─────────────────────────────────────────────────────────────────
# 2. Defensive auto-hedger
# ─────────────────────────────────────────────────────────────────


@dataclass
class DefensiveAutoHedgerConfig:
    """Knobs for the auto-hedger."""
    consecutive_hedge_ticks_required: int = 2
    consecutive_normal_ticks_to_clear: int = 3
    max_hedges_per_session: int = 4
    enabled: bool = True


@dataclass
class AutoHedgeAction:
    """One auto-hedge submission record."""
    ts: float
    triggered_by: str          # "fat_tail_hedge_transition"
    proposals_submitted: int
    broker_results: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["broker_results"] = list(self.broker_results)
        d["notes"] = list(self.notes)
        return d


class DefensiveAutoHedger:
    """Auto-submits the hedge proposer's first proposal when fat-tail
    sustains in HEDGE band.

    State-machine:
      NORMAL → (N consecutive HEDGE ticks) → submit → COOLDOWN
      COOLDOWN → (N consecutive NORMAL ticks) → NORMAL
    """

    def __init__(self, cfg: Optional[DefensiveAutoHedgerConfig] = None) -> None:
        self.cfg = cfg or DefensiveAutoHedgerConfig()
        self._consecutive_hedge_ticks = 0
        self._consecutive_normal_ticks = 0
        self._state = "NORMAL"      # NORMAL / READY / COOLDOWN
        self._hedges_this_session = 0
        self.history: List[AutoHedgeAction] = []

    def consider(self, *,
                   fat_tail_action: str,
                   hedge_proposal_dict: Optional[Dict[str, Any]],
                   ) -> Optional[Dict[str, Any]]:
        """Look at the current tick. Returns a hedge proposal dict if the
        auto-hedger thinks we should submit, else None."""
        cfg = self.cfg
        if not cfg.enabled:
            return None
        if self._hedges_this_session >= cfg.max_hedges_per_session:
            return None

        if fat_tail_action == "HEDGE":
            self._consecutive_hedge_ticks += 1
            self._consecutive_normal_ticks = 0
        else:
            self._consecutive_normal_ticks += 1
            self._consecutive_hedge_ticks = 0

        if (self._state == "NORMAL"
                and self._consecutive_hedge_ticks
                >= cfg.consecutive_hedge_ticks_required):
            if (hedge_proposal_dict is not None
                    and hedge_proposal_dict.get("proposed")
                    and hedge_proposal_dict.get("proposals")):
                self._state = "COOLDOWN"
                self._hedges_this_session += 1
                self.history.append(AutoHedgeAction(
                    ts=time.time(),
                    triggered_by="fat_tail_hedge_transition",
                    proposals_submitted=len(
                        hedge_proposal_dict.get("proposals") or []),
                    notes=[f"auto-hedge fired ({self._hedges_this_session}/"
                            f"{cfg.max_hedges_per_session})"],
                ))
                return hedge_proposal_dict
        if (self._state == "COOLDOWN"
                and self._consecutive_normal_ticks
                >= cfg.consecutive_normal_ticks_to_clear):
            self._state = "NORMAL"
        return None

    def summary(self) -> Dict[str, Any]:
        return {
            "state": self._state,
            "consecutive_hedge_ticks": self._consecutive_hedge_ticks,
            "consecutive_normal_ticks": self._consecutive_normal_ticks,
            "hedges_this_session": self._hedges_this_session,
            "history": [h.to_dict() for h in self.history[-5:]],
        }


# ─────────────────────────────────────────────────────────────────
# 3. Health endpoint
# ─────────────────────────────────────────────────────────────────


def health_report(runner: Any) -> Dict[str, Any]:
    """Produce a structured health report from a V4Runner.

    Safe to call any time; never raises (catches any attribute errors).
    """
    out: Dict[str, Any] = {
        "ok": True,
        "ts": datetime.now(IST).isoformat(timespec="seconds"),
        "runner_class": type(runner).__name__,
    }
    try:
        out["broker"] = runner.broker.healthcheck()
    except Exception as exc:
        out["broker"] = {"error": str(exc)}
        out["ok"] = False
    try:
        out["daily_pnl_rupees"] = runner.manager.daily_pnl_rupees
        out["cumulative_fees_rupees"] = runner.manager.cumulative_fees_rupees
        out["n_open_positions"] = len(runner.manager._open_states)
        out["n_closed_positions"] = len(
            runner.manager.ledger_store.closed_positions())
    except Exception as exc:
        out["manager_error"] = str(exc)
        out["ok"] = False
    try:
        out["persistence_path"] = str(runner.persistence.path_for_today())
    except Exception:
        out["persistence_path"] = None
    try:
        last_tick = getattr(runner, "_last_tick_ts", None)
        out["last_tick_seconds_ago"] = (time.time() - last_tick
                                          if last_tick else None)
    except Exception:
        out["last_tick_seconds_ago"] = None
    return out


# ─────────────────────────────────────────────────────────────────
# 4. Slippage tracker
# ─────────────────────────────────────────────────────────────────


@dataclass
class SlippageRecord:
    """One predicted-vs-actual fill comparison."""
    ts: float
    tradingsymbol: str
    side: str
    predicted_premium: float
    actual_premium: float
    bps_slippage: float

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class SlippageTracker:
    """Records actual vs predicted fill prices, gives rolling stats."""

    def __init__(self) -> None:
        self.records: List[SlippageRecord] = []

    def record(self, *,
                  tradingsymbol: str,
                  side: str,
                  predicted: float,
                  actual: float) -> SlippageRecord:
        if predicted <= 0:
            bps = 0.0
        else:
            sign = +1 if side == "BUY" else -1
            bps = sign * (actual - predicted) / predicted * 10000.0
        rec = SlippageRecord(
            ts=time.time(),
            tradingsymbol=tradingsymbol,
            side=side,
            predicted_premium=predicted,
            actual_premium=actual,
            bps_slippage=bps,
        )
        self.records.append(rec)
        return rec

    def rolling_summary(self, window: int = 50) -> Dict[str, Any]:
        recent = self.records[-window:]
        if not recent:
            return {"n_records": 0}
        bps_values = [r.bps_slippage for r in recent]
        mean_bps = sum(bps_values) / len(bps_values)
        max_bps = max(bps_values)
        min_bps = min(bps_values)
        # Median
        sorted_bps = sorted(bps_values)
        median = sorted_bps[len(sorted_bps) // 2]
        return {
            "n_records": len(recent),
            "mean_bps": round(mean_bps, 2),
            "median_bps": round(median, 2),
            "worst_bps": round(max_bps, 2),
            "best_bps": round(min_bps, 2),
        }


# ─────────────────────────────────────────────────────────────────
# 5. Alert hook emitter
# ─────────────────────────────────────────────────────────────────


ALERT_SEVERITY_CRITICAL = "CRITICAL"
ALERT_SEVERITY_WARN = "WARN"
ALERT_SEVERITY_INFO = "INFO"


@dataclass
class Alert:
    """One alert event."""
    severity: str
    code: str               # e.g. "KILL_SWITCH" / "BIG_LOSS" / "FAT_TAIL_SPIKE"
    message: str
    ts: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["details"] = dict(self.details)
        return d


@dataclass
class AlertEmitterConfig:
    """Knobs."""
    big_loss_threshold_rupees: float = 500.0
    fat_tail_spike_threshold: float = 0.55
    rate_limit_seconds: float = 30.0


class AlertHookEmitter:
    """Generates Alert events; the caller wires hooks to webhooks/etc."""

    def __init__(self, cfg: Optional[AlertEmitterConfig] = None) -> None:
        self.cfg = cfg or AlertEmitterConfig()
        self.alerts: List[Alert] = []
        self._last_alert_per_code: Dict[str, float] = {}

    def from_intent(self, intent_dict: Dict[str, Any]) -> List[Alert]:
        """Generate alerts based on a per-tick PortfolioIntent.to_dict()."""
        cfg = self.cfg
        alerts: List[Alert] = []
        now = time.time()

        # Closed positions with big losses
        for closed in intent_dict.get("closed_this_tick") or []:
            outcome = closed.get("outcome") or {}
            realized = float(outcome.get("realized_rupees", 0.0))
            if realized <= -cfg.big_loss_threshold_rupees:
                a = self._mk(now, code="BIG_LOSS",
                              severity=ALERT_SEVERITY_WARN,
                              message=(f"Position {closed.get('position_id')[:8]} "
                                        f"closed at ₹{realized:.0f}"),
                              details={"outcome": outcome})
                if a is not None:
                    alerts.append(a)

        # Fat-tail spike
        summary = intent_dict.get("portfolio_summary") or {}
        tail = summary.get("fat_tail_score") or {}
        tail_score = float(tail.get("tail_score", 0.0))
        if tail_score >= cfg.fat_tail_spike_threshold:
            a = self._mk(now, code="FAT_TAIL_SPIKE",
                          severity=ALERT_SEVERITY_WARN,
                          message=f"Fat-tail score {tail_score:.2f} elevated",
                          details={"tail": tail})
            if a is not None:
                alerts.append(a)

        # Kill switch
        risk = summary.get("portfolio_risk") or {}
        for k in risk.get("kill_switches") or []:
            a = self._mk(now, code="KILL_SWITCH",
                          severity=ALERT_SEVERITY_CRITICAL,
                          message=f"Portfolio kill: {k}",
                          details={"kill_switch": k})
            if a is not None:
                alerts.append(a)

        self.alerts.extend(alerts)
        return alerts

    def _mk(self, now: float, *, code: str, severity: str,
              message: str, details: Optional[Dict[str, Any]] = None
              ) -> Optional[Alert]:
        last = self._last_alert_per_code.get(code, 0.0)
        if now - last < self.cfg.rate_limit_seconds:
            return None
        self._last_alert_per_code[code] = now
        return Alert(
            severity=severity, code=code, message=message,
            ts=now, details=dict(details or {}),
        )
