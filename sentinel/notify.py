"""Notification filter — convert audit events into operator-actionable
alerts.

Codex's transport (SendGrid / FCM / SES / Pushover / SMS) handles
delivery. This module is the FILTER + the message formatter that
decides:

  * which audit events deserve a real-time alert
  * which channel(s) each event goes to (email / push / both)
  * the rendered message (subject + body, plain + HTML)

The filter is config-driven. Default rules are sensible for a single-
operator launch; SaaS customers will override via per-user prefs in
Supabase later.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .audit_log import register_audit_sink
from .io_decl import IOSpec, declare

LOG = logging.getLogger("sentinel.notify")


@dataclass
class Alert:
    channel: str                       # "email" | "push" | "sms"
    subject: str
    body_plain: str
    body_html: str = ""
    severity: str = "INFO"
    user_id: str = ""
    event: str = ""                    # source audit event name

    def to_row(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "subject": self.subject,
            "body_plain": self.body_plain,
            "body_html": self.body_html,
            "severity": self.severity,
            "user_id": self.user_id,
            "event": self.event,
        }


# ─────────────────────────────────────────────────────────────────
# Rendering — one function per "alert-worthy" audit event
# ─────────────────────────────────────────────────────────────────

def _render_intention_violated(rec: Dict[str, Any]) -> List[Alert]:
    p = rec.get("payload", {})
    reason = p.get("reason") or p.get("evidence") or "limit breached"
    sev = "ERROR"
    return [
        Alert(channel="push", severity=sev, event=rec["event"],
              user_id=rec.get("user_id", ""),
              subject="⚠ Sentinel: intention contract violated",
              body_plain=f"You committed pre-market to a max loss; "
                          f"that line was just crossed.\n\n{reason}\n\n"
                          f"Spine is refusing new non-exit orders until "
                          f"tomorrow. Flatten what you have and stop.",
              body_html=("<h2>⚠ Intention contract violated</h2>"
                          f"<p><b>Reason:</b> {reason}</p>"
                          "<p>Spine refusing non-exit orders until tomorrow. "
                          "Flatten what you have and stop trading today.</p>"))
    ]


def _render_tilt_red(rec: Dict[str, Any]) -> List[Alert]:
    p = rec.get("payload", {})
    band = p.get("tilt_band", "RED")
    return [
        Alert(channel="push", severity="WARN", event=rec["event"],
              user_id=rec.get("user_id", ""),
              subject=f"Sentinel: tilt {band}",
              body_plain=(f"Your tilt index crossed into the {band} band. "
                          f"The spine is refusing non-exit orders. "
                          f"Step away from the screen for 20 minutes."),
              body_html=(f"<h2>Tilt {band}</h2>"
                          "<p>Multiple bias detectors firing concurrently. "
                          "Take a break. Trail your winners, exit your "
                          "losers. Do not open new positions.</p>"))
    ]


def _render_order_blocked(rec: Dict[str, Any]) -> List[Alert]:
    p = rec.get("payload", {})
    reason = p.get("reason", "")
    sym = p.get("symbol", "?")
    return [
        Alert(channel="push", severity="WARN", event=rec["event"],
              user_id=rec.get("user_id", ""),
              subject="Sentinel: order blocked by spine",
              body_plain=(f"The spine blocked an order on {sym} "
                          f"because: {reason}. Your committed limits "
                          f"are working — no action needed."))
    ]


def _render_mind_report_ready(rec: Dict[str, Any]) -> List[Alert]:
    p = rec.get("payload", {})
    tilt_peak = p.get("tilt_peak", "?")
    most_common = p.get("most_common_bias", "—")
    return [
        Alert(channel="email", severity="INFO", event=rec["event"],
              user_id=rec.get("user_id", ""),
              subject="Sentinel: end-of-session Mind Report",
              body_plain=(f"Today's tilt peak: {tilt_peak}.\n"
                          f"Most common bias: {most_common}.\n\n"
                          "Open the cockpit's Mind panel for the full "
                          "reflection with citations."))
    ]


_RENDERERS: Dict[str, Callable[[Dict[str, Any]], List[Alert]]] = {
    "intention_violated": _render_intention_violated,
    "tilt_red": _render_tilt_red,
    "order_blocked": _render_order_blocked,
    "mind_report_ready": _render_mind_report_ready,
}


# ─────────────────────────────────────────────────────────────────
# Notifier — registers an audit sink that fans out to transports
# ─────────────────────────────────────────────────────────────────

class Notifier:
    """Subscribes to the audit log, formats actionable events into
    Alerts, dispatches via Codex's transports."""

    def __init__(self) -> None:
        self._transports: List[Callable[[Alert], None]] = []
        self._delivered: List[Alert] = []           # for tests + ops view

    def register_transport(self, fn: Callable[[Alert], None]) -> None:
        """Codex's email/SMS/push delivery wires here:
            notifier.register_transport(send_via_sendgrid)
            notifier.register_transport(send_via_fcm)"""
        self._transports.append(fn)

    def clear_transports(self) -> None:
        self._transports.clear()

    def delivered(self) -> List[Alert]:
        return list(self._delivered)

    def handle_audit(self, rec: Dict[str, Any]) -> None:
        """Called for every audit record. The sink filters non-
        actionable events."""
        renderer = _RENDERERS.get(rec.get("event"))
        if renderer is None:
            return
        try:
            alerts = renderer(rec)
        except Exception as exc:
            LOG.warning("notify: renderer failed for %s: %s",
                         rec.get("event"), exc)
            return
        for alert in alerts:
            self._delivered.append(alert)
            for transport in self._transports:
                try:
                    transport(alert)
                except Exception as exc:
                    LOG.warning("notify: transport failed: %s", exc)

    def attach_to_audit(self) -> None:
        """Convenience: wire ``self.handle_audit`` as an audit sink."""
        register_audit_sink(self.handle_audit)


# Singleton instance (Codex's transports register here at boot)
GLOBAL_NOTIFIER = Notifier()


declare(IOSpec(
    module="sentinel.notify",
    purpose="audit-event-to-actionable-alert filter — converts the "
            "right audit events into Alerts (subject + plain + HTML) "
            "and dispatches via Codex's registered transports (SendGrid "
            "/ FCM / SES / Pushover). Filters non-actionable events so "
            "the customer's inbox doesn't get spammed",
    inputs=["audit records from sentinel.audit_log"],
    outputs=["Alert per actionable event",
             "dispatched via registered transports"],
    consumes_from=["sentinel.audit_log"],
    produces_for=["Codex's email/push delivery layer"],
    tier="TRUSTED",
))
