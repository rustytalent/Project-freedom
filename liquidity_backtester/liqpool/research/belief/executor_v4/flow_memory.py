"""Flow-level event memory — beyond bars.

The founder's exact concern: *"in the past 60 minutes, you have the data.
What was the pure intention?"* — sequences of micro-events tell stories that
bar-aggregated data hides.

Captures EVENT-LEVEL records (one per tick, not one per bar). Each event
carries the snapshot's key signals plus computed micro-features. Supports
SEQUENCE QUERIES that the manipulation-pattern layer (Sprint 3) and the
scenario web (Sprint 2) will use to detect things like the founder's
example: "surprise up → revert below start → stabilize → up again."
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class FlowEvent:
    """One captured event in the flow memory."""
    ts: Any
    bar_index: int
    spot: float
    spot_pct_change: float            # bps from prior event
    net_intent_z: float
    ce_signed_z: float
    pe_signed_z: float
    ce_dispersion: float
    pe_dispersion: float
    iv_state: str
    battlefield_verdict: str
    thesis_state: str
    winding_zone: str
    decision_action: str
    bull_state: int
    bear_state: int
    sweep_state: int
    bull_thesis_score: float
    bear_thesis_score: float
    avg_friendliness: float
    clean_mark_fraction: float
    abnormal_slot_fraction: float
    # Per-side acceptance fractions (snapshot-level).
    ce_defended_fraction: float
    pe_defended_fraction: float
    ce_rejected_fraction: float
    pe_rejected_fraction: float
    # Derived micro-features computed during push.
    surprise_score: float = 0.0       # 0..1, how unexpected this event is
    flow_signature: str = ""          # short label of the dominant micro-event
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["notes"] = list(self.notes)
        return d


@dataclass
class FlowMemoryConfig:
    cap: int = 3600       # 60 minutes at ~1 event/sec = 3600 events
    surprise_velocity_threshold: float = 1.5  # |net_intent velocity| above this is surprising
    surprise_dispersion_jump: float = 0.3     # dispersion jump above this is surprising


class FlowMemory:
    """Rolling event-level memory with sequence queries.

    Push one event per tick via observe(). Then:
      * recent(n) — last n events
      * sequence_matches([predicates]) — find consecutive subsequences that match
      * find_pattern("surprise_up_then_revert") — pre-baked patterns
      * within_seconds(seconds) — recent events within a time horizon
    """

    def __init__(self, cfg: Optional[FlowMemoryConfig] = None) -> None:
        self.cfg = cfg or FlowMemoryConfig()
        self.events: Deque[FlowEvent] = deque(maxlen=self.cfg.cap)

    def reset(self) -> None:
        self.events.clear()

    def observe(self, snapshot: Dict[str, Any]) -> FlowEvent:
        """Ingest a snapshot, compute micro-features, store event."""
        cfg = self.cfg
        ts = snapshot.get("ts")
        spot = _num(snapshot.get("spot"))
        bar = int(_num(snapshot.get("bars_seen"), 0))

        prior_spot = self.events[-1].spot if self.events else spot
        spot_pct = ((spot - prior_spot) / prior_spot * 10000.0
                     if prior_spot > 0 else 0.0)  # bps

        thesis = _map(snapshot.get("thesis"))
        iv = _map(snapshot.get("iv_state"))
        bf = _map(snapshot.get("battlefield"))
        decision = _map(snapshot.get("decision"))
        ce_rail = _map(bf.get("ce_rail"))
        pe_rail = _map(bf.get("pe_rail"))
        winding = _map(snapshot.get("winding"))

        # Net intent: prefer explicit, fall back to ce_signed - pe_signed
        net_intent_z = _num(iv.get("net_intent_z"))
        ce_signed = _num(ce_rail.get("weighted_mean_signed_z"))
        pe_signed = _num(pe_rail.get("weighted_mean_signed_z"))
        if net_intent_z == 0.0:
            net_intent_z = ce_signed - pe_signed

        slots = list(snapshot.get("slot_readings") or [])
        n_slots = len(slots)
        n_abnormal = sum(1 for s in slots if bool(_map(s).get("is_abnormal")))
        n_dirty = sum(1 for s in slots if str(_map(s).get("mark_source") or "") in ("ltp", "last_valid", "invalid"))
        friend_total = 0.0
        friend_n = 0
        for s in slots:
            f = _num(_map(s).get("friendliness"), math.nan)
            if math.isfinite(f):
                friend_total += f
                friend_n += 1
        avg_friend = (friend_total / friend_n) if friend_n else 1.0
        clean_mark_frac = 1.0 - (n_dirty / n_slots if n_slots else 0.0)
        abnormal_frac = (n_abnormal / n_slots if n_slots else 0.0)

        # Per-side acceptance fractions.
        ce_defended = ce_rejected = pe_defended = pe_rejected = 0
        ce_n = pe_n = 0
        for s in slots:
            sm = _map(s)
            otype = str(sm.get("option_type") or "")
            acc = str(sm.get("acceptance") or "normal")
            if otype == "CE":
                ce_n += 1
                if acc == "defended":
                    ce_defended += 1
                elif acc == "rejected":
                    ce_rejected += 1
            elif otype == "PE":
                pe_n += 1
                if acc == "defended":
                    pe_defended += 1
                elif acc == "rejected":
                    pe_rejected += 1

        ce_def_frac = ce_defended / ce_n if ce_n else 0.0
        ce_rej_frac = ce_rejected / ce_n if ce_n else 0.0
        pe_def_frac = pe_defended / pe_n if pe_n else 0.0
        pe_rej_frac = pe_rejected / pe_n if pe_n else 0.0

        # Surprise score — sudden change in net_intent or dispersion.
        surprise = 0.0
        signature_parts: List[str] = []
        if self.events:
            prior = self.events[-1]
            dintent = net_intent_z - prior.net_intent_z
            if abs(dintent) > cfg.surprise_velocity_threshold:
                surprise += 0.5
                signature_parts.append(
                    f"net_intent jumped {dintent:+.2f}σ"
                )
            ddisp = (_num(ce_rail.get("dispersion_score"))
                      - prior.ce_dispersion)
            if abs(ddisp) > cfg.surprise_dispersion_jump:
                surprise += 0.3
                signature_parts.append(f"CE dispersion {ddisp:+.2f}")
            if abs(spot_pct) > 5.0:   # >5bps move
                surprise += 0.2
                signature_parts.append(f"spot {spot_pct:+.1f}bps")
        surprise = min(1.0, surprise)

        flow_signature = "; ".join(signature_parts) if signature_parts else "calm"

        event = FlowEvent(
            ts=ts, bar_index=bar, spot=spot,
            spot_pct_change=round(spot_pct, 3),
            net_intent_z=round(net_intent_z, 3),
            ce_signed_z=round(ce_signed, 3),
            pe_signed_z=round(pe_signed, 3),
            ce_dispersion=round(_num(ce_rail.get("dispersion_score")), 3),
            pe_dispersion=round(_num(pe_rail.get("dispersion_score")), 3),
            iv_state=str(iv.get("state") or ""),
            battlefield_verdict=str(bf.get("verdict") or ""),
            thesis_state=str(thesis.get("composite_state") or ""),
            winding_zone=str(winding.get("zone") or "NO_WINDING"),
            decision_action=str(decision.get("action") or ""),
            bull_state=int(_num(_map(snapshot.get("bull_state")).get("state_index"), 0)),
            bear_state=int(_num(_map(snapshot.get("bear_state")).get("state_index"), 0)),
            sweep_state=int(_num(_map(snapshot.get("sweep_state")).get("state_index"), 0)),
            bull_thesis_score=round(_num(thesis.get("bull_thesis_score")), 1),
            bear_thesis_score=round(_num(thesis.get("bear_thesis_score")), 1),
            avg_friendliness=round(avg_friend, 3),
            clean_mark_fraction=round(clean_mark_frac, 3),
            abnormal_slot_fraction=round(abnormal_frac, 3),
            ce_defended_fraction=round(ce_def_frac, 3),
            pe_defended_fraction=round(pe_def_frac, 3),
            ce_rejected_fraction=round(ce_rej_frac, 3),
            pe_rejected_fraction=round(pe_rej_frac, 3),
            surprise_score=round(surprise, 3),
            flow_signature=flow_signature,
        )
        self.events.append(event)
        return event

    # ── Query API ──────────────────────────────────────────────────────

    def recent(self, n: int) -> List[FlowEvent]:
        if n <= 0:
            return []
        return list(self.events)[-n:]

    def within_seconds(self, seconds: float) -> List[FlowEvent]:
        """Recent events within a time horizon. Best-effort on ts arithmetic."""
        if not self.events:
            return []
        end = self.events[-1].ts
        out: List[FlowEvent] = []
        for ev in reversed(self.events):
            if end is None or ev.ts is None:
                out.append(ev)
                continue
            try:
                delta = (end - ev.ts).total_seconds()
            except Exception:
                out.append(ev)
                continue
            if delta <= seconds:
                out.append(ev)
            else:
                break
        return list(reversed(out))

    def find_pattern(self, name: str) -> List[Tuple[int, int]]:
        """Detect pre-baked patterns. Returns (start_idx, end_idx) tuples."""
        if name == "surprise_up_then_revert":
            return self._find_surprise_revert(direction=+1)
        if name == "surprise_down_then_revert":
            return self._find_surprise_revert(direction=-1)
        if name == "sustained_ce_defended":
            return self._find_acceptance_streak("CE", "defended")
        if name == "sustained_pe_defended":
            return self._find_acceptance_streak("PE", "defended")
        if name == "sustained_ce_rejected":
            return self._find_acceptance_streak("CE", "rejected")
        if name == "sustained_pe_rejected":
            return self._find_acceptance_streak("PE", "rejected")
        if name == "dispersion_spike":
            return self._find_dispersion_spike()
        if name == "thesis_oscillation":
            return self._find_thesis_oscillation()
        return []

    def sequence_matches(self,
                          predicates: List[Callable[[FlowEvent], bool]],
                          max_gap: int = 0,
                          ) -> List[List[int]]:
        """Find sequences where predicates fire in order, with at most
        ``max_gap`` events between consecutive predicates."""
        events = list(self.events)
        matches: List[List[int]] = []
        n = len(events)
        if not predicates or n == 0:
            return matches
        for start in range(n):
            if not predicates[0](events[start]):
                continue
            indices = [start]
            cursor = start + 1
            ok = True
            for pred in predicates[1:]:
                found = -1
                for j in range(cursor, min(n, cursor + max_gap + 1)):
                    if pred(events[j]):
                        found = j
                        break
                if found < 0:
                    ok = False
                    break
                indices.append(found)
                cursor = found + 1
            if ok:
                matches.append(indices)
        return matches

    def _find_surprise_revert(self, *, direction: int) -> List[Tuple[int, int]]:
        """Pattern: large directional surprise then revert past starting point.

        Example (direction=+1): spot jumps +N bps in event T, then within
        a few events drops below T's starting spot. The founder's exact
        ₹43/₹44 fingerprint at the flow level."""
        events = list(self.events)
        matches: List[Tuple[int, int]] = []
        for i, ev in enumerate(events):
            if ev.surprise_score < 0.5:
                continue
            if direction > 0 and ev.spot_pct_change <= 3.0:
                continue
            if direction < 0 and ev.spot_pct_change >= -3.0:
                continue
            start_spot = ev.spot - ev.spot_pct_change * ev.spot / 10000.0
            # Look ahead 6 events for revert.
            for j in range(i + 1, min(len(events), i + 7)):
                if direction > 0 and events[j].spot < start_spot:
                    matches.append((i, j))
                    break
                if direction < 0 and events[j].spot > start_spot:
                    matches.append((i, j))
                    break
        return matches

    def _find_acceptance_streak(self, side: str, kind: str,
                                 min_streak: int = 4) -> List[Tuple[int, int]]:
        """Streak of bars where >=40% of slots on the given side show the
        given acceptance."""
        events = list(self.events)
        matches: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for i, ev in enumerate(events):
            if side == "CE":
                frac = ev.ce_defended_fraction if kind == "defended" else ev.ce_rejected_fraction
            else:
                frac = ev.pe_defended_fraction if kind == "defended" else ev.pe_rejected_fraction
            if frac >= 0.40:
                if start is None:
                    start = i
            else:
                if start is not None and i - start >= min_streak:
                    matches.append((start, i - 1))
                start = None
        if start is not None and len(events) - start >= min_streak:
            matches.append((start, len(events) - 1))
        return matches

    def _find_dispersion_spike(self) -> List[Tuple[int, int]]:
        events = list(self.events)
        matches: List[Tuple[int, int]] = []
        for i, ev in enumerate(events):
            if ev.ce_dispersion >= 0.65 or ev.pe_dispersion >= 0.65:
                matches.append((i, i))
        return matches

    def _find_thesis_oscillation(self, window: int = 20,
                                  flip_threshold: int = 3) -> List[Tuple[int, int]]:
        """Detect thesis-state flipping between BULL_*/BEAR_* within a window
        — strong signal of indecision / chop / engineered confusion."""
        events = list(self.events)
        matches: List[Tuple[int, int]] = []
        n = len(events)
        for start in range(0, n - window + 1):
            window_events = events[start:start + window]
            flips = 0
            prev_side = ""
            for ev in window_events:
                side = ("bull" if "BULL" in ev.thesis_state
                         else "bear" if "BEAR" in ev.thesis_state else "")
                if side and side != prev_side and prev_side:
                    flips += 1
                if side:
                    prev_side = side
            if flips >= flip_threshold:
                matches.append((start, start + window - 1))
        return matches

    def summary(self) -> Dict[str, Any]:
        """Compact rollup of the last 60 minutes' worth of memory."""
        if not self.events:
            return {"event_count": 0}
        evs = list(self.events)
        surprise_avg = sum(e.surprise_score for e in evs) / len(evs)
        ce_def_avg = sum(e.ce_defended_fraction for e in evs) / len(evs)
        pe_def_avg = sum(e.pe_defended_fraction for e in evs) / len(evs)
        ce_rej_avg = sum(e.ce_rejected_fraction for e in evs) / len(evs)
        pe_rej_avg = sum(e.pe_rejected_fraction for e in evs) / len(evs)
        return {
            "event_count": len(evs),
            "avg_surprise": round(surprise_avg, 3),
            "ce_defended_fraction_avg": round(ce_def_avg, 3),
            "pe_defended_fraction_avg": round(pe_def_avg, 3),
            "ce_rejected_fraction_avg": round(ce_rej_avg, 3),
            "pe_rejected_fraction_avg": round(pe_rej_avg, 3),
            "last_signature": evs[-1].flow_signature,
            "pattern_counts": {
                "surprise_up_then_revert": len(self.find_pattern("surprise_up_then_revert")),
                "surprise_down_then_revert": len(self.find_pattern("surprise_down_then_revert")),
                "sustained_ce_defended": len(self.find_pattern("sustained_ce_defended")),
                "sustained_pe_defended": len(self.find_pattern("sustained_pe_defended")),
                "dispersion_spike": len(self.find_pattern("dispersion_spike")),
                "thesis_oscillation": len(self.find_pattern("thesis_oscillation")),
            },
        }


def _map(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except (TypeError, ValueError):
        pass
    return default
