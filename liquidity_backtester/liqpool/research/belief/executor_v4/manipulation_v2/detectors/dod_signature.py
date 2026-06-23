"""DodSignatureDetector — premium-dislocation signatures across slot grid.

Three signatures we detect from the per-slot ``dod_z`` values
(deviation-of-deviation from substrate):

  1. **Single-strike defense** — one strike has |dod_z| > z_extreme
     while its neighbours sit at baseline. Looks like an isolated
     defense or a fat single-leg hit.
  2. **Wall building** — three or more adjacent strikes on the SAME
     rail with same-sign dod_z above ``z_wall``. Walls indicate
     coordinated positioning (defense or attack).
  3. **Rail tilt** — average dod_z across one rail's OTM strikes is
     consistently negative (cheap) or positive (rich) over time. A
     persistently cheap OTM PE rail means relentless put-selling
     pressure (dealers think downside is contained).

The detector returns whichever signature is strongest. Direction:
  * defense on CE rail = MM expects spot down (selling CE → bearish)
  * defense on PE rail = MM expects spot up (selling PE → bullish)
  * wall building on CE rail = ceiling, expect down
  * wall building on PE rail = floor, expect up
"""
from __future__ import annotations

import statistics
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from .base import DetectorBase, DetectorMagnitude, DetectorPosterior


class DodSignatureDetector(DetectorBase):
    name = "dod_signature"

    def __init__(self, *,
                  z_extreme: float = 2.0,
                  z_wall: float = 1.5,
                  rail_tilt_bars: int = 5,
                  rail_tilt_threshold: float = 0.7,
                  ) -> None:
        self.z_extreme = float(z_extreme)
        self.z_wall = float(z_wall)
        self.rail_tilt_bars = int(rail_tilt_bars)
        self.rail_tilt_threshold = float(rail_tilt_threshold)
        # Per-bar avg dod_z by rail.
        self._rail_history: Deque[Tuple[int, float, float]] = deque(maxlen=64)

    def reset(self) -> None:
        self._rail_history.clear()

    def observe(self, *,
                  snapshot: Dict[str, Any],
                  rich_context: Optional[Any] = None,
                  web_snapshot: Optional[Any] = None,
                  bar_index: int = 0,
                  ) -> DetectorPosterior:
        slots = list(snapshot.get("slot_readings") or [])
        if not slots:
            return DetectorPosterior.quiet()

        # Index slots by (level, option_type) to ease neighbour scans.
        ce_by_level: Dict[int, Dict[str, Any]] = {}
        pe_by_level: Dict[int, Dict[str, Any]] = {}
        for raw in slots:
            slot = _map(raw)
            opt = str(slot.get("option_type") or "")
            try:
                level = int(slot.get("level") or 0)
            except (TypeError, ValueError):
                continue
            if opt == "CE":
                ce_by_level[level] = slot
            elif opt == "PE":
                pe_by_level[level] = slot

        # Per-rail running averages for the tilt analysis.
        ce_avg = _avg(_dod_z_values(ce_by_level.values()))
        pe_avg = _avg(_dod_z_values(pe_by_level.values()))
        self._rail_history.append((bar_index, ce_avg, pe_avg))

        # ── Signature 1: single-strike extremes ───────────────────
        single_strikes: List[Tuple[str, int, float, float]] = []
        for opt, by_lvl in (("CE", ce_by_level), ("PE", pe_by_level)):
            for lvl, slot in by_lvl.items():
                z = float(slot.get("dod_z") or 0.0)
                if abs(z) < self.z_extreme:
                    continue
                # Confirm it's isolated: neighbours within the same opt
                # at |dod_z| < 0.8.
                neighbours = []
                for off in (-1, 1):
                    n = by_lvl.get(lvl + off)
                    if n is not None:
                        neighbours.append(float(n.get("dod_z") or 0.0))
                if neighbours and max(
                        (abs(v) for v in neighbours), default=0.0) > 0.8:
                    continue
                strike = float(slot.get("strike") or 0.0)
                single_strikes.append((opt, lvl, z, strike))

        # ── Signature 2: walls (3+ adjacent same-sign on one rail) ─
        ce_wall = _detect_wall(ce_by_level, self.z_wall)
        pe_wall = _detect_wall(pe_by_level, self.z_wall)

        # ── Signature 3: persistent rail tilt ─────────────────────
        rail_tilt: Optional[Tuple[str, float]] = None
        if len(self._rail_history) >= self.rail_tilt_bars:
            window = list(self._rail_history)[-self.rail_tilt_bars:]
            ce_means = [r[1] for r in window]
            pe_means = [r[2] for r in window]
            avg_ce = statistics.mean(ce_means)
            avg_pe = statistics.mean(pe_means)
            # All bars in the window must agree on the sign.
            ce_consistent = (all(m > 0 for m in ce_means)
                              or all(m < 0 for m in ce_means))
            pe_consistent = (all(m > 0 for m in pe_means)
                              or all(m < 0 for m in pe_means))
            if (ce_consistent
                    and abs(avg_ce) >= self.rail_tilt_threshold):
                rail_tilt = ("CE", avg_ce)
            elif (pe_consistent
                  and abs(avg_pe) >= self.rail_tilt_threshold):
                rail_tilt = ("PE", avg_pe)

        # Pick the strongest signature.
        evidence: List[str] = []
        if ce_wall is not None or pe_wall is not None:
            # Walls dominate.
            wall = ce_wall if (ce_wall is not None and (
                pe_wall is None or abs(ce_wall[2]) >= abs(pe_wall[2])))\
                else pe_wall
            rail, levels, avg_z, target_strike = wall
            # Wall on CE = ceiling above spot = MM expects DOWN.
            # Wall on PE = floor below spot = MM expects UP.
            direction = -1 if rail == "CE" else 1
            classification = f"wall_{rail.lower()}"
            evidence.append(
                f"{rail} wall: {len(levels)} adjacent strikes "
                f"with avg dod_z={avg_z:+.2f}")
            probability = min(0.90, 0.55 + 0.08 * (abs(avg_z) - 1.0))
            confidence = min(0.92, 0.50 + 0.10 * (len(levels) - 2)
                              + 0.05 * (abs(avg_z) - 1.0))
            magnitude = DetectorMagnitude(
                z_score=float(avg_z), raw_value=float(len(levels)),
                units="wall strikes")
            return DetectorPosterior(
                fire=True, probability=probability, direction=direction,
                confidence=confidence,
                horizon_bars=8,
                targeted_strike=target_strike,
                magnitude=magnitude, evidence=evidence,
                classification=classification,
            )

        if single_strikes:
            # Multiple isolated extremes — surface the largest.
            single_strikes.sort(key=lambda x: abs(x[2]), reverse=True)
            opt, lvl, z, strike = single_strikes[0]
            # Single-strike defense (positive dod_z = mark rich vs model
            # = MM holding it up). Defense on CE = MM expects down;
            # defense on PE = MM expects up.
            direction = -1 if opt == "CE" else 1
            # Direction flips if dod_z is NEGATIVE (premium cheap → MM
            # eager seller → expects the opposite outcome on that
            # strike's side).
            if z < 0:
                direction = -direction
            classification = f"single_strike_{opt.lower()}_" + (
                "rich" if z > 0 else "cheap")
            evidence.append(
                f"single {opt} strike {int(strike)} at dod_z={z:+.2f} "
                f"(neighbours quiet)")
            probability = min(0.85, 0.45 + 0.10 * (abs(z) - 1.5))
            confidence = min(0.80, 0.45 + 0.10 * (abs(z) - 1.5))
            magnitude = DetectorMagnitude(
                z_score=float(z), raw_value=float(strike), units="strike")
            return DetectorPosterior(
                fire=True, probability=probability, direction=direction,
                confidence=confidence,
                horizon_bars=5,
                targeted_strike=strike,
                magnitude=magnitude, evidence=evidence,
                classification=classification,
            )

        if rail_tilt is not None:
            rail, avg = rail_tilt
            # Persistent tilt on CE rail rich (positive avg) = MM happily
            # selling CE high = expects spot DOWN. Cheap (negative) = MM
            # bidding CE = expects UP. PE rail interpretation is mirror.
            if rail == "CE":
                direction = -1 if avg > 0 else 1
            else:
                direction = 1 if avg > 0 else -1
            classification = (
                f"rail_tilt_{rail.lower()}_"
                + ("rich" if avg > 0 else "cheap"))
            evidence.append(
                f"{rail} rail tilt: avg dod_z={avg:+.2f} for "
                f"{self.rail_tilt_bars}b")
            probability = min(0.80, 0.45 + 0.10 * (abs(avg) - 0.5))
            confidence = min(0.75, 0.40 + 0.10 * (abs(avg) - 0.5))
            magnitude = DetectorMagnitude(
                z_score=float(avg), raw_value=float(abs(avg)),
                units="avg dod_z")
            return DetectorPosterior(
                fire=True, probability=probability, direction=direction,
                confidence=confidence,
                horizon_bars=12,
                magnitude=magnitude, evidence=evidence,
                classification=classification,
            )

        return DetectorPosterior.quiet()


def _detect_wall(by_level: Dict[int, Dict[str, Any]],
                  z_wall: float,
                  ) -> Optional[Tuple[str, List[int], float, float]]:
    """Find any contiguous run of ≥3 same-sign strikes with |dod_z| >= z_wall."""
    if not by_level:
        return None
    levels = sorted(by_level.keys())
    best: Optional[Tuple[str, List[int], float, float]] = None
    cur: List[int] = []
    cur_sign = 0
    for lvl in levels:
        z = float(by_level[lvl].get("dod_z") or 0.0)
        if abs(z) < z_wall:
            cur = []
            cur_sign = 0
            continue
        sign = 1 if z > 0 else -1
        if cur and sign != cur_sign:
            cur = []
        cur.append(lvl)
        cur_sign = sign
        if len(cur) >= 3:
            avg_z = statistics.mean(
                float(by_level[l].get("dod_z") or 0.0) for l in cur)
            # Targeted strike: middle of the wall.
            mid_lvl = cur[len(cur) // 2]
            target_strike = float(by_level[mid_lvl].get("strike") or 0.0)
            rail = str(by_level[mid_lvl].get("option_type") or "CE")
            cand = (rail, list(cur), avg_z, target_strike)
            if best is None or abs(cand[2]) > abs(best[2]):
                best = cand
    return best


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _dod_z_values(slot_iter: Any) -> List[float]:
    out: List[float] = []
    for slot in slot_iter:
        try:
            out.append(float(slot.get("dod_z") or 0.0))
        except (TypeError, ValueError):
            continue
    return out


def _map(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}
