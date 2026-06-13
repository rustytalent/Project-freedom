"""Calibration engine — turns curator findings into KNOB deltas, and
proves the knob would have helped before accepting it.

The founder's design, exactly: not blind retraining (which overfits one
weird day and ruins weeks). Instead, the shadow ledger reveals biases
("target 12pt too far", "stop 8pt too tight"), and the calibration
engine nudges the KNOBS:

    entry_threshold, target_distance_mult, stop_distance_mult,
    strike_preference, holding_time_min, confidence_haircut,
    partial_booking_ratio, trail_aggressiveness

Then — critically — it RE-RUNS the stored journeys with the proposed
knob applied and only accepts the change if expectancy improves. This
is the founder's "remove the data at that time, apply the calibration,
run the simulation again, see if it actually worked."

Knobs persist to JSON. Weights are never touched here — that is a
deliberate overnight job, not a live mutation.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .curator import CuratorReport, Finding, _hyp, _is_win, _journey
from .io_decl import IOSpec, declare


@dataclass
class Knobs:
    target_distance_mult: float = 1.0     # scales suggested target gap
    stop_distance_mult: float = 1.0       # scales suggested stop gap
    entry_threshold: float = 0.55         # min confidence to surface a trade
    holding_time_min: float = 15.0
    confidence_haircut: float = 0.0       # subtracted from raw confidence
    partial_booking_ratio: float = 0.5
    trail_aggressiveness: float = 0.5     # 0 loose .. 1 tight
    strike_preference: str = "ATM"        # ATM | OTM (curator-driven)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationResult:
    accepted: List[str] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)
    before_expectancy: float = 0.0
    after_expectancy: float = 0.0
    knobs: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


class CalibrationEngine:
    def __init__(self, knobs_path: Optional[Path] = None) -> None:
        self.knobs_path = Path(knobs_path) if knobs_path else None
        self.knobs = self._load() or Knobs()

    def _load(self) -> Optional[Knobs]:
        if self.knobs_path and self.knobs_path.exists():
            try:
                return Knobs(**json.loads(self.knobs_path.read_text()))
            except Exception:
                return None
        return None

    def _save(self) -> None:
        if self.knobs_path:
            self.knobs_path.parent.mkdir(parents=True, exist_ok=True)
            self.knobs_path.write_text(json.dumps(self.knobs.to_dict(), indent=2))

    # -- the re-run test ----------------------------------------------------

    @staticmethod
    def _expectancy(rows: List[Dict[str, Any]],
                    knobs: Knobs) -> float:
        """Replay each completed journey under a candidate knob set and
        return mean R. R = (exit - entry)/risk, where the exit is the
        journey peak capped by the (knob-scaled) target, floored by the
        (knob-scaled) stop. This is the honest counterfactual the
        founder asked for: same journeys, different knobs."""
        rs = []
        for row in rows:
            h, j = _hyp(row), _journey(row)
            if not h or not j.get("complete"):
                continue
            entry = float((row.get("identity") or {}).get("premium") or 0)
            mfe, mae = j.get("mfe"), j.get("mae")
            base_tgt, base_stop = h.get("suggested_target"), h.get("suggested_stop")
            if None in (entry, mfe, mae, base_tgt, base_stop) or entry <= 0:
                continue
            tgt = entry + (base_tgt - entry) * knobs.target_distance_mult
            stop = entry - (entry - base_stop) * knobs.stop_distance_mult
            risk = max(entry - stop, 1e-6)
            # Did the (scaled) stop get hit before the (scaled) target?
            t_tgt, t_stop = j.get("time_to_target_min"), j.get("time_to_stop_min")
            hit_target = mfe >= tgt
            hit_stop = mae <= stop
            if hit_stop and (not hit_target or
                             (t_stop is not None and t_tgt is not None
                              and t_stop < t_tgt)):
                r = (stop - entry) / risk
            elif hit_target:
                r = (tgt - entry) / risk
            else:
                # neither hit: exit at last-known journey value (t+60)
                last = j.get("outcomes", {}).get("t+60") or entry
                r = (last - entry) / risk
            rs.append(r)
        return sum(rs) / len(rs) if rs else 0.0

    # -- proposal + accept --------------------------------------------------

    # candidate grids per knob — the search space the re-run explores.
    TARGET_GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    STOP_GRID = [1.0, 1.25, 1.5, 1.75, 2.0]

    def calibrate(self, report: CuratorReport,
                  rows: List[Dict[str, Any]]) -> CalibrationResult:
        """The founder's method, literally: tune the knob up/down, RE-RUN
        the stored journeys, keep the setting that actually improves
        expectancy. Only FIRM findings open a knob to the search;
        'tentative'/'noise' findings are recorded but never act. The
        re-run is the gate — a proposal that doesn't beat baseline on the
        real journeys is rejected, so we never overfit a guess."""
        res = CalibrationResult()
        before = self._expectancy(rows, self.knobs)
        res.before_expectancy = round(before, 4)

        # which knobs are firm enough to open for search?
        search_target = search_stop = False
        for f in report.findings:
            if f.confidence != "firm":
                res.notes.append(f"ignored (n={f.n}, {f.confidence}): {f.key}")
                continue
            if f.key == "target_distance_bias" and f.value > 0:
                search_target = True
                res.notes.append("proposed: search target_distance_mult")
            elif f.key == "stop_tightness_bias" and f.value > 0:
                search_stop = True
                res.notes.append("proposed: search stop_distance_mult")
            elif f.key == "atm_vs_otm":
                pref = "ATM" if f.value > 0 else "OTM"
                self.knobs.strike_preference = pref
                res.accepted.append(f"prefer {pref}")
                res.notes.append(f"set strike_preference={pref}")

        if not (search_target or search_stop):
            res.after_expectancy = res.before_expectancy
            res.notes.append("no firm directional bias to search")
            res.knobs = self.knobs.to_dict()
            if res.accepted:
                self._save()
            return res

        target_grid = self.TARGET_GRID if search_target else [self.knobs.target_distance_mult]
        stop_grid = self.STOP_GRID if search_stop else [self.knobs.stop_distance_mult]
        best, best_e = None, before
        for tm in target_grid:
            for sm in stop_grid:
                cand = Knobs(**self.knobs.to_dict())
                cand.target_distance_mult = tm
                cand.stop_distance_mult = sm
                e = self._expectancy(rows, cand)
                if e > best_e + 1e-6:
                    best, best_e = cand, e

        res.after_expectancy = round(best_e, 4)
        if best is not None:
            self.knobs = best
            self._save()
            res.accepted.append(
                f"target_mult={best.target_distance_mult}, "
                f"stop_mult={best.stop_distance_mult}")
            res.notes.append(
                f"ACCEPTED via re-run: expectancy {before:.3f} -> {best_e:.3f}")
        else:
            res.rejected.append("target/stop search")
            res.notes.append(
                f"REJECTED: no grid setting beat baseline ({before:.3f})")
        res.knobs = self.knobs.to_dict()
        return res


declare(IOSpec(
    module="sentinel.calibration",
    purpose="curator findings -> knob deltas, accepted only if a journey "
            "re-run proves expectancy improves (no blind retraining)",
    inputs=["sentinel.curator.CuratorReport",
            "file:<journal>/ledger_<session>.jsonl rows"],
    outputs=["sentinel.calibration.Knobs (persisted JSON)",
             "sentinel.calibration.CalibrationResult"],
    consumes_from=["sentinel.curator", "sentinel.shadow_ledger"],
    produces_for=["sentinel.scientists", "sentinel.trails", "sentinel.reports"],
    tier="TRUSTED",
))
