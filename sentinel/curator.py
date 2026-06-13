"""Curator — reads the shadow ledger, judges each experiment, and
extracts the aggregate truths the founder described:

    "our target is usually 12.2 pts too far"
    "our stop is usually 8.2 pts too tight"
    "OTM works only when gamma expansion starts early"
    "ATM is better during chop"
    "confidence 0.7 actually wins only 53%"
    "confidence 0.55 but opening-trend wins 62%"

The curator does NOT retrain weights. It produces FINDINGS — structured
observations with sample sizes — that the calibration engine turns into
knob deltas. Per-event it fills the JUDGMENT block (was entry good, was
stop too tight, did the selection outperform alternatives, etc).

It is the honesty organ: every finding carries its n, so a "truth" from
3 trades is visibly not a truth.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .io_decl import IOSpec, declare
from .shadow_ledger import JOURNEY_CHECKPOINTS

# Optional resolver signature: given an identity row dict, return the
# institutional fair value (e.g. via sentinel.institutional.fair_value_from_svi)
# or None if it can't be priced. Kept as a callable so the curator stays
# decoupled from the institutional module — the wiring lives at the
# caller (server / nightly job).
FairValueResolver = Callable[[Dict[str, Any]], Optional[float]]


@dataclass
class Finding:
    key: str                       # machine id, e.g. "target_distance_bias"
    statement: str                 # human sentence
    value: float                   # the number (pts, ratio, etc.)
    n: int                         # sample size behind it
    confidence: str                # "firm" (n>=30) | "tentative" (10<=n<30) | "noise" (n<10)

    @staticmethod
    def grade(n: int) -> str:
        return "firm" if n >= 30 else ("tentative" if n >= 10 else "noise")


@dataclass
class CuratorReport:
    session: str
    n_events: int
    n_complete: int
    findings: List[Finding] = field(default_factory=list)
    per_scientist: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_moneyness: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    confidence_calibration: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session": self.session, "n_events": self.n_events,
            "n_complete": self.n_complete,
            "findings": [vars(f) for f in self.findings],
            "per_scientist": self.per_scientist,
            "per_moneyness": self.per_moneyness,
            "confidence_calibration": self.confidence_calibration,
        }


def _entry(row: Dict[str, Any]) -> float:
    return float((row.get("identity") or {}).get("premium") or 0.0)


def _hyp(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return row.get("hypothesis")


def _journey(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("journey") or {}


def _is_win(row: Dict[str, Any]) -> Optional[bool]:
    """A virtual long-option trade wins if its MFE reached the target
    before its MAE hit the stop. Returns None if unresolvable."""
    h, j = _hyp(row), _journey(row)
    if not h or not j.get("complete"):
        return None
    mfe, mae = j.get("mfe"), j.get("mae")
    tgt, stp = h.get("suggested_target"), h.get("suggested_stop")
    if None in (mfe, mae, tgt, stp):
        return None
    t_target, t_stop = j.get("time_to_target_min"), j.get("time_to_stop_min")
    if t_target is not None and t_stop is not None:
        return t_target <= t_stop        # whichever was hit first
    if t_target is not None:
        return True
    if t_stop is not None:
        return False
    return mfe >= tgt and mae > stp       # target reachable, stop never hit


class Curator:
    """Judges a session's ledger rows and extracts findings.

    Pass ``fair_value_resolver`` to surface the institutional 'entries
    are paying too rich' finding — the curator calls it per identity
    row and flags when entries sit consistently > ``misprice_threshold``
    (default 5%) above fair value."""

    def __init__(self,
                 fair_value_resolver: Optional[FairValueResolver] = None,
                 misprice_threshold: float = 0.05) -> None:
        self.fair_value_resolver = fair_value_resolver
        self.misprice_threshold = misprice_threshold

    def judge_rows(self, session: str,
                   rows: List[Dict[str, Any]]) -> CuratorReport:
        complete = [r for r in rows if _journey(r).get("complete")]
        rep = CuratorReport(session=session, n_events=len(rows),
                            n_complete=len(complete))
        self._stamp_judgments(complete)
        rep.findings += self._target_stop_bias(complete)
        rep.per_scientist = self._per_scientist(complete)
        rep.per_moneyness = self._per_moneyness(complete)
        rep.confidence_calibration = self._confidence_calibration(complete)
        rep.findings += self._context_findings(complete)
        if self.fair_value_resolver is not None:
            rep.findings += self._fair_value_misprice(rows)
        return rep

    # -- per-event judgment ------------------------------------------------

    def _stamp_judgments(self, rows: List[Dict[str, Any]]) -> None:
        for r in rows:
            h, j = _hyp(r), _journey(r)
            if not h:
                continue
            mfe, mae = j.get("mfe"), j.get("mae")
            tgt, stp, entry = (h.get("suggested_target"),
                               h.get("suggested_stop"), _entry(r))
            jd = r.setdefault("judgment", {})
            if mfe is not None and tgt is not None:
                jd["was_target_too_far"] = mfe < tgt        # never reached it
            if mae is not None and stp is not None and mfe is not None:
                # stop too tight = we got stopped but the move later went our way
                jd["was_stop_too_tight"] = (mae <= stp and mfe >= (tgt or mfe))
            win = _is_win(r)
            if win is not None:
                jd["was_entry_good"] = win

    # -- aggregate findings ---------------------------------------------------

    def _target_stop_bias(self, rows: List[Dict[str, Any]]) -> List[Finding]:
        target_gaps, stop_gaps = [], []
        for r in rows:
            h, j = _hyp(r), _journey(r)
            if not h:
                continue
            mfe, mae = j.get("mfe"), j.get("mae")
            tgt, stp = h.get("suggested_target"), h.get("suggested_stop")
            if mfe is not None and tgt is not None:
                target_gaps.append(tgt - mfe)     # +ve = target set beyond reach
            if mae is not None and stp is not None:
                stop_gaps.append(stp - mae)       # +ve = stop sat below the dip (fine);
                                                  # -ve = stop above the dip (too tight)
        out = []
        if target_gaps:
            med = statistics.median(target_gaps)
            out.append(Finding(
                "target_distance_bias",
                f"target is typically {med:+.1f} premium beyond the achieved "
                f"peak — {'too far, pull it in' if med > 0 else 'conservative'}",
                round(med, 2), len(target_gaps), Finding.grade(len(target_gaps))))
        if stop_gaps:
            med = statistics.median(stop_gaps)
            out.append(Finding(
                "stop_tightness_bias",
                f"stop typically sits {med:+.1f} from the realized dip — "
                f"{'too tight, widen it' if med > 0 else 'has room'}",
                round(med, 2), len(stop_gaps), Finding.grade(len(stop_gaps))))
        return out

    def _per_scientist(self, rows):
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(r.get("scientist", "?"), []).append(r)
        out = {}
        for name, rs in groups.items():
            wins = [w for w in (_is_win(r) for r in rs) if w is not None]
            n = len(wins)
            out[name] = {
                "n": n,
                "hit_rate": round(sum(wins) / n, 3) if n else None,
                "grade": Finding.grade(n),
            }
        return out

    def _per_moneyness(self, rows):
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            key = (r.get("identity") or {}).get("moneyness_key", "?")
            groups.setdefault(key, []).append(r)
        out = {}
        for key, rs in groups.items():
            wins = [w for w in (_is_win(r) for r in rs) if w is not None]
            n = len(wins)
            out[key] = {"n": n,
                        "hit_rate": round(sum(wins) / n, 3) if n else None}
        return out

    def _confidence_calibration(self, rows):
        # bucket predicted confidence vs realized win-rate
        buckets: Dict[str, List[bool]] = {}
        for r in rows:
            h = _hyp(r)
            win = _is_win(r)
            if not h or win is None:
                continue
            c = float(h.get("confidence") or 0)
            b = f"{int(c*10)*10}-{int(c*10)*10+10}%"
            buckets.setdefault(b, []).append(win)
        out = []
        for b in sorted(buckets):
            w = buckets[b]
            out.append({"confidence_bucket": b, "n": len(w),
                        "actual_win_rate": round(sum(w)/len(w), 3)})
        return out

    def _fair_value_misprice(self, rows: List[Dict[str, Any]]) -> List[Finding]:
        """Per-event: entry premium vs the institutional fair value (SVI
        BS). Aggregate finding: 'X% of entries are paying > Y% above fair
        value' — surfaces the 'we're chasing rich premium' bias the
        curator otherwise can't see."""
        deltas: List[float] = []
        for r in rows:
            entry = _entry(r)
            if entry <= 0:
                continue
            fv = self.fair_value_resolver(r) if self.fair_value_resolver else None
            if fv is None or fv <= 0:
                continue
            misprice_pct = (entry - fv) / fv
            deltas.append(misprice_pct)
            r.setdefault("judgment", {})["misprice_vs_fair_value_pct"] = round(
                misprice_pct, 4)
        if not deltas:
            return []
        med = statistics.median(deltas)
        rich = [d for d in deltas if d > self.misprice_threshold]
        cheap = [d for d in deltas if d < -self.misprice_threshold]
        verdict = ("paying rich vs fair value, tighten entry filter"
                   if len(rich) > len(cheap) else
                   "entries near or below fair value — healthy")
        return [Finding(
            "fair_value_misprice_bias",
            f"median entry sits {med*100:+.1f}% from fair value — "
            f"{verdict} (n={len(deltas)}, {len(rich)} rich, {len(cheap)} cheap)",
            round(med, 4), len(deltas), Finding.grade(len(deltas)),
        )]

    def _context_findings(self, rows) -> List[Finding]:
        """The founder's contextual truths: does ATM beat OTM in chop,
        does OTM need early gamma, etc. Computed from the judged rows."""
        out: List[Finding] = []
        # ATM vs OTM overall
        atm = [w for r in rows if "+0" in (r.get("identity") or {}).get("moneyness_key", "")
               for w in [_is_win(r)] if w is not None]
        otm = [w for r in rows if ("+2" in (r.get("identity") or {}).get("moneyness_key", "")
               or "+3" in (r.get("identity") or {}).get("moneyness_key", ""))
               for w in [_is_win(r)] if w is not None]
        if len(atm) >= 5 and len(otm) >= 5:
            ar, orr = sum(atm)/len(atm), sum(otm)/len(otm)
            out.append(Finding(
                "atm_vs_otm",
                f"ATM win {ar:.0%} (n={len(atm)}) vs OTM win {orr:.0%} "
                f"(n={len(otm)}) — {'ATM' if ar>orr else 'OTM'} edge",
                round(ar - orr, 3), min(len(atm), len(otm)),
                Finding.grade(min(len(atm), len(otm)))))
        return out


declare(IOSpec(
    module="sentinel.curator",
    purpose="judges ledger experiments + extracts aggregate truths (n-weighted)",
    inputs=["file:<journal>/ledger_<session>.jsonl (via shadow_ledger.read_session)"],
    outputs=["sentinel.curator.CuratorReport (findings, per-scientist, "
             "per-moneyness, confidence calibration)"],
    consumes_from=["sentinel.shadow_ledger"],
    produces_for=["sentinel.calibration", "sentinel.reports"],
    tier="LOGGED",
))
