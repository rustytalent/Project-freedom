"""Advisor — the profit-maximisation suggestions + the dip recommender
+ the self-scoring suggestion ledger.

Three pieces, deliberately in one module because they share one
contract: every output is a SUGGESTION with a rule id, never an
auto-trade. Sentinel acts only on trails the operator armed.

  Maximizer  — rule-based portfolio suggestions (R1..R5), each rule
               carries a rolling hit-rate learned from the ledger.
  Recommender — the founder's dip-buy scanner: direction view →
               which options just got cheap (fell hard from day high,
               near the money, liquid) and are viable if spot
               reverts.
  Ledger     — every suggestion logged with a premium snapshot;
               at T+resolve_minutes the live premium scores it
               right/wrong; per-rule EWMA hit-rates feed back into
               the maximizer (low-accuracy rules get visibly MUTED,
               not silently dropped — the operator sees the system
               learn).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .kite_client import InstrumentMeta, Quote
from .portfolio import PortfolioState

EWMA_ALPHA = 0.2
MUTE_BELOW = 0.45


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Suggestions (maximizer)
# ---------------------------------------------------------------------------

@dataclass
class Suggestion:
    suggestion_id: str
    rule_id: str
    tradingsymbol: str
    action: str                 # ARM_TRAIL | CLOSE | TIGHTEN | WARN
    reason: str
    premium_at_suggestion: float
    created_at_utc: str = field(default_factory=_utc_now)
    rule_hit_rate: Optional[float] = None
    muted: bool = False


class Maximizer:
    """Rule registry. Each rule: (predicate over portfolio) → suggestions."""

    def __init__(self, ledger: "SuggestionLedger") -> None:
        self.ledger = ledger

    def run(self, pf: PortfolioState,
            armed_symbols: List[str]) -> List[Suggestion]:
        out: List[Suggestion] = []
        out += self._r1_unprotected_winner(pf, armed_symbols)
        out += self._r2_paired_leg_bleed(pf)
        out += self._r3_peak_giveback(pf)
        out += self._r4_theta_burn(pf)
        out += self._r5_concentration(pf)
        # annotate with learned hit-rates + mute flags
        for s in out:
            hr = self.ledger.rule_hit_rate(s.rule_id)
            s.rule_hit_rate = round(hr, 3) if hr is not None else None
            s.muted = hr is not None and hr < MUTE_BELOW
        self.ledger.record_many(out)
        return out

    def _mk(self, rule: str, sym: str, action: str, reason: str,
            premium: float) -> Suggestion:
        return Suggestion(
            suggestion_id=f"SG_{rule}_{sym}_{int(time.time()*1000)}",
            rule_id=rule, tradingsymbol=sym, action=action,
            reason=reason, premium_at_suggestion=premium,
        )

    def _r1_unprotected_winner(self, pf: PortfolioState,
                               armed: List[str]) -> List[Suggestion]:
        out = []
        for v in pf.positions.values():
            if v.pnl > 1500 and v.tradingsymbol not in armed:
                out.append(self._mk(
                    "R1_unprotected_winner", v.tradingsymbol, "ARM_TRAIL",
                    f"position +₹{v.pnl:,.0f} with no trail armed — one "
                    f"pullback erases it", v.ltp))
        return out

    def _r2_paired_leg_bleed(self, pf: PortfolioState) -> List[Suggestion]:
        out = []
        for pair in pf.pairs:
            legs = [pf.positions[s] for s in pair.legs if s in pf.positions]
            losers = [l for l in legs if l.pnl < -1000]
            winners = [l for l in legs if l.pnl > 1000]
            if losers and winners:
                lo = min(losers, key=lambda l: l.pnl)
                out.append(self._mk(
                    "R2_paired_leg_bleed", lo.tradingsymbol, "CLOSE",
                    f"losing {lo.option_type} leg (-₹{abs(lo.pnl):,.0f}) is "
                    f"bleeding against the winning leg in {pair.underlying} "
                    f"{pair.expiry}", lo.ltp))
        return out

    def _r3_peak_giveback(self, pf: PortfolioState) -> List[Suggestion]:
        out = []
        for v in pf.positions.values():
            if (v.peak_pnl > 2000 and v.pnl > 0
                    and v.drawdown_from_peak > 0.4 * v.peak_pnl):
                out.append(self._mk(
                    "R3_peak_giveback", v.tradingsymbol, "TIGHTEN",
                    f"gave back ₹{v.drawdown_from_peak:,.0f} of a "
                    f"₹{v.peak_pnl:,.0f} peak (>40%) — tighten or exit",
                    v.ltp))
        return out

    def _r4_theta_burn(self, pf: PortfolioState) -> List[Suggestion]:
        out = []
        for v in pf.positions.values():
            if (v.quantity > 0 and v.theta_per_day is not None
                    and v.ltp > 0 and v.t_years * 365.0 < 2.0
                    and abs(v.theta_per_day) > 0.10 * v.ltp
                    and abs(v.pnl) < 1000):
                out.append(self._mk(
                    "R4_theta_burn", v.tradingsymbol, "CLOSE",
                    f"<2 days to expiry, theta ₹{abs(v.theta_per_day):,.1f}/day "
                    f"is {abs(v.theta_per_day)/v.ltp:.0%} of premium and the "
                    f"position is going nowhere", v.ltp))
        return out

    def _r5_concentration(self, pf: PortfolioState) -> List[Suggestion]:
        out = []
        if not pf.positions:
            return out
        exposure: Dict[str, float] = {}
        total = 0.0
        for v in pf.positions.values():
            notional = abs(v.ltp * v.quantity)
            exposure[v.underlying or v.tradingsymbol] = (
                exposure.get(v.underlying or v.tradingsymbol, 0.0) + notional)
            total += notional
        for und, notional in exposure.items():
            if total > 0 and notional / total > 0.8 and len(pf.positions) > 1:
                out.append(self._mk(
                    "R5_concentration", und, "WARN",
                    f"{notional/total:.0%} of option exposure in {und} — "
                    f"one underlying decides the day", 0.0))
        return out


# ---------------------------------------------------------------------------
# Dip recommender
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    tradingsymbol: str
    option_type: str
    strike: float
    ltp: float
    day_high: float
    fall_from_high_pct: float
    distance_from_spot_pct: float
    oi: float
    volume: float
    spread_pct: float
    score: float
    rationale: str


def recommend_dips(direction: str, spot: float,
                   metas: List[InstrumentMeta],
                   quotes: Dict[str, Quote],
                   top_n: int = 6) -> List[Recommendation]:
    """The founder's scanner. direction: "UP" | "DOWN" | "NEUTRAL".

    Spot going UP makes puts cheaper → scan PE for the deepest
    intraday markdowns that stay near the money and liquid: the
    candidates that recover hardest if spot mean-reverts. DOWN
    mirrors to CE. NEUTRAL returns both halves interleaved.

    Score = 0.5·fall + 0.3·liquidity + 0.2·proximity, each min-max
    normalised in-batch. Composite is intentionally simple and
    VISIBLE — the dashboard shows every component so the operator
    can disagree with the weighting.
    """
    if direction == "NEUTRAL":
        ups = recommend_dips("UP", spot, metas, quotes, top_n=top_n // 2 or 1)
        downs = recommend_dips("DOWN", spot, metas, quotes, top_n=top_n // 2 or 1)
        return ups + downs
    want_type = "PE" if direction == "UP" else "CE"
    rows = []
    for m in metas:
        if m.instrument_type != want_type:
            continue
        q = quotes.get(m.tradingsymbol)
        if q is None or q.ltp <= 0 or q.day_high <= 0:
            continue
        fall = max(0.0, (q.day_high - q.ltp) / q.day_high)
        dist = abs(m.strike - spot) / spot
        if dist > 0.06:           # beyond ±6% — lottery tickets, skip
            continue
        spread = ((q.ask - q.bid) / q.ltp) if (q.ask > q.bid > 0) else 1.0
        rows.append((m, q, fall, dist, spread))
    if not rows:
        return []
    max_fall = max(r[2] for r in rows) or 1.0
    max_liq = max((r[1].volume + r[1].oi) for r in rows) or 1.0
    recs = []
    for m, q, fall, dist, spread in rows:
        liq = (q.volume + q.oi) / max_liq
        proximity = 1.0 - min(dist / 0.06, 1.0)
        score = 0.5 * (fall / max_fall) + 0.3 * liq + 0.2 * proximity
        if spread > 0.05:
            score *= 0.5          # punishing wide spreads, visibly
        recs.append(Recommendation(
            tradingsymbol=m.tradingsymbol, option_type=m.instrument_type,
            strike=m.strike, ltp=q.ltp, day_high=q.day_high,
            fall_from_high_pct=round(fall * 100, 1),
            distance_from_spot_pct=round(dist * 100, 2),
            oi=q.oi, volume=q.volume,
            spread_pct=round(spread * 100, 2),
            score=round(score, 4),
            rationale=(f"fell {fall:.0%} from day high; "
                       f"{dist:.1%} from spot; "
                       f"{'liquid' if liq > 0.3 else 'thin'}"),
        ))
    recs.sort(key=lambda r: r.score, reverse=True)
    return recs[:top_n]


# ---------------------------------------------------------------------------
# Self-scoring ledger
# ---------------------------------------------------------------------------

class SuggestionLedger:
    """Every suggestion gets resolved against the live premium at
    T+resolve_minutes:

      CLOSE / TIGHTEN — right if the premium FELL after the call
                        (exiting was the better move)
      ARM_TRAIL       — right if premium later gave back >= 20% from
                        its post-suggestion peak (a trail would have
                        captured the difference)
      WARN            — informational; never scored

    Per-rule EWMA hit-rates persist across restarts (JSON sidecar)
    and feed straight back into the maximizer's mute flags — the
    visible v1 of 'it checks itself and improves immediately'.
    """

    def __init__(self, journal_path: Path,
                 resolve_minutes: float = 10.0) -> None:
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.resolve_minutes = float(resolve_minutes)
        self._pending: List[Dict[str, Any]] = []
        self._rates_path = self.journal_path.with_suffix(".rates.json")
        self._rates: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._post_peak: Dict[str, float] = {}
        self._load_rates()

    def _load_rates(self) -> None:
        try:
            blob = json.loads(self._rates_path.read_text())
            self._rates = {k: float(v) for k, v in blob.get("rates", {}).items()}
            self._counts = {k: int(v) for k, v in blob.get("counts", {}).items()}
        except Exception:
            pass

    def _save_rates(self) -> None:
        self._rates_path.write_text(json.dumps(
            {"rates": self._rates, "counts": self._counts}))

    def record_many(self, suggestions: List[Suggestion]) -> None:
        now = time.time()
        seen_live = {p["tradingsymbol"] + p["rule_id"] for p in self._pending}
        for s in suggestions:
            key = s.tradingsymbol + s.rule_id
            if key in seen_live:       # don't restack the same live call
                continue
            row = {**asdict(s), "recorded_monotonic": now}
            self._pending.append(row)
            with open(self.journal_path, "a") as f:
                f.write(json.dumps({k: v for k, v in row.items()
                                    if k != "recorded_monotonic"}) + "\n")

    def resolve_due(self, live_premiums: Dict[str, float]) -> int:
        """Score every pending suggestion older than resolve_minutes.
        Returns how many were resolved this call."""
        now = time.time()
        horizon = self.resolve_minutes * 60.0
        still: List[Dict[str, Any]] = []
        resolved = 0
        for row in self._pending:
            sym = row["tradingsymbol"]
            ltp = live_premiums.get(sym)
            # track post-suggestion peak for ARM_TRAIL scoring
            if ltp and ltp > 0:
                pk = self._post_peak.get(row["suggestion_id"], 0.0)
                self._post_peak[row["suggestion_id"]] = max(pk, ltp)
            if now - row["recorded_monotonic"] < horizon:
                still.append(row)
                continue
            verdict = self._score(row, ltp)
            if verdict is None:
                resolved += 1   # unscorable; drop without rating
                continue
            rule = row["rule_id"]
            prev = self._rates.get(rule, 0.5)
            self._rates[rule] = (1 - EWMA_ALPHA) * prev + EWMA_ALPHA * (
                1.0 if verdict else 0.0)
            self._counts[rule] = self._counts.get(rule, 0) + 1
            resolved += 1
        self._pending = still
        if resolved:
            self._save_rates()
        return resolved

    def _score(self, row: Dict[str, Any],
               ltp: Optional[float]) -> Optional[bool]:
        if ltp is None or ltp <= 0:
            return None
        p0 = float(row["premium_at_suggestion"] or 0.0)
        action = row["action"]
        if action in ("CLOSE", "TIGHTEN"):
            return ltp < p0          # premium fell -> exiting was right
        if action == "ARM_TRAIL":
            peak = self._post_peak.get(row["suggestion_id"], p0)
            return peak > 0 and (peak - ltp) >= 0.2 * peak
        return None                  # WARN — informational

    def rule_hit_rate(self, rule_id: str) -> Optional[float]:
        if self._counts.get(rule_id, 0) < 3:
            return None              # too few resolutions to judge
        return self._rates.get(rule_id)

    def stats(self) -> Dict[str, Any]:
        return {
            "pending": len(self._pending),
            "rules": {
                r: {"hit_rate": round(self._rates.get(r, 0.5), 3),
                    "n_resolved": self._counts.get(r, 0),
                    "muted": (self._counts.get(r, 0) >= 3
                              and self._rates.get(r, 0.5) < MUTE_BELOW)}
                for r in sorted(set(self._rates) | set(self._counts))
            },
        }
