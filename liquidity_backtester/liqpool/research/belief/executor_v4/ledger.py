"""Per-position bar-by-bar audit ledger.

Every position the executor opens has a :class:`PositionLedger` attached
to it. The ledger records:
  * the full :class:`PositionHypothesis` at birth
  * a :class:`BarRecord` for every bar of the position's life
  * the :class:`LedgerOutcome` at close

Saturday post-mortems read the ledger. Future learning sweeps query
ledgers to update scenario priors. Operator reads it to understand why a
trade was taken and how it played out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .hypothesis import PositionHypothesis


@dataclass(frozen=True)
class BarRecord:
    """One bar's record of a held position."""
    bar_index: int
    ts: Any
    spot: float
    held_premium: float                           # the held leg's mark this bar
    current_r: float                              # premium-based R
    best_r_so_far: float
    worst_r_so_far: float
    cumulative_realized_pnl_rupees: float
    confidence: float
    high_water_confidence: float
    thesis_state: str
    iv_state: str
    battlefield_verdict: str
    winding_zone: str
    decision_action: str
    held_spread_state: str
    held_friendliness: float
    held_acceptance: str
    held_dod_z: float
    # Which validation criteria are CURRENTLY met (semantic checks)
    validations_met: List[str] = field(default_factory=list)
    # Which invalidation criteria are currently active (semantic checks)
    invalidations_active: List[str] = field(default_factory=list)
    # Any contradictions from temporal layer
    contradictions_this_bar: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        for k in ("validations_met", "invalidations_active",
                  "contradictions_this_bar"):
            d[k] = list(getattr(self, k))
        return d


@dataclass(frozen=True)
class LedgerOutcome:
    """The final summary of how a position resolved."""
    exit_reason: str
    exit_severity: str            # "hard" / "soft" / "operator"
    exit_premium: float
    exit_spot: float
    exit_bar: int
    realized_r: float
    realized_rupees: float        # gross realized — fees + slippage subtracted
    gross_rupees: float            # before fees
    fees_rupees: float
    slippage_rupees: float
    bars_held: int
    # Post-mortem scorecard
    validation_ever_fully_met: bool
    invalidation_count: int
    hypothesis_vs_reality_score: float    # 0..1, how well reality matched the thesis

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class PositionLedger:
    """The full audit trail for one position."""
    hypothesis: PositionHypothesis
    bar_records: List[BarRecord] = field(default_factory=list)
    outcome: Optional[LedgerOutcome] = None

    @property
    def is_open(self) -> bool:
        return self.outcome is None

    @property
    def position_id(self) -> str:
        return self.hypothesis.position_id

    @property
    def direction(self) -> int:
        return self.hypothesis.direction

    def append_bar(self, record: BarRecord) -> None:
        self.bar_records.append(record)

    def close(self, outcome: LedgerOutcome) -> None:
        self.outcome = outcome

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis": self.hypothesis.to_dict(),
            "bar_records": [b.to_dict() for b in self.bar_records],
            "outcome": self.outcome.to_dict() if self.outcome else None,
            "is_open": self.is_open,
            "position_id": self.position_id,
        }

    def post_mortem(self) -> Dict[str, Any]:
        """Run the post-mortem analysis on a closed ledger.

        Compares what the hypothesis predicted to what actually happened,
        and produces a structured scorecard the operator (or a future
        learning sweep) can use to update priors.
        """
        if self.is_open:
            return {"status": "still_open"}
        out = self.outcome
        h = self.hypothesis
        n_bars = len(self.bar_records)
        # Which validations EVER met during life?
        validations_ever = set()
        for b in self.bar_records:
            for v in b.validations_met:
                validations_ever.add(v)
        validation_hit_rate = (
            len(validations_ever) / max(1, len(h.validation_criteria))
        )
        # Which invalidations triggered EVER?
        invalid_ever = set()
        for b in self.bar_records:
            for v in b.invalidations_active:
                invalid_ever.add(v)
        # Max drawdown during life
        worst_r = min((b.worst_r_so_far for b in self.bar_records),
                      default=0.0)
        # Did profit lock ever fire (i.e., best_r exceeded profit_lock threshold)
        best_r = max((b.best_r_so_far for b in self.bar_records),
                     default=0.0)
        return {
            "status": "closed",
            "direction": h.direction,
            "profile": h.profile,
            "contract_label": h.contract_label,
            "thesis_summary": h.thesis_summary,
            "bars_held": out.bars_held,
            "exit_reason": out.exit_reason,
            "exit_severity": out.exit_severity,
            "realized_r": out.realized_r,
            "realized_rupees": out.realized_rupees,
            "gross_rupees": out.gross_rupees,
            "fees_rupees": out.fees_rupees,
            "slippage_rupees": out.slippage_rupees,
            "best_r_during_life": best_r,
            "worst_r_during_life": worst_r,
            "validation_hit_rate": round(validation_hit_rate, 3),
            "validations_ever_met": sorted(validations_ever),
            "invalidations_ever_triggered": sorted(invalid_ever),
            "hypothesis_vs_reality_score": out.hypothesis_vs_reality_score,
            "n_bars_recorded": n_bars,
            "antithesis_score_at_entry": h.antithesis_score,
            "antithesis_was_right": (
                h.antithesis_score >= 0.5 and out.realized_r <= 0
            ),
        }


class LedgerStore:
    """In-memory store of ledgers. Persistence is a serialization concern
    handled by callers (write to JSONL, parquet, sqlite, whatever)."""

    def __init__(self) -> None:
        self._open: Dict[str, PositionLedger] = {}
        self._closed: List[PositionLedger] = []

    def open(self, ledger: PositionLedger) -> None:
        self._open[ledger.position_id] = ledger

    def find_open(self, position_id: str) -> Optional[PositionLedger]:
        return self._open.get(position_id)

    def close(self, position_id: str, outcome: LedgerOutcome) -> Optional[PositionLedger]:
        ledger = self._open.pop(position_id, None)
        if ledger is not None:
            ledger.close(outcome)
            self._closed.append(ledger)
        return ledger

    def open_positions(self) -> List[PositionLedger]:
        return list(self._open.values())

    def closed_positions(self) -> List[PositionLedger]:
        return list(self._closed)

    def summary(self) -> Dict[str, Any]:
        closed = self._closed
        n = len(closed)
        if n == 0:
            return {"open": len(self._open), "closed": 0}
        wins = sum(1 for l in closed if (l.outcome.realized_rupees or 0) > 0)
        losses = sum(1 for l in closed if (l.outcome.realized_rupees or 0) <= 0)
        gross = sum(l.outcome.realized_rupees for l in closed)
        fees = sum(l.outcome.fees_rupees for l in closed)
        slippage = sum(l.outcome.slippage_rupees for l in closed)
        return {
            "open": len(self._open),
            "closed": n,
            "wins": wins,
            "losses": losses,
            "win_rate": round(wins / max(1, n), 3),
            "net_realized_rupees": round(gross, 2),
            "cumulative_fees_rupees": round(fees, 2),
            "cumulative_slippage_rupees": round(slippage, 2),
            "best_trade_rupees": round(max((l.outcome.realized_rupees
                                              for l in closed), default=0.0), 2),
            "worst_trade_rupees": round(min((l.outcome.realized_rupees
                                               for l in closed), default=0.0), 2),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "open_positions": [l.to_dict() for l in self._open.values()],
            "closed_positions": [l.to_dict() for l in self._closed],
            "summary": self.summary(),
        }
