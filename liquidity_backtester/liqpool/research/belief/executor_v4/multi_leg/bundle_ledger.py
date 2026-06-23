"""BundleLedger — tracks open multi-leg bundles and their combined P&L.

Mirrors the single-leg ``LedgerStore`` design but at the bundle
granularity. Each open bundle has one ledger entry holding:

  * the bundle dataclass (legs + structure + targets)
  * a per-leg fill price (post-reconciliation)
  * the current combined premium (refreshed each tick from live marks)
  * the current combined R = (net_credit_now − net_credit_at_entry)
    × bundle_lot_size, divided by max_loss_estimate
  * a tape of the last N premium samples for the cockpit's combined-R
    chart

Closure: at exit, the entry moves into the ``closed`` list with a
realised P&L, exit reason, and bars-held.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from .bundle import (
    BUNDLE_CLOSED, BUNDLE_CLOSING, BUNDLE_OPEN, MultiLegBundle,
)
from .leg_spec import LegSpec


@dataclass
class BundleLedgerEntry:
    """One open or closed bundle's tracked state."""
    bundle: MultiLegBundle
    # Per-leg actual fill price at entry (rupees per share).
    leg_fill_prices: List[float]
    # Combined premium right now (rupees per single bundle-unit).
    current_combined_premium: float = 0.0
    # Combined R: (current - entry) / max_loss (signed for credit
    # structures so 'good for us' is always positive).
    current_combined_r: float = 0.0
    best_combined_r: float = 0.0
    worst_combined_r: float = 0.0
    bars_held: int = 0
    # Realised at close — None while OPEN.
    realised_rupees: Optional[float] = None
    exit_reason: Optional[str] = None
    bars_to_resolution: Optional[int] = None
    # Recent combined-premium tape for charting.
    premium_tape: Deque[float] = field(default_factory=lambda: deque(maxlen=120))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle": self.bundle.to_dict(),
            "leg_fill_prices": list(self.leg_fill_prices),
            "current_combined_premium": round(self.current_combined_premium, 2),
            "current_combined_r": round(self.current_combined_r, 3),
            "best_combined_r": round(self.best_combined_r, 3),
            "worst_combined_r": round(self.worst_combined_r, 3),
            "bars_held": self.bars_held,
            "realised_rupees": (round(self.realised_rupees, 2)
                                  if self.realised_rupees is not None else None),
            "exit_reason": self.exit_reason,
            "bars_to_resolution": self.bars_to_resolution,
            "premium_tape": list(self.premium_tape),
        }


class BundleLedger:
    """Holds open and closed bundle entries; provides combined-R updates."""

    def __init__(self, *, lot_size: int = 65) -> None:
        self.lot_size = int(lot_size)
        self._open: Dict[str, BundleLedgerEntry] = {}
        self._closed: List[BundleLedgerEntry] = []

    # ── Open / close ───────────────────────────────────────────────

    def open(self, bundle: MultiLegBundle,
              leg_fill_prices: List[float]) -> BundleLedgerEntry:
        bundle.state = BUNDLE_OPEN
        bundle.ts_opened = bundle.ts_opened or time.time()
        entry = BundleLedgerEntry(
            bundle=bundle,
            leg_fill_prices=list(leg_fill_prices),
            current_combined_premium=bundle.net_credit_at_entry,
            current_combined_r=0.0,
        )
        self._open[bundle.bundle_id] = entry
        return entry

    def close(self, bundle_id: str, *,
                realised_rupees: float,
                exit_reason: str,
                bars_held: Optional[int] = None,
                ) -> Optional[BundleLedgerEntry]:
        entry = self._open.pop(bundle_id, None)
        if entry is None:
            return None
        entry.bundle.state = BUNDLE_CLOSED
        entry.bundle.ts_closed = time.time()
        entry.realised_rupees = realised_rupees
        entry.exit_reason = exit_reason
        if bars_held is not None:
            entry.bars_to_resolution = bars_held
        self._closed.append(entry)
        return entry

    # ── Per-tick updates ───────────────────────────────────────────

    def update_combined_marks(self, bundle_id: str,
                                 *,
                                 leg_marks: List[float],
                                 bar_index: int,
                                 ) -> Optional[BundleLedgerEntry]:
        """Refresh the combined premium + R for the given bundle from
        live per-leg marks. Returns the entry if updated."""
        entry = self._open.get(bundle_id)
        if entry is None:
            return None
        if len(leg_marks) != len(entry.bundle.legs):
            return entry
        net_now = 0.0
        for leg, mark in zip(entry.bundle.legs, leg_marks):
            sign = 1.0 if leg.is_short else -1.0
            net_now += sign * mark * leg.lots
        entry.current_combined_premium = net_now
        # Combined R: how much of the entry credit decayed for us.
        # For a credit structure (short-vol) the "good" direction is
        # net_now decreasing toward zero — we keep the credit. For a
        # debit structure (long-vol) the "good" direction is net_now
        # going more negative (premium expansion).
        credit_at_entry = entry.bundle.net_credit_at_entry
        max_loss = max(1.0, entry.bundle.max_loss_estimate / max(1, self.lot_size))
        if credit_at_entry >= 0:
            # Credit structure: good = premium going down.
            entry.current_combined_r = (credit_at_entry - net_now) / max_loss
        else:
            # Debit structure: good = premium going down (more negative).
            entry.current_combined_r = (credit_at_entry - net_now) / max_loss
        entry.best_combined_r = max(entry.best_combined_r,
                                       entry.current_combined_r)
        entry.worst_combined_r = min(entry.worst_combined_r,
                                         entry.current_combined_r)
        entry.premium_tape.append(net_now)
        if entry.bundle.bar_opened is not None:
            entry.bars_held = max(0, bar_index - entry.bundle.bar_opened)
        return entry

    # ── Reads ───────────────────────────────────────────────────────

    def open_bundles(self) -> List[BundleLedgerEntry]:
        return list(self._open.values())

    def closed_bundles(self) -> List[BundleLedgerEntry]:
        return list(self._closed)

    def get(self, bundle_id: str) -> Optional[BundleLedgerEntry]:
        return self._open.get(bundle_id)

    def n_open(self) -> int:
        return len(self._open)

    def n_closed(self) -> int:
        return len(self._closed)

    def summary(self) -> Dict[str, Any]:
        wins = sum(1 for e in self._closed
                   if (e.realised_rupees or 0.0) > 0)
        losses = sum(1 for e in self._closed
                     if (e.realised_rupees or 0.0) <= 0)
        gross = sum((e.realised_rupees or 0.0) for e in self._closed)
        return {
            "n_open_bundles": self.n_open(),
            "n_closed_bundles": self.n_closed(),
            "open": [e.to_dict() for e in self._open.values()],
            "closed_recent": [e.to_dict()
                               for e in self._closed[-10:]],
            "wins": wins, "losses": losses,
            "total_realised_rupees": round(gross, 2),
            "win_rate": (round(wins / max(1, len(self._closed)), 3)
                          if self._closed else 0.0),
        }
