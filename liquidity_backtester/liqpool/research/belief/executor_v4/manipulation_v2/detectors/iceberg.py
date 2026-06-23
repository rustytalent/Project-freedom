"""IcebergDetector — recurring same-size fills at the same price.

An iceberg order shows only a small slice on the book; when filled, a
new same-size slice replaces it at the same price. The signature:

  * A specific (strike, price) sees ``last_trade_size`` repeatedly
    matching a small value over multiple ticks.
  * The depth at that price stays roughly the same after fills (the
    hidden child orders refresh).
  * The persistence_score for that book stays high.

Robust to spoofing: spoofed orders cancel without trading. Iceberg
orders TRADE (the whole point is hidden execution), so we look at
the tape, not just depth.

Direction interpretation:
  * Iceberg on the BUY side of a strike = persistent hidden buying
    pressure at that strike → spot UP toward the strike (if OTM CE
    or ITM PE) or DOWN toward it (if OTM PE or ITM CE).
  * Simpler signal: iceberg in a CE = bid for upside protection =
    expect spot UP. Iceberg in a PE = bid for downside protection =
    expect spot DOWN.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from .base import DetectorBase, DetectorMagnitude, DetectorPosterior


class IcebergDetector(DetectorBase):
    name = "iceberg"

    def __init__(self, *,
                  history_ticks: int = 12,
                  min_recurrences: int = 4,
                  size_tolerance_pct: float = 0.10,
                  min_persistence_score: float = 0.40,
                  ) -> None:
        self.history_ticks = int(history_ticks)
        self.min_recurrences = int(min_recurrences)
        self.size_tolerance_pct = float(size_tolerance_pct)
        self.min_persistence_score = float(min_persistence_score)
        # Map: tradingsymbol → deque of (size, ltp).
        self._tape_history: Dict[str, Deque[Tuple[int, float]]] = {}

    def reset(self) -> None:
        self._tape_history.clear()

    def observe(self, *,
                  snapshot: Dict[str, Any],
                  rich_context: Optional[Any] = None,
                  web_snapshot: Optional[Any] = None,
                  bar_index: int = 0,
                  ) -> DetectorPosterior:
        slots = list(snapshot.get("slot_readings") or [])
        if not slots:
            return DetectorPosterior.quiet()

        ce_iceberg = 0
        pe_iceberg = 0
        targeted_strikes: List[float] = []
        magnitudes: List[float] = []
        for raw in slots:
            slot = raw if isinstance(raw, dict) else {}
            sym = str(slot.get("label") or "")
            last_size = int(slot.get("last_trade_size") or 0)
            if last_size <= 0 or not sym:
                continue
            persistence = float(slot.get("persistence_score") or 1.0)
            if persistence < self.min_persistence_score:
                continue
            ltp = float(slot.get("ltp") or slot.get("mark_price") or 0.0)
            history = self._tape_history.setdefault(
                sym, deque(maxlen=self.history_ticks))
            history.append((last_size, ltp))
            # Look for recurrent same-size fills at the same price.
            if len(history) < self.min_recurrences:
                continue
            ref_size = last_size
            recurrences = 0
            for (sz, px) in history:
                if (abs(sz - ref_size)
                        <= self.size_tolerance_pct * ref_size
                        and abs(px - ltp) <= 0.5):
                    recurrences += 1
            if recurrences < self.min_recurrences:
                continue
            strike = float(slot.get("strike") or 0.0)
            opt = str(slot.get("option_type") or "")
            targeted_strikes.append(strike)
            magnitudes.append(float(recurrences))
            if opt == "CE":
                ce_iceberg += 1
            elif opt == "PE":
                pe_iceberg += 1

        if max(ce_iceberg, pe_iceberg) < 1:
            return DetectorPosterior.quiet()

        if ce_iceberg > pe_iceberg:
            direction = 1     # iceberg on CE → upside bid
            classification = "ce_iceberg"
            count = ce_iceberg
        elif pe_iceberg > ce_iceberg:
            direction = -1
            classification = "pe_iceberg"
            count = pe_iceberg
        else:
            return DetectorPosterior.quiet()

        avg_recurrences = (sum(magnitudes) / len(magnitudes)
                            if magnitudes else float(self.min_recurrences))
        targeted = (sum(targeted_strikes) / len(targeted_strikes)
                     if targeted_strikes else None)
        probability = min(0.80, 0.45 + 0.06 * (
            avg_recurrences - self.min_recurrences))
        confidence = min(0.78, 0.40 + 0.08 * (count - 1)
                          + 0.04 * (avg_recurrences
                                      - self.min_recurrences))
        magnitude = DetectorMagnitude(
            z_score=float(avg_recurrences / max(1, self.min_recurrences)),
            raw_value=float(avg_recurrences),
            units="recurrent fills")
        return DetectorPosterior(
            fire=True, probability=probability,
            direction=direction, confidence=confidence,
            horizon_bars=12,
            targeted_strike=(float(targeted)
                              if targeted is not None else None),
            magnitude=magnitude,
            evidence=[
                f"{count} slots show iceberg pattern "
                f"({classification})",
                f"avg recurrences per slot: {avg_recurrences:.1f} "
                f"(threshold {self.min_recurrences})",
                f"persistence ≥ {self.min_persistence_score:.2f}",
            ],
            classification=classification,
        )
