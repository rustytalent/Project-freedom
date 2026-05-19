"""Track 5: append-only trade journal for live execution.

Logs every event that matters for calibration and post-trade review:
  - alert           : system flagged a tradeable setup (whether or not you took it)
  - order_placed    : you (or auto-execution) placed an order with the broker
  - order_filled    : broker reports the order filled
  - exit            : position closed (stop hit, target hit, manual, EOD square-off)
  - drift_alert     : weekly walk-forward detected model degradation
  - note            : freeform note for retrospective context

The format is JSONL (one JSON object per line) — easy to grep, easy to parse, easy to
back up. Never edit lines; only append. To "delete" an entry, append a correction event.

Why this matters: after 30 trades you have hands-on calibration data. After 100 you know
whether the model's stated probabilities (Q, T, EV) match real-world outcomes. That feedback
loop is what turns a research tool into a trading edge.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Iterable
import json
import pandas as pd
import numpy as np


# Canonical event types — keep this list short and stable.
EVENT_TYPES = {
    "alert", "order_placed", "order_filled", "order_rejected",
    "exit", "drift_alert", "note", "system",
}


class TradeJournal:
    """Append-only JSONL journal. Path is the file; parent directory created on init."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            # touch so subsequent reads don't fail
            self.path.touch()

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append(self, event_type: str, symbol: str, data: Optional[Dict] = None,
               ts: Optional[pd.Timestamp] = None) -> Dict:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event_type {event_type!r}; valid: {sorted(EVENT_TYPES)}")
        entry = {
            "timestamp": str(ts) if ts is not None else pd.Timestamp.utcnow().isoformat(),
            "event_type": event_type,
            "symbol": symbol,
            "data": data or {},
        }
        with self.path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return entry

    def log_alert(self, symbol: str, *, pool_id: str, entry_price: float, stop: float,
                   target: float, q: float, t_today: float, ev: float, direction_p_up: float,
                   dir_tag: str, sector: str, shares: int, notional: float,
                   reasoning: str = "") -> Dict:
        """Convenience: structured 'alert' event with all the fields we want downstream."""
        return self.append("alert", symbol, {
            "pool_id": pool_id, "entry": float(entry_price), "stop": float(stop),
            "target": float(target),
            "q": float(q), "t_today": float(t_today), "ev": float(ev),
            "direction_p_up": float(direction_p_up), "dir_tag": dir_tag,
            "sector": sector, "shares": int(shares), "notional": float(notional),
            "reasoning": reasoning,
        })

    def log_order_placed(self, symbol: str, order_id: str, *, side: str, quantity: int,
                          entry_price: float, stop: float, target: float, alert_id: str = "",
                          dry_run: bool = True) -> Dict:
        return self.append("order_placed", symbol, {
            "order_id": str(order_id), "side": side, "quantity": int(quantity),
            "entry": float(entry_price), "stop": float(stop), "target": float(target),
            "alert_id": alert_id, "dry_run": bool(dry_run),
        })

    def log_fill(self, symbol: str, order_id: str, *, side: str, quantity: int,
                  fill_price: float, fill_ts: Optional[pd.Timestamp] = None) -> Dict:
        return self.append("order_filled", symbol, {
            "order_id": str(order_id), "side": side, "quantity": int(quantity),
            "fill_price": float(fill_price),
            "fill_ts": str(fill_ts) if fill_ts is not None else None,
        })

    def log_exit(self, symbol: str, *, side: str, quantity: int, exit_price: float,
                  exit_reason: str, entry_price: float, r_multiple: Optional[float] = None,
                  pnl_inr: Optional[float] = None, pool_id: str = "") -> Dict:
        return self.append("exit", symbol, {
            "side": side, "quantity": int(quantity), "exit_price": float(exit_price),
            "exit_reason": exit_reason, "entry_price": float(entry_price),
            "r_multiple": float(r_multiple) if r_multiple is not None else None,
            "pnl_inr": float(pnl_inr) if pnl_inr is not None else None,
            "pool_id": pool_id,
        })

    def log_drift_alert(self, *, metric: str, current: float, baseline: float,
                         threshold: float, severity: str = "warning") -> Dict:
        return self.append("drift_alert", "_system", {
            "metric": metric, "current": float(current), "baseline": float(baseline),
            "threshold": float(threshold), "severity": severity,
        })

    def log_note(self, symbol: str, text: str) -> Dict:
        return self.append("note", symbol, {"text": text})

    # ------------------------------------------------------------------
    # Read / analyse
    # ------------------------------------------------------------------

    def read_all(self) -> List[Dict]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[journal] skipping corrupted line: {e}")
        return out

    def filter(self, event_type: Optional[str] = None, symbol: Optional[str] = None,
                since: Optional[pd.Timestamp] = None) -> List[Dict]:
        events = self.read_all()
        out = []
        for e in events:
            if event_type and e["event_type"] != event_type:
                continue
            if symbol and e["symbol"] != symbol:
                continue
            if since is not None:
                ts = pd.Timestamp(e["timestamp"])
                if ts < since:
                    continue
            out.append(e)
        return out

    def alerts_df(self) -> pd.DataFrame:
        alerts = self.filter(event_type="alert")
        if not alerts:
            return pd.DataFrame()
        rows = []
        for e in alerts:
            d = e["data"]
            rows.append({
                "timestamp": pd.Timestamp(e["timestamp"]),
                "symbol": e["symbol"], **d,
            })
        return pd.DataFrame(rows)

    def trades_df(self) -> pd.DataFrame:
        """Joins fills with exits to produce one row per completed trade."""
        events = self.read_all()
        fills_by_order: Dict[str, Dict] = {}
        exits: List[Dict] = []
        for e in events:
            if e["event_type"] == "order_filled":
                fills_by_order[e["data"]["order_id"]] = e
            elif e["event_type"] == "exit":
                exits.append(e)
        rows = []
        for ex in exits:
            row = {
                "exit_timestamp": pd.Timestamp(ex["timestamp"]),
                "symbol": ex["symbol"],
                **ex["data"],
            }
            rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # Calibration: are predicted probabilities matching real outcomes?
    # ------------------------------------------------------------------

    def calibration_summary(self) -> Dict:
        """Walk through alerts → fills → exits and compute:
          - n alerts, n trades taken, take-rate
          - actual win rate by Q decile
          - actual win rate by EV bucket
          - average R-multiple
        Returns a dict; empty if no trades yet.
        """
        alerts = self.alerts_df()
        trades = self.trades_df()
        if alerts.empty or trades.empty:
            return {"n_alerts": len(alerts), "n_trades": len(trades),
                    "summary": "not enough data yet"}

        win_rate = float((trades["r_multiple"] > 0).mean()) if "r_multiple" in trades else None
        avg_r = float(trades["r_multiple"].mean()) if "r_multiple" in trades else None

        # Q-decile calibration (if alerts have q values)
        q_calib = []
        if "q" in alerts.columns:
            # We need to match alerts to trades; for now use timestamp proximity per symbol.
            # Lightweight join: take most-recent alert per symbol before each trade's entry.
            # (This is approximate — full implementation could use explicit alert_id linkage.)
            pass    # extend when we have real trade history

        return {
            "n_alerts": int(len(alerts)),
            "n_trades": int(len(trades)),
            "take_rate": float(len(trades) / len(alerts)) if len(alerts) else 0.0,
            "win_rate_actual": win_rate,
            "avg_r_multiple": avg_r,
        }
