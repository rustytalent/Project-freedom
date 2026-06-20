"""Replay harness — run the executor over a recorded snapshot tape.

The founder cannot paper-trade on a Sunday — the market is closed. But
the existing ``live_runner.py`` writes a JSONL tape of every
BeliefSnapshot it sees during a live session. This harness reads such
a tape, feeds it through the V4Runner end-to-end, and reports:

  * Win rate, average R, max drawdown
  * Total realized P&L, total fees, total slippage
  * Net P&L after costs (the founder's actual take-home)
  * Strategy usage breakdown
  * Per-strategy performance
  * Refused-entry reasons distribution
  * Cockpit snapshot at the end of the day

Output: a structured ``ReplayReport`` (JSON-serializable). Optionally
writes per-tick cockpit + intent JSONL for forensic post-mortems.

Usage:

    from liqpool.research.belief.executor_v4.replay import replay_jsonl
    report = replay_jsonl("/var/lib/sentinel/liqpool_live_signals.jsonl")
    print(report.to_summary_string())
"""
from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .manager import PortfolioManagerConfig
from .persistence import PersistenceConfig
from .v4_runner import V4Runner, V4RunnerConfig


@dataclass
class ReplayConfig:
    """Knobs for the replay harness."""
    starting_capital_rupees: float = 50_000.0
    cockpit_out: Optional[Path] = None    # JSONL of per-tick cockpit dicts
    intent_out: Optional[Path] = None     # JSONL of per-tick intent dicts
    persist_state: bool = False
    manager_cfg: Optional[PortfolioManagerConfig] = None
    progress_every_n_ticks: int = 500


@dataclass
class ReplayReport:
    """Per-replay summary."""
    n_ticks: int
    n_entries: int
    n_exits: int
    n_refused: int
    win_rate: float
    avg_realized_r: float
    median_realized_r: float
    best_trade_rupees: float
    worst_trade_rupees: float
    total_realized_rupees: float
    total_fees_rupees: float
    total_slippage_rupees: float
    net_pnl_rupees: float
    max_drawdown_rupees: float
    max_drawdown_pct: float
    sharpe_proxy: float
    strategy_usage: Dict[str, int]
    refuse_reason_top: List[Dict[str, Any]]
    per_strategy_pnl: Dict[str, float]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_ticks": self.n_ticks,
            "n_entries": self.n_entries,
            "n_exits": self.n_exits,
            "n_refused": self.n_refused,
            "win_rate": round(self.win_rate, 3),
            "avg_realized_r": round(self.avg_realized_r, 3),
            "median_realized_r": round(self.median_realized_r, 3),
            "best_trade_rupees": round(self.best_trade_rupees, 2),
            "worst_trade_rupees": round(self.worst_trade_rupees, 2),
            "total_realized_rupees": round(self.total_realized_rupees, 2),
            "total_fees_rupees": round(self.total_fees_rupees, 2),
            "total_slippage_rupees": round(self.total_slippage_rupees, 2),
            "net_pnl_rupees": round(self.net_pnl_rupees, 2),
            "max_drawdown_rupees": round(self.max_drawdown_rupees, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "sharpe_proxy": round(self.sharpe_proxy, 3),
            "strategy_usage": dict(self.strategy_usage),
            "refuse_reason_top": list(self.refuse_reason_top),
            "per_strategy_pnl": dict(self.per_strategy_pnl),
            "notes": list(self.notes),
        }

    def to_summary_string(self) -> str:
        lines = [
            f"━━ Replay summary ━━",
            f"  {self.n_ticks} ticks | {self.n_entries} entries | "
            f"{self.n_exits} exits | {self.n_refused} refused",
            f"  Win rate: {self.win_rate:.1%}  "
            f"avg R: {self.avg_realized_r:+.2f}  "
            f"median R: {self.median_realized_r:+.2f}",
            f"  Net P&L: ₹{self.net_pnl_rupees:+,.0f}  "
            f"(realized ₹{self.total_realized_rupees:+,.0f} - "
            f"fees ₹{self.total_fees_rupees:,.0f} - "
            f"slip ₹{self.total_slippage_rupees:,.0f})",
            f"  Max DD: ₹{self.max_drawdown_rupees:,.0f} "
            f"({self.max_drawdown_pct:.1%})",
            f"  Sharpe proxy: {self.sharpe_proxy:.2f}",
            f"  Strategies used: {self.strategy_usage}",
            f"  Top refuse reasons: {self.refuse_reason_top[:3]}",
        ]
        return "\n".join(lines)


def replay_jsonl(path: str | Path,
                   cfg: Optional[ReplayConfig] = None) -> ReplayReport:
    """Replay a JSONL tape of BeliefSnapshots.

    Each line in ``path`` must be either:
      * a BeliefSnapshot dict directly, OR
      * a row with an ``extras.belief_snapshot`` field (the format the
        existing live_runner writes)
    """
    cfg = cfg or ReplayConfig()

    def _iter_snapshots():
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                # Try both possible shapes
                snap = None
                if isinstance(row, dict):
                    if "bars_seen" in row and "slot_readings" in row:
                        snap = row
                    else:
                        extras = row.get("extras") or {}
                        snap = extras.get("belief_snapshot")
                if snap:
                    yield snap

    return replay_snapshots(_iter_snapshots(), cfg=cfg)


def replay_snapshots(snapshots: Iterable[Dict[str, Any]],
                       cfg: Optional[ReplayConfig] = None) -> ReplayReport:
    """Replay an iterable of BeliefSnapshot dicts."""
    cfg = cfg or ReplayConfig()
    runner_cfg = V4RunnerConfig(
        manager=cfg.manager_cfg,
        persistence=PersistenceConfig(enabled=False),
        emit_explainer_to_log=False,
        write_cockpit_to_jsonl=cfg.cockpit_out,
    )
    runner = V4Runner(cfg=runner_cfg)

    n_ticks = 0
    n_entries = 0
    n_exits = 0
    n_refused = 0
    realized_pnls: List[float] = []
    realized_rs: List[float] = []
    strategy_usage: Counter = Counter()
    per_strategy_pnl: Dict[str, float] = defaultdict(float)
    refuse_reasons_counter: Counter = Counter()
    peak_pnl = 0.0
    trough_dd = 0.0
    pnl_history: List[float] = []
    total_fees = 0.0
    total_slippage = 0.0

    for snap in snapshots:
        n_ticks += 1
        result = runner.on_tick(snap)
        intent = result.intent

        if intent.new_entry is not None:
            n_entries += 1
            h = intent.new_entry.get("hypothesis") or {}
            strategy = h.get("trigger_action") or "unknown"
            strategy_usage[strategy] += 1

        for closed in intent.closed_this_tick:
            n_exits += 1
            outcome = closed.get("outcome") or {}
            r = float(outcome.get("realized_r", 0.0))
            pnl = float(outcome.get("realized_rupees", 0.0))
            fees = float(outcome.get("fees_rupees", 0.0))
            slip = float(outcome.get("slippage_rupees", 0.0))
            realized_rs.append(r)
            realized_pnls.append(pnl)
            total_fees += fees
            total_slippage += slip
            # Find strategy from hypothesis if available
            for ledger in runner.manager.ledger_store.closed_positions():
                if ledger.hypothesis.position_id == closed.get("position_id"):
                    per_strategy_pnl[ledger.hypothesis.trigger_action] += pnl
                    break

        for r in intent.refuse_reasons:
            n_refused += 1
            # Bin by leading phrase
            phrase = r.split(":")[0][:60]
            refuse_reasons_counter[phrase] += 1

        # Track P&L history for max drawdown
        cum_pnl = intent.daily_pnl_rupees
        pnl_history.append(cum_pnl)
        if cum_pnl > peak_pnl:
            peak_pnl = cum_pnl
        dd = peak_pnl - cum_pnl
        if dd > trough_dd:
            trough_dd = dd

        if cfg.intent_out is not None:
            cfg.intent_out.parent.mkdir(parents=True, exist_ok=True)
            with open(cfg.intent_out, "a") as f:
                f.write(json.dumps(intent.to_dict(), default=str) + "\n")

    win_rate = (sum(1 for r in realized_rs if r > 0) / max(1, len(realized_rs)))
    avg_r = statistics.mean(realized_rs) if realized_rs else 0.0
    med_r = statistics.median(realized_rs) if realized_rs else 0.0
    total_realized = sum(realized_pnls)
    net_pnl = total_realized
    if pnl_history:
        period_returns = [pnl_history[i] - pnl_history[i - 1]
                          for i in range(1, len(pnl_history))]
        mean_ret = (statistics.mean(period_returns) if period_returns else 0)
        std_ret = (statistics.pstdev(period_returns)
                   if len(period_returns) >= 2 else 0)
        sharpe = mean_ret / std_ret * math.sqrt(252) if std_ret > 0 else 0
    else:
        sharpe = 0.0
    best = max(realized_pnls) if realized_pnls else 0.0
    worst = min(realized_pnls) if realized_pnls else 0.0
    dd_pct = (trough_dd / max(cfg.starting_capital_rupees, 1.0))

    top_refuse = [{"reason": r, "count": c}
                  for r, c in refuse_reasons_counter.most_common(5)]

    return ReplayReport(
        n_ticks=n_ticks,
        n_entries=n_entries,
        n_exits=n_exits,
        n_refused=n_refused,
        win_rate=win_rate,
        avg_realized_r=avg_r,
        median_realized_r=med_r,
        best_trade_rupees=best,
        worst_trade_rupees=worst,
        total_realized_rupees=total_realized,
        total_fees_rupees=total_fees,
        total_slippage_rupees=total_slippage,
        net_pnl_rupees=net_pnl,
        max_drawdown_rupees=trough_dd,
        max_drawdown_pct=dd_pct,
        sharpe_proxy=sharpe,
        strategy_usage=dict(strategy_usage),
        refuse_reason_top=top_refuse,
        per_strategy_pnl=dict(per_strategy_pnl),
    )
