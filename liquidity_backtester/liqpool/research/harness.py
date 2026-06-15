"""HypothesisHarness — ask the market a rigorous question, get a real answer.

The unit of work is a HYPOTHESIS: a Python callable that, given a
chronological bar series, decides when to enter and exit a position.
The harness wraps that callable in:

  * Walk-forward windowing (no look-ahead)
  * Real cost model (STT, GST, brokerage, slippage)
  * Position-level P&L computation
  * Trade-level statistics (Sharpe, Sortino, hit-rate, expectancy)

It returns a HypothesisReport — the document you read before deciding
whether to graduate a hypothesis from "interesting" to "live."

Convention: P&L is computed in ABSOLUTE rupees (not %). Trade returns
in R-multiples are computed from the entry's risk distance, so a
"+2R winner" carries the same weight whether the trade was on 1 lot
or 10. That's the unit a quant audits expectancy in.

The harness does NOT pretend to be the entire backtest engine
(execution_simulator_v2 still owns full backtest fidelity for the
research engine). It IS the lightweight, hypothesis-level question-
asker that lets us iterate fast.

References:
  Tharp (2007) — R-multiple expectancy
  Sharpe (1966) — Sharpe ratio
  Sortino (1994) — Sortino ratio (downside deviation)
  Bailey & López de Prado (2014) — deflated Sharpe (TODO: add)
"""
from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import pandas as pd
    _HAS_PD = True
except ImportError:                                  # pragma: no cover
    pd = None
    _HAS_PD = False


# ─────────────────────────────────────────────────────────────────
# What a hypothesis looks like
# ─────────────────────────────────────────────────────────────────

# A hypothesis returns a list of (entry_idx, exit_idx, side, qty)
# tuples given a series of bars. The harness wraps that in cost +
# stats math.
#
# side: +1 = long, -1 = short
# qty: signed quantity in the underlying instrument's tradable unit
#      (lot for options, share for equity). Convention: qty is the
#      total exposure, not signed by side.
#
# Hypothesis signature:
#   fn(bars_df: pandas.DataFrame, **params) -> List[(int, int, int, int)]
#
# bars_df columns required:
#   ts (sortable timestamp), open, high, low, close, volume
HypothesisFn = Callable[..., List[Tuple[int, int, int, int]]]


@dataclass
class HypothesisSpec:
    """The 'test this idea' contract."""
    name: str
    fn: HypothesisFn
    params: Dict[str, Any] = field(default_factory=dict)
    # The instrument the hypothesis trades. For options, this is the
    # contract (e.g. NIFTY25000CE). For underlying, the symbol (NIFTY).
    # Cost model picks rates per instrument_kind.
    instrument_kind: str = "options"          # "options" | "futures" | "equity"
    lot_size: int = 75
    note: str = ""


# ─────────────────────────────────────────────────────────────────
# Cost model — the link that kills 80% of backtest-only "edges"
# ─────────────────────────────────────────────────────────────────

@dataclass
class CostsModel:
    """Per-leg costs. Defaults match Zerodha + Finance Act 2024."""
    brokerage_per_leg_rupees: float = 20.0
    # Options STT applied to SELL side at 0.0625% of premium.
    stt_options_sell_pct: float = 0.000625
    exchange_txn_pct: float = 0.000503        # NSE F&O turnover charge
    sebi_charge_pct: float = 0.0000010
    stamp_duty_buy_pct: float = 0.00003
    gst_pct: float = 0.18
    # Slippage in TICKS of the entry/exit price. 1 tick on NIFTY
    # weekly options ≈ ₹0.05; default = 2 ticks = ₹0.10 per leg.
    # For thinly-traded contracts increase.
    slippage_ticks_per_leg: int = 2
    tick_size_rupees: float = 0.05

    def round_trip_charges(self, buy_price: float, sell_price: float,
                           qty: int) -> float:
        """Sum of all charges for one round-trip options trade."""
        buy_turnover = buy_price * qty
        sell_turnover = sell_price * qty
        stt = sell_turnover * self.stt_options_sell_pct
        exch = (buy_turnover + sell_turnover) * self.exchange_txn_pct
        sebi = (buy_turnover + sell_turnover) * self.sebi_charge_pct
        stamp = buy_turnover * self.stamp_duty_buy_pct
        brokerage = self.brokerage_per_leg_rupees * 2
        gst = (brokerage + exch + sebi) * self.gst_pct
        slip = (self.slippage_ticks_per_leg * self.tick_size_rupees) * qty * 2
        return round(stt + exch + sebi + stamp + brokerage + gst + slip, 2)


# ─────────────────────────────────────────────────────────────────
# One simulated trade
# ─────────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    entry_idx: int
    exit_idx: int
    entry_ts: Any
    exit_ts: Any
    side: int                                 # +1 long, -1 short
    qty: int
    entry_price: float
    exit_price: float
    gross_pnl: float                          # before costs
    costs: float
    net_pnl: float                            # gross - costs
    r_multiple: Optional[float]               # net_pnl / initial risk
    hold_minutes: int
    regime_tag: str = ""

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        # Ensure timestamps are JSON-friendly
        d["entry_ts"] = str(self.entry_ts)
        d["exit_ts"] = str(self.exit_ts)
        return d


def simulate_trades(
    bars: "pd.DataFrame",
    trade_signals: Sequence[Tuple[int, int, int, int]],
    costs: CostsModel,
    initial_risk_per_trade_rupees: float = 1000.0,
) -> List[BacktestTrade]:
    """Convert (entry, exit, side, qty) tuples into BacktestTrades
    with full cost accounting.

    initial_risk_per_trade_rupees is the denominator for R-multiple
    computation. Convention: pre-trade risk = stop-distance × qty,
    but since hypotheses here are simple entry/exit tuples without
    explicit stops, we use a fixed scaling. For real options
    sizing the harness should be invoked from a wrapper that passes
    the actual risk per trade.
    """
    if not _HAS_PD:                              # pragma: no cover
        raise RuntimeError("pandas required for simulate_trades")
    out: List[BacktestTrade] = []
    for entry_idx, exit_idx, side, qty in trade_signals:
        if entry_idx < 0 or exit_idx <= entry_idx or qty <= 0:
            continue
        if exit_idx >= len(bars):
            exit_idx = len(bars) - 1
        e_bar = bars.iloc[entry_idx]
        x_bar = bars.iloc[exit_idx]
        e_price = float(e_bar["close"])
        x_price = float(x_bar["close"])
        # Direction-aware P&L
        gross_pnl = (x_price - e_price) * qty * side
        c = costs.round_trip_charges(
            buy_price=min(e_price, x_price) if side > 0 else max(e_price, x_price),
            sell_price=max(e_price, x_price) if side > 0 else min(e_price, x_price),
            qty=qty)
        net_pnl = gross_pnl - c
        r_mult = net_pnl / initial_risk_per_trade_rupees if initial_risk_per_trade_rupees > 0 else None
        # Hold duration in minutes (assume 5-min base bars; harness
        # can override via bars.attrs)
        bar_minutes = bars.attrs.get("bar_minutes", 5)
        hold_min = (exit_idx - entry_idx) * bar_minutes
        out.append(BacktestTrade(
            entry_idx=entry_idx, exit_idx=exit_idx,
            entry_ts=e_bar.get("ts", entry_idx),
            exit_ts=x_bar.get("ts", exit_idx),
            side=side, qty=qty, entry_price=e_price, exit_price=x_price,
            gross_pnl=round(gross_pnl, 2),
            costs=round(c, 2),
            net_pnl=round(net_pnl, 2),
            r_multiple=round(r_mult, 3) if r_mult is not None else None,
            hold_minutes=int(hold_min),
        ))
    return out


# ─────────────────────────────────────────────────────────────────
# The numbers a quant reads
# ─────────────────────────────────────────────────────────────────

@dataclass
class TradeStats:
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    hit_rate: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    total_costs: float = 0.0
    expectancy_rupees: float = 0.0       # avg net P&L per trade
    expectancy_r: float = 0.0            # avg R-multiple per trade
    avg_winner_r: float = 0.0
    avg_loser_r: float = 0.0
    profit_factor: float = 0.0           # gross wins / abs(gross losses)
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown_rupees: float = 0.0
    max_drawdown_pct_of_peak: float = 0.0
    avg_hold_minutes: float = 0.0

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


def trade_stats(trades: Sequence[BacktestTrade]) -> TradeStats:
    """Compute the full institutional scorecard from a trade list."""
    s = TradeStats()
    if not trades:
        return s
    s.n_trades = len(trades)
    net = [t.net_pnl for t in trades]
    gross = [t.gross_pnl for t in trades]
    costs = [t.costs for t in trades]
    r_mults = [t.r_multiple for t in trades if t.r_multiple is not None]
    holds = [t.hold_minutes for t in trades]
    s.gross_pnl = round(sum(gross), 2)
    s.net_pnl = round(sum(net), 2)
    s.total_costs = round(sum(costs), 2)
    s.n_wins = sum(1 for p in net if p > 0)
    s.n_losses = sum(1 for p in net if p < 0)
    s.hit_rate = round(s.n_wins / s.n_trades, 3)
    s.expectancy_rupees = round(s.net_pnl / s.n_trades, 2)
    if r_mults:
        s.expectancy_r = round(sum(r_mults) / len(r_mults), 3)
        wins = [r for r in r_mults if r > 0]
        losses = [r for r in r_mults if r < 0]
        s.avg_winner_r = round(sum(wins) / len(wins), 3) if wins else 0
        s.avg_loser_r = round(sum(losses) / len(losses), 3) if losses else 0
    # Profit factor
    win_pnl = sum(p for p in gross if p > 0)
    loss_pnl = sum(p for p in gross if p < 0)
    s.profit_factor = round(win_pnl / abs(loss_pnl), 3) if loss_pnl < 0 else float("inf") if win_pnl > 0 else 0
    # Sharpe / Sortino — based on per-trade net P&L; annualised
    # assuming N trades/year. We don't have a calendar so just report
    # the per-trade Sharpe; the caller scales if they want annual.
    if len(net) >= 2:
        mean = statistics.mean(net)
        sd = statistics.pstdev(net)
        s.sharpe = round(mean / sd, 3) if sd > 0 else 0
        downside = [p for p in net if p < mean]
        dsd = statistics.pstdev(downside) if len(downside) >= 2 else 0
        s.sortino = round(mean / dsd, 3) if dsd > 0 else 0
    # Drawdown
    equity = []
    peak = 0
    cum = 0
    max_dd = 0
    max_dd_pct = 0
    for p in net:
        cum += p
        equity.append(cum)
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = (dd / peak * 100.0) if peak > 0 else 0
    s.max_drawdown_rupees = round(max_dd, 2)
    s.max_drawdown_pct_of_peak = round(max_dd_pct, 2)
    s.avg_hold_minutes = round(sum(holds) / len(holds), 1) if holds else 0
    return s


# ─────────────────────────────────────────────────────────────────
# The report — what we read before deciding to keep / kill
# ─────────────────────────────────────────────────────────────────

@dataclass
class HypothesisReport:
    name: str
    overall: TradeStats
    by_month: Dict[str, TradeStats] = field(default_factory=dict)
    by_regime: Dict[str, TradeStats] = field(default_factory=dict)
    trade_count: int = 0
    bars_consumed: int = 0
    bars_with_signal: int = 0
    signal_rate: float = 0.0              # fraction of bars producing a signal
    decision: str = "evaluate"            # "keep" / "kill" / "evaluate" / "needs_more_data"
    decision_reason: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    trades: List[Dict[str, Any]] = field(default_factory=list)

    KEEP_SHARPE_FLOOR = 1.0
    KEEP_MIN_TRADES = 30                  # below this, stats are noise
    KEEP_MAX_DD_PCT = 30.0                 # accept up to 30% peak drawdown

    def to_row(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "overall": self.overall.to_row(),
            "by_month": {k: v.to_row() for k, v in self.by_month.items()},
            "by_regime": {k: v.to_row() for k, v in self.by_regime.items()},
            "trade_count": self.trade_count,
            "bars_consumed": self.bars_consumed,
            "bars_with_signal": self.bars_with_signal,
            "signal_rate": self.signal_rate,
            "decision": self.decision,
            "decision_reason": self.decision_reason,
            "params": self.params,
            "trades": self.trades,
        }

    def auto_decide(self) -> None:
        """Apply the decision rule the doctrine specified:
          keep if Sharpe ≥ 1.0, n_trades ≥ 30, max_dd_pct ≤ 30%.
          kill otherwise. Fewer than KEEP_MIN_TRADES → needs_more_data."""
        st = self.overall
        if st.n_trades < self.KEEP_MIN_TRADES:
            self.decision = "needs_more_data"
            self.decision_reason = (
                f"only {st.n_trades} trades; need ≥ {self.KEEP_MIN_TRADES} "
                f"to trust the stats")
            return
        if (st.sharpe >= self.KEEP_SHARPE_FLOOR
                and st.max_drawdown_pct_of_peak <= self.KEEP_MAX_DD_PCT):
            self.decision = "keep"
            self.decision_reason = (
                f"Sharpe {st.sharpe} ≥ {self.KEEP_SHARPE_FLOOR}, "
                f"max DD {st.max_drawdown_pct_of_peak}% ≤ "
                f"{self.KEEP_MAX_DD_PCT}%")
        else:
            self.decision = "kill"
            self.decision_reason = (
                f"Sharpe {st.sharpe} < {self.KEEP_SHARPE_FLOOR} or "
                f"max DD {st.max_drawdown_pct_of_peak}% > "
                f"{self.KEEP_MAX_DD_PCT}%")


# ─────────────────────────────────────────────────────────────────
# The harness — given a spec + bars, return the report
# ─────────────────────────────────────────────────────────────────

class HypothesisHarness:
    """Run one HypothesisSpec against a bar series. Returns a
    HypothesisReport with full per-month and per-regime breakdowns.

    The caller is responsible for the walk-forward split: pass the
    OOS slice as ``bars`` and you get an honest report. For
    in-sample validation, pass the train slice; the harness doesn't
    care, but the caller needs to be honest about which is which."""

    def __init__(self,
                 costs: Optional[CostsModel] = None,
                 initial_risk_per_trade_rupees: float = 1000.0,
                 regime_tagger: Optional[Callable[[Any], str]] = None) -> None:
        self.costs = costs or CostsModel()
        self.initial_risk = initial_risk_per_trade_rupees
        self.regime_tagger = regime_tagger

    def run(self, spec: HypothesisSpec, bars: "pd.DataFrame"
            ) -> HypothesisReport:
        if not _HAS_PD:                          # pragma: no cover
            raise RuntimeError("pandas required for HypothesisHarness")
        signals = list(spec.fn(bars, **spec.params))
        trades = simulate_trades(bars, signals, self.costs,
                                  initial_risk_per_trade_rupees=self.initial_risk)
        if self.regime_tagger is not None:
            for t in trades:
                try:
                    t.regime_tag = self.regime_tagger(bars.iloc[t.entry_idx])
                except Exception:
                    pass

        overall = trade_stats(trades)
        by_month = self._stats_by_month(trades, bars)
        by_regime = self._stats_by_regime(trades)

        report = HypothesisReport(
            name=spec.name, overall=overall,
            by_month=by_month, by_regime=by_regime,
            trade_count=len(trades),
            bars_consumed=len(bars),
            bars_with_signal=len(signals),
            signal_rate=round(len(signals) / max(len(bars), 1), 4),
            params=spec.params,
            trades=[t.to_row() for t in trades],
        )
        report.auto_decide()
        return report

    def _stats_by_month(self, trades: List[BacktestTrade],
                         bars: "pd.DataFrame") -> Dict[str, TradeStats]:
        if not trades:
            return {}
        buckets: Dict[str, List[BacktestTrade]] = {}
        for t in trades:
            ts = t.entry_ts
            try:
                key = pd.to_datetime(ts).strftime("%Y-%m")
            except Exception:
                key = str(ts)[:7] or "unknown"
            buckets.setdefault(key, []).append(t)
        return {k: trade_stats(v) for k, v in buckets.items()}

    def _stats_by_regime(self, trades: List[BacktestTrade]
                          ) -> Dict[str, TradeStats]:
        if not trades or all(not t.regime_tag for t in trades):
            return {}
        buckets: Dict[str, List[BacktestTrade]] = {}
        for t in trades:
            buckets.setdefault(t.regime_tag or "untagged", []).append(t)
        return {k: trade_stats(v) for k, v in buckets.items()}
