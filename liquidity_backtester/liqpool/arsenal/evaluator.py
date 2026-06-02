"""Arsenal evaluator: shared MIS + cost + barrier pipeline for all alphas.

Every alpha emits AlphaSignal candidates. The evaluator runs them through
the same execution pipeline so they're judged on a level playing field:

  1. Reject signals outside the NSE session or after NO_NEW_ENTRY_AFTER_IST_MIN
     (14:30 IST) — every alpha lives under the same MIS constraint.
  2. Enter at the next bar's open (mirrors V1 simulator's confirmation modes).
  3. Compute realized R via :func:`liqpool.triple_barrier.triple_barrier_label`
     with the alpha's preferred (stop, target, horizon). EOD square-off is
     baked into the label.
  4. Apply Zerodha intraday costs + flat slippage to derive net R.
  5. Record per-trade regime tags from alpha.regime_tags().
  6. Aggregate per-alpha, per-regime, and per-combination metrics with 95% CI.

The point: no alpha can hide behind generous assumptions. If alpha X looks
better than alpha Y, it's because X's signals genuinely produce more
realized R after identical costs and constraints.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from liqpool.costs import ZerodhaEquityCostConfig, estimate_round_trip_charges
from liqpool.execution_backtest import (
    NO_NEW_ENTRY_AFTER_IST_MIN,
    SESSION_CLOSE_IST_MIN,
    SESSION_OPEN_IST_MIN,
    _ist_minute_of_day,
)
from liqpool.indicators import atr
from liqpool.sectors import sector_of
from liqpool.triple_barrier import triple_barrier_label

from .base import Alpha, AlphaSignal


# Columns produced by the evaluator. Documented here so downstream
# analyses (regime aggregation, null tests) can rely on them.
TRADE_COLUMNS: Tuple[str, ...] = (
    "alpha_name", "symbol", "sector",
    "decision_at", "decision_idx", "entry_at", "exit_at",
    "side", "entry_reference", "entry_price", "exit_price",
    "stop_atr", "target_atr", "horizon_bars",
    "confidence",
    "gross_r", "cost_inr", "risk_inr",
    "net_r", "net_pnl", "win",
    "exit_reason", "bars_held",
)


@dataclass
class EvaluatorConfig:
    """Per-evaluator settings. Defaults match the v1 simulator we trust."""
    cost_cfg: ZerodhaEquityCostConfig = field(default_factory=ZerodhaEquityCostConfig)
    notional_inr: float = 50_000.0      # default position size (₹50k)
    atr_period: int = 14
    slippage_bps_per_side: float = 1.0  # flat bps slippage on entry + exit

    def slippage_fraction(self) -> float:
        return float(self.slippage_bps_per_side) / 10_000.0


# ---------------------------------------------------------------------------
# Single-trade execution
# ---------------------------------------------------------------------------

def _execute_signal(signal: AlphaSignal,
                    df_base: pd.DataFrame,
                    atr_series: pd.Series,
                    sector: str,
                    config: EvaluatorConfig,
                    ) -> Optional[Dict[str, Any]]:
    """Run a single AlphaSignal through the shared pipeline. Returns a
    dict matching :data:`TRADE_COLUMNS` or None if the signal was filtered
    out (outside session, late-entry refusal, no same-day room)."""
    idx = df_base.index
    if signal.decision_idx < 0 or signal.decision_idx >= len(idx):
        return None

    # Entry on NEXT bar's open (no look-ahead).
    entry_idx = signal.decision_idx + 1
    if entry_idx >= len(idx):
        return None

    entry_ts = idx[entry_idx]
    entry_min = _ist_minute_of_day(entry_ts)
    if not (SESSION_OPEN_IST_MIN <= entry_min <= SESSION_CLOSE_IST_MIN):
        return None
    if entry_min > NO_NEW_ENTRY_AFTER_IST_MIN:
        return None

    open_price = float(df_base["open"].iloc[entry_idx])
    # Apply flat bps slippage in the adverse direction at entry.
    slip = config.slippage_fraction()
    if signal.side == "long":
        entry_price = open_price * (1 + slip)
    else:
        entry_price = open_price * (1 - slip)

    atr_val = float(atr_series.iloc[entry_idx])
    if atr_val <= 0 or not math.isfinite(atr_val):
        return None

    r, exit_reason, exit_idx = triple_barrier_label(
        df_base, entry_idx=entry_idx, side=signal.side,
        entry_price=entry_price,
        stop_atr=signal.stop_atr, target_atr=signal.target_atr,
        horizon_bars=signal.horizon_bars, atr_value=atr_val,
    )
    if exit_reason == "no_room":
        return None

    # Translate gross R to gross INR by sizing notional/entry_price.
    qty = max(1, int(config.notional_inr / max(entry_price, 1e-9)))
    risk_per_share = float(signal.stop_atr) * atr_val
    risk_inr = risk_per_share * qty
    gross_pnl = r * risk_inr

    # Realistic exit slippage: shift exit price by slip in the adverse
    # direction relative to side.
    if signal.side == "long":
        # Reconstructing approximate exit price for cost computation; for net_r
        # we use the gross_pnl-based delta. Exit fill at the barrier suffers
        # negative slippage.
        exit_price = entry_price + (r * risk_per_share) - (entry_price * slip * (1 if r > 0 else 0))
    else:
        exit_price = entry_price - (r * risk_per_share) + (entry_price * slip * (1 if r > 0 else 0))

    cost_breakdown = estimate_round_trip_charges(
        entry_price, exit_price, qty, config.cost_cfg,
        side="short" if signal.side == "short" else "long",
    )
    cost_inr = float(cost_breakdown["total_cost"])
    net_pnl = gross_pnl - cost_inr
    net_r = net_pnl / max(risk_inr, 1e-9)
    bars_held = max(0, int(exit_idx - entry_idx))

    return {
        "alpha_name": signal.alpha_name,
        "symbol": signal.symbol,
        "sector": sector,
        "decision_at": signal.decision_at,
        "decision_idx": signal.decision_idx,
        "entry_at": entry_ts,
        "exit_at": idx[int(exit_idx)],
        "side": signal.side,
        "entry_reference": signal.entry_reference,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_atr": signal.stop_atr,
        "target_atr": signal.target_atr,
        "horizon_bars": signal.horizon_bars,
        "confidence": signal.confidence,
        "gross_r": float(r),
        "cost_inr": cost_inr,
        "risk_inr": risk_inr,
        "net_r": float(net_r),
        "net_pnl": float(net_pnl),
        "win": bool(net_pnl > 0),
        "exit_reason": exit_reason,
        "bars_held": bars_held,
    }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ArsenalEvaluator:
    """Run a list of alphas across a multi-asset bundle and produce a
    unified trade frame plus aggregate views.

    Stateless across run() calls: the same evaluator instance can be
    reused with different alpha lists or different bundles.
    """

    def __init__(self, alphas: Sequence[Alpha],
                 config: Optional[EvaluatorConfig] = None) -> None:
        if not alphas:
            raise ValueError("ArsenalEvaluator needs at least one alpha")
        for a in alphas:
            if not isinstance(a, Alpha):
                raise TypeError(f"expected Alpha subclass, got {type(a).__name__}")
        self.alphas = list(alphas)
        self.config = config or EvaluatorConfig()

    def run(self, report,
            extras_by_alpha: Optional[Dict[str, Dict[str, Any]]] = None,
            ) -> pd.DataFrame:
        """Walk every asset × every alpha, produce a unified trade frame.

        ``report`` is a multi_asset_report.pkl-style bundle (object with
        ``assets`` dict of AssetData).

        ``extras_by_alpha`` lets the caller pass per-alpha kwargs (e.g.
        the pool list for the liquidity-pool-reach alpha, or the
        force-model bundle for a learned-policy alpha).
        """
        extras_by_alpha = extras_by_alpha or {}
        trades: List[Dict[str, Any]] = []
        for symbol, ad in report.assets.items():
            df_base = ad.base_df
            if df_base is None or df_base.empty:
                continue
            atr_series = atr(df_base, self.config.atr_period).bfill()
            sec = sector_of(symbol)

            asset_extras = {
                "asset_data": ad,                    # for alphas that need pools/results
                "sector": sec,
                "report": report,                    # for wrapped-model alphas that need
                                                     # access to report.unified_ml, etc.
            }
            for alpha in self.alphas:
                extra = {**asset_extras, **(extras_by_alpha.get(alpha.name, {}))}
                signals = alpha.candidates(
                    symbol=symbol, df_base=df_base,
                    atr_series=atr_series, extra=extra,
                )
                for sig in signals:
                    row = _execute_signal(sig, df_base, atr_series, sec, self.config)
                    if row is None:
                        continue
                    # Regime tags annotate the row in-place (with a prefix).
                    tags = alpha.regime_tags(sig, df_base, atr_series)
                    for k, v in tags.items():
                        row[f"regime_{k}"] = v
                    trades.append(row)
        if not trades:
            return pd.DataFrame(columns=list(TRADE_COLUMNS))
        return pd.DataFrame(trades)

    # ---- aggregation helpers ----

    @staticmethod
    def per_alpha_summary(trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return pd.DataFrame()
        rows = []
        for alpha_name, g in trades.groupby("alpha_name"):
            r = g["net_r"].astype(float).to_numpy()
            r = r[np.isfinite(r)]
            n = int(len(r))
            mean_R = float(r.mean()) if n else 0.0
            se = float(r.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
            gw = float(g.loc[g["net_pnl"] > 0, "net_pnl"].sum())
            gl = float(-g.loc[g["net_pnl"] < 0, "net_pnl"].sum())
            rows.append({
                "alpha_name": alpha_name,
                "trades": n,
                "win": float((g["net_pnl"] > 0).mean()),
                "mean_R": mean_R,
                "median_R": float(np.median(r)) if n else 0.0,
                "se_R": se,
                "ci95_lo": mean_R - 1.96 * se,
                "ci95_hi": mean_R + 1.96 * se,
                "PF": (gw / gl) if gl > 0 else float("inf"),
                "avg_cost_inr": float(g["cost_inr"].mean()) if n else 0.0,
            })
        return pd.DataFrame(rows).sort_values("mean_R", ascending=False).reset_index(drop=True)

    @staticmethod
    def per_regime_summary(trades: pd.DataFrame,
                           dimension: str = "regime_session") -> pd.DataFrame:
        if trades.empty or dimension not in trades.columns:
            return pd.DataFrame()
        rows = []
        for (alpha_name, regime), g in trades.groupby(["alpha_name", dimension]):
            r = g["net_r"].astype(float).to_numpy()
            r = r[np.isfinite(r)]
            n = int(len(r))
            mean_R = float(r.mean()) if n else 0.0
            se = float(r.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
            rows.append({
                "alpha_name": alpha_name,
                "regime": regime,
                "dimension": dimension,
                "trades": n,
                "win": float((g["net_pnl"] > 0).mean()) if n else 0.0,
                "mean_R": mean_R,
                "ci95_lo": mean_R - 1.96 * se,
                "ci95_hi": mean_R + 1.96 * se,
            })
        return pd.DataFrame(rows).sort_values(
            ["alpha_name", "mean_R"], ascending=[True, False]
        ).reset_index(drop=True)

    @staticmethod
    def pairwise_combinations(trades: pd.DataFrame,
                              min_overlap: int = 30) -> pd.DataFrame:
        """For each pair (A, B), find decision timestamps where BOTH alphas
        fired on the SAME symbol within the same bar, then report the
        combined trade-level metrics.

        This isn't a perfect intersection (in reality A might fire 5 bars
        before B), but it's the simplest signal-overlap proxy that says
        "when both alphas agree, what happens?"
        """
        if trades.empty:
            return pd.DataFrame()
        rows = []
        alphas = sorted(trades["alpha_name"].unique())
        for i, a in enumerate(alphas):
            for b in alphas[i + 1:]:
                ga = trades[trades["alpha_name"] == a][["symbol", "decision_idx", "net_r", "net_pnl"]]
                gb = trades[trades["alpha_name"] == b][["symbol", "decision_idx", "net_r", "net_pnl"]]
                inner = pd.merge(ga, gb, on=["symbol", "decision_idx"],
                                 suffixes=("_a", "_b"), how="inner")
                if len(inner) < min_overlap:
                    continue
                # Average R across both alphas as a simple "both fired -> avg outcome".
                inner["combined_r"] = (inner["net_r_a"] + inner["net_r_b"]) / 2.0
                r = inner["combined_r"].to_numpy(dtype=float)
                r = r[np.isfinite(r)]
                n = len(r)
                mean_R = float(r.mean()) if n else 0.0
                se = float(r.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
                rows.append({
                    "alpha_a": a,
                    "alpha_b": b,
                    "overlap_n": n,
                    "combined_mean_R": mean_R,
                    "ci95_lo": mean_R - 1.96 * se,
                    "ci95_hi": mean_R + 1.96 * se,
                    "alpha_a_mean_R_overall": float(trades.loc[trades["alpha_name"] == a, "net_r"].mean()),
                    "alpha_b_mean_R_overall": float(trades.loc[trades["alpha_name"] == b, "net_r"].mean()),
                })
        return pd.DataFrame(rows).sort_values("combined_mean_R", ascending=False).reset_index(drop=True)

    @staticmethod
    def daily_alpha_returns(trades: pd.DataFrame) -> pd.DataFrame:
        """Daily net-R series by alpha for portfolio/diversification checks.

        Uses sum of trade ``net_r`` per alpha per day. This is not a capital
        allocator yet; it is the simplest apples-to-apples view of whether
        alpha P&L streams move together.
        """
        if trades.empty or "entry_at" not in trades.columns:
            return pd.DataFrame()
        frame = trades.copy()
        frame["entry_day"] = pd.to_datetime(frame["entry_at"]).dt.date.astype(str)
        pivot = frame.pivot_table(
            index="entry_day",
            columns="alpha_name",
            values="net_r",
            aggfunc="sum",
            fill_value=0.0,
        )
        return pivot.sort_index().reset_index()

    @staticmethod
    def alpha_correlation(daily_returns: pd.DataFrame) -> pd.DataFrame:
        """Pairwise correlation of alpha daily net-R streams."""
        if daily_returns.empty or len(daily_returns.columns) <= 2:
            return pd.DataFrame()
        values = daily_returns.drop(columns=["entry_day"], errors="ignore")
        if values.empty:
            return pd.DataFrame()
        corr = values.corr().reset_index().rename(columns={"alpha_name": "alpha"})
        if "index" in corr.columns:
            corr = corr.rename(columns={"index": "alpha"})
        return corr
