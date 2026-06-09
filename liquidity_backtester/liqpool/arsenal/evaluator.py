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
    "pool_mid_at_touch", "poc_today_at_touch",
    "vah_today_at_touch", "val_today_at_touch",
    "pool_volume_nearest_atr", "pool_volume_confirmed_at_touch",
    "pool_q_pred",
    "gross_r", "cost_inr", "risk_inr",
    "net_r", "net_pnl", "win",
    "exit_reason", "bars_held",
)


@dataclass
class EvaluatorConfig:
    """Per-evaluator settings. Defaults match the v1 simulator we trust."""
    cost_cfg: ZerodhaEquityCostConfig = field(default_factory=ZerodhaEquityCostConfig)
    notional_inr: float = 100_000.0     # default position size (₹1L). Bumped from
                                        # ₹50k because at ₹50k a ₹770 stock gives
                                        # qty=64 — below the practitioner floor
                                        # where a 3-rupee move outearns the fees.
                                        # See user HDFC sizing example.
    atr_period: int = 14
    slippage_bps_per_side: float = 1.0  # flat bps slippage on entry + exit

    # Minimum-economic-position filter: reject any signal whose target reward
    # in INR is less than ``min_target_to_cost_ratio`` * round-trip cost.
    # Default 3.0 = "the target must outearn the fee by 3×". Set to 0.0 to
    # disable (recover the old behaviour where every signal trades regardless
    # of size). This filter materially changes which trades enter the book —
    # at ₹50k notional + low-ATR stocks the historical strategy was paying
    # ~1.1R per trade in cost (see analysis docs), so this is the structural
    # fix, not a research dial.
    min_target_to_cost_ratio: float = 3.0

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
                    state_featurizer: Optional[Any] = None,
                    shadow_log_writer: Optional[Any] = None,
                    trading_date_ist: Optional[str] = None,
                    symbol: Optional[str] = None,
                    alpha_name: Optional[str] = None,
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

    # Minimum-economic-position filter. The position size implied by
    # notional / entry_price determines how many rupees a unit-ATR target
    # actually pays. If even the BEST-CASE outcome (target hit) doesn't
    # clear k× the round-trip cost, this trade is structurally unprofitable
    # — no edge in the model can rescue it. Reject before triple-barrier.
    if config.min_target_to_cost_ratio > 0.0:
        qty_check = max(1, int(config.notional_inr / max(entry_price, 1e-9)))
        target_reward_inr = float(signal.target_atr) * atr_val * qty_check
        # Approximate round-trip cost at target-hit price (favourable exit).
        if signal.side == "long":
            optimistic_exit = entry_price + signal.target_atr * atr_val
        else:
            optimistic_exit = entry_price - signal.target_atr * atr_val
        rt_cost = estimate_round_trip_charges(
            entry_price, optimistic_exit, qty_check, config.cost_cfg,
            side="short" if signal.side == "short" else "long",
        )
        if target_reward_inr < config.min_target_to_cost_ratio * float(rt_cost["total_cost"]):
            # Stream L hook: record the rejection BEFORE returning so the
            # Stream M.4 regret estimator can learn whether the threshold
            # is too tight. Defensive: never block evaluation.
            if shadow_log_writer is not None and trading_date_ist is not None:
                try:
                    from ..products.shadow_log import (
                        EVENT_KIND_BELOW_TARGET_TO_COST,
                    )
                    shadow_log_writer.record(
                        event_kind=EVENT_KIND_BELOW_TARGET_TO_COST,
                        trading_date_ist=trading_date_ist,
                        symbol=symbol,
                        detail_token=(
                            f"{alpha_name or 'unknown'}_"
                            f"{int(signal.decision_idx)}"
                        ),
                        decision_context={
                            "alpha_name": alpha_name,
                            "side": signal.side,
                            "entry_price": float(entry_price),
                            "target_atr": float(signal.target_atr),
                            "stop_atr": float(signal.stop_atr),
                            "atr_at_entry": atr_val,
                            "target_reward_inr": float(target_reward_inr),
                            "round_trip_cost_inr": float(rt_cost["total_cost"]),
                            "target_to_cost_ratio": (
                                float(target_reward_inr) /
                                max(float(rt_cost["total_cost"]), 1e-9)
                            ),
                            "min_target_to_cost_ratio": float(
                                config.min_target_to_cost_ratio
                            ),
                            "qty_check": int(qty_check),
                            "sector": sector,
                        },
                    )
                except Exception:
                    pass
            return None

    pool_mid_at_touch = float(signal.state.get("pool_mid", np.nan))
    pool_q_pred = float(signal.state.get("q_pred", signal.confidence))
    poc_today_at_touch = float("nan")
    vah_today_at_touch = float("nan")
    val_today_at_touch = float("nan")
    pool_volume_nearest_atr = float("nan")
    pool_volume_confirmed_at_touch = False
    if state_featurizer is not None and 0 <= signal.decision_idx < len(df_base):
        try:
            poc_today_at_touch = float(state_featurizer._poc_today[signal.decision_idx])
            vah_today_at_touch = float(state_featurizer._vah_today[signal.decision_idx])
            val_today_at_touch = float(state_featurizer._val_today[signal.decision_idx])
        except Exception:
            pass
    if math.isfinite(pool_mid_at_touch):
        try:
            touch_atr = float(atr_series.iloc[signal.decision_idx])
        except Exception:
            touch_atr = float("nan")
        refs = (poc_today_at_touch, vah_today_at_touch, val_today_at_touch)
        distances = [
            abs(pool_mid_at_touch - ref) / max(touch_atr, 1e-9)
            for ref in refs
            if math.isfinite(ref) and math.isfinite(touch_atr) and touch_atr > 0
        ]
        if distances:
            pool_volume_nearest_atr = float(min(distances))
            pool_volume_confirmed_at_touch = bool(pool_volume_nearest_atr <= 0.25)

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
        "pool_mid_at_touch": pool_mid_at_touch,
        "poc_today_at_touch": poc_today_at_touch,
        "vah_today_at_touch": vah_today_at_touch,
        "val_today_at_touch": val_today_at_touch,
        "pool_volume_nearest_atr": pool_volume_nearest_atr,
        "pool_volume_confirmed_at_touch": pool_volume_confirmed_at_touch,
        "pool_q_pred": pool_q_pred,
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
            shadow_log_writer: Optional[Any] = None,
            trading_date_ist: Optional[str] = None,
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
            try:
                from liqpool.timing import StateFeaturizer
                state_featurizer = StateFeaturizer(df_base)
            except Exception:
                state_featurizer = None
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
                    row = _execute_signal(
                        sig, df_base, atr_series, sec, self.config,
                        state_featurizer=state_featurizer,
                        shadow_log_writer=shadow_log_writer,
                        trading_date_ist=trading_date_ist,
                        symbol=symbol,
                        alpha_name=alpha.name,
                    )
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
        """Per-alpha aggregate. Now includes Bailey-style honesty
        columns alongside the raw means:

          * ``sharpe`` — per-trade Sharpe = mean_R / std_R
          * ``psr`` — Probabilistic Sharpe Ratio vs 0 (Bailey 2012)
          * ``dsr`` — Deflated Sharpe Ratio across the alphas in
            *this* run (Bailey 2014). Deflates by the search budget
            implied by this evaluator's alpha registry.

        The DSR is the headline number that survives the search.
        A DSR > 0.95 means: even after correcting for trying multiple
        alphas, the strategy's true Sharpe is positive with 95%
        confidence.
        """
        from .honesty import deflated_sharpe, probabilistic_sharpe
        if trades.empty:
            return pd.DataFrame()
        # First pass: per-alpha raw stats + raw Sharpe.
        per_alpha_returns: Dict[str, np.ndarray] = {}
        per_alpha_basics: List[Dict[str, Any]] = []
        for alpha_name, g in trades.groupby("alpha_name"):
            r = g["net_r"].astype(float).to_numpy()
            r = r[np.isfinite(r)]
            n = int(len(r))
            mean_R = float(r.mean()) if n else 0.0
            std_R = float(r.std(ddof=1)) if n > 1 else 0.0
            se = std_R / math.sqrt(n) if n > 1 else 0.0
            sharpe = (mean_R / std_R) if std_R > 0 else 0.0
            gw = float(g.loc[g["net_pnl"] > 0, "net_pnl"].sum())
            gl = float(-g.loc[g["net_pnl"] < 0, "net_pnl"].sum())
            per_alpha_returns[str(alpha_name)] = r
            per_alpha_basics.append({
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
                "sharpe": sharpe,
                "_returns": r,
            })

        # Second pass: PSR (each alpha vs 0) and DSR (each alpha vs
        # the max-of-N-trials null built from THIS registry's Sharpes).
        all_sharpes = [b["sharpe"] for b in per_alpha_basics]
        rows = []
        for b in per_alpha_basics:
            r = b.pop("_returns")
            psr = probabilistic_sharpe(r) if len(r) >= 3 else float("nan")
            dsr = (
                deflated_sharpe(r, all_sharpes) if len(r) >= 3 and len(all_sharpes) >= 2
                else psr
            )
            b["psr"] = psr
            b["dsr"] = dsr
            rows.append(b)
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
