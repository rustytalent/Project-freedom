"""NSE sector classification + sector-level analytics for the basket workflow.

The basket-of-stocks workflow (Track 4) trains models across stocks. This module groups them
into sectors, computes per-sector aggregates, and surfaces money-flow signals (which sector is
leading / lagging) so Track 5 can use sector context for live execution decisions.

Sector taxonomy below covers the most-liquid NSE names plus the two indices. Add new symbols
to SECTOR_MAP as you trade them — `sector_of(unknown_symbol)` defaults to 'OTHER'.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import pandas as pd


SECTOR_MAP: Dict[str, str] = {
    # ─── Banking & financial services ──────────────────────────────────
    "HDFCBANK.NS":   "BANKING",
    "ICICIBANK.NS":  "BANKING",
    "SBIN.NS":       "BANKING",
    "AXISBANK.NS":   "BANKING",
    "KOTAKBANK.NS":  "BANKING",
    "INDUSINDBK.NS": "BANKING",
    "FEDERALBNK.NS": "BANKING",
    "BANKBARODA.NS": "BANKING",
    "PNB.NS":        "BANKING",
    "BAJFINANCE.NS": "BANKING",
    "BAJAJFINSV.NS": "BANKING",
    "SBILIFE.NS":    "BANKING",
    "HDFCLIFE.NS":   "BANKING",

    # ─── IT / technology services ─────────────────────────────────────
    "TCS.NS":      "IT",
    "INFY.NS":     "IT",
    "HCLTECH.NS":  "IT",
    "WIPRO.NS":    "IT",
    "TECHM.NS":    "IT",
    "LTIM.NS":     "IT",

    # ─── Energy & oil ─────────────────────────────────────────────────
    "RELIANCE.NS": "ENERGY",
    "ONGC.NS":     "ENERGY",
    "BPCL.NS":     "ENERGY",
    "IOC.NS":      "ENERGY",
    "GAIL.NS":     "ENERGY",
    "POWERGRID.NS": "ENERGY",
    "NTPC.NS":     "ENERGY",
    "COALINDIA.NS": "ENERGY",

    # ─── Auto ─────────────────────────────────────────────────────────
    "MARUTI.NS":      "AUTO",
    "TATAMOTORS.NS":  "AUTO",
    "M&M.NS":         "AUTO",
    "BAJAJ-AUTO.NS":  "AUTO",
    "EICHERMOT.NS":   "AUTO",
    "HEROMOTOCO.NS":  "AUTO",

    # ─── FMCG / consumer ──────────────────────────────────────────────
    "HINDUNILVR.NS":  "FMCG",
    "ITC.NS":         "FMCG",
    "NESTLEIND.NS":   "FMCG",
    "BRITANNIA.NS":   "FMCG",
    "DABUR.NS":       "FMCG",
    "GODREJCP.NS":    "FMCG",
    "MARICO.NS":      "FMCG",
    "TATACONSUM.NS":  "FMCG",

    # ─── Pharma & healthcare ──────────────────────────────────────────
    "SUNPHARMA.NS":  "PHARMA",
    "CIPLA.NS":      "PHARMA",
    "DIVISLAB.NS":   "PHARMA",
    "DRREDDY.NS":    "PHARMA",
    "APOLLOHOSP.NS": "PHARMA",

    # ─── Metals ───────────────────────────────────────────────────────
    "TATASTEEL.NS": "METALS",
    "HINDALCO.NS":  "METALS",
    "JSWSTEEL.NS":  "METALS",
    "VEDL.NS":      "METALS",
    "SAIL.NS":      "METALS",

    # ─── Telecom ──────────────────────────────────────────────────────
    "BHARTIARTL.NS": "TELECOM",
    "IDEA.NS":       "TELECOM",

    # ─── Infrastructure / capital goods ──────────────────────────────
    "LT.NS":        "INFRA",
    "ADANIENT.NS":  "INFRA",
    "ADANIPORTS.NS": "INFRA",
    "ULTRACEMCO.NS": "INFRA",
    "GRASIM.NS":    "INFRA",

    # ─── Indices (for breadth) ────────────────────────────────────────
    "^NSEI":      "INDEX",
    "^NSEBANK":   "INDEX",
    "^CNXIT":     "INDEX",
}


def sector_of(symbol: str) -> str:
    return SECTOR_MAP.get(symbol, "OTHER")


def group_by_sector(symbols: List[str]) -> Dict[str, List[str]]:
    """Symbols grouped by sector. Symbols not in SECTOR_MAP land in 'OTHER'."""
    out: Dict[str, List[str]] = defaultdict(list)
    for s in symbols:
        out[sector_of(s)].append(s)
    return dict(out)


# ---------------------------------------------------------------------------
# Sector-level return time series
# ---------------------------------------------------------------------------

def sector_index_returns(asset_dfs: Dict[str, pd.DataFrame],
                          resample_rule: str = "1D") -> Dict[str, pd.Series]:
    """For each sector, build a sector "index" = mean of constituent daily close returns.

    This is the simplest equal-weighted sector aggregator. Works on whatever base TF was
    fetched — defaults to resampling to daily.
    """
    by_sector = group_by_sector(list(asset_dfs.keys()))
    out: Dict[str, pd.Series] = {}
    for sector, symbols in by_sector.items():
        rets = []
        for s in symbols:
            df = asset_dfs[s]
            if df is None or df.empty:
                continue
            daily_close = df["close"].resample(resample_rule).last().dropna()
            ret = daily_close.pct_change().dropna()
            if len(ret):
                rets.append(ret)
        if not rets:
            continue
        # Outer-join then average; missing days for a stock are treated as 0% return.
        combined = pd.concat(rets, axis=1).fillna(0.0).mean(axis=1)
        out[sector] = combined
    return out


# ---------------------------------------------------------------------------
# Per-sector point-in-time metrics
# ---------------------------------------------------------------------------

def compute_sector_metrics(asset_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """Per-sector cumulative returns over 5d / 20d / 60d windows + annualised volatility.

    Returns: { sector: { symbols, ret_5d, ret_20d, ret_60d, vol_20d_annualised } }
    """
    returns = sector_index_returns(asset_dfs, "1D")
    by_sector = group_by_sector(list(asset_dfs.keys()))
    out: Dict[str, Dict] = {}
    for sector, ret in returns.items():
        if len(ret) < 5:
            continue

        def cum(n: int) -> float:
            # Use whatever rows we have. Resampling 60 days of 5m data gives ~59 daily returns
            # (one less than the bar count). The previous strict `if len(ret) < n: return 0.0`
            # check meant ret_60d returned 0% whenever the data window was exactly 60 days —
            # a silent bug that broke the rotation signal across most sectors.
            n_actual = min(n, len(ret))
            if n_actual < 2:
                return 0.0
            return float((1.0 + ret.iloc[-n_actual:]).prod() - 1.0)

        out[sector] = {
            "symbols": by_sector.get(sector, []),
            "ret_5d": cum(5),
            "ret_20d": cum(20),
            "ret_60d": cum(60),
            "vol_20d_annualised": float(ret.iloc[-20:].std() * np.sqrt(252))
                                   if len(ret) >= 20 else 0.0,
        }
    return out


def sector_correlation_matrix(asset_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pairwise correlation of sector daily-return series (Pearson). NaN-aligned outer join."""
    returns = sector_index_returns(asset_dfs, "1D")
    if not returns:
        return pd.DataFrame()
    sectors = sorted(returns.keys())
    aligned = pd.concat({s: returns[s] for s in sectors}, axis=1).fillna(0.0)
    return aligned.corr()


# ---------------------------------------------------------------------------
# Money-flow rotation signal
# ---------------------------------------------------------------------------

def detect_rotation(sector_metrics: Dict[str, Dict]) -> Dict:
    """Identify sectors with momentum divergence (where money is rotating).

    Heuristic: rank sectors by 5d return and by 60d return.
      - A sector that ranks HIGH on 5d but LOW on 60d → money is rotating INTO it.
      - A sector that ranks LOW on 5d but HIGH on 60d → money is rotating OUT of it.

    Returns a dict with:
      leader_5d, laggard_5d                  : recent winner / loser
      ranked_5d, ranked_60d                  : full ranking lists
      rotation_in                            : list of sectors gaining momentum
      rotation_out                           : list of sectors losing momentum
      narrative                              : one-line plain-English summary
    """
    if not sector_metrics or len(sector_metrics) < 2:
        return {"narrative": "need at least 2 sectors for a rotation signal"}

    sectors = list(sector_metrics.keys())
    n = len(sectors)
    ranked_5d = sorted(sectors, key=lambda s: -sector_metrics[s]["ret_5d"])
    ranked_60d = sorted(sectors, key=lambda s: -sector_metrics[s]["ret_60d"])

    pos_5d = {s: i for i, s in enumerate(ranked_5d)}
    pos_60d = {s: i for i, s in enumerate(ranked_60d)}

    # A sector "rotated in" if its 5d rank is meaningfully better than its 60d rank.
    # Threshold: rank improvement of >= n/3 positions (e.g., went from 6th to 2nd of 6).
    threshold = max(1, n // 3)
    rotation_in = [s for s in sectors
                    if pos_60d[s] - pos_5d[s] >= threshold]
    rotation_out = [s for s in sectors
                     if pos_5d[s] - pos_60d[s] >= threshold]

    # Sort by magnitude of rotation
    rotation_in.sort(key=lambda s: -(pos_60d[s] - pos_5d[s]))
    rotation_out.sort(key=lambda s: -(pos_5d[s] - pos_60d[s]))

    # Narrative
    leader = ranked_5d[0]
    laggard = ranked_5d[-1]
    leader_5d = sector_metrics[leader]["ret_5d"]
    leader_60d = sector_metrics[leader]["ret_60d"]
    laggard_5d = sector_metrics[laggard]["ret_5d"]
    laggard_60d = sector_metrics[laggard]["ret_60d"]

    parts = []
    parts.append(f"{leader} is the 5d leader ({leader_5d:+.1%}; "
                  f"60d {leader_60d:+.1%})")
    parts.append(f"{laggard} is the 5d laggard ({laggard_5d:+.1%}; "
                  f"60d {laggard_60d:+.1%})")
    if rotation_in:
        parts.append(f"money rotating INTO: {', '.join(rotation_in)}")
    if rotation_out:
        parts.append(f"money rotating OUT of: {', '.join(rotation_out)}")
    narrative = ". ".join(parts) + "."

    return {
        "leader_5d": leader, "laggard_5d": laggard,
        "ranked_5d": ranked_5d, "ranked_60d": ranked_60d,
        "rotation_in": rotation_in, "rotation_out": rotation_out,
        "narrative": narrative,
    }


# ---------------------------------------------------------------------------
# Per-sector pooled OOS respect rate (uses Pool.asset → sector mapping)
# ---------------------------------------------------------------------------

def per_sector_oos(oos_pools, oos_results) -> Dict[str, Dict]:
    """Pool the OOS respect outcomes per sector.

    Returns: { sector: { n, tested, respect, strict_respect, symbols } }
    """
    by_sector: Dict[str, list] = defaultdict(list)
    for p, r in zip(oos_pools, oos_results):
        by_sector[sector_of(p.asset)].append((p, r))

    out: Dict[str, Dict] = {}
    for sector, items in by_sector.items():
        tested = [r for _, r in items if r.is_tested]
        respected = [r for r in tested if r.is_respect]
        decisive_resp = [r for r in tested
                          if r.outcome in ("respected_strong", "swept_and_reclaimed")]
        decisive_break = [r for r in tested if r.outcome == "broken_strong"]
        decisive_total = len(decisive_resp) + len(decisive_break)
        out[sector] = {
            "symbols": sorted({p.asset for p, _ in items}),
            "n": len(items),
            "tested": len(tested),
            "respect": (len(respected) / len(tested)) if tested else 0.0,
            "strict_respect": (len(decisive_resp) / decisive_total) if decisive_total else 0.0,
            "n_decisive": decisive_total,
        }
    return out


# ---------------------------------------------------------------------------
# Helpers for runtime serialisation (Track 5 will ingest sector_data.json)
# ---------------------------------------------------------------------------

def per_asset_reliability(report, pooled_floor: float = 0.30) -> Dict[str, float]:
    """For each asset in the report, return its OOS broad respect rate (a 'reliability score').
    Used downstream by live_run.py to multiply EV.

    Floors at `pooled_floor` so an asset with 0 tested pools doesn't kill all its setups.
    Prefer `per_asset_reliability_shrunk` for live trading — it Bayesian-shrinks small
    samples toward basket baseline so a 55-sample 33% rate doesn't get treated the same
    as a 300-sample 33% rate.
    """
    out: Dict[str, float] = {}
    for sym, ad in report.assets.items():
        outs = ad.walkforward.raw_oos_outcomes
        if not outs:
            out[sym] = pooled_floor
            continue
        out[sym] = max(pooled_floor, float(sum(outs) / len(outs)))
    return out


def per_asset_reliability_shrunk(report, basket_baseline: float,
                                   shrinkage_n: int = 80,
                                   low_conf_n: int = 80
                                   ) -> Dict[str, Dict]:
    """Beta-Binomial-style empirical Bayes: pull the per-asset OOS respect rate toward
    `basket_baseline` with weight `shrinkage_n`. A 55-pool asset with 33% rate gets pulled
    toward the basket average (45%); a 300-pool asset stays close to its empirical rate.

    Returns per-asset dict { 'shrunk_respect', 'raw_respect', 'n', 'is_low_conf' }.
    `is_low_conf` flags assets with n < low_conf_n so the dashboard can warn.
    """
    out: Dict[str, Dict] = {}
    for sym, ad in report.assets.items():
        outs = ad.walkforward.raw_oos_outcomes
        n = len(outs)
        raw = (sum(outs) / n) if n else basket_baseline
        # Beta(α=baseline*shrinkage_n, β=(1-baseline)*shrinkage_n) prior; posterior mean is:
        # (sum_y + α) / (n + α + β) = (sum_y + baseline*shrinkage_n) / (n + shrinkage_n)
        shrunk = (sum(outs) + basket_baseline * shrinkage_n) / (n + shrinkage_n)
        out[sym] = {
            "shrunk_respect": float(shrunk),
            "raw_respect": float(raw),
            "n": int(n),
            "is_low_conf": n < low_conf_n,
        }
    return out


def reliability_multiplier(asset_reliability: float, basket_baseline: float,
                            cap_low: float = 0.5, cap_high: float = 1.5) -> float:
    """Convert raw reliability into an EV multiplier, capped to [cap_low, cap_high].
    1.0 means 'asset is exactly at basket avg' — neutral weight."""
    if basket_baseline <= 0:
        return 1.0
    raw = asset_reliability / basket_baseline
    return max(cap_low, min(cap_high, raw))


def sector_momentum_alignment(sector_metrics: Dict[str, Dict], sector: str,
                                trade_side: str) -> float:
    """Multiplier that boosts trades aligned with sector momentum and penalises ones against it.

    `trade_side`: "buy" for setups below current price (long bias), "sell" for above (short bias).
    Sector ret_5d > 0 + buy trade → momentum-aligned. Returns 1.15.
    Sector ret_5d < 0 + sell trade → momentum-aligned. Returns 1.15.
    Aligned against sector momentum → 0.85.
    Within ±0.5% → neutral 1.0 (avoid noise).
    """
    m = sector_metrics.get(sector)
    if not m:
        return 1.0
    ret_5d = float(m.get("ret_5d", 0.0))
    if abs(ret_5d) < 0.005:
        return 1.0
    sector_bullish = ret_5d > 0
    trade_bullish = (trade_side == "buy")
    return 1.15 if sector_bullish == trade_bullish else 0.85


def serialise_sector_intel(sector_metrics: Dict[str, Dict],
                           sector_oos: Dict[str, Dict],
                           rotation: Dict,
                           correlation_df: pd.DataFrame) -> Dict:
    """Build the JSON-serialisable blob Track 5 will read from disk to know:
      - per-sector OOS performance (which sectors the model handles best)
      - per-sector recent momentum (5/20/60-day)
      - rotation signal (which sectors are gaining / losing)
      - cross-sector correlation matrix
    """
    return {
        "sectors": {
            sec: {
                **sector_metrics.get(sec, {}),
                "pooled_oos_respect": sector_oos.get(sec, {}).get("respect"),
                "pooled_oos_strict": sector_oos.get(sec, {}).get("strict_respect"),
                "n_pools": sector_oos.get(sec, {}).get("n"),
                "n_tested": sector_oos.get(sec, {}).get("tested"),
            }
            for sec in set(list(sector_metrics.keys()) + list(sector_oos.keys()))
        },
        "rotation": rotation,
        "correlation_matrix": (correlation_df.to_dict()
                                if correlation_df is not None
                                and not correlation_df.empty else {}),
    }
