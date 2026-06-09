"""Phase 3C leakage probes for pool/timing research runs.

The goal is not to prove a strategy leak-free in one pass. It is to make the
main causal assumptions visible in every run:

- pools must not be consumed before all contributors are known
- labels must not begin before pool availability
- outcome windows and embargo settings must be consistent with active horizons
- bar indexes must be monotonic and deterministic enough for replay-style work
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import Config
from .feature_store import PROXIMITY_HORIZONS
from .pools import Pool
from .tester import PoolResult


@dataclass
class LeakageIssue:
    severity: str
    check: str
    symbol: str
    pool_idx: Optional[int]
    message: str
    detail: Dict

    def to_dict(self) -> Dict:
        return asdict(self)


def _ts(value) -> Optional[pd.Timestamp]:
    if value is None or pd.isna(value):
        return None
    try:
        return pd.Timestamp(value)
    except Exception:
        return None


def _issue(severity: str, check: str, symbol: str, pool_idx: Optional[int],
           message: str, **detail) -> LeakageIssue:
    return LeakageIssue(
        severity=severity,
        check=check,
        symbol=symbol,
        pool_idx=pool_idx,
        message=message,
        detail=detail,
    )


def _bar_period_seconds(index: pd.Index) -> float:
    if len(index) < 2:
        return 300.0
    diffs = pd.Series(index).diff().dropna()
    if diffs.empty:
        return 300.0
    return max(float(pd.Timedelta(diffs.median()).total_seconds()), 1.0)


def max_active_label_horizon(cfg: Config) -> int:
    return max([int(cfg.test_horizon_bars), *[int(h) for h in PROXIMITY_HORIZONS]])


def _tf_period(tf: str, fallback_seconds: float) -> Optional[pd.Timedelta]:
    label = str(tf)
    if label in ("PD", "PW", "PM"):
        return None
    if label == "base":
        return pd.Timedelta(seconds=float(fallback_seconds))
    if label.endswith("min"):
        try:
            return pd.Timedelta(minutes=int(label[:-3]))
        except Exception:
            return None
    if label.endswith("m") and label[:-1].isdigit():
        return pd.Timedelta(minutes=int(label[:-1]))
    if label.endswith("H") and label[:-1].isdigit():
        return pd.Timedelta(hours=int(label[:-1]))
    if label == "1D":
        return pd.Timedelta(days=1)
    if label == "1W":
        return pd.Timedelta(weeks=1)
    return None


def _source_family(source: str) -> str:
    return str(source or "").split("@", 1)[0]


def _contributors_lag_rows(symbol: str, pools: List[Pool],
                           base_period_seconds: float) -> List[Dict]:
    rows: List[Dict] = []
    for pool_idx, pool in enumerate(pools):
        for c_idx, contributor in enumerate(pool.contributors):
            known_at = _ts(getattr(contributor, "known_at", None))
            ts = _ts(getattr(contributor, "ts", None))
            if known_at is None or ts is None:
                continue
            tf = getattr(contributor, "tf", "base")
            period = _tf_period(tf, base_period_seconds)
            source = getattr(contributor, "source", "")
            lag_seconds = float((known_at - ts).total_seconds())
            expected_close = ts + period if period is not None else None
            close_lag_seconds = (
                float((known_at - expected_close).total_seconds())
                if expected_close is not None else None
            )
            rows.append({
                "symbol": symbol,
                "pool_idx": int(pool_idx),
                "contributor_index": int(c_idx),
                "tf": str(tf),
                "source_family": _source_family(source),
                "source": source,
                "lag_seconds": lag_seconds,
                "close_lag_seconds": close_lag_seconds,
            })
    return rows


def _summarise_lags(rows: List[Dict]) -> List[Dict]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    out = []
    for (tf, family), g in df.groupby(["tf", "source_family"], dropna=False):
        out.append({
            "tf": tf,
            "source_family": family,
            "n": int(len(g)),
            "min_lag_seconds": float(g["lag_seconds"].min()),
            "median_lag_seconds": float(g["lag_seconds"].median()),
            "min_close_lag_seconds": (
                float(g["close_lag_seconds"].dropna().min())
                if g["close_lag_seconds"].notna().any() else None
            ),
            "median_close_lag_seconds": (
                float(g["close_lag_seconds"].dropna().median())
                if g["close_lag_seconds"].notna().any() else None
            ),
        })
    return sorted(out, key=lambda r: (str(r["tf"]), str(r["source_family"])))


def _audit_bar_index(symbol: str, df: pd.DataFrame) -> List[LeakageIssue]:
    issues: List[LeakageIssue] = []
    if df is None or df.empty:
        issues.append(_issue("ERROR", "bar_index", symbol, None, "base dataframe is empty"))
        return issues
    if not df.index.is_monotonic_increasing:
        issues.append(_issue("ERROR", "bar_index", symbol, None,
                             "base dataframe index is not monotonic increasing"))
    if df.index.has_duplicates:
        issues.append(_issue("ERROR", "bar_index", symbol, None,
                             "base dataframe index has duplicate timestamps"))
    missing = [c for c in ("open", "high", "low", "close", "volume") if c not in df.columns]
    if missing:
        issues.append(_issue("ERROR", "bar_columns", symbol, None,
                             "base dataframe is missing required OHLCV columns",
                             missing=",".join(missing)))
    return issues


def _audit_pool(symbol: str, pool_idx: int, pool: Pool, result: PoolResult,
                df: pd.DataFrame, cfg: Config,
                base_period_seconds: float) -> List[LeakageIssue]:
    issues: List[LeakageIssue] = []
    idx = df.index
    available_at = _ts(pool.available_at)
    formed_at = _ts(pool.formed_at)

    if available_at is None:
        return [_issue("ERROR", "pool_availability", symbol, pool_idx,
                       "pool.available_at is missing or invalid")]
    if formed_at is not None and available_at < formed_at:
        issues.append(_issue("ERROR", "pool_availability", symbol, pool_idx,
                             "pool is available before its formation timestamp",
                             formed_at=str(formed_at), available_at=str(available_at)))

    latest_contributor_known = None
    for c_idx, contributor in enumerate(pool.contributors):
        known_at = _ts(getattr(contributor, "known_at", None))
        ts = _ts(getattr(contributor, "ts", None))
        source = getattr(contributor, "source", "")
        if known_at is None:
            issues.append(_issue("ERROR", "contributor_availability", symbol, pool_idx,
                                 "contributor known_at is missing",
                                 contributor_index=c_idx, source=source))
            continue
        if ts is not None and known_at < ts:
            issues.append(_issue("ERROR", "contributor_availability", symbol, pool_idx,
                                 "contributor is known before its source timestamp",
                                 contributor_index=c_idx, source=source,
                                 ts=str(ts), known_at=str(known_at)))
        if ts is not None:
            tf = getattr(contributor, "tf", "base")
            period = _tf_period(tf, base_period_seconds)
            if period is not None:
                source_close = ts + period
                if known_at < source_close:
                    issues.append(_issue(
                        "ERROR", "mtf_closed_bar", symbol, pool_idx,
                        "contributor is known before its source bar is fully closed",
                        contributor_index=c_idx,
                        source=source,
                        tf=str(tf),
                        ts=str(ts),
                        source_close=str(source_close),
                        known_at=str(known_at),
                    ))
        if available_at < known_at:
            issues.append(_issue("ERROR", "pool_availability", symbol, pool_idx,
                                 "pool available_at is earlier than a contributor known_at",
                                 contributor_index=c_idx, source=source,
                                 contributor_known_at=str(known_at),
                                 pool_available_at=str(available_at)))
        if latest_contributor_known is None or known_at > latest_contributor_known:
            latest_contributor_known = known_at

    if latest_contributor_known is not None and available_at < latest_contributor_known:
        issues.append(_issue("ERROR", "pool_availability", symbol, pool_idx,
                             "pool available_at is not the max contributor known_at",
                             latest_contributor_known=str(latest_contributor_known),
                             pool_available_at=str(available_at)))

    start_idx = int(idx.searchsorted(available_at, side="left"))
    if start_idx >= len(idx):
        issues.append(_issue("WARN", "pool_availability", symbol, pool_idx,
                             "pool becomes available after the final base bar",
                             available_at=str(available_at)))
    else:
        full_end = start_idx + int(cfg.test_horizon_bars)
        if result.outcome != "horizon_insufficient" and full_end > len(idx):
            issues.append(_issue("WARN", "label_window", symbol, pool_idx,
                                 "label was produced with less than the configured full horizon",
                                 start_idx=start_idx, horizon_bars=cfg.test_horizon_bars,
                                 available_forward_bars=max(0, len(idx) - start_idx)))

    touched_at = _ts(result.touched_at)
    broken_at = _ts(result.broken_at)
    if touched_at is not None:
        if touched_at < available_at:
            issues.append(_issue("ERROR", "label_window", symbol, pool_idx,
                                 "pool touch occurs before pool availability",
                                 touched_at=str(touched_at), available_at=str(available_at)))
        if result.bars_to_touch is None:
            issues.append(_issue("ERROR", "label_window", symbol, pool_idx,
                                 "touched pool has missing bars_to_touch"))
        elif start_idx < len(idx):
            touch_idx = int(idx.searchsorted(touched_at, side="left"))
            expected = touch_idx - start_idx
            if abs(int(result.bars_to_touch) - expected) > 1:
                issues.append(_issue("WARN", "label_window", symbol, pool_idx,
                                     "bars_to_touch does not match base-index distance",
                                     recorded_bars_to_touch=result.bars_to_touch,
                                     expected_bars_to_touch=expected))
    elif result.bars_to_touch is not None:
        issues.append(_issue("ERROR", "label_window", symbol, pool_idx,
                             "bars_to_touch is set but touched_at is missing",
                             bars_to_touch=result.bars_to_touch))

    if broken_at is not None:
        if broken_at < available_at:
            issues.append(_issue("ERROR", "label_window", symbol, pool_idx,
                                 "pool break occurs before pool availability",
                                 broken_at=str(broken_at), available_at=str(available_at)))
        if touched_at is not None and broken_at < touched_at:
            issues.append(_issue("WARN", "label_window", symbol, pool_idx,
                                 "pool break timestamp precedes touch timestamp",
                                 broken_at=str(broken_at), touched_at=str(touched_at)))

    return issues


def _audit_embargo(report, cfg: Config,
                   requested_embargo_bars: Optional[int] = None,
                   allow_short_embargo: bool = False) -> Tuple[List[LeakageIssue], Dict]:
    max_active_horizon = max_active_label_horizon(cfg)
    requested = int(cfg.embargo_bars if requested_embargo_bars is None
                    else requested_embargo_bars)
    effective = int(cfg.embargo_bars)
    issues: List[LeakageIssue] = []
    if effective < max_active_horizon:
        issues.append(_issue(
            "WARN" if allow_short_embargo else "ERROR",
            "embargo_horizon", "ALL", None,
            "configured embargo is shorter than the maximum active label horizon",
            requested_embargo_bars=requested,
            effective_embargo_bars=effective,
            max_active_horizon=max_active_horizon,
            quality_horizon=int(cfg.test_horizon_bars),
            proximity_horizons=",".join(str(h) for h in PROXIMITY_HORIZONS),
            allow_short_embargo=bool(allow_short_embargo),
        ))
    return issues, {
        "requested_embargo_bars": requested,
        "effective_embargo_bars": effective,
        "embargo_bars": effective,
        "max_active_horizon": int(max_active_horizon),
        "quality_horizon": int(cfg.test_horizon_bars),
        "proximity_horizons": [int(h) for h in PROXIMITY_HORIZONS],
        "allow_short_embargo": bool(allow_short_embargo),
        "embargo_covers_max_horizon": bool(effective >= max_active_horizon),
    }


def _truncate_closed(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Return only the bars whose CLOSE timestamp is <= ``as_of``.

    Engine-wide convention (asserted on load in ``warehouse.py``):
        ``df.index[i]`` is the OPEN timestamp of bar i, in tz-naive
        UTC. The bar represents the half-open interval
        ``[index[i], index[i] + period)`` and closes at
        ``index[i] + period``.

    Under this convention, ``index[i] + period`` is the close
    timestamp, and a bar is "knowable" at time ``as_of`` only when
    its close <= as_of. That's what this function returns.

    If warehouse data ever changes the convention to bar-close
    timestamps, this filter shifts by one bar in the wrong direction
    and the replay audit produces silent off-by-ones. Keep the
    convention pinned at the loader.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    period = pd.Timedelta(seconds=_bar_period_seconds(df.index))
    close_ts = pd.to_datetime(df.index) + period
    return df.loc[close_ts <= as_of].copy()


def _pool_match(original: Pool, replayed: Pool, tolerance: float) -> bool:
    if original.side != replayed.side:
        return False
    overlap = not (
        replayed.price_high + tolerance < original.price_low
        or original.price_high + tolerance < replayed.price_low
    )
    mid_near = abs(float(replayed.mid) - float(original.mid)) <= tolerance
    return bool(overlap or mid_near)


def _replay_audit_for_symbol(symbol: str, ad, samples: int,
                             severity: str = "WARN") -> Tuple[List[LeakageIssue], Dict]:
    issues: List[LeakageIssue] = []
    summary = {
        "requested": int(max(samples, 0)),
        "checked": 0,
        "misses": 0,
        "skipped": 0,
    }
    if samples <= 0:
        return issues, summary
    base = getattr(ad, "base_df", None)
    tf_data = getattr(ad, "tf_data", None)
    if base is None or base.empty or not tf_data:
        summary["skipped"] = int(samples)
        return issues, summary

    min_start_idx = max(20, int(getattr(ad.final_cfg, "test_horizon_bars", 150) // 10))
    candidates = []
    for pool_idx, pool in enumerate(getattr(ad, "final_pools", [])):
        available_at = _ts(pool.available_at)
        if available_at is None:
            continue
        start_idx = int(base.index.searchsorted(available_at, side="left"))
        if start_idx < min_start_idx or start_idx >= len(base):
            continue
        candidates.append((pool_idx, pool))
    candidates.sort(key=lambda item: (_ts(item[1].available_at) or pd.Timestamp.min,
                                      float(item[1].score)), reverse=True)

    from .pools import build_pools

    for pool_idx, pool in candidates[:samples]:
        available_at = _ts(pool.available_at)
        if available_at is None:
            summary["skipped"] += 1
            continue
        truncated = {
            tf: _truncate_closed(df, available_at)
            for tf, df in tf_data.items()
            if df is not None and not df.empty
        }
        if "base" not in truncated or truncated["base"].empty:
            summary["skipped"] += 1
            continue
        try:
            replayed_pools = build_pools(truncated, ad.final_cfg)
        except Exception as exc:
            summary["misses"] += 1
            issues.append(_issue(
                severity, "replay_determinism", symbol, pool_idx,
                "detector replay failed on data truncated at pool availability",
                available_at=str(available_at),
                error=str(exc),
            ))
            continue

        recent_base = truncated["base"].tail(100)
        median_range = 0.0
        if not recent_base.empty and {"high", "low"}.issubset(recent_base.columns):
            median_range = float((recent_base["high"] - recent_base["low"]).median())
        tolerance = max(float(pool.width) * 2.0, median_range, 1e-9)
        matched = any(_pool_match(pool, rp, tolerance) for rp in replayed_pools)
        summary["checked"] += 1
        if not matched:
            summary["misses"] += 1
            issues.append(_issue(
                severity, "replay_determinism", symbol, pool_idx,
                "batch pool was not reproduced by detector replay truncated at available_at",
                available_at=str(available_at),
                side=pool.side,
                pool_low=float(pool.price_low),
                pool_high=float(pool.price_high),
                tolerance=float(tolerance),
                replayed_pool_count=int(len(replayed_pools)),
            ))

    return issues, summary


def build_leakage_audit_for_report(report, cfg: Config,
                                   max_issues: int = 5000,
                                   requested_embargo_bars: Optional[int] = None,
                                   allow_short_embargo: bool = False,
                                   replay_audit_samples: int = 0,
                                   replay_severity: str = "WARN") -> Tuple[Dict, pd.DataFrame]:
    issues: List[LeakageIssue] = []
    per_symbol: Dict[str, Dict] = {}
    lag_rows: List[Dict] = []
    replay_summary = {
        "samples_per_symbol": int(max(0, replay_audit_samples)),
        "checked": 0,
        "misses": 0,
        "skipped": 0,
        "severity": replay_severity.upper(),
        "per_symbol": {},
    }

    embargo_issues, embargo_summary = _audit_embargo(
        report,
        cfg,
        requested_embargo_bars=requested_embargo_bars,
        allow_short_embargo=allow_short_embargo,
    )
    issues.extend(embargo_issues)

    for symbol, ad in report.assets.items():
        base = ad.base_df
        symbol_issues: List[LeakageIssue] = []
        symbol_issues.extend(_audit_bar_index(symbol, base))
        period_seconds = _bar_period_seconds(base.index) if base is not None and not base.empty else 0.0
        n_pools = 0
        n_touched = 0
        if base is not None and not base.empty:
            for pool_idx, (pool, result) in enumerate(zip(ad.final_pools, ad.final_results)):
                n_pools += 1
                n_touched += int(result.touched_at is not None)
                symbol_issues.extend(_audit_pool(
                    symbol, pool_idx, pool, result, base, ad.final_cfg, period_seconds,
                ))
            lag_rows.extend(_contributors_lag_rows(symbol, list(ad.final_pools), period_seconds))
            replay_issues, symbol_replay = _replay_audit_for_symbol(
                symbol,
                ad,
                samples=int(max(0, replay_audit_samples)),
                severity=replay_severity.upper(),
            )
            symbol_issues.extend(replay_issues)
            replay_summary["checked"] += int(symbol_replay.get("checked", 0))
            replay_summary["misses"] += int(symbol_replay.get("misses", 0))
            replay_summary["skipped"] += int(symbol_replay.get("skipped", 0))
            replay_summary["per_symbol"][symbol] = symbol_replay
        issues.extend(symbol_issues)
        per_symbol[symbol] = {
            "base_rows": int(len(base)) if base is not None else 0,
            "base_start": str(base.index[0]) if base is not None and not base.empty else None,
            "base_end": str(base.index[-1]) if base is not None and not base.empty else None,
            "base_period_seconds": float(period_seconds),
            "final_pools": int(n_pools),
            "touched_pools": int(n_touched),
            "replay_checked": int(replay_summary["per_symbol"].get(symbol, {}).get("checked", 0)),
            "replay_misses": int(replay_summary["per_symbol"].get(symbol, {}).get("misses", 0)),
            "issues": int(len(symbol_issues)),
            "errors": int(sum(1 for i in symbol_issues if i.severity == "ERROR")),
            "warnings": int(sum(1 for i in symbol_issues if i.severity == "WARN")),
        }

    issue_rows = [i.to_dict() for i in issues[:max_issues]]
    df = pd.DataFrame(issue_rows)
    counts = {
        "ERROR": int(sum(1 for i in issues if i.severity == "ERROR")),
        "WARN": int(sum(1 for i in issues if i.severity == "WARN")),
        "INFO": int(sum(1 for i in issues if i.severity == "INFO")),
    }
    status = "FAIL" if counts["ERROR"] else "WARN" if counts["WARN"] else "PASS"
    summary = {
        "status": status,
        "issue_count": int(len(issues)),
        "issue_count_truncated_to": int(min(len(issues), max_issues)),
        "severity_counts": counts,
        "embargo": embargo_summary,
        "replay_audit": replay_summary,
        "mtf_lag_summary": _summarise_lags(lag_rows),
        "per_symbol": per_symbol,
        "checks": [
            "bar_index",
            "bar_columns",
            "contributor_availability",
            "mtf_closed_bar",
            "pool_availability",
            "label_window",
            "embargo_horizon",
            "replay_determinism",
        ],
        "notes": [
            "This audit is a release gate, not a proof of no leakage.",
            "Replay audit is sampled and price-tolerant; a pass is not a full proof of detector causality.",
        ],
    }
    return summary, df
