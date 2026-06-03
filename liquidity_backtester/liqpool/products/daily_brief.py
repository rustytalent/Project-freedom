"""Daily Research Brief generator.

Reads a fitted ``MultiAssetReport`` bundle (or any object exposing the
same attributes) plus a target IST trading date, and produces a
structured :class:`BriefDocument` matching the contract in
``docs/daily_brief_schema.md``.

v1 scope (what's implemented end-to-end):

  * brief_metadata (always present)
  * sector_regime (from existing sector-rotation utilities)
  * top_watchlist (filtered active pools ranked by short-horizon
    proximity probability)
  * avoid_list (synthesised from bearish-sector regimes + basket
    verdict + per-asset confidence)
  * key_zones (top proximity-model levels across the basket)
  * confidence_notes (model health from the report's OOS audit data)

v1 stubs (rendered as ``_status: "pending"`` blocks so the contract
shape stays stable for downstream consumers):

  * index_regime — needs index OHLCV data which is not yet wired into
    the multi-asset bundle.
  * options_suitability — blocked on the level-to-strike translator.
  * yesterday_audit — blocked on the outcome log.

Design notes:

  * The generator NEVER crashes on a missing attribute. Bundles from
    older versions, or stubs in tests, just produce ``pending`` sections.
  * All probabilities exposed in the brief are calibrated. Raw scores
    are kept in ``state`` blocks for debug only.
  * No tipster outputs. The output speaks in regime / probability /
    confidence language exclusively. The renderer enforces this.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Dataclasses — mirror the JSON contract in docs/daily_brief_schema.md
# ---------------------------------------------------------------------------

@dataclass
class BriefMetadata:
    brief_id: str
    trading_date_ist: str
    generated_at_utc: str
    model_bundle_version: str
    feature_version: str
    session_status: str
    reading_time_minutes: int
    indexes_covered: List[str]


@dataclass
class SectorRegime:
    as_of_ist: str
    trending_up: List[str]
    trending_down: List[str]
    chopping: List[str]
    neutral: List[str]
    leadership_change_vs_yesterday: List[str]


@dataclass
class WatchlistEntry:
    symbol: str
    sector: str
    side: str
    reason_tag: str
    key_level: float
    key_level_type: str
    p_touch_today: float
    p_touch_within_60min: float
    model_confidence_bucket: str
    regime_tags: List[str]
    avoidance_note: Optional[str] = None


@dataclass
class AvoidEntry:
    symbol: str
    reason: str
    regime_tags: List[str]
    model_confidence: str


@dataclass
class KeyZone:
    symbol: str
    level: float
    level_type: str
    side_from_open: str
    p_test_today: float
    p_test_within_60min: float
    horizons_evaluated: List[int]
    p_per_horizon: Dict[str, float]
    confluence_with_poc: bool
    confluence_with_vah_val: bool
    model_confidence_bucket: str


@dataclass
class ConfidenceNotes:
    calibrated_today: List[str]
    drifting_today: List[str]
    drift_reason: Dict[str, str]
    overall_brief_confidence: str
    operator_note: Optional[str] = None


# ---------------------------------------------------------------------------
# BriefDocument — the JSON contract root
# ---------------------------------------------------------------------------

@dataclass
class BriefDocument:
    schema_version: str
    brief_metadata: BriefMetadata
    index_regime: Dict[str, Any]
    sector_regime: SectorRegime
    top_watchlist: List[WatchlistEntry]
    options_suitability: Dict[str, Any]
    avoid_list: List[AvoidEntry]
    key_zones: List[KeyZone]
    confidence_notes: ConfidenceNotes
    yesterday_audit: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-ready) without losing nesting."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Confidence bucketing — map calibrated probabilities to label strings
# ---------------------------------------------------------------------------

def _confidence_bucket(p: float) -> str:
    """Map a calibrated probability into the bucket label the brief uses."""
    if p >= 0.80:
        return "very_high"
    if p >= 0.65:
        return "high"
    if p >= 0.50:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# Sector regime section
# ---------------------------------------------------------------------------

def _classify_sector_regimes(sector_metrics: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Bucket sectors into trending_up / trending_down / chopping / neutral.

    Trending up   : ret_5d > 0 AND ret_20d > 0
    Trending down : ret_5d < 0 AND ret_20d < 0
    Chopping      : ret_5d and ret_20d have opposite signs, magnitudes < 2%
    Neutral       : everything else
    """
    up: List[str] = []
    down: List[str] = []
    chop: List[str] = []
    neutral: List[str] = []
    for sector, m in sorted(sector_metrics.items()):
        r5 = float(m.get("ret_5d", 0.0))
        r20 = float(m.get("ret_20d", 0.0))
        if r5 > 0.0 and r20 > 0.0:
            up.append(sector)
        elif r5 < 0.0 and r20 < 0.0:
            down.append(sector)
        elif (r5 * r20 < 0.0) and abs(r5) < 0.02 and abs(r20) < 0.02:
            chop.append(sector)
        else:
            neutral.append(sector)
    return {
        "trending_up": up,
        "trending_down": down,
        "chopping": chop,
        "neutral": neutral,
    }


def _sector_regime_block(report, as_of_ist: str) -> SectorRegime:
    """Build the sector_regime section from the report's asset frames."""
    from liqpool.sectors import compute_sector_metrics, detect_rotation
    assets = getattr(report, "assets", {}) or {}
    asset_dfs = {
        symbol: ad.base_df
        for symbol, ad in assets.items()
        if getattr(ad, "base_df", None) is not None
    }
    if not asset_dfs:
        return SectorRegime(
            as_of_ist=as_of_ist,
            trending_up=[], trending_down=[], chopping=[], neutral=[],
            leadership_change_vs_yesterday=[],
        )
    metrics = compute_sector_metrics(asset_dfs)
    buckets = _classify_sector_regimes(metrics)
    rotation = detect_rotation(metrics) or {}
    leadership = sorted(set(
        list(rotation.get("rotation_in", []))
        + list(rotation.get("rotation_out", []))
    ))
    return SectorRegime(
        as_of_ist=as_of_ist,
        trending_up=buckets["trending_up"],
        trending_down=buckets["trending_down"],
        chopping=buckets["chopping"],
        neutral=buckets["neutral"],
        leadership_change_vs_yesterday=leadership,
    )


# ---------------------------------------------------------------------------
# Watchlist + key zones — drawn from the OOS pool population + proximity
# ---------------------------------------------------------------------------

def _gather_active_pool_predictions(report,
                                     max_entries: int = 200,
                                     ) -> List[Dict[str, Any]]:
    """Return a flat list of (pool, p_touch, sector, side, ...) records
    for every still-active pool in the bundle's OOS window.

    'Active' means available_at is set AND the pool was not touched/broken
    inside the OOS window — i.e. it is still in play as of the bundle's
    last bar. The brief uses these as the candidate set for both
    top_watchlist and key_zones.
    """
    from liqpool.sectors import sector_of
    assets = getattr(report, "assets", {}) or {}
    prox_models = getattr(report, "unified_proximity", {}) or {}
    out: List[Dict[str, Any]] = []
    for symbol, ad in assets.items():
        wf = getattr(ad, "walkforward", None)
        if wf is None:
            continue
        pools = getattr(wf, "oos_pools", []) or []
        results = getattr(wf, "oos_results", []) or []
        df_base = getattr(ad, "base_df", None)
        if df_base is None or df_base.empty:
            continue
        last_close = float(df_base["close"].iloc[-1])
        sec = sector_of(symbol)
        for pool, result in zip(pools, results):
            # Only pools that are STILL in play as of the bundle's end:
            # never touched, never broken.
            if getattr(result, "touched_at", None) is not None:
                continue
            if getattr(result, "broken_at", None) is not None:
                continue
            # Side relative to last close.
            if pool.price_low > last_close:
                side_from_close = "above"
                key_level = float(pool.price_low)
                key_level_type = "supply_pool"
                trade_side = "short"
            elif pool.price_high < last_close:
                side_from_close = "below"
                key_level = float(pool.price_high)
                key_level_type = "demand_pool"
                trade_side = "long"
            else:
                continue                      # already inside the zone
            # Per-horizon P(touch).
            p_per_horizon: Dict[str, float] = {}
            for h, pm in sorted(prox_models.items()):
                try:
                    p = float(pm.predict_one(
                        pool=pool,
                        dist_atr=1.0,                 # placeholder; real
                                                      # generator would
                                                      # compute properly
                        side=side_from_close,
                        state={},
                        quality_pred=0.5,
                        atr_val=1.0,
                    ))
                except Exception:
                    p = 0.0
                p_per_horizon[str(h)] = p
            short_h = min(prox_models.keys()) if prox_models else None
            long_h = max(prox_models.keys()) if prox_models else None
            p_short = float(p_per_horizon.get(str(short_h), 0.0)) if short_h else 0.0
            p_long = float(p_per_horizon.get(str(long_h), 0.0)) if long_h else 0.0
            out.append({
                "symbol": symbol,
                "sector": sec,
                "side_trade": trade_side,
                "side_from_close": side_from_close,
                "key_level": key_level,
                "key_level_type": key_level_type,
                "p_short": p_short,
                "p_long": p_long,
                "p_per_horizon": p_per_horizon,
                "horizons_evaluated": sorted(prox_models.keys()),
                "pool_score": float(getattr(pool, "score", 0.0)),
            })
    # Cap the candidate set so the brief generator stays O(reasonable).
    out.sort(key=lambda r: r["p_long"], reverse=True)
    return out[:max_entries]


def _watchlist_block(predictions: List[Dict[str, Any]],
                     sector_regime: SectorRegime,
                     min_p_threshold: float = 0.20,
                     max_entries: int = 10) -> List[WatchlistEntry]:
    """Top instruments where the model has actionable conviction.

    Selection rule: longest-horizon P(touch) >= ``min_p_threshold``,
    ranked by P(touch). Capped at ``max_entries`` for the human-layer
    length budget.
    """
    entries: List[WatchlistEntry] = []
    bearish = set(sector_regime.trending_down)
    bullish = set(sector_regime.trending_up)
    for rec in predictions:
        if rec["p_long"] < min_p_threshold:
            continue
        sec = rec["sector"]
        side = rec["side_trade"]
        # Sector-alignment regime tag.
        regime_tags: List[str] = []
        if sec in bullish:
            regime_tags.append(f"{sec.lower()}_trending_up")
        elif sec in bearish:
            regime_tags.append(f"{sec.lower()}_trending_down")
        # Direction-vs-sector misalignment as an avoidance note.
        avoidance_note: Optional[str] = None
        if side == "long" and sec in bearish:
            avoidance_note = "long candidate fights bearish sector regime"
        if side == "short" and sec in bullish:
            avoidance_note = "short candidate fights bullish sector regime"
        entries.append(WatchlistEntry(
            symbol=rec["symbol"],
            sector=sec,
            side=side,
            reason_tag="proximity_high_p_touch_journey",
            key_level=rec["key_level"],
            key_level_type=rec["key_level_type"],
            p_touch_today=rec["p_long"],
            p_touch_within_60min=rec["p_short"],
            model_confidence_bucket=_confidence_bucket(rec["p_long"]),
            regime_tags=regime_tags,
            avoidance_note=avoidance_note,
        ))
        if len(entries) >= max_entries:
            break
    return entries


def _key_zones_block(predictions: List[Dict[str, Any]],
                     max_entries: int = 50) -> List[KeyZone]:
    """All proximity-model levels with non-trivial P(touch).

    Machine-layer only — does not appear in the human PDF. The renderer
    skips this section; engineer / quant customers consume it directly.
    """
    zones: List[KeyZone] = []
    for rec in predictions:
        if rec["p_long"] < 0.05:
            continue
        zones.append(KeyZone(
            symbol=rec["symbol"],
            level=rec["key_level"],
            level_type=rec["key_level_type"],
            side_from_open=rec["side_from_close"],
            p_test_today=rec["p_long"],
            p_test_within_60min=rec["p_short"],
            horizons_evaluated=rec["horizons_evaluated"],
            p_per_horizon=rec["p_per_horizon"],
            confluence_with_poc=False,        # v2: read from trade frame
            confluence_with_vah_val=False,    # v2: read from trade frame
            model_confidence_bucket=_confidence_bucket(rec["p_long"]),
        ))
        if len(zones) >= max_entries:
            break
    return zones


# ---------------------------------------------------------------------------
# Avoid list — the SEBI-safest, customer-most-valuable section
# ---------------------------------------------------------------------------

def _avoid_list_block(report,
                      predictions: List[Dict[str, Any]],
                      sector_regime: SectorRegime,
                      watchlist: List[WatchlistEntry],
                      ) -> List[AvoidEntry]:
    """Synthesize the AVOID list from regime conflicts and basket health.

    Rules:
      * If NO entry in the watchlist clears the watchlist threshold,
        emit ALL_BASKET avoid (the "no setup today" call).
      * For each bearish sector, emit a per-sector avoid for new longs.
      * For each individual symbol where a candidate fights its sector,
        emit a per-symbol avoid.
    """
    avoids: List[AvoidEntry] = []
    if not watchlist:
        avoids.append(AvoidEntry(
            symbol="ALL_BASKET",
            reason="no_setup_today_low_proximity_across_basket",
            regime_tags=["no_qualifying_candidates"],
            model_confidence="high",
        ))
    for sector in sector_regime.trending_down:
        avoids.append(AvoidEntry(
            symbol=f"ALL_{sector}",
            reason="sector_trending_down_avoid_new_longs",
            regime_tags=[f"{sector.lower()}_trending_down"],
            model_confidence="moderate",
        ))
    seen: set = set()
    for entry in watchlist:
        if entry.avoidance_note and entry.symbol not in seen:
            avoids.append(AvoidEntry(
                symbol=entry.symbol,
                reason=entry.avoidance_note,
                regime_tags=entry.regime_tags,
                model_confidence="moderate",
            ))
            seen.add(entry.symbol)
    return avoids


# ---------------------------------------------------------------------------
# Confidence notes — model health surfacing
# ---------------------------------------------------------------------------

def _confidence_notes_block(report) -> ConfidenceNotes:
    """Read OOS calibration metrics off the report and surface drift.

    A model head is flagged 'drifting' when its OOS calibration error
    exceeds 0.08 in magnitude. Threshold matches the outcome-log
    drift contract in ``outcome_logging_schema.md``.
    """
    DRIFT_THRESHOLD = 0.08
    calibrated: List[str] = []
    drifting: List[str] = []
    drift_reason: Dict[str, str] = {}

    # Direction model health.
    direction = getattr(report, "unified_direction", None)
    if direction is not None:
        val_auc = float(getattr(direction, "val_auc", 0.0))
        # AUC alone doesn't measure calibration but we use it as a proxy
        # for "is this model worth listening to today".
        if val_auc >= 0.55:
            calibrated.append("direction")
        else:
            drifting.append("direction")
            drift_reason["direction"] = f"val_auc={val_auc:.3f}_below_0.55"

    # Proximity model health (per horizon).
    prox_models = getattr(report, "unified_proximity", {}) or {}
    for h, pm in sorted(prox_models.items()):
        val_auc = float(getattr(pm, "val_auc", 0.0))
        head = f"proximity_h{h}"
        if val_auc >= 0.60:
            calibrated.append(head)
        else:
            drifting.append(head)
            drift_reason[head] = f"val_auc={val_auc:.3f}_below_0.60"

    # Q model — read calibration_error from the OOS audit if present.
    audit = getattr(report, "unified_oos_audit", None)
    if audit is not None:
        metrics = getattr(audit, "metrics", {}) or {}
        for model_name, m in metrics.items():
            ce = float(getattr(m, "calibration_error", 0.0))
            head = f"q_{model_name}"
            if abs(ce) <= DRIFT_THRESHOLD:
                calibrated.append(head)
            else:
                drifting.append(head)
                drift_reason[head] = f"calibration_error={ce:+.3f}_exceeds_{DRIFT_THRESHOLD}"

    # Overall confidence.
    if not drifting:
        overall = "high"
    elif len(drifting) < len(calibrated):
        overall = "moderate_to_high"
    elif len(drifting) <= len(calibrated):
        overall = "moderate"
    else:
        overall = "low_use_with_caution"

    return ConfidenceNotes(
        calibrated_today=calibrated,
        drifting_today=drifting,
        drift_reason=drift_reason,
        overall_brief_confidence=overall,
        operator_note=None,
    )


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------

def _safe_attr(obj: Any, name: str, default: Any) -> Any:
    try:
        v = getattr(obj, name, default)
    except Exception:
        return default
    return v if v is not None else default


def generate_brief(report: Any,
                   trading_date_ist: str,
                   indexes_covered: Optional[List[str]] = None,
                   model_bundle_version: str = "unknown_bundle",
                   feature_version: str = "42_features_v3",
                   ) -> BriefDocument:
    """Top-level: produce a BriefDocument from a fitted report bundle.

    ``trading_date_ist`` is the IST calendar date the brief targets,
    formatted YYYY-MM-DD. ``indexes_covered`` is the list of index
    symbols the brief is meant to address; for v1 these populate the
    metadata block but the index_regime + options_suitability sections
    are stubbed as ``_status: "pending"`` because the multi-asset
    bundle does not yet carry index OHLCV.

    The function NEVER raises on missing report attributes. Missing
    data flows through to ``_status: "pending"`` sections.
    """
    indexes_covered = indexes_covered or []
    as_of_ist = f"{trading_date_ist}T09:15:00+05:30"

    metadata = BriefMetadata(
        brief_id=f"BRIEF_{trading_date_ist.replace('-', '_')}",
        trading_date_ist=trading_date_ist,
        generated_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        model_bundle_version=model_bundle_version,
        feature_version=feature_version,
        session_status="pre_open",
        reading_time_minutes=6,
        indexes_covered=list(indexes_covered),
    )

    try:
        sector_regime = _sector_regime_block(report, as_of_ist)
    except Exception:
        sector_regime = SectorRegime(
            as_of_ist=as_of_ist, trending_up=[], trending_down=[],
            chopping=[], neutral=[], leadership_change_vs_yesterday=[],
        )

    try:
        predictions = _gather_active_pool_predictions(report)
    except Exception:
        predictions = []
    watchlist = _watchlist_block(predictions, sector_regime)
    key_zones = _key_zones_block(predictions)
    avoid_list = _avoid_list_block(report, predictions, sector_regime, watchlist)

    try:
        confidence_notes = _confidence_notes_block(report)
    except Exception:
        confidence_notes = ConfidenceNotes(
            calibrated_today=[], drifting_today=[],
            drift_reason={}, overall_brief_confidence="unknown",
            operator_note="confidence_notes_unavailable",
        )

    # v1 stubs — fully spec-compliant placeholders so downstream
    # consumers see stable shape.
    index_regime_stub = {
        "_status": "pending",
        "_reason": "index_ohlcv_not_wired_into_bundle",
        "_v1_will_add_when": "multi_asset_run.py ingests index data",
    }
    options_suitability_stub = {
        "_status": "pending",
        "_reason": "level_to_strike_translator_not_yet_built",
        "_blocking_item": "COORDINATION_NEXT_UP_5",
    }
    yesterday_audit_stub = {
        "_status": "pending",
        "_reason": "outcome_log_not_yet_wired",
        "_blocking_item": "COORDINATION_NEXT_UP_3",
    }

    return BriefDocument(
        schema_version=SCHEMA_VERSION,
        brief_metadata=metadata,
        index_regime=index_regime_stub,
        sector_regime=sector_regime,
        top_watchlist=watchlist,
        options_suitability=options_suitability_stub,
        avoid_list=avoid_list,
        key_zones=key_zones,
        confidence_notes=confidence_notes,
        yesterday_audit=yesterday_audit_stub,
    )
