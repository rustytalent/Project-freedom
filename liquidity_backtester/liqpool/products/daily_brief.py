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

from .outcome_log import (
    OutcomeLogWriter,
    PredictionRecord,
    make_prediction_id,
)
from .strike_translator import (
    INDEX_CONFIGS,
    is_known_index,
    premium_regime_for_buyers_and_sellers,
    theta_danger_score,
    translate_proximity,
)

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

    Per-pool inputs to the proximity model:

      * ``dist_atr`` — true distance (in ATR units) from the bundle's
        last close to the pool's near boundary. CRITICAL: this used to
        be a hardcoded 1.0 placeholder, which caused the proximity model
        to return ~98% probability for every pool regardless of how far
        away it actually was. Codex's first real-bundle backfill caught
        this immediately (calibration_error 0.98 on the very_high bucket
        for proximity predictions). Now computed from real prices.
      * ``state`` — the StateFeaturizer's per-bar features at the last
        bar of the asset's OOS window. Falls back to {} on featurizer
        failure; the proximity model treats missing features as 0.0.
      * ``quality_pred`` — per-pool Q from report.unified_ml if both
        unified_ml and unified_featurizer are present in the report;
        otherwise 0.5 (neutral). Q is a model input feature for the
        proximity head.
    """
    from liqpool.sectors import sector_of
    assets = getattr(report, "assets", {}) or {}
    prox_models = getattr(report, "unified_proximity", {}) or {}
    out: List[Dict[str, Any]] = []

    # Q + featurizer: optional. When present, batch-predict Q for the
    # full active-pool set so each predict_one call has real Q context.
    unified_ml = getattr(report, "unified_ml", None)
    unified_featurizer = getattr(report, "unified_featurizer", None)

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
        # ATR at the last bar — the denominator for distance_atr. Use
        # the same window as the StateFeaturizer (atr_14) to keep
        # downstream feature comparisons honest.
        try:
            from liqpool.indicators import atr as _atr
            atr_series = _atr(df_base, 14).bfill()
            atr_at_last = max(float(atr_series.iloc[-1]), 1e-9)
        except Exception:
            atr_at_last = 1.0

        # Build the state feature dict at the last bar.
        last_state: Dict[str, float] = {}
        try:
            from liqpool.timing import StateFeaturizer
            sf = StateFeaturizer(df_base)
            last_state = sf.features_at(len(df_base) - 1, active_pools=[])
        except Exception:
            last_state = {}

        # Per-pool Q predictions — best effort. If anything fails, every
        # pool gets neutral q=0.5.
        per_pool_q: Dict[int, float] = {}
        if unified_ml is not None and unified_featurizer is not None and pools:
            try:
                X_pools = unified_featurizer.transform_batch(pools)
                q_preds = unified_ml.predict(X_pools, pools=pools)
                for i, q in enumerate(q_preds):
                    q_val = float(q) if q is not None else 0.5
                    if not np.isfinite(q_val):
                        q_val = 0.5
                    per_pool_q[i] = q_val
            except Exception:
                per_pool_q = {}

        sec = sector_of(symbol)
        for pool_idx, (pool, result) in enumerate(zip(pools, results)):
            # Only pools that are STILL in play as of the bundle's end:
            # never touched, never broken.
            if getattr(result, "touched_at", None) is not None:
                continue
            if getattr(result, "broken_at", None) is not None:
                continue
            # Side + true distance relative to last close.
            if pool.price_low > last_close:
                side_from_close = "above"
                key_level = float(pool.price_low)
                key_level_type = "supply_pool"
                trade_side = "short"
                dist_atr = (pool.price_low - last_close) / atr_at_last
            elif pool.price_high < last_close:
                side_from_close = "below"
                key_level = float(pool.price_high)
                key_level_type = "demand_pool"
                trade_side = "long"
                dist_atr = (last_close - pool.price_high) / atr_at_last
            else:
                continue                      # already inside the zone
            q_for_pool = per_pool_q.get(pool_idx, 0.5)
            # Per-horizon P(touch) — predict_one fed the REAL distance,
            # the real state at the last bar, and the real per-pool Q.
            # Previously dist_atr was a 1.0 placeholder which caused
            # the proximity model to return ~98% for every pool. Fixed.
            p_per_horizon: Dict[str, float] = {}
            for h, pm in sorted(prox_models.items()):
                try:
                    p = float(pm.predict_one(
                        pool=pool,
                        dist_atr=float(dist_atr),
                        side=side_from_close,
                        state=last_state,
                        quality_pred=q_for_pool,
                        atr_val=atr_at_last,
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
                "dist_atr_at_last_bar": float(dist_atr),
                "q_pred": float(q_for_pool),
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
                   outcome_log_writer: Optional[OutcomeLogWriter] = None,
                   index_data: Optional[Dict[str, Dict[str, Any]]] = None,
                   retrospective: bool = False,
                   publish_to_website: bool = False,
                   website_tier: str = "paid_intraday",
                   ) -> BriefDocument:
    """Top-level: produce a BriefDocument from a fitted report bundle.

    ``trading_date_ist`` is the IST calendar date the brief targets,
    formatted YYYY-MM-DD.

    ``indexes_covered`` populates the metadata block. When ``index_data``
    is also passed AND an entry exists for an index in ``indexes_covered``,
    the options_suitability section is populated via the
    ``strike_translator`` (level-to-strike + theta-danger / premium
    regime). When ``index_data`` is missing for an index, the
    options_suitability stays a ``pending`` stub for that index only.

    ``outcome_log_writer`` — when provided, every prediction the brief
    makes is logged via the writer with a deterministic prediction_id
    for later resolution joining. When omitted, predictions are not
    logged (used for synthetic / test runs).

    ``retrospective`` — when True, every PredictionRecord written via
    the writer is tagged ``is_retrospective=True``. The backfill driver
    (``analysis/backfill_outcome_log.py``) sets this to True so that
    tomorrow's Yesterday Audit can disclose whether yesterday's
    calibration came from retrospective replay rather than live history.
    Default False (live brief generation).

    ``publish_to_website`` — when True AND the env vars
    ``WEBSITE_BASE_URL`` + ``ENGINE_INGEST_TOKEN`` are both set, the
    generated brief's JSON representation is POSTed to the website's
    artifact endpoint at ``website_tier`` access. The push is fire-
    and-forget: a network or auth failure logs a warning but never
    raises, so a website outage cannot break brief generation. Default
    False so existing test/backfill callers are unaffected.

    ``index_data`` schema per index::

        {
          "NIFTY50": {
            "previous_close": 24521.30,
            "expected_gap_atr": 0.42,
            "vol_regime": "normal",
            "vol_regime_zscore_20d": 0.34,
            "directional_bias": "mild_up",
            "expected_range_today_atr": 1.6,
            "path_efficiency_30": 0.55,
            "direction_changes_30": 12.0,
            "proximity_predictions": [
              {"level": 24500, "p_test_today": 0.78,
               "p_test_within_60min": 0.34,
               "side_from_open": "below", "key_level_type": "demand_pool"},
              ...
            ],
          },
          ...
        }

    The function NEVER raises on missing report attributes. Missing
    data flows through to ``_status: "pending"`` sections.
    """
    indexes_covered = indexes_covered or []
    index_data = index_data or {}
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

    # Index regime stays a stub block per-index until index OHLCV is
    # wired into the bundle. When index_data carries it, the per-index
    # block can populate; until then we surface the partial shape.
    index_regime_payload: Dict[str, Any] = {}
    for idx_name in indexes_covered:
        idx_payload = index_data.get(idx_name)
        if idx_payload is None:
            index_regime_payload[idx_name] = {
                "_status": "pending",
                "_reason": "index_data_not_provided_for_this_index",
            }
            continue
        index_regime_payload[idx_name] = {
            "previous_close": idx_payload.get("previous_close"),
            "expected_gap_atr": idx_payload.get("expected_gap_atr"),
            "vol_regime": idx_payload.get("vol_regime"),
            "vol_regime_zscore_20d": idx_payload.get("vol_regime_zscore_20d"),
            "today_session_character_prediction": idx_payload.get(
                "today_session_character_prediction"),
            "trap_risk_score": idx_payload.get("trap_risk_score"),
            "trap_pattern_active": idx_payload.get("trap_pattern_active"),
        }

    # options_suitability section — populated per index via the
    # strike translator when index_data is available. Indexes without
    # data fall through to a per-index pending block, so the rest of
    # the structure is still consumable.
    options_suitability_payload: Dict[str, Any] = {}
    for idx_name in indexes_covered:
        idx_payload = index_data.get(idx_name)
        if idx_payload is None or not is_known_index(idx_name):
            options_suitability_payload[idx_name] = {
                "_status": "pending",
                "_reason": ("index_data_not_provided_for_this_index"
                             if idx_payload is None
                             else "index_not_in_strike_grid_config"),
            }
            continue
        prox_predictions = idx_payload.get("proximity_predictions") or []
        try:
            strike_levels = translate_proximity(prox_predictions, idx_name)
        except Exception:
            strike_levels = []
        theta_score = theta_danger_score(
            path_efficiency_30=float(
                idx_payload.get("path_efficiency_30") or 0.5),
            direction_changes_30=float(
                idx_payload.get("direction_changes_30") or 0.0),
            vol_regime_zscore_20d=float(
                idx_payload.get("vol_regime_zscore_20d") or 0.0),
        )
        regime_pair = premium_regime_for_buyers_and_sellers(theta_score)
        options_suitability_payload[idx_name] = {
            "directional_bias": idx_payload.get("directional_bias", "neutral"),
            "expected_range_today_atr": idx_payload.get(
                "expected_range_today_atr"),
            "expected_range_today_points": idx_payload.get(
                "expected_range_today_points"),
            "movement_quality_prediction": idx_payload.get(
                "movement_quality_prediction"),
            "theta_danger_score": theta_score,
            "strike_levels_in_play": [
                {
                    "strike": s.strike,
                    "p_test_today": s.p_test_today,
                    "p_test_within_60min": s.p_test_within_60min,
                    "side_from_open": s.side_from_open,
                    "key_level_type": s.key_level_type,
                    "underlying_level": s.underlying_level,
                }
                for s in strike_levels
            ],
            **regime_pair,
            "session_recommendation": "context_only_no_directional_call",
        }
    if not options_suitability_payload:
        options_suitability_payload = {
            "_status": "pending",
            "_reason": "no_indexes_covered",
        }

    # Yesterday audit — populated from the joined outcome log if a
    # writer was provided. We compute the previous IST calendar date by
    # subtracting one calendar day; this is approximate (weekends and
    # holidays will miss), but the writer returns an empty frame in
    # those cases and we fall back to the pending stub.
    yesterday_audit_payload: Dict[str, Any]
    if outcome_log_writer is not None:
        try:
            prev_ist = (pd.Timestamp(trading_date_ist)
                        - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            joined = outcome_log_writer.read_joined(prev_ist)
            if joined is None or joined.empty:
                yesterday_audit_payload = {
                    "_status": "pending",
                    "_reason": "no_predictions_logged_for_previous_date",
                    "_previous_ist": prev_ist,
                }
            else:
                from .outcome_log import calibration_by_bucket
                cal = calibration_by_bucket(joined)
                # Stream G — retrospective-share disclosure. If the
                # joined frame predates the flag, treat NaN as live
                # (False) so old partitions render the same as before.
                if "is_retrospective" in joined.columns:
                    retro_series = joined["is_retrospective"].fillna(False)
                    retro_share = float(retro_series.astype(bool).mean())
                else:
                    retro_share = 0.0
                yesterday_audit_payload = {
                    "yesterday_brief_id": (
                        joined["brief_id"].iloc[0]
                        if "brief_id" in joined.columns else "unknown"),
                    "predictions_made": int(len(joined)),
                    "predictions_resolved": int(joined["resolved"].sum()
                                                  if "resolved" in joined.columns
                                                  else 0),
                    "hit_rate_by_confidence_bucket": cal.to_dict(orient="records")
                        if not cal.empty else [],
                    "retrospective_share": retro_share,
                    "is_retrospective_calibration": retro_share > 0.5,
                }
        except Exception as exc:
            yesterday_audit_payload = {
                "_status": "pending",
                "_reason": f"yesterday_audit_compute_failed_{type(exc).__name__}",
            }
    else:
        yesterday_audit_payload = {
            "_status": "pending",
            "_reason": "outcome_log_writer_not_provided",
            "_blocking_item": "COORDINATION_NEXT_UP_3",
        }

    # Log predictions to the outcome log writer (the flywheel).
    if outcome_log_writer is not None:
        _log_brief_predictions(
            outcome_log_writer,
            metadata=metadata,
            watchlist=watchlist,
            avoid_list=avoid_list,
            options_suitability=options_suitability_payload,
            model_bundle_version=model_bundle_version,
            feature_version=feature_version,
            retrospective=retrospective,
        )

    doc = BriefDocument(
        schema_version=SCHEMA_VERSION,
        brief_metadata=metadata,
        index_regime=index_regime_payload or {
            "_status": "pending",
            "_reason": "no_indexes_covered",
        },
        sector_regime=sector_regime,
        top_watchlist=watchlist,
        options_suitability=options_suitability_payload,
        avoid_list=avoid_list,
        key_zones=key_zones,
        confidence_notes=confidence_notes,
        yesterday_audit=yesterday_audit_payload,
    )

    if publish_to_website:
        _publish_brief_to_website(
            doc, trading_date_ist=trading_date_ist,
            tier=website_tier, retrospective=retrospective,
        )

    return doc


def _log_brief_predictions(writer: OutcomeLogWriter,
                            *,
                            metadata: BriefMetadata,
                            watchlist: List[WatchlistEntry],
                            avoid_list: List[AvoidEntry],
                            options_suitability: Dict[str, Any],
                            model_bundle_version: str,
                            feature_version: str,
                            retrospective: bool = False) -> None:
    """Emit one PredictionRecord per call the brief makes, then commit.

    Predictions get logged for three call types:
      * proximity (one per watchlist entry's key level)
      * avoidance (one per avoid_list entry)
      * options_strike (one per strike_levels_in_play entry per index)

    Each gets a deterministic prediction_id (via ``make_prediction_id``)
    so the end-of-day resolver can find and update them.
    """
    common = dict(
        brief_id=metadata.brief_id,
        generated_at_utc=metadata.generated_at_utc,
        trading_date_ist=metadata.trading_date_ist,
        model_bundle_version=model_bundle_version,
        feature_version=feature_version,
        is_retrospective=bool(retrospective),
    )

    # 1) Proximity predictions from the watchlist.
    for entry in watchlist:
        pid = make_prediction_id(
            brief_id=metadata.brief_id,
            symbol=entry.symbol,
            prediction_type="proximity",
            detail_token=f"level_{entry.key_level:.2f}_today",
        )
        writer.write_prediction(PredictionRecord(
            prediction_id=pid,
            instrument_kind="equity",
            symbol=entry.symbol,
            prediction_type="proximity",
            target_description={
                "kind": "level_touch",
                "level": float(entry.key_level),
                "level_type": entry.key_level_type,
                "side_from_open": (
                    "above" if entry.side == "short" else "below"),
                "horizon_label": "today",
            },
            predicted_value=float(entry.p_touch_today),
            predicted_value_kind="calibrated_probability",
            confidence_bucket=entry.model_confidence_bucket,
            raw_score_diagnostic=float(entry.p_touch_today),
            regime_tags_at_prediction={
                "sector": entry.sector,
                "regime_tags": entry.regime_tags,
            },
            **common,
        ))

    # 2) Avoidance predictions.
    for avoid in avoid_list:
        pid = make_prediction_id(
            brief_id=metadata.brief_id,
            symbol=avoid.symbol,
            prediction_type="avoidance",
            detail_token=avoid.reason[:40],
        )
        writer.write_prediction(PredictionRecord(
            prediction_id=pid,
            instrument_kind=("basket" if avoid.symbol.startswith("ALL_")
                              else "equity"),
            symbol=avoid.symbol,
            prediction_type="avoidance",
            target_description={
                "kind": "avoid_today",
                "reason": avoid.reason,
            },
            predicted_value=1.0,         # we recommend avoid with conviction
            predicted_value_kind="boolean",
            confidence_bucket=avoid.model_confidence,
            raw_score_diagnostic=1.0,
            regime_tags_at_prediction={"regime_tags": avoid.regime_tags},
            **common,
        ))

    # 3) Options-strike predictions per index.
    if isinstance(options_suitability, dict):
        for idx_name, idx_block in options_suitability.items():
            if not isinstance(idx_block, dict):
                continue
            for strike_entry in idx_block.get("strike_levels_in_play",
                                                []) or []:
                if not isinstance(strike_entry, dict):
                    continue
                strike = strike_entry.get("strike")
                if strike is None:
                    continue
                pid = make_prediction_id(
                    brief_id=metadata.brief_id,
                    symbol=idx_name,
                    prediction_type="options_strike",
                    detail_token=f"strike_{int(strike)}_today",
                )
                writer.write_prediction(PredictionRecord(
                    prediction_id=pid,
                    instrument_kind="index",
                    symbol=idx_name,
                    prediction_type="options_strike",
                    target_description={
                        "kind": "strike_test",
                        "strike": float(strike),
                        "side_from_open": strike_entry.get(
                            "side_from_open", "above"),
                        "underlying_level": strike_entry.get(
                            "underlying_level"),
                        "horizon_label": "today",
                    },
                    predicted_value=float(
                        strike_entry.get("p_test_today", 0.0)),
                    predicted_value_kind="calibrated_probability",
                    confidence_bucket=_confidence_bucket(
                        float(strike_entry.get("p_test_today", 0.0))),
                    raw_score_diagnostic=float(
                        strike_entry.get("p_test_today", 0.0)),
                    regime_tags_at_prediction={
                        "key_level_type": strike_entry.get(
                            "key_level_type", "unknown"),
                    },
                    **common,
                ))

    writer.commit()


def _publish_brief_to_website(doc: "BriefDocument", *,
                               trading_date_ist: str,
                               tier: str,
                               retrospective: bool) -> None:
    """Fire-and-forget upload of the brief's JSON to the website.

    Skipped silently when WEBSITE_BASE_URL / ENGINE_INGEST_TOKEN are
    not configured — that's normal in dev, test, and backfill runs.
    Any error is logged and swallowed; a website outage must NOT
    break brief generation.
    """
    import json as _json
    import logging as _logging
    import os as _os
    import tempfile as _tempfile
    from pathlib import Path as _Path

    _log = _logging.getLogger(__name__)

    base = _os.environ.get("WEBSITE_BASE_URL", "").strip()
    token = _os.environ.get("ENGINE_INGEST_TOKEN", "").strip()
    if not base or not token:
        _log.info(
            "skipping website publish: WEBSITE_BASE_URL or "
            "ENGINE_INGEST_TOKEN not set")
        return

    # Lazy import so the test suite doesn't carry a hard dep on the
    # pusher when publish_to_website is False (the default).
    try:
        from .artifact_pusher import push_artifact
    except Exception as exc:                                 # pragma: no cover
        _log.warning("artifact_pusher import failed: %s", exc)
        return

    payload = _json.dumps(doc.to_dict(), default=str, indent=2)
    description = (
        f"Daily Brief for IST {trading_date_ist} "
        f"(retrospective replay)" if retrospective else
        f"Daily Brief for IST {trading_date_ist}")
    try:
        with _tempfile.TemporaryDirectory() as td:
            p = _Path(td) / f"brief-{trading_date_ist}.json"
            p.write_text(payload, encoding="utf-8")
            push_artifact(
                path=p,
                kind="daily_brief_email",  # JSON serialisation, text-oid
                tier=tier,                  # type: ignore[arg-type]
                trading_date_ist=trading_date_ist,
                description=description,
                meta={"retrospective": bool(retrospective)},
            )
        _log.info("published brief %s to website", trading_date_ist)
    except Exception as exc:
        _log.warning(
            "website publish failed for brief %s: %s — "
            "brief generation continuing",
            trading_date_ist, exc,
        )
