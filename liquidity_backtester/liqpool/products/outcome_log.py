"""Outcome log — the data flywheel.

Two append-only tables, partitioned by IST trading date:

  * predictions  — one row per call the Daily Brief makes, written
                    at brief-generation time.
  * resolutions  — one row per prediction once the outcome is known,
                    written by the end-of-day resolver.

Joining the two by ``prediction_id`` gives:
  * Per-bucket calibration error (feeds tomorrow's confidence_notes).
  * The Yesterday Audit section of every subsequent brief.
  * A continuously-growing track record that compounds into the moat.

See ``docs/outcome_logging_schema.md`` for the full contract.

Design notes:

  * Append-only. We NEVER overwrite a partition. Corrections of an
    earlier resolution use ``correction_of_resolution_id`` and add a
    new row; old rows stay for audit.
  * Schema evolution is additive-only. Adding a new field is fine; if
    the field is missing in an old partition, the reader fills NaN.
    Removing a field requires a schema version bump.
  * Parquet for v1. Postgres deferred to v2 once paying customers > 10.
  * Storage layout::

        <root>/
          predictions/
            trading_date_ist=2026-06-03/predictions.parquet
          resolutions/
            trading_date_ist=2026-06-03/resolutions.parquet

  * The brief generator passes a writer instance; the writer batches
    in-memory and flushes on ``commit()``. This keeps brief generation
    O(in-memory) and writes a single file per partition per session.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Records — mirror the contract in docs/outcome_logging_schema.md
# ---------------------------------------------------------------------------

@dataclass
class PredictionRecord:
    """Written at brief-generation time. One per call the brief makes."""
    prediction_id: str
    brief_id: str
    generated_at_utc: str
    trading_date_ist: str
    model_bundle_version: str
    feature_version: str

    prediction_type: str            # proximity | direction | avoidance |
                                    # regime | options_strike
    instrument_kind: str            # equity | index | option_strike
    symbol: str

    target_description: Dict[str, Any]
    predicted_value: float
    predicted_value_kind: str       # calibrated_probability | direction_sign
                                    # | boolean | continuous_atr_units
    confidence_bucket: str
    raw_score_diagnostic: float

    regime_tags_at_prediction: Dict[str, Any]
    logged_for_audit: bool = True
    schema_version: str = SCHEMA_VERSION
    # Stream G — retrospective audit flag.
    # ``False`` => prediction was logged in real time by the live brief
    #             generator (the canonical case).
    # ``True``  => prediction was synthesised by ``backfill_outcome_log``
    #             from a historical bundle. The Yesterday Audit must
    #             disclose this to the customer, because a retrospective
    #             calibration window can encode hindsight (the bundle was
    #             fit on data that overlaps the replayed dates) and is
    #             therefore weaker evidence than live-collected outcomes.
    is_retrospective: bool = False


@dataclass
class ResolutionRecord:
    """Written when the outcome is known. One per prediction."""
    prediction_id: str
    resolved_at_utc: str
    resolution_method: str          # level_touched |
                                    # level_not_touched_horizon_expired |
                                    # direction_matched | direction_wrong |
                                    # direction_neutral | avoidance_validated |
                                    # avoidance_unnecessary | data_gap

    outcome_boolean: Optional[bool]
    outcome_continuous: Optional[float]

    resolution_details: Dict[str, Any]
    had_data_gap: bool
    resolution_quality: str         # clean | suspect | data_gap
    trading_date_ist: Optional[str] = None
    correction_of_resolution_id: Optional[str] = None
    schema_version: str = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class OutcomeLogWriter:
    """Append-only Parquet writer for predictions + resolutions.

    Buffered in memory; flushes once at ``commit()`` per (table, IST date)
    partition. Idempotent on the partition file: re-running commit
    after a successful commit on the same buffer is a no-op (the
    in-memory buffers are cleared on flush).

    Usage::

        log = OutcomeLogWriter(root="/data/outcome_log")
        log.write_prediction(record)
        log.write_prediction(another_record)
        log.commit()                          # flush both to parquet

    The writer NEVER overwrites a partition. If a partition file
    already exists for the same IST date, new rows are APPENDED to it.
    Schema evolution: missing-from-old / missing-from-new columns get
    filled with pandas NA on append.
    """

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "predictions").mkdir(parents=True, exist_ok=True)
        (self.root / "resolutions").mkdir(parents=True, exist_ok=True)
        self._pred_buffer: List[Dict[str, Any]] = []
        self._reso_buffer: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def write_prediction(self, record: PredictionRecord) -> None:
        """Buffer a prediction row. Flush via :meth:`commit`."""
        d = asdict(record)
        d["target_description_json"] = json.dumps(d.pop("target_description"))
        d["regime_tags_json"] = json.dumps(d.pop("regime_tags_at_prediction"))
        self._pred_buffer.append(d)

    def write_resolution(self, record: ResolutionRecord) -> None:
        """Buffer a resolution row. Flush via :meth:`commit`."""
        d = asdict(record)
        d["resolution_details_json"] = json.dumps(d.pop("resolution_details"))
        self._reso_buffer.append(d)

    # ------------------------------------------------------------------
    # Commit / read
    # ------------------------------------------------------------------

    def commit(self) -> Dict[str, int]:
        """Flush buffered rows to Parquet, partitioned by trading_date_ist.

        Returns a dict of {table_name: rows_written}.
        """
        n_pred = self._flush_table("predictions",
                                    self._pred_buffer,
                                    date_field="trading_date_ist")
        # For resolutions, we partition by the prediction's trading date
        # too. Resolutions are coupled to the brief day, not the resolve
        # day, so trailing-30-day audit windows roll cleanly.
        n_reso = self._flush_resolutions()
        self._pred_buffer = []
        self._reso_buffer = []
        return {"predictions": n_pred, "resolutions": n_reso}

    def _flush_table(self, table: str, rows: List[Dict[str, Any]],
                     date_field: str) -> int:
        if not rows:
            return 0
        frame = pd.DataFrame(rows)
        total = 0
        for date_value, group in frame.groupby(date_field):
            partition_dir = self.root / table / f"trading_date_ist={date_value}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            path = partition_dir / f"{table}.parquet"
            if path.exists():
                existing = pd.read_parquet(path)
                merged = pd.concat([existing, group], ignore_index=True,
                                     sort=False)
            else:
                merged = group
            merged.to_parquet(path, index=False)
            total += len(group)
        return total

    def _flush_resolutions(self) -> int:
        """Resolutions partition by the prediction's trading date, which
        means we have to join to predictions to know where to write. v1
        keeps it simple: we store the prediction's IST date inside the
        resolution row (the brief generator passes it through) so the
        join is just a column read. Falls back to today's UTC date if
        the field is missing — defensive but should never happen in
        normal operation."""
        if not self._reso_buffer:
            return 0
        frame = pd.DataFrame(self._reso_buffer)
        if "trading_date_ist" not in frame.columns:
            today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            frame["trading_date_ist"] = today_utc
        elif frame["trading_date_ist"].isna().any():
            today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            frame["trading_date_ist"] = frame["trading_date_ist"].fillna(today_utc)
        total = 0
        for date_value, group in frame.groupby("trading_date_ist"):
            partition_dir = self.root / "resolutions" / f"trading_date_ist={date_value}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            path = partition_dir / "resolutions.parquet"
            if path.exists():
                existing = pd.read_parquet(path)
                merged = pd.concat([existing, group], ignore_index=True,
                                     sort=False)
            else:
                merged = group
            merged.to_parquet(path, index=False)
            total += len(group)
        return total

    # ------------------------------------------------------------------
    # Read API — for the Yesterday Audit + calibration analysis
    # ------------------------------------------------------------------

    def read_predictions(self, trading_date_ist: str) -> pd.DataFrame:
        """Load all predictions for a specific IST trading date."""
        path = (self.root / "predictions"
                / f"trading_date_ist={trading_date_ist}"
                / "predictions.parquet")
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def read_resolutions(self, trading_date_ist: str) -> pd.DataFrame:
        """Load all resolutions for a specific IST trading date."""
        path = (self.root / "resolutions"
                / f"trading_date_ist={trading_date_ist}"
                / "resolutions.parquet")
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def read_joined(self, trading_date_ist: str) -> pd.DataFrame:
        """Join predictions + resolutions for one IST date.

        Used by the brief's Yesterday Audit section to compute per-bucket
        hit rate, calibration error, and notable misses.
        """
        preds = self.read_predictions(trading_date_ist)
        if preds.empty:
            return preds
        resos = self.read_resolutions(trading_date_ist)
        if resos.empty:
            preds["resolved"] = False
            return preds
        joined = preds.merge(resos, on="prediction_id", how="left",
                              suffixes=("", "_resolution"))
        joined["resolved"] = joined.get("resolved_at_utc").notna()
        return joined


# ---------------------------------------------------------------------------
# Audit helpers — used by the brief generator's Yesterday Audit
# ---------------------------------------------------------------------------

def calibration_by_bucket(joined: pd.DataFrame) -> pd.DataFrame:
    """Per (prediction_type, confidence_bucket): n, hit_rate, mean_p,
    calibration_error.

    Used to populate the ``hit_rate_by_confidence_bucket`` block of
    the Yesterday Audit and to compute the ``drifting_today`` list
    in the next morning's confidence_notes.
    """
    if joined.empty or "outcome_boolean" not in joined.columns:
        return pd.DataFrame()
    sub = joined[joined["outcome_boolean"].notna()].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["hit"] = sub["outcome_boolean"].astype(int)
    rows = []
    grouped = sub.groupby(["prediction_type", "confidence_bucket"])
    for (ptype, bucket), g in grouped:
        n = int(len(g))
        hit_rate = float(g["hit"].mean())
        mean_p = float(g["predicted_value"].mean())
        rows.append({
            "prediction_type": ptype,
            "confidence_bucket": bucket,
            "n": n,
            "hit_rate": hit_rate,
            "mean_predicted_p": mean_p,
            "calibration_error": mean_p - hit_rate,
        })
    return pd.DataFrame(rows)


def make_prediction_id(brief_id: str, symbol: str,
                       prediction_type: str,
                       detail_token: str) -> str:
    """Deterministic ID factory: BRIEF_<date>_<symbol>_<type>_<detail>.

    ``detail_token`` is alpha-specific: for proximity it's the level +
    horizon; for direction it's the horizon; for avoidance it's the
    regime tag set hash. Keeping the format stable lets us join
    predictions to resolutions purely on this ID.
    """
    def _sanitise(s: str) -> str:
        for bad in (" ", "-", "&", ":", "/", "\\", ".", ","):
            s = s.replace(bad, "_")
        return s
    return (f"{brief_id}_{_sanitise(symbol)}_"
            f"{prediction_type}_{_sanitise(detail_token)}")
