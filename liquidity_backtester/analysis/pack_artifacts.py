"""Pack a training-run output directory into 3-4 consolidated files.

A normal multi_asset_run emits ~20 files (multi_asset_summary.json,
research_summary.json, live_plan.json, per-phase CSVs, leakage_audit,
policy_return tables, reaction_alerts, daily_brief.json/.txt, etc).

That's too many for an individual to ship to a chat tool or a customer
in one go. This script reads the standard output directory and emits
three rich consolidated JSON files plus passes the daily-brief
artifacts through unchanged.

Output files:

  * ``consolidated_summary.json`` — top-level metrics + verdict + brief
    metadata. The file you read FIRST.
  * ``consolidated_models.json`` — per-model OOS metrics, calibration,
    feature importance, audit tables.
  * ``consolidated_backtest.json`` — execution v1/v2, arsenal, policy_R,
    cost wall, reaction alerts.
  * (passes through) ``daily_brief.json`` + ``daily_brief.txt`` if
    present — these are the customer-facing product and stay separate.

Source files NOT modified. The script is additive: re-running is safe.

Usage::

    PYTHONPATH=. python analysis/pack_artifacts.py \\
        --input output_core25_head_alpha_710362b/ \\
        --output packed/core25_head_alpha_710362b/

If --output is omitted, the consolidated files are written into the
input directory itself (``packed_*.json``).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from liqpool.products.outcome_log import (
    OutcomeLogWriter,
    calibration_by_bucket,
)
from liqpool.scoring import normalize_customer_tier


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source -> bundle mapping
# ---------------------------------------------------------------------------

# Files routed to consolidated_summary.json
SUMMARY_FILES = {
    "multi_asset_summary":      "multi_asset_summary.json",
    "research_summary":         "research_summary.json",
    "live_plan":                "live_plan.json",
    "sector_data":              "sector_data.json",
    "leakage_audit":            "leakage_audit.json",
}

# Files routed to consolidated_models.json
MODEL_FILES = {
    "phase3_oos_prediction_audit":    "phase3_oos_prediction_audit.csv",
    "phase3_sector_calibration":      "phase3_sector_calibration.csv",
    "reaction_model_report":          "reaction_model_report.csv",
    "reaction_model_calibration":     "reaction_model_calibration.csv",
    "reaction_feature_importance":    "reaction_feature_importance.csv",
    "policy_return_model_report":     "policy_return_model_report.csv",
    "policy_return_model_calibration":"policy_return_model_calibration.csv",
    "policy_return_model_feature_importance":
        "policy_return_model_feature_importance.csv",
    "policy_return_model_oos_predictions":
        "policy_return_model_oos_predictions.csv",
    "policy_label_summary":           "policy_label_summary.csv",
    "leakage_issues":                 "leakage_issues.csv",
}

# Files routed to consolidated_backtest.json
BACKTEST_FILES = {
    "execution_backtest_summary":     "execution_backtest_summary.csv",
    "execution_backtest_by_direction":"execution_backtest_by_direction.csv",
    "execution_backtest_v2_summary":  "execution_backtest_v2_summary.csv",
    "execution_backtest_v2_vs_v1_delta":
        "execution_backtest_v2_vs_v1_delta.csv",
    "reaction_alerts":                "reaction_alerts.csv",
    "track_a_pretouch_setups":        "track_a_pretouch_setups.csv",
}

# Per-row caps so a single CSV doesn't bloat the bundle. Top-200 by
# default for trade-row files (which can have millions of rows).
MAX_ROWS_PER_FILE = {
    "execution_backtest_trades":            500,
    "execution_backtest_v2_trades":         500,
    "reaction_alerts":                      300,
    "policy_return_model_oos_predictions":  300,
    "leakage_issues":                       200,
}

# Files copied through (not consolidated).
PASS_THROUGH = ("daily_brief.json", "daily_brief.txt")
DELIVERY_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Loaders — graceful, return None on missing / unreadable
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        LOGGER.warning("failed to load %s: %s", path, exc)
        return None


def _load_csv(path: Path, key: str) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        cap = MAX_ROWS_PER_FILE.get(key)
        if cap is not None and len(df) > cap:
            df = df.head(cap)
        return df.to_dict(orient="records")
    except Exception as exc:
        LOGGER.warning("failed to load %s: %s", path, exc)
        return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_manifest(path: Path) -> Dict[str, Any]:
    return {
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _hash_unit(text: str) -> float:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(16 ** 16 - 1)


def _abstract_price(value: Any, *, customer_id: str, day: str,
                    salt: str) -> Any:
    """Deterministically coarsen an exact price for demo/public packs."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value
    m = max(abs(v), 1.0)
    step = max(1.0 if m >= 100.0 else 0.25, 0.005 * m)
    u1 = _hash_unit(f"price|{customer_id}|{day}|{salt}|jitter")
    u2 = _hash_unit(f"price|{customer_id}|{day}|{salt}|canary")
    jitter = (u1 - 0.5) * 0.90 * step
    canary = (u2 - 0.5) * 0.36 * step
    out = round((v + jitter) / step) * step + canary
    if abs(out - v) < 0.10 * step:
        out += 0.25 * step
    return round(out, 4)


def _brief_trading_date(brief: Dict[str, Any]) -> str:
    md = brief.get("brief_metadata") or {}
    return str(md.get("trading_date_ist") or "unknown")


def _audit_date_from_brief(brief: Dict[str, Any]) -> Optional[str]:
    audit = brief.get("yesterday_audit")
    if not isinstance(audit, dict):
        return None
    brief_id = str(audit.get("yesterday_brief_id") or "")
    # BRIEF_2026_06_03 -> 2026-06-03
    parts = brief_id.split("_")
    if len(parts) >= 4 and parts[0] == "BRIEF":
        y, m, d = parts[1], parts[2], parts[3]
        if len(y) == 4 and len(m) == 2 and len(d) == 2:
            return f"{y}-{m}-{d}"
    return None


def _abstract_brief(brief: Dict[str, Any], *, customer_id: str) -> Dict[str, Any]:
    """Return a demo/public-safe brief dict with exact levels coarsened."""
    out = copy.deepcopy(brief)
    day = _brief_trading_date(out)
    for i, row in enumerate(out.get("top_watchlist") or []):
        if isinstance(row, dict) and "key_level" in row:
            row["key_level"] = _abstract_price(
                row["key_level"],
                customer_id=customer_id,
                day=day,
                salt=f"watchlist|{i}|{row.get('symbol', '')}",
            )
    for i, row in enumerate(out.get("key_zones") or []):
        if isinstance(row, dict) and "level" in row:
            row["level"] = _abstract_price(
                row["level"],
                customer_id=customer_id,
                day=day,
                salt=f"key_zone|{i}|{row.get('symbol', '')}",
            )
    options = out.get("options_suitability") or {}
    if isinstance(options, dict):
        for index_name, block in options.items():
            if not isinstance(block, dict):
                continue
            for i, strike in enumerate(block.get("strike_levels_in_play") or []):
                if isinstance(strike, dict) and "strike" in strike:
                    strike["strike"] = _abstract_price(
                        strike["strike"],
                        customer_id=customer_id,
                        day=day,
                        salt=f"strike|{index_name}|{i}",
                    )
    return out


def _brief_document_from_dict(data: Dict[str, Any]):
    from liqpool.products.daily_brief import (
        AvoidEntry,
        BriefDocument,
        BriefMetadata,
        ConfidenceNotes,
        KeyZone,
        SectorRegime,
        WatchlistEntry,
    )

    return BriefDocument(
        schema_version=str(data.get("schema_version", "1.0")),
        brief_metadata=BriefMetadata(**(data.get("brief_metadata") or {})),
        index_regime=data.get("index_regime") or {},
        sector_regime=SectorRegime(**(data.get("sector_regime") or {})),
        top_watchlist=[
            WatchlistEntry(**row) for row in (data.get("top_watchlist") or [])
        ],
        options_suitability=data.get("options_suitability") or {},
        avoid_list=[
            AvoidEntry(**row) for row in (data.get("avoid_list") or [])
        ],
        key_zones=[
            KeyZone(**row) for row in (data.get("key_zones") or [])
        ],
        confidence_notes=ConfidenceNotes(**(data.get("confidence_notes") or {})),
        yesterday_audit=data.get("yesterday_audit") or {},
    )


def _render_brief_text(brief: Dict[str, Any]) -> str:
    from liqpool.products.brief_renderer import render_email

    return render_email(_brief_document_from_dict(brief))


def _write_brief_artifacts(input_dir: Path, output_dir: Path, *,
                           customer_id: str,
                           customer_tier: str) -> Dict[str, Any]:
    tier = normalize_customer_tier(customer_tier)
    brief_path = input_dir / "daily_brief.json"
    text_path = input_dir / "daily_brief.txt"
    status: Dict[str, Any] = {
        "status": "missing",
        "tier_policy": "exact" if tier == "licensed" else "abstracted",
    }
    if not brief_path.exists():
        return status

    brief = _load_json(brief_path)
    if not isinstance(brief, dict):
        status["status"] = "unreadable"
        return status

    if tier == "licensed":
        shutil.copy2(brief_path, output_dir / "daily_brief.json")
        if text_path.exists():
            shutil.copy2(text_path, output_dir / "daily_brief.txt")
        else:
            (output_dir / "daily_brief.txt").write_text(_render_brief_text(brief))
        status["status"] = "exact"
        return status

    safe_brief = _abstract_brief(brief, customer_id=customer_id)
    (output_dir / "daily_brief.json").write_text(
        json.dumps(safe_brief, default=str, indent=2)
    )
    try:
        text = _render_brief_text(safe_brief)
    except Exception as exc:
        LOGGER.warning("failed to render abstracted brief text: %s", exc)
        text = (
            "Daily Research Brief\n\n"
            "Demo/public package: human text was withheld because the "
            "renderer could not rebuild the abstracted brief safely. "
            "Use daily_brief.json and README.txt for this sample."
        )
    (output_dir / "daily_brief.txt").write_text(text)
    status["status"] = "abstracted"
    return status


def _outcome_log_summary(outcome_log_root: Optional[Path],
                         audit_date: Optional[str]) -> Dict[str, Any]:
    if outcome_log_root is None:
        return {"status": "not_provided"}
    root = Path(outcome_log_root)
    if not root.exists():
        return {"status": "missing", "root": str(root)}
    if not audit_date:
        return {"status": "no_audit_date", "root": str(root)}

    writer = OutcomeLogWriter(root=str(root))
    joined = writer.read_joined(audit_date)
    if joined.empty:
        return {"status": "empty", "root": str(root), "audit_date": audit_date}

    resolved = joined.get("resolved")
    resolved_count = int(resolved.fillna(False).sum()) if resolved is not None else 0
    data_gap_count = 0
    if "had_data_gap" in joined.columns:
        data_gap_count = int(joined["had_data_gap"].fillna(False).sum())
    retro_share = 0.0
    if "is_retrospective" in joined.columns:
        retro_share = float(joined["is_retrospective"].fillna(False).mean())
    cal = calibration_by_bucket(joined)
    return {
        "status": "ok",
        "root": str(root),
        "audit_date": audit_date,
        "predictions": int(len(joined)),
        "resolved": resolved_count,
        "data_gaps": data_gap_count,
        "retrospective_share": retro_share,
        "is_retrospective_calibration": retro_share > 0.5,
        "calibration_by_bucket": (
            [] if cal.empty else cal.to_dict(orient="records")
        ),
    }


def _write_readme(output_dir: Path, *, customer_id: str, customer_tier: str,
                  brief_status: Dict[str, Any],
                  outcome_summary: Dict[str, Any]) -> Path:
    tier = normalize_customer_tier(customer_tier)
    zone_policy = "exact" if tier == "licensed" else "abstracted"
    retro = outcome_summary.get("retrospective_share")
    retro_line = (
        f"Outcome-log calibration is {retro:.0%} retrospective replay."
        if isinstance(retro, float)
        else "Outcome-log calibration was not included in this pack."
    )
    body = f"""Daily Research Brief Delivery Pack

Customer id: {customer_id}
Customer tier: {customer_tier}
Reference-zone policy: {zone_policy}

Files:
- daily_brief.txt: human-readable research brief.
- daily_brief.json: machine-readable brief contract.
- consolidated_summary.json / consolidated_models.json / consolidated_backtest.json: internal audit summaries for review.
- outcome_log_summary.json: compact audit/log status for the brief window.
- manifest.json: hashes, source paths, and delivery metadata.

Brief status: {brief_status.get("status")}
{retro_line}

Compliance:
This is non-recommendatory market-structure research context. It is not
investment advice, not a recommendation, and not an instruction to trade.
"""
    path = output_dir / "README.txt"
    path.write_text(body)
    return path


def pack_customer_delivery(input_dir: Path,
                           output_dir: Path,
                           *,
                           outcome_log_root: Optional[Path] = None,
                           customer_id: str = "pilot",
                           customer_tier: str = "licensed",
                           audit_date: Optional[str] = None,
                           include_consolidated: bool = True,
                           make_zip: bool = False) -> Dict[str, Path]:
    """Create a customer-ready delivery pack for a Daily Brief run.

    The pack keeps the old consolidated JSON summaries, adds a manifest,
    README, compact outcome-log summary, and applies tier-aware brief
    redaction for demo/public/sample customers.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    if include_consolidated:
        written.update(pack(input_dir, output_dir))

    raw_brief = _load_json(input_dir / "daily_brief.json")
    if isinstance(raw_brief, dict) and audit_date is None:
        audit_date = _audit_date_from_brief(raw_brief)
    brief_status = _write_brief_artifacts(
        input_dir,
        output_dir,
        customer_id=customer_id,
        customer_tier=customer_tier,
    )
    for name in PASS_THROUGH:
        path = output_dir / name
        if path.exists():
            written[name] = path

    outcome_summary = _outcome_log_summary(outcome_log_root, audit_date)
    outcome_path = output_dir / "outcome_log_summary.json"
    outcome_path.write_text(json.dumps(outcome_summary, default=str, indent=2))
    written["outcome_log_summary"] = outcome_path

    readme = _write_readme(
        output_dir,
        customer_id=customer_id,
        customer_tier=customer_tier,
        brief_status=brief_status,
        outcome_summary=outcome_summary,
    )
    written["README"] = readme

    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_version": DELIVERY_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "source_input_dir": str(input_dir),
        "outcome_log_root": str(outcome_log_root) if outcome_log_root else None,
        "customer_id": customer_id,
        "customer_tier": customer_tier,
        "normalized_customer_tier": normalize_customer_tier(customer_tier),
        "reference_zone_policy": (
            "exact"
            if normalize_customer_tier(customer_tier) == "licensed"
            else "abstracted"
        ),
        "audit_date": audit_date,
        "brief_status": brief_status,
        "outcome_log_summary": outcome_summary,
        "files": {},
    }
    manifest_path.write_text(json.dumps(manifest, default=str, indent=2))
    written["manifest"] = manifest_path

    # Fill hashes after all payload files exist. The manifest intentionally
    # excludes itself because rewriting it would invalidate its own hash.
    manifest["files"] = {
        path.name: _file_manifest(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    manifest_path.write_text(json.dumps(manifest, default=str, indent=2))

    if make_zip:
        archive_base = str(output_dir)
        zip_path = Path(shutil.make_archive(archive_base, "zip", root_dir=output_dir))
        written["zip"] = zip_path
    return written


def _build_bundle(input_dir: Path,
                  file_map: Dict[str, str]) -> Dict[str, Any]:
    bundle: Dict[str, Any] = {}
    for key, filename in file_map.items():
        path = input_dir / filename
        if filename.endswith(".json"):
            bundle[key] = _load_json(path)
        elif filename.endswith(".csv"):
            bundle[key] = _load_csv(path, key)
        else:
            bundle[key] = None
    return bundle


# ---------------------------------------------------------------------------
# Pack driver
# ---------------------------------------------------------------------------

def pack(input_dir: Path, output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Pack the given training-output directory into 3 consolidated
    JSON files + pass through the daily-brief artifacts.

    Returns a dict of {bundle_name: output_path} for all files written.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    if output_dir is None:
        output_dir = input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}

    for bundle_name, file_map in (
        ("consolidated_summary", SUMMARY_FILES),
        ("consolidated_models", MODEL_FILES),
        ("consolidated_backtest", BACKTEST_FILES),
    ):
        bundle = _build_bundle(input_dir, file_map)
        out_path = output_dir / f"{bundle_name}.json"
        with out_path.open("w") as f:
            json.dump(bundle, f, default=str, indent=2)
        written[bundle_name] = out_path
        n_keys = sum(1 for v in bundle.values() if v is not None)
        total_keys = len(file_map)
        LOGGER.info("wrote %s — %d/%d source files present",
                    out_path.name, n_keys, total_keys)

    # Pass-through copies for the daily-brief artifacts.
    for name in PASS_THROUGH:
        src = input_dir / name
        if src.exists() and src.resolve() != (output_dir / name).resolve():
            shutil.copy2(src, output_dir / name)
            written[name] = output_dir / name
            LOGGER.info("passed through %s", name)

    return written


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pack a training-output directory into 3 consolidated files.")
    p.add_argument("--input", type=Path, required=True,
                   help="Training output directory (contains "
                         "multi_asset_summary.json etc.)")
    p.add_argument("--output", type=Path, default=None,
                   help="Where to write the consolidated files. "
                         "Default: same as --input.")
    p.add_argument("--customer-delivery", action="store_true",
                   help="Create a pilot/customer delivery pack with README, "
                        "manifest, tier-aware brief artifacts, and optional zip.")
    p.add_argument("--outcome-log-root", type=Path, default=None,
                   help="Outcome-log root to summarize in customer-delivery mode.")
    p.add_argument("--customer-id", default="pilot",
                   help="Customer id for delivery metadata and demo watermarking.")
    p.add_argument("--customer-tier", default="licensed",
                   help="licensed/paid/premium keep exact zones; demo/public/free "
                        "use abstracted zones.")
    p.add_argument("--audit-date", default=None,
                   help="IST date to summarize from the outcome log. Default: "
                        "derive from daily_brief.yesterday_audit when possible.")
    p.add_argument("--zip", action="store_true",
                   help="Also write <output>.zip in customer-delivery mode.")
    p.add_argument("--no-consolidated", action="store_true",
                   help="In customer-delivery mode, skip consolidated_*.json files.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.customer_delivery:
        if args.output is None:
            raise ValueError("--output is required with --customer-delivery")
        written = pack_customer_delivery(
            args.input,
            args.output,
            outcome_log_root=args.outcome_log_root,
            customer_id=args.customer_id,
            customer_tier=args.customer_tier,
            audit_date=args.audit_date,
            include_consolidated=not args.no_consolidated,
            make_zip=args.zip,
        )
    else:
        written = pack(args.input, args.output)
    print(f"packed {len(written)} files into {args.output or args.input}:")
    for key, path in written.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
