"""Smoke test for the engine → website artifact pipeline.

Run this on the VPS once after configuring ``WEBSITE_BASE_URL`` and
``ENGINE_INGEST_TOKEN``. It uploads a tiny test artifact, prints the
website's acceptance response, and tells you whether the pipeline is
live.

The artifact uses ``tier=public`` so the upload doesn't need a real
subscription to be downloadable. The artifact ``kind`` is
``calibration_summary_csv`` because it's the most innocuous category
to use as a smoke fixture.

Usage::

    # ensure env vars are set:
    export WEBSITE_BASE_URL="https://<your-vercel-domain>"
    export ENGINE_INGEST_TOKEN="<the-shared-secret>"

    PYTHONPATH=. python analysis/smoke_test_artifact_pipeline.py

Expected output: a JSON acceptance dict containing ``accepted=true``,
the artifact id, and the sha256 the server computed. If you see
``HTTP 401`` the token doesn't match between VPS and Vercel. If you
see ``ConnectionError`` the URL is wrong or Vercel isn't reachable.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from liqpool.products.artifact_pusher import (
    ArtifactPushError,
    push_artifact,
)


def main() -> int:
    base = os.environ.get("WEBSITE_BASE_URL", "").strip()
    token = os.environ.get("ENGINE_INGEST_TOKEN", "").strip()
    if not base or not token:
        print("[smoke] FAIL: WEBSITE_BASE_URL or ENGINE_INGEST_TOKEN unset",
              file=sys.stderr)
        print(f"        WEBSITE_BASE_URL={base!r}")
        print(f"        ENGINE_INGEST_TOKEN={'<set>' if token else '<unset>'}")
        return 2

    print(f"[smoke] target website: {base}")
    print(f"[smoke] token configured: yes ({len(token)} chars)")

    # Build a tiny CSV that looks like a real calibration export.
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    body = (
        "trading_date_ist,prediction_type,confidence_bucket,n,hit_rate,"
        "mean_predicted_p,calibration_error,is_retrospective_share\n"
        "2026-06-05,proximity,high,100,0.71,0.72,0.01,0.35\n"
        f"# smoke-test marker {ts}\n"
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / f"smoke-{ts}.csv"
        path.write_text(body)
        try:
            resp = push_artifact(
                path=path,
                kind="calibration_summary_csv",
                tier="public",
                description=(
                    f"Pipeline smoke test — uploaded {ts}. "
                    "Safe to delete after verifying the portal shows it."
                ),
                meta={"smoke_test": True, "uploaded_at": ts},
            )
        except ArtifactPushError as exc:
            print(f"[smoke] FAIL: {exc}", file=sys.stderr)
            return 3

    print("[smoke] OK — website accepted the upload")
    print(json.dumps(resp, indent=2))
    print()
    print(f"[smoke] open {base}/portal/artifacts in your browser to see it")
    print("        (look in the 'Calibration & audit exports' section)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
