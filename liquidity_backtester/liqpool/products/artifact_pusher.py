"""Push generated artifacts to the customer-facing website.

After the engine produces a research artifact (Daily Brief PDF,
Swing Brief PDF, Diagnosis report, calibration CSV, etc.), this
module POSTs it to the website's ``/api/v1/artifacts`` endpoint.
The website stores the file, makes it discoverable on the
subscriber portal's Artifacts page, and serves tier-gated downloads.

Authentication: a shared bearer token (``ENGINE_INGEST_TOKEN``) is
configured both in the engine's environment and in the website's
Vercel env vars. The token gates uploads only — it never appears in
any rendered customer content.

The endpoint accepts JSON with base64-encoded content. Files larger
than ~25 MB should use a presigned-URL upload path (not implemented
in v1; today's artifacts are all under 1 MB).

Usage::

    from liqpool.products.artifact_pusher import push_artifact

    push_artifact(
        path=Path("output/daily-brief-2026-06-05.pdf"),
        kind="daily_brief_pdf",
        tier="paid_intraday",
        trading_date_ist="2026-06-05",
        description="Today's brief — equities + indexes.",
    )

The pusher is intentionally fire-and-forget at the engine level. If
the website is unreachable, the engine logs a warning and continues
— the file is still on disk and can be retried by re-invoking
``push_artifact`` later.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional


LOGGER = logging.getLogger(__name__)


# Valid kinds — kept in sync with `website/lib/artifacts.ts`.
VALID_KINDS = frozenset({
    "daily_brief_pdf",
    "daily_brief_email",
    "swing_brief_pdf",
    "diagnosis_report_pdf",
    "calibration_summary_csv",
    "outcome_log_export_csv",
    "yesterday_audit_csv",
    "options_strikes_csv",
    "options_executor_audit_csv",       # Stream D.7: Table A/B/SKIP audit
    "options_executor_calls_csv",       # per-strike executor decisions
    "weekly_research_note_pdf",
})

VALID_TIERS = frozenset({
    "public",
    "free_signup",
    "paid_intraday",
    "paid_multi_product",
    "paid_diagnosis",
})


class ArtifactPushError(RuntimeError):
    """Raised on persistent upload failure (non-network)."""


def _resolve_website_url() -> str:
    raw = os.environ.get("WEBSITE_BASE_URL", "").strip()
    if not raw:
        raise ArtifactPushError(
            "WEBSITE_BASE_URL not set. Configure to "
            "https://<your-vercel-domain> before pushing artifacts.")
    return raw.rstrip("/")


def _resolve_token() -> str:
    raw = os.environ.get("ENGINE_INGEST_TOKEN", "").strip()
    if not raw:
        raise ArtifactPushError(
            "ENGINE_INGEST_TOKEN not set. The website rejects unauthenticated "
            "uploads. Configure the same token here and in the website's "
            "Vercel env vars.")
    return raw


def push_artifact(*,
                  path: Path,
                  kind: str,
                  tier: str,
                  trading_date_ist: Optional[str] = None,
                  description: Optional[str] = None,
                  meta: Optional[Dict[str, Any]] = None,
                  filename: Optional[str] = None,
                  timeout_seconds: float = 30.0,
                  ) -> Dict[str, Any]:
    """Upload ``path`` to the website's artifact endpoint.

    Returns the server's acceptance dict (with the assigned ``id``
    and the sha256 the server computed).
    """
    if kind not in VALID_KINDS:
        raise ValueError(
            f"unknown artifact kind={kind!r}; must be one of "
            f"{sorted(VALID_KINDS)}")
    if tier not in VALID_TIERS:
        raise ValueError(
            f"unknown access tier={tier!r}; must be one of "
            f"{sorted(VALID_TIERS)}")
    # Resolve env BEFORE touching the filesystem: misconfiguration of
    # the website URL / token is the most common operational failure
    # and should fail loud and early, not after a path traversal.
    base = _resolve_website_url()
    token = _resolve_token()
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"not a file: {path}")

    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    body: Dict[str, Any] = {
        "kind": kind,
        "tier": tier,
        "filename": filename or path.name,
        "content_b64": base64.b64encode(raw).decode("ascii"),
        "sha256": sha,
    }
    if trading_date_ist:
        body["trading_date_ist"] = trading_date_ist
    if description:
        body["description"] = description
    if meta:
        body["meta"] = meta

    url = f"{base}/api/v1/artifacts"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "engine-artifact-pusher/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            status = resp.status
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise ArtifactPushError(
            f"website rejected upload: HTTP {exc.code} {body_text}") from exc
    except urllib.error.URLError as exc:
        LOGGER.warning(
            "artifact upload network failure to %s for %s: %s; "
            "leaving file on disk for retry", url, path, exc,
        )
        raise

    if status not in (200, 202):
        raise ArtifactPushError(
            f"unexpected status {status} from website: {payload!r}")
    LOGGER.info(
        "pushed artifact %s (%s, %d bytes, tier=%s) -> id=%s",
        path.name, kind, len(raw), tier, payload.get("id"),
    )
    return payload


def push_daily_brief(*,
                     pdf_path: Path,
                     trading_date_ist: str,
                     ) -> Dict[str, Any]:
    """Convenience wrapper for the canonical Daily Brief PDF case."""
    return push_artifact(
        path=pdf_path,
        kind="daily_brief_pdf",
        tier="paid_intraday",
        trading_date_ist=trading_date_ist,
        description=(
            f"Daily Brief for IST trading session {trading_date_ist}. "
            "Print-friendly PDF mirror of the email + portal render."
        ),
    )


def push_calibration_summary(*,
                              csv_path: Path,
                              ) -> Dict[str, Any]:
    """Push the rolling calibration summary CSV."""
    return push_artifact(
        path=csv_path,
        kind="calibration_summary_csv",
        tier="free_signup",
        description=(
            "Per-bucket calibration error for the last 90 trading days."
        ),
    )
