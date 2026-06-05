"""Tests for the engine-side artifact pusher.

Pinned contracts:
  * Validates ``kind`` against the allow-list.
  * Validates ``tier`` against the allow-list.
  * Fails closed when ``WEBSITE_BASE_URL`` or ``ENGINE_INGEST_TOKEN``
    are not configured.
  * Builds a well-formed JSON body with base64-encoded content and
    a server-side-checkable sha256.
  * The convenience wrappers (Daily Brief, calibration summary) call
    ``push_artifact`` with the correct kind+tier defaults.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import unittest
from pathlib import Path
from unittest import mock

from liqpool.products.artifact_pusher import (
    ArtifactPushError,
    push_artifact,
    push_calibration_summary,
    push_daily_brief,
    VALID_KINDS,
    VALID_TIERS,
)


class _Resp:
    """Minimal stand-in for urllib's HTTPResponse context manager."""

    def __init__(self, status: int, body: bytes):
        self.status = status
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._body


def _ok_response(body: dict) -> _Resp:
    return _Resp(202, json.dumps(body).encode("utf-8"))


def _tmpfile(tmpdir: Path, content: bytes,
             name: str = "artifact.pdf") -> Path:
    p = tmpdir / name
    p.write_bytes(content)
    return p


class KindAndTierValidationTests(unittest.TestCase):

    def test_invalid_kind_raises(self):
        with mock.patch.dict(os.environ, {
            "WEBSITE_BASE_URL": "https://example.test",
            "ENGINE_INGEST_TOKEN": "token",
        }), self.assertRaises(ValueError):
            push_artifact(path=Path("/tmp/x"), kind="bogus",
                          tier="public")

    def test_invalid_tier_raises(self):
        with mock.patch.dict(os.environ, {
            "WEBSITE_BASE_URL": "https://example.test",
            "ENGINE_INGEST_TOKEN": "token",
        }), self.assertRaises(ValueError):
            push_artifact(path=Path("/tmp/x"), kind="daily_brief_pdf",
                          tier="bogus")

    def test_valid_kinds_include_all_documented(self):
        # If website/lib/artifacts.ts changes, this test catches drift.
        expected = {
            "daily_brief_pdf", "daily_brief_email", "swing_brief_pdf",
            "diagnosis_report_pdf", "calibration_summary_csv",
            "outcome_log_export_csv", "yesterday_audit_csv",
            "options_strikes_csv", "weekly_research_note_pdf",
        }
        self.assertEqual(set(VALID_KINDS), expected)

    def test_valid_tiers_include_all_documented(self):
        expected = {
            "public", "free_signup", "paid_intraday",
            "paid_multi_product", "paid_diagnosis",
        }
        self.assertEqual(set(VALID_TIERS), expected)


class EnvironmentTests(unittest.TestCase):

    def test_missing_base_url_fails_closed(self):
        env = {"ENGINE_INGEST_TOKEN": "token"}
        env.pop("WEBSITE_BASE_URL", None)
        with mock.patch.dict(os.environ, env, clear=True), \
             self.assertRaises(ArtifactPushError):
            push_artifact(path=Path("/tmp/x"), kind="daily_brief_pdf",
                          tier="public")

    def test_missing_token_fails_closed(self):
        env = {"WEBSITE_BASE_URL": "https://example.test"}
        env.pop("ENGINE_INGEST_TOKEN", None)
        with mock.patch.dict(os.environ, env, clear=True), \
             self.assertRaises(ArtifactPushError):
            push_artifact(path=Path("/tmp/x"), kind="daily_brief_pdf",
                          tier="public")


class PostBodyShapeTests(unittest.TestCase):

    def test_builds_well_formed_payload(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = _tmpfile(Path(tmp), b"hello world",
                            name="daily-brief-2026-06-05.pdf")
            captured = {}

            def fake_urlopen(req, timeout):
                captured["url"] = req.full_url
                captured["headers"] = dict(req.headers)
                captured["body"] = json.loads(req.data.decode("utf-8"))
                return _ok_response({
                    "accepted": True,
                    "id": "daily_brief_pdf-deadbeef",
                    "bytes": 11,
                    "sha256": hashlib.sha256(b"hello world").hexdigest(),
                    "tier": "paid_intraday",
                })

            env = {"WEBSITE_BASE_URL": "https://example.test/",
                   "ENGINE_INGEST_TOKEN": "tkn-123"}
            with mock.patch.dict(os.environ, env), \
                 mock.patch("urllib.request.urlopen", fake_urlopen):
                resp = push_artifact(
                    path=path,
                    kind="daily_brief_pdf",
                    tier="paid_intraday",
                    trading_date_ist="2026-06-05",
                    description="Today's brief.",
                )

            # URL stripped of trailing slash, joined correctly.
            self.assertEqual(captured["url"],
                             "https://example.test/api/v1/artifacts")
            self.assertEqual(captured["headers"]["Authorization"],
                             "Bearer tkn-123")
            self.assertEqual(captured["headers"]["Content-type"],
                             "application/json")
            body = captured["body"]
            self.assertEqual(body["kind"], "daily_brief_pdf")
            self.assertEqual(body["tier"], "paid_intraday")
            self.assertEqual(body["filename"], "daily-brief-2026-06-05.pdf")
            self.assertEqual(body["trading_date_ist"], "2026-06-05")
            self.assertEqual(body["description"], "Today's brief.")
            self.assertEqual(
                base64.b64decode(body["content_b64"]), b"hello world")
            self.assertEqual(
                body["sha256"],
                hashlib.sha256(b"hello world").hexdigest())
            # Server response surfaced to caller.
            self.assertEqual(resp["id"], "daily_brief_pdf-deadbeef")

    def test_convenience_wrappers_apply_correct_defaults(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            pdf = _tmpfile(Path(tmp), b"pdfbody", name="brief.pdf")
            csv = _tmpfile(Path(tmp), b"a,b\n1,2\n", name="cal.csv")
            captured = []

            def fake_urlopen(req, timeout):
                captured.append(json.loads(req.data.decode("utf-8")))
                return _ok_response(
                    {"accepted": True, "id": "x", "bytes": 1,
                     "sha256": "x", "tier": "x"})

            env = {"WEBSITE_BASE_URL": "https://example.test",
                   "ENGINE_INGEST_TOKEN": "tkn"}
            with mock.patch.dict(os.environ, env), \
                 mock.patch("urllib.request.urlopen", fake_urlopen):
                push_daily_brief(pdf_path=pdf,
                                 trading_date_ist="2026-06-05")
                push_calibration_summary(csv_path=csv)

            self.assertEqual(captured[0]["kind"], "daily_brief_pdf")
            self.assertEqual(captured[0]["tier"], "paid_intraday")
            self.assertEqual(captured[0]["trading_date_ist"], "2026-06-05")
            self.assertEqual(captured[1]["kind"],
                             "calibration_summary_csv")
            self.assertEqual(captured[1]["tier"], "free_signup")


if __name__ == "__main__":
    unittest.main()
