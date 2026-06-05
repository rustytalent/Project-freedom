"""HTTP-layer smoke tests for the analytics feed (FastAPI adapter)."""
from __future__ import annotations

import importlib
import os

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from liqpool.scoring import PUBLIC_KEYS


def _client(**env):
    for k, v in env.items():
        os.environ[k] = v
    import service.app as app_mod
    importlib.reload(app_mod)
    return TestClient(app_mod.app)


def test_requires_api_key():
    client = _client(GFEED_API_KEYS="k1:custA", GFEED_RATE_PER_MIN="100")
    r = client.get("/v1/levels", params={"symbol": "X.NS", "date": "2026-05-29"})
    assert r.status_code == 401


def test_valid_key_returns_opaque_feed():
    client = _client(GFEED_API_KEYS="k1:custA", GFEED_RATE_PER_MIN="100")
    r = client.get("/v1/levels", params={"symbol": "X.NS", "date": "2026-05-29"},
                   headers={"X-API-Key": "k1"})
    assert r.status_code == 200
    body = r.json()
    assert body["instrument"] == "X.NS"
    assert "not investment advice" in body["compliance_notice"]["notice"].lower()
    for rec in body["observations"]:
        assert set(rec.keys()) == set(PUBLIC_KEYS)
        score = rec["feature_intensity_score"]
        assert isinstance(score, int) and 0 <= score <= 100


def test_demo_key_uses_abstracted_geometry():
    # NB: ``_client`` reloads ``service.app``, which rebinds the
    # module-level ``API_KEYS`` dict that ``require_customer`` resolves
    # via Python module-globals lookup. If we build both clients up
    # front and only THEN make the requests, both clients observe the
    # demo-tier auth registry (set by the second reload) and both
    # responses come out abstracted. We therefore interleave: build,
    # request, build, request.
    params = {"symbol": "X.NS", "date": "2026-05-29"}
    h = {"X-API-Key": "k1"}
    licensed = _client(GFEED_API_KEYS="k1:custA:licensed",
                       GFEED_RATE_PER_MIN="100")
    exact_zone = licensed.get("/v1/levels", params=params,
                              headers=h).json()["observations"][0]["level_zone"]
    demo = _client(GFEED_API_KEYS="k1:custA:demo",
                   GFEED_RATE_PER_MIN="100")
    demo_zone = demo.get("/v1/levels", params=params,
                         headers=h).json()["observations"][0]["level_zone"]
    assert exact_zone != demo_zone


def test_rate_limit():
    client = _client(GFEED_API_KEYS="k1:custA", GFEED_RATE_PER_MIN="2")
    h = {"X-API-Key": "k1"}
    p = {"symbol": "X.NS", "date": "2026-05-29"}
    assert client.get("/v1/levels", params=p, headers=h).status_code == 200
    assert client.get("/v1/levels", params=p, headers=h).status_code == 200
    assert client.get("/v1/levels", params=p, headers=h).status_code == 429


def test_healthz():
    client = _client(GFEED_API_KEYS="k1:custA")
    assert client.get("/healthz").status_code == 200
