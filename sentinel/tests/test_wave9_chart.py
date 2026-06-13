"""Wave 9 tests — interactive payoff chart + stepper UX.

The scenario / payoff charts on both pages now ship:
  * profit/loss zone fills (green above zero, red below)
  * a hover crosshair with a tooltip showing spot + book P&L
  * elegant breakeven diamonds with labels
  * a numeric stepper (replacing the outdated range slider)

These tests pin the markup contract so a future rewrite can't silently
drop the interactivity.
"""
from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def srv(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DEMO", "1")
    monkeypatch.setenv("SENTINEL_JOURNAL_DIR", str(tmp_path))
    monkeypatch.delenv("SENTINEL_TOKEN", raising=False)
    import sentinel.server as server
    importlib.reload(server)
    return server


# ---------------------------------------------------------------------------
# Cockpit — scenario chart upgrade
# ---------------------------------------------------------------------------

def test_cockpit_replaces_slider_with_stepper(srv):
    """The outdated <input type=range> is gone; a numeric stepper with
    quick-step buttons takes its place."""
    with TestClient(srv.app) as c:
        html = c.get("/").text
    # no range slider on the scenario panel any more
    assert 'type="range"' not in html
    # stepper exists with the quick-step buttons
    assert 'class="stepper"' in html
    for label in ("−100", "−50", "−10", "+10", "+50", "+100"):
        assert label in html
    # numeric input still drives the what-if
    assert 'id="mv"' in html and 'type="number"' in html


def test_cockpit_chart_has_crosshair_and_tooltip(srv):
    """The scenario chart ships a hover crosshair group + tooltip div, and
    the wrap exposes the IDs the JS needs."""
    with TestClient(srv.app) as c:
        html = c.get("/").text
    for marker in ('id="curve_wrap"', 'id="curve"', 'id="curve_tip"',
                   'id="curve_xhair"', 'id="xh_v"', 'id="xh_dot"'):
        assert marker in html
    # zone gradients for profit vs loss
    assert 'id="gp"' in html and 'id="gl"' in html


def test_cockpit_chart_handlers_wire_hover_and_click(srv):
    """mousemove fills the tooltip, mouseleave hides it, click sets mv."""
    with TestClient(srv.app) as c:
        html = c.get("/").text
    for hook in ("mousemove", "mouseleave", "addEventListener"):
        assert hook in html
    # click on the chart should drive setMv (drag-to-pan)
    assert 'svg.addEventListener("click"' in html
    assert "function setMv(" in html


# ---------------------------------------------------------------------------
# Console — auditor payoff chart upgrade
# ---------------------------------------------------------------------------

def test_console_audit_chart_is_interactive(srv):
    """The console's auditor chart now uses the same crosshair UX as the
    cockpit, with green/red zone fills and a hover tooltip."""
    with TestClient(srv.app) as c:
        html = c.get("/console").text
    for marker in ('id="aud_wrap"', 'id="aud_curve"', 'id="aud_tip"',
                   'id="aud_xhair"', 'id="aud_xh_v"', 'id="aud_xh_dot"',
                   "drawAuditCurve("):
        assert marker in html
    # both halves of the curve coloured separately
    assert 'id="agp"' in html and 'id="agl"' in html


def test_console_audit_tooltip_shows_pnl(srv):
    """The hover tooltip surfaces Spot + P&L (the founder's explicit ask)."""
    with TestClient(srv.app) as c:
        html = c.get("/console").text
    assert "Book P&amp;L" in html or "P&amp;L" in html
    # the chart-wrap tip CSS class is the one the design-system defines
    assert 'class="tip"' in html


# ---------------------------------------------------------------------------
# Design system — stepper + tip styles ship
# ---------------------------------------------------------------------------

def test_stylesheet_carries_stepper_and_tip(srv):
    with TestClient(srv.app) as c:
        css = c.get("/sentinel.css").text
    assert ".stepper" in css
    assert ".chart-wrap .tip" in css
    # the tooltip becomes visible via .on (matches the JS toggle)
    assert ".chart-wrap .tip.on" in css
