"""Compliance-language guard.

Fails if hard-banned recommendation/tip/signal language appears in the
ENFORCED (customer-facing) scope. Internal engine code is inventoried by the
linter but not enforced here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LINT_PATH = ROOT / "scripts" / "compliance_lint.py"


def _load_linter():
    spec = importlib.util.spec_from_file_location("compliance_lint", LINT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_hard_banned_in_enforced_scope():
    lint = _load_linter()
    findings = lint.scan_repo(ROOT)
    enforced_hard = [f for f in findings
                     if f["enforced"] and f["category"] == "hard"]
    assert not enforced_hard, (
        "hard-banned language in customer-facing scope:\n"
        + "\n".join(f"  {f['file']}:{f['line']} '{f['term']}'" for f in enforced_hard)
    )


def test_disclaimer_is_exempt_not_flagged():
    lint = _load_linter()
    # The disclaimer negates banned words; it must not trip the linter.
    line = ("This output is non-recommendatory market analytics ... not a "
            "buy/sell/hold recommendation ...")
    assert lint._line_exempt(line) is True


def test_linter_flags_a_real_violation():
    lint = _load_linter()
    hits = [term for term, rx in lint.HARD_RX if rx.search("strong BUY recommendation today")]
    assert "buy" in hits and "recommendation" in hits
