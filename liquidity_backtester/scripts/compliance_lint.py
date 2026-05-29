#!/usr/bin/env python3
"""Compliance linter — pre-launch regulatory red-team for user-facing language.

Goal: keep the product positioned as *non-recommendatory market-structure
analytics*, not stock tips / trading signals / investment advice. It scans for
banned and amber-zone language.

Two scopes, by design:

  * ENFORCED — the customer-facing product surface (docs, the HTTP service, the
    feed-scoring modules, READMEs). A hard-banned hit here exits non-zero so it
    can gate deployment.
  * REPORT-ONLY — everything else (the in-browser research UI, the model/engine
    code, analysis scripts, tests, generated research reports). These
    legitimately contain words like "buy"/"entry" as internal mechanics, so they
    are inventoried for the audit but never fail the build.

Exemptions:
  * Any line containing the marker ``COMPLIANCE_ALLOWED_INTERNAL``.
  * The non-recommendatory disclaimer text (negated banned words).
  * The compliance docs' own word-lists, this linter, and the language test.

Usage:
  python scripts/compliance_lint.py                 # lint enforced scope, exit!=0 on hard hits
  python scripts/compliance_lint.py --report docs/compliance_audit_report.md   # full audit
  python scripts/compliance_lint.py --strict        # also fail on amber in enforced scope
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# --- word lists ------------------------------------------------------------
HARD_BANNED = [
    "buy", "sell", "hold", "recommendation", "recommended", "advise", "advice",
    "stock call", "stock tip", "tip", "trading signal", "entry zone",
    "exit zone", "stoploss", "stop loss", "price target", "take profit",
    "go long", "go short", "long this", "short this", "bullish call",
    "bearish call", "trade this", "tradeable", "best trade", "best stock",
    "best pick", "top pick", "multibagger", "jackpot", "sure shot", "guaranteed",
    "profit guaranteed", "high conviction", "intraday call", "option call",
    "option tip", "model portfolio", "portfolio recommendation",
    "allocation recommendation", "execute trade",
]
AMBER = [
    "bullish", "bearish", "upside", "downside", "directional bias", "long bias",
    "short bias", "expected move", "likely up", "likely down", "reversal likely",
    "breakout likely", "breakdown likely", "opportunity", "setup", "confluence",
    "conviction", "avoid buying", "avoid selling", "avoid shorting", "winner",
]

REPLACEMENTS = {
    "signal": "analytics event / statistical observation",
    "trade setup": "market-structure scenario",
    "setup": "market-structure scenario",
    "tradeable": "scenario meeting statistical filters",
    "buy": "(remove — non-recommendatory)",
    "sell": "(remove — non-recommendatory)",
    "hold": "(remove — non-recommendatory)",
    "entry zone": "reference level / liquidity zone (do not instruct to act)",
    "exit zone": "reference level / liquidity zone (do not instruct to act)",
    "stoploss": "reference level (do not instruct to act)",
    "stop loss": "reference level (do not instruct to act)",
    "price target": "observed level / reaction zone (do not instruct to act)",
    "target": "reference level / reaction zone",
    "best stock": "highest data-quality observations",
    "best trade": "highest sample-size scenarios",
    "top pick": "highest data-quality observations",
    "bullish": "directional context metric",
    "bearish": "directional context metric",
    "upside": "forward-return distribution estimate",
    "downside": "forward-return distribution estimate",
    "recommendation": "non-recommendatory analytics output",
}

# --- scope -----------------------------------------------------------------
ENFORCED_GLOBS = [
    "docs/**/*.md", "service/**/*.py", "README.md",
    "liqpool/scoring.py", "liqpool/serving.py", "liqpool/ingest.py",
]
# Files exempt entirely (they legitimately enumerate banned words).
EXEMPT_FILES = {
    "docs/sebi_compliance_notes.md",
    "docs/compliance_audit_report.md",
    "docs/legal/compliance.md",   # legal disclaimer must enumerate what we do NOT do
    "scripts/compliance_lint.py",
    "tests/test_compliance_language.py",
    "liqpool/scoring.py",  # contains FORBIDDEN_PUBLIC_TOKENS + the disclaimer
}
SKIP_DIR_PARTS = {".git", "__pycache__", ".pytest_cache", "data_cache",
                  "node_modules", ".venv", "output", "output_models"}
SCAN_EXTS = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".html",
             ".tsx", ".jsx", ".ts", ".js", ".csv"}

EXEMPT_LINE_SUBSTRINGS = [
    "COMPLIANCE_ALLOWED_INTERNAL",
    "non-recommendatory market analytics",  # the disclaimer signature
]
# Code idioms that are false positives (DOM, internal variables).
SKIP_MATCH_PATTERNS = [
    re.compile(r"\.target\b"),          # e.target, event.target
    re.compile(r"\btarget\s*[:=]"),      # target= attr / target: key in code
    re.compile(r"data-\w+"),
    re.compile(r"take[_]?[Pp]rofit"),    # internal variable name
]


def _compile(words: list[str]) -> list[tuple[str, re.Pattern]]:
    out = []
    for w in words:
        out.append((w, re.compile(r"(?<![\w-])" + re.escape(w) + r"(?![\w-])",
                                  re.IGNORECASE)))
    return out


HARD_RX = _compile(HARD_BANNED)
AMBER_RX = _compile(AMBER)


def _rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def _is_enforced(rel: str) -> bool:
    p = Path(rel)
    return any(p.match(g) for g in ENFORCED_GLOBS)


def _iter_files(root: Path):
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix not in SCAN_EXTS:
            continue
        if any(part in SKIP_DIR_PARTS for part in p.parts):
            continue
        yield p


def _line_exempt(line: str) -> bool:
    return any(s in line for s in EXEMPT_LINE_SUBSTRINGS) or \
        any(rx.search(line) for rx in SKIP_MATCH_PATTERNS)


def scan_file(path: Path, root: Path) -> list[dict]:
    rel = _rel(path, root)
    if rel in EXEMPT_FILES:
        return []
    findings = []
    enforced = _is_enforced(rel)
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    for lineno, line in enumerate(text.splitlines(), 1):
        if _line_exempt(line):
            continue
        for term, rx in HARD_RX:
            if rx.search(line):
                findings.append(dict(file=rel, line=lineno, term=term,
                                     category="hard", enforced=enforced,
                                     suggestion=REPLACEMENTS.get(term, "remove / rephrase"),
                                     text=line.strip()[:160]))
        for term, rx in AMBER_RX:
            if rx.search(line):
                findings.append(dict(file=rel, line=lineno, term=term,
                                     category="amber", enforced=enforced,
                                     suggestion=REPLACEMENTS.get(term, "needs legal review"),
                                     text=line.strip()[:160]))
    return findings


def scan_repo(root: Path) -> list[dict]:
    findings = []
    for f in _iter_files(root):
        findings.extend(scan_file(f, root))
    return findings


def write_report(findings: list[dict], out_path: Path, root: Path) -> None:
    hard = [f for f in findings if f["category"] == "hard"]
    amber = [f for f in findings if f["category"] == "amber"]
    enf_hard = [f for f in hard if f["enforced"]]
    lines = [
        "# Compliance Audit Report",
        "",
        "Generated by `scripts/compliance_lint.py`. This is a language red-team, "
        "NOT legal advice. See `docs/sebi_compliance_notes.md`.",
        "",
        "## Summary",
        "",
        f"- Hard-banned hits (total): **{len(hard)}**",
        f"- Hard-banned hits in ENFORCED scope (launch blockers): **{len(enf_hard)}**",
        f"- Amber-zone hits (total): **{len(amber)}**",
        "",
        "Most hits outside the enforced scope are internal engine / research-UI "
        "code that legitimately models execution mechanics (e.g. `buy`/`entry` in "
        "the backtester). These are NOT shown to customers and are low priority; "
        "they are inventoried below for completeness.",
        "",
        "## Enforced-scope hard-banned (must fix before launch)",
        "",
    ]
    if enf_hard:
        lines += ["| file | line | term | suggestion |", "|---|---|---|---|"]
        for f in enf_hard:
            lines.append(f"| {f['file']} | {f['line']} | `{f['term']}` | {f['suggestion']} |")
    else:
        lines.append("_None._")
    lines += ["", "## All findings (inventory)", "",
              "| file | line | category | term | enforced |",
              "|---|---|---|---|---|"]
    for f in findings:
        lines.append(f"| {f['file']} | {f['line']} | {f['category']} | "
                     f"`{f['term']}` | {'yes' if f['enforced'] else 'no'} |")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="repo root to scan (default: parent of this script's dir)")
    ap.add_argument("--report", default=None, help="write a markdown audit report here")
    ap.add_argument("--strict", action="store_true",
                    help="also fail on amber-zone hits in enforced scope")
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    findings = scan_repo(root)

    if args.report:
        write_report(findings, Path(args.report), root)
        print(f"[compliance] wrote audit report: {args.report}")

    enforced_fail = [f for f in findings if f["enforced"]
                     and (f["category"] == "hard" or (args.strict and f["category"] == "amber"))]
    for f in enforced_fail:
        print(f"FAIL {f['file']}:{f['line']} [{f['category']}] '{f['term']}' -> {f['suggestion']}")

    total_hard = sum(1 for f in findings if f["category"] == "hard")
    total_amber = sum(1 for f in findings if f["category"] == "amber")
    print(f"[compliance] hard={total_hard} amber={total_amber} "
          f"enforced_failures={len(enforced_fail)}")
    return 1 if enforced_fail else 0


if __name__ == "__main__":
    sys.exit(main())
