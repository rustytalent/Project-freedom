"""Internal report artifacts — the two documents the founder generates
and pastes back so the model can gauge everything.

  ARTIFACT 1  system_report.md   — literally everything: module health,
              ledger stats, curator findings (with sample-size grades),
              calibration result + re-run expectancy, profit-lock state,
              leakage verdicts, scientist hit-rates, and an explicit
              "what's working / what's weak / what to improve" section.

  ARTIFACT 2  connection_map.txt + connection_map.dot — the wiring graph
              built from every module's self-declared IO, so we can see
              how the organism is connected and where the gaps are.

Both run with empty data (weekend, market closed): they then report the
STRUCTURE and flag every organ that has no data yet — which is itself
the most useful first read.

CLI:
  python -m sentinel.reports --journal ~/.sentinel --session 2026-06-13 \
      --out sentinel/reports_out
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Importing these runs their declare() calls -> populates io_decl.REGISTRY.
from . import (  # noqa: F401
    advisor, audit_log, auditor, auth, byok, calibration, crux_signal,
    curator, equity_layer, greeks,
    institutional, journey_audit, kite_client, leakage_guard, live_equity,
    live_models, live_publisher, monte_carlo, moneyness, orchestration,
    portfolio, premium_tracker, profit_lock, psychology, saas, scenario_engine,
    scientists, shadow_ledger, strategy_builder, stress, trails,
)
from .calibration import CalibrationEngine
from .curator import Curator
from .io_decl import REGISTRY, render_dot, render_map
from .leakage_guard import LeakageGuard
from .shadow_ledger import read_session


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# Artifact 2 — connection map
# ---------------------------------------------------------------------------

def connection_map() -> Dict[str, str]:
    return {"text": render_map(), "dot": render_dot()}


# ---------------------------------------------------------------------------
# Artifact 1 — system report
# ---------------------------------------------------------------------------

def _ledger_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_kind: Dict[str, int] = {}
    by_sci: Dict[str, int] = {}
    by_money: Dict[str, int] = {}
    complete = 0
    for r in rows:
        by_kind[r.get("kind", "?")] = by_kind.get(r.get("kind", "?"), 0) + 1
        by_sci[r.get("scientist", "?")] = by_sci.get(r.get("scientist", "?"), 0) + 1
        mk = (r.get("identity") or {}).get("moneyness_key", "?")
        by_money[mk] = by_money.get(mk, 0) + 1
        if (r.get("journey") or {}).get("complete"):
            complete += 1
    return {"total": len(rows), "complete": complete,
            "by_kind": by_kind, "by_scientist": by_sci,
            "by_moneyness": by_money}


def generate_system_report(journal_dir: Path, session: str,
                           profit_lock_snapshot: Optional[Dict] = None
                           ) -> str:
    rows = read_session(journal_dir, session)
    stats = _ledger_stats(rows)

    cur = Curator()
    report = cur.judge_rows(session, rows)

    eng = CalibrationEngine(knobs_path=journal_dir / "knobs.json")
    cal = eng.calibrate(report, rows)

    guard = LeakageGuard(history_path=journal_dir / "leakage_history.json")
    # Use the best scientist hit-rate as the headline metric to leakage-check.
    best_hr = 0.0
    for v in report.per_scientist.values():
        if v.get("hit_rate") is not None:
            best_hr = max(best_hr, v["hit_rate"])
    leak = guard.check("best_scientist_hit_rate", best_hr,
                       cal.after_expectancy > cal.before_expectancy)

    L: List[str] = []
    L.append(f"# Sentinel System Report")
    L.append(f"_generated {_now()} · session {session}_\n")

    # 1. headline
    L.append("## 1. Headline")
    L.append(f"- modules registered: **{len(REGISTRY)}**")
    L.append(f"- ledger events this session: **{stats['total']}** "
             f"({stats['complete']} journeys complete)")
    L.append(f"- scientists active: **{len(scientists.DEFAULT_SCIENTISTS)}** "
             f"({', '.join(s.name for s in scientists.DEFAULT_SCIENTISTS)})")
    if stats["total"] == 0:
        L.append("- ⚠ **no ledger data yet** — this is a STRUCTURE report. "
                 "Run a live/demo session to populate the laboratory.")
    L.append("")

    # 2. ledger composition
    L.append("## 2. Shadow ledger composition")
    L.append(f"- by kind: `{stats['by_kind'] or '—'}`")
    L.append(f"- by scientist: `{stats['by_scientist'] or '—'}`")
    L.append(f"- by moneyness: `{stats['by_moneyness'] or '—'}`")
    L.append("")

    # 3. curator findings
    L.append("## 3. Curator findings (n-weighted; noise<10, tentative<30, firm≥30)")
    if report.findings:
        for f in report.findings:
            mark = {"firm": "✓", "tentative": "~", "noise": "✗"}.get(f.confidence, "?")
            L.append(f"- [{mark} {f.confidence} n={f.n}] **{f.key}** — {f.statement}")
    else:
        L.append("- (no completed journeys yet — no findings)")
    L.append("")
    L.append("### Per-scientist hit-rate")
    if report.per_scientist:
        for name, v in sorted(report.per_scientist.items()):
            hr = "—" if v["hit_rate"] is None else f"{v['hit_rate']:.0%}"
            L.append(f"- {name}: {hr} (n={v['n']}, {v['grade']})")
    else:
        L.append("- (none)")
    L.append("")
    L.append("### Confidence calibration (predicted vs actual)")
    if report.confidence_calibration:
        for c in report.confidence_calibration:
            L.append(f"- predicted {c['confidence_bucket']}: actual "
                     f"{c['actual_win_rate']:.0%} (n={c['n']})")
    else:
        L.append("- (insufficient data)")
    L.append("")

    # 4. calibration
    L.append("## 4. Calibration engine (knobs, re-run tested)")
    L.append(f"- re-run expectancy: **{cal.before_expectancy:.3f} → "
             f"{cal.after_expectancy:.3f}** R")
    L.append(f"- accepted: `{cal.accepted or '—'}`")
    L.append(f"- rejected: `{cal.rejected or '—'}`")
    L.append(f"- current knobs: `{cal.knobs}`")
    for n in cal.notes:
        L.append(f"  - {n}")
    L.append("")

    # 5. leakage
    L.append("## 5. Leakage guard")
    L.append(f"- metric: {leak.metric} = {leak.current} "
             f"(prev {leak.previous}, jump {leak.jump})")
    L.append(f"- verdict: **{leak.verdict}** — {leak.note}")
    L.append("")

    # 6. profit lock
    L.append("## 6. Profit lock")
    if profit_lock_snapshot:
        s = profit_lock_snapshot
        L.append(f"- current ₹{s['current_pnl']:,.0f} · peak ₹{s['peak_pnl']:,.0f} "
                 f"· **locked ₹{s['locked_pnl']:,.0f}** · floating ₹{s['floating_pnl']:,.0f}")
        L.append(f"- worst-case day profit if it all fell: "
                 f"**₹{s['worst_case_day_profit']:,.0f}**")
    else:
        L.append("- not armed this session")
    L.append("")

    # 7. self-assessment
    L.append("## 7. Self-assessment — what's working / weak / to improve")
    working, weak, improve = _self_assess(stats, report, cal, leak)
    L.append("**Working:**")
    for w in working:
        L.append(f"- {w}")
    L.append("\n**Weak / unproven:**")
    for w in weak:
        L.append(f"- {w}")
    L.append("\n**Improve next:**")
    for w in improve:
        L.append(f"- {w}")
    L.append("")

    # 8. institutional surfaces available (the SaaS catalog)
    L.append("## 8. Institutional surfaces available")
    L.append("Named, citation-bearing methodologies in this build:")
    L.append("- `institutional`: SVI smile (Gatheral 2004), fair value, "
             "25-delta skew, vol cone (Burghardt & Lane 1990), Cornish-Fisher "
             "VaR (BIS 1996), Historical VaR, Expected Shortfall (Basel III), "
             "Yang-Zhang RV, Crux Liquidity / Slippage scores, roll curves")
    L.append("- `stress`: 7 canned crisis scenarios "
             f"({', '.join(sorted(stress.SCENARIOS))})")
    L.append("- `auditor`: multi-leg payoff curve + Hull-taxonomy detection "
             "+ uncapped-loss + regime-mismatch flags")
    L.append("- `strategy_builder`: 6 customer intents "
             f"({', '.join(strategy_builder.INTENTS)})")
    L.append("- `equity_layer`: NIFTY top-10 HIDDEN_BULL/HIDDEN_BEAR/"
             "BROAD_BULL/BROAD_BEAR/TRUE_FLAT regime classifier "
             f"(weights effective {equity_layer.NIFTY_TOP10_EFFECTIVE_DATE})")
    L.append("- `saas`: tier gating across "
             f"{len(saas.FEATURE_CATALOG)} features (RETAIL/PRO/QUANT/FOUNDER)")
    L.append("")

    # 9. module health
    L.append("## 9. Module health (self-declared IO)")
    for name in sorted(REGISTRY):
        s = REGISTRY[name]
        L.append(f"- `{name}` [{s.tier}] — {s.purpose}")
    L.append("")
    L.append("_(full wiring graph in connection_map.txt / .dot)_")
    return "\n".join(L)


def _self_assess(stats, report, cal, leak):
    working, weak, improve = [], [], []
    if stats["total"] > 0:
        working.append(f"shadow ledger capturing {stats['total']} events with "
                       f"full context+journey schema")
    else:
        weak.append("no ledger data — laboratory hasn't run a session yet")
        improve.append("run one demo/live session to populate ledger, then "
                       "regenerate this report")
    firm = [f for f in report.findings if f.confidence == "firm"]
    if firm:
        working.append(f"{len(firm)} firm finding(s) — enough data to act on")
    else:
        weak.append("no FIRM findings yet (need ≥30 completed journeys per truth)")
    if cal.accepted:
        working.append(f"calibration accepted {len(cal.accepted)} knob change(s) "
                       f"that passed the re-run test")
    elif cal.before_expectancy or cal.after_expectancy:
        weak.append("calibration proposed changes but re-run rejected them "
                    "(not enough proven edge yet)")
    if leak.verdict == "SUSPICIOUS":
        weak.append("leakage guard flagged a suspicious jump — investigate "
                    "before trusting the scientist metrics")
    if leak.verdict == "EARNED":
        working.append("a fast metric jump was backed by real expectancy gain "
                       "(EARNED, not leaked)")
    improve.append("bridge the research codebase context (trend/vol/levels) "
                   "into MarketSnapshot so scientists think with real context")
    improve.append("accumulate ≥1 week of sessions so per-moneyness behaviour "
                   "and confidence calibration become firm")
    return working, weak, improve


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def write_artifacts(journal_dir: Path, session: str, out_dir: Path,
                    profit_lock_snapshot: Optional[Dict] = None) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = generate_system_report(journal_dir, session, profit_lock_snapshot)
    cmap = connection_map()
    paths = {
        "system_report": out_dir / "system_report.md",
        "connection_map_txt": out_dir / "connection_map.txt",
        "connection_map_dot": out_dir / "connection_map.dot",
    }
    paths["system_report"].write_text(rep)
    paths["connection_map_txt"].write_text(cmap["text"])
    paths["connection_map_dot"].write_text(cmap["dot"])
    return paths


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--journal", type=Path,
                   default=Path.home() / ".sentinel")
    p.add_argument("--session", default=datetime.now().strftime("%Y-%m-%d"))
    p.add_argument("--out", type=Path, default=Path("sentinel/reports_out"))
    args = p.parse_args()
    paths = write_artifacts(args.journal, args.session, args.out)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
