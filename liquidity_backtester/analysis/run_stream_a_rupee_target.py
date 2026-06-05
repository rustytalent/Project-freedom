"""Stream A rupee-target replay for post-touch execution.

This script does not retrain models and does not add alpha. It replays the
existing touched-pool population through the V2 simulator under the user's
actual operating discipline:

- at least Rs 600 gross reward per trade,
- at least Rs 6 per-share target movement,
- maximum Rs 2L notional,
- minimum Rs 30K notional,
- stop distance = 50% of the target move.

Acceptance per MASTER_PLAN Stream A: any cell crosses zero with p < 0.05 on
n >= 200; otherwise post-touch remains closed under rupee-floor sizing too.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from analysis.run_geometry_mode_sweep import (
    MODE_LABELS,
    _load_bundle,
    _simulate_report,
    _summarise,
    _write_json,
)
from liqpool.config import Config
from liqpool.execution_simulator_v2 import (
    ExecutionV2Config,
    RupeeTargetExecutionConfig,
)


DEFAULT_MODES = ("touch_confirmed", "break_confirmed", "sweep_reclaim")


def _mode_label(mode: str) -> str:
    return MODE_LABELS.get(str(mode), str(mode))


def _summarise_groups(trades: pd.DataFrame, keys: Sequence[str]) -> List[Dict]:
    if trades.empty:
        return []
    rows: List[Dict] = []
    for values, group in trades.groupby(list(keys), dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        row = {key: str(value) for key, value in zip(keys, values)}
        if "mode" in row:
            row["mode_label"] = _mode_label(row["mode"])
        row.update(_summarise(group))
        rows.append(row)
    rows.sort(key=lambda r: (r.get("mean_R", 0.0), r.get("n", 0)), reverse=True)
    return rows


def _candidate_rows(rows: Sequence[Dict], *, min_n: int, p_threshold: float) -> List[Dict]:
    candidates = [
        dict(row)
        for row in rows
        if int(row.get("n", 0)) >= int(min_n)
        and float(row.get("mean_R", 0.0)) > 0.0
        and float(row.get("p_value", 1.0)) < float(p_threshold)
    ]
    candidates.sort(key=lambda r: (r.get("mean_R", 0.0), r.get("n", 0)), reverse=True)
    return candidates


def _fmt_float(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{digits}f}"


def _markdown_table(rows: Sequence[Dict], columns: Sequence[str], *, max_rows: int = 20) -> List[str]:
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in list(rows)[:max_rows]:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                if col in {"win_rate"}:
                    vals.append(f"{value:.1%}")
                elif col == "p_value":
                    vals.append(_fmt_float(value, 4))
                else:
                    vals.append(_fmt_float(value))
            else:
                vals.append(str(value))
        out.append("| " + " | ".join(vals) + " |")
    return out


def _write_report(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headline = payload["headline_findings"]
    cfg = payload["rupee_target_config"]
    lines: List[str] = [
        "# Stream A Rupee-Target Replay",
        "",
        "## Verdict",
        "",
        f"- Decision: `{headline['decision']}`",
        f"- Any accepted cell: `{headline['any_accepted_cell']}`",
        f"- Trades replayed: `{headline['total_trades']}`",
        f"- Modes: `{', '.join(payload['modes'])}`",
        f"- Acceptance gate: `n >= {headline['min_n']}`, `mean_R > 0`, `p < {headline['p_threshold']}`.",
        "",
        "## Rupee Target Config",
        "",
        f"- Required gross reward: Rs {cfg['required_reward_inr']:,.0f}",
        f"- Minimum per-share move: Rs {cfg['min_per_share_move']:,.2f}",
        f"- Notional range: Rs {cfg['min_notional']:,.0f} to Rs {cfg['max_notional']:,.0f}",
        f"- Stop distance: `{cfg['stop_ratio']:.2f}x` target move",
        "",
        "## Top Overall Mode Cells",
        "",
    ]
    lines.extend(_markdown_table(
        payload["by_mode"],
        ("mode_label", "n", "win_rate", "mean_R", "mean_gross_R", "mean_cost_R", "PF", "p_value"),
    ))
    lines.extend(["", "## Top Factor X Mode Cells", ""])
    lines.extend(_markdown_table(
        payload["by_factor_mode"],
        ("factor", "mode_label", "n", "win_rate", "mean_R", "mean_gross_R", "mean_cost_R", "PF", "p_value"),
        max_rows=30,
    ))
    lines.extend(["", "## Accepted Cells", ""])
    accepted = payload["accepted_cells"]
    if accepted:
        lines.extend(_markdown_table(
            accepted,
            ("family", "factor", "mode_label", "n", "win_rate", "mean_R", "mean_gross_R", "mean_cost_R", "PF", "p_value"),
            max_rows=50,
        ))
    else:
        lines.append("- None.")
    lines.extend([
        "",
        "## Interpretation Guardrail",
        "",
        "- This is a Stream A execution-rescue test, not live approval.",
        "- Passing cells would still need same-slice nulls, cost stress, and forward outcome-log confirmation.",
        "- Failing cells close post-touch cash-equity MIS under the user's rupee-floor discipline.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args) -> Dict:
    bundle = Path(args.bundle).expanduser()
    if not bundle.exists():
        raise SystemExit(f"missing bundle: {bundle}")
    modes = tuple(m.strip() for m in str(args.modes).split(",") if m.strip())
    if not modes:
        raise SystemExit("at least one mode is required")

    report = _load_bundle(bundle)
    cfg = getattr(report, "config", None) or Config()
    rupee_target = RupeeTargetExecutionConfig(
        required_reward_inr=args.required_reward_inr,
        min_per_share_move=args.min_per_share_move,
        max_notional=args.max_notional,
        min_notional=args.min_notional,
        stop_ratio=args.stop_ratio,
    )
    v2_cfg = ExecutionV2Config(
        fill_policy=args.fill_policy,
        use_1m_resolution=args.use_1m_resolution,
        slippage_model=args.slippage_model,
        base_slippage_bps=args.base_slippage_bps,
        exchange=args.exchange,
        rupee_target=rupee_target,
    )

    print(
        f"[stream-a] replaying modes={modes} workers={args.workers} "
        f"required_reward=Rs {args.required_reward_inr:g}",
        flush=True,
    )
    trades = _simulate_report(
        report,
        cfg,
        modes=modes,
        v2_cfg=v2_cfg,
        raw_1m_dir=args.raw_1m_dir,
        split=args.split,
        workers=args.workers,
    )
    print(f"[stream-a] trades={len(trades)}", flush=True)

    by_mode = _summarise_groups(trades, ("mode",))
    by_factor = _summarise_groups(trades, ("factor",))
    by_factor_mode = _summarise_groups(trades, ("factor", "mode"))
    accepted: List[Dict] = []
    for family, rows in (
        ("mode", by_mode),
        ("factor", by_factor),
        ("factor_mode", by_factor_mode),
    ):
        for row in _candidate_rows(rows, min_n=args.min_n, p_threshold=args.p_threshold):
            enriched = {"family": family}
            enriched.update(row)
            accepted.append(enriched)
    accepted.sort(key=lambda r: (r.get("mean_R", 0.0), r.get("n", 0)), reverse=True)

    headline = {
        "decision": "RUPEE_TARGET_PASS" if accepted else "RUPEE_TARGET_FAIL",
        "any_accepted_cell": bool(accepted),
        "total_trades": int(len(trades)),
        "best_mode": by_mode[0] if by_mode else {},
        "best_factor_mode": by_factor_mode[0] if by_factor_mode else {},
        "min_n": int(args.min_n),
        "p_threshold": float(args.p_threshold),
    }
    payload = {
        "headline_findings": headline,
        "bundle": str(bundle),
        "split": args.split,
        "modes": list(modes),
        "rupee_target_config": {
            "required_reward_inr": float(args.required_reward_inr),
            "min_per_share_move": float(args.min_per_share_move),
            "max_notional": float(args.max_notional),
            "min_notional": float(args.min_notional),
            "stop_ratio": float(args.stop_ratio),
        },
        "by_mode": by_mode,
        "by_factor": by_factor,
        "by_factor_mode": by_factor_mode,
        "accepted_cells": accepted,
    }

    out = Path(args.out).expanduser()
    _write_json(out, payload)
    print(f"wrote {out}", flush=True)

    csv_out = Path(args.csv_out).expanduser()
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    csv_rows: List[Dict] = []
    for family, rows in (("mode", by_mode), ("factor", by_factor), ("factor_mode", by_factor_mode)):
        for row in rows:
            enriched = {"family": family}
            enriched.update(row)
            csv_rows.append(enriched)
    pd.DataFrame(csv_rows).to_csv(csv_out, index=False)
    print(f"wrote {csv_out}", flush=True)

    if args.trades_out:
        trades_out = Path(args.trades_out).expanduser()
        trades_out.parent.mkdir(parents=True, exist_ok=True)
        trades.to_parquet(trades_out, index=False)
        print(f"wrote {trades_out}", flush=True)

    report_out = Path(args.report_out).expanduser()
    _write_report(report_out, payload)
    print(f"wrote {report_out}", flush=True)
    print(f"decision: {headline['decision']}", flush=True)
    return payload


def parse_args(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="Path to multi_asset_report.pkl")
    ap.add_argument("--raw-1m-dir", default=None)
    ap.add_argument("--out", default="output_core25_head_alpha_710362b/stream_a_rupee_target.json")
    ap.add_argument("--csv-out", default="reports/stream_a_rupee_target_cells.csv")
    ap.add_argument("--report-out", default="reports/stream_a_rupee_target.md")
    ap.add_argument("--trades-out", default="")
    ap.add_argument("--split", default="oos", choices=("oos", "train"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--modes", default=",".join(DEFAULT_MODES))
    ap.add_argument("--exchange", default="NSE")
    ap.add_argument("--fill-policy", default="neutral", choices=("generous", "neutral", "conservative"))
    ap.add_argument("--use-1m-resolution", action="store_true")
    ap.add_argument("--slippage-model", default="state_dependent", choices=("flat", "state_dependent"))
    ap.add_argument("--base-slippage-bps", type=float, default=2.0)
    ap.add_argument("--required-reward-inr", type=float, default=600.0)
    ap.add_argument("--min-per-share-move", type=float, default=6.0)
    ap.add_argument("--max-notional", type=float, default=200_000.0)
    ap.add_argument("--min-notional", type=float, default=30_000.0)
    ap.add_argument("--stop-ratio", type=float, default=0.5)
    ap.add_argument("--min-n", type=int, default=200)
    ap.add_argument("--p-threshold", type=float, default=0.05)
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
