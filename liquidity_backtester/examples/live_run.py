"""Track 5: morning live runner.

Workflow:
  1. Multi-asset analysis (Track 4.5)
  2. Drift check vs baseline (alert if model degraded)
  3. Position sizing per setup using your account state
  4. Interactive confirmation before each order
  5. Place orders via Zerodha Kite (dry_run=True by default, real with --live)
  6. Log everything to the journal

Daily routine:
  # 1. Once a day (manually, since Kite access_token expires daily):
  python examples/zerodha_login.py
  # ↑ outputs the access_token. Set it in shell:
  export KITE_API_KEY=your_api_key
  export KITE_ACCESS_TOKEN=your_access_token

  # 2. Run live (dry-run by default — no real orders sent):
  python examples/live_run.py --capital 500000

  # 3. To actually place orders, add --live (still requires per-trade confirmation):
  python examples/live_run.py --capital 500000 --live

  # 4. EOD reconcile:
  python examples/live_run.py --capital 500000 --eod
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

from liqpool import Config
from liqpool.multi_asset import run_multi_asset, print_multi_asset_summary
from liqpool.featurize import MultiAssetFeaturizer
from liqpool.timing import StateFeaturizer
from liqpool.sectors import sector_of
from liqpool.journal import TradeJournal
from liqpool.sizing import SizingConfig, AccountState, size_setup, format_sizing_line
from liqpool.drift import (DriftThresholds, extract_drift_metrics, check_drift,
                            load_baseline, save_baseline, print_drift_report)
from liqpool.broker_zerodha import broker_from_env


DEFAULT_BASKET = ("HDFCBANK.NS,ICICIBANK.NS,SBIN.NS,AXISBANK.NS,"
                  "TCS.NS,INFY.NS,HCLTECH.NS,"
                  "MARUTI.NS,TATAMOTORS.NS,"
                  "HINDUNILVR.NS")


def _ask_yes_no(prompt: str, default_no: bool = True) -> bool:
    suffix = " [y/N]: " if default_no else " [Y/n]: "
    try:
        ans = input(prompt + suffix).strip().lower()
    except EOFError:
        return False
    if not ans:
        return not default_no
    return ans in ("y", "yes")


def cmd_morning(args):
    # ----- 1. Multi-asset analysis -----
    cfg = Config(
        symbol=args.symbols.split(",")[0].strip(),
        base_interval=args.base, period=args.period,
        higher_tfs=args.tfs.split(","),
        test_horizon_bars=args.horizon, opt_iterations=args.iters,
        opt_explore_frac=0.35, opt_seed=11,
        min_pool_score=args.min_score,
    )
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    print(f"=== LIVE RUN ({len(symbols)} assets) ===")

    def prog(symbol, step):
        print(f"  [{symbol}]  {step}")

    report = run_multi_asset(symbols, cfg,
                              n_folds=args.folds, train_frac=args.train_frac,
                              iters_per_fold=args.iters, final_iters=args.final_iters,
                              min_train_days=args.min_train_days, progress=prog)

    if report.skipped_symbols:
        print("\n⚠  Assets dropped from run:")
        for sym in report.skipped_symbols:
            print(f"   - {sym}: {report.skip_reasons.get(sym, 'unknown')}")

    # ----- 2. Drift check -----
    drift_baseline_path = Path(args.out) / "drift_baseline.json"
    baseline = load_baseline(drift_baseline_path)
    current_metrics = extract_drift_metrics(report)
    drift_report = check_drift(current_metrics, baseline, DriftThresholds())
    print_drift_report(drift_report)

    # If status is OK, update baseline. If drift detected, KEEP old baseline for next comparison.
    if drift_report.overall_status == "OK":
        save_baseline(current_metrics, drift_baseline_path)
        print(f"  baseline updated → {drift_baseline_path}")

    # ----- 3. Summary -----
    print_multi_asset_summary(report)

    # ----- 4. Build today's setups & size them -----
    journal = TradeJournal(args.journal)
    journal.append("system", "_run", {
        "type": "morning_run_start",
        "n_assets": len(report.assets),
        "skipped": report.skipped_symbols,
        "drift_status": drift_report.overall_status,
    })

    if drift_report.overall_status == "DRIFT_DETECTED":
        print("\n🛑 DRIFT_DETECTED — pausing trading. Investigate before continuing.")
        for a in drift_report.alerts:
            journal.log_drift_alert(metric=a["metric"], current=a["value"],
                                     baseline=a["threshold"], threshold=a["threshold"],
                                     severity=a["severity"])
        return

    # Compute current state and rank candidates cross-asset (same logic as multi_asset_run.py)
    primary_h = sorted(report.unified_proximity.keys())[0] if report.unified_proximity else None
    TRADEABLE_T_TODAY = args.min_t
    TRADEABLE_Q = args.min_q
    DIR_ALIGN_MARGIN = 0.10

    candidates = []
    for symbol, ad in report.assets.items():
        if not ad.final_pools:
            continue
        base = ad.base_df
        atr_proxy = float((base["high"] - base["low"]).rolling(14).mean().iloc[-1])
        if atr_proxy <= 0:
            atr_proxy = 1.0
        current = float(base["close"].iloc[-1])
        now_ts = base.index[-1]
        live = [p for p, r in zip(ad.final_pools, ad.final_results)
                if p.available_at <= now_ts and not r.is_break
                and (r.touched_at is None or r.touched_at > now_ts)]
        state_feat = StateFeaturizer(base)
        current_state = state_feat.features_at(len(base) - 1, live)
        dir_p_up = (report.unified_direction.predict_state(current_state)
                     if report.unified_direction else None)

        def dir_tag(side_str, p_up=dir_p_up):
            if p_up is None: return "DIR_NA"
            if p_up >= 0.5 + DIR_ALIGN_MARGIN:
                return "DIR_ALIGN" if side_str == "above" else "DIR_FIGHT"
            if p_up <= 0.5 - DIR_ALIGN_MARGIN:
                return "DIR_ALIGN" if side_str == "below" else "DIR_FIGHT"
            return "DIR_NEUTRAL"

        active = [(p, r) for p, r in zip(ad.final_pools, ad.final_results)
                   if not r.is_break and r.outcome != "horizon_insufficient"
                   and (r.touched_at is None or r.touched_at > now_ts)]
        above = sorted([(p, r) for p, r in active if p.price_low > current],
                        key=lambda x: -x[0].score)[:5]
        below = sorted([(p, r) for p, r in active if p.price_high < current],
                        key=lambda x: -x[0].score)[:5]

        for side_str, plist in (("above", above), ("below", below)):
            for p, r in plist:
                dist_atr = (abs(p.mid - current)) / max(atr_proxy, 1e-9)
                X = report.unified_featurizer.transform_batch([p])
                q = float(report.unified_ml.predict(X, pools=[p])[0])
                t_today = (report.unified_proximity[primary_h].predict_one(
                    p, dist_atr, side_str, current_state, q) if primary_h is not None else 0.0)
                tag = dir_tag(side_str)
                bonus = 1.15 if tag == "DIR_ALIGN" else (0.85 if tag == "DIR_FIGHT" else 1.0)
                ev = q * t_today * bonus
                candidates.append({
                    "symbol": symbol, "side": side_str, "pool": p, "result": r,
                    "dist_atr": dist_atr, "q": q, "t_today": t_today,
                    "dir_tag": tag, "dir_p_up": dir_p_up, "ev": ev,
                    "current": current, "atr_proxy": atr_proxy,
                    "sector": sector_of(symbol),
                })

    tradeable = sorted([c for c in candidates
                          if c["q"] >= TRADEABLE_Q and c["t_today"] >= TRADEABLE_T_TODAY],
                         key=lambda c: -c["ev"])

    print("\n================ TODAY'S TRADEABLE SETUPS (sized) ================")
    if not tradeable:
        print("  No tradeable setups today.")
        journal.append("system", "_run", {"type": "no_tradeable_today"})
        return

    sizing_cfg = SizingConfig(
        risk_per_trade_pct=args.risk_pct,
        max_concurrent_positions=args.max_positions,
        max_sector_concentration_pct=args.max_sector_pct,
        daily_loss_cap_pct=args.daily_loss_cap_pct,
        leverage=args.leverage,
    )

    # Optional broker (read-only positions / LTP)
    broker = broker_from_env(dry_run=not args.live) if args.use_broker else None

    open_positions = []
    if broker is not None:
        try:
            pos = broker.get_positions()
            for p in pos:
                qty = p.get("quantity", 0) or 0
                if qty == 0:
                    continue
                open_positions.append({
                    "symbol": p.get("tradingsymbol", ""),
                    "sector": sector_of(p.get("tradingsymbol", "") + ".NS"),
                    "notional_inr": float(qty) * float(p.get("average_price", 0)),
                    "side": "BUY" if qty > 0 else "SELL",
                })
            print(f"  [broker] {len(open_positions)} open positions retrieved")
        except Exception as e:
            print(f"  [broker] failed to fetch positions: {e}")

    account = AccountState(
        capital_inr=float(args.capital),
        realised_pnl_today_inr=float(args.realised_pnl_today),
        open_positions=open_positions,
    )

    confirmed_orders = 0
    for i, c in enumerate(tradeable, 1):
        # Build entry/stop/target
        p = c["pool"]; a = c["atr_proxy"]
        if c["side"] == "below":      # BUY at pool high, stop below pool low, target above
            entry = p.price_high
            stop = p.price_low - 0.5 * a
            target = p.price_high + 2 * a
            order_side = "BUY"
        else:                           # SELL at pool low, stop above pool high, target below
            entry = p.price_low
            stop = p.price_high + 0.5 * a
            target = p.price_low - 2 * a
            order_side = "SELL"

        verdict = "TRADE_HIGH_CONFIDENCE" if c["ev"] >= 0.20 else "TRADE_CAUTIOUS"
        setup_dict = {
            "symbol": c["symbol"], "side": c["side"], "entry": entry, "stop": stop,
            "target": target, "sector": c["sector"], "dir_tag": c["dir_tag"],
        }
        decision = size_setup(setup_dict, sizing_cfg, account, verdict=verdict)

        print(f"\n  #{i} [{c['symbol']:<14}] {c['sector']:<9} {order_side}  pool "
              f"₹{p.price_low:.2f}-{p.price_high:.2f}  "
              f"Q={c['q']:.0%} T_today={c['t_today']:.0%} EV={c['ev']:.0%} "
              f"({c['dir_tag']}, {verdict})")
        print(f"     entry=₹{entry:.2f}  stop=₹{stop:.2f}  target=₹{target:.2f}  "
              f"R:R={(abs(target-entry)/abs(entry-stop)):.1f}:1")
        print(format_sizing_line(setup_dict, decision))

        alert_entry = journal.log_alert(
            symbol=c["symbol"], pool_id=f"{c['symbol']}_{p.formed_at}",
            entry_price=entry, stop=stop, target=target,
            q=c["q"], t_today=c["t_today"], ev=c["ev"],
            direction_p_up=c["dir_p_up"] if c["dir_p_up"] is not None else 0.5,
            dir_tag=c["dir_tag"], sector=c["sector"],
            shares=decision.shares, notional=decision.notional_inr,
            reasoning=f"verdict={verdict}; conf_mult={decision.confidence_multiplier}",
        )

        if decision.blocked or decision.shares <= 0:
            continue

        # Interactive confirmation
        place = False
        if args.auto_confirm:
            place = True
        elif args.interactive:
            place = _ask_yes_no(f"     >> Place this order?")

        if not place:
            journal.append("note", c["symbol"], {"text": "alert skipped by user / not confirmed"})
            continue

        # Place order (dry-run unless --live + confirmed)
        if broker is None:
            print(f"     [no broker configured — would have placed "
                  f"{order_side} {decision.shares} @ ₹{entry:.2f}]")
            journal.log_order_placed(symbol=c["symbol"], order_id="NO_BROKER",
                                      side=order_side, quantity=decision.shares,
                                      entry_price=entry, stop=stop, target=target,
                                      alert_id=alert_entry["timestamp"],
                                      dry_run=True)
        else:
            ts = c["symbol"].replace(".NS", "")    # Kite tradingsymbol
            result = broker.place_bracket_order(
                tradingsymbol=ts, side=order_side, quantity=decision.shares,
                entry_price=entry, stop=stop, target=target,
                confirm=(args.live and args.auto_confirm) or
                          (args.live and _ask_yes_no(f"     >> CONFIRM live order to broker?",
                                                       default_no=True)),
            )
            journal.log_order_placed(symbol=c["symbol"], order_id=result.order_id,
                                      side=order_side, quantity=decision.shares,
                                      entry_price=entry, stop=stop, target=target,
                                      alert_id=alert_entry["timestamp"],
                                      dry_run=result.dry_run)
            if result.error:
                print(f"     [order error: {result.error}]")
                journal.append("order_rejected", c["symbol"], {"error": result.error})
            else:
                confirmed_orders += 1
                print(f"     [order_id={result.order_id} status={result.status}]")

    print(f"\n[summary] {confirmed_orders} orders placed (of {len(tradeable)} tradeable setups). "
          f"Journal: {args.journal}")


def cmd_eod(args):
    """End-of-day reconcile: read broker positions, close out journal entries."""
    journal = TradeJournal(args.journal)
    broker = broker_from_env(dry_run=not args.live) if args.use_broker else None
    if broker is None:
        print("[eod] no broker configured — skipping reconcile")
        return
    try:
        positions = broker.get_positions()
        orders = broker.get_orders()
    except Exception as e:
        print(f"[eod] failed to fetch positions/orders: {e}")
        return

    print(f"[eod] {len(positions)} net positions, {len(orders)} orders today")
    journal.append("system", "_eod", {
        "n_positions": len(positions), "n_orders": len(orders),
        "positions": positions, "orders": orders,
    })
    print(f"[eod] journal updated → {args.journal}")


def cmd_journal(args):
    """Show the journal's calibration summary."""
    journal = TradeJournal(args.journal)
    summary = journal.calibration_summary()
    print("\n================ JOURNAL CALIBRATION ================")
    print(json.dumps(summary, indent=2))
    alerts_df = journal.alerts_df()
    if not alerts_df.empty:
        print(f"\n[recent alerts] ({len(alerts_df)})")
        print(alerts_df.tail(10).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # `morning`: full daily run
    m = sub.add_parser("morning", help="run morning analysis + sized alerts + (optional) orders")
    m.add_argument("--symbols", default=DEFAULT_BASKET)
    m.add_argument("--base", default="5m")
    m.add_argument("--period", default="60d")
    m.add_argument("--tfs", default="15min,60min,180min,1D,1W")
    m.add_argument("--folds", type=int, default=4)
    m.add_argument("--train-frac", type=float, default=0.6, dest="train_frac")
    m.add_argument("--iters", type=int, default=80)
    m.add_argument("--final-iters", type=int, default=200, dest="final_iters")
    m.add_argument("--horizon", type=int, default=150)
    m.add_argument("--min-score", type=float, default=2.0, dest="min_score")
    m.add_argument("--min-train-days", type=int, default=10, dest="min_train_days")
    m.add_argument("--capital", type=float, required=True,
                    help="Total trading capital in INR (used for position sizing)")
    m.add_argument("--realised-pnl-today", type=float, default=0.0,
                    dest="realised_pnl_today",
                    help="Today's realised P&L so far in INR (negative if losses, "
                          "drives daily-loss circuit)")
    m.add_argument("--risk-pct", type=float, default=1.0,
                    help="Per-trade risk as %% of capital. Default 1.0.")
    m.add_argument("--max-positions", type=int, default=4)
    m.add_argument("--max-sector-pct", type=float, default=40.0)
    m.add_argument("--daily-loss-cap-pct", type=float, default=3.0)
    m.add_argument("--leverage", type=float, default=5.0)
    m.add_argument("--min-q", type=float, default=0.55)
    m.add_argument("--min-t", type=float, default=0.05)
    m.add_argument("--use-broker", action="store_true",
                    help="Read positions/LTP from Kite (needs KITE_API_KEY + KITE_ACCESS_TOKEN)")
    m.add_argument("--live", action="store_true",
                    help="Enable LIVE order placement (default is dry-run). "
                          "Each order still requires confirmation.")
    m.add_argument("--interactive", action="store_true", default=True,
                    help="Prompt before each order. Default True.")
    m.add_argument("--auto-confirm", action="store_true",
                    help="Skip per-order prompt (DANGEROUS with --live).")
    m.add_argument("--journal", default="output/journal.jsonl")
    m.add_argument("--out", default="output")
    m.set_defaults(func=cmd_morning)

    # `eod`: end-of-day reconcile
    e = sub.add_parser("eod", help="end-of-day reconcile with broker")
    e.add_argument("--journal", default="output/journal.jsonl")
    e.add_argument("--use-broker", action="store_true", default=True)
    e.add_argument("--live", action="store_true")
    e.set_defaults(func=cmd_eod)

    # `journal`: show calibration summary
    j = sub.add_parser("journal", help="show journal calibration")
    j.add_argument("--journal", default="output/journal.jsonl")
    j.set_defaults(func=cmd_journal)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
