"""Demo: the premium-vs-spot divergence detector on a synthetic session.

This codifies the founder's live observation (2026-06-16): when spot
moves but the option that *should* be dying refuses to (the contrarian
leg holds up), the move is a trap and a reversal is coming.

It builds a synthetic NIFTY session with two planted events:
  * a BULL TRAP around 11:00 — spot pushes up, but puts are quietly
    accumulated (held up), so the up-move reverses.
  * a clean IV expansion later — both legs lift together (vega), which
    must NOT be mistaken for a directional trap.

Usage (from ``liquidity_backtester/`` root):

    PYTHONPATH=. python examples/run_premium_divergence.py

When you have a real option chain, feed a frame with columns
``ts, spot, call_premium, put_premium`` (one strike, ATM or near) into
``compute_divergence_frame`` / ``divergence_signals``. Optionally add
``call_delta`` / ``put_delta`` columns if you have model deltas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from liqpool.research import (
    PremiumDivergenceConfig,
    compute_divergence_frame,
    divergence_signals,
    summarize_divergence,
)


def _session(n: int = 375, seed: int = 7) -> pd.DataFrame:
    """One trading day of 1-min bars with two planted events:
      * a sustained bull trap (puts boosted during an up-move) — produces
        BULL_TRAPs and, when the wick→hold geometry aligns, STRONG_BULL_TRAPs.
      * a clean IV expansion (both legs lifted by vega) — must NOT trap.

    The canonical ₹43/₹44 fair-value rejection geometry is covered by the
    unit test :func:`tests.test_premium_divergence._fair_value_rejection_frame`
    — its timing is sensitive to the noise profile and the test is the
    right place to pin it. This demo shows the tier ladder in motion."""
    rng = np.random.default_rng(seed)
    spot = 23000 + np.cumsum(rng.normal(0, 4, n))
    d_spot = np.diff(spot, prepend=spot[0])
    call = 110 + np.cumsum(0.5 * d_spot + rng.normal(0, 0.8, n)) - np.arange(n) * 0.03
    put = 110 + np.cumsum(-0.5 * d_spot + rng.normal(0, 0.8, n)) - np.arange(n) * 0.03

    # Event 1 — bull trap window (90..120): spot pushes up, puts boosted.
    # The put leg's *systematic* upward residual through this window
    # creates one or more bars where Layer 1 + 2 + 3 all align →
    # STRONG_BULL_TRAP.
    a, b = 90, 120
    ramp = np.linspace(0, 90, b - a)
    spot[a:b] += ramp
    put[a:b] += np.linspace(0, 45, b - a)
    call[a:b] += 0.5 * ramp

    # Event 2 — clean IV expansion (180..210): both legs lifted equally.
    c, d = 180, 210
    iv = np.linspace(0, 35, d - c)
    spot[c:d] += np.linspace(0, 60, d - c)
    call[c:d] += 0.5 * np.linspace(0, 60, d - c) + iv
    put[c:d] += -0.5 * np.linspace(0, 60, d - c) + iv

    return pd.DataFrame({
        "ts": pd.date_range("2026-06-16 09:15", periods=n, freq="1min"),
        "spot": spot,
        "call_premium": np.clip(call, 1, None),
        "put_premium": np.clip(put, 1, None),
    })


def main() -> None:
    cfg = PremiumDivergenceConfig()
    df = _session()
    enriched = compute_divergence_frame(df, cfg)
    summ = summarize_divergence(enriched, cfg)

    print(f"[session]  {summ['n_bars']} bars  "
          f"signals={summ['n_signals']} ({summ['signal_rate']:.1%})")
    print(f"  strong: bull={summ['strong_bull_traps']} bear={summ['strong_bear_traps']}  "
          f"basic: bull={summ['bull_traps']} bear={summ['bear_traps']}  "
          f"confirmed: up={summ['confirmed_up']} down={summ['confirmed_down']}")

    print("\n[trap signals]  (one entry per event — cooldown 10 bars)")
    print("  legend: ★ = STRONG (Layer 1 + 2 + 3 agree, trade-grade)")
    print("          · = basic   (Layer 1 only — research-grade)\n")
    sigs = divergence_signals(enriched, cfg, cooldown_bars=10)
    for s in sigs:
        ts = pd.Timestamp(s.ts).strftime("%H:%M")
        marker = "★" if s.tier == "strong" else "·"
        print(f"  {marker} {ts}  {s.verdict:<17} take {s.leg.upper():<4} "
              f"(expect spot {'DOWN' if s.direction < 0 else 'UP'})  "
              f"spot={s.spot:.0f}  move={s.spot_move_norm:+.1f}σ  "
              f"intent_z={s.net_intent:+.1f}  "
              f"anomaly_z={s.residual_anomaly_z:+.1f}  "
              f"fair={s.fair_price:.1f}  "
              f"rejected={'Y' if s.fair_value_rejected else 'N'}  "
              f"entry≈{s.entry_premium:.1f} [{s.entry_zone_low:.1f}-{s.entry_zone_high:.1f}]")

    iv_window_traps = [s for s in sigs if 178 <= s.index <= 215]
    print(f"\n[IV expansion window 180-210]  traps fired = {len(iv_window_traps)}  "
          f"(want 0 — pure vega must not look like positioning)")
    bull_window_strong = [s for s in sigs
                           if 88 <= s.index <= 122 and s.tier == "strong"]
    print(f"[bull-trap window 90-120]  STRONG traps = {len(bull_window_strong)}  "
          f"(Layer 1 + 2 + 3 aligned — this is the trade-grade signal)")


if __name__ == "__main__":
    main()
