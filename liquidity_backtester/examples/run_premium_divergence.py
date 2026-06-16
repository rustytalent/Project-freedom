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
    """One trading day of 1-min bars with a planted bull trap + IV spike."""
    rng = np.random.default_rng(seed)
    spot = 23000 + np.cumsum(rng.normal(0, 4, n))
    d_spot = np.diff(spot, prepend=spot[0])
    call = 110 + np.cumsum(0.5 * d_spot + rng.normal(0, 0.8, n)) - np.arange(n) * 0.03
    put = 110 + np.cumsum(-0.5 * d_spot + rng.normal(0, 0.8, n)) - np.arange(n) * 0.03

    # Bull trap around bar 90-120: spot pushes up, puts held up (accumulation).
    a, b = 90, 120
    L = b - a
    ramp = np.linspace(0, 90, L)
    spot[a:b] += ramp
    put[a:b] += np.linspace(0, 45, L)        # puts should fall ~ -45; instead rise
    call[a:b] += 0.5 * ramp                   # calls track honestly

    # Clean IV expansion around bar 250-280: both legs lift equally (vega).
    c, d = 250, 280
    L2 = d - c
    iv = np.linspace(0, 35, L2)
    spot[c:d] += np.linspace(0, 60, L2)
    call[c:d] += 0.5 * np.linspace(0, 60, L2) + iv
    put[c:d] += -0.5 * np.linspace(0, 60, L2) + iv

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
    print(f"  bull_traps={summ['bull_traps']}  bear_traps={summ['bear_traps']}  "
          f"confirmed_up={summ['confirmed_up']}  confirmed_down={summ['confirmed_down']}")

    print("\n[trap signals]  (one entry per event — cooldown 10 bars)")
    sigs = divergence_signals(enriched, cfg, cooldown_bars=10)
    for s in sigs:
        ts = pd.Timestamp(s.ts).strftime("%H:%M")
        print(f"  {ts}  {s.verdict:<10} take {s.leg.upper():<4} "
              f"(expect spot {'DOWN' if s.direction < 0 else 'UP'})  "
              f"spot={s.spot:.0f}  move={s.spot_move_norm:+.1f}σ  "
              f"intent_z={s.net_intent:+.1f}  "
              f"entry≈{s.entry_premium:.1f} [{s.entry_zone_low:.1f}-{s.entry_zone_high:.1f}]")

    # Confirm the IV window (bars 250-280) produced no traps.
    iv_window_traps = [s for s in sigs if 248 <= s.index <= 285]
    print(f"\n[IV expansion window 250-280]  traps fired = {len(iv_window_traps)}  "
          f"(want 0 — pure vega must not look like positioning)")


if __name__ == "__main__":
    main()
