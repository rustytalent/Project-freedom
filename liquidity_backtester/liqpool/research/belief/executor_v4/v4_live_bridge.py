"""KiteV4LiveRunner — the clean live bridge (latest brain only, no v3).

The founder's serious concern (2026-06-21, paraphrased): "What if we go
live and then discover oops — these old stale modules are connected?"

That concern was JUSTIFIED. The old ``live_runner.KiteBeliefLiveRunner``
still constructs and uses the v3 ``executor.py``. Its KITE DATA path is
excellent and reusable; only its brain is stale.

This bridge reuses the proven Kite data machinery from
``KiteBeliefLiveRunner`` (connection, contract selection, quote polling,
``BeliefEngine.observe``) but routes every ``BeliefSnapshot`` into the
LATEST ``V4Runner`` brain — which carries the full v4 stack: probability
web, MM-mind, manipulation board, fat-tail amplifier, crowd mirror,
strategy library, portfolio risk, hedge proposer, AND the Adaptive Exit
Quote Engine.

The v3 executor is NEVER constructed here (we override __init__ to skip
it), so there is zero chance of a stale module silently handling money.

Wiring:

    Kite WebSocket / REST  (KiteBeliefLiveRunner data path — proven)
            ↓
    BeliefEngine.observe(ts, spot, quotes)  →  BeliefSnapshot
            ↓
    snapshot.to_dict()
            ↓
    V4Runner.on_tick(snapshot_dict)   ← LATEST brain, exit engine, broker
            ↓
    BrokerAdapter (paper / Kite dry-run / Kite live)

Held premiums for open positions are resolved from the snapshot's
``slot_readings`` (which carry per-contract mark_price + friendliness +
spread_state), so no separate held-quote plumbing is required for a
clean first wire.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from liqpool.research.belief.live_runner import (
    BeliefLiveConfig,
    IST,
    KiteBeliefLiveRunner,
)

from .broker import BrokerAdapter, KiteBrokerAdapter, KiteBrokerConfig, PaperBrokerAdapter
from .v4_runner import V4Runner, V4RunnerConfig


LOG = logging.getLogger(__name__)


@dataclass
class KiteV4BridgeConfig:
    """Knobs for the live bridge."""
    belief: BeliefLiveConfig = None       # data-path config (contracts, polling)
    v4: Optional[V4RunnerConfig] = None    # brain config
    confirm_real_orders: bool = False      # MUST be True to send real orders
    cockpit_server_port: Optional[int] = None   # e.g. 8765 to serve the UI
    write_cockpit_jsonl: Optional[Path] = None
    print_explainer_each_tick: bool = True


class KiteV4LiveRunner(KiteBeliefLiveRunner):
    """Live runner that feeds Kite data into the v4 brain — never v3.

    Subclasses ``KiteBeliefLiveRunner`` to inherit the proven Kite data
    machinery (``_conn``, ``_snapshot_quotes``, ``_maybe_refresh_contracts``,
    ``_stream_read``, ``engine``), but:
      * overrides ``__init__`` so the v3 executor is NEVER constructed
      * overrides ``run_forever`` to route snapshots into ``V4Runner``
    """

    def __init__(self, *,
                  api_key: str,
                  access_token: str,
                  bridge_cfg: KiteV4BridgeConfig,
                  broker: Optional[BrokerAdapter] = None) -> None:
        cfg = bridge_cfg.belief or BeliefLiveConfig()
        # Initialise the data path WITHOUT calling the parent __init__
        # (which would construct the v3 executor). We set up only the
        # pieces we actually use.
        self.api_key = api_key
        self.access_token = access_token
        self.cfg = cfg
        # Reuse the parent's engine + streaming construction by importing
        # the same classes it uses.
        from liqpool.research.belief.engine import (
            BeliefEngine, BeliefEngineConfig,
        )
        from liqpool.research.streaming_divergence import (
            StreamingDivergenceEngine,
        )
        self.engine = BeliefEngine(BeliefEngineConfig(
            strike_step=cfg.strike_step,
            levels=cfg.levels,
            warmup_bars=cfg.warmup_bars,
        ))
        # CRITICAL: the v3 executor is deliberately NOT constructed.
        self.executor = None
        self.streaming = (StreamingDivergenceEngine()
                           if cfg.include_streaming_divergence else None)
        self._kite = None
        self._contracts = {}
        self._last_contract_refresh_tick = -10**9
        self._last_quote_call = 0.0

        # Build the v4 brain.
        self.bridge_cfg = bridge_cfg
        v4_cfg = bridge_cfg.v4 or V4RunnerConfig(
            cockpit_server_port=bridge_cfg.cockpit_server_port,
            write_cockpit_to_jsonl=bridge_cfg.write_cockpit_jsonl,
            emit_explainer_to_log=False,
            confirm_real_orders=bridge_cfg.confirm_real_orders,
        )
        if broker is None:
            broker = self._build_broker(bridge_cfg)
        self.v4 = V4Runner(cfg=v4_cfg, broker=broker)
        LOG.info("KiteV4LiveRunner ready — brain=v4 (latest), v3 executor "
                  "NOT constructed, confirm_real=%s",
                  bridge_cfg.confirm_real_orders)

    def _build_broker(self, bridge_cfg: KiteV4BridgeConfig) -> BrokerAdapter:
        """Build the broker adapter. Paper unless confirm_real is set AND
        a real Kite account is available."""
        if not bridge_cfg.confirm_real_orders:
            LOG.info("paper broker (confirm_real=False)")
            return PaperBrokerAdapter(mark_lookup=None)
        # Live Kite broker — wraps the existing rate-limited KiteAccount.
        try:
            from sentinel.kite_client import KiteAccount
            account = KiteAccount(
                label="founder_v4",
                api_key=self.api_key,
                access_token=self.access_token,
                dry_run=False,
            )
            LOG.warning("LIVE Kite broker — real orders WILL be sent")
            return KiteBrokerAdapter(
                kite_account=account,
                cfg=KiteBrokerConfig(confirm_real=True),
            )
        except Exception as exc:
            LOG.error("could not build live Kite broker (%s); "
                       "falling back to PAPER for safety", exc)
            return PaperBrokerAdapter(mark_lookup=None)

    # ── The v4 tick loop ────────────────────────────────────────────

    def run_forever(self) -> None:
        """Poll Kite, build a BeliefSnapshot, route it into the v4 brain.

        Identical data path to the parent, but the brain is V4Runner and
        the v3 executor is never invoked.
        """
        cfg = self.cfg
        if cfg.output_jsonl:
            cfg.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        LOG.info("starting KiteV4LiveRunner shadow=%s confirm_real=%s",
                  not self.bridge_cfg.confirm_real_orders,
                  self.bridge_cfg.confirm_real_orders)
        tick_no = 0
        try:
            while cfg.max_ticks is None or tick_no < cfg.max_ticks:
                started = time.monotonic()
                ts = datetime.now(IST)
                try:
                    if not self._contracts:
                        initial_spot = self._spot()
                        self._maybe_refresh_contracts(spot=initial_spot,
                                                        tick_no=tick_no)
                    spot, quotes = self._snapshot_quotes(ts)
                    if (tick_no - self._last_contract_refresh_tick
                            >= cfg.refresh_contracts_every):
                        self._maybe_refresh_contracts(spot=spot,
                                                        tick_no=tick_no)
                        spot, quotes = self._snapshot_quotes(ts)
                    if not quotes:
                        raise RuntimeError("no option quotes returned")
                    stream = self._stream_read(ts, spot, quotes)
                    snapshot = self.engine.observe(
                        ts=ts.isoformat(),
                        spot=spot,
                        quotes=quotes,
                        hunt_verdict=(stream.stable_verdict
                                       if stream is not None else ""),
                        trap_verdict=(stream.stable_verdict
                                       if stream is not None else ""),
                    )
                    # ── route into the LATEST v4 brain ──────────────
                    snapshot_dict = snapshot.to_dict()
                    result = self.v4.on_tick(snapshot_dict)

                    if self.bridge_cfg.print_explainer_each_tick:
                        print(result.cockpit.explainer_text)

                    LOG.info(
                        "tick=%s spot=%.2f action=%s warm=%s "
                        "v4_entry=%s closed=%s broker_orders=%s",
                        tick_no, spot,
                        snapshot.decision.action,
                        snapshot.is_warm,
                        result.intent.new_entry is not None,
                        len(result.intent.closed_this_tick),
                        len(result.broker_results),
                    )
                except KeyboardInterrupt:
                    raise
                except Exception:
                    LOG.exception("v4 live tick failed")
                tick_no += 1
                elapsed = time.monotonic() - started
                sleep_for = max(0.0, cfg.poll_seconds - elapsed)
                if sleep_for:
                    time.sleep(sleep_for)
        finally:
            # Clean shutdown of the cockpit server if running.
            try:
                self.v4.stop()
            except Exception:
                pass


def build_kite_v4_runner(*,
                           api_key: str,
                           access_token: str,
                           underlying: str = "NIFTY",
                           confirm_real: bool = False,
                           cockpit_port: Optional[int] = 8765,
                           poll_seconds: float = 1.0,
                           ) -> KiteV4LiveRunner:
    """Convenience factory for the Monday launcher."""
    belief_cfg = BeliefLiveConfig(
        underlying=underlying,
        poll_seconds=poll_seconds,
    )
    bridge_cfg = KiteV4BridgeConfig(
        belief=belief_cfg,
        confirm_real_orders=confirm_real,
        cockpit_server_port=cockpit_port,
    )
    return KiteV4LiveRunner(
        api_key=api_key,
        access_token=access_token,
        bridge_cfg=bridge_cfg,
    )
