"""V4 live runner — the Monday entry point.

Wires the existing BeliefEngine to the v4 PortfolioManager, routes
intents to a pluggable BrokerAdapter, persists state for crash safety,
and emits a CockpitSnapshot per tick for the operator UI.

Usage (paper trading, no real money):

    from liqpool.research.belief.executor_v4.v4_runner import (
        V4Runner, V4RunnerConfig,
    )
    runner = V4Runner.paper()
    for tick in tick_stream():
        intent = runner.on_tick(tick)
        print(runner.cockpit().explainer_text)

Usage (live, founder Monday):

    from liqpool.research.belief.executor_v4.v4_runner import V4Runner
    runner = V4Runner.kite(
        api_key="...", access_token="...",
        confirm_real=True,    # MUST be set explicitly
    )

The runner does NOT pull data itself — it's tick-driven. Wire the
``on_tick(belief_snapshot, held_quotes)`` method to whatever ingestion
loop you have (the existing ``live_runner.py`` is the typical source).
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .broker import (
    BrokerAdapter,
    BrokerOrder,
    BrokerOrderResult,
    KiteBrokerAdapter,
    KiteBrokerConfig,
    PaperBrokerAdapter,
)
from .cockpit import CockpitSnapshot, build_cockpit_snapshot
from .economics import NIFTY_LOT_SIZE
from .engine_upgrades import EngineUpgrades, EngineUpgradesConfig
from .manager import (
    PortfolioIntent,
    PortfolioManager,
    PortfolioManagerConfig,
)
from .persistence import (
    ManagerPersistence,
    PersistenceConfig,
    capture_state,
    restore_pnl_only,
)


log = logging.getLogger(__name__)


@dataclass
class V4RunnerConfig:
    """Top-level runner configuration."""
    manager: Optional[PortfolioManagerConfig] = None
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    engine_upgrades: EngineUpgradesConfig = field(
        default_factory=EngineUpgradesConfig)
    apply_engine_upgrades: bool = True
    write_cockpit_to_jsonl: Optional[Path] = None
    cockpit_server_port: Optional[int] = None  # set to e.g. 8765 to enable
    emit_explainer_to_log: bool = True
    confirm_real_orders: bool = False
    # Sizing — lots translate to shares via lot_size.
    lot_size: int = NIFTY_LOT_SIZE
    # Exchange + product for orders sent through the broker.
    exchange: str = "NFO"
    product: str = "MIS"
    # Heart-beat: if no tick in this many seconds, do nothing dangerous.
    stale_tick_threshold_seconds: float = 30.0


@dataclass
class TickResult:
    """Per-tick output of the runner."""
    intent: PortfolioIntent
    cockpit: CockpitSnapshot
    broker_results: List[BrokerOrderResult]
    persisted: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.to_dict(),
            "cockpit": self.cockpit.to_dict(),
            "broker_results": [r.to_dict() for r in self.broker_results],
            "persisted": self.persisted,
            "notes": list(self.notes),
        }


class V4Runner:
    """Top-level executor v4 runtime — drives the manager + broker + UI."""

    def __init__(self, *,
                  cfg: Optional[V4RunnerConfig] = None,
                  broker: Optional[BrokerAdapter] = None) -> None:
        self.cfg = cfg or V4RunnerConfig()
        self.manager = PortfolioManager(self.cfg.manager)
        self.broker = broker or PaperBrokerAdapter()
        self.persistence = ManagerPersistence(self.cfg.persistence)
        self.engine_upgrades = (EngineUpgrades(self.cfg.engine_upgrades)
                                  if self.cfg.apply_engine_upgrades else None)
        self._cockpit_server = None
        if self.cfg.cockpit_server_port is not None:
            from .cockpit_server import CockpitServer
            self._cockpit_server = CockpitServer(self,
                                                   port=self.cfg.cockpit_server_port)
            self._cockpit_server.start()
        self._last_tick_ts: Optional[float] = None
        # Try to restore from disk if a snapshot exists for today.
        snap = self.persistence.load_today()
        if snap is not None:
            restore_pnl_only(self.manager, snap)
            log.info("restored manager P&L state from %s",
                      self.persistence.path_for_today())

    # ── Convenience factories ────────────────────────────────────────

    @classmethod
    def paper(cls, *,
                manager_cfg: Optional[PortfolioManagerConfig] = None,
                mark_lookup=None,
                slippage_bps: float = 0.0,
                ) -> "V4Runner":
        cfg = V4RunnerConfig(manager=manager_cfg, confirm_real_orders=False)
        broker = PaperBrokerAdapter(
            mark_lookup=mark_lookup,
            default_slippage_bps=slippage_bps,
        )
        return cls(cfg=cfg, broker=broker)

    @classmethod
    def kite(cls, *,
              kite_account,
              confirm_real: bool = False,
              manager_cfg: Optional[PortfolioManagerConfig] = None,
              broker_cfg: Optional[KiteBrokerConfig] = None,
              ) -> "V4Runner":
        cfg = V4RunnerConfig(manager=manager_cfg,
                                confirm_real_orders=confirm_real)
        broker = KiteBrokerAdapter(
            kite_account=kite_account,
            cfg=broker_cfg or KiteBrokerConfig(confirm_real=confirm_real),
        )
        return cls(cfg=cfg, broker=broker)

    # ── Per-tick public API ──────────────────────────────────────────

    def on_tick(self, belief_snapshot: Mapping[str, Any],
                  held_quotes: Optional[Dict[str, Any]] = None,
                  ) -> TickResult:
        """One belief snapshot in → one TickResult out.

        Steps:
          1. Hand snapshot to manager.evaluate()
          2. Translate the PortfolioIntent into broker orders for new
             entries / closures
          3. Persist state to disk
          4. Build the cockpit snapshot
          5. Emit / log
        """
        cfg = self.cfg
        notes: List[str] = []
        self._last_tick_ts = time.time()

        # 1. Engine upgrades (adaptive warmup, dirty-quote healing, etc.)
        snap_for_manager = dict(belief_snapshot)
        if self.engine_upgrades is not None:
            snap_for_manager, upgrade_report = self.engine_upgrades.transform(
                snap_for_manager)
            if upgrade_report.actions_taken:
                notes.append("upgrades: "
                              + "; ".join(upgrade_report.actions_taken[:3]))
        # 1b. Manager evaluation.
        if hasattr(self.broker, "begin_tick"):
            self.broker.begin_tick()
        intent = self.manager.evaluate(snap_for_manager,
                                         held_quotes=held_quotes)

        # 2. Broker routing.
        broker_results = self._route_to_broker(intent)
        # Keep the paper adapter's marks fresh for accurate unrealized P&L.
        if isinstance(self.broker, PaperBrokerAdapter):
            for slot in (belief_snapshot.get("slot_readings") or []):
                sym = self._infer_tradingsymbol(belief_snapshot, slot)
                if sym:
                    mark = slot.get("mark_price")
                    if mark and math.isfinite(mark) and mark > 0:
                        self.broker.update_mark(sym, float(mark))

        # 3. Persistence.
        snap = capture_state(self.manager)
        persisted = self.persistence.write(snap)

        # 4. Cockpit.
        cockpit = build_cockpit_snapshot(intent.to_dict())
        if self._cockpit_server is not None:
            try:
                self._cockpit_server.publish(cockpit.to_dict())
            except Exception:
                pass    # never let UI plumbing break the trading loop

        # 5. Emit to JSONL if configured.
        if cfg.write_cockpit_to_jsonl is not None:
            self._append_jsonl(cfg.write_cockpit_to_jsonl,
                                 cockpit.to_dict())
        if cfg.emit_explainer_to_log:
            log.info(cockpit.explainer_text)

        return TickResult(intent=intent, cockpit=cockpit,
                           broker_results=broker_results,
                           persisted=persisted, notes=notes)

    def stop(self) -> None:
        """Clean shutdown (stops the cockpit server if running)."""
        if self._cockpit_server is not None:
            try:
                self._cockpit_server.stop()
            except Exception:
                pass
            self._cockpit_server = None

    def cockpit(self) -> Optional[CockpitSnapshot]:
        """Latest cockpit snapshot. Useful for serving a status endpoint."""
        # Re-derive from the last manager state.
        # For simplicity we ask the manager for a HOLD-like summary.
        intent = self.manager.evaluate(self._empty_snapshot(), held_quotes={})
        return build_cockpit_snapshot(intent.to_dict())

    # ── Broker routing ───────────────────────────────────────────────

    def _route_to_broker(self,
                          intent: PortfolioIntent) -> List[BrokerOrderResult]:
        """Translate per-tick PortfolioIntent → BrokerOrder calls."""
        results: List[BrokerOrderResult] = []
        # New entry → BUY order for each leg.
        if intent.new_entry:
            h = intent.new_entry.get("hypothesis") or {}
            tradingsymbol = self._build_tradingsymbol(h)
            if tradingsymbol:
                qty = int(h.get("size_lots", 0)) * self.cfg.lot_size
                if qty > 0:
                    side = "BUY" if int(h.get("direction", +1)) > 0 else "SELL"
                    order = BrokerOrder(
                        client_order_id=self.broker._mk_client_id(
                            h.get("position_id", "?"), "ENTRY"),
                        tradingsymbol=tradingsymbol,
                        exchange=self.cfg.exchange,
                        side=side, quantity=qty,
                        order_type="LIMIT",
                        product=self.cfg.product,
                        limit_price=float(h.get("entry_premium", 0.0)),
                        tag="ENTRY",
                        position_id=h.get("position_id"),
                    )
                    results.append(self.broker.place_order(order))
        # Closures → opposite side.
        for closed in intent.closed_this_tick:
            pid = closed.get("position_id", "")
            outcome = closed.get("outcome") or {}
            # We don't have the hypothesis on closure (it's already in the
            # ledger), but the runner kept track via the open_states. Look
            # up the closed ledger from the ledger store.
            ledger = None
            for cl in self.manager.ledger_store.closed_positions():
                if cl.hypothesis.position_id == pid:
                    ledger = cl
                    break
            if ledger is None:
                continue
            h = ledger.hypothesis
            tradingsymbol = self._build_tradingsymbol({
                "contract_side": h.contract_side,
                "strike_price": h.strike_price,
            })
            if not tradingsymbol:
                continue
            qty = h.size_lots * self.cfg.lot_size
            side = "SELL" if h.direction > 0 else "BUY"
            order = BrokerOrder(
                client_order_id=self.broker._mk_client_id(pid, "EXIT"),
                tradingsymbol=tradingsymbol,
                exchange=self.cfg.exchange,
                side=side, quantity=qty,
                order_type="LIMIT",
                product=self.cfg.product,
                limit_price=float(outcome.get("exit_premium", 0.0)),
                tag="EXIT",
                position_id=pid,
            )
            results.append(self.broker.place_order(order))
        return results

    # ── Tradingsymbol construction ──────────────────────────────────

    def _build_tradingsymbol(self, hypothesis_dict: Mapping[str, Any]) -> str:
        """Build a NIFTY weekly tradingsymbol from hypothesis fields.

        Best-effort; if the caller has a richer tradingsymbol mapper,
        they can monkey-patch this method.
        """
        side = str(hypothesis_dict.get("contract_side", ""))
        strike = int(round(float(hypothesis_dict.get("strike_price", 0))))
        if not side or strike <= 0:
            return ""
        # Caller is expected to override this; default returns a
        # placeholder so paper trading and tests still work.
        return f"NIFTYWK{strike}{side}"

    def _infer_tradingsymbol(self, snapshot: Mapping[str, Any],
                                slot: Mapping[str, Any]) -> str:
        """Inverse: given a slot reading, infer the tradingsymbol used
        for our orders."""
        side = str(slot.get("option_type") or "")
        strike = int(round(float(slot.get("strike") or 0)))
        if not side or strike <= 0:
            return ""
        return f"NIFTYWK{strike}{side}"

    def _empty_snapshot(self) -> Dict[str, Any]:
        return {"ts": "", "spot": 0.0, "bars_seen": 0, "is_warm": False,
                "thesis": {}, "iv_state": {}, "battlefield": {},
                "decision": {}, "winding": {}, "bull_state": {},
                "bear_state": {}, "sweep_state": {}, "slot_readings": []}

    @staticmethod
    def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
