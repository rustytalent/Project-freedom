"""Broker adapter layer — pluggable order execution.

Three implementations ship out of the box:

  * ``PaperBrokerAdapter``  — paper trading. Tracks fills internally;
    no real orders. Default for safety and the only adapter used until
    the founder explicitly opts in to live trading.

  * ``KiteBrokerAdapter``   — Zerodha Kite Connect adapter. Hooks the
    existing ``sentinel.kite_client.KiteAccount`` so all of its rate
    limiting + daily order cap protections kick in.

  * ``BrokerAdapter``       — abstract base. Any custom integration
    (Upstox, IBKR, etc.) implements this 3-method protocol.

Every adapter is invoked through the same ``BrokerOrder`` request /
``BrokerOrderResult`` reply shapes, so the manager doesn't know or
care which broker it's talking to.
"""
from __future__ import annotations

from .base import (
    BrokerAdapter,
    BrokerOrder,
    BrokerOrderResult,
    BrokerOrderState,
    BrokerPosition,
)
from .paper import PaperBrokerAdapter
from .kite import KiteBrokerAdapter, KiteBrokerConfig

__all__ = [
    "BrokerAdapter",
    "BrokerOrder",
    "BrokerOrderResult",
    "BrokerOrderState",
    "BrokerPosition",
    "KiteBrokerAdapter",
    "KiteBrokerConfig",
    "PaperBrokerAdapter",
]
