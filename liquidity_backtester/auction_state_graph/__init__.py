"""Standalone Auction State Graph package.

This module is intentionally isolated from model bundles, Sentinel, Kite live
loops, and execution code. It can be run as a replay-only candle-path layer.
"""

from .config import AuctionGraphConfig
from .runner import run_auction_state_graph

__all__ = ["AuctionGraphConfig", "run_auction_state_graph"]
