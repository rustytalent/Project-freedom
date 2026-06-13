"""Sentinel — standalone live trading copilot.

A single program for the VPS: live portfolio intelligence, Greeks
from first principles, per-position trailing stops, a portfolio-level
profit lock, dip recommendations, and a self-scoring suggestion
ledger — behind one professional dashboard.

Run:
    SENTINEL_DEMO=1 uvicorn sentinel.server:app --port 8800

See sentinel/README.md for the full runbook.
"""
__version__ = "0.1.0"
