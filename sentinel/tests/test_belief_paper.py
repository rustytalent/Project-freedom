from __future__ import annotations

from sentinel.belief_paper import BeliefPaperLedger
from sentinel.live_publisher import ModelSignal


def _sig(ts: str, signal: str, spot: float, direction: int, confidence: float = 0.8,
         action: str | None = None, bars: int = 1) -> ModelSignal:
    return ModelSignal(
        ts_ist=ts,
        asset="NIFTY_DEMO",
        model="premium_belief_engine",
        signal=signal,
        confidence=confidence,
        trust_tier="SHADOW",
        reason_codes=[f"thesis={signal}"],
        extras={
            "spot": spot,
            "direction": direction,
            "bars_seen": bars,
            "executor": {
                "action": action or signal,
                "intent": action or signal,
                "allowed": True,
                "direction": direction,
                "confidence": confidence,
                "size_fraction": 0.0,
                "reason_codes": [signal],
                "reject_reasons": [],
            },
        },
    )


def test_hold_signal_can_open_shadow_position_and_mark_pnl(tmp_path):
    ledger = BeliefPaperLedger(
        enabled=True, journal_path=tmp_path / "paper.jsonl",
        lot_size=75, lots=1, min_confidence=0.5, trade_holds=True,
    )
    assert ledger.ingest(_sig("10:00:00", "HOLD_BEAR", 24100.0, -1, bars=1))
    assert ledger.snapshot()["side"] == "SHORT"
    ledger.ingest(_sig("10:00:01", "HOLD_BEAR", 24090.0, -1, bars=2))
    snap = ledger.snapshot()
    assert snap["position_open"] is True
    assert snap["unrealized_points"] == 10.0
    assert snap["unrealized_pnl"] == 750.0
    assert snap["paper_size_fraction"] == 1.0


def test_exit_signal_realizes_pnl(tmp_path):
    ledger = BeliefPaperLedger(
        enabled=True, journal_path=tmp_path / "paper.jsonl",
        lot_size=75, lots=1, min_confidence=0.5, trade_holds=True,
    )
    ledger.ingest(_sig("10:00:00", "HOLD_BULL", 100.0, 1, bars=1))
    assert ledger.ingest(_sig("10:00:02", "EXIT_BULL", 104.0, 1,
                              action="EXIT", bars=2))
    snap = ledger.snapshot()
    assert snap["position_open"] is False
    assert snap["realized_pnl"] == 300.0
    assert snap["n_trades"] == 1
    assert snap["wins"] == 1


def test_duplicate_signal_is_ignored(tmp_path):
    ledger = BeliefPaperLedger(
        enabled=True, journal_path=tmp_path / "paper.jsonl",
        lot_size=75, lots=1, min_confidence=0.5, trade_holds=True,
    )
    sig = _sig("10:00:00", "HOLD_BULL", 100.0, 1, bars=1)
    assert ledger.ingest(sig)
    assert ledger.ingest(sig) is False
    assert ledger.snapshot()["position_age_ticks"] == 0


def test_disabled_ledger_does_not_trade(tmp_path):
    ledger = BeliefPaperLedger(enabled=False, journal_path=tmp_path / "paper.jsonl")
    assert ledger.ingest(_sig("10:00:00", "HOLD_BULL", 100.0, 1, bars=1)) is False
    snap = ledger.snapshot()
    assert snap["enabled"] is False
    assert snap["position_open"] is False
