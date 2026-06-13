"""Wave 6 tests — the canonical ledger (Codex Problem S2).

SuggestionLedger now mirrors every suggestion into the ShadowLedger as a
``model_suggestion`` event, so there is ONE system of record. These
tests pin: the mirror happens, resolution stamps the canonical event's
judgment, the canonical events flow through ledger_export, and the
backward-compatible no-shadow path still works.
"""
from __future__ import annotations

from pathlib import Path

from sentinel.advisor import Suggestion, SuggestionLedger
from sentinel.ledger_export import to_decision_event
from sentinel.shadow_ledger import KIND_MODEL_SUGGESTION, ShadowLedger, read_session


def _sug(rule="R1_unprotected_winner", sym="NIFTY25000CE", action="ARM_TRAIL",
         premium=180.0, ot="CE", strike=25000.0):
    return Suggestion(
        suggestion_id=f"{rule}_{sym}", rule_id=rule, tradingsymbol=sym,
        action=action, reason="test reason", premium_at_suggestion=premium,
        option_type=ot, strike=strike, underlying="NIFTY")


def test_suggestion_mirrors_into_shadow_ledger(tmp_path):
    sl = ShadowLedger(tmp_path)
    led = SuggestionLedger(tmp_path / "suggestions.jsonl",
                           shadow_ledger=sl, session="2026-06-13")
    led.record_many([_sug()])
    rows = read_session(tmp_path, "2026-06-13")
    assert len(rows) == 1
    r = rows[0]
    assert r["kind"] == KIND_MODEL_SUGGESTION
    assert r["scientist"] == "R1_unprotected_winner"      # rule id preserved
    assert r["identity"]["instrument"] == "NIFTY25000CE"
    assert r["identity"]["option_type"] == "CE"
    assert r["identity"]["strike"] == 25000.0
    assert r["identity"]["premium"] == 180.0
    # advisory events carry no fabricated price hypothesis
    assert r["hypothesis"] is None


def test_resolution_stamps_canonical_judgment(tmp_path):
    sl = ShadowLedger(tmp_path)
    led = SuggestionLedger(tmp_path / "suggestions.jsonl", resolve_minutes=10.0,
                           shadow_ledger=sl, session="2026-06-13")
    led.record_many([_sug(action="CLOSE", premium=180.0)])
    # force the pending row past its horizon, then resolve with a lower
    # premium so CLOSE scores TRUE (exiting was right)
    led._pending[0]["recorded_monotonic"] -= 10 * 60 + 1
    n = led.resolve_due({"NIFTY25000CE": 150.0})
    assert n == 1
    rows = read_session(tmp_path, "2026-06-13")
    r = rows[0]          # last-write-wins -> the resolved version
    assert r["journey"]["complete"] is True
    assert r["judgment"]["was_entry_good"] is True


def test_canonical_suggestion_flows_through_ledger_export(tmp_path):
    sl = ShadowLedger(tmp_path)
    led = SuggestionLedger(tmp_path / "suggestions.jsonl",
                           shadow_ledger=sl, session="2026-06-13")
    led.record_many([_sug()])
    rows = read_session(tmp_path, "2026-06-13")
    ev = to_decision_event(rows[0])
    # model_suggestion -> MODEL_PREDICTION in the shared DecisionEvent vocab
    assert ev.event_type == "MODEL_PREDICTION"
    assert ev.scientist == "R1_unprotected_winner"
    assert ev.option_type == "CE"


def test_no_shadow_ledger_is_backward_compatible(tmp_path):
    """Without a shadow_ledger the SuggestionLedger behaves exactly as
    before — JSONL projection + EWMA scoring, no canonical mirror."""
    led = SuggestionLedger(tmp_path / "suggestions.jsonl", resolve_minutes=10.0)
    led.record_many([_sug(action="CLOSE")])
    led._pending[0]["recorded_monotonic"] -= 10 * 60 + 1
    n = led.resolve_due({"NIFTY25000CE": 150.0})
    assert n == 1
    # the compat JSONL still exists and has the suggestion
    assert (tmp_path / "suggestions.jsonl").exists()
    # no ledger_<session>.jsonl was created
    assert not list(tmp_path.glob("ledger_*.jsonl"))


def test_resolution_drains_transient_state(tmp_path):
    """Both scored and unscored resolutions must drop their in-flight
    entries (_canonical_events, _post_peak) so a long session can't leak."""
    sl = ShadowLedger(tmp_path)
    led = SuggestionLedger(tmp_path / "suggestions.jsonl", resolve_minutes=10.0,
                           shadow_ledger=sl, session="2026-06-13")
    led.record_many([
        _sug(rule="R1_unprotected_winner", action="ARM_TRAIL",
             sym="NIFTY25000CE"),
        _sug(rule="R5_concentration", action="WARN", ot="", strike=0.0,
             sym="NIFTY"),
    ])
    assert len(led._canonical_events) == 2
    for row in led._pending:
        row["recorded_monotonic"] -= 10 * 60 + 1
    led.resolve_due({"NIFTY25000CE": 150.0, "NIFTY": 100.0})
    # everything resolved -> transient maps fully drained
    assert led._canonical_events == {}
    assert led._post_peak == {}
    assert led._pending == []


def test_warn_suggestion_mirrors_but_never_scores(tmp_path):
    sl = ShadowLedger(tmp_path)
    led = SuggestionLedger(tmp_path / "suggestions.jsonl", resolve_minutes=10.0,
                           shadow_ledger=sl, session="2026-06-13")
    led.record_many([_sug(rule="R5_concentration", action="WARN",
                          ot="", strike=0.0, sym="NIFTY")])
    led._pending[0]["recorded_monotonic"] -= 10 * 60 + 1
    led.resolve_due({"NIFTY": 100.0})
    # WARN is unscorable -> EWMA count stays 0 for the rule
    assert led._counts.get("R5_concentration", 0) == 0
    # but it was still mirrored into the canonical substrate
    rows = read_session(tmp_path, "2026-06-13")
    assert any(r["scientist"] == "R5_concentration" for r in rows)
