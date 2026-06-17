"""Standalone live runner for the Premium Belief Engine.

This module is deliberately boring: it does not invent alpha and it does
not place orders. It turns live Kite quotes into the exact inputs that
``BeliefEngine`` expects, then writes append-only JSONL rows that can be
tailed by Sentinel's ``LiveSignalsTail`` or inspected directly.

The engine remains independent from Sentinel. Sentinel is an optional
consumer of the stream, not a required dependency.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from liqpool.contracts.signals import ModelSignal
from liqpool.research.belief.engine import BeliefEngine, BeliefEngineConfig, BeliefSnapshot
from liqpool.research.belief.executor import BeliefExecutionGovernor, ExecutionGovernorConfig
from liqpool.research.belief.mark_price import Quote
from liqpool.research.streaming_divergence import StreamingDivergenceEngine, StreamingRead


IST = timezone(timedelta(hours=5, minutes=30))
LOG = logging.getLogger(__name__)
ContractKey = Tuple[float, str]


@dataclass(frozen=True)
class OptionContract:
    """One option contract selected for the live battlefield."""

    tradingsymbol: str
    strike: float
    option_type: str
    expiry: date
    instrument_token: Optional[int] = None
    exchange: str = "NFO"

    @property
    def key(self) -> ContractKey:
        return (float(self.strike), self.option_type)

    @property
    def kite_key(self) -> str:
        return f"{self.exchange}:{self.tradingsymbol}"


@dataclass
class BeliefLiveConfig:
    """Operational knobs for the standalone live runner."""

    underlying: str = "NIFTY"
    spot_key: str = "NSE:NIFTY 50"
    exchange: str = "NFO"
    strike_step: float = 50.0
    levels: int = 5
    poll_seconds: float = 1.0
    warmup_bars: int = 80
    refresh_contracts_every: int = 30
    output_jsonl: Path = Path("/var/lib/sentinel/liqpool_live_signals.jsonl")
    executor_output_jsonl: Optional[Path] = None
    shadow_only: bool = True
    max_ticks: Optional[int] = None
    include_streaming_divergence: bool = True
    enable_executor: bool = True
    executor_min_entry_confidence: float = 0.66
    executor_min_scalp_confidence: float = 0.72
    min_quote_gap_seconds: float = 1.05
    terminal: bool = False
    clear_terminal: bool = True
    demo_start_spot: float = 23500.0


@dataclass(frozen=True)
class LiveBeliefRow:
    """Serializable row written by the runner."""

    signal: ModelSignal
    snapshot: Dict[str, Any]
    stream: Optional[Dict[str, Any]]
    contracts: List[Dict[str, Any]]
    executor: Optional[Dict[str, Any]] = None

    def to_json_row(self) -> Dict[str, Any]:
        row = self.signal.to_row()
        row.setdefault("extras", {})
        row["extras"] = dict(row["extras"])
        row["extras"]["belief_snapshot"] = self.snapshot
        if self.stream is not None:
            row["extras"]["streaming_divergence"] = self.stream
        row["extras"]["contracts"] = self.contracts
        if self.executor is not None:
            row["extras"]["executor"] = self.executor
        return row


def now_ist_hms() -> str:
    return datetime.now(IST).strftime("%H:%M:%S")


def _parse_expiry(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(IST).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)).date()
    except Exception:
        return None


def _row_name(row: Mapping[str, Any]) -> str:
    return str(row.get("name") or row.get("underlying") or "").upper()


def _row_type(row: Mapping[str, Any]) -> str:
    return str(row.get("instrument_type") or row.get("option_type") or "").upper()


def _row_symbol(row: Mapping[str, Any]) -> str:
    return str(row.get("tradingsymbol") or row.get("symbol") or "")


def _row_exchange(row: Mapping[str, Any], default: str) -> str:
    return str(row.get("exchange") or default)


def _nearest_atm(spot: float, strike_step: float) -> float:
    if strike_step <= 0:
        raise ValueError("strike_step must be positive")
    return round(float(spot) / strike_step) * strike_step


def select_belief_contracts(
    instruments: Iterable[Mapping[str, Any]],
    *,
    spot: float,
    underlying: str = "NIFTY",
    exchange: str = "NFO",
    strike_step: float = 50.0,
    levels: int = 5,
    now: Optional[date] = None,
) -> Dict[ContractKey, OptionContract]:
    """Pick ATM ± ``levels`` CE/PE contracts from the nearest live expiry."""

    if levels < 0:
        raise ValueError("levels must be non-negative")

    today = now or datetime.now(IST).date()
    underlying_upper = underlying.upper()
    rows: List[Tuple[Mapping[str, Any], date]] = []
    for row in instruments:
        if _row_name(row) != underlying_upper:
            continue
        if _row_type(row) not in {"CE", "PE"}:
            continue
        exp = _parse_expiry(row.get("expiry"))
        if exp is None or exp < today:
            continue
        try:
            float(row.get("strike") or 0.0)
        except Exception:
            continue
        rows.append((row, exp))

    if not rows:
        raise RuntimeError(f"no live {underlying} CE/PE instruments found")

    nearest_expiry = min(exp for _row, exp in rows)
    atm = _nearest_atm(float(spot), strike_step)
    wanted = {atm + i * strike_step for i in range(-levels, levels + 1)}

    selected: Dict[ContractKey, OptionContract] = {}
    for row, exp in rows:
        if exp != nearest_expiry:
            continue
        strike = float(row.get("strike") or 0.0)
        if strike not in wanted:
            continue
        option_type = _row_type(row)
        key = (strike, option_type)
        selected[key] = OptionContract(
            tradingsymbol=_row_symbol(row),
            strike=strike,
            option_type=option_type,
            expiry=exp,
            instrument_token=(int(row["instrument_token"])
                              if row.get("instrument_token") is not None else None),
            exchange=_row_exchange(row, exchange),
        )

    expected = (2 * levels + 1) * 2
    if len(selected) < expected:
        LOG.warning(
            "selected %s/%s contracts for %s expiry=%s atm=%s",
            len(selected), expected, underlying, nearest_expiry, atm,
        )
    return selected


def kite_quote_to_belief_quote(raw: Mapping[str, Any], *, ts: Any = None) -> Quote:
    """Convert one Kite ``quote`` payload into Belief Engine's ``Quote``."""

    depth = raw.get("depth") or {}
    buy = (depth.get("buy") or [{}])[0] or {}
    sell = (depth.get("sell") or [{}])[0] or {}
    bid = float(buy.get("price") or 0.0)
    ask = float(sell.get("price") or 0.0)
    bid_qty = float(buy.get("quantity") or buy.get("orders") or 0.0)
    ask_qty = float(sell.get("quantity") or sell.get("orders") or 0.0)
    ltp = float(raw.get("last_price") or raw.get("ltp") or math.nan)
    last_trade_time = raw.get("last_trade_time")
    ltp_age_s = 0.0
    if isinstance(last_trade_time, datetime):
        base_ts = ts if isinstance(ts, datetime) else datetime.now(IST)
        if last_trade_time.tzinfo is None:
            last_trade_time = last_trade_time.replace(tzinfo=IST)
        ltp_age_s = max(0.0, (base_ts.astimezone(IST) - last_trade_time.astimezone(IST)).total_seconds())
    return Quote(
        bid=bid,
        ask=ask,
        bid_qty=bid_qty,
        ask_qty=ask_qty,
        ltp=ltp,
        ltp_age_s=ltp_age_s,
        ts=ts,
    )


def _best_atm_marks(quotes: Mapping[ContractKey, Quote], spot: float, strike_step: float) -> Tuple[Optional[float], Optional[float]]:
    atm = _nearest_atm(spot, strike_step)
    ce = quotes.get((atm, "CE"))
    pe = quotes.get((atm, "PE"))
    def mark(q: Optional[Quote]) -> Optional[float]:
        if q is None:
            return None
        if q.bid > 0 and q.ask > 0:
            return (q.bid + q.ask) / 2.0
        if math.isfinite(float(q.ltp)):
            return float(q.ltp)
        return None
    return mark(ce), mark(pe)


def model_signal_from_snapshot(
    snapshot: BeliefSnapshot,
    *,
    asset: str,
    stream: Optional[StreamingRead] = None,
    executor: Optional[Mapping[str, Any]] = None,
) -> ModelSignal:
    decision = snapshot.decision
    reason_codes = []
    if decision.no_trade_reason:
        reason_codes.append(decision.no_trade_reason)
    if decision.thesis_state:
        reason_codes.append(f"thesis={decision.thesis_state}")
    if snapshot.iv_state.state:
        reason_codes.append(f"iv={snapshot.iv_state.state}")
    if snapshot.battlefield.verdict:
        reason_codes.append(f"battlefield={snapshot.battlefield.verdict}")
    if stream is not None and stream.stable_verdict:
        reason_codes.append(f"stream={stream.stable_verdict}")
    if executor is not None and executor.get("intent"):
        reason_codes.append(f"executor={executor.get('intent')}")

    risk = "shadow_only"
    if decision.action in {"NO_TRADE", "WAIT"}:
        risk = "blocked"
    elif decision.spread_friendliness < 0.45:
        risk = "spread_unfriendly"

    return ModelSignal(
        ts_ist=now_ist_hms(),
        asset=asset,
        model="premium_belief_engine",
        signal=decision.action,
        confidence=max(0.0, min(1.0, float(decision.confidence))),
        trust_tier="SHADOW",
        risk=risk,
        reason_codes=reason_codes[:8],
        extras={
            "source": "liqpool",
            "spot": snapshot.spot,
            "trade_allowed": decision.trade_allowed,
            "direction": decision.direction,
            "strike": decision.strike.to_dict(),
            "bars_seen": snapshot.bars_seen,
            "is_warm": snapshot.is_warm,
            "executor": executor,
        },
        source="liqpool",
    )


def make_live_row(
    snapshot: BeliefSnapshot,
    *,
    asset: str,
    contracts: Mapping[ContractKey, OptionContract],
    stream: Optional[StreamingRead] = None,
    executor: Optional[Mapping[str, Any]] = None,
) -> LiveBeliefRow:
    return LiveBeliefRow(
        signal=model_signal_from_snapshot(snapshot, asset=asset, stream=stream, executor=executor),
        snapshot=snapshot.to_dict(),
        stream=stream.to_dict() if stream is not None else None,
        contracts=[
            {
                "strike": c.strike,
                "option_type": c.option_type,
                "tradingsymbol": c.tradingsymbol,
                "expiry": c.expiry.isoformat(),
                "instrument_token": c.instrument_token,
                "exchange": c.exchange,
            }
            for c in sorted(contracts.values(), key=lambda x: (x.strike, x.option_type))
        ],
        executor=dict(executor) if executor is not None else None,
    )


def _executor_for_config(cfg: BeliefLiveConfig) -> Optional[BeliefExecutionGovernor]:
    if not cfg.enable_executor:
        return None
    return BeliefExecutionGovernor(ExecutionGovernorConfig(
        min_warm_bars=max(1, int(cfg.warmup_bars)),
        min_entry_confidence=cfg.executor_min_entry_confidence,
        min_scalp_confidence=cfg.executor_min_scalp_confidence,
    ))


def _executor_intent(
    governor: Optional[BeliefExecutionGovernor],
    snapshot: BeliefSnapshot,
    stream: Optional[StreamingRead],
) -> Optional[Dict[str, Any]]:
    if governor is None:
        return None
    stream_payload = stream.to_dict() if stream is not None else None
    return governor.evaluate(snapshot.to_dict(), stream_payload).to_dict()


def _write_executor_row(
    path: Optional[Path],
    *,
    asset: str,
    snapshot: BeliefSnapshot,
    executor: Optional[Mapping[str, Any]],
) -> None:
    if path is None or executor is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts_ist": now_ist_hms(),
        "asset": asset,
        "spot": snapshot.spot,
        "bars_seen": snapshot.bars_seen,
        "executor": dict(executor),
    }
    with path.open("a", buffering=1) as fh:
        fh.write(json.dumps(row, default=str, separators=(",", ":")) + "\n")


def terminal_view(row: LiveBeliefRow) -> str:
    """Compact operator screen for a single belief row."""

    snap = row.snapshot
    decision = snap.get("decision", {})
    thesis = snap.get("thesis", {})
    iv_state = snap.get("iv_state", {})
    battlefield = snap.get("battlefield", {})
    winding = snap.get("winding", {})
    stream = row.stream or {}
    executor = row.executor or {}
    strike = decision.get("strike", {}) or {}

    lines = [
        "=" * 78,
        "PREMIUM BELIEF ENGINE - LIVE SHADOW TERMINAL",
        "=" * 78,
        f"ts={snap.get('ts')}  spot={float(snap.get('spot', 0.0)):.2f}  "
        f"bars={snap.get('bars_seen')}  warm={snap.get('is_warm')}",
        "",
        f"ACTION: {decision.get('action')}  allowed={decision.get('trade_allowed')}  "
        f"direction={decision.get('direction')}  confidence={float(decision.get('confidence', 0.0)):.2f}",
        f"STRIKE: {strike.get('label', '') or 'NA'}  side={strike.get('side', '') or 'NA'}  "
        f"level={strike.get('level', 0)}",
        f"REASON: {decision.get('no_trade_reason') or decision.get('thesis_state') or 'none'}",
        "",
        f"THESIS: {thesis.get('composite_state')}  "
        f"bull={float(thesis.get('bull_thesis_score', 0.0)):.1f}  "
        f"bear={float(thesis.get('bear_thesis_score', 0.0)):.1f}  "
        f"danger={float(thesis.get('no_trade_score', 0.0)):.1f}",
        f"IV: {iv_state.get('state')}  dir={iv_state.get('direction')}  "
        f"conf={float(iv_state.get('confidence', 0.0)):.2f}  "
        f"clean_marks={float(iv_state.get('clean_mark_fraction', 0.0)):.2f}",
        f"BATTLEFIELD: {battlefield.get('verdict')}  dir={battlefield.get('direction')}  "
        f"conf={float(battlefield.get('confidence', 0.0)):.2f}",
        f"WINDING: {winding.get('zone')}  conf={float(winding.get('confidence', 0.0)):.2f}",
        "",
        f"STREAM: stable={stream.get('stable_verdict', 'NA')}  "
        f"provisional={stream.get('provisional_verdict', 'NA')}  "
        f"action={stream.get('action', 'NA')}  dir={stream.get('direction', 'NA')}",
        f"EXECUTOR: intent={executor.get('intent', 'DISABLED')}  "
        f"allowed={executor.get('allowed', False)}  "
        f"size={float(executor.get('size_fraction') or 0.0):.2f}  "
        f"profile={executor.get('profile', 'NA') or 'NA'}  "
        f"mode={executor.get('order_mode', 'SHADOW_ONLY')}",
        "",
        "CONTRACTS:",
    ]
    for contract in row.contracts[:12]:
        lines.append(
            f"  {contract['tradingsymbol']:<24} {contract['option_type']} "
            f"strike={float(contract['strike']):.0f} exp={contract['expiry']}"
        )
    if len(row.contracts) > 12:
        lines.append(f"  ... {len(row.contracts) - 12} more")
    lines.extend([
        "",
        "This is SHADOW output only. No orders are placed.",
        "=" * 78,
    ])
    return "\n".join(lines)


def print_terminal(row: LiveBeliefRow, *, clear: bool = True) -> None:
    if clear:
        sys.stdout.write("\033[2J\033[H")
    sys.stdout.write(terminal_view(row) + "\n")
    sys.stdout.flush()


class KiteBeliefLiveRunner:
    """Poll Kite quotes and feed the Premium Belief Engine."""

    def __init__(self, *, api_key: str, access_token: str, cfg: BeliefLiveConfig) -> None:
        self.api_key = api_key
        self.access_token = access_token
        self.cfg = cfg
        self.engine = BeliefEngine(BeliefEngineConfig(
            strike_step=cfg.strike_step,
            levels=cfg.levels,
            warmup_bars=cfg.warmup_bars,
        ))
        self.executor = _executor_for_config(cfg)
        self.streaming = StreamingDivergenceEngine() if cfg.include_streaming_divergence else None
        self._kite = None
        self._contracts: Dict[ContractKey, OptionContract] = {}
        self._last_contract_refresh_tick = -10**9
        self._last_quote_call = 0.0

    def _conn(self):
        if self._kite is None:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=self.api_key)
            kite.set_access_token(self.access_token)
            self._kite = kite
        return self._kite

    def _load_instruments(self) -> List[Mapping[str, Any]]:
        return list(self._conn().instruments(self.cfg.exchange) or [])

    def _quote_raw(self, keys: List[str]) -> Dict[str, Any]:
        now = time.monotonic()
        wait = (self._last_quote_call + self.cfg.min_quote_gap_seconds) - now
        if wait > 0:
            time.sleep(wait)
        raw = self._conn().quote(keys) or {}
        self._last_quote_call = time.monotonic()
        return raw

    def _spot(self) -> float:
        raw = self._quote_raw([self.cfg.spot_key])
        spot = float((raw.get(self.cfg.spot_key) or {}).get("last_price") or 0.0)
        if spot <= 0:
            raise RuntimeError(f"invalid spot LTP for {self.cfg.spot_key}: {spot}")
        return spot

    def _maybe_refresh_contracts(self, *, spot: float, tick_no: int) -> None:
        if (self._contracts
                and tick_no - self._last_contract_refresh_tick < self.cfg.refresh_contracts_every):
            return
        self._contracts = select_belief_contracts(
            self._load_instruments(),
            spot=spot,
            underlying=self.cfg.underlying,
            exchange=self.cfg.exchange,
            strike_step=self.cfg.strike_step,
            levels=self.cfg.levels,
        )
        self._last_contract_refresh_tick = tick_no
        LOG.info("selected %s contracts for %s", len(self._contracts), self.cfg.underlying)

    def _snapshot_quotes(self, ts: datetime) -> Tuple[float, Dict[ContractKey, Quote]]:
        if not self._contracts:
            return self._spot(), {}
        raw = self._quote_raw([self.cfg.spot_key] + [c.kite_key for c in self._contracts.values()])
        spot = float((raw.get(self.cfg.spot_key) or {}).get("last_price") or 0.0)
        if spot <= 0:
            raise RuntimeError(f"invalid spot LTP for {self.cfg.spot_key}: {spot}")
        out: Dict[ContractKey, Quote] = {}
        for key, contract in self._contracts.items():
            payload = raw.get(contract.kite_key) or raw.get(contract.tradingsymbol)
            if payload is None:
                continue
            out[key] = kite_quote_to_belief_quote(payload, ts=ts)
        return spot, out

    def _stream_read(self, ts: datetime, spot: float, quotes: Mapping[ContractKey, Quote]) -> Optional[StreamingRead]:
        if self.streaming is None:
            return None
        ce_mark, pe_mark = _best_atm_marks(quotes, spot, self.cfg.strike_step)
        if ce_mark is None or pe_mark is None:
            return None
        return self.streaming.update(ts, spot, ce_mark, pe_mark)

    def run_forever(self) -> None:
        self.cfg.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        LOG.info("starting Premium Belief live runner shadow_only=%s out=%s",
                 self.cfg.shadow_only, self.cfg.output_jsonl)
        tick_no = 0
        with self.cfg.output_jsonl.open("a", buffering=1) as fh:
            while self.cfg.max_ticks is None or tick_no < self.cfg.max_ticks:
                started = time.monotonic()
                ts = datetime.now(IST)
                try:
                    if not self._contracts:
                        initial_spot = self._spot()
                        self._maybe_refresh_contracts(spot=initial_spot, tick_no=tick_no)
                    spot, quotes = self._snapshot_quotes(ts)
                    if tick_no - self._last_contract_refresh_tick >= self.cfg.refresh_contracts_every:
                        self._maybe_refresh_contracts(spot=spot, tick_no=tick_no)
                        spot, quotes = self._snapshot_quotes(ts)
                    if not quotes:
                        raise RuntimeError("no option quotes returned")
                    stream = self._stream_read(ts, spot, quotes)
                    snapshot = self.engine.observe(
                        ts=ts.isoformat(),
                        spot=spot,
                        quotes=quotes,
                        hunt_verdict=(stream.stable_verdict if stream is not None else ""),
                        trap_verdict=(stream.stable_verdict if stream is not None else ""),
                    )
                    executor = _executor_intent(self.executor, snapshot, stream)
                    row = make_live_row(
                        snapshot,
                        asset=self.cfg.underlying,
                        contracts=self._contracts,
                        stream=stream,
                        executor=executor,
                    )
                    fh.write(json.dumps(row.to_json_row(), default=str, separators=(",", ":")) + "\n")
                    _write_executor_row(
                        self.cfg.executor_output_jsonl,
                        asset=self.cfg.underlying,
                        snapshot=snapshot,
                        executor=executor,
                    )
                    if self.cfg.terminal:
                        print_terminal(row, clear=self.cfg.clear_terminal)
                    LOG.info(
                        "tick=%s spot=%.2f action=%s confidence=%.2f warm=%s executor=%s reason=%s",
                        tick_no,
                        spot,
                        snapshot.decision.action,
                        snapshot.decision.confidence,
                        snapshot.is_warm,
                        (executor or {}).get("intent", "disabled"),
                        snapshot.decision.no_trade_reason or snapshot.decision.thesis_state,
                    )
                except KeyboardInterrupt:
                    raise
                except Exception:
                    LOG.exception("belief live tick failed")
                tick_no += 1
                elapsed = time.monotonic() - started
                sleep_for = max(0.0, self.cfg.poll_seconds - elapsed)
                if sleep_for:
                    time.sleep(sleep_for)


class DemoBeliefLiveRunner:
    """Market-closed rehearsal runner.

    This generates a deterministic-ish option battlefield from a synthetic
    spot path. It is for terminal/JOSNL plumbing and warmup/decision checks,
    not for profitability validation.
    """

    def __init__(self, *, cfg: BeliefLiveConfig, seed: int = 7) -> None:
        self.cfg = cfg
        self.engine = BeliefEngine(BeliefEngineConfig(
            strike_step=cfg.strike_step,
            levels=cfg.levels,
            warmup_bars=cfg.warmup_bars,
        ))
        self.executor = _executor_for_config(cfg)
        self.streaming = StreamingDivergenceEngine() if cfg.include_streaming_divergence else None
        self.random = random.Random(seed)
        self._contracts: Dict[ContractKey, OptionContract] = {}

    def _spot(self, tick_no: int) -> float:
        drift = math.sin(tick_no / 18.0) * 38.0 + math.sin(tick_no / 7.0) * 13.0
        slow = tick_no * 0.22
        noise = self.random.uniform(-2.5, 2.5)
        return float(self.cfg.demo_start_spot + drift + slow + noise)

    def _contracts_for_spot(self, spot: float) -> Dict[ContractKey, OptionContract]:
        atm = _nearest_atm(spot, self.cfg.strike_step)
        exp = datetime.now(IST).date() + timedelta(days=3)
        out: Dict[ContractKey, OptionContract] = {}
        for level in range(-self.cfg.levels, self.cfg.levels + 1):
            strike = atm + level * self.cfg.strike_step
            for typ in ("CE", "PE"):
                out[(float(strike), typ)] = OptionContract(
                    tradingsymbol=f"{self.cfg.underlying}DEMO{int(strike)}{typ}",
                    strike=float(strike),
                    option_type=typ,
                    expiry=exp,
                    exchange=self.cfg.exchange,
                )
        return out

    def _quotes(self, spot: float, tick_no: int, ts: datetime) -> Dict[ContractKey, Quote]:
        self._contracts = self._contracts_for_spot(spot)
        quotes: Dict[ContractKey, Quote] = {}
        pressure = math.sin(tick_no / 11.0)
        iv_wave = 1.0 + 0.18 * math.sin(tick_no / 23.0)
        for key, _contract in self._contracts.items():
            strike, typ = key
            intrinsic = max(0.0, spot - strike) if typ == "CE" else max(0.0, strike - spot)
            distance = abs(spot - strike) / max(self.cfg.strike_step, 1.0)
            time_value = max(8.0, 42.0 * math.exp(-0.27 * distance)) * iv_wave
            directional_bump = pressure * (7.0 if typ == "CE" else -7.0)
            fair = max(1.0, intrinsic + time_value + directional_bump)
            spread = max(0.10, fair * (0.006 + 0.002 * distance))
            bid = max(0.05, fair - spread / 2.0)
            ask = max(bid + 0.05, fair + spread / 2.0)
            base_qty = max(50.0, 1600.0 - 135.0 * distance)
            bid_qty = base_qty * (1.0 + max(0.0, pressure) * (1.0 if typ == "CE" else 0.25))
            ask_qty = base_qty * (1.0 + max(0.0, -pressure) * (1.0 if typ == "PE" else 0.25))
            quotes[key] = Quote(
                bid=bid,
                ask=ask,
                bid_qty=bid_qty,
                ask_qty=ask_qty,
                ltp=(bid + ask) / 2.0,
                ltp_age_s=0.0,
                ts=ts,
            )
        return quotes

    def _stream_read(self, ts: datetime, spot: float, quotes: Mapping[ContractKey, Quote]) -> Optional[StreamingRead]:
        if self.streaming is None:
            return None
        ce_mark, pe_mark = _best_atm_marks(quotes, spot, self.cfg.strike_step)
        if ce_mark is None or pe_mark is None:
            return None
        return self.streaming.update(ts, spot, ce_mark, pe_mark)

    def run_forever(self) -> None:
        self.cfg.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        LOG.info("starting DEMO Premium Belief runner out=%s", self.cfg.output_jsonl)
        tick_no = 0
        with self.cfg.output_jsonl.open("a", buffering=1) as fh:
            while self.cfg.max_ticks is None or tick_no < self.cfg.max_ticks:
                ts = datetime.now(IST)
                spot = self._spot(tick_no)
                quotes = self._quotes(spot, tick_no, ts)
                stream = self._stream_read(ts, spot, quotes)
                snapshot = self.engine.observe(
                    ts=ts.isoformat(),
                    spot=spot,
                    quotes=quotes,
                    hunt_verdict=(stream.stable_verdict if stream is not None else ""),
                    trap_verdict=(stream.stable_verdict if stream is not None else ""),
                )
                executor = _executor_intent(self.executor, snapshot, stream)
                row = make_live_row(
                    snapshot,
                    asset=f"{self.cfg.underlying}_DEMO",
                    contracts=self._contracts,
                    stream=stream,
                    executor=executor,
                )
                fh.write(json.dumps(row.to_json_row(), default=str, separators=(",", ":")) + "\n")
                _write_executor_row(
                    self.cfg.executor_output_jsonl,
                    asset=f"{self.cfg.underlying}_DEMO",
                    snapshot=snapshot,
                    executor=executor,
                )
                if self.cfg.terminal:
                    print_terminal(row, clear=self.cfg.clear_terminal)
                LOG.info(
                    "demo_tick=%s spot=%.2f action=%s confidence=%.2f warm=%s executor=%s",
                    tick_no,
                    spot,
                    snapshot.decision.action,
                    snapshot.decision.confidence,
                    snapshot.is_warm,
                    (executor or {}).get("intent", "disabled"),
                )
                tick_no += 1
                if self.cfg.poll_seconds > 0:
                    time.sleep(self.cfg.poll_seconds)


def runner_from_env(cfg: BeliefLiveConfig) -> KiteBeliefLiveRunner:
    api_key = os.getenv("KITE_API_KEY", "").strip()
    access_token = os.getenv("KITE_ACCESS_TOKEN", "").strip()
    if not api_key or not access_token:
        raise RuntimeError("KITE_API_KEY and KITE_ACCESS_TOKEN are required")
    return KiteBeliefLiveRunner(api_key=api_key, access_token=access_token, cfg=cfg)
