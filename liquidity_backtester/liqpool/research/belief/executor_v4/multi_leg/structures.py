"""Structure constructors — turn a strategy class + current state into a
MultiLegBundle ready for the broker.

Each constructor takes:
  * ``spot``                — current underlying spot
  * ``slot_readings``       — the snapshot's per-contract slot readings;
                               the constructor picks strikes from this
                               grid (so it inherits the manager's
                               instrument selection)
  * ``lots``                — the size in lots applied to every leg
                               (kept uniform so the structure stays
                               balanced)
  * ``wing_width_pct``      — optional override for how far the wings
                               sit from the body (default per structure)

The constructors do NOT touch the broker. They produce a
MultiLegBundle that ``broker.place_multi_leg_bundle()`` then executes
atomically.

Strategies implemented:

  * **iron_condor**     — sell OTM CE + OTM PE, buy further-OTM
                            wings. Pure neutral. Wins when spot stays
                            in the body.
  * **jade_lizard**     — sell OTM PE + ATM CE, buy further-OTM CE
                            wing. Asymmetric — defined risk on the
                            upside, undefined on the downside (the
                            short PE). Used when there is a soft
                            downside floor but indecisive top.
  * **butterfly**       — buy 1 ATM-wing + sell 2 ATM body + buy 1
                            ATM+wing. Tight body, pinning play.
  * **strangle**        — buy OTM CE + buy OTM PE. Long volatility.
                            Wins on big moves either way.
  * **vertical_spread** — directional debit/credit spread, two legs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .bundle import MultiLegBundle
from .leg_spec import LEG_LONG, LEG_SHORT, LegSpec, OPT_CE, OPT_PE


STRUCTURE_IRON_CONDOR = "iron_condor"
STRUCTURE_JADE_LIZARD = "jade_lizard"
STRUCTURE_BUTTERFLY = "butterfly"
STRUCTURE_STRANGLE = "strangle"
STRUCTURE_VERTICAL_SPREAD = "vertical_spread"


# ── Strike-grid helpers ───────────────────────────────────────────


def _slot_to_dict(slot: Any) -> Dict[str, Any]:
    return slot if isinstance(slot, dict) else {}


def _build_strike_table(slot_readings: List[Any]
                         ) -> Dict[Tuple[float, str], Dict[str, Any]]:
    """Index slot readings by (strike, option_type) so the constructors
    can ask "give me the CE at 23150" without an O(n) scan."""
    out: Dict[Tuple[float, str], Dict[str, Any]] = {}
    for raw in slot_readings:
        slot = _slot_to_dict(raw)
        try:
            strike = float(slot.get("strike", 0.0))
            opt = str(slot.get("option_type") or "")
        except Exception:
            continue
        if strike <= 0 or opt not in (OPT_CE, OPT_PE):
            continue
        out[(strike, opt)] = slot
    return out


def _nearest_strike(table: Dict[Tuple[float, str], Dict[str, Any]],
                     target_strike: float, option_type: str,
                     exclude: Optional[set] = None,
                     direction: int = 0,
                     ) -> Optional[Tuple[float, Dict[str, Any]]]:
    """Pick the slot whose strike is closest to ``target_strike`` for
    the given option_type.

    ``exclude``: optional set of strikes already used by other legs in
    the same bundle. Prevents wings from collapsing onto body strikes
    when the strike grid is coarse relative to the wing width.

    ``direction``: when +1, only consider strikes ≥ target; when -1,
    only ≤ target. Used to push wings further OTM when the nearest
    strike was already taken.
    """
    excl = exclude or set()
    candidates = [(strike, slot) for (strike, opt), slot in table.items()
                  if opt == option_type and strike not in excl]
    if direction > 0:
        candidates = [c for c in candidates if c[0] >= target_strike]
    elif direction < 0:
        candidates = [c for c in candidates if c[0] <= target_strike]
    if not candidates:
        return None
    return min(candidates, key=lambda x: abs(x[0] - target_strike))


def _atm_strike(table: Dict[Tuple[float, str], Dict[str, Any]],
                 spot: float, option_type: str) -> Optional[float]:
    """Round ``spot`` to the nearest available strike of the given
    option_type."""
    res = _nearest_strike(table, spot, option_type)
    return res[0] if res is not None else None


def _make_leg(role: str, strike: float, option_type: str, side: str,
               lots: int, slot: Optional[Dict[str, Any]],
               ) -> LegSpec:
    """Pack a LegSpec with the trading symbol + estimated premium pulled
    from the slot reading."""
    return LegSpec(
        role=role,
        strike=strike,
        option_type=option_type,
        side=side,
        lots=lots,
        tradingsymbol=str((slot or {}).get("label") or ""),
        limit_price=None,
        estimated_premium=float((slot or {}).get("mark_price") or 0.0),
        metadata={
            "moneyness_label": str((slot or {}).get("moneyness_label") or ""),
            "friendliness": float((slot or {}).get("friendliness") or 0.0),
            "is_abnormal": bool((slot or {}).get("is_abnormal") or False),
            "spread_state": str((slot or {}).get("spread_state") or ""),
        },
    )


# ── Per-structure constructors ────────────────────────────────────


@dataclass
class IronCondor:
    """Body short ± body width, wings long ± wing width.

    Defaults: body 1% OTM from spot, wings 0.5% past the body.
    Premium target: 50% of net credit decayed.
    """
    body_offset_pct: float = 0.010    # 1.0% from spot
    wing_width_pct: float = 0.005     # 0.5% past body
    target_credit_pct: float = 0.50
    stop_loss_pct: float = 1.50

    def construct(self, *, spot: float,
                     slot_readings: List[Any],
                     lots: int = 1,
                     regime_tag: Optional[Dict[str, Any]] = None,
                     ) -> Optional[MultiLegBundle]:
        if spot <= 0 or lots <= 0:
            return None
        table = _build_strike_table(slot_readings)
        body_ce_target = spot * (1 + self.body_offset_pct)
        body_pe_target = spot * (1 - self.body_offset_pct)
        wing_ce_target = spot * (1 + self.body_offset_pct + self.wing_width_pct)
        wing_pe_target = spot * (1 - self.body_offset_pct - self.wing_width_pct)
        # Body picks first so wings can exclude their strikes.
        picks_body = [
            ("short_otm_ce", body_ce_target, OPT_CE, LEG_SHORT, 0),
            ("short_otm_pe", body_pe_target, OPT_PE, LEG_SHORT, 0),
        ]
        legs: List[LegSpec] = []
        used_strikes: set = set()
        for role, target, opt, side, push in picks_body:
            res = _nearest_strike(table, target, opt, exclude=used_strikes,
                                   direction=push)
            if res is None:
                return None
            strike, slot = res
            used_strikes.add(strike)
            legs.append(_make_leg(role, strike, opt, side, lots, slot))
        # Wings push further OTM (+1 for CE wing means strikes ABOVE target,
        # -1 for PE wing means strikes BELOW target) and exclude body strikes.
        picks_wing = [
            ("long_wing_ce", wing_ce_target, OPT_CE, LEG_LONG, +1),
            ("long_wing_pe", wing_pe_target, OPT_PE, LEG_LONG, -1),
        ]
        for role, target, opt, side, push in picks_wing:
            res = _nearest_strike(table, target, opt, exclude=used_strikes,
                                   direction=push)
            if res is None:
                # Fall back to any further strike of that type even if
                # it's not in the "preferred" direction.
                res = _nearest_strike(table, target, opt,
                                        exclude=used_strikes)
                if res is None:
                    return None
            strike, slot = res
            used_strikes.add(strike)
            legs.append(_make_leg(role, strike, opt, side, lots, slot))
        bundle = MultiLegBundle(
            bundle_id="",
            structure_class=STRUCTURE_IRON_CONDOR,
            legs=legs,
            target_credit_pct=self.target_credit_pct,
            stop_loss_pct=self.stop_loss_pct,
            regime_tag_at_entry=dict(regime_tag or {}),
        )
        net_credit_per_lot = bundle.estimated_combined_premium()
        bundle.net_credit_at_entry = net_credit_per_lot
        # Max loss = wing width × lots (defined-risk structure).
        wing_width_rupees = abs(wing_ce_target - body_ce_target)
        bundle.max_loss_estimate = wing_width_rupees * lots
        bundle.max_profit_estimate = max(0.0, net_credit_per_lot * lots)
        bundle.notes.append(
            f"iron_condor body={int(body_ce_target)}/{int(body_pe_target)} "
            f"wings={int(wing_ce_target)}/{int(wing_pe_target)} "
            f"net_credit≈₹{net_credit_per_lot:.0f}/lot")
        return bundle


@dataclass
class JadeLizard:
    """Sell OTM PE + ATM CE; buy further-OTM CE wing.

    Asymmetric structure — defined risk on the upside (covered by the
    long CE wing), undefined on the downside (the short PE). Best used
    when the floor looks reasonably firm but the top is indecisive.
    """
    short_pe_offset_pct: float = 0.012
    wing_offset_pct: float = 0.012
    target_credit_pct: float = 0.50
    stop_loss_pct: float = 1.50

    def construct(self, *, spot: float,
                     slot_readings: List[Any],
                     lots: int = 1,
                     regime_tag: Optional[Dict[str, Any]] = None,
                     ) -> Optional[MultiLegBundle]:
        if spot <= 0 or lots <= 0:
            return None
        table = _build_strike_table(slot_readings)
        atm_ce = _atm_strike(table, spot, OPT_CE)
        wing_ce_target = spot * (1 + self.wing_offset_pct)
        short_pe_target = spot * (1 - self.short_pe_offset_pct)
        if atm_ce is None:
            return None
        picks = [
            ("short_atm_ce", atm_ce, OPT_CE, LEG_SHORT),
            ("short_otm_pe", short_pe_target, OPT_PE, LEG_SHORT),
            ("long_wing_ce", wing_ce_target, OPT_CE, LEG_LONG),
        ]
        legs: List[LegSpec] = []
        for role, target, opt, side in picks:
            res = _nearest_strike(table, target, opt)
            if res is None:
                return None
            strike, slot = res
            legs.append(_make_leg(role, strike, opt, side, lots, slot))
        bundle = MultiLegBundle(
            bundle_id="",
            structure_class=STRUCTURE_JADE_LIZARD,
            legs=legs,
            target_credit_pct=self.target_credit_pct,
            stop_loss_pct=self.stop_loss_pct,
            regime_tag_at_entry=dict(regime_tag or {}),
        )
        net_credit_per_lot = bundle.estimated_combined_premium()
        bundle.net_credit_at_entry = net_credit_per_lot
        # Upside risk is defined: wing distance × lots minus credit.
        wing_distance = abs(wing_ce_target - atm_ce)
        bundle.max_loss_estimate = max(
            0.0, (wing_distance - max(0.0, net_credit_per_lot)) * lots)
        bundle.max_profit_estimate = max(0.0, net_credit_per_lot * lots)
        bundle.notes.append(
            f"jade_lizard short_ce={int(atm_ce)} short_pe={int(short_pe_target)} "
            f"wing={int(wing_ce_target)} net_credit≈₹{net_credit_per_lot:.0f}/lot")
        return bundle


@dataclass
class Butterfly:
    """Long ATM-wing, short 2× ATM, long ATM+wing — pinning play.

    Wins maximally when spot pins the ATM strike at expiry. The
    structure is symmetric in CE space by default; PE-side butterflies
    are constructed by passing ``option_type=OPT_PE``.
    """
    wing_width_pct: float = 0.008
    target_credit_pct: float = 0.50
    stop_loss_pct: float = 1.50

    def construct(self, *, spot: float,
                     slot_readings: List[Any],
                     lots: int = 1,
                     option_type: str = OPT_CE,
                     regime_tag: Optional[Dict[str, Any]] = None,
                     ) -> Optional[MultiLegBundle]:
        if spot <= 0 or lots <= 0:
            return None
        if option_type not in (OPT_CE, OPT_PE):
            return None
        table = _build_strike_table(slot_readings)
        atm = _atm_strike(table, spot, option_type)
        if atm is None:
            return None
        wing_up_target = atm + spot * self.wing_width_pct
        wing_dn_target = atm - spot * self.wing_width_pct
        picks = [
            ("long_lower_wing", wing_dn_target, option_type, LEG_LONG),
            ("short_atm_body", atm, option_type, LEG_SHORT),
            ("long_upper_wing", wing_up_target, option_type, LEG_LONG),
        ]
        legs: List[LegSpec] = []
        for role, target, opt, side in picks:
            res = _nearest_strike(table, target, opt)
            if res is None:
                return None
            strike, slot = res
            # Body is 2× the wing lots to balance.
            leg_lots = lots * 2 if role == "short_atm_body" else lots
            legs.append(_make_leg(role, strike, opt, side, leg_lots, slot))
        bundle = MultiLegBundle(
            bundle_id="",
            structure_class=STRUCTURE_BUTTERFLY,
            legs=legs,
            target_credit_pct=self.target_credit_pct,
            stop_loss_pct=self.stop_loss_pct,
            regime_tag_at_entry=dict(regime_tag or {}),
        )
        bundle.net_credit_at_entry = bundle.estimated_combined_premium()
        # Max loss for a long butterfly = the net debit paid (negative
        # net credit), per lot.
        bundle.max_loss_estimate = max(0.0, -bundle.net_credit_at_entry * lots)
        # Max profit = wing distance − debit, per lot.
        bundle.max_profit_estimate = max(
            0.0, (spot * self.wing_width_pct
                  + bundle.net_credit_at_entry) * lots)
        bundle.notes.append(
            f"butterfly({option_type}) body={int(atm)} "
            f"wings={int(wing_dn_target)}/{int(wing_up_target)}")
        return bundle


@dataclass
class Strangle:
    """Long OTM CE + long OTM PE — long volatility."""
    leg_offset_pct: float = 0.012
    target_credit_pct: float = 0.50   # not used for long structures, kept for shape uniformity
    stop_loss_pct: float = 1.50

    def construct(self, *, spot: float,
                     slot_readings: List[Any],
                     lots: int = 1,
                     short: bool = False,
                     regime_tag: Optional[Dict[str, Any]] = None,
                     ) -> Optional[MultiLegBundle]:
        if spot <= 0 or lots <= 0:
            return None
        side = LEG_SHORT if short else LEG_LONG
        table = _build_strike_table(slot_readings)
        ce_target = spot * (1 + self.leg_offset_pct)
        pe_target = spot * (1 - self.leg_offset_pct)
        picks = [
            (("short" if short else "long") + "_otm_ce", ce_target, OPT_CE, side),
            (("short" if short else "long") + "_otm_pe", pe_target, OPT_PE, side),
        ]
        legs: List[LegSpec] = []
        for role, target, opt, sd in picks:
            res = _nearest_strike(table, target, opt)
            if res is None:
                return None
            strike, slot = res
            legs.append(_make_leg(role, strike, opt, sd, lots, slot))
        bundle = MultiLegBundle(
            bundle_id="",
            structure_class=STRUCTURE_STRANGLE,
            legs=legs,
            target_credit_pct=self.target_credit_pct,
            stop_loss_pct=self.stop_loss_pct,
            regime_tag_at_entry=dict(regime_tag or {}),
        )
        bundle.net_credit_at_entry = bundle.estimated_combined_premium()
        # Long strangle: max loss is the debit (per lot, both legs); max
        # profit is theoretically unbounded so we record None-equivalent
        # via a sentinel large number.
        if short:
            bundle.max_loss_estimate = float("inf")
            bundle.max_profit_estimate = max(
                0.0, bundle.net_credit_at_entry * lots)
        else:
            bundle.max_loss_estimate = max(
                0.0, -bundle.net_credit_at_entry * lots)
            bundle.max_profit_estimate = float("inf")
        kind = "short" if short else "long"
        bundle.notes.append(
            f"{kind}_strangle ce={int(ce_target)} pe={int(pe_target)}")
        return bundle


@dataclass
class VerticalSpread:
    """Directional debit/credit spread — two legs, same option_type."""
    width_pct: float = 0.010
    target_credit_pct: float = 0.50
    stop_loss_pct: float = 1.50

    def construct(self, *, spot: float,
                     slot_readings: List[Any],
                     direction: int,         # +1 = bull, -1 = bear
                     option_type: str = OPT_CE,
                     lots: int = 1,
                     credit: bool = False,
                     regime_tag: Optional[Dict[str, Any]] = None,
                     ) -> Optional[MultiLegBundle]:
        if spot <= 0 or lots <= 0 or direction == 0:
            return None
        if option_type not in (OPT_CE, OPT_PE):
            return None
        table = _build_strike_table(slot_readings)
        atm = _atm_strike(table, spot, option_type)
        if atm is None:
            return None
        far = atm + (spot * self.width_pct) * (1 if direction > 0 else -1)
        # Bull call debit: long ATM, short far OTM (above).
        # Bull put credit: short ATM, long far OTM (below).
        if direction > 0 and not credit:
            picks = [("long_atm", atm, option_type, LEG_LONG),
                     ("short_far", far, option_type, LEG_SHORT)]
        elif direction > 0 and credit:
            picks = [("short_atm", atm, option_type, LEG_SHORT),
                     ("long_far", far, option_type, LEG_LONG)]
        elif direction < 0 and not credit:
            picks = [("long_atm", atm, option_type, LEG_LONG),
                     ("short_far", far, option_type, LEG_SHORT)]
        else:
            picks = [("short_atm", atm, option_type, LEG_SHORT),
                     ("long_far", far, option_type, LEG_LONG)]
        legs: List[LegSpec] = []
        for role, target, opt, side in picks:
            res = _nearest_strike(table, target, opt)
            if res is None:
                return None
            strike, slot = res
            legs.append(_make_leg(role, strike, opt, side, lots, slot))
        bundle = MultiLegBundle(
            bundle_id="",
            structure_class=STRUCTURE_VERTICAL_SPREAD,
            legs=legs,
            target_credit_pct=self.target_credit_pct,
            stop_loss_pct=self.stop_loss_pct,
            regime_tag_at_entry=dict(regime_tag or {}),
        )
        bundle.net_credit_at_entry = bundle.estimated_combined_premium()
        width_rupees = abs(spot * self.width_pct)
        bundle.max_loss_estimate = max(
            0.0, (width_rupees if credit else
                  -bundle.net_credit_at_entry) * lots)
        bundle.max_profit_estimate = max(
            0.0, (bundle.net_credit_at_entry if credit else
                  width_rupees + bundle.net_credit_at_entry) * lots)
        bundle.notes.append(
            f"vertical_{option_type}_{'credit' if credit else 'debit'} "
            f"atm={int(atm)} far={int(far)}")
        return bundle


# ── The composite constructor ─────────────────────────────────────


class StructureConstructor:
    """Single entry point the manager calls.

    Routes to the right per-structure builder based on
    ``strategy_class``. Returns None when the structure cannot be built
    (no slot readings, missing strikes, invalid args) — the manager
    then falls back to single-leg or refuses.
    """

    def __init__(self) -> None:
        self.iron_condor = IronCondor()
        self.jade_lizard = JadeLizard()
        self.butterfly = Butterfly()
        self.strangle = Strangle()
        self.vertical_spread = VerticalSpread()

    def construct(self, *,
                     strategy_class: str,
                     spot: float,
                     slot_readings: List[Any],
                     lots: int = 1,
                     direction: int = 0,
                     option_type: str = OPT_CE,
                     regime_tag: Optional[Dict[str, Any]] = None,
                     short: bool = False,
                     credit: bool = False,
                     ) -> Optional[MultiLegBundle]:
        sc = strategy_class
        kw = dict(spot=spot, slot_readings=slot_readings, lots=lots,
                   regime_tag=regime_tag)
        if sc == STRUCTURE_IRON_CONDOR:
            return self.iron_condor.construct(**kw)
        if sc == STRUCTURE_JADE_LIZARD:
            return self.jade_lizard.construct(**kw)
        if sc == STRUCTURE_BUTTERFLY:
            return self.butterfly.construct(option_type=option_type, **kw)
        if sc == STRUCTURE_STRANGLE:
            return self.strangle.construct(short=short, **kw)
        if sc == STRUCTURE_VERTICAL_SPREAD:
            return self.vertical_spread.construct(
                direction=direction, option_type=option_type, credit=credit,
                **kw)
        return None

    def is_multi_leg(self, strategy_class: str) -> bool:
        return strategy_class in (
            STRUCTURE_IRON_CONDOR, STRUCTURE_JADE_LIZARD,
            STRUCTURE_BUTTERFLY, STRUCTURE_STRANGLE,
            STRUCTURE_VERTICAL_SPREAD,
        )
