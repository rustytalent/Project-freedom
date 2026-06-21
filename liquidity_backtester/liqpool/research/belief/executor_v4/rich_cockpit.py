"""Rich terminal cockpit — beautiful ANSI-rendered live view.

This is the "look at the screen and feel I will win" deliverable.
Stdlib only — no external dependencies. Renders the CockpitSnapshot
into a multi-panel terminal layout with color-coded states, progress
bars, and live updates.

Layout:

  ╔═══════════════ HEADER: timestamp, bar, action ═══════════════╗
  ║  Tick @ 2026-06-23 10:42:15  bar 1234  +₹1,250 today        ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  ACTION CARD       │  P&L                                    ║
  ║  ✓ OPEN CE_ATM     │  Daily:  +₹ 1,250                       ║
  ║    score 0.74      │  Fees:      ₹   140                     ║
  ║    1.8x edge       │  Net:    +₹ 1,110                       ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  PROBABILITY WEB                                             ║
  ║  Consensus: +0.42 ████████░░░ bullish                         ║
  ║  Tail:       0.08 ░░░░░░░░░░░ low                              ║
  ║  Chop:       0.15 ██░░░░░░░░░ low                              ║
  ║                                                              ║
  ║  Top scenarios:                                              ║
  ║    bull_continuation         0.32 →+1 directional             ║
  ║    shakeout_then_bull        0.18 →+1 directional             ║
  ║    chop_range_bound          0.15 → 0 chop                    ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  MM MIND  ── accumulating 72% ── bias +1, vol expansion      ║
  ║  > Bias long; avoid shorts into absorption                   ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  RISK / GREEKS    Δ=+14 ✓  V=-29 ⚠  Γ=0.34 ✓  Θ=-₹85/d        ║
  ║  At risk ₹6,500 / ₹25,000 (26%)   DD 0.4R                    ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  OPEN POSITIONS (3)                                          ║
  ║  CE_ATM  +1L @ ₹98 → ₹105 (+0.65R) [12b] mm_intent_mimicry   ║
  ║  CE_OTM2 +1L @ ₹52 → ₹49 (-0.30R) [ 4b] single_leg           ║
  ║  PE_OTM3 +1L @ ₹28 → ₹31 (+0.18R) [ 8b] anti_crowd_contrarian║
  ╠══════════════════════════════════════════════════════════════╣
  ║  PATTERNS: accumulation 0.65 (15b) ⟂ slow_grind_bull 0.42    ║
  ╚══════════════════════════════════════════════════════════════╝

Used by:
  * V4Runner.evaluate(...) → followed by `print(render_rich_cockpit(snap))`
  * The Monday launcher's --rich-tui mode
  * Sentinel's terminal panel (replaces the third-rate Codex UI)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ── ANSI color helpers ────────────────────────────────────────────


_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

# Foreground colors
FG_RED = "\033[31m"
FG_GREEN = "\033[32m"
FG_YELLOW = "\033[33m"
FG_BLUE = "\033[34m"
FG_MAGENTA = "\033[35m"
FG_CYAN = "\033[36m"
FG_WHITE = "\033[37m"
FG_GRAY = "\033[90m"
FG_BRIGHT_GREEN = "\033[92m"
FG_BRIGHT_RED = "\033[91m"
FG_BRIGHT_YELLOW = "\033[93m"

# Background colors
BG_BLACK = "\033[40m"


@dataclass
class CockpitRenderConfig:
    """Layout knobs."""
    width: int = 78
    enable_color: bool = True
    show_explainer_text: bool = False    # leave the bare text out (already in panels)
    bar_width: int = 18                   # progress-bar character width


def _strip(text: str) -> str:
    """Visible-character length (skip ANSI escapes)."""
    out = []
    in_escape = False
    for c in text:
        if c == "\033":
            in_escape = True
            continue
        if in_escape:
            if c == "m":
                in_escape = False
            continue
        out.append(c)
    return "".join(out)


def _wrap(text: str, color: str, *, cfg: CockpitRenderConfig) -> str:
    if not cfg.enable_color:
        return text
    return f"{color}{text}{_RESET}"


def _pad(text: str, width: int, *, fill: str = " ") -> str:
    """Pad/truncate text to width, accounting for invisible ANSI escapes."""
    visible = _strip(text)
    if len(visible) >= width:
        # Truncate carefully
        out = []
        consumed = 0
        in_escape = False
        for c in text:
            if c == "\033":
                in_escape = True
            if not in_escape and consumed >= width:
                break
            out.append(c)
            if in_escape and c == "m":
                in_escape = False
                continue
            if not in_escape:
                consumed += 1
        result = "".join(out)
        # Only emit a terminator reset if the truncated text contains
        # an ANSI escape (otherwise we'd leak `\033[0m` into plain output).
        if "\033" in result:
            result += _RESET
        return result
    return text + fill * (width - len(visible))


def _bar(value: float, *, low: float = 0.0, high: float = 1.0,
          width: int = 18, fill_char: str = "█",
          empty_char: str = "░") -> str:
    """Simple progress bar."""
    if high <= low:
        return empty_char * width
    frac = max(0.0, min(1.0, (value - low) / (high - low)))
    n_fill = int(round(frac * width))
    return fill_char * n_fill + empty_char * (width - n_fill)


def _signed_bar(value: float, *, low: float = -1.0, high: float = 1.0,
                 width: int = 18, fill_char: str = "█",
                 empty_char: str = "░") -> str:
    """Signed progress bar: centre is 0; left half = negative, right half = positive."""
    half = width // 2
    if value >= 0:
        n_fill = int(round(min(1.0, value / max(high, 1e-6)) * half))
        return (" " * half) + fill_char * n_fill + empty_char * (half - n_fill)
    else:
        n_fill = int(round(min(1.0, -value / max(-low, 1e-6)) * half))
        return empty_char * (half - n_fill) + fill_char * n_fill + (" " * half)


# ── Color helpers based on value semantics ────────────────────────


def _color_for_pnl(value: float, *, cfg: CockpitRenderConfig) -> str:
    if not cfg.enable_color:
        return ""
    if value > 0:
        return FG_BRIGHT_GREEN
    if value < 0:
        return FG_BRIGHT_RED
    return FG_GRAY


def _color_for_action(kind: str, *, cfg: CockpitRenderConfig) -> str:
    if not cfg.enable_color:
        return ""
    return {
        "ENTER": FG_BRIGHT_GREEN,
        "EXIT": FG_BRIGHT_YELLOW,
        "REFUSE": FG_BRIGHT_RED,
        "HOLD": FG_GRAY,
    }.get(kind, FG_WHITE)


def _color_for_tail_action(action: str, *, cfg: CockpitRenderConfig) -> str:
    if not cfg.enable_color:
        return ""
    return {
        "SCALE_UP": FG_BRIGHT_GREEN,
        "NORMAL": FG_GREEN,
        "HEDGE": FG_YELLOW,
        "REFUSE": FG_BRIGHT_RED,
    }.get(action, FG_WHITE)


def _color_for_signed(value: float, *, cfg: CockpitRenderConfig,
                        zero_color: str = "") -> str:
    if not cfg.enable_color:
        return ""
    if value > 0.05:
        return FG_BRIGHT_GREEN
    if value < -0.05:
        return FG_BRIGHT_RED
    return zero_color or FG_GRAY


# ── Box drawing primitives ────────────────────────────────────────


def _hline(width: int, char: str = "═") -> str:
    return char * width


def _box_top(width: int) -> str:
    return "╔" + "═" * (width - 2) + "╗"


def _box_bottom(width: int) -> str:
    return "╚" + "═" * (width - 2) + "╝"


def _box_sep(width: int) -> str:
    return "╠" + "═" * (width - 2) + "╣"


def _box_line(content: str, width: int) -> str:
    return "║" + _pad(content, width - 2) + "║"


# ── Section renderers ────────────────────────────────────────────


def _render_header(snapshot: Dict[str, Any], *, cfg: CockpitRenderConfig
                    ) -> List[str]:
    ts = snapshot.get("ts", "")
    bar = snapshot.get("bar_index", 0)
    pnl = (snapshot.get("pnl_panel") or {}).get("daily_pnl_rupees", 0.0)
    pnl_color = _color_for_pnl(pnl, cfg=cfg)
    title = _wrap(f"PREMIUM BELIEF v4", FG_CYAN + _BOLD, cfg=cfg)
    pnl_text = _wrap(f"₹{pnl:+,.0f} today", pnl_color, cfg=cfg)
    line = f" {title}  bar {bar}  {ts[:19]}  ─  {pnl_text}"
    return [_box_top(cfg.width), _box_line(line, cfg.width),
             _box_sep(cfg.width)]


def _render_action_card(snapshot: Dict[str, Any], *, cfg: CockpitRenderConfig
                          ) -> List[str]:
    card = snapshot.get("action_card") or {}
    kind = card.get("kind", "HOLD")
    headline = card.get("headline", "")
    color = _color_for_action(kind, cfg=cfg)
    lines: List[str] = []
    lines.append(_box_line("  " + _wrap("ACTION", FG_CYAN + _BOLD, cfg=cfg),
                            cfg.width))
    symbol = {"ENTER": "✓", "EXIT": "✗", "REFUSE": "⊘", "HOLD": "·"}.get(kind, "·")
    lines.append(_box_line(f"  {_wrap(symbol, color, cfg=cfg)} "
                              f"{_wrap(headline, color, cfg=cfg)}", cfg.width))
    if kind == "ENTER":
        score = card.get("aggregator_score")
        size_mult = card.get("size_multiplier")
        if score is not None and size_mult is not None:
            lines.append(_box_line(
                f"    score {score:.2f} | size×{size_mult:.2f}",
                cfg.width))
        sp = card.get("stop_premium")
        tp = card.get("target_premium")
        if sp is not None and tp is not None:
            lines.append(_box_line(
                f"    stop ₹{sp:.2f}  target ₹{tp:.2f}", cfg.width))
    elif kind == "EXIT":
        rupees = card.get("realized_rupees", 0.0)
        rcolor = _color_for_pnl(rupees, cfg=cfg)
        lines.append(_box_line(
            f"    realized {_wrap(f'₹{rupees:+,.0f}', rcolor, cfg=cfg)} "
            f"({card.get('realized_r', 0):+.2f}R) "
            f"[{card.get('bars_held', 0)}b]", cfg.width))
        reason = (card.get("exit_reason") or "")[:60]
        lines.append(_box_line(f"    reason: {reason}", cfg.width))
    elif kind == "REFUSE":
        for r in (card.get("all_reasons") or [])[:2]:
            lines.append(_box_line(f"    → {r[:60]}", cfg.width))
    lines.append(_box_sep(cfg.width))
    return lines


def _render_web_panel(snapshot: Dict[str, Any], *, cfg: CockpitRenderConfig
                       ) -> List[str]:
    web = snapshot.get("web_panel") or {}
    lines: List[str] = []
    lines.append(_box_line(
        "  " + _wrap("PROBABILITY WEB", FG_CYAN + _BOLD, cfg=cfg),
        cfg.width))
    consensus = float(web.get("directional_consensus", 0.0))
    tail = float(web.get("tail_mass", 0.0))
    chop = float(web.get("chop_mass", 0.0))
    cons_bar = _signed_bar(consensus, width=cfg.bar_width)
    tail_bar = _bar(tail, high=0.5, width=cfg.bar_width)
    chop_bar = _bar(chop, high=0.8, width=cfg.bar_width)
    cons_color = _color_for_signed(consensus, cfg=cfg)
    tail_color = (FG_BRIGHT_RED if tail >= 0.30
                  else FG_YELLOW if tail >= 0.15
                  else FG_GREEN) if cfg.enable_color else ""
    chop_color = (FG_YELLOW if chop >= 0.40 else FG_GREEN) \
                  if cfg.enable_color else ""
    cons_label = ("bullish" if consensus > 0.20
                   else "bearish" if consensus < -0.20 else "neutral")
    lines.append(_box_line(
        f"  Consensus: {consensus:+5.2f} {_wrap(cons_bar, cons_color, cfg=cfg)} {cons_label}",
        cfg.width))
    lines.append(_box_line(
        f"  Tail:       {tail:4.2f} {_wrap(tail_bar, tail_color, cfg=cfg)}",
        cfg.width))
    lines.append(_box_line(
        f"  Chop:       {chop:4.2f} {_wrap(chop_bar, chop_color, cfg=cfg)}",
        cfg.width))
    lines.append(_box_line("", cfg.width))
    top = web.get("top_scenarios") or []
    if top:
        lines.append(_box_line(
            "  " + _wrap("Top scenarios:", _BOLD, cfg=cfg), cfg.width))
        for sc in top[:5]:
            name = sc.get("name", "")
            prob = float(sc.get("probability", 0.0))
            fam = sc.get("family", "")
            direction = sc.get("direction", 0)
            arrow = ("↑" if direction > 0 else "↓" if direction < 0 else "·")
            arrow_color = _color_for_signed(direction, cfg=cfg)
            lines.append(_box_line(
                f"    {name:30s} {prob:5.2f} "
                f"{_wrap(arrow, arrow_color, cfg=cfg)} {fam}",
                cfg.width))
    lines.append(_box_sep(cfg.width))
    return lines


def _render_mm_panel(snapshot: Dict[str, Any], *, cfg: CockpitRenderConfig
                      ) -> List[str]:
    mm = snapshot.get("mm_panel") or {}
    crowd = snapshot.get("crowd_panel") or {}
    lines: List[str] = []
    intent = mm.get("dominant_intent", "neutral_inventory")
    prob = float(mm.get("dominant_probability", 0.0))
    bias = mm.get("implied_bias", 0)
    vol_view = mm.get("implied_volatility_view", "neutral")
    bias_arrow = "↑" if bias > 0 else "↓" if bias < 0 else "·"
    bias_color = _color_for_signed(bias, cfg=cfg)
    lines.append(_box_line(
        "  " + _wrap("MM MIND", FG_CYAN + _BOLD, cfg=cfg) +
        f"  {intent} @ {prob:.0%}  bias {_wrap(bias_arrow, bias_color, cfg=cfg)} "
        f" vol: {vol_view}",
        cfg.width))
    guidance = mm.get("operator_guidance", "")
    if guidance:
        lines.append(_box_line(
            "  " + _wrap("> ", FG_GRAY, cfg=cfg) + guidance[:cfg.width - 6],
            cfg.width))
    # Crowd line
    if crowd:
        retail = crowd.get("we_look_like_retail", False)
        score = float(crowd.get("retail_similarity_score", 0.0))
        retail_label = ("YES" if retail else "no")
        retail_color = (FG_BRIGHT_RED if retail
                        else FG_GREEN) if cfg.enable_color else ""
        lines.append(_box_line(
            f"  CROWD MIRROR  retail: "
            f"{_wrap(retail_label, retail_color, cfg=cfg)}  "
            f"score {score:.2f}",
            cfg.width))
    lines.append(_box_sep(cfg.width))
    return lines


def _render_fat_tail(snapshot: Dict[str, Any], *, cfg: CockpitRenderConfig
                       ) -> List[str]:
    dial = snapshot.get("fat_tail_dial") or {}
    lines: List[str] = []
    score = float(dial.get("tail_score", 0.0))
    action = dial.get("action", "NORMAL")
    color = _color_for_tail_action(action, cfg=cfg)
    bar = _bar(score, high=1.0, width=cfg.bar_width)
    lines.append(_box_line(
        "  " + _wrap("FAT-TAIL", FG_CYAN + _BOLD, cfg=cfg) +
        f"   {score:.2f} {_wrap(bar, color, cfg=cfg)} {_wrap(action, color, cfg=cfg)}",
        cfg.width))
    lines.append(_box_sep(cfg.width))
    return lines


def _render_risk_panel(snapshot: Dict[str, Any], *, cfg: CockpitRenderConfig
                         ) -> List[str]:
    risk = snapshot.get("risk_panel") or {}
    lines: List[str] = []
    delta = float(risk.get("net_delta", 0.0))
    vega = float(risk.get("net_vega", 0.0))
    gamma = float(risk.get("net_gamma", 0.0))
    theta = float(risk.get("net_theta", 0.0))
    par = float(risk.get("total_premium_at_risk_rupees", 0.0))
    dd = float(risk.get("portfolio_drawdown_r", 0.0))
    par_color = (FG_BRIGHT_RED if par >= 20000
                  else FG_YELLOW if par >= 12000
                  else FG_GREEN) if cfg.enable_color else ""
    dd_color = (FG_BRIGHT_RED if dd >= 2.0
                  else FG_YELLOW if dd >= 1.0
                  else FG_GREEN) if cfg.enable_color else ""
    lines.append(_box_line(
        "  " + _wrap("RISK / GREEKS", FG_CYAN + _BOLD, cfg=cfg) +
        f"   Δ={delta:+.1f}  V={vega:+.1f}  Γ={gamma:+.3f}  Θ={theta:+.1f}/d",
        cfg.width))
    lines.append(_box_line(
        f"  premium at risk {_wrap(f'₹{par:,.0f}', par_color, cfg=cfg)}  "
        f"DD {_wrap(f'{dd:.1f}R', dd_color, cfg=cfg)}",
        cfg.width))
    kills = risk.get("kill_switches") or []
    if kills:
        for k in kills[:2]:
            lines.append(_box_line(
                "  " + _wrap("⚠ KILL: " + k[:cfg.width - 12],
                              FG_BRIGHT_RED + _BOLD, cfg=cfg),
                cfg.width))
    lines.append(_box_sep(cfg.width))
    return lines


def _render_positions(snapshot: Dict[str, Any], *, cfg: CockpitRenderConfig
                       ) -> List[str]:
    pos = snapshot.get("positions_panel") or {}
    lines: List[str] = []
    n = int(pos.get("n_open", 0))
    lines.append(_box_line(
        "  " + _wrap(f"OPEN POSITIONS ({n})", FG_CYAN + _BOLD, cfg=cfg),
        cfg.width))
    for p in (pos.get("open") or [])[:6]:
        label = p.get("contract_label", "?")
        direction = p.get("direction", 1)
        size = p.get("size_lots", 0)
        best_r = float(p.get("best_r", 0.0))
        worst_r = float(p.get("worst_r", 0.0))
        last_premium = float(p.get("last_premium", 0.0))
        # We don't have current R in summary; approximate from best/worst
        cur_r = best_r if best_r != 0 else worst_r
        cur_color = _color_for_signed(cur_r, cfg=cfg)
        direction_sym = "+" if direction > 0 else "-"
        lines.append(_box_line(
            f"  {label:10s} {direction_sym}{size}L "
            f"→ ₹{last_premium:6.2f}  "
            f"best {_wrap(f'{best_r:+.2f}R', _color_for_signed(best_r, cfg=cfg), cfg=cfg)}  "
            f"worst {_wrap(f'{worst_r:+.2f}R', _color_for_signed(-worst_r, cfg=cfg), cfg=cfg)}",
            cfg.width))
    lines.append(_box_sep(cfg.width))
    return lines


def _render_patterns(snapshot: Dict[str, Any], *, cfg: CockpitRenderConfig
                       ) -> List[str]:
    p = snapshot.get("patterns_panel") or {}
    lines: List[str] = []
    n = int(p.get("n_active", 0))
    if n == 0:
        lines.append(_box_line(
            "  " + _wrap("PATTERNS", FG_CYAN + _BOLD, cfg=cfg) + "   none active",
            cfg.width))
    else:
        for pat in (p.get("patterns") or [])[:4]:
            name = pat.get("name", "?")
            conf = float(pat.get("confidence", 0.0))
            intent = pat.get("implied_mm_intent", "")
            conf_color = (FG_BRIGHT_YELLOW if conf >= 0.65
                           else FG_YELLOW if conf >= 0.40
                           else FG_GRAY) if cfg.enable_color else ""
            lines.append(_box_line(
                f"  ⟂ {name:24s} "
                f"{_wrap(f'conf {conf:.2f}', conf_color, cfg=cfg)} "
                f"intent: {intent}",
                cfg.width))
    return lines


def _render_pnl_panel(snapshot: Dict[str, Any], *,
                        cfg: CockpitRenderConfig) -> List[str]:
    pnl = snapshot.get("pnl_panel") or {}
    lines: List[str] = []
    daily = float(pnl.get("daily_pnl_rupees", 0.0))
    fees = float(pnl.get("cumulative_fees_rupees", 0.0))
    net = daily   # daily_pnl is already net by construction in the manager
    win_rate = pnl.get("win_rate")
    daily_color = _color_for_pnl(daily, cfg=cfg)
    lines.append(_box_sep(cfg.width))
    msg = (f"  P&L: {_wrap(f'₹{daily:+,.0f}', daily_color, cfg=cfg)} "
           f"(fees ₹{fees:,.0f})")
    if win_rate is not None:
        msg += f"  win rate {float(win_rate):.0%}"
    lines.append(_box_line(msg, cfg.width))
    return lines


# ── Top-level renderer ────────────────────────────────────────────


def render_rich_cockpit(cockpit_dict: Dict[str, Any],
                          cfg: Optional[CockpitRenderConfig] = None,
                          ) -> str:
    """Render a CockpitSnapshot.to_dict() as a multi-panel string.

    The output is a multi-line string with ANSI color codes. The caller
    can print() it directly to a terminal or strip the colors for plain
    text.
    """
    cfg = cfg or CockpitRenderConfig()
    out: List[str] = []
    out.extend(_render_header(cockpit_dict, cfg=cfg))
    out.extend(_render_action_card(cockpit_dict, cfg=cfg))
    out.extend(_render_web_panel(cockpit_dict, cfg=cfg))
    out.extend(_render_mm_panel(cockpit_dict, cfg=cfg))
    out.extend(_render_fat_tail(cockpit_dict, cfg=cfg))
    out.extend(_render_risk_panel(cockpit_dict, cfg=cfg))
    out.extend(_render_positions(cockpit_dict, cfg=cfg))
    out.extend(_render_patterns(cockpit_dict, cfg=cfg))
    out.extend(_render_pnl_panel(cockpit_dict, cfg=cfg))
    out.append(_box_bottom(cfg.width))
    return "\n".join(out)


def render_plain_cockpit(cockpit_dict: Dict[str, Any],
                           width: int = 78) -> str:
    """Render without colors — for log files / non-TTY emit."""
    cfg = CockpitRenderConfig(width=width, enable_color=False)
    return render_rich_cockpit(cockpit_dict, cfg=cfg)
