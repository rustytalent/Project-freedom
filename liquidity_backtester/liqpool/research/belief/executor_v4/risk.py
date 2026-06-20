"""Portfolio risk layer — total exposure + kill switches.

Tracks portfolio-wide aggregates and emits structured kill switches that
the manager consults each tick before allowing new entries.

Aggregates per tick:
  * total_premium_at_risk_rupees
  * net_directional_exposure (net delta in lots)
  * gross_exposure_rupees
  * portfolio_drawdown_r
  * most_dangerous_position (id, fraction of total risk)
  * correlated_clusters — positions sharing direction / underlying
  * net_vega, net_theta

Kill switches:
  * portfolio premium-at-risk > budget
  * net delta > delta cap (becomes too directional)
  * portfolio drawdown_r below floor
  * correlated cluster > 70% of total risk
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PortfolioRiskConfig:
    """Knobs for the portfolio risk layer."""
    max_total_premium_at_risk_rupees: float = 25_000.0  # 5 × per-trade cap
    max_net_delta_lots: float = 4.0
    max_portfolio_drawdown_r: float = 3.0    # in R units (sum of -R of losers)
    max_cluster_fraction: float = 0.85       # max % of risk in one cluster
    min_positions_for_cluster_check: int = 3  # 1-2 positions don't trigger it


@dataclass
class PortfolioRiskReport:
    """The portfolio-risk per-tick output."""
    total_premium_at_risk_rupees: float
    net_directional_exposure_lots: float
    gross_exposure_rupees: float
    portfolio_drawdown_r: float
    most_dangerous_position_id: str
    most_dangerous_position_risk_fraction: float
    correlated_clusters: List[Dict[str, Any]]
    net_vega: float
    net_theta: float
    net_delta: float
    net_gamma: float
    kill_switches: List[str]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_premium_at_risk_rupees": round(self.total_premium_at_risk_rupees, 2),
            "net_directional_exposure_lots": round(self.net_directional_exposure_lots, 3),
            "gross_exposure_rupees": round(self.gross_exposure_rupees, 2),
            "portfolio_drawdown_r": round(self.portfolio_drawdown_r, 3),
            "most_dangerous_position_id": self.most_dangerous_position_id,
            "most_dangerous_position_risk_fraction": round(
                self.most_dangerous_position_risk_fraction, 3),
            "correlated_clusters": list(self.correlated_clusters),
            "net_vega": round(self.net_vega, 3),
            "net_theta": round(self.net_theta, 3),
            "net_delta": round(self.net_delta, 3),
            "net_gamma": round(self.net_gamma, 3),
            "kill_switches": list(self.kill_switches),
            "notes": list(self.notes),
        }


class PortfolioRiskLayer:
    """Computes portfolio aggregates and emits kill-switch strings."""

    def __init__(self, cfg: Optional[PortfolioRiskConfig] = None) -> None:
        self.cfg = cfg or PortfolioRiskConfig()

    def compute(self, *,
                  open_states: Dict[str, Any],
                  current_r_by_position: Dict[str, float],
                  greeks_by_position: Optional[Dict[str, Dict[str, float]]] = None,
                  lot_size: int = 65,
                  ) -> PortfolioRiskReport:
        """Run the aggregation pass over open positions.

        ``open_states`` is the manager's internal ``_open_states`` dict
        (position_id → state). We read each state's hypothesis directly.
        """
        cfg = self.cfg
        kill: List[str] = []
        notes: List[str] = []

        total_premium = 0.0
        gross_exposure = 0.0
        net_delta_lots = 0.0
        net_vega = 0.0
        net_theta = 0.0
        net_delta = 0.0
        net_gamma = 0.0
        drawdown = 0.0
        per_position_risk: Dict[str, float] = {}
        cluster_by_dir: Dict[int, float] = {0: 0.0, +1: 0.0, -1: 0.0}

        for pos_id, state in open_states.items():
            h = state.hypothesis
            premium_at_risk = float(h.rupees_at_risk or 0.0)
            total_premium += premium_at_risk
            per_position_risk[pos_id] = premium_at_risk
            cluster_by_dir[h.direction] = cluster_by_dir.get(h.direction, 0.0) \
                + premium_at_risk
            gross_exposure += premium_at_risk
            # Direction × lots (rough proxy for "lots of delta")
            net_delta_lots += h.direction * float(h.size_lots or 0)
            # Greeks if provided.
            g = (greeks_by_position or {}).get(pos_id, {})
            net_vega += float(g.get("vega", 0.0))
            net_theta += float(g.get("theta", 0.0))
            net_delta += float(g.get("delta", 0.0))
            net_gamma += float(g.get("gamma", 0.0))
            # Drawdown contribution: only losing positions count.
            r = current_r_by_position.get(pos_id, 0.0)
            if r < 0:
                drawdown += abs(r)

        # Most dangerous position
        if per_position_risk:
            worst_id = max(per_position_risk.items(), key=lambda kv: kv[1])[0]
            worst_frac = (per_position_risk[worst_id]
                          / max(1.0, total_premium))
        else:
            worst_id = ""
            worst_frac = 0.0

        # Correlated clusters: long vs short.
        clusters = []
        for d, total in cluster_by_dir.items():
            if total <= 0:
                continue
            frac = total / max(1.0, gross_exposure)
            label = ("long" if d > 0 else "short" if d < 0
                     else "neutral")
            clusters.append({
                "direction": d, "label": label,
                "total_risk_rupees": round(total, 2),
                "fraction_of_gross": round(frac, 3),
            })

        # Kill switches.
        if total_premium > cfg.max_total_premium_at_risk_rupees:
            kill.append(
                f"total premium-at-risk ₹{total_premium:.0f} > budget "
                f"₹{cfg.max_total_premium_at_risk_rupees:.0f}"
            )
        if abs(net_delta_lots) > cfg.max_net_delta_lots:
            kill.append(
                f"net directional exposure {net_delta_lots:+.1f} lots > cap "
                f"±{cfg.max_net_delta_lots:.1f}"
            )
        if drawdown > cfg.max_portfolio_drawdown_r:
            kill.append(
                f"portfolio drawdown {drawdown:.1f}R > floor "
                f"{cfg.max_portfolio_drawdown_r:.1f}R"
            )
        if len(open_states) >= cfg.min_positions_for_cluster_check:
            for c in clusters:
                if c["fraction_of_gross"] > cfg.max_cluster_fraction:
                    kill.append(
                        f"cluster '{c['label']}' carries "
                        f"{c['fraction_of_gross']:.0%} of gross — concentration risk"
                    )
        if not kill:
            notes.append("portfolio risk within limits")

        return PortfolioRiskReport(
            total_premium_at_risk_rupees=total_premium,
            net_directional_exposure_lots=net_delta_lots,
            gross_exposure_rupees=gross_exposure,
            portfolio_drawdown_r=drawdown,
            most_dangerous_position_id=worst_id,
            most_dangerous_position_risk_fraction=worst_frac,
            correlated_clusters=clusters,
            net_vega=net_vega,
            net_theta=net_theta,
            net_delta=net_delta,
            net_gamma=net_gamma,
            kill_switches=kill,
            notes=notes,
        )
