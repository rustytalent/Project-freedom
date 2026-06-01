"""Configuration dataclasses for detection params, factor weights, and runtime."""
from dataclasses import dataclass, field, asdict
from typing import List, Dict


@dataclass
class DetectionParams:
    # Swing fractal window: a swing high needs `left` lower highs to the left and `right` to the right.
    swing_left: int = 3
    swing_right: int = 3

    # EQH/EQL tolerance as multiple of ATR(atr_period) at the first swing.
    eqhl_tol_atr: float = 0.15
    eqhl_max_bars: int = 240        # max bar distance between the two swings on the source TF
    eqhl_min_touches: int = 2       # 2 = double top/bottom, 3 = triple, etc.

    # ATR settings used everywhere a "small price tolerance" is needed.
    atr_period: int = 14

    # Fair value gap: bullish gap when bar[i-1].high < bar[i+1].low (and the middle bar is the displacement).
    # min gap size as fraction of ATR — filters noise gaps.
    fvg_min_atr: float = 0.25

    # Order block: last opposite-color candle preceding a displacement of `ob_displacement_atr`*ATR.
    ob_displacement_atr: float = 1.5
    ob_lookback: int = 30

    # In-candle imbalance: wick/body asymmetry threshold to flag a candle as a rejection / liquidity grab.
    wick_dominance: float = 0.55    # dominant wick must be >= this fraction of full range
    body_max_ratio: float = 0.35    # body must be <= this fraction of full range to count as rejection

    # Opening Range Breakout: minutes after session open. For US equities, default 15m ORB.
    orb_minutes: int = 15

    # Volume node: rolling window for volume-by-price, top-N price bins flagged as HVN.
    vp_bins: int = 50
    vp_window_bars: int = 390 * 5   # ~5 trading days of 1m == 5d of bars on 1m. Auto-scaled later.
    vp_top_n: int = 4

    # Pool merging: pools whose price ranges overlap (or are within merge_atr*ATR) on the same side
    # are merged into a single zone.
    merge_atr: float = 0.20

    # Default pool half-width in ATRs when a single price level needs a zone.
    pool_halfwidth_atr: float = 0.10


@dataclass
class FactorWeights:
    """Weights applied when scoring a candidate pool. Optimizer tunes these."""
    eqhl: float = 1.0
    prev_day: float = 1.0
    prev_week: float = 1.2
    prev_month: float = 1.4
    fvg: float = 0.7
    order_block: float = 0.9
    in_candle_imbalance: float = 0.5
    volume_node: float = 0.8
    orb_extreme: float = 0.6
    multi_tf_overlap: float = 1.5   # multiplier per additional TF that confirms the zone

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class Config:
    symbol: str = "AAPL"
    base_interval: str = "5m"
    period: str = "60d"                 # used if start/end not given
    start: str | None = None
    end: str | None = None

    # Higher TFs to compute features on. Their pools get projected onto the base 5m chart.
    # Default: 5m base + 15m + 1H + 3H + 1D + 1W (the multi-resolution stack).
    higher_tfs: List[str] = field(default_factory=lambda: ["15min", "60min", "180min", "1D", "1W"])

    # Tester: how far forward (in 5m bars) we evaluate each pool. The tester now classifies into
    # tiers via two symmetric thresholds.
    test_horizon_bars: int = 200

    # MIS-aware label generation. When True (default), tester.test_pools and
    # timing.generate_snapshots cap each label's forward window at the
    # same-IST-trading-day EOD bar (minute <= 15:15 IST). This makes the
    # respect/break/direction/proximity LABELS match what an intraday-MIS
    # trader could actually realise. Set False to revert to legacy
    # swing-style multi-day labels (e.g. for research on swing strategies).
    intraday_session_only: bool = True

    # Two-tier reaction / break thresholds (in ATR units).
    #   weak_reaction_atr  : enough reverse move post-touch to count as a 'mild' respect.
    #   strong_reaction_atr: definitive rejection.
    #   weak_break_atr     : close beyond zone by this much = mild break (could be a wick close).
    #   strong_break_atr   : close beyond zone by this much = decisive break.
    weak_reaction_atr: float = 0.5
    strong_reaction_atr: float = 1.5
    weak_break_atr: float = 0.20
    strong_break_atr: float = 0.50
    # Reaction must occur within this many bars after the first touch. Raised from 20 to 40 so
    # pools formed on higher TFs (1D/1W/3H) have a fair chance to react on the 5m chart —
    # institutional reactions often take 2-3 hours.
    respect_within_bars: int = 40
    # Confirmation: a single bar closing past the zone by >=strong_break_atr can be a stop hunt.
    # Require N consecutive bars closing past before calling it broken_strong. N=2 = "the break
    # held for one more bar" — classical breakout-confirmation principle.
    strong_break_confirm_bars: int = 2
    # If the first strong-break bar is followed by a close BACK INSIDE the zone within this many
    # bars, classify as swept_and_reclaimed (the canonical SMC stop-hunt-then-hold pattern).
    reclaim_within_bars: int = 12

    # Minimum confluence score for a pool to be drawn / tested.
    min_pool_score: float = 1.0

    # Optimizer
    opt_iterations: int = 80
    opt_explore_frac: float = 0.4       # fraction of iterations used for random exploration
    opt_seed: int = 7

    # Financial-ML validation/model controls. Pool labels can overlap because a pool remains
    # live across many future bars, so quality-model validation defaults to purged chronological
    # folds plus an embargo after each validation slice.
    validation_method: str = "purged_embargoed_walk_forward"
    embargo_bars: int = 78
    regularization_preset: str = "default"

    detect: DetectionParams = field(default_factory=DetectionParams)
    weights: FactorWeights = field(default_factory=FactorWeights)
