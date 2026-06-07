"""LightGBM structural priors for the OptionsExpectedReturnModel.

Encodes the cascade DAG (executor doc §3.5) as LightGBM's
``interaction_constraints`` and ``monotone_constraints`` so the
model is structurally prevented from learning implausible
relationships even when the OOS sample is small.

Interaction constraints — a tree split path may only combine
features that share at least one group. A group is a "cluster" of
features that the cascade DAG says can causally interact. Features
NOT sharing a group cannot appear together in the same split path.

Monotone constraints — pin known directional relationships per
side. For BUY: higher IV percentile → lower expected R (premium is
over-priced; mean-reversion risk). For SELL: the sign flips
(richer premium = more theta to collect).

When the CCV-aware features land in D.5, this module's group
template grows; the trainer automatically picks them up if they're
present in the feature frame.
"""
from __future__ import annotations

from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Feature catalogue — the union of what's USABLE as model input
# ---------------------------------------------------------------------------

NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = (
    # Structural distance + DTE + ToD
    "dist_strike_to_spot_atr",
    "abs_dist_pct",
    "dte_trading_days",
    "tod_minutes_since_open",
    "tod_minutes_to_eod",
    # Underlying intraday regime
    "underlying_5m_return",
    "underlying_30m_return",
    "underlying_realized_vol_30m",
    # Lagged daily Greeks
    "delta_yday",
    "gamma_yday",
    "theta_per_day_pct_yday",
    "vega_per_volpoint_pct_yday",
    "iv_yday",
    "iv_dod_change_bps",
    "iv_percentile_60d",
    "iv_rank_60d",
    "pcr_oi_yday",
    # Lagged daily macro
    "india_vix_close_yday",
    "india_vix_dod_change",
    "usdinr_dod_change_bps",
)

CATEGORICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "moneyness_bucket",
    "weekly_expiry_flag",
)

# Moneyness bucket levels — fixed encoding so the one-hot columns
# are deterministic across runs.
MONEYNESS_BUCKET_LEVELS: tuple[str, ...] = (
    "ATM", "ATM_or_near", "1OTM", "2OTM", "3OTM_plus",
)


def expand_categorical_features(frame_columns: Iterable[str]) -> list[str]:
    """Return the post-one-hot feature column names that would appear
    when ``CATEGORICAL_FEATURE_COLUMNS`` are expanded against the
    fixed ``MONEYNESS_BUCKET_LEVELS`` levels.

    The expansion is fixed (does NOT come from the data) so train and
    predict frames produce identical feature vectors even when the
    predict frame contains a moneyness bucket the train frame missed.
    """
    out: list[str] = list(NUMERIC_FEATURE_COLUMNS)
    for level in MONEYNESS_BUCKET_LEVELS:
        out.append(f"moneyness_bucket__{level}")
    out.append("weekly_expiry_flag_int")
    # Preserve the input order for columns we don't recognise — useful
    # when D.5 adds CCV features.
    seen = set(out)
    for c in frame_columns:
        if c not in seen and c not in {
            "moneyness_bucket", "weekly_expiry_flag", "side",
            "tod_bucket", "underlying", "strike", "bar_ts",
            "trading_date_ist", "spot", "atr_underlying",
        }:
            # Only pick up CCV-shaped columns: those starting with
            # macro_/regime_/pool_/options_/micro_/manipulation_ or
            # ending with one of the established alignment suffixes.
            if any(c.startswith(p) for p in (
                "macro_", "regime_", "pool_", "micro_",
                "manipulation_", "options_",
            )) or c.endswith("_alignment") or c.endswith("_flag"):
                out.append(c)
                seen.add(c)
    return out


# ---------------------------------------------------------------------------
# Interaction groups (the cascade DAG)
# ---------------------------------------------------------------------------

# Group names refer to the FINAL (post-one-hot) feature column names.
# A future commit (D.5) can append the CCV cluster without breaking
# this contract — features that don't exist at fit time are simply
# filtered out by ``build_constraints``.
INTERACTION_GROUPS_TEMPLATE: tuple[tuple[str, ...], ...] = (
    # Top-of-cascade: macro
    (
        "india_vix_close_yday",
        "india_vix_dod_change",
        "usdinr_dod_change_bps",
    ),
    # Intraday underlying regime
    (
        "underlying_5m_return",
        "underlying_30m_return",
        "underlying_realized_vol_30m",
    ),
    # Structural distance / DTE / ToD / moneyness
    (
        "dist_strike_to_spot_atr",
        "abs_dist_pct",
        "dte_trading_days",
        "tod_minutes_since_open",
        "tod_minutes_to_eod",
        "moneyness_bucket__ATM",
        "moneyness_bucket__ATM_or_near",
        "moneyness_bucket__1OTM",
        "moneyness_bucket__2OTM",
        "moneyness_bucket__3OTM_plus",
        "weekly_expiry_flag_int",
    ),
    # Options-specific cluster (Greeks + IV)
    (
        "delta_yday", "gamma_yday",
        "theta_per_day_pct_yday", "vega_per_volpoint_pct_yday",
        "iv_yday", "iv_dod_change_bps",
        "iv_percentile_60d", "iv_rank_60d",
        "pcr_oi_yday",
    ),
    # CCV cluster — populated when D.5 ships the cascade adjustments.
    (
        "macro_score", "regime_score", "macro_regime_alignment",
    ),
    (
        "regime_score", "pool_score", "options_score",
        "regime_pool_alignment", "pool_holding_strength",
    ),
    (
        "pool_score", "micro_score", "manipulation_score",
        "manip_micro_alignment", "manip_pool_alignment",
        "micro_forced_flag", "pool_distortion_flag",
        "micro_organic_score", "pool_holding_strength",
    ),
)


# ---------------------------------------------------------------------------
# Monotone constraints
# ---------------------------------------------------------------------------

# BUY side: higher IV is bad (mean reversion), higher delta is good,
# higher 30m return is good (momentum carries into 60m horizon),
# more DTE is good (theta less aggressive), farther OTM is bad.
MONOTONE_BUY: dict[str, int] = {
    "underlying_30m_return": +1,
    "underlying_5m_return": +1,
    "delta_yday": +1,
    "vega_per_volpoint_pct_yday": +1,
    "iv_percentile_60d": -1,
    "iv_rank_60d": -1,
    "abs_dist_pct": -1,
    "dte_trading_days": +1,
    # CCV (D.5)
    "macro_score": +1,
    "micro_organic_score": +1,
    "options_score": +1,
    "micro_forced_flag": -1,
    "pool_distortion_flag": -1,
}

# SELL side: most signs flip. Higher IV is GOOD (rich premium),
# tight DTE is GOOD (theta accrues fast), less directional move is
# good.
MONOTONE_SELL: dict[str, int] = {
    "underlying_30m_return": -1,
    "underlying_5m_return": -1,
    "delta_yday": -1,
    "vega_per_volpoint_pct_yday": -1,
    "iv_percentile_60d": +1,
    "iv_rank_60d": +1,
    "abs_dist_pct": +1,
    "dte_trading_days": -1,
    # CCV (D.5)
    "macro_score": -1,
    "micro_organic_score": -1,
    "options_score": -1,
    "micro_forced_flag": -1,
    "pool_distortion_flag": -1,
}


# ---------------------------------------------------------------------------
# LightGBM hyperparameters
# ---------------------------------------------------------------------------

# Per docs/options_strategy_methodology.md §4.
LIGHTGBM_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 0.9,                # Huber transition quantile
    "num_leaves": 31,
    "min_data_in_leaf": 80,
    "bagging_fraction": 0.8,
    "bagging_freq": 3,
    "feature_fraction": 0.85,
    "lambda_l2": 12.0,
    "learning_rate": 0.04,
    "verbose": -1,
}

# Training schedule.
NUM_BOOST_ROUND: int = 400
EARLY_STOPPING_ROUNDS: int = 30


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_constraints(feature_names: list[str], side: str,
                      ) -> tuple[list[list[int]], list[int]]:
    """Build LightGBM ``interaction_constraints`` (as lists of feature
    indices) and ``monotone_constraints`` (as a per-feature sign
    vector) from the templates above.

    Groups containing features not in ``feature_names`` are filtered
    silently. A feature absent from the monotone map gets 0
    (unconstrained).
    """
    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    interaction: list[list[int]] = []
    for group in INTERACTION_GROUPS_TEMPLATE:
        idxs = [name_to_idx[n] for n in group if n in name_to_idx]
        if idxs:
            interaction.append(idxs)

    mono_map = MONOTONE_BUY if side == "buy" else MONOTONE_SELL
    monotone = [mono_map.get(n, 0) for n in feature_names]

    return interaction, monotone


def derive_tenor(dte_trading_days: float | int) -> Optional[str]:
    """Bucket DTE into the v1 tenor categories."""
    if dte_trading_days is None:
        return None
    try:
        dte = int(dte_trading_days)
    except (TypeError, ValueError):
        return None
    if dte < 1:
        return None
    if dte <= 7:
        return "weekly"
    if dte <= 14:
        return "2week"
    return None        # out of scope at v1
