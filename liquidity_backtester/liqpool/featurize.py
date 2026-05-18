"""Convert a Pool into a flat feature vector for ML.

All features are CAUSAL — computed using only data available at `pool.available_at` or
properties of the pool itself (which are determined by data <= available_at).

Feature groups:
  COUNTS:
    n_contributors           total contributing detectors
    n_distinct_tfs           how many TFs contributed
    factor_count_<family>    per-family count (EQHL, OB, FVG, PM, PW, REJ, PD, ORB, HVN, SWING)
    tf_count_<tf_name>       per-TF count (base, 15min, 60min, 180min, 1D, 1W, PD, PW, PM)
  POOL GEOMETRY:
    width_atr                pool width / median ATR
    score                    Track 1.5 heuristic score (kept as one input among many)
    median_contributor_strength
    age_at_availability_bars hours between earliest contributor formation and available_at (staleness)
  CONTEXT AT available_at (causal):
    adx_14                   trend strength on base TF
    vol_ratio                short vs long ATR ratio
    session_<label>          one-hot NSE session
  POOL TYPE:
    side_high                1 if pool is on the high side (supply), 0 if low (demand)
    headline_factor_<f>      one-hot of the headline-factor family
"""
from __future__ import annotations
from typing import List, Dict
import pandas as pd
import numpy as np

from typing import Optional

from .pools import Pool
from .indicators import atr
from .regime import compute_regime_series, lookup_regime, SESSION_LABELS
from .correlation import pool_factors
from .stratified import _headline_factor, _FACTOR_PRIORITY


FACTOR_FAMILIES = ("EQHL", "OB", "FVG", "PM", "PW", "REJ", "PD", "ORB", "HVN", "SWING")
TF_KEYS = ("base", "15min", "60min", "180min", "1D", "1W", "PD", "PW", "PM")


class Featurizer:
    """Fit once on a base TF dataframe (precomputes ATR and regime series), then call .transform
    on individual pools or batches. Stateless wrt training — same Featurizer is used for train
    and OOS pools.

    Multi-asset support: pass `known_assets=[...]` to enable an asset one-hot block. All
    Featurizers in a multi-asset run share the same `known_assets` list so feature columns are
    identical across them; transform() reads pool.asset to populate the correct one-hot."""

    def __init__(self, df_base: pd.DataFrame, atr_period: int = 14,
                 asset_name: str = "", known_assets: Optional[List[str]] = None):
        self.df_base = df_base
        self.atr_series = atr(df_base, atr_period).bfill()
        self.median_atr = float(self.atr_series.median())
        self.regime_df = compute_regime_series(df_base)
        # Base-period in seconds, computed from the actual index. Used for age normalisation.
        # Defaults to 300s (5m) if we can't infer (e.g., single-bar df).
        diffs = pd.Series(df_base.index).diff().dropna()
        self.base_period_seconds = float(diffs.median().total_seconds()) if len(diffs) else 300.0
        if self.base_period_seconds <= 0:
            self.base_period_seconds = 300.0
        self.asset_name = asset_name
        self.known_assets = list(known_assets) if known_assets else []
        # Pre-build the canonical feature column list for stable ML input.
        self.feature_names = self._build_feature_names()

    def _build_feature_names(self) -> List[str]:
        cols: List[str] = ["n_contributors", "n_distinct_tfs",
                            "width_atr", "score", "median_contributor_strength",
                            "age_at_availability_bars",
                            "adx_14", "vol_ratio", "side_high"]
        cols += [f"factor_count_{f}" for f in FACTOR_FAMILIES]
        cols += [f"tf_count_{t}" for t in TF_KEYS]
        cols += [f"session_{s}" for s in SESSION_LABELS]
        cols += [f"headline_factor_{f}" for f in FACTOR_FAMILIES]
        if self.known_assets:
            cols += [f"asset_{a}" for a in self.known_assets]
        return cols

    def transform(self, pool: Pool) -> Dict[str, float]:
        ctrs = pool.contributors
        n_ctr = len(ctrs)
        med_atr = max(self.median_atr, 1e-9)

        # Factor + TF counts
        factor_counts = {f: 0 for f in FACTOR_FAMILIES}
        tf_counts = {t: 0 for t in TF_KEYS}
        from .correlation import _FAMILY as FAMILY_MAP
        for c in ctrs:
            head = c.source.split("@", 1)[0]
            fam = FAMILY_MAP.get(head, head)
            if fam in factor_counts:
                factor_counts[fam] += 1
            tf = c.tf
            if tf in tf_counts:
                tf_counts[tf] += 1

        # Headline factor (for one-hot)
        hf = _headline_factor(pool)

        # Pool geometry
        width_atr = pool.width / med_atr
        med_strength = float(np.median([c.strength for c in ctrs])) if ctrs else 0.0
        earliest_ts = min((c.ts for c in ctrs), default=pool.formed_at)
        if pool.available_at and earliest_ts and pool.available_at >= earliest_ts:
            age_bars = (pool.available_at - earliest_ts).total_seconds() / self.base_period_seconds
        else:
            age_bars = 0.0

        # Regime at available_at
        regime = lookup_regime(self.regime_df, pool.available_at)
        session = regime["session"]

        # Build dict
        feats: Dict[str, float] = {
            "n_contributors": float(n_ctr),
            "n_distinct_tfs": float(len(set(pool.tfs))),
            "width_atr": float(width_atr),
            "score": float(pool.score),
            "median_contributor_strength": float(med_strength),
            "age_at_availability_bars": float(age_bars),
            "adx_14": float(regime["adx_14"]) if not np.isnan(regime["adx_14"]) else 0.0,
            "vol_ratio": float(regime["vol_ratio"]) if not np.isnan(regime["vol_ratio"]) else 1.0,
            "side_high": 1.0 if pool.side == "high" else 0.0,
        }
        for f in FACTOR_FAMILIES:
            feats[f"factor_count_{f}"] = float(factor_counts[f])
        for t in TF_KEYS:
            feats[f"tf_count_{t}"] = float(tf_counts[t])
        for s in SESSION_LABELS:
            feats[f"session_{s}"] = 1.0 if session == s else 0.0
        for f in FACTOR_FAMILIES:
            feats[f"headline_factor_{f}"] = 1.0 if hf == f else 0.0
        # Multi-asset one-hot. Reads pool.asset, falls back to self.asset_name if pool didn't
        # have one set (single-asset compatibility).
        if self.known_assets:
            tag = pool.asset or self.asset_name
            for a in self.known_assets:
                feats[f"asset_{a}"] = 1.0 if tag == a else 0.0
        return feats

    def transform_batch(self, pools: List[Pool]) -> pd.DataFrame:
        rows = [self.transform(p) for p in pools]
        # Ensure stable column order even if some pools are missing factors.
        return pd.DataFrame(rows, columns=self.feature_names).fillna(0.0)


class MultiAssetFeaturizer:
    """Single-entry featurizer for multi-asset runs. Holds one Featurizer per asset (so per-asset
    ATR/regime series are correct) and dispatches based on `pool.asset` to the right one. All
    underlying Featurizers share the same `known_assets` list so feature columns are identical."""

    def __init__(self, asset_dfs: Dict[str, pd.DataFrame], atr_period: int = 14):
        self.assets = sorted(asset_dfs.keys())
        self._per_asset: Dict[str, Featurizer] = {
            a: Featurizer(df, atr_period, asset_name=a, known_assets=self.assets)
            for a, df in asset_dfs.items()
        }
        if not self._per_asset:
            raise ValueError("MultiAssetFeaturizer needs at least one asset df")
        first = self._per_asset[self.assets[0]]
        self.feature_names = first.feature_names

    def transform(self, pool: Pool) -> Dict[str, float]:
        if pool.asset not in self._per_asset:
            raise KeyError(f"Pool.asset={pool.asset!r} not in featurizer assets "
                            f"{self.assets}; tag pools before featurizing.")
        return self._per_asset[pool.asset].transform(pool)

    def transform_batch(self, pools: List[Pool]) -> pd.DataFrame:
        rows = [self.transform(p) for p in pools]
        return pd.DataFrame(rows, columns=self.feature_names).fillna(0.0)

    def for_asset(self, asset: str) -> Featurizer:
        """Return the underlying single-asset Featurizer (e.g. to access base_period_seconds)."""
        return self._per_asset[asset]
