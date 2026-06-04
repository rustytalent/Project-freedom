"""Kite Indian Market Data warehouse reader.

Single source of truth for reading the Google-Drive / Colab data
warehouse the user built via Zerodha Kite Connect + NSE F&O bhavcopy.
Every loader is:

  * existence-checked    — missing files raise a clear, named error.
  * column-validated     — wrong schema fails loudly, not silently.
  * lookahead-safe       — frames are sorted by time on load; helpers
                           assert no future rows leak into features.
  * timeframe-aware      — intraday vs daily layers are kept distinct;
                           daily-options data is NEVER silently joined
                           to intraday-equity data.
  * timezone-bridged     — raw_1m / resampled / spot bars arrive
                           tz-aware Asia/Kolkata; the existing liqpool
                           pipeline is tz-naive UTC. ``normalize_utc``
                           converts IST→UTC-naive on demand.

WAREHOUSE LAYOUT (exact, as built):

  raw_1m/{SYMBOL}_1m.parquet                          equity 1-min OHLCV
  resampled/{tf}/{SYMBOL}_{tf}.parquet                tf in 5m/15m/60m/180m/1D/1W
  spot/{INDEX}/5min.parquet                           index/VIX 5-min
  spot/{INDEX}/day.parquet                            index/VIX daily
  equity_daily/{SYMBOL}/day.parquet                   equity daily
  futures/{UNDERLYING}/{EXPIRY}/5min.parquet          active futures 5-min+OI
  options_active/{INDEX}/{EXPIRY}/{STRIKE}_{CE|PE}.parquet  active options 5-min+OI
  bhavcopy/{INDEX}_options_eod.parquet                expired options EOD raw
  model_options/{INDEX}_model_options_eod.parquet     filtered EOD
  model_options_with_greeks/{INDEX}_..._with_greeks.parquet  EOD + Greeks
  macro/risk_free_rate_daily.parquet                  daily r

Default root is the Colab mount; override via the ``KITE_WAREHOUSE_ROOT``
env var or the ``root`` constructor argument so the same code runs on
the VPS, locally, or in CI against a synthetic fixture warehouse.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_ROOT = "/content/drive/MyDrive/kite_indian_market_data"

# Canonical index names as the warehouse stores them.
INDEX_NAMES = ("NIFTY50", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "INDIAVIX")
# Index names as the OPTIONS layers store them (no '50' suffix on Nifty).
OPTION_INDEX_NAMES = ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY")
RESAMPLE_TFS = ("5m", "15m", "60m", "180m", "1D", "1W")

# ---------------------------------------------------------------------------
# Per-layer schema contracts
# ---------------------------------------------------------------------------

# OHLCV intraday/daily bar layers share this required column set.
_BAR_COLS = ("timestamp", "open", "high", "low", "close", "volume")
_BAR_OI_COLS = _BAR_COLS + ("open_interest",)

# Options EOD raw bhavcopy.
_BHAV_COLS = ("trading_date", "symbol", "expiry_date", "strike", "side",
              "open", "high", "low", "close", "settlement",
              "volume", "open_interest", "change_in_oi")

# Model-ready EOD (filtered) adds DTE + ATM context.
_MODEL_OPT_COLS = _BHAV_COLS + ("days_to_expiry", "atm_strike_est",
                                 "strike_distance_pct")

# Model-ready EOD + Greeks adds the BS-inverted columns.
_GREEKS_COLS = _MODEL_OPT_COLS + (
    "spot_close", "risk_free_rate", "t", "flag", "moneyness",
    "log_moneyness", "distance_from_spot_pct", "option_price_for_iv",
    "implied_volatility", "delta", "gamma", "theta", "vega",
)

_MACRO_COLS = ("trading_date", "risk_free_rate")


@dataclass
class LayerReport:
    """Diagnostic summary for one loaded frame."""
    layer: str
    key: str
    path: str
    n_rows: int
    timeframe_class: str            # "intraday" | "daily" | "eod_options"
    date_start: Optional[str]
    date_end: Optional[str]
    missing_summary: Dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        miss = (", ".join(f"{k}={v}" for k, v in self.missing_summary.items())
                if self.missing_summary else "none")
        return (f"[{self.layer}:{self.key}] rows={self.n_rows} "
                f"tf={self.timeframe_class} "
                f"range=({self.date_start}..{self.date_end}) missing={miss}")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class WarehouseError(Exception):
    """Base error for warehouse access problems."""


class WarehouseFileMissing(WarehouseError):
    pass


class WarehouseSchemaError(WarehouseError):
    pass


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class WarehouseReader:
    """Typed, validated, lookahead-safe reader over the data warehouse."""

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = Path(root or os.environ.get("KITE_WAREHOUSE_ROOT",
                                                  DEFAULT_ROOT))

    # ------------------------------------------------------------------
    # Path helpers (no IO — pure path construction so tests are cheap)
    # ------------------------------------------------------------------

    def equity_1m_path(self, symbol: str) -> Path:
        return self.root / "raw_1m" / f"{symbol}_1m.parquet"

    def resampled_path(self, symbol: str, tf: str) -> Path:
        if tf not in RESAMPLE_TFS:
            raise WarehouseError(f"unknown resample tf {tf!r}; "
                                  f"known: {RESAMPLE_TFS}")
        return self.root / "resampled" / tf / f"{symbol}_{tf}.parquet"

    def spot_path(self, index: str, tf: str = "5min") -> Path:
        if tf not in ("5min", "day"):
            raise WarehouseError(f"spot tf must be '5min' or 'day', got {tf!r}")
        return self.root / "spot" / index / f"{tf}.parquet"

    def equity_daily_path(self, symbol: str) -> Path:
        return self.root / "equity_daily" / symbol / "day.parquet"

    def futures_path(self, underlying: str, expiry: str) -> Path:
        return self.root / "futures" / underlying / expiry / "5min.parquet"

    def option_active_path(self, index: str, expiry: str, strike: int,
                            side: str) -> Path:
        side = side.upper()
        if side not in ("CE", "PE"):
            raise WarehouseError(f"option side must be CE/PE, got {side!r}")
        return (self.root / "options_active" / index / expiry
                / f"{strike}_{side}.parquet")

    def bhavcopy_path(self, index: str) -> Path:
        return self.root / "bhavcopy" / f"{index}_options_eod.parquet"

    def model_options_path(self, index: str) -> Path:
        return self.root / "model_options" / f"{index}_model_options_eod.parquet"

    def model_options_greeks_path(self, index: str) -> Path:
        return (self.root / "model_options_with_greeks"
                / f"{index}_model_options_with_greeks.parquet")

    def macro_path(self) -> Path:
        return self.root / "macro" / "risk_free_rate_daily.parquet"

    # ------------------------------------------------------------------
    # Core load + validate
    # ------------------------------------------------------------------

    def _read(self, path: Path, required_cols: Tuple[str, ...],
              ts_col: str, layer: str, key: str,
              timeframe_class: str,
              normalize_utc: bool = False,
              sort: bool = True) -> Tuple[pd.DataFrame, LayerReport]:
        """Load a parquet, validate schema, sort by time, build a report.

        ``normalize_utc`` converts a tz-aware Asia/Kolkata ``ts_col`` into
        a tz-naive UTC index (the liqpool pipeline convention). Only valid
        for the timestamp-indexed bar layers.
        """
        if not path.exists():
            raise WarehouseFileMissing(
                f"warehouse file not found: {path}. "
                f"(layer={layer}, key={key}). Check the warehouse root "
                f"and that the Drive is mounted.")
        df = pd.read_parquet(path)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise WarehouseSchemaError(
                f"{path} missing required columns {missing}; "
                f"found {list(df.columns)}")

        # Time handling.
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col])
            if normalize_utc:
                df = self._to_utc_naive(df, ts_col)
            if sort:
                df = df.sort_values(ts_col).reset_index(drop=True)

        date_start = date_end = None
        if ts_col in df.columns and len(df):
            date_start = str(pd.to_datetime(df[ts_col].iloc[0]))
            date_end = str(pd.to_datetime(df[ts_col].iloc[-1]))

        # Missing-value summary on the columns that matter.
        miss_summary: Dict[str, int] = {}
        for c in required_cols:
            if c in df.columns:
                n_na = int(df[c].isna().sum())
                if n_na:
                    miss_summary[c] = n_na

        report = LayerReport(
            layer=layer, key=key, path=str(path), n_rows=int(len(df)),
            timeframe_class=timeframe_class,
            date_start=date_start, date_end=date_end,
            missing_summary=miss_summary,
        )
        return df, report

    @staticmethod
    def _to_utc_naive(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
        """Convert a tz-aware (or IST-implied) timestamp column to tz-naive
        UTC, matching the liqpool convention (tz-naive == UTC, IST=UTC+5:30).
        """
        df = df.copy()
        ts = pd.to_datetime(df[ts_col])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        else:
            # tz-naive but stored as IST wall-clock → subtract 5:30 to get UTC.
            ts = ts - pd.Timedelta(hours=5, minutes=30)
        df[ts_col] = ts
        return df

    # ------------------------------------------------------------------
    # Public typed loaders — intraday / daily bars
    # ------------------------------------------------------------------

    def load_equity_1m(self, symbol: str, normalize_utc: bool = True
                       ) -> Tuple[pd.DataFrame, LayerReport]:
        return self._read(self.equity_1m_path(symbol), _BAR_COLS,
                          "timestamp", "raw_1m", symbol, "intraday",
                          normalize_utc=normalize_utc)

    def load_resampled(self, symbol: str, tf: str,
                       normalize_utc: bool = True
                       ) -> Tuple[pd.DataFrame, LayerReport]:
        tf_class = "daily" if tf in ("1D", "1W") else "intraday"
        return self._read(self.resampled_path(symbol, tf), _BAR_COLS,
                          "timestamp", f"resampled/{tf}", symbol, tf_class,
                          normalize_utc=normalize_utc)

    def load_spot(self, index: str, tf: str = "5min",
                  normalize_utc: bool = True
                  ) -> Tuple[pd.DataFrame, LayerReport]:
        tf_class = "daily" if tf == "day" else "intraday"
        # spot layer carries open_interest (usually 0 for indexes).
        return self._read(self.spot_path(index, tf), _BAR_OI_COLS,
                          "timestamp", f"spot/{tf}", index, tf_class,
                          normalize_utc=normalize_utc)

    def load_equity_daily(self, symbol: str, normalize_utc: bool = True
                          ) -> Tuple[pd.DataFrame, LayerReport]:
        return self._read(self.equity_daily_path(symbol), _BAR_OI_COLS,
                          "timestamp", "equity_daily", symbol, "daily",
                          normalize_utc=normalize_utc)

    def load_futures(self, underlying: str, expiry: str,
                     normalize_utc: bool = True
                     ) -> Tuple[pd.DataFrame, LayerReport]:
        return self._read(self.futures_path(underlying, expiry), _BAR_OI_COLS,
                          "timestamp", "futures", f"{underlying}/{expiry}",
                          "intraday", normalize_utc=normalize_utc)

    def load_option_active(self, index: str, expiry: str, strike: int,
                           side: str, normalize_utc: bool = True
                           ) -> Tuple[pd.DataFrame, LayerReport]:
        return self._read(self.option_active_path(index, expiry, strike, side),
                          _BAR_OI_COLS, "timestamp", "options_active",
                          f"{index}/{expiry}/{strike}_{side}", "intraday",
                          normalize_utc=normalize_utc)

    # ------------------------------------------------------------------
    # Public typed loaders — daily EOD options (kept SEPARATE from intraday)
    # ------------------------------------------------------------------

    def load_bhavcopy(self, index: str) -> Tuple[pd.DataFrame, LayerReport]:
        return self._read(self.bhavcopy_path(index), _BHAV_COLS,
                          "trading_date", "bhavcopy", index, "eod_options",
                          normalize_utc=False)

    def load_model_options(self, index: str
                           ) -> Tuple[pd.DataFrame, LayerReport]:
        return self._read(self.model_options_path(index), _MODEL_OPT_COLS,
                          "trading_date", "model_options", index,
                          "eod_options", normalize_utc=False)

    def load_model_options_with_greeks(self, index: str,
                                        drop_missing_iv: bool = False
                                        ) -> Tuple[pd.DataFrame, LayerReport]:
        """Load the EOD options + Greeks file.

        ~84-90% of rows have a finite implied_volatility (deep ITM/OTM
        inversion fails). ``drop_missing_iv=True`` removes rows where
        implied_volatility is NaN; the default keeps them so callers can
        decide. The report's missing_summary always shows the IV gap.
        """
        df, rep = self._read(self.model_options_greeks_path(index),
                             _GREEKS_COLS, "trading_date",
                             "model_options_with_greeks", index,
                             "eod_options", normalize_utc=False)
        if drop_missing_iv and "implied_volatility" in df.columns:
            before = len(df)
            df = df[df["implied_volatility"].notna()].reset_index(drop=True)
            rep.missing_summary["dropped_missing_iv"] = before - len(df)
            rep.n_rows = len(df)
        return df, rep

    def load_macro(self) -> Tuple[pd.DataFrame, LayerReport]:
        return self._read(self.macro_path(), _MACRO_COLS, "trading_date",
                          "macro", "risk_free_rate", "daily",
                          normalize_utc=False)

    # ------------------------------------------------------------------
    # Discovery + summary
    # ------------------------------------------------------------------

    def available_equity_symbols(self) -> List[str]:
        d = self.root / "raw_1m"
        if not d.exists():
            return []
        return sorted(p.stem.replace("_1m", "") for p in d.glob("*_1m.parquet"))

    def available_spot_indexes(self) -> List[str]:
        d = self.root / "spot"
        if not d.exists():
            return []
        return sorted(p.name for p in d.iterdir() if p.is_dir())

    def summary(self, equity_symbols: Optional[List[str]] = None,
                indexes: Optional[List[str]] = None,
                print_report: bool = True) -> List[LayerReport]:
        """Walk a representative slice of the warehouse, validating and
        reporting row counts + date ranges for each layer. Robust: a
        missing layer is reported as a warning line, not a crash."""
        reports: List[LayerReport] = []
        equity_symbols = equity_symbols or self.available_equity_symbols()[:3]
        indexes = indexes or [i for i in self.available_spot_indexes()][:3]

        def _try(loader, *args, **kwargs):
            try:
                _, rep = loader(*args, **kwargs)
                reports.append(rep)
                if print_report:
                    print(rep)
            except WarehouseError as exc:
                if print_report:
                    print(f"[WARN] {exc}")

        for sym in equity_symbols:
            _try(self.load_equity_1m, sym)
            _try(self.load_equity_daily, sym)
        for idx in indexes:
            _try(self.load_spot, idx, "5min")
        for opt_idx in OPTION_INDEX_NAMES:
            _try(self.load_model_options_with_greeks, opt_idx)
        _try(self.load_macro)
        return reports


# ---------------------------------------------------------------------------
# Lookahead-safety helpers — used by every feature builder downstream
# ---------------------------------------------------------------------------

def assert_sorted_by_time(df: pd.DataFrame, ts_col: str = "timestamp") -> None:
    """Raise if the frame is not strictly non-decreasing in time. Every
    causal feature builder should call this before computing anything."""
    if ts_col not in df.columns:
        raise WarehouseSchemaError(f"no {ts_col!r} column to check ordering")
    ts = pd.to_datetime(df[ts_col]).values
    if len(ts) > 1 and not np.all(ts[1:] >= ts[:-1]):
        raise WarehouseError(
            f"frame is not sorted ascending by {ts_col!r}; sort before "
            f"feature engineering or you risk lookahead bias")


def causal_rolling(series: pd.Series, window: int, fn: str = "mean",
                   min_periods: Optional[int] = None) -> pd.Series:
    """Causal rolling reduction with NO bfill/ffill.

    The audit flagged bfill() warmups injecting future-smoothed values
    into the earliest bars. This helper never back-fills: warmup rows
    that lack a full window get the expanding value (min_periods=1 by
    default) or NaN, never a value computed from future bars.
    """
    min_periods = 1 if min_periods is None else min_periods
    roll = series.rolling(window, min_periods=min_periods)
    if not hasattr(roll, fn):
        raise WarehouseError(f"unknown rolling fn {fn!r}")
    return getattr(roll, fn)()


def align_daily_to_intraday(intraday: pd.DataFrame,
                            daily_feature: pd.DataFrame,
                            intraday_ts: str = "timestamp",
                            daily_date: str = "trading_date",
                            feature_cols: Optional[List[str]] = None,
                            ) -> pd.DataFrame:
    """Join a DAILY feature frame onto an INTRADAY bar frame WITHOUT
    lookahead: each intraday bar on date D gets the daily feature row
    from date D-1 (the most recent CLOSED daily value).

    This is the only sanctioned way to mix daily-options/daily-regime
    features into an intraday-equity model. Using same-day daily close
    on an intraday bar would leak the day's outcome into morning bars.
    """
    intraday = intraday.copy()
    daily = daily_feature.copy()
    feature_cols = feature_cols or [c for c in daily.columns
                                    if c != daily_date]

    intraday["_date"] = pd.to_datetime(intraday[intraday_ts]).dt.normalize()
    daily["_date"] = pd.to_datetime(daily[daily_date]).dt.normalize()
    daily = daily.sort_values("_date").reset_index(drop=True)
    # Shift daily features forward by one trading row so date D bar sees
    # date D-1's value.
    shifted = daily[["_date"] + feature_cols].copy()
    shifted[feature_cols] = shifted[feature_cols].shift(1)

    merged = pd.merge_asof(
        intraday.sort_values("_date"),
        shifted.sort_values("_date"),
        on="_date", direction="backward",
    )
    return merged.drop(columns=["_date"])
