"""Resample local Zerodha/Kite 1-minute parquet into research timeframes.

Input can be either:
  - one all-symbol parquet file with columns date/ts, symbol, open, high, low, close, volume
  - a directory of per-symbol parquet files; symbol is read from the column or filename

Output files:
  all_5m.parquet, all_15m.parquet, all_60m.parquet, all_180m.parquet, all_1D.parquet, all_1W.parquet
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from liqpool.data import _normalise_parquet_ohlcv, resample


TIMEFRAMES = ("5m", "15m", "60m", "180m", "1D", "1W")


def _read_raw(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no parquet files found under {path}")
    frames = []
    for fp in files:
        df = pd.read_parquet(fp)
        df = df.rename(columns={c: str(c).lower() for c in df.columns})
        if "symbol" not in df.columns:
            stem = fp.stem.upper()
            for suffix in ("_1M", "_1MIN", "_MINUTE"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            df["symbol"] = stem
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    if "symbol" not in raw.columns:
        raise ValueError("raw parquet must contain symbol column or use per-symbol filenames")
    return raw


def _resample_one_symbol(symbol: str, df: pd.DataFrame, tf: str) -> pd.DataFrame:
    clean = _normalise_parquet_ohlcv(df, symbol)
    out = resample(clean, tf)
    out = out.reset_index().rename(columns={"ts": "date"})
    out.insert(1, "symbol", symbol)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Raw 1m parquet file or directory")
    ap.add_argument("--out", required=True, help="Output resampled parquet directory")
    args = ap.parse_args()

    raw_path = Path(args.raw).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = _read_raw(raw_path)
    raw["symbol"] = raw["symbol"].astype(str).str.upper()
    symbols = sorted(raw["symbol"].dropna().unique())
    print(f"[resample] loaded {len(raw):,} raw rows across {len(symbols)} symbols")

    for tf in TIMEFRAMES:
        pieces = []
        for symbol, sdf in raw.groupby("symbol", sort=True):
            try:
                pieces.append(_resample_one_symbol(symbol, sdf, tf))
            except Exception as e:
                print(f"[resample] {symbol} {tf} skipped: {e}")
        if not pieces:
            print(f"[resample] no output for {tf}")
            continue
        out = pd.concat(pieces, ignore_index=True)
        fp = out_dir / f"all_{tf}.parquet"
        out.to_parquet(fp, index=False)
        print(f"[resample] wrote {fp} rows={len(out):,}")


if __name__ == "__main__":
    main()
