"""
CLI script to clean raw CSV and generate features for training.
Example usage:
$ python -m src.features.build_features \
      --csv data/raw/SPY_20220101_20250630_5m.csv \
      --out data/processed/SPY_features.parquet
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pandas_ta as ta

from .cleaning import clean

UTC = ZoneInfo("UTC")
_LOG = logging.getLogger(__name__)


def _as_utc(d: str) -> dt.datetime:
    return dt.datetime.fromisoformat(d).replace(tzinfo=UTC)


def _relative_ma(series: pd.Series, window: int, ema: bool = False) -> pd.Series:
    ma = (series.ewm(span=window, adjust=False).mean()
          if ema else series.rolling(window).mean())
    return (series - ma) / ma


def _vwap_distance(df: pd.DataFrame) -> pd.Series:
    vwaps = (df["close"] * df["volume"]).groupby(df.index.date).cumsum() / \
            df["volume"].groupby(df.index.date).cumsum()
    return (df["close"] - vwaps) / df["close"]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    prc = df["close"]

    out = pd.DataFrame(index=df.index)
    # Log returns
    for lag in (1, 5, 10):
        out[f"ret_lag{lag}"] = np.log(prc / prc.shift(lag))

    # Trend
    out["sma10"] = _relative_ma(prc, 10)
    out["ema20"] = _relative_ma(prc, 20, ema=True)
    out["ema50"] = _relative_ma(prc, 50, ema=True)

    # Momentum
    out["rsi14"] = df.ta.rsi(length=14)
    out["mom10"] = prc - prc.shift(10)

    # Volatility
    hlc = df.ta.atr(length=10)
    out["atr10"] = hlc / prc
    bb = df.ta.bbands(length=20, std=2)
    out["bb_width20"] = (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) / prc

    # Volume
    vol_mean = df["volume"].rolling("30D").mean()
    vol_std = df["volume"].rolling("30D").std()
    out["vol_z30"] = (df["volume"] - vol_mean) / vol_std

    # Relative price
    out["dist_vwap_d"] = _vwap_distance(df)

    # Session time
    idx_east = out.index.tz_convert("America/New_York")
    out["hour"] = idx_east.hour
    out["minute"] = idx_east.minute
    out["is_opening"] = (out["hour"] == 9) & (out["minute"] <= 45)
    out["is_closing"] = (out["hour"] == 15) & (out["minute"] >= 30)

    # Clean NaNs
    out = out.dropna().astype(np.float32)
    _LOG.info("Feature matrix shape: %s", out.shape)
    return out


def cli():
    p = argparse.ArgumentParser("build_features")
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--start", type=_as_utc, default=None)
    p.add_argument("--end", type=_as_utc, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    df_raw = pd.read_csv(args.csv, parse_dates=["timestamp"], index_col="timestamp")
    df_clean = clean(df_raw, args.start, args.end)
    prices_out = args.out.with_name(args.csv.stem + "_prices.parquet")
    df_clean.to_parquet(prices_out, compression="snappy")
    _LOG.info("Saved cleaned prices → %s", prices_out)

    feats = engineer(df_clean)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(args.out, compression="snappy")
    _LOG.info("Saved features → %s", args.out)


if __name__ == "__main__":
    cli()
