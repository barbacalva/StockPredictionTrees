"""
Generates multiclass labels (UP / FLAT / DOWN) using **triple‑barrier**
meta‑labeling.  The method is path‑aware: during a look‑ahead horizon we check
whether price first crosses an upper or lower barrier (take‑profit / stop‑loss);
 otherwise, the observation is neutral.

Usage
------
$ python -m src.labels.build_labels \
        --price data/processed/SPY_prices.parquet \
        --out data/processed/SPY_labels.parquet \
        --horizon 30min --vol-days 30 --threshold 1.0
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Union
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

UTC = ZoneInfo("UTC")
_LOG = logging.getLogger(__name__)


def halflife_from_days(
        index: pd.DatetimeIndex,
        vol_days: int
) -> int:
    counts = pd.Series(1, index=index).groupby(index.date).size()
    bars_per_day = int(counts.median())
    return int(bars_per_day * vol_days / 2)


def dynamic_volatility(
        price: pd.Series,
        horizon: pd.Timedelta,
        halflife: float | None = None,
        rolling_window: int | None = None,
) -> pd.Series:
    if halflife is None and rolling_window is None:
        raise ValueError("One of 'halflife' or 'rolling_window' must be provided.")

    lagged = price.reindex(price.index - horizon, method="ffill")
    lagged.index = price.index
    ret = np.log(price / lagged)

    if halflife is not None:
        vol = ret.ewm(halflife=halflife, adjust=False).std()
    else:
        vol = ret.rolling(rolling_window, min_periods=rolling_window).std()

    return vol


def compute_vertical_barriers(
        idx: pd.DatetimeIndex,
        horizon: Union[str, pd.Timedelta],
) -> pd.Series:
    if not isinstance(horizon, pd.Timedelta):
        horizon = pd.Timedelta(horizon)

    last_bar = pd.Series(idx).groupby(idx.date).max()

    naive = idx + horizon

    last_arr = pd.to_datetime(
        pd.Series(idx.date).map(last_bar).values, utc=True
    ).tz_convert(idx.tz)

    barrier_int = np.minimum(naive, last_arr)

    return pd.Series(barrier_int, index=idx)


def compute_triple_barrier(
        price_series: pd.Series,
        vertical_barriers: pd.Series,
        thresholds: pd.Series) -> pd.Series:
    def get_first_barrier_touched_time(bet_event):
        price_path = price_series[bet_event['start']:bet_event['vertical_barrier']]
        path_returns = (price_path / price_series.asof(bet_event['start']) - 1)

        earliest_sl = path_returns[path_returns < -bet_event['target']].dropna().index.min()
        earliest_pt = path_returns[path_returns > bet_event['target']].dropna().index.min()

        return pd.Series([earliest_sl, earliest_pt], index=['sl_touch', 'pt_touch'])

    barriers = pd.DataFrame({'start': vertical_barriers.index,
                             'vertical_barrier': vertical_barriers,
                             'target': thresholds})
    barriers[['sl_touch', 'pt_touch']] = barriers.apply(get_first_barrier_touched_time, axis=1)

    barrier_columns = [col for col in ['vertical_barrier', 'sl_touch', 'pt_touch']
                       if not barriers[col].isna().all()]
    barriers['end_ts'] = barriers[barrier_columns].min(axis=1)
    barr_touched = np.select(
        [barriers['end_ts'] == barriers['pt_touch'], barriers['end_ts'] == barriers['sl_touch']],
        [1, -1], default=0
    ).astype("int8")
    return pd.Series(barr_touched, index=price_series.index)


def triple_barrier_labels(
        df: pd.DataFrame,
        horizon: pd.Timedelta,
        vol_days: int = 30,
        thresh_mult: float = 1.0,
) -> pd.DataFrame:
    """
    Returns a DataFrame with column 'y' ∈ {‑1,0,+1}:
    +1 → price hits the upper barrier first
    -1 → price hits the lower barrier first
     0 → neither hit before vertical horizon
    """
    price = df["close"]

    # Dynamic volatility‑based barriers
    halflife = halflife_from_days(price.index, vol_days)
    sigma = dynamic_volatility(price, horizon, halflife=halflife)
    thresh = sigma * thresh_mult

    v_barriers = compute_vertical_barriers(price.index, horizon)

    y = compute_triple_barrier(price, v_barriers, thresh)
    labels = pd.DataFrame({"y": y}, index=df.index).dropna()
    return labels


def cli() -> None:
    p = argparse.ArgumentParser("build_labels")
    p.add_argument("--price", required=True, type=Path,
                   help="Parquet with at least a 'close' column.")
    p.add_argument("--out", required=True, type=Path,
                   help="Output Parquet with the column 'y'.")
    p.add_argument("--horizon", default="30min",
                   help="Look‑ahead horizon (e.g. '30min', '2h').")
    p.add_argument("--vol-days", type=int, default=30,
                   help="Rolling window (in days) for volatility estimate.")
    p.add_argument("--threshold", type=float, default=1.0,
                   help="Multiplier of sigma for the barriers.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    price_df = pd.read_parquet(args.price)
    horizon_td = pd.Timedelta(args.horizon)

    labels = triple_barrier_labels(
        price_df,
        horizon=horizon_td,
        vol_days=args.vol_days,
        thresh_mult=args.threshold,
    )
    _LOG.info("Labels shape: %s | distribution: %s",
              labels.shape, labels["y"].value_counts(normalize=True).round(3).to_dict())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(args.out, compression="snappy")
    _LOG.info("Saved labels → %s", args.out)


if __name__ == "__main__":
    cli()
