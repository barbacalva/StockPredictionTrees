"""
Cleans the 5-minute OHLCV bars.
Guarantees UTC index, NYSE regular session and gap forward-fill.
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import pandas_market_calendars as mcal

_LOG = logging.getLogger(__name__)


def expected_timestamps(start: datetime, end: datetime, freq: str = "5min") -> pd.DatetimeIndex:
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start.date(), end.date())
    return mcal.date_range(schedule, frequency=freq)


def clean(df: pd.DataFrame, start: datetime | None, end: datetime | None) -> pd.DataFrame:
    df = df.copy().sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
    else:
        df.index = df.index.tz_convert("UTC")

    if start is None:
        start = df.index[0]
    if end is None:
        end = df.index[-1]

    mask = (df.index >= start) & (df.index <= end)
    df = df.loc[mask]

    expected = expected_timestamps(start, end)
    df = df.reindex(expected)
    df.ffill(inplace=True)
    df.drop_duplicates(inplace=True)

    _LOG.info("Cleaned dataframe: %d → %d rows", len(expected), len(df))
    return df
