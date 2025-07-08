"""
Downloads 5 min OHLCV bars for a single symbol (ex. SPY) and saves it in
a reproducible csv. Uses Alpha Vantage if API KEY exists; otherwise, tries
yfinance (only 60 d). As a last resource, it searches for an existing CSV.

Usage
----
$ python -m src.data.download_intraday \
    --symbol SPY --start 2022-01-01 --end 2025-07-04 --dest data/raw/
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()


class IntradayDownloader:

    def __init__(self, symbol: str, start: datetime, end: datetime, dest: Path):
        self.symbol = symbol.upper()
        self.start, self.end = _as_utc(start), _as_utc(end)
        self.dest = dest
        self.dest.mkdir(parents=True, exist_ok=True)
        self.max_span = timedelta(days=59)
        self.logger = logging.getLogger(self.__class__.__name__)

    def get(self) -> Optional[pd.DataFrame]:

        data = None
        key = os.getenv("ALPHAVANTAGE_API_KEY")
        if key:
            self.logger.info("Using Alpha Vantage")
            data = self._from_alpha(key)
        else:
            self.logger.warning("ALPHAVANTAGE_API_KEY not defined: using yfinance (max 60 d)")

        if data is None:
            limit = _as_utc(datetime.now()) - self.max_span

            if self.end < limit:
                self.logger.error(
                    "Requested range ends before last 60 day limit; cannot be obtained through yfinance."
                )
                return None

            if self.start < limit:
                self.logger.warning(
                    "Requested range exceeds last 60 day limit; trimming to last 60."
                )
                self.start = limit
            data = self._from_yahoo()

        return data

    def _from_alpha(self, key: str) -> Optional[pd.DataFrame]:
        months = pd.period_range(self.start, self.end, freq="M")
        chunks: list[pd.DataFrame] = []

        for m in months:
            url = (
                "https://www.alphavantage.co/query"
                f"?function=TIME_SERIES_INTRADAY"
                f"&symbol={self.symbol}"
                f"&interval=5min&month={m.strftime('%Y-%m')}"
                "&adjusted=false&extended_hours=false&datatype=csv"
                f"&outputsize=full&apikey={key}"
            )
            csv = self._safe_request(url)
            if csv:
                df = pd.read_csv(io.StringIO(csv))
                chunks.append(df)
            else:
                break

        if len(chunks) == 0:
            logging.error('Could not fetch data from Alpha Vantage')
            return None

        data = pd.concat(chunks, ignore_index=True)
        data = data.assign(timestamp=pd.to_datetime(data["timestamp"]))
        data = data.set_index("timestamp")
        data.index = data.index.tz_localize("US/Eastern")
        data = data.sort_index()
        data = data.loc[self.start:self.end]
        return data

    def _from_yahoo(self) -> pd.DataFrame:
        df = yf.download(
            self.symbol,
            start=self.start,
            end=self.end + timedelta(days=1),
            interval="5m",
            progress=False,
            auto_adjust=False,
            prepost=False,
        )
        if df.empty:
            raise RuntimeError(
                "yfinance did not return anything; review symbol or range."
            )
        df = df.droplevel(1, axis=1)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df.index.name = 'timestamp'
        df = df[~df.index.duplicated(keep="first")]
        df = df.loc[self.start:self.end]
        return df

    def _safe_request(self, url: str, retries: int = 3, pause: int = 12) -> Optional[str]:
        for attempt in range(1, retries + 1):
            r = requests.get(url, timeout=30)
            if "Note" in r.text or "Information" in r.text:
                self.logger.warning("Reached API key limit")
                return None
            if r.ok and r.text.strip():
                return r.text
            self.logger.warning("Try # %d/%d failed (%s)", attempt, retries, url)
            time.sleep(pause)
        return None

    @staticmethod
    def cli() -> None:
        p = argparse.ArgumentParser("download_intraday")
        p.add_argument("--symbol", required=True)
        p.add_argument("--start", type=lambda s: datetime.fromisoformat(s), required=True)
        p.add_argument("--end", type=lambda s: datetime.fromisoformat(s), required=True)
        p.add_argument("--dest", type=Path, default=Path("../../data/raw"))
        args = p.parse_args()

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s : %(message)s",
        )
        out = args.dest / f"{args.symbol}_{args.start:%Y%m%d}_{args.end:%Y%m%d}_5m.csv"
        if out.exists():
            logging.info("File %s already exists — omitting", out)
            return

        dl = IntradayDownloader(args.symbol, args.start, args.end, args.dest)
        df = dl.get()
        if df is None:
            logging.error("Could not fetch any data")
        df.to_csv(out)
        logging.info("Saved %s (%d rows)", out, len(df))


def _as_utc(dt_like: datetime) -> datetime:
    if dt_like.tzinfo is None:
        return dt_like.replace(tzinfo=ZoneInfo("UTC"))
    return dt_like.astimezone(ZoneInfo("UTC"))


if __name__ == "__main__":
    IntradayDownloader.cli()
