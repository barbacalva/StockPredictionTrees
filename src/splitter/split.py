from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Tuple

import pandas as pd

_LOG = logging.getLogger(__name__)


@dataclass
class Block:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    idx: int


class WalkForwardSplitter:
    def __init__(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            *,
            train_days: int = 60,
            test_days: int = 10,
            step_days: int = 0,
    ):
        self.X, self.y = self._align(X, y)
        self.train_td = pd.Timedelta(days=train_days)
        self.test_td = pd.Timedelta(days=test_days)
        self.step_td = pd.Timedelta(days=step_days)

    @staticmethod
    def _align(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        common = X.index.intersection(y.index)
        if common.empty:
            raise ValueError("Features y labels no comparten índice.")
        X, y = X.loc[common], y.loc[common]
        return X.sort_index(), y.sort_index()

    def split(self) -> Iterator[Block]:
        start = self.X.index.min()
        end = self.X.index.max()
        block = 0

        while True:
            train_end = start + self.train_td
            test_end = train_end + self.test_td

            if test_end > end:
                break

            mask_train = (self.X.index >= start) & (self.X.index < train_end)
            mask_test = (self.X.index >= train_end) & (self.X.index < test_end)

            if mask_train.sum() == 0 or mask_test.sum() == 0:
                _LOG.warning("Bloque %d vacío; se omite.", block)
                break

            yield Block(
                X_train=self.X.loc[mask_train],
                y_train=self.y.loc[mask_train],
                X_test=self.X.loc[mask_test],
                y_test=self.y.loc[mask_test],
                idx=block,
            )

            if self.step_td == pd.Timedelta(0):
                break

            start += self.step_td
            block += 1
