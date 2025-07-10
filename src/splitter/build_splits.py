from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd

from src.splitter.split import WalkForwardSplitter


def cli() -> None:
    p = argparse.ArgumentParser("build_splits")
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--train-days", type=int, default=60)
    p.add_argument("--test-days", type=int, default=10)
    p.add_argument("--step-days", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    X = pd.read_parquet(args.features)
    y = pd.read_parquet(args.labels)["y"]

    splitter = WalkForwardSplitter(
        X, y,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    n_blocks = 0
    for block in splitter.split():
        fname = args.outdir / f"block_{block.idx:02d}.pkl"
        with open(fname, "wb") as f:
            pickle.dump(block, f, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info("Saved %s (%d train, %d test)",
                     fname.name, len(block.X_train), len(block.X_test))
        n_blocks += 1

    logging.info("Total splits generated: %d", n_blocks)


if __name__ == "__main__":
    cli()
