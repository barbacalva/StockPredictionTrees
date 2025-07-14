"""
Trains and evaluates a XGBoost model for each walk-forward block.
Ejemplo:
$ python -m src.training.train_xgb \
      --splits  data/processed/splits_wf \
      --models  models \
      --reports reports \
      --n-estimators 300 --max-depth 6 --eta 0.05
"""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


_LOG = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, avg: str = "macro") -> dict:
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=avg, zero_division=0)
    recall = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
    return {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def train_and_evaluate(
        splits_dir: Path,
        models_dir: Path,
        signals_dir: Path,
        **xgb_params,
) -> pd.DataFrame:
    models_dir.mkdir(parents=True, exist_ok=True)
    signals_dir.mkdir(parents=True, exist_ok=True)

    class_map = {-1: 0, 0: 1, 1: 2}  # DOWN→0  FLAT→1  UP→2
    rows = []
    signals = []

    for pkl_path in sorted(splits_dir.glob("block_*.pkl")):
        block_id = int(pkl_path.stem.split("_")[-1])
        _LOG.info("Processing %s …", pkl_path.name)

        with open(pkl_path, "rb") as f:
            block = pickle.load(f)

        y_train = block.y_train.map(class_map).to_numpy()
        y_test = block.y_test.map(class_map).to_numpy()

        dtrain = xgb.DMatrix(block.X_train, label=y_train)
        dtest = xgb.DMatrix(block.X_test, label=y_test)

        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "subsample": 0.8,
            "eval_metric": "mlogloss",
            **xgb_params,
        }

        bst = xgb.train(
            params, dtrain,
            num_boost_round=params.pop("n_estimators", 300),
            verbose_eval=False,
        )

        proba = bst.predict(dtest)
        y_pred = np.argmax(proba, axis=1)

        sig_series = pd.Series(y_pred, index=block.X_test.index, name="labels")
        sig_series = sig_series.map({0: -1, 1: 0, 2: 1})

        metrics = compute_metrics(block.y_test.to_numpy(), y_pred)
        _LOG.info("Bloque %02d | acc=%.3f f1=%.3f", block_id,
                  metrics["accuracy"], metrics["f1"])

        model_path = models_dir / f"xgb_block_{block_id:02d}.json"
        bst.save_model(model_path)

        rows.append({"block": block_id, **metrics})
        signals.append(sig_series)

    signals = pd.concat(signals)
    signals_path = signals_dir / "signals.csv"
    signals.to_csv(signals_path)

    return pd.DataFrame(rows).set_index("block")


def cli() -> None:
    p = argparse.ArgumentParser("train_xgb")
    p.add_argument("--splits", required=True, type=Path)
    p.add_argument("--models", type=Path, default="models")
    p.add_argument("--signals", type=Path, default="signals")
    p.add_argument("--reports", type=Path, default="reports")

    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--eta", type=float, default=0.05)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    summary = train_and_evaluate(
        splits_dir=args.splits,
        models_dir=args.models,
        signals_dir=args.signals,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        eta=args.eta,
    )

    agg = summary.agg(["mean", "std"]).round(4)
    agg.index = ["GLOBAL_MEAN", "GLOBAL_STD"]
    summary_full = pd.concat([summary, agg])

    summary_path = args.reports / "summary_metrics.csv"
    summary_full.to_csv(summary_path)
    logging.info("Saved report → %s", summary_path)
    logging.info("\n%s", summary_full)


if __name__ == "__main__":
    cli()
