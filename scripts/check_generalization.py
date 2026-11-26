#!/usr/bin/env python3
"""Overfitting check: compare train / val / test metrics of the RNN score and the decision model.

A large train-vs-test gap means overfitting; close values mean the model generalises.

Run with:
    .venv-app/bin/python scripts/check_generalization.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> str:
    pred = p >= threshold
    return (f"n={len(y):6d}  ROC-AUC={roc_auc_score(y, p):.4f}  PR-AUC={average_precision_score(y, p):.4f}  "
            f"F1={f1_score(y, pred, zero_division=0):.4f}")


def main() -> int:
    rnn_meta = json.loads((ROOT / "models/rnn/rnn_meta.json").read_text())
    scores = pd.read_parquet(ROOT / "models/rnn/rnn_scores.parquet")
    print("RNN columns:", list(scores.columns))
    print("RNN score (as used by the decision model), threshold =",
          round(rnn_meta["calibration"]["threshold_calibrated"], 4))
    for split in ("train", "val", "test"):
        part = scores[scores["split"] == split]
        print(f"  {split:5s} " + metrics(part["label"].to_numpy(), part["rnn_score"].to_numpy(),
                                         rnn_meta["calibration"]["threshold_calibrated"]))

    dec_meta = json.loads((ROOT / "models/decision/decision_meta.json").read_text())
    audit = pd.read_parquet(ROOT / "models/decision/decision_features.parquet")
    print("\nLogistic regression, threshold =", round(dec_meta["threshold"], 4))
    for split in ("train", "val", "test"):
        part = audit[audit["split"] == split]
        print(f"  {split:5s} " + metrics(part["label"].to_numpy(), part["probability"].to_numpy(),
                                         dec_meta["threshold"]))
    cnn_meta = json.loads((ROOT / "models/cnn/cnn_meta.json").read_text())
    print("\nCNN (from cnn_meta.json, crops never seen in training for val/test)")
    splits = {"train": cnn_meta.get("train_metrics_clean"), "val": cnn_meta.get("val_metrics"),
              "test": cnn_meta.get("test_metrics_full", cnn_meta["test_metrics"])}
    for split, m in splits.items():
        if m:
            print(f"  {split:5s} n={m.get('n_crops', m.get('n_test_crops', 0)):6d}  "
                  f"accuracy={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}  ECE={m['ece']:.4f}")
        else:
            print(f"  {split:5s} not exported by this model version")
    if "overfitting_gap_train_minus_test_accuracy" in cnn_meta:
        print(f"  train - test accuracy gap: {cnn_meta['overfitting_gap_train_minus_test_accuracy']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
