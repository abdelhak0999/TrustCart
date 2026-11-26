#!/usr/bin/env python3
"""Train the TrustCart decision model (logistic regression) and export its metadata.

Zero-leakage protocol
---------------------
1. Features are derived only from the raw session events (no label is ever used).
2. The existing ``split`` column of ``models/rnn/rnn_scores.parquet`` is respected verbatim:
   train -> fit the scaler and the classifier; val -> choose the alert threshold and the
   class-weight variant; test -> final, untouched evaluation.
3. The test split is never used for any choice.

The checkout features mirror ``app/trustcart/decision.py:features`` exactly:
    rnn_risk, items_picked_not_presented, cnn_confidence, cart_total, visit_duration_min

Because the RPC dataset has no per-session checkout photos, ``cnn_confidence`` is a deterministic,
label-independent proxy derived from a hash of each presented item id. This is documented in the
exported metadata and is replaced by the real CNN confidence at inference time when crops are
available.

Run with:
    .venv-app/bin/python scripts/train_decision.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES = ROOT / "models" / "rnn" / "rnn_scores.parquet"
DEFAULT_EVENTS = ROOT / "outputs" / "rnn_v5" / "sessions_labeled.parquet"
DEFAULT_OUT = ROOT / "models" / "decision"

FEATURES = ("rnn_risk", "items_picked_not_presented", "cnn_confidence", "cart_total",
            "visit_duration_min")
FEATURE_LABELS_FR = {
    "rnn_risk": "risque du parcours (RNN)",
    "items_picked_not_presented": "articles pris non présentés",
    "cnn_confidence": "confiance de la reconnaissance (CNN)",
    "cart_total": "montant du panier",
    "visit_duration_min": "durée de la visite",
}
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- feature engineering

def _proxy_cnn_confidence(item_id: int) -> float:
    """Deterministic, label-independent confidence proxy in [0.55, 0.99].

    The RPC pipeline does not export per-session checkout images, so a real CNN confidence cannot
    be recomputed for the 33,218 historical sessions. Using the label here would leak the target;
    a seeded item hash keeps the feature honest (it carries no theft information) and lets the
    coefficient learn whether the signal is useful. At inference, the real CNN confidence is used.
    """
    digest = hashlib.sha256(f"trustcart-cnn-proxy:{int(item_id)}".encode()).digest()
    uniform = int.from_bytes(digest[:8], "big") / 2 ** 64
    return 0.55 + 0.44 * uniform


def session_features(events: pd.DataFrame) -> pd.DataFrame:
    """Derive the five decision features from raw session events, one row per session."""
    rows = []
    for session_id, group in events.groupby("session_id", sort=False):
        picked = tuple(dict.fromkeys(group.loc[group["action"] == "PICK", "itemid"]))
        paid = tuple(group.loc[group["action"] == "PAY", "itemid"])
        price_by_item = group.drop_duplicates("itemid").set_index("itemid")["price"]
        presented = set(paid)

        confidences = [_proxy_cnn_confidence(item) for item in presented]
        rows.append({
            "session_id": int(session_id),
            "items_picked_not_presented": sum(1 for item in picked if item not in presented),
            "cart_total": float(sum(price_by_item.get(item, 0.0) for item in presented)),
            "visit_duration_min": float(group["dt_sec"].sum() / 60.0),
            "cnn_confidence": float(np.mean(confidences)) if confidences else np.nan,
        })
    return pd.DataFrame(rows)


def build_dataset(scores_path: Path, events_path: Path) -> tuple[pd.DataFrame, float]:
    scores = pd.read_parquet(scores_path, columns=["session_id", "label", "split", "rnn_score"])
    events = pd.read_parquet(events_path, columns=["session_id", "action", "itemid", "price", "dt_sec"])

    features = session_features(events)
    data = scores.merge(features, on="session_id", how="left", validate="one_to_one")
    if data["split"].isna().any() or data["rnn_score"].isna().any():
        raise ValueError("Missing values after merging scores and events")

    unknown_splits = set(data["split"].unique()) - set(SPLITS)
    if unknown_splits:
        raise ValueError(f"Unexpected split values: {unknown_splits}")

    # The CNN confidence is only missing for sessions where nothing was paid/presented. Impute with
    # the training median, stored in the metadata and reused at inference when no crop is supplied.
    fill = float(data.loc[data["split"] == "train", "cnn_confidence"].median())
    data["cnn_confidence"] = data["cnn_confidence"].fillna(fill)
    data["rnn_risk"] = data["rnn_score"].astype(float)

    if data[list(FEATURES)].isna().any().any():
        raise ValueError("Feature matrix still contains NaN values")
    return data, fill


# --------------------------------------------------------------------------- metrics & threshold

def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lo) & (probabilities < hi)
        if not mask.any():
            continue
        error += mask.mean() * abs(y_true[mask].mean() - probabilities[mask].mean())
    return float(error)


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Best F1 on validation, tie-broken by recall then by proximity to 0.5."""
    candidates = np.unique(np.concatenate((np.linspace(0.01, 0.99, 197), probabilities)))
    best = (-1.0, -1.0, -1.0, 0.5)
    for threshold in candidates:
        predicted = probabilities >= threshold
        f1 = f1_score(y_true, predicted, zero_division=0)
        recall = recall_score(y_true, predicted, zero_division=0)
        score = (float(f1), float(recall), -abs(float(threshold) - 0.5))
        if score > best[:3]:
            best = (score[0], score[1], score[2], float(threshold))
    return best[3], best[0]


def evaluate(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "brier": float(brier_score_loss(y_true, probabilities)),
        "ece": expected_calibration_error(y_true, probabilities),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# --------------------------------------------------------------------------- training

def train(data: pd.DataFrame, fill: float, out_dir: Path) -> dict:
    train_df = data[data["split"] == "train"]
    val_df = data[data["split"] == "val"]
    test_df = data[data["split"] == "test"]

    if not (len(train_df) and len(val_df) and len(test_df)):
        raise ValueError("One of the train/val/test splits is empty")
    overlap = set(train_df["session_id"]) & set(val_df["session_id"]) | \
        set(train_df["session_id"]) & set(test_df["session_id"]) | \
        set(val_df["session_id"]) & set(test_df["session_id"])
    if overlap:
        raise ValueError(f"Session leakage between splits: {len(overlap)} shared ids")

    scaler = StandardScaler().fit(train_df[list(FEATURES)])
    x_train = scaler.transform(train_df[list(FEATURES)])
    x_val = scaler.transform(val_df[list(FEATURES)])
    x_test = scaler.transform(test_df[list(FEATURES)])
    y_train, y_val, y_test = train_df["label"].to_numpy(), val_df["label"].to_numpy(), test_df["label"].to_numpy()

    candidates = []
    for class_weight in (None, "balanced"):
        model = LogisticRegression(class_weight=class_weight, solver="lbfgs", max_iter=5000,
                                   random_state=42)
        model.fit(x_train, y_train)
        val_probabilities = model.predict_proba(x_val)[:, 1]
        threshold, val_f1 = choose_threshold(y_val, val_probabilities)
        val_pr_auc = average_precision_score(y_val, val_probabilities)
        candidates.append({"class_weight": class_weight, "model": model,
                           "threshold": threshold, "val_f1": float(val_f1),
                           "val_pr_auc": float(val_pr_auc)})

    # Model selection is done on validation only (never on test).
    best = max(candidates, key=lambda c: (c["val_f1"], c["val_pr_auc"], -c["threshold"]))
    model = best["model"]
    threshold = best["threshold"]

    val_probabilities = model.predict_proba(x_val)[:, 1]
    test_probabilities = model.predict_proba(x_test)[:, 1]
    val_metrics = evaluate(y_val, val_probabilities, threshold)
    test_metrics = evaluate(y_test, test_probabilities, threshold)

    metadata = {
        "model": "logistic_regression",
        "artifact_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": list(FEATURES),
        "feature_labels_fr": FEATURE_LABELS_FR,
        "coefficients": [float(c) for c in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
        "scaler": {
            "mean": [float(m) for m in scaler.mean_],
            "scale": [float(s) for s in scaler.scale_],
        },
        "threshold": float(threshold),
        "threshold_selection": {
            "split": "val",
            "criterion": "F1 maximisé",
            "tie_break": "rappel, puis proximité de 0,5",
            "candidates": 197 + len(val_probabilities),
        },
        "cnn_confidence_fill": float(fill),
        "training": {
            "class_weight": best["class_weight"],
            "solver": "lbfgs",
            "max_iter": 5000,
            "random_state": 42,
            "n_train": int(len(train_df)),
            "class_weight_candidates": [
                {"class_weight": c["class_weight"], "val_f1": c["val_f1"],
                 "val_pr_auc": c["val_pr_auc"], "threshold": c["threshold"]}
                for c in candidates
            ],
        },
        "validation": val_metrics,
        "test": test_metrics,
        "splits": {name: int((data["split"] == name).sum()) for name in SPLITS},
        "provenance": {
            "scores": "models/rnn/rnn_scores.parquet",
            "events": "outputs/rnn_v5/sessions_labeled.parquet",
            "label_column": "label",
            "split_column": "split",
            "presented_items": "unique PAY item ids (same default as the demo app)",
            "cnn_confidence": (
                "deterministic label-independent item-hash proxy in [0.55, 0.99]; the RPC dataset "
                "does not provide per-session checkout images. At inference, the real CNN "
                "confidence is used when crops are supplied, otherwise cnn_confidence_fill."
            ),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "decision_meta.json", "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    audit = data[["session_id", "split", "label", *FEATURES]].copy()
    audit["probability"] = np.nan
    audit.loc[audit["split"] == "train", "probability"] = model.predict_proba(x_train)[:, 1]
    audit.loc[audit["split"] == "val", "probability"] = val_probabilities
    audit.loc[audit["split"] == "test", "probability"] = test_probabilities
    audit["alert"] = (audit["probability"] >= threshold).astype(int)
    audit.to_parquet(out_dir / "decision_features.parquet", index=False)

    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.scores.is_file():
        print(f"Missing scores file: {args.scores}", file=sys.stderr)
        return 1
    if not args.events.is_file():
        print(f"Missing events file: {args.events}", file=sys.stderr)
        return 1

    data, fill = build_dataset(args.scores, args.events)
    metadata = train(data, fill, args.out_dir)

    print("Decision model trained — zero-leakage protocol respected")
    print(f"  features      : {', '.join(metadata['features'])}")
    print(f"  class_weight  : {metadata['training']['class_weight']}")
    print(f"  threshold     : {metadata['threshold']:.4f} (chosen on validation)")
    print(f"  validation    : F1={metadata['validation']['f1']:.4f}  "
          f"PR-AUC={metadata['validation']['pr_auc']:.4f}  "
          f"recall={metadata['validation']['recall']:.4f}")
    print(f"  test          : F1={metadata['test']['f1']:.4f}  "
          f"PR-AUC={metadata['test']['pr_auc']:.4f}  "
          f"precision={metadata['test']['precision']:.4f}  "
          f"recall={metadata['test']['recall']:.4f}  "
          f"false alarms={metadata['test']['false_alarm_rate']:.2%}")
    print(f"  metadata      : {args.out_dir / 'decision_meta.json'}")
    print(f"  audit parquet : {args.out_dir / 'decision_features.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
