#!/usr/bin/env python3
"""Train the TrustCart decision model (logistic regression) and export its metadata.

Zero-leakage protocol
---------------------
1. Features are derived only from the raw session events (no label is ever used).
2. The existing ``split`` column of ``models/rnn/rnn_scores.parquet`` is respected verbatim:
   train -> fit the scaler and the classifier; val -> choose the alert threshold and the
   class-weight variant; test -> final, untouched evaluation.
3. The test split is never used for any choice.

The features are taken from ``app/trustcart/decision.py:features``:
    rnn_risk, items_picked_not_presented, cart_total, visit_duration_min, items_rejected_by_cnn

Checkout-camera simulation (v3)
-------------------------------
At checkout the CNN photographs every declared item and confirms it or not
(``app/trustcart/cnn.py:verify_basket``). The training sessions have no photos, so the camera is
simulated, separately inside each split and with a fixed seed:
  * honest camera noise: every paid item of every session is rejected with probability
    ``1 - CNN test accuracy`` (the CNN's measured error rate);
  * substitution fraud: a ``--substitution-rate`` fraction of honest sessions is cloned with
    1 (80 %) or 2 (20 %) paid items swapped for another product. Label 1, pattern "substitution".
    The RNN score of the clone is the original one: the path is honest, only the camera can see it.
A rejected item is treated exactly as in the app: not presented, removed from the cart total.

``cnn_confidence`` is deliberately NOT a feature. The RetailRocket sessions have no checkout photos,
so a real CNN confidence cannot be computed for training; v1 used a label-independent hash proxy,
which taught the model nothing (coefficient ~0.05) and created a train/inference mismatch once the
real CNN (test accuracy 0.994, confidence ~1.0) was plugged in. The CNN's contribution to the
decision is the list of items presented at checkout (-> items_picked_not_presented); its confidence
is shown to staff, not used as a risk signal.

Run with:
    .venv-app/bin/python scripts/train_decision.py
"""
from __future__ import annotations

import argparse
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

DEFAULT_CNN_META = ROOT / "models" / "cnn" / "cnn_meta.json"

FEATURES = ("rnn_risk", "items_picked_not_presented", "cart_total", "visit_duration_min",
            "items_rejected_by_cnn")
FEATURE_LABELS_FR = {
    "rnn_risk": "risque du parcours (RNN)",
    "items_picked_not_presented": "articles pris non présentés",
    "cart_total": "montant du panier",
    "visit_duration_min": "durée de la visite",
    "items_rejected_by_cnn": "articles non confirmés par la caméra (CNN)",
}
SUBSTITUTION_ID_OFFSET = 1_000_000_000
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- feature engineering

def session_features(events: pd.DataFrame) -> pd.DataFrame:
    """Derive the decision features from raw session events, one row per session."""
    rows = []
    for session_id, group in events.groupby("session_id", sort=False):
        picked = tuple(dict.fromkeys(group.loc[group["action"] == "PICK", "itemid"]))
        paid = tuple(group.loc[group["action"] == "PAY", "itemid"])
        price_by_item = group.drop_duplicates("itemid").set_index("itemid")["price"]
        presented = tuple(dict.fromkeys(paid))
        picked_set = set(picked)

        rows.append({
            "session_id": int(session_id),
            "items_picked_not_presented": sum(1 for item in picked if item not in set(presented)),
            "cart_total": float(sum(price_by_item.get(item, 0.0) for item in presented)),
            "visit_duration_min": float(group["dt_sec"].sum() / 60.0),
            # (price, was picked in the aisles) of every declared item, for the camera simulation
            "declared": [(float(price_by_item.get(item, 0.0)), item in picked_set) for item in presented],
        })
    return pd.DataFrame(rows)


def _reject(row: dict, rejected: list[tuple[float, bool]]) -> dict:
    """Apply camera rejections exactly as the app does: not presented, out of the cart total."""
    row = dict(row)
    row["items_rejected_by_cnn"] = len(rejected)
    row["items_picked_not_presented"] += sum(1 for _, picked in rejected if picked)
    row["cart_total"] = max(0.0, row["cart_total"] - sum(price for price, _ in rejected))
    return row


def simulate_camera(data: pd.DataFrame, cnn_error_rate: float, substitution_rate: float,
                    seed: int = 42) -> pd.DataFrame:
    """Honest camera noise on every session + substitution-fraud clones, split by split."""
    rng = np.random.default_rng(seed)
    out = []
    for split in SPLITS:
        part = data[data["split"] == split]
        for row in part.to_dict("records"):
            declared = row["declared"]
            noisy = [item for item in declared if rng.random() < cnn_error_rate]
            out.append(_reject({**row, "pattern_full": row["pattern"]}, noisy))

        honest = part[(part["label"] == 0) & (part["declared"].map(len) > 0)]
        clones = honest.sample(frac=substitution_rate, random_state=int(rng.integers(1 << 31)))
        for row in clones.to_dict("records"):
            declared = row["declared"]
            k = min(len(declared), 2 if rng.random() < 0.2 else 1)
            swapped = [declared[i] for i in rng.choice(len(declared), size=k, replace=False)]
            clone = _reject({**row, "pattern_full": "substitution"}, swapped)
            clone.update(session_id=SUBSTITUTION_ID_OFFSET + int(row["session_id"]), label=1)
            out.append(clone)
    return pd.DataFrame(out)


def build_dataset(scores_path: Path, events_path: Path, cnn_meta_path: Path,
                  substitution_rate: float) -> pd.DataFrame:
    scores = pd.read_parquet(scores_path, columns=["session_id", "label", "split", "pattern",
                                                   "rnn_score"])
    events = pd.read_parquet(events_path, columns=["session_id", "action", "itemid", "price", "dt_sec"])

    features = session_features(events)
    data = scores.merge(features, on="session_id", how="left", validate="one_to_one")
    if data["split"].isna().any() or data["rnn_score"].isna().any():
        raise ValueError("Missing values after merging scores and events")

    unknown_splits = set(data["split"].unique()) - set(SPLITS)
    if unknown_splits:
        raise ValueError(f"Unexpected split values: {unknown_splits}")

    data["rnn_risk"] = data["rnn_score"].astype(float)
    cnn_error_rate = 1.0 - float(json.loads(cnn_meta_path.read_text())["test_metrics"]["accuracy"])
    data = simulate_camera(data, cnn_error_rate, substitution_rate)
    data.attrs["cnn_error_rate"] = cnn_error_rate
    data.attrs["substitution_rate"] = substitution_rate

    if data[list(FEATURES)].isna().any().any():
        raise ValueError("Feature matrix still contains NaN values")
    return data


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

def train(data: pd.DataFrame, out_dir: Path) -> dict:
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
    alerts = test_probabilities >= threshold
    patterns = test_df["pattern_full"].fillna("none").to_numpy()
    test_metrics["recall_by_pattern"] = {
        str(name): float(alerts[(patterns == name) & (y_test == 1)].mean())
        for name in sorted(set(patterns[y_test == 1]))
    }
    original = test_df["session_id"].to_numpy() < SUBSTITUTION_ID_OFFSET
    test_metrics["without_substitution_clones"] = evaluate(
        y_test[original], test_probabilities[original], threshold)

    metadata = {
        "model": "logistic_regression",
        "artifact_version": 3,
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
            "camera_simulation": {
                "cnn_error_rate": data.attrs["cnn_error_rate"],
                "substitution_rate": data.attrs["substitution_rate"],
                "substitution_items": "1 (80 %) or 2 (20 %) declared items swapped",
                "seed": 42,
                "note": ("simulated per split: honest rows get CNN-noise rejections, substitution "
                         "clones of honest sessions are labelled 1 and keep their RNN score"),
            },
            "cnn_confidence": (
                "not a feature: no checkout photos exist for the training sessions, so it cannot "
                "be learned honestly. The CNN feeds the decision through the presented items."
            ),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "decision_meta.json", "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    audit = data[["session_id", "split", "label", "pattern_full", *FEATURES]].copy()
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
    parser.add_argument("--cnn-meta", type=Path, default=DEFAULT_CNN_META)
    parser.add_argument("--substitution-rate", type=float, default=0.05,
                        help="fraction of honest sessions cloned as substitution fraud, per split")
    args = parser.parse_args()

    if not args.scores.is_file():
        print(f"Missing scores file: {args.scores}", file=sys.stderr)
        return 1
    if not args.events.is_file():
        print(f"Missing events file: {args.events}", file=sys.stderr)
        return 1

    data = build_dataset(args.scores, args.events, args.cnn_meta, args.substitution_rate)
    metadata = train(data, args.out_dir)

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
    print(f"  coefficients  : " + ", ".join(
        f"{name}={coef:+.3f}" for name, coef in zip(metadata["features"], metadata["coefficients"])))
    print(f"  test recall by pattern: " + ", ".join(
        f"{name} {value:.0%}" for name, value in metadata["test"]["recall_by_pattern"].items()))
    base = metadata["test"]["without_substitution_clones"]
    print(f"  test without clones: F1={base['f1']:.4f} recall={base['recall']:.4f} "
          f"false alarms={base['false_alarm_rate']:.2%}")
    print(f"  metadata      : {args.out_dir / 'decision_meta.json'}")
    print(f"  audit parquet : {args.out_dir / 'decision_features.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
