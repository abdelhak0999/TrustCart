"""Score a TrustCart customer session with the trained RNN (exported by rnn.ipynb, Phase 13)."""
import json

import numpy as np
from tensorflow import keras


class SessionScorer:
    """Raw session events -> anomaly score, with exactly the preprocessing used in training."""

    def __init__(self, model_path, encoders_path, meta_path):
        self.model = keras.models.load_model(model_path)
        with open(encoders_path) as f:
            self.enc = json.load(f)
        with open(meta_path) as f:
            meta = json.load(f)
        calibration = meta["calibration"]
        self.cal_a, self.cal_b = calibration["a"], calibration["b"]
        self.threshold = float(calibration["threshold_calibrated"])     # threshold on the calibrated score

    def encode(self, actions, aisles, dt_sec, price):
        """actions: LOOK/PICK/PAY, aisles: store aisle (0 = unknown), dt_sec: seconds since previous action."""
        max_len, pad, n_aisles = self.enc["max_len"], self.enc["pad_id"], self.enc["n_aisles"]
        actions = list(actions)[-max_len:]
        if not actions:
            raise ValueError("Cannot score an empty session")
        aisles = np.asarray(aisles, dtype="int64")[-max_len:]
        dt_sec = np.clip(np.asarray(dt_sec, dtype="float32")[-max_len:], 0, None).copy()
        price = np.clip(np.asarray(price, dtype="float32")[-max_len:], 0, None)
        dt_sec[0] = 0.0
        aisles = np.where((aisles >= 0) & (aisles < n_aisles), aisles, 0)

        n = len(actions)
        action_ids = np.full((1, max_len), pad, dtype="int32")
        aisle_ids = np.full((1, max_len), pad, dtype="int32")
        numeric = np.zeros((1, max_len, len(self.enc["num_features"])), dtype="float32")
        action_ids[0, :n] = [self.enc["action_vocab"][a] for a in actions]
        aisle_ids[0, :n] = aisles + self.enc["aisle_offset"]
        numeric[0, :n, 0] = np.log1p(dt_sec)
        numeric[0, :n, 1] = np.log1p(price)
        last_step = np.zeros((1, max_len), dtype="float32")
        last_step[0, n - 1] = 1.0
        return {"action": action_ids, "aisle": aisle_ids, "numeric": numeric, "last_step": last_step}

    def raw_score(self, actions, aisles, dt_sec, price):
        """Uncalibrated model output."""
        inputs = self.encode(actions, aisles, dt_sec, price)
        return float(np.asarray(self.model(inputs, training=False)).ravel()[0])

    def score(self, actions, aisles, dt_sec, price):
        """Calibrated theft risk in [0, 1] (Platt scaling fitted on validation)."""
        p = min(max(self.raw_score(actions, aisles, dt_sec, price), 1e-6), 1 - 1e-6)
        return float(1 / (1 + np.exp(-(self.cal_a * np.log(p / (1 - p)) + self.cal_b))))

    def is_anomaly(self, actions, aisles, dt_sec, price):
        """True when the calibrated risk of the session reaches the alert threshold."""
        return self.score(actions, aisles, dt_sec, price) >= self.threshold
