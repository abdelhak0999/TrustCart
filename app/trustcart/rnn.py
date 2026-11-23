"""RNN (LSTM) path-risk model: a thin wrapper around the scorer exported by notebooks/rnn.ipynb."""
import importlib.util
import json
import os
from functools import lru_cache

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")      # hide TensorFlow start-up logs

from .config import RNN_DIR  # noqa: E402

AVAILABLE = True


@lru_cache(maxsize=1)
def load_scorer():
    """Load the trained model once; preprocessing and calibration come with it."""
    spec = importlib.util.spec_from_file_location("rnn_inference", RNN_DIR / "rnn_inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SessionScorer(str(RNN_DIR / "rnn_model.keras"), str(RNN_DIR / "encoders.json"),
                                str(RNN_DIR / "rnn_meta.json"))


@lru_cache(maxsize=1)
def metadata() -> dict:
    with open(RNN_DIR / "rnn_meta.json") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def cue_thresholds() -> dict:
    """Thresholds used to build the simulated thefts (Phase 6), reused to describe a path in words."""
    with open(RNN_DIR / "injection_config.json") as f:
        return json.load(f)


def path_risk(events: list[dict]) -> float:
    """Calibrated theft risk of a visit, between 0 and 1."""
    return load_scorer().score([e["action"] for e in events], [e["aisle"] for e in events],
                               [e["dt_sec"] for e in events], [e["price"] for e in events])


def alert_threshold() -> float:
    """Risk above which the visit is flagged (chosen on validation data, best F1)."""
    return load_scorer().threshold


def reliability() -> dict:
    """Test-set figures averaged over 4 trainings (Phase 14)."""
    return metadata()["robustness"]["seeds"]["mean"]
