"""Final decision: VALIDATED or ALERT.

The decision is 100% deterministic and belongs to the trained logistic regression. The LLM agents
in ``agents/`` may explain it, but they can never change it. The model combines the RNN path risk,
the basket consistency measured at checkout, the CNN confidence and two context features.

The metadata (coefficients, intercept, scaler, threshold, metrics) lives in
``models/decision/decision_meta.json`` and was produced by ``scripts/train_decision.py`` with a
strict train/val/test protocol: fit on train, threshold on validation, final evaluation on test.
"""
import json
import math
from dataclasses import dataclass
from functools import lru_cache

from .cnn import CheckoutView
from .config import DECISION_DIR
from .visit import VisitSummary

VALIDATED, ALERT = "VALIDATED", "ALERT"
LOGISTIC_REGRESSION_AVAILABLE = True


@dataclass(frozen=True)
class Decision:
    outcome: str                     # VALIDATED or ALERT
    source: str                      # "logistic_regression" or "provisional_rule"
    reason: str                      # one sentence, in French
    probability: float | None = None  # model probability of anomaly, in [0, 1]
    threshold: float | None = None    # alert threshold used for this decision


@lru_cache(maxsize=1)
def metadata() -> dict:
    """Trained coefficients and scaler, loaded once."""
    with open(DECISION_DIR / "decision_meta.json") as f:
        return json.load(f)


def reliability() -> dict:
    """Final test-set metrics of the decision model (for the Decision Agent's facts)."""
    return metadata()["test"]


def features(path_risk: float, visit: VisitSummary, checkout: CheckoutView) -> dict:
    """Inputs of the logistic regression, as planned in the project design."""
    presented = set(checkout.item_ids)
    return {
        "rnn_risk": path_risk,
        "items_picked_not_presented": sum(1 for i in visit.picked if i not in presented),
        "cnn_confidence": checkout.confidence,
        "cart_total": sum(visit.items[i].price for i in checkout.item_ids if i in visit.items),
        "visit_duration_min": visit.duration_min,
    }


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exponential = math.exp(z)
    return exponential / (1.0 + exponential)


def _standardised_vector(inputs: dict, meta: dict) -> tuple[list[float], list[str]]:
    names = meta["features"]
    fill = float(meta.get("cnn_confidence_fill", 0.0))
    values = []
    for name in names:
        value = inputs.get(name)
        values.append(fill if value is None else float(value))
    means, scales = meta["scaler"]["mean"], meta["scaler"]["scale"]
    return [(value - mean) / scale for value, mean, scale in zip(values, means, scales)], names


def _main_factors(standardised: list[float], names: list[str], meta: dict,
                  outcome: str, count: int = 2) -> list[str]:
    """The features that push hardest toward the decision actually taken."""
    contributions = [coefficient * value
                     for coefficient, value in zip(meta["coefficients"], standardised)]
    if outcome == ALERT:
        ranked = sorted((item for item in zip(names, contributions) if item[1] > 0),
                        key=lambda item: item[1], reverse=True)
    else:
        ranked = sorted((item for item in zip(names, contributions) if item[1] < 0),
                        key=lambda item: item[1])
    if not ranked:
        ranked = sorted(zip(names, contributions), key=lambda item: abs(item[1]), reverse=True)
    labels = meta["feature_labels_fr"]
    return [labels[name] for name, contribution in ranked[:count] if abs(contribution) > 1e-9]


def _logistic_regression(inputs: dict) -> Decision:
    """Deterministic logistic-regression decision. This is the only source of ALERT/VALIDATED."""
    meta = metadata()
    standardised, names = _standardised_vector(inputs, meta)
    logit = float(meta["intercept"]) + sum(
        coefficient * value
        for coefficient, value in zip(meta["coefficients"], standardised)
    )
    probability = _sigmoid(logit)
    threshold = float(meta["threshold"])
    outcome = ALERT if probability >= threshold else VALIDATED

    factors = _main_factors(standardised, names, meta, outcome)
    if outcome == ALERT:
        factor_text = f" Facteurs principaux : {', '.join(factors)}." if factors else ""
        reason = (f"Régression logistique : probabilité d'anomalie {probability:.0%}, "
                  f"au-dessus du seuil de {threshold:.0%}.{factor_text}")
    else:
        factor_text = f" Éléments rassurants : {', '.join(factors)}." if factors else ""
        reason = (f"Régression logistique : probabilité d'anomalie {probability:.0%}, "
                  f"sous le seuil de {threshold:.0%}.{factor_text}")
    return Decision(outcome, "logistic_regression", reason, probability, threshold)


def decide(path_risk: float, threshold: float, visit: VisitSummary, checkout: CheckoutView) -> Decision:
    if LOGISTIC_REGRESSION_AVAILABLE:
        return _logistic_regression(features(path_risk, visit, checkout))
    if path_risk >= threshold:
        return Decision(ALERT, "provisional_rule",
                        f"Risque du parcours {path_risk:.0%}, au-dessus du seuil d'alerte de {threshold:.0%}.",
                        path_risk, threshold)
    return Decision(VALIDATED, "provisional_rule",
                    f"Risque du parcours {path_risk:.0%}, sous le seuil d'alerte de {threshold:.0%}.",
                    path_risk, threshold)
