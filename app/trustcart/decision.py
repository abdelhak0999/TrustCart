"""Final decision: VALIDATED or ALERT.

The logistic regression (teammate) is NOT TRAINED YET. Until then a provisional rule decides from
the RNN risk alone, with the threshold chosen on validation data.

To plug in the trained model: set LOGISTIC_REGRESSION_AVAILABLE = True and implement
`_logistic_regression` on the inputs built by `features`.
"""
from dataclasses import dataclass

from .cnn import CheckoutView
from .visit import VisitSummary

VALIDATED, ALERT = "VALIDATED", "ALERT"
LOGISTIC_REGRESSION_AVAILABLE = False


@dataclass(frozen=True)
class Decision:
    outcome: str        # VALIDATED or ALERT
    source: str         # "provisional_rule" or "logistic_regression"
    reason: str         # one sentence, in French


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


def _logistic_regression(inputs: dict) -> Decision:
    raise NotImplementedError("Plug the trained logistic regression in here")


def decide(path_risk: float, threshold: float, visit: VisitSummary, checkout: CheckoutView) -> Decision:
    if LOGISTIC_REGRESSION_AVAILABLE:
        return _logistic_regression(features(path_risk, visit, checkout))
    if path_risk >= threshold:
        return Decision(ALERT, "provisional_rule",
                        f"Risque du parcours {path_risk:.0%}, au-dessus du seuil d'alerte de {threshold:.0%}.")
    return Decision(VALIDATED, "provisional_rule",
                    f"Risque du parcours {path_risk:.0%}, sous le seuil d'alerte de {threshold:.0%}.")
