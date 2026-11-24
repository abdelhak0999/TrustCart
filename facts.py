"""Single source of truth for the facts agents may talk about, plus the fixed customer wording.

The facts dict is built exclusively from deterministic model outputs. Agents receive it and may
summarise it; they cannot add facts, and they cannot change the decision.
"""
from dataclasses import dataclass

from .. import decision as decision_module
from .. import rnn
from ..cnn import CheckoutView
from ..decision import ALERT, Decision
from ..visit import VisitSummary, timeline


@dataclass(frozen=True)
class CustomerMessage:
    title: str
    body: str


def build_facts(decision: Decision, path_risk: float, threshold: float, visit: VisitSummary,
                checkout: CheckoutView, events: list[dict]) -> dict:
    """Everything the agents may say, and nothing else: their only source of truth."""
    presented = set(checkout.item_ids)
    reliability = rnn.reliability()
    cnn_labels = list(dict.fromkeys(checkout.labels))
    facts = {
        "decision": "ALERTE" if decision.outcome == ALERT else "VALIDÉ",
        "raison_de_la_decision": decision.reason,
        "qui_decide": ("régression logistique (modèle déterministe)" if decision.source == "logistic_regression"
                       else "règle provisoire sur le risque du parcours"),
        "probabilite_anomalie_pourcent": (
            round(decision.probability * 100) if decision.probability is not None else None),
        "seuil_decision_pourcent": (
            round(decision.threshold * 100) if decision.threshold is not None else round(threshold * 100)),
        "risque_du_parcours_pourcent": round(path_risk * 100),
        "seuil_alerte_pourcent": round(threshold * 100),
        "indices_dans_le_parcours": list(visit.cues),
        "articles_pris": [visit.items[i].label for i in visit.picked],
        "articles_poses_en_caisse": [visit.items[i].label for i in checkout.item_ids if i in visit.items],
        "articles_pris_non_presentes": [visit.items[i].label for i in visit.picked if i not in presented],
        "total_articles_poses_eur": round(sum(visit.items[i].price for i in presented if i in visit.items), 2),
        "duree_visite_min": round(visit.duration_min, 1),
        "plus_longue_pause_min": round(visit.longest_pause_min, 1),
        "reconnaissance_produits": (
            f"CNN : confiance moyenne {checkout.confidence:.0%} ; catégories reconnues : "
            f"{', '.join(cnn_labels) if cnn_labels else 'aucune'}"
            if checkout.source == "cnn" and checkout.confidence is not None
            else ("saisie manuelle : aucune photo de caisse analysée"
                  + (f" ; catégories déclarées : {', '.join(cnn_labels)}" if cnn_labels else ""))),
        "categories_reconnues_par_cnn": cnn_labels,
        "confiance_moyenne_cnn": (round(checkout.confidence, 3)
                                  if checkout.confidence is not None else None),
        "source_du_panier": "cnn" if checkout.source == "cnn" else "saisie manuelle",
        "fiabilite_du_modele_de_parcours": (
            f"sur des données de test, {reliability['recall']:.0%} des vols simulés détectés, "
            f"{reliability['false_alarm_rate']:.2%} de clients honnêtes signalés à tort"),
        "chronologie": timeline(events),
    }
    if decision_module.LOGISTIC_REGRESSION_AVAILABLE:
        metrics = decision_module.reliability()
        facts["fiabilite_du_modele_de_decision"] = (
            f"sur des données de test, précision {metrics['precision']:.0%}, "
            f"rappel {metrics['recall']:.0%}, {metrics['false_alarm_rate']:.2%} de fausses alertes")
    return facts


def customer_message(decision: Decision, facts: dict) -> CustomerMessage:
    """Fixed wording on purpose: the customer screen must never vary or accuse."""
    if decision.outcome == ALERT:
        return CustomerMessage("Un instant, s'il vous plaît",
                               "Un conseiller arrive pour vous aider à finaliser votre passage en caisse.")
    n = len(facts["articles_poses_en_caisse"])
    return CustomerMessage("Votre panier est validé",
                           f"{n} article{'s' if n > 1 else ''}, {facts['total_articles_poses_eur']:.2f} €. "
                           "Présentez votre carte pour payer. Bonne journée !")
