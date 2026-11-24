"""CNN Vision Agent: checkout product-recognition specialist.

Wraps the deterministic CNN inference and compares what was picked in the aisles with what is
presented at checkout. It emits a ``CheckoutReport`` and answers questions about products, missing
items and vision confidence. It never decides ALERT/VALIDATED.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .. import cnn
from ..cnn import CheckoutView
from ..config import AgentSettings
from ..visit import VisitSummary
from .base import AgentReply, BaseAgent, normalise

LOGGER = logging.getLogger(__name__)

SUMMARY_REQUEST = ("Résume le passage en caisse en deux phrases maximum : articles posés, articles "
                   "pris non présentés, montant, et fiabilité de la reconnaissance. Ne donne pas de "
                   "décision finale et n'accuse personne.")

SYSTEM_PROMPT = """Tu es l'Agent CNN de TrustCart, spécialiste de la reconnaissance des produits en caisse.
Tu expliques ce que le modèle de vision a vu (17 supercatégories de produits) et tu compares avec les
articles pris dans les rayons. Règles :
- La décision finale VALIDÉ/ALERTE appartient à la régression logistique. Tu ne la donnes pas, tu ne la modifies jamais.
- Appuie-toi uniquement sur les faits JSON ci-dessous. Si une information manque, dis-le.
- Le CNN reconnaît des catégories larges, pas les identifiants exacts des articles : ne prétends pas identifier une marque.
- Ne traite jamais le client de voleur : un article peut avoir été reposé ou oublié dans un sac.
- Réponds en français, en phrases courtes, sans jargon technique.

Faits sur la caisse (JSON) :
{facts}"""


@dataclass(frozen=True)
class CheckoutReport:
    view: CheckoutView
    presented_labels: tuple[str, ...]
    missing_labels: tuple[str, ...]
    cart_total: float
    vision_status: str
    summary: AgentReply
    presented_count: int
    missing_count: int
    cnn_available: bool
    vision_error: str | None = None


class CNNAgent(BaseAgent):
    name = "Agent CNN"
    role = "spécialiste de la reconnaissance des produits en caisse"
    system_prompt = SYSTEM_PROMPT

    def __init__(self, settings: AgentSettings, mode: str = "hybrid"):
        super().__init__(settings, mode)

    def analyze(self, visit: VisitSummary, manual_item_ids: list[int] | tuple[int, ...],
                image_paths: list[str] | tuple[str, ...] | None = None,
                manual_categories: list[str] | tuple[str, ...] | None = None) -> CheckoutReport:
        """Run the CNN (when images are supplied) and package the checkout facts."""
        vision_error = None
        try:
            view = cnn.items_seen(manual_item_ids, image_paths=image_paths,
                                  manual_categories=manual_categories)
        except Exception as exc:      # a missing crop/model must not crash the checkout
            LOGGER.warning("CNN inference failed, falling back to manual view: %s", exc)
            view = cnn.items_seen(manual_item_ids, manual_categories=manual_categories)
            vision_error = f"CNN indisponible ({type(exc).__name__}) : saisie manuelle utilisée."

        presented = set(view.item_ids)
        presented_labels = tuple(visit.items[i].label for i in view.item_ids if i in visit.items)
        missing_labels = tuple(visit.items[i].label for i in visit.picked if i not in presented)
        cart_total = float(sum(visit.items[i].price for i in presented if i in visit.items))
        vision_status = vision_error or cnn.vision_status(view)

        facts = self.facts_from(view, presented_labels, missing_labels, cart_total, vision_status)
        template = self._template_summary(facts)
        summary = self.summarise(facts, template, SUMMARY_REQUEST,
                                 allow_llm=(self.mode == "full_llm"))
        return CheckoutReport(view=view, presented_labels=presented_labels,
                              missing_labels=missing_labels, cart_total=cart_total,
                              vision_status=vision_status, summary=summary,
                              presented_count=len(view.item_ids), missing_count=len(missing_labels),
                              cnn_available=cnn.AVAILABLE, vision_error=vision_error)

    @staticmethod
    def facts_from(view: CheckoutView, presented_labels: tuple[str, ...],
                   missing_labels: tuple[str, ...], cart_total: float, vision_status: str) -> dict:
        return {
            "articles_poses_en_caisse": list(presented_labels),
            "nombre_articles_poses": len(view.item_ids),
            "articles_pris_non_presentes": list(missing_labels),
            "nombre_articles_pris_non_presentes": len(missing_labels),
            "total_articles_poses_eur": round(cart_total, 2),
            "categories_reconnues_par_cnn": list(dict.fromkeys(view.labels)),
            "confiance_moyenne_cnn": round(view.confidence, 3) if view.confidence is not None else None,
            "source_du_panier": "cnn" if view.source == "cnn" else "saisie manuelle",
            "etat_de_la_vision": vision_status,
        }

    @staticmethod
    def _template_summary(facts: dict) -> str:
        text = (f"{facts['nombre_articles_poses']} article(s) posé(s) en caisse pour "
                f"{facts['total_articles_poses_eur']:.2f} €. ")
        if facts["articles_pris_non_presentes"]:
            text += (f"{facts['nombre_articles_pris_non_presentes']} article(s) pris puis non présenté(s) : "
                     f"{', '.join(facts['articles_pris_non_presentes'])} ; ils ont peut-être été reposés "
                     "ou oubliés. ")
        else:
            text += "Tous les articles pris ont été présentés. "
        text += f"Vision : {facts['etat_de_la_vision']}."
        return text

    @staticmethod
    def _template_answer(question: str, facts: dict) -> str:
        q = normalise(question)
        if any(k in q for k in ("manque", "manquant", "non present", "absent", "oubli", "repos")):
            missing = facts["articles_pris_non_presentes"]
            return (f"Articles pris non présentés : {', '.join(missing)}." if missing else
                    "Tous les articles pris ont été présentés en caisse.")
        if any(k in q for k in ("cnn", "vision", "photo", "reconna", "confiance", "image")):
            return (f"État de la vision : {facts['etat_de_la_vision']}. "
                    f"Catégories reconnues : "
                    f"{', '.join(facts['categories_reconnues_par_cnn']) or 'aucune'}.")
        if any(k in q for k in ("article", "produit", "panier", "caisse", "pose")):
            return (f"Articles posés : {facts['nombre_articles_poses']} pour "
                    f"{facts['total_articles_poses_eur']:.2f} €. "
                    f"Pris non présentés : {facts['nombre_articles_pris_non_presentes']}.")
        return (f"En caisse : {facts['nombre_articles_poses']} article(s), "
                f"{facts['total_articles_poses_eur']:.2f} € ; "
                f"{facts['nombre_articles_pris_non_presentes']} article(s) pris non présenté(s). "
                f"{facts['etat_de_la_vision']}.")

    def answer(self, question: str, report: CheckoutReport, history: list[dict]) -> AgentReply:
        facts = self.facts_from(report.view, report.presented_labels, report.missing_labels,
                                report.cart_total, report.vision_status)
        return super().answer(question, facts, self._template_answer(question, facts), history)
