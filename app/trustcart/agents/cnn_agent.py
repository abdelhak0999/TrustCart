"""CNN Vision Agent: checkout product-recognition specialist.

Photographs every product placed on the tray, runs the CNN on the photos and keeps as *presented*
only the items the CNN confirms (``cnn.verify_basket``). Those CNN-verified items are what the
decision model receives, so the CNN directly drives ``items_picked_not_presented`` and the cart
total. It emits a ``CheckoutReport`` and answers questions about products, missing
items and vision confidence. It never decides ALERT/VALIDATED.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

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
    """Output of the CNN Agent: items confirmed or rejected by the camera, missing items, total."""
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
    rejected_labels: tuple[str, ...] = ()


class CheckoutState(TypedDict, total=False):
    """LangGraph state of the CNN agent: ``infer`` fills the checkout facts, ``narrate`` the report."""
    visit: VisitSummary
    presented: list[int] | tuple[int, ...]
    image_paths: list[str] | tuple[str, ...] | None
    manual_categories: list[str] | tuple[str, ...] | None
    camera: bool
    swaps: dict[int, str] | None
    view: CheckoutView
    presented_labels: tuple[str, ...]
    missing_labels: tuple[str, ...]
    rejected_labels: tuple[str, ...]
    cart_total: float
    vision_status: str
    vision_error: str | None
    facts: dict
    report: CheckoutReport


class CNNAgent(BaseAgent):
    """CNN Agent: runs the camera check at checkout and explains what was seen or rejected."""
    name = "Agent CNN"
    role = "spécialiste de la reconnaissance des produits en caisse"
    system_prompt = SYSTEM_PROMPT

    def __init__(self, settings: AgentSettings, mode: str = "hybrid"):
        super().__init__(settings, mode)
        self.graph = self._build_graph()

    def _build_graph(self):
        """infer (CNN or manual basket) -> narrate (template or LLM summary)."""
        builder = StateGraph(CheckoutState)
        builder.add_node("infer", self._infer)
        builder.add_node("narrate", self._narrate)
        builder.add_edge(START, "infer")
        builder.add_edge("infer", "narrate")
        builder.add_edge("narrate", END)
        return builder.compile(name="cnn_agent")

    def _infer(self, state: CheckoutState) -> dict:
        """Run the CNN (when images are supplied) and compute the checkout facts."""
        visit = state["visit"]
        manual_item_ids = state["presented"]
        manual_categories = state.get("manual_categories")
        vision_error = None
        image_paths = state.get("image_paths")
        if image_paths is None and state.get("camera", True) and cnn.AVAILABLE:
            image_paths = cnn.camera_photos(manual_item_ids, state.get("swaps"))
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
        rejected_labels = tuple(visit.items[i].label if i in visit.items else f"Article {i}"
                                for i in view.rejected_ids)
        cart_total = float(sum(visit.items[i].price for i in presented if i in visit.items))
        vision_status = vision_error or cnn.vision_status(view)
        return {"view": view, "presented_labels": presented_labels, "missing_labels": missing_labels,
                "rejected_labels": rejected_labels,
                "cart_total": cart_total, "vision_status": vision_status, "vision_error": vision_error,
                "facts": self.facts_from(view, presented_labels, missing_labels, cart_total,
                                         vision_status, rejected_labels)}

    def _narrate(self, state: CheckoutState) -> dict:
        facts = state["facts"]
        summary = self.summarise(facts, self._template_summary(facts), SUMMARY_REQUEST,
                                 allow_llm=(self.mode == "full_llm"))
        view = state["view"]
        return {"report": CheckoutReport(
            view=view, presented_labels=state["presented_labels"],
            missing_labels=state["missing_labels"], cart_total=state["cart_total"],
            vision_status=state["vision_status"], summary=summary,
            presented_count=len(view.item_ids), missing_count=len(state["missing_labels"]),
            cnn_available=cnn.AVAILABLE, vision_error=state["vision_error"],
            rejected_labels=state["rejected_labels"])}

    def analyze(self, visit: VisitSummary, manual_item_ids: list[int] | tuple[int, ...],
                image_paths: list[str] | tuple[str, ...] | None = None,
                manual_categories: list[str] | tuple[str, ...] | None = None,
                camera: bool = True, swaps: dict[int, str] | None = None) -> CheckoutReport:
        """Run the agent graph; the decision is not taken here.

        With ``camera`` (default) and no explicit ``image_paths``, every declared item is
        photographed (``cnn.camera_photos``) and the CNN confirms it or not.
        """
        return self.graph.invoke({"visit": visit, "presented": manual_item_ids,
                                  "image_paths": image_paths,
                                  "manual_categories": manual_categories,
                                  "camera": camera, "swaps": swaps})["report"]

    @staticmethod
    def facts_from(view: CheckoutView, presented_labels: tuple[str, ...],
                   missing_labels: tuple[str, ...], cart_total: float, vision_status: str,
                   rejected_labels: tuple[str, ...] = ()) -> dict:
        """Facts the CNN Agent is allowed to talk about (built only from the CNN outputs)."""
        return {
            "articles_poses_en_caisse": list(presented_labels),
            "articles_declares_non_reconnus_par_la_camera": list(rejected_labels),
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
        if facts["articles_declares_non_reconnus_par_la_camera"]:
            text += ("Non confirmé(s) par la caméra : "
                     f"{', '.join(facts['articles_declares_non_reconnus_par_la_camera'])}. ")
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
            rejected = facts["articles_declares_non_reconnus_par_la_camera"]
            return (f"État de la vision : {facts['etat_de_la_vision']}. "
                    + (f"Non confirmé(s) par la caméra : {', '.join(rejected)}. " if rejected else "")
                    + 
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
        """Answer a staff question about the items, the photos or the camera."""
        facts = self.facts_from(report.view, report.presented_labels, report.missing_labels,
                                report.cart_total, report.vision_status, report.rejected_labels)
        return super().answer(question, facts, self._template_answer(question, facts), history)
