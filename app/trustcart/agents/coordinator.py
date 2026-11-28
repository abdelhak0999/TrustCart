"""Coordinator / Staff Liaison Agent.

LangGraph orchestrator over the three model agents (each one is itself a compiled LangGraph
``infer/decide -> narrate`` graph). Two graphs:

  * checkout: START -> visit -> (rnn_agent || cnn_agent) -> decision_agent -> briefing -> END
  * Q&A:      START -> route -> Send(rnn_qa | cnn_qa | decision_qa)* -> synthesise -> END
                              `-> coordinator_qa -> END

It keeps the fixed neutral customer message, writes the staff
briefing and routes staff questions to the right specialist. It never overrides a model decision:
the ``DecisionReport`` produced by the logistic regression is final.

Modes (``AGENT_MODE``):
  * ``hybrid`` (default): deterministic reports; the Coordinator uses the LLM for the briefing and
    specialist agents use it only when a staff question is routed to them.
  * ``full_llm``: the three model agents also narrate their own reports, then the Coordinator
    synthesises. More LLM calls, richer demo.
  * ``templates``: no LLM at all; every answer comes from the offline French templates.
"""
from __future__ import annotations

import logging
import operator
import re
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .. import rnn
from ..config import AgentSettings, agent_mode, agent_settings
from ..decision import ALERT
from ..visit import VisitSummary, summarize
from .base import AgentReply, BaseAgent, normalise
from .cnn_agent import CNNAgent, CheckoutReport
from .decision_agent import DecisionAgent, DecisionReport
from .facts import CustomerMessage, customer_message
from .rnn_agent import RNNAgent, PathReport

LOGGER = logging.getLogger(__name__)

ROUTER_PROMPT = """Tu es le routeur de TrustCart. Choisis quel(s) agent(s) doivent répondre à la question du
personnel. Agents disponibles :
- rnn : parcours du client en magasin, pauses, durée, vitesse, rayons, risque comportemental ;
- cnn : produits posés en caisse, articles manquants, reconnaissance visuelle, montant du panier ;
- decision : décision VALIDÉ/ALERTE, seuil, probabilité, raisons, poids des signaux, fiabilité de la décision ;
- coordinator : conduite à tenir avec le client, salutations, questions générales ou hors sujet.
Réponds UNIQUEMENT par un ou plusieurs noms parmi rnn, cnn, decision, coordinator, séparés par des
virgules, sans autre texte. N'utilise coordinator que seul."""

BRIEFING_REQUEST = ("Rédige le message affiché sur la tablette du personnel pour cette visite : trois phrases "
                    "au maximum, qui donnent la décision déjà prise, les principaux indices et, en cas "
                    "d'alerte, comment aborder le client avec discrétion. N'accuse jamais le client.")

SYSTEM_PROMPT = """Tu es le Coordinateur TrustCart, l'assistant du personnel en caisse autonome.
Tu synthétises les rapports de trois agents spécialisés : l'Agent RNN (parcours), l'Agent CNN (produits
en caisse) et l'Agent Décision (régression logistique). Règles :
- La décision VALIDÉ/ALERTE est définitive et vient du modèle statistique. Tu ne la modifies jamais.
- Appuie-toi uniquement sur les faits et les rapports JSON ci-dessous. Si une information manque, dis-le.
- Ne traite jamais le client de voleur. Parle d'indices, et rappelle qu'un article reposé ou oublié reste possible.
- Réponds en français, en phrases courtes, sans jargon technique. Trois phrases maximum pour un briefing.

Faits et rapports (JSON) :
{facts}"""

ROUTE_KEYWORDS = {
    "rnn": ("parcours", "risque", "rnn", "rayon", "pause", "arret", "vitesse", "deplacement",
            "comportement", "trajectoire", "fiabilite", "fiable", "duree", "long", "lent", "rapide"),
    "cnn": ("article", "produit", "cnn", "vision", "caisse", "panier", "manquant", "presente",
            "confiance", "reconnaissance", "image", "photo", "pose", "reconnu"),
    "decision": ("decision", "seuil", "alerte", "valide", "logistique", "probabilite", "facteur",
                 "poids", "pourquoi", "raison", "justif", "fiabilite", "fiable"),
}
CONDUCT_KEYWORDS = ("aborder", "approcher", "faire", "conseil", "comment", "discret", "discretion",
                    "procedure", "expliquer au client")


def _normalise(text: str) -> str:
    return normalise(text)


@dataclass(frozen=True)
class AgentActivity:
    """One line of the agent trace shown to staff (who did what, with which backend)."""
    agent: str
    role: str
    status: str                 # "ok", "fallback" or "offline"
    summary: str
    source: str
    note: str | None
    metrics: dict


@dataclass(frozen=True)
class CheckoutResult:
    """Everything produced by one checkout: the three reports, customer message and briefing."""
    path: PathReport
    checkout: CheckoutReport
    decision: DecisionReport
    customer: CustomerMessage
    briefing: AgentReply
    activities: tuple[AgentActivity, ...]

    @property
    def outcome(self) -> str:
        """VALIDATED or ALERT, as decided by the logistic regression."""
        return self.decision.outcome

    @property
    def is_alert(self) -> bool:
        """True when the visit must be checked by staff."""
        return self.decision.outcome == ALERT


class OrchestratorState(TypedDict, total=False):
    """LangGraph state of the checkout graph (inputs, then each agent's report)."""
    events: list[dict]
    presented: list[int] | tuple[int, ...]
    image_paths: list[str] | tuple[str, ...] | None
    manual_categories: list[str] | tuple[str, ...] | None
    camera: bool
    swaps: dict[int, str] | None
    visit: VisitSummary
    path: PathReport
    checkout: CheckoutReport
    decision: DecisionReport
    result: CheckoutResult


class QAState(TypedDict, total=False):
    """LangGraph state of the staff Q&A graph (question, chosen routes, specialist replies)."""
    question: str
    result: CheckoutResult
    history: list[dict]
    routes: list[str]
    replies: Annotated[list[tuple[int, AgentReply]], operator.add]   # (route order, reply)
    answer: AgentReply


class CoordinatorAgent(BaseAgent):
    """Coordinator: orchestrates the three agents, writes the staff briefing and routes questions."""
    name = "Coordinateur"
    role = "orchestrateur et liaison avec le personnel"
    system_prompt = SYSTEM_PROMPT

    def __init__(self, settings: AgentSettings | None = None, mode: str | None = None):
        self.mode = mode or agent_mode()
        if settings is not None and settings.backend == "templates":
            self.mode = "templates"
        self.settings = settings or agent_settings("coordinator")
        super().__init__(self.settings, self.mode)
        self.rnn_agent = RNNAgent(settings or agent_settings("rnn"), self.mode)
        self.cnn_agent = CNNAgent(settings or agent_settings("cnn"), self.mode)
        self.decision_agent = DecisionAgent(settings or agent_settings("decision"), self.mode)
        self.checkout_graph = self._build_checkout_graph()
        self.qa_graph = self._build_qa_graph()

    # ------------------------------------------------------------------ orchestration

    def _build_checkout_graph(self):
        builder = StateGraph(OrchestratorState)
        builder.add_node("visit", self._visit_node)
        builder.add_node("rnn_agent", self._rnn_node)
        builder.add_node("cnn_agent", self._cnn_node)
        builder.add_node("decision_agent", self._decision_node)
        builder.add_node("briefing", self._briefing_node)
        builder.add_edge(START, "visit")
        builder.add_edge("visit", "rnn_agent")          # fan-out: both specialists run in parallel
        builder.add_edge("visit", "cnn_agent")
        builder.add_edge(["rnn_agent", "cnn_agent"], "decision_agent")   # join: waits for both
        builder.add_edge("decision_agent", "briefing")
        builder.add_edge("briefing", END)
        return builder.compile(name="trustcart_checkout")

    @staticmethod
    def _visit_node(state: OrchestratorState) -> dict:
        return {"visit": summarize(state["events"], rnn.cue_thresholds())}

    def _rnn_node(self, state: OrchestratorState) -> dict:
        return {"path": self.rnn_agent.analyze(state["events"], state["visit"])}

    def _cnn_node(self, state: OrchestratorState) -> dict:
        return {"checkout": self.cnn_agent.analyze(state["visit"], state["presented"],
                                                   state.get("image_paths"),
                                                   state.get("manual_categories"),
                                                   camera=state.get("camera", True),
                                                   swaps=state.get("swaps"))}

    def _decision_node(self, state: OrchestratorState) -> dict:
        return {"decision": self.decision_agent.evaluate(state["path"], state["checkout"],
                                                         state["events"])}

    def _briefing_node(self, state: OrchestratorState) -> dict:
        path, checkout, decision = state["path"], state["checkout"], state["decision"]
        customer = customer_message(decision.decision, decision.facts)
        briefing = self._briefing(path, checkout, decision)
        activities = self._activities(path, checkout, decision, briefing)
        return {"result": CheckoutResult(path=path, checkout=checkout, decision=decision,
                                         customer=customer, briefing=briefing,
                                         activities=activities)}

    def run_checkout(self, events: list[dict], presented: list[int] | tuple[int, ...],
                     image_paths: list[str] | tuple[str, ...] | None = None,
                     manual_categories: list[str] | tuple[str, ...] | None = None,
                     camera: bool = True, swaps: dict[int, str] | None = None) -> CheckoutResult:
        """Run the full deterministic pipeline, then let the Coordinator narrate it.

        ``presented`` are the items the customer places on the tray; with ``camera`` the CNN
        confirms each one from its photo before the decision model sees the basket.
        """
        return self.checkout_graph.invoke({
            "events": events, "presented": presented, "image_paths": image_paths,
            "manual_categories": manual_categories, "camera": camera, "swaps": swaps,
        })["result"]

    def _briefing(self, path: PathReport, checkout: CheckoutReport,
                  decision: DecisionReport) -> AgentReply:
        facts = dict(decision.facts)
        facts["rapport_rnn"] = path.summary.text
        facts["rapport_cnn"] = checkout.summary.text
        facts["rapport_decision"] = decision.summary.text
        template = self._template_briefing(decision)
        return self.summarise(facts, template, BRIEFING_REQUEST,
                              allow_llm=(self.mode in ("hybrid", "full_llm")))

    @staticmethod
    def _template_briefing(decision: DecisionReport) -> str:
        facts = decision.facts
        probability = facts.get("probabilite_anomalie_pourcent")
        threshold = facts.get("seuil_decision_pourcent")
        if facts["decision"] == "ALERTE":
            signals = []
            if facts["risque_du_parcours_pourcent"] >= facts["seuil_alerte_pourcent"]:
                signals.append(f"risque du parcours {facts['risque_du_parcours_pourcent']} %")
            if facts["indices_dans_le_parcours"]:
                signals.append(facts["indices_dans_le_parcours"][0])
            if facts["articles_pris_non_presentes"]:
                signals.append(f"{len(facts['articles_pris_non_presentes'])} article(s) pris non présenté(s)")
            if facts.get("articles_declares_non_reconnus_par_la_camera"):
                signals.append(f"{len(facts['articles_declares_non_reconnus_par_la_camera'])} "
                               "article(s) déclaré(s) non confirmé(s) par la caméra")
            first = (f"Vérification recommandée : décision ALERTE, probabilité d'anomalie "
                     f"{probability} % (seuil {threshold} %). ")
            second = ("Signaux : " + " ; ".join(signals) + ". " if signals else
                      "Le modèle combine plusieurs signaux faibles. ")
            return first + second + ("Abordez le client avec discrétion et proposez votre aide : "
                                     "un oubli reste possible.")
        first = (f"Rien à signaler : décision VALIDÉ, probabilité d'anomalie {probability} % "
                 f"(seuil {threshold} %). ")
        if facts["articles_pris_non_presentes"]:
            second = (f"{len(facts['articles_pris_non_presentes'])} article(s) pris puis non présenté(s), "
                      "probablement reposé(s) en rayon. ")
        else:
            second = "Le parcours est cohérent et tous les articles pris ont été présentés. "
        return first + second + "Vous pouvez laisser le client finaliser son paiement."

    # ------------------------------------------------------------------ Q&A routing

    def route(self, question: str) -> list[str]:
        """Return the specialist agents relevant to a staff question."""
        q = _normalise(question)
        routes = [name for name, keywords in ROUTE_KEYWORDS.items()
                  if any(keyword in q for keyword in keywords)]
        if routes:
            return routes
        if any(keyword in q for keyword in CONDUCT_KEYWORDS):
            return ["coordinator"]
        return ["coordinator"]

    def _build_qa_graph(self):
        builder = StateGraph(QAState)
        builder.add_node("route", self._route_node)
        builder.add_node("coordinator_qa", self._coordinator_qa_node)
        builder.add_node("rnn_qa", self._specialist_node("rnn"))
        builder.add_node("cnn_qa", self._specialist_node("cnn"))
        builder.add_node("decision_qa", self._specialist_node("decision"))
        builder.add_node("synthesise", self._synthesise_node)
        builder.add_edge(START, "route")
        builder.add_conditional_edges("route", self._dispatch,
                                      ["coordinator_qa", "rnn_qa", "cnn_qa", "decision_qa"])
        for specialist in ("rnn_qa", "cnn_qa", "decision_qa"):
            builder.add_edge(specialist, "synthesise")
        builder.add_edge("coordinator_qa", END)
        builder.add_edge("synthesise", END)
        return builder.compile(name="trustcart_qa")

    def llm_route(self, question: str) -> list[str] | None:
        """Ask the LLM which specialists should answer; None when unavailable or invalid."""
        if self._llm is None or self.mode == "templates":
            return None
        try:
            raw = self._llm.invoke([{"role": "system", "content": ROUTER_PROMPT},
                                    {"role": "user", "content": question}])
        except Exception as exc:      # any LLM failure -> keyword router
            LOGGER.warning("LLM router failed, using keywords: %s", type(exc).__name__)
            return None
        names = re.findall(r"[a-z]+", normalise(raw or ""))
        valid = ("rnn", "cnn", "decision", "coordinator")
        if not names or any(name not in valid for name in names):
            LOGGER.warning("LLM router returned an invalid answer %r, using keywords", raw)
            return None
        routes = [name for name in ("rnn", "cnn", "decision") if name in names]
        return routes or ["coordinator"]

    def _route_node(self, state: QAState) -> dict:
        """LLM routing first; deterministic keyword routing as the offline / invalid-answer fallback."""
        routes = self.llm_route(state["question"])
        return {"routes": routes if routes is not None else self.route(state["question"])}

    @staticmethod
    def _dispatch(state: QAState) -> list[Send] | str:
        if state["routes"] == ["coordinator"]:
            return "coordinator_qa"
        return [Send(f"{route}_qa", {**state, "routes": [route], "replies": []})
                for route in state["routes"]]

    def _specialist_node(self, route: str):
        agents = {"rnn": (self.rnn_agent, "path"), "cnn": (self.cnn_agent, "checkout"),
                  "decision": (self.decision_agent, "decision")}

        def node(state: QAState) -> dict:
            """Ask one specialist agent and tag its reply with the route order."""
            agent, report_attr = agents[route]
            reply = agent.answer(state["question"], getattr(state["result"], report_attr),
                                 state.get("history", []))
            return {"replies": [(list(ROUTE_KEYWORDS).index(route), reply)]}

        node.__name__ = f"{route}_qa"
        return node

    def _coordinator_qa_node(self, state: QAState) -> dict:
        return {"answer": self._coordinator_answer(state["question"], state["result"],
                                                   state.get("history", []))}

    def _synthesise_node(self, state: QAState) -> dict:
        replies = [reply for _, reply in sorted(state["replies"], key=lambda item: item[0])]
        if len(replies) == 1:
            return {"answer": replies[0]}
        return {"answer": self._synthesise(state["question"], state["result"], replies,
                                           state.get("history", []))}

    def answer(self, question: str, result: CheckoutResult, history: list[dict]) -> AgentReply:
        """Route a staff question, answer with the specialist(s), and synthesise if needed."""
        return self.qa_graph.invoke({"question": question, "result": result,
                                     "history": history, "replies": []})["answer"]

    def _coordinator_answer(self, question: str, result: CheckoutResult,
                            history: list[dict]) -> AgentReply:
        facts = dict(result.decision.facts)
        facts["rapport_rnn"] = result.path.summary.text
        facts["rapport_cnn"] = result.checkout.summary.text
        facts["rapport_decision"] = result.decision.summary.text
        template = self._template_answer(question, result)
        return super().answer(question, facts, template, history)

    def _synthesise(self, question: str, result: CheckoutResult, replies: list[AgentReply],
                    history: list[dict]) -> AgentReply:
        facts = dict(result.decision.facts)
        facts["question"] = question
        facts["reponses_des_specialistes"] = [
            {"agent": reply.agent_name, "reponse": reply.text} for reply in replies
        ]
        template = "Synthèse du coordinateur : " + " ".join(
            f"[{reply.agent_name}] {reply.text}" for reply in replies)
        if self.llm_available:
            return self._with_fallback(facts, history, question, template)
        return AgentReply(template, replies[0].source, self.name,
                          next((reply.note for reply in replies if reply.note), None))

    @staticmethod
    def _template_answer(question: str, result: CheckoutResult) -> str:
        facts = result.decision.facts
        q = _normalise(question)
        if any(keyword in q for keyword in CONDUCT_KEYWORDS):
            if facts["decision"] == "ALERTE":
                return ("Restez discret et bienveillant : proposez votre aide pour finaliser le passage "
                        "en caisse et demandez si un article est resté dans un sac. Ne parlez jamais de vol.")
            return ("Le passage est validé : remerciez le client et laissez-le payer. Aucune intervention "
                    "supplémentaire n'est nécessaire.")
        return (f"Décision {facts['decision']} (probabilité d'anomalie "
                f"{facts.get('probabilite_anomalie_pourcent')} %, seuil "
                f"{facts.get('seuil_decision_pourcent')} %). "
                f"Parcours : {facts['risque_du_parcours_pourcent']} % de risque ; "
                f"{len(facts['articles_pris_non_presentes'])} article(s) pris non présenté(s). "
                f"{facts['reconnaissance_produits']}.")

    # ------------------------------------------------------------------ UI activity trace

    def _activities(self, path: PathReport, checkout: CheckoutReport,
                    decision: DecisionReport, briefing: AgentReply) -> tuple[AgentActivity, ...]:
        def status(reply: AgentReply) -> str:
            """Trace status of a reply: ok (LLM), fallback (LLM failed) or offline (templates)."""
            if reply.note:
                return "fallback"
            return "offline" if reply.source.startswith("Réponses") else "ok"

        return (
            AgentActivity(
                agent=self.rnn_agent.name, role=self.rnn_agent.role, status=status(path.summary),
                summary=path.summary.text, source=path.summary.source, note=path.summary.note,
                metrics={"risque": f"{path.risk:.0%}", "seuil": f"{path.threshold:.0%}",
                         "indices": len(path.cues), "événements": path.n_events}),
            AgentActivity(
                agent=self.cnn_agent.name, role=self.cnn_agent.role, status=status(checkout.summary),
                summary=checkout.summary.text, source=checkout.summary.source, note=checkout.summary.note,
                metrics={"déclarés": len(checkout.view.declared_ids),
                         "confirmés CNN": checkout.presented_count,
                         "non présentés": checkout.missing_count,
                         "panier": f"{checkout.cart_total:.2f} €",
                         "confiance": (f"{checkout.view.confidence:.0%}"
                                       if checkout.view.confidence is not None else "n/a")}),
            AgentActivity(
                agent=self.decision_agent.name, role=self.decision_agent.role,
                status=status(decision.summary), summary=decision.summary.text,
                source=decision.summary.source, note=decision.summary.note,
                metrics={"décision": decision.outcome,
                         "probabilité": (f"{decision.decision.probability:.0%}"
                                         if decision.decision.probability is not None else "n/a"),
                         "seuil": (f"{decision.decision.threshold:.0%}"
                                   if decision.decision.threshold is not None else "n/a"),
                         "modèle": decision.decision.source}),
            AgentActivity(
                agent=self.name, role=self.role, status=status(briefing),
                summary=briefing.text, source=briefing.source, note=briefing.note,
                metrics={"mode": self.mode, "décision": decision.outcome}),
        )
