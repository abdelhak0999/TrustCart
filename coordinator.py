"""Coordinator / Staff Liaison Agent.

Orchestrates the three model agents, keeps the fixed neutral customer message, writes the staff
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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .. import rnn
from ..config import AgentSettings, agent_mode, agent_settings
from ..decision import ALERT
from ..visit import summarize
from .base import AgentReply, BaseAgent, normalise
from .cnn_agent import CNNAgent, CheckoutReport
from .decision_agent import DecisionAgent, DecisionReport
from .facts import CustomerMessage, customer_message
from .rnn_agent import RNNAgent, PathReport

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
    agent: str
    role: str
    status: str                 # "ok", "fallback" or "offline"
    summary: str
    source: str
    note: str | None
    metrics: dict


@dataclass(frozen=True)
class CheckoutResult:
    path: PathReport
    checkout: CheckoutReport
    decision: DecisionReport
    customer: CustomerMessage
    briefing: AgentReply
    activities: tuple[AgentActivity, ...]

    @property
    def outcome(self) -> str:
        return self.decision.outcome

    @property
    def is_alert(self) -> bool:
        return self.decision.outcome == ALERT


class CoordinatorAgent(BaseAgent):
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

    # ------------------------------------------------------------------ orchestration

    def run_checkout(self, events: list[dict], presented: list[int] | tuple[int, ...],
                     image_paths: list[str] | tuple[str, ...] | None = None,
                     manual_categories: list[str] | tuple[str, ...] | None = None) -> CheckoutResult:
        """Run the full deterministic pipeline, then let the Coordinator narrate it."""
        visit = summarize(events, rnn.cue_thresholds())

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="trustcart-agent") as pool:
            path_future = pool.submit(self.rnn_agent.analyze, events, visit)
            checkout_future = pool.submit(self.cnn_agent.analyze, visit, presented,
                                           image_paths, manual_categories)
            path_report = path_future.result()
            checkout_report = checkout_future.result()

        decision_report = self.decision_agent.evaluate(path_report, checkout_report, events)
        customer = customer_message(decision_report.decision, decision_report.facts)
        briefing = self._briefing(path_report, checkout_report, decision_report)
        activities = self._activities(path_report, checkout_report, decision_report, briefing)
        return CheckoutResult(path=path_report, checkout=checkout_report, decision=decision_report,
                              customer=customer, briefing=briefing, activities=activities)

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

    def answer(self, question: str, result: CheckoutResult, history: list[dict]) -> AgentReply:
        """Route a staff question, answer with the specialist(s), and synthesise if needed."""
        routes = self.route(question)
        if routes == ["coordinator"]:
            return self._coordinator_answer(question, result, history)

        replies: list[AgentReply] = []
        for route in routes:
            if route == "rnn":
                replies.append(self.rnn_agent.answer(question, result.path, history))
            elif route == "cnn":
                replies.append(self.cnn_agent.answer(question, result.checkout, history))
            elif route == "decision":
                replies.append(self.decision_agent.answer(question, result.decision, history))

        if len(replies) == 1:
            return replies[0]
        return self._synthesise(question, result, replies, history)

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
                metrics={"posés": checkout.presented_count, "non présentés": checkout.missing_count,
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
