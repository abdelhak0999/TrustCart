"""Decision / Fusion Agent: risk-fusion and policy specialist.

Owns the deterministic logistic-regression decision and reconciles the RNN and CNN reports. It
emits a ``DecisionReport`` and answers questions about the threshold, the features and the reasons
behind an alert. It cannot change the outcome: the statistical model is the only decider.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .. import decision as decision_module
from ..config import AgentSettings
from ..decision import ALERT, Decision
from .base import AgentReply, BaseAgent, normalise
from .cnn_agent import CheckoutReport
from .facts import build_facts
from .rnn_agent import PathReport

SUMMARY_REQUEST = ("Explique en deux phrases maximum comment les signaux du parcours et de la caisse "
                   "ont conduit à la décision déjà prise. Ne change pas la décision et n'accuse personne.")

SYSTEM_PROMPT = """Tu es l'Agent Décision de TrustCart, spécialiste de la fusion des signaux et de la règle d'alerte.
Tu expliques une décision déjà prise par la régression logistique, à partir du risque du parcours,
des articles non présentés, de la confiance de la vision et du contexte du panier. Règles :
- La décision VALIDÉ/ALERTE est définitive et vient du modèle. Tu ne la modifies jamais, tu ne la remets pas en cause.
- Appuie-toi uniquement sur les faits JSON ci-dessous. Si une information manque, dis-le.
- Ne traite jamais le client de voleur : parle d'indices et d'une vérification éventuelle.
- Réponds en français, en phrases courtes, sans jargon technique (pas de « logit », « coefficient », « calibration »).

Faits sur la décision (JSON) :
{facts}"""


@dataclass(frozen=True)
class DecisionReport:
    """Output of the Decision Agent: the final decision, its five features and the allowed facts."""
    decision: Decision
    features: dict
    facts: dict
    reliability: dict | None
    summary: AgentReply

    @property
    def outcome(self) -> str:
        """VALIDATED or ALERT."""
        return self.decision.outcome

    @property
    def is_alert(self) -> bool:
        """True when the visit must be checked by staff."""
        return self.decision.outcome == ALERT


class DecisionState(TypedDict, total=False):
    """LangGraph state of the Decision agent."""
    path: PathReport
    checkout: CheckoutReport
    events: list[dict]
    decision: Decision
    features: dict
    facts: dict
    report: DecisionReport


class DecisionAgent(BaseAgent):
    """Decision Agent: applies the logistic regression and explains which factors drove it.

    It is the only agent that produces VALIDATED / ALERT, and the decision comes from the
    deterministic model, never from the language model.
    """
    name = "Agent Décision"
    role = "spécialiste de la fusion des signaux et de la règle d'alerte"
    system_prompt = SYSTEM_PROMPT

    def __init__(self, settings: AgentSettings, mode: str = "hybrid"):
        super().__init__(settings, mode)
        self.graph = self._build_graph()

    def _build_graph(self):
        """decide (deterministic logistic regression) -> narrate (template or LLM summary)."""
        builder = StateGraph(DecisionState)
        builder.add_node("decide", self._decide)
        builder.add_node("narrate", self._narrate)
        builder.add_edge(START, "decide")
        builder.add_edge("decide", "narrate")
        builder.add_edge("narrate", END)
        return builder.compile(name="decision_agent")

    def _decide(self, state: DecisionState) -> dict:
        """No LLM participates in the outcome: this node is the only place it is computed."""
        path_report, checkout_report = state["path"], state["checkout"]
        final = decision_module.decide(path_report.risk, path_report.threshold,
                                       path_report.visit, checkout_report.view)
        model_features = decision_module.features(path_report.risk, path_report.visit,
                                                  checkout_report.view)
        facts = build_facts(final, path_report.risk, path_report.threshold, path_report.visit,
                            checkout_report.view, state["events"])
        facts["features_du_modele_de_decision"] = {
            key: (round(value, 3) if isinstance(value, float) else value)
            for key, value in model_features.items()
        }
        facts["source_de_la_decision"] = final.source
        return {"decision": final, "features": model_features, "facts": facts}

    def _narrate(self, state: DecisionState) -> dict:
        facts = state["facts"]
        reliability = (decision_module.reliability()
                       if decision_module.LOGISTIC_REGRESSION_AVAILABLE else None)
        summary = self.summarise(facts, self._template_summary(facts), SUMMARY_REQUEST,
                                 allow_llm=(self.mode == "full_llm"))
        return {"report": DecisionReport(decision=state["decision"], features=state["features"],
                                         facts=facts, reliability=reliability, summary=summary)}

    def evaluate(self, path_report: PathReport, checkout_report: CheckoutReport,
                 events: list[dict]) -> DecisionReport:
        """Run the agent graph: deterministic decision, then narration."""
        return self.graph.invoke({"path": path_report, "checkout": checkout_report,
                                  "events": events})["report"]

    @staticmethod
    def _template_summary(facts: dict) -> str:
        decision = facts["decision"]
        probability = facts.get("probabilite_anomalie_pourcent")
        threshold = facts.get("seuil_decision_pourcent")
        if decision == "ALERTE":
            text = (f"Décision ALERTE : probabilité d'anomalie {probability} % "
                    f"(seuil {threshold} %). ")
            signals = []
            if facts["risque_du_parcours_pourcent"] >= facts["seuil_alerte_pourcent"]:
                signals.append(f"parcours à {facts['risque_du_parcours_pourcent']} % de risque")
            if facts["articles_pris_non_presentes"]:
                signals.append(f"{len(facts['articles_pris_non_presentes'])} article(s) pris non présenté(s)")
            if facts.get("articles_declares_non_reconnus_par_la_camera"):
                signals.append(f"{len(facts['articles_declares_non_reconnus_par_la_camera'])} "
                               "article(s) déclaré(s) non confirmé(s) par la caméra")
            text += ("Signaux : " + " et ".join(signals) + ". " if signals else
                     "Le modèle combine plusieurs signaux faibles. ")
            return text + "Une vérification discrète est recommandée, sans accuser le client."
        text = (f"Décision VALIDÉ : probabilité d'anomalie {probability} %, "
                f"sous le seuil de {threshold} %. ")
        if facts["articles_pris_non_presentes"]:
            text += (f"{len(facts['articles_pris_non_presentes'])} article(s) pris non présenté(s), "
                     "probablement reposé(s) en rayon. ")
        return text + "Aucune vérification supplémentaire n'est nécessaire."

    @staticmethod
    def _template_answer(question: str, facts: dict) -> str:
        q = normalise(question)
        if any(k in q for k in ("seuil", "probabil", "pourquoi", "raison", "alerte", "valid")):
            return (f"{facts['raison_de_la_decision']} Décision prise par : {facts['qui_decide']}. "
                    f"Probabilité d'anomalie : {facts.get('probabilite_anomalie_pourcent')} %, "
                    f"seuil : {facts.get('seuil_decision_pourcent')} %.")
        if any(k in q for k in ("facteur", "poids", "feature", "signal", "variable")):
            features = facts.get("features_du_modele_de_decision", {})
            return ("Signaux utilisés par le modèle : "
                    + ", ".join(f"{key} = {value}" for key, value in features.items()) + ".")
        if any(k in q for k in ("fiab", "fiable", "erreur", "sur ", "confiance")):
            reliability = facts.get("fiabilite_du_modele_de_decision")
            return (f"Fiabilité du modèle de décision : {reliability}." if reliability else
                    "La fiabilité du modèle de décision n'est pas disponible.")
        return (f"Décision {facts['decision']} : probabilité d'anomalie "
                f"{facts.get('probabilite_anomalie_pourcent')} %, seuil "
                f"{facts.get('seuil_decision_pourcent')} %. {facts['raison_de_la_decision']}")

    def answer(self, question: str, report: DecisionReport, history: list[dict]) -> AgentReply:
        """Answer a staff question about the decision, threshold or reliability."""
        return super().answer(question, report.facts, self._template_answer(question, report.facts),
                              history)
