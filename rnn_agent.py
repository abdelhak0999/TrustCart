"""RNN Path Agent: behavioural-sequence specialist.

Owns the deterministic RNN inference and the readable visit summary. It emits a ``PathReport`` and
answers staff questions about the trajectory, pauses, speed of picks and model reliability. It never
decides ALERT/VALIDATED and never overrides the decision model.
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import rnn
from ..config import AgentSettings
from ..visit import VisitSummary, format_duration, summarize, timeline
from .base import AgentReply, BaseAgent, normalise

SUMMARY_REQUEST = ("Résume le parcours de ce client en deux phrases maximum : niveau de risque, "
                   "indices éventuels, durée et pause la plus longue. Ne donne pas de décision finale.")

SYSTEM_PROMPT = """Tu es l'Agent RNN de TrustCart, spécialiste du parcours client en magasin.
Tu expliques le score de risque comportemental produit par le modèle RNN (LSTM) et les faits du parcours.
Règles :
- La décision finale VALIDÉ/ALERTE appartient à la régression logistique. Tu ne la donnes pas, tu ne la modifies jamais.
- Appuie-toi uniquement sur les faits JSON ci-dessous. Si une information manque, dis-le.
- Ne traite jamais le client de voleur : parle d'indices, et rappelle qu'un article reposé ou oublié reste possible.
- Réponds en français, en phrases courtes, sans jargon (pas de « LSTM », « logit », « calibration »).

Faits sur le parcours (JSON) :
{facts}"""


@dataclass(frozen=True)
class PathReport:
    risk: float
    threshold: float
    reliability: dict
    visit: VisitSummary
    timeline: tuple[str, ...]
    cues: tuple[str, ...]
    summary: AgentReply
    n_events: int
    above_threshold: bool


class RNNAgent(BaseAgent):
    name = "Agent RNN"
    role = "spécialiste du parcours et du risque comportemental"
    system_prompt = SYSTEM_PROMPT

    def __init__(self, settings: AgentSettings, mode: str = "hybrid"):
        super().__init__(settings, mode)

    def analyze(self, events: list[dict], visit: VisitSummary | None = None) -> PathReport:
        """Run the RNN and package its output; the decision is not taken here."""
        visit = visit or summarize(events, rnn.cue_thresholds())
        risk = float(rnn.path_risk(events))
        threshold = float(rnn.alert_threshold())
        reliability = rnn.reliability()
        lines = tuple(timeline(events))
        facts = self.facts_from(risk, threshold, reliability, visit, lines)
        template = self._template_summary(facts)
        summary = self.summarise(facts, template, SUMMARY_REQUEST,
                                 allow_llm=(self.mode == "full_llm"))
        return PathReport(risk=risk, threshold=threshold, reliability=reliability, visit=visit,
                          timeline=lines, cues=tuple(visit.cues), summary=summary,
                          n_events=len(events), above_threshold=risk >= threshold)

    @staticmethod
    def facts_from(risk: float, threshold: float, reliability: dict, visit: VisitSummary,
                   lines: tuple[str, ...]) -> dict:
        return {
            "risque_du_parcours_pourcent": round(risk * 100),
            "seuil_alerte_pourcent": round(threshold * 100),
            "risque_au_dessus_du_seuil": risk >= threshold,
            "indices_dans_le_parcours": list(visit.cues),
            "duree_visite_min": round(visit.duration_min, 1),
            "plus_longue_pause_min": round(visit.longest_pause_min, 1),
            "articles_pris": [visit.items[i].label for i in visit.picked],
            "nombre_evenements": len(lines),
            "fiabilite_du_modele_de_parcours": (
                f"sur des données de test, {reliability['recall']:.0%} des vols simulés détectés, "
                f"{reliability['false_alarm_rate']:.2%} de clients honnêtes signalés à tort"),
            "chronologie": list(lines[:60]) + (["…"] if len(lines) > 60 else []),
        }

    @staticmethod
    def _template_summary(facts: dict) -> str:
        level = "élevé" if facts["risque_au_dessus_du_seuil"] else "faible"
        text = (f"Risque du parcours {level} : {facts['risque_du_parcours_pourcent']} % "
                f"(seuil {facts['seuil_alerte_pourcent']} %). ")
        cues = facts["indices_dans_le_parcours"]
        text += ("Indices : " + " ; ".join(cues) + ". " if cues else
                 "Aucun indice simple : c'est l'enchaînement des actions qui pèse sur le score. ")
        text += (f"Visite de {format_duration(facts['duree_visite_min'] * 60)}, "
                 f"pause la plus longue {format_duration(facts['plus_longue_pause_min'] * 60)}.")
        return text

    @staticmethod
    def _template_answer(question: str, facts: dict) -> str:
        q = normalise(question)
        if any(k in q for k in ("pause", "arret", "duree", "long")):
            return (f"La visite dure {format_duration(facts['duree_visite_min'] * 60)}, avec une pause "
                    f"maximale de {format_duration(facts['plus_longue_pause_min'] * 60)}. "
                    f"Risque du parcours : {facts['risque_du_parcours_pourcent']} %.")
        if any(k in q for k in ("indice", "comportement", "vitesse", "rapide", "rayon", "trajectoire")):
            cues = facts["indices_dans_le_parcours"]
            return ("Indices relevés : " + " ; ".join(cues) + "." if cues else
                    "Aucun indice simple n'est ressorti : le modèle réagit à l'ordre et au rythme des actions.")
        if any(k in q for k in ("fiab", "fiable", "erreur", "sur ", "confiance")):
            return f"Le modèle de parcours peut se tromper : {facts['fiabilite_du_modele_de_parcours']}."
        return (f"Le parcours présente un risque de {facts['risque_du_parcours_pourcent']} % "
                f"(seuil {facts['seuil_alerte_pourcent']} %). "
                + ("Indices : " + " ; ".join(facts["indices_dans_le_parcours"]) + "."
                   if facts["indices_dans_le_parcours"] else
                   "Aucun indice simple, c'est l'enchaînement des actions qui compte."))

    def answer(self, question: str, report: PathReport, history: list[dict]) -> AgentReply:
        facts = self.facts_from(report.risk, report.threshold, report.reliability, report.visit,
                                report.timeline)
        return super().answer(question, facts, self._template_answer(question, facts), history)
