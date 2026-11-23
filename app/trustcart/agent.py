"""Explanation agent: turns the models' decision into words for customers and staff.

The decision is always taken by the models (see decision.py); the agent only explains it.
The customer screen always uses fixed, neutral wording. The staff briefing and the answers to
staff questions come from a language model (Gemini or Ollama) when one is configured, and from
written templates otherwise or whenever the model fails, so the app never stops working.
"""
import json
from dataclasses import dataclass

import openai

from . import rnn
from .cnn import CheckoutView
from .config import AgentSettings
from .decision import ALERT, Decision
from .visit import VisitSummary, timeline

TEMPLATE_SOURCE = "Réponses pré-écrites (hors ligne)"

SYSTEM_PROMPT = """Tu es l'assistant de TrustCart, la caisse autonome sans friction d'un supermarché.
Tu aides le personnel à comprendre une décision prise par les modèles du magasin pour une visite client.

Règles :
- La décision (VALIDÉ ou ALERTE) est déjà prise par les modèles. Tu l'expliques ; tu ne la modifies jamais et tu ne la remets pas en cause.
- Appuie-toi uniquement sur les faits fournis ci-dessous. Si une information manque, dis-le simplement (par exemple : la reconnaissance des produits en caisse n'est pas encore disponible, les articles posés sont saisis à la main).
- Ne traite jamais le client de voleur. Parle d'indices, et rappelle quand c'est utile qu'un article peut avoir été reposé ou oublié dans un sac.
- Réponds en français, en phrases courtes et claires, sans jargon technique (pas de « LSTM », « logit », « calibration »).

Faits sur la visite (JSON) :
{facts}"""

BRIEFING_REQUEST = ("Rédige le message affiché sur la tablette du personnel pour cette visite : trois phrases au "
                    "maximum, qui donnent la décision, les principaux indices et, en cas d'alerte, comment "
                    "aborder le client avec discrétion.")


@dataclass(frozen=True)
class CustomerMessage:
    title: str
    body: str


@dataclass(frozen=True)
class AgentReply:
    text: str
    source: str             # which brain wrote it
    note: str | None = None  # why the templates were used instead of the language model


def build_facts(decision: Decision, path_risk: float, threshold: float, visit: VisitSummary,
                checkout: CheckoutView, events: list[dict]) -> dict:
    """Everything the agent may say, and nothing else: its only source of truth."""
    presented = set(checkout.item_ids)
    reliability = rnn.reliability()
    return {
        "decision": "ALERTE" if decision.outcome == ALERT else "VALIDÉ",
        "raison_de_la_decision": decision.reason,
        "qui_decide": ("règle provisoire sur le risque du parcours (la régression logistique n'est pas encore "
                       "entraînée)" if decision.source == "provisional_rule" else "régression logistique"),
        "risque_du_parcours_pourcent": round(path_risk * 100),
        "seuil_alerte_pourcent": round(threshold * 100),
        "indices_dans_le_parcours": list(visit.cues),
        "articles_pris": [visit.items[i].label for i in visit.picked],
        "articles_poses_en_caisse": [visit.items[i].label for i in checkout.item_ids if i in visit.items],
        "articles_pris_non_presentes": [visit.items[i].label for i in visit.picked if i not in presented],
        "total_articles_poses_eur": round(sum(visit.items[i].price for i in presented if i in visit.items), 2),
        "duree_visite_min": round(visit.duration_min, 1),
        "plus_longue_pause_min": round(visit.longest_pause_min, 1),
        "reconnaissance_produits": ("non disponible : le modèle de vision n'est pas encore entraîné, "
                                    "les articles posés sont saisis à la main" if checkout.source == "manual"
                                    else f"confiance moyenne {checkout.confidence:.0%}"),
        "fiabilite_du_modele_de_parcours": (
            f"sur des données de test : {reliability['recall']:.0%} des vols simulés détectés, "
            f"{reliability['false_alarm_rate']:.2%} de clients honnêtes signalés à tort"),
        "chronologie": timeline(events),
    }


def customer_message(decision: Decision, facts: dict) -> CustomerMessage:
    """Fixed wording on purpose: the customer screen must never vary or accuse."""
    if decision.outcome == ALERT:
        return CustomerMessage("Un instant, s'il vous plaît",
                               "Un conseiller arrive pour vous aider à finaliser votre passage en caisse.")
    n = len(facts["articles_poses_en_caisse"])
    return CustomerMessage("Votre panier est validé",
                           f"{n} article{'s' if n > 1 else ''}, {facts['total_articles_poses_eur']:.2f} €. "
                           "Présentez votre carte pour payer. Bonne journée !")


# ---------------------------------------------------------------- templates (no language model)

def _template_briefing(facts: dict) -> str:
    missing = facts["articles_pris_non_presentes"]
    if facts["decision"] == "VALIDÉ":
        text = (f"Rien à signaler. Risque du parcours : {facts['risque_du_parcours_pourcent']} %, "
                f"sous le seuil d'alerte de {facts['seuil_alerte_pourcent']} %.")
        if missing:
            text += f" {len(missing)} article(s) pris puis non présenté(s), probablement reposé(s) en rayon."
        return text
    cues = facts["indices_dans_le_parcours"]
    text = (f"Vérification recommandée. Risque du parcours : {facts['risque_du_parcours_pourcent']} % "
            f"(seuil {facts['seuil_alerte_pourcent']} %). ")
    text += ("Indices : " + " ; ".join(cues) + ". " if cues else
             "Aucun indice simple identifié : c'est l'enchaînement des actions qui est inhabituel. ")
    if missing:
        text += f"Pris mais non présenté(s) : {', '.join(missing)}. "
    return text + "Abordez le client avec discrétion et proposez votre aide : un oubli reste possible."


def _template_answer(question: str, facts: dict) -> str:
    q = question.lower()
    if "pourquoi" in q or "raison" in q:
        cues = facts["indices_dans_le_parcours"]
        return (f"{facts['raison_de_la_decision']} Décision prise par : {facts['qui_decide']}. "
                + ("Indices relevés : " + " ; ".join(cues) + "." if cues else
                   "Aucun indice simple n'explique le score : c'est l'ordre des actions qui est inhabituel."))
    if "article" in q or "produit" in q or "panier" in q:
        missing = facts["articles_pris_non_presentes"]
        return (f"Articles pris : {len(facts['articles_pris'])}. Posés en caisse : "
                f"{len(facts['articles_poses_en_caisse'])}. "
                + (f"Non présentés : {', '.join(missing)}." if missing else "Tous les articles pris ont été présentés.")
                + f" Reconnaissance des produits : {facts['reconnaissance_produits']}.")
    if any(k in q for k in ("faire", "approch", "conseil", "comment")):
        return ("Restez discret et bienveillant : proposez votre aide pour finaliser le passage en caisse et "
                "demandez si un article est resté dans un sac. Ne parlez jamais de vol.")
    if any(k in q for k in ("fiab", "erreur", "sûr", "sur ")):
        return f"Le modèle de parcours peut se tromper : {facts['fiabilite_du_modele_de_parcours']}."
    return ("En mode hors ligne, je réponds aux questions sur : pourquoi cette décision, les articles, "
            "la conduite à tenir et la fiabilité du modèle.")


# ---------------------------------------------------------------- agent

class ExplanationAgent:
    def __init__(self, settings: AgentSettings):
        self.settings = settings
        self._client = None
        if settings.backend != "templates":
            self._client = openai.OpenAI(api_key=settings.api_key, base_url=settings.base_url,
                                         timeout=30.0, max_retries=1)

    @property
    def source(self) -> str:
        if self._client is None:
            return TEMPLATE_SOURCE
        return f"{'Gemini' if self.settings.backend == 'gemini' else 'Ollama'} ({self.settings.model})"

    def _ask(self, facts: dict, history: list[dict], question: str) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(facts=json.dumps(facts, ensure_ascii=False, indent=1))},
                    *history, {"role": "user", "content": question}]
        response = self._client.chat.completions.create(model=self.settings.model, messages=messages)
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("empty answer")
        return text

    def _with_fallback(self, facts: dict, history: list[dict], question: str, template: str) -> AgentReply:
        if self._client is None:
            return AgentReply(template, TEMPLATE_SOURCE)
        try:
            return AgentReply(self._ask(facts, history, question), self.source)
        except openai.RateLimitError:
            note = "quota du modèle de langage atteint"
        except openai.AuthenticationError:
            note = "clé API refusée"
        except openai.BadRequestError as exc:
            # Gemini answers an invalid key with 400 rather than 401, in a body the client does not parse
            invalid_key = "api key" in exc.response.text.lower()
            note = "clé API refusée" if invalid_key else "requête refusée par le modèle de langage"
        except openai.APIConnectionError:
            note = "modèle de langage injoignable"
        except (openai.APIError, ValueError) as exc:
            note = f"erreur du modèle de langage ({type(exc).__name__})"
        return AgentReply(template, TEMPLATE_SOURCE, note)

    def staff_briefing(self, facts: dict) -> AgentReply:
        return self._with_fallback(facts, [], BRIEFING_REQUEST, _template_briefing(facts))

    def answer(self, question: str, facts: dict, history: list[dict]) -> AgentReply:
        return self._with_fallback(facts, history, question, _template_answer(question, facts))
