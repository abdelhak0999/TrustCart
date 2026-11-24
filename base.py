"""Shared contracts and offline-safe language-model wrapper for the TrustCart agents.

Every agent inherits the same zero-crash guarantee:
  * no API key / ``AGENT_BACKEND=templates`` -> deterministic French templates;
  * rate limit, refused key, unreachable endpoint or any unexpected error -> template fallback
    with a short note explaining why, so the demo never stops working.

The language model is only a narrator. The statistical models own every ALERT/VALIDATED decision.
"""
from __future__ import annotations

import json
import logging
import unicodedata
from dataclasses import dataclass

import openai

from ..config import AgentSettings

LOGGER = logging.getLogger(__name__)
TEMPLATE_SOURCE = "Réponses pré-écrites (hors ligne)"


def normalise(text: str) -> str:
    """Lowercase and strip French accents so keyword routing/templates are robust."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


@dataclass(frozen=True)
class AgentReply:
    text: str
    source: str                 # which brain wrote it
    agent_name: str             # "Agent RNN", "Agent CNN", "Agent Décision", "Coordinateur"
    note: str | None = None     # why templates were used instead of the language model


class BaseAgent:
    """Common OpenAI-compatible client (Gemini / Ollama) with automatic template fallback."""

    name = "Agent"
    role = "assistant TrustCart"
    system_prompt = (
        "Tu es un assistant TrustCart. La décision VALIDÉ/ALERTE est déjà prise par les modèles "
        "statistiques : tu l'expliques, tu ne la modifies jamais et tu ne la remets pas en cause. "
        "Réponds en français, en phrases courtes, sans jargon technique. Ne traite jamais le client "
        "de voleur : parle d'indices et rappelle qu'un oubli ou un article reposé reste possible."
    )

    def __init__(self, settings: AgentSettings, mode: str = "hybrid"):
        self.settings = settings
        self.mode = mode
        self._client = None
        if settings.backend != "templates" and mode != "templates":
            self._client = openai.OpenAI(api_key=settings.api_key, base_url=settings.base_url,
                                         timeout=30.0, max_retries=1)

    @property
    def llm_available(self) -> bool:
        return self._client is not None

    @property
    def source(self) -> str:
        if self._client is None:
            return TEMPLATE_SOURCE
        backend = "Gemini" if self.settings.backend == "gemini" else "Ollama"
        return f"{backend} ({self.settings.model})"

    def _ask(self, facts: dict, history: list[dict], question: str,
             system_prompt: str | None = None) -> str:
        prompt = (system_prompt or self.system_prompt).format(
            facts=json.dumps(facts, ensure_ascii=False, indent=1, default=str)
        )
        messages = [{"role": "system", "content": prompt},
                    *history,
                    {"role": "user", "content": question}]
        response = self._client.chat.completions.create(model=self.settings.model, messages=messages)
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("empty answer")
        return text

    def _with_fallback(self, facts: dict, history: list[dict], question: str, template: str,
                       system_prompt: str | None = None) -> AgentReply:
        """Ask the LLM, but never raise: on any failure return the offline French template."""
        if self._client is None:
            return AgentReply(template, TEMPLATE_SOURCE, self.name)
        try:
            return AgentReply(self._ask(facts, history, question, system_prompt), self.source, self.name)
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
        except Exception as exc:      # zero-crash guarantee at the LLM boundary
            LOGGER.warning("%s: unexpected LLM error: %s", self.name, exc)
            note = f"erreur inattendue ({type(exc).__name__})"
        LOGGER.warning("%s: falling back to templates (%s)", self.name, note)
        return AgentReply(template, TEMPLATE_SOURCE, self.name, note)

    def summarise(self, facts: dict, template: str, question: str, allow_llm: bool) -> AgentReply:
        """Return an LLM-written summary when allowed, otherwise the deterministic template."""
        if allow_llm and self._client is not None:
            return self._with_fallback(facts, [], question, template)
        return AgentReply(template, TEMPLATE_SOURCE, self.name)

    def answer(self, question: str, facts: dict, template: str, history: list[dict]) -> AgentReply:
        """Answer a staff question, with the same offline fallback."""
        return self._with_fallback(facts, history, question, template)
