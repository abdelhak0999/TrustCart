"""Paths and settings, read once from the project's .env file."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

RNN_DIR = ROOT / "models" / "rnn"
CNN_DIR = ROOT / "models" / "cnn"
DECISION_DIR = ROOT / "models" / "decision"
DEMO_SESSIONS_PATH = ROOT / "app" / "data" / "demo_sessions.json"

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"

AGENT_MODES = ("hybrid", "full_llm", "templates")


@dataclass(frozen=True)
class AgentSettings:
    """Language-model backend chosen for one agent (Gemini, Ollama or offline templates)."""
    backend: str            # "gemini", "ollama" or "templates"
    model: str
    base_url: str | None
    api_key: str | None


def agent_settings(role: str | None = None) -> AgentSettings:
    """Pick the agent's language model from .env; without a usable backend, fall back to templates.

    ``role`` may be "rnn", "cnn", "decision" or "coordinator". A role-specific model can be set
    with e.g. ``GEMINI_MODEL_CNN``; otherwise the shared ``GEMINI_MODEL`` / ``OLLAMA_MODEL`` is used.
    """
    backend = os.getenv("AGENT_BACKEND", "").strip().lower()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    if backend in ("", "gemini") and gemini_key:
        model = (os.getenv(f"GEMINI_MODEL_{role.upper()}", "").strip() if role else "") \
            or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
        return AgentSettings("gemini", model, GEMINI_BASE_URL, gemini_key)
    if backend == "ollama":
        # Ollama ignores the key, but the client library requires a non-empty one
        model = (os.getenv(f"OLLAMA_MODEL_{role.upper()}", "").strip() if role else "") \
            or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()
        return AgentSettings("ollama", model,
                             os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL).strip(), "ollama")
    return AgentSettings("templates", "", None, None)


def agent_mode() -> str:
    """How much narration the language model does: hybrid (default), full_llm or templates."""
    mode = os.getenv("AGENT_MODE", "hybrid").strip().lower()
    return mode if mode in AGENT_MODES else "hybrid"
