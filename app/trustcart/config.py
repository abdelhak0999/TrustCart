"""Paths and settings, read once from the project's .env file."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

RNN_DIR = ROOT / "models" / "rnn"
DEMO_SESSIONS_PATH = ROOT / "app" / "data" / "demo_sessions.json"

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"


@dataclass(frozen=True)
class AgentSettings:
    backend: str            # "gemini", "ollama" or "templates"
    model: str
    base_url: str | None
    api_key: str | None


def agent_settings() -> AgentSettings:
    """Pick the agent's language model from .env; without a usable backend, fall back to templates."""
    backend = os.getenv("AGENT_BACKEND", "").strip().lower()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    if backend in ("", "gemini") and gemini_key:
        return AgentSettings("gemini", os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip(),
                             GEMINI_BASE_URL, gemini_key)
    if backend == "ollama":
        # Ollama ignores the key, but the client library requires a non-empty one
        return AgentSettings("ollama", os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip(),
                             os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL).strip(), "ollama")
    return AgentSettings("templates", "", None, None)
