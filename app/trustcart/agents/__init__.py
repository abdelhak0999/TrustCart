"""Multi-agent explanation layer for TrustCart.

Three model agents (RNN, CNN, Decision) wrap deterministic ML inference and emit structured
reports. The Coordinator orchestrates them, writes the staff briefing and routes Q&A. None of them
can override the statistical decision: models decide, agents explain.
"""
from .base import AgentReply, BaseAgent, TEMPLATE_SOURCE
from .cnn_agent import CNNAgent, CheckoutReport
from .coordinator import AgentActivity, CheckoutResult, CoordinatorAgent
from .decision_agent import DecisionAgent, DecisionReport
from .facts import CustomerMessage, build_facts, customer_message
from .rnn_agent import PathReport, RNNAgent

__all__ = [
    "AgentActivity", "AgentReply", "BaseAgent", "CNNAgent", "CheckoutReport", "CheckoutResult",
    "CoordinatorAgent", "CustomerMessage", "DecisionAgent", "DecisionReport", "PathReport",
    "RNNAgent", "TEMPLATE_SOURCE", "build_facts", "customer_message",
]
