#!/usr/bin/env python3
"""Offline verification for the TrustCart multi-agent layer.

Checks, with no API key and no network:
  1. all five demo scenarios run end-to-end in template mode;
  2. the deterministic decision is never overridden by an agent;
  3. the exported logistic-regression metadata reproduces the training probabilities;
  4. the CNN loads and classifies the shipped sample crops;
  5. staff questions route to the expected specialist agents;
  6. rate-limit / connection / unexpected LLM errors fall back to French templates.

Run with:
    .venv-app/bin/python scripts/verify_agents.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
os.environ["AGENT_BACKEND"] = "templates"
os.environ["AGENT_MODE"] = "templates"

import httpx2  # noqa: E402
import openai  # noqa: E402
import pandas as pd  # noqa: E402

from trustcart import cnn, decision, rnn  # noqa: E402
from trustcart.agents import CoordinatorAgent  # noqa: E402
from trustcart.agents.base import BaseAgent, ChatModel, TEMPLATE_SOURCE  # noqa: E402
from trustcart.config import AgentSettings, agent_settings  # noqa: E402
from trustcart.visit import summarize  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    """Print one check result and remember failures."""
    print(("  ✅ " if condition else "  ❌ ") + message)
    if not condition:
        FAILURES.append(message)


class FailingLLM:
    """Stand-in chat client whose every call raises ``error``."""

    def __init__(self, error: Exception):
        self.error = error

    def invoke(self, _messages):
        """Fake LLM call."""
        raise self.error


def failing_llm(error: Exception) -> FailingLLM:
    """A fake language model that always raises ``error`` (to test the template fallback)."""
    return FailingLLM(error)


def http_errors() -> dict[str, Exception]:
    """Typical API errors (quota, refused key, network) used to test the fallback."""
    request = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx2.Response(429, request=request)
    return {
        "rate limit": openai.RateLimitError("quota", response=response, body=None),
        "connection": openai.APIConnectionError(request=request),
        "unexpected": RuntimeError("boom"),
    }


def sentence_count(text: str) -> int:
    """Number of sentences in a text (the staff briefing must stay short)."""
    return len(re.findall(r"[.!?](?:\s|$)", text))


def main() -> int:
    """Run every offline check and return a non-zero exit code on failure."""
    print("1. Demo scenarios (templates, zero network)")
    sessions = json.load(open(ROOT / "app" / "data" / "demo_sessions.json"))
    coordinator = CoordinatorAgent(agent_settings("coordinator"), mode="templates")
    results = {}
    for session in sessions:
        kind, events = session["kind"], session["events"]
        visit = summarize(events, rnn.cue_thresholds())
        presented = list(dict.fromkeys(visit.paid))
        result = coordinator.run_checkout(events, presented)
        results[kind] = result

        expected = decision.decide(result.path.risk, result.path.threshold, visit,
                                   result.checkout.view)
        check(result.outcome == expected.outcome,
              f"{kind}: agent outcome equals deterministic decision ({result.outcome})")
        check(result.decision.decision.source == "logistic_regression",
              f"{kind}: decision source is logistic_regression")
        check(result.customer.title in ("Votre panier est validé", "Un instant, s'il vous plaît"),
              f"{kind}: fixed neutral customer message")
        check(sentence_count(result.briefing.text) <= 3,
              f"{kind}: staff briefing has at most 3 sentences")
        check(all(activity.summary for activity in result.activities),
              f"{kind}: all four agents produced a report")

    check(results["normal"].outcome == decision.VALIDATED, "normal scenario is VALIDATED")
    for kind in ("conceal", "alibi", "spree", "false_alarm"):
        check(results[kind].outcome == decision.ALERT, f"{kind} scenario is ALERT")

    print("\n2. Decision metadata reproduces the exported probabilities")
    meta = decision.metadata()
    check(len(meta["coefficients"]) == len(meta["features"]) == 5,
          "metadata has 5 coefficients and 5 features")
    check("items_rejected_by_cnn" in meta["features"],
          "decision model uses the CNN camera check as a feature")
    check("cnn_confidence" not in meta["features"],
          "decision model does not use the untrainable CNN-confidence proxy")
    check(0.0 <= meta["threshold"] <= 1.0, "threshold is in [0, 1]")
    check(meta["threshold_selection"]["split"] == "val",
          "threshold was selected on validation, not test")
    audit = pd.read_parquet(ROOT / "models" / "decision" / "decision_features.parquet")
    check(set(audit["split"]) == {"train", "val", "test"}, "audit parquet has the three splits")
    split_ids = {name: set(audit.loc[audit["split"] == name, "session_id"])
                 for name in ("train", "val", "test")}
    check(not (split_ids["train"] & split_ids["val"])
          and not (split_ids["train"] & split_ids["test"])
          and not (split_ids["val"] & split_ids["test"]),
          "no session leaks between train/val/test")
    sample = audit.sample(n=25, random_state=42)
    max_delta = 0.0
    for _, row in sample.iterrows():
        inputs = {feature: row[feature] for feature in meta["features"]}
        reproduced = decision._logistic_regression(inputs)
        max_delta = max(max_delta, abs(reproduced.probability - float(row["probability"])))
        if reproduced.outcome != (decision.ALERT if int(row["alert"]) else decision.VALIDATED):
            check(False, f"decision mismatch for session {row['session_id']}")
            break
    check(max_delta < 1e-6, f"exported metadata matches training probabilities (max Δ={max_delta:.2e})")

    print("\n3. CNN inference on the shipped sample crops")
    index = cnn.sample_index()
    correct = 0
    for entry in index:
        prediction = cnn.predict_image(entry["file"])
        correct += prediction.label == entry["supercategory"]
        check(0.0 <= prediction.confidence <= 1.0,
              f"{entry['supercategory']}: confidence in [0, 1]")
    check(correct / len(index) >= 0.60,
          f"sample-crop accuracy {correct}/{len(index)} ≥ 60%")
    view = cnn.items_seen([1, 2], image_paths=cnn.camera_photos([1, 2]))
    check(view.source == "cnn" and view.confidence is not None and len(view.labels) == 2,
          "items_seen returns a CNN view with mean confidence")
    check(view.item_ids == (1, 2) and not view.rejected_ids,
          "CNN confirms declared items whose photo matches their category")
    swapped = cnn.items_seen([1, 2], image_paths=cnn.camera_photos([1, 2], {2: "milk"}))
    check(swapped.item_ids == (1,) and swapped.rejected_ids == (2,),
          "CNN rejects a declared item whose photo shows another product")

    print("\n3b. The CNN drives the decision (substitution fraud)")
    normal = next(s for s in sessions if s["kind"] == "normal")
    normal_visit = summarize(normal["events"], rnn.cue_thresholds())
    declared = list(dict.fromkeys(normal_visit.paid))
    picked_declared = [i for i in declared if i in normal_visit.picked]
    target = max(picked_declared, key=lambda i: normal_visit.items[i].price)
    other = next(c for c in cnn.categories() if c != cnn.catalogue_category(target))
    honest = coordinator.run_checkout(normal["events"], declared)
    fraud = coordinator.run_checkout(normal["events"], declared, swaps={target: other})
    check(honest.checkout.view.source == "cnn" and honest.checkout.missing_count == 0,
          "camera mode is the default and confirms every honest item")
    check(fraud.checkout.missing_count == 1 and target in fraud.checkout.view.rejected_ids,
          "swapped item is counted as not presented")
    check(fraud.decision.features["items_picked_not_presented"]
          > honest.decision.features["items_picked_not_presented"],
          "CNN output changes the decision model's input")
    check(fraud.decision.decision.probability > honest.decision.decision.probability,
          f"substitution raises the anomaly probability "
          f"({honest.decision.decision.probability:.0%} -> {fraud.decision.decision.probability:.0%})")
    check(honest.outcome == decision.VALIDATED and fraud.outcome == decision.ALERT,
          "honest normal checkout is VALIDATED, the same visit with a substitution is ALERT")
    off = coordinator.run_checkout(normal["events"], declared, camera=False)
    check(off.checkout.view.source == "manual", "camera=False keeps the manual fallback")
    manual = cnn.items_seen([1, 2])
    check(manual.source == "manual" and manual.confidence is None,
          "items_seen keeps a manual fallback")

    print("\n4. Q&A routing")
    expected_routes = {
        "Pourquoi cette alerte ?": {"decision"},
        "Quels articles manquent ?": {"cnn"},
        "Quelle est la pause la plus longue ?": {"rnn"},
        "Comment aborder le client ?": {"coordinator"},
        "Le modèle est-il fiable ?": {"rnn", "decision"},
    }
    for question, expected in expected_routes.items():
        routes = set(coordinator.route(question))
        check(routes == expected, f"route {question!r} -> {sorted(routes)}")
    reply = coordinator.answer("Quels articles manquent ?", results["conceal"], [])
    check(reply.agent_name == "Agent CNN", "missing-items question is answered by Agent CNN")
    reply = coordinator.answer("Pourquoi cette alerte ?", results["conceal"], [])
    check(reply.agent_name == "Agent Décision", "why-alert question is answered by Agent Décision")
    reply = coordinator.answer("Comment aborder le client ?", results["conceal"], [])
    check(reply.agent_name == "Coordinateur", "conduct question is answered by the Coordinator")

    print("\n5. Zero-crash LLM fallbacks & per-model chat clients")
    settings = AgentSettings("gemini", "fake-model", None, "fake-key")
    for label, error in http_errors().items():
        agent = BaseAgent(settings, mode="hybrid")
        check(isinstance(agent._llm, ChatModel), f"{label}: BaseAgent uses the OpenAI-compatible ChatModel")
        agent._llm = failing_llm(error)
        reply = agent._with_fallback({"fait": 1}, [], "question", "Réponse de secours")
        check(reply.text == "Réponse de secours", f"{label}: template text returned")
        check(reply.source == TEMPLATE_SOURCE, f"{label}: source marked offline")
        check(reply.note is not None, f"{label}: failure note exposed")

    fallback_coordinator = CoordinatorAgent(settings, mode="hybrid")
    for role_agent in (fallback_coordinator, fallback_coordinator.rnn_agent,
                       fallback_coordinator.cnn_agent, fallback_coordinator.decision_agent):
        check(isinstance(role_agent._llm, ChatModel),
              f"{role_agent.name}: owns its own ChatModel client")
    fallback_coordinator._llm = failing_llm(http_errors()["rate limit"])
    session = next(s for s in sessions if s["kind"] == "normal")
    visit = summarize(session["events"], rnn.cue_thresholds())
    result = fallback_coordinator.run_checkout(session["events"], list(dict.fromkeys(visit.paid)))
    check(result.briefing.source == TEMPLATE_SOURCE and bool(result.briefing.text),
          "coordinator briefing falls back to a French template on rate limit")
    check(result.briefing.note is not None and "quota" in result.briefing.note,
          "coordinator briefing exposes the rate-limit note")

    print("\n6. LLM router (LangGraph Q&A) with keyword fallback")

    class ScriptedLLM:
        """A fake language model that always returns the same answer."""
        def __init__(self, answer):
            self.answer = answer

        def invoke(self, _messages):
            """Fake LLM call."""
            return self.answer

    router = CoordinatorAgent(settings, mode="hybrid")
    paraphrase = "Il est parti vite sans payer le chocolat ?"
    router._llm = ScriptedLLM("rnn, cnn")
    check(router.llm_route(paraphrase) == ["rnn", "cnn"], "LLM route parsed: 'rnn, cnn' -> [rnn, cnn]")
    router._llm = ScriptedLLM("Decision")
    check(router.llm_route("Pourquoi ?") == ["decision"], "LLM route is case/space tolerant")
    router._llm = ScriptedLLM("coordinator, decision")
    check(router.llm_route("x") == ["decision"], "coordinator is dropped when a specialist is also chosen")
    router._llm = ScriptedLLM("je pense que c'est le rnn")
    check(router.llm_route("x") is None, "free-text router answer is rejected (keyword fallback)")
    router._llm = failing_llm(http_errors()["rate limit"])
    check(router.llm_route("x") is None, "router LLM error -> keyword fallback")
    check(coordinator.llm_route("x") is None, "templates mode never calls the router LLM")

    router._llm = ScriptedLLM("cnn")
    state = router.qa_graph.invoke({"question": paraphrase, "result": results["conceal"],
                                    "history": [], "replies": []})
    check(state["routes"] == ["cnn"], "Q&A graph uses the LLM route when valid")
    router._llm = ScriptedLLM("n'importe quoi")
    state = router.qa_graph.invoke({"question": "Quels articles manquent ?", "result": results["conceal"],
                                    "history": [], "replies": []})
    check(state["routes"] == ["cnn"], "Q&A graph falls back to keywords on an invalid LLM route")

    print("\n" + ("FAILED: " + str(len(FAILURES)) + " check(s)" if FAILURES else "ALL CHECKS PASSED"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
