#!/usr/bin/env python3
"""Run a single TrustCart demo scenario from the CLI and print every workflow stage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from trustcart import rnn  # noqa: E402
from trustcart.agents import CoordinatorAgent  # noqa: E402
from trustcart.visit import summarize  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate a TrustCart checkout scenario.")
    parser.add_argument("kind", choices=["normal", "conceal", "alibi", "spree", "false_alarm"],
                        help="Demo scenario to run.")
    parser.add_argument("--mode", default="templates", choices=["templates", "hybrid", "full_llm"],
                        help="Agent narration mode (default: templates).")
    parser.add_argument("--swap", metavar="CATEGORY",
                        help="Substitution fraud: the most expensive declared item is photographed "
                             "as this CNN category instead of its own.")
    args = parser.parse_args()

    sessions = json.loads((ROOT / "app" / "data" / "demo_sessions.json").read_text(encoding="utf-8"))
    events = next(s["events"] for s in sessions if s["kind"] == args.kind)
    visit = summarize(events, rnn.cue_thresholds())
    presented = list(dict.fromkeys(visit.paid))
    swaps = None
    if args.swap:
        target = max(presented, key=lambda i: visit.items[i].price)
        swaps = {target: args.swap}

    coordinator = CoordinatorAgent(mode=args.mode)
    res = coordinator.run_checkout(events, presented, swaps=swaps)
    question = "Pourquoi cette alerte ?" if res.is_alert else "Comment aborder le client ?"
    reply = coordinator.answer(question, res, [])

    print(f"OUTCOME: {res.outcome}")
    print(f"[1] RNN: {res.path.summary.text}")
    print(f"[2] CNN: {res.checkout.summary.text}")
    print(f"[3] DECISION: {res.decision.summary.text}")
    print(f"[4] CUSTOMER: {res.customer.title} | {res.customer.body}")
    print(f"[5] BRIEFING: {res.briefing.text}")
    print(f"[6] Q&A ({reply.agent_name}): {reply.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
