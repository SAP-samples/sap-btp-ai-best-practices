"""Run synthetic decisions examples against an existing AI Core deployment.

From the skill folder (requires existing credentials in the process or .env):
    python scripts/live_test_rpt_decisions.py
    python scripts/live_test_rpt_decisions.py --env-file /path/to/.env --output /tmp/decisions.json

Sends eight synthetic requests and reports labels, probabilities, confidence and
score arithmetic. Quality mismatches are reported rather than treated as API
failures. Does not create deployments or submit customer datasets.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "assets"))
sys.path.insert(0, str(SKILL_ROOT.parent / "rpt-1" / "assets"))
from rpt_decisions_client import RPTDecisionsClientAPI, most_likely


def summarize(response: dict) -> dict:
    """Return selected labels and separate probabilities/confidence from raw JSON."""
    summary = {}
    for key, answer in response["answers"].items():
        selected = most_likely(answer)
        if answer["type"] == "noul":
            summary[key] = {"selected": selected, "probability_of_true": answer["noul"]}
            continue
        summary[key] = {"selected": answer.get("legend", {}).get(selected, selected),
                        "winning_probability": answer["probabilities"][selected],
                        "endpoint_confidence": answer.get("confidence")}
        if answer["type"] == "score":
            summary[key]["score"] = answer["score"]
            summary[key]["weighted_score"] = sum(int(level) * p for level, p in answer["probabilities"].items())
    return summary


def main() -> None:
    """Load optional environment, run fixture requests with progress, and save evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.env_file:
        load_dotenv(args.env_file, override=False)
    examples = json.loads((SKILL_ROOT / "assets" / "decision_examples.json").read_text())
    client = RPTDecisionsClientAPI()
    groups = [("triage", examples["triage"], examples["new_messages"]),
              ("structured_tickets", examples["ticket_routing"], examples["tickets"]),
              ("conversations", examples["chat_triage"], examples["conversations"])]
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "model_configuration": client.model_name, "model_version": client.model_version,
              "route": "/systemone", "groups": []}
    for name, request, states in groups:
        responses = client.decide_many(request, states, max_concurrency=1)
        records = [{"state": state, "response": response, "summary": summarize(response)}
                   for state, response in zip(states, responses)]
        report["groups"].append({"name": name, "records": records})
        print(json.dumps({"name": name, "summaries": [record["summary"] for record in records]}, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
