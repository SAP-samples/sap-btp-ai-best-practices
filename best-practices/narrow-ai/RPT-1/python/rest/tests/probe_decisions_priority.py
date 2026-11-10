"""Diagnose priority predictions with controlled synthetic endpoint requests.

Run from rest/ with existing AI Core credentials and dependencies:
    python tests/probe_decisions_priority.py --output /tmp/priority-probe.json

Compare joint questions, isolated score, categorical priority, reversed levels,
and changed instructions. Query example states as a memorization control only;
these controls are not held-out accuracy measurements. No deployment changes.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone

from dotenv import load_dotenv
from tqdm import tqdm

from probe_decisions_confidence import REST_DIRECTORY, notebook_request
from rpt_decisions_client import RPTDecisionsClientAPI
from decision_examples import expanded_triage_request, priority_checks


def variants(request: dict) -> dict[str, dict]:
    """Return controlled variants of the supplied synthetic triage request."""
    isolated = copy.deepcopy(request)
    isolated["questions"] = {"priority": isolated["questions"]["priority"]}
    for example in isolated["examples"]:
        example["answers"] = {"priority": example["answers"]["priority"]}
    numeric_choice = copy.deepcopy(isolated)
    numeric_choice["questions"]["priority"].update(
        type="choice", criteria={str(i): label for i, label in enumerate(
            request["questions"]["priority"]["criteria"])}
    )
    named_choice = copy.deepcopy(numeric_choice)
    names = {"0": "can_wait", "1": "this_week", "2": "today"}
    named_choice["questions"]["priority"]["criteria"] = {
        names[key]: value for key, value in numeric_choice["questions"]["priority"]["criteria"].items()
    }
    for example in named_choice["examples"]:
        example["answers"]["priority"] = names[example["answers"]["priority"]]
    reversed_score = copy.deepcopy(isolated)
    reversed_score["questions"]["priority"]["criteria"].reverse()
    for example in reversed_score["examples"]:
        example["answers"]["priority"] = str(2 - int(example["answers"]["priority"]))
    instructed = copy.deepcopy(isolated)
    instructed["questions"]["priority"]["instructions"] = (
        "Assign today to any billing complaint or access problem, this week to "
        "routine questions, and can wait to praise. Never assign can wait to a complaint."
    )
    return {"joint_score": request, "isolated_score": isolated,
            "numeric_choice": numeric_choice, "named_choice": named_choice,
            "reversed_score": reversed_score, "explicit_instructions": instructed,
            "expanded_joint_score": expanded_triage_request(request)}


def summarize(name: str, response: dict) -> dict:
    """Return actual priority, probability and arithmetic checks for one response."""
    answer = response["answers"]["priority"]
    probabilities = answer["probabilities"]
    winner = max(probabilities, key=probabilities.get)
    row = {"variant": name, "type": answer["type"], "winner": winner,
           "winning_probability": probabilities[winner],
           "confidence": answer.get("confidence"), "probabilities": probabilities}
    if answer["type"] == "score":
        row.update(score=answer["score"], label=answer["legend"][winner],
                   weighted_score=sum(int(key) * value for key, value in probabilities.items()))
    else:
        row["label"] = answer["choice"]
    return row


def main() -> None:
    """Run controlled probes; save raw requests/responses and print summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expanded-only", action="store_true",
                        help="Run just the four broader-context probes")
    parser.add_argument("--held-out-only", action="store_true",
                        help="Compare sparse and broader examples on nine fresh messages")
    args = parser.parse_args()
    load_dotenv(REST_DIRECTORY.parent / ".env", override=False)
    client = RPTDecisionsClientAPI(timeout_s=45)
    request = notebook_request()
    states = [request["state"], "Love the new dashboard, great work team!",
              "Where can I download last year's invoices?",
              "This is the worst support ever, my account is locked AGAIN."]
    tasks = [(name, {**body, "state": state})
             for name, body in variants(request).items() for state in states]
    tasks += [("example_replay", {**request, "state": example["state"]})
              for example in request["examples"]]
    if args.expanded_only:
        tasks = [(name, body) for name, body in tasks if name == "expanded_joint_score"]
    if args.held_out_only:
        tasks = [(name, {**body, "state": state})
                 for name, body in [("sparse_check", request),
                                    ("expanded_check", expanded_triage_request(request))]
                 for state, _ in priority_checks()]
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "probes": []}
    for name, body in tqdm(tasks, desc="priority controls"):
        response = client.decide(body)
        summary = summarize(name, response)
        report["probes"].append({"request": body, "response": response, "summary": summary})
        with open(args.output, "w") as output:
            json.dump(report, output, indent=2)
        tqdm.write(json.dumps({"state": body["state"], **summary}))


if __name__ == "__main__":
    main()
