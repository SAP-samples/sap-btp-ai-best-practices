"""Inspect confidence returned by the existing AI Core decisions deployment.

Run from the rest/ folder with its Python dependencies and AI Core environment:
    python tests/probe_decisions_confidence.py
    python tests/probe_decisions_confidence.py --output /tmp/decisions-confidence.json

Sends two synthetic requests: the notebook's triage example and the public
Playground documentation's three-example variant. No deployments are created.
Only inference responses and comparisons are printed/saved, never credentials.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

REST_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REST_DIRECTORY))

from dotenv import load_dotenv  # noqa: E402
from rpt_decisions_client import RPTDecisionsClientAPI  # noqa: E402


def notebook_request() -> dict:
    """Read the notebook's literal triage request; return synthetic input only."""
    notebook = json.loads((REST_DIRECTORY / "04_RPT_decisions.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for statement in ast.parse("".join(cell["source"])).body:
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "triage"
                for target in statement.targets
            ):
                return ast.literal_eval(statement.value)
    raise ValueError("The notebook has no literal triage request.")


def confidence_comparison(response: dict) -> list[dict]:
    """Return server confidence beside winning probability for each typed answer.

    Input is an unmodified decisions response. Output records preserve missing
    confidence as None; no substitute confidence is calculated.
    """
    comparisons = []
    for key, answer in response["answers"].items():
        if answer["type"] == "noul":
            comparisons.append({"question": key, "type": "noul",
                                "probability_of_true": answer["noul"],
                                "returned_confidence": answer.get("confidence")})
            continue
        probabilities = answer["probabilities"]
        winner = answer["choice"] if answer["type"] == "choice" else max(
            probabilities, key=probabilities.get
        )
        comparisons.append({"question": key, "type": answer["type"],
                            "returned_confidence": answer.get("confidence"),
                            "winning_probability": probabilities[winner]})
    return comparisons


def main() -> None:
    """Parse optional output path, run two live probes, and emit a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON evidence file")
    args = parser.parse_args()
    load_dotenv(REST_DIRECTORY.parent / ".env", override=False)
    client = RPTDecisionsClientAPI(timeout_s=45)
    request = notebook_request()
    variants = [("notebook_five_examples", request),
                ("playground_docs_three_examples", {**request, "examples": request["examples"][:3]})]
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "model_configuration": client.model_name,
              "model_version": client.model_version, "route": "/systemone", "probes": []}
    for name, body in tqdm(variants, desc="confidence probes"):
        response = client.decide(body)
        report["probes"].append({"name": name, "request": body, "response": response,
                                 "comparison": confidence_comparison(response)})
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
