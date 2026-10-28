---
name: rpt-decisions
description: Use when answering typed questions about an input (yes/no, choice among named options, ordinal scale) with probabilities from labeled examples using SAP RPT-1.6 on SAP AI Core, such as ticket triage, intent routing, tone or priority scoring, escalation flags, or confidence-gated human review.
---

# SAP RPT Decisions

A decision request describes an input (`state`: text, JSON object or chat log), up to 10 typed `questions`, and labeled `examples`. SAP RPT returns a probability for every possible answer, without training.

RPT on AI Core is a table endpoint (`/predict` requires `prediction_config` and rejects other fields with HTTP 422), so the asset translates the decision into one RPT table and one joint call: examples become context rows, states become `[PREDICT]` query rows, and each question becomes a classification target with `top_k` equal to its number of options.

## Reusable Client

Copy two files into the target project, side by side:

- [assets/rpt_decisions.py](assets/rpt_decisions.py): `decide`, `validate_decision_request`, `build_decision_payload`, `parse_decision_answers`.
- [../rpt-1/assets/rpt_client_api.py](../rpt-1/assets/rpt_client_api.py): shared OAuth, deployment discovery and `/predict`. Do not modify it; it is the same file the `rpt-1` skill ships.

```bash
python -m pip install requests requests-toolbelt python-dotenv
```

```python
from rpt_client_api import RPTClientAPI
from rpt_decisions import decide

client = RPTClientAPI(model_name="sap-rpt-1.6")
result = decide(client, request)                  # request["state"]
batch = decide(client, request, states=messages)  # up to 512 states, one call
batch["answers"][0]["tone"]  # {"type": "choice", "choice": "angry", "probabilities": {...}, "confidence": 0.46}
```

Read [docs/rpt-decisions.md](docs/rpt-decisions.md) for the request format, the table translation and encodings, answer shapes, and measured accuracy.

## Key Rules

- Every option of every question must appear in at least one example; RPT only predicts labels it has seen. `validate_decision_request` enforces this and the limits (1-10 questions, 2-2,048 examples, 2-255 options).
- `encoding="described"` (default) writes question instructions and option descriptions into the table; `"plain"` writes raw keys. Descriptions help most with few examples and terse option names; test both on labeled data.
- Use `confidence` to route uncertain answers to a person; pick the threshold from a held-out evaluation.
- Design examples so no single structured field gives the answer away.
- Use `sap-rpt-1.6`; above 2,048 examples use `sap-rpt-1.6-large`.
- Store the decision definition (questions, criteria, examples, encoding, model) with every stored answer.

## Validation

```bash
python tests/test_rpt_decisions.py
```

Offline only: validation rules, table translation for both encodings, structured and chat states, and answer parsing.
