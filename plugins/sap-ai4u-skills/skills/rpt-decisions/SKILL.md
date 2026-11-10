---
name: rpt-decisions
description: Use when implementing or troubleshooting typed yes/no, categorical choice, or ordered-scale decisions from labeled examples through the sap-rpt-decisions model on SAP AI Core, including priority policies, intent routing, structured tickets, conversation topics, endpoint confidence, and held-out review thresholds.
---

# SAP RPT Decisions

Use the existing `sap-rpt-decisions` deployment (model version `1` in the verified setup) with `POST {deployment_url}/systemone`. Send one `state`, a map of typed `questions`, and labeled `examples`. The service uses examples in context; there is no separate training step. Authentication and deployment discovery reuse the `rpt-1` client, while inference uses the decisions model's own contract.

Read [docs/rpt-decisions.md](docs/rpt-decisions.md) before implementing a flow. It documents the request, answer types, confidence, validated examples, evaluation, errors and script usage.

## Resources and setup

Copy these two modules side by side into the target project:

- [assets/rpt_decisions_client.py](assets/rpt_decisions_client.py): `RPTDecisionsClientAPI`, `decide`, `decide_many`, `most_likely`, `answers_to_frame`.
- [../rpt-1/assets/rpt_client_api.py](../rpt-1/assets/rpt_client_api.py): shared OAuth, credential redaction, and discovery of an existing running deployment. Reuse unchanged.

Install [assets/requirements.txt](assets/requirements.txt), and configure the variables in [assets/.env.example](assets/.env.example). Never copy or commit populated credentials.

```bash
python -m pip install -r assets/requirements.txt
```

[assets/decision_examples.json](assets/decision_examples.json) contains complete synthetic support examples from the executed tutorial: 18-example triage, structured tickets, and conversation topics. Use them to learn the contract, then replace them with representative labeled cases for the actual policy.

```python
from rpt_decisions_client import RPTDecisionsClientAPI, answers_to_frame

client = RPTDecisionsClientAPI()  # existing sap-rpt-decisions deployment
result = client.decide(request)  # raw response; answers is a map, not a list
confidence = result["answers"]["request_kind"]["confidence"]
results = client.decide_many(request, states=messages, max_concurrency=1)
frame = answers_to_frame(results)
```

## Implementation rules

- Call `/systemone` directly. Do not build a `/predict` table, add `prediction_config`, choose `encoding`, or calculate replacement confidence. The former translator asset has been retired.
- Use 1–10 questions, 2–2,048 examples, required `instructions`, and 2–255 options/levels for `choice`/`score`. Every declared label must occur in at least one example. The service validates requests; this client forwards them unchanged.
- One state per call. `decide_many` sends one request per state, preserves order and displays progress. Start with concurrency 1 for long runs. Rate limits and transient upstream HTTP 500/502/503/504 errors use bounded retries; validation failures do not.
- `noul` is probability of true and has no separate confidence. `choice` selects a category. `score` returns an expected numeric level and probabilities: its most likely level need not equal its rounded score.
- Read `confidence` exactly as returned. It differs from winning probability. Score confidence can be zero while the selected level has probability 0.50 or higher. Keep absent confidence missing. Validate thresholds separately for each question on held-out labels.
- Define the business policy before labeling. Use relevant structured facts, representative examples and unambiguous options; avoid shortcuts such as anger always implying urgency. Joint question sets and labels can change predictions, so evaluate the exact final request design.
- Mandatory deadlines and escalation rules belong in explicit workflow rules. A plausible answer or a valid API response is not evidence of accuracy. Store the decision definition, example version and model configuration with operational results.
- Do not use the tabular model's large-context options to bypass this endpoint's 2,048-example limit. Never deploy automatically; the user performs any deployment manually.
- For applications that persist decision definitions or outputs, use HANA and automatically initialize required schemas/tables. The bundled fixed JSON is a teaching fixture, not application persistence.

## Validation scripts

```bash
python tests/test_rpt_decisions.py
python -I tests/test_rpt_decisions.py
```

Offline tests cover the route and unchanged body/answers, retries, redaction, ordered batching, flattening and missing confidence. They need no credentials.

When an existing deployment and authorized credentials are available, the [live script](scripts/live_test_rpt_decisions.py) sends eight synthetic requests, prints per-question summaries with progress, and optionally saves raw diagnostic evidence:

```bash
python scripts/live_test_rpt_decisions.py --env-file /path/to/.env --output /tmp/decisions.json
```

It checks the endpoint contract and exposes quality differences without imposing probabilistic accuracy assertions. It neither deploys nor submits customer datasets. Keep any larger labeled evaluation separate from the teaching fixtures.
