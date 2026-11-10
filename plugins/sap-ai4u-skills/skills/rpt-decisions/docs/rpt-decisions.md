# SAP RPT decisions: direct endpoint guide

## What the model does

`sap-rpt-decisions` answers fixed, typed questions about one input using labeled examples within that same request. Inputs can be text, a JSON object, or a list such as conversation turns. The endpoint produces distributions over the demonstrated labels, with typed summaries. It is statistical prediction, not enforcement of instructions as hard rules, and needs at least two examples.

The verified AI Core configuration is scenario `foundation-models`, executable `aicore-sap`, model name `sap-rpt-decisions`, model version `1`. The client discovers an existing unique `RUNNING` deployment in the configured resource group; it creates no resources. Model version identifies the deployment configuration, and should not be inferred from an underlying tabular model.

## Resource map and installation

| Resource | Purpose |
|---|---|
| `assets/rpt_decisions_client.py` | Direct inference, ordered batching, returned-answer helpers |
| `../rpt-1/assets/rpt_client_api.py` | Shared authentication, redaction and deployment discovery |
| `assets/requirements.txt` | requests, requests-toolbelt, python-dotenv, pandas and tqdm dependencies |
| `assets/.env.example` | Credential variable names, without secrets |
| `assets/decision_examples.json` | Complete synthetic requests and demonstration query states |
| `tests/test_rpt_decisions.py` | Offline contract checks with fake responses |
| `scripts/live_test_rpt_decisions.py` | Eight synthetic live requests and optional raw JSON evidence |

Copy the two clients into one project folder and install the requirements. Supply `AICORE_AUTH_URL`, `AICORE_CLIENT_ID`, `AICORE_CLIENT_SECRET`, `AICORE_BASE_URL`, and `AICORE_RESOURCE_GROUP` in the process or a `.env` discoverable from its working directory. The live script additionally supports an explicit `--env-file`. Existing environment values take precedence. No credentials or deployment URLs are included in report output.

```bash
python -m pip install -r assets/requirements.txt
python tests/test_rpt_decisions.py
python scripts/live_test_rpt_decisions.py --env-file /path/to/.env --output /tmp/decisions.json
```

## Request contract

```text
POST {deployment_url}/systemone
Content-Type: application/json
Authorization: Bearer <existing OAuth token>
AI-Resource-Group: <configured group>
```

```python
request = {
    "state": "Please restore access today: the entire department cannot sign in.",
    "questions": {
        "blocked": {
            "type": "noul",
            "instructions": "Is work blocked by an access problem?",
            "criteria": {"true": "Work cannot continue", "false": "Work continues"},
        },
    },
    "examples": [
        {"state": "All staff cannot sign in and cannot work.", "answers": {"blocked": True}},
        {"state": "Please send documentation; work continues.", "answers": {"blocked": False}},
    ],
}
result = client.decide(request)
```

| Field | Meaning |
|---|---|
| `state` | Exactly one string, JSON object or list; each list is one query |
| `questions` | Map of 1–10 named questions |
| `examples` | 2–2,048 states with answers to every question |

| Type | Criteria | Example label | Response |
|---|---|---|---|
| `noul` | Optional descriptions keyed by `true` and `false` | Boolean | `noul`: probability of true |
| `choice` | Required map of 2–255 option keys to descriptions | Option key string | `choice`, `probabilities`, `confidence` |
| `score` | Required ordered array of 2–255 level descriptions, lowest first | Index string such as `"0"` | `score`, `legend`, `probabilities`, `confidence` |

`instructions` is required for every question. Every declared choice option and score level must occur in an example. The client performs no local translation or payload validation: server errors describe rejected fields and labels. Use only `state`, `questions`, and `examples`; a `states` field, model selector in the inference body, `encoding`, and tabular prediction configuration are not part of this contract.

Text becomes one state column; structured objects contribute individual fields, and a list remains a single state. The service handles the internal representation. Do not depend on a locally invented table encoding or assume adding more example rows beyond the endpoint limit is supported.

## Response and direct confidence

The successful JSON contains an `answers` map, `metadata`, `model`, and `usage`; it does not require `status.code`.

Actual response from the 18-example support fixture, checked on 10 November 2026:

```json
{
  "billing": {"type": "noul", "noul": 0.0},
  "request_kind": {
    "type": "choice", "choice": "technical",
    "probabilities": {"billing": 0.0, "feedback": 0.0, "technical": 1.0},
    "confidence": 1.0
  },
  "priority": {
    "type": "score", "score": 1.89,
    "legend": {"0": "can wait", "1": "this week", "2": "today"},
    "probabilities": {"0": 0.01, "1": 0.09, "2": 0.9},
    "confidence": 0.703
  }
}
```

The snippet is the `answers` member, not the entire response. For this text request metadata reports 19 rows (18 examples plus one query), four columns (state plus three questions), three predictions and one query row. `model` is reported by the service, and `usage` contains reported input/output tokens.

- `noul=0.0` means probability of true is zero; a 0.5 threshold gives false in the demonstration. Different costs can justify another threshold.
- `choice` is the selected category, while `probabilities` is the full returned distribution.
- `score` is a probability-weighted expected index: `0*0.01 + 1*0.09 + 2*0.90 = 1.89`. The most likely level is `2`, interpreted using `legend`. Rounding the expected index can differ from choosing the largest probability, especially if the distribution spans opposite extremes.
- `confidence` is copied directly, not reconstructed. It is a separate quantity from winning probability and is not measured accuracy. Score confidence behaves like concentration in observed tests, but the exact formula is not established as a stable model contract.

A deadline query — “Please correct my invoice before today's payment cutoff in two hours.” — returned expected level `1.26`, most likely level today, `P(today)=0.50`, and confidence `0.0`. This means the leading label has substantial uncertainty; it does not mean today has probability zero. Use the full distribution and enforce mandatory deadlines through workflow rules. Confidence-zero responses are valid results and must not be overwritten.

```python
answer = result["answers"]["priority"]
level = most_likely(answer)
label = answer["legend"][level]
expected_score = answer["score"]
winning_probability = answer["probabilities"][level]
endpoint_confidence = answer["confidence"]
```

`most_likely` thresholds `noul` at 0.5, returns the selected `choice`, and selects the highest-probability `score` index. Equal score probabilities use the first encountered maximum; do not treat a tie as decisive. `answers_to_frame` keeps raw expected scores and confidence, adds a most-likely score level, and leaves absent confidence missing. It does not flatten full distributions; retain raw responses when they matter.

## Several inputs

```python
responses = client.decide_many(request, messages, max_concurrency=1)
frame = answers_to_frame(responses, index=message_ids)
```

This makes one inference call per state; it is not a 512-query single-call batch. Responses remain in input order. The default concurrency is four, but concurrency one reduces bursts in long evaluations. `tqdm` reports progress. When a request exhausts retries, the batch raises an error; it does not return a partial-success envelope. Authentication is fetched before workers start.

HTTP 429 and transient 500/502/503/504 responses are retried up to five times after the initial attempt. `Retry-After` is respected when numeric; otherwise delays are 10, 20, 30, 40 and 50 seconds. HTTP validation errors are not retried. Transport failures raise `RPTAPIError`. Retrying inference can add usage, and server rate limits remain in effect.

## Designing the examples

The bundled triage fixture covers billing (`noul`), request kind (`choice`), and priority (`score`) with 18 examples. Its priority policy is:

- Can wait: feedback or future suggestions without action now.
- This week: routine information or a single billing correction without blocked work or a deadline today.
- Today: blocked access, outage, repeated charges, or a deadline today.

Polite, factual and frustrated examples occur at every priority. This prevents anger becoming a substitute for urgency. Representative labels matter more than merely satisfying every-label coverage.

Structured ticket examples use affected-user count and blocked-work status alongside message and channel; routine and critical cases occur for every team. These are relevant facts, while arbitrary customer tiers or channel correlations can create shortcuts. Conversation examples classify access, billing and export topics across both resolved and unresolved conversations; the whole list is one input.

The source tutorial's revised demonstration inputs returned their expected labels. That does not establish general accuracy. Earlier sparse examples gave poor priorities; subjective tone and free-text resolution also proved fragile. Pick concrete policy definitions, include relevant history and facts, inspect mistakes, and evaluate each question independently. Changing the joint question set or option keys can alter predictions; validate the exact production request shape.

These fixtures are fixed synthetic teaching resources. For an application storing decision definitions, cases or results, use HANA and automatically create required schema/tables before inserting; do not build runtime persistence around local JSON/CSV/Excel. Keep versioned fixture loading separate from operational persistence.

## Held-out evaluation and review

The complete source notebook ran 1,600 direct inference requests on 10 November 2026: four context sizes, each tested on 400 held-out banking77 messages with ten intents. Training examples and test messages came from separate original splits. Intent is `choice`; card-relatedness is `noul`, derived from intent. No priority labels were evaluated in this benchmark.

| Examples per intent | Total examples | Intent accuracy | Card accuracy |
|---|---:|---:|---:|
| 2 | 20 | 0.925 | 0.962 |
| 5 | 50 | 0.962 | 0.970 |
| 20 | 200 | 0.968 | 0.982 |
| 80 | 800 | 0.970 | 0.985 |

At 80 examples per intent, a returned intent-confidence threshold of 0.8 accepted 94.8% of messages at 98.9% accuracy, referring 21 messages. These are results on one dataset, not guarantees or transferable priority thresholds.

```python
accepted = [response for response in responses
            if response["answers"]["intent"].get("confidence") is not None
            and response["answers"]["intent"]["confidence"] >= threshold]
```

Select thresholds by measuring accepted coverage and accepted accuracy on held-out labels, including cases where no answer is accepted. Assess class imbalance, costs and mandatory rules. The live script supplies contract and example checks, not an independent benchmark; it reports responses rather than failing on expected-label mismatches.

## Errors and troubleshooting

| Symptom | Meaning and response |
|---|---|
| No matching configuration or running deployment | Verify existing credentials, resource group, model name/version; user deploys manually if needed |
| Multiple running deployments match | Resolve the configuration ambiguity rather than selecting silently |
| HTTP 400: subpath `predict` not allowed | Use `/systemone` via `RPTDecisionsClientAPI` |
| HTTP 422: missing instructions / extra fields | Correct the typed request; do not add tabular or batch fields |
| HTTP 400: label never demonstrated | Add representative labeled examples covering every declared option/level |
| HTTP 429 after retries | Reduce concurrency, wait for quota recovery, and check deployment limits |
| HTTP 500/502/503/504 after retries | Upstream remains unavailable; preserve diagnostics and retry later |
| Score confidence zero | Inspect distribution and business policy; zero is a valid returned value |
| Confident but wrong answers | Check labels, shortcut fields, overlap and held-out errors; confidence is not a quality guarantee |

## Offline and live verification

```bash
python tests/test_rpt_decisions.py
python -I tests/test_rpt_decisions.py
python -m py_compile assets/rpt_decisions_client.py scripts/live_test_rpt_decisions.py
python scripts/live_test_rpt_decisions.py --env-file /path/to/.env --output /tmp/decisions.json
```

The offline suite mocks transport and checks unchanged route/body/answers, HTTP retries, redacted errors, order, score/Boolean helpers, flattening and absent confidence. The live script loads the synthetic fixture, sends four triage messages, two structured tickets and two conversations, prints selected labels with probabilities/confidence, and optionally writes raw diagnostic JSON. It does not read banking77, create tables, deploy, or submit customer data.

The former `rpt_decisions.py` translator and its `build_decision_payload`, `parse_decision_answers`, `validate_decision_request`, and `encoding` APIs have been removed. Update consumers to import `rpt_decisions_client`, use the dedicated deployment and direct request, and read an `answers` map per response rather than a locally built list envelope.
