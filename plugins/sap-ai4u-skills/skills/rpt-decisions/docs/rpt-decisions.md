# SAP RPT Decisions

## Purpose

Many process steps are small decisions about one input: is this message about billing, how angry is the customer, how urgent is it, which team owns it. Each question has a fixed set of answers, and past cases with correct answers usually exist. `rpt_decisions.py` answers such questions with SAP RPT-1.6 from a handful to a few thousand labeled cases and returns a probability for every possible answer.

The request format follows the decision format of SAP's RPT Playground so requests can be reused; the translation to an AI Core table request is this asset's own design.

## Setup

Copy `assets/rpt_decisions.py` and `../rpt-1/assets/rpt_client_api.py` into one folder of the target project. Credentials and deployment discovery are those of the `rpt-1` skill (`AICORE_*` variables in the environment or a `.env` file; one `RUNNING` deployment of the chosen model in the resource group).

## Request Format

```python
triage = {
    "state": "I was charged $99 instead of $49 on my last bill, please fix this!",
    "questions": {
        "billing": {"type": "noul", "instructions": "Is this message about billing?"},
        "tone": {
            "type": "choice",
            "instructions": "What is the tone of the message?",
            "criteria": {
                "calm": "Polite, positive, or appreciative",
                "angry": "Hostile, frustrated, or threatening",
                "neutral": "Factual, neither positive nor negative",
            },
        },
        "priority": {
            "type": "score",
            "instructions": "How urgently does this need attention?",
            "criteria": ["can wait", "this week", "today"],
        },
    },
    "examples": [
        {"state": "Thanks so much, the new feature is great!",
         "answers": {"billing": False, "tone": "calm", "priority": "0"}},
        {"state": "You billed my card three times, refund me!",
         "answers": {"billing": True, "tone": "angry", "priority": "2"}},
        # ... every option of every question must appear at least once
    ],
}
```

| Field | Meaning |
| --- | --- |
| `state` | Input to decide about: a string, a JSON object (one column per key) or a list such as a chat log (stored as JSON text) |
| `questions` | 1-10 questions under short keys |
| `examples` | 2-2,048 past cases, each with a `state` and `answers` to every question |

| `type` | Use for | `criteria` | Example answer | Returned |
| --- | --- | --- | --- | --- |
| `noul` | yes/no | optional `{"true": "...", "false": "..."}` | `true` / `false` | probability of yes |
| `choice` | one of 2-255 named options | required `{"option": "description"}` | option name | winner, distribution, confidence |
| `score` | ordered scale of 2-255 levels, lowest first | required list of level labels | level index as text: `"0"`, `"1"`, ... | expected level, distribution, confidence |

## Translation to an RPT Table

`build_decision_payload` produces the `/predict` request:

- `row_id` index column: `example-0`, ..., `query-0`, ...
- the state becomes a `state` column (text, or JSON text for lists) or one column per key (objects); missing keys are filled with `null`;
- each question becomes one target column; example rows hold the answer, query rows hold `[PREDICT]`;
- each target is `task_type="classification"` with `top_k` equal to the number of options, so the full distribution returns;
- state keys that clash with `row_id` or a target column name raise `ValueError`.

`decide` sends one joint call for all questions and maps the response back; `parse_decision_answers` gives options RPT did not return probability 0.

### Encodings

RPT reads column names and cell texts for their meaning, so how the table is written can change answers. Answers are mapped back to option keys either way.

| Table part | `encoding="plain"` | `encoding="described"` (default) |
| --- | --- | --- |
| Question column name | key (`tone`) | `instructions`, cut to 100 characters; falls back to the key when missing or colliding |
| `noul` cell | `true` / `false` | `yes` / `no`, plus criteria text if given (`yes: Needs a supervisor now`) |
| `choice` cell | `angry` | `angry: Hostile, frustrated, or threatening` |
| `score` cell | `2` | level label (`today`) |
| State columns | unchanged | unchanged |

## Answer Shapes

Live output for the triage request above (5 examples, `sap-rpt-1.6`):

```python
{
    "billing":  {"type": "noul", "noul": 0.75},
    "tone":     {"type": "choice", "choice": "angry",
                 "probabilities": {"calm": 0.24, "angry": 0.46, "neutral": 0.3}, "confidence": 0.46},
    "priority": {"type": "score", "score": 1.22, "legend": {"0": "can wait", "1": "this week", "2": "today"},
                 "probabilities": {"0": 0.20, "1": 0.38, "2": 0.42}, "confidence": 0.42},
}
```

- `noul`: probability of yes.
- `choice`: `confidence` is the winner's probability; 0.46 is a much weaker answer than 0.95.
- `score`: `score` is the expected level (sum of level x probability); use it to sort by urgency, and the most likely level when one label is needed.

`decide` returns `{"model", "answers": [one map per state], "metadata"}`.

## Confidence-Gated Automation

Accept an answer automatically when `confidence` is at or above a threshold and route the rest to a person:

```python
accepted = [a for a in batch["answers"] if a["intent"]["confidence"] >= 0.8]
```

Pick the threshold from held-out labeled cases: measure, per threshold, the share handled automatically and the accuracy of those answers.

## Measured Results

400 held-out real customer messages (banking77, 10 intents), one `choice` question (intent) and one derived `noul` question (is it about the card), all 400 answered in one call per run:

| Examples per intent | Intent accuracy (plain / described) | Card yes/no accuracy (plain / described) |
| --- | --- | --- |
| 2 | 0.888 / 0.880 | 0.94 / 0.92 |
| 5 | 0.950 / 0.945 | 0.96 / 0.96 |
| 20 | 0.960 / 0.972 | 0.98 / 0.98 |
| 80 | 0.978 / 0.978 | 0.99 / 0.99 |

- Random guessing gives 10% intent accuracy.
- Confidence threshold 0.8 (80 examples per intent, described): 95% handled automatically at 99.7% accuracy, 20 of 400 sent to a person.
- Remaining mistakes were mostly between related intents (card not arrived vs card lost); clearer option descriptions or more examples of those options are the levers.
- Encodings were equal here because the intent names are self-explanatory. On a 5-example hand-written triage set, `described` fixed a wrong tone answer.

These numbers describe one dataset; evaluate on held-out labeled customer cases before using answers in a process.

## Design Guidance

- **Few examples are fragile.** With 5 or 8 hand-written examples, answers about concepts the examples do not cover (an angry message not about billing, escalation from free text) were wrong. Add labeled history before trusting a question.
- **Avoid leaking fields.** In a structured-state draft where every gold-tier ticket was escalated, the model learned "gold means escalate". Make the answer depend on the content, not on one field.
- **Model choice.** `sap-rpt-1.6` handles up to 2,048 examples; above that use `sap-rpt-1.6-large` (see `rpt-1` for limits). Up to 512 states per call.
- **Traceability.** Store questions, criteria, examples, encoding and model with every stored answer.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `ValueError: ... never appear in the examples` | Add an example for every option; RPT only predicts labels seen in context |
| `ValueError: ... clash with reserved columns` | Rename the state key that equals `row_id` or a question column name |
| `ValueError: Provide 1-512 states per call` | Split the batch |
| HTTP 422 on `/predict` | Request was sent with non-table fields; send `build_decision_payload` output only |
| Confident but wrong answers | Examples let one field give the answer away, or option meanings overlap; revise examples or descriptions |

## Validation

```bash
python tests/test_rpt_decisions.py
python -m py_compile assets/rpt_decisions.py
```
