# Diagnosing unreliable priority decisions

## Purpose

The decisions tutorial now separates a valid response from a useful business prediction. Its original five-example request returns implausible tone and priority labels. This investigation preserves that failure, explains score arithmetic and confidence, and compares broader example coverage under an explicit policy.

## Inputs and outputs

Inputs are synthetic support messages, typed questions and labeled examples. No customer data or HANA tables are required for these fixed teaching fixtures. The original request uses five examples; the broader request uses 18, six per priority with each tone represented at every priority. Nine additional synthetic messages have policy-based expected labels and are excluded from the broader context.

Outputs are unmodified endpoint responses, expected scores, most likely level labels, winning probabilities, server confidence and per-case correctness. Optional JSON probe files are diagnostic evidence. Existing environment credentials authenticate to an already running deployment; no deployment is changed.

## Findings from live tests on 10 November 2026

1. The original billing complaint returns probabilities `{0: 0.54, 1: 0.15, 2: 0.31}`, expected score `0.77`, most likely level `can wait`, and confidence `0.0`. The arithmetic is correct: `0*0.54 + 1*0.15 + 2*0.31 = 0.77`. The predicted distribution is unsuitable for the support task.
2. Expected score is an average, not a selected level. Rounding `0.77` gives level 1, while the most likely level is 0. An average can conceal a split between opposite ends of the scale.
3. Confidence is not winning probability. Replaying a labeled repeated-charge example produced `P(today)=0.74`, confidence `0.0`; praise produced `P(can wait)=0.98`, confidence `0.8824`. Score confidence behaves like a measure of concentration in these tests. Do not infer a guaranteed formula or interpret confidence as measured accuracy.
4. Removing the other questions changed the billing complaint from `{0: 0.54, 1: 0.15, 2: 0.31}` to `{0: 0.27, 1: 0.42, 2: 0.31}`. An isolated numeric `choice` question returned exactly the same distribution as the isolated `score` question, with a different confidence field. This supports a distinction between prediction distribution and typed answer summarization.
5. Named categories changed predictions; reversing numeric level assignments did not merely reverse the old probabilities. Encoding and the exact joint question set matter. These controls do not establish an endpoint defect or a universal alternative encoding.
6. Instructions alone did not enforce the priority policy. The original five examples also confound anger with urgent billing and omit blocked access. The expanded context changes both policy instructions and examples, so the improvement cannot be attributed to example count alone.
7. The broader context assigns this week to the single billing correction, can wait to praise, this week to invoice lookup, and today to blocked access. The blocked-account case still has confidence `0.2116` with winning probability `0.73`.
8. Nine fresh synthetic cases improved from **5/9** correct to **8/9**. The retained failure escalates a nonurgent single-fee correction to today (`P(today)=0.70`, confidence `0.349`). This is a small development check authored after choosing the policy and examples, not an independent production benchmark. One of the four demonstration messages also appears in the broader context, so those four examples must not be presented as held-out accuracy.
9. The saved escalation and conversation-resolution examples also contain errors. Correct team routing does not validate escalation. The banking77 benchmark evaluates intent/card questions and gives no evidence for priority, tone, escalation or resolution quality.

## Policy

- **Can wait:** feedback or future ideas with no action needed now.
- **This week:** routine information or a single billing correction without blocked work or a deadline today.
- **Today:** blocked access, outage, repeated incorrect charges or an explicit deadline today.

This policy is illustrative. Production priorities should follow the organization's documented SLA. Hard escalation requirements and deadlines should be enforced by workflow rules; statistical predictions alone do not guarantee compliance.

## Related files and key code

- `narrow-ai/RPT-1/python/rest/04_RPT_decisions.ipynb`: original failures, explanation, broader-context comparison, nine-case check and retained mistake. New cells were executed and actual outputs saved; the long banking benchmark was not rerun.
- `narrow-ai/RPT-1/python/rest/decision_examples.py`: 18 synthetic context cases and nine fresh policy checks, kept separate from client logic.
- `narrow-ai/RPT-1/python/rest/tests/probe_decisions_priority.py`: controlled endpoint probes, example replays and fresh-message checks with progress reporting.
- `narrow-ai/RPT-1/python/rest/README.md`: tutorial summary and limitations.

```python
broader_triage = expanded_triage_request(triage)
response = client.decide(broader_triage)
answer = response["answers"]["priority"]
level = most_likely(answer)
label = answer["legend"][level]
winning_probability = answer["probabilities"][level]
endpoint_confidence = answer["confidence"]
```

## Repeat the checks

From `narrow-ai/RPT-1/python/rest/`, using the configured Python environment:

```bash
python tests/probe_decisions_priority.py --output /tmp/priority-controls.json
python tests/probe_decisions_priority.py --expanded-only --output /tmp/priority-expanded.json
python tests/probe_decisions_priority.py --held-out-only --output /tmp/priority-checks.json
python tests/test_rest_offline.py
```

The default probe runs 33 synthetic requests. Expanded-only runs four; held-out-only runs 18. Five example replays in the default probe are memorization controls, not accuracy evidence. The notebook visibly shows context examples, predictions, aggregate accuracy and the remaining error. Read confidence directly; do not replace it with a reconstructed value. Validate all questions and review thresholds on a larger independently labeled set before operational use.
