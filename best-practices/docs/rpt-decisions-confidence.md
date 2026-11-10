# RPT decisions: endpoint-provided confidence

## Purpose

Notebook `04_RPT_decisions.ipynb` now reads confidence directly from the existing AI Core decisions endpoint and distinguishes it from the probability of the winning answer. The former local confidence formulas were diagnostic comparisons; neither the REST client nor its DataFrame conversion used them to generate confidence. Those formulas have been removed to avoid suggesting a guaranteed service contract.

## Inputs and outputs

Inputs are the notebook's synthetic customer-message triage request (five labeled examples) and the public Playground documentation's three-example variant, each containing `state`, `questions` and `examples`. Questions cover `noul`, `choice` and `score`. Existing AI Core credentials come from the environment or `narrow-ai/RPT-1/python/.env`; no credentials are printed or saved.

The probe returns the unmodified inference JSON and a comparison of returned confidence against winning probability. Its optional JSON output is diagnostic evidence, not application data storage. No customer dataset is submitted and no deployment is created or changed.

## Live results — 10 November 2026

Configuration: `sap-rpt-decisions`, model version `1`; route: `POST {deployment_url}/systemone`. Both requests succeeded.

| Request | Question | Returned confidence | Winning probability |
|---|---|---:|---:|
| Notebook, five examples | tone (`choice`) | 0.415 | 0.61 |
| Notebook, five examples | priority (`score`) | 0.0 | 0.54 |
| Playground docs, three examples | tone (`choice`) | 0.175 | 0.45 |
| Playground docs, three examples | priority (`score`) | 0.0 | 0.36 |

`billing` (`noul`) returned probability of true `0.85` and `0.45`, respectively, with no separate confidence member.

The [Playground documentation](https://rpt.cloud.sap/docs/api/decisions), also reproduced in the user-supplied PDF, describes `/api/systemone` using `sap-rpt-1.6`. It defines confidence as the winning choice probability or most likely score-level probability. The tested AI Core configuration instead returns `model: sap-rpt-decisions`, and the values above do not follow that definition. Playground's example numbers are illustrative, not expected outputs for our deployment.

The endpoint does supply confidence directly. A response cannot establish whether that field comes from the underlying model itself or service-side postprocessing, or whether another deployment/version behaves differently. We did not call the authenticated Playground API: the configured credentials are for AI Core. Do not replace returned confidence with a locally inferred value or claim the observed formula as a documented contract.

## Key code

```python
result = client.decide(triage)  # unchanged server JSON
answer = result["answers"]["tone"]
endpoint_confidence = answer["confidence"]
winning_probability = answer["probabilities"][answer["choice"]]
```

For `score`, `max(answer["probabilities"].values())` gives the most likely level's probability, while `answer["confidence"]` remains a separate returned value. For `noul`, read `answer["noul"]` as probability of true. `answers_to_frame` preserves absent confidence as missing rather than deriving a substitute. Validate confidence thresholds on held-out data for the deployed model.

## Related files

- `narrow-ai/RPT-1/python/rest/04_RPT_decisions.ipynb`: direct confidence reading, live comparison and historical benchmark diagnostic.
- `narrow-ai/RPT-1/python/rest/rpt_decisions_client.py`: direct `/systemone` call and confidence-preserving DataFrame conversion.
- `narrow-ai/RPT-1/python/rest/rpt_client_api.py`: unchanged OAuth and existing-deployment discovery.
- `narrow-ai/RPT-1/python/rest/tests/probe_decisions_confidence.py`: repeatable two-request synthetic probe with progress reporting.
- `narrow-ai/RPT-1/python/rest/tests/test_rest_offline.py`: raw response preservation and absent-confidence checks.
- `narrow-ai/RPT-1/python/rest/README.md`: corrected direct endpoint usage; removed references to a nonexistent local translator.

## Verification

From `narrow-ai/RPT-1/python/rest/`, using the repository virtual environment:

```bash
python tests/probe_decisions_confidence.py --output /tmp/decisions-confidence.json
python tests/test_rest_offline.py
```

The live probe authenticates, discovers an existing deployment and sends two requests; it incurs normal inference usage. The offline suite sends no requests. Notebook Python cells were compiled for syntax. The notebook's earlier 400-message banking77 results were retained and were not rerun during this focused investigation.

## Raw synthetic responses

### notebook_five_examples

```json
{
  "answers": {
    "billing": {
      "noul": 0.85,
      "type": "noul"
    },
    "priority": {
      "confidence": 0.0,
      "legend": {
        "0": "can wait",
        "1": "this week",
        "2": "today"
      },
      "probabilities": {
        "0": 0.54,
        "1": 0.15,
        "2": 0.31
      },
      "score": 0.77,
      "type": "score"
    },
    "tone": {
      "choice": "calm",
      "confidence": 0.415,
      "probabilities": {
        "angry": 0.28,
        "calm": 0.61,
        "neutral": 0.11
      },
      "type": "choice"
    }
  },
  "metadata": {
    "context_mode": "default",
    "num_columns": 4,
    "num_predictions": 3,
    "num_query_rows": 1,
    "num_rows": 6
  },
  "model": "sap-rpt-decisions",
  "usage": {
    "input_tokens": 21,
    "output_tokens": 3
  }
}
```

### playground_docs_three_examples

```json
{
  "answers": {
    "billing": {
      "noul": 0.45,
      "type": "noul"
    },
    "priority": {
      "confidence": 0.0,
      "legend": {
        "0": "can wait",
        "1": "this week",
        "2": "today"
      },
      "probabilities": {
        "0": 0.3,
        "1": 0.34,
        "2": 0.36
      },
      "score": 1.06,
      "type": "score"
    },
    "tone": {
      "choice": "neutral",
      "confidence": 0.175,
      "probabilities": {
        "angry": 0.13,
        "calm": 0.42,
        "neutral": 0.45
      },
      "type": "choice"
    }
  },
  "metadata": {
    "context_mode": "default",
    "num_columns": 4,
    "num_predictions": 3,
    "num_query_rows": 1,
    "num_rows": 4
  },
  "model": "sap-rpt-decisions",
  "usage": {
    "input_tokens": 13,
    "output_tokens": 3
  }
}
```

