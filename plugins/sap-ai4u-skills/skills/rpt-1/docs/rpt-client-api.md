# SAP RPT-1.5/1.6 Direct AI Core Client

## Purpose

`RPTClientAPI` is a copyable Python client for current SAP RPT inference features that are not represented by older `gen_ai_hub.proxy.native.sap` request and response models. It sends and returns plain dictionaries, preserving the service contract without maintaining a parallel Pydantic schema.

The template supports classification, regression, joint multi-target inference, `top_k`, confidence intervals, explainability, row/column JSON, gzip, Parquet, RPT-1.6 context modes, and synchronous or asynchronous call sites.

## Install and Copy

Copy `assets/rpt_client_api.py` into the consuming project. It requires:

```bash
python -m pip install requests requests-toolbelt python-dotenv
```

The asynchronous methods use `asyncio.to_thread`; they share the same tested `requests` transport instead of introducing a second HTTP client.

## Environment and Deployment Resolution

Put AI Core credentials in the environment or a local `.env` excluded from version control:

```dotenv
AICORE_AUTH_URL=https://your-auth-host
AICORE_CLIENT_ID=...
AICORE_CLIENT_SECRET=...
AICORE_BASE_URL=https://your-ai-api-host/v2
AICORE_RESOURCE_GROUP=default
```

Construction obtains an OAuth token, lists `foundation-models` configurations for executable `aicore-sap`, matches exact `modelName` and `modelVersion` bindings, and joins running deployments through `configurationId`. The base URL may include or omit `/v2`. Zero matches and multiple running matches are errors.

```python
from rpt_client_api import RPTClientAPI

client = RPTClientAPI()  # sap-rpt-1.6, model version 1
large_client = RPTClientAPI(model_name="sap-rpt-1.6-large")
```

OAuth tokens are cached and refreshed before expiry. Credentials and bearer tokens are never included in client errors.

## JSON Prediction

Context rows have known target values. Query rows use the configured prediction placeholder. Include an index column so responses can be joined back to inputs.

```python
from rpt_client_api import RPTClientAPI

payload = {
    "index_column": "row_id",
    "prediction_config": {
        "target_columns": [
            {
                "name": "segment",
                "prediction_placeholder": "[PREDICT]",
                "task_type": "classification",
                "top_k": 2,
            },
            {
                "name": "amount",
                "prediction_placeholder": "[PREDICT]",
                "task_type": "regression",
            },
        ],
        "explanations": {
            "top_column_scores": 4,
            "top_relevant_context_rows": 3,
        },
        "context_mode": "default",
    },
    "parse_data_types": True,
    "data_schema": {
        "row_id": {"dtype": "string"},
        "feature": {"dtype": "numeric"},
        "segment": {"dtype": "string"},
        "amount": {"dtype": "numeric"},
    },
    "rows": [
        {"row_id": "c-1", "feature": 10, "segment": "A", "amount": 90.0},
        {"row_id": "c-2", "feature": 20, "segment": "B", "amount": 130.0},
        {
            "row_id": "q-1",
            "feature": 15,
            "segment": "[PREDICT]",
            "amount": "[PREDICT]",
        },
    ],
}

response = RPTClientAPI().predict_json(payload)
```

Set `compress=True` to gzip the serialized body. For column-oriented JSON, replace `rows` with equal-length arrays under `columns`. RPT evaluates multiple target columns jointly, so results can differ from independent single-target requests.

## Asynchronous Calls

The async methods have the same arguments and return the same raw dictionaries:

```python
response = await client.apredict_json(payload, compress=True)
```

`apredict_json` and `apredict_parquet` move the blocking request to a worker thread so an existing event loop remains responsive.

## Parquet Prediction

`/predict_parquet` sends `file`, JSON-string `prediction_config`, optional `index_column`, and `parse_data_types` as multipart fields.

```python
prediction_config = {
    "target_columns": [
        {
            "name": "amount",
            "prediction_placeholder": "[PREDICT]",
            "task_type": "regression",
        }
    ],
    "explanations": {
        "top_column_scores": 4,
        "top_relevant_context_rows": 3,
    },
}

response = client.predict_parquet(
    "request.parquet",
    prediction_config,
    index_column="row_id",
    parse_data_types=True,
)
```

Parquet columns have one physical type. If a numeric target also contains a string placeholder, convert every known target value to a string, use the placeholder for query rows, set `task_type="regression"`, and keep `parse_data_types=True`.

## Response Contract and Helpers

Successful responses remain unmodified dictionaries:

- `status.code`: `0` success, `1` warning, `2` invalid request, `3` server error.
- `predictions`: one record per query row and one candidate list per target.
- `explanations`: optional arrays ordered by query row.

HTTP failures and application codes 2/3 raise `RPTAPIError`. Code 1 is returned unchanged.

```python
from rpt_client_api import flatten_predictions, map_explanation_rows

records = flatten_predictions(response, index_column="row_id")
mapped = map_explanation_rows(response, payload["rows"])
```

Flattened records contain `query_position`, optional `index`, `target`, `rank`, `prediction`, `confidence`, and `confidence_interval`. Explanation mapping also accepts `payload["columns"]`. Relevant-context values are zero-based positions into the complete input—not index-column values or positions in a context-only subset.

## How to Interpret Explanations

`top_column_scores` are attention-based importance scores for each query row. They describe what the model attended to for that prediction; they do not establish causality. Strongly correlated features can share importance rather than producing one dominant column.

`top_relevant_context_rows` are context examples ranked by model relevance. They are not similarity distances, counterfactuals, or proof that a context row caused the prediction.

Live validation on deterministic Boston and Titanic holdouts produced these behaviors:

| Probe | Observed behavior | Usage guidance |
| --- | --- | --- |
| Explanations omitted | The response had no `explanations` object | Request explanations only when consumers use them |
| Scores only / rows only | Either explanation field worked independently | Ask only for the diagnostic needed |
| Counts 1, 4, and 20 | Returned rankings were stable prefixes for all eight tested queries | Smaller counts are sufficient for concise displays |
| `top_column_scores=20` with 13 eligible features | Returned 13 scores; their rounded sum was approximately 1 | The result is capped by available feature columns |
| Top four scores | Rounded sums were 0.50–0.61 | Do not renormalize a truncated map or label it as percentages |
| Relevant rows | Every returned index was unique and pointed to a context row, never a query row | Resolve indices against the complete submitted table |
| Explanation depth changed | Predictions matched the no-explanation request in every tested variant | Explanation settings did not alter these tested predictions |
| RPT-1.5 versus RPT-1.6 | Top feature agreed on 8/8 regression queries; top-three relevant-row mean Jaccard overlap was 0.675; predictions differed | Revalidate explanations after changing model version |
| Titanic classification | Relevant context labels agreed with the predicted class 75% of the time, versus a 38.3% positive-class context rate | Relevant rows can reveal which labeled examples support a prediction, but agreement is not guaranteed |
| Joint two-target request | Returned one explanation entry per query row, not one per query-target pair, and contained no target label | Treat joint explanations as row-level; use separate calls when target-specific attribution is required |
| Count 21 | Rejected with HTTP 422 and application status 2 | Keep each requested explanation count between 0 and 20 |

The numerical observations above describe one live validation run, not universal quality guarantees. For customer data, review explanation stability across representative batches, model versions, and correlated-feature groups before using the output in a business decision or user-facing rationale.

When target-specific explanations matter, separate calls provide separately scoped explanations but no longer reproduce the joint multi-target prediction problem. Store both the prediction configuration and model version with any explanation used for audit.

## Model Guidance and Limits

| Model | Use |
| --- | --- |
| `sap-rpt-1.5` | Small compatibility model with explainability |
| `sap-rpt-1.5-large` | Large compatibility model with explainability |
| `sap-rpt-1.6` | Default small RPT-1.6 model |
| `sap-rpt-1.6-large` | Large context, high cardinality, and deep mode |

Current RPT-1.6 guidance:

| Limit | Small | Large |
| --- | ---: | ---: |
| Context rows | 2,048 | 65,536 |
| Columns | 100 | 256 |
| Recommended target classes | 250 | 5,000 |
| Query rows per request | 512 | 512 |
| Target columns per request | 10 | 10 |

Large-model context up to about 36,000 rows is recommended. Use `context_mode="deep"` only with `sap-rpt-1.6-large` and more than 8,000 context rows; expect higher latency and cost. Explanation counts may each be at most 20.

## Platform Boundaries

| Surface | Supported API behavior |
| --- | --- |
| SAP AI Core production inference | Deployment `/predict` and `/predict_parquet`; explanations live inside `prediction_config` |
| RPT Playground evaluation API | `https://rpt.cloud.sap/api/predict`; API token; different request fields and wrapped response |
| Chat-RPT and what-if | Web-UI capabilities; no documented public chat endpoint as of 2026-09-10 |
| Tabular Orchestration | Separate service surface; not part of `RPTClientAPI` |

The Playground can accept inline data or an HTTPS CSV `dataSource`. It samples and wraps the underlying AI request/response and is not a production substitute for AI Core.

## Validation

```bash
python tests/test_rpt_client_api.py
python -m py_compile assets/rpt_client_api.py scripts/live_test_rpt_client_api.py
python scripts/live_test_rpt_client_api.py --dataset-root /path/to/datasets
```

The live matrix checks all four models, both task types, `top_k`, confidence intervals, explanations, joint targets, row/column JSON, gzip, Parquet, sync/async paths, and RPT-1.6-large deep mode. Contract failures return nonzero. MAE, RMSE, R-squared, interval coverage/width, accuracy, macro-F1, and top-k accuracy are report-only.

## Troubleshooting

- Missing environment variable: define every `AICORE_*` value above; never copy credentials into code.
- No configuration match: confirm the exact model name and `model_version="1"`.
- Multiple running deployments: resolve the duplicate outside the client; it will not guess.
- `401`/`403`: verify OAuth credentials, scopes, and resource group.
- Status 2: inspect `RPTAPIError.response_payload` for validation details.
- Weak predictions: improve context relevance, label consistency, class balance, and leakage before adding rows.
- Missing advanced SDK fields: use this direct dictionary client.
