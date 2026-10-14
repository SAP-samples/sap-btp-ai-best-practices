---
name: tabpfn
description: >-
  Call the SAP TabPFN tabular foundation model (deployed as "tabpfn-3.5-plus")
  through direct SAP AI Core REST for classification and regression. Use this
  skill whenever the task involves the tabpfn-3.5-plus deployment, the TabPFN
  /predict endpoint on AI Core, building tabular classification/regression with
  TabPFN, choosing TabPFN task_config / tabpfn_config settings (n_estimators,
  random_state, categorical_features_indices), TabPFN limits/sizing/cost (cells,
  rows, features), or any TabPFN call where the SAP AI SDK has no wrapper. Also
  use for "TabPFN vs SAP-RPT / RPT-1.6" questions on SAP AI Core and for
  debugging TabPFN 400/422 responses. Prefer this skill over hand-rolling AI Core
  requests any time TabPFN on SAP is mentioned, even if the user does not name
  the endpoint explicitly.
---

# SAP TabPFN (tabpfn-3.5-plus) via direct AI Core REST

TabPFN is a tabular foundation model (Prior Labs, acquired by SAP). SAP AI Core
hosts it as the deployment **`tabpfn-3.5-plus`** (the TabPFN Plus checkpoint).
The SAP AI SDK has **no TabPFN wrapper**, so it is called over direct REST. This
skill provides a tested client, the exact request/response contract, the
per-task settings, and the limits/cost model.

TabPFN is an **in-context learner**: labelled training rows and the rows to
predict go in a *single* `/predict` call — there is no separate fit/training
step.

## When to use

Use this for any TabPFN-on-SAP task: making predictions with `tabpfn-3.5-plus`,
wiring it into an app, picking settings, estimating cost, or comparing it to
SAP-RPT. If the SAP AI SDK is what's wanted and it's an RPT model, use the RPT
path instead — TabPFN specifically has no SDK support and needs this REST client.

## Quickstart

1. Ensure a `.env` (in the working directory) provides the AI Core service key:
   `AICORE_AUTH_URL`, `AICORE_CLIENT_ID`, `AICORE_CLIENT_SECRET`,
   `AICORE_BASE_URL`, `AICORE_RESOURCE_GROUP`. (Optionally
   `AICORE_TABPFN_DEPLOYMENT_URL` / `_ID` to skip deployment discovery.)
2. Install deps: `pip install requests python-dotenv pandas`.
3. Copy `scripts/tabpfn_client_api.py` into the project and call it:

```python
from tabpfn_client_api import TabPFNClient

client = TabPFNClient()  # resolves the tabpfn-3.5-plus deployment from env

# pandas convenience (train_df has features + target; query_df has features)
result = client.predict(train_df, query_df, target="Survived", task="classification")
result["labels"]         # argmax class per query row
result["probabilities"]  # per-row class-probability vectors
result["raw"]["usage"]   # billed cells etc.

# or raw lists:
out = client.predict_raw(
    x_train=[[1.0, "a"], [2.0, "b"]], y_train=["yes", "no"],
    x_test=[[1.5, "a"]], task="regression",
    tabpfn_config={"n_estimators": 8, "categorical_features_indices": [1]},
)
```

Run `python scripts/example_predict.py` for a live classification + regression
demo against the deployment.

## The contract in one screen

`POST {deploymentUrl}/predict` with:

```json
{
  "X_train": [[...]], "y_train": [...], "X_test": [[...]],
  "task_config": { "task": "classification", "tabpfn_config": { "n_estimators": 8 } }
}
```

- `task_config.task` ∈ {`classification`, `regression`} (the discriminator).
- Response — classification: `prediction` = probability vectors ordered by
  `metadata.classes` (hard label = argmax). Regression: `prediction` = scalars.
- `usage.num_cells` = the billed size unit.

Full details, worked examples, and error semantics: **`references/api-contract.md`**.

## Settings (the part people get wrong)

Hyper-parameters do **not** go at the `task_config` top level — that layer is
strict and returns `400 extra_forbidden`. They nest under
`task_config.tabpfn_config`. The knobs that matter most:

- `n_estimators` — **capped at 8** on this deployment (`>8` → 400). Fewer = faster.
- `random_state` — reproducible per seed.
- `categorical_features_indices` — set it for integer-coded categoricals; free accuracy.
- **No thinking mode** — `thinking_effort`/`thinking_metric` are rejected (that's
  the unreleased TabPFN Thinking checkpoint, not TabPFN Plus).

Full per-task knob table and validated behavior: **`references/settings.md`**.

## Limits & cost

- FAQ: up to 1M rows / 200 features; 10k-row classification < 3s.
- Validated: `n_estimators ≤ 8`; strict config; tested to ~4k context / 100 query.
- Cost = cells: `context_rows×cols + query_rows×predict_cols` (= `usage.num_cells`).

Details and the FAQ-vs-validated distinction: **`references/limits.md`**.

## Files

- `scripts/tabpfn_client_api.py` — self-contained client (OAuth, discovery,
  `TabPFNClient.predict` / `predict_raw`). Copy into the target project.
- `scripts/example_predict.py` — runnable classification + regression demo.
- `references/api-contract.md` — endpoint, auth, request/response, errors.
- `references/settings.md` — per-task `task_config` / `tabpfn_config` knobs.
- `references/limits.md` — model limits, validated API caps, cost/cells.

## Gotchas

- **Missing values must be JSON `null`**, never `NaN` (the client handles this).
- The endpoint's `openapi.json` is disabled (405) — when something is rejected,
  read the 400/422 body; pydantic errors name the exact field/constraint.
- If deployment discovery finds zero or multiple matches, set
  `AICORE_TABPFN_DEPLOYMENT_ID` to target one directly.
