# TabPFN 3.5-plus — direct AI Core REST contract

Recovered by probing the live `tabpfn-3.5-plus` deployment (its `openapi.json`
is disabled, returning HTTP 405, so this was reconstructed from the endpoint's
validation errors and confirmed with 200 responses).

## Endpoint & auth

- **Inference:** `POST {deploymentUrl}/predict` (JSON body).
  `/v1/predict` and `/infer` do not exist (404).
- **Auth:** OAuth 2.0 client-credentials. `POST {AICORE_AUTH_URL}/oauth/token`
  with `grant_type=client_credentials` + client id/secret (form-encoded) → bearer
  token. Every call carries `Authorization: Bearer <token>` and
  `AI-Resource-Group: <AICORE_RESOURCE_GROUP>`.
- **Deployment discovery:** `GET {AICORE_BASE_URL}/v2/lm/configurations` and
  `GET .../v2/lm/deployments`, join on `configurationId`, pick the RUNNING one.
  Prefer its `deploymentUrl`; else build `{base}/v2/inference/deployments/{id}`.
  The bundled client (`scripts/tabpfn_client_api.py`) does all of this; you can
  also skip it by setting `AICORE_TABPFN_DEPLOYMENT_URL` / `_ID`.

## Request

```json
{
  "X_train": [[f1, f2, ...], ...],
  "y_train": [label, ...],
  "X_test":  [[f1, f2, ...], ...],
  "task_config": {
    "task": "classification",
    "tabpfn_config": { "n_estimators": 8, "random_state": 0 }
  }
}
```

- **In-context learner:** labelled context (`X_train`/`y_train`) and query rows
  (`X_test`) go in the **same** request. No separate fit/train step.
- `X_train` / `X_test`: list of feature rows (list-of-lists, positional columns —
  same order in both). Mixed numeric/string cells are fine; **missing values must
  be JSON `null`** (do not send `NaN`).
- `y_train`: targets aligned to `X_train`.
- `task_config`: a pydantic tagged union (`ClassifierConfig | RegressorConfig`)
  discriminated by `task` ∈ {`classification`, `regression`}. It is **strict** —
  any unknown top-level key returns 400. Hyper-parameters go **nested** under
  `task_config.tabpfn_config` (see `settings.md`).

## Response

Classification — `prediction` is one probability vector per query row, ordered by
`metadata.classes` (hard label = argmax):

```json
{
  "prediction": [[0.196, 0.803]],
  "metadata": { "classes": ["no", "yes"], "n_estimators": 8, "task": "classification",
                "tabpfn_config": { ... }, "test_set_num_rows": 1, "top_k": null },
  "usage": { "num_cells": 14, "num_predictions": 1, "X_train": {...}, "X_test": {...} }
}
```

Regression — `prediction` is one scalar per query row:

```json
{
  "prediction": [2.81, 6.07],
  "metadata": { "task": "regression", "n_estimators": 8, "tabpfn_config": { ... },
                "test_set_num_rows": 2 },
  "usage": { "num_cells": 19, "num_predictions": 2 }
}
```

- `metadata.tabpfn_config` echoes the **resolved** settings (defaults + your
  overrides) — useful to confirm what actually ran.
- `usage.num_cells` is the billable size unit (see `limits.md`).

## Errors

- **400** custom envelope `{"error_code": "USER_ERROR", "message": ..., "trace_id": ...}`
  — semantic problems (missing `X_test`; no valid `task_config`; value out of
  range, e.g. `n_estimators` > 8).
- **422** raw pydantic validation — schema/type violations. These are the most
  informative signal when reverse-engineering: a discriminated-union error names
  the discriminator (`task`) and the valid tags; `extra_forbidden` names the
  rejected key. Read them rather than guessing.

## Minimal call (Python)

```python
from tabpfn_client_api import TabPFNClient
client = TabPFNClient()  # resolves tabpfn-3.5-plus from AICORE_* env
out = client.predict_raw(
    x_train=[[1.0, "a"], [2.0, "b"], [1.5, "a"], [3.0, "b"]],
    y_train=["yes", "no", "yes", "no"],
    x_test=[[1.2, "a"]],
    task="classification",
)
# out["prediction"] -> [[P(no), P(yes)]]; out["metadata"]["classes"] -> ["no","yes"]
```

See `scripts/example_predict.py` for a runnable classification + regression demo,
and `scripts/tabpfn_client_api.py` for a pandas-DataFrame convenience method.
