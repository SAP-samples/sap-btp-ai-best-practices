# SAP RPT-1 Code Examples (Python)

This folder contains Python code examples for working with **SAP RPT-1**, a Relational Pretrained Transformer for tabular data hosted on SAP AI Core. You provide tabular rows with one or more cells marked as prediction targets, and the model returns predicted values for those cells based on patterns in the surrounding data.

The examples use the **SAP Generative AI Hub native SDK** (`gen_ai_hub.proxy.native.sap`), which provides a high-level `RPTClient` that handles authentication, deployment resolution, and request serialization internally.

## Project Structure

```
python/
├── .env.example              # Template for required environment variables
├── requirements.txt          # Python dependencies
├── RPT-1_native_sdk.ipynb    # Tutorial notebook: complete RPT-1 workflow
└── sample_orders.csv         # Sample CSV dataset used in the notebook
```

## Notebook Overview

`RPT-1_native_sdk.ipynb` is a step-by-step tutorial covering:

1. **Setup** — load credentials, initialize `RPTClient`
2. **Classification** — Pydantic models + row-based data
3. **Regression** — plain dictionary + column-based data
4. **Mixed task types** — classification + regression in a single call
5. **Response interpretation** — `Prediction` / `PredictionItem` objects, status codes, metadata
6. **Classification with `top_k`** — multiple ranked predictions per cell
7. **Async usage** — `await client.apredict()` for async workflows
8. **Pandas DataFrame workflow** — load a CSV, split context/query rows, inject placeholders, predict

## Key Concepts

| Concept | Description |
|---|---|
| `RPTClient` | SDK client that handles auth, deployment resolution, and prediction calls. |
| `RPTRequest` | Pydantic model for structured, type-safe request construction. The client also accepts plain dictionaries. |
| `TargetColumn` | Declares a column to predict, its task type (`classification` or `regression`), and optionally `top_k`. |
| `prediction_placeholder` | A sentinel value (default `"[PREDICT]"`) placed in target column cells you want the model to fill. |
| `index_column` | The column that uniquely identifies each row, so predictions can be mapped back to their source. |
| `data_schema` | A mapping of column names to `{"dtype": "string" | "numeric" | "date"}`. Optional but recommended. |
| `rows` vs `columns` | Two payload formats for tabular data. `rows` sends an array of record objects; `columns` sends a dict of column arrays. |
| Status codes | `0` = OK, `1` = warning, `2` = invalid input, `3` = server error. |

## Prerequisites

- Python 3.10+
- An SAP AI Core instance with RPT-1 available
- A running RPT-1 deployment (create one via the [AI Launchpad](https://developers.sap.com/tutorials/ai-core-generative-ai.html#6c4a539e-2bdf-4ddb-97a0-0f8d0f1bd00e))

## Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required variables:

| Variable | Description |
|---|---|
| `AICORE_AUTH_URL` | OAuth2 token endpoint for your SAP AI Core instance. |
| `AICORE_CLIENT_ID` | OAuth2 client ID from your service key. |
| `AICORE_CLIENT_SECRET` | OAuth2 client secret from your service key. |
| `AICORE_BASE_URL` | SAP AI Core API base URL. |
| `AICORE_RESOURCE_GROUP` | Resource group for deployments (default: `"default"`). |

The SDK resolves the deployment automatically — no deployment URL or ID is needed in the code or environment.

## Install

```bash
cd best-practices/narrow-ai/RPT-1/python
pip install -r requirements.txt
```

## Run

```bash
jupyter notebook RPT-1_native_sdk.ipynb
```

Or use the SDK directly in a script:

```python
from dotenv import load_dotenv
load_dotenv()

from gen_ai_hub.proxy.native.sap.client import RPTClient
from gen_ai_hub.proxy.native.sap.models import RPTRequest, PredictionConfig, TargetColumn

client = RPTClient()

body = RPTRequest(
    prediction_config=PredictionConfig(
        target_columns=[
            TargetColumn(name="COSTCENTER", task_type="classification")
        ]
    ),
    rows=[
        {"PRODUCT": "Couch",        "PRICE": 999.99, "ID": "35",  "COSTCENTER": "[PREDICT]"},
        {"PRODUCT": "Office Chair", "PRICE": 150.80, "ID": "44",  "COSTCENTER": "Office Furniture"},
        {"PRODUCT": "Server Rack",  "PRICE": 2200.00,"ID": "104", "COSTCENTER": "Data Infrastructure"},
    ],
    index_column="ID",
)

response = client.predict(body=body, model_name="sap-rpt-1-small")
print(response.predictions)
```

## What to Expect

The response object contains:

- **`status`**: `code` (0 = OK) and `message`.
- **`metadata`**: `num_rows`, `num_columns`, `num_predictions`, `num_query_rows`.
- **`predictions`**: a list of `Prediction` objects, one per query row, each containing target columns with `prediction` and `confidence` values.

Example classification output:

```
Prediction(root={
    'COSTCENTER': [PredictionItem(prediction='Office Furniture', confidence=0.89)],
    'ID': '35'
})
```

Example regression output:

```
Prediction(root={
    'DISCOUNT_RATE': [PredictionItem(prediction=0.092, confidence=None)],
    'ID': '35'
})
```

For regression, `confidence` is `None` — the model does not produce confidence scores for numeric predictions.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `401` / `403` | Invalid credentials, expired token, wrong scopes, or mismatched resource group. |
| `404` on predict | No running RPT-1 deployment, or the deployment is not in `RUNNING` state. |
| `400` with `data_schema` | Date columns must use ISO format (`YYYY-MM-DD`) when declared as `{"dtype": "date"}`. |
| `422` or status code `2` | Payload shape issue — check `target_columns`, placeholder values, and column name consistency. |
| Weak predictions | Add more representative context rows, verify data quality, and ensure consistent column types. Prefer explicit `data_schema` and `task_type`. |
