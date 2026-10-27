# SAP RPT on SAP AI Core with direct REST calls (Python)

Jupyter notebooks and small support modules for the current SAP RPT models on SAP AI Core, called with plain HTTP requests:

- **SAP-RPT-1.5 / SAP-RPT-1.6**: tabular classification and regression with explanations, confidence intervals and `context_mode`.
- **SAP-RPT-1.6-large deep context**: more than 8,000 context rows with `context_mode="deep"`.
- **SAP-RPT-TS**: time-series forecasting (`sap-rpt-ts`).
- **Decisions**: typed questions about an input (yes/no, choice, ordered scale) answered with probabilities, built on the RPT table endpoint.

## Why direct REST

The native SAP AI SDK (`gen_ai_hub.proxy.native.sap.RPTClient`, used in `../RPT-1_native_sdk.ipynb`) models the original RPT-1 request and response fields only. It drops the newer fields (explanations, confidence intervals, `context_mode`) and has no client for SAP-RPT-TS. These notebooks send and receive plain dictionaries, so every field of the current service contract is kept.

## Files

```
rest/
  01_RPT-1.6_predict.ipynb         RPT-1.5/1.6: raw call anatomy, joint classification + regression, holdout comparison,
                                   top_k, confidence intervals, explanations, model comparison
  02_RPT-1.6_deep_context.ipynb    sap-rpt-1.6 vs sap-rpt-1.6-large default vs deep, 3 test months, ~15,000 context rows
  03_RPT-TS_forecast.ipynb         SAP-RPT-TS: request fields, rules, rolling backtest, quantiles, hierarchy
  04_RPT_decisions.ipynb           decisions (noul / choice / score) as RPT tables, accuracy on banking77
  rpt_client_api.py                support: OAuth, deployment discovery, /predict for tabular RPT
  rpt_ts_client.py                 support: RPTTSClientAPI, forecast_to_frame, hierarchy_to_frame
  rpt_decisions.py                 support: decision validation, table translation, answer parsing
  tests/test_rest_offline.py       offline checks of the support modules
  data/hotel_nights_2005_2026.csv  Swiss hotel nights per region and guest origin, 2005-2026 (notebook 03)
  data/OVERNIGHTSTAYS.csv          Swiss hotel nights per region and country, 2022-2024 (notebooks 02, 03)
  data/employee_sample.csv         2,500-row sample of the SAP HANA ML Challenge employee data (notebook 01)
  data/banking77_10_intents.csv    banking77 customer messages, 10 intents (notebook 04)
  requirements.txt, .env.example
```

`rpt_client_api.py` is a copy of the client shipped with the `sap-ai4u-skills:rpt-1` skill; keep it unchanged so both stay in sync.

## Prerequisites

1. SAP AI Core instance with running deployments in scenario `foundation-models`, executable `aicore-sap`, model version `1`:

   | Model name | Used in |
   |---|---|
   | `sap-rpt-1.5`, `sap-rpt-1.5-large` | 01 |
   | `sap-rpt-1.6` | 01, 02, 04 |
   | `sap-rpt-1.6-large` | 01, 02 |
   | `sap-rpt-ts` | 03 |

   The clients find the deployment by model name; exactly one `RUNNING` deployment per model is expected in the resource group. No deployment ID or URL is configured.

2. Service key values in a `.env` file (never commit it). `find_dotenv` searches upwards, so `rest/.env` or the existing `RPT-1/python/.env` both work:

   ```dotenv
   AICORE_AUTH_URL=https://<subaccount>.authentication.<region>.hana.ondemand.com
   AICORE_CLIENT_ID=...
   AICORE_CLIENT_SECRET=...
   AICORE_BASE_URL=https://api.ai.<region>.ml.hana.ondemand.com
   AICORE_RESOURCE_GROUP=default
   ```

3. Python 3.10+.

## Install and run

```bash
cd best-practices/narrow-ai/RPT-1/python/rest
```

```bash
python -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt jupyter
```

```bash
jupyter notebook
```

Run the notebooks top to bottom. Each one is self-contained and committed with the outputs of a live run, so you can read the results without credentials.

## How to test

Offline checks (no credentials, no network):

```bash
python tests/test_rest_offline.py
```

Live end-to-end check (calls your AI Core deployments; costs inference):

```bash
jupyter nbconvert --to notebook --execute --inplace 01_RPT-1.6_predict.ipynb
```

Repeat for notebooks 02-04. A notebook fails on the first HTTP or RPT error.

---

## 1. Tabular prediction with SAP-RPT-1.5 / 1.6 (notebook 01)

**What:** fill `[PREDICT]` cells of query rows from labeled context rows sent in the same request (in-context learning, no training).

**Input:** `POST {deployment_url}/predict`, headers `Authorization: Bearer <token>`, `AI-Resource-Group`, JSON body:

```python
payload = {
    "index_column": "ID",
    "prediction_config": {
        "target_columns": [
            {"name": "COSTCENTER", "prediction_placeholder": "[PREDICT]", "task_type": "classification", "top_k": 2},
            {"name": "DISCOUNT_RATE", "prediction_placeholder": "[PREDICT]", "task_type": "regression"},
        ],
        "explanations": {"top_column_scores": 3, "top_relevant_context_rows": 2},
    },
    "data_schema": {"PRICE": {"dtype": "numeric"}, "ORDERDATE": {"dtype": "date"}},
    "rows": rows,  # context rows + query rows with "[PREDICT]"
}
response = RPTClientAPI(model_name="sap-rpt-1.6").predict_json(payload)
```

**Output:** `status.code` (0 ok, 1 warning, 2 invalid input, 3 server error), `metadata`, `predictions` (one record per query row; classification candidates with `confidence`, regression with `confidence_interval` `[lower, upper]`) and, if requested, `explanations` (one entry per query row). `flatten_predictions` and `map_explanation_rows` turn them into tables.

| Model | Max context rows | Recommended context | Max columns |
|---|---|---|---|
| `sap-rpt-1.5` | 2,048 | 500-2,000 | 100 |
| `sap-rpt-1.5-large` | 65,536 | 4,000-8,000 | 256 |
| `sap-rpt-1.6` | 2,048 | 500-2,000 | 100 |
| `sap-rpt-1.6-large` | 65,536 | 4,000-36,000 | 256 |

All variants: up to 512 query rows and 10 target columns per request. Multiple targets in one request are predicted jointly.

Measured on a holdout of the employee data (2,000 context rows, 500 hidden answers, `SALARY` regression and `FLIGHT_RISK` classification in one request): salary error about 8,200 (RPT-1.6) and 8,800 (RPT-1.5) against 18,200 for a median baseline; among the 50 employees ranked riskiest, 42% (RPT-1.6) and 40% (RPT-1.5) were real `Yes` cases against 9.6% at random. The large variants returned the same answers as the small ones at this context size. Numeric-looking IDs are returned as numbers, so use text IDs (`emp-29089`).

## 2. Deep context with SAP-RPT-1.6-large (notebook 02)

`prediction_config.context_mode` is `"default"` or `"deep"`. Deep mode exists only for `sap-rpt-1.6-large` and is recommended only above 8,000 context rows, after trying `sap-rpt-1.6` and `sap-rpt-1.6-large` in default mode.

Measured on the overnight-stays dataset (`OVERNIGHTSTAYS.csv`), predicting each of the last three months (1,001 region x country rows each) from all earlier months (14,000-16,000 context rows), log target with lag features:

| Setup | Log error, average of 3 months | Typical factor off | Seconds per month (2 requests) |
|---|---|---|---|
| Baseline (same month last year) | 0.403 | 1.50 | - |
| `sap-rpt-1.6`, 2,048 sampled rows | 0.364 | 1.44 | ~1.6 |
| `sap-rpt-1.6-large`, all rows, default | 0.341 | 1.41 | ~2.2 |
| `sap-rpt-1.6-large`, all rows, deep | 0.343 | 1.41 | ~4.7 |

All setups beat the baseline in every month; all rows on `sap-rpt-1.6-large` beat the 2,048-row sample in every month; default and deep were close, deep about twice as slow. On raw (heavy-tailed) counts RPT lost to the baseline in a trial; a `log1p` target and lag columns were needed.

## 3. Time-series forecasting with SAP-RPT-TS (notebook 03)

**What:** zero-shot probabilistic multi-step forecasts for one or many series, with optional past/future covariates.

**Input:** `POST {deployment_url}/predict` on the `sap-rpt-ts` deployment:

```python
from rpt_ts_client import RPTTSClientAPI, forecast_to_frame

client = RPTTSClientAPI()
response = client.forecast({
    "horizon": 3,
    "data": [{
        "timestamp": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
        "target": [120, 135, 128, 142],
        "past_covariates": {"temperature": [15, 18, 17, 21]},
        "future_covariates": {"temperature": [22, 24, 23]},
    }],
    "options": {"confidence_interval": 0.9, "cross_learning": True},
})
forecast_to_frame(response)
```

**Request fields per series:** `target` (history to forecast from, `null` for missing values; a list of lists for several targets on one timeline), `timestamp` (optional, same length, evenly spaced), `past_covariates` (optional, same length as the history), `future_covariates` (optional, length `horizon`, only for values known in advance, each also present in `past_covariates`).

**Output:** `result[series][target][step]` (forecast), `confidence_interval_output[series][target][0|1][step]` (80% range by default), `quantile_result` / `selected_quantile_result`, `result_timestamp`, `detected_interval`, `hierarchical_result`, `all_null_targets`, `usage` (input/output time points, the billing unit). There is no `status.code`; errors are HTTP 400 / 422 with a `detail` message. Limits: 8,192 history points per series, horizon 1,024.

Main `options`: `confidence_interval`, `quantile_levels`, `output_quantile` (21-level grid), `point_forecast` (`median` default, `mean`, `hierarchical_forecast`), `clip_min`/`clip_max`, `cross_learning`, `enhanced_forecast` (input points billed 4x), `rules`, `hierarchical_labels`, `hierarchical_levels`.

**Rules** (`options.rules`): list of `{"covariate", "operator", "threshold", "forecast_value"}`. For each forecast step where the numeric future covariate satisfies `value <operator> threshold` (`==`, `!=`, `>`, `>=`, `<`, `<=`), the forecast becomes `forecast_value` for all targets of the series. The covariate must be a numeric future covariate of every series in the request (else HTTP 400). Rules apply in list order, the last match wins, and ranges/quantiles are not adjusted.

Findings from the live runs: 13 regional series of hotel nights with history since 2005, rolling backtest over 24 cut-offs (forecasts starting July 2024 to June 2026), 312 comparisons per horizon:

| Variant | MAPE 1 month ahead | MAPE 3 months ahead | 80% range hit rate (1 / 3 months) |
|---|---|---|---|
| Seasonal naive (same month last year) | 0.047 | 0.046 | - |
| Full history | 0.046 | 0.050 | 0.84 / 0.83 |
| Full history + `last_year` covariate | 0.045 | 0.049 | 0.83 / 0.85 |
| Full history + `last_year`, `enhanced_forecast` | 0.044 | 0.048 | 0.85 / 0.86 |
| Last 24 months only | 0.058 | 0.052 | 0.89 / 0.94 |

- On these very regular regional totals the model is about as accurate as seasonal naive; long history matters (24 months only is clearly worse) and gives well-sized ranges. `cross_learning` made no difference here.
- On the noisier 104 region x guest-origin series the model with the `last_year` covariate beat seasonal naive one month ahead (WAPE 5.8% vs 6.3%) and was close three months ahead (6.4% vs 6.2%).
- With only 29 months of history (earlier version of the notebook) seasonal naive stayed clearly ahead, and a single 3-month holdout had made the model look better than it was: always backtest over several cut-offs.
- A backtest fits in a few requests when each (cut-off, series) pair is sent as its own series; do not combine this with `cross_learning`, which would let later cut-offs leak future values into earlier ones. The client retries HTTP 429 automatically.
- On 23 intermittent series (39% zero months), `median` forecasts summed to 45% of the real nights, `mean` to 99%; `median` had the lower monthly error. Use `mean` when totals matter.
- **Hierarchy labels must nest explicitly.** Each level is aggregated by its own label value. With labels `{"region": r, "country": c}` and levels `[null, "region", "country"]`, the nodes under each region are cross-region country totals. Use a label that is unique within its parent, e.g. `{"region": r, "region_country": f"{r} | {c}"}`. With that, total = sum of regions and region = sum of countries exactly. `hierarchy_to_frame` flattens the tree.
- Over 6 cut-offs, regional forecasts from the reconciled tree (104 region x origin leaves) were as accurate as direct regional forecasts (MAPE 5.4% vs 5.6%); with the short 2022-2024 data the reconciled version had been worse. Compare on your own data.

## 4. Decisions (notebook 04)

**What:** answer typed questions about a `state` (a text, a JSON object or a chat log) from labeled `examples`, with a probability for every possible answer:

| Question type | Example answer | Returned |
|---|---|---|
| `noul` | `true` / `false` | probability of true |
| `choice` | option name | winning option, distribution, confidence |
| `score` | level index as string | expected level, distribution, confidence |

**How:** RPT on AI Core is a table endpoint (`/predict` requires `prediction_config` and rejects other fields with HTTP 422), so `rpt_decisions.py` translates the decision into one RPT table and one joint `/predict` call:

- examples become context rows, states become query rows (`row_id` index);
- the state becomes a `state` column (text, JSON text for lists) or one column per key (objects);
- each question becomes a classification target with `top_k` equal to its number of options, so the full distribution is returned;
- the response is mapped back to one answer per question.

```python
from rpt_client_api import RPTClientAPI
from rpt_decisions import decide

result = decide(RPTClientAPI(model_name="sap-rpt-1.6"), request)              # one state
batch = decide(RPTClientAPI(model_name="sap-rpt-1.6"), request, states=texts)  # up to 512 states
result["answers"][0]["tone"]  # {"type": "choice", "choice": "angry", "probabilities": {...}, "confidence": 0.46}
```

`encoding="described"` (default) uses the question `instructions` as column names and the option descriptions as cell values (`"angry: Hostile, frustrated, or threatening"`); `encoding="plain"` writes the raw keys.

Measured on 400 held-out real customer messages (banking77, 10 intents, one `choice` and one derived `noul` question):

| Examples per intent | Intent accuracy (plain / described) | Card yes/no accuracy |
|---|---|---|
| 2 | 0.888 / 0.880 | 0.94 / 0.92 |
| 5 | 0.950 / 0.945 | 0.96 / 0.96 |
| 20 | 0.960 / 0.972 | 0.98 / 0.98 |
| 80 | 0.978 / 0.978 | 0.99 / 0.99 |

Accepting only answers with confidence of at least 0.8 handled 95% of the messages automatically at 99.7% accuracy. On the small hand-written triage example the described encoding fixed a wrong tone answer; on the banking messages both encodings were equal.

`validate_decision_request` enforces the limits before the call: 1-10 questions, 2-2,048 examples, 2-255 options, and every option must appear in at least one example. The request format follows the decision format of SAP's RPT Playground; the table translation is our own design.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No AI Core configuration matched 'sap-rpt-ts' version '1'` | No configuration with that `modelName`/`modelVersion` in `AICORE_RESOURCE_GROUP`. Check the service key points to the instance and resource group that hold the deployment. |
| `Found multiple running deployments` | More than one `RUNNING` deployment for the model; stop the duplicate. |
| `401` / `403` | Wrong credentials or resource group. |
| HTTP 422 on `/predict` | Schema error; the message lists the failing fields (`loc`). Unknown fields are rejected. |
| RPT `status.code` 2 | Invalid input (e.g. `context_mode` on RPT-1.5, explanation counts above 20). |
| RPT-TS HTTP 400 | Business rule, e.g. a future covariate whose length differs from `horizon`, or too many missing timestamps (more than 20%). |
| `Invalid value '[PREDICT]' for dtype 'float64'` (pandas) | Cast target columns to `object` before writing the placeholder. |
| Weak predictions | Add representative context, fix label consistency, transform heavy-tailed targets, add lag columns for temporal data. |

## References

- SAP Help: [SAP-RPT-1.5](https://help.sap.com/docs/sap-ai-core/generative-ai/sap-rpt-1-5), [SAP-RPT-1.6](https://help.sap.com/docs/sap-ai-core/generative-ai/sap-rpt-1-6), [SAP-RPT-TS](https://help.sap.com/docs/sap-ai-core/generative-ai/sap-rpt-ts), [SAP-RPT-TS inference request](https://help.sap.com/docs/sap-ai-core/generative-ai/inference-request-overview)
- Datasets:
  - Swiss Federal Statistical Office (FSO), HESTA, [table px-x-1003020000_103](https://www.pxweb.bfs.admin.ch/pxweb/en/px-x-1003020000_103/): `hotel_nights_2005_2026.csv` (downloaded via the PxWeb API; 7 named guest origins, the rest summed as "Other countries") and `OVERNIGHTSTAYS.csv`. [FSO terms of use](https://www.bfs.admin.ch/bfs/en/home/fso/swiss-federal-statistical-office/terms-of-use.html): open use, source must be cited.
  - [banking77](https://github.com/PolyAI-LDN/task-specific-datasets) by PolyAI, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): `banking77_10_intents.csv` keeps 10 of the 77 intents, original train/test split in column `split`, columns renamed to `message` and `intent`.
  - Employee data: [SAP HANA Cloud Machine Learning Challenge](https://community.sap.com/t5/technology-blogs-by-sap/quot-i-quit-quot-how-to-predict-employee-churn-sap-hana-cloud-machine/ba-p/13560188), as used in this repository's classification best practice.
