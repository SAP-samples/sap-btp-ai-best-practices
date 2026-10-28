# SAP RPT on SAP AI Core with direct REST calls (Python)

SAP RPT is a family of SAP models that make predictions from business tables. You show the model rows where the answer is already known, and it fills in the answer for new rows. Nothing is trained beforehand: the examples are sent with each request, and the model, pretrained by SAP on many tables, is used as it is.

The four notebooks in this folder show four uses:

| Notebook | What it does | Example in the notebook | Model |
|---|---|---|---|
| 01 | Fill in a missing value in a table: a category or a number | the cost center and discount of an order; an employee's salary and flight risk | SAP-RPT-1.5, SAP-RPT-1.6 |
| 02 | The same with very large example tables (more than 8,000 rows) | hotel nights per region and guest country | SAP-RPT-1.6-large, `context_mode="deep"` |
| 03 | Forecast the next steps of a time series, with a range of likely values | 20 years of monthly hotel nights in 13 Swiss regions | SAP-RPT-TS |
| 04 | Answer fixed questions about a text, with a probability for each answer | "Is this message about billing?", "Which of 10 needs does this bank customer have?" | SAP-RPT-1.6 |

Every notebook is committed with the results of a live run, so you can read it without an SAP AI Core account.

## Words used on this page

| Word | Meaning |
|---|---|
| context rows, examples | rows where the answer is known; they are sent with the request so the model can learn from them |
| query rows | rows where the answer is missing, marked `[PREDICT]` |
| classification, regression | predicting a category (`Yes`/`No`, a cost center) or a number (a salary, a discount) |
| holdout | rows whose answers we know but hide from the model, so its answers can be checked |
| baseline | a simple rule the model must beat to be useful, for example "the same as last year" |
| confidence, confidence interval, range | how sure the model is: a probability for a category, a likely low-to-high range for a number |
| time series, horizon | values measured at regular steps (every month); the horizon is how many future steps to forecast |
| covariate | another value that helps explain the one you forecast, for example temperature or a planned promotion |
| backtest | pretend to be in the past, forecast, and compare with what really happened |
| MAPE, WAPE | average forecast error as a percentage of the actual value; WAPE gives large series more weight |
| quantile | a value with a probability attached: the actual result should fall below the 0.9 quantile in about 9 of 10 cases |

## Files

```
rest/
  01_RPT-1.6_predict.ipynb         RPT-1.5/1.6: raw call anatomy, joint classification + regression, holdout comparison,
                                   top_k, confidence intervals, explanations, model comparison
  02_RPT-1.6_deep_context.ipynb    sap-rpt-1.6 vs sap-rpt-1.6-large default vs deep, 3 test months, ~15,000 context rows
  03_RPT-TS_forecast.ipynb         SAP-RPT-TS: request fields, rules, rolling backtest, quantiles, hierarchy
  04_RPT_decisions.ipynb           decisions (yes/no, choice, ordered scale) as RPT tables, accuracy on banking77
  rpt_client_api.py                support: OAuth, deployment discovery, /predict for tabular RPT
  rpt_ts_client.py                 support: RPTTSClientAPI, forecast_to_frame, hierarchy_to_frame
  rpt_decisions.py                 support: decision validation, table translation, answer parsing
  tests/test_rest_offline.py       offline checks of the support modules
  data/hotel_nights_2005_2026.csv  Swiss hotel nights per region and guest origin, 2005-2026 (notebook 03)
  data/OVERNIGHTSTAYS.csv          Swiss hotel nights per region and country, 2022-2024 (notebooks 02, 03)
  data/employee_sample.csv         2,500-row sample of the SAP HANA ML Challenge employee data (notebook 01)
  data/banking77_10_intents.csv    banking77 customer messages, 10 intents (notebook 04)
  ../sample_orders.csv             small order table shared with the other RPT examples (notebook 01)
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

**What:** fill the `[PREDICT]` cells of query rows. The model learns from the context rows sent in the same request; there is no training step.

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

Results on the employee data: 2,000 employees with known answers were sent as examples, and the model predicted `SALARY` (a number) and `FLIGHT_RISK` (`Yes`/`No`) for 500 other employees whose answers were hidden, both in one request.

- Salary: the average error was about 8,200 with RPT-1.6 and 8,800 with RPT-1.5. Guessing the median salary for everyone gave 18,200.
- Flight risk: among the 50 employees the model ranked most likely to leave, 42% (RPT-1.6) and 40% (RPT-1.5) really were `Yes` cases. A random pick of 50 gave 9.6%.
- The large variants gave the same answers as the small ones at this size; they are built for much more context (section 2).

The service reads number-like IDs as numbers, so an ID such as `29089` comes back as a number and no longer matches your key. Use text IDs such as `emp-29089`.

## 2. Deep context with SAP-RPT-1.6-large (notebook 02)

`prediction_config.context_mode` is `"default"` or `"deep"`. Deep mode spends more time per request to make better use of a large context. It exists only for `sap-rpt-1.6-large` and is recommended only above 8,000 context rows, after trying `sap-rpt-1.6` and `sap-rpt-1.6-large` in default mode.

Test on `OVERNIGHTSTAYS.csv`: predict the hotel nights of each of the last three months in the data (1,001 region and guest-country combinations per month), using all earlier months as examples (14,000 to 16,000 rows). The baseline is "the same as twelve months ago". Lower error is better; "typical factor off" 1.41 means the estimate is typically 1.41 times too high or too low:

| Setup | Log error, average of 3 months | Typical factor off | Seconds per month (2 requests) |
|---|---|---|---|
| Baseline (same month last year) | 0.403 | 1.50 | - |
| `sap-rpt-1.6`, 2,048 sampled rows | 0.364 | 1.44 | ~1.6 |
| `sap-rpt-1.6-large`, all rows, default | 0.341 | 1.41 | ~2.2 |
| `sap-rpt-1.6-large`, all rows, deep | 0.343 | 1.41 | ~4.7 |

- All three RPT setups beat the baseline in every test month.
- Sending all rows to `sap-rpt-1.6-large` beat the 2,048-row sample in every month.
- Default and deep mode were close; deep took about twice as long.
- Two preparation steps were needed for these results. The target is predicted as `log(1 + nights)`, so that small country groups and large regions count equally. Last month's and last year's values are copied into each row as extra columns ("lag columns"), because RPT reads each row on its own and does not see the timeline otherwise.

## 3. Time-series forecasting with SAP-RPT-TS (notebook 03)

**What:** forecast several steps ahead for one or many series, with a range of likely values, without training on your data. Other values that help explain the series (covariates) can be added, both for the past and, when known in advance, for the forecast steps.

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

Results on 13 regional series of monthly hotel nights, with history since 2005. The backtest was repeated for 24 starting points (forecasts starting July 2024 to June 2026), which gives 312 comparisons each for 1 and 3 months ahead. MAPE is the average error as a share of the actual value (0.045 = 4.5%); the 80% range should contain the actual value about 80% of the time:

| Variant | MAPE 1 month ahead | MAPE 3 months ahead | 80% range hit rate (1 / 3 months) |
|---|---|---|---|
| Full history | 0.046 | 0.050 | 0.84 / 0.83 |
| Full history + `last_year` covariate | 0.045 | 0.049 | 0.83 / 0.85 |
| Full history + `last_year`, `enhanced_forecast` | 0.044 | 0.048 | 0.85 / 0.86 |

- Forecasts one month ahead were off by 4.4-4.6% on average, and the 80% range held the actual value in 83-86% of cases. `cross_learning` made no difference here.
- Send all the history you have (up to 8,192 steps per series); every variant above uses the full 20 years.
- On the noisier region by guest-origin series (104 series, for example German guests in Ticino), adding the `last_year` covariate cut the error one month ahead from 6.4% to 5.8% (WAPE).
- Always backtest over several starting points: one month alone can make any setup look better or worse than it is.
- A whole backtest fits in a few requests when each (starting point, series) pair is sent as its own series. Do not combine this with `cross_learning`: it lets the model learn across all series of a request, so later starting points would leak future values into earlier ones. The client retries HTTP 429 (too many requests) automatically.
- On 23 series with many empty months (39% of the months had zero nights), `median` forecasts added up to 45% of the real nights and `mean` forecasts to 99%; `median` was closer month by month. Use `mean` when totals matter.
- In a hierarchy, labels must nest explicitly, because each level is aggregated by its own label value. With labels `{"region": r, "country": c}` and levels `[null, "region", "country"]`, the nodes under each region are cross-region country totals. Use a label that is unique within its parent, e.g. `{"region": r, "region_country": f"{r} | {c}"}`. With that, total = sum of regions and region = sum of countries exactly. `hierarchy_to_frame` flattens the tree.
- Over 6 starting points, regional forecasts taken from the hierarchy (built from the 104 region by origin series and adjusted to add up) were as accurate as forecasting the 13 regions directly (MAPE 5.4% vs 5.6%). Compare on your own data.

## 4. Decisions (notebook 04)

**What:** answer typed questions about a `state` (a text, a JSON object or a chat log) from labeled `examples`, with a probability for every possible answer:

| Question type | Example answer | Returned |
|---|---|---|
| `noul` (yes/no) | `true` / `false` | probability of true |
| `choice` | option name | winning option, distribution, confidence |
| `score` | level index as string | expected level, distribution, confidence |

`noul` is the yes/no type, named as in the decision format of SAP's RPT Playground.

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

Results on 400 real customer messages to a bank whose answers were hidden (banking77 dataset, 10 intents, i.e. 10 kinds of customer need). Two questions per message: which need it is (`choice`), and whether it is about the customer's card (yes/no, derived from the intent):

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
