# SAP-RPT-TS Direct AI Core Client

## Purpose

`RPTTSClientAPI` calls a running `sap-rpt-ts` deployment on SAP AI Core. It inherits environment-based OAuth and deployment discovery from `RPTClientAPI` (the `rpt-1` skill) and only replaces the inference call, because the time-series contract differs from the tabular RPT models:

| | Tabular RPT (`rpt-1`) | SAP-RPT-TS |
| --- | --- | --- |
| Request | `rows` + `prediction_config` | `horizon` + `data` (series list) + `options` |
| Response | `status.code` envelope + `predictions` | forecast arrays + `usage`, no `status.code` |
| Errors | HTTP error or application code 2/3 | HTTP 400 (business rule) or 422 (schema) with `detail` |

## Setup

Copy `assets/rpt_ts_client.py` and `../rpt-1/assets/rpt_client_api.py` into one folder of the target project. Credentials come from the environment or a `.env` file excluded from version control:

```dotenv
AICORE_AUTH_URL=https://<subaccount>.authentication.<region>.hana.ondemand.com
AICORE_CLIENT_ID=...
AICORE_CLIENT_SECRET=...
AICORE_BASE_URL=https://api.ai.<region>.ml.hana.ondemand.com
AICORE_RESOURCE_GROUP=default
```

Deployment: scenario `foundation-models`, executable `aicore-sap`, model `sap-rpt-ts`, version `1`. Exactly one `RUNNING` deployment per model is expected in the resource group; no deployment ID or URL is configured.

## Request Contract

`POST {deployment_url}/predict`:

```python
request = {
    "horizon": 3,
    "data": [  # one entry per series
        {
            "timestamp": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01"],
            "target": [120, 135, 128, 142, 150, 149],
            "past_covariates": {
                "temperature": [15, 18, 17, 21, 24, 26],
                "promotion": [0, 1, 0, 0, 1, 0],
                "web_visits": [900, 1100, 950, 990, 1200, 1150],  # past only: unknown in the future
            },
            "future_covariates": {
                "temperature": [27, 25, 22],
                "promotion": [1, 0, 0],
            },
        }
    ],
    "options": {"confidence_interval": 0.8},
}
response = RPTTSClientAPI().forecast(request)  # compress=True gzips large bodies
```

| Field | Meaning | Length | Required |
| --- | --- | --- | --- |
| `horizon` (top level) | Number of steps to forecast, up to 1,024 | - | yes |
| `target` | History to forecast from; `null` marks a missing value. A list of lists forecasts several targets that share one timeline and covariates | history steps | yes |
| `timestamp` | Evenly spaced times of the history. At most 20% may be gap-filled. Formats: integers, `YYYY-MM-DD`, `YYYY-MM-DD HH:MM:SS`, `YYYY-MM-DD HH:MM:SS.fffffff`, `HH:MM:SS`; read as written, no time-zone conversion | same as `target` | no (no dates in output without it) |
| `past_covariates` | Variables measured over the history; numbers or text | same as `target` | no |
| `future_covariates` | Values for the forecast steps, only for variables known in advance (planned promotions, holidays, weather forecasts); each must also be in `past_covariates` | `horizon` | no |

Several series in one request may have different lengths and are forecast independently unless `options.cross_learning` is `true`. Keep targets plus covariates below 128 per series. History beyond 8,192 points per series is cut off.

## Response Contract

Arrays are nested series, then target, then step:

| Field | Shape | Content |
| --- | --- | --- |
| `result` | `[series][target][step]` | Point forecast (see `point_forecast`) |
| `confidence_interval_output` | `[series][target][0=lower, 1=upper][step]` | Range, 80% by default |
| `quantile_result` | `[series][target][21][step]` | Only with `output_quantile: true` |
| `selected_quantile_result` | `[series][target][quantile][step]` | Only with `quantile_levels` |
| `result_timestamp` | `[series][step]` | Dates of the forecast steps |
| `detected_interval` | `[series]` | Spacing read from the timestamps |
| `hierarchical_result` | tree | Only with `point_forecast: "hierarchical_forecast"` |
| `all_null_targets` | list | Series or targets whose history was entirely `null` |
| `usage` | object | Input and output time points: the billing unit |

```python
from rpt_ts_client import forecast_to_frame

frame = forecast_to_frame(response, series_ids=["store-1"])
# columns: series, variate, step, timestamp, point, lower, upper
```

## Errors and Retries

The service answers HTTP 400 (business rule, for example a future covariate whose length differs from `horizon`, or more than 20% missing timestamps) or HTTP 422 (schema error; unknown fields are rejected) with a `detail` message. `forecast` raises `RPTAPIError` with `http_status`, the `detail` text and credentials redacted. HTTP 429 is retried up to 5 times, waiting `Retry-After` seconds when sent, else 10 s, 20 s, ...

## Options

| Option | Effect |
| --- | --- |
| `confidence_interval` | Width of the range: `0.8` default (0.1-0.9 quantiles), `0.9` (0.05-0.95), `null` switches it off |
| `quantile_levels` | Return exactly these quantiles in `selected_quantile_result` |
| `output_quantile` | Return the fixed 21-level grid (0.01, 0.05, 0.10, ..., 0.95, 0.99) in `quantile_result` |
| `point_forecast` | `"median"` (default), `"mean"`, or `"hierarchical_forecast"` |
| `clip_min` / `clip_max` | Bound the forecast, e.g. `clip_min: 0` for counts |
| `cross_learning` | Learn across all series of the request |
| `enhanced_forecast` | Extra inference passes; input time points billed 4x |
| `rules` | Deterministic overrides (below) |
| `hierarchical_labels`, `hierarchical_levels` | Hierarchy definition (below) |

## Deterministic Rules

`options.rules` sets the forecast to a fixed value whenever a future covariate meets a condition, e.g. zero sales on closed days:

```python
"options": {"rules": [
    {"covariate": "closed", "operator": "==", "threshold": 1, "forecast_value": 0},
    {"covariate": "promotion", "operator": ">=", "threshold": 2, "forecast_value": 300},
]}
```

Observed behavior in live requests:

1. For each step, the step's covariate value is tested as `value <operator> threshold`; operators `==`, `!=`, `>`, `>=`, `<`, `<=`.
2. A match sets that step's forecast to `forecast_value` for every target of the series.
3. Rules apply in list order; when several match a step, the last one wins.
4. Only `result` changes. Ranges and quantiles are not adjusted, so a rule value can lie outside the reported range.
5. The covariate must be a numeric future covariate of every series in the request, otherwise HTTP 400.

## Ranges, Quantiles and the Point Forecast

A quantile q is a value the actual should fall below in a share q of cases; the 80% range runs from the 0.1 to the 0.9 quantile. Typical uses: capacity at the 0.9 quantile, conservative budgets at 0.1, the median as the expected value.

Pick `point_forecast` by the decision the number feeds:

| Value | Choose when |
| --- | --- |
| `median` | Each step should be close to reality; smooth, rarely zero |
| `mean` | Totals matter (stock, budget) and the series is lopsided, e.g. many zero months. Computed from the quantiles, so pass a dense `quantile_levels` list (e.g. every 0.05) |
| `hierarchical_forecast` | Several levels must add up |

Measured on 23 intermittent series (39% zero months) over 11 cut-offs: `median` forecasts summed to 45% of the real volume, `mean` to 99%; `median` had the lower monthly error.

## Hierarchical Forecasting

Send only the lowest level; the service forecasts every level and reconciles them (MinT-WLS) so that each parent equals the sum of its children at every step.

```python
from rpt_ts_client import hierarchy_to_frame

request = {
    "horizon": 3,
    "data": [leaf_series(region, child) for region, child in leaves],
    "options": {
        "point_forecast": "hierarchical_forecast",
        "hierarchical_labels": [
            {"region": region, "region_child": f"{region} | {child}"} for region, child in leaves
        ],
        "hierarchical_levels": [None, "region", "region_child"],  # None first adds a grand total
    },
}
tree = hierarchy_to_frame(client.forecast(request, compress=True))
# columns: level (0 = total), path ("total", "North", "North / North | A"), variate, step, point
```

`result` is `[]` and no ranges are returned for hierarchical forecasts.

**Labels must be unique within their parent.** Each level is grouped by its own label value only. With labels `{"region": r, "child": c}`, the nodes under each region become totals of that child across all regions and do not add up to the region. Combine the parent into the child label as above. With that, total equals the sum of regions and each region equals the sum of its children exactly; verify this after every change.

Reconciliation can change accuracy in either direction: on 20 years of regional data the reconciled regional numbers matched direct regional forecasts (MAPE 5.4% vs 5.6%); on a short 29-month history they had been worse. Compare both on your own data.

## Rolling Backtest Recipe

One holdout can make any method look better or worse than it is. Forecast from many cut-offs and compare against a simple baseline (seasonal naive: same period last year):

```python
def make_series(values, timestamps, cutoff, horizon, with_last_year=False):
    """History of one series up to `cutoff` (exclusive), optionally with a 12-step lag covariate."""
    series = {"timestamp": timestamps[:cutoff], "target": values[:cutoff]}
    if with_last_year:
        lagged = [None] * 12 + values[:-12]  # unknown for the first year: send null
        series["past_covariates"] = {"last_year": lagged[:cutoff]}
        series["future_covariates"] = {"last_year": lagged[cutoff:cutoff + horizon]}
    return series

keys = [(cutoff, name) for cutoff in cutoffs for name in series_by_name]
response = client.forecast({
    "horizon": 3,
    "data": [make_series(series_by_name[name], timestamps, cutoff, 3) for cutoff, name in keys],
    "options": {"clip_min": 0, "confidence_interval": 0.8},  # cross_learning stays off
}, compress=True)
frame = forecast_to_frame(response)
frame["cutoff"] = [keys[s][0] for s in frame["series"]]
frame["name"] = [keys[s][1] for s in frame["series"]]
```

Sending each (cut-off, series) pair as its own series fits a whole backtest into a few requests (chunk to ~100 series per request). Never combine this with `cross_learning`: later cut-offs would leak future values into earlier ones. Test `cross_learning` with one request per cut-off instead.

Metrics: MAPE (mean absolute percentage error), WAPE (sum of absolute errors / sum of actuals; prefer it when some series are small) and the 80% range hit rate (share of actuals inside the range; near 0.80 means well-sized ranges).

## Measured Results

13 regional series of Swiss hotel nights (history since 2005), 24 cut-offs, 312 comparisons per horizon:

| Variant | MAPE 1 step ahead | MAPE 3 steps ahead | 80% range hit rate (1 / 3) |
| --- | ---: | ---: | --- |
| Seasonal naive | 0.047 | 0.046 | - |
| Full history | 0.046 | 0.050 | 0.84 / 0.83 |
| Full history + `last_year` covariate | 0.045 | 0.049 | 0.83 / 0.85 |
| + `enhanced_forecast` | 0.044 | 0.048 | 0.85 / 0.86 |
| Last 24 months only | 0.058 | 0.052 | 0.89 / 0.94 |

- Very regular series: about as accurate as seasonal naive. Long history mattered and gave well-sized ranges. `cross_learning` made no difference.
- Noisier 104 region x origin series: with `last_year` the model beat seasonal naive one step ahead (WAPE 5.8% vs 6.3%) and was close three steps ahead (6.4% vs 6.2%).
- With only 29 months of history seasonal naive stayed clearly ahead, and a single 3-month holdout had overstated the model.

These numbers describe one dataset; backtest on the customer's data before relying on the forecasts or ranges.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `No AI Core configuration matched 'sap-rpt-ts' version '1'` | Service key points to another instance or resource group |
| `Found multiple running deployments` | Stop the duplicate deployment |
| `401` / `403` | Wrong credentials or resource group |
| HTTP 422 | Schema error; `detail` names the failing field; unknown fields are rejected |
| HTTP 400 | Business rule: future covariate length differs from `horizon`, rule covariate missing or non-numeric in a series, more than 20% missing timestamps |
| Hierarchy children do not add up to the parent | Child labels not unique within their parent |
| Rule value outside the range | Expected: rules do not adjust ranges |
| Forecast totals far below actuals on sparse series | Use `point_forecast: "mean"` with dense `quantile_levels` |

## Validation

```bash
python tests/test_rpt_ts_client.py
python -m py_compile assets/rpt_ts_client.py
```

The offline tests use a fake HTTP session; they check flattening, hierarchy paths, the success path, and that bearer tokens and client secrets never appear in raised errors.
