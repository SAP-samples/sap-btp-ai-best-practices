---
name: rpt-ts
description: Use when building or troubleshooting SAP-RPT-TS (`sap-rpt-ts`) zero-shot time-series forecasting through SAP AI Core, including multi-series requests, past and future covariates, quantiles and confidence ranges, deterministic forecast rules, point_forecast choice, hierarchical reconciliation, and rolling backtests.
---

# SAP-RPT-TS Direct API

SAP-RPT-TS is SAP's pretrained time-series foundation model on AI Core. A request carries the history of one or more series; the response returns a probabilistic multi-step forecast. There is no training step and the SAP AI SDK has no client for it, so call the deployment over REST.

For forecasting models trained inside SAP HANA (PAL additive models through hana-ml), use `narrow-ai-time-series-forecasting` instead.

## Reusable Client

Copy two files into the target project, side by side:

- [assets/rpt_ts_client.py](assets/rpt_ts_client.py): `RPTTSClientAPI.forecast`, `forecast_to_frame`, `hierarchy_to_frame`.
- [../rpt-1/assets/rpt_client_api.py](../rpt-1/assets/rpt_client_api.py): shared OAuth, deployment discovery and `RPTAPIError`. Do not modify it; it is the same file the `rpt-1` skill ships.

```bash
python -m pip install requests requests-toolbelt python-dotenv pandas
```

```python
from rpt_ts_client import RPTTSClientAPI, forecast_to_frame

client = RPTTSClientAPI()  # finds the one RUNNING sap-rpt-ts deployment
response = client.forecast({"horizon": 3, "data": [{"target": [120, 135, 128, 142]}]})
forecast_to_frame(response)
```

Read [docs/rpt-ts.md](docs/rpt-ts.md) before building a request: it holds the request/response contract, options, rule semantics, the hierarchy label trap, the backtest recipe and measured results.

## Key Rules

- `future_covariates` have length `horizon`, hold only values known in advance, and must also appear in `past_covariates`.
- Send long history (up to 8,192 points per series); 24 months gave clearly worse forecasts and wider ranges than 20 years.
- Judge a setup with a rolling backtest over many cut-offs against seasonal naive, never one holdout.
- One request can carry each (cut-off, series) pair as its own series, but only with `cross_learning` off.
- `point_forecast="mean"` when totals matter on intermittent series; `median` (default) for month-by-month closeness.
- Hierarchy labels must be unique within their parent (`"Region | Child"`), or child nodes aggregate across parents.
- `enhanced_forecast` bills input points at 4x; keep it only if the backtest shows a gain.

## Validation

```bash
python tests/test_rpt_ts_client.py
```

Offline only: response flattening, hierarchy flattening and credential redaction on HTTP errors. A live check costs inference; run one small `forecast` call against the deployment.
