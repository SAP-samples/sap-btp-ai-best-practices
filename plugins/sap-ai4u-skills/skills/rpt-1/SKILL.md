---
name: rpt-1
description: Use when building or troubleshooting SAP RPT-1.5 or RPT-1.6 tabular classification, regression, multi-target, explainability, confidence interval, JSON, gzip, Parquet, or deep-context inference through SAP AI Core.
---

# SAP RPT Direct API

Use direct AI Core REST when the native SAP AI SDK omits RPT-1.5/1.6 request or response fields. Keep payloads and responses as dictionaries so `top_k`, `confidence_interval`, `explanations`, and `context_mode` survive unchanged.

## Reusable Client

Copy [assets/rpt_client_api.py](assets/rpt_client_api.py) into the target project. It provides:

- `RPTClientAPI` with environment-based OAuth and deployment resolution.
- Sync and async JSON and Parquet methods.
- Gzip JSON, joint multi-target requests, and RPT application-status handling.
- `flatten_predictions` and `map_explanation_rows` helpers.

Read [docs/rpt-client-api.md](docs/rpt-client-api.md) before implementing a new flow or diagnosing a request. It contains the supported contract, examples, limits, platform boundaries, and Parquet placeholder rule.

## Model Choice

| Need | Model |
| --- | --- |
| Default RPT-1.6 inference | `sap-rpt-1.6` |
| Large context or high cardinality | `sap-rpt-1.6-large` |
| Compatibility validation | `sap-rpt-1.5` or `sap-rpt-1.5-large` |
| More than 8,000 context rows | `sap-rpt-1.6-large` with `context_mode="deep"` |

Batch query rows and retain an `index_column`. Multi-target predictions in one request are joint predictions and can differ from separate calls.

## Validation

Run the offline contract tests first:

```bash
python tests/test_rpt_client_api.py
```

When live AI Core credentials and the validation datasets are available, run:

```bash
python scripts/live_test_rpt_client_api.py --dataset-root /path/to/datasets
```

The live command fails on contract errors and reports quality metrics without enforcing probabilistic thresholds. It does not deploy anything.

## Product Boundary

This skill's reusable client targets AI Core `/predict` and `/predict_parquet`. The Playground exposes a separate `/api/predict` evaluation contract. Chat-RPT and what-if interactions are web-UI features unless SAP publishes a supported API; Tabular Orchestration is a separate surface.
