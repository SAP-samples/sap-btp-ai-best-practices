# SAP RPT-1 Best Practices

This area showcases three ways to work with SAP-RPT table-completion models on SAP AI Core.

SAP-RPT is a relational pretrained transformer for structured business data. You send rows of example data as context, mark one or more target cells with a placeholder, and the model predicts the values of those cells through in-context learning, without task-specific training. It supports classification and regression tasks.

- [python](./python): Python SDK example using `RPTClient` from the SAP Cloud SDK for AI for Python (`sap-ai-sdk-gen`), covering classification, regression, mixed tasks, `top_k`, async calls and a pandas DataFrame workflow.
- [Java](./Java): Java SDK example for inference-only RPT-1 predictions against an existing deployment.
- [typescript](./typescript): TypeScript SDK example (`@sap-ai-sdk/rpt`) for inference-only RPT-1 predictions against an existing deployment.

## What Each Language Track Covers

The Java and TypeScript tracks intentionally mirror the same teaching flow:

1. Schema-free classification example predicting `COSTCENTER`
2. Structured prediction example with explicit column typing predicting `DISCOUNT_RATE`

The TypeScript sample expects an existing deployment ID through `RPT1_DEPLOYMENT_ID`.

The current Java SDK resolves the deployed model through `RptModel` and AI Core configuration, so the Java sample uses `RPT1_MODEL_NAME` instead of an explicit deployment ID.

The Python track is a single tutorial notebook that goes further: it also covers mixed classification and regression in one call, ranked predictions with `top_k`, async usage and response interpretation.

## When To Use Which Track

- Choose [python](./python) if you work in notebooks or data pipelines, or want the broadest coverage of request options.
- Choose [Java](./Java) or [typescript](./typescript) if you are building an application backend in those languages.

All three tracks use the official SAP Cloud SDK for AI.

## Prerequisites

- An SAP AI Core instance with the **`extended`** service plan. See [Prerequisites](../../generative-ai/Prerequisites.md).
- A running SAP-RPT deployment in the generative AI hub. The samples use the `sap-rpt-1-small` model (lower latency and cost, suited to prototyping and high-volume workloads); `sap-rpt-1-large` targets higher accuracy. See the [SAP developer tutorial on using SAP-RPT in the generative AI hub](https://developers.sap.com/tutorials/ai-core-genaihub-sap-rpt-model-usage.html) for deployment steps.

## Model versions and newer request options

SAP has released newer versions of SAP-RPT since these samples were first written (SAP-RPT-1.5, and SAP-RPT-1.6, announced in the [Q3 2026 AI release highlights](https://news.sap.com/2026/10/ai-release-highlights-q3-2026/)). Check the generative AI hub model library for the model names and versions available in your region.

Recent SDK releases add request and response options for these versions. They are optional and not yet used in the samples:

| Option | Purpose |
|---|---|
| Explanations | Feature-importance explanations for predictions |
| Confidence intervals | Uncertainty bounds for regression predictions |
| `context_mode` | Controls how the context data is used; SAP-RPT-1.6 adds a Deep Context Mode for the large variant that uses more reference data at the cost of higher latency |

Minimum SDK versions:

- TypeScript: `@sap-ai-sdk/rpt` 2.16.0 adds explanations, confidence intervals and `context_mode`. That release also contains breaking changes: there is no longer a default model name, and `parse_data_types` defaults to `false`. Review the [release notes](https://sap.github.io/ai-sdk/docs/js/release-notes) before upgrading the TypeScript sample from the pinned `2.8.0`.
- Python: RPT-1.6 support was added to the RPT client in [SAP/ai-sdk-python#174](https://github.com/SAP/ai-sdk-python/pull/174). Use a release of `sap-ai-sdk-gen` that includes it.

### Tabular orchestration

For production scenarios with a fixed data source, SAP also offers **Tabular Orchestration**, a managed workflow in which you configure the data source once and each prediction is a single API call that handles data ingestion and context selection. In the JavaScript SDK this is available as an experimental `TabularOrchestrationClient` from version 2.17.0.
