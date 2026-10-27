# SAP RPT-1 Best Practices

This area showcases three ways to work with SAP's RPT-1 table-completion models on SAP AI Core.

- [python](./python): native SAP AI SDK notebook for RPT-1 inference.
- [python/rest](./python/rest): direct REST notebooks for the current models on SAP AI Core: SAP-RPT-1.5/1.6 (explanations, confidence intervals, deep context), SAP-RPT-TS time-series forecasting, and Playground-style decisions.
- [Java](./Java): Java SDK example for inference-only RPT-1 predictions against an existing deployment.
- [typescript](./typescript): TypeScript SDK example for inference-only RPT-1 predictions against an existing deployment.

## What Each Language Track Covers

The Java and TypeScript tracks intentionally mirror the same teaching flow:

1. Schema-free classification example predicting `COSTCENTER`
2. Structured prediction example with explicit column typing predicting `DISCOUNT_RATE`

The TypeScript sample expects an existing deployment ID through `RPT1_DEPLOYMENT_ID`.

The current Java SDK resolves the deployed model through `RptModel` and AI Core configuration, so the Java sample uses `RPT1_MODEL_NAME` instead of an explicit deployment ID.

## When To Use Which Track

- Choose [python](./python) for the native SAP AI SDK abstractions in Python.
- Choose [python/rest](./python/rest) for features the SDKs do not model yet: SAP-RPT-1.5/1.6 explanations and `context_mode`, the `sap-rpt-ts` time-series model, and decisions.
- Choose [Java](./Java) or [typescript](./typescript) if you want to use the official SAP AI SDK abstractions for RPT-1 inference.
