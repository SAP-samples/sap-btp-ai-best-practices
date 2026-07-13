# SAP RPT-1 Best Practices

This area showcases three ways to work with SAP's RPT-1 table-completion models on SAP AI Core.

- [python](./python): raw HTTP requests and a custom example client for RPT-1 inference. An official Python SDK is planned; these examples serve as interim guidance.
- [Java](./Java): Java SDK example for inference-only RPT-1 predictions against an existing deployment.
- [typescript](./typescript): TypeScript SDK example for inference-only RPT-1 predictions against an existing deployment.

## What Each Language Track Covers

The Java and TypeScript tracks intentionally mirror the same teaching flow:

1. Schema-free classification example predicting `COSTCENTER`
2. Structured prediction example with explicit column typing predicting `DISCOUNT_RATE`

The TypeScript sample expects an existing deployment ID through `RPT1_DEPLOYMENT_ID`.

The current Java SDK resolves the deployed model through `RptModel` and AI Core configuration, so the Java sample uses `RPT1_MODEL_NAME` instead of an explicit deployment ID.

## When To Use Which Track

- Choose [python](./python) for direct API access and as a reference for building custom wrappers. An official SAP AI SDK for Python is planned but not yet publicly available.
- Choose [Java](./Java) or [typescript](./typescript) if you want to use the official SAP AI SDK abstractions for RPT-1 inference.
