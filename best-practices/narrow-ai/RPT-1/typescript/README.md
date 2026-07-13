# SAP RPT-1 SDK Demo (TypeScript)

This sample demonstrates how to call an existing **SAP RPT-1** deployment from TypeScript using the [`@sap-ai-sdk/rpt`](https://www.npmjs.com/package/@sap-ai-sdk/rpt) package.

RPT-1 is a foundation model hosted on SAP AI Core for tabular tasks. You provide tabular rows with one or more cells marked as prediction targets, and the model returns predicted values for those cells based on the patterns in the surrounding data. This sample focuses on **inference only** — it assumes you already have a deployed RPT-1 model and a valid deployment ID.

## What This Sample Covers

The project includes two runnable prediction examples that illustrate the two main ways to interact with RPT-1:

### 1. Schema-Free Classification (`predict:without-schema`)

The model **infers column types automatically** from the data you send. This is the simplest path when your columns are straightforward.

- **Task type**: `classification` — predicting a categorical value.
- **Target column**: `COSTCENTER` — the model predicts which cost center a product belongs to.
- **Placeholder**: the string `"[PREDICT]"` marks the cell to be filled.
- **SDK method**: `RptClient.predictWithoutSchema()`

### 2. Schema-Based Regression (`predict:with-schema`)

You provide an **explicit column schema** declaring each column's data type (`string`, `numeric`, `date`). This gives the model stronger type information, which is useful for numeric or date-heavy datasets.

- **Task type**: `regression` — predicting a continuous numeric value.
- **Target column**: `DISCOUNT_RATE` — the model predicts the discount percentage for a product.
- **Placeholder**: the numeric value `-1` marks the cell to be filled.
- **Schema flag**: `parse_data_types: true` tells the API to interpret values according to the declared types.
- **SDK method**: `RptClient.predictWithSchema()`

## Project Structure

```text
typescript/
├── .env.example        # Template for required environment variables
├── .gitignore          # Ignores node_modules/, dist/, and .env
├── package.json        # Scripts, dependencies, and project metadata
├── tsconfig.json       # TypeScript compiler configuration (ES2022, NodeNext)
└── src/
    ├── index.ts        # CLI entry point — reads config, selects example, runs it
    ├── config.ts       # Loads .env, validates AICORE_SERVICE_KEY and RPT1_DEPLOYMENT_ID
    ├── data.ts         # Sample row data and column schemas for both examples
    ├── examples.ts     # Creates RptClient and calls predictWithoutSchema / predictWithSchema
    └── output.ts       # Formats and prints prediction responses to the console
```

### How the files connect

```
index.ts
  ├── config.ts      →  readConfig()   →  validates env vars, returns { deploymentId }
  ├── config.ts      →  readSelector() →  parses CLI arg: "without-schema" | "with-schema" | "all"
  ├── examples.ts    →  runWithoutSchema() / runWithSchema()
  │     ├── data.ts  →  provides row data and schemas
  │     └── @sap-ai-sdk/rpt  →  RptClient handles auth and HTTP calls
  └── output.ts      →  printResult()  →  formats response to console
```

## Key Concepts

These are the core RPT-1 API building blocks visible in the sample code:

| Concept | Description |
|---|---|
| `RptClient` | SDK client instantiated with a `deploymentId`. Handles authentication via the SAP AI Core service key. |
| `prediction_config.target_columns` | Declares which column(s) to predict, the placeholder value, and the task type (`classification` or `regression`). |
| `prediction_placeholder` | A sentinel value placed in the target column cells you want the model to fill. Use a string like `"[PREDICT]"` for classification or a numeric value like `-1` for regression. |
| `index_column` | The column that uniquely identifies each row (e.g., `"ID"`), so predictions can be mapped back to their source rows. |
| `parse_data_types` | When `true`, the API interprets row values according to the schema's declared `dtype` entries. Only used with `predictWithSchema`. |
| `predictWithoutSchema` | Sends rows directly — the model infers column types from the data. |
| `predictWithSchema` | Sends rows together with a column schema array — gives the model explicit type information. |

## Prerequisites

- Node.js 20+
- npm 10+
- An SAP AI Core RPT-1 deployment ID

## Configuration

Copy `.env.example` to `.env` and fill in your credentials, or export the variables directly in your shell.

```bash
cp .env.example .env
```

Required variables:

| Variable | Description |
|---|---|
| `AICORE_SERVICE_KEY` | JSON service key for your SAP AI Core instance. Contains `clientid`, `clientsecret`, `url`, and `serviceurls.AI_API_URL`. |
| `RPT1_DEPLOYMENT_ID` | The deployment ID of your RPT-1 model on SAP AI Core. |

Example `.env` content:

```bash
AICORE_SERVICE_KEY='{"clientid":"...","clientsecret":"...","url":"...","serviceurls":{"AI_API_URL":"https://..."}}'
RPT1_DEPLOYMENT_ID='<your-rpt1-deployment-id>'
```

`AICORE_SERVICE_KEY` is the documented local-development path for this sample. If your setup uses SAP Cloud SDK destinations or another supported SAP AI SDK connection mechanism, adapt the environment accordingly.

## Install and Verify

```bash
cd best-practices/narrow-ai/RPT-1/typescript
npm install
npm run typecheck   # validates types without emitting files
npm run build       # compiles TypeScript to dist/
```

## Run

```bash
npm run predict:without-schema   # classification example (COSTCENTER)
npm run predict:with-schema      # regression example (DISCOUNT_RATE)
npm run predict:all              # runs both examples sequentially
```

## What to Expect

Each example prints a summary to the console:

```
=== without-schema ===
requestId: <uuid>
status: 200 Predictions generated successfully.
firstPrediction:
{
  "PRODUCT": "Couch",
  "PRICE": 999.99,
  "ORDERDATE": "2025-11-28",
  "ID": "35",
  "COSTCENTER": "<predicted value>"
}
numPredictions: 1
```

The output includes:
- **requestId** — a unique identifier for the prediction request.
- **status** — HTTP-style code and message from the RPT-1 API.
- **firstPrediction** — the full row with the predicted value filled in.
- **numPredictions** — total number of rows that received predictions.