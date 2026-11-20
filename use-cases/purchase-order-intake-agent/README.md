# Purchase Order Intake Agent

Agentic intake of customer purchase orders: customers send orders by e-mail (PDF, Word, Excel,
images or plain text); a LangGraph ReAct agent on **gpt-6-luna** (SAP Generative AI Hub, high
reasoning effort) classifies and extracts them, resolves customer, ship-to and materials against
SAP ECC master data, fixes what it can, and hands the order to a **deterministic validation gate**.
Orders that pass are posted automatically as ECC sales orders; everything else waits in a review
UI where a human corrects the highlighted fields and confirms.

SAP ECC is **mocked** with HANA Cloud tables that mirror ECC (`KNA1_POAGENT`, `KNVV_POAGENT`,
`VBAK_POAGENT`, ...). Matching, validation, posting payload and review are real; replacing the mock
with SAP Integration Suite calls to `BAPI_SALESORDER_CREATEFROMDAT2` is described in
[docs/ecc-integration.md](docs/ecc-integration.md).

```
Upload / e-mail trigger ─► intake (parse .eml, PDFs, Office, images) ─► QUEUED
   ─► worker pool ─► order agent (skills + tools: customers, partners, materials, units, submit_order)
   ─► deterministic gate ─► POSTED (mock ECC sales order)  |  NEEDS_REVIEW  |  NOT_AN_ORDER  |  FAILED
   ─► review UI: Save / Confirm / Reprocess; reviewer corrections become precedent for later runs
```

## Repository layout

```
api/                         FastAPI backend (Python 3.12)
  app/main.py                app, lifespan (schema bootstrap, order pipeline)
  app/db/                    HANA engine, table catalogue + automatic DDL, row helpers
  app/ecc/                   mock ECC access: master data (read), sales documents (write)  <- production swap points
  app/intake/                .eml / Office / image parsing, upload grouping and storage
  app/agent_core/            vendored LangGraph ReAct runtime (create-langgraph-react-agent template)
  app/order_agent/           agent config, skills, tools, prompt, run wrapper
  app/orders/                matching, quantity conversion, validation, service, posting, pipeline, repository
  app/routers/, app/models/  REST API
  app/cli/                   seed_master_data, run_samples
ui/                          React + UI5 Web Components (orders list, upload dialog, review page)
tests/unit/                  unit tests (no credentials needed)
docs/                        feature documentation (see index below)
data/seed/                   mock ECC master-data seed and matching acceptance list (fictitious data)
data/orders/                 sample purchase orders as PDF (fictitious data, see data/README.md)
```

## Prerequisites

* Python 3.12, Node.js 20+
* SAP AI Core service key with a `gpt-6-luna` deployment (Generative AI Hub)
* SAP HANA Cloud instance (the user needs `CREATE TABLE`; `CREATE SCHEMA` if `HANA_SCHEMA` is set)
* Cloud Foundry CLI for deployment

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r api/requirements.txt
npm --prefix ui install
cp api/.env.example api/.env      # fill AICORE_*, HANA_*, API_KEY (optional HANA_SCHEMA)
cp ui/.env.example ui/.env        # VITE_API_BASE_URL, VITE_API_KEY (= API_KEY)
```

All tables are created automatically at API start-up (or by the seed CLI). Load the mock ECC
master data (seed file in `data/seed/master_data.json`, see
[docs/mock-ecc-master-data.md](docs/mock-ecc-master-data.md)):

```bash
cd api
../.venv/bin/python -m app.cli.seed_master_data --dry-run
../.venv/bin/python -m app.cli.seed_master_data
```

## Run locally

```bash
.venv/bin/python -m uvicorn app.main:app --app-dir api --port 8000
npm --prefix ui run dev -- --port 5173
```

Open http://localhost:5173, click **Upload**, drop an `.eml` / PDF or paste an order e-mail.

Command-line end-to-end run over sample files (progress bar, per-order report):

```bash
cd api
../.venv/bin/python -m app.cli.run_samples ../data/orders --concurrency 4
../.venv/bin/python -m app.cli.run_samples ../data/orders --post
../.venv/bin/python -m app.cli.seed_master_data --tables ZUOM_ALIAS_POAGENT --reset-transactions   # clean demo state
```

## Tests

```bash
cd api && ../.venv/bin/python -m unittest discover -s ../tests/unit -t ..
npm --prefix ui run typecheck && npm --prefix ui run build
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `AICORE_*` | - | Generative AI Hub credentials (model and reasoning effort: `api/app/order_agent/config/agent.yaml`) |
| `HANA_ADDRESS`, `HANA_PORT`, `HANA_USER`, `HANA_PASSWORD`, `HANA_ENCRYPT` | - | HANA Cloud |
| `HANA_SCHEMA` | user schema | optional dedicated schema |
| `API_KEY` | - | shared `X-API-Key` (UI: `VITE_API_KEY`) |
| `ORDER_WORKERS` | 4 | parallel agent runs per instance |
| `ORDER_AGENT_TIMEOUT_SECONDS` | 900 | max agent run |
| `AUTO_POST_ENABLED` | true | post orders that pass the gate |
| `CONFIDENCE_THRESHOLD` | 0.8 | review threshold |
| `ORDER_PIPELINE_ENABLED` | true | run workers in this instance |

More in [docs/order-processing-pipeline.md](docs/order-processing-pipeline.md).

## Deployment to SAP BTP Cloud Foundry (run by you)

Deployment is never automated from this repository's tooling; run it yourself:

```bash
cf login -a <API_ENDPOINT> --sso
./deploy.sh
```

`deploy.sh` generates a fresh API key, reads `AICORE_*` / `HANA_*` (and optional `HANA_SCHEMA`) from
`api/.env`, pushes `po-intake-agent-api` and `po-intake-agent-ui` (`manifest.yaml`, adjust routes
first) and binds SAP Cloud Logging. Load the master data from your machine with the seed CLI (same
HANA instance) before processing orders.

## Documentation

| Document | Content |
|---|---|
| [docs/hana-data-model.md](docs/hana-data-model.md) | All tables and columns, ECC mapping and deviations, schema bootstrap |
| [docs/mock-ecc-master-data.md](docs/mock-ecc-master-data.md) | Master-data access, matching rules and scores, seed CLI |
| [docs/order-intake.md](docs/order-intake.md) | Upload API, e-mail parsing, inline images, what the model receives |
| [docs/order-agent.md](docs/order-agent.md) | Agent runtime, configuration, tools, skills, confidence |
| [docs/order-processing-pipeline.md](docs/order-processing-pipeline.md) | Statuses, validation gate, posting, review API, concurrency and recovery |
| [docs/correction-learning.md](docs/correction-learning.md) | Reviewer corrections as precedent |
| [docs/ui-order-intake.md](docs/ui-order-intake.md) | Orders list, upload dialog, review page |
| [docs/frontend-react-ui5.md](docs/frontend-react-ui5.md) | UI framework conventions |
| [docs/ecc-integration.md](docs/ecc-integration.md) | BAPI / IDoc mapping, CPI connectivity, production checklist |
