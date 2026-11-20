# ECN Impact Analysis

An application on SAP Business AI Platform that helps manufacturing engineers understand what a released Engineering Change Notice (ECN) means for their plants.

The application reads a PLM export of the ECN together with staged SAP extracts (engineering and manufacturing BOMs, variant-configuration rules, material master, purchase info records and stock). It builds a deterministic, evidence-backed impact snapshot of the affected parts, BOMs and plants. SAP Generative AI Hub then writes a business-readable impact analysis on top of that evidence. An engineer reviews the result, asks follow-up questions in a chat, and records an attributed decision. Every step is logged in an audit trail.

The application is review-first: the AI never takes a decision on its own, and accepting an ECN does not write to any SAP system (the SAP update is simulated).

## What it does

- **ECN intake:** imports a JSON export of change notices; the user selects which ECNs to load.
- **Deterministic evidence:** matches every ECN part against the engineering BOM (EBOM) and manufacturing BOM (MBOM), resolves the affected plants from the configuration rules and the plant mapping, and adds purchasing and stock context.
- **AI impact analysis:** writes an executive summary and staged business perspectives (what changes, what is impacted, where, what is still open) from the evidence snapshot only.
- **ECN Assistant:** a chat panel that answers questions about the open ECN ("which parts are affected?", "where are they used?"), or across all loaded ECNs from the inbox, with tools that query the snapshots and a knowledge graph of the ECNs.
- **Material detail:** per part, shows BOM occurrences, plants, procurement data and, where a drawing exists, an AI-assisted drawing-feasibility check against a pipe standard.
- **Engineering review:** Accept ECN or Send back to engineering, with reviewer name and comment, shown in the ECN list and detail.
- **Reference data management:** the Settings page publishes and versions the reference workbooks, and checks AI provider readiness.

## Architecture

```text
Browser (UI5 Web Components)
        |
        |  HTTPS + X-API-Key
        v
FastAPI backend (Cloud Foundry)
   |-- ECN intake and evidence builder (deterministic)
   |-- Analysis jobs (SAP Generative AI Hub)
   |-- ECN Assistant chat (tool-calling agent)
   |-- Drawing-feasibility assessment (multimodal model)
        |
        v
SAP HANA Cloud
   |-- Relational tables: ECNs, snapshots, reference datasets, analyses, reviews, audit log
   |-- Knowledge graph: rebuildable RDF projection of each ECN
```

## Key components

| Area | Path | Purpose |
|---|---|---|
| UI pages | `ui/src/pages/` | ECN Inbox (`ecns`), ECN detail (`ecn-detail`), Settings (`settings`) |
| UI components | `ui/src/components/` | Plant impact table, BOM change evidence, material dialog, ECN Assistant chat, audit log |
| UI routes | `ui/src/routes.js` | `/ecns`, `/ecns/:id` (+ `/impact`, `/attention`, `/references`), `/settings` |
| API entry point | `api/app/main.py` | App startup; creates the HANA schema and tables if they are missing |
| API routes | `api/app/routers/` | Uploads, ECNs, snapshots, analysis progress, assistant, reference datasets, drawing feasibility, provider status |
| ECN domain | `api/app/ecn/` | Intake, evidence, rules, LLM prompts, assistant tools, RDF projection |
| Drawing feasibility | `api/app/drawing_feasibility/` | Pipe-drawing assessment service and policy files |
| Persistence | `api/app/persistence/` | HANA schema, repositories, codecs |
| Sample data | `data/` | Fictitious, consistent dataset (BOM, rules, master data, ECN export) |
| Scripts | `scripts/` | Publish the sample datasets, import rules, anonymization tool |
| Tests | `tests/`, `ui/tests/` | Python unit and integration tests, UI unit tests |
| Docs | `docs/` | One file per feature; start at [docs/README.md](docs/README.md) |

## AI capabilities

- SAP Generative AI Hub text model for the impact analysis and the chat (default `gpt-5.4`, set with `LLM_MODEL`).
- Tool-calling chat agent (ECN Assistant) scoped to the open ECN, or to all loaded ECNs from the inbox.
- Knowledge graph in SAP HANA Cloud (RDF projection) used by the chat to discover related parts and ECNs.
- Multimodal model for the drawing-feasibility check.

All model calls go through SAP AI Core. The evidence snapshot is deterministic; the model only writes on top of it.

## Prerequisites

- Python 3.12 and Node.js 20.19+
- An SAP AI Core service key with the models above deployed
- An SAP HANA Cloud user that can create tables in its schema
- Cloud Foundry CLI (for deployment only)

## Getting started

### 1. Configure

```bash
cp api/.env.example api/.env
cp ui/.env.example ui/.env
```

- `api/.env`: fill in `AICORE_*`, `HANA_*` and a random `API_KEY` (for example `openssl rand -hex 32`).
- `ui/.env`: set `VITE_API_BASE_URL=http://127.0.0.1:8000` and `VITE_API_KEY` to the same value as `API_KEY`.

### 2. Start the API

```bash
cd api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

On startup the API checks the HANA schema and creates any missing tables. The Swagger UI is at http://127.0.0.1:8000/docs.

### 3. Start the UI

In a second terminal:

```bash
cd ui
npm install
npm run dev
```

Open the URL that Vite prints (default http://localhost:5173).

### 4. Load the sample data

With the API running, from the repository root:

```bash
api/.venv/bin/python scripts/publish_reference_datasets.py --manifest scripts/manifests/q900v-2026-09.json
```

This publishes the BOM, configuration rules, material master, purchase info and stock workbooks, and imports the two demonstration ECNs `ECN345274` and `ECN346863`. See [docs/reference-datasets.md](docs/reference-datasets.md).

## Tests

```bash
PYTHONPATH=api api/.venv/bin/python -m pytest -q
npm test --prefix ui
```

Tests that need live HANA or AI Core credentials are marked `hana_live` and `llm_live` and are skipped unless enabled. See [docs/ecn-impact-testing.md](docs/ecn-impact-testing.md).

## Deployment to SAP BTP Cloud Foundry

1. Change the app names and routes in `manifest.yaml`.
2. Log in: `cf login -a <API_ENDPOINT> --sso`
3. Install the UI dependencies: `npm install --prefix ui`
4. Run `./deploy.sh`. It generates a new API key, builds the UI locally with it, and pushes the API and the static UI.

Build details: [docs/cf-local-ui-build.md](docs/cf-local-ui-build.md).

## Sample data

`data/` is an anonymized copy of the source extraction the application was built with. All company names, plants, brands, part numbers and people are fictitious and consistent across files. How the data and code were anonymized is described in [docs/anonymization.md](docs/anonymization.md).

## Limitations

- Proof of concept: no live connection to PLM or SAP; all source data comes from files.
- Accepting an ECN does not write to SAP; the update is simulated.
- AI output is a recommendation for a human reviewer, not a decision.
