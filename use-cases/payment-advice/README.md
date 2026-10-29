# Payment Advice Extractor

Automates the cash-application work behind customer payment advices (remittances):
it reads incoming advices from a mailbox, extracts their lines with SAP Document AI,
interprets the deductions with the customer's documented rules, lets a person review
the result, matches every invoice line against the open items in SAP S/4HANA, and
creates the S/4 payment advice that S/4 later uses to clear the incoming payment.

- **Frontend:** UI5 Web Components + Vite (`ui/`)
- **Backend:** FastAPI (`api/`), SAP HANA Cloud (configuration, workspace, job queue),
  SAP Document AI (extraction), SAP Generative AI Hub (mapping and deduction agent),
  SAP S/4HANA OData (lookup, payment advice)
- **Deployment:** SAP BTP Cloud Foundry (`manifest.yaml`, `deploy.sh`)

## What it does

```text
Gmail inbox / manual upload
   -> extraction (Document AI, customer schema or canonical schema + LLM mapping)   UC-01
   -> customer rules + deduction interpretation (agent)                             UC-02
   -> review in the UI (edit cells, accept, mark reviewed)                          UC-03
   -> S/4 check: invoice references -> open items, customers, company code
   -> S/4 payment advice (API_PAYMENT_ADVICE_SRV)
```

| Use case | Status |
|---|---|
| UC-01 extraction to the canonical payment advice | built, live-validated |
| UC-02 deduction interpretation and rules authoring | built |
| Customer rule engine (accounts, company code, reason codes from documented rules) | built |
| UC-03 email intake (Gmail destination or manual), review workspace, parallel processing | built |
| Document AI schema administration from the Rules and Advice chats | built |
| S/4 integration: check and post payment advices | built, first live post on 2026-10-03 |

The target S/4 system, client and company code are configured through the `S4_*`
variables in `api/.env` (see Setup).

## Project structure

```text
api/                FastAPI backend (deployed)
  app/payment_advice/   extraction pipeline, schemas, customer playbooks, rule engine
  app/deduction_agent/  UC-02 agent runtime
  app/email_ingestion/  Gmail/manual intake, HANA workspace store, workers
  app/s4/               S/4 connectivity, lookup, validation, payment advice, probe
  app/routers/          HTTP endpoints (/api/payment-advice, /api/email-ingestion, /api/joule)
ui/                 UI5 Web Components frontend (deployed)
  src/pages/home/       inbox, advice review, S/4 posting tab
  src/pages/rules/      deduction rules assistant (chat)
tests/unit/         offline Python tests
manifest.yaml       Cloud Foundry apps: payment-advice-automation-api, payment-advice-automation
deploy.sh           deployment helper (run by the user)
```

## Prerequisites

- Python 3.12 and Node.js (npm)
- Access to: SAP HANA Cloud, SAP AI Core (Generative AI Hub), SAP Document AI (service key)
- For S/4 locally: the SAP VPN and an S/4 user (direct mode)
- For deployment: the Cloud Foundry CLI

## Setup

```bash
cp api/.env.example api/.env
```

```bash
cp ui/.env.example ui/.env
```

Fill in `api/.env`:

- `AICORE_*`: SAP AI Core credentials; `HANA_*`: HANA Cloud connection.
- Document AI: place the service key at `api/app/payment_advice/schema/service_key.json`
  (git-ignored), or use one of the `DOCUMENT_AI_SERVICE_KEY_*` variables.
- `API_KEY`: any secret; `ui/.env` `VITE_API_KEY` must have the same value.
- S/4 (optional): `S4_BASE_URL`, `S4_USERNAME`, `S4_PASSWORD`, `S4_CLIENT`, `S4_VERIFY`
  for direct mode; Destination/Connectivity values for btp mode; demo aliases
  `S4_COMPANY_CODE_ALIASES="CA01=Z291"` and
  `S4_REASON_CODE_ALIASES="316=060,319=060,321=060,323=060"` (demo only, empty in production).
- `DOCUMENT_AI_CLIENT_ID` (optional): Document AI client id; default `ai4u_payment_advice`.
- `PAYMENT_ADVICE_SEED_CUSTOMERS_PATH` (optional): JSON list of priority customers seeded
  at startup (`[{"client_key": "...", "display_name": "..."}]`, path relative to `api/`).
  Default: the packaged demo list `api/app/payment_advice/seed_customers.json` with one
  fictional customer, Northwind (no S/4 data: never check or post its advices).

Create the Python environment at the repository root (the test commands use `.venv`):

```bash
python3.12 -m venv .venv
```

```bash
.venv/bin/pip install -r api/requirements.txt
```

The HANA tables (customers, schemas, playbooks, workspace, job queue) are created
automatically on first start if they do not exist, and missing priority customers from
the seed list are inserted (existing rows are never changed).

## Run locally

Backend (http://127.0.0.1:8000, Swagger UI at `/docs`):

```bash
cd api && ../.venv/bin/uvicorn app.main:app --reload
```

Frontend (Vite dev server, http://localhost:5173):

```bash
cd ui && npm install && npm run dev
```

**Shared job queue:** every running API instance (local and Cloud Foundry) processes
jobs from the same HANA queue. When you test a local code change, stop the deployed API
or deploy the same version, otherwise the deployed version may process your uploads.

## Use the application

1. **Inbox** (`/home`): fetch new emails (Gmail destination) or add one manually with
   its attachments. Each advice is extracted and interpreted in the background.
2. **Review:** pick the advice, check the header and line table, correct cells and
   **Accept Changes** (or **Undo Changes**), then **Mark reviewed**. Tabs show the
   original email, attachments and a per-advice chat.
3. **S/4 posting** tab: **Check against S/4** matches every invoice line to an open S/4
   item and lists blocking issues; deduction lines show "New deduction".
   **Post to S/4** creates the payment advice and shows what S/4 stored.
4. **Rules** (`/rules`): chat with the rules assistant to read or update a customer's
   deduction playbook (PDF/Word/Excel rule documents can be attached), create customers,
   make them priority, and inspect or change a priority customer's Document AI schema
   (plan, prepare a new version, test it on a sample, publish). The Advice chat can change
   the schema of the advice's own customer.

## S/4 connectivity probe

Read-only check of the S/4 system (services, company codes, customers; every call is a GET). In Cloud Foundry:

```bash
cf run-task payment-advice-automation-api --name s4-probe --command "python -m app.s4.probe <company code>"
```

Locally, from the `api/` folder:

```bash
python -m app.s4.probe <company code>
```

## Tests

```bash
PYTHONPATH=api .venv/bin/python -m unittest discover -s tests/unit -q
```

```bash
node --test ui/tests/inbox-concurrency.test.mjs
```

## Gmail intake

Gmail is read through the BTP destination `PAYMENT_ADVICE_GMAIL_READONLY`
(`OAuth2RefreshToken`, refresh token in the additional property `GMAIL_REFRESH_TOKEN`)
and a Destination service instance bound to the API. Manual intake works without it.
Destination URL `https://gmail.googleapis.com/gmail/v1`, token URL
`https://oauth2.googleapis.com/token`, client id and secret of the Google OAuth web client.
Obtain the refresh token once with Google's OAuth authorization-code flow for the mailbox
owner (scope `https://www.googleapis.com/auth/gmail.readonly`, `access_type=offline`,
`prompt=consent`).

## Deployment (Cloud Foundry)

Deployments are run by the user, never automatically.

```bash
cf login -a https://api.cf.eu10-005.hana.ondemand.com --sso
```

```bash
./deploy.sh
```

`deploy.sh` generates a fresh API key, reads AI Core and HANA credentials from
`api/.env`, pushes both apps from `manifest.yaml`, sets the S/4 btp-mode variables
(`S4_CONNECTIVITY_MODE=btp`, Destination/Connectivity values, advice type, and the demo
aliases), and sets or unsets `DOCUMENT_AI_CLIENT_ID` and
`PAYMENT_ADVICE_SEED_CUSTOMERS_PATH` as defined in `api/.env`. Direct S/4 credentials
never leave the laptop. Apps:

- UI: https://payment-advice-automation.cfapps.eu10-005.hana.ondemand.com
- API: https://payment-advice-automation-api.cfapps.eu10-005.hana.ondemand.com
