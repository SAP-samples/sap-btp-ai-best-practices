## Invoice Workspace

The Invoice Workspace screens receivable invoices for eligibility and builds credit
recommendations with a multi-week optimizer. The backend is FastAPI (`api/`). The UI is
Vite with UI5 Web Components (`ui/`). All state lives in SAP HANA. A read-only
assistant is available in the UI and in Joule.

Main capabilities:
- **Eligibility.** Upload an offer workbook, apply rules R13, R1, R16, R17, R11 and R2,
  and inspect the diagnostics for each invoice.
- **Recommendation.** Start from an eligible selection or from an upstream-approved
  extraction. Configure credit settings (limits, mappings, opening exposure, FX,
  repayments), estimate invoice lifetimes with SAP RPT-1, then solve a Monday-based
  weekly schedule with OR-Tools CP-SAT.
- **Results.** Selected and not-selected invoices, capacity per facility, customer and
  group, and downloadable artifacts.
- **Assistant.** Explains saved analyses and runs (LangGraph, SAP Gen AI Hub). It is
  exposed over A2A JSON-RPC for Joule.

## Eligibility rules

The rules live in `api/app/services/eligibility/rules.py` and run in this order:

- R13: the invoice must not be overdue. `due_date` must be strictly after `purchase_date`.
- R1: the due date must be at least `NDDT` days after `purchase_date` (default 6).
- R16: the tenor must be less than `TEIH` days, `due_date - issuance_date < TEIH` (default 15).
- R17: issuance must be at least `ISSPUR` days before `purchase_date` (default 0).
- R11: the currency must be in the eligible list (default `EUR,USD`).
- R2: document number, fiscal year and reference must be unique within the batch.

The defaults come from environment variables (`ELIGIBILITY_NDDT`, `ELIGIBILITY_TEIH`,
`ELIGIBILITY_ISSPUR`, `ELIGIBILITY_CURRENCIES`). An upload can also override them.

## API

Every route except health and agent-card discovery requires the `X-API-Key` header.

- `/api/workspace/*` covers analyses, selections, runs, credit settings, preparation,
  acknowledgement, artifacts and conversations.
- `/api/workspace/lifecycle/*` manages the RPT-1 lifetime history datasets stored in
  HANA: `POST /datasets` (multipart `.xlsx` upload with `dataset_id`, optional `activate`),
  `GET /datasets`, `GET /datasets/{dataset_id}`, `GET /datasets/{dataset_id}/rows`,
  `GET /active` and `PUT /active`. No history workbook ships with the application.
- `POST /api/a2a` is the A2A JSON-RPC transport. `GET /.well-known/agent-card.json`
  serves the agent card.
- `GET /api/health` is the health check.

The interactive API documentation is at `http://127.0.0.1:8000/docs` when running locally.

## Local development

```bash
python3 -m venv .venv
.venv/bin/pip install -r api/requirements.txt
cp api/.env.example api/.env   # fill in API_KEY, AICORE_*, RPT1_MODEL_NAME, hana_*
PYTHONPATH=api .venv/bin/uvicorn app.main:app --reload --port 8000
```

```bash
npm --prefix ui install
npm --prefix ui run dev
```

Then open `/workspace`. The workspace needs a reachable HANA instance and creates its
tables automatically on the first request.

## Tests

```bash
cd api && PYTHONPATH=. ../.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
node --test ui/tests/*.test.mjs
```

## Deployment

The target is Cloud Foundry, with a BTP destination and a Joule capability. Deployments
are always run manually by an operator.

Prerequisites:
- A HANA Cloud technical user with `CREATE TABLE`, `SELECT`, `INSERT`, `UPDATE` and
  `DELETE` in its default schema. The tables are created automatically on first use.
- An AI Core service key, with exactly one RUNNING deployment of `RPT1_MODEL_NAME`
  (for example `sap-rpt-1-small`) and the assistant model `A2A_MODEL`.
- A service instance named `Cloud Logging` in the CF space.
- Cloud Foundry CLI v8, SAP BTP CLI, Python 3.12, Node.js 20.12 to 24 for the Joule
  Studio CLI, `openssl` and `curl`.

Steps (Bash, from this folder):
1. Create `api/.env` from `api/.env.example` and fill in the `AICORE_*` and `hana_*` values.
2. Set your own routes in `manifest.yaml` (`routes`, `API_BASE_URL`, `A2A_BASE_URL`,
   `A2A_ENDPOINT_URL`, `ALLOWED_ORIGIN`, `VITE_API_BASE_URL`, `VITE_APP_HOST`).
3. `cf login --sso`, `cf target -o <org> -s <space>`, `btp login --sso`, `btp target`.
4. `./deploy.sh`. It reuses or generates the API key, pushes both apps, binds Cloud
   Logging, creates the `RECEIVABLES_AGENT` destination, validates the Joule capability
   and smoke-checks the health route. `./deploy.sh --rotate-api-key` rotates the key.
5. `joule login`, `joule deploy -c -n receivables_assistant`, `joule launch receivables_assistant`.
6. Upload and activate an RPT-1 history workbook with
   `POST /api/workspace/lifecycle/datasets`. Without an active dataset every lifetime
   falls back to 28 days and every run asks for acknowledgement.

Known limitation: there is no XSUAA login yet. The UI calls the API with a shared key
(`VITE_API_KEY`) compiled into the browser bundle. Add an approuter with XSUAA before
production use.
