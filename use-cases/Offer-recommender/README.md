# Customer Offer Advisor

The recommendation source of truth is HANA runtime data plus source-controlled anonymized catalogs. The chat layer uses LangGraph for orchestration and durable state, but it does not decide eligibility or final offers.

## Architecture

- `api/`: FastAPI backend, deterministic recommendation domain, LangGraph chat workflow, batch exports
- `ui/`: UI5 frontend for chat assistant, single-account lookup, and batch analysis

The new runtime does not import the archived Streamlit app or the archived backend package copies.

## Intended Process Logic

For each account, the backend preserves this flow:

1. Load HANA runtime tables and catalogs
2. Compute typed account facts
3. Evaluate deterministic offers
4. Generate missing-fact follow-up questions
5. Rank and select the final offer
6. Expose the result through chat, single-account lookup, and batch analysis

## Sources Of Truth

The backend reads from HANA tables or compatible views that match the `COA_*` defaults in `api/app/nbo/config.py`:

- customer, segment, active offering, program contract, profile, DER, and event-history tables in HANA
- active catalog JSON under `api/app/nbo/catalogs/`
- public anonymized seed/source artifacts under `anonymized/api/`

Batch outputs are written under `api/output/batch_runs/`.
Chat thread persistence is stored in `api/output/chat_threads.sqlite`.

Local seed workbooks are only supported for explicit HANA bootstrap workflows through `api/scripts/load_hana_seed_data.py`. They are not runtime inputs and are excluded from deployment payloads.

To populate a demo HANA target with public anonymized seed data, run:

```bash
python api/scripts/load_hana_seed_data.py \
  --customer-workbook anonymized/api/data_seed/customer_seed.xlsx \
  --program-codes-workbook anonymized/api/data_seed/program_seed.xlsx
```

The command recreates the configured `COA_*` runtime tables, so run it only against the intended demo schema.

## Backend Design

- Deterministic recommendation logic lives under `api/app/nbo/` and `api/app/services/recommendations.py`
- Shared response contracts live under `api/app/models/nbo.py`
- Chat orchestration lives under `api/app/chat/service.py`
- Batch execution lives under `api/app/services/batch.py`
- API routes live under `api/app/routers/`

The chat flow is intentionally constrained:

- account extraction is deterministic
- account evaluation is deterministic
- question planning is deterministic
- answer application is deterministic
- LangGraph is used for thread state, transitions, and persistence

## API Surface

The backend exposes:

- `POST /api/accounts/evaluate`
- `POST /api/chat/threads`
- `GET /api/chat/threads/{thread_id}`
- `POST /api/chat/threads/{thread_id}/messages`
- `POST /api/chat/threads/{thread_id}/decline`
- `POST /api/batch/runs`
- `GET /api/batch/runs/{run_id}`
- `GET /api/batch/runs/{run_id}/artifacts/{artifact_name}`

## Local Development

### Backend

```bash
cd api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

### Frontend

```bash
cd ui
npm install
cp .env.example .env
npm run dev
```

By default, the UI expects the API at `http://127.0.0.1:8000/`.

If `API_KEY` is unset in the backend environment, local/test requests are accepted without auth. In deployed environments, set `API_KEY` and pass the same value from the UI.

## Verification

Backend tests:

```bash
python3 -m pytest api/tests -q
```

Frontend tests:

```bash
cd ui
npm test
```

Frontend production build:

```bash
cd ui
npm run build
```

## Deployment

Cloud Foundry deployment is configured through:

- `manifest.yaml`
- `deploy.sh`

`deploy.sh` generates a temporary API key and injects it into the manifest deployment.
It also injects the HANA runtime variables required by the API app:

- `hana_address`
- `hana_port`
- `hana_user`
- `hana_password`
- `hana_encrypt`

Set those values in your shell or in `api/.env` before running `./deploy.sh`.
To use a different env file, run `DEPLOY_ENV_FILE=/path/to/env ./deploy.sh`.
The env file is read locally and is not uploaded to Cloud Foundry.


## Current Limitations

- Runtime requires populated HANA tables or aliases matching the configured `COA_*` table names
- Batch execution is synchronous in-process
- The UI build still emits a chunk-size warning because the UI shell bundle remains large
- Legacy code is still present under `archived/` for migration/reference, even though it is no longer part of the active runtime
