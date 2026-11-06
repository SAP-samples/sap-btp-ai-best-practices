# Customer Service Assistant

This PoC processes customer Gmail or manually entered requests for shipment documents and order status. The API classifies incoming email, retrieves matching order context and shipment documents stored in SAP HANA, creates reviewable drafts with HANA-backed attachments when applicable, and waits for a user to approve or reject each draft in the UI.

- **Frontend:** UI5 Web Components, plain JavaScript, Vite
- **Backend:** FastAPI, Pydantic, SQLAlchemy HANA, LangGraph
- **Runtime stores:** SAP HANA for documents, source emails, extractions, draft rows, fetch state, and history
- **Gmail access:** SAP BTP Destination service with an OAuth2 refresh-token destination

## Project Structure

```text
project-root/
├── api/          # FastAPI backend application
├── data/         # data/anonymized/ holds the demo shipment files read by the HANA setup script
├── ui/           # UI5 Web Components frontend application
├── deploy.sh     # Cloud Foundry deployment helper
└── manifest.yaml # Cloud Foundry application manifest
```

## Prerequisites

- Node.js and npm
- Python 3.12 or compatible project virtual environment
- Cloud Foundry CLI for deployment
- SAP HANA credentials
- SAP AI Core / GenAI Hub credentials
- SAP BTP Destination service instance for Gmail
- `cds` CLI when running the API locally with a BTP service binding

## Local Setup

Create local environment files:

```bash
cp api/.env.example api/.env
cp ui/.env.example ui/.env
```

Set these values in `api/.env`:

```dotenv
API_KEY=replace-me
HANA_ADDRESS=replace-me
HANA_PORT=443
HANA_USER=replace-me
HANA_PASSWORD=replace-me
HANA_ENCRYPT=true
AICORE_AUTH_URL=replace-me
AICORE_CLIENT_ID=replace-me
AICORE_CLIENT_SECRET=replace-me
AICORE_BASE_URL=replace-me
AICORE_RESOURCE_GROUP=replace-me
FETCH_MAX_DAYS=7
FETCH_MAX_WORKERS=4
GMAIL_MAILBOX_ID=me
GMAIL_DESTINATION_NAME=GMAIL_API_READONLY
GMAIL_DESTINATION_LEVEL=subaccount
GMAIL_REFRESH_TOKEN_PROPERTY=GMAIL_REFRESH_TOKEN
DESTINATION_SERVICE_NAME=shipment-document-request-destination
DOCUMENT_REQUEST_EXTRACTOR_MODEL=gpt-6-luna
DOCUMENT_REQUEST_AGENT_MODEL=gpt-6-luna
DOCUMENT_REQUEST_FROM_EMAIL=
```

Set `ui/.env` so `VITE_API_KEY` matches `API_KEY`:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_API_KEY=replace-me
```

Install dependencies:

```bash
source .venv/bin/activate
python -m pip install -r api/requirements.txt
npm install --prefix ui
```

## Gmail Destination And VCAP_SERVICES

The API does not store Google OAuth JSON locally. It expects `VCAP_SERVICES` from a bound SAP BTP Destination service and then resolves the Gmail destination configured by `GMAIL_DESTINATION_NAME`.

Create the Cloud Foundry Destination service instance in the target org and space:

```bash
cf login -a https://api.cf.eu10-004.hana.ondemand.com --sso
cf marketplace -s destination
cf create-service destination lite shipment-document-request-destination
cf services
```

If the `lite` plan is not available in your landscape, use the plan shown by `cf marketplace -s destination`.

The application can reuse the existing subaccount-level destination `GMAIL_API_READONLY` shown in the BTP cockpit. The name is not a technical limitation. What matters is that the `GMAIL_REFRESH_TOKEN` value was authorized with scopes that allow the app to list and read messages, create Gmail drafts with attachments, send drafts, and delete rejected drafts. If the refresh token was minted with read-only scopes only, Gmail will allow inbox reads but return an authorization error when the app tries to create, send, or delete a draft.

Draft creation uses the standard Gmail REST API endpoint `POST /gmail/v1/users/{userId}/drafts`; there is no separate send-mail API for creating the draft. Google requires one of these OAuth scopes for draft creation:

- `https://www.googleapis.com/auth/gmail.compose`
- `https://www.googleapis.com/auth/gmail.modify`
- `https://mail.google.com/`

Current expected destination:

```text
Name: GMAIL_API_READONLY
Type: HTTP
Proxy Type: Internet
URL: https://gmail.googleapis.com/gmail/v1
Authentication: OAuth2RefreshToken
Client ID: <Google OAuth client id>
Client Secret: <Google OAuth client secret>
Token Service URL: https://oauth2.googleapis.com/token
Token Service URL Type: Dedicated
Use default client trust store: enabled
Additional Property: GMAIL_REFRESH_TOKEN=<fresh Google refresh_token>
```

If draft operations fail with `HTTP 403` and an insufficient-scope message, keep the same destination but replace `GMAIL_REFRESH_TOKEN` with a fresh token granted compose/send draft access, or duplicate the destination under a clearer name such as `GMAIL_API_COMPOSE` and update `GMAIL_DESTINATION_NAME`.

For local development, create the CDS binding from the API folder so `api/.cdsrc-private.json` is created in the folder that is actually run by Uvicorn:

```bash
cd api
cds bind --to shipment-document-request-destination
```

Then start the API through the binding:

```bash
cd api
cds bind --exec -- ../.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

In Cloud Foundry, do not use `cds bind --exec`; `VCAP_SERVICES` is supplied by the `services` binding in `manifest.yaml`.

## HANA Seeding

The runtime API never reads shipment documents from disk. The setup script reads anonymized local scenario files from `data/anonymized/`, clears previously seeded order/document rows by default, stores PDF metadata plus binary content in HANA, parses scenario `.eml` tables, and derives order status from HANA line rows.

```bash
cd api
../.venv/bin/python scripts/setup_hana_document_request_data.py
```

Reset and reload the PoC tables:

```bash
cd api
../.venv/bin/python scripts/setup_hana_document_request_data.py --drop-existing
```

## Run Locally

Backend:

```bash
cd api
cds bind --exec -- ../.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Frontend:

```bash
cd ui
npm run dev -- --host 127.0.0.1 --port 5173
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). Use **Fetch new emails** to process Gmail messages, then approve or reject draft responses. Fetches and scenario loads run in the background: the UI shows live progress ("Processing X of Y emails") and refreshes the draft list as drafts appear.

For local testing without Gmail/BTP Destination configured, use the **Manual email** form or **Load scenario emails**. These paths create review rows even when Gmail draft creation is unavailable. **Fetch new emails** remains Gmail-dependent and will return a service error if the Gmail destination is not configured.

## Deployment

The Cloud Foundry manifest deploys:

- `shipment-document-request-agent-api`
- `shipment-document-request-agent-ui`
- API route: `https://shipment-document-request-agent-api.cfapps.eu10-004.hana.ondemand.com`
- UI route: `https://shipment-document-request-agent.cfapps.eu10-004.hana.ondemand.com`
- Destination service binding: `shipment-document-request-destination`

Before deployment:

1. Confirm `api/.env` contains HANA and AI Core values.
2. Confirm the Destination service instance `shipment-document-request-destination` exists in the target CF space.
3. Confirm the subaccount destination `GMAIL_API_READONLY` exists and has a valid `GMAIL_REFRESH_TOKEN` with draft permissions.
4. Log in to Cloud Foundry:

```bash
cf login -a https://api.cf.eu10-004.hana.ondemand.com --sso
```

Deploy:

```bash
chmod +x deploy.sh
./deploy.sh
```

The deploy script:

1. Generates a temporary API key.
2. Builds the UI with the production API route and API key.
3. Pushes API and UI using `manifest.yaml`.
4. Syncs HANA and AI Core runtime environment variables from `api/.env` to the API app without printing secret values.
5. Restarts the API app so the runtime environment is active.

Override optional deployment values:

```bash
FETCH_MAX_DAYS=14 ./deploy.sh
DOCUMENT_REQUEST_FROM_EMAIL=documents@example.com ./deploy.sh
```

Manual deployment uses the same manifest variables:

```bash
cf push \
  --var api_key="$(openssl rand -hex 32)" \
  --var api_route="https://shipment-document-request-agent-api.cfapps.eu10-004.hana.ondemand.com" \
  --var ui_host="shipment-document-request-agent.cfapps.eu10-004.hana.ondemand.com" \
  --var fetch_max_days="7" \
  --var document_request_from_email=""
```

After manual `cf push`, set the HANA and AI Core values on `shipment-document-request-agent-api` with `cf set-env`, then restart the API app.

## Tests

Frontend helper tests and build:

```bash
node --test ui/tests/reviewHelpers.test.js
npm --prefix ui run build
```
