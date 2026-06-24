# Email Price Change Review Assistant

This repository contains a working price-change review assistant for supplier emails.
It ingests Gmail messages or manually submitted emails, extracts requested price
changes with SAP GenAI Hub, resolves supplier/material context through HANA and
S/4HANA, and gives reviewers a UI to edit, approve, or reject draft price changes.

## Current state

- `api/`: FastAPI backend. It owns Gmail ingestion, manual email processing,
  attachment handling, HANA persistence, S/4 lookup/write integration, progress
  reporting, and protected JSON APIs.
- `ui/`: Vite SPA using UI5 Web Components. It provides the incoming request
  review queue and the approved/rejected history page.
- `deploy.sh`: manual Cloud Foundry deployment helper. It builds the UI,
  generates a per-deploy API key, runs `cf push`, copies runtime secrets from
  `api/.env` to the API app, binds Cloud Logging, and restarts the API app.
- `manifest.yaml`: Cloud Foundry manifest for the API and UI apps.

## Repository layout

```text
.
├── api/
│   ├── app/
│   │   ├── main.py
│   │   ├── routers/price_changes.py
│   │   ├── price_changes/
│   │   ├── models/
│   │   └── security.py
│   ├── scripts/
│   │   ├── setup_hana_demo_data.py
│   │   ├── encode_google_oauth.py
│   │   └── s4_connectivity_probe.py
│   ├── seed_data/
│   ├── .env.example
│   ├── .cdsrc-private.json
│   └── requirements.txt
├── ui/
│   ├── src/
│   │   ├── main.js
│   │   ├── routes.js
│   │   ├── services/api.js
│   │   └── pages/
│   ├── tests/
│   ├── .env.example
│   └── package.json
├── docs/
├── scripts/
├── tests/
├── deploy.sh
└── manifest.yaml
```

## Backend API

The FastAPI app starts from `api/app/main.py` and mounts
`api/app/routers/price_changes.py` under `/api`. All price-change routes require
the shared `X-API-Key` header configured with `API_KEY`.

Current endpoints:

- `GET /api/health`
- `POST /api/emails/fetch-new`
- `POST /api/emails/manual`
- `POST /api/emails/manual-with-attachments`
- `POST /api/processing-runs`
- `GET /api/processing-runs/{processing_run_id}`
- `GET /api/email-attachments/{attachment_id}/download`
- `GET /api/price-change-drafts`
- `GET /api/price-change-history`
- `DELETE /api/price-change-history`
- `PATCH /api/price-change-drafts/{draft_id}`
- `POST /api/price-change-drafts/{draft_id}/approve`
- `POST /api/price-change-drafts/{draft_id}/reject`

Important backend modules:

- `api/app/price_changes/gmail_service.py`: Gmail REST access through BTP
  Destination.
- `api/app/price_changes/btp_destination.py`: Destination service binding and
  runtime destination resolution.
- `api/app/price_changes/extraction_service.py`: email and attachment
  extraction entry point.
- `api/app/price_changes/batch_resolution.py`: batch draft generation flow.
- `api/app/price_changes/repositories.py`: HANA persistence.
- `api/app/price_changes/s4_lookup.py`: S/4 lookup access.
- `api/app/price_changes/s4_price_write.py`: S/4 approval write path.
- `api/app/price_changes/settings.py`: environment-driven runtime settings.

## Frontend UI

The UI lives in `ui/` and is a Vite application. `ui/src/services/api.js` reads:

- `VITE_API_BASE_URL`
- `VITE_API_KEY`

Routes are defined in `ui/src/routes.js`:

- `home`: incoming requests and draft review.
- `past`: approved/rejected history.

Common commands:

```sh
cd ui
npm install
npm run dev
npm run build
```

## Local setup

Create environment files:

```sh
cp api/.env.example api/.env
cp ui/.env.example ui/.env
```

Set at least:

- `API_KEY` in `api/.env` and `VITE_API_KEY` in `ui/.env`.
- `HANA_*` values for SAP HANA.
- `AICORE_*` values for SAP GenAI Hub.
- `GMAIL_*` and `DESTINATION_SERVICE_NAME` for Gmail ingestion.
- `S4_*`, `DESTINATION_*`, and `CONNECTIVITY_*` values for S/4 access.


Run the API without local CF service binding:

```sh
cd api
source .venv/bin/activate
uvicorn app.main:app --reload
```

Run the UI:

```sh
cd ui
npm install
npm run dev
```

## Local BTP Destination binding

For Gmail destination access in local development, create an `api/.cdsrc-private.json`
file with `cds bind`. The file is local/private and must not be committed.

Prerequisites:

- You are logged in to Cloud Foundry with `cf login`.
- The Destination service instance exists. In the current repo config its name is
  `email-price-classifier-destination`.
- A service key exists. The checked-in local example references
  `email-price-classifier-destination-key`.
- You run the command from `api/`, because the generated private file must live
  next to the backend app.

Generate or update the private binding file:

```sh
cd api
cds bind --to email-price-classifier-destination
```

If you need to target a specific service key explicitly:

```sh
cd api
cds bind --to email-price-classifier-destination:email-price-classifier-destination-key
```

The generated file should look like this shape, with your CF endpoint, org,
space, service instance, and key:

```json
{
  "requires": {
    "[hybrid]": {
      "destinations": {
        "binding": {
          "type": "cf",
          "apiEndpoint": "https://api.cf.<region>.hana.ondemand.com",
          "org": "<cf-org>",
          "space": "<cf-space>",
          "instance": "email-price-classifier-destination",
          "key": "email-price-classifier-destination-key"
        },
        "kind": "destinations",
        "vcap": {
          "name": "destinations"
        }
      }
    }
  }
}
```

Run the API with the bound `VCAP_SERVICES` environment:

```sh
cd api
source .venv/bin/activate
cds bind --exec -- uvicorn app.main:app --reload
```

`cds bind --exec` injects the resolved Cloud Foundry service binding as
`VCAP_SERVICES` for the command. The backend then uses
`DESTINATION_SERVICE_NAME`, `GMAIL_DESTINATION_NAME`, `GMAIL_DESTINATION_LEVEL`,
and `GMAIL_REFRESH_TOKEN_PROPERTY` to resolve the Gmail runtime destination.

## Cloud Foundry deployment

Current manifest apps:

- `email-price-classifier-api`
- `email-price-classifier-ui`

Current manifest service binding:

- `email-price-classifier-destination`

Current manifest routes:

- `email-price-classifier-api.cfapps.eu10-005.hana.ondemand.com`
- `email-price-classifier.cfapps.eu10-005.hana.ondemand.com`

Manual helper:

```sh
./deploy.sh
```

Optional overrides:

```sh
FETCH_MAX_DAYS=14 ./deploy.sh
PRICE_CHANGE_AGENT_REASONING_EFFORT=medium ./deploy.sh
CF_TRACE=true ./deploy.sh
```

`deploy.sh` expects `api/.env` to contain all required runtime values. It checks
that the active CF API endpoint is `https://api.cf.eu10-005.hana.ondemand.com`,
builds the UI with the generated API key, runs `cf push` with manifest
variables, applies backend secrets with `cf set-env`, binds Cloud Logging, and
restarts the API app.

## Tests

Backend tests:

```sh
pytest tests
```

UI tests:

The UI has helper tests under `ui/tests/`, but `ui/package.json` does not
currently define a test script.

## Known limitations

- Gmail ingestion depends on a valid BTP Destination with a working Google
  refresh token stored in the configured destination additional property.
- S/4 approval writes require the configured S/4 destination/connectivity path
  or direct S/4 credentials to be available.
- The API still uses a shared `X-API-Key`; there is no per-user identity or role
  model yet.
