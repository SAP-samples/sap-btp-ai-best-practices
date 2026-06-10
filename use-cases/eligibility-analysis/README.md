## Eligibility Analysis

This repository provides an invoice eligibility analysis service with a UI for uploading offer files and reviewing results. The core logic lives in the FastAPI backend under `api/`, while the UI in `ui/` allows users to set eligibility parameters, upload Excel files, and review funded versus non-funded invoices.

## Eligibility Rules

Eligibility checks are implemented in `api/app/services/eligibility/rules.py` and applied in order. Each rule returns a specific rejection reason when it fails.

- R13: Invoice must not be overdue. `due_date` must be strictly after `purchase_date`.
- R1: Due date must be at least `NDDT` days after `purchase_date`.
- R16: Tenor must be less than `TEIH` days. `due_date - issuance_date < TEIH`.
- R17: Issuance must be at least `ISSPUR` days before `purchase_date`.
- R11: Currency must be in the eligible list.
- R2: Invoice reference must be unique per seller within the batch.

These rules are combined per invoice. An invoice is eligible only if it passes all rules.

## Rule Parameters and Defaults

Rule thresholds are configurable by API query parameters and environment variables. Query parameters take precedence for a single request.

Defaults are defined in `api/app/config/eligibility_config.py`:

- `ELIGIBILITY_NDDT` (default `6`)
- `ELIGIBILITY_TEIH` (default `15`)
- `ELIGIBILITY_ISSPUR` (default `0`)
- `ELIGIBILITY_CURRENCIES` (default `EUR,USD`)

The UI configuration panel on the Eligibility page maps directly to these parameters:

- Purchase Date
- NDDT (minimum days to due date)
- TEIH (maximum tenor days)
- ISSPUR (minimum days since issuance)
- Eligible Currencies (comma-separated)

## How Eligibility Works End-to-End

1. The UI uploads an Excel offer file to the backend.
2. The backend parses the offer file into `OfferInvoice` objects.
3. The `EligibilityEngine` applies each rule to every invoice.
4. Results are grouped into funded and non-funded invoices.
5. A summary Excel file is generated for download.
6. Historical results are stored for seller statistics and rejection breakdowns.

## API Endpoints (Eligibility)

All endpoints are protected by the API key dependency.

- `POST /api/eligibility/analyze`
  - Uploads an Excel file and applies all rules.
  - Optional query parameters: `purchase_date`, `nddt`, `teih`, `isspur`, `eligible_currencies`.
- `GET /api/eligibility/config`
  - Returns the default rule configuration used by the UI.
- `GET /api/eligibility/download/{filename}`
  - Downloads the generated summary Excel file.
- `GET /api/eligibility/seller/{seller_id}/summary`
  - Seller-level eligibility statistics and rejection breakdown.
- `GET /api/eligibility/seller/{seller_id}/history`
  - Paginated processing history for a seller.

## A2A Agent Endpoint

The backend exposes an A2A JSON-RPC endpoint for eligibility questions and customer log analysis:

- `POST /api/a2a` (requires `X-API-Key`)
- `GET /api/a2a/.well-known/agent-card.json`

Example request:

```bash
curl -s http://127.0.0.1:8000/api/a2a \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "parts": [{ "kind": "text", "text": "Why was invoice INV-100 rejected?" }],
        "messageId": "msg-1"
      },
      "configuration": { "historyLength": 4 }
    }
  }'
```

## Local Development

### Backend (FastAPI)

1. Create and activate a virtual environment.
2. Install dependencies from `api/requirements.txt`.
3. Run the server.

```bash
cd api
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API documentation is available at `http://127.0.0.1:8000/docs`.

### Frontend (UI)

```bash
cd ui
npm install
npm run dev
```

Open the UI and navigate to the Eligibility page to upload a file and run the analysis.

## Deployment

Deployment uses `manifest.yaml` for Cloud Foundry. If you need a unique app name, replace the placeholder values in `manifest.yaml`.
