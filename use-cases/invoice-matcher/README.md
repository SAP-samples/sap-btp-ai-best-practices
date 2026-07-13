# Invoice-Payment Matcher

AI-powered invoice-to-payment matching application. Matches bank payments to customer invoices using rule-based matching and LLM-powered name resolution, designed for Japanese payment reconciliation.

- **Frontend:** [UI5 Web Components](https://sap.github.io/ui5-webcomponents/) (plain JavaScript + Vite)
- **Backend:** [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **AI:** SAP AI Core (generative-ai-hub-sdk)

## Project Structure

```
invoice-matcher/
├── api/             # FastAPI backend (AI matching service)
│   ├── app/
│   │   ├── main.py
│   │   ├── routers/matching.py
│   │   ├── services/ai_matching_service.py
│   │   ├── services/csv_parser.py
│   │   └── models/
│   └── requirements.txt
├── ui/              # UI5 Web Components frontend
│   ├── src/
│   │   ├── pages/invoice-matcher/
│   │   ├── worker.js
│   │   └── services/api.js
│   └── package.json
├── manifest.yaml    # Cloud Foundry deployment
└── deploy.sh        # Deployment script
```

## How It Works

1. **Rule-based matching** (client-side Web Worker): Matches invoices to payments by amount (within tolerance) + invoice number found in payment text fields.
2. **AI matching** (server-side): For unmatched invoices, uses LLM to match payer names to customer names accounting for Japanese romanization variants, abbreviations, and field-width splits.

## Local Development

### 1. Backend

```bash
cd api
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Fill in SAP AI Core credentials
uvicorn app.main:app --reload
```

API docs at http://127.0.0.1:8000/docs

### 2. Frontend

```bash
cd ui
npm install
cp .env.example .env
npm run dev
```

Opens at http://localhost:5173

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for full step-by-step instructions.

## Matching Logic

See [MATCHING_PROCESS.md](MATCHING_PROCESS.md) for detailed explanation of the matching algorithm.
