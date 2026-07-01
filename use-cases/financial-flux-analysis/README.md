# Financial Flux Analysis

AI-powered financial variance analysis platform built on **SAP BTP**. The system ingests financial data from SAP HANA Cloud, detects significant variances at the cost center / GL account / WBS level, classifies their drivers, and generates natural-language explanations using **SAP AI Core (GenAI Hub)**.

It exposes both a **REST API** (FastAPI) and a **SAP Fiori / UI5** frontend with an interactive chat assistant.

---

## 🏗️ Architecture

```
┌────────────────────────┐        ┌────────────────────────┐        ┌────────────────────────┐
│   SAP Fiori / UI5      │  HTTP  │    FastAPI Backend     │  SQL   │    SAP HANA Cloud      │
│   (Vite + UI5 1.124)   │ ─────► │      (Python 3.11)     │ ─────► │   (Datasphere schema)  │
│   Chat • Variances     │        │  Variance • LLM • Chat │        │   Financial datasets   │
└────────────────────────┘        └──────────┬─────────────┘        └────────────────────────┘
                                             │
                                             ▼
                                   ┌────────────────────┐
                                   │  SAP AI Core /     │
                                   │  GenAI Hub (LLM)   │
                                   └────────────────────┘
```

Both apps are deployed to **Cloud Foundry** (`cflinuxfs4` stack) via `manifest.yaml`.

---

## 📁 Repository Structure

```
.
├── App/
│   ├── backend/                 # FastAPI + Python 3.11 service
│   │   ├── api.py               # REST endpoints (variance, chat, WBS)
│   │   ├── app.py               # Application entry / wiring
│   │   ├── database/            # HANA connection layer
│   │   ├── loaders/             # Raw data loaders
│   │   ├── normalizers/         # ID & text normalization
│   │   ├── enrichers/           # Dataset enrichment
│   │   ├── validators/          # Data validation
│   │   ├── services/            # Financial processing, hierarchies
│   │   ├── src/                 # Variance detection, drilldown, LLM
│   │   │   └── chat/            # Conversational AI module
│   │   ├── utils/               # Currency parser, helpers
│   │   └── requirements.txt
│   │
│   └── fiori/                   # SAP UI5 / Fiori frontend
│       ├── webapp/              # UI5 app (controllers, views, fragments)
│       ├── nginx.conf
│       └── package.json
```

---

## ✨ Key Features

- **Variance detection** at multiple aggregation levels (Cost Center, GL Account, Profit Center, WBS).
- **Driver classification** to explain the *why* behind significant deviations.
- **Drill-down analysis** from totals down to individual postings.
- **LLM-based explanations** powered by SAP AI Core (GenAI Hub).
- **Interactive chat assistant** with context awareness over the financial datasets.
- **SAP HANA Cloud** native integration (Datasphere schema).
- **Fiori 3 / Horizon** UI with responsive layout.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** (for the Fiori frontend)
- **SAP HANA Cloud** instance with the financial schema
- **SAP AI Core** service key (Client ID / Secret)
- **Cloud Foundry CLI** (for deployment)

---

### 1. Backend — local setup

```bash
cd App/backend

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment (copy and edit)
cp manifest.yaml.example manifest.yaml   # for CF deploy
# OR for local dev, create a .env file:
cat > .env <<EOF
HANA_ADDRESS=<your-hana-host>.hanacloud.ondemand.com
HANA_PORT=443
HANA_USER=<your-user>
HANA_PASSWORD=<your-password>
HANA_ENCRYPT=True
HANA_SCHEMA=<your-schema>

AICORE_AUTH_URL=https://<tenant>.authentication.<region>.hana.ondemand.com
AICORE_CLIENT_ID=<client-id>
AICORE_CLIENT_SECRET=<client-secret>
AICORE_BASE_URL=https://api.ai.prod.<region>.aws.ml.hana.ondemand.com/v2
AICORE_RESOURCE_GROUP=default
EOF

# Run the API
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

---

### 2. Frontend — local setup

```bash
cd App/fiori

# Install dependencies
npm install

# Start Vite dev server
npm run dev
```

The UI5 app will be available at `http://localhost:5173` (or the port Vite reports).

> Make sure the backend is running and that the `apiService.js` base URL points to it.

---

## ☁️ Cloud Foundry Deployment

### Backend

```bash
cd App/backend

# Create your manifest.yaml from the template (NEVER commit it with real secrets)
cp manifest.yaml.example manifest.yaml
# Edit manifest.yaml and fill in your real credentials

cf push
```

### Frontend

```bash
cd App/fiori
npm run build         # produces ./dist
cf push
```

---

## 🔐 Security & Secrets

This repository **does not contain real credentials**. The following files are git-ignored and must be created locally:

| File                              | Purpose                                  |
|-----------------------------------|------------------------------------------|
| `App/backend/.env`                | Local development environment variables  |
| `App/backend/manifest.yaml`       | Cloud Foundry manifest with real secrets |

Use the provided templates as a starting point:

- `App/backend/manifest.yaml.example`

> ⚠️ **Never commit real HANA passwords, AI Core client secrets, or any other credentials.** The `.gitignore` files at the repo root and inside `App/backend/` are configured to prevent this.

---

## 📚 Additional Documentation

- [`App/README.md`](./App/README.md) — Detailed application architecture and API reference.
- [`App/backend/README_API.md`](./App/backend/README_API.md) — REST API endpoints and contracts.
- [`App/backend/SYSTEM_DOCUMENTATION.md`](./App/backend/SYSTEM_DOCUMENTATION.md) — System-level technical documentation.
- [`App/fiori/webapp/annotations/README_ANNOTATIONS.md`](./App/fiori/webapp/annotations/README_ANNOTATIONS.md) — Fiori annotations guide.

---

## 🛠️ Tech Stack

| Layer        | Technology                                      |
|--------------|-------------------------------------------------|
| Frontend     | SAP UI5 1.124 · Fiori 3 / Horizon · Vite        |
| Backend      | Python 3.11 · FastAPI · Uvicorn · Pandas        |
| Database     | SAP HANA Cloud (via Datasphere)                 |
| AI / LLM     | SAP AI Core · GenAI Hub                         |
| Deployment   | SAP BTP · Cloud Foundry (`cflinuxfs4`)          |

---

## 📄 License

Proprietary — SAP BTP AI Services CoE. All rights reserved.