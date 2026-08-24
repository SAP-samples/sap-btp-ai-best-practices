# SAP Document AI + GenAI Hub — Frontend

SAP Fiori frontend application for the Invoice Extraction Pipeline.

## Tech Stack

- **Vite** — Build tool & dev server
- **TypeScript** — Type-safe development
- **SAP UI5 Web Components v2** — Official SAP Fiori components
- **SAP Horizon Theme** — Latest SAP visual design system
- **Axios** — HTTP client for backend API

## Project Structure

```
frontend/
├── src/
│   ├── api/
│   │   ├── client.ts       # Centralized API client (axios)
│   │   └── types.ts        # TypeScript interfaces for all API responses
│   ├── components/
│   │   ├── app.ts          # Main App class (render + event binding + execution)
│   │   └── app-render.ts   # Auxiliary render functions (schemas, templates, eval)
│   ├── utils/
│   │   └── formatters.ts   # Confidence, score, HTML escape utilities
│   ├── main.ts             # Entry point — UI5 imports + Horizon theme + App init
│   └── vite-env.d.ts       # TypeScript declarations for Vite env + UI5 modules
├── index.html              # App shell with SAP Fiori CSS variables
├── vite.config.ts          # Vite config with API proxy
├── tsconfig.json           # TypeScript config
├── package.json
├── .env                    # Local environment variables
└── .env.example            # Environment variables template
```

## Setup

### 1. Install dependencies

```bash
cd frontend
npm install
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set VITE_API_BASE_URL to your backend URL
```

Default: `VITE_API_BASE_URL=http://localhost:8000`

### 3. Start backend

```bash
cd ../backend
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

### 4. Start frontend dev server

```bash
cd frontend
npm run dev
```

Open: http://localhost:3000

## Build for Production

```bash
npm run build
npm run preview
```

## API Endpoints Used

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API health check |
| GET | `/api/v1/schemas` | List SAP DocAI schemas |
| GET | `/api/v1/templates` | List SAP DocAI templates |
| POST | `/api/v1/invoice/process` | Process invoice with SAP DocAI |
| POST | `/api/v1/genai/pipeline` | Full GenAI pipeline (SAP + LLM1 + LLM2 + comparison) |
| POST | `/api/v1/evaluation/run` | Run quality evaluation |
| GET | `/api/v1/output/genai` | List GenAI output files |
| GET | `/api/v1/output/genai/{filename}` | Get specific GenAI output file |
| GET | `/api/v1/output/evaluation` | List evaluation output files |

## Features

- **SAP ShellBar** with logo and API health indicator
- **Scenario Selection** — 5 pipeline modes with radio buttons
- **File Upload** — Drag & drop or click to browse (PDF, JPG, PNG, TIFF)
- **Processing Steps** — Visual step indicators with busy states
- **Results Dashboard** — Tabbed view with:
  - SAP Document AI extraction (fields + confidence + line items)
  - LLM Technique 1 (free prompting) results
  - LLM Technique 2 (structured JSON) results
  - Side-by-side comparison with conflict detection
  - Executive summary
- **Quality Evaluation** — KPI cards, metrics table, AI assessment
- **Toast Notifications** — Success/error feedback
- **Responsive Layout** — Works on desktop and tablet
- **SAP Horizon Theme** — Official SAP visual design
