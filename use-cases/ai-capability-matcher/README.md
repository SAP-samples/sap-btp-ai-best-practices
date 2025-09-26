# AI Capability Matcher

Streamlit UI + FastAPI backend to match a client's product list against an AI product catalog. The app builds text from selected columns, computes embeddings, retrieves nearest candidates with SAP HANA vector search, and optionally ranks them with an LLM to provide concise reasoning. Upload two CSVs, monitor progress as batches run, and download a results CSV.

## Overview

At a high level, the workflow is:

1. Upload AI catalog CSV and client CSV in the UI.
2. Select the columns that best describe each item on both sides.
3. The application generates embeddings for both datasets (via SAP Gen AI Hub), stores AI vectors in a temporary SAP HANA vector table, and retrieves the best matches.
4. Optionally, an LLM ranks the candidates and generates a brief reason per match.
5. The UI shows progress per batch and provides a results CSV for download.

## Features

- Upload two CSVs and preview their heads
- Choose columns for text construction (frontend and backend stay aligned)
- Adjustable matches per row (1–10) and batch size
- Optional custom system prompt for LLM ranking
- Progress indicator for long-running jobs and CSV download
- Backend supports API-key protection and production CORS
- Works without LLM/HANA credentials for local dev (best-effort fallbacks)

## Project layout
```
resources/
  api/                      # FastAPI backend
    app/
      main.py               # App entry; includes routers and CORS
      routers/match.py      # /api/match implementation
      models/               # Pydantic schemas
      utils/                # HANA vector + optional helpers
    requirements.txt
    README.md
  ui/                       # Streamlit UI
    src/
      Home.py               # Landing page
      pages/Capability_Matcher.py   # Main comparator page
      api_client.py         # HTTP client to backend
      utils.py              # CSS loader
    static/                 # Branding and styles
    requirements.txt
    streamlit_app.py        # Optional entry helper
    README.md
data/
  ai_catalog.csv            # Example data
  client_catalog.csv        # Example data
```

## Quick start (local)

Run the backend and UI in separate shells, each with its own virtual environment.

### 1) Start the API
```bash
cd resources/api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2) Start the UI
```bash
cd resources/ui
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export API_BASE_URL=${API_BASE_URL:-http://localhost:8000}
streamlit run src/Home.py --server.port 8501 --server.address 0.0.0.0
```

Open `http://localhost:8501` and navigate to the `Comparator` page from the sidebar.


## Configuration

Backend (`resources/api`):
- CORS/UI: `ALLOWED_ORIGIN`, `APP_ENV`
- Security: `API_KEY` (clients send header `X-API-Key`)
- Server: `PORT` (default 8000)
- SAP HANA: `HANA_ADDRESS`, `HANA_PORT`, `HANA_USER`, `HANA_PASSWORD`
- Gen AI Hub (optional for embeddings/LLM): environment consistent with `gen_ai_hub.proxy.native.openai`

UI (`resources/ui`):
- `API_BASE_URL` (default `http://localhost:8000`)
- `API_KEY` (optional, sent as `X-API-Key`)
- Timeouts (optional): `API_TIMEOUT_SECONDS`, `API_CONNECT_TIMEOUT_SECONDS`, `API_READ_TIMEOUT_SECONDS`

## Usage

1. On the `Capability_Matcher` page, upload the AI Catalog CSV and Client CSV. Use the data examples in `data/ai_catalog.csv` and `data/client_catalog.csv`.
2. Select the columns to build text for each side, and choose a display column for candidate naming.
3. Configure matches per row, batch size, and whether to enable LLM ranking. Optionally set a custom batch system prompt.
4. Click `Run Matching`. Watch progress and preview output; download the final CSV when complete.

## Notes

- Example CSVs are provided under `data/` to get started quickly.
- For detailed backend and UI instructions, see `resources/api/README.md` and `resources/ui/README.md`.

