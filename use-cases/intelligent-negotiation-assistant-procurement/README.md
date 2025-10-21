## Template: UI and API Structure

This directory contains the deployable application scaffold. It is divided into a Streamlit UI and a FastAPI backend, plus deployment assets for SAP BTP (Cloud Foundry).

- **UI:** Streamlit app in `ui/`
- **API:** FastAPI service in `api/`
- **Deployment:** `manifest.yaml` and `deploy.sh`

### UI (`ui/`)
- `streamlit_app.py`: Entry point that composes pages from `src/pages` and shared components.
- `src/`: UI logic and helpers
  - `api_client.py`: HTTP client used by the UI to call the backend
  - `data_loader.py`: Data access utilities and caching
  - `utils.py`: UI helpers and formatting
  - `Offers_Comparison.py`: Core comparison view logic
  - `pages/`: Streamlit multi-page views
    - `1_Detailed_Comparison.py`, `3_Detailed_Dashboard.py`, `4_Part_Comparison.py`, `5_Cost_Breakdown.py`, `6_Risk_Assessment.py`
- `static/`: Assets served by the UI
  - `images/`, `font/`, `styles/`
- `requirements.txt`: Python dependencies for the UI

How UI talks to the API:
- Uses `src/api_client.py` to call backend endpoints for analyses, grounding, and knowledge graph operations.
- Reads `API_KEY` and backend base URL from environment variables (`ui/.env`).

### API (`api/`)
- `app/main.py`: FastAPI application entry point and router mounting
- `app/routers/`: HTTP endpoints
  - `analysis.py`, `chat.py`, `grounding.py`, `kg.py`, `example_router.py`
- `app/services/`: Business logic
  - `analysis_service.py`, `chat_service.py`, `kg_service.py`
- `app/models/`: Pydantic schemas and domain models
- `app/core/`: Core modules
  - `kg_creation/`: Extraction, chunking, unification, validation, serialization
  - `llm/`: LLM factories/config/prompts
  - `structured_llm_analyzers/`: Structured analyzers (parts, cost, risk, TQDCS)
- `app/utils/`: Shared utilities (e.g., grounding)
- `app/data/`: Local caches and KG data
- `requirements.txt`: Python dependencies for the API

Security and config:
- API expects `API_KEY` for simple key-based auth (`app/security.py`).
- AI Core and model credentials configured via `api/.env`.

### Local Development
Set up environment files from examples and run UI and API in separate terminals.

UI:
```bash
cd ui
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python streamlit_app.py
```

API:
```bash
cd api
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Deployment
- Update names in `manifest.yaml` as needed.
- Run `./deploy.sh` to deploy UI and API to Cloud Foundry with a generated API key, or use `cf push --var api_key=...` to deploy manually.
