## AI Capability Matcher API (FastAPI)

This service exposes a FastAPI backend that matches client product rows against an AI catalog using embeddings, SAP HANA native vector search, and optional LLM-based ranking/reasons.

### Features
- Health check at `/api/health`
- Matching endpoint at `/api/match` that:
  - Builds embeddings for AI catalog and client rows
  - Stores AI vectors in SAP HANA temporary vector table
  - Finds nearest neighbors via `COSINE_SIMILARITY`
  - Optionally ranks candidates and generates reasons via SAP Gen AI Hub

### Project layout
```
resources/api/
  app/
    main.py                # FastAPI app, CORS, router include
    routers/match.py       # /api/match implementation
    models/common.py       # Health and error models
    models/match.py        # Request/response schemas for matching
    utils/hana_vectors.py  # HANA vector table helpers
    utils/grounding_utils.py # (optional) AICore helpers
    security.py            # API key validation (optional)
  requirements.txt
```

### Prerequisites
- Python 3.10+
- Access to SAP HANA (for real vector search). For local/dev, the service can still run but matching will fallback only for embeddings; HANA is required for vector search.
- Optional: SAP Gen AI Hub access for embeddings and LLM. Without credentials, the service falls back to deterministic random embeddings and passthrough ranking.

### Environment variables
Create a `.env` file in `resources/api/` by renaming `.env.example` to `.env` or export these variables in your environment.

- CORS/UI
  - `ALLOWED_ORIGIN` (optional): Production UI origin to allow via CORS
  - `APP_ENV` (optional): Set to `production` to restrict CORS to `ALLOWED_ORIGIN`
- API security
  - `API_KEY` (optional): If set, clients should send header `X-API-Key: <value>`
- Server
  - `PORT` (optional): Port to bind (default 8000)
- SAP HANA connection (required for vector search)
  - `HANA_ADDRESS`
  - `HANA_PORT`
  - `HANA_USER`
  - `HANA_PASSWORD`
- Gen AI Hub (optional: embeddings/LLM)
  - The code uses `gen_ai_hub.proxy.native.openai` interface if available. Configure per your environment/client.
- AICore (optional: used by `utils/grounding_utils.py`)
  - `AICORE_AUTH_URL`, `AICORE_CLIENT_ID`, `AICORE_CLIENT_SECRET`, `AICORE_BASE_URL`

### Install and run (local)
Use a virtual environment. From the repo root:

```bash
cd resources/api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl -s http://localhost:8000/api/health | jq .
```

### API reference

#### GET /api/health
Health probe.

Response 200:
```json
{
  "status": "healthy",
  "timestamp": 1700000000.0,
  "service": "api",
  "version": null
}
```

#### POST /api/match
Match client data rows to AI catalog rows. See `app/models/match.py` for full schema.

Request body fields:
- `ai_rows` (List[object]): AI catalog rows as dictionaries
- `client_rows` (List[object]): Client rows as dictionaries
- `selected_ai_columns` (List[str]): Columns from AI catalog to build text
- `selected_client_columns` (List[str]): Columns from client data to build text
- `matching_column` (str, optional): Column from AI catalog to display as candidate name
- `num_matches` (int, default 5, 1..10): Matches per client row
- `batch_size` (int, default 5, 1..10): Processed together for LLM calls
- `use_llm` (bool, default true): Call LLM to rank and explain
- `batch_system_prompt` (str, optional): Custom system prompt for LLM

Response 200 schema (simplified):
```json
{
  "success": true,
  "message": "Matching completed",
  "model": "gpt-4o",
  "result_columns": ["LLM Match 1", "LLM Reasoning 1", ...],
  "matches": [
    { "results": { "LLM Match 1": "...", "LLM Reasoning 1": "...", ... } },
    ...
  ]
}
```

Notes:
- The service computes embeddings for both datasets in one operation to guarantee dimensional consistency.
- AI vectors are inserted into a temporary HANA table; the table is dropped after the request.
- If `use_llm` is false or LLM is unavailable, nearest-neighbor names are returned with a generic reason.

### Example requests

Minimal example payload:
```json
{
  "ai_rows": [{"id": 1, "name": "Product A", "desc": "Foundation model"}],
  "client_rows": [{"id": "c1", "name": "Client Prod Alpha", "desc": "GenAI service"}],
  "selected_ai_columns": ["name", "desc"],
  "selected_client_columns": ["name", "desc"],
  "matching_column": "name",
  "num_matches": 3,
  "batch_size": 1,
  "use_llm": true
}
```

curl:
```bash
API=http://localhost:8000
PAYLOAD='{"ai_rows":[{"id":1,"name":"Product A","desc":"Foundation model"}],"client_rows":[{"id":"c1","name":"Client Prod Alpha","desc":"GenAI service"}],"selected_ai_columns":["name","desc"],"selected_client_columns":["name","desc"],"matching_column":"name","num_matches":3,"batch_size":1,"use_llm":true}'
curl -s -X POST "$API/api/match" \
  -H "Content-Type: application/json" \
  ${API_KEY:+-H "X-API-Key: $API_KEY"} \
  -d "$PAYLOAD" | jq .
```

Python (requests):
```python
import os, requests
API = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY")
payload = {
  "ai_rows": [{"id": 1, "name": "Product A", "desc": "Foundation model"}],
  "client_rows": [{"id": "c1", "name": "Client Prod Alpha", "desc": "GenAI service"}],
  "selected_ai_columns": ["name", "desc"],
  "selected_client_columns": ["name", "desc"],
  "matching_column": "name",
  "num_matches": 3,
  "batch_size": 1,
  "use_llm": True,
}
headers = {"Content-Type": "application/json"}
if API_KEY:
    headers["X-API-Key"] = API_KEY
r = requests.post(f"{API}/api/match", json=payload, headers=headers, timeout=(10, 900))
r.raise_for_status()
print(r.json())
```

### Deployment notes
- Bind to `PORT` on `0.0.0.0` (configured in `app/main.py` and typical `uvicorn` command)
- Ensure HANA connectivity from the runtime environment
- Configure CORS via `ALLOWED_ORIGIN` in production
- Provide `API_KEY` if you require authenticated access


