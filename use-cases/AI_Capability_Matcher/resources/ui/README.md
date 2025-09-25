## AI Capability Matcher UI (Streamlit)

Streamlit UI to upload two CSVs (AI catalog and client data), configure which columns to use, and call the backend API to get ranked matches and reasoning.

### Features
- Upload CSVs and preview heads
- Column selection for text construction (mirrors backend logic)
- Adjustable matches per row, batch size, and optional custom LLM prompt
- Progress bar for long jobs and CSV download of results

### Project layout
```
resources/ui/
  streamlit_app.py          # Entry point (helpful for CF or local run)
  src/
    Home.py                # Landing page
    pages/Capability_Matcher.py    # Main comparator page
    api_client.py          # HTTP client to call backend
    utils.py               # CSS loader
  static/
    styles/*.css, images/* # Branding and styling
  requirements.txt
```

### Prerequisites
- Python 3.10+
- A running backend API (see `resources/api/README.md`), default at `http://localhost:8000`

### Environment variables
Create a `.env` in `resources/ui/` by renaming `.env.example` into `.env` (optional):
- `API_BASE_URL` (default `http://localhost:8000`)
- `API_KEY` (optional) If the API enforces an API key, it will be sent as `X-API-Key`
- Timeouts (optional, seconds)
  - `API_TIMEOUT_SECONDS` (default 900)
  - `API_CONNECT_TIMEOUT_SECONDS` (default 10)
  - `API_READ_TIMEOUT_SECONDS` (default = `API_TIMEOUT_SECONDS`)

### Install and run (local)
Use a virtual environment in this folder:

```bash
cd resources/ui
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export API_BASE_URL=${API_BASE_URL:-http://localhost:8000}
streamlit run src/Home.py --server.port 8501 --server.address 0.0.0.0
```

Alternatively, use the provided entry script (also useful on Cloud Foundry):

```bash
python streamlit_app.py
```

Then open `http://localhost:8501`.

### Usage
1. Open the app and go to the `Comparator` page from the sidebar.
2. Upload the AI Catalog CSV and Client CSV.
3. Select which columns to use to construct text for embeddings on each side.
4. Choose a display column from the AI catalog (shown as candidate name).
5. Configure matches per row, batch size, and whether to use LLM ranking.
6. Optionally provide a custom batch system prompt.
7. Click `Run Matching`.
8. Inspect the preview of results and download the CSV.

### Backend contract
The UI calls the backend endpoint `POST /api/match` with the payload assembled in `src/pages/Comparator.py`. See `resources/api/README.md` for full schema and examples. Timeouts for long-running batches are configured in `src/api_client.py`.

### Troubleshooting
- Cannot connect to API: ensure the backend runs on `API_BASE_URL` and CORS allows the origin (for browser-hosted UIs). The Streamlit app runs server-side requests, so CORS is usually not an issue.
- Request timed out: increase the timeout minutes in the UI or adjust `API_READ_TIMEOUT_SECONDS`.
- 403 from API: provide `API_KEY` if the backend requires it.


