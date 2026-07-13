# ai-core-video

AI-powered Video & Audio Incident Monitoring built with:
- Frontend: SAP UI5 (standalone Fiori-style page)
- Backend: Python FastAPI with OData-style endpoints
- AI: SAP AI Core calling Gemini 2.5 Pro

This app lets users upload or select media, configure analysis parameters, and receive incident detection results, severity classification, and usage metrics.

## Components

- Frontend (static): `video-incident-monitor/webapp/analyze-standalone.html`
  - Loads `config.js` to decide the backend base URL at runtime (localhost vs Cloud Foundry).
  - Sends requests to the backend endpoint, optionally including `X-API-Key` header if provided via meta tag.
- Backend (FastAPI): `backend_odata_service.py`
  - OData-style endpoints under `/odata/v4/VideoIncidentService/…`
  - Integrates with SAP AI Core (Gemini 2.5 Pro) via OAuth and deployment ID.
  - Saves uploaded files on the instance filesystem (ephemeral on Cloud Foundry).

## Runtime configuration

Frontend resolves backend URL via `config.js`:
- Local: `http://localhost:5000`
- Cloud Foundry: `https://ai-core-video-backend.cfapps.eu10-004.hana.ondemand.com`
- Optional override via meta tag:
  ```html
  <meta name="backend-base-url" content="https://ai-core-video.cfapps.eu10-004.hana.ondemand.com/api">
  <meta name="backend-api-key" content="YOUR_API_KEY">
  ```

Backend environment variables:
- `AICORE_AUTH_URL`, `AICORE_CLIENT_ID`, `AICORE_CLIENT_SECRET`, `AICORE_BASE_URL`, `AICORE_RESOURCE_GROUP`
- `API_KEY` (if set, backend requires `X-API-Key` header)
- `ALLOWED_ORIGIN` (controls CORS; set to your frontend route on CF)

## Cloud Foundry deployment

Two apps via `manifest.yml`:
- Frontend (staticfile buildpack) at `ai-core-video.cfapps.eu10-004.hana.ondemand.com`
- Backend (python buildpack) at `ai-core-video-backend.cfapps.eu10-004.hana.ondemand.com`

Use the provided `deploy.sh` to generate a secure API key and push:
```bash
./deploy.sh
```
It prints the generated API key and how to inject it in the frontend via meta tag.

## Local development

Backend:
```bash
pip install -r requirements.txt
uvicorn backend_odata_service:app --host 0.0.0.0 --port 5000
```

Frontend:
- Open `video-incident-monitor/webapp/analyze-standalone.html` directly, or
```bash
cd video-incident-monitor/webapp
npx serve -p 8080
# open http://localhost:8080/analyze-standalone.html
```

## Troubleshooting

- 401 Unauthorized: Ensure backend has `API_KEY` set and the frontend includes `X-API-Key` header (via meta tag).
- CORS errors: Check `ALLOWED_ORIGIN` matches your frontend route exactly (including scheme).
- 404 media files: Confirm static server serves `Video/` or `Audio/` paths under `webapp`.

## License

Internal prototype for AI CORE Video Incident Monitoring. Adapt for production as needed.
