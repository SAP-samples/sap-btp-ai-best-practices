# API Reference — ai-core-video Backend

Base URL (runtime-dependent):
- Local: `http://localhost:5000`
- Cloud Foundry: `https://ai-core-video-backend.cfapps.eu10-004.hana.ondemand.com`
- Optional override (frontend): add meta tag  
  `<meta name="backend-base-url" content="https://ai-core-video.cfapps.eu10-004.hana.ondemand.com/api">`

Authentication:
- If `API_KEY` is set on the backend, clients must send `X-API-Key: <key>` header on protected endpoints.

CORS:
- Controlled by `ALLOWED_ORIGIN` env var (e.g., `https://ai-core-video.cfapps.eu10-004.hana.ondemand.com`).

## Health

GET `/health`

Example:
```bash
curl -s https://ai-core-video-backend.cfapps.eu10-004.hana.ondemand.com/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-12-05T10:00:00.000000",
  "service": "Video Incident Monitoring OData Service"
}
```

## OData Service Document

GET `/odata/v4/VideoIncidentService/`  
HEAD `/odata/v4/VideoIncidentService/`

Headers (if API_KEY is enabled):
- `X-API-Key: <key>`

Response (example):
```json
{
  "@odata.context": "$metadata",
  "value": [
    { "name": "MediaAnalysis", "kind": "EntitySet", "url": "MediaAnalysis" }
  ]
}
```

## OData Metadata

GET `/odata/v4/VideoIncidentService/$metadata`

Headers (if API_KEY is enabled):
- `X-API-Key: <key>`

Response:
- XML content of `metadata.xml`

## MediaAnalysis — Collection

GET `/odata/v4/VideoIncidentService/MediaAnalysis`

Headers (if API_KEY is enabled):
- `X-API-Key: <key>`

Response:
```json
{
  "@odata.context": "$metadata#MediaAnalysis",
  "value": [ /* array of analysis entities */ ]
}
```

## MediaAnalysis — Single Entity

GET `/odata/v4/VideoIncidentService/MediaAnalysis({id})`

Headers (if API_KEY is enabled):
- `X-API-Key: <key>`

Response:
```json
{
  "@odata.context": "$metadata#MediaAnalysis/$entity",
  "ID": "550e8400-e29b-41d4-a716-446655440001",
  "fileName": "example.mp4",
  "fileType": "video",
  "mimeType": "video/mp4",
  "fileSize": 123456,
  "filePath": "...",
  "instruction": "Analyze...",
  "temperature": 0.7,
  "maxTokens": 2000,
  "status": "completed",
  "analysisResult": "text",
  "incidentDetected": true,
  "severity": "high",
  "promptTokens": 1000,
  "completionTokens": 500,
  "totalTokens": 1500,
  "processingTime": 42,
  "analyzedAt": "2025-12-05T10:00:00.000000",
  "analyzedBy": "System"
}
```

## Create and Analyze Media

POST `/odata/v4/VideoIncidentService/MediaAnalysis`

Headers:
- `X-API-Key: <key>` (if enabled)
- `Content-Type: multipart/form-data`

Form fields:
- `file` (required): media file (video/audio)
- `instruction` (optional): text instruction
- `temperature` (optional, default `0.7`): model temperature (0–2)
- `maxTokens` (optional, default `2000`): max output tokens
- `autoAnalyze` (optional): `"true"` to run analysis immediately

Example:
```bash
curl -s -X POST \
  -H "X-API-Key: <key>" \
  -F "file=@Video/14_Team000884-xx_xx_xxxx - walking along the tube.mp4" \
  -F "instruction=Analyze this media for safety incidents" \
  -F "temperature=0.7" \
  -F "maxTokens=2000" \
  -F "autoAnalyze=true" \
  https://ai-core-video-backend.cfapps.eu10-004.hana.ondemand.com/odata/v4/VideoIncidentService/MediaAnalysis
```

Response:
- The created `MediaAnalysis` entity (status `pending` or `completed` depending on `autoAnalyze`).

## Analyze Existing Media

POST `/odata/v4/VideoIncidentService/MediaAnalysis({id})/analyze`

Headers:
- `X-API-Key: <key>` (if enabled)

Response:
- The updated `MediaAnalysis` entity with `analysisResult`, `incidentDetected`, `severity`, and token/processing metrics.

## Delete Media Analysis

DELETE `/odata/v4/VideoIncidentService/MediaAnalysis({id})`

Headers:
- `X-API-Key: <key>` (if enabled)

Response:
- 204 No Content on success

## Error Handling

Common errors:
- `401 Unauthorized` — Missing or invalid API key
- `400 Invalid UUID format` — Malformed entity ID
- `404 Not Found` — Metadata or entity/file not found
- `500 Internal Server Error` — Unexpected error during processing/inference

## Notes

- File storage on CF instances is ephemeral; for production, use persistent storage (e.g., object storage).
- CORS should be constrained to your frontend route in `ALLOWED_ORIGIN`.
- AI Core credentials must be managed securely via CF environment variables (`cf set-env`) or secrets managers.
