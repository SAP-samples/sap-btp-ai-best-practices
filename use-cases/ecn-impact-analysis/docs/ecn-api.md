# ECN impact API

This page records the original MVP contract. For current summary generation,
see `docs/summary-regeneration.md`.

## Purpose

The API accepts source files, publishes deterministic ECN/BOM/rule evidence to HANA, schedules independent evidence-grounded analyses, and exposes stored ECN reads. Generation starts asynchronously after publication: a successful upload response means deterministic HANA state and queued jobs exist, not that every LLM result is complete.

## Authentication and startup

All `/api` business routes require `X-API-Key`; `/api/health` is public. Authentication uses a constant-time comparison against `API_KEY`. Public errors always use:

```json
{"code":"ERROR_CODE","message":"Safe message","details":null}
```

Local `.env` values load through `python-dotenv` before application creation. Do not shell-source `api/.env`: shell-significant characters in client secrets can be expanded or removed before Python starts. Startup validates or creates the seven HANA tables, builds one shared ingestion preparation gate, lifecycle mutation guard, provider, and generation runtime, and marks interrupted `QUEUED`/`RUNNING` jobs as `FAILED/PROCESS_RESTARTED`. Shutdown gathers generation tasks, closes the provider, then disposes the HANA engine even if an earlier cleanup fails.

## Inputs and outputs

The original MVP OpenAPI surface contained eight paths:

- `GET /api/health`
- `POST /api/uploads`
- `GET /api/ecns`
- `GET /api/ecns/{ecn_row_id}`
- `GET /api/ecns/{ecn_row_id}/evidence/{evidence_id}`
- `GET /api/analysis-progress`
- `POST /api/ecns/{ecn_row_id}/analysis/retry`
- `DELETE /api/history`

The original MVP API had no regeneration route. The current application uses
`POST /api/summary-batches` for explicit generation and regeneration; see
`docs/summary-regeneration.md` for its request and replacement behavior.

### Upload

`POST /api/uploads` consumes multipart form data:

- `request_id`: required client-generated UUID used for idempotency.
- `ecn_file`: required `.json` file with an accepted JSON media type.
- `bom_file`: optional `.xlsx` EBOM/MBOM workbook. The first successful upload requires it; later uploads may reuse the latest persisted snapshot.

Files are metadata-validated and streamed into bounded memory before parsing. A newly committed upload schedules the exact job IDs returned by publication. Replaying the same `request_id` returns the current representation with `created=false` and schedules nothing again. The `202` response includes upload/request/BOM identities, five mutually exclusive state counts, and stable list/progress URLs.

### Historical reads

- `GET /api/ecns` accepts `page`, `page_size`, `query`, `ecn_number`, `plant`, `analysis_state`, `has_warnings`, `sort`, and `order`. Each item carries `affected_plants` (see [ecn-list-plants-column.md](ecn-list-plants-column.md)).
- `GET /api/ecns/{ecn_row_id}` returns one exact immutable occurrence, source provenance, deterministic evidence groups, current job state, and any stored analysis. `linked_ecns` contains outgoing source-description references and available target row UUIDs; see [Linked ECN navigation](ecn-linked-navigation.md).
- `GET /api/ecns/{ecn_row_id}/evidence/{evidence_id}` returns evidence only after membership is checked inside that ECN's persisted bundle.
- `GET /api/ecns/{ecn_row_id}/drawing?number=...` returns the exact packaged sample PDF for an ECN member. Material details advertise availability as `engineering_drawing`; see [drawing-feasibility.md](drawing-feasibility.md).
- `GET /api/analysis-progress` returns global counts or accepts an optional `upload_id` UUID.

The sort allowlist is `uploaded_at`, `resolution_date`, `affected_plants`, `ecn_number`, `description`, `affected_item_count`, `ebom_match_count`, `mbom_match_count`, `warning_count`, and `analysis_state`; order is `asc` or `desc`. Evidence IDs are deterministic `E-` plus sixteen lowercase hexadecimal characters, not UUIDs. Every GET makes zero provider calls and is side-effect free, except that `GET /api/ecns` saves plant labels for page rows that were never assessed (an idempotent cache fill).

### Retry and reset

`POST /api/ecns/{ecn_row_id}/analysis/retry` permits only an existing failed job without a stored result. The committed job is scheduled through the same runtime and concurrency limit used by upload work. Missing ECNs return `404`; known non-retryable states return `409`.

`DELETE /api/history` requires this exact JSON body:

```json
{"confirmation":"RESET HISTORY"}
```

Reset returns `409` immediately while any job is queued or running. A successful reset deletes attempts, analyses, jobs, ECNs, uploads, BOM rows, and ECN/BOM source artifacts in dependency order. HANA table definitions and conditional-rule artifacts are preserved.

## Error behavior

- `400`: malformed or unacceptable source content and first-upload BOM omissions.
- `403`: missing or invalid API key.
- `404`: missing ECN occurrence or evidence membership.
- `409`: invalid state transition, active reset conflict, or publication race.
- `413`: configured upload byte limit exceeded.
- `422`: path, query, form, or JSON shape does not match the API contract.
- `500`: startup/configuration failure or an unexpected server error.

Validation details contain field locations and error types only; submitted values, source content, provider exception/refusal text, credentials, prompts, and evidence payloads are never echoed.

## Related code

- `api/app/main.py`: application factory, safe handlers, production lifespan, and route protection.
- `api/app/dependencies.py`: immutable application service container.
- `api/app/routers/uploads.py`: bounded multipart conversion and post-commit scheduling.
- `api/app/routers/ecns.py`: list, detail, and same-occurrence evidence reads.
- `api/app/routers/analysis_progress.py`: aggregate progress and failed-only retry.
- `api/app/routers/history.py`: exact-confirmation reset.
- `api/app/ecn/api_mapping.py`: pure field-complete workflow response mapping.
- `api/app/ecn/services/retry.py`: cancellation-safe guarded retry transition.

## Testing

Run the focused transport, mapping, routing, and lifespan suite:

```bash
PYTHONPATH=api python -m pytest -q \
  tests/unit/test_api_mapping.py \
  tests/unit/test_upload_api.py \
  tests/unit/test_ecn_api.py \
  tests/unit/test_analysis_api.py \
  tests/unit/test_retry_service.py \
  tests/unit/test_history_api.py \
  tests/unit/test_app_lifespan.py
```

Then run the complete backend unit suite and import compilation:

```bash
PYTHONPATH=api python -m pytest -q tests/unit
python -m compileall -q api/app
```

Live HANA and SAP Gen AI Hub checks are opt-in, require real credentials, and use isolated suffixed test tables. No deployment command is part of this workflow.

```bash
RUN_FULL_LIVE_TESTS=1 PYTHONPATH=api .venv/bin/python -m pytest -q -s \
  tests/integration/test_generation_api_live.py
```
