# Async Extraction Jobs

## What This Feature Is

The extraction API now accepts PDF uploads as asynchronous jobs. `POST /api/extraction/run` validates the upload, stores the PDFs in HANA, creates a job row, and returns `202 Accepted` with a `job_id` instead of keeping the HTTP request open until classification finishes.

The first implementation still runs work inside the API process with a bounded `ThreadPoolExecutor`. The persisted HANA job and artifact model makes the API/UI contract compatible with a future separate worker.

The existing extraction pipeline still creates a local workbook as an intermediate artifact. The job manager reads that file into HANA and removes the transient local copy after the successful store.

## What It Is For

This removes the previous long-running request/response behavior from the UI. The Streamlit app submits once, polls the job endpoint, renders coarse progress, then downloads the generated Excel workbook by job ID when the job succeeds.

## Inputs And Outputs

Input endpoint:

```text
POST /api/extraction/run
Content-Type: multipart/form-data
Header: X-API-Key: <API_KEY> when configured
Fields: files[], llm_verify, llm_model, llm_min_confidence, top_k, merge_headers, output_name, embedding_model, show_preview
```

Immediate output:

```json
{
  "job_id": "uuid",
  "status": "QUEUED",
  "status_url": "/api/extraction/jobs/uuid",
  "download_url": "/api/extraction/jobs/uuid/download",
  "created_at": "2026-05-26T13:00:00Z"
}
```

Polling endpoint:

```text
GET /api/extraction/jobs/{job_id}
```

Terminal statuses are `SUCCEEDED` and `FAILED`. Successful jobs include previews, runtime metadata, `reference_data_version`, `output_filename`, `output_size`, and a `download_url`.

Download endpoint:

```text
GET /api/extraction/jobs/{job_id}/download
```

The old path-based endpoint `GET /api/extraction/download?path=...` has been removed.

## Related Files

- `api/app/routers/extraction.py` defines the submit, poll, and job download endpoints.
- `api/app/services/extraction_jobs.py` contains the HANA repository, in-memory test repository, and in-process job manager.
- `api/app/services/extraction_service.py` exposes `run_extraction_for_paths()` so jobs can materialize PDFs from HANA BLOBs before running the existing pipeline.
- `api/app/services/auth.py` enforces `X-API-Key` when `API_KEY` is configured.
- `ui/src/api_client.py` submits the job, polls status, and downloads by job ID.
- `ui/src/Home.py` renders polling progress and completed results.

## HANA Tables

`EXTRACTION_JOBS` stores one row per submitted job:

```sql
CREATE COLUMN TABLE "EXTRACTION_JOBS" (
  "JOB_ID" NVARCHAR(64) NOT NULL,
  "STATUS" NVARCHAR(20) NOT NULL,
  "CREATED_AT" TIMESTAMP NOT NULL,
  "UPDATED_AT" TIMESTAMP NOT NULL,
  "STARTED_AT" TIMESTAMP,
  "FINISHED_AT" TIMESTAMP,
  "CONFIG_JSON" NCLOB NOT NULL,
  "PROGRESS" INTEGER NOT NULL,
  "STAGE" NVARCHAR(80) NOT NULL,
  "MESSAGE" NVARCHAR(500) NOT NULL,
  "FILE_COUNT" INTEGER NOT NULL,
  "LLM_VERIFY" BOOLEAN NOT NULL,
  "TOP_K" INTEGER NOT NULL,
  "RUNTIME_SECONDS" DOUBLE,
  "REFERENCE_DATA_VERSION" NVARCHAR(100),
  "OUTPUT_FILENAME" NVARCHAR(255),
  "OUTPUT_MIME_TYPE" NVARCHAR(128),
  "OUTPUT_SIZE" BIGINT,
  "RESULT_BLOB" BLOB,
  "RESULT_METADATA_JSON" NCLOB,
  "HEADERS_PREVIEW_JSON" NCLOB,
  "LINE_ITEMS_PREVIEW_JSON" NCLOB,
  "ERRORS_JSON" NCLOB,
  "WARNINGS_JSON" NCLOB,
  PRIMARY KEY ("JOB_ID")
);
```

`EXTRACTION_JOB_FILES` stores uploaded PDF inputs:

```sql
CREATE COLUMN TABLE "EXTRACTION_JOB_FILES" (
  "JOB_ID" NVARCHAR(64) NOT NULL,
  "FILE_INDEX" INTEGER NOT NULL,
  "FILENAME" NVARCHAR(255) NOT NULL,
  "MIME_TYPE" NVARCHAR(128) NOT NULL,
  "SIZE_BYTES" BIGINT NOT NULL,
  "CONTENT_BLOB" BLOB NOT NULL,
  PRIMARY KEY ("JOB_ID", "FILE_INDEX")
);
```

The repository creates the tables idempotently by checking `SYS.TABLES` and validates required column names and HANA data types through `SYS.TABLE_COLUMNS` before accepting writes.

## Key Code Snippets

Submitting a job from the router:

```python
payloads = await _read_uploads(files)
return get_job_manager().submit(files=payloads, config=config)
```

Polling from the UI client:

```python
while True:
    status_payload = get_job_status(job_id)
    if status_payload.get("status") in {"SUCCEEDED", "FAILED"}:
        return status_payload
    time.sleep(interval)
```

## Operational Settings

- `API_KEY`: when set, extraction endpoints require `X-API-Key`; in production, missing `API_KEY` fails closed.
- `EXTRACTION_JOB_WORKERS`: number of in-process worker threads; default `1`.
- `EXTRACTION_MAX_QUEUED_JOBS`: maximum queued plus running jobs; default `20`.
- `HANA_EXTRACTION_JOBS_TABLE`: optional jobs table name override; default `EXTRACTION_JOBS`.
- `HANA_EXTRACTION_JOB_FILES_TABLE`: optional uploaded-files table name override; default `EXTRACTION_JOB_FILES`.

## Limitations

- Jobs run inside the API process. API restarts can interrupt running work, though submitted job state and artifacts are persisted in HANA.
- There is no cancel, retry, websocket, or external queue in this version.
- Progress is intentionally coarse: queued `0`, preparing `10`, pipeline running `25`, storing result `90`, succeeded `100`.

## How To Test

Run backend tests:

```bash
cd anonymized/api
python -m unittest discover -s tests
```

Run UI client tests:

```bash
cd anonymized/ui
python -m unittest discover -s tests
```
