# Joule PDF Commodity-Code Results

## Purpose

This feature lets an SAP Joule capability submit one PDF as raw bytes, reuse the
existing asynchronous extraction and commodity-code matching pipeline, and read
all detected line items as deterministic JSON pages. The existing Streamlit
multipart upload and Excel download remain unchanged.

## Input

```http
POST /api/extraction/jobs
Content-Type: application/octet-stream
Accept: application/json
X-API-Key: <configured API key>

%PDF-...
```

The body must be nonempty, start with `%PDF-`, and be no larger than 10 MiB.
The backend owns the filename and extraction settings, including
`llm_verify=true`; Joule cannot override pipeline options through this route.

An accepted request returns `202` with the existing `job_id`, `status`,
`status_url`, `download_url`, and `created_at` fields. Joule should preserve the
root `job_id`, poll the existing status URL, and request results after the status
becomes `SUCCEEDED`.

## Output

```http
GET /api/extraction/jobs/{job_id}/result?page=1
Accept: application/json
X-API-Key: <configured API key>
```

Each page contains at most 30 rows:

```json
{
  "job_id": "uuid",
  "status": "SUCCEEDED",
  "line_items": [
    {
      "description": "Brake pads",
      "net_amount": 100,
      "quantity": 2,
      "unit_price": 50,
      "ai_suggested_commodity_code": "RC0001",
      "ai_confidence_score": "91%",
      "ai_reasoning": "Best semantic match."
    }
  ],
  "pagination": {
    "current_page": 1,
    "page_size": 30,
    "total_items": 1,
    "total_pages": 1,
    "previous_page": null,
    "next_page": null
  }
}
```

Missing values are returned as `Not detected`; detected monetary and quantity
values remain numeric. `UNSURE` suggestions and fallback reasoning are retained.
Invalid pages return `400`, unknown jobs `404`, unfinished results `409`, and
HANA storage failures `503`. `previous_page` is null on the first page and
`next_page` is null on the last page.

## Processing and Persistence

All-null placeholder rows created for documents without detected line items are
removed before commodity-code matching. A job fails when no genuine line item
remains. Every enriched row is normalized to the seven fields above and stored
inside the existing `RESULT_METADATA_JSON` NCLOB. The HANA schema, uploaded PDF
BLOB storage, Excel artifact, status endpoint, and download endpoint do not
change.

## Related Files

- `api/app/routers/extraction.py`: raw submission and paginated result routes.
- `api/app/services/extraction_service.py`: placeholder filtering and result serialization.
- `api/app/services/extraction_jobs.py`: HANA metadata retrieval and 30-row pagination.
- `api/app/models/extraction.py`: public Joule result models.
- `api/tests/`: route, pipeline, serialization, persistence, and pagination checks.

## Test

From the `api` directory:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```
