# Joule Commodity Code Assistant Integration

## Purpose

The direct SAPDAS capability in `joule_agent/` lets a Joule user upload one PDF, start the existing asynchronous extraction and commodity-code matching pipeline, check its status, and browse all detected line items in numbered pages. The Joule response replaces the previous user-facing Excel-only workflow; the existing workbook can remain an internal artifact for the Streamlit UI and backend compatibility.

The capability does not contain extraction, retrieval, matching, or AI logic. It reuses the FastAPI backend, HANA-backed job storage, LLM document extraction, HANA reference data, embedding matching, and final LLM verification already implemented in `api/`.

## Inputs and Outputs

Input:

- Exactly one PDF uploaded through Joule's `file-upload` action.
- Maximum file size: 10 MB, matching Joule attachment limits and the backend boundary.
- No scenario file slot is used for the attachment. The dialog function reads `uploaded_document.files.get(0).file_reference_id` and forwards it as an octet-stream body.

Each result line item contains exactly these seven fields:

- `description`
- `net_amount`
- `quantity`
- `unit_price`
- `ai_suggested_commodity_code`
- `ai_confidence_score`
- `ai_reasoning`

The API returns up to 30 line items per page plus pagination metadata. Joule exposes both root results through `response_context`, allowing generated cards or lists while retaining the backend values as the source of truth.

## End-to-End Flow

1. The submit scenario routes to `fn_submit_pdf`.
2. `file-upload` collects one PDF and returns an opaque attachment reference.
3. Joule resolves that reference and sends the PDF bytes to the backend with `Content-Type: application/octet-stream`.
4. The backend validates the body, stores the PDF and job in HANA, and returns `202 Accepted` with a `job_id`.
5. The submit scenario stores only that root `job_id` as `last_job_id` in conversation-scoped capability context.
6. The user explicitly checks status. Joule never polls automatically.
7. When the backend reports `SUCCEEDED`, the user selects **Show page 1**.
8. The first-page or numbered-page scenario calls the same result function, which retrieves one structured page.
9. Joule renders all seven fields for each returned line item. Pagination replies include the explicit adjacent page number, for example, `Show page 2 of the latest commodity code results`.

Recognized backend states are `QUEUED`, `RUNNING`, `SUCCEEDED`, and `FAILED`. A missing job, malformed response, unsupported status, or unavailable page produces a bounded recovery message.

## Related Files

- `joule_agent/assistant.da.sapdas.yaml`: assistant wrapper named `commodity_code_assistant`.
- `joule_agent/commodity_code_capability/capability.sapdas.yaml`: capability metadata and destination alias.
- `joule_agent/commodity_code_capability/capability_context.yaml`: conversation-scoped `last_job_id` only.
- `joule_agent/commodity_code_capability/functions/fn_submit_pdf.yaml`: PDF upload and raw-byte submission.
- `joule_agent/commodity_code_capability/functions/fn_check_job.yaml`: user-driven status lookup.
- `joule_agent/commodity_code_capability/functions/fn_show_results.yaml`: shared first-page and numbered-page result retrieval.
- `joule_agent/commodity_code_capability/scenarios/`: intent routing, capability-context injection, page slot, and response grounding.
- `api/app/routers/extraction.py`: raw submit, status, and structured paginated result endpoints.
- `api/app/models/extraction.py`: structured line-item and pagination response models.
- `api/app/services/extraction_jobs.py`: asynchronous job lifecycle and result-page retrieval.
- `api/app/services/extraction_service.py`: extraction, matching, and seven-field Joule result mapping.
- `docs/joule-commodity-results.md`: backend implementation and test contract.

## Exact Backend Contract

All routes are relative to the Cloud Foundry API application URL and require the same `X-API-Key` when `API_KEY` is configured.

### Submit one PDF

```http
POST /api/extraction/jobs
Content-Type: application/octet-stream
Accept: application/json
X-API-Key: <API_KEY>

%PDF-...
```

Successful response:

```http
202 Accepted
Content-Type: application/json
```

```json
{
  "job_id": "uuid",
  "status": "QUEUED",
  "status_url": "/api/extraction/jobs/uuid",
  "download_url": "/api/extraction/jobs/uuid/download",
  "created_at": "2026-07-18T09:00:00Z"
}
```

### Check status

```http
GET /api/extraction/jobs/{job_id}
Accept: application/json
X-API-Key: <API_KEY>
```

The `status` field is `QUEUED`, `RUNNING`, `SUCCEEDED`, or `FAILED`.

### Read one result page

```http
GET /api/extraction/jobs/{job_id}/result?page={page_number}
Accept: application/json
X-API-Key: <API_KEY>
```

Successful response shape:

```json
{
  "job_id": "uuid",
  "status": "SUCCEEDED",
  "line_items": [
    {
      "description": "Brake pad set",
      "net_amount": 120.5,
      "quantity": 2,
      "unit_price": 60.25,
      "ai_suggested_commodity_code": "12345678",
      "ai_confidence_score": "91%",
      "ai_reasoning": "The item purpose matches the selected taxonomy entry."
    }
  ],
  "pagination": {
    "current_page": 1,
    "page_size": 30,
    "total_items": 31,
    "total_pages": 2,
    "previous_page": null,
    "next_page": 2
  }
}
```

## BTP Destination

Create or update a destination manually with these values:

```text
Name: CommodityCodePipelineAPI
Type: HTTP
URL: https://<commodity-code-api-route>
Proxy Type: Internet
Authentication: NoAuthentication
URL.headers.X-API-Key: <same API_KEY configured on the API application>
```

The SAPDAS capability maps its only system alias as follows:

```yaml
system_aliases:
  CommodityCodeAPI:
    destination: CommodityCodePipelineAPI
```

Do not put the API key in SAPDAS YAML or commit it to the repository. Destination administrators own the secret value and rotation.

## Validation and Tests

Run backend tests from the repository root:

```bash
cd api
python3 -m unittest discover -s tests
cd ..
```

Run the attachment contract verifier:

```bash
python3 /Users/I760054/.codex/skills/sap-joule-attachment-capabilities/scripts/check_attachment_contract.py \
  joule_agent \
  --strict
```

Validate and compile with the tenant-compatible Joule CLI:

```bash
joule lint joule_agent/assistant.da.sapdas.yaml
joule compile joule_agent/commodity_code_capability /tmp/commodity_code_joule_compile
```

Lint and compile validate design-time structure. Runtime testing must additionally confirm that Joule sends real PDF bytes, the destination adds `X-API-Key`, the backend returns `202`, capability context retains `job_id`, and all status/result branches render correctly.

### Runtime troubleshooting

- Dialog-function result expressions that feed capability context use YAML's `>-` block scalar. The trailing `-` is required: plain `>` appends a line break to a string result, which would turn the stored job ID into `<uuid>\n` and make the Business Connector reject the status and result paths before calling the backend.
- In a Joule debug export, `W_API_REQUEST_PREPARATION` with `Invalid character '\n' for PATH` identifies that stale job-ID value. Redeploy the corrected assistant and start a new conversation before uploading the PDF again, because an existing conversation can still retain the old capability-context value.
- Quick replies are routed as new user utterances. If a status request opens the upload flow, inspect `SCENARIO_SELECTED` and keep the submit description focused on starting a new document job while the status description focuses on checking, polling, or refreshing an existing job.
- `status_code` expression errors after a request-preparation failure are downstream symptoms. Confirm the request path first; the API-result guards then provide the user-facing recovery message.

## Manual Cloud Foundry and Joule Deployment

Deployment is always user-run. Codex must not execute these commands.

Deploy the Cloud Foundry applications after reviewing variables and the existing manifest:

```bash
cf push --var api_key="<API_KEY>"
```

After the backend route and destination are available, deploy the Joule assistant manually with Joule CLI 1.5.21:

```bash
joule deploy --compile \
  joule_agent/assistant.da.sapdas.yaml \
  --name commodity_code_assistant
```

Then perform a runtime smoke test in one conversation: upload a known PDF, correlate Joule and backend logs, check pending and terminal states, open page 1, request a numbered page, and verify all seven fields against the API JSON.
