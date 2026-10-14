# SAP Document AI Core Client

## What This Feature Is

This feature completes the core Python client flow for the SAP Document AI REST
API. It lets a developer authenticate with a service key, create or select a SAP
Document AI client, create and configure a schema, upload a PDF, poll for
extraction results, and manage uploaded document jobs.

The client intentionally focuses on the easy workflow. Full administration APIs
for configuration, enrichment-data CRUD, ground truth, confirmation, exports,
and template CRUD are out of scope for this pass.

## Expected Inputs And Outputs

Inputs:

- A SAP Document AI service key JSON with `uaa.url`, `uaa.clientid`,
  `uaa.clientsecret`, `url`, and `swagger`.
- Client definitions with `clientId` and `clientName`.
- Schema metadata and field definitions.
- PDF or image document paths for upload.
- Optional SAP upload options such as `document_type`, `received_date`,
  `custom_label`, `enrichment`, `template_id`, and `candidate_template_ids`.

Outputs:

- Normalized Python lists for client, schema, schema version, and document list
  operations.
- SAP response dictionaries for create, update, upload, polling, catalog search,
  and deletion operations.
- `DoxApiError` exceptions containing HTTP status and SAP error details for
  failed API calls.

## Related Files

- `dox_client/sap_dox_client.py`: main REST client, request layer, normalized
  list handling, schema helpers, document helpers, and `DoxApiError`.
- `dox_client/models.py`: Pydantic field definitions, upload/catalog option
  models, and extraction parsing helpers.
- `dox_client/__init__.py`: package exports for the client, error type, and
  models.
- `tests/test_sap_dox_client.py`: mocked HTTP tests covering the core client
  behavior.
- `README.md`: user-facing setup and usage guide.

## Key Code Snippets

Create and configure a schema:

```python
from dox_client import FieldDefinition, SapDoxClient

client = SapDoxClient.from_service_key("dox_client/schemas/service_key.json")

schema = client.create_schema(
    client_id="invoice_client",
    schema_name="Invoice_Schema",
    schema_description="Invoice fields for extraction",
    document_type="invoice",
)

client.configure_schema_version(
    schema_id=schema["id"],
    version=1,
    client_id="invoice_client",
    header_fields=[FieldDefinition(name="documentNumber")],
    line_item_fields=[FieldDefinition(name="netAmount", formattingType="number")],
)
```

Upload and poll:

```python
job = client.upload_document(
    file_path="invoice.pdf",
    client_id="invoice_client",
    schema_id=schema["id"],
    schema_version=1,
    document_type="invoice",
)

result = client.wait_for_result(job["id"], timeout_seconds=180, poll_interval_seconds=3)
```

Search uploaded documents:

```python
catalog = client.search_document_catalog(
    client_id="invoice_client",
    filter_query="status eq done",
    like_filter='fileName like "invoice"',
    limit=10,
    order="created desc",
)
```

## Endpoint Coverage

Included:

- `GET /capabilities`
- `POST /clients`, `GET /clients`, `DELETE /clients`
- `GET /schemas/capabilities`
- `POST /schemas`, `GET /schemas`, `PUT /schemas/<schemaId>`,
  `DELETE /schemas`
- `POST /schemas/<schemaId>`
- `GET /schemas/<schemaId>`
- `GET /schemas/<schemaId>/versions`
- `GET /schemas/<schemaId>/versions/<version>`
- `PUT /schemas/<schemaId>/versions/<version>`
- `POST /schemas/<schemaId>/versions/<version>/fields`
- `POST /schemas/<schemaId>/versions/<version>/activate`
- `POST /schemas/<schemaId>/versions/<version>/deactivate`
- `POST /document/jobs`, `GET /document/jobs`, `GET /document/jobs/<id>`,
  `DELETE /document/jobs`
- `POST /document/catalog`

Deferred:

- Configuration API
- Enrichment Data API CRUD
- Ground-truth save and document confirmation
- Export and OCR text endpoints
- Template CRUD and document-template association endpoints

## How To Test

Run the mocked unit test suite:

```bash
python3 -m pytest
```

Run only the client tests:

```bash
python3 -m pytest tests/test_sap_dox_client.py -q
```

These tests do not call SAP and do not require credentials.
