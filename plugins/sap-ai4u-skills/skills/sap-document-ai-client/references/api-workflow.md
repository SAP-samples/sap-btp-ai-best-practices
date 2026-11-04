# SAP Document AI Client API Workflow

## Scope

This skill supports the core easy flow only:

1. Create or select a SAP Document AI client.
2. Create a schema and configure fields.
3. Upload a PDF or supported document with schema or ad-hoc extraction fields.
4. Poll or retrieve extraction results.
5. List, search, and delete document jobs.
6. Instant learning: read/set configuration, save ground truth, confirm documents, export confirmed documents (details in `references/instant-learning.md`).

Deferred unless explicitly requested: Enrichment Data CRUD, OCR text endpoints, and full Template API CRUD.

## Public Client Surface

Discovery:

- `get_capabilities()`
- `get_schema_capabilities()`

Clients:

- `create_clients(clients)`
- `list_clients(limit=100, offset=0, client_id_starts_with=None)`
- `delete_clients(client_ids)`

Schemas:

- `create_schema(client_id, schema_name, schema_description=None, document_type=None, document_type_description=None)`
- `update_schema(schema_id, client_id="default", name=None, schema_description=None, document_type_description=None)`
- `delete_schema(schema_ids, client_id="default")`
- `list_schemas(client_id="default", limit=100, offset=0, document_type=None, order=None, predefined=None)`
- `get_schema_by_name(schema_name, client_id="default", limit=1000, document_type=None, predefined=None)`
- `get_schema_details(schema_id, client_id="default")`
- `create_schema_version(schema_id, client_id="default")`
- `update_schema_version(schema_id, version, client_id="default", schema_description=...)`
- `list_schema_versions(schema_id, client_id="default")`
- `get_schema_version_details(schema_id, version, client_id="default")`
- `is_schema_version_editable(schema_id, version, client_id="default")`
- `add_fields_to_schema_version(...)`
- `configure_schema_version(...)`
- `activate_schema_version(schema_id, version, client_id="default")`
- `deactivate_schema_version(schema_id, version, client_id="default")`

Documents:

- `upload_document(...)`
- `get_job(job_id, extracted_values=None, return_null_values=None)`
- `wait_for_result(job_id, timeout_seconds=180, poll_interval_seconds=2, terminal_statuses=None, extracted_values=None, return_null_values=None)`
- `list_documents(client_id=None)`
- `search_document_catalog(client_id=None, filter_query=None, like_filter=None, limit=None, offset=None, order=None)`
- `delete_jobs(job_ids)`
- `delete_job(job_id)`

Configuration and instant learning:

- `get_configuration(key=None, scope=None, client_id=None)`
- `set_configuration(values, scope="tenant", client_id=None)`
- `ground_truth_from_job(job)` (static)
- `save_ground_truth(job_id, extraction, enrichment=None, languages=None, country=None)`
- `confirm_document(job_id, data_for_retraining=False)`
- `export_document(job_id, client_id)`

## Endpoint Map

- `GET /capabilities`
- `POST /clients`, `GET /clients`, `DELETE /clients`
- `GET /schemas/capabilities`
- `POST /schemas`, `GET /schemas`, `DELETE /schemas`
- `GET /schemas/<schemaId>`, `PUT /schemas/<schemaId>`, `POST /schemas/<schemaId>`
- `GET /schemas/<schemaId>/versions`
- `GET /schemas/<schemaId>/versions/<version>`
- `PUT /schemas/<schemaId>/versions/<version>`
- `POST /schemas/<schemaId>/versions/<version>/fields`
- `POST /schemas/<schemaId>/versions/<version>/activate`
- `POST /schemas/<schemaId>/versions/<version>/deactivate`
- `POST /document/jobs`, `GET /document/jobs`, `DELETE /document/jobs`
- `GET /document/jobs/<id>`
- `POST /document/catalog`
- `GET /configuration`, `GET /configuration/<key>`, `POST /configuration`, `DELETE /configuration`
- `POST /document/jobs/<id>` (save ground truth)
- `POST /document/jobs/<id>/confirm?dataForRetraining=<bool>`
- `GET /document/jobs/<id>/export?clientId=<client>`

## SAP Response Normalization

SAP often wraps arrays:

```python
{"id": "tenant", "payload": [{"clientId": "c_00"}]}
{"schemas": [[{"id": "schema-1", "version": "1"}]]}
{"results": [[{"id": "job-1"}]]}
```

Core list methods should flatten these into:

```python
[{"clientId": "c_00"}]
[{"id": "schema-1", "version": "1"}]
[{"id": "job-1"}]
```

## Schema Field Contract

Validate the complete field set before creating the schema. SAP creates the schema shell first; a later field failure can leave an empty inactive schema, and activation will not run.

Field rules:

| Property | Required contract |
|---|---|
| `name` | Required and unique. No whitespace. Use letters, numbers, `_`, `-`, `.`, `,`, `&`, `$`, `#`, or `~`. |
| `label` | Free-form, maximum 200 characters, and unique across all header and line-item fields. Separate technical names do not permit duplicate labels. |
| `description` | Free-form, maximum 500 characters. Put conditional business requirements here when needed. |
| `formattingType` | `string`, `number`, `date`, `discount`, `currency`, `country/region`, or `listOfValues`. Normalize agent-facing `integer` to `number` and `list of values` to `listOfValues`. |
| setup | Always use `setupType: static`, `setupTypeVersion: 2.0.0`, and `setup: {type: auto, priority: 1}` for this client workflow. |

The field payload must contain both arrays, even when one is empty:

```python
payload = {
    "headerFields": [
        {
            "name": "invoice_number",
            "label": "Invoice Number",
            "description": "Unique invoice identifier; mandatory",
            "defaultExtractor": {},
            "setupType": "static",
            "setupTypeVersion": "2.0.0",
            "setup": {"type": "auto", "priority": 1},
            "formattingType": "string",
            "formatting": {},
            "formattingTypeVersion": "1.0.0",
        }
    ],
    "lineItemFields": [],
}
```

`POST /schemas/<schemaId>/versions/<version>/fields` represents the complete field set. It replaces the stored arrays rather than appending one field. To add fields sequentially, resend all previously accepted fields plus the new field and include both arrays. Verify the stored field counts after each diagnostic request.

Observed failure signatures:

- Omitting either array can return HTTP 500 with only `Internal Server Error`.
- Reusing a label can return HTTP 400 with SAP code `ES130` and `Schema label names shouldn't be repeated.` Disambiguate duplicate business labels, for example by adding the section name.

The current schema-field API has no mandatory flag. Preserve unconditional and conditional requirements in descriptions or application metadata; do not invent a `mandatory` payload property.

After the field POST succeeds, retrieve the schema version, verify the field counts, and only then call the activation endpoint.

## Schema Maintenance

Observed against the live service:

- `PUT /schemas/<schemaId>` (name or description change) fails on an active schema with `Schema ... is in active state`. Deactivate the version, `PUT`, then reactivate (reactivate in a `finally` block so a failed update does not leave the schema inactive).
- The `PUT /schemas/<schemaId>` body must contain `name` and `documentTypeDescription` in addition to the changed `schemaDescription`; partial bodies return HTTP 400 naming the missing field.
- `POST /schemas` also requires `documentTypeDescription`.
- Schema names follow the field-name rule: no whitespace (`Field name shouldn't contain spaces`). Use underscores for readable display names.
- `DELETE /schemas` fails for schemas with an active version. Deactivate every active version first.
- `GET /schemas/<schemaId>/versions` returns its list under the `schemas` key, possibly nested one level.

## Upload Rules

Schema upload:

```python
client.upload_document(
    "invoice.pdf",
    client_id="c_00",
    schema_id="schema-id",
    schema_version=1,
    document_type="invoice",
)
```

Ad-hoc upload:

```python
client.upload_document(
    "document.pdf",
    client_id="c_00",
    document_type="custom",
    header_fields=["documentNumber"],
    line_item_fields=["netAmount"],
)
```

The ad-hoc multipart `options` JSON must use:

```json
{
  "clientId": "c_00",
  "documentType": "custom",
  "extraction": {
    "headerFields": ["documentNumber"],
    "lineItemFields": ["netAmount"]
  }
}
```

Template detection requires a schema:

```python
client.upload_document(
    "invoice.pdf",
    client_id="c_00",
    schema_id="schema-id",
    template_id="detect",
    candidate_template_ids=["template-a", "template-b"],
)
```

## Test Pattern

Use a fake session that records requests and returns queued fake responses. Assert paths, query parameters, JSON bodies, and multipart `options` payloads. Keep tests independent from SAP credentials.

Minimum behavior coverage for new changes:

- Token caching or request auth if authentication changes.
- `DoxApiError` details for failed API responses.
- Response normalization for new list-like APIs.
- Upload validation and multipart `options` payloads.
- Correct official endpoint and body shape for delete/update operations.
- Ground-truth payload shape, `dataForRetraining` query value, and configuration scope parameters for instant learning changes.
