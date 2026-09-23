---
name: sap-document-ai-client
description: Use when building or debugging the SAP Document AI Python REST client, especially SapDoxClient schema fields, schema activation, client or document-job flows, upload options, DoxApiError handling, mocked tests, or Document AI core-client documentation.
---

# SAP Document AI Client

## Core Workflow

Use this skill when working on the Python `dox_client` package for SAP Document AI REST API integration.

1. Inspect the current repository files first; prefer live code over bundled snapshots.
2. Read `references/api-workflow.md` before changing schema fields, public APIs, or tests.
3. Use the source snapshots in `assets/source-snapshot/` only when the target repository is missing context or when porting this client pattern elsewhere.
4. Preserve the core easy flow: create/select client, create/configure schema, upload PDF, poll/get extraction results, list/search/delete jobs.
5. Keep automated tests mocked. Do not require live SAP credentials for the default test suite.

## Repository Anchors

Expected files in the `dox_client` repository:

- `dox_client/sap_dox_client.py`: REST client, request layer, normalized wrappers, schema/document helpers.
- `dox_client/models.py`: Pydantic field definitions, upload/catalog options, extraction parsing helpers.
- `dox_client/__init__.py`: package exports.
- `tests/test_sap_dox_client.py`: fake-session tests for the core API.
- `docs/document-ai-core-client.md`: feature documentation.
- `README.md`: user-facing setup and examples.

If these files are absent, inspect `assets/source-snapshot/` for the canonical implementation from the original project.

## Implementation Rules

- Normalize SAP list responses into plain `list[dict]` for `list_clients`, `list_schemas`, `list_schema_versions`, and `list_documents`.
- Raise `DoxApiError` for failed SAP API calls and preserve `status_code`, `sap_code`, `sap_message`, `details`, and `response_text`.
- Keep schema deletion aligned with SAP docs: `DELETE /schemas?clientId=...` with `{"value": ["schema-id"]}`.
- Keep document deletion aligned with SAP docs: `DELETE /document/jobs` with `{"value": ["job-id"]}`.
- Validate upload option combinations before HTTP calls. Reject schema plus ad-hoc fields, two schema identifiers, templates without schema, and candidate templates without template detection.
- Send ad-hoc upload fields under `options["extraction"]`.
- Convert `FieldDefinition`, field dictionaries, and field-name strings consistently.
- Validate the complete schema field contract before creating a schema shell. For field names, labels, supported data types, setup payloads, full-array replacement behavior, and activation order, follow `references/api-workflow.md`.
- Do not update `tutorial.ipynb` unless the user explicitly asks for notebook migration.

## Testing

Run:

```bash
python3 -m pytest -q
python3 -m compileall dox_client
git diff --check
```

Add or update tests before production code when changing behavior. Use a fake `requests.Session`; do not call SAP during unit tests.

## Documentation

When adding a feature, update `docs/document-ai-core-client.md` or create a focused Markdown file under `docs/` per the repository instruction. Include purpose, expected inputs and outputs, related files, snippets, and test commands.
