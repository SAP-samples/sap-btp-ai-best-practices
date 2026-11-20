> **Current workflow:** See [ECN snapshot review redesign](snapshot-review-redesign.md).
> Registration/import now make no model calls; explicit generation uses one synthesis
> stage. The older flow below remains historical implementation context.

# JSON-only ECN intake and source pinning

## Purpose

ECN intake is deliberately separate from reusable source management. Operators
publish BOM and other source workbooks in Settings, then upload only the ECN
JSON. This prevents every ECN batch from carrying a duplicate workbook and
makes the exact source versions used by an analysis auditable.

## API contract

`POST /api/uploads` accepts exactly:

- `request_id`: caller-generated idempotency UUID.
- `ecn_file`: bounded `.json` file.

Obsolete multipart fields such as `bom_file` are rejected. If Settings has no
validated global BOM publication, intake returns `NO_ACTIVE_BOM` and writes
nothing.

## Snapshot behavior

At publication time the ingestion service reads the current active reference
metadata, loads the active BOM rows from HANA, and writes one
`ECN_IMPACT_UPLOAD_DATASET` link per active dataset type. These links are
immutable: replacing a Settings source affects only later ECN uploads.

Legacy upload/BOM columns remain populated for historical compatibility. A
history reset removes ECN history and its dataset links but preserves all
Settings publications, their normalized rows, and their source artifacts.

## Related files

- `api/app/routers/uploads.py`: JSON-only transport contract.
- `api/app/ecn/services/ingestion.py`: active-source selection and pinning.
- `api/app/persistence/upload_repository.py`: atomic provenance-link inserts.
- `api/app/persistence/history_repository.py`: reference-safe history reset.

## Tests

```bash
PYTHONPATH=api .venv/bin/python -m pytest \
  tests/unit/test_upload_api.py \
  tests/unit/test_ingestion_service.py \
  tests/unit/test_upload_repository.py \
  tests/unit/test_history_repository.py -q
```
