# Business evidence presentation

## Purpose

The deterministic engine stores immutable, audit-oriented evidence. Its
internal IDs, enum values, and raw payloads are useful for traceability but are
not the primary user experience. The business evidence projection converts the
stored bundle into readable material changes, BOM matches, configuration scope,
data-quality issues, and genuine missing evidence.

Historical evidence is not rewritten. The projection can therefore improve
without invalidating an earlier upload or its evidence hash.

## Inputs and outputs

Input: one stored `EvidenceBundle` for an ECN occurrence.

Output sections:

- Added materials
- Removed materials
- Affected or modified materials
- EBOM matches
- MBOM matches
- Configuration and site applicability
- Data quality
- Missing evidence

Each output row carries business fields when they exist: material number,
description, revision, direction, parent, BOM path, quantity, effectivity,
plant, represented-row count, and representative source coordinates. Repeated
source-quality issues are grouped by code and impact. Parsing problems are in
Data quality; Missing evidence is limited to unsupported conclusions and
required inputs.

Internal `E-...` IDs, raw enums, related UUIDs, qualifiers, and raw values are
returned only under `technical_details`. Paginated lists omit that object by
default. It is included only when `include_technical=true` is requested, or
when an exact evidence item is opened for audit.

## API

```http
GET /api/ecns/{ecn_row_id}/evidence?section=added_materials&page=1&page_size=25
GET /api/ecns/{ecn_row_id}/evidence?section=data_quality&page=1&page_size=25
GET /api/ecns/{ecn_row_id}/evidence/{evidence_id}
```

`page_size` is bounded to 1-100. Evidence is loaded only for these explicit
requests; the ECN detail endpoint will use metadata and lazy evidence pages.

## Related files

- `api/app/ecn/evidence/business.py` - grouping and pagination rules
- `api/app/ecn/api_models.py` - public business DTOs
- `api/app/persistence/ecn_repository.py` - isolated evidence-bundle read
- `api/app/routers/ecns.py` - list and exact lookup routes
- `tests/unit/test_business_evidence.py` - grouping and pagination tests

## Test

```bash
PYTHONPATH=api .venv/bin/python -m pytest -q \
  tests/unit/test_business_evidence.py \
  tests/unit/test_ecn_api.py \
  tests/unit/test_ecn_repository.py
```
