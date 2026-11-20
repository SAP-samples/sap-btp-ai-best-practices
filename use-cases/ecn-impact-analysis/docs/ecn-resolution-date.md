# ECN Resolution Date (release date)

## What this is

Every imported ECN carries a source-system `ResolutionDate` — the timestamp at
which the change notice was *released* (resolved) in Windchill. This feature
surfaces that date in two places:

1. **ECN inbox table** — a sortable, viewable "Resolution date" column, placed
   immediately before the "Uploaded" column.
2. **ECN detail page** — a "Resolution date" row in the *Change summary* card,
   rendered immediately **above** the "Uploaded" field.

"Uploaded" is when the snapshot was imported into this system; "Resolution date"
is when the ECN was released at the source. They are distinct dates and are now
shown side by side.

## Why it is built this way (no new column)

The raw ECN object is already persisted verbatim in the `SOURCE_JSON` NCLOB
column of `ECN_IMPACT_ECN` (written by `ecn_publication_to_row`), and
`ResolutionDate` is a top-level key of that object. So the value already exists
for every ECN ever imported — including the rows imported before this feature.

Instead of adding a dedicated `RESOLUTION_DATE` column (which would require
editing the strict startup schema contract in `schema.py`, running an additive
`ALTER TABLE`, and backfilling every existing row), the read queries extract the
value directly from the stored JSON with HANA's `JSON_VALUE`:

```sql
JSON_VALUE(E."SOURCE_JSON", '$.ResolutionDate')
```

This needs **no schema migration, no backfill, and works immediately for all
existing rows**.

### Known ceiling and upgrade path

`JSON_VALUE` parses the `SOURCE_JSON` NCLOB per row when sorting — perfectly fine
for hundreds to low-thousands of ECNs, but it is a full scan with no index. The
sort is also a lexical comparison of the ISO-8601 string, which is chronological
only while the source timezone offset stays consistent (Windchill exports use a
consistent offset). If the ECN table grows large, or offsets vary, promote
`ResolutionDate` to an indexed UTC `TIMESTAMP` column using the additive-`ALTER`
mechanism already present in `schema.py` (see the `additive` dict handling), and
point `RESOLUTION_DATE_EXPR` at that column instead. This is marked with a
`ponytail:` comment in `ecn_repository.py`.

## Inputs and outputs

**Input**: the `ResolutionDate` field of each ECN object in an imported ECN JSON
(e.g. `data/ECN.json`), an ISO-8601 string such as
`"2017-01-25T11:56:05+01:00"`, or `null` when the source has no value.

**Outputs** (new fields on existing API responses):

- `GET /api/ecns` → each `EcnListItem` gains `resolution_date: datetime | null`.
- `GET /api/ecns/{ecn_row_id}` → `EcnDetailResponse` gains
  `resolution_date: datetime | null`.
- `GET /api/ecns?sort=resolution_date&order=asc|desc` — new allowlisted sort key.

A stored value that is absent, blank, or not valid ISO-8601 maps to `null`
(never a 500); the raw string always remains in `SOURCE_JSON` for audit.

## Related files

Backend:
- `api/app/persistence/ecn_repository.py` — `RESOLUTION_DATE_EXPR`, the
  `resolution_date` entry in `SORT_EXPRESSIONS`, `_NULLS_LAST_SORT_KEYS`, the
  projected `RESOLUTION_DATE` column in `_list_sql` and `_detail_sql`, and the
  read in `_stored_ecn_metadata_from_detail_row`.
- `api/app/ecn/api_models.py` — `resolution_date` on `EcnListItem` and
  `EcnDetailResponse`; `resolution_date` added to the `EcnListQuery.sort` literal.
- `api/app/routers/ecns.py` — `resolution_date` added to the `sort` query regex.
- `api/app/persistence/records.py` — `StoredEcnRecord.resolution_date`.
- `api/app/persistence/codec.py` — `_parse_iso_datetime`, and population in
  `ecn_list_item_from_row`, `ecn_detail_response_from_record`, and
  `stored_ecn_from_row`.

Frontend:
- `ui/src/modules/list-state.js` — `resolution_date` in `LIST_SORTS`.
- `ui/src/pages/ecns/ecns.html` — sort `<option>`, `<col>`, and `<th>`.
- `ui/src/pages/ecns/ecns.js` — the rendered cell; `formatDate` hardened for
  null/empty.
- `ui/src/pages/ecn-detail/ecn-detail.js` — the "Resolution date" row above
  "Uploaded" in `renderSourceSummary`.

Tests:
- `tests/unit/test_ecn_repository.py` — sort SQL, list-item parse (value + null),
  detail mapper (value + null); refined the detail SQL invariant test to allow the
  bounded `JSON_VALUE` scalar while still forbidding full `SOURCE_JSON` hydration.
- `ui/tests/fixtures/api-responses.js`, `ui/tests/ecns-page.test.js`,
  `ui/tests/ecn-detail-page.test.js`.

## Key code

Read expression, reused for sort and both projections
(`api/app/persistence/ecn_repository.py`):

```python
RESOLUTION_DATE_EXPR = "JSON_VALUE(E.\"SOURCE_JSON\", '$.ResolutionDate')"

SORT_EXPRESSIONS = {
    "uploaded_at": 'U."CREATED_AT"',
    "resolution_date": RESOLUTION_DATE_EXPR,
    # ...
}
```

Defensive ISO parsing at the API boundary (`api/app/persistence/codec.py`):

```python
def _parse_iso_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        return None
```

Detail Change-summary card, resolution above uploaded
(`ui/src/pages/ecn-detail/ecn-detail.js`):

```js
appendDefinition(
  list,
  "Resolution date",
  detail.resolution_date ? dateTimeFormat.format(new Date(detail.resolution_date)) : null,
);
appendDefinition(list, "Uploaded", dateTimeFormat.format(new Date(detail.uploaded_at)));
```

## How to test

Backend unit tests (from the repository root, using the project venv):

```bash
PYTHONPATH=api .venv/bin/python -m pytest tests/unit/test_ecn_repository.py -q
```

Relevant cases: `test_sort_resolution_date_uses_json_value_nulls_last`,
`test_list_item_parses_resolution_date`,
`test_list_item_resolution_date_absent_is_none`,
`test_detail_mapper_parses_resolution_date`,
`test_detail_mapper_resolution_date_absent_is_none`.

Frontend unit tests:

```bash
cd ui && npx vitest run tests/ecns-page.test.js tests/ecn-detail-page.test.js
```

Manual: import `data/ECN.json`, open the ECN inbox, sort by "Resolution date",
and open any ECN to see the "Resolution date" row above "Uploaded" in the Change
summary card.
