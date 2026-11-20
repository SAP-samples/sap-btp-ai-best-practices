# Delete selected ECNs

## Purpose

The Incoming ECNs table can remove selected ECN line items as if they had never been imported. It is a scoped version of the [analysis history reset](ecn-history-reset.md): only the selected ECNs are removed, and everything else stays in place.

A line item means **every assessment version** of one original ECN source position. A reassessment reuses the original `ECN_JSON` artifact and keeps its `SOURCE_ORDINAL`, so the pair `(ECN_ARTIFACT_ID, SOURCE_ORDINAL)` groups all versions together. Deleting any version deletes the whole line item. Importing the same ECN again later creates a new artifact, which makes it a new, independent line item.

## Inputs and outputs

| Input | Output |
|---|---|
| Selection from the list: explicit checked rows, or "Select all matching ECNs" (the filter minus unchecked rows) | Every version of each selected ECN removed from HANA and from the knowledge graph |
| Exact confirmation phrase `DELETE` | Deletion counts plus `graph_cleanup: "complete" \| "failed"` |

```http
POST /api/ecns/delete
{"confirmation": "DELETE",
 "selection": {"mode": "explicit", "ecn_ids": ["<ecn_row_id>", "..."]}}
```

`selection` accepts the same payload as `POST /api/summary-batches`, so `{"mode": "filtered", ...}` works too.

Response:

```json
{"ecns": 2, "analyses": 1, "attempts": 1, "jobs": 1, "reviews": 0,
 "uploads": 1, "source_artifacts": 1, "feasibility_assessments": 0,
 "graph_cleanup": "complete"}
```

| Code | HTTP | Meaning |
|---|---:|---|
| `ECN_DELETE_CONFIRMATION_INVALID` | 400 | The phrase was not exactly `DELETE`. Nothing was read or deleted. |
| `SUMMARY_SELECTION_EMPTY` | 400 | The selection resolved to no ECNs. |
| `ECN_DELETE_ACTIVE_JOBS` | 409 | A selected ECN (any version) has a queued or running summary job. Nothing was deleted. |
| `ECN_DELETE_EXTERNAL_REFERENCES` | 409 | A table the app does not manage still references a selected ECN. The message names each table and its row count. Nothing was deleted. See [Tables the app does not manage](#tables-the-app-does-not-manage). |
| `ECN_DELETE_REFERENCED` | 409 | HANA refused the delete with a foreign-key violation (error 462) that the preflight did not catch, for example a deeper chain of foreign tables. The transaction was rolled back and the server log names the table. |

The analysis history reset uses the same checks with the codes `HISTORY_RESET_EXTERNAL_REFERENCES` and `HISTORY_RESET_REFERENCED`.

Repeating a request is harmless because IDs that no longer exist are ignored.

## What is deleted and what is kept

| Deleted (only rows of the selected ECNs) | Kept |
|---|---|
| `REVIEW`, `BATCH_JOB`, `ANALYSIS`, `PROVIDER_CALL`, `ANALYSIS_STAGE`, `LLM_ATTEMPT`, `ANALYSIS_JOB`, `RDF_PROJECTION`, `MANUFACTURING_FEASIBILITY_ASSESSMENT`, `ECN` | Reference datasets: BOM, material master, purchase info, stock, conditional rules |
| `UPLOAD` / `UPLOAD_DATASET` rows once no ECN references them | Uploads and artifacts still shared with ECNs that are not deleted |
| `ECN_JSON` source artifacts once no upload references them | All table definitions |
| `GENERATION_BATCH` rows once no batch member remains | Other ECNs' knowledge-graph triples |
| Knowledge-graph triples with `belongsToEcn <ecn root>` | |

A single import of up to 25 ECNs shares one upload and one artifact. The upload and artifact survive until the last ECN from that import is deleted. While they survive, the stored JSON still contains the deleted ECN's record, but no ECN row points to it, so it is never displayed or analysed.

## How it works

1. The router resolves the selection with `_resolve_summary_ids` (shared with summary generation).
2. `HistoryService.delete_ecns` checks the phrase, then takes the shared `HistoryMutationGuard` so the delete cannot race an upload publication.
3. `HistoryRepository.delete_ecns_if_idle` runs **one HANA transaction**: it expands the selection to all versions, runs the active-job check, runs the external-reference preflight, deletes in foreign-key order, and removes orphaned parents with `NOT IN (SELECT ...)` guards.
4. After the commit, `RdfProjectionService.delete_ecns` removes the per-ECN graph triples. The graph is cleaned last on purpose. If this step fails, the relational delete stays committed, the error is logged, and the response reports `graph_cleanup: "failed"`. Graph reads are always scoped to an existing ECN, so leftover triples are never shown.

Key snippet (`api/app/persistence/history_repository.py`):

```python
f'SELECT DISTINCT V."ECN_ROW_ID", V."UPLOAD_ID", VU."ECN_ARTIFACT_ID" '
f'FROM {ecn} S JOIN {upload} SU ON S."UPLOAD_ID" = SU."UPLOAD_ID" '
f'JOIN {upload} VU ON VU."ECN_ARTIFACT_ID" = SU."ECN_ARTIFACT_ID" '
f'JOIN {ecn} V ON V."UPLOAD_ID" = VU."UPLOAD_ID" '
f'AND V."SOURCE_ORDINAL" = S."SOURCE_ORDINAL" '
f'WHERE S."ECN_ROW_ID" IN :ids'
```

## Tables the app does not manage

A HANA schema can hold tables that were created outside this codebase but still declare a `RESTRICT` foreign key on `ECN_IMPACT_ECN`. A known case is the `ECN_IMPACT_MAKE_BUY_*` family, left over from the removed make-or-buy feature. Before this check existed, deleting a referenced ECN failed with HANA error 462, the API returned a 500, and the dialog showed only "Failed to fetch".

`find_external_ecn_references` (`api/app/persistence/external_references.py`) runs inside the delete transaction, after the active-job check and before any DELETE:

1. It reads every foreign key in `CURRENT_SCHEMA` that references the ECN table from `REFERENTIAL_CONSTRAINTS`.
2. It skips the tables listed in `PersistenceTableNames`, because the delete order already handles them.
3. For each remaining table it counts the rows that reference the ECNs in scope (all non-null rows for the history reset).
4. If any count is above zero, it raises `ExternalEcnReferenceError` and nothing is deleted.

The app never deletes rows from tables it does not own. Catalog names are interpolated into SQL only when they match `^[A-Z0-9_]{1,127}$`; any other name is reported as blocking without being queried.

```python
blocking = find_external_ecn_references(connection, tables, ids)
if blocking:
    raise ExternalEcnReferenceError(blocking)
```

### Removing the leftover make-or-buy tables

`scripts/cleanup_legacy_make_buy_tables.py` lists the `ECN_IMPACT_MAKE_BUY_*` tables in the connected schema. It orders them children first from the catalog foreign keys and removes them only when asked. It stops if a table outside that family references them, or if one of them is owned by the app.

```bash
# Read-only: list the tables, their row counts and the cleanup order
.venv/bin/python scripts/cleanup_legacy_make_buy_tables.py
```

```bash
# Delete every row in one transaction; keep the table definitions
.venv/bin/python scripts/cleanup_legacy_make_buy_tables.py --apply
```

```bash
# Delete the rows, then drop the tables
.venv/bin/python scripts/cleanup_legacy_make_buy_tables.py --apply --drop
```

## UI

The **Delete selected ECNs** button sits next to **Generate summaries** and is enabled whenever the selection is not empty. The dialog lists the selected ECN numbers (or "N matching ECNs will be deleted." for a filter-wide selection) and requires typing `DELETE`. On success it clears the selection, returns to page 1, reloads the list and progress, and shows the count of deleted assessments. A 409 or other error stays inside the dialog and shows the API `message`, for example the tables that block the delete.

## Related files

- `api/app/routers/workspace_actions.py`: `POST /api/ecns/delete`
- `api/app/ecn/services/history.py`: `HistoryService.delete_ecns`
- `api/app/persistence/history_repository.py`: `delete_ecns_if_idle`, `_ecn_scope_sql`, `_scoped_delete_statements`
- `api/app/persistence/external_references.py`: `find_external_ecn_references`, `ExternalEcnReferenceError`
- `scripts/cleanup_legacy_make_buy_tables.py`: user-run removal of the leftover make-or-buy tables
- `api/app/persistence/rdf_repository.py` and `api/app/ecn/rdf/service.py`: per-ECN graph cleanup
- `api/app/persistence/records.py`: `EcnDeletionCounts`
- `ui/src/pages/ecns/ecns.html`, `ecns.js`, `ecns.css`, and `ui/src/services/workspace-api.js`
- `tests/unit/test_ecn_deletion.py`, `tests/unit/test_external_ecn_references.py`, `ui/tests/ecns-page.test.js`

## How to test

```bash
PYTHONPATH=api .venv/bin/python -m pytest -q tests/unit/test_ecn_deletion.py tests/unit/test_external_ecn_references.py tests/unit/test_history_service.py tests/unit/test_history_repository.py
npm --prefix ui test -- --run tests/ecns-page.test.js
```

Manual test: select one ECN that has a reassessment, delete it, and confirm that both versions are gone from the list. Then confirm that Settings still shows the same reference datasets and that another ECN from the same import still opens.
