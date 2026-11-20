# Analysis history reset

To remove only some ECNs, use [Delete selected ECNs](ecn-scoped-delete.md) on the Incoming ECNs table instead.

The service accepts only the exact case-sensitive phrase `RESET HISTORY`. Under the same process mutation guard used by final upload publication, the HANA repository checks for `QUEUED` or `RUNNING` jobs and performs deletion only when none exist. The check and all deletes share one transaction; a conflict deletes nothing and does not wait.

| Deleted | Preserved |
|---|---|
| analyses and attempt telemetry | all seven HANA table definitions |
| analysis jobs and ECN occurrences | conditional-rule workbook artifacts/configuration |
| uploads and normalized BOM rows | HANA schema and credentials |
| ECN JSON and BOM workbook artifacts | application code and runtime configuration |

Deletion order is analysis, LLM attempt, job, RDF projection status, manufacturing feasibility assessment, ECN, upload, BOM row, then only ECN/BOM source artifacts. Before any delete, the repository checks for tables the app does not manage that still reference ECN rows (see [Delete selected ECNs](ecn-scoped-delete.md#tables-the-app-does-not-manage)); if any hold rows, nothing is deleted. After success there is no current BOM snapshot, so the next upload must include an EBOM/MBOM workbook. This action does not delete or re-import conditional rules.

## Service and transaction contract

`HistoryService.reset_history(confirmation)` validates the phrase before touching the repository. It then takes the shared `HistoryMutationGuard` and calls `HistoryRepository.clear_history_if_idle()` exactly once. The repository, not the service, owns both the active-job check and dependency-ordered deletion in one HANA transaction. The service performs no separate preflight query, retry loop, polling, sleep, or SQL operation.

The guard is process-local and deliberately short-lived. It makes final upload publication and reset mutually exclusive inside the single-process MVP. It is not a distributed lock or a work queue and does not block upload parsing, evidence preparation, list/detail reads, or provider requests.

Reset may therefore finish while a long upload is still preparing against a reused BOM. Before that upload can publish, ingestion revalidates the selected artifact under this same guard with an ID-only `SELECT 1` that excludes persisted payload columns. A removed snapshot yields `BOM_SNAPSHOT_INVALIDATED` and no partial publication; the upload must be resubmitted with a workbook.

On success, `HistoryDeletionCounts` reports exact deleted row counts for attempts, analyses, jobs, ECNs, uploads, BOM rows, and source artifacts. The counts come directly from the committed repository transaction.

## Errors and future HTTP behavior

| Code | HTTP | Meaning |
|---|---:|---|
| `HISTORY_RESET_CONFIRMATION_INVALID` | 400 | The confirmation was not exactly `RESET HISTORY`; the repository was not called. |
| `HISTORY_RESET_ACTIVE_JOBS` | 409 | The repository found at least one queued or running job atomically; nothing was deleted. |
| `HISTORY_RESET_EXTERNAL_REFERENCES` | 409 | A table the app does not manage still references ECN rows. The message names each table; nothing was deleted. |
| `HISTORY_RESET_REFERENCED` | 409 | HANA refused the delete with a foreign-key violation (error 462). The transaction was rolled back; the server log names the table. |

The `/settings` page supplies the exact confirmation phrase. It fails closed until `GET /api/analysis-progress` establishes that queued and running counts are both zero. While work is active it polls serially; if the status request fails, reset remains disabled and the operator can retry the status check. A race-time HTTP 409 disables reset again and resumes status polling without deleting history.

After a successful reset, Settings stores the returned seven counts in session-only browser storage and navigates to `/ecns`. The empty historical list displays the exact counts once, removes the storage entry, and reminds the operator through the upload copy that the next upload requires a workbook.

## Owning files and verification

- `api/app/ecn/services/history.py` validates confirmation, owns service-level error mapping, and uses the shared guard.
- `api/app/ecn/services/lifecycle.py` defines the shared process mutation guard.
- `api/app/persistence/history_repository.py` owns the one-transaction active check, delete order, and preservation boundary.
- `ui/src/pages/settings/settings.html`, `settings.js`, and `settings.css` own the exact confirmation and fail-closed activity gate.
- `ui/src/pages/ecns/ecns.js` consumes the one-shot deletion summary after redirect.
- `tests/unit/test_history_service.py` covers confirmation, conflict, counts, and publication/reset serialization.
- `tests/unit/test_history_repository.py` covers SQL order, one-transaction behavior, active-job blocking, and rule preservation.
- `ui/tests/settings-page.test.js` and `ui/tests/e2e/shell.spec.js` cover browser safety and reset flow.

Run the focused verification with:

```bash
PYTHONPATH=api python -m pytest -q \
  tests/unit/test_history_service.py \
  tests/unit/test_history_repository.py \
  tests/unit/test_ingestion_service.py
python -m compileall -q api/app/ecn/services
npm --prefix ui test -- tests/settings-page.test.js tests/ecns-page.test.js
npm --prefix ui run test:e2e -- shell.spec.js
```
