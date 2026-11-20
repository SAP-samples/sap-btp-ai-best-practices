# One-shot selected ECN import

## Purpose

The ECN inbox imports a fresh source file without registering or retaining the
complete export. The browser parses the JSON in a short-lived Web Worker, lets
the operator search and select ECNs across preview pages, and sends only the
selected records to the existing ingestion API. Importing makes no AI calls.

## Inputs and outputs

| Input | Processing | Output |
|---|---|---|
| One UTF-8 `.json` array of ECN objects | Worker-only parse, searchable 25-row preview, whole-file Select/Deselect all | No server-side artifact until ECNs are selected |
| One or more selected rows | Selected rows are serialized in original order and uploaded in batches of at most 25 | HANA-backed `READY` ECN snapshots whose source artifacts contain only their selected batch |

The original filename is retained for every selected batch. The complete JSON
exists only in the worker while the modal is open. Success or cancellation
terminates the worker and clears its file, preview, selection, and retry state.

## Retry behavior

Each 25-record batch receives its own idempotency UUID. Completed batches stay
committed if a later batch fails. The failed batch and all remaining batches
stay in the dialog with their original UUIDs, so **Import selected ECNs** safely
continues without re-creating successful imports. Opening **Import new ECN**
later always starts a new session.

After all batches succeed, the inbox returns to page 1 sorted by newest upload
so the newly selected ECNs appear above earlier imports. Unselected source
records are discarded and cannot be recovered from the application.

Lifecycle display follows this order: `State.Display`, `State.Value`, a string
`State`, then **Not available**. Source values are rendered as text only.

The API receives each selected-only batch through the existing upload boundary:

```http
POST /api/uploads
Content-Type: multipart/form-data
request_id=<stable-batch-uuid>&ecn_file=<selected-only-json-batch>
```

## Related files

- `ui/src/pages/ecns/ecns.html` — statistics-first layout and accessible import dialog.
- `ui/src/pages/ecns/ecns.js` — refreshes the default newest-first list after import.
- `ui/src/components/export-import.js` — dialog state, selection, resumable batching, and cleanup.
- `ui/src/components/ecn-import-worker.js` — worker-owned parse, preview, and selected serialization.
- `ui/src/modules/ecn-import-data.js` — root validation, lifecycle display, search, and serialization.
- `api/app/routers/uploads.py` — selected-only multipart ingestion boundary.

## Verification

```bash
npm test --prefix ui -- --run tests/export-import.test.js tests/ecns-page.test.js
npm run build --prefix ui
E2E_PORT=4174 npm run test:e2e --prefix ui
PYTHONPATH=api .venv/bin/python -m pytest -q tests/integration/test_snapshot_workspace_live.py
```

The live HANA test is opt-in and uses isolated suffixed tables. No deployment is
part of this workflow.
