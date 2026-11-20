# Regenerate an ECN summary

## Purpose

The **Generate summary** action on an ECN detail page and **Generate summaries**
for selected incoming ECNs request a fresh AI assessment even when an ECN already
has a completed summary. This lets a user explicitly rerun a summary for the same
stored ECN occurrence.

## Input and output

Both controls call `POST /api/summary-batches` with a new `request_id` and either
explicit ECN row IDs or a filtered selection. The server resolves the selected
IDs, requires current source projections, and queues `READY`, `FAILED`, `BLOCKED`,
and `COMPLETED` jobs. A `QUEUED` or `RUNNING` job is shared with the new batch
instead of starting a concurrent duplicate. Replaying the same `request_id`
returns its existing batch without scheduling another call.

When a completed job is queued again, its previous analysis remains available
and is labeled as previous on the detail page. A successful generation replaces
that row atomically; a failure leaves it available and the job retryable.
Prior attempt and provider-call records remain available for diagnostics. The
job keeps its pinned model configuration and moves to the current staged prompt
and schema versions. See [Staged ECN impact analysis](ecn-staged-impact-analysis.md).

```python
# The completed-to-queued transition retains the accepted analysis until
# a validated replacement commits.
result, job_ids = workspace_repository.create_batch(request_id, selected_ecn_ids)
runtime.schedule_jobs(job_ids)
```

## Related files and check

- `api/app/persistence/workspace_repository.py`: transactional batch scheduling.
- `api/app/routers/workspace_actions.py`: shared endpoint for both controls.
- `ui/src/pages/ecns/ecns.js` and `ui/src/pages/ecn-detail/ecn-detail.js`: actions
  and progress display.
- `tests/unit/test_snapshot_workspace.py`, `tests/integration/test_ecn_workflow.py`,
  `ui/tests/ecns-page.test.js`, and `ui/tests/ecn-detail-page.test.js`:
  completed-summary regressions.

Run `PYTHONPATH=api .venv/bin/python -m pytest -q tests/unit/test_snapshot_workspace.py tests/integration/test_ecn_workflow.py`
and, from `ui/`, `npm test -- --run tests/ecns-page.test.js tests/ecn-detail-page.test.js`.
