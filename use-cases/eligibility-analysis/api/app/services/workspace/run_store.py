"""Persist run snapshots and claim revisions with atomic SQL predicates."""

import json
from uuid import uuid4
from ...models.workspace import RevisionConflict, RunStatus
from .schema import decode_json, ensure_schema, transaction


class RunStore:
    """Own durable run state; callers cannot overwrite stale or completed inputs."""

    def __init__(self, backend, db_path=None):
        """Initialize the supplied HANA backend or explicit isolated test database."""
        self.backend, self.db_path = backend, db_path
        ensure_schema(backend, db_path)

    def create(self, analysis_id, row_ids):
        """Create one draft with an immutable exact list of eligible source IDs."""
        if not row_ids or len(set(row_ids)) != len(row_ids):
            raise ValueError("A run needs a nonempty unique candidate list")
        result = RunStatus(run_id=str(uuid4()), analysis_id=analysis_id, revision=0,
                           status="draft", row_ids=row_ids, readiness_issues=[{
                               "path": "settings", "message": "Configure credit limits and opening exposure"}]).model_dump()
        with transaction(self.backend, self.db_path) as cursor:
            cursor.execute("INSERT INTO RECEIVABLES_RUNS (run_id, analysis_id, revision, status, payload) "
                           "VALUES (?, ?, ?, ?, ?)", (result["run_id"], analysis_id, 0, "draft", json.dumps(result)))
        return result

    def get(self, run_id):
        """Read the authoritative run snapshot or raise LookupError if absent."""
        with transaction(self.backend, self.db_path) as cursor:
            cursor.execute("SELECT payload FROM RECEIVABLES_RUNS WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if row is None:
                raise LookupError("Run not found")
            return decode_json(row[0])

    def compare_and_swap(self, run_id, expected_revision, changes):
        """Apply an allowed snapshot update only if its revision is still current."""
        if set(changes) - {"status", "settings", "readiness_issues", "preparation_id", "preparation", "result"}:
            raise ValueError("Cannot change a run's source identity")
        current = self.get(run_id)
        if current["revision"] != expected_revision or current["status"] in ("completed", "cancelled"):
            raise RevisionConflict("Run revision changed or inputs are immutable")
        if "settings" in changes and current["status"] != "draft":
            raise RevisionConflict("Only draft settings can be edited")
        updated = RunStatus(**{**current, **changes, "revision": expected_revision + 1}).model_dump()
        with transaction(self.backend, self.db_path) as cursor:
            cursor.execute("UPDATE RECEIVABLES_RUNS SET revision = ?, status = ?, payload = ? "
                           "WHERE run_id = ? AND revision = ? AND status = ?",
                           (updated["revision"], updated["status"], json.dumps(updated),
                            run_id, expected_revision, current["status"]))
            if cursor.rowcount != 1:
                raise RevisionConflict("Run was changed by another request")
        return updated

    def list_for_analysis(self, analysis_id):
        """List durable runs for reopening, including pending acknowledgements."""
        with transaction(self.backend, self.db_path) as cursor:
            cursor.execute('SELECT payload FROM RECEIVABLES_RUNS WHERE analysis_id = ?', (analysis_id,))
            return [decode_json(row[0]) for row in cursor.fetchall()]

    def recover_interrupted(self):
        """Mark abandoned active stages failed on the single workspace worker's startup.

        Call once before accepting work. Pending acknowledgements are deliberately
        untouched. This deployment uses one API process; multiple workers require
        a distributed lease before enabling this startup recovery.
        """
        with transaction(self.backend, self.db_path) as cursor:
            cursor.execute("SELECT payload FROM RECEIVABLES_RUNS WHERE status IN ('estimating_lifetimes','optimizing')")
            active = [decode_json(row[0]) for row in cursor.fetchall()]
        for run in active:
            try:
                self.compare_and_swap(run['run_id'], run['revision'], {'status': 'failed', 'result': {
                    'stage': run['status'], 'error': 'Work was interrupted by an API restart. Retry explicitly.', 'retryable': True}})
            except RevisionConflict:
                continue
        return len(active)
