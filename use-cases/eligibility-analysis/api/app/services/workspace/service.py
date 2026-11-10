"""Narrow facade for strict eligibility uploads and direct candidate handoff."""

import hashlib
from ...config.eligibility_config import EligibilitySettings
from ...models.workspace import WorkspaceValidationError
from ..eligibility.analysis import evaluate_offer
from .candidates import choose_candidates
from .schema import transaction


class WorkspaceService:
    """Coordinate existing business logic with explicit, durable source boundaries."""

    def __init__(self, analyses, runs):
        """Accept stores explicitly so tests and routes control database access."""
        self.analyses, self.runs = analyses, runs

    def analyze(self, content, filename, analysis_date, settings, request_key):
        """Strictly evaluate a single workbook and save original rows plus diagnostics."""
        if not request_key or len(request_key) > 128:
            raise WorkspaceValidationError("Provide a request key of 1–128 characters", "request_key")
        if not isinstance(settings, dict) or set(settings) - {"nddt", "teih", "isspur", "eligible_currencies"}:
            raise WorkspaceValidationError("Unknown eligibility setting", "settings")
        for key in ("nddt", "teih", "isspur"):
            if key in settings and (type(settings[key]) is not int or settings[key] < (1 if key == "teih" else 0)):
                raise WorkspaceValidationError("Provide a valid nonnegative threshold", f"settings.{key}")
        if "eligible_currencies" in settings:
            values = settings["eligible_currencies"]
            if not isinstance(values, list) or not values or any(not isinstance(v, str) or len(v) != 3 for v in values):
                raise WorkspaceValidationError("Provide a list of three-letter currencies", "settings.eligible_currencies")
            settings = {**settings, "eligible_currencies": sorted(set(v.upper() for v in values))}
        rules = EligibilitySettings(**settings)
        invoices, results, _, _ = evaluate_offer(content, filename, analysis_date, rules, strict=True)
        if not invoices:
            raise WorkspaceValidationError("The offer contains no invoice rows", "file")
        source_hash = hashlib.sha256(content).hexdigest()
        rows = []
        for invoice, result in zip(invoices, results):
            source_key = f"{source_hash}:{invoice.source_sheet}:{invoice.source_row_number}"
            rows.append(dict(row_id=hashlib.sha256(source_key.encode()).hexdigest(),
                             source_event_id=hashlib.sha256(source_key.encode()).hexdigest(),
                             source_row_number=invoice.source_row_number, source_sheet=invoice.source_sheet,
                             eligible=result.is_eligible, invoice=invoice.model_dump(mode="json"),
                             diagnostics=result.model_dump(mode="json")))
        # This transaction is also the diagnostic event log: no second historical write
        # can partially commit or inflate counts on a retried upload.
        return self.analyses.create(content, filename, analysis_date, rules.to_dict(), rows, request_key)

    def import_selection(self, content, filename, analysis_date, request_key):
        """Persist one upstream-approved extraction without an eligibility evaluation."""
        from .selection_import import selection_rows
        if not request_key or len(request_key) > 128:
            raise WorkspaceValidationError("Provide a request key of 1–128 characters", "request_key")
        rows, metadata = selection_rows(content)
        return self.analyses.create(content, filename, analysis_date, metadata, rows, request_key)

    def create_run(self, analysis_id, scope):
        """Snapshot eligible source IDs from one saved analysis without export/re-upload."""
        rows = choose_candidates(self.analyses.all_rows(analysis_id), scope)
        if not rows:
            raise WorkspaceValidationError("No eligible invoices are available for optimization")
        run = self.runs.create(analysis_id, [row["row_id"] for row in rows])
        analysis = self.analyses.get(analysis_id)
        if analysis['settings'].get('source_kind') == 'selection':
            from datetime import date, timedelta
            start = date.fromisoformat(analysis['analysis_date'])
            start += timedelta(days=(7-start.weekday()) % 7)
            associations = {}
            for row in rows:
                invoice = row['invoice']
                associations.setdefault(invoice['debtor_id'], set()).add(invoice['seller_id'])
            settings = dict(planning_start=start.isoformat(), horizon_weeks=12,
                seller_to_facility={row['invoice']['seller_id']:row['invoice']['seller_id'] for row in rows},
                customer_to_facility={key:next(iter(values)) for key,values in associations.items() if len(values)==1})
            run = self.runs.compare_and_swap(run['run_id'], run['revision'], {'settings':settings})
        return run

    def delete_analyses(self, analysis_ids):
        """Atomically delete selected uploads, their runs, exports, rows, and run artifacts."""
        requested_ids = list(dict.fromkeys(analysis_ids))
        if not requested_ids:
            return {"deleted_analysis_ids": [], "deleted_count": 0}
        from ..optimizer.artifact_store import OptimizerArtifactStore
        OptimizerArtifactStore(db_path=self.analyses.db_path, backend=self.analyses.backend)
        placeholders = ",".join("?" for _ in requested_ids)
        with transaction(self.analyses.backend, self.analyses.db_path) as cursor:
            cursor.execute(
                f"SELECT analysis_id FROM RECEIVABLES_ANALYSES WHERE analysis_id IN ({placeholders})",
                tuple(requested_ids),
            )
            existing = {row[0] for row in cursor.fetchall()}
            deleted_ids = [analysis_id for analysis_id in requested_ids if analysis_id in existing]
            if not deleted_ids:
                return {"deleted_analysis_ids": [], "deleted_count": 0}
            delete_placeholders = ",".join("?" for _ in deleted_ids)
            cursor.execute(
                f"SELECT run_id FROM RECEIVABLES_RUNS WHERE analysis_id IN ({delete_placeholders})",
                tuple(deleted_ids),
            )
            run_ids = [row[0] for row in cursor.fetchall()]
            if run_ids:
                run_placeholders = ",".join("?" for _ in run_ids)
                cursor.execute(
                    f"DELETE FROM optimizer_process_artifacts WHERE process_id IN ({run_placeholders})",
                    tuple(run_ids),
                )
            for table in (
                "RECEIVABLES_ANALYSIS_EXPORTS",
                "RECEIVABLES_SOURCE_ROWS",
                "RECEIVABLES_RUNS",
                "RECEIVABLES_ANALYSES",
            ):
                cursor.execute(
                    f"DELETE FROM {table} WHERE analysis_id IN ({delete_placeholders})",
                    tuple(deleted_ids),
                )
        return {"deleted_analysis_ids": deleted_ids, "deleted_count": len(deleted_ids)}
