"""Snapshot-bound funding outcomes and exports do not change eligibility."""
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from app.services.database.backend import BackendType, DatabaseBackend
from app.services.workspace.run_store import RunStore
from app.services.optimizer.artifact_store import OptimizerArtifactStore
from app.services.workspace.artifacts import WorkspaceArtifacts, funding_outcome, report_markdown


class ResultTests(unittest.TestCase):
    """Exercise export bytes and failures against explicitly isolated SQL storage."""

    def setUp(self):
        """Save a small completed plan with separate source and solver populations."""
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        db = DatabaseBackend(BackendType.SQLITE); path = Path(self.temp.name) / 'results.db'
        self.runs = RunStore(db, path)
        run = self.runs.create('analysis', ['a', 'b'])
        self.run = self.runs.compare_and_swap(run['run_id'], 0, {'status': 'completed', 'result': {
            'solver_status': 'FEASIBLE', 'objective_amount': 20, 'selected': [{'row_id': 'a', 'Original Amount': '20'}],
            'not_selected': [{'row_id': 'b'}], 'pre_excluded': [], 'weekly_plan': [], 'exposure': {}}})
        self.artifacts = WorkspaceArtifacts(self.runs, OptimizerArtifactStore(path, backend=db))

    def test_outside_scope_is_not_solver_rejection(self):
        """Preserve the distinction between unconsidered, unselected and selected."""
        self.assertEqual(funding_outcome('outside', {'a','b'}, {'a'}), 'not_in_run')
        self.assertEqual(funding_outcome('b', {'a','b'}, {'a'}), 'not_selected')

    def test_report_failure_preserves_other_files_and_completed_run(self):
        """Retrying report generation consumes saved results and never runs optimization."""
        with patch('app.services.workspace.artifacts.render_pdf', side_effect=RuntimeError('renderer')):
            manifest = self.artifacts.generate_report(self.run['run_id'])
        self.assertEqual(self.runs.get(self.run['run_id'])['status'], 'completed')
        self.assertEqual(next(item for item in manifest if item['artifact_id']=='report-pdf')['status'], 'failed')
        with patch('app.services.workspace.artifacts.render_pdf', return_value=b'%PDF-example'):
            self.artifacts.generate_report(self.run['run_id'])
        content = self.artifacts.download(self.run['run_id'], 'all-files')
        self.assertIn('selected.xlsx', zipfile.ZipFile(io.BytesIO(content)).namelist())
        with self.assertRaises(LookupError): self.artifacts.download(self.run['run_id'], '../../secret')

    def test_report_describes_a_recommendation_without_execution_claims(self):
        """Keep customer-facing report language within the optimizer's actual scope."""
        markdown = report_markdown(self.run)

        self.assertIn('# Invoice Recommendation Report', markdown)
        self.assertIn('recommended amount', markdown)
        self.assertNotIn('funding executed', markdown.lower())
        self.assertNotIn('disbursed', markdown.lower())

    def test_report_handles_an_empty_candidate_scope(self):
        """Render a zero-candidate saved result without dividing by zero."""
        empty = dict(self.run)
        empty['row_ids'] = []
        empty['result'] = dict(self.run['result'], selected=[], not_selected=[])

        self.assertIn('0 of 0 invoices', report_markdown(empty))
