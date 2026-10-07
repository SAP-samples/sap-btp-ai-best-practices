"""Assistant model isolation and request-local source references."""
import json
import unittest
from unittest.mock import Mock
from app.a2a.model_config import resolve_assistant_model
from app.a2a.workspace_context import resolve_workspace_context
from app.a2a.tools.workspace_tools import summarize_workspace_run


class WorkspaceContextTests(unittest.TestCase):
    """Reject stale or cross-offer scope before invoking any model/tool."""

    def test_luna_is_independent_of_report_model(self):
        """Legacy report configuration must not defeat the requested assistant default."""
        self.assertEqual(resolve_assistant_model({'AICORE_MODEL':'gpt-4.1'}),'gpt-5.6-luna')

    def test_invalid_scope_and_stale_run(self):
        """Source rows and run revisions belong to one explicit analysis."""
        analyses=Mock();analyses.all_rows.return_value=[{'row_id':'a'}]
        runs=Mock();runs.get.return_value={'analysis_id':'offer','revision':3}
        for context in [{'analysis_id':'offer','row_ids':['outside']},{'analysis_id':'other','run_id':'r','revision':3},{'run_id':'r','revision':2}]:
            with self.assertRaises(ValueError):resolve_workspace_context(context,analyses,runs)
        self.assertEqual(resolve_workspace_context({'run_id':'r','revision':3},analyses,runs).analysis_id,'offer')

    def test_complete_model_run_exposes_recommendation_without_internal_claim(self):
        """A null acknowledgement is not an inconsistency when no fallback required consent."""
        run = {
            'run_id': 'run-1', 'status': 'completed', 'revision': 4,
            'row_ids': ['a', 'b'], 'readiness_issues': [],
            'preparation': {
                'predictions': [
                    {'row_id': 'a', 'source': 'rpt1'},
                    {'row_id': 'b', 'source': 'rpt1'},
                ],
                'fallback_count': 0, 'acknowledgement': None,
                'execution_claimed': True,
                'history': {'dataset_id': 'reference-v1'},
            },
            'result': {
                'solver_status': 'FEASIBLE', 'objective_amount': 1250.0,
                'currency': 'EUR', 'week_starts': ['2025-02-03'],
                'exposure': {}, 'selected': [{'row_id': 'a'}],
                'not_selected': [{'row_id': 'b'}],
            },
        }

        summary = summarize_workspace_run(run)

        self.assertEqual(summary['lifetime_estimation']['acknowledgement_status'], 'not_required')
        self.assertEqual(summary['lifetime_estimation']['rpt1_count'], 2)
        self.assertTrue(summary['recommendation']['available'])
        self.assertEqual(summary['recommendation']['recommended_amount'], 1250.0)
        self.assertNotIn('execution_claimed', json.dumps(summary))
        self.assertNotIn('preparation', summary['run'])

    def test_fallback_acknowledgement_statuses_are_explicit(self):
        """Pending, accepted and impossible saved states remain distinguishable to the model."""
        base = {
            'run_id': 'run-2', 'revision': 2, 'row_ids': ['a'],
            'readiness_issues': [], 'result': None,
            'preparation': {
                'predictions': [{'row_id': 'a', 'source': 'fallback_default_weeks'}],
                'fallback_count': 1, 'acknowledgement': None,
            },
        }
        pending = summarize_workspace_run({**base, 'status': 'awaiting_lifetime_acknowledgement'})
        accepted = summarize_workspace_run({**base, 'status': 'optimizing', 'preparation': {
            **base['preparation'], 'acknowledgement': {'accepted_at': '2026-09-08T08:00:00Z'}}})
        inconsistent = summarize_workspace_run({**base, 'status': 'completed'})

        self.assertEqual(pending['lifetime_estimation']['acknowledgement_status'], 'pending')
        self.assertEqual(accepted['lifetime_estimation']['acknowledgement_status'], 'accepted')
        self.assertEqual(inconsistent['lifetime_estimation']['acknowledgement_status'], 'inconsistent')
