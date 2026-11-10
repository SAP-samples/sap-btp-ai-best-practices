"""Durable cancellation, interrupted-work recovery and one-time resume contracts."""
from test_workspace_preparation import PreparationTests
from app.services.workspace.preparation import PreparationService


class RecoveryTests(PreparationTests):
    """Extend gate fixtures with actual SQL recovery and malformed-result cases."""

    def test_restart_fails_active_work_but_preserves_pending(self):
        """Startup never reports abandoned work complete or spends model calls automatically."""
        active = self.preparation.start(self.run['run_id'], self.run['revision'])
        self.assertEqual(self.store.recover_interrupted(), 1)
        self.assertEqual(self.store.get(active['run_id'])['status'], 'failed')
        self.assertEqual(self.model_calls, 0)
        failed = self.store.get(active['run_id'])
        pending = self.preparation.prepare(failed['run_id'], failed['revision'])
        self.assertEqual(self.store.recover_interrupted(), 0)
        self.assertEqual(self.store.get(pending['run_id'])['status'], 'awaiting_lifetime_acknowledgement')

    def test_cancel_during_estimation_cannot_start_solver(self):
        """A late model response cannot overwrite a user cancellation."""
        active = self.preparation.start(self.run['run_id'], self.run['revision'])
        self.preparation.cancel(active['run_id'], active['revision'])
        self.assertEqual(self.preparation.finish(active)['status'], 'cancelled')
        self.assertEqual(self.solver_calls, 0)

    def test_unknown_prediction_id_fails_without_solving(self):
        """Malformed output cannot leave a run permanently estimating."""
        self.predictions[0]['row_id'] = 'outside'
        result = self.preparation.prepare(self.run['run_id'], self.run['revision'])
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(self.solver_calls, 0)

    def test_report_failure_keeps_completed_plan(self):
        """A downstream export failure never invalidates a successful solve."""
        def fail_report(run):
            """Simulate a renderer failure after persisted completion."""
            raise RuntimeError('renderer unavailable')
        self.execution.on_completed = fail_report
        self.predictions[1]['source'] = 'rpt1'
        with self.assertLogs('app.services.workspace.execution', level='ERROR'):
            result = self.preparation.prepare(self.run['run_id'], self.run['revision'])
        self.assertEqual(self.store.get(result['run_id'])['status'], 'completed')
