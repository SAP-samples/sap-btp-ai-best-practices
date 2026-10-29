"""The backend, not the browser, must enforce prediction-fallback acknowledgement."""
import tempfile
import unittest
from pathlib import Path
from app.services.database.backend import BackendType,DatabaseBackend
from app.services.workspace.run_store import RunStore
from app.services.workspace.preparation import PreparationService,preparation_status
from app.services.workspace.execution import ExecutionService


class PreparationTests(unittest.TestCase):
    """Use injected model/solver boundaries and real SQL snapshots to verify the gate."""

    def setUp(self):
        """Create a ready draft with a model response that uses a fallback."""
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.store=RunStore(DatabaseBackend(BackendType.SQLITE),Path(self.temp.name)/'test.db')
        self.run=self.store.create('analysis',['a','b'])
        self.run=self.store.compare_and_swap(self.run['run_id'],0,{'settings':{'ready':True},'readiness_issues':[]})
        self.model_calls=0;self.solver_calls=0
        self.predictions=[{'row_id':'a','expected_lifetime_days':35,'source':'rpt1'},
                          {'row_id':'b','expected_lifetime_days':28,'source':'fallback_default_weeks','reason':'Missing prediction'}]
        self.execution=ExecutionService(self.store,self.solve)
        self.preparation=PreparationService(self.store,self.estimate,self.execution)

    def estimate(self,run):
        """Represent the external estimator boundary with deterministic test outputs."""
        self.model_calls+=1
        return self.predictions,{'dataset_id':'history-v1'}

    def solve(self,run):
        """Count real orchestration entries into the expensive solver boundary."""
        self.solver_calls+=1
        return {'selected_ids':['a'],'solver_status':'OPTIMAL'}

    def test_fallback_waits_then_acknowledged_resume_solves_once(self):
        """A fallback may not reach the solver until its exact preparation is accepted."""
        pending=self.preparation.prepare(self.run['run_id'],self.run['revision'])
        self.assertEqual(pending['status'],'awaiting_lifetime_acknowledgement')
        self.assertEqual(self.solver_calls,0)
        with self.assertRaises(ValueError):
            self.preparation.acknowledge(pending['run_id'],'wrong',pending['revision'],True,None)
        result=self.preparation.acknowledge(pending['run_id'],pending['preparation_id'],pending['revision'],True,None)
        self.assertEqual(result['status'],'completed')
        self.assertEqual((self.model_calls,self.solver_calls),(1,1))
        self.preparation.acknowledge(pending['run_id'],pending['preparation_id'],pending['revision'],True,None)
        self.assertEqual(self.solver_calls,1)
        self.assertIsNotNone(result['preparation']['acknowledgement']['accepted_at'])

    def test_complete_predictions_need_no_fallback_approval(self):
        """All-model preparation proceeds directly to the solver once."""
        self.predictions[1]['source']='rpt1'
        result=self.preparation.prepare(self.run['run_id'],self.run['revision'])
        self.assertEqual(result['status'],'completed')
        self.assertEqual(self.solver_calls,1)

    def test_decline_and_missing_prediction_never_solve(self):
        """Missing rows become explicit four-week fallbacks; declining leaves them pending."""
        self.predictions=self.predictions[:1]
        pending=self.preparation.prepare(self.run['run_id'],self.run['revision'])
        result=self.preparation.acknowledge(pending['run_id'],pending['preparation_id'],pending['revision'],False,None)
        self.assertEqual(result['status'],'awaiting_lifetime_acknowledgement')
        self.assertEqual(result['preparation']['predictions'][1]['expected_lifetime_days'],28)
        self.assertEqual(self.solver_calls,0)

    def test_invalid_duration_is_not_silently_treated_as_a_model_prediction(self):
        """Nonfinite or nonpositive values are fallbacks with an explicit reason."""
        self.predictions[0]['expected_lifetime_days']=float('nan')
        result=self.preparation.prepare(self.run['run_id'],self.run['revision'])
        self.assertEqual(result['preparation']['fallback_count'],2)
        self.assertEqual(self.solver_calls,0)

    def test_new_service_instance_can_resume_persisted_pending_state(self):
        """Pending acknowledgement survives process-local service replacement."""
        pending=self.preparation.prepare(self.run['run_id'],self.run['revision'])
        restarted=PreparationService(self.store,self.estimate,self.execution)
        self.assertEqual(restarted.acknowledge(pending['run_id'],pending['preparation_id'],pending['revision'],True,None)['status'],'completed')
        self.assertEqual(self.model_calls,1)
