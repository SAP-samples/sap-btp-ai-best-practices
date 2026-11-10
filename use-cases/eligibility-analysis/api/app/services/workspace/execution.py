"""Execute a saved preparation exactly once and preserve completed solver outcomes."""
from ...models.workspace import RevisionConflict


class ExecutionService:
    """Claim solver entry atomically; reporting failures cannot erase a successful plan."""

    def __init__(self,runs,solver,on_completed=None):
        """Inject the solver and optional artifact callback around durable state transitions."""
        self.runs,self.solver,self.on_completed=runs,solver,on_completed

    def execute(self,run_id,preparation_id):
        """Consume saved predictions only after a valid optimization claim."""
        run=self.runs.get(run_id)
        preparation=run.get('preparation') or {}
        if run['preparation_id']!=preparation_id: raise RevisionConflict('Preparation changed')
        if preparation.get('execution_claimed'): return run
        if run['status']!='optimizing': raise RevisionConflict('The run has not been approved for optimization')
        if any(row['source']!='rpt1' for row in preparation['predictions']) and not preparation.get('acknowledgement'):
            raise RevisionConflict('Fallback acknowledgement is required before solving')
        run=self.runs.compare_and_swap(run_id,run['revision'],{'preparation':{**preparation,'execution_claimed':True}})
        try:
            result=self.solver(run)
        except Exception as error:
            return self.runs.compare_and_swap(run_id,run['revision'],{'status':'failed','result':{
                'stage':'optimization','error':str(error),'retryable':True}})
        completed=self.runs.compare_and_swap(run_id,run['revision'],{'status':'completed','result':result})
        if self.on_completed:
            # Artifact service owns its own failure manifest; optimization stays complete.
            try: self.on_completed(completed)
            except Exception:
                import logging
                logging.getLogger(__name__).exception('Run completed but artifact generation failed')
        return completed

    def retry(self, run_id, expected_revision):
        """Explicitly retry an interrupted/failed solve with saved predictions and acknowledgement."""
        run = self.runs.get(run_id)
        preparation = run.get('preparation') or {}
        if run['status'] != 'failed' or not preparation.get('predictions'):
            raise RevisionConflict('Only a failed solve with saved predictions can resume')
        return self.runs.compare_and_swap(run_id, expected_revision, {'status': 'optimizing', 'result': None,
            'preparation': {**preparation, 'execution_claimed': False}})
