"""Persist model preparation and require explicit acknowledgement of every fallback."""
import hashlib
import json
import math
from decimal import Decimal
from datetime import datetime,timezone
from uuid import uuid4
from ...models.workspace import RevisionConflict,WorkspaceValidationError


def preparation_status(predictions):
    """One non-model prediction is sufficient to block entry into optimization."""
    if not predictions: raise ValueError('Cannot prepare an empty candidate set')
    return 'awaiting_lifetime_acknowledgement' if any(row['source']!='rpt1' for row in predictions) else 'optimizing'


def normalize_predictions(row_ids,predictions):
    """Require complete source identity and finite positive durations, exposing missing values."""
    by_id={}
    for prediction in predictions:
        identity=prediction.get('row_id')
        if identity not in row_ids or identity in by_id: raise ValueError('Estimator returned unknown or duplicate source row IDs')
        by_id[identity]=prediction
    output=[]
    for identity in row_ids:
        prediction=dict(by_id.get(identity,{}))
        try:
            duration=float(prediction.get('expected_lifetime_days'))
            valid=math.isfinite(duration) and duration>0
        except (ValueError,TypeError): valid=False
        if not valid or prediction.get('source')!='rpt1':
            prediction.update(expected_lifetime_days=28,source='fallback_default_weeks',
                              reason=prediction.get('reason') or 'A valid model lifetime is unavailable')
        else: prediction['expected_lifetime_days']=max(1,int(math.ceil(duration)))
        prediction['row_id']=identity
        confidence=prediction.get('confidence')
        if confidence is not None and (not isinstance(confidence,(float,int)) or not math.isfinite(confidence)):
            prediction['confidence']=None
        output.append(prediction)
    return output


class PreparationService:
    """Own durable model preparation, exact acknowledgement and monotonic solver handoff."""

    def __init__(self,runs,estimator,execution):
        """Inject external estimation/execution boundaries around the real persisted state."""
        self.runs,self.estimator,self.execution=runs,estimator,execution

    def start(self,run_id,expected_revision):
        """Claim a ready draft before launching expensive model work."""
        run=self.runs.get(run_id)
        if run['status'] not in ('draft','failed') or run['revision']!=expected_revision:
            raise RevisionConflict('Only a current draft or explicitly retried failure can prepare')
        if run['readiness_issues']:
            error=WorkspaceValidationError('Review credit settings before optimizing','settings')
            error.fields=run['readiness_issues'];raise error
        payload=json.dumps({'row_ids':run['row_ids'],'settings':run['settings']},sort_keys=True)
        preparation={'input_hash':hashlib.sha256(payload.encode()).hexdigest(),
                     'started_at':datetime.now(timezone.utc).isoformat(),'input_revision':expected_revision,
                     'predictions':[],'acknowledgement':None,'execution_claimed':False}
        return self.runs.compare_and_swap(run_id,expected_revision,{'status':'estimating_lifetimes',
            'preparation_id':str(uuid4()),'preparation':preparation,'result':None})

    def prepare(self,run_id,expected_revision):
        """Synchronously prepare a ready draft; API workers call start/finish separately."""
        return self.finish(self.start(run_id,expected_revision))

    def finish(self,claimed):
        """Persist normalized prediction rows before publishing the acknowledgement state."""
        try:
            predictions,metadata=self.estimator(claimed)
        except Exception as error:
            predictions=[];metadata={'status':'estimation_failed','error':type(error).__name__,
                                     'message':'Lifetime estimation failed; review the four-week fallback assumptions'}
        try:
            predictions=normalize_predictions(claimed['row_ids'],predictions)
        except (ValueError, TypeError, KeyError) as error:
            return self.runs.compare_and_swap(claimed['run_id'],claimed['revision'],{'status':'failed',
                'result':{'stage':'estimation','error':str(error),'retryable':True}})
        fallback=[row for row in predictions if row['source']!='rpt1']
        totals={}
        for row in fallback:
            if row.get('original_amount') is not None:
                currency=row['original_currency']
                totals[currency]=str(Decimal(totals.get(currency,'0'))+Decimal(row['original_amount']))
        preparation={**claimed['preparation'],'predictions':predictions,'history':metadata,
                     'fallback_amounts_by_currency':totals,
                     'fallback_count':len(fallback),'prepared_at':datetime.now(timezone.utc).isoformat()}
        state=preparation_status(predictions)
        try:
            run=self.runs.compare_and_swap(claimed['run_id'],claimed['revision'],{'status':state,'preparation':preparation})
        except RevisionConflict:
            return self.runs.get(claimed['run_id']) # e.g. user cancelled while the model was running
        return self.execution.execute(run['run_id'],run['preparation_id']) if state=='optimizing' else run

    def acknowledge(self,run_id,preparation_id,expected_revision,accepted,actor_id=None,execute=True):
        """Accept only the displayed preparation and resume using its saved predictions."""
        run=self.runs.get(run_id)
        if run['preparation_id']!=preparation_id: raise RevisionConflict('Preparation changed; review its current assumptions')
        previous=run['preparation'].get('acknowledgement')
        if previous and previous['accepted_revision']==expected_revision and accepted:
            return run
        if run['status']!='awaiting_lifetime_acknowledgement' or run['revision']!=expected_revision:
            raise RevisionConflict('Preparation revision changed')
        if not accepted: return run
        preparation={**run['preparation'],'acknowledgement':{
            'accepted_at':datetime.now(timezone.utc).isoformat(),'actor_id':actor_id,
            'accepted_revision':expected_revision,'input_hash':run['preparation']['input_hash'],
            'fallback_row_ids':[row['row_id'] for row in run['preparation']['predictions'] if row['source']!='rpt1'],
            'fallback_days':28}}
        try:
            run=self.runs.compare_and_swap(run_id,expected_revision,{'status':'optimizing','preparation':preparation})
        except RevisionConflict:
            current=self.runs.get(run_id)
            acknowledgement=(current.get('preparation') or {}).get('acknowledgement') or {}
            if current['preparation_id']==preparation_id and acknowledgement.get('accepted_revision')==expected_revision:
                return current
            raise
        return self.execution.execute(run_id,preparation_id) if execute else run

    def cancel(self,run_id,expected_revision):
        """Cancel before solver entry; cancellation never fabricates a completed result."""
        run=self.runs.get(run_id)
        if run['status'] not in ('draft','estimating_lifetimes','awaiting_lifetime_acknowledgement'):
            raise RevisionConflict('This run can no longer be cancelled')
        return self.runs.compare_and_swap(run_id,expected_revision,{'status':'cancelled'})
