"""Authorized additive workspace endpoints; business services own state and validation."""

import json
import logging
from datetime import date
from functools import lru_cache
from threading import Lock

_ORCHESTRATION_LOCK = Lock()
# ponytail: process-local serialization matches the single API instance; use a
# distributed database lock before horizontally scaling artifact generation.
_ARTIFACT_MUTATION_LOCK = Lock()
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from ..models.workspace import InsightsRequest
from ..services.workspace.insights import WorkspaceInsights
from ..services.workspace.analysis_exports import save_analysis_exports, read_analysis_export
from ..models.workspace import SettingsRequest
from ..services.workspace.settings import preview_run_settings, save_run_settings
from ..services.workspace.settings_import import import_settings
from ..models.workspace import CreateRun, DeleteAnalyses, RevisionConflict, WorkspaceValidationError
from ..security import get_api_key
from ..services.database.backend import get_backend
from ..services.workspace.analysis_store import AnalysisStore
from ..services.workspace.run_store import RunStore
from ..services.workspace.service import WorkspaceService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/workspace", tags=["workspace"], dependencies=[Depends(get_api_key)])


@lru_cache(maxsize=1)
def get_workspace_service():
    """Initialize HANA-backed workspace tables before accepting the first request."""
    backend = get_backend()
    if not backend.is_hana:
        raise HTTPException(503, detail={"code": "storage_unavailable", "message": "Configure HANA for the workspace"})
    try:
        return WorkspaceService(AnalysisStore(backend), RunStore(backend))
    except Exception:
        logger.exception("Workspace database initialization failed")
        raise HTTPException(503, detail={"code": "storage_unavailable", "message": "Workspace storage is unavailable"}) from None


def domain_call(function, *args, **kwargs):
    """Translate expected domain failures into consistent, non-SQL HTTP errors."""
    try:
        return function(*args, **kwargs)
    except LookupError:
        raise HTTPException(404, detail={"code": "not_found"}) from None
    except RevisionConflict as error:
        raise HTTPException(409, detail={"code": "revision_conflict", "message": str(error)}) from None
    except ValueError as error:
        fields = error.fields if isinstance(error, WorkspaceValidationError) else [{"path": "file", "message": str(error), "row_ids": []}]
        raise HTTPException(422, detail={"code": "validation_error", "fields": fields}) from None


@router.post("/analyses")
def create_analysis(file: UploadFile = File(...), analysis_date: date = Form(...),
                    settings: str = Form("{}"), request_key: str = Form(...),
                    service=Depends(get_workspace_service)):
    """Analyze one uploaded offer with explicit date/settings and a retry key."""
    parsed = domain_call(json.loads, settings)
    return domain_call(service.analyze, file.file.read(), file.filename or "offer.xlsx", analysis_date, parsed, request_key)


@router.post("/selections")
def import_selection(file: UploadFile = File(...), analysis_date: date = Form(...),
                     request_key: str = Form(...), service=Depends(get_workspace_service)):
    """Upload already eligible candidates independently of the eligibility parser."""
    return domain_call(service.import_selection, file.file.read(), file.filename or "extraction.xlsx",
                       analysis_date, request_key)


@router.get("/analyses")
def list_analyses(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
                  service=Depends(get_workspace_service)):
    """Return saved analysis summaries in newest-first order."""
    return domain_call(service.analyses.list_analyses, limit, offset)


@router.delete("/analyses")
def delete_analyses(request: DeleteAnalyses, service=Depends(get_workspace_service)):
    """Delete selected uploads and every HANA record owned by their workflows."""
    with _ARTIFACT_MUTATION_LOCK:
        return domain_call(service.delete_analyses, [str(analysis_id) for analysis_id in request.analysis_ids])


@router.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: str, service=Depends(get_workspace_service)):
    """Reopen a saved analysis and its immutable rule snapshot."""
    result = domain_call(service.analyses.get, analysis_id)
    rows = domain_call(service.analyses.all_rows, analysis_id)
    result['filter_options'] = {key:sorted({row['invoice'][key] for row in rows if row['invoice'].get(key)})
                                for key in ('seller_id','debtor_id','programa','insurer_id','original_currency')}
    return result


@router.get("/analyses/{analysis_id}/invoices")
def get_invoices(analysis_id: str, search: str = "", status: str = "", seller_id: str = "",
                 debtor_id: str = "", programa: str = "", insurer_id: str = "", original_currency: str = "",
                 limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
                 service=Depends(get_workspace_service)):
    """Filter the table view without implicitly changing any funding run's scope."""
    filters = dict(search=search, status=status, seller_id=seller_id, debtor_id=debtor_id,
                   programa=programa, insurer_id=insurer_id, original_currency=original_currency)
    return domain_call(service.analyses.list_rows, analysis_id, filters, limit, offset)


@router.post("/runs")
def create_run(request: CreateRun, service=Depends(get_workspace_service)):
    """Create a draft using all eligible rows or an explicitly selected eligible subset."""
    return domain_call(service.create_run, request.analysis_id, request.candidate_scope)


@router.get("/runs/{run_id}")
def get_run(run_id: str, service=Depends(get_workspace_service)):
    """Return the authoritative revision, readiness issues, preparation and results."""
    return domain_call(service.runs.get, run_id)


@router.post("/analyses/{analysis_id}/insights")
def get_insights(analysis_id: str, request: InsightsRequest, service=Depends(get_workspace_service)):
    """Compare an exact current population against prior unique source events."""
    snapshot = domain_call(WorkspaceInsights(service.analyses).analyze, analysis_id,
                           request.row_ids, request.filters, request.lookback_days)
    token = save_analysis_exports(service.analyses, analysis_id, snapshot)
    return {key:value for key,value in {**snapshot,'scope_token':token}.items() if key != 'current_rows'}


@router.get("/analyses/{analysis_id}/exports/{kind}")
def get_analysis_export(analysis_id: str, kind: str, scope_token: str, service=Depends(get_workspace_service)):
    """Download an authorized immutable file matching the displayed diagnostic scope."""
    content = domain_call(read_analysis_export, service.analyses, analysis_id, scope_token, kind)
    pdf = kind == 'insights-pdf'
    return Response(content, media_type='application/pdf' if pdf else 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    headers={'Content-Disposition':f'attachment; filename="{kind}.{ "pdf" if pdf else "xlsx"}"'})


@router.post('/runs/{run_id}/settings/preview')
def preview_settings(run_id:str, request:SettingsRequest, service=Depends(get_workspace_service)):
    """Validate credit settings and project opening exposure without saving edits."""
    return domain_call(preview_run_settings,service,run_id,request.expected_revision,request.settings)


@router.put('/runs/{run_id}/settings')
def save_settings(run_id:str, request:SettingsRequest, service=Depends(get_workspace_service)):
    """Save reviewed manual settings with compare-and-swap revision protection."""
    return domain_call(save_run_settings,service,run_id,request.expected_revision,request.settings)


@router.post('/runs/{run_id}/settings/import')
def preview_settings_import(run_id:str,file:UploadFile=File(...),expected_revision:int=Form(...),service=Depends(get_workspace_service)):
    """Parse uploaded settings into a reviewable preview without committing a run change."""
    run=domain_call(service.runs.get,run_id)
    draft=domain_call(import_settings,file.file.read(),file.filename or '',run['settings'])
    return domain_call(preview_run_settings,service,run_id,expected_revision,draft)


@lru_cache(maxsize=1)
def _create_orchestration():
    """Wire HANA context, the saved preparation gate and bounded worker execution."""
    from ..services.lifecycle.store import LifecycleStore
    from ..services.workspace.runtime import WorkspaceRuntime
    from ..services.workspace.execution import ExecutionService
    from ..services.workspace.preparation import PreparationService
    from ..services.workspace.jobs import WorkspaceJobs
    service = get_workspace_service()
    runtime = WorkspaceRuntime(service, LifecycleStore(service.runs.backend))
    service.runs.recover_interrupted()
    execution = ExecutionService(service.runs, runtime.solve,
        lambda run: generate_workspace_report(run["run_id"]))
    return PreparationService(service.runs, runtime.estimate, execution), WorkspaceJobs(), runtime


def get_orchestration():
    """Serialize singleton initialization so recovery cannot interrupt another first request."""
    with _ORCHESTRATION_LOCK:
        return _create_orchestration()


@router.get('/analyses/{analysis_id}/runs')
def list_analysis_runs(analysis_id: str, service=Depends(get_workspace_service)):
    """Reopen exact saved drafts, pending assumptions or immutable completed results."""
    domain_call(service.analyses.get, analysis_id)
    return {'items': service.runs.list_for_analysis(analysis_id)}


from ..models.workspace import RevisionRequest, AcknowledgementRequest


@router.post('/runs/{run_id}/prepare')
def prepare_run(run_id: str, request: RevisionRequest):
    """Persist a preparation claim and estimate in a bounded background worker."""
    preparation, jobs, runtime = get_orchestration()
    domain_call(runtime.inputs, domain_call(preparation.runs.get, run_id))
    try:
        return domain_call(jobs.submit, lambda: preparation.start(run_id, request.expected_revision), preparation.finish)
    except RuntimeError as error:
        raise HTTPException(429, detail={'message': str(error)}) from None


@router.post('/runs/{run_id}/acknowledgement')
def acknowledge_run(run_id: str, request: AcknowledgementRequest):
    """Claim explicit fallback acceptance; solve using only the persisted predictions."""
    preparation, jobs, _ = get_orchestration()
    if not request.accepted:
        return domain_call(preparation.acknowledge, run_id, request.preparation_id, request.expected_revision, False)
    try:
        return domain_call(jobs.submit,
            lambda: preparation.acknowledge(run_id, request.preparation_id, request.expected_revision, True, execute=False),
            lambda run: preparation.execution.execute(run['run_id'], run['preparation_id']))
    except RuntimeError as error:
        raise HTTPException(429, detail={'message': str(error)}) from None


@router.post('/runs/{run_id}/cancel')
def cancel_run(run_id: str, request: RevisionRequest):
    """Cancel a draft, pending assumption review or in-flight estimation before solving."""
    preparation, _, _ = get_orchestration()
    return domain_call(preparation.cancel, run_id, request.expected_revision)


@router.post('/runs/{run_id}/retry-solve')
def retry_saved_solve(run_id: str, request: RevisionRequest):
    """Retry only the solver stage; do not spend another lifetime estimation call."""
    preparation, jobs, _ = get_orchestration()
    try:
        return domain_call(jobs.submit, lambda: preparation.execution.retry(run_id, request.expected_revision),
            lambda run: preparation.execution.execute(run['run_id'], run['preparation_id']))
    except RuntimeError as error:
        raise HTTPException(429, detail={'message': str(error)}) from None


@lru_cache(maxsize=1)
def get_workspace_artifacts():
    """Use the existing HANA artifact store for all run-bound download bytes."""
    from ..services.optimizer.artifact_store import OptimizerArtifactStore
    from ..services.workspace.artifacts import WorkspaceArtifacts
    service = get_workspace_service()
    return WorkspaceArtifacts(service.runs, OptimizerArtifactStore(backend=service.runs.backend), service.analyses)


def generate_workspace_report(run_id: str):
    """Serialize file generation with whole-workflow deletion in this API process."""
    with _ARTIFACT_MUTATION_LOCK:
        return get_workspace_artifacts().generate_report(run_id)


@router.get('/runs/{run_id}/artifacts')
def list_run_artifacts(run_id: str):
    """Expose file readiness independently of optimization completion."""
    return {'items': domain_call(get_workspace_artifacts().manifest, run_id)}


@router.post('/runs/{run_id}/report/retry')
def retry_run_report(run_id: str):
    """Regenerate missing reports/files from saved results without model or solver calls."""
    return {'items': domain_call(generate_workspace_report, run_id)}


@router.get('/runs/{run_id}/artifacts/{artifact_id}')
def download_run_artifact(run_id: str, artifact_id: str):
    """Download an allowlisted file after the router's shared authorization check."""
    from ..services.workspace.artifacts import FILES
    content = domain_call(get_workspace_artifacts().download, run_id, artifact_id)
    _, filename, media = FILES[artifact_id]
    return Response(content, media_type=media, headers={'Content-Disposition': f'attachment; filename="{filename}"'})


@router.get('/conversations/{context_id}')
def get_conversation(context_id: str):
    """Restore only visible complete-turn messages under the existing authenticated domain."""
    from ..a2a.persistence import get_conversation_store, access_domain
    saved=domain_call(get_conversation_store().load,context_id,access_domain())
    visible=[]
    for item in saved['messages']:
        if item['type'] not in ('human','ai'):continue
        data=item['data']
        if data.get('tool_calls') or not data.get('content'):continue
        visible.append({'role':'user' if item['type']=='human' else 'assistant','content':data['content']})
    return {'context_id':context_id,'revision':saved['revision'],'messages':visible}
