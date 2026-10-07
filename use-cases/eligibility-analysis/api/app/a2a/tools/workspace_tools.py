"""Read-only tools constrained to validated request-local receivables context."""
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from ..workspace_context import resolve_workspace_context


def summarize_workspace_run(run):
    """Project a saved run into stable, business-facing assistant fields.

    Args:
        run: Saved workspace run dictionary containing preparation and result state.

    Returns:
        A dictionary with run, lifetime estimation and recommendation summaries.
    """
    preparation = run.get('preparation') or {}
    predictions = preparation.get('predictions') or []
    fallback_count = int(preparation.get('fallback_count') or
                         sum(row.get('source') != 'rpt1' for row in predictions))
    acknowledgement = preparation.get('acknowledgement')
    if fallback_count == 0:
        acknowledgement_status = 'not_required'
    elif acknowledgement:
        acknowledgement_status = 'accepted'
    elif run.get('status') == 'awaiting_lifetime_acknowledgement':
        acknowledgement_status = 'pending'
    else:
        acknowledgement_status = 'inconsistent'
    result = run.get('result') or {}
    return {
        'run': {
            'run_id': run['run_id'], 'status': run['status'], 'revision': run['revision'],
            'candidate_count': len(run.get('row_ids') or []),
            'readiness_issues': run.get('readiness_issues') or [],
        },
        'credit_settings': run.get('settings') or {},
        'lifetime_estimation': {
            'rpt1_count': sum(row.get('source') == 'rpt1' for row in predictions),
            'fallback_count': fallback_count,
            'fallback_days': 28 if fallback_count else None,
            'history': preparation.get('history') or {},
            'acknowledgement_status': acknowledgement_status,
            'acknowledged_at': acknowledgement.get('accepted_at') if acknowledgement else None,
        },
        'recommendation': {
            'available': run.get('status') == 'completed' and bool(run.get('result')),
            'solver_status': result.get('solver_status'),
            'selected_count': len(result.get('selected') or []),
            'not_selected_count': len(result.get('not_selected') or []),
            'pre_excluded_count': len(result.get('pre_excluded') or []),
            'recommended_amount': result.get('objective_amount'),
            'currency': result.get('currency'),
            'week_starts': result.get('week_starts') or [],
            'exposure': result.get('exposure') or {},
        },
    }


def scoped_service(config):
    """Resolve current references again so stale scope never silently switches runs."""
    from ...routers.workspace import get_workspace_service
    service=get_workspace_service()
    payload=config.get('configurable',{}).get('workspace_context') or {}
    context=resolve_workspace_context(payload,service.analyses,service.runs)
    if not context.analysis_id:raise ValueError('Choose an offer in the workspace or supply its analysis reference')
    return service,context


@tool
def get_workspace_overview(config: RunnableConfig) -> dict:
    """Read the active saved offer, exact recommendation scope, credit assumptions and model coverage."""
    service,context=scoped_service(config);analysis=service.analyses.get(context.analysis_id)
    result={'analysis':analysis,'context':context.model_dump()}
    if context.run_id:
        run=service.runs.get(context.run_id)
        result.update(summarize_workspace_run(run))
    return result


@tool
def get_workspace_invoice_rows(config: RunnableConfig, offset: int=0, limit: int=50) -> dict:
    """Read scoped invoice diagnostics and distinct saved recommendation outcomes, with pagination."""
    from ...services.workspace.artifacts import funding_outcome
    service,context=scoped_service(config)
    if offset<0 or not 1<=limit<=100:raise ValueError('Use offset >= 0 and limit 1–100')
    from ...services.workspace.analysis_store import filter_rows
    rows=filter_rows(service.analyses.all_rows(context.analysis_id),context.filters)
    if context.row_ids:rows=[row for row in rows if row['row_id'] in set(context.row_ids)]
    if context.run_id:
        run=service.runs.get(context.run_id);selected={row['row_id'] for row in (run.get('result') or {}).get('selected',[])}
        for row in rows:row['funding_outcome']=funding_outcome(row['row_id'],set(run['row_ids']),selected) if run['status']=='completed' else 'not_yet_optimized'
    return {'analysis_id':context.analysis_id,'run_id':context.run_id,'total':len(rows),'rows':rows[offset:offset+limit]}


@tool
def get_workspace_pattern_insights(config: RunnableConfig, lookback_days: int=90) -> dict:
    """Compare selected/current filtered eligibility outcomes with strict historical diagnostics."""
    from ...services.workspace.insights import WorkspaceInsights
    service,context=scoped_service(config)
    if not 1<=lookback_days<=3650:raise ValueError('Lookback must be 1–3650 days')
    snapshot=WorkspaceInsights(service.analyses).analyze(context.analysis_id,context.row_ids or None,context.filters,lookback_days)
    return {key:value for key,value in snapshot.items() if key!='current_rows'}


WORKSPACE_TOOLS=[get_workspace_overview,get_workspace_invoice_rows,get_workspace_pattern_insights]


@tool
def inspect_saved_workspace(analysis_id: str|None=None, run_id: str|None=None) -> dict:
    """Read an explicitly named saved offer/run, useful in Joule without browser context.

    Use an ID supplied or selected by the user. Never pick another latest run implicitly.
    Returns authoritative IDs, revision, settings, preparation and recommendation summary.
    """
    from ...routers.workspace import get_workspace_service
    service=get_workspace_service()
    if not analysis_id and not run_id:raise ValueError('Supply an explicit saved analysis or run ID')
    revision=service.runs.get(run_id)['revision'] if run_id else None
    config={'configurable':{'workspace_context':{'analysis_id':analysis_id,'run_id':run_id,'revision':revision}}}
    return get_workspace_overview.func(config=config)


@tool
def list_saved_workspace_offers(limit: int=10) -> dict:
    """List saved offer names and IDs for the user to choose, without selecting an active run."""
    from ...routers.workspace import get_workspace_service
    if not 1<=limit<=50:raise ValueError('Use a limit of 1–50')
    return get_workspace_service().analyses.list_analyses(limit,0)


WORKSPACE_TOOLS += [inspect_saved_workspace,list_saved_workspace_offers]
