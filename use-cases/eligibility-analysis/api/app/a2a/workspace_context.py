"""Validate source references separately from untrusted free-form assistant prompts."""
from pydantic import BaseModel,ConfigDict,Field
from ..models.workspace import RevisionConflict


class WorkspaceContext(BaseModel):
    """Identify an authorized saved offer, optional exact run and selected source IDs."""
    model_config=ConfigDict(extra='forbid')
    analysis_id: str|None=None
    run_id: str|None=None
    revision: int|None=Field(None,ge=0)
    row_ids: list[str]=Field(default_factory=list)
    filters: dict[str,str]=Field(default_factory=dict)


def resolve_workspace_context(payload,analyses,runs,principal=None):
    """Reject unknown IDs, cross-offer rows and stale runs before any model invocation.

    The current API has one shared-key access domain. Authentication is enforced by
    the router; principal is reserved for a future real tenant identity, never guessed.
    """
    context=WorkspaceContext(**(payload or {}))
    if context.run_id:
        run=runs.get(context.run_id)
        if context.analysis_id and context.analysis_id!=run['analysis_id']:raise ValueError('Run belongs to a different offer')
        if context.revision is None or context.revision!=run['revision']:raise RevisionConflict('Refresh the saved run before asking about it')
        context.analysis_id=run['analysis_id']
    if context.analysis_id:
        analyses.get(context.analysis_id)
        known={row['row_id'] for row in analyses.all_rows(context.analysis_id)}
        if len(context.row_ids)!=len(set(context.row_ids)) or not set(context.row_ids)<=known:raise ValueError('Unknown or duplicate source rows')
    elif context.row_ids or context.filters:raise ValueError('Choose a saved offer before passing row/filter context')
    allowed={'status','search','seller_id','debtor_id','programa','insurer_id','original_currency'}
    if set(context.filters)-allowed:raise ValueError('Unknown workspace filters')
    return context
