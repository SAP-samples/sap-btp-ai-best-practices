/** Pure workspace state: visibility is separate from the exact funding population. */
export function createWorkspaceState() {
  return {analysisId:null, runId:null, revision:null, selectedRowIds:new Set(),
    eligibleRowIds:new Set(), filters:{}, offset:0, view:'invoices',abortController:new AbortController()};
}

/** Reset source-bound state when opening a different analysis. */
export function switchAnalysis(state, analysisId) {
  Object.assign(state, {analysisId, runId:null, revision:null, selectedRowIds:new Set(),
    eligibleRowIds:new Set(), filters:{}, offset:0,view:'invoices'});
}

/** Merge a visible page's selection while retaining off-page selections. */
export function updatePageSelection(state, pageIds, selectedIds) {
  for (const id of pageIds) state.selectedRowIds.delete(id);
  for (const id of selectedIds) state.selectedRowIds.add(id);
}

/** Build recommendation scope without copying display filters into optimizer inputs. */
export function candidateScope(mode, selectedRowIds, eligibleRowIds) {
  if (mode === 'all_eligible') return {mode, row_ids:[]};
  if (mode !== 'selected') throw new Error('Unknown scope mode');
  const row_ids = [...selectedRowIds].filter(id => eligibleRowIds.has(id)).sort();
  if (!row_ids.length) throw new Error('Select at least one eligible invoice');
  return {mode, row_ids};
}

/** Describe persisted run state in business language rather than storage enum names. */
export function runStatusLabel(run) {
  const labels={draft:'Credit settings needed',estimating_lifetimes:'Estimating lifetimes',
    awaiting_lifetime_acknowledgement:'Review lifetime assumptions',optimizing:'Computing recommendation',
    completed:'Recommendation ready',failed:'Run failed',cancelled:'Cancelled'};
  return run.status==='draft'&&!run.readiness_issues?.length ? 'Ready to compute' : labels[run.status]||'Unknown status';
}

/** Identify saved runs by their meaning and scope, retaining a short ID for distinction. */
export function savedRunLabel(run) {
  return `${runStatusLabel(run)} · ${run.row_ids.length.toLocaleString()} invoices · ${run.run_id.slice(0,8)}`;
}

/**
 * Confirm that a durable recommendation can be opened without inferring funding execution.
 * @param {object|null} run Current saved run.
 * @returns {boolean} True only when a completed result snapshot exists.
 */
export function resultsAreReady(run) {
  return run?.status==='completed'&&Boolean(run.result);
}

/**
 * Resolve the primary workspace view from the explicit URL and current run state.
 * @param {string} search Current query string.
 * @param {object|null} run Current saved run.
 * @returns {'invoices'|'results'} Safe primary view selection.
 */
export function requestedWorkspaceView(search,run) {
  return new URLSearchParams(search).get('view')==='results'&&resultsAreReady(run) ? 'results' : 'invoices';
}
