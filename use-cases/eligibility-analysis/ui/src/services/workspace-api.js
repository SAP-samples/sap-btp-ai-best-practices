/** Authenticated workspace requests, preserving field-level errors and cancellation. */
import {API_BASE_URL, API_KEY} from './api.js';
import {analysisForm} from './workspace-form.js';

/** Send JSON or multipart data; never place credentials in download URLs. */
export async function workspaceRequest(path, {method='GET', body, signal, blob=false}={}) {
  const multipart = body instanceof FormData;
  const response = await fetch(`${API_BASE_URL}/api/workspace${path}`, {
    method, signal, headers:{'X-API-Key':API_KEY, ...(!multipart && body ? {'Content-Type':'application/json'} : {})},
    body: body ? (multipart ? body : JSON.stringify(body)) : undefined,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const detail = payload.detail || {};
    const error = new Error(detail.message || detail.fields?.map(issue => `${issue.path}: ${issue.message}`).join('\n') ||
      (typeof detail === 'string' ? detail : `Request failed (${response.status})`));
    Object.assign(error, {status:response.status, detail});
    throw error;
  }
  return blob ? response.blob() : response.json();
}

/** Read immutable analysis metadata and complete eligible source IDs. */
export const getAnalysis = (id, signal) => workspaceRequest(`/analyses/${encodeURIComponent(id)}`, {signal});
/** List saved offers, independently of local browser state. */
export const listAnalyses = (query={}, signal) => workspaceRequest(`/analyses?${new URLSearchParams(query)}`, {signal});
/** Permanently delete selected saved analyses and their HANA-owned child records. */
export const deleteAnalyses = (analysisIds, signal) => workspaceRequest('/analyses', {method:'DELETE',body:{analysis_ids:analysisIds},signal});
/** Fetch one visible page without changing run scope. */
export const listInvoices = (id, query={}, signal) => workspaceRequest(`/analyses/${encodeURIComponent(id)}/invoices?${new URLSearchParams(query)}`, {signal});
/** Create a saved draft from exact server-validated eligible IDs. */
export const createRun = (analysisId, scope) => workspaceRequest('/runs', {method:'POST', body:{analysis_id:analysisId, candidate_scope:scope}});
/** Reopen authoritative progress or a completed funding result. */
export const getRun = (id, signal) => workspaceRequest(`/runs/${encodeURIComponent(id)}`, {signal});
/** Analyze one file with an explicit date and a stable key for uncertain retries. */
export const analyzeFile = (file, settings, signal) => workspaceRequest('/analyses', {method:'POST', body:analysisForm(file,settings), signal});
/** Compute a frozen eligibility comparison and its export token. */
export const getInsights = (id, body, signal) => workspaceRequest(`/analyses/${encodeURIComponent(id)}/insights`, {method:'POST', body, signal});
/** Fetch a scope-bound analysis export as an authenticated blob. */
export const analysisExport = (id, kind, token) => workspaceRequest(`/analyses/${encodeURIComponent(id)}/exports/${kind}?scope_token=${encodeURIComponent(token)}`, {blob:true});

/** Save a fetched artifact and promptly revoke its temporary browser object URL. */
export function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url; anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}
/** Validate a panel draft without persisting it. */
export const previewSettings = (id, revision, settings) => workspaceRequest(`/runs/${id}/settings/preview`, {method:'POST',body:{expected_revision:revision,settings}});
/** Save reviewed settings against their original revision. */
export const saveSettings = (id, revision, settings) => workspaceRequest(`/runs/${id}/settings`, {method:'PUT',body:{expected_revision:revision,settings}});
/** Import only into a reviewed panel draft. */
export function previewSettingsImport(id,file,revision) {
  const body=new FormData();body.append('file',file);body.append('expected_revision',String(revision));
  return workspaceRequest(`/runs/${id}/settings/import`,{method:'POST',body});
}
/** List persisted runs for the current saved offer. */
export const listRuns = id => workspaceRequest(`/analyses/${id}/runs`);
/** Start explicit lifetime preparation once. */
export const prepareRun = run => workspaceRequest(`/runs/${run.run_id}/prepare`,{method:'POST',body:{expected_revision:run.revision}});
/** Accept exactly the displayed preparation, never client-supplied prediction values. */
export const acknowledgeRun = run => workspaceRequest(`/runs/${run.run_id}/acknowledgement`,{method:'POST',body:{expected_revision:run.revision,preparation_id:run.preparation_id,accepted:true}});
/** Cancel only before the solver has entered execution. */
export const cancelRun = run => workspaceRequest(`/runs/${run.run_id}/cancel`,{method:'POST',body:{expected_revision:run.revision}});
/** Retry optimization with the saved prediction snapshot. */
export const retrySolve = run => workspaceRequest(`/runs/${run.run_id}/retry-solve`,{method:'POST',body:{expected_revision:run.revision}});
/** Fetch run-bound file readiness. */
export const listArtifacts = id => workspaceRequest(`/runs/${id}/artifacts`);
/** Rebuild missing files independently of optimization. */
export const retryReport = id => workspaceRequest(`/runs/${id}/report/retry`,{method:'POST'});
/** Download one authenticated allowlisted file. */
export const runArtifact = (id,key) => workspaceRequest(`/runs/${id}/artifacts/${key}`,{blob:true});

/** Import upstream-approved candidates without running eligibility rules. */
export const importSelection = (file, settings, signal) => workspaceRequest('/selections', {method:'POST', body:analysisForm(file,settings), signal});
