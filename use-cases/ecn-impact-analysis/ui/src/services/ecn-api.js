import { request } from "./api.js";

/** Upload one ECN JSON document and pin the active Settings datasets. */
export function uploadSources({ requestId, ecnFile, signal, dataGroup = "customer", assessmentDate = "" }) {
  const body = new FormData();
  body.set("request_id", requestId);
  body.set("ecn_file", ecnFile);
  body.set("data_group", dataGroup);
  if (assessmentDate) body.set("assessment_date", assessmentDate);
  return request("/api/uploads", { method: "POST", body, signal });
}

/** Return one server-paginated historical list, omitting only null, undefined, and empty filters. */
export function listEcns(query = {}, { signal } = {}) {
  const entries = Object.entries(query)
    .filter(([, value]) => value !== null && value !== undefined && value !== "")
    .map(([key, value]) => [key, String(value)]);
  const params = new URLSearchParams(entries);
  const suffix = params.size ? `?${params.toString()}` : "";
  return request(`/api/ecns${suffix}`, { signal });
}

/** Load one immutable ECN occurrence. */
export function getEcn(ecnRowId, { signal } = {}) {
  return request(`/api/ecns/${encodeURIComponent(ecnRowId)}`, { signal });
}

/** Read bounded rebuildable RDF projection diagnostics for one ECN occurrence. */
export function getRdfProjection(ecnRowId, { signal } = {}) {
  return request(`/api/ecns/${encodeURIComponent(ecnRowId)}/rdf-projection`, { signal });
}

/** Request a post-commit rebuild using only the selected ECN identity. */
export function retryRdfProjection(ecnRowId, { signal } = {}) {
  return request(`/api/ecns/${encodeURIComponent(ecnRowId)}/rdf-projection/retry`, {
    method: "POST",
    signal,
  });
}

/** Resolve one controlled evidence ID within its owning occurrence. */
export function getEvidence(ecnRowId, evidenceId, { signal } = {}) {
  return request(
    `/api/ecns/${encodeURIComponent(ecnRowId)}/evidence/${encodeURIComponent(evidenceId)}`,
    { signal },
  );
}

/** Load one bounded page from a business evidence section. */
export function listBusinessEvidence(
  ecnRowId,
  section,
  { page = 1, pageSize = 25, includeTechnical = false, signal } = {},
) {
  const params = new URLSearchParams({
    section,
    page: String(page),
    page_size: String(pageSize),
    include_technical: String(includeTechnical),
  });
  return request(`/api/ecns/${encodeURIComponent(ecnRowId)}/evidence?${params}`, { signal });
}

/** Return immutable Settings-managed reference dataset publications. */
export function listReferenceDatasets({ signal } = {}) {
  return request("/api/reference-datasets", { signal });
}

/** Delete one Settings workbook by ID when no ECN upload depends on it.
 * @param {string} datasetId - The publication UUID shown in Settings.
 * @param {{signal?: AbortSignal}} options - Optional request cancellation.
 * @returns {Promise<null>} Resolves after the server removes the workbook.
 */
export function deleteReferenceDataset(datasetId, { signal } = {}) {
  return request(`/api/reference-datasets/${encodeURIComponent(datasetId)}`, {
    method: "DELETE",
    signal,
  });
}

/** Validate and atomically publish one reusable Settings workbook.
 * Plants are not declared here: they are derived from conditional rules.
 */
export function publishReferenceDataset({
  datasetType,
  sourceFile,
  context = {},
  signal,
}) {
  const body = new FormData();
  body.set("dataset_type", datasetType);
  body.set("context_json", JSON.stringify(context));
  body.set("source_file", sourceFile);
  return request("/api/reference-datasets", { method: "POST", body, signal });
}

/** Read the Plant Code Mapping, the excluded rule states and the states seen in rule uploads.
 * @param {{signal?: AbortSignal}} options - Optional request cancellation.
 * @returns {Promise<{mappings: Array<{suffix: string, plant_code: string, plant_name: string}>,
 *   excluded_rule_states: string[], available_rule_states: string[]}>}
 */
export function getPlantSettings({ signal } = {}) {
  return request("/api/plant-settings", { signal });
}

/** Replace the Plant Code Mapping and excluded rule states in one request.
 * @param {{mappings: Array<object>, excluded_rule_states: string[]}} settings - Complete new values.
 * @param {{signal?: AbortSignal}} options - Optional request cancellation.
 * @returns {Promise<object>} The saved settings as read back by the server.
 */
export function savePlantSettings(settings, { signal } = {}) {
  return request("/api/plant-settings", { method: "PUT", body: settings, signal });
}

/** Read the cached safe provider preflight result. */
export function getProviderStatus({ signal } = {}) {
  return request("/api/provider-status", { signal });
}

/** Refresh safe provider preflight and resume eligible blocked jobs. */
export function checkProviderStatus({ signal } = {}) {
  return request("/api/provider-status/check", { method: "POST", signal });
}

/** Read global progress or progress for one upload. */
export function getAnalysisProgress({ uploadId = null, signal } = {}) {
  const suffix = uploadId ? `?upload_id=${encodeURIComponent(uploadId)}` : "";
  return request(`/api/analysis-progress${suffix}`, { signal });
}

/** Queue a new attempt only for a failed analysis. */
export function retryAnalysis(ecnRowId, { signal } = {}) {
  return request(`/api/ecns/${encodeURIComponent(ecnRowId)}/analysis/retry`, {
    method: "POST",
    signal,
  });
}

/** Delete historical records after exact server-side confirmation. */
export function resetHistory(confirmation, { signal } = {}) {
  return request("/api/history", {
    method: "DELETE",
    body: { confirmation },
    signal,
  });
}
