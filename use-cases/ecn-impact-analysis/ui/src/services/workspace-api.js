import { request } from "./api.js";

/** Submit one transient PS4 drawing assessment; the drawing is not part of ECN ingestion. */
export const assessDrawing = (file, signal) => {
  const form = new FormData();
  form.append("drawing_file", file);
  return request("/api/drawing-feasibility/assessments", { method: "POST", body: form, signal });
};

export const generateSummaries = (selection, requestId, signal) => request("/api/summary-batches", { method: "POST", body: { request_id: requestId, selection }, signal });
/** Permanently delete selected ECN line items (all versions); reference data is kept. */
export const deleteEcns = (selection, confirmation, signal) => request("/api/ecns/delete", { method: "POST", body: { confirmation, selection }, signal });
/** Read complete snapshot aggregates, independent of AI state. */
export const getSnapshot = (id, signal) => request(`/api/ecns/${encodeURIComponent(id)}/snapshot`, { signal });
/** Read validated perspective outputs while the final synthesis is still pending. */
export const getAnalysisStages = (id, signal) => request(`/api/ecns/${encodeURIComponent(id)}/analysis-stages`, { signal });
/** Page source-referenced objects, optionally by an observed ancestor. */
export const listObjects = (id, params, signal) => request(`/api/ecns/${encodeURIComponent(id)}/objects?${new URLSearchParams(params)}`, { signal });
/** Page a member object's exact source references and BOM occurrences. */
export const getObject = (id, params, signal) => request(`/api/ecns/${encodeURIComponent(id)}/object?${new URLSearchParams(params)}`, { signal });
/** Read the available material drawing through the authenticated API. */
export const getEngineeringDrawing = (id, number, signal) => request(`/api/ecns/${encodeURIComponent(id)}/drawing?${new URLSearchParams({ number })}`, { signal, responseType: "blob" });
/** Read saved reviewer statements against this exact snapshot. */
export const listReviews = (id, signal) => request(`/api/ecns/${encodeURIComponent(id)}/reviews`, { signal });
/** Append a separately attributed review; source facts remain unchanged. */
export const saveReview = (id, body, signal) => request(`/api/ecns/${encodeURIComponent(id)}/reviews`, { method: "POST", body, signal });
/** Read the read-only preview of SAP master data updates that accepting would imply (no SAP write). */
export const getAcceptancePlan = (id, signal) => request(`/api/ecns/${encodeURIComponent(id)}/acceptance-plan`, { signal });
/** Send a user-initiated, read-only assistant turn with browser-memory history. */
export const chat = (id, body, signal) => request(`/api/ecns/${encodeURIComponent(id)}/chat`, { method: "POST", body, signal });
/** Send a read-only assistant turn about all stored ECNs (no ECN open). */
export const globalChat = (body, signal) => request("/api/chat", { method: "POST", body, signal });

/** Create a fresh assessment with current datasets; no model calls. */
export const reassess = (id, requestId, signal, assessmentDate = null) => request(`/api/ecns/${encodeURIComponent(id)}/reassess`, { method: "POST", body: { request_id: requestId, assessment_date: assessmentDate }, signal });
/** Retrieve earlier assessments of this exact source position. */
export const versions = (id, signal) => request(`/api/ecns/${encodeURIComponent(id)}/versions`, { signal });

/** Page deterministic lifecycle and usage findings. */
export const listChanges = (id, params, signal) => request(`/api/ecns/${encodeURIComponent(id)}/changes?${new URLSearchParams(params)}`, { signal });
