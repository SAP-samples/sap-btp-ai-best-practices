/**
 * Handover workspace API service.
 *
 * CRUD operations for handover workspaces, file uploads, and context sources.
 */

import { request, requestMultipart, API_BASE_URL } from "./api.js";

// ── Handover CRUD ──────────────────────────────────────────────────────

/**
 * Create a new handover workspace.
 * @param {Object} [options]
 * @param {string} [options.linkedConversationId] - Optional ideation conversation to link
 * @param {Object} [options.payload] - Initial form payload
 * @returns {Promise<{id: string, status: string}>}
 */
export async function createHandover({ linkedConversationId, payload } = {}) {
  return request("/api/handover", "POST", {
    linked_conversation_id: linkedConversationId || null,
    payload: payload || null,
  });
}

/**
 * List all handover workspaces for the current user.
 * @returns {Promise<{items: Array, total: number}>}
 */
export async function listHandovers() {
  return request("/api/handover", "GET");
}

/**
 * Get a handover workspace with full payload (flattened at top level).
 * @param {string} handoverId
 * @returns {Promise<Object>}
 */
export async function getHandover(handoverId) {
  return request(`/api/handover/${handoverId}`, "GET");
}

/**
 * Update handover form fields (partial merge).
 * @param {string} handoverId
 * @param {Object} payload - Fields to update
 * @returns {Promise<{id: string, status: string}>}
 */
export async function updateHandover(handoverId, payload) {
  return request(`/api/handover/${handoverId}`, "PATCH", { payload });
}

/**
 * Mark a handover as validated/complete.
 * @param {string} handoverId
 * @returns {Promise<{id: string, status: string}>}
 */
export async function completeHandover(handoverId) {
  return request(`/api/handover/${handoverId}/complete`, "POST");
}

/**
 * Delete a handover workspace.
 * @param {string} handoverId
 * @returns {Promise<{deleted: boolean}>}
 */
export async function deleteHandover(handoverId) {
  return request(`/api/handover/${handoverId}`, "DELETE");
}

/**
 * Export handover as clean JSON for downstream consumption.
 * @param {string} handoverId
 * @returns {Promise<Object>}
 */
export async function exportHandover(handoverId) {
  return request(`/api/handover/${handoverId}/export`, "GET");
}

/**
 * Export handover in a specific format and trigger a browser download.
 * For JSON, returns the parsed object (use exportHandover instead for data access).
 * For PDF/Markdown, downloads the file directly.
 * @param {string} handoverId
 * @param {"json"|"pdf"|"markdown"} format
 * @returns {Promise<void>}
 */
export async function exportHandoverAs(handoverId, format = "json") {
  if (format === "json") {
    // JSON: parse and return for downloadJson() in the caller
    return request(`/api/handover/${handoverId}/export`, "GET");
  }

  // PDF/Markdown: binary download with auth headers
  const { getAuthHeaders, API_BASE_URL } = await import("./api.js");
  const headers = await getAuthHeaders();
  const res = await fetch(
    `${API_BASE_URL}/api/handover/${handoverId}/export?format=${encodeURIComponent(format)}`,
    { headers },
  );
  if (!res.ok) throw new Error(`Export failed: ${res.status}`);

  const blob = await res.blob();
  const disposition = res.headers.get("Content-Disposition");
  const match = disposition?.match(/filename="?(.+?)"?$/);
  const ext = format === "pdf" ? "pdf" : format === "zip" ? "zip" : "md";
  const filename = match?.[1] || `handover-export.${ext}`;

  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Context Sources ────────────────────────────────────────────────────

/**
 * Add a context source to the handover.
 * @param {string} handoverId
 * @param {string} sourceType - ideation | transcript | document | manual
 * @param {string} [sourceRef] - Conversation ID or filename
 * @returns {Promise<Object>}
 */
export async function addContext(handoverId, sourceType, sourceRef) {
  const formData = new FormData();
  formData.append("source_type", sourceType);
  if (sourceRef) formData.append("source_ref", sourceRef);
  return requestMultipart(`/api/handover/${handoverId}/context`, formData);
}

/**
 * Add a file as a context source (document type).
 * The backend extracts text, runs LLM extraction, and stores the file.
 * @param {string} handoverId
 * @param {File} file - The file to upload
 * @returns {Promise<Object>}
 */
export async function addContextFile(handoverId, file) {
  const formData = new FormData();
  formData.append("source_type", "document");
  formData.append("source_ref", file.name);
  formData.append("file", file);
  return requestMultipart(`/api/handover/${handoverId}/context`, formData);
}

/**
 * List context sources for a handover.
 * @param {string} handoverId
 * @returns {Promise<Array>}
 */
export async function listContext(handoverId) {
  return request(`/api/handover/${handoverId}/context`, "GET");
}

/**
 * Remove a context source.
 * @param {string} handoverId
 * @param {string} contextId
 * @returns {Promise<{deleted: boolean}>}
 */
export async function removeContext(handoverId, contextId) {
  return request(`/api/handover/${handoverId}/context/${contextId}`, "DELETE");
}

// ── File Uploads ───────────────────────────────────────────────────────

/**
 * Upload a file to a specific zone.
 * @param {string} handoverId
 * @param {File} file
 * @param {string} zone - asis | data | tobe | logic | transcript | docs
 * @returns {Promise<{file_id: string, filename: string, size: number, zone: string}>}
 */
export async function uploadFile(handoverId, file, zone) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("zone", zone);
  return requestMultipart(`/api/handover/${handoverId}/files`, formData);
}

/**
 * List file metadata for a handover.
 * @param {string} handoverId
 * @returns {Promise<Array>}
 */
export async function listFiles(handoverId) {
  return request(`/api/handover/${handoverId}/files`, "GET");
}

/**
 * Delete a single uploaded file.
 * @param {string} handoverId
 * @param {string} fileId
 * @returns {Promise<{deleted: boolean}>}
 */
export async function deleteFile(handoverId, fileId) {
  return request(`/api/handover/${handoverId}/files/${fileId}`, "DELETE");
}

/**
 * Get the full URL for downloading/previewing a handover file.
 * The caller must add auth headers when fetching this URL.
 * @param {string} handoverId
 * @param {string} fileId
 * @returns {string}
 */
export function getHandoverFileUrl(handoverId, fileId) {
  return `${API_BASE_URL}/api/handover/${handoverId}/files/${fileId}/content`;
}

// ── AI Assist ─────────────────────────────────────────────────────────

/**
 * Run pre-fill extraction across all linked context sources.
 * Returns suggested field values (does NOT auto-write to the form).
 * @param {string} handoverId
 * @returns {Promise<{fields: Object, field_count: number, source_count: number}>}
 */
export async function prefillFromContext(handoverId) {
  return request(`/api/handover/${handoverId}/prefill`, "POST");
}

/**
 * Send a message to the handover agent assistant.
 * @param {string} handoverId
 * @param {string} message - User's message text
 * @param {string} [targetSection] - Optional section ID to focus the agent on
 * @returns {Promise<{kind: string, message: string, fields?: Object, gaps?: Array}>}
 */
export async function sendAgentMessage(handoverId, message, targetSection) {
  return request(`/api/handover/${handoverId}/agent`, "POST", {
    message,
    target_section: targetSection || null,
  });
}

/**
 * Get the agent conversation history for replay on load.
 * @param {string} handoverId
 * @returns {Promise<{messages: Array}>}
 */
export async function getAgentHistory(handoverId) {
  return request(`/api/handover/${handoverId}/agent/history`, "GET");
}
