function parseJsonField(value, fallback) {
  if (!value) return fallback;
  if (Array.isArray(value)) return value;
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
}

function blankToNull(value) {
  const trimmed = String(value ?? "").trim();
  return trimmed ? trimmed : null;
}

function secondsSince(value, nowMs) {
  const timestamp = Date.parse(normalizeUtcDateTime(value || ""));
  if (!Number.isFinite(timestamp)) {
    return null;
  }
  return Math.max(0, Math.round((nowMs - timestamp) / 1000));
}

/**
 * Trim a required manual email field and fail when it is blank.
 *
 * @param {unknown} value Raw form field value.
 * @returns {string} Trimmed field value.
 */
function requireManualField(value) {
  const trimmed = String(value ?? "").trim();
  if (!trimmed) {
    throw new Error("Sender email, subject, and body are required.");
  }
  return trimmed;
}

/**
 * Create a browser-safe processing run id for progress polling.
 *
 * @param {string} prefix Source prefix such as manual or gmail-fetch.
 * @returns {string} Unique processing run id.
 */
export function createProcessingRunId(prefix) {
  const safePrefix = String(prefix || "run").replace(/[^a-z0-9_-]/gi, "-").toLowerCase();
  const randomId = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `${safePrefix}-${randomId}`;
}

/**
 * Convert a processing snapshot into UI-ready progress text.
 *
 * @param {{ run?: object, events?: object[] }} snapshot API processing snapshot.
 * @param {number} nowMs Current timestamp in milliseconds.
 * @returns {{ status: string, currentMessage: string, helperText: string, isStale: boolean, recentEvents: object[] }} View model.
 */
export function buildProcessingProgressView(snapshot, nowMs = Date.now()) {
  const run = snapshot?.run || {};
  const events = Array.from(snapshot?.events || []);
  const currentMessage = run.current_message || events.at(-1)?.message || "Starting analysis";
  const lastUpdateSeconds = secondsSince(run.last_heartbeat_at || events.at(-1)?.event_time, nowMs);
  const isStale = run.status === "running" && lastUpdateSeconds !== null && lastUpdateSeconds > 45;
  const helperText = isStale
    ? `Still waiting on the model or S/4. Last update ${lastUpdateSeconds}s ago.`
    : lastUpdateSeconds === null
      ? "Waiting for first progress update"
      : `Last update ${lastUpdateSeconds}s ago`;
  const recentEvents = events
    .slice()
    .sort((left, right) => Number(left.sequence_number || 0) - Number(right.sequence_number || 0))
    .slice(-5);
  return {
    status: run.status || "running",
    currentMessage,
    helperText,
    isStale,
    recentEvents
  };
}

function normalizeUtcDateTime(value) {
  if (!value || typeof value !== "string") {
    return value || "";
  }
  if (/[zZ]$|[+-]\d{2}:?\d{2}$/.test(value)) {
    return value;
  }
  return `${value.replace(" ", "T")}Z`;
}

const validationFieldLabels = {
  supplier_id: "Supplier ID",
  material_number: "Material number",
  original_price: "Original price",
  requested_new_price: "Requested price",
  currency: "Currency",
  effective_from: "Effective from"
};

/**
 * Convert a persisted draft status code into the reviewer-facing label.
 *
 * @param {unknown} status Persisted draft status such as ready_for_review.
 * @returns {string} Human-readable status label for review screens.
 */
export function getDraftStatusLabel(status) {
  const labels = {
    ready_for_review: "Ready to Approve",
    needs_human_review: "Needs Review",
    not_price_request: "Non-price",
    failed: "Failed",
    approved: "Approved",
    rejected: "Rejected"
  };
  return labels[status] || status || "Unknown";
}

/**
 * Return the editor control behavior for a draft field in the review form.
 *
 * @param {unknown} fieldName Draft edit field suffix, such as "supplier-id" or "explanation".
 * @returns {{ component: "input" } | { component: "textarea", growing: boolean, rows: number }} Editor configuration.
 */
export function getDraftFieldEditorConfig(fieldName) {
  if (String(fieldName ?? "").trim() === "explanation") {
    return {
      component: "textarea",
      growing: true,
      rows: 4
    };
  }
  return {
    component: "input"
  };
}

/**
 * Read the current value from a draft edit control.
 *
 * @param {{ tagName?: string, value?: string, shadowRoot?: { querySelector?: (selector: string) => ({ value?: string } | null) } } | null | undefined} element UI5 edit control.
 * @returns {string} Current editable value shown to the reviewer.
 */
export function readDraftEditorValue(element) {
  if (!element) {
    return "";
  }
  if (element.tagName === "UI5-TEXTAREA") {
    const nativeTextArea = element.shadowRoot?.querySelector?.("textarea");
    return nativeTextArea?.value ?? "";
  }
  return element.value ?? "";
}

export function formatValidationError(error) {
  const message = String(error ?? "");
  const match = message.match(/^([a-z0-9_]+) is required$/i);
  if (!match) {
    return message;
  }
  const label = validationFieldLabels[match[1]] || match[1].replaceAll("_", " ");
  return `${label} is required`;
}

/**
 * Convert a failed API request into the message shown in review toasts.
 *
 * @param {Error & { detail?: unknown }} error Request error from the API service.
 * @returns {string} User-facing error message.
 */
export function formatApiErrorMessage(error) {
  const detail = error?.detail;
  if (detail && typeof detail === "object" && detail.message) {
    return String(detail.message);
  }
  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }
  return error?.message || "Request failed.";
}

export function mapDraftFromApi(item) {
  return {
    rowId: item.draft_id,
    gmailMessageId: item.gmail_message_id,
    senderEmail: item.sender_email,
    subject: item.subject || "",
    supplierId: item.supplier_id,
    supplierName: item.supplier_name,
    supplierEmail: item.supplier_email || item.sender_email,
    materialCode: item.material_code,
    materialDescription: item.material_description,
    originalPrice: item.original_price,
    requestedNewPrice: item.requested_new_price,
    currency: item.currency,
    uom: item.uom,
    effectiveFrom: item.effective_from,
    effectiveTo: item.effective_to,
    status: item.status,
    confidence: item.confidence,
    explanation: item.explanation,
    candidateMaterials: parseJsonField(item.candidate_materials_json, []),
    candidateSuppliers: parseJsonField(item.candidate_suppliers_json, []),
    validationErrors: parseJsonField(item.validation_errors_json, []),
    attachments: parseJsonField(item.attachments_json || item.attachments, []),
    s4WriteResult: item.s4_write_result || null,
    originalEmail: {
      from: item.sender_email,
      subject: item.subject || "",
      date: normalizeUtcDateTime(item.email_date),
      body: item.body || ""
    }
  };
}

/**
 * Build the source label shown for original email audit metadata.
 *
 * @param {unknown} messageId Gmail or synthetic manual message id.
 * @returns {string} Human-readable source label.
 */
export function formatEmailSourceLabel(messageId) {
  const value = String(messageId ?? "").trim();
  if (!value) {
    return "Source not set";
  }
  if (value.startsWith("manual-")) {
    return `Manual entry ${value}`;
  }
  return `Gmail ${value}`;
}

/**
 * Convert manual email form values into the API request payload.
 *
 * @param {{ senderEmail: unknown, subject: unknown, body: unknown }} formValues Manual form values.
 * @returns {{ sender_email: string, subject: string, body: string }} API payload.
 */
export function buildManualEmailPayload(formValues, processingRunId = null) {
  const payload = {
    sender_email: requireManualField(formValues.senderEmail),
    subject: requireManualField(formValues.subject),
    body: requireManualField(formValues.body)
  };
  if (processingRunId) {
    payload.processing_run_id = String(processingRunId);
  }
  return payload;
}

/**
 * Return whether the manual route has user-selected attachments.
 *
 * @param {ArrayLike<unknown> | null | undefined} files Selected browser files.
 * @returns {boolean} True when at least one file is selected.
 */
export function manualEmailHasAttachments(files) {
  return Array.from(files || []).length > 0;
}

/**
 * Build multipart form data for the manual email upload route.
 *
 * @param {{ senderEmail: unknown, subject: unknown, body: unknown }} formValues Manual form values.
 * @param {ArrayLike<File | { file: Blob, name?: string }> | null | undefined} files Selected files.
 * @returns {FormData} Multipart API payload.
 */
export function buildManualEmailFormData(formValues, files, processingRunId = null) {
  const formData = new FormData();
  formData.append("sender_email", requireManualField(formValues.senderEmail));
  formData.append("subject", requireManualField(formValues.subject));
  formData.append("body", requireManualField(formValues.body));
  if (processingRunId) {
    formData.append("processing_run_id", String(processingRunId));
  }
  Array.from(files || []).forEach((entry) => {
    const file = entry?.file || entry;
    const filename = entry?.name || file?.name || "attachment";
    formData.append("attachments", file, filename);
  });
  return formData;
}

/**
 * Format attachment byte size for audit displays.
 *
 * @param {unknown} value Raw byte count.
 * @returns {string} Human-readable size.
 */
export function formatAttachmentSize(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes < 0) {
    return "Size unknown";
  }
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  const kb = bytes / 1024;
  if (kb < 1024) {
    return `${Number.isInteger(kb) ? kb : kb.toFixed(1).replace(/\.0$/, "")} KB`;
  }
  const mb = kb / 1024;
  return `${Number.isInteger(mb) ? mb : mb.toFixed(1).replace(/\.0$/, "")} MB`;
}

/**
 * Build the authenticated attachment download endpoint.
 *
 * @param {unknown} attachmentId Persisted attachment id.
 * @returns {string} API endpoint path.
 */
export function attachmentDownloadEndpoint(attachmentId) {
  return `/api/email-attachments/${encodeURIComponent(String(attachmentId ?? ""))}/download`;
}

export function buildDraftPatchPayload(row) {
  return {
    supplier_id: blankToNull(row.supplierId),
    supplier_name: blankToNull(row.supplierName),
    supplier_email: blankToNull(row.supplierEmail),
    material_number: blankToNull(row.materialCode),
    material_description: blankToNull(row.materialDescription),
    original_price: blankToNull(row.originalPrice),
    requested_new_price: blankToNull(row.requestedNewPrice),
    currency: blankToNull(row.currency),
    uom: blankToNull(row.uom),
    effective_from: blankToNull(row.effectiveFrom),
    effective_to: blankToNull(row.effectiveTo),
    explanation: blankToNull(row.explanation)
  };
}

export function getDraftDecisionActions(status) {
  return [
    {
      key: "reject",
      label: "Reject",
      design: "Negative",
      className: "reject-draft-button",
      disabled: false
    },
    {
      key: "approve",
      label: "Accept",
      design: "Positive",
      className: "approve-draft-button",
      disabled: status !== "ready_for_review"
    }
  ];
}

export function summarizeFetchResult(summary) {
  const base = `Fetched ${summary.fetched}, skipped ${summary.skipped_existing}, non-price ${summary.not_price_request}, extraction failures ${summary.extraction_failed}, agent failures ${summary.agent_failed}, drafts ${summary.drafts_created}`;
  const attachmentTotal =
    Number(summary.attachments_downloaded || 0) +
    Number(summary.attachments_skipped || 0) +
    Number(summary.attachments_failed || 0);
  if (!attachmentTotal) {
    return base;
  }
  return `${base}, attachments ${summary.attachments_downloaded || 0} downloaded, ${summary.attachments_skipped || 0} skipped, ${summary.attachments_failed || 0} failed`;
}
