export function parseJsonField(value, fallback) {
  if (!value) return fallback;
  if (Array.isArray(value)) return value;
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
}

export function normalizeUtcDateTime(value) {
  if (!value || typeof value !== "string") {
    return value || "";
  }
  if (/[zZ]$|[+-]\d{2}:?\d{2}$/.test(value)) {
    return value;
  }
  return `${value.replace(" ", "T")}Z`;
}

export function formatFileSize(sizeBytes) {
  const numeric = Number(sizeBytes);
  if (!Number.isFinite(numeric) || numeric < 0) {
    return "0 B";
  }
  if (numeric < 1024) {
    return `${numeric} B`;
  }
  return `${(numeric / 1024).toFixed(1)} KB`;
}

export function mapDraftFromApi(item) {
  const attachments = parseJsonField(item.selected_documents_json, []).map((attachment) => ({
    documentId: attachment.document_id,
    fileName: attachment.file_name,
    documentType: attachment.document_type,
    mimeType: attachment.mime_type,
    sizeBytes: attachment.size_bytes
  }));

  return {
    rowId: item.draft_id,
    gmailMessageId: item.gmail_message_id,
    country: item.client_country,
    toEmail: item.to_email,
    subject: item.subject || "",
    body: item.body || "",
    status: item.status,
    confidence: item.confidence,
    explanation: item.explanation || "",
    attachments,
    validationErrors: parseJsonField(item.validation_errors_json, []),
    originalEmail: {
      from: item.sender_email,
      subject: item.original_subject || "",
      date: normalizeUtcDateTime(item.email_date),
      body: item.original_body || ""
    }
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
      label: "Approve",
      design: "Positive",
      className: "approve-draft-button",
      disabled: status !== "ready_for_review"
    }
  ];
}

export function summarizeFetchResult(summary) {
  return `Fetched ${summary.fetched}, skipped ${summary.skipped_existing}, non-requests ${summary.not_document_request}, extraction failures ${summary.extraction_failed}, agent failures ${summary.agent_failed}, drafts ${summary.drafts_created}`;
}

/**
 * Describe a background run (GET /emails/fetch-status) for the status line.
 * @param {object} run FetchRunStatus payload: status, summary, error_message.
 * @returns {string} Progress text while running, final summary or error otherwise.
 */
export function describeRun(run) {
  const summary = run.summary || {};
  if (run.status === "running") {
    const done = (summary.fetched || 0) + (summary.skipped_existing || 0);
    return summary.total ? `Processing ${done} of ${summary.total} emails...` : "Looking for new emails...";
  }
  if (run.status === "failed") {
    return `Run failed: ${run.error_message || "unknown error"}`;
  }
  return summarizeFetchResult(summary);
}

export function validateManualEmail(values) {
  const errors = [];
  if (!String(values.from || "").trim()) {
    errors.push("From is required");
  }
  if (!String(values.subject || "").trim()) {
    errors.push("Subject is required");
  }
  if (!String(values.body || "").trim()) {
    errors.push("Body is required");
  }
  return errors;
}

export function manualEmailPayload(values) {
  return {
    from: String(values.from || "").trim(),
    subject: String(values.subject || "").trim(),
    body: String(values.body || "").trim()
  };
}

/**
 * Build the review decision payload using the exact draft body currently shown to the reviewer.
 *
 * @param {string} body - Current editable draft text from the expanded review field.
 * @returns {{body: string}} Request body sent to approve or reject endpoints.
 */
export function draftDecisionPayload(body) {
  return {
    body: String(body ?? "")
  };
}
