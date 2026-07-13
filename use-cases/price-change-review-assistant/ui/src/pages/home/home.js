import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/DatePicker.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/TextArea.js";
import "@ui5/webcomponents/dist/Toast.js";
import "@ui5/webcomponents-icons/dist/attachment.js";
import "@ui5/webcomponents-icons/dist/delete.js";
import "@ui5/webcomponents-icons/dist/download.js";

import { API_BASE_URL, API_KEY, request, requestFormData } from "../../services/api.js";
import {
  attachmentDownloadEndpoint,
  buildProcessingProgressView,
  buildManualEmailFormData,
  buildManualEmailPayload,
  buildDraftPatchPayload,
  createProcessingRunId,
  formatApiErrorMessage,
  formatAttachmentSize,
  formatEmailSourceLabel,
  formatValidationError,
  getDraftFieldEditorConfig,
  getDraftDecisionActions,
  getDraftStatusLabel,
  manualEmailHasAttachments,
  mapDraftFromApi,
  readDraftEditorValue,
  summarizeFetchResult
} from "./reviewHelpers.js";

async function fetchNewEmails(processingRunId) {
  const suffix = processingRunId ? `?processing_run_id=${encodeURIComponent(processingRunId)}` : "";
  return request(`/api/emails/fetch-new${suffix}`, "POST");
}

/**
 * Submit one manually-entered supplier email for normal backend processing.
 *
 * @param {{ sender_email: string, subject: string, body: string }} payload Manual email API payload.
 * @returns {Promise<object>} FetchSummary-compatible processing response.
 */
async function submitManualEmail(payload) {
  return request("/api/emails/manual", "POST", payload);
}

/**
 * Submit one manually-entered supplier email plus browser-selected attachments.
 *
 * @param {{ senderEmail: string, subject: string, body: string }} formValues Manual email form values.
 * @param {File[]} attachments Selected attachment files.
 * @returns {Promise<object>} FetchSummary-compatible processing response.
 */
async function submitManualEmailWithAttachments(formValues, attachments, processingRunId) {
  return requestFormData(
    "/api/emails/manual-with-attachments",
    "POST",
    buildManualEmailFormData(formValues, attachments, processingRunId)
  );
}

async function getProcessingRun(processingRunId) {
  return request(`/api/processing-runs/${encodeURIComponent(processingRunId)}`);
}

async function createProcessingRun(processingRunId, sourceType) {
  return request("/api/processing-runs", "POST", {
    processing_run_id: processingRunId,
    source_type: sourceType
  });
}

async function getPriceChangeDrafts() {
  return request("/api/price-change-drafts");
}

async function patchPriceChangeDraft(draftId, payload) {
  return request(`/api/price-change-drafts/${draftId}`, "PATCH", payload);
}

async function approvePriceChangeDraft(draftId) {
  return request(`/api/price-change-drafts/${draftId}/approve`, "POST");
}

async function rejectPriceChangeDraft(draftId) {
  return request(`/api/price-change-drafts/${draftId}/reject`, "POST");
}

/**
 * Download an attachment through the authenticated API route.
 *
 * @param {{ attachment_id?: string, filename?: string }} attachment Attachment metadata.
 * @returns {Promise<void>} Resolves after the browser download is triggered.
 */
async function downloadAttachmentFile(attachment) {
  const endpoint = attachmentDownloadEndpoint(attachment.attachment_id);
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    headers: {
      "X-API-Key": API_KEY
    }
  });
  if (!response.ok) {
    throw new Error(`Download failed with HTTP ${response.status}`);
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = attachment.filename || "attachment";
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatDisplayValue(value) {
  return value ? escapeHtml(value) : '<span class="review-cell-secondary">Not set</span>';
}

function formatMoney(row, field) {
  const value = row[field];
  if (!value) {
    return '<span class="review-cell-secondary">Not set</span>';
  }
  return escapeHtml(row.currency ? `${row.currency} ${value}` : value);
}

function statusTag(status) {
  return `<span class="status-pill status-pill-${escapeHtml(status || "unknown")}">${escapeHtml(getDraftStatusLabel(status))}</span>`;
}

function formatConfidence(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "n/a";
  }
  return `${Math.round(numeric * 100)}%`;
}

function formatDateTime(value) {
  if (!value) {
    return '<span class="review-cell-secondary">Not set</span>';
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return escapeHtml(value);
  }
  return escapeHtml(
    new Intl.DateTimeFormat(undefined, {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      timeZoneName: "short"
    }).format(date)
  );
}

function summaryFromItems(items) {
  return {
    total: items.length,
    ready: items.filter((item) => item.status === "ready_for_review").length,
    needsReview: items.filter((item) => item.status === "needs_human_review").length,
    approved: items.filter((item) => item.status === "approved").length,
    failed: items.filter((item) => item.status === "failed").length
  };
}

function renderSummary(summaryContainer, items) {
  const summary = summaryFromItems(items);
  const cards = [
    { label: "Drafts", value: summary.total },
    { label: getDraftStatusLabel("ready_for_review"), value: summary.ready },
    { label: getDraftStatusLabel("needs_human_review"), value: summary.needsReview },
    { label: "Approved", value: summary.approved },
    { label: "Failed", value: summary.failed }
  ];

  summaryContainer.innerHTML = cards
    .map(
      (card) => `
        <div class="summary-card">
          <span class="summary-label">${escapeHtml(card.label)}</span>
          <div class="summary-value">${escapeHtml(card.value)}</div>
        </div>
      `
    )
    .join("");
}

function renderCandidateList(title, candidates, renderCandidate) {
  if (!candidates.length) {
    return "";
  }

  return `
    <div class="candidate-section">
      <h4>${escapeHtml(title)}</h4>
      <div class="candidate-list">
        ${candidates.map((candidate) => `<div class="candidate-item">${renderCandidate(candidate)}</div>`).join("")}
      </div>
    </div>
  `;
}

function renderValidationErrors(errors) {
  if (!errors.length) {
    return "";
  }
  return `<div class="validation-errors">${errors.map((error) => `<div>${escapeHtml(formatValidationError(error))}</div>`).join("")}</div>`;
}

/**
 * Render the latest persisted S/4 write attempt for an expanded draft.
 *
 * @param {object | null} result Latest S/4 write result.
 * @returns {string} HTML snippet.
 */
function renderS4WriteResult(result) {
  if (!result) {
    return "";
  }
  const status = result.status || "unknown";
  const message = result.message || "Last S/4 write attempt did not return a message.";
  return `
    <div class="s4-write-result is-${escapeHtml(status)}">
      <strong>S/4 price write</strong>
      <span>${escapeHtml(status)}</span>
      <div>${escapeHtml(message)}</div>
    </div>
  `;
}

function renderProcessingProgress(container, snapshot) {
  if (!container) return;
  const view = buildProcessingProgressView(snapshot);
  const isRunning = view.status === "running";
  const statusLabel = view.status === "success" ? "Analysis complete" : view.status === "failed" ? "Analysis failed" : "Analysis running";
  container.hidden = false;
  container.innerHTML = `
    <div class="processing-progress-main">
      ${isRunning ? '<ui5-busy-indicator active size="S"></ui5-busy-indicator>' : ""}
      <div class="processing-progress-copy">
        <span class="processing-progress-title">${escapeHtml(statusLabel)}: ${escapeHtml(view.currentMessage)}</span>
        <span class="processing-progress-helper${view.isStale ? " is-stale" : ""}">${escapeHtml(view.helperText)}</span>
      </div>
    </div>
    ${
      view.recentEvents.length
        ? `
          <div class="processing-progress-events">
            ${view.recentEvents
              .map((event) => `<span class="processing-progress-event">${escapeHtml(event.message || "")}</span>`)
              .join("")}
          </div>
        `
        : ""
    }
  `;
}

function renderInitialProcessingProgress(container, message) {
  if (!container) return;
  container.hidden = false;
  container.innerHTML = `
    <div class="processing-progress-main">
      <ui5-busy-indicator active size="S"></ui5-busy-indicator>
      <div class="processing-progress-copy">
        <span class="processing-progress-title">${escapeHtml(message)}</span>
        <span class="processing-progress-helper">Waiting for first progress update</span>
      </div>
    </div>
  `;
}

function renderAttachmentAuditList(attachments) {
  if (!attachments.length) {
    return "";
  }
  return `
    <div class="email-attachments">
      <h4>Attachments</h4>
      <div class="email-attachment-list">
        ${attachments
          .map(
            (attachment) => `
              <div class="email-attachment-item">
                <div class="email-attachment-main">
                  <span>${escapeHtml(attachment.filename || "Attachment")}</span>
                  <span class="review-cell-secondary">
                    ${escapeHtml(String(attachment.file_extension || "").toUpperCase())}
                    ${escapeHtml(formatAttachmentSize(attachment.size_bytes))}
                  </span>
                </div>
                <ui5-button
                  design="Transparent"
                  icon="download"
                  class="download-attachment-button"
                  data-attachment-id="${escapeHtml(attachment.attachment_id || "")}"
                  data-attachment-filename="${escapeHtml(attachment.filename || "attachment")}"
                >
                  Download
                </ui5-button>
              </div>
            `
          )
          .join("")}
      </div>
    </div>
  `;
}

function renderDraftDecisionButtons(item) {
  return getDraftDecisionActions(item.status)
    .map(
      (action) => `
        <ui5-button
          design="${escapeHtml(action.design)}"
          class="${escapeHtml(action.className)}"
          data-row-id="${escapeHtml(item.rowId)}"
          ${action.disabled ? "disabled" : ""}
        >
          ${escapeHtml(action.label)}
        </ui5-button>
      `
    )
    .join("");
}

/**
 * Render one editable scalar draft field in the expanded review form.
 *
 * @param {object} item Draft row being edited.
 * @param {string} fieldName Draft edit field suffix used in the DOM id.
 * @param {unknown} value Current draft field value.
 * @param {{ type?: string }} options Optional UI5 input attributes for single-line fields.
 * @returns {string} UI5 input or textarea HTML for the edit table cell.
 */
function renderDraftFieldControl(item, fieldName, value, options = {}) {
  const config = getDraftFieldEditorConfig(fieldName);
  const id = `edit-${fieldName}-${escapeHtml(item.rowId)}`;
  const escapedValue = escapeHtml(value ?? "");

  if (config.component === "textarea") {
    return `
      <ui5-textarea
        id="${id}"
        class="draft-field-textarea"
        value="${escapedValue}"
        rows="${escapeHtml(config.rows)}"
        ${config.growing ? "growing" : ""}
      ></ui5-textarea>
    `;
  }

  const typeAttribute = options.type ? ` type="${escapeHtml(options.type)}"` : "";
  return `<ui5-input id="${id}" value="${escapedValue}"${typeAttribute}></ui5-input>`;
}

function renderRows(tableBody, items, expandedRowId) {
  tableBody.innerHTML = items
    .map((item) => {
      const isExpanded = item.rowId === expandedRowId;
      const supplierDisplay = item.supplierName || item.supplierId || item.senderEmail;
      const materialDisplay = item.materialDescription || item.materialCode;
      const candidateMaterials = renderCandidateList(
        "Candidate materials",
        item.candidateMaterials,
        (candidate) => `
          <strong>${escapeHtml(candidate.material_number || candidate.material_code || "")}</strong>
          <div>${escapeHtml(candidate.material_description || "")}</div>
          <div class="review-cell-secondary">${escapeHtml(candidate.currency || "")} ${escapeHtml(candidate.current_price || "")}</div>
        `
      );
      const candidateSuppliers = renderCandidateList(
        "Candidate suppliers",
        item.candidateSuppliers,
        (candidate) => `
          <strong>${escapeHtml(candidate.supplier_id || "")}</strong>
          <div>${escapeHtml(candidate.supplier_name || candidate.company || "")}</div>
          <div class="review-cell-secondary">${escapeHtml(candidate.supplier_email || candidate.email || "")}</div>
        `
      );

      return `
        <tr class="review-row is-${escapeHtml(item.status || "unknown")}${isExpanded ? " is-expanded" : ""}" data-row-id="${escapeHtml(item.rowId)}">
          <td>
            <div class="review-cell-main">
              <span>${formatDisplayValue(supplierDisplay)}</span>
              <span class="review-cell-secondary">${formatDisplayValue(item.supplierEmail || item.senderEmail)}</span>
            </div>
          </td>
          <td>
            <div class="review-cell-main">
              <span>${formatDisplayValue(item.subject)}</span>
              <span class="review-cell-secondary">${escapeHtml(formatEmailSourceLabel(item.gmailMessageId))}</span>
            </div>
          </td>
          <td>
            <div class="review-cell-main">
              <span>${formatDisplayValue(item.materialCode)}</span>
              <span class="review-cell-secondary">${formatDisplayValue(materialDisplay)}</span>
            </div>
          </td>
          <td>${formatMoney(item, "originalPrice")}</td>
          <td>${formatMoney(item, "requestedNewPrice")}</td>
          <td>${formatDisplayValue(item.currency)}</td>
          <td>${formatDisplayValue(item.effectiveFrom)}</td>
          <td>
            <div class="review-cell-main">
              ${statusTag(item.status)}
              <span class="review-cell-secondary">Confidence ${escapeHtml(formatConfidence(item.confidence))}</span>
            </div>
          </td>
          <td class="review-row-actions">
            <div class="decision-actions decision-actions-table" aria-label="Review actions">
              ${renderDraftDecisionButtons(item)}
            </div>
          </td>
        </tr>
        ${
          isExpanded
            ? `
              <tr class="review-expanded-row">
                <td colspan="9">
                  <div class="expanded-card">
                    <div class="expanded-panel">
                      <h3>Draft fields</h3>
                      ${renderValidationErrors(item.validationErrors)}
                      ${renderS4WriteResult(item.s4WriteResult)}
                      <table class="edit-table">
                        <tbody>
                          <tr>
                            <th>Supplier ID</th>
                            <td>${renderDraftFieldControl(item, "supplier-id", item.supplierId)}</td>
                          </tr>
                          <tr>
                            <th>Supplier name</th>
                            <td>${renderDraftFieldControl(item, "supplier-name", item.supplierName)}</td>
                          </tr>
                          <tr>
                            <th>Supplier email</th>
                            <td>${renderDraftFieldControl(item, "supplier-email", item.supplierEmail)}</td>
                          </tr>
                          <tr>
                            <th>Material number</th>
                            <td>${renderDraftFieldControl(item, "material-code", item.materialCode)}</td>
                          </tr>
                          <tr>
                            <th>Material description</th>
                            <td>${renderDraftFieldControl(item, "material-description", item.materialDescription)}</td>
                          </tr>
                          <tr>
                            <th>Original price</th>
                            <td>${renderDraftFieldControl(item, "original-price", item.originalPrice, { type: "Number" })}</td>
                          </tr>
                          <tr>
                            <th>Requested price</th>
                            <td>${renderDraftFieldControl(item, "requested-price", item.requestedNewPrice, { type: "Number" })}</td>
                          </tr>
                          <tr>
                            <th>Currency</th>
                            <td>${renderDraftFieldControl(item, "currency", item.currency)}</td>
                          </tr>
                          <tr>
                            <th>UOM</th>
                            <td>${renderDraftFieldControl(item, "uom", item.uom)}</td>
                          </tr>
                          <tr>
                            <th>Effective from</th>
                            <td><ui5-date-picker id="edit-effective-from-${escapeHtml(item.rowId)}" format-pattern="yyyy-MM-dd" value="${escapeHtml(item.effectiveFrom || "")}"></ui5-date-picker></td>
                          </tr>
                          <tr>
                            <th>Effective to</th>
                            <td><ui5-date-picker id="edit-effective-to-${escapeHtml(item.rowId)}" format-pattern="yyyy-MM-dd" value="${escapeHtml(item.effectiveTo || "")}"></ui5-date-picker></td>
                          </tr>
                          <tr>
                            <th>Explanation</th>
                            <td>${renderDraftFieldControl(item, "explanation", item.explanation)}</td>
                          </tr>
                        </tbody>
                      </table>
                      <div class="edit-actions">
                        <div class="decision-actions decision-actions-edit" aria-label="Review actions">
                          ${renderDraftDecisionButtons(item)}
                        </div>
                        <ui5-button design="Emphasized" class="save-edits-button" data-row-id="${escapeHtml(item.rowId)}">Save</ui5-button>
                      </div>
                      ${candidateMaterials}
                      ${candidateSuppliers}
                    </div>
                    <div class="expanded-panel audit-panel">
                      <h3>Original email</h3>
                      <div class="status-metadata">
                        <ui5-label>Status</ui5-label>
                        <div class="review-cell-pill-row">
                          ${statusTag(item.status)}
                          <span class="review-cell-secondary">Confidence ${escapeHtml(formatConfidence(item.confidence))}</span>
                        </div>
                      </div>
                      <div class="email-metadata">
                        <div class="email-metadata-line">
                          <ui5-label>From</ui5-label>
                          <span>${formatDisplayValue(item.originalEmail.from || item.senderEmail)}</span>
                        </div>
                        <div class="email-metadata-line">
                          <ui5-label>Date</ui5-label>
                          <span>${formatDateTime(item.originalEmail.date)}</span>
                        </div>
                        <div class="email-metadata-line">
                          <ui5-label>Subject</ui5-label>
                          <span>${formatDisplayValue(item.originalEmail.subject)}</span>
                        </div>
                      </div>
                      <div class="email-body">${escapeHtml(item.originalEmail.body)}</div>
                      ${renderAttachmentAuditList(item.attachments || [])}
                    </div>
                  </div>
                </td>
              </tr>
            `
            : ""
        }
      `;
    })
    .join("");
}

function readRowEdits(rowId) {
  const valueOf = (name) => readDraftEditorValue(document.getElementById(`edit-${name}-${rowId}`));
  return {
    supplierId: valueOf("supplier-id"),
    supplierName: valueOf("supplier-name"),
    supplierEmail: valueOf("supplier-email"),
    materialCode: valueOf("material-code"),
    materialDescription: valueOf("material-description"),
    originalPrice: valueOf("original-price"),
    requestedNewPrice: valueOf("requested-price"),
    currency: valueOf("currency"),
    uom: valueOf("uom"),
    effectiveFrom: valueOf("effective-from"),
    effectiveTo: valueOf("effective-to"),
    explanation: valueOf("explanation")
  };
}

/**
 * Read the manual email form fields from the current page.
 *
 * @returns {{ senderEmail: string, subject: string, body: string }} Manual form values.
 */
function readManualEmailForm() {
  return {
    senderEmail: document.getElementById("manual-email-sender")?.value ?? "",
    subject: document.getElementById("manual-email-subject")?.value ?? "",
    body: document.getElementById("manual-email-body")?.value ?? ""
  };
}

/**
 * Clear all manual email form fields after cancel or successful processing.
 *
 * @returns {void}
 */
function clearManualEmailForm() {
  const senderInput = document.getElementById("manual-email-sender");
  const subjectInput = document.getElementById("manual-email-subject");
  const bodyInput = document.getElementById("manual-email-body");
  if (senderInput) senderInput.value = "";
  if (subjectInput) subjectInput.value = "";
  if (bodyInput) bodyInput.value = "";
}

function renderManualAttachmentSelection(container, files) {
  if (!container) return;
  if (!files.length) {
    container.innerHTML = "";
    return;
  }
  container.innerHTML = `
    <div class="manual-attachment-list">
      ${files
        .map(
          (file, index) => `
            <div class="manual-attachment-item">
              <div class="manual-attachment-main">
                <span>${escapeHtml(file.name || "Attachment")}</span>
                <span class="review-cell-secondary">${escapeHtml(formatAttachmentSize(file.size))}</span>
              </div>
              <ui5-button
                design="Transparent"
                icon="delete"
                class="manual-attachment-remove-button"
                data-attachment-index="${escapeHtml(index)}"
              ></ui5-button>
            </div>
          `
        )
        .join("")}
    </div>
  `;
}

function mergeUpdatedDraft(previousItem, apiDraft) {
  const updatedItem = mapDraftFromApi(apiDraft);
  return {
    ...previousItem,
    ...updatedItem,
    senderEmail: updatedItem.senderEmail || previousItem.senderEmail,
    subject: updatedItem.subject || previousItem.subject,
    originalEmail: {
      ...previousItem.originalEmail,
      ...updatedItem.originalEmail,
      from: updatedItem.originalEmail.from || previousItem.originalEmail.from,
      subject: updatedItem.originalEmail.subject || previousItem.originalEmail.subject,
      date: updatedItem.originalEmail.date || previousItem.originalEmail.date,
      body: updatedItem.originalEmail.body || previousItem.originalEmail.body
    }
  };
}

export default function initHomePage() {
  const summaryContainer = document.getElementById("summary-grid");
  const toolbarText = document.getElementById("review-toolbar-text");
  const fetchSummaryText = document.getElementById("fetch-summary-text");
  const processingProgress = document.getElementById("processing-progress");
  const fetchButton = document.getElementById("fetch-emails-button");
  const manualToggleButton = document.getElementById("manual-email-toggle-button");
  const manualForm = document.getElementById("manual-email-form");
  const manualSubmitButton = document.getElementById("manual-email-submit-button");
  const manualCancelButton = document.getElementById("manual-email-cancel-button");
  const manualAttachmentButton = document.getElementById("manual-email-attachments-button");
  const manualAttachmentInput = document.getElementById("manual-email-attachments-input");
  const manualAttachmentList = document.getElementById("manual-email-attachments-list");
  const loadingState = document.getElementById("review-loading");
  const errorState = document.getElementById("review-error");
  const tableContainer = document.getElementById("review-table-container");
  const tableBody = document.getElementById("review-table-body");
  const toast = document.getElementById("review-toast");

  let items = [];
  let expandedRowId = null;
  let manualAttachments = [];
  let processingPollTimer = null;

  const pollProcessingRun = async (processingRunId) => {
    try {
      const snapshot = await getProcessingRun(processingRunId);
      renderProcessingProgress(processingProgress, snapshot);
      return snapshot;
    } catch (error) {
      if (error.status !== 404 && processingProgress) {
        renderInitialProcessingProgress(processingProgress, `Progress unavailable: ${error.message}`);
      }
      return null;
    }
  };

  const stopProgressPolling = () => {
    if (processingPollTimer) {
      clearInterval(processingPollTimer);
      processingPollTimer = null;
    }
  };

  const startProgressPolling = (processingRunId, initialMessage) => {
    stopProgressPolling();
    renderInitialProcessingProgress(processingProgress, initialMessage);
    pollProcessingRun(processingRunId);
    processingPollTimer = setInterval(() => {
      pollProcessingRun(processingRunId);
    }, 1500);
  };

  const clearManualEmailFormAndAttachments = () => {
    clearManualEmailForm();
    manualAttachments = [];
    if (manualAttachmentInput) {
      manualAttachmentInput.value = "";
    }
    renderManualAttachmentSelection(manualAttachmentList, manualAttachments);
  };

  const loadData = async () => {
    try {
      errorState.hidden = true;
      const payload = await getPriceChangeDrafts();
      items = (payload.items || []).map(mapDraftFromApi);
      loadingState.hidden = true;
      tableContainer.hidden = false;
      updateView();
    } catch (error) {
      loadingState.hidden = true;
      errorState.hidden = false;
      errorState.classList.add("error");
      errorState.textContent = `Failed to load draft queue: ${error.message}`;
    }
  };

  const updateView = () => {
    renderSummary(summaryContainer, items);
    renderRows(tableBody, items, expandedRowId);

    const summary = summaryFromItems(items);
    toolbarText.textContent = `${summary.ready} ready, ${summary.needsReview} needing review, ${summary.total} total drafts`;

    tableBody.querySelectorAll(".review-row").forEach((row) => {
      row.addEventListener("click", () => {
        const { rowId } = row.dataset;
        expandedRowId = expandedRowId === rowId ? null : rowId;
        updateView();
      });
    });

    tableBody.querySelectorAll(".save-edits-button").forEach((button) => {
      button.addEventListener("click", async (event) => {
        event.stopPropagation();
        const { rowId } = button.dataset;
        button.disabled = true;
        try {
          const previousItem = items.find((item) => item.rowId === rowId);
          const updatedDraft = await patchPriceChangeDraft(rowId, buildDraftPatchPayload(readRowEdits(rowId)));
          const updatedItem = previousItem ? mergeUpdatedDraft(previousItem, updatedDraft) : mapDraftFromApi(updatedDraft);
          items = items.map((item) => (item.rowId === rowId ? updatedItem : item));
          expandedRowId = rowId;
          if (toast) {
            toast.textContent = `Draft ${rowId} saved.`;
            toast.show();
          }
          updateView();
        } catch (error) {
          if (toast) {
            toast.textContent = `Save failed: ${error.message}`;
            toast.show();
          }
        } finally {
          button.disabled = false;
        }
      });
    });

    tableBody.querySelectorAll(".approve-draft-button").forEach((button) => {
      button.addEventListener("click", async (event) => {
        event.stopPropagation();
        const { rowId } = button.dataset;
        button.disabled = true;
        try {
          const previousItem = items.find((item) => item.rowId === rowId);
          let itemForApproval = previousItem;
          if (expandedRowId === rowId) {
            const updatedDraft = await patchPriceChangeDraft(rowId, buildDraftPatchPayload(readRowEdits(rowId)));
            itemForApproval = previousItem ? mergeUpdatedDraft(previousItem, updatedDraft) : mapDraftFromApi(updatedDraft);
            items = items.map((item) => (item.rowId === rowId ? itemForApproval : item));
          }
          await approvePriceChangeDraft(rowId);
          items = items.filter((item) => item.rowId !== rowId);
          expandedRowId = null;
          if (toast) {
            toast.textContent = `Draft ${rowId} accepted and S/4 price update applied.`;
            toast.show();
          }
          updateView();
        } catch (error) {
          const detailDraft = error.detail?.draft;
          if (detailDraft) {
            const previousItem = items.find((item) => item.rowId === rowId);
            const updatedItem = previousItem ? mergeUpdatedDraft(previousItem, detailDraft) : mapDraftFromApi(detailDraft);
            items = items.map((item) => (item.rowId === rowId ? updatedItem : item));
            expandedRowId = rowId;
            updateView();
          }
          if (toast) {
            toast.textContent = `Accept failed: ${formatApiErrorMessage(error)}`;
            toast.show();
          }
        } finally {
          button.disabled = false;
        }
      });
    });

    tableBody.querySelectorAll(".reject-draft-button").forEach((button) => {
      button.addEventListener("click", async (event) => {
        event.stopPropagation();
        const { rowId } = button.dataset;
        button.disabled = true;
        try {
          await rejectPriceChangeDraft(rowId);
          items = items.filter((item) => item.rowId !== rowId);
          expandedRowId = null;
          if (toast) {
            toast.textContent = `Draft ${rowId} rejected.`;
            toast.show();
          }
          updateView();
        } catch (error) {
          if (toast) {
            toast.textContent = `Reject failed: ${error.message}`;
            toast.show();
          }
        } finally {
          button.disabled = false;
        }
      });
    });

    tableBody.querySelectorAll(".download-attachment-button").forEach((button) => {
      button.addEventListener("click", async (event) => {
        event.stopPropagation();
        const attachmentId = button.dataset.attachmentId;
        const filename = button.dataset.attachmentFilename || "attachment";
        const attachment =
          items.flatMap((item) => item.attachments || []).find((item) => item.attachment_id === attachmentId) || {
            attachment_id: attachmentId,
            filename
          };
        button.disabled = true;
        try {
          await downloadAttachmentFile(attachment);
        } catch (error) {
          if (toast) {
            toast.textContent = `Download failed: ${error.message}`;
            toast.show();
          }
        } finally {
          button.disabled = false;
        }
      });
    });
  };

  fetchButton.addEventListener("click", async () => {
    const processingRunId = createProcessingRunId("gmail-fetch");
    fetchButton.disabled = true;
    fetchButton.textContent = "Fetching";
    try {
      await createProcessingRun(processingRunId, "gmail_fetch");
      startProgressPolling(processingRunId, "Fetching new emails");
      const summary = await fetchNewEmails(processingRunId);
      await pollProcessingRun(processingRunId);
      fetchSummaryText.textContent = summarizeFetchResult(summary);
      await loadData();
    } catch (error) {
      fetchSummaryText.textContent = `Fetch failed: ${error.message}`;
      await pollProcessingRun(processingRunId);
    } finally {
      stopProgressPolling();
      fetchButton.disabled = false;
      fetchButton.textContent = "Fetch new emails";
    }
  });

  manualToggleButton.addEventListener("click", () => {
    manualForm.hidden = !manualForm.hidden;
    if (!manualForm.hidden) {
      document.getElementById("manual-email-sender")?.focus();
    }
  });

  manualCancelButton.addEventListener("click", () => {
    clearManualEmailFormAndAttachments();
    manualForm.hidden = true;
  });

  manualAttachmentButton.addEventListener("click", () => {
    manualAttachmentInput?.click();
  });

  manualAttachmentInput.addEventListener("change", () => {
    manualAttachments = Array.from(manualAttachmentInput.files || []);
    renderManualAttachmentSelection(manualAttachmentList, manualAttachments);
  });

  manualAttachmentList.addEventListener("click", (event) => {
    const button = event.target.closest(".manual-attachment-remove-button");
    if (!button) return;
    const index = Number(button.dataset.attachmentIndex);
    manualAttachments = manualAttachments.filter((_file, fileIndex) => fileIndex !== index);
    if (manualAttachmentInput) {
      manualAttachmentInput.value = "";
    }
    renderManualAttachmentSelection(manualAttachmentList, manualAttachments);
  });

  manualSubmitButton.addEventListener("click", async () => {
    const processingRunId = createProcessingRunId("manual");
    manualSubmitButton.disabled = true;
    manualCancelButton.disabled = true;
    try {
      const formValues = readManualEmailForm();
      await createProcessingRun(processingRunId, "manual");
      startProgressPolling(processingRunId, "Analyzing manual email");
      const summary = manualEmailHasAttachments(manualAttachments)
        ? await submitManualEmailWithAttachments(formValues, manualAttachments, processingRunId)
        : await submitManualEmail(buildManualEmailPayload(formValues, processingRunId));
      await pollProcessingRun(processingRunId);
      fetchSummaryText.textContent = summarizeFetchResult(summary);
      clearManualEmailFormAndAttachments();
      manualForm.hidden = true;
      await loadData();
    } catch (error) {
      fetchSummaryText.textContent = `Manual email failed: ${error.message}`;
      await pollProcessingRun(processingRunId);
    } finally {
      stopProgressPolling();
      manualSubmitButton.disabled = false;
      manualCancelButton.disabled = false;
    }
  });

  loadData();
}
