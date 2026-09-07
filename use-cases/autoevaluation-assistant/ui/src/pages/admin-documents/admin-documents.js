import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Icon.js";
import "@ui5/webcomponents/dist/Title.js";

import "@ui5/webcomponents-icons/dist/delete.js";
import "@ui5/webcomponents-icons/dist/download.js";
import "@ui5/webcomponents-icons/dist/refresh.js";
import "@ui5/webcomponents-icons/dist/upload.js";
import "@ui5/webcomponents-icons/dist/upload-to-cloud.js";

import { getLanguage, t } from "../../modules/i18n.js";
import { request, requestBlob, requestForm } from "../../services/api.js";
import {
  applyBenchmarkValidation,
  beginBenchmarkHistoryLoad,
  beginBenchmarkMutation,
  benchmarkHistoryView,
  buildBenchmarkImportForm,
  canActivateBenchmark,
  createBenchmarkAdminState,
  completeBenchmarkHistoryLoad,
  failBenchmarkHistoryLoad,
  formatBenchmarkTimestamp,
  isBenchmarkWorkbook,
  isCurrentBenchmarkHistoryRequest,
  isCurrentBenchmarkSelection,
  selectBenchmarkWorkbook
} from "./benchmarkAdminState.js";

const ADMIN_DOCUMENTS_ENDPOINT = "/api/admin-documents";
const BENCHMARK_IMPORT_ENDPOINT = "/api/import/assessment-benchmarks";
const DOCUMENT_POLL_INTERVAL_MS = 2500;
let benchmarkLanguageListenerAttached = false;
let benchmarkState = createBenchmarkAdminState();

const state = {
  documents: [],
  selectedFiles: [],
  activeJobId: null,
  pollTimer: null
};

/**
 * Escape backend-provided text before rendering it into page HTML.
 *
 * @param {unknown} value - Value to render as text.
 * @returns {string} HTML-escaped string.
 */
function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

/**
 * Format one timestamp for display in the document table.
 *
 * @param {string | null | undefined} value - ISO timestamp from the API.
 * @returns {string} Localized date/time label or a fallback dash.
 */
function formatTimestamp(value) {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "-";
  }
  return date.toLocaleString();
}

/**
 * Format a byte count using browser-native units.
 *
 * @param {number} byteCount - Document size in bytes.
 * @returns {string} Human-readable file size.
 */
function formatBytes(byteCount) {
  const value = Number(byteCount || 0);
  if (value < 1024) {
    return `${value} B`;
  }
  const units = ["KB", "MB", "GB"];
  let size = value / 1024;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size.toFixed(size >= 10 ? 0 : 1)} ${units[unitIndex]}`;
}

/**
 * Show a page-level status message.
 *
 * @param {string} message - User-visible message to show.
 * @param {"info" | "error" | "success"} tone - Visual tone for the message.
 * @returns {void}
 */
function showSummary(message, tone = "info") {
  const summary = document.getElementById("document-manager-summary");
  if (!summary) {
    return;
  }
  summary.hidden = false;
  summary.className = `document-manager-summary ${tone}`;
  summary.textContent = message;
}

/**
 * Hide the page-level status message.
 *
 * @returns {void}
 */
function hideSummary() {
  const summary = document.getElementById("document-manager-summary");
  if (summary) {
    summary.hidden = true;
    summary.textContent = "";
  }
}

/**
 * Apply translated labels for the standalone benchmark administration panel.
 *
 * @returns {void}
 */
function renderBenchmarkLabels() {
  const labels = {
    "benchmark-admin-title": "admin.benchmark.title",
    "benchmark-admin-description": "admin.benchmark.description",
    "benchmark-history-refresh-button": "admin.benchmark.refresh",
    "benchmark-select-button": "admin.benchmark.select",
    "benchmark-validate-button": "admin.benchmark.validate",
    "benchmark-activate-button": "admin.benchmark.activate"
  };
  Object.entries(labels).forEach(([elementId, translationKey]) => {
    const element = document.getElementById(elementId);
    if (element) {
      element.textContent = t(translationKey);
    }
  });
}

/**
 * Show or clear the benchmark-panel request status region.
 *
 * @param {string | null} message - Localized message, or null to hide it.
 * @param {"info" | "error" | "success"} [tone] - Visual status tone.
 * @returns {void}
 */
function showBenchmarkStatus(message, tone = "info") {
  const element = document.getElementById("benchmark-admin-status");
  if (!element) {
    return;
  }
  element.hidden = !message;
  element.className = `benchmark-admin-status ${tone}`;
  element.textContent = message || "";
}

/**
 * Render one safe count dictionary without exposing company identities.
 *
 * @param {Record<string, number> | null | undefined} counts - Aggregate counts by label.
 * @returns {string} Escaped comma-separated count summary.
 */
function renderAggregateCounts(counts) {
  const entries = Object.entries(counts || {});
  return entries.length
    ? entries
        .map(([label, count]) => `${escapeHtml(label)} (${Number(count || 0)})`)
        .join(", ")
    : escapeHtml(t("admin.benchmark.none"));
}

/**
 * Translate a stable public validation code without rendering backend prose.
 *
 * @param {string} code - Machine-readable warning or error category.
 * @param {"warning" | "error"} kind - Generic fallback category.
 * @returns {string} English/Italian operator guidance for the code.
 */
function localizedBenchmarkIssue(code, kind) {
  const key = `admin.benchmark.issue.${code}`;
  const translated = t(key);
  return translated === key
    ? t(`admin.benchmark.issue.${kind}Fallback`)
    : translated;
}

/**
 * Translate persisted import status while retaining a safe generic fallback.
 *
 * @param {string | null | undefined} status - Import lifecycle status.
 * @returns {string} Localized status label.
 */
function localizedBenchmarkStatus(status) {
  const key = `admin.benchmark.importStatus.${status || "unknown"}`;
  const translated = t(key);
  return translated === key ? t("admin.benchmark.importStatus.unknown") : translated;
}

/**
 * Render the active HANA benchmark version or a clear unavailable state.
 *
 * @returns {void}
 */
function renderBenchmarkHistory() {
  const container = document.getElementById("benchmark-active-summary");
  if (!container) {
    return;
  }
  if (benchmarkState.historyLoading) {
    container.className = "benchmark-active-summary";
    container.textContent = t("admin.benchmark.loadingHistory");
    return;
  }
  const view = benchmarkHistoryView(benchmarkState.history);
  if (view.kind === "unavailable") {
    container.className = "benchmark-active-summary unavailable";
    container.innerHTML = `<strong>${escapeHtml(t("admin.benchmark.noActiveTitle"))}</strong><p>${escapeHtml(t("admin.benchmark.noActiveDetail"))}</p>`;
    return;
  }
  const active = view.active;
  container.className = "benchmark-active-summary";
  container.innerHTML = `
    <strong>${escapeHtml(t("admin.benchmark.activeTitle"))}</strong>
    <dl class="benchmark-summary-grid">
      <div><dt>${escapeHtml(t("admin.benchmark.file"))}</dt><dd>${escapeHtml(active.source_filename)}</dd></div>
      <div><dt>${escapeHtml(t("admin.benchmark.status"))}</dt><dd>${escapeHtml(localizedBenchmarkStatus(active.status))}</dd></div>
      <div><dt>${escapeHtml(t("admin.benchmark.activated"))}</dt><dd>${escapeHtml(formatBenchmarkTimestamp(active.activated_at, getLanguage()))}</dd></div>
      <div><dt>${escapeHtml(t("admin.benchmark.importId"))}</dt><dd>${escapeHtml(active.import_id)}</dd></div>
      <div><dt>${escapeHtml(t("admin.benchmark.rows"))}</dt><dd>${Number(active.row_count || 0)}</dd></div>
      <div><dt>${escapeHtml(t("admin.benchmark.companies"))}</dt><dd>${Number(active.company_count || 0)}</dd></div>
      <div><dt>${escapeHtml(t("admin.benchmark.questionnaires"))}</dt><dd>${Number(active.questionnaire_count || 0)}</dd></div>
      <div><dt>${escapeHtml(t("admin.benchmark.questions"))}</dt><dd>${Number(active.question_count || 0)}</dd></div>
    </dl>
  `;
}

/**
 * Render dry-run counts, aggregate warnings, and sampled safe row errors.
 *
 * @returns {void}
 */
function renderBenchmarkValidation() {
  const container = document.getElementById("benchmark-validation-result");
  if (!container) {
    return;
  }
  const validation = benchmarkState.validation;
  if (!validation) {
    container.hidden = true;
    container.innerHTML = "";
    return;
  }
  const warnings = validation.warnings || [];
  const errors = validation.sampled_errors || [];
  container.hidden = false;
  container.innerHTML = `
    <h4>${escapeHtml(
      validation.success
        ? t("admin.benchmark.validationPassed")
        : t("admin.benchmark.validationFailed")
    )}</h4>
    <code class="benchmark-hash">SHA-256: ${escapeHtml(validation.source_sha256 || "-")}</code>
    <dl class="benchmark-count-grid">
      <div><dt>${escapeHtml(t("admin.benchmark.rows"))}</dt><dd>${Number(validation.row_count || 0)}</dd></div>
      <div><dt>${escapeHtml(t("admin.benchmark.companies"))}</dt><dd>${Number(validation.company_count || 0)}</dd></div>
      <div><dt>${escapeHtml(t("admin.benchmark.questionnaires"))}</dt><dd>${Number(validation.questionnaire_count || 0)}</dd></div>
      <div><dt>${escapeHtml(t("admin.benchmark.questions"))}</dt><dd>${Number(validation.question_count || 0)}</dd></div>
      <div><dt>${escapeHtml(t("admin.benchmark.accepted"))}</dt><dd>${Number(validation.accepted_count || 0)}</dd></div>
      <div><dt>${escapeHtml(t("admin.benchmark.rejected"))}</dt><dd>${Number(validation.rejected_count || 0)}</dd></div>
      <div><dt>${escapeHtml(t("admin.benchmark.classes"))}</dt><dd>${renderAggregateCounts(validation.class_counts)}</dd></div>
      <div><dt>${escapeHtml(t("admin.benchmark.naceCohorts"))}</dt><dd>${renderAggregateCounts(validation.nace1_counts)}</dd></div>
    </dl>
    <h5>${escapeHtml(t("admin.benchmark.warnings", { count: warnings.length }))}</h5>
    ${
      warnings.length
        ? `<ul>${warnings
            .map(
              (warning) =>
                `<li><strong>${escapeHtml(warning.code)}</strong> (${Number(warning.count || 0)}): ${escapeHtml(localizedBenchmarkIssue(warning.code, "warning"))}${
                  warning.row_numbers?.length
                    ? ` — ${escapeHtml(t("admin.benchmark.rowsList", { rows: warning.row_numbers.join(", ") }))}`
                    : ""
                }</li>`
            )
            .join("")}</ul>`
        : `<p>${escapeHtml(t("admin.benchmark.noWarnings"))}</p>`
    }
    <h5>${escapeHtml(t("admin.benchmark.sampledErrors", { count: errors.length }))}</h5>
    ${
      errors.length
        ? `<ul>${errors
            .map(
              (error) =>
                `<li>${escapeHtml(t("admin.benchmark.row", { row: error.row_number }))}: <strong>${escapeHtml(error.code)}</strong> — ${escapeHtml(localizedBenchmarkIssue(error.code, "error"))}</li>`
            )
            .join("")}</ul>`
        : `<p>${escapeHtml(t("admin.benchmark.noErrors"))}</p>`
    }
  `;
}

/**
 * Synchronize selected-file text, busy states, and activation permission.
 *
 * @returns {void}
 */
function renderBenchmarkControls() {
  const fileLabel = document.getElementById("benchmark-selected-file");
  const validateButton = document.getElementById("benchmark-validate-button");
  const activateButton = document.getElementById("benchmark-activate-button");
  const selectButton = document.getElementById("benchmark-select-button");
  const refreshButton = document.getElementById("benchmark-history-refresh-button");
  if (fileLabel) {
    fileLabel.textContent = benchmarkState.selectedFile
      ? `${benchmarkState.selectedFile.name} · ${formatBytes(benchmarkState.selectedFile.size)}`
      : t("admin.benchmark.noSelection");
  }
  const mutationBusy = Boolean(benchmarkState.mutationLoading);
  if (validateButton) {
    validateButton.disabled = mutationBusy || !benchmarkState.selectedFile;
    validateButton.loading = benchmarkState.mutationLoading === "validation";
  }
  if (activateButton) {
    activateButton.disabled = !canActivateBenchmark(benchmarkState);
    activateButton.loading = benchmarkState.mutationLoading === "activation";
  }
  if (selectButton) {
    selectButton.disabled = mutationBusy;
  }
  if (refreshButton) {
    refreshButton.disabled = mutationBusy || benchmarkState.historyLoading;
    refreshButton.loading = benchmarkState.historyLoading;
  }
  renderBenchmarkHistory();
  renderBenchmarkValidation();
}

/**
 * Load safe active/recent import metadata for the admin status card.
 *
 * @param {boolean} [preserveStatus] - Keep an activation success message visible.
 * @returns {Promise<void>} Promise resolved after the active summary renders.
 */
async function loadBenchmarkHistory(preserveStatus = false) {
  const historyRequest = beginBenchmarkHistoryLoad(benchmarkState);
  benchmarkState = historyRequest.state;
  renderBenchmarkControls();
  try {
    const history = await request(BENCHMARK_IMPORT_ENDPOINT);
    if (!isCurrentBenchmarkHistoryRequest(benchmarkState, historyRequest.requestRevision)) {
      return;
    }
    benchmarkState = completeBenchmarkHistoryLoad(
      benchmarkState,
      history,
      historyRequest.requestRevision
    );
    if (!preserveStatus && !benchmarkState.mutationLoading) {
      showBenchmarkStatus(null);
    }
  } catch (error) {
    if (!isCurrentBenchmarkHistoryRequest(benchmarkState, historyRequest.requestRevision)) {
      return;
    }
    benchmarkState = failBenchmarkHistoryLoad(
      benchmarkState,
      error.message,
      historyRequest.requestRevision
    );
    if (!benchmarkState.mutationLoading) {
      showBenchmarkStatus(t("admin.benchmark.historyFailed"), "error");
    }
  }
  renderBenchmarkControls();
}

/**
 * Validate the selected workbook without writing or activating any HANA data.
 *
 * @returns {Promise<void>} Promise resolved after safe validation results render.
 */
async function validateBenchmarkWorkbook() {
  if (!benchmarkState.selectedFile) {
    showBenchmarkStatus(t("admin.benchmark.selectRequired"), "error");
    return;
  }
  const file = benchmarkState.selectedFile;
  const selectionRevision = benchmarkState.selectionRevision;
  benchmarkState = beginBenchmarkMutation(benchmarkState, "validation");
  showBenchmarkStatus(t("admin.benchmark.validating"), "info");
  renderBenchmarkControls();
  try {
    const validation = await requestForm(
      BENCHMARK_IMPORT_ENDPOINT,
      buildBenchmarkImportForm(file, false)
    );
    if (!isCurrentBenchmarkSelection(benchmarkState, selectionRevision)) {
      return;
    }
    benchmarkState = applyBenchmarkValidation(
      benchmarkState,
      validation,
      selectionRevision
    );
    showBenchmarkStatus(
      validation.success
        ? t("admin.benchmark.validationReady")
        : t("admin.benchmark.validationFailed"),
      validation.success ? "success" : "error"
    );
  } catch (error) {
    if (!isCurrentBenchmarkSelection(benchmarkState, selectionRevision)) {
      return;
    }
    const validation = error.detail?.validation || null;
    benchmarkState = validation
      ? applyBenchmarkValidation(benchmarkState, validation, selectionRevision)
      : { ...benchmarkState, mutationLoading: null, error: error.message };
    showBenchmarkStatus(
      validation
        ? t("admin.benchmark.validationFailed")
        : t("admin.benchmark.validationRequestFailed"),
      "error"
    );
  }
  renderBenchmarkControls();
}

/**
 * Explicitly activate the currently validated workbook and refresh HANA status.
 *
 * @returns {Promise<void>} Promise resolved after activation/history refresh.
 */
async function activateBenchmarkWorkbook() {
  if (!canActivateBenchmark(benchmarkState)) {
    showBenchmarkStatus(t("admin.benchmark.validationRequired"), "error");
    return;
  }
  if (!window.confirm(t("admin.benchmark.confirmActivation"))) {
    return;
  }
  const file = benchmarkState.selectedFile;
  const selectionRevision = benchmarkState.selectionRevision;
  benchmarkState = beginBenchmarkMutation(benchmarkState, "activation");
  showBenchmarkStatus(t("admin.benchmark.activating"), "info");
  renderBenchmarkControls();
  try {
    const result = await requestForm(
      BENCHMARK_IMPORT_ENDPOINT,
      buildBenchmarkImportForm(file, true)
    );
    if (!isCurrentBenchmarkSelection(benchmarkState, selectionRevision)) {
      return;
    }
    benchmarkState = applyBenchmarkValidation(
      benchmarkState,
      result,
      selectionRevision
    );
    showBenchmarkStatus(
      result.no_op ? t("admin.benchmark.alreadyActive") : t("admin.benchmark.activationDone"),
      "success"
    );
    await loadBenchmarkHistory(true);
  } catch (error) {
    if (!isCurrentBenchmarkSelection(benchmarkState, selectionRevision)) {
      return;
    }
    const validation = error.detail?.validation || null;
    benchmarkState = validation
      ? applyBenchmarkValidation(benchmarkState, validation, selectionRevision)
      : { ...benchmarkState, mutationLoading: null, error: error.message };
    showBenchmarkStatus(t("admin.benchmark.activationFailed"), "error");
  }
  renderBenchmarkControls();
}

/**
 * Replace the selected benchmark workbook and clear stale validation state.
 *
 * @param {File | null} file - Browser file selected through the XLSX input.
 * @returns {void}
 */
function handleBenchmarkFileSelection(file) {
  benchmarkState = selectBenchmarkWorkbook(benchmarkState, null);
  if (file && !isBenchmarkWorkbook(file)) {
    showBenchmarkStatus(t("admin.benchmark.xlsxOnly"), "error");
    renderBenchmarkControls();
    return;
  }
  benchmarkState = selectBenchmarkWorkbook(benchmarkState, file);
  showBenchmarkStatus(null);
  renderBenchmarkControls();
}

/**
 * Render the selected upload queue.
 *
 * @returns {void}
 */
function renderSelectedFiles() {
  const container = document.getElementById("selected-document-files");
  if (!container) {
    return;
  }
  if (state.selectedFiles.length === 0) {
    container.hidden = true;
    container.innerHTML = "";
    return;
  }

  container.hidden = false;
  container.innerHTML = `
    <div class="selected-file-list">
      ${state.selectedFiles
        .map(
          (file) => `
            <span class="selected-file-chip">
              <strong>${escapeHtml(file.name)}</strong>
              <span>${escapeHtml(formatBytes(file.size))}</span>
            </span>
          `
        )
        .join("")}
    </div>
    <div class="selected-file-actions">
      <ui5-button id="clear-selected-documents-button">Clear</ui5-button>
      <ui5-button id="upload-selected-documents-button" design="Emphasized" icon="upload">Start upload</ui5-button>
    </div>
  `;

  document
    .getElementById("clear-selected-documents-button")
    ?.addEventListener("click", clearSelectedFiles);
  document
    .getElementById("upload-selected-documents-button")
    ?.addEventListener("click", () => {
      uploadSelectedFiles().catch((error) => showSummary(error.message, "error"));
    });
}

/**
 * Render the current admin document list table.
 *
 * @returns {void}
 */
function renderDocumentTable() {
  const body = document.getElementById("document-table-body");
  const countLabel = document.getElementById("document-count-label");
  if (countLabel) {
    countLabel.textContent = `${state.documents.length} file${state.documents.length === 1 ? "" : "s"}`;
  }
  if (!body) {
    return;
  }

  if (state.documents.length === 0) {
    body.innerHTML = `
      <tr>
        <td colspan="5" class="document-empty-row">
          No admin documents have been uploaded yet.
        </td>
      </tr>
    `;
    return;
  }

  body.innerHTML = state.documents
    .map(
      (document) => `
        <tr>
          <td>
            <strong>${escapeHtml(document.file_name)}</strong>
            <span>${escapeHtml(document.content_type || "application/octet-stream")} · ${escapeHtml(formatBytes(document.file_size))}</span>
          </td>
          <td><span class="document-status ${escapeHtml(document.status)}">${escapeHtml(document.status)}</span></td>
          <td>${Number(document.chunk_count || 0)}</td>
          <td>${escapeHtml(formatTimestamp(document.created_at))}</td>
          <td class="document-row-actions">
            <ui5-button icon="download" design="Transparent" data-download-document="${escapeHtml(document.document_id)}" tooltip="Download"></ui5-button>
            <ui5-button icon="delete" design="Transparent" data-delete-document="${escapeHtml(document.document_id)}" tooltip="Delete"></ui5-button>
          </td>
        </tr>
      `
    )
    .join("");

  body.querySelectorAll("[data-download-document]").forEach((button) => {
    button.addEventListener("click", () => {
      downloadDocument(button.dataset.downloadDocument).catch((error) =>
        showSummary(error.message, "error")
      );
    });
  });
  body.querySelectorAll("[data-delete-document]").forEach((button) => {
    button.addEventListener("click", () => {
      deleteDocument(button.dataset.deleteDocument).catch((error) =>
        showSummary(error.message, "error")
      );
    });
  });
}

/**
 * Load admin document metadata from the backend.
 *
 * @returns {Promise<void>} Promise resolved after table state is refreshed.
 */
async function loadDocuments() {
  const payload = await request(ADMIN_DOCUMENTS_ENDPOINT);
  state.documents = payload.documents || [];
  renderDocumentTable();
}

/**
 * Add files selected through input or drag-and-drop to the upload queue.
 *
 * @param {File[]} files - Browser File objects selected by the user.
 * @returns {void}
 */
function addSelectedFiles(files) {
  const allowedExtensions = [".pdf", ".docx", ".xlsx", ".xlsm", ".eml"];
  const acceptedFiles = files.filter((file) =>
    allowedExtensions.some((extension) => file.name.toLowerCase().endsWith(extension))
  );
  state.selectedFiles = [...state.selectedFiles, ...acceptedFiles];
  renderSelectedFiles();
  if (acceptedFiles.length !== files.length) {
    showSummary("Only PDF, Word, Excel, and email documents can be uploaded.", "error");
  } else {
    hideSummary();
  }
}

/**
 * Clear the local upload queue.
 *
 * @returns {void}
 */
function clearSelectedFiles() {
  state.selectedFiles = [];
  const input = document.getElementById("document-file-input");
  if (input) {
    input.value = "";
  }
  renderSelectedFiles();
}

/**
 * Upload queued files and start polling the ingestion job.
 *
 * @returns {Promise<void>} Promise resolved after the upload request succeeds.
 */
async function uploadSelectedFiles() {
  if (state.selectedFiles.length === 0) {
    showSummary("Select at least one document before uploading.", "error");
    return;
  }

  const uploadButton = document.getElementById("upload-selected-documents-button");
  if (uploadButton) {
    uploadButton.loading = true;
  }
  try {
    const form = new FormData();
    state.selectedFiles.forEach((file) => {
      form.append("files", file, file.name);
    });
    const response = await requestForm(ADMIN_DOCUMENTS_ENDPOINT, form);
    state.activeJobId = response.job_id;
    clearSelectedFiles();
    showSummary(`Admin ingestion job ${response.job_id} created for ${response.document_count} file(s).`, "success");
    await loadDocuments();
    startIngestionPolling(response.job_id);
  } finally {
    if (uploadButton) {
      uploadButton.loading = false;
    }
  }
}

/**
 * Poll one ingestion job until it reaches a terminal state.
 *
 * @param {string} jobId - Document ingestion job identifier.
 * @returns {void}
 */
function startIngestionPolling(jobId) {
  stopIngestionPolling();
  const poll = async () => {
    try {
      const payload = await request(
        `${ADMIN_DOCUMENTS_ENDPOINT}/ingestion-jobs/${encodeURIComponent(jobId)}`
      );
      await loadDocuments();
      showSummary(
        `Ingestion ${payload.status}: ${payload.processed_document_count}/${payload.document_count} file(s), ${payload.indexed_chunk_count} chunk(s).`,
        payload.status === "failed" ? "error" : "info"
      );
      if (["completed", "failed"].includes(payload.status)) {
        stopIngestionPolling();
        state.activeJobId = null;
        if (payload.status === "completed") {
          showSummary(`Ingestion completed with ${payload.indexed_chunk_count} chunk(s).`, "success");
        }
      }
    } catch (error) {
      stopIngestionPolling();
      showSummary(error.message, "error");
    }
  };

  state.pollTimer = window.setInterval(poll, DOCUMENT_POLL_INTERVAL_MS);
  poll();
}

/**
 * Stop any active ingestion polling timer.
 *
 * @returns {void}
 */
function stopIngestionPolling() {
  if (state.pollTimer) {
    window.clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

/**
 * Download one stored HANA admin document blob.
 *
 * @param {string} documentId - Document identifier selected by the user.
 * @returns {Promise<void>} Promise resolved after the browser download starts.
 */
async function downloadDocument(documentId) {
  const document = state.documents.find((item) => item.document_id === documentId);
  if (!document) {
    return;
  }
  const blob = await requestBlob(
    `${ADMIN_DOCUMENTS_ENDPOINT}/${encodeURIComponent(documentId)}/download`
  );
  const url = window.URL.createObjectURL(blob);
  const link = window.document.createElement("a");
  link.href = url;
  link.download = document.file_name || "document";
  link.click();
  window.URL.revokeObjectURL(url);
}

/**
 * Delete one admin document from the HANA corpus and refresh the table.
 *
 * @param {string} documentId - Document identifier selected by the user.
 * @returns {Promise<void>} Promise resolved after deletion and refresh complete.
 */
async function deleteDocument(documentId) {
  const document = state.documents.find((item) => item.document_id === documentId);
  if (!document) {
    return;
  }
  const confirmed = window.confirm(`Delete ${document.file_name}?`);
  if (!confirmed) {
    return;
  }
  await request(
    `${ADMIN_DOCUMENTS_ENDPOINT}/${encodeURIComponent(documentId)}`,
    "DELETE"
  );
  showSummary(`${document.file_name} deleted.`, "success");
  await loadDocuments();
}

/**
 * Wire drag-and-drop upload behavior for the drop zone.
 *
 * @returns {void}
 */
function attachDropZoneHandlers() {
  const dropZone = document.getElementById("document-drop-zone");
  if (!dropZone) {
    return;
  }
  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.add("drag-over");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.remove("drag-over");
    });
  });
  dropZone.addEventListener("drop", (event) => {
    addSelectedFiles(Array.from(event.dataTransfer?.files || []));
  });
}

/**
 * Initialize the Super Admin Document Manager page after its HTML has mounted.
 *
 * @returns {void}
 */
export default function initAdminDocumentsPage() {
  const selectButton = document.getElementById("select-documents-button");
  const input = document.getElementById("document-file-input");
  const refreshButton = document.getElementById("refresh-documents-button");
  const benchmarkInput = document.getElementById("benchmark-workbook-input");
  const benchmarkSelectButton = document.getElementById("benchmark-select-button");
  const benchmarkValidateButton = document.getElementById("benchmark-validate-button");
  const benchmarkActivateButton = document.getElementById("benchmark-activate-button");
  const benchmarkRefreshButton = document.getElementById("benchmark-history-refresh-button");

  // Page navigation remounts this DOM; each mount starts with no trusted local
  // validation so activation can never inherit permission from an older file.
  benchmarkState = createBenchmarkAdminState();
  renderBenchmarkLabels();
  renderBenchmarkControls();

  selectButton?.addEventListener("click", () => input?.click());
  input?.addEventListener("change", () => {
    addSelectedFiles(Array.from(input.files || []));
    input.value = "";
  });
  refreshButton?.addEventListener("click", () => {
    refreshButton.loading = true;
    loadDocuments()
      .catch((error) => showSummary(error.message, "error"))
      .finally(() => {
        refreshButton.loading = false;
      });
  });

  benchmarkSelectButton?.addEventListener("click", () => benchmarkInput?.click());
  benchmarkInput?.addEventListener("change", () => {
    handleBenchmarkFileSelection(benchmarkInput.files?.[0] || null);
    benchmarkInput.value = "";
  });
  benchmarkValidateButton?.addEventListener("click", () => {
    validateBenchmarkWorkbook();
  });
  benchmarkActivateButton?.addEventListener("click", () => {
    activateBenchmarkWorkbook();
  });
  benchmarkRefreshButton?.addEventListener("click", () => {
    loadBenchmarkHistory();
  });

  if (!benchmarkLanguageListenerAttached) {
    document.addEventListener("language-change", () => {
      if (!document.getElementById("benchmark-admin-title")) {
        return;
      }
      renderBenchmarkLabels();
      renderBenchmarkControls();
    });
    benchmarkLanguageListenerAttached = true;
  }

  attachDropZoneHandlers();
  renderSelectedFiles();
  renderDocumentTable();
  loadDocuments().catch((error) => showSummary(error.message, "error"));
  loadBenchmarkHistory();
}
