import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Icon.js";
import "@ui5/webcomponents/dist/Title.js";

import "@ui5/webcomponents-icons/dist/delete.js";
import "@ui5/webcomponents-icons/dist/download.js";
import "@ui5/webcomponents-icons/dist/refresh.js";
import "@ui5/webcomponents-icons/dist/upload.js";
import "@ui5/webcomponents-icons/dist/upload-to-cloud.js";

import { request, requestBlob, requestForm } from "../../services/api.js";

const ASSESSMENT_ID = "demo-assessment";
const DOCUMENT_POLL_INTERVAL_MS = 2500;

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
 * Render the current document list table.
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
          No evidence documents have been uploaded yet.
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
 * Load document metadata from the backend.
 *
 * @returns {Promise<void>} Promise resolved after table state is refreshed.
 */
async function loadDocuments() {
  const payload = await request(
    `/api/documents?assessment_id=${encodeURIComponent(ASSESSMENT_ID)}`
  );
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
    form.append("assessment_id", ASSESSMENT_ID);
    state.selectedFiles.forEach((file) => {
      form.append("files", file, file.name);
    });
    const response = await requestForm("/api/documents", form);
    state.activeJobId = response.job_id;
    clearSelectedFiles();
    showSummary(`Document ingestion job ${response.job_id} created for ${response.document_count} file(s).`, "success");
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
        `/api/documents/ingestion-jobs/${encodeURIComponent(jobId)}`
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
 * Download one stored HANA document blob.
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
    `/api/documents/${encodeURIComponent(documentId)}/download?assessment_id=${encodeURIComponent(ASSESSMENT_ID)}`
  );
  const url = window.URL.createObjectURL(blob);
  const link = window.document.createElement("a");
  link.href = url;
  link.download = document.file_name || "document";
  link.click();
  window.URL.revokeObjectURL(url);
}

/**
 * Delete one document from the HANA corpus and refresh the table.
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
    `/api/documents/${encodeURIComponent(documentId)}?assessment_id=${encodeURIComponent(ASSESSMENT_ID)}`,
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
 * Initialize the Document Manager page after its HTML has mounted.
 *
 * @returns {void}
 */
export default function initHomePage() {
  const selectButton = document.getElementById("select-documents-button");
  const input = document.getElementById("document-file-input");
  const refreshButton = document.getElementById("refresh-documents-button");

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

  attachDropZoneHandlers();
  renderSelectedFiles();
  renderDocumentTable();
  loadDocuments().catch((error) => showSummary(error.message, "error"));
}
