import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Dialog.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/Toast.js";

import { request } from "../../services/api.js";
import { formatFileSize, mapDraftFromApi } from "../home/reviewHelpers.js";
import {
  attachPaginationHandlers,
  normalizePagination,
  paginatedEndpoint,
  renderPaginationControls
} from "../shared/pagination.js";

async function getHistory(page = 1) {
  return request(paginatedEndpoint("/api/document-requests/history", page));
}

async function clearHistory() {
  return request("/api/document-requests/history", "DELETE");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function statusLabel(status) {
  return status === "approved" ? "Approved" : status === "rejected" ? "Rejected" : status || "Unknown";
}

function statusTag(status) {
  return `<span class="status-pill status-pill-${escapeHtml(status || "unknown")}">${escapeHtml(statusLabel(status))}</span>`;
}

function summaryFromMetadata(meta, items) {
  const summary = meta?.summary || {};
  const approved = Number.isFinite(Number(summary.approved))
    ? Number(summary.approved)
    : items.filter((item) => item.status === "approved").length;
  const rejected = Number.isFinite(Number(summary.rejected))
    ? Number(summary.rejected)
    : items.filter((item) => item.status === "rejected").length;
  const total = Number.isFinite(Number(summary.total)) ? Number(summary.total) : items.length;
  return { total, approved, rejected };
}

function renderSummary(summaryContainer, summary) {
  summaryContainer.innerHTML = [
    { label: "Past requests", value: summary.total },
    { label: "Approved", value: summary.approved },
    { label: "Rejected", value: summary.rejected }
  ]
    .map((card) => `
      <div class="summary-card">
        <span class="summary-label">${escapeHtml(card.label)}</span>
        <div class="summary-value">${escapeHtml(card.value)}</div>
      </div>
    `)
    .join("");
}

function renderAttachments(attachments) {
  if (!attachments.length) return "No attachments";
  return attachments
    .map((attachment) => `${attachment.fileName} (${formatFileSize(attachment.sizeBytes)})`)
    .join(", ");
}

function renderRows(tableBody, items) {
  tableBody.innerHTML = items
    .map((item) => `
      <tr class="review-row is-${escapeHtml(item.status || "unknown")}">
        <td>${escapeHtml(item.originalEmail.from || item.toEmail || "Unknown")}</td>
        <td>${escapeHtml(item.country || "Not set")}</td>
        <td>${escapeHtml(item.subject)}</td>
        <td>${escapeHtml(renderAttachments(item.attachments))}</td>
        <td>${statusTag(item.status)}</td>
      </tr>
    `)
    .join("");
}

export default function initPastPage() {
  const summaryContainer = document.getElementById("summary-grid");
  const toolbarText = document.getElementById("review-toolbar-text");
  const loadingState = document.getElementById("review-loading");
  const errorState = document.getElementById("review-error");
  const tableContainer = document.getElementById("review-table-container");
  const tableBody = document.getElementById("review-table-body");
  const paginationContainer = document.getElementById("review-pagination");
  const clearButton = document.getElementById("clear-history-button");
  const clearDialog = document.getElementById("clear-history-dialog");
  const confirmClearButton = document.getElementById("confirm-clear-history-button");
  const cancelClearButton = document.getElementById("cancel-clear-history-button");
  const toast = document.getElementById("review-toast");

  let items = [];
  let currentPage = 1;
  let paginationMeta = normalizePagination({});

  const showToast = (message) => {
    if (!toast) return;
    toast.textContent = message;
    toast.show();
  };

  const updateView = () => {
    const summary = summaryFromMetadata(paginationMeta, items);
    renderSummary(summaryContainer, summary);
    renderRows(tableBody, items);
    toolbarText.textContent = `${summary.approved} approved, ${summary.rejected} rejected, ${summary.total} total`;
    if (clearButton) {
      clearButton.disabled = summary.total === 0;
    }
    renderPaginationControls(paginationContainer, paginationMeta);
    attachPaginationHandlers(paginationContainer, paginationMeta, async (page) => {
      await loadData(page);
    });
  };

  const loadData = async (page = currentPage) => {
    try {
      errorState.hidden = true;
      const payload = await getHistory(page);
      const nextMeta = normalizePagination(payload);
      if (page > nextMeta.totalPages) {
        await loadData(nextMeta.totalPages);
        return;
      }
      currentPage = nextMeta.page;
      paginationMeta = nextMeta;
      items = (payload.items || []).map(mapDraftFromApi);
      loadingState.hidden = true;
      tableContainer.hidden = false;
      updateView();
    } catch (error) {
      loadingState.hidden = true;
      errorState.hidden = false;
      errorState.classList.add("error");
      errorState.textContent = `Failed to load history: ${error.message}`;
    }
  };

  clearButton?.addEventListener("click", () => {
    clearDialog.open = true;
  });

  cancelClearButton?.addEventListener("click", () => {
    clearDialog.open = false;
  });

  confirmClearButton?.addEventListener("click", async () => {
    confirmClearButton.disabled = true;
    clearButton.disabled = true;
    try {
      const result = await clearHistory();
      clearDialog.open = false;
      showToast(`Cleared ${result.drafts} requests, ${result.extractions} extractions, ${result.emails} emails.`);
      await loadData(1);
    } catch (error) {
      showToast(`Clear failed: ${error.message}`);
    } finally {
      confirmClearButton.disabled = false;
      if (clearButton) {
        clearButton.disabled = paginationMeta.total === 0;
      }
    }
  });

  loadData();
}
