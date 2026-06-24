import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Dialog.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents-icons/dist/delete.js";
import "@ui5/webcomponents-icons/dist/navigation-left-arrow.js";
import "@ui5/webcomponents-icons/dist/navigation-right-arrow.js";

import { request } from "../../services/api.js";
import { mapDraftFromApi } from "../home/reviewHelpers.js";
import {
  HISTORY_PAGE_SIZE,
  buildHistoryEndpoint,
  buildHistoryToolbarText,
  historyPageCount
} from "./pastHelpers.js";

/**
 * Fetch one page of approved/rejected price-change history.
 *
 * @param {number} offset Number of newest-first rows to skip.
 * @returns {Promise<object>} Paginated history API response.
 */
async function getPriceChangeHistory(offset) {
  return request(buildHistoryEndpoint({ limit: HISTORY_PAGE_SIZE, offset }));
}

/**
 * Delete all historical approved/rejected price-change rows.
 *
 * @returns {Promise<object>} Reset response containing the deleted row count.
 */
async function resetPriceChangeHistory() {
  return request("/api/price-change-history", "DELETE");
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

function statusLabel(status) {
  const labels = {
    approved: "Approved",
    rejected: "Rejected"
  };
  return labels[status] || status || "Unknown";
}

function statusTag(status) {
  return `<span class="status-pill status-pill-${escapeHtml(status || "unknown")}">${escapeHtml(statusLabel(status))}</span>`;
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

function renderSummary(summaryContainer, summary) {
  const cards = [
    { label: "Past requests", value: summary.total },
    { label: "Approved", value: summary.approved },
    { label: "Rejected", value: summary.rejected }
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

function readOnlyRow(label, value) {
  return `
    <tr>
      <th>${escapeHtml(label)}</th>
      <td><span class="read-only-value">${formatDisplayValue(value)}</span></td>
    </tr>
  `;
}

function renderRows(tableBody, items, expandedRowId) {
  tableBody.innerHTML = items
    .map((item) => {
      const isExpanded = item.rowId === expandedRowId;
      const supplierDisplay = item.supplierName || item.supplierId || item.senderEmail;
      const materialDisplay = item.materialDescription || item.materialCode;

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
              <span class="review-cell-secondary">Gmail ${escapeHtml(item.gmailMessageId || "")}</span>
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
        </tr>
        ${
          isExpanded
            ? `
              <tr class="review-expanded-row">
                <td colspan="8">
                  <div class="expanded-card">
                    <div class="expanded-panel">
                      <h3>Final values</h3>
                      <table class="edit-table">
                        <tbody>
                          ${readOnlyRow("Supplier ID", item.supplierId)}
                          ${readOnlyRow("Supplier name", item.supplierName)}
                          ${readOnlyRow("Supplier email", item.supplierEmail)}
                          ${readOnlyRow("Material number", item.materialCode)}
                          ${readOnlyRow("Material description", item.materialDescription)}
                          ${readOnlyRow("Original price", item.originalPrice)}
                          ${readOnlyRow("Approved price", item.requestedNewPrice)}
                          ${readOnlyRow("Currency", item.currency)}
                          ${readOnlyRow("UOM", item.uom)}
                          ${readOnlyRow("Effective from", item.effectiveFrom)}
                          ${readOnlyRow("Effective to", item.effectiveTo)}
                          ${readOnlyRow("Explanation", item.explanation)}
                        </tbody>
                      </table>
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

export default function initPastPage() {
  const summaryContainer = document.getElementById("summary-grid");
  const toolbarText = document.getElementById("review-toolbar-text");
  const loadingState = document.getElementById("review-loading");
  const errorState = document.getElementById("review-error");
  const tableContainer = document.getElementById("review-table-container");
  const tableBody = document.getElementById("review-table-body");
  const previousButton = document.getElementById("history-previous-button");
  const nextButton = document.getElementById("history-next-button");
  const pageIndicator = document.getElementById("history-page-indicator");
  const resetButton = document.getElementById("history-reset-button");
  const resetDialog = document.getElementById("history-reset-dialog");
  const resetNoButton = document.getElementById("history-reset-no-button");
  const resetYesButton = document.getElementById("history-reset-yes-button");

  let items = [];
  let expandedRowId = null;
  let currentPage = 1;
  let summary = {
    total: 0,
    approved: 0,
    rejected: 0,
    limit: HISTORY_PAGE_SIZE,
    offset: 0
  };

  const updateView = () => {
    renderSummary(summaryContainer, summary);
    renderRows(tableBody, items, expandedRowId);

    const pageCount = historyPageCount(summary.total, summary.limit);
    toolbarText.textContent = buildHistoryToolbarText(summary);
    pageIndicator.textContent = `Page ${currentPage} of ${pageCount}`;
    previousButton.disabled = currentPage <= 1;
    nextButton.disabled = currentPage >= pageCount;
    resetButton.disabled = summary.total <= 0;

    tableBody.querySelectorAll(".review-row").forEach((row) => {
      row.addEventListener("click", () => {
        const { rowId } = row.dataset;
        expandedRowId = expandedRowId === rowId ? null : rowId;
        updateView();
      });
    });
  };

  /**
   * Show one page-level error message.
   *
   * @param {string} message User-facing error text.
   * @returns {void}
   */
  const showError = (message) => {
    errorState.hidden = false;
    errorState.classList.add("error");
    errorState.textContent = message;
  };

  const loadData = async (page = currentPage) => {
    try {
      errorState.hidden = true;
      loadingState.hidden = false;
      tableContainer.hidden = true;
      const offset = (page - 1) * HISTORY_PAGE_SIZE;
      const payload = await getPriceChangeHistory(offset);
      items = (payload.items || []).map(mapDraftFromApi);
      summary = {
        total: Number(payload.total || 0),
        approved: Number(payload.approved || 0),
        rejected: Number(payload.rejected || 0),
        limit: Number(payload.limit || HISTORY_PAGE_SIZE),
        offset: Number(payload.offset || offset)
      };
      const pageCount = historyPageCount(summary.total, summary.limit);
      if (page > pageCount) {
        currentPage = pageCount;
        await loadData(currentPage);
        return;
      }
      currentPage = page;
      expandedRowId = null;
      loadingState.hidden = true;
      tableContainer.hidden = false;
      updateView();
    } catch (error) {
      loadingState.hidden = true;
      showError(`Failed to load past requests: ${error.message}`);
    }
  };

  previousButton.addEventListener("click", () => {
    if (currentPage > 1) {
      loadData(currentPage - 1);
    }
  });

  nextButton.addEventListener("click", () => {
    const pageCount = historyPageCount(summary.total, summary.limit);
    if (currentPage < pageCount) {
      loadData(currentPage + 1);
    }
  });

  resetButton.addEventListener("click", () => {
    resetDialog.open = true;
  });

  resetNoButton.addEventListener("click", () => {
    resetDialog.open = false;
  });

  resetYesButton.addEventListener("click", async () => {
    try {
      resetYesButton.disabled = true;
      resetNoButton.disabled = true;
      resetButton.disabled = true;
      await resetPriceChangeHistory();
      resetDialog.open = false;
      currentPage = 1;
      await loadData(1);
    } catch (error) {
      resetDialog.open = false;
      showError(`Failed to reset historical requests: ${error.message}`);
    } finally {
      resetYesButton.disabled = false;
      resetNoButton.disabled = false;
      updateView();
    }
  });

  loadData();
}
