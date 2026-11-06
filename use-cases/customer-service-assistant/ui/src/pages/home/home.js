import "@ui5/webcomponents/dist/BusyIndicator.js";
import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/TextArea.js";
import "@ui5/webcomponents/dist/Toast.js";

import { API_BASE_URL, API_KEY, request } from "../../services/api.js";
import {
  draftDecisionPayload,
  formatFileSize,
  getDraftDecisionActions,
  manualEmailPayload,
  mapDraftFromApi,
  validateManualEmail,
  summarizeFetchResult,
  describeRun
} from "./reviewHelpers.js";
import {
  attachPaginationHandlers,
  normalizePagination,
  paginatedEndpoint,
  renderPaginationControls
} from "../shared/pagination.js";

const RUN_ENDPOINTS = {
  gmail: "/api/document-requests/emails/fetch-new",
  scenarios: "/api/document-requests/emails/seed-scenarios"
};
const RUN_POLL_INTERVAL_MS = 2000;

/**
 * Start a background run. A 409 means a run for this source is already active,
 * which is fine: the caller simply follows that run's progress.
 * @param {"gmail"|"scenarios"} source Run source.
 */
async function startRun(source) {
  try {
    await request(RUN_ENDPOINTS[source], "POST");
  } catch (error) {
    if (error.status !== 409) throw error;
  }
}

/**
 * Poll the run status until it is no longer running.
 * @param {"gmail"|"scenarios"} source Run source.
 * @param {(run: object) => Promise<void>|void} onUpdate Called with every status payload.
 * @returns {Promise<object>} Final FetchRunStatus payload.
 */
async function followRun(source, onUpdate) {
  while (true) {
    const run = await request(`/api/document-requests/emails/fetch-status?source=${source}`);
    await onUpdate(run);
    if (run.status !== "running") return run;
    await new Promise((resolve) => setTimeout(resolve, RUN_POLL_INTERVAL_MS));
  }
}

async function processManualEmail(payload) {
  return request("/api/document-requests/emails/manual", "POST", payload);
}

async function getDrafts(page = 1) {
  return request(paginatedEndpoint("/api/document-requests/drafts", page));
}

async function approveDraft(draftId, payload) {
  return request(`/api/document-requests/drafts/${draftId}/approve`, "POST", payload);
}

async function rejectDraft(draftId, payload) {
  return request(`/api/document-requests/drafts/${draftId}/reject`, "POST", payload);
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
  const labels = {
    ready_for_review: "Ready",
    needs_human_review: "Needs review",
    failed: "Failed",
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
  if (!Number.isFinite(numeric)) return "n/a";
  return `${Math.round(numeric * 100)}%`;
}

function formatDateTime(value) {
  if (!value) return "Not set";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return escapeHtml(value);
  return escapeHtml(new Intl.DateTimeFormat(undefined, {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short"
  }).format(date));
}

function summaryFromItems(items) {
  return {
    total: items.length,
    ready: items.filter((item) => item.status === "ready_for_review").length,
    needsReview: items.filter((item) => item.status === "needs_human_review").length,
    failed: items.filter((item) => item.status === "failed").length
  };
}

function summaryFromMetadata(meta, items) {
  const fallback = summaryFromItems(items);
  const summary = meta?.summary || {};
  return {
    total: Number.isFinite(Number(summary.total)) ? Number(summary.total) : fallback.total,
    ready: Number.isFinite(Number(summary.ready_for_review)) ? Number(summary.ready_for_review) : fallback.ready,
    needsReview: Number.isFinite(Number(summary.needs_human_review))
      ? Number(summary.needs_human_review)
      : fallback.needsReview,
    failed: Number.isFinite(Number(summary.failed)) ? Number(summary.failed) : fallback.failed
  };
}

function renderSummary(summaryContainer, summary) {
  const cards = [
    { label: "Drafts", value: summary.total },
    { label: "Ready", value: summary.ready },
    { label: "Needs review", value: summary.needsReview },
    { label: "Failed", value: summary.failed }
  ];
  summaryContainer.innerHTML = cards
    .map((card) => `
      <div class="summary-card">
        <span class="summary-label">${escapeHtml(card.label)}</span>
        <div class="summary-value">${escapeHtml(card.value)}</div>
      </div>
    `)
    .join("");
}

function renderAttachments(attachments) {
  if (!attachments.length) {
    return '<span class="review-cell-secondary">No attachments selected</span>';
  }
  return attachments
    .map((attachment) => `
      <div class="attachment-line">
        <a href="#" class="attachment-link" data-document-id="${escapeHtml(attachment.documentId)}">${escapeHtml(attachment.fileName)}</a>
        <span>${escapeHtml(attachment.documentType)} · ${escapeHtml(formatFileSize(attachment.sizeBytes))}</span>
      </div>
    `)
    .join("");
}

function renderValidationErrors(errors) {
  if (!errors.length) return "";
  return `<div class="validation-errors">${errors.map((error) => `<div>${escapeHtml(error)}</div>`).join("")}</div>`;
}

function renderDecisionButtons(item) {
  return getDraftDecisionActions(item.status)
    .map((action) => `
      <ui5-button
        design="${escapeHtml(action.design)}"
        class="${escapeHtml(action.className)}"
        data-row-id="${escapeHtml(item.rowId)}"
        ${action.disabled ? "disabled" : ""}
      >
        ${escapeHtml(action.label)}
      </ui5-button>
    `)
    .join("");
}

/**
 * Read the draft body currently visible in the expanded editor.
 *
 * @param {HTMLElement} tableBody - Review table body containing expanded draft editors.
 * @param {string} rowId - Draft row id being approved or rejected.
 * @param {Array<object>} items - Loaded draft rows used as a fallback when the row is collapsed.
 * @returns {string} Current reviewer-edited draft body.
 */
function getCurrentDraftBody(tableBody, rowId, items) {
  const editor = Array.from(tableBody.querySelectorAll(".draft-body-editor"))
    .find((field) => field.dataset.rowId === rowId);
  if (editor) {
    return editor.value;
  }
  const draft = items.find((item) => item.rowId === rowId);
  return draft?.body || "";
}

function renderRows(tableBody, items, expandedRowId) {
  tableBody.innerHTML = items
    .map((item) => {
      const isExpanded = item.rowId === expandedRowId;
      return `
        <tr class="review-row is-${escapeHtml(item.status || "unknown")}${isExpanded ? " is-expanded" : ""}" data-row-id="${escapeHtml(item.rowId)}">
          <td>
            <div class="review-cell-main">
              <span>${escapeHtml(item.originalEmail.from || item.toEmail || "Unknown")}</span>
              <span class="review-cell-secondary">${escapeHtml(formatDateTime(item.originalEmail.date))}</span>
            </div>
          </td>
          <td>${escapeHtml(item.country || "Not set")}</td>
          <td>
            <div class="review-cell-main">
              <span>${escapeHtml(item.subject)}</span>
              <span class="review-cell-secondary">Gmail ${escapeHtml(item.gmailMessageId || "")}</span>
            </div>
          </td>
          <td>${escapeHtml(item.attachments.length)} selected</td>
          <td>
            <div class="review-cell-main">
              ${statusTag(item.status)}
              <span class="review-cell-secondary">Confidence ${escapeHtml(formatConfidence(item.confidence))}</span>
            </div>
          </td>
          <td><div class="decision-actions">${renderDecisionButtons(item)}</div></td>
        </tr>
        ${
          isExpanded
            ? `
              <tr class="review-expanded-row">
                <td colspan="6">
                  <div class="expanded-layout">
                    <div class="expanded-panel">
                      <h3>Original email</h3>
                      <div class="email-metadata">
                        <ui5-label>From</ui5-label><span>${escapeHtml(item.originalEmail.from || "")}</span>
                        <ui5-label>Subject</ui5-label><span>${escapeHtml(item.originalEmail.subject || "")}</span>
                        <ui5-label>Date</ui5-label><span>${escapeHtml(formatDateTime(item.originalEmail.date))}</span>
                      </div>
                      <div class="email-body">${escapeHtml(item.originalEmail.body || "")}</div>
                    </div>
                    <div class="expanded-panel">
                      <h3>Draft response</h3>
                      ${renderValidationErrors(item.validationErrors)}
                      <div class="draft-fields">
                        <ui5-label>To</ui5-label><span>${escapeHtml(item.toEmail || "")}</span>
                        <ui5-label>Subject</ui5-label><span>${escapeHtml(item.subject)}</span>
                        <ui5-label>Explanation</ui5-label><span>${escapeHtml(item.explanation)}</span>
                      </div>
                      <ui5-textarea
                        class="draft-body-editor"
                        data-row-id="${escapeHtml(item.rowId)}"
                        rows="18"
                        value="${escapeHtml(item.body)}"
                      ></ui5-textarea>
                      <h3>Selected attachments</h3>
                      <div class="attachment-list">${renderAttachments(item.attachments)}</div>
                      <div class="expanded-actions">${renderDecisionButtons(item)}</div>
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

export default function initHomePage() {
  const summaryContainer = document.getElementById("summary-grid");
  const toolbarText = document.getElementById("review-toolbar-text");
  const fetchSummaryText = document.getElementById("fetch-summary-text");
  const fetchButton = document.getElementById("fetch-emails-button");
  const seedButton = document.getElementById("seed-scenarios-button");
  const manualFrom = document.getElementById("manual-email-from");
  const manualSubject = document.getElementById("manual-email-subject");
  const manualBody = document.getElementById("manual-email-body");
  const manualButton = document.getElementById("process-manual-email-button");
  const manualStatus = document.getElementById("manual-email-status");
  const loadingState = document.getElementById("review-loading");
  const errorState = document.getElementById("review-error");
  const tableContainer = document.getElementById("review-table-container");
  const tableBody = document.getElementById("review-table-body");
  const paginationContainer = document.getElementById("review-pagination");
  const toast = document.getElementById("review-toast");

  let items = [];
  let expandedRowId = null;
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
    renderRows(tableBody, items, expandedRowId);
    toolbarText.textContent = `${summary.ready} ready, ${summary.needsReview} needing review, ${summary.failed} failed, ${summary.total} total`;
    renderPaginationControls(paginationContainer, paginationMeta);
    attachPaginationHandlers(paginationContainer, paginationMeta, async (page) => {
      expandedRowId = null;
      await loadData(page);
    });

    tableBody.querySelectorAll(".review-row").forEach((row) => {
      row.addEventListener("click", () => {
        const { rowId } = row.dataset;
        expandedRowId = expandedRowId === rowId ? null : rowId;
        updateView();
      });
    });

    tableBody.querySelectorAll(".approve-draft-button").forEach((button) => {
      button.addEventListener("click", async (event) => {
        event.stopPropagation();
        const { rowId } = button.dataset;
        const body = getCurrentDraftBody(tableBody, rowId, items);
        button.disabled = true;
        try {
          await approveDraft(rowId, draftDecisionPayload(body));
          expandedRowId = null;
          showToast(`Draft ${rowId} sent.`);
          await loadData(currentPage);
        } catch (error) {
          showToast(`Approve failed: ${error.message}`);
        } finally {
          button.disabled = false;
        }
      });
    });

    tableBody.querySelectorAll(".reject-draft-button").forEach((button) => {
      button.addEventListener("click", async (event) => {
        event.stopPropagation();
        const { rowId } = button.dataset;
        const body = getCurrentDraftBody(tableBody, rowId, items);
        button.disabled = true;
        try {
          await rejectDraft(rowId, draftDecisionPayload(body));
          expandedRowId = null;
          showToast(`Draft ${rowId} rejected.`);
          await loadData(currentPage);
        } catch (error) {
          showToast(`Reject failed: ${error.message}`);
        } finally {
          button.disabled = false;
        }
      });
    });

    tableBody.querySelectorAll(".attachment-link").forEach((link) => {
      link.addEventListener("click", async (event) => {
        event.preventDefault();
        event.stopPropagation();
        const documentId = link.dataset.documentId;
        const response = await fetch(
          `${API_BASE_URL}/api/document-requests/documents/${documentId}`,
          { headers: { "X-API-Key": API_KEY } }
        );
        if (!response.ok) {
          showToast("Failed to load document preview");
          return;
        }
        const blob = await response.blob();
        window.open(URL.createObjectURL(blob), "_blank");
      });
    });
  };

  const loadData = async (page = currentPage) => {
    try {
      errorState.hidden = true;
      const payload = await getDrafts(page);
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
      errorState.textContent = `Failed to load drafts: ${error.message}`;
    }
  };

  /**
   * Run a background source end to end: start, show live progress, refresh the
   * draft table whenever new drafts appear, and restore the button afterwards.
   */
  const runInBackground = async (source, button, busyLabel, idleLabel) => {
    button.disabled = true;
    button.textContent = busyLabel;
    let draftsSeen = -1;
    try {
      await startRun(source);
      const finalRun = await followRun(source, async (run) => {
        fetchSummaryText.textContent = describeRun(run);
        const drafts = run.summary?.drafts_created ?? 0;
        // Re-rendering the table recreates the editable textarea, so never
        // refresh while a reviewer has a draft open (unsaved edits would be lost).
        if (drafts !== draftsSeen && expandedRowId === null) {
          draftsSeen = drafts;
          await loadData(1);
        }
      });
      if (expandedRowId === null) {
        await loadData(1);
      } else {
        fetchSummaryText.textContent = `${describeRun(finalRun)}. Close the open draft to see new drafts.`;
      }
    } catch (error) {
      fetchSummaryText.textContent = `Run failed: ${error.message}`;
    } finally {
      button.disabled = false;
      button.textContent = idleLabel;
    }
  };

  fetchButton.addEventListener("click", () => runInBackground("gmail", fetchButton, "Fetching", "Fetch new emails"));
  seedButton.addEventListener("click", () => runInBackground("scenarios", seedButton, "Loading", "Load scenario emails"));

  manualButton.addEventListener("click", async () => {
    const values = {
      from: manualFrom.value,
      subject: manualSubject.value,
      body: manualBody.value
    };
    const errors = validateManualEmail(values);
    if (errors.length) {
      manualStatus.textContent = errors.join(", ");
      return;
    }
    manualButton.disabled = true;
    manualButton.textContent = "Processing";
    try {
      const summary = await processManualEmail(manualEmailPayload(values));
      manualStatus.textContent = summarizeFetchResult(summary);
      expandedRowId = null;
      await loadData(1);
    } catch (error) {
      manualStatus.textContent = `Manual email failed: ${error.message}`;
    } finally {
      manualButton.disabled = false;
      manualButton.textContent = "Process email";
    }
  });

  loadData();
}
