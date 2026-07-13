import { escapeHtml } from "../../utils/html.js";

import {
  SECTION_232_REVIEW_PAGE_SIZE,
  buildSection232HtsSearchControlState,
  buildSection232ReviewDiagnostics,
  buildSection232ReviewInteractivity,
  buildSection232WorkspaceSummary,
  isSection232DraftRowSelected
} from "./section-232-review_state.js";
import { formatReviewTimestamp, trimText } from "./section-232-review_shared.js";

function getCurrentPage(paging = {}) {
  const limit = Number(paging.limit || 0);
  if (!limit) return 1;
  return Math.floor(Number(paging.offset || 0) / limit) + 1;
}

function getTotalPages(paging = {}) {
  const total = Number(paging.total || 0);
  const limit = Number(paging.limit || SECTION_232_REVIEW_PAGE_SIZE);
  return Math.max(1, Math.ceil(total / limit));
}

function getPageLabel(paging = {}, rows = []) {
  const total = Number(paging.total || 0);
  if (!total) {
    return "No rows";
  }
  const offset = Number(paging.offset || 0);
  const start = offset + 1;
  const end = Math.min(offset + rows.length, total);
  return `Rows ${start}-${end} of ${total} · Page ${getCurrentPage(paging)} of ${getTotalPages(paging)}`;
}

function renderMessage({ message = "", messageDesign = "Information" } = {}) {
  const strip = document.getElementById("section-232-review-message");
  if (!strip) return;
  strip.design = messageDesign;
  strip.textContent = message;
  strip.style.display = message ? "block" : "none";
}

function buildInteractivity({
  route = {},
  loading = false,
  mutationInFlight = false,
  publishedBy = "",
  paging = {},
  detailSelection = {},
  selectedCount = 0
} = {}) {
  return buildSection232ReviewInteractivity({
    mode: route.mode,
    loading,
    mutationInFlight,
    publishedBy,
    rowCount: paging.total,
    selection: detailSelection,
    selectedCount
  });
}

function renderDiagnostics({ loading = false, route = {}, rows = [] } = {}) {
  const container = document.getElementById("section-232-review-diagnostics");
  if (!container) return;
  if (loading || route.mode !== "draft") {
    container.innerHTML = "";
    container.style.display = "none";
    return;
  }

  const diagnostics = buildSection232ReviewDiagnostics(rows);
  if (!diagnostics.items.length) {
    container.innerHTML = "";
    container.style.display = "none";
    return;
  }

  container.style.display = "block";
  container.innerHTML = `
    <ui5-message-strip design="Warning" hide-close-button>
      <div class="section-232-review-diagnostics-copy">
        <div class="section-232-review-diagnostics-summary">${escapeHtml(diagnostics.summary)}</div>
        <ul class="section-232-review-diagnostics-list">
          ${diagnostics.items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
        </ul>
      </div>
    </ui5-message-strip>
  `;
}

function renderSummary({ route = {}, workspace = null, paging = {}, publishedBy = "", loading = false, mutationInFlight = false } = {}) {
  const summary = buildSection232WorkspaceSummary(
    {
      ...(workspace || {}),
      total: paging.total
    },
    route
  );
  const title = document.getElementById("section-232-review-title");
  const subtitle = document.getElementById("section-232-review-subtitle");
  const modeTag = document.getElementById("section-232-review-mode-tag");
  const publishControls = document.getElementById("section-232-review-publish-controls");
  const publishedByInput = document.getElementById("section-232-review-published-by-input");
  const refreshButton = document.getElementById("section-232-review-refresh-button");
  const publishButton = document.getElementById("section-232-review-publish-button");
  const acceptedCount = Number(workspace?.batch?.accepted_count || 0);
  const canPublish = acceptedCount > 0 && Boolean(trimText(publishedBy));

  if (title) title.textContent = summary.title;
  if (subtitle) subtitle.textContent = summary.subtitle;
  if (modeTag) {
    modeTag.design = route.mode === "published" ? "Positive" : "Information";
    modeTag.textContent = summary.modeLabel;
  }
  if (publishControls) {
    publishControls.style.display = "";
  }
  if (publishedByInput) {
    publishedByInput.value = publishedBy;
    publishedByInput.disabled = loading || mutationInFlight;
  }
  if (refreshButton) {
    refreshButton.disabled = loading || mutationInFlight;
    refreshButton.loading = loading;
  }
  if (publishButton) {
    publishButton.style.display = route.mode === "draft" ? "" : "none";
    publishButton.disabled = loading || mutationInFlight || !canPublish;
    publishButton.loading = loading || mutationInFlight;
    publishButton.title = canPublish
      ? "Publish the accepted Section 232 draft rows"
      : "Enter Published by and accept at least one row before publishing";
  }
}

function renderPagination({ paging = {}, rows = [], loading = false, mutationInFlight = false } = {}) {
  const copy = document.getElementById("section-232-review-page-copy");
  const previousButton = document.getElementById("section-232-review-page-previous");
  const nextButton = document.getElementById("section-232-review-page-next");
  if (!copy || !previousButton || !nextButton) return;

  copy.textContent = getPageLabel(paging, rows);
  const offset = Number(paging.offset || 0);
  const limit = Number(paging.limit || SECTION_232_REVIEW_PAGE_SIZE);
  const total = Number(paging.total || 0);
  const hasPrevious = offset > 0;
  const hasNext = offset + limit < total;
  previousButton.disabled = loading || mutationInFlight || !hasPrevious;
  nextButton.disabled = loading || mutationInFlight || !hasNext;
}

function renderSearchControls({
  route = {},
  htsSearchInput = "",
  appliedHtsSearchQuery = "",
  loading = false,
  mutationInFlight = false
} = {}) {
  const searchInput = document.getElementById("section-232-review-hts-search");
  const searchButton = document.getElementById("section-232-review-submit-search");
  const clearButton = document.getElementById("section-232-review-clear-search");
  const controlState = buildSection232HtsSearchControlState({
    inputQuery: htsSearchInput,
    appliedQuery: appliedHtsSearchQuery,
    routeValid: route.isValid,
    loading,
    mutationInFlight
  });

  if (searchInput) {
    searchInput.value = controlState.inputQuery;
    searchInput.disabled = controlState.disabled;
  }
  if (searchButton) {
    searchButton.disabled = !controlState.canSubmit;
    searchButton.title = controlState.canSubmit ? "Search matching HTS codes" : "Enter or change an HTS code before searching";
  }
  if (clearButton) {
    clearButton.disabled = !controlState.canClear;
    clearButton.title = "Clear HTS search";
  }
}

function renderTable({
  route = {},
  rows = [],
  paging = {},
  draftSelection = {},
  loading = false,
  mutationInFlight = false,
  publishedBy = "",
  detailSelection = {},
  selectedCount = 0
} = {}) {
  const container = document.getElementById("section-232-review-table");
  const interactivity = buildInteractivity({
    route,
    loading,
    mutationInFlight,
    publishedBy,
    paging,
    detailSelection,
    selectedCount
  });
  const summaryNode = document.getElementById("section-232-review-selection-summary");
  const selectAllButton = document.getElementById("section-232-review-select-all");
  const acceptButton = document.getElementById("section-232-review-accept-selected");
  const rejectButton = document.getElementById("section-232-review-reject-selected");
  const clearButton = document.getElementById("section-232-review-clear-selection");

  if (summaryNode) {
    if (route.mode === "draft") {
      summaryNode.textContent = selectedCount
        ? `${selectedCount} row${selectedCount === 1 ? "" : "s"} selected across all pages.`
        : "Select rows to review or click one row to inspect details.";
    } else {
      summaryNode.textContent = detailSelection.selectionLabel;
    }
  }
  if (selectAllButton) {
    selectAllButton.style.display = interactivity.showBulkActions ? "" : "none";
    selectAllButton.disabled = !interactivity.canSelectAllRows;
  }
  if (acceptButton) {
    acceptButton.style.display = interactivity.showBulkActions ? "" : "none";
    acceptButton.disabled = !interactivity.canBulkReview;
    acceptButton.loading = mutationInFlight;
  }
  if (rejectButton) {
    rejectButton.style.display = interactivity.showBulkActions ? "" : "none";
    rejectButton.disabled = !interactivity.canBulkReview;
    rejectButton.loading = mutationInFlight;
  }
  if (clearButton) {
    clearButton.style.display = interactivity.showBulkActions ? "" : "none";
    clearButton.disabled = !interactivity.canClearSelection;
  }

  if (!container) return;
  if (loading) {
    container.innerHTML = `<div class="section-232-review-empty">Loading workspace…</div>`;
    return;
  }
  if (!rows.length) {
    container.innerHTML = `<div class="section-232-review-empty">No review rows were returned for this page.</div>`;
    return;
  }

  const showSelectionColumn = interactivity.showBulkActions;
  container.innerHTML = `
    <table class="section-232-review-table">
      <thead>
        <tr>
          ${showSelectionColumn ? '<th class="section-232-review-col-select">Select</th>' : ""}
          <th class="section-232-review-col-legal">Legal HTS</th>
          <th class="section-232-review-col-catalog">Catalog</th>
          <th class="section-232-review-col-rule">Rule</th>
          <th class="section-232-review-col-decision">Decision</th>
        </tr>
      </thead>
      <tbody>
        ${rows
          .map((row) => {
            const isDetailRow = detailSelection.selectedRow?.candidateId === row.candidateId;
            const isSelected = isSection232DraftRowSelected(draftSelection, row.candidateId);
            return `
              <tr class="section-232-review-row ${isDetailRow ? "is-selected" : ""}" data-row-id="${escapeHtml(row.candidateId)}">
                ${
                  showSelectionColumn
                    ? `
                <td class="section-232-review-col-select">
                  <ui5-checkbox
                    data-row-select="${escapeHtml(row.candidateId)}"
                    ${isSelected ? "checked" : ""}
                    ${interactivity.enableRowCheckboxes ? "" : "disabled"}
                  ></ui5-checkbox>
                </td>`
                    : ""
                }
                <td class="section-232-review-col-legal">
                  <div class="section-232-review-code">${escapeHtml(row.legalHtsCode)}</div>
                  <div class="section-232-review-cell-subtitle">${escapeHtml(row.description)}</div>
                  ${row.isSuspect ? '<ui5-tag design="Negative">Suspect</ui5-tag>' : ""}
                </td>
                <td class="section-232-review-col-catalog">
                  <ui5-tag design="${escapeHtml(row.catalogStatus.state)}">${escapeHtml(row.catalogStatus.label)}</ui5-tag>
                  <div class="section-232-review-cell-subtitle">${escapeHtml(row.catalogStatus.detail)}</div>
                </td>
                <td class="section-232-review-col-rule">
                  <div class="section-232-review-cell-title">${escapeHtml(row.coverageEffect)}</div>
                  <div class="section-232-review-cell-subtitle">${escapeHtml(row.effectiveWindowSummary)}</div>
                  <div class="section-232-review-cell-subtitle">${escapeHtml(row.ruleType)} · ${escapeHtml(row.metalScope)}</div>
                  <div class="section-232-review-cell-subtitle">Rate ${escapeHtml(row.rateText)}</div>
                  <div class="section-232-review-cell-subtitle">Added ${escapeHtml(formatReviewTimestamp(row.sourceUploadedAt))}</div>
                </td>
                <td class="section-232-review-col-decision">
                  <div class="section-232-review-cell-title">${escapeHtml(row.reviewDecision)}</div>
                  <div class="section-232-review-cell-subtitle">${escapeHtml(row.sourceSummary)}</div>
                  <div class="section-232-review-cell-subtitle">${escapeHtml(row.candidateFlags.join(", ") || "No flags")}</div>
                  <div class="section-232-review-cell-subtitle">Processed ${escapeHtml(formatReviewTimestamp(row.processedAt))}</div>
                </td>
              </tr>
            `;
          })
          .join("")}
      </tbody>
    </table>
  `;
}

function renderSourceDocumentsMarkup(row) {
  if (!row.sourceDocuments.length) {
    return `<div class="section-232-review-empty">No source-document timing is available for this row.</div>`;
  }
  return row.sourceDocuments
    .map(
      (document) => `
        <div class="section-232-review-source-doc-row">
          <div class="section-232-review-source-doc-name">${escapeHtml(document.filename || "Unknown source")}</div>
          <div class="section-232-review-cell-subtitle">Added ${escapeHtml(formatReviewTimestamp(document.uploadedAt))}</div>
        </div>
      `
    )
    .join("");
}

function renderHistoryMarkup(row) {
  if (!row.historyCount) {
    return `<div class="section-232-review-empty">No prior published changes are available for this HTS code.</div>`;
  }
  return row.history
    .map((item) => {
      const publishedLabel = item.version
        ? `published in ${item.version}`
        : "published in an earlier version";
      const timingLabel = item.publishedAt ? ` · ${formatReviewTimestamp(item.publishedAt)}` : "";
      return `
        <div class="section-232-review-history-row">
          <div class="section-232-review-history-title">
            ${escapeHtml(item.actionLabel)} in ${escapeHtml(item.sourceSummary)}
          </div>
          <div class="section-232-review-cell-subtitle">
            ${escapeHtml(publishedLabel)}${escapeHtml(timingLabel)}
          </div>
          <div class="section-232-review-cell-subtitle">
            ${escapeHtml(item.coverageEffect)} · ${escapeHtml(item.ruleType)} · ${escapeHtml(item.legalHtsCode)}
          </div>
        </div>
      `;
    })
    .join("");
}

function renderMatchEvidenceMarkup(row) {
  if (!row.matchEvidenceCount) {
    return `<div class="section-232-review-empty">No occurrence context is available for this review row.</div>`;
  }
  return row.matchEvidence
    .map(
      (evidence) => `
        <div class="section-232-review-evidence-card">
          <div class="section-232-review-evidence-meta">
            <div class="section-232-review-evidence-title">
              ${escapeHtml(evidence.pageLabel)} · ${escapeHtml(evidence.sourceFilename || "Unknown source")}
            </div>
            <ui5-tag design="Information">${escapeHtml(evidence.textSourceLabel)}</ui5-tag>
          </div>
          <div class="section-232-review-detail-copy">Matched text: ${escapeHtml(evidence.matchedText)}</div>
          <div class="section-232-review-evidence-context">${escapeHtml(evidence.contextText)}</div>
        </div>
      `
    )
    .join("");
}

function renderDetail({
  route = {},
  loading = false,
  mutationInFlight = false,
  publishedBy = "",
  paging = {},
  detailSelection = {},
  selectedCount = 0
} = {}) {
  const container = document.getElementById("section-232-review-detail");
  const summaryNode = document.getElementById("section-232-review-detail-summary");
  if (!container || !summaryNode) return;

  const interactivity = buildInteractivity({
    route,
    loading,
    mutationInFlight,
    publishedBy,
    paging,
    detailSelection,
    selectedCount
  });
  summaryNode.textContent = detailSelection.selectionLabel;

  if (loading) {
    container.innerHTML = `
      <div class="section-232-review-empty">
        Loading detail pane…
      </div>
    `;
    return;
  }

  if (!detailSelection.selectedRow) {
    container.innerHTML = `
      <div class="section-232-review-empty">
        The detail pane stays inactive until exactly one table row is selected.
      </div>
    `;
    return;
  }

  const row = detailSelection.selectedRow;
  const status = row.catalogStatus;
  const pagesLabel = row.sourcePages.length ? row.sourcePages.join(", ") : "—";
  const candidateFlagsLabel = row.candidateFlags.length ? row.candidateFlags.join(", ") : "No flags";
  const sourceDocumentsMarkup = renderSourceDocumentsMarkup(row);
  const historyMarkup = renderHistoryMarkup(row);
  const matchEvidenceMarkup = renderMatchEvidenceMarkup(row);

  container.innerHTML = `
    <div class="section-232-review-detail-card">
      <div class="section-232-review-detail-meta">
        <ui5-tag design="${escapeHtml(status.state)}">${escapeHtml(status.label)}</ui5-tag>
        ${row.isSuspect ? '<ui5-tag design="Negative">Suspect candidate</ui5-tag>' : ""}
        <div class="section-232-review-detail-copy">${escapeHtml(status.detail)}</div>
      </div>

      <div class="section-232-review-detail-grid">
        <div class="section-232-review-detail-section">
          <div class="section-232-review-detail-label">Legal HTS</div>
          <div class="section-232-review-detail-value section-232-review-code">${escapeHtml(row.legalHtsCode)}</div>
        </div>
        <div class="section-232-review-detail-section">
          <div class="section-232-review-detail-label">Source pages</div>
          <div class="section-232-review-detail-value">${escapeHtml(pagesLabel)}</div>
        </div>
        <div class="section-232-review-detail-section">
          <div class="section-232-review-detail-label">Coverage</div>
          <div class="section-232-review-detail-value">${escapeHtml(`${row.coverageEffect} · ${row.ruleType}`)}</div>
        </div>
        <div class="section-232-review-detail-section">
          <div class="section-232-review-detail-label">Metal scope</div>
          <div class="section-232-review-detail-value">${escapeHtml(row.metalScope)}</div>
        </div>
        <div class="section-232-review-detail-section">
          <div class="section-232-review-detail-label">Effective window</div>
          <div class="section-232-review-detail-value">${escapeHtml(row.effectiveWindowLabel)}</div>
        </div>
        <div class="section-232-review-detail-section">
          <div class="section-232-review-detail-label">Rate text</div>
          <div class="section-232-review-detail-value">${escapeHtml(row.rateText)}</div>
        </div>
        <div class="section-232-review-detail-section">
          <div class="section-232-review-detail-label">Added</div>
          <div class="section-232-review-detail-value">${escapeHtml(formatReviewTimestamp(row.sourceUploadedAt))}</div>
        </div>
        <div class="section-232-review-detail-section">
          <div class="section-232-review-detail-label">Processed</div>
          <div class="section-232-review-detail-value">${escapeHtml(formatReviewTimestamp(row.processedAt))}</div>
        </div>
        <div class="section-232-review-detail-section">
          <div class="section-232-review-detail-label">Candidate flags</div>
          <div class="section-232-review-detail-value">${escapeHtml(candidateFlagsLabel)}</div>
        </div>
      </div>

      ${
        interactivity.showDecisionEditor
          ? `
      <div class="section-232-review-detail-field">
        <div class="section-232-review-detail-label">Review decision</div>
        <div class="section-232-review-detail-value">${escapeHtml(row.reviewDecision)}</div>
        <div class="section-232-review-detail-actions">
          <ui5-button
            design="Positive"
            data-detail-decision="accepted"
            ${interactivity.canEditDetailDecision && row.reviewDecision !== "accepted" ? "" : "disabled"}
          >
            Accept
          </ui5-button>
          <ui5-button
            design="Negative"
            data-detail-decision="rejected"
            ${interactivity.canEditDetailDecision && row.reviewDecision !== "rejected" ? "" : "disabled"}
          >
            Reject
          </ui5-button>
          <ui5-button
            design="Negative"
            data-detail-delete-hts-code="${escapeHtml(row.legalHtsCode)}"
            ${interactivity.canDeleteDetailCode ? "" : "disabled"}
          >
            Delete HTS code
          </ui5-button>
        </div>
      </div>`
          : `
      <div class="section-232-review-detail-field">
        <div class="section-232-review-detail-label">Review decision</div>
        <div class="section-232-review-detail-value">${escapeHtml(row.reviewDecision)}</div>
        ${
          interactivity.showDeleteAction
            ? `
        <div class="section-232-review-detail-actions">
          <ui5-button
            design="Negative"
            data-detail-delete-hts-code="${escapeHtml(row.legalHtsCode)}"
            ${interactivity.canDeleteDetailCode ? "" : "disabled"}
          >
            Delete HTS code
          </ui5-button>
        </div>`
            : ""
        }
      </div>`
      }

      <div class="section-232-review-detail-field">
        <div class="section-232-review-detail-label">Source files</div>
        <div class="section-232-review-source-doc-list">
          ${sourceDocumentsMarkup}
        </div>
      </div>

      ${
        route.mode === "published"
          ? `
      <div class="section-232-review-detail-field">
        <div class="section-232-review-detail-label">History</div>
        <div class="section-232-review-history-list">
          ${historyMarkup}
        </div>
      </div>`
          : ""
      }

      <div class="section-232-review-detail-field">
        <div class="section-232-review-detail-label">Matched occurrences</div>
        <div class="section-232-review-evidence-list">
          ${matchEvidenceMarkup}
        </div>
      </div>

      <div class="section-232-review-detail-field">
        <ui5-label for="section-232-review-detail-excerpt">Source excerpt</ui5-label>
        <ui5-textarea id="section-232-review-detail-excerpt" readonly rows="6" value="${escapeHtml(row.sourceExcerpt)}"></ui5-textarea>
      </div>
    </div>
  `;
}

export function renderSection232ReviewView(context = {}) {
  renderSummary(context);
  renderSearchControls(context);
  renderPagination(context);
  renderTable(context);
  renderDetail(context);
  renderDiagnostics(context);
  renderMessage(context);
}
