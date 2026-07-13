import { escapeHtml } from "../../utils/html.js";
import { formatTimestamp } from "../../utils/time.js";
import {
  buildCancelDraftState,
  buildPublishedReviewHref,
  buildRulesetSummaryCards,
  buildSection232ResetImpactCards,
  buildSettingsLaunchState,
  isSection232ResetConfirmationValid
} from "./settings_review_state.js";
import { buildGCCTrackerSelectionText, isGCCTrackerWorkbookFile } from "./gcc_tracker_upload_state.js";

const MAX_CODE_PREVIEW = 16;

function formatChapterCoverage(chapters) {
  if (!chapters?.length) return "None";
  const values = [...chapters]
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value))
    .sort((left, right) => left - right);
  const groups = [];
  let start = values[0];
  let end = values[0];
  for (let index = 1; index < values.length; index += 1) {
    const current = values[index];
    if (current === end + 1) {
      end = current;
      continue;
    }
    groups.push(start === end ? `${start}` : `${start}-${end}`);
    start = current;
    end = current;
  }
  groups.push(start === end ? `${start}` : `${start}-${end}`);
  return groups.join(", ");
}

function statusDesign(status) {
  if (status === "completed") return "Positive";
  if (status === "partial") return "Critical";
  if (status === "seeded") return "Information";
  return "Negative";
}

function setButtonState(buttonId, { busy = false, disabled = busy } = {}) {
  const button = document.getElementById(buttonId);
  if (!button) return;
  button.loading = busy;
  button.disabled = disabled;
}

function renderMessage({ message = "", messageDesign = "Information" } = {}) {
  const strip = document.getElementById("settings-message");
  if (!strip) return;
  if (!message) {
    strip.style.display = "none";
    strip.textContent = "";
    return;
  }
  strip.design = messageDesign;
  strip.textContent = message;
  strip.style.display = "block";
}

function renderStatCards(container, items) {
  if (!container) return;
  container.innerHTML = items
    .map(
      (item) => `
        <div class="settings-stat-card">
          <div class="settings-stat-label">${escapeHtml(item.label)}</div>
          <div class="settings-stat-value">${escapeHtml(item.value)}</div>
        </div>
      `
    )
    .join("");
}

function renderSummary({ sources = [], rulesetSummary = {}, classificationStats = {} } = {}) {
  const summaryContainer = document.getElementById("settings-summary-cards");
  const corpusContainer = document.getElementById("settings-corpus-stats");
  const section232ResetContainer = document.getElementById("settings-section-232-reset-stats");
  const classificationContainer = document.getElementById("settings-classification-stats");
  if (!summaryContainer || !corpusContainer || !section232ResetContainer || !classificationContainer) return;

  const latestSource = sources[0] || null;
  const formattedRulesetSummary = {
    ...rulesetSummary,
    last_published_at: rulesetSummary.last_published_at ? formatTimestamp(rulesetSummary.last_published_at) : null
  };
  const summaryCards = buildRulesetSummaryCards(formattedRulesetSummary);
  const publishedReviewHref = buildPublishedReviewHref(rulesetSummary.active_ruleset_version);
  summaryContainer.innerHTML = summaryCards
    .map((item) => {
      if (item.label === "Eligible code count" && publishedReviewHref) {
        return `
          <button type="button" class="settings-summary-card settings-summary-card-button" data-summary-action="open-published-review">
            <div class="settings-summary-label">${escapeHtml(item.label)}</div>
            <div class="settings-summary-value">${escapeHtml(item.value)}</div>
          </button>
        `;
      }
      return `
        <div class="settings-summary-card">
          <div class="settings-summary-label">${escapeHtml(item.label)}</div>
          <div class="settings-summary-value">${escapeHtml(item.value)}</div>
        </div>
      `;
    })
    .join("");

  renderStatCards(corpusContainer, [
    { label: "Stored source PDFs", value: String(sources.length) },
    { label: "Pending drafts", value: String(rulesetSummary.pending_draft_batch_count || 0) },
    { label: "Latest source upload", value: latestSource ? formatTimestamp(latestSource.uploaded_at) : "No uploads yet" }
  ]);

  renderStatCards(
    section232ResetContainer,
    buildSection232ResetImpactCards({
      sourceCount: sources.length,
      rulesetSummary
    })
  );

  renderStatCards(classificationContainer, [
    { label: "Saved classifications", value: String(classificationStats.saved_classification_count || 0) },
    { label: "Latest saved run", value: formatTimestamp(classificationStats.latest_classified_at) }
  ]);
}

function renderCompositionModeSettings({
  appSettings = {},
  appSettingsDraft = false,
  compositionModeSaveBusy = false
} = {}) {
  const toggle = document.getElementById("settings-gcc-composition-toggle");
  const summary = document.getElementById("settings-composition-mode-summary");
  const status = document.getElementById("settings-composition-mode-status");
  if (!toggle || !summary || !status) return;

  toggle.checked = Boolean(appSettingsDraft);
  summary.textContent = appSettingsDraft
    ? "GCC tracker mode is active for GCC-backed items."
    : "Legacy PDF-derived composition mode is active for all items.";

  if (appSettingsDraft !== Boolean(appSettings.use_gcc_tracker_metal_composition)) {
    status.textContent = "Unsaved change.";
  } else {
    status.textContent = appSettings.updated_at
      ? `Last updated ${formatTimestamp(appSettings.updated_at)}`
      : "Current default uses GCC tracker composition with optional PDF evidence.";
  }

  setButtonState("settings-composition-mode-save-button", {
    busy: compositionModeSaveBusy
  });
}

function renderFileSelection(containerId, files, emptyLabel) {
  const container = document.getElementById(containerId);
  if (!container) return;
  if (!files.length) {
    container.textContent = emptyLabel;
    return;
  }
  container.textContent = `${files.length} file${files.length === 1 ? "" : "s"} selected: ${files.map((file) => file.name).join(", ")}`;
}

function renderSources({ sources = [] } = {}) {
  const container = document.getElementById("settings-source-list");
  if (!container) return;
  if (!sources.length) {
    container.innerHTML = `<div class="settings-empty">No Section 232 source documents have been stored yet.</div>`;
    return;
  }

  container.innerHTML = `
    <div class="settings-source-list">
      ${sources
        .map(
          (source) => `
            <div class="settings-source-item">
              <div class="settings-source-head">
                <div>
                  <div class="settings-source-name">${escapeHtml(source.filename)}</div>
                  <div class="settings-source-meta">
                    Uploaded ${escapeHtml(formatTimestamp(source.uploaded_at))} ·
                    ${escapeHtml(String(source.page_count || 0))} pages ·
                    ${escapeHtml(String(source.hts_mention_count || 0))} HTS mentions
                  </div>
                </div>
                <ui5-tag design="${statusDesign(source.extraction_status)}">${escapeHtml(source.extraction_status)}</ui5-tag>
              </div>
              ${
                source.warnings?.length
                  ? `
                    <div class="settings-source-warnings">
                      ${source.warnings
                        .map((warning) => `<ui5-tag design="Critical">${escapeHtml(warning)}</ui5-tag>`)
                        .join("")}
                    </div>
                  `
                  : ""
              }
            </div>
          `
        )
        .join("")}
    </div>
  `;
}

function renderCodePreview({ eligibleCodes = [], rulesetSummary = {} } = {}) {
  const container = document.getElementById("settings-code-preview");
  const actionButton = document.getElementById("settings-code-review-button");
  if (!container) return;
  const publishedReviewHref = buildPublishedReviewHref(rulesetSummary.active_ruleset_version);

  if (actionButton) {
    actionButton.disabled = !publishedReviewHref;
  }

  if (!eligibleCodes.length) {
    container.innerHTML = publishedReviewHref
      ? `<div class="settings-empty">No eligible HTS codes are currently active in the published ruleset. Open published review to inspect the full ruleset.</div>`
      : `<div class="settings-empty">No eligible HTS codes are currently active in the published ruleset.</div>`;
    return;
  }

  const previewCodes = eligibleCodes.slice(0, MAX_CODE_PREVIEW);
  const remainingCount = Math.max(0, eligibleCodes.length - previewCodes.length);
  container.innerHTML = `
    <div class="settings-code-meta">${escapeHtml(String(eligibleCodes.length))} codes are currently eligible under the active published ruleset.</div>
    <div class="settings-code-chip-list">
      ${previewCodes.map((code) => `<div class="settings-code-chip">${escapeHtml(code)}</div>`).join("")}
    </div>
    ${
      remainingCount
        ? `<div class="settings-code-meta settings-code-preview">+${escapeHtml(String(remainingCount))} more codes available in the review workspace.</div>`
        : ""
    }
  `;
}

function renderDraftBatchState({
  currentDraftBatch = null,
  rulesetSummary = {},
  draftCancelBusy = false
} = {}) {
  const statsContainer = document.getElementById("settings-draft-batch-stats");
  const copyContainer = document.getElementById("settings-draft-batch-copy");
  const openButton = document.getElementById("settings-review-open-button");
  const cancelButton = document.getElementById("settings-review-cancel-button");
  if (!statsContainer || !copyContainer || !openButton || !cancelButton) return;

  const launchState = buildSettingsLaunchState({
    currentDraftBatch,
    rulesetSummary
  });
  const cancelState = buildCancelDraftState({
    currentDraftBatch,
    rulesetSummary,
    busy: draftCancelBusy
  });
  setButtonState("settings-review-cancel-button", {
    busy: cancelState.busy,
    disabled: cancelState.disabled
  });

  if (!launchState.hasLaunchTarget) {
    statsContainer.innerHTML = "";
    copyContainer.textContent =
      launchState.pendingDraftCount > 0
        ? "Pending draft batches exist on the server. Refresh the launchpad to reopen the latest pending batch."
        : "No draft batch has been processed in this session.";
    openButton.disabled = true;
    return;
  }

  const isCurrentBatchLoaded = currentDraftBatch?.batch_id === launchState.batch?.batch_id;
  const batch = launchState.batch;
  const statItems = [
    { label: "Batch ID", value: batch.batch_id },
    { label: "Source PDFs", value: String(batch.source_count || batch.source_filenames?.length || 0) },
    { label: "Rule candidates", value: String(batch.rule_candidate_count || 0) },
    { label: "Pending decisions", value: String(batch.pending_count || 0) },
    { label: "Catalog warnings", value: String(batch.warning_count || 0) },
    { label: "Accepted rows", value: String(batch.accepted_count || 0) }
  ];

  renderStatCards(statsContainer, statItems);

  if (isCurrentBatchLoaded) {
    copyContainer.textContent = `Draft batch ${currentDraftBatch.batch_id} created ${formatTimestamp(
      currentDraftBatch.created_at
    )}. Open review to continue in the dedicated Section 232 workspace.`;
  } else {
    copyContainer.textContent = `Latest pending batch ${launchState.batch.batch_id} is ready to reopen in the dedicated Section 232 workspace.`;
  }
  openButton.disabled = false;
}

function renderHTSCatalogSummary({ htsCatalog = {} } = {}) {
  const container = document.getElementById("settings-hts-catalog-stats");
  if (!container) return;
  const summary = htsCatalog.summary || {};
  renderStatCards(container, [
    { label: "Managed files", value: String(summary.managed_file_count || 0) },
    { label: "Chapter coverage", value: formatChapterCoverage(summary.loaded_chapters || []) },
    { label: "Code map", value: summary.has_code_map ? "Managed" : "Not uploaded" },
    { label: "Catalog rows", value: String(summary.catalog_row_count || 0) },
    { label: "Code map rows", value: String(summary.code_map_row_count || 0) },
    {
      label: "Last refresh",
      value: summary.last_refresh_at
        ? `${formatTimestamp(summary.last_refresh_at)} (${summary.last_refresh_status || "unknown"})`
        : summary.last_refresh_status || "unknown"
    }
  ]);

  const errorContainer = document.getElementById("settings-hts-refresh-error");
  if (!errorContainer) return;
  if (summary.last_refresh_error) {
    errorContainer.innerHTML = `<ui5-tag design="Negative">${escapeHtml(summary.last_refresh_error)}</ui5-tag>`;
  } else {
    errorContainer.innerHTML = "";
  }
}

/**
 * Render GCC Tracker workbook upload state and the latest refresh counts.
 *
 * @param {object} context Settings page render context.
 * @returns {void}
 */
function renderGCCTrackerUpload({
  selectedGCCTrackerFile = null,
  lastGCCTrackerRefresh = null,
  gccTrackerRefreshBusy = false
} = {}) {
  const selectionContainer = document.getElementById("settings-gcc-tracker-selection");
  const resultContainer = document.getElementById("settings-gcc-tracker-refresh-result");
  if (selectionContainer) {
    selectionContainer.textContent = buildGCCTrackerSelectionText(selectedGCCTrackerFile);
  }
  if (resultContainer) {
    if (!lastGCCTrackerRefresh) {
      resultContainer.innerHTML = `<div class="settings-empty">No GCC Tracker refresh has been run from Settings yet.</div>`;
    } else {
      renderStatCards(resultContainer, [
        { label: "Uploaded file", value: lastGCCTrackerRefresh.uploaded_filename || "GCC Tracker workbook" },
        { label: "GCC source rows", value: String(lastGCCTrackerRefresh.source_row_count || 0) },
        { label: "Prepared rows", value: String(lastGCCTrackerRefresh.prepared_row_count || 0) },
        { label: "Classifications cleared", value: String(lastGCCTrackerRefresh.cleared_classification_count || 0) },
        { label: "Jobs cancelled", value: String(lastGCCTrackerRefresh.cancelled_job_count || 0) },
        {
          label: "HANA target",
          value: [lastGCCTrackerRefresh.hana_schema, lastGCCTrackerRefresh.hana_table].filter(Boolean).join(".") || "Configured table"
        }
      ]);
    }
  }

  setButtonState("settings-gcc-tracker-select-button", {
    disabled: Boolean(gccTrackerRefreshBusy)
  });
  setButtonState("settings-gcc-tracker-refresh-button", {
    busy: Boolean(gccTrackerRefreshBusy),
    disabled: Boolean(gccTrackerRefreshBusy) || !isGCCTrackerWorkbookFile(selectedGCCTrackerFile)
  });
}

function renderHTSCatalogSources({
  htsCatalog = {},
  htsCatalogDeletingFilename = null
} = {}) {
  const container = document.getElementById("settings-hts-catalog-list");
  if (!container) return;
  if (!htsCatalog.items?.length) {
    container.innerHTML = `<div class="settings-empty">No managed HTS catalog CSV files are available.</div>`;
    return;
  }

  container.innerHTML = `
    <div class="settings-source-list">
      ${htsCatalog.items
        .map((source) => {
          const isChapterSource = source.source_kind === "chapter";
          const isDeleting = htsCatalogDeletingFilename === source.filename;
          const canDelete =
            isChapterSource && Number(htsCatalog.summary?.managed_chapter_file_count || 0) > 1;
          return `
            <div class="settings-source-item">
              <div class="settings-source-head">
                <div>
                  <div class="settings-source-name">${escapeHtml(source.filename)}</div>
                  <div class="settings-source-meta">
                    ${escapeHtml(source.source_kind === "code_map" ? "Code map" : `Chapter ${source.chapter_number}`)} ·
                    ${escapeHtml(String(source.size_bytes || 0))} bytes ·
                    Uploaded ${escapeHtml(formatTimestamp(source.uploaded_at))}
                  </div>
                </div>
                <div class="settings-source-actions-inline">
                  <ui5-tag design="Information">${escapeHtml(source.source_kind === "code_map" ? "code map" : "chapter")}</ui5-tag>
                  ${
                    isChapterSource
                      ? `
                        <ui5-button
                          design="Transparent"
                          icon="delete"
                          data-hts-catalog-action="delete"
                          data-filename="${escapeHtml(source.filename)}"
                          ${canDelete && !isDeleting ? "" : "disabled"}
                        >
                          ${escapeHtml(isDeleting ? "Deleting..." : "Delete")}
                        </ui5-button>
                      `
                      : ""
                  }
                </div>
              </div>
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderDialogs({
  isClassificationResetDialogOpen = false,
  isSection232ResetDialogOpen = false,
  section232ResetConfirmation = "",
  classificationResetBusy = false,
  section232ResetBusy = false
} = {}) {
  const classificationDialog = document.getElementById("settings-reset-dialog");
  const section232Dialog = document.getElementById("settings-section-232-reset-dialog");
  const confirmationInput = document.getElementById("settings-section-232-reset-confirmation-input");
  const section232ConfirmDisabled = !isSection232ResetConfirmationValid(section232ResetConfirmation) || section232ResetBusy;

  if (classificationDialog) {
    classificationDialog.open = isClassificationResetDialogOpen;
  }
  if (section232Dialog) {
    section232Dialog.open = isSection232ResetDialogOpen;
  }
  if (confirmationInput) {
    confirmationInput.value = section232ResetConfirmation;
  }

  setButtonState("settings-reset-confirm-button", {
    busy: classificationResetBusy
  });
  setButtonState("settings-section-232-reset-confirm-button", {
    busy: section232ResetBusy,
    disabled: section232ConfirmDisabled
  });
}

export function renderSettingsView(context = {}) {
  renderMessage(context);
  renderCompositionModeSettings(context);
  renderFileSelection("settings-source-selection", context.selectedSourceFiles || [], "No PDF files selected.");
  renderFileSelection("settings-hts-catalog-selection", context.selectedCatalogFiles || [], "No HTS catalog CSV files selected.");
  renderSummary(context);
  renderSources(context);
  renderCodePreview(context);
  renderDraftBatchState(context);
  renderGCCTrackerUpload(context);
  renderHTSCatalogSummary(context);
  renderHTSCatalogSources(context);
  renderDialogs(context);
  setButtonState("settings-source-process-button", {
    busy: Boolean(context.sourceProcessBusy)
  });
  setButtonState("settings-hts-catalog-upload-button", {
    busy: Boolean(context.htsCatalogUploadBusy)
  });
}
