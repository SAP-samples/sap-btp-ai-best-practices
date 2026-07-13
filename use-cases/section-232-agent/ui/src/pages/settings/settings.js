import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/Card.js";
import "@ui5/webcomponents/dist/CardHeader.js";
import "@ui5/webcomponents/dist/Dialog.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/MessageStrip.js";
import "@ui5/webcomponents/dist/Switch.js";
import "@ui5/webcomponents/dist/Tag.js";

import { pageRouter } from "../../modules/router.js";
import { request } from "../../services/api.js";
import {
  SECTION232_RESET_CONFIRMATION_TEXT,
  buildCancelDraftState,
  buildPublishedReviewHref,
  buildSettingsLaunchState,
  decideDraftRestoreSource,
  isSection232ResetConfirmationValid,
  isPendingReviewDraftBatch,
  selectLatestPendingDraftBatch
} from "./settings_review_state.js";
import {
  buildDraftBatchSnapshot,
  clearPersistedDraftBatch,
  persistDraftBatch,
  readPersistedDraftBatch
} from "./settings_persistence.js";
import {
  buildMaterialMasterRefreshSuccessMessage,
  isMaterialMasterWorkbookFile
} from "./material_master_upload_state.js";
import { renderSettingsView } from "./settings_view.js";

const DEFAULT_RULESET_SUMMARY = {
  active_ruleset_version: null,
  eligible_hts_code_count: 0,
  pending_draft_batch_count: 0,
  last_published_at: null
};

const DEFAULT_CLASSIFICATION_STATS = {
  saved_classification_count: 0,
  latest_classified_at: null
};

const DEFAULT_APP_SETTINGS = {
  use_material_master_metal_composition: false,
  updated_at: null
};

const DEFAULT_HTS_CATALOG_SUMMARY = {
  managed_file_count: 0,
  managed_chapter_file_count: 0,
  loaded_chapters: [],
  has_code_map: false,
  last_refresh_status: "unknown",
  last_refresh_at: null,
  last_refresh_error: null,
  catalog_row_count: 0,
  code_map_row_count: 0
};

function createInitialState() {
  return {
    sources: [],
    rulesetSummary: {
      ...DEFAULT_RULESET_SUMMARY
    },
    eligibleCodes: [],
    classificationStats: {
      ...DEFAULT_CLASSIFICATION_STATS
    },
    appSettings: {
      ...DEFAULT_APP_SETTINGS
    },
    appSettingsDraft: false,
    selectedSourceFiles: [],
    selectedCatalogFiles: [],
    selectedMaterialMasterFile: null,
    lastMaterialMasterRefresh: null,
    currentDraftBatch: null,
    htsCatalog: {
      items: [],
      summary: {
        ...DEFAULT_HTS_CATALOG_SUMMARY
      }
    },
    htsCatalogDeletingFilename: null,
    message: "",
    messageDesign: "Information",
    sourceProcessBusy: false,
    htsCatalogUploadBusy: false,
    materialMasterRefreshBusy: false,
    compositionModeSaveBusy: false,
    classificationResetBusy: false,
    section232ResetBusy: false,
    draftCancelBusy: false,
    isClassificationResetDialogOpen: false,
    isSection232ResetDialogOpen: false,
    section232ResetConfirmation: ""
  };
}

const state = createInitialState();

function resetState() {
  Object.assign(state, createInitialState());
}

function applyMessage(text = "", design = "Information") {
  state.message = text;
  state.messageDesign = design;
}

function showMessage(text = "", design = "Information") {
  applyMessage(text, design);
  render();
}

function buildViewState() {
  return {
    ...state
  };
}

function render() {
  renderSettingsView(buildViewState());
}

function navigateTo(path) {
  const normalizedPath = String(path ?? "").trim();
  if (!normalizedPath) {
    return;
  }
  pageRouter.navigate(normalizedPath);
}

function clearInputValue(inputId) {
  const input = document.getElementById(inputId);
  if (input) {
    input.value = "";
  }
}

function setCurrentDraftBatch(batch) {
  state.currentDraftBatch = buildDraftBatchSnapshot(batch);
  persistDraftBatch(state.currentDraftBatch);
}

function clearCurrentDraftBatch() {
  state.currentDraftBatch = null;
  clearPersistedDraftBatch();
}

function findPendingDraftBatch(batchId, rulesetSummary = state.rulesetSummary) {
  const normalizedBatchId = String(batchId ?? "").trim();
  if (!normalizedBatchId) {
    return null;
  }
  const pendingDraftBatches = Array.isArray(rulesetSummary?.pending_draft_batches)
    ? rulesetSummary.pending_draft_batches
    : [];
  return pendingDraftBatches.find((batch) => String(batch?.batch_id ?? "").trim() === normalizedBatchId) || null;
}

function syncCurrentDraftBatchFromRulesetSummary() {
  if (!state.currentDraftBatch?.batch_id) {
    return;
  }
  const pendingBatch = findPendingDraftBatch(state.currentDraftBatch.batch_id);
  if (pendingBatch) {
    setCurrentDraftBatch(pendingBatch);
    return;
  }
  clearCurrentDraftBatch();
}

function restorePersistedDraftBatchSnapshot(persistedBatch = null) {
  const batch = persistedBatch || readPersistedDraftBatch();
  if (!isPendingReviewDraftBatch(batch)) {
    if (batch?.batch_id) {
      clearPersistedDraftBatch();
    }
    return false;
  }
  const serverBatch = findPendingDraftBatch(batch.batch_id);
  if (!serverBatch) {
    clearCurrentDraftBatch();
    return false;
  }
  setCurrentDraftBatch(serverBatch);
  return true;
}

function restoreLatestServerPendingDraftBatch() {
  const pendingBatch = selectLatestPendingDraftBatch(state.rulesetSummary);
  if (!pendingBatch?.batch_id) {
    return false;
  }
  setCurrentDraftBatch(pendingBatch);
  return true;
}

function buildDraftRestoreCandidate() {
  if (isPendingReviewDraftBatch(state.currentDraftBatch)) {
    return state.currentDraftBatch;
  }
  return readPersistedDraftBatch();
}

function restoreDraftBatchLaunchState({
  preferredPersistedDraftBatch = null,
  showUnavailableMessage = false
} = {}) {
  const persistedDraftBatch = preferredPersistedDraftBatch || readPersistedDraftBatch();
  const restorablePersistedDraftBatch = isPendingReviewDraftBatch(persistedDraftBatch) ? persistedDraftBatch : null;
  const serverPendingDraftCount = Number(state.rulesetSummary.pending_draft_batch_count || 0);
  if (persistedDraftBatch?.batch_id && !restorablePersistedDraftBatch) {
    clearPersistedDraftBatch();
  }

  const restoreSource = decideDraftRestoreSource({
    serverPendingDraftCount,
    persistedDraftBatch: restorablePersistedDraftBatch
  });

  let restored = false;
  if (restoreSource === "persisted") {
    restored = restorePersistedDraftBatchSnapshot(restorablePersistedDraftBatch);
    if (!restored && serverPendingDraftCount > 0) {
      restored = restoreLatestServerPendingDraftBatch();
    }
  } else if (restoreSource === "server") {
    restored = restoreLatestServerPendingDraftBatch();
  }

  if (!restored) {
    clearCurrentDraftBatch();
    if (showUnavailableMessage && serverPendingDraftCount > 0) {
      applyMessage(
        "Pending draft batches exist on the server, but Settings could not reopen the latest pending batch. Stale local draft state was ignored.",
        "Information"
      );
    }
  }

  render();
  return restored;
}

function reconcileDraftBatchLaunchState() {
  return restoreDraftBatchLaunchState({
    preferredPersistedDraftBatch: buildDraftRestoreCandidate()
  });
}

async function refreshPageState() {
  const [appSettingsResponse, sourceResponse, rulesetSummaryResponse, codeResponse, statsResponse, htsCatalogResponse] = await Promise.all([
    request("/api/metal-composition/app-settings"),
    request("/api/metal-composition/section-232/sources"),
    request("/api/metal-composition/section-232/ruleset-summary"),
    request("/api/metal-composition/section-232/eligible-hts-codes"),
    request("/api/metal-composition/classifications/stats"),
    request("/api/metal-composition/hts-catalog/sources")
  ]);

  state.appSettings = appSettingsResponse || {
    ...DEFAULT_APP_SETTINGS
  };
  state.appSettingsDraft = Boolean(state.appSettings.use_material_master_metal_composition);
  state.sources = sourceResponse.items || [];
  state.rulesetSummary = rulesetSummaryResponse || {
    ...DEFAULT_RULESET_SUMMARY
  };
  state.eligibleCodes = codeResponse.codes || [];
  state.classificationStats = statsResponse || {
    ...DEFAULT_CLASSIFICATION_STATS
  };
  state.htsCatalog = htsCatalogResponse || {
    items: [],
    summary: {
      ...DEFAULT_HTS_CATALOG_SUMMARY
    }
  };
  syncCurrentDraftBatchFromRulesetSummary();
  render();
}

async function processSection232DraftBatch() {
  if (!state.selectedSourceFiles.length) {
    showMessage("Choose at least one PDF file before processing a draft batch.", "Critical");
    return;
  }

  const formData = new FormData();
  state.selectedSourceFiles.forEach((file) => formData.append("files", file));

  state.sourceProcessBusy = true;
  render();
  try {
    const response = await request("/api/metal-composition/section-232/draft-batches/process", "POST", formData);
    setCurrentDraftBatch(response.batch || null);
    state.rulesetSummary = response.ruleset_summary || state.rulesetSummary;
    state.selectedSourceFiles = [];
    clearInputValue("settings-source-upload-input");
    await refreshPageState();
    applyMessage(
      `Processed ${response.batch?.source_count || 0} PDF file${response.batch?.source_count === 1 ? "" : "s"} into draft batch ${response.batch?.batch_id || "unknown"}. Use Open review to continue in the dedicated workspace.`,
      "Positive"
    );
    render();
  } catch (error) {
    await refreshPageState();
    reconcileDraftBatchLaunchState();
    applyMessage(error.message, "Negative");
    render();
  } finally {
    state.sourceProcessBusy = false;
    render();
  }
}

async function uploadHTSCatalogSources() {
  if (!state.selectedCatalogFiles.length) {
    showMessage("Choose at least one HTS catalog CSV file before uploading.", "Critical");
    return;
  }

  const formData = new FormData();
  state.selectedCatalogFiles.forEach((file) => formData.append("files", file));

  state.htsCatalogUploadBusy = true;
  render();
  try {
    const response = await request("/api/metal-composition/hts-catalog/sources", "POST", formData);
    state.selectedCatalogFiles = [];
    clearInputValue("settings-hts-catalog-upload-input");
    await refreshPageState();
    applyMessage(
      `Uploaded ${response.uploaded_file_count} HTS catalog file${response.uploaded_file_count === 1 ? "" : "s"} and refreshed the managed HTS catalog.`,
      "Positive"
    );
    render();
  } catch (error) {
    applyMessage(error.message, "Negative");
    render();
  } finally {
    state.htsCatalogUploadBusy = false;
    render();
  }
}

/**
 * Upload the selected Material Master workbook and refresh the HANA serving table.
 *
 * @returns {Promise<void>} Resolves after the refresh attempt updates Settings state.
 */
async function uploadMaterialMasterFile() {
  if (!state.selectedMaterialMasterFile) {
    showMessage("Choose a Material Master .xlsb or .xlsx file before refreshing HANA.", "Critical");
    return;
  }
  if (!isMaterialMasterWorkbookFile(state.selectedMaterialMasterFile)) {
    showMessage("The Material Master upload must be a .xlsb or .xlsx workbook.", "Critical");
    return;
  }

  const formData = new FormData();
  formData.append("file", state.selectedMaterialMasterFile);

  state.materialMasterRefreshBusy = true;
  render();
  try {
    const response = await request("/api/metal-composition/admin/material-master/refresh-hana", "POST", formData);
    state.selectedMaterialMasterFile = null;
    state.lastMaterialMasterRefresh = response;
    clearInputValue("settings-material-master-upload-input");
    await refreshPageState();
    applyMessage(buildMaterialMasterRefreshSuccessMessage(response), "Positive");
    render();
  } catch (error) {
    applyMessage(error.message, "Negative");
    render();
  } finally {
    state.materialMasterRefreshBusy = false;
    render();
  }
}

async function saveCompositionMode() {
  const nextValue = Boolean(state.appSettingsDraft);
  if (nextValue === Boolean(state.appSettings.use_material_master_metal_composition)) {
    showMessage("No composition-mode change to save.", "Information");
    return;
  }

  state.compositionModeSaveBusy = true;
  render();
  try {
    const response = await request("/api/metal-composition/app-settings", "PUT", {
      use_material_master_metal_composition: nextValue
    });
    state.appSettings = response || state.appSettings;
    state.appSettingsDraft = Boolean(state.appSettings.use_material_master_metal_composition);
    applyMessage(
      nextValue
        ? "Saved Material Master metal composition mode."
        : "Saved legacy PDF-derived metal composition mode.",
      "Positive"
    );
    render();
  } catch (error) {
    state.appSettingsDraft = Boolean(state.appSettings.use_material_master_metal_composition);
    applyMessage(error.message, "Negative");
    render();
  } finally {
    state.compositionModeSaveBusy = false;
    render();
  }
}

async function deleteHTSCatalogSource(filename) {
  const normalizedFilename = String(filename || "").trim();
  const source = state.htsCatalog.items.find((item) => item.filename === normalizedFilename);
  if (!source || source.source_kind !== "chapter") {
    return;
  }
  if (Number(state.htsCatalog.summary?.managed_chapter_file_count || 0) <= 1) {
    showMessage("At least one HTS chapter CSV must remain in the managed catalog.", "Critical");
    return;
  }

  const targetLabel = source.chapter_number != null ? `Chapter ${source.chapter_number}` : normalizedFilename;
  if (!window.confirm(`Delete ${targetLabel} (${normalizedFilename}) from the managed HTS catalog?`)) {
    return;
  }

  state.htsCatalogDeletingFilename = normalizedFilename;
  render();
  try {
    const response = await request(
      `/api/metal-composition/hts-catalog/sources/${encodeURIComponent(normalizedFilename)}`,
      "DELETE"
    );
    await refreshPageState();
    applyMessage(`Deleted ${response.deleted_filename} and refreshed the managed HTS catalog.`, "Positive");
    render();
  } catch (error) {
    applyMessage(error.message, "Negative");
    render();
  } finally {
    state.htsCatalogDeletingFilename = null;
    render();
  }
}

function openResetDialog() {
  state.isClassificationResetDialogOpen = true;
  render();
}

function closeResetDialog() {
  state.isClassificationResetDialogOpen = false;
  render();
}

function syncSection232ResetConfirmation(value = "") {
  state.section232ResetConfirmation = value;
  render();
}

function openSection232ResetDialog() {
  state.section232ResetConfirmation = "";
  state.isSection232ResetDialogOpen = true;
  render();
}

function closeSection232ResetDialog() {
  state.section232ResetConfirmation = "";
  state.isSection232ResetDialogOpen = false;
  render();
}

function buildSection232ResetMessage(response) {
  const parts = [];
  if (response.cleared_source_count > 0) {
    parts.push(
      `removed ${response.cleared_source_count} stored source PDF${response.cleared_source_count === 1 ? "" : "s"}`
    );
  }
  if (response.cleared_draft_batch_count > 0) {
    parts.push(
      `deleted ${response.cleared_draft_batch_count} draft batch${response.cleared_draft_batch_count === 1 ? "" : "es"}`
    );
  }
  if (response.cleared_published_ruleset_count > 0) {
    parts.push(
      `deleted ${response.cleared_published_ruleset_count} published ruleset${response.cleared_published_ruleset_count === 1 ? "" : "s"}`
    );
  }
  if (!parts.length) {
    return "No Section 232 data was stored.";
  }
  return `Reset Section 232 data: ${parts.join(", ")}.`;
}

async function resetClassifications() {
  state.classificationResetBusy = true;
  render();
  try {
    const response = await request("/api/metal-composition/classifications/reset", "POST");
    state.isClassificationResetDialogOpen = false;
    await refreshPageState();
    const parts = [];
    if (response.cleared_classification_count > 0) {
      parts.push(
        `Cleared ${response.cleared_classification_count} saved classification snapshot${response.cleared_classification_count === 1 ? "" : "s"}.`
      );
    }
    if (response.cancelled_job_count > 0) {
      parts.push(`Cancelled ${response.cancelled_job_count} active classification job${response.cancelled_job_count === 1 ? "" : "s"}.`);
    }
    applyMessage(parts.length > 0 ? parts.join(" ") : "No classifications to reset.", "Positive");
    render();
  } catch (error) {
    applyMessage(error.message, "Negative");
    render();
  } finally {
    state.classificationResetBusy = false;
    render();
  }
}

async function resetSection232Data() {
  if (!isSection232ResetConfirmationValid(state.section232ResetConfirmation)) {
    showMessage(`Type ${SECTION232_RESET_CONFIRMATION_TEXT} exactly to confirm the reset.`, "Critical");
    return;
  }

  state.section232ResetBusy = true;
  render();
  try {
    const response = await request("/api/metal-composition/section-232/reset", "POST");
    clearCurrentDraftBatch();
    state.selectedSourceFiles = [];
    state.section232ResetConfirmation = "";
    state.isSection232ResetDialogOpen = false;
    clearInputValue("settings-source-upload-input");
    await refreshPageState();
    applyMessage(buildSection232ResetMessage(response), "Positive");
    render();
  } catch (error) {
    applyMessage(error.message, "Negative");
    render();
  } finally {
    state.section232ResetBusy = false;
    render();
  }
}

function openPublishedReviewWorkspace() {
  const href = buildPublishedReviewHref(state.rulesetSummary.active_ruleset_version);
  if (!href) {
    showMessage("No published Section 232 ruleset is available to review yet.", "Information");
    return;
  }
  navigateTo(href);
}

function openDraftReviewWorkspace() {
  const launchState = buildSettingsLaunchState({
    currentDraftBatch: state.currentDraftBatch,
    rulesetSummary: state.rulesetSummary
  });
  if (!launchState.hasLaunchTarget) {
    showMessage("Process a draft batch before opening the Section 232 review workspace.", "Critical");
    return;
  }
  navigateTo(launchState.href);
}

async function cancelCurrentDraftBatch() {
  const cancelState = buildCancelDraftState({
    currentDraftBatch: state.currentDraftBatch,
    rulesetSummary: state.rulesetSummary,
    busy: state.draftCancelBusy
  });
  if (!cancelState.hasCancelTarget) {
    showMessage("No pending Section 232 draft batch is available to cancel.", "Information");
    return;
  }
  if (
    !window.confirm(
      `Cancel draft batch ${cancelState.batchId} and remove its source PDFs?`
    )
  ) {
    return;
  }

  state.draftCancelBusy = true;
  render();
  try {
    const response = await request(
      `/api/metal-composition/section-232/draft-batches/${encodeURIComponent(cancelState.batchId)}`,
      "DELETE"
    );
    clearCurrentDraftBatch();
    await refreshPageState();
    restoreDraftBatchLaunchState();
    applyMessage(
      `Cancelled draft batch ${response.batch_id}. Removed ${response.deleted_source_count || 0} source PDF${
        response.deleted_source_count === 1 ? "" : "s"
      } and ${response.deleted_draft_rule_count || 0} draft rule${
        response.deleted_draft_rule_count === 1 ? "" : "s"
      }.`,
      "Positive"
    );
    render();
  } catch (error) {
    await refreshPageState();
    if (error.status === 404 || error.status === 409) {
      clearCurrentDraftBatch();
      restoreDraftBatchLaunchState();
      applyMessage(
        error.status === 404
          ? `Draft batch ${cancelState.batchId} is no longer available. Local draft state was cleared.`
          : `Draft batch ${cancelState.batchId} can no longer be cancelled. Local draft state was refreshed.`,
        "Information"
      );
    } else {
      reconcileDraftBatchLaunchState();
      applyMessage(error.message, "Negative");
    }
    render();
  } finally {
    state.draftCancelBusy = false;
    render();
  }
}

function handleSummaryClick(event) {
  const actionElement = event.target.closest?.("[data-summary-action]");
  if (!actionElement) return;
  if (actionElement.dataset.summaryAction === "open-published-review") {
    openPublishedReviewWorkspace();
  }
}

function handleHTSCatalogAction(event) {
  const actionElement = event.target.closest?.("[data-hts-catalog-action]");
  if (!actionElement) return;
  if (actionElement.dataset.htsCatalogAction === "delete") {
    void deleteHTSCatalogSource(actionElement.dataset.filename || "");
  }
}

function bindEvents() {
  document.getElementById("settings-summary-cards")?.addEventListener("click", handleSummaryClick);
  document.getElementById("settings-mm-composition-toggle")?.addEventListener("change", (event) => {
    state.appSettingsDraft = Boolean(event.target?.checked);
    render();
  });
  document.getElementById("settings-composition-mode-save-button")?.addEventListener("click", saveCompositionMode);

  document.getElementById("settings-source-select-button")?.addEventListener("click", () => {
    document.getElementById("settings-source-upload-input")?.click();
  });
  document.getElementById("settings-source-upload-input")?.addEventListener("change", (event) => {
    state.selectedSourceFiles = Array.from(event.target.files || []);
    render();
  });
  document.getElementById("settings-source-process-button")?.addEventListener("click", processSection232DraftBatch);
  document.getElementById("settings-review-open-button")?.addEventListener("click", openDraftReviewWorkspace);
  document.getElementById("settings-review-cancel-button")?.addEventListener("click", cancelCurrentDraftBatch);

  document.getElementById("settings-hts-catalog-select-button")?.addEventListener("click", () => {
    document.getElementById("settings-hts-catalog-upload-input")?.click();
  });
  document.getElementById("settings-hts-catalog-upload-input")?.addEventListener("change", (event) => {
    state.selectedCatalogFiles = Array.from(event.target.files || []);
    render();
  });
  document.getElementById("settings-hts-catalog-upload-button")?.addEventListener("click", uploadHTSCatalogSources);
  document.getElementById("settings-hts-catalog-list")?.addEventListener("click", handleHTSCatalogAction);

  document.getElementById("settings-material-master-select-button")?.addEventListener("click", () => {
    document.getElementById("settings-material-master-upload-input")?.click();
  });
  document.getElementById("settings-material-master-upload-input")?.addEventListener("change", (event) => {
    state.selectedMaterialMasterFile = Array.from(event.target.files || [])[0] || null;
    render();
  });
  document.getElementById("settings-material-master-refresh-button")?.addEventListener("click", uploadMaterialMasterFile);

  document.getElementById("settings-code-review-button")?.addEventListener("click", openPublishedReviewWorkspace);

  document.getElementById("settings-section-232-reset-button")?.addEventListener("click", openSection232ResetDialog);
  document.getElementById("settings-section-232-reset-cancel-button")?.addEventListener("click", closeSection232ResetDialog);
  document.getElementById("settings-section-232-reset-confirmation-input")?.addEventListener("input", (event) => {
    syncSection232ResetConfirmation(event.target?.value || "");
  });
  document.getElementById("settings-section-232-reset-confirm-button")?.addEventListener("click", resetSection232Data);

  document.getElementById("settings-reset-button")?.addEventListener("click", openResetDialog);
  document.getElementById("settings-reset-cancel-button")?.addEventListener("click", closeResetDialog);
  document.getElementById("settings-reset-confirm-button")?.addEventListener("click", resetClassifications);
}

export default async function initSettingsPage() {
  resetState();
  bindEvents();
  render();
  try {
    await refreshPageState();
    restoreDraftBatchLaunchState({
      showUnavailableMessage: true
    });
  } catch (error) {
    applyMessage(error.message, "Negative");
    render();
  }
}
