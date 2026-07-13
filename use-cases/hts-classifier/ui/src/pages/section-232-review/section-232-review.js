import "@ui5/webcomponents/dist/Button.js";
import "@ui5/webcomponents/dist/CheckBox.js";
import "@ui5/webcomponents/dist/Input.js";
import "@ui5/webcomponents/dist/Label.js";
import "@ui5/webcomponents/dist/MessageStrip.js";
import "@ui5/webcomponents/dist/Tag.js";
import "@ui5/webcomponents/dist/TextArea.js";
import "@ui5/webcomponents/dist/Title.js";

import { pageRouter } from "../../modules/router.js";
import { buildQuery, request } from "../../services/api.js";
import {
  SECTION_232_REVIEW_PAGE_SIZE,
  buildSection232DraftBulkReviewPayload,
  buildSection232ReviewMutationRecoveryState,
  buildSection232ReviewRowViewModel,
  buildSection232ReviewRequestQuery,
  clearSection232DraftSelection,
  createSection232DraftSelectionState,
  getSection232DraftSelectionCount,
  parseSection232ReviewRoute,
  resolveSection232ReviewSelection,
  selectAllSection232DraftRows,
  toggleSection232DraftSelection
} from "./section-232-review_state.js";
import {
  clearPersistedPublishedBy,
  clearPersistedSettingsDraftBatch,
  persistPublishedBy,
  readPersistedPublishedBy
} from "./section-232-review_persistence.js";
import { trimText } from "./section-232-review_shared.js";
import { renderSection232ReviewView } from "./section-232-review_view.js";

const state = {
  route: parseSection232ReviewRoute(""),
  loading: false,
  mutationInFlight: false,
  rows: [],
  workspace: null,
  detailCandidateId: "",
  htsSearchInput: "",
  appliedHtsSearchQuery: "",
  draftSelection: createSection232DraftSelectionState(),
  publishedBy: "",
  message: "",
  messageDesign: "Information",
  loadRequestId: 0,
  paging: {
    total: 0,
    limit: SECTION_232_REVIEW_PAGE_SIZE,
    offset: 0
  }
};

function applyMessage(text = "", design = "Information") {
  state.message = text;
  state.messageDesign = design;
}

function isSameWorkspace(left, right) {
  return trimText(left?.mode) === trimText(right?.mode)
    && trimText(left?.batchId) === trimText(right?.batchId)
    && trimText(left?.version) === trimText(right?.version);
}

function resetWorkspaceState(
  route,
  {
    loading = false,
    message = "",
    messageDesign = "Information",
    preserveDraftSelection = false,
    preserveDetailSelection = false
  } = {}
) {
  const sameWorkspace = isSameWorkspace(state.route, route);
  state.route = route;
  state.loading = Boolean(loading);
  state.workspace = sameWorkspace ? state.workspace : null;
  state.rows = sameWorkspace ? state.rows : [];
  state.paging = {
    total: sameWorkspace ? state.paging.total : 0,
    limit: SECTION_232_REVIEW_PAGE_SIZE,
    offset: sameWorkspace ? state.paging.offset : 0
  };
  if (!sameWorkspace) {
    state.htsSearchInput = "";
    state.appliedHtsSearchQuery = "";
  }
  if (!(preserveDraftSelection && sameWorkspace && route?.mode === "draft")) {
    state.draftSelection = clearSection232DraftSelection();
  }
  if (!(preserveDetailSelection && sameWorkspace)) {
    state.detailCandidateId = "";
  }
  state.publishedBy = route?.mode === "draft"
    ? sameWorkspace
      ? state.publishedBy
      : readPersistedPublishedBy(route.batchId)
    : "";
  applyMessage(message, messageDesign);
}

function getDraftBatchId() {
  return trimText(state.workspace?.batch?.batch_id || state.route?.batchId);
}

function getDraftSelectionCount() {
  return getSection232DraftSelectionCount(state.draftSelection, state.paging.total);
}

function getDetailSelection() {
  return resolveSection232ReviewSelection(
    state.rows,
    state.detailCandidateId ? [state.detailCandidateId] : [],
    { mode: state.route.mode }
  );
}

async function recoverFromMutationFailure(message, design = "Negative") {
  Object.assign(state, buildSection232ReviewMutationRecoveryState(message, design));
  render();
  await loadWorkspace(state.route, {
    preserveMessage: true,
    offset: state.paging.offset,
    preserveDraftSelection: state.route.mode === "draft",
    preserveDetailSelection: true
  });
}

function buildViewState() {
  return {
    route: state.route,
    loading: state.loading,
    mutationInFlight: state.mutationInFlight,
    rows: state.rows,
    workspace: state.workspace,
    paging: state.paging,
    draftSelection: state.draftSelection,
    publishedBy: state.publishedBy,
    message: state.message,
    messageDesign: state.messageDesign,
    htsSearchInput: state.htsSearchInput,
    appliedHtsSearchQuery: state.appliedHtsSearchQuery,
    detailSelection: getDetailSelection(),
    selectedCount: getDraftSelectionCount()
  };
}

function render() {
  renderSection232ReviewView(buildViewState());
}

function setDraftSelection(nextSelection) {
  state.draftSelection = nextSelection;
  render();
}

function toggleRowSelection(candidateId, isSelected) {
  setDraftSelection(toggleSection232DraftSelection(state.draftSelection, candidateId, isSelected));
}

function selectAllRows() {
  if (state.loading || state.mutationInFlight || state.route.mode !== "draft") return;
  setDraftSelection(selectAllSection232DraftRows());
}

async function reviewDraftRule(candidateId, decision) {
  const batchId = getDraftBatchId();
  if (!batchId || state.route.mode !== "draft" || state.loading || state.mutationInFlight) return;
  if (!trimText(state.publishedBy)) {
    applyMessage("Enter Published by before reviewing draft rows.", "Information");
    render();
    return;
  }

  state.mutationInFlight = true;
  render();
  try {
    await request(
      `/api/metal-composition/section-232/draft-batches/${encodeURIComponent(batchId)}/rules/${encodeURIComponent(candidateId)}`,
      "PATCH",
      { decision }
    );
    state.mutationInFlight = false;
    applyMessage(`Marked ${candidateId} as ${decision}.`, "Positive");
    await loadWorkspace(state.route, {
      preserveMessage: true,
      offset: state.paging.offset,
      preserveDraftSelection: true,
      preserveDetailSelection: true
    });
    return;
  } catch (error) {
    await recoverFromMutationFailure(error.message || "Failed to update draft review decision.");
    return;
  } finally {
    if (state.mutationInFlight) {
      state.mutationInFlight = false;
      render();
    }
  }
}

async function reviewSelectedRows(decision) {
  const batchId = getDraftBatchId();
  const selectedCount = getDraftSelectionCount();
  if (!batchId || state.route.mode !== "draft" || state.loading || state.mutationInFlight || !selectedCount) return;
  if (!trimText(state.publishedBy)) {
    applyMessage("Enter Published by before reviewing draft rows.", "Information");
    render();
    return;
  }

  state.mutationInFlight = true;
  render();
  try {
    const response = await request(
      `/api/metal-composition/section-232/draft-batches/${encodeURIComponent(batchId)}/rules`,
      "PATCH",
      buildSection232DraftBulkReviewPayload(state.draftSelection, decision)
    );
    state.mutationInFlight = false;
    state.draftSelection = clearSection232DraftSelection();
    const processedCount = Number(response?.updated_count) || selectedCount;
    applyMessage(`Marked ${processedCount} selected row${processedCount === 1 ? "" : "s"} as ${decision}.`, "Positive");
    await loadWorkspace(state.route, {
      preserveMessage: true,
      offset: state.paging.offset,
      preserveDraftSelection: false,
      preserveDetailSelection: true
    });
    return;
  } catch (error) {
    await recoverFromMutationFailure(error.message || "Failed to update selected draft review rows.");
    return;
  } finally {
    if (state.mutationInFlight) {
      state.mutationInFlight = false;
      render();
    }
  }
}

async function deleteDraftHtsCode(htsCode) {
  const batchId = getDraftBatchId();
  const normalizedCode = trimText(htsCode);
  if (!batchId || !normalizedCode || state.route.mode !== "draft" || state.loading || state.mutationInFlight) return;
  if (!window.confirm(`Delete all draft rows for HTS code ${normalizedCode} from this batch?`)) {
    return;
  }

  state.mutationInFlight = true;
  render();
  try {
    const response = await request(
      `/api/metal-composition/section-232/draft-batches/${encodeURIComponent(batchId)}/hts-codes/${encodeURIComponent(normalizedCode)}`,
      "DELETE"
    );
    state.mutationInFlight = false;
    state.draftSelection = clearSection232DraftSelection();
    state.detailCandidateId = "";
    applyMessage(
      `Deleted ${response?.deleted_count || 0} draft row${Number(response?.deleted_count) === 1 ? "" : "s"} for ${normalizedCode}.`,
      "Positive"
    );
    await loadWorkspace(state.route, {
      preserveMessage: true,
      offset: state.paging.offset,
      preserveDraftSelection: false,
      preserveDetailSelection: false
    });
    return;
  } catch (error) {
    await recoverFromMutationFailure(error.message || "Failed to delete draft HTS code.");
    return;
  } finally {
    if (state.mutationInFlight) {
      state.mutationInFlight = false;
      render();
    }
  }
}

async function deletePublishedHtsCode(htsCode) {
  const normalizedCode = trimText(htsCode);
  if (!normalizedCode || state.route.mode !== "published" || state.loading || state.mutationInFlight) return;
  if (!trimText(state.publishedBy)) {
    applyMessage("Enter Published by before deleting a published HTS code.", "Information");
    render();
    return;
  }
  if (!window.confirm(`Delete HTS code ${normalizedCode} from the active published ruleset? This creates a new published version.`)) {
    return;
  }

  state.mutationInFlight = true;
  render();
  try {
    const response = await request(
      `/api/metal-composition/section-232/published/hts-codes/${encodeURIComponent(normalizedCode)}/delete`,
      "POST",
      { published_by: trimText(state.publishedBy) }
    );
    const publishedVersion = trimText(response?.published_version);
    if (!publishedVersion) {
      throw new Error("Delete succeeded but no published version was returned.");
    }
    state.mutationInFlight = false;
    pageRouter.navigate(`/settings/section-232/review?version=${encodeURIComponent(publishedVersion)}`);
    return;
  } catch (error) {
    await recoverFromMutationFailure(error.message || "Failed to delete published HTS code.");
    return;
  } finally {
    if (state.mutationInFlight) {
      state.mutationInFlight = false;
      render();
    }
  }
}

async function publishDraftBatch() {
  const batchId = getDraftBatchId();
  const acceptedCount = Number(state.workspace?.batch?.accepted_count || 0);
  if (
    !batchId
    || state.route.mode !== "draft"
    || state.loading
    || state.mutationInFlight
    || acceptedCount <= 0
    || !trimText(state.publishedBy)
  ) {
    return;
  }

  state.mutationInFlight = true;
  render();
  try {
    const response = await request(
      `/api/metal-composition/section-232/draft-batches/${encodeURIComponent(batchId)}/publish`,
      "POST",
      { published_by: trimText(state.publishedBy) }
    );
    const publishedVersion = trimText(response?.published_version);
    if (!publishedVersion) {
      throw new Error("Publish succeeded but no published version was returned.");
    }
    clearPersistedPublishedBy(batchId);
    clearPersistedSettingsDraftBatch(batchId);
    state.mutationInFlight = false;
    pageRouter.navigate(`/settings/section-232/review?version=${encodeURIComponent(publishedVersion)}`);
    return;
  } catch (error) {
    await recoverFromMutationFailure(error.message || "Failed to publish draft batch.");
    return;
  } finally {
    if (state.mutationInFlight) {
      state.mutationInFlight = false;
      render();
    }
  }
}

async function loadWorkspace(
  routeInput = state.route,
  {
    preserveMessage = false,
    offset = state.paging.offset,
    preserveDraftSelection = false,
    preserveDetailSelection = false
  } = {}
) {
  const nextRoute = typeof routeInput === "string" || routeInput instanceof URLSearchParams || routeInput?.querystring
    ? parseSection232ReviewRoute(routeInput)
    : routeInput;
  const sameWorkspace = isSameWorkspace(state.route, nextRoute);
  const nextOffset = sameWorkspace ? Math.max(0, Number(offset) || 0) : 0;
  const loadRequestId = state.loadRequestId + 1;
  state.loadRequestId = loadRequestId;

  resetWorkspaceState(nextRoute, {
    loading: nextRoute.isValid,
    message: preserveMessage ? state.message : "",
    messageDesign: preserveMessage ? state.messageDesign : "Information",
    preserveDraftSelection,
    preserveDetailSelection
  });
  state.paging.offset = nextOffset;

  if (!nextRoute.isValid) {
    applyMessage(nextRoute.error, "Critical");
    render();
    return;
  }

  render();
  try {
    const workspace = await request(
      `/api/metal-composition/section-232/review${buildQuery({
        ...buildSection232ReviewRequestQuery(nextRoute, state.appliedHtsSearchQuery),
        limit: SECTION_232_REVIEW_PAGE_SIZE,
        offset: nextOffset
      })}`
    );
    if (loadRequestId !== state.loadRequestId) {
      return;
    }
    state.workspace = workspace;
    state.rows = (workspace.rows || []).map(buildSection232ReviewRowViewModel);
    state.paging = {
      total: Number(workspace.total || 0),
      limit: Number(workspace.limit || SECTION_232_REVIEW_PAGE_SIZE),
      offset: Number(workspace.offset || 0)
    };
    if (!state.rows.some((row) => row.candidateId === state.detailCandidateId)) {
      state.detailCandidateId = "";
    }
    state.loading = false;
    if (state.route.mode !== "draft") {
      state.draftSelection = clearSection232DraftSelection();
    }
    if (!preserveMessage) {
      applyMessage("", "Information");
    }
  } catch (error) {
    if (loadRequestId !== state.loadRequestId) {
      return;
    }
    state.loading = false;
    applyMessage(error.message || "Failed to load review workspace.", "Negative");
  }
  render();
}

function goToOffset(nextOffset) {
  if (state.loading || state.mutationInFlight) return;
  loadWorkspace(state.route, {
    preserveMessage: true,
    offset: nextOffset,
    preserveDraftSelection: state.route.mode === "draft",
    preserveDetailSelection: false
  });
}

function updateHtsSearchInput(value) {
  const normalizedQuery = trimText(value);
  if (normalizedQuery === state.htsSearchInput) return;
  state.htsSearchInput = normalizedQuery;
  render();
}

function submitHtsSearch() {
  const normalizedQuery = trimText(state.htsSearchInput);
  if (normalizedQuery === state.appliedHtsSearchQuery || state.loading || state.mutationInFlight) return;
  state.appliedHtsSearchQuery = normalizedQuery;
  state.paging.offset = 0;
  loadWorkspace(state.route, {
    preserveMessage: true,
    offset: 0,
    preserveDraftSelection: state.route.mode === "draft",
    preserveDetailSelection: true
  });
}

function clearHtsSearch() {
  const hadAppliedSearch = Boolean(trimText(state.appliedHtsSearchQuery));
  state.htsSearchInput = "";
  state.appliedHtsSearchQuery = "";
  state.paging.offset = 0;
  render();
  if (!hadAppliedSearch || state.loading || state.mutationInFlight) return;
  loadWorkspace(state.route, {
    preserveMessage: true,
    offset: 0,
    preserveDraftSelection: state.route.mode === "draft",
    preserveDetailSelection: true
  });
}

function bindEvents() {
  document.getElementById("section-232-review-back-button")?.addEventListener("click", () => {
    pageRouter.navigate("/settings");
  });

  document.getElementById("section-232-review-refresh-button")?.addEventListener("click", () => {
    loadWorkspace(state.route, {
      preserveMessage: true,
      offset: state.paging.offset,
      preserveDraftSelection: state.route.mode === "draft",
      preserveDetailSelection: true
    });
  });

  document.getElementById("section-232-review-hts-search")?.addEventListener("input", (event) => {
    updateHtsSearchInput(event.target?.value ?? "");
  });

  document.getElementById("section-232-review-submit-search")?.addEventListener("click", () => {
    submitHtsSearch();
  });

  document.getElementById("section-232-review-clear-search")?.addEventListener("click", () => {
    clearHtsSearch();
  });

  document.getElementById("section-232-review-publish-button")?.addEventListener("click", () => {
    publishDraftBatch();
  });
  document.getElementById("section-232-review-published-by-input")?.addEventListener("input", (event) => {
    state.publishedBy = event.target?.value ?? "";
    if (state.route.mode === "draft") {
      persistPublishedBy(getDraftBatchId(), state.publishedBy);
    }
    render();
  });

  document.getElementById("section-232-review-accept-selected")?.addEventListener("click", () => {
    reviewSelectedRows("accepted");
  });

  document.getElementById("section-232-review-reject-selected")?.addEventListener("click", () => {
    reviewSelectedRows("rejected");
  });

  document.getElementById("section-232-review-select-all")?.addEventListener("click", () => {
    selectAllRows();
  });

  document.getElementById("section-232-review-clear-selection")?.addEventListener("click", () => {
    setDraftSelection(clearSection232DraftSelection());
  });

  document.getElementById("section-232-review-page-previous")?.addEventListener("click", () => {
    goToOffset(Math.max(0, state.paging.offset - state.paging.limit));
  });

  document.getElementById("section-232-review-page-next")?.addEventListener("click", () => {
    goToOffset(state.paging.offset + state.paging.limit);
  });

  document.getElementById("section-232-review-table")?.addEventListener("change", (event) => {
    if (state.loading || state.mutationInFlight) return;
    const candidateId = event.target?.dataset?.rowSelect;
    if (!candidateId) return;
    toggleRowSelection(candidateId, Boolean(event.target.checked));
  });

  document.getElementById("section-232-review-table")?.addEventListener("click", (event) => {
    if (state.loading || state.mutationInFlight) return;
    const rowElement = event.target?.closest?.("[data-row-id]");
    if (!rowElement || event.target?.dataset?.rowSelect) return;
    state.detailCandidateId = rowElement.dataset.rowId;
    render();
  });

  document.getElementById("section-232-review-detail")?.addEventListener("click", (event) => {
    const decision = event.target?.dataset?.detailDecision;
    const candidateId = getDetailSelection().selectedRow?.candidateId;
    const deleteHtsCode = event.target?.dataset?.detailDeleteHtsCode;
    if (decision && candidateId) {
      reviewDraftRule(candidateId, decision);
      return;
    }
    if (!deleteHtsCode) return;
    if (state.route.mode === "draft") {
      deleteDraftHtsCode(deleteHtsCode);
      return;
    }
    deletePublishedHtsCode(deleteHtsCode);
  });
}

export default function init(ctx) {
  const initialRoute = parseSection232ReviewRoute(ctx?.querystring || window.location.search);
  resetWorkspaceState(initialRoute, { loading: initialRoute.isValid });
  bindEvents();
  render();
  loadWorkspace(initialRoute, { offset: 0 });
}
