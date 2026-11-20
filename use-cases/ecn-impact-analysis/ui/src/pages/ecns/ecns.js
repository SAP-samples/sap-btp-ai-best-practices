import "@ui5/webcomponents/dist/Tag.js";
import "@ui5/webcomponents/dist/MessageStrip.js";
import { reviewPresentation } from "../../modules/review-outcomes.js";
import { suppliedPlantLabel } from "../../modules/plant-display.js";
import { bindEcnImportDialog } from "../../components/export-import.js";
import { deleteEcns, generateSummaries } from "../../services/workspace-api.js";
import { getAnalysisProgress, listEcns, uploadSources } from "../../services/ecn-api.js";
import { startAnalysisPolling } from "../../modules/analysis-polling.js";
import { createLatestRequestGate } from "../../modules/request-generation.js";
import { detailHref, DEFAULT_LIST_STATE, parseListState, serializeListState } from "../../modules/list-state.js";

const defaultApi = { deleteEcns, generateSummaries, getAnalysisProgress, listEcns, uploadSources };

const statusPresentation = Object.freeze({
  READY: ["Ready", "Neutral"],
  QUEUED: ["Waiting", "Neutral"],
  BLOCKED: ["Blocked", "Critical"],
  RUNNING: ["Running", "Information"],
  COMPLETED: ["Available", "Positive"],
  COMPLETED_WITH_GAPS: ["Available", "Positive"],
  FAILED: ["Unavailable", "Negative"],
  PARTIAL: ["Partial", "Critical"],
});

const impactPresentation = Object.freeze({
  HIGH: ["High", "Negative"],
  MEDIUM: ["Medium", "Critical"],
  LOW: ["Low", "Positive"],
  UNKNOWN: ["Unknown", "Neutral"],
  NOT_APPLICABLE: ["N/A", "Neutral"],
});

const tagDesignByTone = Object.freeze({
  success: "Positive",
  information: "Information",
  warning: "Critical",
  error: "Negative",
  neutral: "Neutral",
});

/** Assign nullable API values to a text-only DOM node. */
function setText(node, value) {
  node.textContent = value == null ? "" : String(value);
}

/** Render one Inbox feedback message while retaining its explicit live-region semantics. */
function setFeedback(root, selector, value) {
  const message = root.querySelector(selector);
  setText(message, value);
  message.hidden = !value;
}

/** Format one deletion count with a readable singular or optional irregular plural noun. */
function countLabel(count, singular, plural = `${singular}s`) {
  return `${count} ${count === 1 ? singular : plural}`;
}

/** Consume and render the one-shot summary left by a successful history reset. */
function consumeResetSummary(root) {
  const rawSummary = sessionStorage.getItem("ecn-history-reset-summary");
  if (!rawSummary) return;
  sessionStorage.removeItem("ecn-history-reset-summary");
  try {
    const counts = JSON.parse(rawSummary);
    setFeedback(root, "#reset-summary", `History reset: ${[
      countLabel(counts.deleted_upload_count, "upload"),
      countLabel(counts.deleted_ecn_count, "ECN"),
      countLabel(counts.deleted_analysis_count, "analysis", "analyses"),
      countLabel(counts.deleted_job_count, "job"),
      countLabel(counts.deleted_attempt_count, "attempt"),
      countLabel(counts.deleted_bom_row_count, "BOM row"),
    ].join(", ")}, and ${countLabel(counts.deleted_source_artifact_count, "source artifact")} deleted.`);
  } catch {
    // Corrupt session-only state is discarded rather than blocking the list page.
  }
}

/** Format an API timestamp as a compact local date while preserving invalid text. */
function formatDate(value) {
  if (value == null || value === "") return "";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? String(value || "") : new Intl.DateTimeFormat().format(date);
}

/** Return one accessible UI5 semantic tag without exposing internal enums. */
function createStatusTag(value, presentations, fallback = "Unknown") {
  const [label, presentation] = presentations[value] || [fallback, "Neutral"];
  const tag = document.createElement("ui5-tag");
  tag.className = "ecn-status-tag";
  tag.setAttribute("design", tagDesignByTone[presentation] || presentation);
  tag.setAttribute("size", "S");
  tag.setAttribute("aria-label", label);
  tag.textContent = label;
  return tag;
}

/** Append one table cell with an optional semantic class. */
function appendCell(row, content, className = "") {
  const cell = document.createElement("td");
  if (className) cell.className = className;
  if (content instanceof Node) cell.append(content);
  else setText(cell, content);
  row.append(cell);
  return cell;
}

/**
 * Render the plants affected by one ECN, one label per line.
 *
 * Input: `affected_plants` from the list API (`null` = not assessed yet, `[]` = none linked).
 * Output: a node showing each label, or a muted "—" when there is nothing to show.
 */
function createPlantList(plants) {
  const list = document.createElement("div");
  if (!plants?.length) {
    list.className = "muted-value";
    list.textContent = "—";
    list.title = plants ? "No plant linked" : "Not assessed yet";
    return list;
  }
  list.className = "plant-list";
  for (const label of plants) {
    const line = document.createElement("span");
    line.textContent = suppliedPlantLabel(label);
    list.append(line);
  }
  return list;
}

function displayAnalysisState(item) {
  return item.analysis_state === "FAILED" && item.has_partial_analysis
    ? "PARTIAL" : item.analysis_state;
}

function createEcnLink(item, state) {
  const link = document.createElement("a");
  link.href = detailHref(item.ecn_row_id, state);
  link.textContent = item.display_ecn_number;
  link.dataset.ecnRowId = item.ecn_row_id;
  link.className = "ecn-link";
  return link;
}

function createSelectionControl(item, isSelected) {
  const select = document.createElement("input");
  select.type = "checkbox";
  select.className = "summary-selection";
  select.value = item.ecn_row_id;
  select.checked = isSelected(item.ecn_row_id);
  select.setAttribute("aria-label", `Select ${item.display_ecn_number} for summary`);
  return select;
}

function createViewLink(item, state) {
  const link = document.createElement("a");
  link.href = detailHref(item.ecn_row_id, state);
  link.className = "view-action";
  link.textContent = "View";
  link.setAttribute("aria-label", `View ${item.display_ecn_number}`);
  return link;
}

function appendCardDetail(list, label, value) {
  const term = document.createElement("dt");
  term.textContent = label;
  const detail = document.createElement("dd");
  if (value instanceof Node) detail.append(value);
  else setText(detail, value);
  list.append(term, detail);
}

/** Render the narrow-screen ECN list from the same API-backed rows as the desktop table. */
export function renderEcnCards(host, items, state, isSelected = () => false) {
  host.replaceChildren();
  for (const item of items) {
    const card = document.createElement("article");
    card.className = "ecn-result-card";
    card.setAttribute("aria-label", `${item.display_ecn_number}: ${item.description_summary || item.title || "No description supplied"}`);

    const header = document.createElement("header");
    header.className = "ecn-result-card-header";
    const identity = document.createElement("div");
    identity.className = "ecn-result-card-identity";
    identity.append(createSelectionControl(item, isSelected), createEcnLink(item, state));
    header.append(identity, createViewLink(item, state));

    const description = document.createElement("p");
    description.className = "ecn-result-card-description";
    description.textContent = item.description_summary || item.title || "No description supplied";

    const details = document.createElement("dl");
    details.className = "ecn-result-card-details";
    appendCardDetail(details, "Analysis", createStatusTag(displayAnalysisState(item), statusPresentation));
    appendCardDetail(details, "Review", createStatusTag(item.review_outcome, reviewPresentation, "Unreviewed"));
    appendCardDetail(details, "Impact", createStatusTag(item.overall_impact, impactPresentation));
    const warnings = document.createElement("span");
    warnings.className = item.counts.warning_count ? "warning-count" : "muted-value";
    warnings.textContent = String(item.counts.warning_count);
    appendCardDetail(details, "Warnings", warnings);
    const plants = item.affected_plants?.length;
    appendCardDetail(details, "Plants", plants ? `${plants} affected` : item.affected_plants ? "None linked" : "Not assessed");

    card.append(header, description, details);
    host.append(card);
  }
}

/** Render only API-owned fields; source strings are always text nodes. */
export function renderEcnRows(tbody, items, state, isSelected = () => false) {
  tbody.replaceChildren();
  for (const item of items) {
    const row = document.createElement("tr");
    const ecnLink = createEcnLink(item, state);

    const fullDescription = item.description_summary || item.title || "No description supplied";
    const description = document.createElement("span");
    description.className = "description-clamp";
    description.textContent = fullDescription;
    description.title = fullDescription;
    description.tabIndex = 0;
    description.setAttribute("aria-label", fullDescription);

    const bomMatches = item.counts.ebom_match_count + item.counts.mbom_match_count;
    const materialCount = document.createElement("span");
    materialCount.className = "metric-stack";
    materialCount.textContent = `${item.counts.affected_item_count}`;
    materialCount.title = `${bomMatches} total EBOM and MBOM matches`;

    const warningCount = document.createElement("span");
    warningCount.className = item.counts.warning_count ? "warning-count" : "muted-value";
    warningCount.textContent = String(item.counts.warning_count);

    const viewLink = createViewLink(item, state);

    const identity = document.createElement("div");
    const select = createSelectionControl(item, isSelected);
    identity.append(select, ecnLink);
    appendCell(row, identity, "ecn-cell");
    appendCell(row, description, "description-cell");
    appendCell(row, createPlantList(item.affected_plants), "plant-cell");
    appendCell(row, formatDate(item.resolution_date), "date-cell");
    appendCell(row, formatDate(item.uploaded_at), "date-cell");
    appendCell(row, materialCount, "count-cell");
    appendCell(row, warningCount, "count-cell");
    const displayState = displayAnalysisState(item);
    appendCell(row, createStatusTag(displayState, statusPresentation), "status-cell");
    appendCell(row, createStatusTag(item.overall_impact, impactPresentation), "impact-cell");
    appendCell(row, createStatusTag(item.review_outcome, reviewPresentation, "Unreviewed"), "status-cell");
    appendCell(row, viewLink, "action-cell");
    tbody.append(row);
  }
}

/** Build one ECN-list initializer with injectable IO for deterministic tests. */
export function createEcnsPage({
  api = defaultApi,
  startPolling = startAnalysisPolling,
  bindImportDialog = bindEcnImportDialog,
  uuid = () => crypto.randomUUID(),
} = {}) {
  /** Initialize list, upload, and progress behavior for one route generation. */
  return async function init(context) {
    const root = context.container;
    consumeResetSummary(root);
    let state = parseListState(context.query);
    let stopPolling = () => {};
    let disposeImport = () => {};
    let progressSignature = null;
    let totalMatching = 0;
    let allFiltered = false;
    let selectedIds = new Set();
    let excludedIds = new Set();
    let generationRequestId = null;
    // Display numbers of rows seen on any loaded page, so the delete dialog can name explicit selections.
    const ecnNumbers = new Map();
    const listReads = createLatestRequestGate(context.signal);
    const progressReads = createLatestRequestGate(context.signal);
    const listeners = [];
    /** Register an event listener and retain its exact cleanup. */
    const on = (node, type, handler) => {
      node.addEventListener(type, handler);
      listeners.push(() => node.removeEventListener(type, handler));
    };

    const overviewFilters = [
      { id: "kpi-total-filter", value: "", message: "Showing ECNs with any analysis state." },
      { id: "kpi-complete-filter", value: "SUMMARIES_AVAILABLE", message: "Showing ECNs with completed summaries, including summaries with gaps." },
      { id: "kpi-progress-filter", value: "IN_PROGRESS", message: "Showing ECNs whose analysis is waiting or running." },
      { id: "kpi-attention", value: "FAILED", message: "Showing ECNs whose analysis needs attention." },
    ];

    /** Indicate the applied analysis group, including changes made through the form. */
    const syncOverviewFilters = () => {
      for (const { id, value } of overviewFilters) {
        root.querySelector(`#${id}`).setAttribute("aria-pressed", String(state.analysis_state === value));
      }
    };

    /** Copy current allowlisted state into visible filter controls and overview cards. */
    const fillFilters = () => {
      for (const [key, value] of Object.entries(state)) {
        const field = root.querySelector(`[name="${key}"]`);
        if (field) field.value = String(value);
      }
      syncOverviewFilters();
    };

    const filtersForm = root.querySelector("#filters");
    const secondaryFilters = root.querySelector("#secondary-filters");
    const filtersDisclosure = root.querySelector("#filters-disclosure");
    let secondaryFiltersExpanded = false;

    /** Count non-default secondary filter values without treating default sort/paging as active. */
    const activeSecondaryFilterCount = () => {
      const form = new FormData(filtersForm);
      return [
        form.get("ecn_number"),
        form.get("plant"),
        form.get("has_warnings"),
      ].filter((value) => value !== "").length
        + (form.get("order") !== DEFAULT_LIST_STATE.order ? 1 : 0)
        + (String(form.get("page_size")) !== String(DEFAULT_LIST_STATE.page_size) ? 1 : 0);
    };

    /** Keep the disclosure name synchronized with the non-default secondary criteria. */
    const updateActiveFilterCount = () => {
      const count = activeSecondaryFilterCount();
      const label = `${count} active`;
      setText(root.querySelector("#active-filter-count"), label);
      filtersDisclosure.setAttribute(
        "aria-label",
        count === 1 ? "Filters, 1 active filter" : `Filters, ${count} active filters`,
      );
    };

    /** Reveal secondary filters only on narrow layouts; desktop retains the direct form. */
    const syncFilterDisclosure = () => {
      const narrow = window.matchMedia?.("(max-width: 600px)").matches ?? window.innerWidth <= 600;
      secondaryFilters.hidden = narrow && !secondaryFiltersExpanded;
      filtersDisclosure.setAttribute("aria-expanded", String(!secondaryFilters.hidden));
    };

    /** Return whether one visible row belongs to the current cross-page selection. */
    const isSelected = (ecnRowId) => allFiltered
      ? !excludedIds.has(ecnRowId)
      : selectedIds.has(ecnRowId);

    /** Count explicit or filter-wide members of the current selection. */
    const selectedCount = () => allFiltered
      ? Math.max(0, totalMatching - excludedIds.size)
      : selectedIds.size;

    /** Render the selection count and generation/deletion availability. */
    const showSummarySelection = () => {
      const count = selectedCount();
      setText(
        root.querySelector("#summary-selection-status"),
        allFiltered ? `${count} of ${totalMatching} matching ECNs selected` : `${count} selected`,
      );
      root.querySelector("#generate-selected").disabled = count === 0;
      root.querySelector("#delete-selected").disabled = count === 0;
    };

    /** Clear explicit/filter-wide membership whenever its defining filters change. */
    const clearSummarySelection = ({ keepResult = false } = {}) => {
      allFiltered = false;
      selectedIds = new Set();
      excludedIds = new Set();
      generationRequestId = null;
      root.querySelectorAll(".summary-selection").forEach((checkbox) => { checkbox.checked = false; });
      if (!keepResult) setFeedback(root, "#generation-result", "");
      showSummarySelection();
    };

    /** Build the API selection without pagination or presentation sorting. */
    const summarySelection = () => allFiltered ? {
      mode: "filtered",
      query: state.query,
      ecn_number: state.ecn_number,
      plant: state.plant,
      analysis_state: state.analysis_state || null,
      has_warnings: state.has_warnings === "" ? null : state.has_warnings === "true",
      excluded_ecn_ids: [...excludedIds],
    } : {
      mode: "explicit",
      ecn_ids: [...selectedIds],
    };

    /** Refresh the historical page while suppressing stale results. */
    const loadList = async () => {
      const request = listReads.next();
      try {
        const focusedRowId = document.activeElement?.dataset?.ecnRowId || null;
        setText(root.querySelector("#list-loading"), "Loading ECNs.");
        root.querySelector("#list-loading").hidden = false;
        root.querySelector("#list-error").hidden = true;
        setText(root.querySelector("#list-error"), "");
        const result = await api.listEcns(state, { signal: request.signal });
        if (!request.isCurrent()) return;
        root.querySelector("#list-loading").hidden = true;
        totalMatching = result.total_items;
        result.items.forEach((item) => ecnNumbers.set(item.ecn_row_id, item.display_ecn_number));
        renderEcnRows(root.querySelector("#table-scroll tbody"), result.items, state, isSelected);
        renderEcnCards(root.querySelector("#ecn-card-list"), result.items, state, isSelected);
        showSummarySelection();
        if (focusedRowId) {
          const visibleResults = window.matchMedia?.("(max-width: 600px)").matches ?? window.innerWidth <= 600
            ? root.querySelector("#ecn-card-list")
            : root.querySelector("#table-scroll");
          visibleResults?.querySelector(`a[data-ecn-row-id="${focusedRowId}"]`)?.focus();
        }
        root.querySelector("#empty-state").hidden = result.total_items !== 0;
        setText(root.querySelector("#page-summary"), `Page ${result.page} of ${Math.max(result.total_pages, 1)}`);
        root.querySelector("#previous-page").disabled = result.page <= 1;
        root.querySelector("#next-page").disabled = result.page >= result.total_pages;
        setText(root.querySelector("#list-error"), "");
        root.querySelector("#list-error").hidden = true;
      } catch (error) {
        if (error.name !== "AbortError" && request.isCurrent()) {
          root.querySelector("#list-loading").hidden = true;
          setText(root.querySelector("#list-error"), error.message);
          root.querySelector("#list-error").hidden = false;
        }
      }
    };

    /** Read aggregate analysis progress through its own request lane. */
    const loadProgress = async () => {
      const request = progressReads.next();
      const result = await api.getAnalysisProgress({ signal: request.signal });
      if (!request.isCurrent()) throw new DOMException("Superseded", "AbortError");
      return result;
    };

    /** Render progress and refresh list rows only when job counts changed. */
    const showProgress = (value, refreshOnChange = false) => {
      const nextSignature = [
        value.ready_count || 0,
        value.queued_count,
        value.blocked_count || 0,
        value.running_count,
        value.completed_count,
        value.completed_with_gaps_count,
        value.failed_count,
      ].join(":");
      const changed = progressSignature !== null && nextSignature !== progressSignature;
      progressSignature = nextSignature;
      setText(root.querySelector("#kpi-total"), value.total_count);
      setText(root.querySelector("#kpi-complete"), value.completed_count + value.completed_with_gaps_count);
      setText(root.querySelector("#kpi-progress"), value.queued_count + value.running_count);
      setText(root.querySelector("#kpi-attention-count"), value.failed_count);
      if (refreshOnChange && changed) void loadList();
    };

    /** Start one serial progress loop for the current page. */
    const beginPolling = () => {
      stopPolling();
      stopPolling = startPolling({
        load: loadProgress,
        onUpdate: (value) => showProgress(value, true),
        onError: (error) => {
          setText(root.querySelector("#list-error"), error.message);
          root.querySelector("#list-error").hidden = false;
        },
        isFinished: (value) => value.unfinished_count === 0,
        signal: context.signal,
      });
    };

    const applyFilters = (statusMessage = "") => {
      clearSummarySelection();
      const form = new FormData(filtersForm);
      state = parseListState(new URLSearchParams(form));
      state.page = 1;
      syncOverviewFilters();
      updateActiveFilterCount();
      setFeedback(root, "#filter-result-status", statusMessage);
      history.replaceState({}, "", `/ecns?${serializeListState(state)}`);
      void loadList();
    };
    on(filtersForm, "submit", (event) => {
      event.preventDefault();
      applyFilters();
    });
    on(root.querySelector("#apply-filters"), "click", () => applyFilters());
    on(filtersDisclosure, "click", () => {
      secondaryFiltersExpanded = !secondaryFiltersExpanded;
      syncFilterDisclosure();
    });
    on(filtersForm, "input", (event) => {
      if (event.target.closest?.("#secondary-filters")) updateActiveFilterCount();
    });
    on(filtersForm, "change", (event) => {
      if (event.target.closest?.("#secondary-filters")) updateActiveFilterCount();
    });
    on(root.querySelector("#clear-filters"), "click", () => {
      state = { ...DEFAULT_LIST_STATE };
      fillFilters();
      applyFilters("Filters cleared.");
    });
    for (const { id, value, message } of overviewFilters) {
      on(root.querySelector(`#${id}`), "click", () => {
        state = { ...state, analysis_state: value, page: 1 };
        fillFilters();
        applyFilters(message);
      });
    }
    on(window, "resize", syncFilterDisclosure);
    on(root.querySelector("#previous-page"), "click", () => {
      state.page -= 1;
      history.replaceState({}, "", `/ecns?${serializeListState(state)}`);
      void loadList();
    });
    on(root.querySelector("#next-page"), "click", () => {
      state.page += 1;
      history.replaceState({}, "", `/ecns?${serializeListState(state)}`);
      void loadList();
    });
    on(root.querySelector("#ecn-results"), "change", (event) => {
      const checkbox = event.target.closest?.(".summary-selection");
      if (!checkbox) return;
      if (allFiltered) {
        if (checkbox.checked) excludedIds.delete(checkbox.value);
        else excludedIds.add(checkbox.value);
      } else if (checkbox.checked) selectedIds.add(checkbox.value);
      else selectedIds.delete(checkbox.value);
      root.querySelectorAll(".summary-selection").forEach((control) => {
        if (control.value === checkbox.value) control.checked = checkbox.checked;
      });
      generationRequestId = null;
      setFeedback(root, "#generation-result", "");
      showSummarySelection();
    });

    disposeImport = bindImportDialog({
      root,
      signal: context.signal,
      uploadSources: api.uploadSources,
      uuid,
      imported: async () => {
        state = parseListState("");
        fillFilters();
        updateActiveFilterCount();
        history.replaceState({}, "", `/ecns?${serializeListState(state)}`);
        await Promise.all([loadList(), loadProgress().then(showProgress)]);
      },
    });
    on(root.querySelector("#select-all-matching"), "click", () => {
      allFiltered = true;
      selectedIds.clear();
      excludedIds.clear();
      generationRequestId = null;
      root.querySelectorAll(".summary-selection").forEach((checkbox) => { checkbox.checked = true; });
      setFeedback(root, "#generation-result", "");
      showSummarySelection();
    });
    on(root.querySelector("#clear-summary-selection"), "click", () => clearSummarySelection());
    on(root.querySelector("#generate-selected"), "click", async () => {
      const button = root.querySelector("#generate-selected");
      button.disabled = true;
      generationRequestId ||= uuid();
      try {
        const batch = await api.generateSummaries(summarySelection(), generationRequestId, context.signal);
        const scheduled = batch.selection.scheduled_job_ids.length;
        const completed = batch.jobs.filter((job) => job.state === "COMPLETED").length;
        const alreadyRunning = Math.max(0, batch.jobs.length - scheduled - completed);
        setFeedback(root, "#generation-result", `${scheduled} scheduled · ${alreadyRunning} already running · ${completed} completed`);
        clearSummarySelection({ keepResult: true });
        await Promise.all([loadList(), loadProgress().then(showProgress)]);
        beginPolling();
      } catch (error) {
        if (error.name !== "AbortError") {
          setText(root.querySelector("#list-error"), error.message);
          root.querySelector("#list-error").hidden = false;
        }
      } finally {
        showSummarySelection();
      }
    });
    const deleteDialog = root.querySelector("#delete-dialog");
    const deleteInput = root.querySelector("#delete-confirmation");
    const confirmDelete = root.querySelector("#confirm-delete");
    /** Close the delete dialog and forget its typed phrase and last error. */
    const closeDelete = () => {
      if (typeof deleteDialog.close === "function" && deleteDialog.open) deleteDialog.close();
      deleteInput.value = "";
      confirmDelete.disabled = true;
      setText(root.querySelector("#delete-error"), "");
    };
    on(root.querySelector("#delete-selected"), "click", () => {
      const list = root.querySelector("#delete-list");
      list.replaceChildren();
      if (allFiltered) {
        setText(root.querySelector("#delete-summary"), `${selectedCount()} matching ECNs will be deleted.`);
      } else {
        setText(root.querySelector("#delete-summary"), `${selectedIds.size} ECNs will be deleted:`);
        for (const id of selectedIds) {
          const item = document.createElement("li");
          item.textContent = ecnNumbers.get(id) || id;
          list.append(item);
        }
      }
      closeDelete();
      if (typeof deleteDialog.showModal === "function") deleteDialog.showModal();
      deleteInput.focus();
    });
    on(deleteInput, "input", () => { confirmDelete.disabled = deleteInput.value !== "DELETE"; });
    on(root.querySelector("#cancel-delete"), "click", closeDelete);
    on(deleteDialog, "cancel", (event) => { event.preventDefault(); if (!deleteInput.disabled) closeDelete(); });
    on(confirmDelete, "click", async () => {
      confirmDelete.disabled = true;
      deleteInput.disabled = true;
      try {
        const result = await api.deleteEcns(summarySelection(), deleteInput.value, context.signal);
        deleteInput.disabled = false;
        closeDelete();
        clearSummarySelection();
        const graphNote = result.graph_cleanup === "failed" ? " · knowledge-graph cleanup failed (see server log)" : "";
        setFeedback(root, "#generation-result", `${result.ecns} ECN assessments deleted${graphNote}`);
        state.page = 1;
        history.replaceState({}, "", `/ecns?${serializeListState(state)}`);
        await Promise.all([loadList(), loadProgress().then(showProgress)]);
      } catch (error) {
        deleteInput.disabled = false;
        if (error.name !== "AbortError") setText(root.querySelector("#delete-error"), error.message);
        confirmDelete.disabled = deleteInput.value !== "DELETE";
      }
    });
    fillFilters();
    updateActiveFilterCount();
    syncFilterDisclosure();
    await loadList();
    try {
      const progress = await loadProgress();
      showProgress(progress);
      if (progress.unfinished_count > 0) beginPolling();
    } catch (error) {
      if (error.name !== "AbortError") {
        setText(root.querySelector("#list-error"), error.message);
        root.querySelector("#list-error").hidden = false;
      }
    }

    /** Stop polling, abort each lane, and detach every page-owned listener. */
    return () => {
      stopPolling();
      disposeImport();
      listReads.dispose();
      progressReads.dispose();
      listeners.forEach((remove) => remove());
    };
  };
}

export const init = createEcnsPage();
