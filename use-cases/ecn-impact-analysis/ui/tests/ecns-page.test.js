import { beforeEach, expect, it, vi } from "vitest";
import html from "../src/pages/ecns/ecns.html?raw";
import { createEcnsPage } from "../src/pages/ecns/ecns.js";
import { listEnvelope, missingIdentifierRow, progressEnvelope, rowOne } from "./fixtures/api-responses.js";

/** Mount the ECN page template and return a router-compatible context. */
function context(query = "sort=warning_count&order=desc&page=1&page_size=25") {
  document.body.innerHTML = `<main>${html}</main>`;
  return {
    container: document.querySelector("main"),
    query: new URLSearchParams(query),
    signal: new AbortController().signal,
    navigate: vi.fn(),
  };
}

beforeEach(() => {
  history.replaceState({}, "", "/ecns");
  sessionStorage.clear();
});

it("places statistics before the one-shot import action", () => {
  const page = context();
  const statistics = page.container.querySelector(".kpi-grid");
  const importAction = page.container.querySelector("#open-import");

  expect(statistics.compareDocumentPosition(importAction) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  expect(page.container.querySelector("#export-workspace")).toBeNull();
  expect(page.container.querySelector("#upload-panel")).toBeNull();
});

it("keeps inbox guidance in an adjacent information button", () => {
  const page = context();
  const title = page.container.querySelector("#ecns-title");
  const info = page.container.querySelector("#ecns-info");

  expect(info.getAttribute("icon")).toBe("information");
  expect(info.getAttribute("tooltip")).toBe(
    "Review imported snapshots and explicitly choose which ECNs receive AI summaries.",
  );
  expect(title.compareDocumentPosition(info) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  expect(page.container.querySelector(".page-header p")).toBeNull();
});

it("does not render a redundant ECN list heading", () => {
  const page = context();

  expect(page.container.querySelector("#history-title")).toBeNull();
  expect(page.container.textContent).not.toContain("Stored Occurrences");
});

it("uses compact UI5 buttons for ECN table actions", () => {
  const page = context();
  const actions = [
    ["open-import", null, "Emphasized"],
    ["apply-filters", null, "Emphasized"],
    ["select-all-matching", "multiselect-all", "Transparent"],
    ["clear-summary-selection", "decline", "Transparent"],
    ["generate-selected", "document-text", "Transparent"],
    ["delete-selected", "delete", "Negative"],
  ];

  for (const [id, icon, design] of actions) {
    const button = page.container.querySelector(`#${id}`);
    expect(button.tagName).toBe("UI5-BUTTON");
    expect(button.getAttribute("icon")).toBe(icon);
    expect(button.getAttribute("design")).toBe(design);
  }
  const applyFilters = page.container.querySelector("#apply-filters");
  expect(applyFilters.textContent).toBe("Go");
  expect(page.container.querySelector("#filters").contains(applyFilters)).toBe(true);
  expect(page.container.querySelector(".workspace-controls")).toBeNull();
  expect(page.container.querySelector("#summary-selection-status").parentElement.classList.contains("table-toolbar")).toBe(true);
});

it("keeps secondary filters directly visible on desktop and counts only non-default secondary values", async () => {
  const api = {
    listEcns: vi.fn().mockResolvedValue(listEnvelope()),
    getAnalysisProgress: vi.fn().mockResolvedValue(progressEnvelope()),
  };
  await createEcnsPage({ api, startPolling: () => vi.fn() })(context("ecn_number=ECN100&plant=PL10&has_warnings=true&order=asc&page_size=50"));

  const disclosure = document.querySelector("#filters-disclosure");
  const secondary = document.querySelector("#secondary-filters");
  expect(secondary.hidden).toBe(false);
  expect(disclosure.getAttribute("aria-controls")).toBe("secondary-filters");
  expect(disclosure.getAttribute("aria-expanded")).toBe("true");
  expect(document.querySelector("#active-filter-count").textContent).toBe("5 active");
  expect(disclosure.getAttribute("aria-label")).toBe("Filters, 5 active filters");
});

it("discloses secondary filters on narrow screens with button semantics", async () => {
  const originalMatchMedia = window.matchMedia;
  window.matchMedia = vi.fn().mockReturnValue({ matches: true });
  try {
    const api = {
      listEcns: vi.fn().mockResolvedValue(listEnvelope()),
      getAnalysisProgress: vi.fn().mockResolvedValue(progressEnvelope()),
    };
    await createEcnsPage({ api, startPolling: () => vi.fn() })(context("plant=PL10"));

    const disclosure = document.querySelector("#filters-disclosure");
    const secondary = document.querySelector("#secondary-filters");
    expect(disclosure.tagName).toBe("BUTTON");
    expect(disclosure.getAttribute("aria-expanded")).toBe("false");
    expect(secondary.hidden).toBe(true);
    expect(document.querySelector("#active-filter-count").textContent).toBe("1 active");
    disclosure.focus();
    disclosure.click();
    expect(document.activeElement).toBe(disclosure);
    expect(disclosure.getAttribute("aria-expanded")).toBe("true");
    expect(secondary.hidden).toBe(false);
  } finally {
    window.matchMedia = originalMatchMedia;
  }
});

it("restores focus to the visible ECN card after a narrow-screen progress refresh", async () => {
  const originalMatchMedia = window.matchMedia;
  window.matchMedia = vi.fn().mockReturnValue({ matches: true });
  let onProgress;
  try {
    const api = {
      listEcns: vi.fn().mockResolvedValue(listEnvelope()),
      getAnalysisProgress: vi.fn().mockResolvedValue(progressEnvelope()),
    };
    await createEcnsPage({
      api,
      startPolling: ({ onUpdate }) => { onProgress = onUpdate; return vi.fn(); },
    })(context());

    const cardLink = document.querySelector("#ecn-card-list a[data-ecn-row-id]");
    cardLink.focus();
    onProgress(progressEnvelope({ running_count: 0, completed_count: 2, unfinished_count: 0 }));
    await vi.waitFor(() => expect(document.activeElement).toBe(document.querySelector("#ecn-card-list a[data-ecn-row-id]")));
  } finally {
    window.matchMedia = originalMatchMedia;
  }
});

it("uses the needs-attention KPI action to apply the existing FAILED analysis filter", async () => {
  const api = {
    listEcns: vi.fn().mockResolvedValue(listEnvelope()),
    getAnalysisProgress: vi.fn().mockResolvedValue(progressEnvelope({ blocked_count: 1, failed_count: 2 })),
  };
  await createEcnsPage({ api, startPolling: () => vi.fn() })(context("query=shaft&sort=warning_count&order=asc&page=2&page_size=50"));

  const action = document.querySelector("#kpi-attention");
  expect(action.tagName).toBe("BUTTON");
  expect(document.querySelector("#kpi-attention-count").textContent).toBe("2");
  action.click();
  await vi.waitFor(() => expect(api.listEcns.mock.lastCall[0]).toMatchObject({
    query: "shaft", analysis_state: "FAILED", sort: "warning_count", order: "asc", page: 1, page_size: 50,
  }));
  expect(document.querySelector('[name="analysis_state"]').value).toBe("FAILED");
  const filterFeedback = document.querySelector("#filter-result-status");
  expect(filterFeedback.tagName).toBe("UI5-MESSAGE-STRIP");
  expect(filterFeedback.getAttribute("design")).toBe("Information");
  expect(filterFeedback.getAttribute("role")).toBe("status");
  expect(filterFeedback.hidden).toBe(false);
  expect(filterFeedback.textContent).toBe("Showing ECNs whose analysis needs attention.");
  expect(window.location.search).toContain("analysis_state=FAILED");
  expect(window.location.search).toContain("page=1");
});

it("renders list failure in an assertive negative UI5 message strip", async () => {
  const api = {
    listEcns: vi.fn().mockRejectedValue(new Error("ECN list is unavailable.")),
    getAnalysisProgress: vi.fn().mockResolvedValue(progressEnvelope()),
  };
  await createEcnsPage({ api, startPolling: () => vi.fn() })(context());

  const listError = document.querySelector("#list-error");
  expect(listError.tagName).toBe("UI5-MESSAGE-STRIP");
  expect(listError.getAttribute("design")).toBe("Negative");
  expect(listError.getAttribute("role")).toBe("alert");
  expect(listError.getAttribute("aria-live")).toBe("assertive");
  expect(listError.hidden).toBe(false);
  expect(listError.textContent).toBe("ECN list is unavailable.");
});

it("renders bounded business rows and routes missing IDs by internal row ID", async () => {
  const api = {
    listEcns: vi.fn().mockResolvedValue(listEnvelope()),
    getAnalysisProgress: vi.fn().mockResolvedValue(progressEnvelope()),
    uploadSources: vi.fn(),
  };
  const stop = vi.fn();
  await createEcnsPage({ api, startPolling: () => stop, uuid: () => "request-1" })(context());

  const rows = [...document.querySelectorAll("#table-scroll tbody tr")];
  expect(document.querySelectorAll("#import-rows tr")).toHaveLength(0);
  expect(rows[0].textContent).toContain("Change bracket geometry");
  expect(rows[0].textContent).toContain("3");
  expect(rows[0].textContent).toContain("2");
  expect(rows[0].textContent).toContain("Running");
  expect(rows[1].textContent).toContain("Medium");
  expect(rows[0].textContent).not.toContain("RUNNING");
  const statusTags = rows[0].querySelectorAll(".status-cell ui5-tag");
  expect(statusTags).toHaveLength(2);
  expect(statusTags[0].getAttribute("design")).toBe("Information");
  expect(statusTags[0].getAttribute("size")).toBe("S");
  expect(statusTags[0].classList.contains("ecn-status-tag")).toBe(true);
  const description = rows[0].querySelector(".description-clamp");
  expect(description.title).toBe("Change bracket geometry");
  expect(description.getAttribute("aria-label")).toBe("Change bracket geometry");
  expect(description.tabIndex).toBe(0);
  expect(document.querySelector("#kpi-total").textContent).toBe("2");
  expect(document.querySelector("#kpi-complete").textContent).toBe("1");
  const missing = document.querySelector('a[href^="/ecns/22222222"]');
  expect(missing.textContent).toBe("Missing ECN ID");
  expect(missing.href).toContain("sort=warning_count");
  expect(document.querySelectorAll("#table-scroll .view-action")).toHaveLength(2);
  const cards = [...document.querySelectorAll("#ecn-card-list .ecn-result-card")];
  expect(cards).toHaveLength(2);
  expect(cards[0].textContent).toContain("Change bracket geometry");
  expect(cards[0].textContent).toContain("Analysis");
  expect(cards[0].textContent).toContain("Review");
  expect(cards[0].textContent).toContain("Warnings");
  expect(cards[0].querySelectorAll("ui5-tag")).toHaveLength(3);
  expect(cards[0].querySelector("ui5-tag").getAttribute("design")).toBe("Information");
  expect(cards[0].querySelector(".view-action").getAttribute("aria-label")).toContain("ECN100");

  // Resolution date renders in its own column (first of the two date cells),
  // ahead of the Uploaded date, using the shared locale formatter.
  const headers = [...document.querySelectorAll("#table-scroll thead th")].map((th) => th.textContent);
  expect(headers).toContain("Resolution date");
  const resolutionCell = rows[0].querySelectorAll("td.date-cell")[0];
  expect(resolutionCell.textContent).toBe(
    new Intl.DateTimeFormat().format(new Date(rowOne.resolution_date)),
  );
});

it("shows partial analysis for a failed job with compatible saved stages", async () => {
  const api = {
    listEcns: vi.fn().mockResolvedValue(listEnvelope([{
      ...rowOne, analysis_state: "FAILED", has_partial_analysis: true,
    }])),
    getAnalysisProgress: vi.fn().mockResolvedValue(progressEnvelope({ failed_count: 1 })),
  };
  await createEcnsPage({ api, startPolling: () => vi.fn() })(context());

  expect(document.querySelector(".status-cell").textContent).toBe("Partial");
  expect(document.querySelector('option[value="FAILED"]').textContent).toBe("Needs attention");
});

it("refreshes imported ECNs on page one in newest-upload order and resets the active-filter count", async () => {
  let onImported;
  const api = {
    listEcns: vi.fn().mockResolvedValue(listEnvelope([])),
    getAnalysisProgress: vi.fn().mockResolvedValue(progressEnvelope()),
  };
  const init = createEcnsPage({
    api,
    startPolling: () => vi.fn(),
    bindImportDialog: ({ imported }) => { onImported = imported; return vi.fn(); },
  });
  await init(context("plant=PL10&sort=warning_count&order=asc&page=2&page_size=50"));
  expect(document.querySelector("#active-filter-count").textContent).toBe("3 active");
  await onImported();

  expect(api.listEcns.mock.lastCall[0]).toMatchObject({
    page: 1,
    sort: "uploaded_at",
    order: "desc",
  });
  expect(document.querySelector("#active-filter-count").textContent).toBe("0 active");
  expect(document.querySelector("#filters-disclosure").getAttribute("aria-label")).toBe("Filters, 0 active filters");
  expect(window.location.search).toContain("sort=uploaded_at");
});

it("consumes reset counts exactly once on the empty history", async () => {
  sessionStorage.setItem("ecn-history-reset-summary", JSON.stringify({
    deleted_attempt_count: 2,
    deleted_analysis_count: 1,
    deleted_job_count: 2,
    deleted_ecn_count: 2,
    deleted_upload_count: 1,
    deleted_bom_row_count: 99,
    deleted_source_artifact_count: 2,
  }));
  const api = {
    listEcns: vi.fn().mockResolvedValue(listEnvelope([])),
    getAnalysisProgress: vi.fn().mockResolvedValue(progressEnvelope({
      running_count: 0,
      completed_count: 0,
      unfinished_count: 0,
      total_count: 0,
    })),
    uploadSources: vi.fn(),
  };
  await createEcnsPage({ api, startPolling: () => vi.fn() })(context());
  const resetFeedback = document.querySelector("#reset-summary");
  expect(resetFeedback.tagName).toBe("UI5-MESSAGE-STRIP");
  expect(resetFeedback.getAttribute("design")).toBe("Positive");
  expect(resetFeedback.getAttribute("role")).toBe("status");
  expect(resetFeedback.hidden).toBe(false);
  expect(resetFeedback.textContent).toContain(
    "1 upload, 2 ECNs, 1 analysis, 2 jobs, 2 attempts, 99 BOM rows, and 2 source artifacts deleted.",
  );
  expect(document.querySelector("#empty-state").hidden).toBe(false);
  expect(sessionStorage.getItem("ecn-history-reset-summary")).toBeNull();
});

it("keeps explicit selections across pages and generates without a dialog", async () => {
  const page = (items, number) => ({
    ...listEnvelope(items),
    page: number,
    total_items: 2,
    total_pages: 2,
  });
  const api = {
    listEcns: vi.fn(async (state) => page(
      state.page === 1 ? [rowOne] : [missingIdentifierRow],
      state.page,
    )),
    getAnalysisProgress: vi.fn().mockResolvedValue(progressEnvelope()),
    uploadSources: vi.fn(),
    generateSummaries: vi.fn().mockResolvedValue({
      selection: { scheduled_job_ids: ["job-1", "job-2"] },
      jobs: [{ state: "QUEUED" }, { state: "QUEUED" }],
    }),
  };
  await createEcnsPage({ api, startPolling: () => vi.fn(), bindImportDialog: () => vi.fn(), uuid: () => "request-1" })(context());
  document.querySelector(".summary-selection").click();
  document.querySelector("#next-page").click();
  await vi.waitFor(() => expect(document.querySelector(".summary-selection").value).toBe(missingIdentifierRow.ecn_row_id));
  document.querySelector(".summary-selection").click();
  document.querySelector("#generate-selected").click();
  await vi.waitFor(() => expect(api.generateSummaries).toHaveBeenCalledOnce());

  expect(api.generateSummaries.mock.calls[0][0]).toEqual({
    mode: "explicit",
    ecn_ids: [rowOne.ecn_row_id, missingIdentifierRow.ecn_row_id],
  });
  const generationFeedback = document.querySelector("#generation-result");
  expect(generationFeedback.tagName).toBe("UI5-MESSAGE-STRIP");
  expect(generationFeedback.getAttribute("design")).toBe("Positive");
  expect(generationFeedback.getAttribute("role")).toBe("status");
  expect(generationFeedback.hidden).toBe(false);
  expect(generationFeedback.textContent).toContain("2 scheduled");
  expect(document.querySelector(".summary-dialog")).toBeNull();
});

it("submits a completed ECN for regeneration from the incoming list", async () => {
  const api = {
    listEcns: vi.fn().mockResolvedValue(listEnvelope([missingIdentifierRow])),
    getAnalysisProgress: vi.fn().mockResolvedValue(progressEnvelope()),
    generateSummaries: vi.fn().mockResolvedValue({
      selection: { scheduled_job_ids: ["job-2"] },
      jobs: [{ state: "QUEUED" }],
    }),
  };
  await createEcnsPage({ api, startPolling: () => vi.fn(), bindImportDialog: () => vi.fn(), uuid: () => "request-3" })(context());
  document.querySelector(".summary-selection").click();
  document.querySelector("#generate-selected").click();
  await vi.waitFor(() => expect(api.generateSummaries).toHaveBeenCalledOnce());
  expect(api.generateSummaries.mock.calls[0][0]).toEqual({
    mode: "explicit",
    ecn_ids: [missingIdentifierRow.ecn_row_id],
  });
  expect(document.querySelector("#generation-result").textContent).toContain("1 scheduled");
});

it("submits filter-wide exclusions and clears selection when filters change", async () => {
  const api = {
    listEcns: vi.fn().mockResolvedValue(listEnvelope()),
    getAnalysisProgress: vi.fn().mockResolvedValue(progressEnvelope()),
    uploadSources: vi.fn(),
    generateSummaries: vi.fn().mockResolvedValue({
      selection: { scheduled_job_ids: [] },
      jobs: [{ state: "COMPLETED" }],
    }),
  };
  await createEcnsPage({ api, startPolling: () => vi.fn(), bindImportDialog: () => vi.fn(), uuid: () => "request-2" })(context("query=shaft&page=1&page_size=25"));
  document.querySelector("#select-all-matching").click();
  const first = document.querySelector(".summary-selection");
  first.click();
  document.querySelector("#generate-selected").click();
  await vi.waitFor(() => expect(api.generateSummaries).toHaveBeenCalledOnce());
  expect(api.generateSummaries.mock.calls[0][0]).toEqual({
    mode: "filtered",
    query: "shaft",
    ecn_number: "",
    plant: "",
    analysis_state: null,
    has_warnings: null,
    excluded_ecn_ids: [rowOne.ecn_row_id],
  });
  expect(document.querySelector("#generation-result").textContent).toContain("1 completed");

  document.querySelector("#select-all-matching").click();
  document.querySelector("#apply-filters").click();
  expect(document.querySelector("#summary-selection-status").textContent).toContain("0 selected");
});

it("deletes explicit selections only after the exact DELETE phrase", async () => {
  const api = {
    listEcns: vi.fn().mockResolvedValue(listEnvelope()),
    getAnalysisProgress: vi.fn().mockResolvedValue(progressEnvelope()),
    deleteEcns: vi.fn().mockResolvedValue({ ecns: 2, graph_cleanup: "complete" }),
  };
  await createEcnsPage({ api, startPolling: () => vi.fn(), bindImportDialog: () => vi.fn() })(context());
  const deleteButton = document.querySelector("#delete-selected");
  expect(deleteButton.disabled).toBe(true);
  document.querySelector(".summary-selection").click();
  expect(deleteButton.disabled).toBe(false);
  deleteButton.click();
  expect(document.querySelector("#delete-list").textContent).toContain(rowOne.display_ecn_number);

  const input = document.querySelector("#delete-confirmation");
  const confirm = document.querySelector("#confirm-delete");
  input.value = "delete";
  input.dispatchEvent(new Event("input"));
  expect(confirm.disabled).toBe(true);
  input.value = "DELETE";
  input.dispatchEvent(new Event("input"));
  expect(confirm.disabled).toBe(false);
  confirm.click();

  await vi.waitFor(() => expect(api.deleteEcns).toHaveBeenCalledOnce());
  expect(api.deleteEcns.mock.calls[0].slice(0, 2)).toEqual([
    { mode: "explicit", ecn_ids: [rowOne.ecn_row_id] },
    "DELETE",
  ]);
  await vi.waitFor(() => expect(document.querySelector("#generation-result").textContent).toBe("2 ECN assessments deleted"));
  expect(document.querySelector("#summary-selection-status").textContent).toBe("0 selected");
  expect(api.listEcns).toHaveBeenCalledTimes(2);
});

it("deletes filter-wide selections and keeps conflicts inside the dialog", async () => {
  const api = {
    listEcns: vi.fn().mockResolvedValue(listEnvelope()),
    getAnalysisProgress: vi.fn().mockResolvedValue(progressEnvelope()),
    deleteEcns: vi.fn().mockRejectedValue(Object.assign(new Error("Summaries are running"), { name: "ApiError" })),
  };
  await createEcnsPage({ api, startPolling: () => vi.fn(), bindImportDialog: () => vi.fn() })(context("query=shaft&page=1&page_size=25"));
  document.querySelector("#select-all-matching").click();
  document.querySelector("#delete-selected").click();
  expect(document.querySelector("#delete-summary").textContent).toBe("2 matching ECNs will be deleted.");
  const input = document.querySelector("#delete-confirmation");
  input.value = "DELETE";
  input.dispatchEvent(new Event("input"));
  document.querySelector("#confirm-delete").click();

  await vi.waitFor(() => expect(document.querySelector("#delete-error").textContent).toBe("Summaries are running"));
  expect(api.deleteEcns.mock.calls[0][0]).toMatchObject({ mode: "filtered", query: "shaft", excluded_ecn_ids: [] });
  expect(document.querySelector("#confirm-delete").disabled).toBe(false);
  expect(document.querySelector("#summary-selection-status").textContent).toBe("2 of 2 matching ECNs selected");
});

it("shows plants right after Description and keeps the plant filter in list and selection requests", async () => {
  const api = {
    listEcns: vi.fn().mockResolvedValue(listEnvelope([rowOne, missingIdentifierRow, { ...rowOne, ecn_row_id: "33333333-3333-4333-8333-333333333333", affected_plants: [] }])),
    getAnalysisProgress: vi.fn().mockResolvedValue(progressEnvelope()),
    generateSummaries: vi.fn().mockResolvedValue({ selection: { scheduled_job_ids: [] }, jobs: [] }),
  };
  await createEcnsPage({ api, startPolling: () => vi.fn(), bindImportDialog: () => vi.fn(), uuid: () => "request-3" })(context("plant=PL10&sort=affected_plants"));

  const headers = [...document.querySelectorAll("#table-scroll thead th")].map((th) => th.textContent);
  expect(headers.indexOf("Plants")).toBe(headers.indexOf("Description") + 1);
  const cells = [...document.querySelectorAll("td.plant-cell")];
  expect([...cells[0].querySelectorAll(".plant-list span")].map((line) => line.textContent)).toEqual(["Plant A (PL10)", "Plant C (DE01)"]);
  expect(cells[1].textContent).toBe("—");
  expect(cells[1].firstChild.title).toBe("Not assessed yet");
  expect(cells[2].textContent).toBe("—");
  expect(cells[2].firstChild.title).toBe("No plant linked");

  expect(api.listEcns.mock.calls[0][0]).toMatchObject({ plant: "PL10", sort: "affected_plants" });
  expect(document.querySelector('[name="plant"]').value).toBe("PL10");
  document.querySelector("#select-all-matching").click();
  document.querySelector("#generate-selected").click();
  await vi.waitFor(() => expect(api.generateSummaries).toHaveBeenCalledOnce());
  expect(api.generateSummaries.mock.calls[0][0]).toMatchObject({ mode: "filtered", plant: "PL10" });
});
