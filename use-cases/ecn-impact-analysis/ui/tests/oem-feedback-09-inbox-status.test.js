import { afterEach, beforeEach, expect, it, vi } from "vitest";
import html from "../src/pages/ecns/ecns.html?raw";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
const css = readFileSync(resolve("src/pages/ecns/ecns.css"), "utf8");
import { createEcnsPage } from "../src/pages/ecns/ecns.js";
import { listEnvelope, rowOne } from "./fixtures/api-responses.js";

const cleanups = [];

/** Synthetic progress with the same derived equations as AnalysisProgressResponse. */
function progress(counts = {}) {
  const value = {
    ready_count: 0, queued_count: 0, blocked_count: 0, running_count: 0,
    completed_count: 0, completed_with_gaps_count: 0, failed_count: 0,
    ...counts,
  };
  return {
    ...value,
    unfinished_count: value.queued_count + value.blocked_count + value.running_count,
    total_count: Object.values(value).reduce((sum, count) => sum + count, 0),
  };
}

function mount(query = "") {
  document.body.innerHTML = `<main>${html}</main>`;
  const controller = new AbortController();
  cleanups.push(() => controller.abort());
  return {
    container: document.querySelector("main"),
    query: new URLSearchParams(query), signal: controller.signal, navigate: vi.fn(),
  };
}

async function initialize({ value = progress(), items = [], query = "", apiOverrides = {}, startPolling = vi.fn(() => vi.fn()) } = {}) {
  const page = mount(query);
  const api = {
    listEcns: vi.fn().mockResolvedValue(listEnvelope(items)),
    getAnalysisProgress: vi.fn().mockResolvedValue(value),
    ...apiOverrides,
  };
  const disposeImport = vi.fn();
  const cleanup = await createEcnsPage({
    api, startPolling, bindImportDialog: () => disposeImport,
  })(page);
  cleanups.push(cleanup);
  return { page, api, cleanup, startPolling, disposeImport };
}

beforeEach(() => {
  history.replaceState({}, "", "/ecns");
  sessionStorage.clear();
});
afterEach(() => {
  cleanups.splice(0).reverse().forEach((cleanup) => cleanup());
  vi.restoreAllMocks();
  document.body.replaceChildren();
});

it.each([
  { name: "empty", counts: {}, inProgress: 0, attention: 0, complete: 0 },
  { name: "queued", counts: { queued_count: 3 }, inProgress: 3, attention: 0, complete: 0 },
  { name: "running", counts: { running_count: 4 }, inProgress: 4, attention: 0, complete: 0 },
  { name: "blocked only", counts: { blocked_count: 7 }, inProgress: 0, attention: 0, complete: 0 },
  { name: "ready and complete", counts: { ready_count: 2, completed_count: 3, completed_with_gaps_count: 5 }, inProgress: 0, attention: 0, complete: 8 },
  { name: "mixed", counts: { ready_count: 1, queued_count: 2, blocked_count: 7, running_count: 3, completed_count: 4, completed_with_gaps_count: 5, failed_count: 6 }, inProgress: 5, attention: 6, complete: 9 },
])("preserves exact global counter sums: $name", async ({ counts, inProgress, attention, complete }) => {
  const value = progress(counts);
  await initialize({ value });
  expect(document.querySelector("#kpi-progress").textContent).toBe(String(inProgress));
  expect(document.querySelector("#kpi-attention-count").textContent).toBe(String(attention));
  expect(document.querySelector("#kpi-complete").textContent).toBe(String(complete));
  expect(document.querySelector("#kpi-total").textContent).toBe(String(value.total_count));
});

it("offers collapsed native keyboard help outside the attention action and named scrollable evidence", () => {
  mount();
  const help = document.querySelector("#generation-count-help");
  const summary = help.querySelector("summary");
  const region = help.querySelector('[role="region"]');
  expect(help.tagName).toBe("DETAILS");
  expect(help.open).toBe(false);
  expect(summary.textContent).toBe("What triggers these generation counters?");
  expect(region.getAttribute("aria-label")).toBe("Generation counter explanations");
  expect(region.tabIndex).toBe(0);
  expect(document.querySelector("#kpi-attention").contains(help)).toBe(false);
  expect(document.querySelector("#kpi-attention").getAttribute("aria-describedby").split(" "))
    .toEqual(["generation-count-scope"]);
  expect(document.querySelector("#kpi-attention-action")).toBeNull();
  expect(document.body.textContent).not.toContain("Show needs attention");
  expect(css).toContain(".generation-count-guidance summary:focus-visible");
  expect(css).toContain(".generation-count-help-content:focus-visible");
  expect(css).toContain("max-height: 20rem;");
  // Native details activation/keyboard behavior is browser-owned; no fake toggle handler.
  help.open = true;
  region.focus();
  expect(document.activeElement).toBe(region);
});

it("explains state entry/exit, technical failure retry, blocked exclusions and non-generation distinctions", () => {
  mount();
  const waiting = document.querySelector("#kpi-progress-help").textContent;
  const failed = document.querySelector("#kpi-attention-help").textContent;
  const help = document.querySelector("#generation-count-help").textContent;
  expect(waiting).toContain("queued jobs waiting to be claimed plus running jobs");
  expect(waiting).toContain("leaves when generation completes, fails, or is held as Waiting for AI");
  expect(waiting).toContain("Waiting for AI (blocked) jobs are not included");
  expect(failed).toContain("technical failure");
  expect(failed).toContain("even if partial results were saved");
  expect(failed).toContain("View to inspect its details");
  expect(failed).toContain("Retry analysis when appropriate");
  expect(failed).toContain("leaves this counter when it is queued again for retry");
  expect(help).toContain("Ready for summary, completed summaries (including complete with gaps), and Waiting for AI do not contribute");
  for (const distinction of ["source-data warnings", "source release status", "manufacturing feasibility", "engineering-review decisions"]) {
    expect(help).toContain(distinction);
  }
});

it.each([
  ["kpi-total-filter", ""],
  ["kpi-complete-filter", "SUMMARIES_AVAILABLE"],
  ["kpi-progress-filter", "IN_PROGRESS"],
  ["kpi-attention", "FAILED"],
])("keeps counts global while %s filters and preserves other filters, sort and selection reset", async (id, analysisState) => {
  const { api } = await initialize({
    value: progress({ failed_count: 8, blocked_count: 6, queued_count: 2 }),
    items: [{ ...rowOne, analysis_state: "BLOCKED", counts: { ...rowOne.counts, warning_count: 99 } }],
    query: "query=synthetic&ecn_number=DEMO&plant=DE01&has_warnings=true&analysis_state=BLOCKED&sort=warning_count&order=asc&page=2&page_size=50",
  });
  const scope = document.querySelector("#generation-count-scope");
  expect(scope.hidden).toBe(false);
  expect(scope.textContent).toContain("all imported ECNs across uploads, not just the filtered list or current page");
  expect(scope.textContent).toContain("keeps your other list filters");
  expect(api.getAnalysisProgress.mock.calls[0]).toHaveLength(1);
  expect(Object.keys(api.getAnalysisProgress.mock.calls[0][0])).toEqual(["signal"]);
  document.querySelector(".summary-selection").click();
  expect(document.querySelector("#summary-selection-status").textContent).toBe("1 selected");
  const action = document.querySelector(`#${id}`);
  expect(action.tagName).toBe("BUTTON");
  expect(action.type).toBe("button");
  expect(action.getAttribute("aria-controls")).toBe("ecn-results");
  action.focus();
  action.click();
  await vi.waitFor(() => expect(api.listEcns).toHaveBeenCalledTimes(2));
  expect(api.listEcns.mock.lastCall[0]).toMatchObject({
    query: "synthetic", ecn_number: "DEMO", plant: "DE01", has_warnings: "true",
    analysis_state: analysisState, sort: "warning_count", order: "asc", page: 1, page_size: 50,
  });
  expect(document.activeElement).toBe(action);
  expect(document.querySelector("#summary-selection-status").textContent).toBe("0 selected");
  expect(document.querySelector("#filter-result-status").hidden).toBe(false);
  expect(document.querySelector("#kpi-attention-count").textContent).toBe("8");
  expect(document.querySelector("#kpi-progress").textContent).toBe("2");
  expect(api.getAnalysisProgress).toHaveBeenCalledOnce();
  expect(new URLSearchParams(location.search).get("analysis_state") || "").toBe(analysisState);
  expect(document.querySelector('[name="analysis_state"]').value).toBe(analysisState);
  expect(action.getAttribute("aria-pressed")).toBe("true");
  expect(document.querySelectorAll('.kpi-card[aria-pressed="true"]')).toHaveLength(1);
});

it("synchronizes active cards with restored URLs, applied dropdown filters and clear filters", async () => {
  await initialize({ query: "analysis_state=SUMMARIES_AVAILABLE" });
  expect(document.querySelector("#kpi-complete-filter").getAttribute("aria-pressed")).toBe("true");
  const select = document.querySelector('[name="analysis_state"]');
  select.value = "IN_PROGRESS";
  document.querySelector("#apply-filters").click();
  expect(document.querySelector("#kpi-progress-filter").getAttribute("aria-pressed")).toBe("true");
  expect(document.querySelector("#kpi-complete-filter").getAttribute("aria-pressed")).toBe("false");
  select.value = "QUEUED";
  document.querySelector("#apply-filters").click();
  expect(document.querySelectorAll('.kpi-card[aria-pressed="true"]')).toHaveLength(0);
  document.querySelector("#clear-filters").click();
  expect(document.querySelector("#kpi-total-filter").getAttribute("aria-pressed")).toBe("true");
});

it.each([
  ["kpi-complete-filter", "SUMMARIES_AVAILABLE"],
  ["kpi-progress-filter", "IN_PROGRESS"],
])("uses the %s group for filter-wide generation selections", async (id, analysisState) => {
  const generateSummaries = vi.fn().mockResolvedValue({ selection: { scheduled_job_ids: [] }, jobs: [] });
  const { api } = await initialize({ items: [rowOne], apiOverrides: { generateSummaries } });
  document.querySelector(`#${id}`).click();
  await vi.waitFor(() => expect(api.listEcns).toHaveBeenCalledTimes(2));
  document.querySelector("#select-all-matching").click();
  document.querySelector("#generate-selected").click();
  await vi.waitFor(() => expect(generateSummaries).toHaveBeenCalledOnce());
  expect(generateSummaries.mock.calls[0][0]).toMatchObject({ mode: "filtered", analysis_state: analysisState });
  await vi.waitFor(() => expect(api.getAnalysisProgress).toHaveBeenCalledTimes(2));
});

it("preserves polling transitions, unchanged-signature suppression and BLOCKED unfinished behavior", async () => {
  const startPolling = vi.fn(() => vi.fn());
  const { api, page } = await initialize({ value: progress({ queued_count: 2, failed_count: 1 }), startPolling });
  const loop = startPolling.mock.calls[0][0];
  expect(loop.signal).toBe(page.signal);
  expect(loop.isFinished(progress({ blocked_count: 3 }))).toBe(false);
  expect(loop.isFinished(progress({ failed_count: 3 }))).toBe(true);
  const help = document.querySelector("#generation-count-help");
  help.open = true;
  loop.onUpdate(progress({ queued_count: 2, failed_count: 1 }));
  expect(api.listEcns).toHaveBeenCalledOnce();
  loop.onUpdate(progress({ running_count: 2, failed_count: 1 }));
  await vi.waitFor(() => expect(api.listEcns).toHaveBeenCalledTimes(2));
  loop.onUpdate(progress({ completed_with_gaps_count: 2, failed_count: 1 }));
  await vi.waitFor(() => expect(api.listEcns).toHaveBeenCalledTimes(3));
  expect(document.querySelector("#kpi-progress").textContent).toBe("0");
  expect(document.querySelector("#kpi-complete").textContent).toBe("2");
  expect(document.querySelector("#kpi-attention-count").textContent).toBe("1");
  loop.onUpdate(progress({ completed_with_gaps_count: 2, queued_count: 1 }));
  await vi.waitFor(() => expect(api.listEcns).toHaveBeenCalledTimes(4));
  expect(document.querySelector("#kpi-attention-count").textContent).toBe("0");
  expect(help.open).toBe(true);
});

it("keeps guidance available during pending progress, empty success and failed progress without substituting zero", async () => {
  let resolveProgress;
  const page = mount();
  const api = {
    listEcns: vi.fn().mockResolvedValue(listEnvelope([])),
    getAnalysisProgress: vi.fn(() => new Promise((resolve) => { resolveProgress = resolve; })),
  };
  const pending = createEcnsPage({ api, bindImportDialog: () => vi.fn() })(page);
  await vi.waitFor(() => expect(api.getAnalysisProgress).toHaveBeenCalledOnce());
  expect(document.querySelector("#kpi-progress").textContent).toBe("—");
  expect(document.querySelector("#generation-count-scope").hidden).toBe(false);
  expect(document.querySelector("#empty-state").hidden).toBe(false);
  resolveProgress(progress());
  cleanups.push(await pending);
  expect(document.querySelector("#kpi-progress").textContent).toBe("0");
  await initialize({ apiOverrides: { getAnalysisProgress: vi.fn().mockRejectedValue(new Error("Synthetic progress unavailable.")) } });
  expect(document.querySelector("#kpi-progress").textContent).toBe("—");
  expect(document.querySelector("#kpi-attention-count").textContent).toBe("—");
  expect(document.querySelector("#list-error").textContent).toBe("Synthetic progress unavailable.");
  expect(document.querySelector("#list-error").getAttribute("role")).toBe("alert");
  expect(document.querySelector("#list-error").hidden).toBe(false);
  expect(document.querySelector("#generation-count-help")).not.toBeNull();
});

it("stops polling, aborts progress reads and detaches all card listeners on cleanup", async () => {
  const stop = vi.fn();
  const { api, cleanup, disposeImport } = await initialize({
    value: progress({ blocked_count: 1 }), startPolling: vi.fn(() => stop),
  });
  const progressSignal = api.getAnalysisProgress.mock.calls[0][0].signal;
  cleanups.splice(cleanups.indexOf(cleanup), 1);
  cleanup();
  expect(stop).toHaveBeenCalledOnce();
  expect(disposeImport).toHaveBeenCalledOnce();
  expect(progressSignal.aborted).toBe(true);
  document.querySelectorAll("button.kpi-card").forEach((card) => card.click());
  expect(api.listEcns).toHaveBeenCalledOnce();
  expect(location.search).not.toContain("analysis_state=FAILED");
});
