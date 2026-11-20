import { beforeEach, expect, it, vi } from "vitest";
import html from "../src/pages/settings/settings.html?raw";
import { createSettingsPage } from "../src/pages/settings/settings.js";
import {
  plantSettingsEnvelope,
  progressEnvelope,
  providerStatusEnvelope,
  referenceDatasetsEnvelope,
} from "./fixtures/api-responses.js";

/** Mount the Settings template and return a route-compatible context. */
function settingsContext() {
  document.body.innerHTML = `<main>${html}</main>`;
  return {
    container: document.querySelector("main"),
    query: new URLSearchParams(),
    params: {},
    signal: new AbortController().signal,
    navigate: vi.fn(),
  };
}

/** Return the complete Settings API surface with per-test overrides. */
function settingsApi(overrides = {}) {
  return {
    getAnalysisProgress: vi.fn().mockResolvedValue(progressEnvelope()),
    getProviderStatus: vi.fn().mockResolvedValue(providerStatusEnvelope()),
    checkProviderStatus: vi.fn().mockResolvedValue(providerStatusEnvelope()),
    listReferenceDatasets: vi.fn().mockResolvedValue(referenceDatasetsEnvelope()),
    publishReferenceDataset: vi.fn(),
    deleteReferenceDataset: vi.fn(),
    resetHistory: vi.fn(),
    getPlantSettings: vi.fn().mockResolvedValue(plantSettingsEnvelope()),
    savePlantSettings: vi.fn(async (body) => plantSettingsEnvelope(body)),
    ...overrides,
  };
}

it("focuses the requested purchasing source from an acceptance-material link", async () => {
  const context = settingsContext();
  context.query.set("source", "PURCHASE_INFO_RECORDS");
  const cleanup = await createSettingsPage({ api: settingsApi(), startPolling: () => vi.fn() })(context);
  expect(document.activeElement).toBe(document.querySelector('.source-card[data-dataset-type="PURCHASE_INFO_RECORDS"]'));
  cleanup();
});

beforeEach(() => sessionStorage.clear());

it("renders safe provider readiness and exactly six maintained source controls", async () => {
  const api = settingsApi();
  await createSettingsPage({ api, startPolling: () => vi.fn() })(settingsContext());
  expect(document.body.textContent).toContain("gpt-5.4");
  expect(document.body.textContent).toContain("Text Generation");
  expect(document.querySelectorAll(".source-card")).toHaveLength(6);
  expect(document.querySelectorAll(".source-upload-step")).toHaveLength(18);
  expect(document.querySelector("#advanced-administration-title").textContent).toBe("Advanced Administration");
  expect(document.querySelector("#plant-mapping").closest(".advanced-administration")).not.toBeNull();
  expect(document.querySelector("#history-reset-details").open).toBe(false);
  expect(document.body.textContent).toContain("Purchase Info Prices");
  expect(document.querySelector('form[data-dataset-type="PURCHASE_INFO_PRICES"] [name="source_file"]').accept)
    .toBe(".csv,text/csv");
  expect(document.body.textContent).toContain("EBOM MBOM Q900V.xlsx");
  expect(document.body.textContent).toContain("sap-procurement-status-v1");
  expect(document.body.textContent).toContain("External procurement");
  // Plants are derived from conditional rules, never typed at upload.
  expect(document.querySelector('.source-upload-form [name="plant_code"]')).toBeNull();
  // Retired source types must not appear in the Settings page.
  expect(document.body.textContent).not.toContain("Make/Buy history");
  expect(document.body.textContent).not.toContain("Make/Buy business rules");
  expect(document.body.textContent).not.toContain("Historical Make/Buy");
  expect(document.body.textContent).not.toContain("https://");
  expect(document.body.textContent).not.toContain("Bearer");
});

it("labels a stopped text-generation deployment as unavailable", async () => {
  const unavailable = providerStatusEnvelope();
  unavailable.models = [{ ...unavailable.models[0], status: "NOT_AVAILABLE" }];
  const api = settingsApi({ getProviderStatus: vi.fn().mockResolvedValue(unavailable) });
  await createSettingsPage({ api, startPolling: () => vi.fn() })(settingsContext());
  expect(document.body.textContent).toContain("Deployment unavailable");
});

it("refreshes provider readiness only after an explicit check", async () => {
  const api = settingsApi({
    getProviderStatus: vi.fn().mockResolvedValue(providerStatusEnvelope({ checked_at: null })),
  });
  await createSettingsPage({ api, startPolling: () => vi.fn() })(settingsContext());
  expect(api.checkProviderStatus).not.toHaveBeenCalled();
  document.querySelector("#check-provider").click();
  await vi.waitFor(() => expect(api.checkProviderStatus).toHaveBeenCalledOnce());
  expect(document.querySelector("#provider-checked-at").textContent).toContain("Sep");
});

it("accepts the source-specific replacement format and identifies an invalid selection", async () => {
  const published = {
    ...referenceDatasetsEnvelope().items[0],
    dataset_type: "PURCHASE_INFO_PRICES",
    file_name: "prices.csv",
    warning_count: 2,
  };
  const api = settingsApi({ publishReferenceDataset: vi.fn().mockResolvedValue(published) });
  await createSettingsPage({ api, startPolling: () => vi.fn() })(settingsContext());

  const stockForm = document.querySelector('form[data-dataset-type="STOCK"]');
  const invalidFile = new File(["not a workbook"], "stock.csv", { type: "text/csv" });
  Object.defineProperty(stockForm.querySelector('[name="source_file"]'), "files", { value: [invalidFile] });
  stockForm.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
  expect(api.publishReferenceDataset).not.toHaveBeenCalled();
  expect(stockForm.querySelector('[name="source_file"]').getAttribute("aria-invalid")).toBe("true");
  expect(stockForm.textContent).toContain("Select a XLSX workbook.");

  const priceForm = document.querySelector('form[data-dataset-type="PURCHASE_INFO_PRICES"]');
  const priceFile = new File(["Info Rec.,Net Price"], "prices.csv", { type: "text/csv" });
  Object.defineProperty(priceForm.querySelector('[name="source_file"]'), "files", { value: [priceFile] });
  priceForm.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
  await vi.waitFor(() => expect(api.publishReferenceDataset).toHaveBeenCalledWith(expect.objectContaining({
    datasetType: "PURCHASE_INFO_PRICES",
    sourceFile: priceFile,
    signal: expect.any(AbortSignal),
  })));
  await vi.waitFor(() => expect(document.querySelector("#sources-error").textContent).toContain("2 warnings"));
});

it("publishes a stock workbook and activates it without retired Make/Buy source controls", async () => {
  const published = {
    ...referenceDatasetsEnvelope().items[0],
    dataset_type: "STOCK",
    file_name: "stock-snapshot.xlsx",
  };
  const api = settingsApi({ publishReferenceDataset: vi.fn().mockResolvedValue(published) });
  await createSettingsPage({ api, startPolling: () => vi.fn() })(settingsContext());
  // Neither retired type must render a form.
  expect(document.querySelector('form[data-dataset-type="MAKE_BUY_HISTORY"]')).toBeNull();
  expect(document.querySelector('form[data-dataset-type="MAKE_BUY_RULES"]')).toBeNull();
  // The STOCK form must still be present.
  const form = document.querySelector('form[data-dataset-type="STOCK"]');
  const file = new File(["workbook"], "stock-snapshot.xlsx", {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  });
  Object.defineProperty(form.querySelector('[name="source_file"]'), "files", { value: [file] });
  form.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
  await vi.waitFor(() => expect(api.publishReferenceDataset).toHaveBeenCalledOnce());
  expect(api.publishReferenceDataset).toHaveBeenCalledWith(expect.objectContaining({
    datasetType: "STOCK",
    sourceFile: file,
    signal: expect.any(AbortSignal),
  }));
  await vi.waitFor(() => expect(document.querySelector("#sources-error").textContent).toContain("now active"));
  expect(api.listReferenceDatasets).toHaveBeenCalledTimes(2);
});

it("shows an X for every version and deletes only the confirmed file", async () => {
  const original = referenceDatasetsEnvelope();
  const older = { ...original.items[0], dataset_id: "77777777-7777-4777-8777-777777777777", file_name: "older-bom.xlsx", active: false };
  let items = [...original.items, older];
  const api = settingsApi({
    listReferenceDatasets: vi.fn(async () => referenceDatasetsEnvelope({ items })),
    deleteReferenceDataset: vi.fn(async (id) => { items = items.filter((item) => item.dataset_id !== id); }),
  });
  const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
  try {
    await createSettingsPage({ api, startPolling: () => vi.fn() })(settingsContext());
    expect(document.querySelectorAll(".publication-delete")).toHaveLength(3);
    expect(document.querySelector(".publication-inactive").textContent).toContain("older-bom.xlsx");
    document.querySelector('[aria-label="Delete older-bom.xlsx"]').click();
    expect(api.deleteReferenceDataset).not.toHaveBeenCalled();
    confirm.mockReturnValue(true);
    document.querySelector('[aria-label="Delete older-bom.xlsx"]').click();
    await vi.waitFor(() => expect(api.deleteReferenceDataset).toHaveBeenCalledWith(
      older.dataset_id,
      { signal: expect.any(AbortSignal) },
    ));
    await vi.waitFor(() => expect(document.querySelector('[aria-label="Delete older-bom.xlsx"]')).toBeNull());
    expect(document.querySelector("#sources-error").textContent).toContain("was deleted");
  } finally {
    confirm.mockRestore();
  }
});

it("keeps a referenced file visible when deletion is rejected", async () => {
  const api = settingsApi({
    deleteReferenceDataset: vi.fn().mockRejectedValue(new Error("This file is used by an ECN upload.")),
  });
  const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
  try {
    await createSettingsPage({ api, startPolling: () => vi.fn() })(settingsContext());
    document.querySelector('[aria-label="Delete EBOM MBOM Q900V.xlsx"]').click();
    await vi.waitFor(() => expect(document.querySelector("#sources-error").textContent).toContain("used by an ECN upload"));
    expect(document.querySelector('[aria-label="Delete EBOM MBOM Q900V.xlsx"]')).not.toBeNull();
  } finally {
    confirm.mockRestore();
  }
});

it("fails reset closed for queued, blocked, running, or unknown activity", async () => {
  const api = settingsApi({
    getAnalysisProgress: vi.fn().mockResolvedValue(progressEnvelope({
      queued_count: 0,
      blocked_count: 1,
      running_count: 0,
      unfinished_count: 1,
    })),
  });
  await createSettingsPage({ api, startPolling: () => vi.fn() })(settingsContext());
  const input = document.querySelector("#reset-confirmation");
  input.value = "RESET HISTORY";
  input.dispatchEvent(new Event("input"));
  expect(document.querySelector("#reset-history").disabled).toBe(true);
  expect(document.querySelector("#active-status").textContent).toContain("1 waiting for AI");
  expect(document.body.textContent).toContain("Settings-managed reference datasets");
});

it("status reads cannot abort reset and success records counts before list navigation", async () => {
  let resetSignal;
  let releaseReset;
  const context = settingsContext();
  const api = settingsApi({
    getAnalysisProgress: vi.fn().mockResolvedValue(progressEnvelope({
      running_count: 0,
      completed_count: 2,
      unfinished_count: 0,
    })),
    resetHistory: vi.fn((confirmation, { signal }) => {
      resetSignal = signal;
      return new Promise((resolve) => {
        releaseReset = () => resolve({
          deleted_attempt_count: 2,
          deleted_analysis_count: 1,
          deleted_job_count: 2,
          deleted_ecn_count: 2,
          deleted_upload_count: 1,
          deleted_bom_row_count: 99,
          deleted_source_artifact_count: 2,
        });
      });
    }),
  });
  await createSettingsPage({ api, startPolling: () => vi.fn() })(context);
  const input = document.querySelector("#reset-confirmation");
  input.value = "RESET HISTORY";
  input.dispatchEvent(new Event("input"));
  document.querySelector("#reset-history").click();
  await Promise.resolve();
  document.querySelector("#retry-status").click();
  await Promise.resolve();
  expect(resetSignal.aborted).toBe(false);
  releaseReset();
  await vi.waitFor(() => expect(context.navigate).toHaveBeenCalledWith("/ecns"));
  expect(JSON.parse(sessionStorage.getItem("ecn-history-reset-summary")).deleted_bom_row_count).toBe(99);
});

it("handles a race-time 409 by disabling reset and resuming status checks", async () => {
  const poll = vi.fn(() => vi.fn());
  const api = settingsApi({
    getAnalysisProgress: vi.fn().mockResolvedValue(progressEnvelope({
      running_count: 0,
      completed_count: 2,
      unfinished_count: 0,
    })),
    resetHistory: vi.fn().mockRejectedValue(
      Object.assign(new Error("Analysis became active"), { status: 409 }),
    ),
  });
  await createSettingsPage({ api, startPolling: poll })(settingsContext());
  const input = document.querySelector("#reset-confirmation");
  input.value = "RESET HISTORY";
  input.dispatchEvent(new Event("input"));
  document.querySelector("#reset-history").click();
  await vi.waitFor(() => expect(poll).toHaveBeenCalledOnce());
  expect(document.querySelector("#reset-history").disabled).toBe(true);
  expect(document.querySelector("#reset-error").textContent).toContain("became active");
});


it("edits, validates and saves the Plant Code Mapping with rule states", async () => {
  const api = settingsApi();
  await createSettingsPage({ api, startPolling: () => vi.fn() })(settingsContext());
  const rows = () => [...document.querySelectorAll("#plant-mapping-rows tr")];
  expect(rows().map((row) => row.querySelector('[name="suffix"]').value)).toEqual(["JB", "JD"]);
  const states = Object.fromEntries([...document.querySelectorAll('[name="rule_state"]')].map((box) => [box.value, box.checked]));
  expect(states).toEqual({ "App For Series": true, "In Work": true, "To be Obsolete": false });

  document.querySelector("#add-plant-mapping").click();
  const added = rows()[2];
  added.querySelector('[name="suffix"]').value = "is";
  added.querySelector('[name="plant_code"]').value = "PL10";
  added.querySelector('[name="plant_name"]').value = "Plant A";
  rows()[1].querySelector(".plant-mapping-remove").click();
  document.querySelector('[name="rule_state"][value="In Work"]').checked = false;
  document.querySelector("#add-plant-mapping").click();
  document.querySelector("#plant-mapping-form").requestSubmit();
  await vi.waitFor(() => expect(api.savePlantSettings).toHaveBeenCalled());

  expect(api.savePlantSettings.mock.calls[0][0]).toEqual({
    mappings: [
      { suffix: "JB", plant_code: "PL10", plant_name: "Plant A" },
      { suffix: "is", plant_code: "PL10", plant_name: "Plant A" },
    ],
    excluded_rule_states: ["In Work", "To be Obsolete"],
  });
  await vi.waitFor(() => expect(document.querySelector("#plant-mapping-status").textContent).toContain("saved"));
});

it("shows the server's validation message when the mapping is rejected", async () => {
  const api = settingsApi({
    savePlantSettings: vi.fn().mockRejectedValue(Object.assign(new Error("Brand suffix JB is mapped twice."), { status: 400 })),
  });
  await createSettingsPage({ api, startPolling: () => vi.fn() })(settingsContext());
  document.querySelector("#plant-mapping-form").requestSubmit();
  await vi.waitFor(() => expect(document.querySelector("#plant-mapping-status").textContent).toBe("Brand suffix JB is mapped twice."));
  expect(document.querySelector('#plant-mapping-form [type="submit"]').disabled).toBe(false);
});
