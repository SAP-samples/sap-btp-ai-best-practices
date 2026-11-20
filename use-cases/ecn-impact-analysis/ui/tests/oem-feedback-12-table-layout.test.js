import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { materialDialog } from "../src/components/material-dialog.js";
import * as api from "../src/services/workspace-api.js";
vi.mock("../src/services/workspace-api.js", () => ({ getObject: vi.fn(), assessDrawing: vi.fn() }));
let controller;
beforeEach(() => {
  document.body.replaceChildren(); vi.clearAllMocks(); controller = new AbortController();
  HTMLDialogElement.prototype.showModal = function () { this.open = true; };
  HTMLDialogElement.prototype.close = function () { this.open = false; this.dispatchEvent(new Event("close")); };
});
afterEach(() => controller.abort());
function fixture() {
  const long = "SYNTHETIC-UNBROKEN-" + "X".repeat(180);
  return { number: "SIM-PIPE", description: long, types: [], material_brief: {
    bom_coverage: { engineering_occurrences: 1, manufacturing_occurrences: 0, coverage: "ENGINEERING_ONLY", effective_interval_count: 1, plants: [], occurrences: [{ source: "Enriched EBOM", path: [{ description: long, number: "SIM-ASSEMBLY" }], item_or_position: long, quantity: "123456789.125", unit: "EA", valid_from_date: "2026-03-13", valid_to_date: "9999-12-31", change_number: long }] },
    sourcing: [{ plant_code: "SIM", plant_name: long, procurement_type_labels: ["External procurement"], source_lead_time: { status: "SOURCE_LEAD_TIME_AVAILABLE", display: "30 days" }, source_purchase_price: { status: "SOURCE_PRICE_AVAILABLE", net_price: "12.50", currency: "EUR", price_per: "1", order_price_unit: "EA" } }],
  }, manufacturing_feasibility: { overall_result: "Feasible against the assessed PS4 requirements", candidate_reports: [{ id: "synthetic", label: "Synthetic primary", result: "Feasible against the assessed PS4 requirements", criteria: [{ id: "raw_material", title: "Raw material", status: "Pass", calculator: { raw_material: { designation: long, guidelineCoverage: "listed" } } }], limitations: [] }] } };
}
async function open(value = fixture()) {
  api.getObject.mockResolvedValue(value);
  await materialDialog("synthetic-ecn", controller.signal, vi.fn())("SIM-PIPE");
  return document.querySelector("dialog");
}
it("names and focuses all four scoped semantic table scroll regions without rewriting source facts", async () => {
  const value = fixture(); const before = JSON.stringify(value); const dialog = await open(value);
  const regions = dialog.querySelectorAll(".material-evidence-table"); expect(regions).toHaveLength(4);
  for (const region of regions) {
    expect(region.classList.contains("table-scroll")).toBe(true);
    expect(region.getAttribute("role")).toBe("region"); expect(region.tabIndex).toBe(0);
    expect(region.getAttribute("aria-label")).toBeTruthy();
    expect(region.querySelector("thead th")).not.toBeNull(); expect(region.querySelector("tbody td")).not.toBeNull();
  }
  const row = dialog.querySelector(".observed-occurrences-table tbody tr");
  expect([...row.cells].map(cell => cell.textContent)).toEqual(["Enriched EBOM", `${value.description} (SIM-ASSEMBLY)`, value.description, "123456789.125 EA", "13 Mar 2026 – No planned end", value.description]);
  expect(dialog.querySelector(".bom-coverage-table").textContent).toContain("Enriched EBOM: 1");
  expect(dialog.querySelector(".drawing-feasibility-calculation-table").textContent).toContain(value.description);
  expect(dialog.querySelector(".material-procurement-table").textContent).toContain("30 days");
  expect(dialog.querySelector(".material-procurement-table").textContent).toContain("€12.50 / 1 EA");
  expect(JSON.stringify(value)).toBe(before); expect(api.getObject).toHaveBeenCalledTimes(1); expect(api.assessDrawing).not.toHaveBeenCalled();
});
it("retains collapsed native disclosures, expansion without IO, and opener focus return", async () => {
  const opener = document.createElement("button"); document.body.append(opener); opener.focus();
  const dialog = await open();
  for (const disclosure of dialog.querySelectorAll(".observed-occurrences-details, .drawing-feasibility-calculations, .drawing-feasibility-calculation-group")) {
    expect(disclosure.tagName).toBe("DETAILS"); expect(disclosure.open).toBe(false);
    expect(disclosure.firstElementChild.tagName).toBe("SUMMARY"); expect(disclosure.firstElementChild.hasAttribute("role")).toBe(false);
    disclosure.open = true;
  }
  expect(api.getObject).toHaveBeenCalledTimes(1); expect(api.assessDrawing).not.toHaveBeenCalled();
  dialog.close(); expect(document.activeElement).toBe(opener);
});
it("preserves missing source states instead of fabricating evidence tables", async () => {
  const dialog = await open({ number: "SIM-EMPTY", types: [], material_brief: { sourcing: [] } });
  expect(dialog.querySelectorAll(".material-evidence-table")).toHaveLength(0);
  expect(dialog.textContent).toContain("No source-backed procurement permission is available");
  expect(dialog.querySelector(".drawing-preview-input")).not.toBeNull();
});
it("keeps pending and failed object reads from presenting fabricated success", async () => {
  let complete; api.getObject.mockImplementation(() => new Promise(resolve => { complete = resolve; }));
  const failure = vi.fn(); const pending = materialDialog("synthetic-ecn", controller.signal, failure)("SIM-PIPE");
  expect(document.querySelectorAll(".material-evidence-table")).toHaveLength(0);
  complete(fixture()); await pending; expect(document.querySelectorAll(".material-evidence-table")).toHaveLength(4);
  api.getObject.mockRejectedValue(new Error("Synthetic read failure"));
  await materialDialog("synthetic-ecn", controller.signal, failure)("SIM-ERROR");
  expect(failure).toHaveBeenCalledWith(expect.objectContaining({ message: "Synthetic read failure" }));
});
it("uses bounded fixed tracks and wrapping only within material evidence, overriding intrinsic hierarchy width", () => {
  // Resolve from the lead's documented `cd ui` invocation; avoid Vite's jsdom import.meta URL and CSS-disabled transform.
  const css = readFileSync(resolve(process.cwd(), "src/style.css"), "utf8");
  expect(css).toContain(".object-dialog .material-evidence-table { min-width: 0; max-width: 100%; overflow: auto; }");
  expect(css).toContain(".object-dialog .material-evidence-table table { table-layout: fixed; }");
  expect(css).toContain("white-space: normal; overflow-wrap: anywhere; word-break: normal;");
  expect(css).toContain(".object-dialog .observed-occurrences-table .hierarchy-stack { width: 100%; min-width: 0; max-width: 100%; }");
  for (const width of ["76rem", "78rem", "48rem", "64rem"]) expect(css).toContain(`width: max(100%, ${width})`);
  expect(css).toContain("max-height: min(28rem, 45vh)");
  expect(css).toContain(".table-scroll:focus-visible");
  expect(css).not.toMatch(/\.object-dialog \.material-evidence-table[^}]*?(?:text-overflow:\s*ellipsis|overflow:\s*hidden)/);
});
