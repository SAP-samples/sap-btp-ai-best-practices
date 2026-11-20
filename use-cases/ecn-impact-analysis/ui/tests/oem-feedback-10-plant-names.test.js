import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { buildPlantNameIndex, describeRenderedPlants, plantLabel, suppliedPlantLabel } from "../src/modules/plant-display.js";
import { buildIdentifierIndex, linkifyIdentifiers } from "../src/components/identifier-glossary.js";
import { plantsLabel } from "../src/components/plant-impact-table.js";
import { renderBomChanges } from "../src/components/bom-changes-table.js";
import { referenceLine, mountReferenceDocuments } from "../src/components/reference-documents.js";
import { materialDialog } from "../src/components/material-dialog.js";
import { createEcnDetailPage } from "../src/pages/ecn-detail/ecn-detail.js";
import * as api from "../src/services/workspace-api.js";
import html from "../src/pages/ecn-detail/ecn-detail.html?raw";
import { detailEnvelope } from "./fixtures/api-responses.js";

vi.mock("../src/services/workspace-api.js", () => ({
  getObject: vi.fn(), assessDrawing: vi.fn(), generateSummaries: vi.fn(),
  getSnapshot: vi.fn(), getAnalysisStages: vi.fn(),
}));
vi.mock("../src/components/snapshot-workspace.js", () => ({ snapshotWorkspace: () => ({
  inspect: inspectPart, updateAnalysisDescriptions: vi.fn(), updateAuditDetail: vi.fn(),
}) }));
const inspectPart = vi.hoisted(() => vi.fn());
const pairs = [{ plant_code: "PK01", plant_name: "Plant B" }, { plant_code: "PL10", plant_name: "Plant A" }];
let controller;
let originalShowModal;
beforeEach(() => {
  originalShowModal = Object.getOwnPropertyDescriptor(HTMLDialogElement.prototype, "showModal");
  Object.defineProperty(HTMLDialogElement.prototype, "showModal", { configurable: true, writable: true,
    value: vi.fn(function () { this.open = true; }) });
  controller = new AbortController();
  document.body.replaceChildren(); vi.clearAllMocks();
});
afterEach(() => {
  controller.abort(); vi.restoreAllMocks();
  if (originalShowModal) Object.defineProperty(HTMLDialogElement.prototype, "showModal", originalShowModal);
  else delete HTMLDialogElement.prototype.showModal;
});

it("uses supplied names only, trims pairs and rejects conflicting mappings regardless of order", () => {
  const rows = [...pairs, { plant_code: "PK01", plant_name: " Plant B " }, { plant_code: "XX01", plant_name: "" }];
  expect([...buildPlantNameIndex(rows)]).toEqual([["PK01", "Plant B"], ["PL10", "Plant A"]]);
  for (const conflict of [rows.concat({ plant_code: "PK01", plant_name: "Other" }), [{ plant_code: "PK01", plant_name: "Other" }, ...rows]]) {
    expect(buildPlantNameIndex(conflict).has("PK01")).toBe(false);
    expect(plantLabel("PK01", buildPlantNameIndex(conflict))).toBe("PK01");
  }
  expect(buildPlantNameIndex(null).size).toBe(0);
  const host = document.createElement("p"); host.textContent = "PK01 XX01";
  describeRenderedPlants(host, buildPlantNameIndex([...pairs, { plant_code: "PK01", plant_name: "Other" }]));
  expect(host.textContent).toBe("PK01 XX01");
  expect(plantLabel("XX01", buildPlantNameIndex(pairs))).toBe("XX01");
});

it("replaces whole tokens, not specification codes, Unicode words, paths, formulas or URL text", () => {
  const host = document.createElement("div");
  host.innerHTML = '<p>Plants PK01 and PL10; (PK01). PK010 XPK01 PK01_A PK01-B éPK01 /PK01 https://example.test?q=PK01 www.example.test?q=PL10 $PK01 + PL10$ PK01=PL10 mailto:PK01@example.test PS4_OK</p><code>PK01</code><pre>PL10</pre><a href="https://example.test/PK01">PK01</a><math>PL10</math>';
  host.id = "PK01";
  describeRenderedPlants(host, buildPlantNameIndex(pairs));
  expect(host.querySelector("p").textContent).toBe("Plants Plant B and Plant A; (Plant B). PK010 XPK01 PK01_A PK01-B éPK01 /PK01 https://example.test?q=PK01 www.example.test?q=PL10 $PK01 + PL10$ PK01=PL10 mailto:PK01@example.test PS4_OK");
  expect(host.id).toBe("PK01");
  expect(host.querySelector("a").getAttribute("href")).toBe("https://example.test/PK01");
  for (const selector of ["code", "pre", "a", "math"]) expect(host.querySelector(selector).textContent).toMatch(/^(PK01|PL10)$/);
  const once = host.innerHTML;
  describeRenderedPlants(host, buildPlantNameIndex(pairs));
  expect(host.innerHTML).toBe(once);
});

it("keeps material descriptions literal and clickable part identities intact", () => {
  const host = document.createElement("div"); host.textContent = "PK01 seal, part PART1 in PL10.";
  const index = buildIdentifierIndex({ parts: [{ number: "PART1", description: "PK01 seal" }] });
  const inspect = vi.fn();
  linkifyIdentifiers(host, index, inspect);
  describeRenderedPlants(host, buildPlantNameIndex(pairs), ["PK01 seal"]);
  expect(host.textContent).toBe("PK01 seal, part PART1 in Plant A.");
  host.querySelector("button").click(); expect(inspect).toHaveBeenCalledWith("PART1");
  expect(host.querySelector("button").getAttribute("aria-label")).toContain("PART1: PK01 seal");
});

it("treats malicious names as text and escapes regex metacharacters", () => {
  const host = document.createElement("div"); host.textContent = "Plant P.1";
  describeRenderedPlants(host, buildPlantNameIndex([{ plant_code: "P.1", plant_name: '<img src=x onerror="alert(1)">' }]));
  expect(host.textContent).toBe('Plant <img src=x onerror="alert(1)">');
  expect(host.querySelector("img")).toBeNull();
});

it("keeps explicit plant fields description-first without inventing names or changing BOM IDs", () => {
  expect(suppliedPlantLabel("PL10 Plant A")).toBe("Plant A (PL10)");
  expect(suppliedPlantLabel("PK01")).toBe("PK01");
  expect(suppliedPlantLabel("Already named (PL10)")).toBe("Already named (PL10)");
  expect(plantsLabel(pairs)).toBe("Plant B (PK01), Plant A (PL10)");
  expect(plantsLabel([{ plant_code: "PL10", plant_name: "A" }, { plant_code: "PL10", plant_name: "B" }])).toBe("PL10, PL10");
  const host = document.createElement("div"), inspect = vi.fn();
  renderBomChanges(host, { lines: [{ incoming: { number: "PART1", description: "PS4_OK PK01 seal" }, plants: pairs }] }, inspect);
  expect(host.textContent).toContain("Plant B (PK01), Plant A (PL10)");
  expect(host.textContent).toContain("PS4_OK PK01 seal");
  host.querySelector("button").click(); expect(inspect).toHaveBeenCalledWith("PART1");
});

it("maps only reference plant fields and preserves other source facts and lazy API behavior", async () => {
  const item = { number: "D1", description: "PK01 spec", plant_code: "PK01", file_name: "PK01.xlsx", sheet: "EBOM", row: 2 };
  expect(referenceLine(item, buildPlantNameIndex(pairs))).toBe("D1 · PK01 spec · plant Plant B (PK01) · PK01.xlsx · EBOM · row 2");
  const host = document.createElement("div"), fetch = vi.fn();
  mountReferenceDocuments(host, fetch, "row", controller.signal, Promise.resolve({ plant_impact: { plants: pairs }, reference_documents: [item] }));
  expect(fetch).not.toHaveBeenCalled();
  const panel = host.querySelector("details"); panel.open = true; panel.dispatchEvent(new Event("toggle"));
  await vi.waitFor(() => expect(panel.textContent).toContain("plant Plant B (PK01)"));
  expect(fetch).not.toHaveBeenCalled(); expect(item.plant_code).toBe("PK01");
});

it.each([{ sourcing: pairs, expected: "Plant B (PK01), XX01" }, { sourcing: [], expected: "PK01, XX01" }, { sourcing: [...pairs, { plant_code: "PK01", plant_name: "Conflicting" }], expected: "PK01, XX01" }])("maps coverage strings only from verified material-response pairs: $expected", async ({ sourcing, expected }) => {
  api.getObject.mockResolvedValue({ number: "PART1", description: "PK01 seal", material_brief: {
    bom_coverage: { plants: ["PK01", "XX01"], occurrences: [], coverage: "NOT_OBSERVED" }, sourcing,
  } });
  const inspect = materialDialog("row", controller.signal, vi.fn()); await inspect("PART1");
  expect(document.querySelector('td[data-label="Plant"]').textContent).toBe(expected);
  expect(document.querySelector("dialog").textContent).toContain("PK01 seal");
  expect(api.getObject).toHaveBeenCalledWith("row", { number: "PART1", technical: true, page: 1, page_size: 25 }, expect.any(AbortSignal));
});

it("repaints stored summary after snapshot and polling without source mutation or losing part links", async () => {
  document.body.innerHTML = `<main>${html}</main>`;
  const detail = detailEnvelope();
  detail.job.state = "RUNNING";
  const markdown = '- **Where:** Plants PK01 and PL10. PART1. Unknown XX01. PK010 PS4_OK.\n- **What changes:** PK01 seal.\n\n`PK01` [PL10](https://example.test/PL10)';
  detail.analysis.sections.executive_summary.assessment_markdown = markdown;
  let resolveSnapshot, update;
  const snapshotRequest = new Promise(resolve => { resolveSnapshot = resolve; });
  const io = { getEcn: vi.fn().mockResolvedValue(detail), getSnapshot: vi.fn().mockReturnValue(snapshotRequest) };
  await createEcnDetailPage({ api: io, startPolling: ({ onUpdate }) => { update = onUpdate; return vi.fn(); } })({
    container: document.querySelector("main"), params: { ecn_row_id: "synthetic-row" }, query: new URLSearchParams(), signal: controller.signal, navigate: vi.fn(),
  });
  expect(document.querySelector(".ai-change-summary").textContent).toContain("Plants PK01 and PL10");
  resolveSnapshot({ plant_impact: { plants: pairs }, parts: [{ number: "PART1", description: "PK01 seal" }] });
  await vi.waitFor(() => expect(document.querySelector(".ai-change-summary").textContent).toContain("Plants Plant B and Plant A"));
  const summary = document.querySelector(".ai-change-summary");
  expect(summary.textContent).toContain("PK01 seal");
  expect(summary.textContent).toContain("PK010 PS4_OK");
  expect(summary.querySelector("code").textContent).toBe("PK01");
  summary.querySelector("button").click(); expect(inspectPart).toHaveBeenCalledWith("PART1");
  expect(update).toBeTypeOf("function");
  update(detail);
  update(detail);
  expect(document.querySelectorAll(".ai-change-summary .plant-name-ref")).toHaveLength(2);
  expect(detail.analysis.sections.executive_summary.assessment_markdown).toBe(markdown);
  expect(io.getSnapshot).toHaveBeenCalledTimes(1);
  expect(io.getSnapshot).toHaveBeenCalledWith("synthetic-row", controller.signal);
});

it.each([{ summary: null, expected: "No AI change summary is available" }, { summary: "Plants PK01 and XX01.", expected: "Plants PK01 and XX01." }])("preserves empty summary/snapshot states: $expected", async ({ summary, expected }) => {
  document.body.innerHTML = `<main>${html}</main>`;
  const detail = detailEnvelope(); detail.analysis.sections.executive_summary.assessment_markdown = summary;
  await createEcnDetailPage({ api: { getEcn: vi.fn().mockResolvedValue(detail), getSnapshot: vi.fn().mockResolvedValue({ plant_impact: { plants: [] } }) } })({
    container: document.querySelector("main"), params: { ecn_row_id: "row" }, query: new URLSearchParams(), signal: controller.signal, navigate: vi.fn(),
  });
  await vi.waitFor(() => expect(document.querySelector("#affected-sites-list").textContent).toContain("No manufacturing site"));
  expect(document.querySelector("#source-record-fields").textContent).toContain(expected);
  expect(document.querySelectorAll(".plant-name-ref")).toHaveLength(0);
});

it("retains honest summary codes when snapshot fails and shows existing plant error", async () => {
  document.body.innerHTML = `<main>${html}</main>`;
  const detail = detailEnvelope(); detail.analysis.sections.executive_summary.assessment_markdown = "Plants PK01 and PL10.";
  await createEcnDetailPage({ api: { getEcn: vi.fn().mockResolvedValue(detail), getSnapshot: vi.fn().mockRejectedValue(new Error("Synthetic unavailable")) } })({
    container: document.querySelector("main"), params: { ecn_row_id: "row" }, query: new URLSearchParams(), signal: controller.signal, navigate: vi.fn(),
  });
  await vi.waitFor(() => expect(document.querySelector("#affected-sites-list").textContent).toContain("unavailable"));
  expect(document.querySelector(".ai-change-summary").textContent).toContain("PK01 and PL10");
});
